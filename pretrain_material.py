"""Pretrain one scene-independent 6D material decoder on procedural BRDFs.

No Cat/Pixiu images or test data enter this stage. The same final decoder is
frozen in both inverse-rendering experiments. Edit configs/material.json for
the canonical experiment; each run saves its configuration and source.
"""

import argparse
import json
import math
import subprocess
import tarfile
import time
from pathlib import Path

import torch

from materials import MaterialDecoder, MaterialEncoder
from materials.procedural import enhanced_brdf, sample_directions, sample_materials


ROOT = Path(__file__).resolve().parent


def relative_brdf_error(predicted, target):
    return ((predicted - target).square() / (target.square() + .01)).mean()


@torch.no_grad()
def validate_highlights(encoder, decoder, output):
    """Compare actual analytic BRDF lobes, resolving the narrowest roughness."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    delta = torch.linspace(-16., 16., 4097)
    angle = math.pi/6
    wi = torch.tensor([math.sin(angle), 0., math.cos(angle)]).expand(len(delta), -1)
    theta = -angle + delta * math.pi/180
    wo = torch.stack((theta.sin(),torch.zeros_like(theta),theta.cos()),-1)
    rows=[]
    fig,axes=plt.subplots(2,4,figsize=(16,7))
    for metal in (0.,1.):
        for column,roughness in enumerate((.001,.005,.02,.1)):
            params=torch.tensor([[1.5,.333333,.333333,.333333,.942809,.942809,.942809,
                                  roughness,.25,1.,metal,0.,.5,0.,1.5,0.,.5,.5,.5,.5,0.,.5]])
            expected,_,_=enhanced_brdf(params.expand(len(delta),-1),wi,wo)
            predicted,_,_=decoder(encoder(params).expand(len(delta),-1),wi,wo)
            teacher=expected.mean(-1); neural=predicted.mean(-1)
            def width(curve):
                selected=delta[curve >= curve.max()*.5]
                return float(selected[-1]-selected[0])
            rows.append(dict(metalness=metal,roughness=roughness,teacher_peak=float(teacher.max()),
                             decoder_peak=float(neural.max()),peak_ratio=float(neural.max()/teacher.max()),
                             teacher_width_deg=width(teacher),decoder_width_deg=width(neural),
                             decoder_peak_offset_deg=float(delta[neural.argmax()]-delta[teacher.argmax()])))
            ax=axes[int(metal),column]
            ax.semilogy(delta,teacher,label='Analytic teacher'); ax.semilogy(delta,neural,label='Decoder')
            ax.set_title(f'metal={metal:g}, roughness={roughness:g}'); ax.set_xlabel('Mirror-angle offset (deg)')
            ax.legend()
    fig.tight_layout(); fig.savefig(output/'highlight_curves.png',dpi=140); plt.close(fig)
    (output/'highlight_metrics.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(rows),flush=True)


def fit_highlight_codes(encoder, decoder, output, steps):
    """Fit only six material codes to analytic lobes; test unseen incident angles.

    This isolates encoder error from the frozen decoder's local fitting ability.
    It uses no scene pixels and does not establish a global capacity bound.
    """
    decoder.requires_grad_(False)
    parameters = torch.tensor([
        [1.5,.333333,.333333,.333333,.942809,.942809,.942809,
         roughness,.25,1.,metal,0.,.5,0.,1.5,0.,.5,.5,.5,.5,0.,.5]
        for metal in (0.,1.) for roughness in (.001,.005,.02,.1)])
    initial = encoder(parameters).detach()
    logits = torch.nn.Parameter(torch.logit(initial))
    optimizer = torch.optim.Adam([logits], lr=.02)

    def directions(angles, offsets):
        angles = torch.tensor(angles)[:, None]*math.pi/180
        outgoing = -angles+offsets[None]*math.pi/180
        incident = torch.stack((angles.sin(), torch.zeros_like(angles), angles.cos()), -1)
        wi = incident.expand(-1,len(offsets),-1).reshape(-1,3)
        wo = torch.stack((outgoing.sin(),torch.zeros_like(outgoing),outgoing.cos()),-1).reshape(-1,3)
        valid = wo[:,2]>0
        return wi[valid],wo[valid]

    def responses(codes, wi, wo):
        count = len(wi)
        return decoder(codes[:,None].expand(-1,count,-1).reshape(-1,6),
                       wi.repeat(8,1),wo.repeat(8,1))[0].reshape(8,count,3)

    offsets = torch.logspace(-3, math.log10(18), 128)
    offsets = torch.cat((-offsets.flip(0),torch.zeros(1),offsets))
    wi,wo = directions([15.,45.,70.],offsets)
    target = enhanced_brdf(parameters[:,None].expand(-1,len(wi),-1).reshape(-1,22),
                           wi.repeat(8,1),wo.repeat(8,1))[0].reshape(8,len(wi),3)
    start = time.monotonic()
    for step in range(steps):
        optimizer.param_groups[0]['lr'] = .0001+.5*(.02-.0001)*(1+math.cos(math.pi*step/steps))
        optimizer.zero_grad(set_to_none=True)
        prediction = responses(logits.sigmoid(),wi,wo)
        loss = ((prediction+.01).log()-(target+.01).log()).square().mean()
        loss.backward(); optimizer.step()
        if step == 0 or (step+1)%200 == 0:
            print(json.dumps({'event':'fit_lobes','step':step+1,'loss':float(loss),
                              'seconds':time.monotonic()-start}),flush=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = []
    delta = torch.linspace(-16.,16.,4097)
    fig,axes = plt.subplots(2,4,figsize=(16,7))
    with torch.no_grad():
        for angle in (30.,60.):
            test_wi,test_wo = directions([angle],delta)
            teacher = enhanced_brdf(parameters[:,None].expand(-1,len(delta),-1).reshape(-1,22),
                                     test_wi.repeat(8,1),test_wo.repeat(8,1))[0].reshape(8,len(delta),3)
            before,after = responses(initial,test_wi,test_wo),responses(logits.sigmoid(),test_wi,test_wo)
            for index,params in enumerate(parameters):
                row = {'metalness':float(params[10]),'roughness':float(params[7]),'incident_angle_deg':angle}
                for label,values in [('teacher',teacher),('encoder_code',before),('fitted_code',after)]:
                    curve=values[index].mean(-1);support=delta[curve>=curve.max()*.5]
                    row[label]={'peak':float(curve.max()),'width_deg':float(support[-1]-support[0]),
                                'peak_offset_deg':float(delta[curve.argmax()]),
                                'log_rmse':float((((values[index]+.01).log()-(teacher[index]+.01).log()).square().mean()).sqrt())}
                    if angle == 30:
                        axes.flat[index].semilogy(delta,curve,label=label)
                rows.append(row)
                if angle == 30:
                    axes.flat[index].set_title(f'metal={params[10]:g}, roughness={params[7]:g}')
                    axes.flat[index].legend();axes.flat[index].set_xlabel('Mirror offset (deg)')
    fig.tight_layout();fig.savefig(output/'fitted_lobe_curves.png',dpi=140);plt.close(fig)
    torch.save({'initial':initial,'fitted':logits.detach().sigmoid(),'parameters':parameters},output/'fitted_codes.pt')
    result={'fit_incident_angles':[15,45,70],'test_incident_angles':[30,60],'steps':steps,
            'optimized':'material logits only; decoder and normals fixed','seconds':time.monotonic()-start,'rows':rows}
    (output/'fitted_lobe_metrics.json').write_text(json.dumps(result,indent=2)+'\n')


@torch.no_grad()
def validate(encoder, decoder, count, seed, teacher):
    rng = torch.Generator(device="cuda").manual_seed(seed)
    params, classes = sample_materials(count, "cuda", rng)
    wi, wo = sample_directions(count, "cuda", rng, params=params)
    target, transmission, reflection = teacher(params, wi, wo)
    predicted, pred_t, pred_r = decoder(encoder(params), wi, wo)
    rows = {}
    for label, selected in [("all", torch.ones_like(classes, dtype=torch.bool))] + [
        (name, classes == i) for i, name in enumerate(("haze", "dust", "clearcoat", "scatter", "infill"))
    ]:
        rows[label] = {
            "count": int(selected.sum()),
            "relative_brdf_l2": float(relative_brdf_error(predicted[selected], target[selected])),
            "log1p_brdf_rmse": float((predicted[selected].log1p()-target[selected].log1p()).square().mean().sqrt()),
            "transmission_mae": float((pred_t[selected]-transmission[selected]).abs().mean()),
            "reflection_mae": float((pred_r[selected]-reflection[selected]).abs().mean()),
        }
    if not all(math.isfinite(value) for row in rows.values() for value in row.values()):
        raise FloatingPointError("Non-finite material validation")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/material.json")
    parser.add_argument("--evaluate", type=Path, help="Only evaluate saved prior highlight curves on CPU")
    parser.add_argument("--output", type=Path, help="Output for --evaluate")
    parser.add_argument("--fit-lobes", type=int, default=0,
                        help="With --evaluate, optimize only material codes for this many CPU steps")
    args = parser.parse_args()
    if args.fit_lobes < 0 or (args.fit_lobes and not args.evaluate):
        parser.error('--fit-lobes requires --evaluate and nonnegative steps')
    if args.evaluate:
        if args.output is None:
            parser.error('--evaluate requires --output')
        torch.set_num_threads(4)
        saved=torch.load(args.evaluate,map_location='cpu',weights_only=False)
        encoder,decoder=MaterialEncoder(),MaterialDecoder()
        encoder.load_state_dict(saved['encoder']); decoder.load_state_dict(saved['decoder'])
        args.output.mkdir(parents=True,exist_ok=False)
        validate_highlights(encoder.eval(),decoder.eval(),args.output)
        if args.fit_lobes:
            (args.output/'config.json').write_text(json.dumps(
                {'checkpoint':str(args.evaluate.resolve()),'steps':args.fit_lobes,'device':'cpu'},indent=2)+'\n')
            with tarfile.open(args.output/'source.tar','w') as archive:
                for name in ('pretrain_material.py','materials'):
                    archive.add(ROOT/name,arcname=name,filter=lambda info:None if '__pycache__' in info.name else info)
            fit_highlight_codes(encoder,decoder,args.output,args.fit_lobes)
        return
    config = json.loads(args.config.read_text())
    output = ROOT / config["output"]
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2)+"\n")
    with tarfile.open(output / "source.tar", "w") as archive:
        for name in ("pretrain_material.py", "configs/material.json", "materials"):
            archive.add(ROOT / name, arcname=name, filter=lambda info: None if "__pycache__" in info.name else info)
    torch.set_num_threads(4)
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    encoder, decoder = MaterialEncoder().cuda(), MaterialDecoder().cuda()
    optimizer = torch.optim.Adam([*encoder.parameters(), *decoder.parameters()],
                                 lr=config["learning_rate"], fused=True)
    teacher = torch.compile(enhanced_brdf, fullgraph=True) if config["compile_teacher"] else enhanced_brdf
    print(json.dumps({"event": "initialized", "gpu": torch.cuda.get_device_name(0),
                      "torch": torch.__version__, "source_revision": subprocess.check_output(
                          ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                      "decoder_parameters": sum(p.numel() for p in decoder.parameters()),
                      "encoder_parameters": sum(p.numel() for p in encoder.parameters())}), flush=True)
    start = time.monotonic()
    history = (output / "history.jsonl").open("w", buffering=1)
    for step in range(1, config["steps"] + 1):
        progress = (step - 1) / max(1, config["steps"] - 1)
        learning_rate = config["final_learning_rate"] + .5 * (
            config["learning_rate"] - config["final_learning_rate"]) * (1 + math.cos(math.pi * progress))
        optimizer.param_groups[0]["lr"] = learning_rate
        with torch.no_grad():
            params, _ = sample_materials(config["batch_size"], "cuda")
            wi, wo = sample_directions(config["batch_size"], "cuda", params=params)
            target, transmission, reflection = teacher(params, wi, wo)
        latent = encoder(params)
        noisy = latent + (torch.rand_like(latent)*2-1)*config["latent_jitter"]
        predicted, pred_t, pred_r = decoder(noisy, wi, wo)
        f_loss = relative_brdf_error(predicted, target)
        # Relative squared error caps the penalty for a missed bright peak at 1,
        # while strongly penalizing overshoot. Log-space regression treats both
        # multiplicative errors symmetrically over the BRDF's dynamic range.
        log_loss = (torch.log(predicted + .01) - torch.log(target + .01)).square().mean()
        t_loss = (pred_t-transmission).abs().mean()
        r_loss = (pred_r-reflection).abs().mean()
        loss = .95*log_loss + .04*t_loss + .01*r_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % config["log_every"] == 0 or step == config["steps"]:
            row = {"step": step, "loss": float(loss), "brdf_relative_l2": float(f_loss),
                   "brdf_log_l2": float(log_loss),
                   "transmission_l1": float(t_loss), "reflection_l1": float(r_loss),
                   "learning_rate": learning_rate,
                   "seconds": time.monotonic()-start}
            if not all(math.isfinite(value) for value in row.values()):
                raise FloatingPointError(f"Non-finite material training at {step}")
            history.write(json.dumps(row)+"\n")
            print(json.dumps(row), flush=True)
    metrics = validate(encoder, decoder, config["validation_samples"], config["seed"]+1, teacher)
    # Neutral dielectric, no decorators; this is only latent initialization.
    neutral = torch.tensor([[1.5, .333333, .333333, .333333, .942809, .942809, .942809,
                             .1, .25, .8, 0., 0., .5, 0., 1.5, 0., .5, .5, .5, .5, 0., .5]], device="cuda")
    with torch.no_grad():
        initial_latent = encoder(neutral)[0].cpu()
    torch.save({"config": config, "step": config["steps"], "encoder": encoder.cpu().state_dict(),
                "decoder": decoder.cpu().state_dict(), "initial_latent": initial_latent,
                "metrics": metrics}, output / "last.pt")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2)+"\n")
    print(json.dumps({"event": "complete", "output": str(output), "validation": metrics["all"]}), flush=True)


if __name__ == "__main__":
    main()
