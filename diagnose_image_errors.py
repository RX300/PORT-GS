"""GPU image-error measurements on existing real-scene fit/validation frames."""

import argparse
import json
import math
import tarfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F

from cameras import load_camera_offsets
from data import SceneDataset
from evaluate import load_model, load_surface_field, load_normal_field, to_device, render_observation, target_image, ssim, save_pair, neutral_peak_mask, neutral_peak_metrics
from renderer import visibility_hint, _rasterize
from surface import SurfacePriors, surface_losses


def block_energy(image, block):
    return F.avg_pool2d(image.permute(2, 0, 1)[None], block).square().mean()


def boundary(mask):
    eroded = -F.max_pool2d(-mask[None, None].float(), 3, 1, 1)[0, 0]
    return mask & (eroded == 0)


def normal_cue_coherence(points, directions, frames, sizes=(.005, .01, .02)):
    """One vote per frame/cell; predict each direction using other views only."""
    result = {}
    for size in sizes:
        cells = (points/size).floor().long()
        keys = torch.cat((cells, frames[:,None]), -1)
        groups, inverse = torch.unique(keys, dim=0, return_inverse=True)
        sums = torch.zeros(len(groups),3).index_add_(0,inverse,directions)
        per_frame = F.normalize(sums,dim=-1)
        unique_cells, inverse = torch.unique(groups[:,:3],dim=0,return_inverse=True)
        counts = torch.bincount(inverse,minlength=len(unique_cells))
        totals = torch.zeros(len(unique_cells),3).index_add_(0,inverse,per_frame)
        supported = counts[inverse]>=3
        others = F.normalize(totals[inverse][supported]-per_frame[supported],dim=-1)
        angles = (others*per_frame[supported]).sum(-1).clamp(-1,1).acos()*180/torch.pi
        result[str(size)] = {
            'frame_cell_observations':len(groups), 'cells_with_3_views':int((counts>=3).sum()),
            'supported_observations':int(supported.sum()),
            'mean_leave_one_view_out_angle_deg':float(angles.mean()) if len(angles) else None,
            'p90_leave_one_view_out_angle_deg':float(angles.quantile(.9)) if len(angles) else None,
            'fraction_under_10_degrees':float((angles<10).float().mean()) if len(angles) else None,
        }
    return result


@torch.no_grad()
def highlight_diagnostics(transport, inputs, foreground, alpha, target, target_alpha, gamma, output, stem):
    """Probe local neutral peaks; these are image proxies, not BRDF ground truth."""
    g, receivers, eye, light, intensity, _, _ = inputs
    covered = alpha[..., 0] > 0
    incident, wi = transport.point_light(receivers['means'], light, intensity, receivers['visibility'])
    wo = F.normalize(eye - receivers['means'], dim=-1)
    normal = transport.material_normal(receivers)
    latent = receivers['features'][:, :6].sigmoid()
    f, transmission, _ = transport.decoder(latent, wi, wo, normal)
    cosine = (normal * wi).sum(-1, keepdim=True).clamp_min(0)
    fraction = transport.exchange(receivers['features']).sigmoid()
    specular = incident * cosine * f
    diffuse = incident * cosine * transmission * receivers['base'].sigmoid() / math.pi
    direct = specular + diffuse
    nonlocal_rgb = foreground - specular - (1 - fraction) * diffuse
    half = F.normalize(wi + wo, dim=-1)
    aligned_f, _, _ = transport.decoder(latent, wi, wo, half)
    aligned_specular = incident * (half * wi).sum(-1, keepdim=True).clamp_min(0) * aligned_f

    def display(values):
        result = torch.zeros_like(target)
        result[covered] = values.clamp_min(0).pow(1 / gamma) * alpha[covered]
        return result.clamp(0, 1)

    peaks = neutral_peak_mask(target,target_alpha)
    selected = peaks[covered]
    images = {'prediction': display(foreground), 'direct': display(direct),
              'specular_before_gate': display(specular), 'specular_retained': display(specular),
              'diffuse_after_gate': display((1-fraction)*diffuse), 'nonlocal': display(nonlocal_rgb),
              'aligned_normal_specular': display(aligned_specular)}
    images['prediction']=(images['prediction']*255).round()/255
    save_pair(output/(stem+'_pair.png'),images['prediction'],target)
    row = {'neutral_peak_pixels': int(peaks.sum()), 'linear_light_scale': float(transport.light_scale)}
    visible=alpha[...,0][covered]>.9
    row['backfacing_shading_normal_fraction']=float(((normal*wo).sum(-1)<0)[visible].float().mean())
    row['shading_normal_correction_mean_deg']=float(torch.acos((normal[visible]*receivers['normals'][visible]).sum(-1).clamp(-1,1)).mean()*180/math.pi)
    row['neutral_peak_metrics']=neutral_peak_metrics(images['prediction'],target,target_alpha)
    geometry_f,geometry_t,_=transport.decoder(latent,wi,wo,receivers['normals'])
    geometry_cosine=(receivers['normals']*wi).sum(-1,keepdim=True).clamp_min(0)
    without_correction=nonlocal_rgb+incident*geometry_cosine*(geometry_f+(1-fraction)*geometry_t*receivers['base'].sigmoid()/math.pi)
    old_gate=foreground-fraction*specular
    half_vector_probe = foreground.clone()
    half_vector_probe[selected] += aligned_specular[selected]-specular[selected]
    row['counterfactual_neutral_peaks']={
        'geometry_normal_only':neutral_peak_metrics((display(without_correction)*255).round()/255,target,target_alpha),
        'gate_full_direct_response':neutral_peak_metrics((display(old_gate)*255).round()/255,target,target_alpha),
        'half_vector_specular_probe':neutral_peak_metrics((display(half_vector_probe)*255).round()/255,target,target_alpha)}
    row['half_vector_probe_note'] = 'GT-selected pixels only: replace specular normal with half-vector, keep diffuse/nonlocal unchanged; diagnostic, not a usable prediction.'
    if selected.any():
        if transport.material_model == 'ggx':
            row['ggx_at_target_peaks'] = {
                'F0_mean_rgb':latent[selected, :3].mean(0).tolist(),
                'alpha_mean_two_lobes':(.001 * (math.log(1000) * latent[selected, 3:5]).exp()).mean(0).tolist(),
                'first_lobe_weight_mean':float(latent[selected, 5].mean()),
            }
        row['peak_rgb_mean'] = {'target': target[peaks].mean(0).tolist(),
                               **{key: value[peaks].mean(0).tolist() for key,value in images.items()}}
        row['peak_normal_half_angle_deg'] = float(torch.acos((normal[selected]*half[selected]).sum(-1).clamp(-1,1)).mean()*180/math.pi)
        row['peak_geometry_normal_half_angle_deg'] = float(torch.acos((receivers['normals'][selected]*half[selected]).sum(-1).clamp(-1,1)).mean()*180/math.pi)
        codes = receivers['features'][:,6:9]
        if 'normal_residual' in receivers:
            codes = codes+receivers['normal_residual']
        row['peak_normal_offset_saturation_fraction']=float((codes[selected].tanh().abs()>.95).float().mean())
        row['peak_visibility_mean'] = float(receivers['visibility'][selected].mean())
        row['peak_gate_rgb'] = fraction[selected].mean(0).tolist()
        row['peak_linear_specular_mean'] = float(specular[selected].mean())
        row['peak_linear_aligned_specular_mean'] = float(aligned_specular[selected].mean())
    panels = [(target,'GT'),(images['prediction'],'Prediction'),(peaks,'Neutral local peaks'),
              (images['direct'],'Direct, before gate'),(images['specular_before_gate'],'Specular, before gate'),
              (images['specular_retained'],'Specular retained in final sum'),
              (images['diffuse_after_gate'],'Diffuse, after gate'),(images['nonlocal'],'Nonlocal'),
              (images['aligned_normal_specular'],'Normal = half-vector (oracle probe)')]
    fig,axes=plt.subplots(3,3,figsize=(12,12))
    for ax,(value,title) in zip(axes.flat,panels):
        ax.imshow(value.cpu().numpy(),vmin=0,vmax=1)
        ax.set_title(title); ax.axis('off')
    fig.suptitle(stem+' | components gamma-encoded separately, not additive in display space')
    fig.tight_layout(); fig.savefig(output/(stem+'_highlights.png'),dpi=140); plt.close(fig)
    return row


@torch.enable_grad()
def fit_sdf_mask_probe(checkpoint_path, output, steps, mask_weight):
    """CPU-only feasibility probe; does not update Gaussians or scene radiance."""
    from sdf import SurfaceSDF, silhouette_rays, silhouette_logits
    torch.set_num_threads(4)
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if not saved['config'].get('sdf', False):
        raise ValueError('The mask probe requires an already fitted SDF')
    field = SurfaceSDF(saved['gaussians']['center'], saved['radius'],
                       detail=saved['config'].get('sdf_detail', False))
    field.load_state_dict(saved['sdf'])
    dataset = SceneDataset(saved['config']['scene'], 'train', 128)
    fit_indices = saved['fit_indices'][::16]
    check_indices = saved['fit_indices'][8::32]
    assert not set(fit_indices) & set(check_indices)
    fit_rays = [silhouette_rays(dataset[i], field) for i in fit_indices]
    check_samples = [dataset[i] for i in check_indices]
    check_rays = [silhouette_rays(s, field) for s in check_samples]
    config = {'source_checkpoint':str(Path(checkpoint_path).resolve()), 'device':'cpu',
              'steps':steps, 'mask_weight':mask_weight, 'eikonal_weight':.1,
              'learning_rate':.0001, 'seed':0, 'resolution':128, 'rays_per_step':64,
              'samples_per_ray':128, 'softness_radius':.005,
              'fit_indices':fit_indices, 'check_indices':check_indices,
              'note':'Check views were excluded from this mask refinement, but participated in the original scene training. No official test views.'}
    (output/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    root = Path(__file__).resolve().parent
    with tarfile.open(output/'source.tar', 'w') as archive:
        for path in [*root.glob('*.py'), root/'methods', root/'materials']:
            archive.add(path, arcname=path.name, filter=lambda item:None if '__pycache__' in item.name else item)

    @torch.no_grad()
    def measure():
        rows, predictions = [], []
        for index, rays in zip(check_indices, check_rays):
            prediction = torch.zeros_like(rays['target'], dtype=torch.bool)
            for ids in rays['valid'].split(256):
                prediction[ids] = silhouette_logits(field, rays, ids) > 0
            target = rays['target'] > .5
            intersection = (prediction & target).sum()
            rows.append({'frame':index, 'iou':float(intersection/(prediction | target).sum()),
                         'precision':float(intersection/prediction.sum().clamp_min(1)),
                         'recall':float(intersection/target.sum())})
            predictions.append(prediction.reshape(rays['shape']))
        return {'rows':rows, 'mean':{k:sum(row[k] for row in rows)/len(rows)
                                    for k in ['iou', 'precision', 'recall']}}, predictions

    start = time.monotonic()
    before, before_images = measure()
    optimizer = torch.optim.Adam(field.parameters(), lr=.0001)
    generator = torch.Generator().manual_seed(0)
    history = []
    for step in range(1, steps+1):
        rays = fit_rays[int(torch.randint(len(fit_rays), (), generator=generator))]
        ids = torch.cat([pool[torch.randint(len(pool), (32,), generator=generator)]
                         for pool in (rays['boundary'], rays['valid'])])
        logits = silhouette_logits(field, rays, ids)
        mask_loss = F.binary_cross_entropy_with_logits(logits, rays['target'][ids])
        points = (torch.rand((128, 3), generator=generator)*2-1).requires_grad_()
        gradient = torch.autograd.grad(field(points).sum(), points, create_graph=True)[0]
        eikonal = (gradient.norm(dim=-1)-1).square().mean()
        loss = mask_weight*mask_loss+.1*eikonal
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Non-finite SDF mask probe at {step}')
        optimizer.zero_grad(set_to_none=True)
        loss.backward(); optimizer.step()
        if step == 1 or step % 100 == 0 or step == steps:
            row = {'step':step, 'mask_loss':float(mask_loss), 'eikonal':float(eikonal),
                   'seconds':time.monotonic()-start}
            history.append(row); print(json.dumps(row), flush=True)
    after, after_images = measure()
    result = {'before':before, 'after':after, 'seconds':time.monotonic()-start,
              'scope':'SDF silhouettes only; Gaussians/materials/radiance unchanged; no relighting quality claim'}
    (output/'metrics.json').write_text(json.dumps(result, indent=2)+'\n')
    (output/'history.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in history))
    torch.save({'sdf':field.state_dict(), 'config':config}, output/'field.pt')
    fig, axes = plt.subplots(4, 3, figsize=(9, 11))
    for axes_row, index in zip(axes, torch.linspace(0, len(check_rays)-1, 4).long().tolist()):
        for ax, values, title in zip(axes_row,
                [check_samples[index]['alpha'][..., 0], before_images[index], after_images[index]],
                ['GT', 'Initial SDF', 'Refined SDF']):
            ax.imshow(values, cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'{title} | train {check_indices[index]}'); ax.axis('off')
    fig.tight_layout(); fig.savefig(output/'comparison.png', dpi=130); plt.close(fig)
    print(json.dumps({'before':before['mean'], 'after':after['mean']}), flush=True)


def _compare_cue_reference(current, reference, path=''):
    """Exact discrete/GT fields, bounded independent-render floating error."""
    differences = {}
    if isinstance(current, dict):
        assert current.keys() == reference.keys(), path
        for key in current:
            differences.update(_compare_cue_reference(current[key], reference[key], path+'.'+key))
    elif isinstance(current, float) and path.rsplit('.', 1)[-1] not in (
            'alpha_L1', 'target_contrast', 'target_contrast_sum'):
        assert math.isclose(current, reference, rel_tol=1e-6, abs_tol=1e-8), (path, current, reference)
        if current != reference:
            differences[path] = {'current':current, 'reference':reference, 'difference':current-reference}
    else:
        assert current == reference, (path, current, reference)
    return differences


def _plot_angular_cue_crops(output, crops, images):
    import numpy as np
    columns = [('target', 'GT'), ('prediction', 'Prediction'),
               ('geometry_angle', 'Geometry angle / deg'),
               ('narrow2', 'Geometry kernel 2 deg'), ('narrow32', 'Geometry kernel 32 deg'),
               ('wide128', 'Geometry kernel 128 deg'), ('material_angle', 'Material angle / deg')]
    selected = [crop for crop in crops if crop['frame_index'] in images]
    if not selected:
        return
    fig, axes = plt.subplots(len(selected), len(columns), figsize=(18, 2.7*len(selected)), squeeze=False)
    for row, crop in enumerate(selected):
        x0,y0,x1,y1 = crop['bounds_xyxy']; frame = crop['frame_index']
        data = images[frame]
        for col,(key,title) in enumerate(columns):
            ax = axes[row,col]
            if key.startswith(('narrow','wide')):
                tau = float(key.replace('narrow','').replace('wide',''))*np.pi/180
                value = np.exp(-data['geometry_angle']**2/(2*tau*tau))
                ax.imshow(value[y0:y1,x0:x1], vmin=0, vmax=1, cmap='magma', interpolation='nearest')
            elif key.endswith('_angle'):
                ax.imshow(data[key][y0:y1,x0:x1]*180/np.pi, vmin=0, vmax=90,
                          cmap='viridis_r', interpolation='nearest')
            else:
                ax.imshow(data[key][y0:y1,x0:x1], interpolation='nearest')
            if row == 0:
                ax.set_title(title)
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(f"Frame {frame}{crop['marker']}\nGT area {crop['gt_component_area']}")
    fig.suptitle('Fixed GT crops | Raw angular cues, not rendered radiance | Same color range per column; angles saturate at 90 deg')
    fig.tight_layout(); fig.savefig(output/'cue_crops.png', dpi=140); plt.close(fig)


@torch.no_grad()
def audit_angular_cues(args, output):
    """Read actual frozen receivers once; compare local cues on training images."""
    import os
    import subprocess
    import sys
    import numpy as np
    from PIL import Image
    from angular_cues import analyze_angular_cues, aggregate_angular_cues
    from evaluate import (load_radiance_residual, observation_image,
                          neutral_peak_components, aggregate_neutral_peak_components)

    checkpoint_path = Path(args.checkpoint).resolve()
    reference_path = args.reference_metrics.resolve()
    reference = json.loads(reference_path.read_text())
    assert reference['split'] == 'fit' and reference['limit'] == 0
    assert Path(reference['checkpoint']).resolve() == checkpoint_path
    assert reference['render_branch'] == 'gaussian_residual'
    resolution = reference['resolution'] if args.resolution is None else args.resolution
    assert resolution == reference['resolution']
    crops = json.loads(args.cue_crops.read_text())['four_saved_fit_diagnostics']['crops']
    g, transport, saved = load_model(checkpoint_path)
    cfg = saved['config']
    assert cfg['representation'] == 'neural_material'
    assert not cfg.get('sdf_shading') and not cfg.get('normal_field')
    head = load_radiance_residual(saved)
    assert head is not None
    g.eval(); transport.eval(); head.eval()
    dataset = SceneDataset(cfg['scene'], 'train', resolution, unit_light_intensity=cfg['unit_light_intensity'])
    fit = saved['fit_indices']
    assert reference['evaluated_frames'] == len(fit)
    assert [row['frame_index'] for row in reference['views']] == fit
    selected = fit if args.limit == 0 else fit[::max(1,len(fit)//args.limit)][:args.limit]
    camera_indices = {index:local for local,index in enumerate(fit)}
    offsets = None
    if saved['camera_offsets'] is not None:
        offsets = load_camera_offsets(saved, g.radius).cuda()
    assert reference['corrected_fit_frames'] == (len(fit) if offsets is not None else 0)
    reference_rows = {row['frame_index']:row for row in reference['views']}
    reference_pairs = {row['frame_index']:ordinal for ordinal,row in enumerate(reference['views'][:4])}
    protocol = {
        'checkpoint':str(checkpoint_path), 'reference_metrics':str(reference_path),
        'crop_source':str(args.cue_crops.resolve()), 'scope':'Training-fit cue localization, no optimization or test evaluation; not geometry truth or generalization.',
        'split':'train', 'resolution':resolution, 'fit_indices':fit, 'evaluated_indices':selected,
        'corrected_cameras':offsets is not None, 'source_full_fit_count':len(fit),
        'primary_normal':'geometry', 'material_normal':'descriptive only; not selected by screening',
        'rays':'Actual renderer transport receivers and residual-head query inputs; ascending GS-alpha-positive pixels.',
        'raw_cues':'Unit-peak isotropic half-vector kernels, no irradiance factor, no learned rotation or GT-conditioned correction.',
        'reference_checks':'Every frame PSNR/SSIM/raw_MSE/alpha/residual stats/global peaks/components; no repeated LPIPS. Exact GT/count/alpha, other floats rtol1e-6 atol1e-8.',
        'no_state_updates':True, 'command':sys.argv, 'python':sys.executable,
        'torch':torch.__version__, 'gpu':torch.cuda.get_device_name(0),
        'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'source_revision':subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
    }
    (output/'config.json').write_text(json.dumps(protocol,indent=2)+'\n')
    project = Path(__file__).resolve().parent
    with tarfile.open(output/'source.tar','w') as archive:
        for path in [*project.glob('*.py'),project/'methods',project/'materials']:
            archive.add(path,arcname=path.name,filter=lambda item:None if '__pycache__' in item.name else item)
    crop_frames = {crop['frame_index'] for crop in crops}
    frame_reports, checks, images, component_metrics = [], [], {}, []
    start = time.monotonic()
    with (output/'components.jsonl').open('w') as raw:
        for ordinal,index in enumerate(selected):
            sample = to_device(dataset[index], 'cuda')
            if offsets is not None:
                sample = offsets.correct(sample,camera_indices[index])
            capture, chunks, query_points, query_normals = {}, [], [], []
            def base_hook(module,inputs,result):
                assert not capture
                capture.update(receivers=inputs[1],base=result.detach())
            def head_hook(module,inputs,result):
                chunks.append((result[0] if isinstance(result,tuple) else result).detach())
                query_points.append(inputs[0].detach())
                query_normals.append(inputs[1].detach())
            hooks = [transport.register_forward_hook(base_hook),head.register_forward_hook(head_hook)]
            try:
                prediction,alpha,info = render_observation(g,transport,sample,cfg['background'],
                    saved['step'] >= cfg['shadow_start'],saved['step'] >= cfg['port_start'],
                    cfg['display_gamma'],cfg['shadow_mode'],radiance_residual=head)
            finally:
                for hook in hooks: hook.remove()
            covered = alpha[...,0] > 0
            receivers = capture['receivers']
            assert torch.equal(receivers['means'],torch.cat(query_points))
            normals = {'geometry':receivers['normals'],
                       'material':transport.material_normal(receivers)}
            assert torch.equal(torch.cat(query_normals),normals[head.normal_source])
            base,delta = capture['base'],torch.cat(chunks)
            assert len(base) == int(covered.sum()) == len(delta)
            foreground = torch.zeros_like(prediction)
            foreground[covered] = (base+delta).clamp_min(0)
            reconstructed = observation_image(foreground*alpha+cfg['background']*(1-alpha),
                cfg['display_gamma'],alpha=None if sample['is_hdr'] else alpha,background=cfg['background'])
            assert torch.equal(prediction,reconstructed)
            target = target_image(sample,cfg['background'],cfg['display_gamma'])
            qpred,qtarget = [(v.clamp(0,1)*255).round()/255 for v in (prediction,target)]
            cpred,ctarget = [(v*255).round().byte().cpu().float()/255 for v in (qpred,qtarget)]
            calpha = sample['alpha'].cpu()
            current = {'frame_index':index,'name':sample['name'],
                'PSNR':float(-10*torch.log10((qpred-qtarget).square().mean())),
                'SSIM':float(ssim(qpred,qtarget)), 'raw_MSE':float((prediction-target).square().mean()),
                'alpha_L1':float((alpha-sample['alpha']).abs().mean()),
                'residual_stats':{key:value.item() if isinstance(value,torch.Tensor) else value
                                  for key,value in info['residual_stats'].items()},
                'neutral_peak_metrics':neutral_peak_metrics(qpred,qtarget,sample['alpha']),
                'neutral_peak_components':neutral_peak_components(cpred,ctarget,calpha)}
            difference = _compare_cue_reference(current,{key:reference_rows[index][key] for key in current})
            pair_equal = None
            if index in reference_pairs:
                pair = np.asarray(Image.open(reference_path.parent/f"pair_{reference_pairs[index]:03d}.png"))
                assert np.array_equal(pair,torch.cat((ctarget,cpred),1).mul(255).round().byte().numpy())
                pair_equal = True
            wi = F.normalize(sample['light_pos']-receivers['means'],dim=-1)
            wo = F.normalize(sample['c2w'][:3,3]-receivers['means'],dim=-1)
            half = F.normalize(wi+wo,dim=-1)
            angle_maps = {}
            for name,normal in normals.items():
                angle = (normal*half).sum(-1).clamp(-1,1).acos()
                assert torch.isfinite(angle).all()
                image = alpha[...,0].new_full(covered.shape,float('nan'))
                image[covered] = angle
                angle_maps[name] = image.cpu()
            report = analyze_angular_cues(cpred,ctarget,calpha[...,0],alpha[...,0].cpu(),angle_maps)
            tiny = current['neutral_peak_components']['buckets']['1_to_4']['sums']
            for key,refkey in [('gt_small_peak_components','component_count'),('gt_small_peak_pixels','gt_peak_pixels'),
                               ('matched_gt_pixels_2px','matched_gt_pixels_2px'),
                               ('components_with_at_least_one_hit','components_with_at_least_one_hit')]:
                assert report['counts'][key] == tiny[refkey], (index,key,report['counts'][key],tiny[refkey])
            report['frame_index'] = index
            for row in report['components']:
                row['frame_index'] = index
                raw.write(json.dumps(row,allow_nan=False)+'\n')
            raw.flush()
            frame_reports.append(report)
            component_metrics.append(current['neutral_peak_components'])
            checks.append({'frame_index':index,'name':sample['name'],
                'camera_offset_local_index':camera_indices[index] if offsets is not None else None,
                'reference_metric_differences':difference,'reference_pair_uint8_equal':pair_equal,
                'actual_head_points_normals_match_receivers':True,'observation_reconstruction_bitwise_equal':True,
                'counts':report['counts'],
                'normal_norm_minmax':{name:[float(n.norm(dim=-1).min()),float(n.norm(dim=-1).max())]
                                      for name,n in normals.items()}})
            if index in crop_frames:
                data = {'target':ctarget.numpy(),'prediction':cpred.numpy(),
                        **{name+'_angle':value.numpy() for name,value in angle_maps.items()}}
                images[index] = data
                np.savez_compressed(output/f'frame_{index:03d}.npz',**data,
                                    gt_alpha=calpha[...,0].numpy(),gs_alpha=alpha[...,0].cpu().numpy())
            if ordinal == 0 or (ordinal+1)%25 == 0 or ordinal+1 == len(selected):
                print(json.dumps({'completed_frames':ordinal+1,'total_frames':len(selected),
                    'frame_index':index,'seconds':round(time.monotonic()-start,2),
                    'small_components':report['counts']['gt_small_peak_components']}),flush=True)
    combined = aggregate_angular_cues(frame_reports)
    pooled_components = aggregate_neutral_peak_components(component_metrics)
    pooled_reference_differences = (_compare_cue_reference(
        pooled_components,reference['metrics']['neutral_peak_components'],'pooled_components')
        if selected == fit else None)
    assert g.radius == saved['radius']
    for module,state in [(g,saved['gaussians']),(transport,saved['transport']),
                          (head,saved['radiance_residual'])]+(
                          [(offsets,saved['camera_offsets'])] if offsets is not None else []):
        assert module.state_dict().keys() == state.keys()
        for key,value in module.state_dict().items():
            reference_value = state[key]
            assert value.dtype == reference_value.dtype and value.shape == reference_value.shape
            assert torch.equal(value.contiguous().reshape(-1).view(torch.uint8),
                               reference_value.contiguous().reshape(-1).view(torch.uint8)), key
    _plot_angular_cue_crops(output,crops,images)
    result = {key:value for key,value in combined.items() if key != 'components'}
    result.update(protocol=protocol,frames=checks,neutral_peak_components=pooled_components,
                  seconds=round(time.monotonic()-start,2),all_model_states_bitwise_unchanged=True,
                  actual_render_reconstruction_and_reference_checks_passed=True,complete=True,
                  raw_components='components.jsonl',cropped_frames=sorted(images),
                  pooled_reference_differences=pooled_reference_differences)
    (output/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'complete':True,'counts':result['counts'],'screen':result['screen'],
                      'output':str(output)}),flush=True)


@torch.no_grad()
@torch.no_grad()
def preview_relighting(checkpoint, output):
    """Fixed-camera preview using the ordinary PORT evaluator's rendering path."""
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    from evaluate import load_model,render_observation
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    g,t,saved=load_model(checkpoint);cfg=saved['config']
    if cfg['representation'] not in ('directional_port_v1','neural_material'):
        raise ValueError('This preview supports the directional and neural-material pipelines')
    data=SceneDataset(cfg['scene'],'test',cfg['resolution']);camera=to_device(data[0],'cuda')
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    fig,axes=plt.subplots(1,5,figsize=(15,3))
    axes[0].imshow(target_image(camera,cfg['background'],cfg['display_gamma']).cpu())
    axes[0].set_title('GT: camera0 / light0',fontsize=9);rows=[]
    for ax,index in zip(axes[1:],np.linspace(0,len(data)-1,4,dtype=int).tolist()):
        light=to_device(data[index],'cuda')
        sample=dict(camera,light_pos=light['light_pos'],light_intensity=light['light_intensity'])
        raw,_,_=render_observation(g,t,sample,cfg['background'],saved['step']>=cfg['shadow_start'],
            saved['step']>=cfg['port_start'],cfg['display_gamma'],shadow_mode=cfg['shadow_mode'])
        rgb=raw.clamp(0,1).cpu();Image.fromarray((rgb*255).round().byte().numpy()).save(out/f'light_{index:03}.png')
        ax.imshow(rgb);ax.set_title(f'camera0 / light{index}',fontsize=9)
        rows.append(dict(light_frame=index,light_position_world=sample['light_pos'].tolist(),
            light_intensity=sample['light_intensity'].tolist(),paired_GT_available=index==0,
            clipped_channel_fraction=float((raw>1).float().mean())))
    for ax in axes:ax.axis('off')
    fig.tight_layout();fig.savefig(out/'relighting.png',dpi=150);plt.close(fig)
    (out/'preview.json').write_text(json.dumps(dict(checkpoint=str(Path(checkpoint).resolve()),
        camera_frame=0,camera=data.frames[0],lights=rows,matmul_tf32=False,
        scope='Fixed camera and four dataset lights. Onlylight0 has pairedGT; other combinations are qualitative, not scored accuracy.'),indent=2)+'\n')


def fit_free_colors(checkpoint, output, frames, steps):
    """Fit independent nonnegative Gaussian colors per fit view; never a relighting result.

    A fixed-geometry projection is reused. Diagonally majorized FISTA first fits
    linear radiance; projected Adam then fits the observed image. Display metrics
    are diagnostic and are not a certificate of global optimality.
    """
    from gsplat.cuda._wrapper import rasterize_to_pixels
    from evaluate import observation_image
    torch.set_num_threads(8)
    g,t,saved = load_model(checkpoint)
    cfg=saved['config']
    if g.geometry != '3dgs' or cfg['representation'] != 'light_atlas':
        raise ValueError('Free-color capacity diagnosis requires a 3D LiSA checkpoint')
    if not set(frames).issubset(saved['fit_indices']):
        raise ValueError('Free-color fitting may use only source fit frames')
    g.requires_grad_(False);t.requires_grad_(False);t.eval()
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    dataset=SceneDataset(cfg['scene'],'train',cfg['resolution'],cfg['unit_light_intensity'])
    cameras=load_camera_offsets(saved,g.radius).cuda() if saved['camera_offsets'] is not None else None
    camera_indices={index:local for local,index in enumerate(saved['fit_indices'])}
    results=[]
    def metrics(predicted,target):
        x=(predicted.clamp(0,1)*255).round()/255
        y=(target.clamp(0,1)*255).round()/255
        return dict(PSNR=float(-10*torch.log10((x-y).square().mean())),SSIM=float(ssim(x,y)))
    for index in frames:
        started=time.monotonic()
        sample=to_device(dataset[index],'cuda')
        if cameras is not None:
            sample=cameras.correct(sample,camera_indices[index])
        target=target_image(sample,cfg['background'],cfg['display_gamma'])
        linear_target=(sample['image'] if sample['is_hdr'] else
                       sample['image'].clamp_min(0).pow(cfg['display_gamma']))
        if sample['alpha'] is not None:
            linear_target=linear_target*sample['alpha']+cfg['background']*(1-sample['alpha'])
        h,w=sample['image'].shape[:2]
        with torch.no_grad():
            native,_,_=render_observation(g,t,sample,cfg['background'],saved['step']>=cfg['shadow_start'],
                saved['step']>=cfg['port_start'],cfg['display_gamma'],cfg['shadow_mode'])
            initial=g.params['means'].new_full((len(g.params['means']),3),.2)
            reference,alpha,info=_rasterize('3dgs',**g.raster_inputs(),colors=initial,
                viewmats=sample['viewmat'][None],Ks=sample['K'][None],width=w,height=h,packed=False)
            alpha=alpha[0]
        def splat(colors):
            rgb,a=rasterize_to_pixels(info['means2d'],info['conics'],colors[None],info['opacities'],
                w,h,info['tile_size'],info['isect_offsets'],info['flatten_ids'],packed=False)
            return rgb[0],a[0]
        with torch.no_grad():
            check,check_alpha=splat(initial)
            torch.testing.assert_close(check,reference[0],rtol=0,atol=0)
            torch.testing.assert_close(check_alpha,alpha,rtol=0,atol=0)
        probe=initial.new_zeros((len(initial),1),requires_grad=True)
        mass=torch.autograd.grad(splat(probe)[0].sum(),probe)[0].detach().clamp_min(1e-12)
        background=cfg['background']*(1-alpha)
        def display(rgb):
            return observation_image(rgb,cfg['display_gamma'],
                alpha=None if sample['is_hdr'] else alpha,background=cfg['background'])
        colors=initial;extrapolated=colors;momentum=1.;trace=[];best_psnr=-math.inf;best_image=None;best_colors=None
        linear_steps=steps//2
        for step in range(1,linear_steps+1):
            variable=extrapolated.detach().requires_grad_()
            rgb=splat(variable)[0]+background
            gradient=torch.autograd.grad(.5*(rgb-linear_target).square().sum(),variable)[0]
            with torch.no_grad():
                updated=(variable-gradient/mass).clamp_min(0)
                next_momentum=.5*(1+math.sqrt(1+4*momentum*momentum))
                extrapolated=updated+(momentum-1)/next_momentum*(updated-colors)
                colors=updated;momentum=next_momentum
                if step==1 or step%100==0 or step==linear_steps:
                    fitted=splat(colors)[0]+background
                    observed=display(fitted)
                    item=dict(step=step,stage='linear_nnls',linear_MSE=float((fitted-linear_target).square().mean()),
                              **metrics(observed,target))
                    trace.append(item)
                    if item['PSNR']>best_psnr:
                        best_psnr=item['PSNR'];best_image=observed.clone();best_colors=colors.clone()
        free=torch.nn.Parameter(best_colors)
        optimizer=torch.optim.Adam([free],lr=.03,eps=1e-15)
        for step in range(linear_steps+1,steps+1):
            optimizer.zero_grad(set_to_none=True)
            observed=display(splat(free)[0]+background)
            (observed-target.clamp(0,1)).square().mean().backward()
            optimizer.step()
            with torch.no_grad():
                free.clamp_(min=0)
                if step%100==0 or step==steps:
                    fitted=splat(free)[0]+background;observed=display(fitted)
                    item=dict(step=step,stage='observation_adam',linear_MSE=float((fitted-linear_target).square().mean()),
                              **metrics(observed,target))
                    trace.append(item)
                    if item['PSNR']>best_psnr:
                        best_psnr=item['PSNR'];best_image=observed.clone()
        save_pair(out/f'frame_{index:04d}_native.png',native,target)
        save_pair(out/f'frame_{index:04d}_free_colors.png',best_image,target)
        row=dict(frame=index,native=metrics(native,target),best_sampled_free_color=metrics(best_image,target),
                 trace=trace,seconds=time.monotonic()-started,projection_matches_native=True,
                 geometry_frozen=True,gaussians=len(colors))
        results.append(row)
        print(json.dumps({k:v for k,v in row.items() if k!='trace'}),flush=True)
    report=dict(checkpoint=str(Path(checkpoint).resolve()),scope='Per-fit-view independent nonnegative colors; '
        'fixed geometry, no shared BRDF/lighting. These images are fitting-capacity diagnostics, not novel-view or '
        'relighting results. Finite iterations do not prove a global display-PSNR ceiling.',
        optimization='First half diagonal-majorized FISTA NNLS in linear radiance; second half projected Adam in observed RGB.',
        steps=steps,results=results)
    (out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')


@torch.no_grad()
def preview_light_atlas(checkpoint, output, split, frames):
    """Export actual LiSA buffers and independently verify the shading decomposition."""
    import sys
    import numpy as np
    from evaluate import observation_image

    torch.set_num_threads(8)
    g, t, saved = load_model(checkpoint)
    cfg = saved['config']
    if cfg['representation'] != 'light_atlas':
        raise ValueError('Light-atlas preview requires a light_atlas checkpoint')
    t.eval()
    dataset = SceneDataset(cfg['scene'], split, cfg['resolution'], cfg['unit_light_intensity'])
    shadow = saved['step'] >= cfg['shadow_start']
    port_active = saved['step'] >= cfg['port_start']
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    report = dict(checkpoint=str(Path(checkpoint).resolve()), scene=cfg['scene'], split=split,
                  material_head=t.material_head, shading=t.shading, step=saved['step'], command=sys.argv,
                  torch_version=torch.__version__, cuda_version=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(), seed=cfg['seed'], shadow=shadow, port_active=port_active,
                  flux_note='Learned channels, not RGB or calibrated irradiance; no PCA used.',
                  depth_note='Normalized light-camera depth; uncovered texels masked.',
                  display_note='Radiance components gamma-encoded separately; only linear components add.',
                  radiance_share_protocol='Alpha-weighted linear foreground radiance summed over camera pixels.',
                  frames=[])
    camera_offsets = (load_camera_offsets(saved, g.radius).cuda()
                      if split == 'train' and saved['camera_offsets'] is not None else None)
    camera_indices = {index: local for local, index in enumerate(saved['fit_indices'])}
    for index in frames:
        sample = to_device(dataset[index], 'cuda')
        if camera_offsets is not None and index in camera_indices:
            sample = camera_offsets.correct(sample, camera_indices[index])
        capture = {}
        def capture_receivers(module, inputs):
            capture['receivers'] = inputs[1]
        hook = t.register_forward_pre_hook(capture_receivers)
        try:
            predicted, alpha, _ = render_observation(
                g, t, sample, cfg['background'], shadow, port_active, cfg['display_gamma'], cfg['shadow_mode'])
        finally:
            hook.remove()
        receivers = capture['receivers']
        points = receivers['means']
        wi = F.normalize(sample['light_pos'] - points, dim=-1)
        wo = F.normalize(sample['c2w'][:3, 3] - points, dim=-1)
        half = F.normalize(wi + wo, dim=-1)
        rho, cosines = t.local_response(receivers, receivers['normals'], wi, wo, half,
                                        (points - g.center) / g.radius)
        atlas = t.light_atlas(g, sample['light_pos'])
        stats, fluxes = t.gather(atlas, points, g.radius)
        visibility = torch.ones_like(cosines[:, :1])
        # Optional specular lobes (2026-10-04) use the geometric level-0 moment test, as in forward.
        specular_visibility = torch.ones_like(cosines[:, :1])
        if shadow and t.visibility_model != 'none':
            visibility = (receivers['visibility'][:, None] if t.per_gaussian_visibility else
                          t.receiver_visibility(stats, receivers['features'], cosines[:, :1]))
            specular_visibility = (visibility if t.per_gaussian_visibility else stats[:, t.STATS - 1:t.STATS])
        specular = (t.specular_response(receivers, cosines, (points - g.center) / g.radius)
                    if t.specular != 'none' else torch.zeros_like(rho))
        transfer = (t.transfer(stats, fluxes, receivers['features'], cosines)
                    if port_active and t.light_transport == 'atlas' else torch.zeros_like(rho))
        incident = sample['light_intensity'][None] / t.light_scale / (
            sample['light_pos'] - points).square().sum(-1, keepdim=True)
        covered = alpha[..., 0] > 0
        local_linear = torch.zeros_like(predicted)
        specular_linear = torch.zeros_like(predicted)
        transfer_linear = torch.zeros_like(predicted)
        visibility_image = predicted.new_full(covered.shape, float('nan'))
        if t.shading == 'gaussian':
            # Composite each contribution with the exact same geometry and order
            # as the RGB renderer, then normalize to the preview's foreground convention.
            colors = torch.cat((incident * visibility * rho, incident * specular_visibility * specular,
                                incident * transfer, visibility), -1)
            components, _, _ = _rasterize('3dgs', **g.raster_inputs(), colors=colors,
                viewmats=sample['viewmat'][None], Ks=sample['K'][None],
                width=predicted.shape[1], height=predicted.shape[0], packed=False)
            values = components[0][covered] / alpha[covered]
            local_linear[covered], specular_linear[covered], transfer_linear[covered] = values[:, :3], values[:, 3:6], values[:, 6:9]
            visibility_image[covered] = values[:, 9]
        else:
            local_linear[covered] = incident * visibility * rho
            specular_linear[covered] = incident * specular_visibility * specular
            transfer_linear[covered] = incident * transfer
            visibility_image[covered] = visibility[:, 0]
        reconstructed = observation_image(
            (local_linear + specular_linear + transfer_linear) * alpha + cfg['background'] * (1 - alpha),
            cfg['display_gamma'], alpha=None if sample['is_hdr'] else alpha, background=cfg['background'])
        torch.testing.assert_close(reconstructed, predicted, rtol=1e-5, atol=1e-6)
        for tensor in atlas['levels'] + [visibility, local_linear, specular_linear, transfer_linear]:
            assert torch.isfinite(tensor).all()
        stem = f'frame_{index:03d}'
        levels = [level[0].cpu().numpy() for level in atlas['levels']]
        raw = levels[0]
        c = t.flux_dim
        coverage = raw[c + 2]
        support = coverage > 0.01
        mean = raw[c] / np.maximum(coverage, 1e-4)
        spread = np.sqrt(np.maximum(raw[c + 1] / np.maximum(coverage, 1e-4) - mean**2, 0))
        depth = np.ma.masked_where(~support, mean)
        depth_spread = np.ma.masked_where(~support, spread)

        fig, axes = plt.subplots(2, 6, figsize=(18, 6.3))
        panels = [(coverage, 'Coverage M0 (light view)', 'gray', 0, 1),
                  (depth, 'Mean depth M1 / M0', 'viridis', *np.quantile(mean[support], [.01, .99])),
                  (depth_spread, 'Depth standard deviation', 'magma', 0, np.quantile(spread[support], .99))]
        ranges = []
        for channel in range(c):
            upper = float(np.quantile(raw[channel][support], .99))
            ranges.append([0, upper])
            panels.append((raw[channel], f'Learned flux feature {channel}', 'inferno', 0, upper))
        for ax, (data, title, cmap, low, high) in zip(axes.flat, panels):
            artist = ax.imshow(data, cmap=cmap, vmin=low, vmax=high)
            ax.set_title(title, fontsize=11)
            ax.axis('off')
            fig.colorbar(artist, ax=ax, fraction=.035, pad=.025)
        for ax in list(axes.flat)[len(panels):]:
            ax.axis('off')
        fig.suptitle(f"{Path(cfg['scene']).name} #{index} | LiSA {t.material_head} | actual {t.atlas_resolution}x{t.atlas_resolution} light-space buffers\n"
                     'Feature heatmaps use separate 99th-percentile display ranges; colors are not object RGB.', fontsize=13)
        fig.tight_layout()
        fig.savefig(out / (stem + '_atlas.png'), dpi=140)
        plt.close(fig)

        fig, axes = plt.subplots(2, len(levels), figsize=(17.5, 5.2))
        for k, level in enumerate(levels):
            axes[0, k].imshow(level[c + 2], cmap='gray', vmin=0, vmax=1, interpolation='nearest')
            axes[1, k].imshow(level[0], cmap='inferno', vmin=0, vmax=ranges[0][1], interpolation='nearest')
            axes[0, k].set_title(f'L{k}: {level.shape[-1]}x{level.shape[-2]}')
            axes[0, k].axis('off'); axes[1, k].axis('off')
        fig.suptitle('Actual atlas pyramid | top: coverage | bottom: flux feature 0 (same color range)')
        fig.tight_layout()
        fig.savefig(out / (stem + '_pyramid.png'), dpi=140)
        plt.close(fig)

        def component_display(values):
            return observation_image(values * alpha, cfg['display_gamma'],
                                     alpha=None if sample['is_hdr'] else alpha, background=0.0).clamp(0, 1).cpu()
        target = target_image(sample, cfg['background'], cfg['display_gamma']).clamp(0, 1).cpu()
        images = [target, predicted.clamp(0, 1).cpu(), visibility_image.cpu(),
                  component_display(local_linear), component_display(transfer_linear)]
        titles = ['GT (camera view)', f'Final LiSA ({t.material_head})', 'Visibility V: black=shadow',
                  'Local contribution E * V * rho', 'Transport contribution E * transfer']
        if t.specular != 'none':
            images.insert(4, component_display(specular_linear))
            titles.insert(4, 'Specular lobes E * V0 * s')
        fig, axes = plt.subplots(1, len(images), figsize=(3.4 * len(images), 3.8))
        for ax, data, title in zip(axes, images, titles):
            ax.imshow(data, cmap='gray', vmin=0, vmax=1)
            ax.set_title(title, fontsize=10); ax.axis('off')
        fig.suptitle(f"{Path(cfg['scene']).name} #{index} | camera view | components encoded separately, not additive in display space")
        fig.tight_layout()
        fig.savefig(out / (stem + '_decomposition.png'), dpi=140)
        plt.close(fig)
        save_pair(out / (stem + '_pair.png'), predicted, target.to(predicted.device))
        np.savez_compressed(out / (stem + '_buffers.npz'),
                            **{f'atlas_level_{k}': level for k, level in enumerate(levels)},
                            visibility=visibility_image.cpu().numpy(),
                            local_linear=local_linear.cpu().numpy(), specular_linear=specular_linear.cpu().numpy(),
                            transfer_linear=transfer_linear.cpu().numpy(),
                            alpha=alpha.cpu().numpy(), light_view=atlas['view'].cpu().numpy(),
                            focal=float(atlas['focal']), depth_center=float(atlas['depth_center']), radius=g.radius)
        error = ((predicted.clamp(0, 1) * 255).round() / 255 - (target.to(predicted.device) * 255).round() / 255).square().mean(-1)
        gt_mask = sample['alpha'][..., 0] > .5
        inner = -F.max_pool2d(-gt_mask[None, None].float(), 5, 1, 2)[0, 0]
        outer = F.max_pool2d(gt_mask[None, None].float(), 5, 1, 2)[0, 0]
        regions = dict(interior=inner, boundary=outer-inner, background=1-outer)
        report['frames'].append(dict(frame_index=index, name=sample['name'],
            light_position=sample['light_pos'].tolist(), intensity=sample['light_intensity'].tolist(),
            PSNR=float(-10 * error.mean().log10()), alpha_L1=float((alpha-sample['alpha']).abs().mean()),
            error_share={key:float((error*mask).sum()/error.sum()) for key,mask in regions.items()},
            channel_display_ranges=ranges, atlas_shapes=[list(x.shape) for x in levels],
            decomposition_max_abs_error=float((reconstructed - predicted).abs().max()),
            visibility_mean=float(visibility.mean()), transport_share_linear=float(
                (transfer_linear * alpha).sum() / ((local_linear + specular_linear + transfer_linear) * alpha).sum()),
            specular_share_linear=float(
                (specular_linear * alpha).sum() / ((local_linear + specular_linear + transfer_linear) * alpha).sum())))
        print(json.dumps(report['frames'][-1]), flush=True)
    (out / 'preview.json').write_text(json.dumps(report, indent=2) + '\n')


def export_initial_support(reference_checkpoint, output, mode):
    """Compare camera-cube seeding with training-silhouette-supported seeding.

    Only the reference split, camera normalization and seed are read. No fitted
    positions, material, test frame or external reconstruction is transferred.
    Both outputs use the existing geometry importer to keep their feature/network
    initialization streams identical in the controlled comparison.
    """

    torch.set_num_threads(8)
    saved = torch.load(reference_checkpoint,map_location='cpu',weights_only=False)
    cfg = saved['config']
    center = saved['gaussians']['center'].cuda()
    radius, count = saved['radius'], cfg['points']
    generator = torch.Generator(device='cuda').manual_seed(cfg['seed'])
    fit = saved['fit_indices']
    frames = fit[::max(1,len(fit)//32)][:32]
    metadata = dict(mode=mode,reference=str(Path(reference_checkpoint).resolve()),scene=cfg['scene'],
                    seed=cfg['seed'],points=count,radius=radius,fit_frames=frames,
                    scope='Training-camera/mask initialization only; no fitted geometry or official test input')
    if mode == 'camera':
        points = (torch.rand(count,3,device='cuda',generator=generator)*2-1)*radius+center
        scales = torch.full_like(points,math.log(radius*1.5/count**(1/3)))
    else:
        dataset = SceneDataset(cfg['scene'],'train',cfg['resolution'],cfg['unit_light_intensity'])
        samples = [to_device(dataset[i],'cuda') for i in frames]
        from training.initialization import silhouette_seeds
        points, scales, support_metadata = silhouette_seeds(samples, center, radius, count, cfg['seed'])
        metadata.update(support_metadata)
    quats = torch.zeros(count,4,device='cuda');quats[:,0]=1
    state={'center':center.cpu(),'params.means':points.cpu(),'params.scales':scales.cpu(),
           'params.quats':quats.cpu(),'params.opacities':torch.full((count,),-2.2)}
    metadata['scale_median']=float(scales[:,0].exp().median())
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    torch.save(dict(config=cfg,gaussians=state,radius=radius,fit_indices=fit,val_indices=saved['val_indices'],
                    initial_support=metadata),out/'geometry.pt')
    (out/'provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata),flush=True)


def export_gs3_geometry(ply_path, reference_checkpoint, output):
    """Geometry-only capacity probe in the existing PORT geometry-import format.

    The GS3 model used all official training frames. These artifacts are for
    fitting-capacity diagnosis, never for train-internal holdout selection or
    the from-scratch final method. No GS3 material or shading normal is copied.
    """
    import numpy as np
    from plyfile import PlyData

    saved = torch.load(reference_checkpoint, map_location='cpu', weights_only=False)
    if not saved['config']['fit_all'] or saved['val_indices']:
        raise ValueError('GS3 full-train geometry needs a fit-all reference checkpoint')
    ply_path = Path(ply_path).resolve()
    if ply_path.parents[2].name != Path(saved['config']['scene']).name:
        raise ValueError('GS3 PLY scene and reference scene differ')
    vertices = PlyData.read(ply_path)['vertex']
    def columns(names):
        return torch.from_numpy(np.stack([vertices[name] for name in names], -1).copy())
    state = {'center':saved['gaussians']['center'],
             'params.means':columns(['x','y','z']),
             'params.scales':columns([f'scale_{i}' for i in range(3)]),
             'params.quats':columns([f'rot_{i}' for i in range(4)]),
             'params.opacities':columns(['opacity'])[:,0]}
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    metadata = dict(source_ply=str(ply_path), reference_checkpoint=str(Path(reference_checkpoint).resolve()),
        scene=saved['config']['scene'], points=len(vertices.data), radius=saved['radius'],
        scope='Fixed-geometry, all-training-data capacity diagnosis only; not an independent validation model',
        imported=['position','covariance','opacity'], excluded=['material','shading normal','neural phase function'])
    torch.save(dict(config=saved['config'], gaussians=state, radius=saved['radius'],
                    fit_indices=saved['fit_indices'], val_indices=[], source_geometry=metadata), out/'geometry.pt')
    (out/'provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata),flush=True)


def audit_atlas_refinement(checkpoint, output, limit):
    """Measure training-view image gradients, actual contribution and residual per primitive."""
    import sys
    import numpy as np

    torch.set_num_threads(8)
    g, t, saved = load_model(checkpoint)
    cfg = saved['config']
    if cfg['representation'] != 'light_atlas':
        raise ValueError('Atlas refinement audit requires light_atlas')
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    data = SceneDataset(cfg['scene'], 'train', cfg['resolution'], cfg['unit_light_intensity'])
    selected = saved['fit_indices'][::max(1, len(saved['fit_indices'])//limit)][:limit]
    offsets = load_camera_offsets(saved, g.radius).cuda() if saved['camera_offsets'] is not None else None
    frame_rows = {index:row for row,index in enumerate(saved['fit_indices'])}
    n = len(g.params['means'])
    grad = g.center.new_zeros(n)
    count = torch.zeros_like(grad)
    contribution = torch.zeros_like(grad)
    residual = torch.zeros_like(grad)
    frames = []
    for index in selected:
        sample = to_device(data[index], 'cuda')
        if offsets is not None:
            sample = offsets.correct(sample, frame_rows[index])
        g.zero_grad(set_to_none=True)
        t.zero_grad(set_to_none=True)
        predicted, alpha, info = render_observation(g, t, sample, cfg['background'],
            saved['step'] >= cfg['shadow_start'], saved['step'] >= cfg['port_start'],
            cfg['display_gamma'], cfg['shadow_mode'], absgrad=True)
        target = target_image(sample, cfg['background'], cfg['display_gamma'])
        error = (predicted.detach().clamp(0, 1) - target.clamp(0, 1)).square().mean(-1)
        loss = .8*(predicted-target).abs().mean() + .2*(1-ssim(predicted.clamp(0,1),target.clamp(0,1)))
        loss = loss + cfg['mask_weight']*(alpha-sample['alpha']).abs().mean()
        loss.backward()
        h, w = predicted.shape[:2]
        with torch.no_grad():
            visible = (info['radii'][0] > 0).all(-1)
            screen = info['means2d'].absgrad[0] * grad.new_tensor([w/2, h/2])
            grad[visible] += screen.norm(dim=-1)[visible]
            count += visible
            frames.append(dict(frame_index=index, loss=float(loss), PSNR=float(-10*error.mean().log10()),
                               alpha_L1=float((alpha-sample['alpha']).abs().mean())))
        del predicted, info, loss
        # A zero-color splat is a linear probe of front-to-back weights. Its
        # color derivative is sum_p T_ip alpha_ip; it cannot change this model.
        probe = g.center.new_zeros((n, 1), requires_grad=True)
        image, _, _ = _rasterize('3dgs', **{key:value.detach() for key,value in g.raster_inputs().items()},
            colors=probe, viewmats=sample['viewmat'].detach()[None], Ks=sample['K'].detach()[None],
            width=w, height=h, packed=False)
        mass = torch.autograd.grad(image, probe, torch.ones_like(image), retain_graph=True)[0][:,0]
        weighted_error = torch.autograd.grad(image, probe, error[None,...,None])[0][:,0]
        torch.testing.assert_close(mass.sum(), alpha.detach().sum(), rtol=5e-5, atol=1e-3)
        contribution += mass.detach()/(h*w)
        residual += weighted_error.detach()/(h*w)
    with torch.no_grad():
        gradient = grad/count.clamp_min(1)
        contribution /= len(selected)
        residual /= len(selected)
        scale = g.params['scales'].exp().amax(-1)/g.radius
        buckets = {}
        for name,mask in [('all', torch.ones_like(count, dtype=torch.bool)),
                          ('scale_lt_0.01', scale < .01), ('scale_0.01_0.03', (scale >= .01)&(scale < .03)),
                          ('scale_ge_0.03', scale >= .03)]:
            values = gradient[mask]
            if not len(values):
                continue
            buckets[name] = dict(count=int(mask.sum()),
                gradient_quantiles=values.quantile(values.new_tensor([0.,.5,.9,.99,1.])).tolist(),
                above_0_0008=int((values>.0008).sum()), above_0_00008=int((values>.00008).sum()),
                zero_contribution=int((contribution[mask]==0).sum()),
                contribution_share=float(contribution[mask].sum()/contribution.sum()),
                residual_share=float(residual[mask].sum()/residual.sum()))
        report = dict(checkpoint=str(Path(checkpoint).resolve()), command=sys.argv, shading=t.shading,
            step=saved['step'], scene=cfg['scene'], frames=frames, buckets=buckets,
            scope='Fixed training views; per-primitive native absgrad and alpha-composited contribution; no fitting')
        np.savez_compressed(out/'primitive_statistics.npz', gradient=gradient.cpu().numpy(),
            contribution=contribution.cpu().numpy(), residual=residual.cpu().numpy(), scale=scale.cpu().numpy())
    (out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(buckets),flush=True)


def score_external_renders(renders, scene, output, radius):
    """Score another method's saved official-test PNGs against canonical PORT targets.

    Uses PORT's quantized PSNR/SSIM/VGG-LPIPS and the same secondary shift-aligned
    protocol as evaluate.py --shift-align, so every method is compared identically.
    Renders must be 00000.png... in official transforms_test.json order, or PORT
    --save-all evaluation pairs (pair_000.png...), whose GT half is verified.
    """
    import lpips
    import numpy as np
    from PIL import Image
    from evaluate import shift_aligned_metrics
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    dataset = SceneDataset(scene, 'test', 512)
    model = lpips.LPIPS(net='vgg').cuda().eval()
    rows = []
    with torch.no_grad():
        for index in range(len(dataset)):
            sample = to_device(dataset[index], 'cuda')
            target = (target_image(sample, 0.0, 1.0).clamp(0, 1) * 255).round() / 255
            pair = Path(renders) / f'pair_{index:03d}.png'
            if pair.exists():
                # PORT evaluation pairs store [GT | prediction]; the GT half must equal the canonical target.
                image = np.asarray(Image.open(pair).convert('RGB'))
                width = image.shape[1] // 2
                saved_target = torch.from_numpy(image[:, :width]).cuda().float() / 255
                if not torch.equal(saved_target, target):
                    raise ValueError(f'Saved GT in {pair} differs from the canonical target')
                image = image[:, width:]
            else:
                image = np.asarray(Image.open(Path(renders) / f'{index:05d}.png').convert('RGB'))
            predicted = torch.from_numpy(np.ascontiguousarray(image)).cuda().float() / 255
            if predicted.shape != target.shape:
                raise ValueError(f'Render {index} has shape {tuple(predicted.shape)}, target {tuple(target.shape)}')
            row = {'frame_index': index, 'name': sample['name'],
                   'PSNR': (-10 * torch.log10((predicted - target).square().mean())).item(),
                   'SSIM': ssim(predicted, target).item(),
                   'LPIPS': model(predicted.permute(2, 0, 1)[None] * 2 - 1, target.permute(2, 0, 1)[None] * 2 - 1).item()}
            if radius:
                row['shift_aligned'] = shift_aligned_metrics(predicted, target, radius, 0.0, model)
            rows.append(row)
    metrics = {key: float(np.mean([row[key] for row in rows])) for key in ('PSNR', 'SSIM', 'LPIPS')}
    if radius:
        aligned = [row['shift_aligned'] for row in rows]
        metrics['shift_aligned'] = {key: float(np.mean([row[key] for row in aligned])) for key in aligned[0]}
        for key in ('dy', 'dx'):
            metrics['shift_aligned'][key + '_std'] = float(np.std([row[key] for row in aligned]))
    report = {'renders': str(Path(renders).resolve()), 'scene': str(scene), 'split': 'test',
              'metrics': metrics, 'views': rows,
              'protocol': 'Canonical PORT targets (512px, black background, uint8); unit PSNR, zero-padded '
                          '11x11 SSIM, VGG LPIPS on [-1,1]. shift_aligned: same secondary protocol as '
                          f'evaluate.py --shift-align {radius}.'}
    (output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(metrics), flush=True)


def silhouette_split_audit(checkpoint_path, output, radius=20):
    """Model-based calibration audit: silhouette shift of fit, held-out train and official test views.

    The checkpoint's geometry is rasterized with gsplat at each view's original
    calibration; the best global 2D shift of its alpha against the GT alpha is
    recorded (integer then 1/4px). With a checkpoint that never saw the held-out
    train frames, a systematic difference between held-out train and test shifts
    measures a train/test calibration offset independent of appearance modeling.
    """
    import numpy as np
    from gsplat import rasterization
    from gggs_reconstruction import author_modules, relighting_state
    author_modules()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    saved = torch.load(checkpoint_path, map_location='cuda', weights_only=False)
    if saved.get('kind') == 'gggs_core':
        state = relighting_state(saved)
        means, scales, quats = state['params.means'], state['params.scales'].exp(), state['params.quats']
        opacities = state['params.opacities'].sigmoid()
        scene, resolution = saved['config']['scene'], saved['config']['resolution']
    else:
        state = saved['gaussians']
        means, scales, quats = state['params.means'], state['params.scales'].exp(), state['params.quats']
        opacities = state['params.opacities'].sigmoid()
        scene, resolution = saved['config']['scene'], saved['config']['resolution']
    if scales.shape[-1] == 2:
        raise ValueError('Silhouette audit supports 3D Gaussian checkpoints')

    def shift(target, predicted):
        h, w = target.shape
        def moved(dy, dx):
            theta = torch.tensor([[1, 0, -2*dx/w], [0, 1, -2*dy/h]], dtype=torch.float32, device=target.device)[None]
            grid = F.affine_grid(theta, (1, 1, h, w), align_corners=False)
            return F.grid_sample(predicted[None, None], grid, align_corners=False)[0, 0]
        best = (float('inf'), 0., 0.)
        for dy in range(-radius, radius+1):
            for dx in range(-radius, radius+1):
                error = (moved(dy, dx)-target).square().mean().item()
                if error < best[0]:
                    best = (error, float(dy), float(dx))
        center_y, center_x = best[1], best[2]
        for dy in np.arange(center_y-1, center_y+1.01, .25):
            for dx in np.arange(center_x-1, center_x+1.01, .25):
                error = (moved(dy, dx)-target).square().mean().item()
                if error < best[0]:
                    best = (error, float(dy), float(dx))
        return best[1], best[2]
    report = {'checkpoint': str(checkpoint_path), 'scene': scene, 'resolution': resolution,
              'fit_frames': len(saved['fit_indices']), 'validation_frames': len(saved['val_indices']),
              'protocol': 'gsplat alpha at original calibration vs GT alpha; best global 2D shift within '
                          f'+-{radius}px (1/4px); shift is applied to the prediction to match GT.', 'splits': {}}
    for split, indices in [('validation', saved['val_indices']), ('test', None), ('fit', saved['fit_indices'][::8])]:
        dataset = SceneDataset(scene, 'test' if split == 'test' else 'train', resolution)
        indices = list(range(len(dataset))) if indices is None else list(indices)
        rows = []
        with torch.no_grad():
            for index in indices:
                sample = to_device(dataset[index], 'cuda')
                h, w = sample['image'].shape[:2]
                _, alpha, _ = rasterization(means, quats, scales, opacities, torch.ones_like(means[:, :1]),
                                            sample['viewmat'][None], sample['K'][None], w, h, packed=False)
                predicted, target = alpha[0, ..., 0], sample['alpha'][..., 0]
                dy, dx = shift(target, predicted)
                union = ((predicted > .5) | (target > .5)).sum().item()
                rows.append({'frame_index': index, 'dy': dy, 'dx': dx,
                             'iou': ((predicted > .5) & (target > .5)).sum().item()/max(1, union)})
        values = np.array([[row['dy'], row['dx']] for row in rows])
        report['splits'][split] = {'frames': len(rows), 'dy_mean': float(values[:, 0].mean()),
            'dy_std': float(values[:, 0].std()), 'dx_mean': float(values[:, 1].mean()),
            'dx_std': float(values[:, 1].std()), 'iou_mean': float(np.mean([row['iou'] for row in rows])),
            'views': rows}
        print(json.dumps({split: {k: v for k, v in report['splits'][split].items() if k != 'views'}}), flush=True)
    (output/'silhouette_audit.json').write_text(json.dumps(report, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", nargs='?')
    parser.add_argument("--output", required=True)
    parser.add_argument('--multiview-probe', type=Path, help='Scene path for train-only calibrated plane-sweep feasibility')
    parser.add_argument('--compare-geometry-buffers', type=Path, help='Candidate evaluation directory; --reference-geometry is the reference evaluation directory')
    parser.add_argument('--reference-geometry', type=Path, help='Reference evaluation directory for saved geometry buffers')
    parser.add_argument('--relight-preview', action='store_true', help='Fixed-camera four-light preview of the directional or neural-material pipeline')
    parser.add_argument('--light-atlas-preview', action='store_true', help='Export actual LiSA atlas channels, pyramid and shading decomposition')
    parser.add_argument('--atlas-refinement-audit', action='store_true', help='Measure LiSA native gradients and per-Gaussian contribution on fit frames')
    parser.add_argument('--free-color-steps', type=int, default=0,
                        help='Diagnostic per-fit-view nonnegative color fitting on fixed geometry; not relighting evaluation')
    parser.add_argument('--gs3-geometry', type=Path, help='Export this GS3 PLY for a fixed-geometry capacity probe, using the positional checkpoint for normalization/splits')
    parser.add_argument('--initial-support', choices=['camera','hull'], help='Export fresh 20k initialization from reference split/camera normalization, optionally restricted by fit silhouettes')
    parser.add_argument('--atlas-frames', type=int, nargs='+', default=[0], help='Dataset frame indices for atlas preview or free-color diagnosis')
    parser.add_argument('--gggs-default-review', action='store_true', help='Compare source GGGS, its default-renderer import and jointly optimized geometry')
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--split", choices=['train','test'], default='train')
    parser.add_argument("--gradient-frames", type=int, default=0)
    parser.add_argument("--geometry-frames", type=int, default=0,
                        help="Save 2DGS normal/depth consistency panels for this many frames")
    parser.add_argument('--cue-coherence', action='store_true',
                        help='Compare multi-view training highlight half-vectors at raw/SDF-projected positions')
    parser.add_argument("--highlight-frames", type=int, default=0,
                        help="Decompose neural-material shading on this many selected frames")
    parser.add_argument("--attention-frames", type=int, default=0,
                        help="Record attention weights for this many frames per split")
    parser.add_argument('--sdf-mask-fit-steps', type=int, default=0,
                        help='CPU-only SDF silhouette refinement probe; does not update the scene model')
    parser.add_argument('--sdf-mask-weight', type=float, default=1.,
                        help='Mask weight for the CPU probe; zero gives an Eikonal-only control')
    parser.add_argument('--camera-matches', type=Path,
                        help='CPU audit of saved camera offsets using fixed NPZ matches in checkpoint-resolution pixels')
    parser.add_argument('--angular-cues', action='store_true',
                        help='Train-only local angle-cue localization audit on a saved radiance-residual model')
    parser.add_argument('--reference-metrics', type=Path, help='Existing full-fit metrics.json for exact diagnostic alignment')
    parser.add_argument('--cue-crops', type=Path, help='Existing interaction comparison.json containing fixed GT crops')
    parser.add_argument('--resolution', type=int, help='Angular-cue resolution; must match the reference evaluation')
    parser.add_argument('--foundation-comparison', type=Path,
                        help='Compare saved foundation PNGs using a JSON manifest and canonical dataset targets')
    parser.add_argument('--foundation-device', choices=['cpu', 'cuda:0'], default='cpu',
                        help='Device for optional foundation LPIPS only; all other saved-image metrics use CPU')
    parser.add_argument('--lpips', action='store_true', help='Compute VGG LPIPS for the foundation comparison')
    parser.add_argument('--silhouette-audit', action='store_true',
                        help='Silhouette shifts of fit, held-out train and official test views for a 3D checkpoint')
    parser.add_argument('--external-renders', type=Path,
                        help='Directory of another method\'s official-test PNGs (00000.png...) to score against canonical targets')
    parser.add_argument('--scene', type=Path, help='Scene directory for --external-renders')
    parser.add_argument('--shift-align', type=int, default=24,
                        help='Search radius of the secondary shift-aligned metrics for --external-renders; 0 disables')
    args = parser.parse_args()
    if args.free_color_steps:
        if not args.checkpoint or args.split != 'train' or args.free_color_steps < 2:
            parser.error('Free-color fitting needs a checkpoint, train split and at least two steps')
        return fit_free_colors(args.checkpoint,args.output,args.atlas_frames,args.free_color_steps)
    if args.initial_support:
        if not args.checkpoint:
            parser.error('Initial support export requires a reference split/normalization checkpoint')
        return export_initial_support(args.checkpoint,args.output,args.initial_support)
    if args.gs3_geometry:
        if not args.checkpoint:
            parser.error('GS3 geometry export requires a fit-all reference checkpoint')
        return export_gs3_geometry(args.gs3_geometry, args.checkpoint, args.output)
    if args.atlas_refinement_audit:
        if not args.checkpoint or args.split != 'train' or args.limit < 1:
            parser.error('Atlas refinement audit requires a checkpoint, train split and positive --limit')
        return audit_atlas_refinement(args.checkpoint, args.output, args.limit)
    if args.light_atlas_preview:
        if not args.checkpoint:parser.error('Light-atlas preview requires a checkpoint')
        return preview_light_atlas(args.checkpoint, args.output, args.split, args.atlas_frames)
    if args.silhouette_audit:
        if not args.checkpoint:parser.error('--silhouette-audit requires a checkpoint')
        return silhouette_split_audit(args.checkpoint, args.output)
    if args.external_renders:
        if not args.scene:parser.error('--external-renders requires --scene')
        return score_external_renders(args.external_renders, args.scene, args.output, args.shift_align)
    if args.relight_preview:
        if not args.checkpoint:parser.error('Relighting preview requires a checkpoint')
        return preview_relighting(args.checkpoint,args.output)
    if args.gggs_default_review:
        if not args.checkpoint:parser.error('GGGS/default geometry review requires a checkpoint')
        from geometry_review import review_default_geometry
        return review_default_geometry(args.checkpoint,args.output)
    if args.compare_geometry_buffers:
        if not args.reference_geometry:parser.error('Geometry buffers require --reference-geometry')
        from geometry_review import compare_geometry_buffers
        return compare_geometry_buffers(args.reference_geometry,args.compare_geometry_buffers,args.output)
    if args.multiview_probe:
        from multiview_geometry import run_probe
        return run_probe(args.multiview_probe,args.output,args.resolution or 256)
    if args.foundation_comparison:
        if (args.checkpoint is not None or args.angular_cues or args.camera_matches
                or args.sdf_mask_fit_steps or args.gradient_frames or args.geometry_frames
                or args.highlight_frames or args.attention_frames or args.cue_coherence
                or args.reference_metrics is not None or args.cue_crops is not None or args.resolution is not None):
            parser.error('--foundation-comparison cannot combine with a checkpoint or other diagnostic modes')
        from foundation_comparison import compare_foundations
        torch.set_num_threads(8)
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=False)
        compare_foundations(args.foundation_comparison, output, args.foundation_device, args.lpips)
        return
    if args.checkpoint is None:
        parser.error('checkpoint is required except with --foundation-comparison')
    if args.lpips or args.foundation_device != 'cpu':
        parser.error('--lpips and --foundation-device require --foundation-comparison')
    if args.sdf_mask_fit_steps < 0 or args.sdf_mask_weight < 0:
        parser.error('SDF mask probe steps/weight must be nonnegative')
    if args.sdf_mask_fit_steps and args.split != 'train':
        parser.error('The SDF mask probe uses training views only')
    if args.camera_matches and (args.split != 'train' or args.sdf_mask_fit_steps):
        parser.error('Camera match audit uses training views and cannot combine with the mask probe')
    if args.angular_cues:
        if (args.split != 'train' or args.limit < 0 or args.reference_metrics is None or args.cue_crops is None
                or args.gradient_frames or args.geometry_frames or args.highlight_frames or args.attention_frames
                or args.cue_coherence or args.sdf_mask_fit_steps or args.camera_matches):
            parser.error('--angular-cues requires train, nonnegative limit (0=all), reference metrics and cue crops, without other diagnostic modes')
    elif args.reference_metrics is not None or args.cue_crops is not None or args.resolution is not None:
        parser.error('Reference metrics, cue crops and resolution require --angular-cues')
    torch.set_num_threads(8)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    if args.angular_cues:
        audit_angular_cues(args,output)
        return
    if args.camera_matches:
        import numpy as np
        from cameras import correspondence_errors
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        dataset = SceneDataset(checkpoint['config']['scene'], 'train', checkpoint['config']['resolution'])
        with np.load(args.camera_matches) as matches:
            result = correspondence_errors(checkpoint, matches, dataset)
        result.update(checkpoint=str(Path(args.checkpoint).resolve()),
                      matches=str(args.camera_matches.resolve()), resolution=dataset.resolution)
        (output/'metrics.json').write_text(json.dumps(result, indent=2)+'\n')
        with tarfile.open(output/'source.tar', 'w') as archive:
            for name in ['cameras.py', 'data.py', 'diagnose_image_errors.py']:
                archive.add(Path(__file__).parent/name, arcname=name)
        print(json.dumps({key:result[key] for key in ['all', 'platform', 'optimized_poses']}), flush=True)
        return
    if args.sdf_mask_fit_steps:
        fit_sdf_mask_probe(args.checkpoint, output, args.sdf_mask_fit_steps, args.sdf_mask_weight)
        return
    g, t, checkpoint = load_model(args.checkpoint)
    cfg = checkpoint["config"]
    sdf = load_surface_field(checkpoint)
    if cfg.get('sdf', False):
        from sdf import sample_surface, visible_primitives
        sdf_generator = torch.Generator(device='cuda').manual_seed(1)
    shading_field = sdf if cfg.get('sdf_shading', False) else None
    normal_field = load_normal_field(checkpoint)
    if args.cue_coherence and (args.split != 'train' or sdf is None or cfg['representation'] != 'neural_material'):
        raise ValueError('Cue coherence requires the training split of neural_material with a fitted SDF')
    dataset = SceneDataset(
        cfg["scene"], args.split, cfg["resolution"], unit_light_intensity=cfg["unit_light_intensity"]
    )
    fit = checkpoint["fit_indices"]
    selected = [("fit", i) for i in fit[:: max(1, len(fit) // args.limit)][:args.limit]]
    selected += [("validation", i) for i in checkpoint["val_indices"][:args.limit]]
    if args.split == 'test':
        selected=[('test',i) for i in range(0,len(dataset),max(1,len(dataset)//args.limit))][:args.limit]
    if args.cue_coherence:
        selected = [(split,index) for split,index in selected if split=='fit']
    priors = None
    if args.gradient_frames and g.geometry == "2dgs":
        priors = SurfacePriors(cfg['surface_priors'], dataset, fit)
    show = set(selected[:2]) | {("validation", index) for index in checkpoint["val_indices"][:2]}
    camera_offsets = None
    if checkpoint["camera_offsets"] is not None:
        camera_offsets = load_camera_offsets(checkpoint, g.radius).cuda()
    camera_indices = {index: local for local, index in enumerate(fit)}
    rows = []
    cues = {key:[] for key in ('points','half','normal','frames')}
    attention_rows = []
    attention_counts = {"fit": 0, "validation": 0, "test": 0}
    for split, index in selected:
        sample = to_device(dataset[index], "cuda")
        if split == "fit" and camera_offsets is not None:
            sample = camera_offsets.correct(sample, camera_indices[index])
        captured = []
        hook = t.register_forward_hook(lambda module, inputs, result: captured.append((inputs, result)))
        inspect_attention = attention_counts[split] < args.attention_frames
        if inspect_attention:
            attention = {}
            pooled_sources = t.pooled_sources
            def capture_tokens(source):
                result = pooled_sources(source)
                attention['tokens'] = result
                return result
            t.pooled_sources = capture_tokens
            query_hooks = [
                t.query.register_forward_hook(lambda module, inputs, result: attention.update(queries=result)),
                t.key.register_forward_hook(lambda module, inputs, result: attention.update(keys=result)),
            ]
        pred, alpha, info = render_observation(
            g,
            t,
            sample,
            cfg["background"],
            checkpoint["step"] >= cfg["shadow_start"],
            checkpoint["step"] >= cfg["port_start"],
            cfg["display_gamma"],
            cfg["shadow_mode"],
            surface_field=shading_field,
            normal_field=normal_field,
        )
        hook.remove()
        if inspect_attention:
            t.pooled_sources = pooled_sources
            for handle in query_hooks:
                handle.remove()
            _, _, _, values, mass = attention['tokens']
            q, k = attention['queries'][None], attention['keys'][None]
            score = t.attention_score
            if score == 'cosine':
                logits = (F.normalize(q,dim=-1) @ F.normalize(k,dim=-1).transpose(-1,-2)) * t.attention_dim**.5
            else:
                logits = (q @ k.transpose(-1,-2)) / t.attention_dim**.5
            weights = (logits+mass.log()).softmax(-1)
            extra = dict(query_norm_mean=float(q.norm(dim=-1).mean()),
                         key_norm_mean=float(k.norm(dim=-1).mean()),
                         logit_min=float(logits.min()),logit_max=float(logits.max()))
            del q,k,logits
            entropy = -(weights*weights.clamp_min(1e-30).log()).sum(-1)
            selected_signal = weights[0]@values
            extra['effective_tokens_per_head'] = entropy.exp().mean(-1).tolist()
            attention_rows.append(dict(
                split=split, frame_index=index, score=score, tokens=len(mass),
                max_weight_mean=float(weights.max(-1).values.mean()),
                max_weight_p99=float(weights.max(-1).values.quantile(.99)),
                effective_tokens_mean=float(entropy.exp().mean()),
                selected_signal_mean=selected_signal.mean(0).tolist(),
                area_mean_signal=(mass@values).tolist(), **extra))
            attention_counts[split] += 1
            del attention, values, mass, weights, entropy, selected_signal
        inputs, foreground = captured[0]
        direct = t(*inputs[:-1], False)
        fraction = t.exchange(inputs[1]['features']).sigmoid()
        nonlocal_rgb = foreground - (1-fraction)*direct
        if cfg['representation'] == 'neural_material':
            receivers=inputs[1]
            incident,wi=t.point_light(receivers['means'],inputs[3],inputs[4],receivers['visibility'])
            wo=F.normalize(inputs[2]-receivers['means'],dim=-1)
            specular,_=t.material_response(receivers,wi,wo)
            nonlocal_rgb -= fraction*incident*specular
        nonlocal_share = float(nonlocal_rgb.sum()/foreground.sum().clamp_min(1e-8))
        pred = (pred.clamp(0, 1) * 255).round() / 255
        target = (
            target_image(sample, cfg["background"], cfg["display_gamma"]).clamp(0, 1) * 255
        ).round() / 255
        error = pred - target
        error2 = error.square().mean(-1)
        gt_mask = sample["alpha"][..., 0] > 0.5
        pred_mask = alpha[..., 0] > 0.5
        mask = gt_mask[None, None].float()
        inner = -F.max_pool2d(-mask, 5, 1, 2)[0, 0]
        outer = F.max_pool2d(mask, 5, 1, 2)[0, 0]
        regions = {"interior": inner, "boundary": outer - inner, "background": 1 - outer}
        distances = torch.cdist(
            torch.stack(torch.where(boundary(gt_mask)), -1).float(),
            torch.stack(torch.where(boundary(pred_mask)), -1).float(),
        )
        edge_dist = torch.cat((distances.amin(0), distances.amin(1)))
        visibility = visibility_hint(g, sample["light_pos"], mode=cfg["shadow_mode"])
        radius = info["radii"][0].amax(-1).float()
        attributes = torch.stack((visibility, radius), -1)
        rendered, _, _ = _rasterize(
            g.geometry,
            **g.raster_inputs(),
            colors=attributes,
            viewmats=sample["viewmat"][None],
            Ks=sample["K"][None],
            width=pred.shape[1],
            height=pred.shape[0],
            packed=False,
        )
        maps = rendered[0] / alpha.clamp_min(1e-8)
        overlap = gt_mask & pred_mask
        mse = error2.mean()
        row = {
            "split": split,
            "frame_index": index,
            "PSNR": float(-10 * mse.log10()),
            "nonlocal_radiance_share": nonlocal_share,
            "exchange_mean": fraction.mean(0).tolist(),
            "alpha_L1": float((alpha-sample['alpha']).abs().mean()),
            "MSE": float(mse),
            "error_share": {
                key: float((error2 * value).sum() / error2.sum()) for key, value in regions.items()
            },
            "block_error_fraction": {
                str(block): float(block_energy(error, block) / mse) for block in [2, 4, 8, 16]
            },
            "fine_detail_energy_ratio": float(
                (block_energy(pred, 1) - block_energy(pred, 2))
                / (block_energy(target, 1) - block_energy(target, 2))
            ),
            "interior_rgb_bias": ((error * inner[..., None]).sum((0, 1)) / inner.sum()).tolist(),
            "silhouette_distance_mean_px": float(edge_dist.mean()),
            "silhouette_distance_p90_px": float(edge_dist.quantile(0.9)),
            "silhouette_iou": float((gt_mask & pred_mask).sum() / (gt_mask | pred_mask).sum()),
            "mean_visible_support_radius_px": float(maps[..., 1][overlap].mean()),
            "shadow_condition_bins": {},
        }
        if sample['alpha'] is not None:
            row['neutral_peaks']=neutral_peak_metrics(pred,target,sample['alpha'])
        if cfg['representation'] == 'neural_material':
            covered = alpha[..., 0] > 0
            selected = (inner.bool() & (sample['alpha'][..., 0] > .99) & (alpha[..., 0] > .9))[covered]
            base = receivers['base'].sigmoid()
            unshadowed, _ = t.point_light(receivers['means'], inputs[3], inputs[4],
                                        torch.ones_like(receivers['visibility']))
            linear_target = sample['image'][covered]
            if not sample['is_hdr']:
                linear_target = linear_target.pow(cfg['display_gamma'])
            # Normal-independent upper bound for unit-albedo *direct Lambertian*
            # under the current normalized light scale. Specular and indirect
            # energy can exceed it: this ratio is not a true albedo estimate.
            ratio = math.pi * linear_target / unshadowed
            row['diffuse_capacity'] = {
                'pixels':int(selected.sum()),
                'base_above_095_rgb':(base[selected] > .95).float().mean(0).tolist(),
                'target_above_unshadowed_lambertian_bound_rgb':(ratio[selected] > 1).float().mean(0).tolist(),
                'target_to_bound_quantiles_rgb':torch.quantile(ratio[selected], ratio.new_tensor([.5, .9, .99]), dim=0).tolist(),
                'note':'Current light normalization and observation gamma; ignores specular/indirect energy. Not ground-truth albedo or proof of exposure error.',
            }
            ys = torch.where(gt_mask)[0]
            y = torch.arange(gt_mask.shape[0], device=gt_mask.device)[:, None].expand_as(gt_mask)
            color = F.avg_pool2d(sample['image'].permute(2, 0, 1)[None], 5, 1, 2)[0].permute(1, 2, 0)
            platform = ((y > ys.min()+.65*(ys.max()-ys.min())) & (color[..., 0] < 1.5*color[..., 1])
                        & ~neutral_peak_mask(target, sample['alpha']))[covered] & selected
            row['diffuse_capacity']['platform_proxy'] = {
                'pixels':int(platform.sum()),
                'note':'Lower 35% of foreground y extent, local R<1.5G, excluding neutral peaks; heuristic, not semantic ground truth.',
            }
            if platform.any():
                row['diffuse_capacity']['platform_proxy'].update(
                    target_above_bound_rgb=(ratio[platform] > 1).float().mean(0).tolist(),
                    target_to_bound_quantiles_rgb=torch.quantile(ratio[platform], ratio.new_tensor([.5, .9, .99]), dim=0).tolist())
        if args.cue_coherence:
            covered = alpha[...,0]>0
            peak = (neutral_peak_mask(target,sample['alpha']) & (alpha[...,0]>.8))[covered]
            if peak.any():
                world = inputs[1]['means'][peak]
                half = F.normalize(F.normalize(sample['light_pos']-world,dim=-1)
                                   + F.normalize(sample['c2w'][:3,3]-world,dim=-1),dim=-1)
                cues['points'].append(sdf.normalized(world).cpu())
                cues['half'].append(half.cpu())
                cues['normal'].append(t.material_normal(inputs[1])[peak].cpu())
                cues['frames'].append(torch.full((len(world),),index,dtype=torch.long))
        if g.geometry == '2dgs':
            from surface import depth_normals
            normal = F.normalize(info['surface_normals'] @ sample['viewmat'][:3, :3].T, dim=-1)
            derived = depth_normals(info['surface_depth'], sample['K'])
            valid = (alpha[..., 0] > .8) & gt_mask & (info['surface_depth'][..., 0] > 0)
            interior = F.avg_pool2d(valid.float()[None,None],3,1,1)[0,0] == 1
            angle = (normal*derived).sum(-1).clamp(-1,1).acos() * (180 / torch.pi)
            row['geometry'] = {
                'interior_pixels':int(interior.sum()),
                'normal_depth_angle_deg':float(angle[interior].mean()),
                'distortion_radius_squared':float(info['surface_distortion'][...,0][interior].mean()/g.radius**2),
            }
            if sdf is not None:
                candidates = visible_primitives(g.params['means'], g.params['opacities'].sigmoid(),
                                                 info, alpha, sample, g.radius)
                if len(candidates):
                    primitive_distance = sdf(sdf.normalized(g.params['means'][candidates])).abs()
                    row['geometry']['sdf_primitives'] = {
                        'candidates':len(candidates),
                        'distance_radius_units':float(primitive_distance.mean()),
                        'fraction_above_02':float((primitive_distance>.02).float().mean()),
                    }
                samples = sample_surface(info, alpha, sample, sdf, 1024, sdf_generator)
                if samples is not None:
                    points, sampled_normals, _ = samples
                    with torch.enable_grad():
                        points = points.detach().requires_grad_()
                        distance = sdf(points)
                        gradient = torch.autograd.grad(distance.sum(), points)[0]
                    agreement = (F.normalize(gradient, dim=-1)*sampled_normals).sum(-1).clamp(-1,1)
                    row['geometry']['sdf'] = {
                        'surface_distance_radius_units':float(distance.abs().mean()),
                        'normal_agreement_deg':float(agreement.acos().mean()*180/torch.pi),
                        'eikonal_abs_error':float((gradient.norm(dim=-1)-1).abs().mean()),
                    }
            if len(rows) < args.geometry_frames:
                panels = [((normal*.5+.5)*valid[...,None], '2DGS normal'),
                          ((derived*.5+.5)*interior[...,None], 'Depth-derived normal'),
                          (info['surface_depth'][...,0]*valid, 'Camera Z depth'),
                          (angle*interior, 'Normal/depth angle (degrees)')]
                if shading_field is not None or normal_field is not None:
                    for values, title in [(inputs[1]['normals'], 'SDF base normal' if shading_field is not None else 'GS base normal'),
                                           (t.material_normal(inputs[1]), 'BRDF shading normal')]:
                        canvas = torch.zeros_like(normal)
                        canvas[alpha[...,0]>0] = (values @ sample['viewmat'][:3,:3].T)*.5+.5
                        panels.append((canvas*valid[...,None], title))
                    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
                else:
                    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
                for ax, (values, title) in zip(axes.flat, panels):
                    artist = ax.imshow(values.cpu().numpy(), **({'vmin':0, 'vmax':90} if title.startswith('Normal/') else {}))
                    ax.set_title(title); ax.axis('off')
                    if values.ndim == 2:
                        fig.colorbar(artist, ax=ax, fraction=.035)
                fig.tight_layout()
                fig.savefig(output/f'{split}_{index:04}_geometry.png', dpi=120)
                plt.close(fig)
        if len(rows) < args.highlight_frames:
            row['highlights'] = highlight_diagnostics(t, inputs, foreground, alpha, target,
                                                      sample['alpha'], cfg['display_gamma'], output, f'{split}_{index:04}')
        for name, low, high in [("deep", 0, 0.1), ("partial", 0.1, 0.5), ("lit", 0.5, 1.01)]:
            selection = overlap & (maps[..., 0] >= low) & (maps[..., 0] < high)
            row["shadow_condition_bins"][name] = {
                "pixels": int(selection.sum()),
                "squared_error": float(error2[selection].sum()),
                "target_luminance_sum": float(target.mean(-1)[selection].sum()),
                "pred_luminance_sum": float(pred.mean(-1)[selection].sum()),
            }
        rows.append(row)
        if priors is not None and split == 'fit' and len(rows) <= args.gradient_frames:
            with torch.enable_grad():
                raw, a, details = render_observation(g,t,sample,cfg['background'],True,True,
                                                     cfg['display_gamma'],cfg['shadow_mode'],surface_field=shading_field,
                                                     normal_field=normal_field)
                gt = target_image(sample,cfg['background'],cfg['display_gamma'])
                losses = surface_losses(details,a,sample,priors.frame(index,'cuda'),g.radius,
                    cfg['normal_weight'],cfg['depth_weight'],cfg['surface_consistency_weight'],cfg['distortion_weight'])
                photo = .8*(raw-gt).abs().mean()+.2*(1-ssim(raw.clamp(0,1),gt.clamp(0,1)))
                params = (g.params['means'],g.params['quats'])
                photo_grads = torch.autograd.grad(photo,params,retain_graph=True)
                row['geometry_gradient'] = {}
                for name,loss in losses.items():
                    gradients = torch.autograd.grad(loss,params,retain_graph=True)
                    row['geometry_gradient'][name] = {}
                    for field,grad,reference in zip(('means','quats'),gradients,photo_grads):
                        row['geometry_gradient'][name][field] = {
                            'relative_norm':float(grad.norm()/reference.norm().clamp_min(1e-12)),
                            'cosine':float((grad*reference).sum()/(grad.norm()*reference.norm()).clamp_min(1e-12))}
            del raw,a,details,losses,photo,photo_grads,gradients
        if (split, index) in show:
            fig, axes = plt.subplots(2, 3, figsize=(12, 8))
            panels = [
                (target, "GT"),
                (pred, f'Prediction {row["PSNR"]:.2f} dB'),
                (error.abs().mean(-1), "Absolute RGB error"),
                (
                    torch.stack(
                        (sample["alpha"][..., 0], alpha[..., 0], torch.zeros_like(alpha[..., 0])),
                        -1,
                    ),
                    "Alpha: red GT, green prediction",
                ),
                (maps[..., 0], "Camera-weighted shadow visibility"),
                (maps[..., 1], "Camera-weighted support radius (px)"),
            ]
            for ax, (values, title) in zip(axes.flat, panels):
                kwargs = {"vmin": 0, "vmax": 1} if title.endswith("visibility") else {}
                if title.startswith("Absolute"):
                    kwargs = {"vmin": 0, "vmax": 0.3}
                if title.endswith("(px)"):
                    kwargs = {"vmin": 0, "vmax": 80}
                artist = ax.imshow(values.cpu().numpy(), **kwargs)
                ax.set_title(title)
                ax.axis("off")
                if values.ndim == 2:
                    fig.colorbar(artist, ax=ax, fraction=0.035)
            camera_label = "saved fit camera" if split == "fit" and camera_offsets is not None else "original camera"
            fig.suptitle(f"{split} frame {index} | {camera_label}")
            fig.tight_layout()
            fig.savefig(output / f"{split}_{index:04}.png", dpi=120)
            plt.close(fig)
    result = {
        "checkpoint": args.checkpoint,
        "material_model": t.material_model if cfg['representation'] == 'neural_material' else None,
        "shading_normals": "sdf" if shading_field is not None else "gaussian",
        "normal_residual_field": normal_field is not None,
        "surface_depth": g.surface_depth,
        "rows": rows,
        "notes": [
            "Saved training camera offsets apply to fit frames; held-out frames use original cameras.",
            "Predictions use the saved observation model and training-stage schedule.",
            "Block energy is squared error after non-overlapping area averaging; differences between levels are orthogonal block-detail energy.",
            "Support radius is camera-alpha-compositing weighted; shadow bins use the model visibility estimate.",
            "Silhouette distances use threshold .5 contours in pixels.",
            "Peak localization uses the same neutral-peak rule and GT foreground on both images; matching allows a 2-pixel Chebyshev radius. These are image proxies, not specular ground truth.",
        ],
    }
    if args.attention_frames:
        result['attention_kernel'] = attention_rows
    if args.cue_coherence:
        if not cues['points']:
            raise ValueError('No covered training peaks available for normal-cue analysis')
        bank = {key:torch.cat(value) for key,value in cues.items()}
        projected, valid = [], []
        for chunk in bank['points'].split(8192):
            with torch.enable_grad():
                points = chunk.to(g.center.device).requires_grad_()
                distance = sdf(points)
                gradient = torch.autograd.grad(distance.sum(),points)[0]
            norm2 = gradient.square().sum(-1,keepdim=True)
            projected.append((points-distance[:,None]*gradient/norm2.clamp_min(1e-6)).detach().cpu())
            valid.append(((norm2[:,0]>.25)&(distance.abs()<.05)).cpu())
        bank['projected_points'] = torch.cat(projected)
        bank['valid_projection'] = torch.cat(valid)
        good = bank['valid_projection']
        cue_report = {
            'pixels':len(bank['points']), 'frames':int(bank['frames'].unique().numel()),
            'projection_supported_pixels':int(good.sum()),
            'model_to_half_angle_deg':float((bank['normal']*bank['half']).sum(-1).clamp(-1,1).acos().mean()*180/torch.pi),
            'raw':normal_cue_coherence(bank['points'][good],bank['half'][good],bank['frames'][good]),
            'sdf_projected':normal_cue_coherence(bank['projected_points'][good],bank['half'][good],bank['frames'][good]),
            'notes':['Half-vectors at GT image peaks are conditional mirror-normal cues, not true surface normals.',
                     'Raw and projected statistics use identical pixels with gradient norm > .5 and |SDF| < .05 radius.',
                     'One Newton projection to SDF; voxel sizes are in scene-radius units; no test/validation frames.'],
        }
        torch.save(bank,output/'normal_cues.pt')
        (output/'cue_coherence.json').write_text(json.dumps(cue_report,indent=2)+'\n')
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"frames": len(rows), "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
