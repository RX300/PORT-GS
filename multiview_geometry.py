"""Train-only calibrated plane-sweep feasibility probe, without learned priors.

This is a conservative matching diagnostic, not a reproduction of CoMVS-GS or
PatchMatch. Frontoparallel 9x9 ZNCC patches deliberately keep the test simple.
"""
from pathlib import Path
import json
import tarfile
import time

import numpy as np
import torch
from torch.nn import functional as F

from data import SceneDataset, split_train_lights
from evaluate import to_device
from gaussians import camera_bounds


def pixel_rays(sample):
    h, w = sample['image'].shape[:2]
    y, x = torch.meshgrid(torch.arange(h, device=sample['K'].device)+.5,
                          torch.arange(w, device=sample['K'].device)+.5, indexing='ij')
    return torch.stack((x, y, torch.ones_like(x)), -1) @ torch.linalg.inv(sample['K']).T


def project_world(points, sample):
    cam = points @ sample['viewmat'][:3, :3].T + sample['viewmat'][:3, 3]
    q = cam @ sample['K'].T
    uv = q[..., :2] / q[..., 2:].clamp_min(1e-8)
    h, w = sample['image'].shape[:2]
    return uv / uv.new_tensor([w, h])*2-1, cam[..., 2]


def world_points(sample, depth):
    cam = pixel_rays(sample)*depth[..., None]
    return (cam-sample['viewmat'][:3, 3]) @ sample['viewmat'][:3, :3]


def select_neighbors(samples, center, count=4):
    """Moderate camera baseline; prefer similar known lamp directions."""
    indices = list(samples)
    origins = torch.stack([samples[i]['c2w'][:3, 3] for i in indices])
    lamps = torch.stack([samples[i]['light_pos'] for i in indices])
    camera_dirs = F.normalize(origins-center, dim=-1)
    light_dirs = F.normalize(lamps-center, dim=-1)
    cam_angle = (camera_dirs @ camera_dirs.T).clamp(-1, 1).acos()*180/torch.pi
    light_angle = (light_dirs @ light_dirs.T).clamp(-1, 1).acos()*180/torch.pi
    result, metadata = {}, {}
    for j, index in enumerate(indices):
        eligible = (cam_angle[j] >= 3) & (cam_angle[j] <= 25)
        candidates = torch.where(eligible)[0]
        if len(candidates) < count:
            raise ValueError(f'Insufficient 3–25 degree neighbors for frame {index}')
        cost = light_angle[j, candidates] + .5*(cam_angle[j, candidates]-10).abs()
        chosen = candidates[cost.argsort()[:count]]
        result[index] = [indices[k] for k in chosen.tolist()]
        metadata[index] = [{'frame':indices[k], 'camera_degrees':float(cam_angle[j,k]),
                            'light_degrees':float(light_angle[j,k])} for k in chosen.tolist()]
    return result, metadata


@torch.no_grad()
def plane_sweep(reference, sources, center, radius, planes=384):
    """Independent depth search; no model depth enters the search interval."""
    h, w = reference['image'].shape[:2]
    device = reference['K'].device
    zcenter = (reference['viewmat'][:3,:3]@center + reference['viewmat'][:3,3])[2]
    depths = torch.linspace(float(zcenter-radius), float(zcenter+radius), planes, device=device)
    if depths[0] <= 0:
        raise ValueError('Camera-bound search volume crosses camera plane')
    gray = lambda s: (s['image']*s['image'].new_tensor([.299,.587,.114])).sum(-1)
    pool = lambda x: F.avg_pool2d(x, 9, stride=1, padding=4)
    r = gray(reference)[None, None]
    mr, vr = pool(r), (pool(r*r)-pool(r).square()).clamp_min(0)
    valid_ref = pool(reference['alpha'].permute(2,0,1)[None]) > .99
    rays = pixel_rays(reference)
    costs = []
    for ds in depths.split(16):
        xyz = (rays[None]*ds[:,None,None,None]-reference['viewmat'][:3,3]) @ reference['viewmat'][:3,:3]
        source_costs = []
        for source in sources:
            grid, z = project_world(xyz, source)
            values = torch.stack((gray(source), source['alpha'][...,0]))[None].expand(len(ds),-1,-1,-1)
            warped = F.grid_sample(values, grid, align_corners=False)
            s = warped[:, :1]
            ms, vs = pool(s), (pool(s*s)-pool(s).square()).clamp_min(0)
            covariance = pool(r*s)-mr*ms
            ncc = (covariance/(vr*vs).sqrt().clamp_min(1e-6)).clamp(-1,1)
            valid = valid_ref & (pool(warped[:,1:2])>.99) & (vr>.0001) & (vs>.0001)
            valid &= (z[:,None]>0)
            source_costs.append(torch.where(valid, 1-ncc, 2.)[:,0])
        # Require agreement in at least three of the four source views.
        costs.append(torch.stack(source_costs).sort(dim=0).values[:3].mean(0))
    volume = torch.cat(costs)
    best, ids = volume.min(0)
    # Compare distinct minima, not neighboring samples of the same smooth
    # minimum. The latter makes confidence depend on plane discretization.
    minima = torch.ones_like(volume,dtype=torch.bool)
    minima[1:-1] = (volume[1:-1]<volume[:-2]) & (volume[1:-1]<=volume[2:])
    separation = (torch.arange(planes,device=device)[:,None,None]-ids[None]).abs()
    second = volume.masked_fill((separation<=4)|~minima, float('inf')).min(0).values
    reliable = (best < .3) & (second-best>.02) & (ids>2) & (ids<planes-3)
    return {'depth':depths[ids], 'cost':best, 'margin':second-best, 'reliable':reliable,
            'depth_bounds':[float(depths[0]),float(depths[-1])], 'spacing':float(depths[1]-depths[0])}


@torch.no_grad()
def consistency(reference, sources, result, source_results, radius):
    points = world_points(reference,result['depth'])
    votes = torch.zeros_like(result['depth'],dtype=torch.int32)
    h,w = result['depth'].shape
    y,x = torch.meshgrid(torch.arange(h,device=points.device)+.5,
                          torch.arange(w,device=points.device)+.5,indexing='ij')
    pixels = torch.stack((x,y),-1)
    for source, other in zip(sources,source_results):
        grid, z = project_world(points,source)
        fields = torch.stack((other['depth'],other['reliable'].float()))[None]
        queried = F.grid_sample(fields,grid[None],align_corners=False)[0]
        uv = (grid+1)*grid.new_tensor([w,h])/2
        rays = torch.cat((uv,torch.ones_like(uv[...,:1])),-1)@torch.linalg.inv(source['K']).T
        back_world = (rays*queried[0,...,None]-source['viewmat'][:3,3])@source['viewmat'][:3,:3]
        back_grid,_ = project_world(back_world,reference)
        error = (((back_grid+1)*grid.new_tensor([w,h])/2-pixels).square().sum(-1)).sqrt()
        valid = (queried[1]>.99) & ((z-queried[0]).abs()<.01*radius) & (error<1.)
        votes += valid.int()
    return votes


def run_probe(scene, output, resolution=256, anchors=4):
    torch.set_num_threads(8)
    torch.manual_seed(0)
    out = Path(output); out.mkdir(parents=True,exist_ok=False)
    dataset = SceneDataset(scene,'train',resolution)
    fit, validation = split_train_lights(dataset.frames)
    samples = {i:dataset[i] for i in fit}
    center,radius = camera_bounds(list(samples.values())[::max(1,len(fit)//32)])
    neighbors,pairs = select_neighbors(samples,center)
    selected = [fit[k] for k in np.linspace(0,len(fit)-1,anchors,dtype=int)]
    needed = sorted(set(selected+[j for i in selected for j in neighbors[i]]))
    gpu_needed = set(needed+[j for i in needed for j in neighbors[i]])
    gpu = {i:to_device(samples[i],'cuda') for i in gpu_needed}
    config = dict(scene=str(dataset.scene_path.resolve()),resolution=resolution,planes=384,
                  anchors=selected,fit_indices=fit,validation_indices=validation,
                  depth_views=needed,pairs=pairs,world_radius=radius,world_center=center.tolist(),
                  source_count=4,aggregate_best=3,window=9,ncc_min=.7,margin_min=.02,
                  uniqueness_exclusion_bins=4,consistency_pixels=1.,consistency_radius=.01,
                  uniqueness_comparison='distinct local minima and interval endpoints',
                  acceptance_votes=3,geometry_teacher=False,sdf=False,
                  note='Matching feasibility only, not CoMVS-GS reproduction or depth ground truth')
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    with tarfile.open(out/'source.tar','w') as archive:
        for name in ['multiview_geometry.py','data.py','gaussians.py','diagnose_image_errors.py']:
            archive.add(Path(__file__).parent/name,arcname=name)
    results = {}; started=time.monotonic()
    for i in needed:
        results[i] = plane_sweep(gpu[i],[gpu[j] for j in neighbors[i]],center.cuda(),radius)
        print(json.dumps({'event':'depth','frame':i,'done':len(results),'total':len(needed),
                          'seconds':time.monotonic()-started}),flush=True)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(selected),4,figsize=(12,3*len(selected)),squeeze=False)
    rows=[]; points=[]; colors=[]
    for row,i in enumerate(selected):
        result=results[i];sample=gpu[i]
        votes=consistency(sample,[gpu[j] for j in neighbors[i]],result,[results[j] for j in neighbors[i]],radius)
        foreground=sample['alpha'][...,0]>.9
        accepted=result['reliable'] & (votes>=3) & foreground
        count=int(foreground.sum())
        rows.append(dict(frame=i,foreground_pixels=count,matched_pixels=int((result['reliable']&foreground).sum()),
                         consistent_pixels=int(accepted.sum()),consistent_fraction=float(accepted.sum()/max(1,count)),
                         depth_spacing=result['spacing'],selected_pairs=pairs[i]))
        points.append(world_points(sample,result['depth'])[accepted].cpu())
        colors.append(sample['image'][accepted].cpu())
        depth=(result['depth']-result['depth_bounds'][0])/(2*radius)
        panels=[sample['image'],depth*foreground,result['reliable'].float(),accepted.float()]
        for ax,values,label in zip(axes[row],panels,['Observed','Best depth (unfiltered)','NCC + uniqueness','3-view geometric check']):
            ax.imshow(values.cpu().numpy(),vmin=0,vmax=1,cmap='viridis' if values.ndim==2 else None)
            ax.set_title(f'{i}: {label}');ax.axis('off')
        # Preserve only anchor depth diagnostics, not every cost volume.
        torch.save({k:v.cpu() if torch.is_tensor(v) else v for k,v in dict(result,votes=votes,accepted=accepted).items()},out/f'depth_{i:04d}.pt')
    fig.tight_layout();fig.savefig(out/'matching_panel.png',dpi=140);plt.close(fig)
    xyz,rgb=torch.cat(points).numpy(),torch.cat(colors).numpy()
    from plyfile import PlyData,PlyElement
    vertices=np.empty(len(xyz),dtype=[('x','f4'),('y','f4'),('z','f4'),('red','u1'),('green','u1'),('blue','u1')])
    for k,name in enumerate(['x','y','z']):vertices[name]=xyz[:,k]
    for k,name in enumerate(['red','green','blue']):vertices[name]=(rgb[:,k]*255).round().clip(0,255).astype(np.uint8)
    PlyData([PlyElement.describe(vertices,'vertex')]).write(str(out/'matched_points.ply'))
    report=dict(status='completed',rows=rows,accepted_points=len(xyz),seconds=time.monotonic()-started,
                consistent_fraction=sum(r['consistent_pixels'] for r in rows)/sum(r['foreground_pixels'] for r in rows),
                limitations=['Frontoparallel plane sweep is simpler than full PatchMatch.',
                             'Consistency is not independent geometric accuracy; no ground truth mesh is available.',
                             'Four predetermined train anchors cannot establish whole-scene quality.'])
    (out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    return report
