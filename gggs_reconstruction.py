"""Read-only GGGS checkpoint import and continuous-depth diagnostics for relighting."""
from pathlib import Path
from types import ModuleType, SimpleNamespace
import argparse
import math
import sys

import torch
from torch.nn import functional as F

from native_reconstruction import projection_from_K, depth_world_normals, reconstruction_coordinates

ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT/'third_party/Geometry-Grounded-Gaussian-Splatting'


@torch.no_grad()
def relighting_state(saved):
    """Filtered GGGS geometry in the default renderer's world/log parameterization.

    This preserves the 3D covariance and opacity compensation, not the author's
    radial sorting, EWA footprint or continuous-depth rendering equations.
    """
    if saved['kind'] != 'gggs_core':
        raise ValueError('Expected a GGGS geometry checkpoint')
    scale = saved['config']['training_coordinate_transform']['scale']
    raw = saved['capture'][5].exp()
    filt = saved['world_filter_3D'] / scale
    filtered = torch.sqrt(raw.square() + filt.square())
    coefficient = torch.sqrt(raw.square().prod(-1) / filtered.square().prod(-1))
    center = saved['gaussians']['center']
    return {'center': center, 'params.means': saved['capture'][2] * scale + center,
            'params.scales': (filtered * scale).log(),
            'params.quats': F.normalize(saved['capture'][6], dim=-1),
            'params.opacities': torch.logit(saved['capture'][7][:, 0].sigmoid() * coefficient)}


def author_modules():
    # Load the author's model package without running its unrelated dataset
    # loader (which imports Open3D). All model/appearance modules are real files.
    if 'scene' not in sys.modules:
        package=ModuleType('scene');package.__path__=[str(UPSTREAM/'scene')]
        sys.modules['scene']=package
    for path in [ROOT/'third_party/python',
                 ROOT/'third_party/2d-gaussian-splatting/submodules/simple-knn',
                 UPSTREAM/'submodules/diff-gaussian-rasterization',UPSTREAM]:
        sys.path.insert(0,str(path))
    from scene.gaussian_model import GaussianModel
    from arguments import OptimizationParams
    from gaussian_renderer import render
    parser=argparse.ArgumentParser()
    opt=OptimizationParams(parser).extract(parser.parse_args([]))
    return GaussianModel,opt,render


def centered_canvas(sample):
    """Recenter by rendering a larger virtual sensor; preserve original rays.

    Upstream CUDA assumes centered principal points even when its projection
    matrix is off-center. Sampling the centered canvas avoids changing CUDA.
    There is one bilinear resampling of rendered fields, no input resampling.
    """
    h,w=sample['image'].shape[:2];K=sample['K']
    cx,cy=float(K[0,2]),float(K[1,2])
    width=2*math.ceil(max(cx,w-cx));height=2*math.ceil(max(cy,h-cy))
    ck=K.clone();ck[0,2]=width/2;ck[1,2]=height/2
    y,x=torch.meshgrid(torch.arange(h,device=K.device)+.5,
                        torch.arange(w,device=K.device)+.5,indexing='ij')
    grid=torch.stack(((x-cx+width/2)/width*2-1,(y-cy+height/2)/height*2-1),-1)
    camera=SimpleNamespace(image_width=width,image_height=height,
        R=sample['viewmat'][:3,:3].T,T=sample['viewmat'][:3,3],Fx=float(K[0,0]),Fy=float(K[1,1]),
        FoVx=2*math.atan(width/(2*float(K[0,0]))),FoVy=2*math.atan(height/(2*float(K[1,1]))),
        world_view_transform=sample['viewmat'].T.contiguous(),
        full_proj_transform=(projection_from_K(ck,width,height)@sample['viewmat']).T.contiguous(),
        camera_center=sample['c2w'][:3,3])
    return camera,grid


def render_geometry(model,sample,render,background=0.,require_depth=True):
    camera,grid=centered_canvas(sample)
    pkg=render(camera,model,SimpleNamespace(debug=False),model.get_xyz.new_full((3,),background),
               kernel_size=0.,require_depth=require_depth)
    def crop(value):
        return F.grid_sample(value[None],grid[None],align_corners=False)[0]
    rgb=crop(pkg['render']).permute(1,2,0)
    alpha=crop(pkg['mask'])[0]
    depth=crop(pkg['median_depth'])[0]
    normal=crop(pkg['normal']).permute(1,2,0)@sample['viewmat'][:3,:3]
    surf=depth_world_normals(sample,depth)*alpha.detach()[...,None]
    return dict(rgb=rgb,alpha=alpha,depth=depth,normal=normal,surface_normal=surf,
                screen=pkg['viewspace_points'],visible=pkg['visibility_filter'],radii=pkg['radii'])


def restore_model(Model,saved):
    model=Model(3,0)
    model.restore(saved['capture'],SimpleNamespace(**saved['config']['optimizer']))
    with torch.no_grad():
        for target,source in [('_xyz','means'),('_scaling','scales'),('_rotation','quats'),('_opacity','opacities')]:
            value=saved['gaussians']['params.'+source]
            if target=='_opacity':value=value[:,None]
            getattr(model,target).copy_(value)
    if saved['config'].get('mip_filter',False):
        model.filter_3D=saved['world_filter_3D']
    else:
        model.reset_3D_filter()
    return model


def restore_evaluation_model(Model,saved):
    """Keep the exact coordinate scale in which continuous depth was optimized."""
    model=Model(3,0)
    model.restore(saved['capture'],SimpleNamespace(**saved['config']['optimizer']))
    if saved['config']['mip_filter']:
        model.filter_3D=saved['world_filter_3D']/saved['config']['training_coordinate_transform']['scale']
    else:
        model.reset_3D_filter()
    return model


def render_evaluation_geometry(model,sample,render,config):
    transform=config['training_coordinate_transform']
    center=sample['viewmat'].new_tensor(transform['center']);scale=transform['scale']
    local=reconstruction_coordinates(sample,center,scale)
    result=render_geometry(model,local,render,config['background'])
    result['depth']=result['depth']*scale
    return result




