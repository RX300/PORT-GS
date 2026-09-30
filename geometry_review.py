"""Common-native-renderer geometry comparison and image-correspondence audit."""
from pathlib import Path
import json
import cv2
import numpy as np
import torch
from torch.nn import functional as F

from data import SceneDataset
from evaluate import to_device,target_image


def continuous_review_model(Model, state, source_config):
    """Replay baked world geometry in the author's original numerical coordinates."""
    model=Model(0,0);means=state['params.means'];n=len(means)
    transform=source_config['training_coordinate_transform']
    center=means.new_tensor(transform['center']);scale=float(transform['scale'])
    model._xyz=((means-center)/scale).contiguous()
    model._scaling=(state['params.scales']-np.log(scale)).contiguous()
    model._rotation=state['params.quats'];model._opacity=state['params.opacities'][:,None].contiguous()
    model._features_dc=means.new_zeros(n,1,3);model._features_rest=means.new_zeros(n,0,3)
    model._sg_axis=means.new_zeros(n,0,3);model._sg_sharpness=means.new_zeros(n,0)
    model._sg_color=means.new_zeros(n,0,3)
    model.reset_3D_filter()  # The source filter is already baked into the imported covariance/opacity.
    return model


@torch.no_grad()
def review_default_geometry(checkpoint, output):
    """Separate renderer conversion from geometry changes on three fixed train views."""
    import matplotlib.pyplot as plt
    from evaluate import load_model
    from gggs_reconstruction import (author_modules as gggs_modules, relighting_state,
        restore_evaluation_model, render_evaluation_geometry)
    from native_reconstruction import depth_world_normals, boundary_metrics
    from renderer import _rasterize
    torch.set_num_threads(8)
    Model,_,backend=gggs_modules()
    final,_,saved=load_model(checkpoint);cfg=saved['config']
    if cfg.get('init_geometry_format')!='gggs':raise ValueError('Expected GGGS-initialized default checkpoint')
    source=torch.load(cfg['init_geometry'],map_location='cuda',weights_only=False)
    initial=relighting_state(source);model=restore_evaluation_model(Model,source)
    initial_replay=continuous_review_model(Model,initial,source['config'])
    final_replay=continuous_review_model(Model,final.state_dict(),source['config'])
    for a,b in zip((model.get_xyz,model.get_rotation,*model.get_scaling_n_opacity_with_3D_filter),
                   (initial_replay.get_xyz,initial_replay.get_rotation,*initial_replay.get_scaling_n_opacity_with_3D_filter)):
        torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-7)
    inputs=dict(means=initial['params.means'],scales=initial['params.scales'].exp(),
                quats=initial['params.quats'],opacities=initial['params.opacities'].sigmoid())
    data=SceneDataset(cfg['scene'],'train',cfg['resolution'])
    indices=[0,len(data)//2,len(data)-1]
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    fig,axes=plt.subplots(3,5,figsize=(20,12),squeeze=False)
    direction=F.normalize(torch.tensor([.4,-.5,.8],device='cuda'),dim=0)
    rows=[]
    for row,index in enumerate(indices):
        sample=to_device(data[index],'cuda');h,w=sample['image'].shape[:2]
        results={'GGGS continuous depth':render_evaluation_geometry(model,sample,backend,source['config'])}
        replay=render_evaluation_geometry(initial_replay,sample,backend,source['config'])
        original=results['GGGS continuous depth']
        valid=(original['alpha']>.9)&(replay['alpha']>.9)&(original['depth']>0)&(replay['depth']>0)
        error=(original['depth'][valid]-replay['depth'][valid]).abs()/source['radius']
        alpha_error=(original['alpha']-replay['alpha']).abs()
        handoff=dict(alpha_max_abs=float(alpha_error.max()),alpha_mean_abs=float(alpha_error.mean()),
            alpha_p99_abs=float(alpha_error.quantile(.99)),physical_parameter_check=True,
            depth_median_abs_over_radius=float(error.median()),depth_p99_abs_over_radius=float(error.quantile(.99)),
            compared_depth_pixels=int(valid.sum()))
        # Roundtripping world/log parameters can cross discrete EWA cutoffs.
        # Bound the worst alpha change by one byte and also require small pooled error.
        if (handoff['alpha_max_abs']>1/255 or handoff['alpha_mean_abs']>1e-5
                or handoff['alpha_p99_abs']>1e-4 or handoff['depth_p99_abs_over_radius']>1e-3):
            raise ValueError(f'Continuous-depth geometry roundtrip failed: {handoff}')
        for label,values in [('Imported / default depth',inputs),('Joint final / default depth',final.raster_inputs())]:
            rendered,alpha,_=_rasterize('3dgs',**values,colors=torch.ones_like(values['means']),
                viewmats=sample['viewmat'][None],Ks=sample['K'][None],width=w,height=h,
                backgrounds=values['means'].new_zeros(1,3),render_mode='RGB+ED',packed=False)
            depth=rendered[0,...,-1];a=alpha[0,...,0]
            results[label]=dict(alpha=a,depth=depth,surface_normal=depth_world_normals(sample,depth))
        results['Joint final / GGGS depth']=render_evaluation_geometry(final_replay,sample,backend,source['config'])
        axes[row,0].imshow(target_image(sample,cfg['background'],cfg['display_gamma']).clamp(0,1).cpu())
        axes[row,0].set_title(f'Observed RGB | train {index}')
        metrics={}
        for col,(label,result) in enumerate(results.items(),1):
            a=result['alpha'];n=F.normalize(result['surface_normal'],dim=-1)
            clay=(.2+.8*(n*direction).sum(-1).abs())*a
            axes[row,col].imshow(clay.cpu(),cmap='gray',vmin=0,vmax=1)
            axes[row,col].set_title(f'{label} | train {index}')
            gt=sample['alpha'][...,0]>.9;mask=a>.5
            metrics[label]=dict(silhouette_IoU=float((gt&mask).sum()/(gt|mask).sum()),
                alpha_L1=float((a-sample['alpha'][...,0]).abs().mean()),**boundary_metrics(result,sample))
        for ax in axes[row]:ax.axis('off')
        rows.append(dict(frame_index=index,metrics=metrics,initial_continuous_roundtrip=handoff))
    fig.suptitle(f"{cfg['representation']} | joint step {saved['step']} | GGGS initialization")
    fig.tight_layout();fig.savefig(out/'geometry.png',dpi=140);plt.close(fig)
    report=dict(checkpoint=str(checkpoint),representation=cfg['representation'],step=saved['step'],
        source=cfg['init_geometry'],split='train',fixed_indices=indices,
        initial_points=len(inputs['means']),final_points=len(final.params['means']),views=rows,
        scope='No geometry GT. Three fixed train views, both initial/final expected-center-depth and initial/final GGGS continuous-depth. Imported filter is baked exactly once. Initial continuous roundtrip checked. Silhouette and roughness are proxies, not shape accuracy.')
    (out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')


def compare_geometry_buffers(reference, candidate, output):
    """Compare fixed validation cameras without reloading either author backend."""
    import matplotlib.pyplot as plt
    from evaluate import target_image
    reference,candidate,output=Path(reference),Path(candidate),Path(output)
    output.mkdir(parents=True,exist_ok=False)
    old_report=json.loads((reference/'metrics.json').read_text())
    new_report=json.loads((candidate/'metrics.json').read_text())
    if old_report['split']!=new_report['split'] or old_report['selected_geometry_frames']!=new_report['selected_geometry_frames']:
        raise ValueError('Geometry comparison requires identical splits and fixed frame indices')
    old=dict(torch.load(reference/'geometry_views.pt',map_location='cpu',weights_only=False)['frames'])
    new=dict(torch.load(candidate/'geometry_views.pt',map_location='cpu',weights_only=False)['frames'])
    cfg=json.loads((candidate.parent/'config.json').read_text())
    old_cfg=json.loads((Path(old_report['checkpoint']).parent/'config.json').read_text())
    names={'native_2dgs':'Native 2DGS','gggs_core':'GGGS core','gaussian_wrapping':'Gaussian Wrapping'}
    old_name=names.get(old_cfg['representation'],old_cfg['representation'])
    new_name=names.get(cfg['representation'],cfg['representation'])
    if old_cfg.get('prior_weight',0):old_name+=' + StableNormal'
    if cfg.get('prior_weight',0):new_name+=' + StableNormal'
    if old_cfg.get('prior_normal_weight',0) or old_cfg.get('prior_depth_weight',0):old_name+=' + normal/depth priors'
    if cfg.get('prior_normal_weight',0) or cfg.get('prior_depth_weight',0):new_name+=' + normal/depth priors'
    data=SceneDataset(cfg['scene'],'train',cfg['resolution'])
    indices=new_report['selected_geometry_frames']
    direction=F.normalize(torch.tensor([.4,-.5,.8]),dim=0)
    for normal_key,filename in [('normal','geometry_comparison.png'),
                                ('surface_normal','depth_geometry_comparison.png')]:
        fig,axes=plt.subplots(len(indices),5,figsize=(15,3*len(indices)),squeeze=False)
        closeups=[]
        for row,i in enumerate(indices):
            target=target_image(data[i],cfg['background'])
            panels=[target];labels=['Observed RGB']
            for results,name in [(old,old_name),(new,new_name)]:
                result=results[i]
                normal=F.normalize(result[normal_key],dim=-1);alpha=result['alpha']
                if alpha.shape!=target.shape[:2]:raise ValueError('Image size mismatch')
                gray=(.2+.8*(normal*direction).sum(-1).abs())*alpha
                panels.extend([gray,(normal*.5+.5)*alpha[...,None]])
                labels.extend([name+'\nclay',name+'\nnormals'])
            if normal_key=='surface_normal':
                foreground=(data[i]['alpha'][...,0]>.05)|(old[i]['alpha']>.05)|(new[i]['alpha']>.05)
                yy,xx=torch.where(foreground)
                y0,y1=max(0,int(yy.min())-8),min(target.shape[0],int(yy.max())+9)
                x0,x1=max(0,int(xx.min())-8),min(target.shape[1],int(xx.max())+9)
                closeups.append((i,[panels[k][y0:y1,x0:x1] for k in [0,1,3]]))
            for ax,values,label in zip(axes[row],panels,labels):
                ax.imshow(values.numpy(),cmap='gray' if values.ndim==2 else None,vmin=0,vmax=1)
                ax.set_title(f'{i}: {label}',fontsize=9);ax.axis('off')
        fig.suptitle('Depth-derived surface normals' if normal_key=='surface_normal' else 'Rasterized geometric normals')
        fig.tight_layout();fig.savefig(output/filename,dpi=140);plt.close(fig)
        if closeups:
            fig,axes=plt.subplots(len(indices),3,figsize=(12,4*len(indices)),squeeze=False)
            for row,(i,panels) in enumerate(closeups):
                for ax,values,label in zip(axes[row],panels,['Observed RGB',old_name,new_name]):
                    ax.imshow(values.numpy(),cmap='gray' if values.ndim==2 else None,vmin=0,vmax=1)
                    ax.set_title(f'{i}: {label}',fontsize=10);ax.axis('off')
            fig.suptitle('Depth-derived clay; shared crop includes observed and both predicted silhouettes')
            fig.tight_layout();fig.savefig(output/'depth_clay_closeup.png',dpi=140);plt.close(fig)
    report=dict(reference=str(reference.resolve()),candidate=str(candidate.resolve()),
        reference_metrics=old_report['metrics'],candidate_metrics=new_report['metrics'],
        split=new_report['split'],evaluated_frames=new_report['evaluated_frames'],fixed_frames=indices,
        note='Original cameras, fixed shared directional gray shading, predicted alpha; no geometry alignment or GT silhouette clipping. No geometry GT. Separate panels use rasterized geometric normals and depth-derived surface normals.')
    (output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    return report




def correspondence_pair(source,target):
    detector=cv2.SIFT_create(nfeatures=3000)
    def detect(sample):
        image=(sample['image'].cpu().numpy()*255).round().clip(0,255).astype(np.uint8)
        gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        mask=(sample['alpha'][...,0].cpu().numpy()>.9).astype(np.uint8)*255
        return detector.detectAndCompute(gray,mask)
    ka,da=detect(source);kb,db=detect(target)
    if da is None or db is None:return np.empty((0,2),np.float32),np.empty((0,2),np.float32),{'reason':'no descriptors'}
    matches=[a for pair in cv2.BFMatcher().knnMatch(da,db,k=2) if len(pair)==2 for a,b in [pair] if a.distance<.7*b.distance]
    if len(matches)<16:return np.empty((0,2),np.float32),np.empty((0,2),np.float32),{'ratio_matches':len(matches),'reason':'too few'}
    a=np.float32([ka[m.queryIdx].pt for m in matches]);b=np.float32([kb[m.trainIdx].pt for m in matches])
    cv2.setRNGSeed(0)
    matrix,mask=cv2.findFundamentalMat(a,b,cv2.FM_RANSAC,1.,.999)
    if matrix is None or matrix.shape!=(3,3):return a[:0],b[:0],{'ratio_matches':len(matches),'reason':'F failed'}
    inliers=mask.ravel().astype(bool)
    return a[inliers],b[inliers],{'ratio_matches':len(matches),'image_F_inliers':int(inliers.sum()),
        'scope':'Classical image-F filtered matches; changing light/repeated texture may still cause false matches.'}


def feature_reprojection(source,target,result,xy):
    xy=torch.tensor(xy,device=result['depth'].device)
    h,w=result['depth'].shape
    grid=((xy+.5)/xy.new_tensor([w,h])*2-1).reshape(1,1,-1,2)
    fields=torch.stack((result['depth'],result['alpha']))
    values=F.grid_sample(fields[None],grid,align_corners=False)[0,:,0].T
    ray=torch.cat((xy+.5,torch.ones_like(xy[:,:1])),-1)@torch.linalg.inv(source['K']).T
    source_camera=ray*values[:,:1]
    world=(source_camera-source['viewmat'][:3,3])@source['viewmat'][:3,:3]
    camera=world@target['viewmat'][:3,:3].T+target['viewmat'][:3,3]
    projected=camera@target['K'].T
    projected=projected[:,:2]/projected[:,2:].clamp_min(1e-8)-.5
    valid=(values[:,0]>0)&(values[:,1]>.9)&(camera[:,2]>.01)
    return projected,valid


def calibrated_epipolar_error(source,target,a,b):
    """Independent camera check against image-F inliers, in pixel units."""
    view=target['viewmat']@torch.linalg.inv(source['viewmat'])
    t=view[:3,3]
    cross=t.new_zeros(3,3)
    cross[0,1],cross[0,2]=-t[2],t[1]
    cross[1,0],cross[1,2]=t[2],-t[0]
    cross[2,0],cross[2,1]=-t[1],t[0]
    matrix=torch.linalg.inv(target['K']).T@cross@view[:3,:3]@torch.linalg.inv(source['K'])
    x=torch.tensor(a,device=t.device)+.5;y=torch.tensor(b,device=t.device)+.5
    x=torch.cat((x,torch.ones_like(x[:,:1])),-1)
    y=torch.cat((y,torch.ones_like(y[:,:1])),-1)
    line_y=x@matrix.T;line_x=y@matrix
    numerator=(y*line_y).sum(-1).abs()
    error=.5*numerator*(line_y[:,:2].norm(dim=-1).clamp_min(1e-10).reciprocal()
                       +line_x[:,:2].norm(dim=-1).clamp_min(1e-10).reciprocal())
    return {'median_pixels':float(error.median()),'p90_pixels':float(torch.quantile(error,.9)),
            'baseline_world':float(t.norm())}


