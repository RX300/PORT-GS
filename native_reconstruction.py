"""Shared camera/depth geometry utilities; no reconstruction training backend."""
import math
import numpy as np
import torch
from torch.nn import functional as F


def projection_from_K(K, width, height, near=.01, far=100.):
    """Native ndc2pix maps to integer indices; K describes i+.5 pixel centers."""
    p = K.new_zeros((4, 4))
    p[0, 0], p[1, 1] = 2*K[0, 0]/width, 2*K[1, 1]/height
    p[0, 2], p[1, 2] = 2*K[0, 2]/width-1, 2*K[1, 2]/height-1
    p[2, 2], p[2, 3], p[3, 2] = far/(far-near), -far*near/(far-near), 1
    return p


def depth_points(sample, depth):
    h, w = depth.shape
    y, x = torch.meshgrid(torch.arange(h, device=depth.device)+.5,
                          torch.arange(w, device=depth.device)+.5, indexing='ij')
    ray = torch.stack((x, y, torch.ones_like(x)), -1) @ torch.linalg.inv(sample['K']).T
    camera = ray*depth[..., None]
    view = sample['viewmat']
    return (camera-view[:3, 3]) @ view[:3, :3]


def depth_world_normals(sample, depth):
    xyz = depth_points(sample, depth)
    dy = xyz[2:, 1:-1]-xyz[:-2, 1:-1]
    dx = xyz[1:-1, 2:]-xyz[1:-1, :-2]
    n = F.normalize(torch.linalg.cross(dy, dx), dim=-1)
    return F.pad(n.permute(2, 0, 1), (1, 1, 1, 1)).permute(1, 2, 0)


def reconstruction_coordinates(sample, center, scale):
    """Apply x_world = scale*x_local + center without changing image rays."""
    sample=dict(sample)
    view=sample['viewmat'].clone()
    view[:3,3]=(view[:3,:3]@center+view[:3,3])/scale
    sample['viewmat']=view
    pose=sample['c2w'].clone();pose[:3,3]=(pose[:3,3]-center)/scale
    sample['c2w']=pose
    return sample


def boundary_metrics(result,sample):
    """Silhouette agreement and boundary depth-normal roughness, not GT shape."""
    import cv2
    gt=(sample['alpha'][...,0]>.9).cpu().numpy().astype(np.uint8)
    pred=(result['alpha']>.5).cpu().numpy().astype(np.uint8)
    kernel=np.ones((3,3),np.uint8)
    gb=gt-cv2.erode(gt,kernel);pb=pred-cv2.erode(pred,kernel)
    dg=cv2.distanceTransform(1-gb,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    dp=cv2.distanceTransform(1-pb,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    outside=cv2.distanceTransform(1-gt,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    distance=outside[pb.astype(bool)]
    precision=float((dg[pb.astype(bool)]<=2).mean()) if pb.any() else 0.
    recall=float((dp[gb.astype(bool)]<=2).mean()) if gb.any() else 0.
    n=F.normalize(result['surface_normal'],dim=-1)
    band=torch.from_numpy((dg<=8)&(gt>0)).to(n.device)
    valid=band&(result['alpha']>.9)&(result['depth']>0)
    values=[]
    for axis in [0,1]:
        other=n.roll(1,axis);both=valid&valid.roll(1,axis)
        both[0,:]=False;both[:,0]=False
        angle=(n*other).sum(-1).clamp(-1,1).acos()*180/math.pi
        values.append(angle[both])
    values=torch.cat(values)
    return dict(boundary_F1_2px=2*precision*recall/max(precision+recall,1e-8),
        boundary_precision_2px=precision,boundary_recall_2px=recall,
        boundary_outward_fraction_over_2px=float((distance>2).mean()) if len(distance) else 0.,
        boundary_outward_p95_px=float(np.quantile(distance,.95)) if len(distance) else None,
        boundary_depth_normal_roughness_p95_deg=float(torch.quantile(values,.95)) if len(values) else None)
