"""GT-only local peak/ring draws and raw observation-RGB pair supervision."""

import torch

from sdf_volume import select_rays


_COMPONENT_BUCKETS = {'1_to_4': (1, 4), '5_to_16': (5, 16), 'gt16': (17, None)}


@torch.no_grad()
def build_residual_pair_pool(rays, peak_mask):
    """Cache eligible GT anchors and their own 11x11 nonpeak foreground rings.

    ``rays`` follows ``silhouette_rays``: valid flat indices, GT alpha target,
    and image shape. Every GT peak is excluded from rings, even invalid peaks.
    Components describe the full GT mask; they do not change draw probabilities.
    """
    import cv2
    import numpy as np

    height, width = rays['shape']
    device = rays['valid'].device
    peaks = peak_mask.reshape(-1)
    anchors = rays['valid'][peaks[rays['valid']]]
    valid_peak_pixels = len(anchors)
    allowed = torch.zeros(height*width, dtype=torch.bool, device=device)
    allowed[rays['valid']] = True
    allowed &= (rays['target'] > .9) & ~peaks
    dy, dx = torch.meshgrid(torch.arange(-5, 6, device=device),
                            torch.arange(-5, 6, device=device), indexing='ij')
    y = anchors[:, None]//width + dy.reshape(1, -1)
    x = anchors[:, None]%width + dx.reshape(1, -1)
    candidates = (y*width+x).clamp(0, height*width-1)
    in_ring = ((y >= 0) & (y < height) & (x >= 0) & (x < width)
               & allowed[candidates])
    counts = in_ring.sum(-1)
    eligible = counts > 0
    anchors, counts = anchors[eligible], counts[eligible]
    rings = candidates[eligible][in_ring[eligible]]

    component_count, labels, component_stats, _ = cv2.connectedComponentsWithStats(
        peak_mask.reshape(height, width).cpu().numpy().astype(np.uint8), connectivity=8)
    areas = component_stats[:, cv2.CC_STAT_AREA]
    anchor_components = labels.reshape(-1)[anchors.cpu().numpy()]
    eligible_components = np.unique(anchor_components)
    component_buckets = np.full(component_count, -1, dtype=np.int64)
    buckets = {}
    for bucket_index, (name, (low, high)) in enumerate(_COMPONENT_BUCKETS.items()):
        ids = np.flatnonzero((areas >= low) & ((areas <= high) if high else True))
        ids = ids[ids != 0]  # Label zero is background regardless of its area.
        component_buckets[ids] = bucket_index
        buckets[name] = {
            'components': int(len(ids)),
            'gt_peak_pixels': int(areas[ids].sum()),
            'eligible_components': int(np.isin(ids, eligible_components).sum()),
            'eligible_anchors': int(np.isin(anchor_components, ids).sum()),
        }
    return {
        'anchors': anchors,
        'rings': rings,
        'ring_counts': counts,
        'ring_offsets': counts.cumsum(0)-counts,
        'anchor_components': torch.as_tensor(anchor_components, device=device, dtype=torch.long),
        'component_buckets': torch.as_tensor(component_buckets, device=device),
        'stats': {'gt_peak_pixels': int(peaks.sum()), 'valid_peak_pixels': valid_peak_pixels,
                  'eligible_anchors': len(anchors), 'component_buckets': buckets},
    }


@torch.no_grad()
def select_residual_pairs(rays, count, generator, pool):
    """Return repeated RGB draws and their flat-pixel (anchor, ring) pairs.

    A quarter of draws are anchors, a quarter their rings, then equal current
    foreground/valid quotas. A frame with no eligible anchor uses the existing
    foreground/valid sampler for the entire budget and returns no fake pairs.
    """
    if count <= 0 or count % 4:
        raise ValueError('paired residual ray count must be positive and divisible by four')
    anchors = pool['anchors']
    if not len(anchors):
        return select_rays(rays, count, generator), anchors.new_empty((0, 2))
    slots = count//4
    rows = torch.randint(len(anchors), (slots,), device=anchors.device, generator=generator)
    ring_choices = (torch.rand(slots, device=anchors.device, generator=generator)
                    * pool['ring_counts'][rows]).long()
    peaks = anchors[rows]
    rings = pool['rings'][pool['ring_offsets'][rows]+ring_choices]
    base = select_rays(rays, 2*slots, generator)
    return torch.cat((peaks, rings, base)), torch.stack((peaks, rings), -1)


@torch.no_grad()
def residual_pair_sample_stats(pool, pairs):
    """Record component coverage without using components to choose anchors."""
    rows = torch.searchsorted(pool['anchors'], pairs[:, 0].contiguous())
    components = pool['anchor_components'][rows]
    bucket_indices = pool['component_buckets'][components]
    buckets = {}
    for index, name in enumerate(_COMPONENT_BUCKETS):
        selected = bucket_indices == index
        buckets[name] = {
            'anchor_draws': int(selected.sum()),
            'sampled_peak_pixels': int(pairs[selected, 0].unique().numel()),
            'component_ids': components[selected].unique().cpu().tolist(),
        }
    return {'component_buckets': buckets}


def residual_pair_loss(predicted, target, alpha, pairs, nominal_pair_count):
    """Raw RGB difference L1, normalized by all nominal slots and 3 channels.

    Both endpoints must have positive frozen GS alpha. Excluded pairs retain
    their ordinary RGB draws; neither support filtering nor repeats renormalize
    the pair term. Empty pairs/support yield a prediction-connected zero.
    """
    if nominal_pair_count <= 0:
        raise ValueError('nominal pair count must be positive')
    peak, ring = pairs.unbind(-1)
    covered = alpha.reshape(-1) > 0
    supported = covered[peak] & covered[ring]
    prediction, observation = predicted.reshape(-1, 3), target.reshape(-1, 3)
    difference = ((prediction[peak]-prediction[ring])
                  - (observation[peak]-observation[ring]))
    return difference[supported].abs().sum()/(nominal_pair_count*3), supported
