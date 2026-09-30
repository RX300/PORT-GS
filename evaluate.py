"""Evaluate saved models using their training and held-out camera contracts."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from gaussians import Gaussians
from methods import build_transport, resolve_config
from renderer import render


def ssim(x, y, core=None):
    """Existing SSIM map, optionally averaged over (y0,y1,x0,x1).

    Bounds are half-open in the supplied image. A padded patch can therefore
    retain its convolution halo while contributing only its central pixels.
    With core=None the original arithmetic and whole-image mean are unchanged.
    """
    x, y = x.permute(2, 0, 1)[None], y.permute(2, 0, 1)[None]
    axis = torch.arange(11, device=x.device, dtype=x.dtype) - 5
    kernel = torch.exp(-axis.square() / (2 * 1.5**2))
    kernel = kernel / kernel.sum()
    window = (kernel[:, None] * kernel[None, :])[None, None].expand(3, 1, 11, 11)
    mx, my = F.conv2d(x, window, padding=5, groups=3), F.conv2d(y, window, padding=5, groups=3)
    vx = F.conv2d(x * x, window, padding=5, groups=3) - mx * mx
    vy = F.conv2d(y * y, window, padding=5, groups=3) - my * my
    cov = F.conv2d(x * y, window, padding=5, groups=3) - mx * my
    score = (
        ((2 * mx * my + 0.01**2) * (2 * cov + 0.03**2))
        / ((mx * mx + my * my + 0.01**2) * (vx + vy + 0.03**2))
    )
    if core is not None:
        y0, y1, x0, x1 = core
        if not (0 <= y0 < y1 <= score.shape[-2] and 0 <= x0 < x1 <= score.shape[-1]):
            raise ValueError('SSIM core must be a nonempty region inside the supplied image')
        score = score[..., y0:y1, x0:x1]
    return score.mean()


def shift_images(image, shifts, background=0.0):
    """Bilinear sub-pixel translations of one HWC image; returns [N,H,W,C].

    ``shifts`` is [N,2] as (dy, dx) in pixels: content moves down/right for
    positive values. Uncovered pixels take the constant background.
    """
    h, w = image.shape[:2]
    theta = image.new_zeros((len(shifts), 2, 3))
    theta[:, 0, 0] = theta[:, 1, 1] = 1
    theta[:, 0, 2] = -2 * shifts[:, 1] / w
    theta[:, 1, 2] = -2 * shifts[:, 0] / h
    grid = F.affine_grid(theta, (len(shifts), image.shape[2], h, w), align_corners=False)
    source = (image - background).permute(2, 0, 1)[None].expand(len(shifts), -1, -1, -1)
    moved = F.grid_sample(source, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    return moved.permute(0, 2, 3, 1) + background


def best_translation(predicted, target, radius=24, background=0.0, chunk=96):
    """Global image translation minimizing MSE: integer search, then 1/8-pixel refinement."""
    def search(candidates):
        errors = []
        for part in candidates.split(chunk):
            moved = shift_images(predicted, part, background)
            errors.append((moved - target).square().flatten(1).mean(-1))
        errors = torch.cat(errors)
        return candidates[errors.argmin()]
    axis = torch.arange(-radius, radius + 1, device=predicted.device, dtype=predicted.dtype)
    grid = torch.stack(torch.meshgrid(axis, axis, indexing='ij'), -1).reshape(-1, 2)
    best = search(grid)
    fine = torch.arange(-8, 9, device=predicted.device, dtype=predicted.dtype) / 8
    fine = torch.stack(torch.meshgrid(fine, fine, indexing='ij'), -1).reshape(-1, 2)
    return search(best + fine)


def shift_aligned_metrics(x, y, radius, background=0.0, lpips_model=None):
    """Secondary protocol: metrics after the best global 2D translation of the render.

    x and y are display-domain HWC images in [0,1] (already quantized when the
    primary metrics are). The translation is fitted against the GT view, so this
    is a calibration-compensated diagnostic, never the primary fixed-camera score.
    """
    shift = best_translation(x, y, radius, background)
    moved = shift_images(x, shift[None], background)[0].clamp(0, 1)
    moved = (moved * 255).round() / 255
    item = {'dy': shift[0].item(), 'dx': shift[1].item(),
            'PSNR': (-10 * torch.log10((moved - y).square().mean())).item(),
            'SSIM': ssim(moved, y).item()}
    if lpips_model is not None:
        item['LPIPS'] = lpips_model(moved.permute(2, 0, 1)[None] * 2 - 1,
                                    y.permute(2, 0, 1)[None] * 2 - 1).item()
    return item


def calibrate_view(gaussians, transport, sample, target, background, shadow, port_active, display_gamma,
                   shadow_mode, steps, light=True, surface_field=None, normal_field=None):
    """Secondary literature-style protocol: fit one view's camera rotation (and light offset) on its GT.

    The scene, materials and networks stay frozen; only a camera-frame rotation about the
    calibrated center (3 values) and optionally a world light-position offset (3 values,
    object-radius units) are optimized with the training image loss. Published real-scene
    baselines calibrate test views on test images; this reproduces that protocol for
    comparison and is never the primary fixed-calibration metric.
    """
    from cameras import TrainCameraRotations, TrainLightOffsets
    device = gaussians.center.device
    camera = TrainCameraRotations(1, device=device)
    parameters = list(camera.parameters())
    lights = None
    if light:
        lights = TrainLightOffsets(1, gaussians.radius, device=device)
        parameters += list(lights.parameters())
    optimizer = torch.optim.SparseAdam(parameters, lr=1e-3)
    frozen = [value for value in list(gaussians.parameters()) + list(transport.parameters()) if value.requires_grad]
    for value in frozen:
        value.requires_grad_(False)
    target = target.clamp(0, 1)

    def corrected():
        view = camera.correct(sample, 0)
        return lights.correct(view, 0) if lights is not None else view
    try:
        # evaluate_samples runs under no_grad; only these calibration parameters need gradients.
        with torch.enable_grad():
            for step in range(steps):
                optimizer.param_groups[0]['lr'] = 1e-3 * (1e-2 ** (step / max(1, steps - 1)))
                optimizer.zero_grad(set_to_none=True)
                predicted, _, _ = render_observation(gaussians, transport, corrected(), background, shadow,
                                                     port_active, display_gamma, shadow_mode,
                                                     surface_field=surface_field, normal_field=normal_field)
                loss = 0.8 * (predicted - target).abs().mean() + 0.2 * (1 - ssim(predicted.clamp(0, 1), target))
                loss.backward()
                optimizer.step()
    finally:
        for value in frozen:
            value.requires_grad_(True)
    with torch.no_grad():
        view = corrected()
        stats = {'rotation_deg': (camera.rotation.weight[0] * 180 / torch.pi).tolist()}
        if lights is not None:
            stats['light_offset'] = (gaussians.radius * lights.offset.weight[0]).tolist()
    return {key: value.detach() if isinstance(value, torch.Tensor) else value for key, value in view.items()}, stats


def to_device(sample, device):
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in sample.items()
    }


def neutral_peak_mask(target, alpha):
    """GT-only bright/desaturated local peaks, an image proxy for highlights."""
    neutral = target.amin(-1)
    smooth = F.avg_pool2d(neutral[None,None],11,1,5)[0,0]
    saturation = (target.amax(-1)-neutral)/target.amax(-1).clamp_min(1e-6)
    local_saturation = F.avg_pool2d(saturation[None,None],11,1,5)[0,0]
    interior = F.avg_pool2d((alpha[...,0]>.9).float()[None,None],5,1,2)[0,0] == 1
    return (neutral>.55) & (neutral-smooth>.15) & (local_saturation-saturation>.15) & interior


def neutral_peak_metrics(predicted, target, alpha):
    """Image-proxy peak counts and errors; inputs share the observation domain."""
    peaks=neutral_peak_mask(target,alpha)
    predicted_peaks=neutral_peak_mask(predicted,alpha)
    target_neighborhood=F.max_pool2d(peaks[None,None].float(),5,1,2)[0,0]>0
    predicted_neighborhood=F.max_pool2d(predicted_peaks[None,None].float(),5,1,2)[0,0]>0
    row={'pixels':int(peaks.sum()), 'predicted_pixels':int(predicted_peaks.sum()),
         'matched_predicted_pixels_2px':int((predicted_peaks & target_neighborhood).sum()),
         'matched_target_pixels_2px':int((peaks & predicted_neighborhood).sum())}
    if peaks.any():
        def contrast(image):
            neutral=image.amin(-1)
            return neutral-F.avg_pool2d(neutral[None,None],11,1,5)[0,0]
        row.update(rgb_mae=float((predicted[peaks]-target[peaks]).abs().mean()),
                   target_contrast=float(contrast(target)[peaks].mean()),
                   predicted_contrast=float(contrast(predicted)[peaks].mean()),
                   contrast_ratio=float(contrast(predicted)[peaks].mean()/contrast(target)[peaks].mean()))
    return row


def aggregate_neutral_peak_metrics(rows):
    """Pool counts and target-peak pixels across frames, without frame weighting."""
    result = {key:sum(row[key] for row in rows) for key in (
        'pixels', 'predicted_pixels', 'matched_predicted_pixels_2px', 'matched_target_pixels_2px')}
    for key in ('rgb_mae', 'target_contrast', 'predicted_contrast'):
        result[key] = (sum(row[key]*row['pixels'] for row in rows if row['pixels'])
                       / result['pixels'] if result['pixels'] else None)
    result['contrast_ratio'] = (result['predicted_contrast']/result['target_contrast']
                                if result['pixels'] else None)
    result['precision_2px'] = (result['matched_predicted_pixels_2px']/result['predicted_pixels']
                              if result['predicted_pixels'] else None)
    result['recall_2px'] = (result['matched_target_pixels_2px']/result['pixels']
                           if result['pixels'] else None)
    return result


_NEUTRAL_PEAK_COMPONENT_BUCKETS = {'1_to_4': (1, 4), '5_to_16': (5, 16), 'gt16': (17, None)}
_NEUTRAL_PEAK_COMPONENT_SUM_KEYS = (
    'component_count', 'gt_peak_pixels', 'absolute_rgb_error_sum',
    'target_contrast_sum', 'predicted_contrast_sum', 'matched_gt_pixels_2px',
    'components_with_at_least_one_hit',
)


def _neutral_peak_component_metrics(sums):
    pixels, components = sums['gt_peak_pixels'], sums['component_count']
    return {
        'component_count': components,
        'gt_peak_pixels': pixels,
        'rgb_mae': sums['absolute_rgb_error_sum'] / (3 * pixels) if pixels else None,
        'target_contrast': sums['target_contrast_sum'] / pixels if pixels else None,
        'predicted_contrast': sums['predicted_contrast_sum'] / pixels if pixels else None,
        'contrast_ratio': sums['predicted_contrast_sum'] / sums['target_contrast_sum'] if pixels else None,
        'matched_gt_pixels_2px': sums['matched_gt_pixels_2px'],
        'pixel_recall_2px': sums['matched_gt_pixels_2px'] / pixels if pixels else None,
        'components_with_at_least_one_hit': sums['components_with_at_least_one_hit'],
        'component_any_hit_fraction': sums['components_with_at_least_one_hit'] / components if components else None,
    }


def neutral_peak_components(predicted, target, alpha):
    """GT 8-connected peak components in observation RGB, using CPU reductions.

    Inputs follow ``neutral_peak_metrics``; the evaluator supplies uint8-quantized
    RGB and target alpha. GT component areas define 1-4, 5-16 and >16 pixel
    buckets. Matching uses a 2-pixel Chebyshev neighborhood of predicted peaks;
    contrast is neutral RGB minus its zero-padded 11x11 mean. Raw sums allow
    pixel/component pooling across frames. These are image proxies, not physical
    specular labels; any-hit also accepts broad or displaced predicted peaks.
    """
    import cv2

    predicted, target, alpha = (image.detach().cpu() for image in (predicted, target, alpha))
    peaks = neutral_peak_mask(target, alpha)
    predicted_peaks = neutral_peak_mask(predicted, alpha)
    hits = peaks & (F.max_pool2d(predicted_peaks[None, None].float(), 5, 1, 2)[0, 0] > 0)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        peaks.numpy().astype(np.uint8), connectivity=8)
    hit_labels = np.unique(labels[hits.numpy()])

    def contrast(image):
        neutral = image.amin(-1)
        return neutral - F.avg_pool2d(neutral[None, None], 11, 1, 5)[0, 0]

    target_contrast, predicted_contrast = contrast(target), contrast(predicted)
    buckets = {}
    for name, (low, high) in _NEUTRAL_PEAK_COMPONENT_BUCKETS.items():
        ids = [index for index in range(1, count)
               if stats[index, cv2.CC_STAT_AREA] >= low
               and (high is None or stats[index, cv2.CC_STAT_AREA] <= high)]
        mask = torch.from_numpy(np.isin(labels, ids))
        sums = {
            'component_count': len(ids),
            'gt_peak_pixels': int(mask.sum()),
            'absolute_rgb_error_sum': float((predicted[mask] - target[mask]).abs().sum()),
            'target_contrast_sum': float(target_contrast[mask].sum()),
            'predicted_contrast_sum': float(predicted_contrast[mask].sum()),
            'matched_gt_pixels_2px': int((hits & mask).sum()),
            'components_with_at_least_one_hit': int(np.isin(ids, hit_labels).sum()),
        }
        buckets[name] = {'sums': sums, 'metrics': _neutral_peak_component_metrics(sums)}
    return {'total_predicted_peak_pixels': int(predicted_peaks.sum()), 'buckets': buckets}


def aggregate_neutral_peak_components(rows):
    """Pool raw component/pixel sums, then derive metrics; empty buckets are None."""
    buckets = {}
    for name in _NEUTRAL_PEAK_COMPONENT_BUCKETS:
        sums = {key: sum(row['buckets'][name]['sums'][key] for row in rows)
                for key in _NEUTRAL_PEAK_COMPONENT_SUM_KEYS}
        buckets[name] = {'sums': sums, 'metrics': _neutral_peak_component_metrics(sums)}
    return {'total_predicted_peak_pixels': sum(row['total_predicted_peak_pixels'] for row in rows),
            'buckets': buckets}


def observation_image(linear, gamma=1.0, alpha=None, background=0.0):
    """Encode HDR RGB or encode PNG foreground before alpha composition."""
    if alpha is not None:
        foreground = (linear - background * (1 - alpha)).clamp_min(0) / alpha.clamp_min(1e-8)
    else:
        foreground = linear
    encoded = foreground if gamma == 1.0 else torch.where(
        foreground > 0,
        foreground.clamp_min(1e-8).pow(1 / gamma),
        torch.zeros_like(foreground),
    )
    return encoded if alpha is None else encoded * alpha + background * (1 - alpha)


def target_image(sample, background, display_gamma=1.0):
    image, alpha = sample["image"], sample["alpha"]
    target = image if alpha is None else image * alpha + background * (1 - alpha)
    return observation_image(target, display_gamma) if sample["is_hdr"] else target


def render_observation(
    gaussians,
    transport,
    sample,
    background,
    shadow,
    port_active,
    display_gamma=1.0,
    shadow_mode="depth",
    absgrad=False,
    geometry_only=False,
    surface_field=None,
    normal_field=None,
    radiance_residual=None,
    residual_indices=None,
):
    """Apply the observation model to a full view."""
    linear, alpha, info = render(
        gaussians,
        transport,
        sample,
        background,
        shadow,
        port_active,
        shadow_mode=shadow_mode,
        absgrad=absgrad,
        geometry_only=geometry_only,
        surface_field=surface_field,
        normal_field=normal_field,
        radiance_residual=radiance_residual,
        residual_indices=residual_indices,
    )
    predicted = observation_image(
        linear,
        display_gamma,
        alpha=None if sample["is_hdr"] else alpha,
        background=background,
    )
    return predicted, alpha, info


def save_pair(path, predicted, target):
    pair = torch.cat((target, predicted), dim=1).detach().clamp(0, 1).cpu().numpy()
    Image.fromarray(np.rint(pair * 255).astype(np.uint8)).save(path)


def aggregate_residual_stats(rows):
    """Sum counts, pool queried-pixel means, and retain minima from nonempty frames."""
    count = sum(row['queried_pixels'] for row in rows)
    counts = ['queried_pixels', 'covered_pixels', 'requested_pixels', 'requested_covered_pixels']
    minima = [key for key in rows[0] if key.endswith('_min')]
    return {
        **{key: sum(row[key] for row in rows) for key in counts},
        **{key: min((row[key] for row in rows if row['queried_pixels']), default=0.) for key in minima},
        **{key: sum(row[key]*row['queried_pixels'] for row in rows)/max(count, 1)
           for key in rows[0] if key not in counts and key not in minima},
    }


@torch.no_grad()
def evaluate_samples(
    gaussians,
    transport,
    samples,
    background,
    shadow,
    port_active,
    output=None,
    perceptual=False,
    display_gamma=1.0,
    quantize=False,
    shadow_mode="depth",
    geometry_only=False,
    surface_field=None,
    normal_field=None,
    sdf_volume=None,
    highlights=False,
    radiance_residual=None,
    save_all=False,
    shift_align=0,
    calibrate_steps=0,
    calibrate_light=True,
):
    if sdf_volume is not None and radiance_residual is not None:
        raise ValueError('SDF volume and Gaussian radiance residual are separate rendering branches')
    transport.eval()
    lpips_model = None
    if perceptual:
        import lpips

        lpips_model = lpips.LPIPS(net="vgg").to(gaussians.center.device).eval()
    results = []
    for idx, sample in enumerate(samples):
        if sdf_volume is None:
            predicted, alpha, info = render_observation(
                gaussians, transport, sample, background, shadow, port_active,
                display_gamma, shadow_mode, geometry_only=geometry_only,
                surface_field=surface_field, normal_field=normal_field,
                radiance_residual=radiance_residual)
        else:
            from sdf_volume import render_image
            field, radiance, samples_per_ray = sdf_volume
            linear, alpha, _ = render_image(field, radiance, sample, transport.light_scale,
                                            samples=samples_per_ray, background=background)
            predicted = observation_image(linear, display_gamma,
                alpha=None if sample['is_hdr'] else alpha, background=background)
        target = target_image(sample, background, display_gamma)
        # Unit data range; clamp only for explicitly reported display-domain metrics.
        x, y = predicted.clamp(0, 1), target.clamp(0, 1)
        if quantize:
            x, y = (x * 255).round() / 255, (y * 255).round() / 255
        item = {
            "frame_index": sample["frame_index"],
            "name": sample["name"],
            "PSNR": (-10 * torch.log10((x - y).square().mean())).item(),
            "SSIM": ssim(x, y).item(),
            "raw_MSE": (predicted - target).square().mean().item(),
        }
        if radiance_residual is not None:
            item['residual_stats'] = {key: value.item() if isinstance(value, torch.Tensor) else value
                                      for key, value in info['residual_stats'].items()}
        if sdf_volume is None and 'surface_fragment_stats' in info:
            item['surface_fragment_stats'] = info['surface_fragment_stats']
        if sample["alpha"] is not None:
            item["alpha_L1"] = (alpha - sample["alpha"]).abs().mean().item()
        if highlights:
            if sample['alpha'] is None:
                raise ValueError('Neutral-peak evaluation requires target alpha for the interior mask')
            peak_x, peak_y = (x, y) if quantize else ((x*255).round()/255, (y*255).round()/255)
            item['neutral_peak_metrics'] = neutral_peak_metrics(peak_x, peak_y, sample['alpha'])
            # Reconstruct on CPU from bytes, as in offline saved-PNG diagnostics;
            # transferring GPU /255 floats can otherwise differ by one ULP.
            component_x, component_y = ((image * 255).round().byte().cpu().float() / 255
                                        for image in (peak_x, peak_y))
            item['neutral_peak_components'] = neutral_peak_components(component_x, component_y, sample['alpha'])
        if lpips_model is not None:
            item["LPIPS"] = lpips_model(
                x.permute(2, 0, 1)[None] * 2 - 1, y.permute(2, 0, 1)[None] * 2 - 1
            ).item()
        if shift_align:
            with torch.no_grad():
                item['shift_aligned'] = shift_aligned_metrics(x, y, shift_align, background, lpips_model)
        if calibrate_steps:
            if sdf_volume is not None or radiance_residual is not None:
                raise ValueError('Test-view calibration supports the Gaussian rendering branch only')
            calibrated, stats = calibrate_view(gaussians, transport, sample, target, background, shadow,
                                               port_active, display_gamma, shadow_mode, calibrate_steps,
                                               calibrate_light, surface_field, normal_field)
            with torch.no_grad():
                fitted, _, _ = render_observation(gaussians, transport, calibrated, background, shadow, port_active,
                                                  display_gamma, shadow_mode, geometry_only=geometry_only,
                                                  surface_field=surface_field, normal_field=normal_field)
                fx = (fitted.clamp(0, 1) * 255).round() / 255 if quantize else fitted.clamp(0, 1)
                stats.update(PSNR=(-10 * torch.log10((fx - y).square().mean())).item(), SSIM=ssim(fx, y).item())
                if lpips_model is not None:
                    stats['LPIPS'] = lpips_model(fx.permute(2, 0, 1)[None] * 2 - 1,
                                                 y.permute(2, 0, 1)[None] * 2 - 1).item()
            item['test_time_calibrated'] = stats
            if output and (save_all or idx < 4):
                save_pair(Path(output) / f"calibrated_pair_{idx:03}.png", fx, y)
        results.append(item)
        if output and (save_all or idx < 4):
            save_pair(Path(output) / f"pair_{idx:03}.png", x, y)
    metrics = {
        key: sum(row[key] for row in results) / len(results)
        for key in results[0]
        if key not in ["frame_index", "name", "neutral_peak_metrics", "neutral_peak_components",
                       "residual_stats", "surface_fragment_stats", "shift_aligned", "test_time_calibrated"]
    }
    if calibrate_steps:
        fitted = [row['test_time_calibrated'] for row in results]
        metrics['test_time_calibrated'] = {key: sum(row[key] for row in fitted) / len(fitted)
                                           for key in ('PSNR', 'SSIM', 'LPIPS') if key in fitted[0]}
    if shift_align:
        aligned = [row['shift_aligned'] for row in results]
        metrics['shift_aligned'] = {key: sum(row[key] for row in aligned) / len(aligned) for key in aligned[0]}
        for key in ('dy', 'dx'):
            values = torch.tensor([row[key] for row in aligned])
            metrics['shift_aligned'][key + '_std'] = values.std(unbiased=False).item()
    if radiance_residual is not None:
        metrics['residual_stats'] = aggregate_residual_stats([row['residual_stats'] for row in results])
    if highlights:
        metrics['neutral_peak_metrics'] = aggregate_neutral_peak_metrics(
            [row['neutral_peak_metrics'] for row in results])
        metrics['neutral_peak_components'] = aggregate_neutral_peak_components(
            [row['neutral_peak_components'] for row in results])
    transport.train()
    return metrics, results


def load_model(path):
    checkpoint = torch.load(path, map_location="cuda", weights_only=False)
    config = checkpoint["config"]
    state = checkpoint["gaussians"]
    count = len(state["params.means"])
    gaussians = Gaussians(
        count,
        state["center"],
        checkpoint["radius"],
        config["feature_dim"],
        geometry=resolve_config(config)['geometry'],
    )
    gaussians.load_state_dict(state)
    gaussians.surface_depth = config.get('surface_depth', 'center')
    transport = build_transport(
        config, checkpoint["transport"]["light_scale"].item()
    ).cuda()
    transport.load_state_dict(checkpoint["transport"])
    return gaussians, transport, checkpoint


def load_surface_field(checkpoint):
    if not checkpoint['config'].get('sdf', False):
        return None
    from sdf import SurfaceSDF
    field = SurfaceSDF(checkpoint['gaussians']['center'], checkpoint['radius'],
                       detail=checkpoint['config'].get('sdf_detail', False))
    field.load_state_dict(checkpoint['sdf'])
    return field


def load_normal_field(checkpoint):
    if not checkpoint['config'].get('normal_field', False):
        return None
    from materials.normal_field import NormalResidualField
    field = NormalResidualField(checkpoint['gaussians']['center'], checkpoint['radius'])
    field.load_state_dict(checkpoint['normal_field'])
    return field


def load_radiance_residual(checkpoint):
    config = checkpoint['config']
    if not config.get('radiance_residual', False):
        if 'radiance_residual' in checkpoint:
            raise ValueError('Radiance residual weights require an enabled checkpoint configuration')
        if config.get('residual_interaction', 'none') != 'none':
            raise ValueError('Radiance residual interaction requires an enabled checkpoint configuration')
        if config.get('residual_angular_bank', 'none') != 'none':
            raise ValueError('Radiance residual angular bank requires an enabled checkpoint configuration')
        return None
    if config.get('representation') != 'neural_material':
        raise ValueError('Radiance residual checkpoints require neural_material')
    if config.get('sdf_shading', False) or config.get('normal_field', False):
        raise ValueError('Radiance residual does not support SDF shading or normal-field checkpoints')
    if 'radiance_residual' not in checkpoint or 'residual_steps' not in checkpoint:
        raise ValueError('Enabled radiance residual checkpoint is missing its state or step count')
    from materials.radiance_residual import RadianceResidual
    residual = RadianceResidual(checkpoint['gaussians']['center'], checkpoint['radius'],
                                normal_source=config['residual_normal'],
                                interaction=config.get('residual_interaction', 'none'),
                                angular_bank=config.get('residual_angular_bank', 'none'))
    expected_scales = residual.angular_scales.clone() if residual.angular_bank != 'none' else None
    residual.load_state_dict(checkpoint['radiance_residual'], strict=True)
    if expected_scales is not None and not torch.equal(residual.angular_scales, expected_scales):
        raise ValueError('Radiance residual angular scales disagree with the saved angular-bank configuration')
    return residual


def load_sdf_volume(checkpoint):
    if not checkpoint['config'].get('sdf_volume_weight', 0):
        raise ValueError('Checkpoint has no image-supervised SDF volume branch')
    from sdf_volume import SDFRadiance
    field = load_surface_field(checkpoint)
    radiance = SDFRadiance(field.center.device,
                           detail=checkpoint['config'].get('sdf_volume_detail', False),
                           hint_encoding=checkpoint['config'].get('sdf_volume_hint_encoding', False))
    radiance.load_state_dict(checkpoint['sdf_volume'])
    return field, radiance, checkpoint['config']['sdf_volume_samples']


def main():
    torch.set_num_threads(8)
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument(
        "--split", choices=["validation", "fit", "test", "train"], default="validation"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Deterministic frame subsample for diagnostics; 0 evaluates every frame",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--lpips", action="store_true")
    parser.add_argument('--resolution', type=int, help='Evaluation longest-edge resolution; defaults to the saved training resolution')
    parser.add_argument('--highlights', action='store_true', help='Report quantized neutral-peak image proxies using target alpha')
    parser.add_argument('--sdf-volume', action='store_true', help='Evaluate the independent SDF rendering branch')
    parser.add_argument('--save-all', action='store_true', help='Export every evaluated GT/prediction pair instead of the first four')
    parser.add_argument('--calibrate-test-views', type=int, default=0,
                        help='Also report a secondary literature-style score after fitting each held-out view\'s '
                             'camera rotation and light offset on its GT for this many steps (scene frozen)')
    parser.add_argument('--calibrate-camera-only', action='store_true',
                        help='With --calibrate-test-views, fit only the camera rotation')
    parser.add_argument('--shift-align', type=int, default=0,
                        help='Also report secondary metrics after each view\'s best global 2D translation within '
                             'this many pixels (calibration-compensated diagnostic; primary metrics unchanged)')
    args = parser.parse_args()
    from data import SceneDataset

    gaussians, transport, checkpoint = load_model(args.checkpoint)
    config = checkpoint["config"]
    resolution = config['resolution'] if args.resolution is None else args.resolution
    surface_field = load_surface_field(checkpoint) if config.get('sdf_shading', False) else None
    normal_field = load_normal_field(checkpoint)
    radiance_residual = load_radiance_residual(checkpoint)
    if args.sdf_volume and radiance_residual is not None:
        parser.error('--sdf-volume cannot evaluate an enabled Gaussian radiance residual checkpoint')
    dataset = SceneDataset(
        config["scene"],
        "train" if args.split in ["validation", "fit"] else args.split,
        resolution,
        unit_light_intensity=config["unit_light_intensity"],
    )
    indices = (
        checkpoint["val_indices"]
        if args.split == "validation"
        else checkpoint["fit_indices"] if args.split == "fit" else list(range(len(dataset)))
    )
    if not len(indices):
        raise ValueError(
            "This checkpoint has no held-out validation frames; select an existing split explicitly"
        )
    if args.limit:
        indices = indices[:: max(1, len(indices) // args.limit)][: args.limit]
    camera_offsets = None
    fit_camera_indices = {}
    if args.split in ["fit", "train"] and checkpoint["camera_offsets"] is not None:
        from cameras import load_camera_offsets

        camera_offsets = load_camera_offsets(checkpoint, gaussians.radius).cuda()
        fit_camera_indices = {index: local for local, index in enumerate(checkpoint["fit_indices"])}
    light_offsets = None
    if args.split in ["fit", "train"] and checkpoint.get("light_offsets") is not None:
        from cameras import TrainLightOffsets

        light_offsets = TrainLightOffsets(len(checkpoint["fit_indices"]), gaussians.radius).cuda()
        light_offsets.load_state_dict(checkpoint["light_offsets"])
        fit_light_indices = {index: local for local, index in enumerate(checkpoint["fit_indices"])}

    def evaluation_samples():
        for index in indices:
            sample = to_device(dataset[index], "cuda")
            if index in fit_camera_indices:
                sample = camera_offsets.correct(sample, fit_camera_indices[index])
            if light_offsets is not None and index in fit_light_indices:
                sample = light_offsets.correct(sample, fit_light_indices[index])
            yield sample

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    metrics, rows = evaluate_samples(
        gaussians,
        transport,
        evaluation_samples(),
        config["background"],
        checkpoint["step"] >= config["shadow_start"],
        checkpoint["step"] >= config["port_start"],
        output,
        args.lpips,
        config["display_gamma"],
        surface_field=surface_field,
        normal_field=normal_field,
        quantize=True,
        shadow_mode=config["shadow_mode"],
        geometry_only=checkpoint["step"] <= config.get("geometry_warmup_steps", 0),
        sdf_volume=load_sdf_volume(checkpoint) if args.sdf_volume else None,
        highlights=args.highlights,
        radiance_residual=radiance_residual,
        save_all=args.save_all,
        shift_align=args.shift_align,
        calibrate_steps=args.calibrate_test_views,
        calibrate_light=not args.calibrate_camera_only,
    )
    report = {
        "split": args.split,
        "checkpoint": args.checkpoint,
        'render_branch':('sdf_volume' if args.sdf_volume else
                         'gaussian_residual' if radiance_residual is not None else
                         'gaussian'),
        "metrics": metrics,
        "views": rows,
        "protocol": "uint8-quantized observation RGB, unit PSNR, zero-padded 11x11 SSIM sigma1.5; LPIPS standard [-1,1]; raw_MSE is unclipped observation MSE; original held-out camera calibration",
        "training_resolution": config["resolution"],
        "resolution": resolution,
        "background": config["background"],
        "evaluated_frames": len(indices),
        "limit": args.limit,
        "save_all": args.save_all,
        "display_gamma": config["display_gamma"],
        "camera_protocol": "saved training offsets on fit frames; original calibration on held-out frames",
        "light_protocol": ("saved per-frame light offsets on fit frames; original calibration on held-out frames"
                           if checkpoint.get("light_offsets") is not None else "original calibration"),
        "corrected_fit_frames": sum(index in fit_camera_indices for index in indices),
        "observation_protocol": "PNG foreground gamma then alpha composition; HDR full-image gamma",
    }
    if args.calibrate_test_views:
        if args.split == 'fit':
            parser.error('--calibrate-test-views is for held-out views; fit views already use saved corrections')
        report['test_time_calibrated_protocol'] = (
            f'Secondary literature-style protocol: per held-out view, a camera rotation about the calibrated center'
            f'{" and a world light-position offset" if not args.calibrate_camera_only else ""} are fitted on that view\'s GT '
            f'for {args.calibrate_test_views} Adam steps (lr 1e-3 decayed to 1e-5, 0.8 L1 + 0.2 SSIM loss) with the '
            'scene, materials and networks frozen, then the view is re-rendered and scored. Uses test GT; '
            'never the primary fixed-camera metric.')
    if args.shift_align:
        report['shift_aligned_protocol'] = (
            f'Secondary diagnostic: each quantized render is translated by the global 2D shift (integer search '
            f'within +-{args.shift_align}px, then 1/8px bilinear refinement) minimizing MSE to its GT view, '
            'requantized, then scored with the same PSNR/SSIM/LPIPS. The shift is fitted on GT, so it '
            'compensates per-view calibration error and is never the primary fixed-camera metric.')
    if args.highlights:
        report['highlight_protocol'] = ('Neutral bright/desaturated local peaks in uint8-quantized observation RGB; '
            'target-alpha interior; 11x11 contrast window; 2px Chebyshev matching; '
            'counts and target-peak pixels pooled across frames. GT 8-connected component area buckets '
            '1-4, 5-16 and >16 pixels pool raw pixel/component sums; component any-hit also accepts '
            'broad or displaced predicted peaks. Image proxy, not physical specular ground truth.')
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
