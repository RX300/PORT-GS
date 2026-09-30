"""CPU diagnostics of fixed-normal angular cues at GT-defined tiny highlights.

Kernel values are unit-peak features, not predicted radiance. Selecting a kernel
using a GT core/ring contrast is an offline optimistic bound, not model input.
"""

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from evaluate import neutral_peak_mask


NARROW_TAU_DEGREES = np.geomspace(2., 32., 8)
WIDE_TAU_DEGREES = np.geomspace(8., 128., 8)
_BANKS = {'narrow': NARROW_TAU_DEGREES, 'wide': WIDE_TAU_DEGREES}
_GROUPS = ('all', 'missed', 'partial', 'all_hit')
_QUANTILES = (0., .1, .25, .5, .75, .9, 1.)
_COUNT_KEYS = ('all_gt_peak_components', 'all_gt_peak_pixels', 'predicted_peak_pixels',
               'gt_small_peak_components', 'gt_small_peak_pixels',
               'matched_gt_pixels_2px', 'components_with_at_least_one_hit')


def angular_kernel_bank(angles, bank='narrow'):
    """Return float64 [...,8] isotropic unit-peak cues for radian angles."""
    angles = np.asarray(angles, dtype=np.float64)
    tau = np.deg2rad(_BANKS[bank])
    return np.exp(-.5*np.square(angles[..., None]/tau))


def _metadata(normals):
    return {
        'normal_sources': list(normals),
        'primary_normal_source': 'geometry',
        'kernel': 'exp(-theta^2/(2*tau^2)); unit peak; angles in radians',
        'tau_degrees': {name: values.tolist() for name,values in _BANKS.items()},
        'component_protocol': 'GT neutral_peak_mask;8-connected components with1–4 pixels.',
        'match_protocol': 'Per-GT-pixel Chebyshev radius2 from predicted neutral peaks.',
        'ring_protocol': 'Own11x11 dilation minus all GT peaks;target alpha>.9;GS alpha>0 and finite angle.',
        'core_protocol': 'All component pixels must have GS alpha>0 and finite angle;low GS alpha<=.9 is descriptive.',
        'localization_protocol': 'Exact angle minimum over valid core union valid ring;best/worst Chebyshev distance to GT core. '
            'Reliable only with complete core/nonempty ring,worst tie distance<=2,nonplateau,and no tied minima spanning core and ring.',
        'oracle_protocol': 'Each bank independently maximizes GT core-minus-ring mean;offline optimistic feature contrast only.',
        'quantile_probabilities': list(_QUANTILES),
        'quantile_protocol': 'NumPy linear empirical quantiles;pixel weighting repeats each component scalar by GT component area.',
        'aggregate_weighting': 'Equal component weights or GT core-area weights;ring means retain each component own ring. '
            'Metric distributions include eligible components only;unsupported counts remain explicit and fail the primary screen.',
        'screen': {'missed_denominator': 'All completely missed components, including unsupported components as failures.',
            'max_worst_distance_px': 2, 'min_best_narrow_delta': .1, 'min_narrow_minus_wide_delta': .05,
            'min_qualifying_fraction': .6, 'stop_nonpositive_or_mislocalized_fraction_above': .5,
            'material': 'Descriptive only;never selects the primary decision.'},
    }


def _normal_component(angle, core, ring, covered):
    valid = np.isfinite(angle) & covered
    valid_core, valid_ring = core & valid, ring & valid
    core_count, ring_count = int(valid_core.sum()), int(valid_ring.sum())
    complete_core = core_count == int(core.sum())
    eligible = complete_core and ring_count > 0
    core_values, ring_values = angle[valid_core], angle[valid_ring]
    banks = {}
    for name in _BANKS:
        core_mean = angular_kernel_bank(core_values, name).mean(0) if core_count else None
        ring_mean = angular_kernel_bank(ring_values, name).mean(0) if ring_count else None
        delta = core_mean-ring_mean if core_mean is not None and ring_mean is not None else None
        index = int(delta.argmax()) if delta is not None else None
        banks[name] = {'core_mean': core_mean.tolist() if core_mean is not None else None,
            'ring_mean': ring_mean.tolist() if ring_mean is not None else None,
            'delta': delta.tolist() if delta is not None else None,
            'best_delta': float(delta[index]) if index is not None else None, 'best_index': index,
            'best_tie_count': int((delta == delta[index]).sum()) if index is not None else 0}
    region = valid_core | valid_ring
    if region.any():
        minimum = float(angle[region].min())
        extrema = region & (angle == minimum)
        extrema_yx, core_yx = np.argwhere(extrema), np.argwhere(core)
        distances = np.abs(extrema_yx[:, None]-core_yx[None]).max(2).min(1)
        best, worst = int(distances.min()), int(distances.max())
        count = len(extrema_yx)
        plateau = bool((angle[region] == minimum).all())
        spans = bool((extrema & core).any() and (extrema & valid_ring).any())
        localization = {'minimum_angle_radians': minimum, 'best_distance_px': best, 'worst_distance_px': worst,
            'tie_count': count, 'plateau': plateau, 'tie_ambiguous': count > 1,
            'tie_spans_core_and_ring': spans,
            'reliable': bool(eligible and worst <= 2 and not plateau and not spans)}
    else:
        localization = {'minimum_angle_radians': None, 'best_distance_px': None, 'worst_distance_px': None,
            'tie_count': 0, 'plateau': False, 'tie_ambiguous': False,
            'tie_spans_core_and_ring': False, 'reliable': False}
    narrow, wide = banks['narrow']['best_delta'], banks['wide']['best_delta']
    gap = narrow-wide if narrow is not None and wide is not None else None
    qualifies = bool(eligible and localization['reliable'] and narrow >= .1 and gap >= .05)
    return {'eligible': bool(eligible), 'qualifies': qualifies, 'valid_core_pixels': core_count,
        'valid_ring_pixels': ring_count, 'ring_pixels_before_GS_and_angle_mask': int(ring.sum()),
        'complete_core': complete_core, 'has_valid_ring': ring_count > 0,
        'banks': banks, 'narrow_minus_wide_best_delta': gap, 'localization': localization}


def _distribution(values, weights):
    pairs = [(value, weight) for value,weight in zip(values, weights) if value is not None]
    if not pairs:
        return {'components': 0, 'pixels': 0, 'component_weighted': {'mean': None, 'quantiles': None},
                'pixel_weighted': {'mean': None, 'quantiles': None}}
    values = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
    weights = np.asarray([pair[1] for pair in pairs], dtype=np.int64)
    return {'components': len(values), 'pixels': int(weights.sum()),
        'component_weighted': {'mean': float(values.mean()), 'quantiles': np.quantile(values, _QUANTILES).tolist()},
        'pixel_weighted': {'mean': float(np.average(values, weights=weights)),
            'quantiles': np.quantile(np.repeat(values, weights), _QUANTILES).tolist()}}


def _array_means(values, weights):
    pairs = [(value, weight) for value,weight in zip(values, weights) if value is not None]
    if not pairs:
        return {'components': 0, 'pixels': 0, 'component_weighted': None, 'pixel_weighted': None}
    values = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
    weights = np.asarray([pair[1] for pair in pairs], dtype=np.int64)
    return {'components': len(values), 'pixels': int(weights.sum()),
        'component_weighted': values.mean(0).tolist(), 'pixel_weighted': np.average(values, axis=0, weights=weights).tolist()}


def _normal_summary(components, source):
    rows = [row['normals'][source] for row in components]
    weights = [row['area_pixels'] for row in components]
    eligible = [i for i,row in enumerate(rows) if row['eligible']]
    good_rows, good_weights = [rows[i] for i in eligible], [weights[i] for i in eligible]
    metrics = {
        'best_narrow_delta': [row['banks']['narrow']['best_delta'] for row in good_rows],
        'best_wide_delta': [row['banks']['wide']['best_delta'] for row in good_rows],
        'narrow_minus_wide_best_delta': [row['narrow_minus_wide_best_delta'] for row in good_rows],
        'minimum_angle_degrees': [np.rad2deg(row['localization']['minimum_angle_radians']) for row in good_rows],
        'best_distance_px': [row['localization']['best_distance_px'] for row in good_rows],
        'worst_distance_px': [row['localization']['worst_distance_px'] for row in good_rows],
    }
    result = {'components': len(rows), 'pixels': sum(weights),
        'eligible_components': len(eligible), 'eligible_pixels': sum(good_weights),
        'unsupported_components': sum(not row['eligible'] for row in rows),
        'unsupported_pixels': sum(weight for row,weight in zip(rows,weights) if not row['eligible']),
        'incomplete_core_components': sum(not row['complete_core'] for row in rows),
        'no_valid_ring_components': sum(not row['has_valid_ring'] for row in rows),
        'qualifying_components': sum(row['qualifies'] for row in rows),
        'qualifying_pixels': sum(weight for row,weight in zip(rows,weights) if row['qualifies']),
        'plateau_components': sum(row['localization']['plateau'] for row in rows),
        'tied_minimum_components': sum(row['localization']['tie_ambiguous'] for row in rows),
        'tied_minimum_spanning_core_ring_components': sum(row['localization']['tie_spans_core_and_ring'] for row in rows),
        'reliably_localized_components': sum(row['localization']['reliable'] for row in rows),
        'eligible_metric_distributions': {name: _distribution(values, good_weights) for name,values in metrics.items()},
        'eligible_bank_means': {}, 'eligible_selected_bandwidth_counts': {}}
    for bank in _BANKS:
        result['eligible_bank_means'][bank] = {name: _array_means([row['banks'][bank][name] for row in good_rows], good_weights)
                                              for name in ('core_mean', 'ring_mean', 'delta')}
        indices = np.asarray([row['banks'][bank]['best_index'] for row in good_rows], dtype=np.int64)
        result['eligible_selected_bandwidth_counts'][bank] = {
            'components': np.bincount(indices, minlength=8).tolist(),
            'pixels': np.bincount(indices, weights=np.asarray(good_weights, dtype=np.int64), minlength=8).astype(np.int64).tolist()}
    return result


def _aggregate(components, normals):
    result = {}
    for group in _GROUPS:
        rows = components if group == 'all' else [row for row in components if row['hit_group'] == group]
        result[group] = {'components': len(rows), 'pixels': sum(row['area_pixels'] for row in rows),
            'matched_gt_pixels_2px': sum(row['matched_gt_pixels_2px'] for row in rows),
            'coverage': {key: sum(row['coverage'][key] for row in rows)
                         for key in ('covered_core_pixels', 'uncovered_core_pixels', 'low_alpha_core_pixels')},
            'components_with_uncovered_core': sum(row['coverage']['uncovered_core_pixels'] > 0 for row in rows),
            'components_with_low_alpha_core': sum(row['coverage']['low_alpha_core_pixels'] > 0 for row in rows),
            'normals': {source: _normal_summary(rows, source) for source in normals}}
    return result


def _screen(components):
    missed = [row['normals']['geometry'] for row in components if row['hit_group'] == 'missed']
    count = len(missed)
    qualifying = sum(row['qualifies'] for row in missed)
    nonpositive = [row['eligible'] and row['banks']['narrow']['best_delta'] <= 0 for row in missed]
    mislocalized = [row['eligible'] and row['localization']['worst_distance_px'] > 2 for row in missed]
    failing = sum(a or b for a,b in zip(nonpositive, mislocalized))
    fraction = qualifying/count if count else None
    stop_fraction = failing/count if count else None
    return {'normal_source': 'geometry', 'missed_components_denominator': count,
        'qualifying_components': qualifying, 'qualifying_fraction': fraction, 'minimum_qualifying_fraction': .6,
        'unsupported_components': sum(not row['eligible'] for row in missed),
        'eligible_nonpositive_best_narrow_components': sum(nonpositive),
        'eligible_mislocalized_components': sum(mislocalized),
        'eligible_nonpositive_or_mislocalized_components': failing,
        'nonpositive_or_mislocalized_fraction_of_all_missed': stop_fraction,
        'stop_fixed_normal_narrow_training': bool(count and stop_fraction > .5),
        'numerical_screen_passes': bool(count and fraction >= .6),
        'manual_fixed_crop_support_required': True,
        'scope': 'Practical frozen-normal feature screen, not a capacity theorem or unseen-light evaluation.'}


def analyze_angular_cues(prediction, target, target_alpha, gs_alpha, angle_maps):
    """Analyze one CPU frame; RGB floats must be rebuilt from quantized uint8.

    Alphas have shape HxW; angle maps are HxW radians and use NaN outside GS
    coverage. Geometry is the primary cue; material, when supplied, is only a
    descriptive reference. All kernel math and component reductions are float64.
    """
    assert prediction.device.type == target.device.type == target_alpha.device.type == gs_alpha.device.type == 'cpu'
    assert prediction.shape == target.shape and prediction.ndim == 3 and prediction.shape[-1] == 3
    assert target_alpha.shape == gs_alpha.shape == target.shape[:2]
    assert 'geometry' in angle_maps
    assert all(value.device.type == 'cpu' and value.shape == target_alpha.shape for value in angle_maps.values())
    peaks = neutral_peak_mask(target, target_alpha[..., None])
    predicted_peaks = neutral_peak_mask(prediction, target_alpha[..., None])
    hits = peaks & (F.max_pool2d(predicted_peaks[None, None].float(), 5, 1, 2)[0, 0] > 0)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(peaks.numpy().astype(np.uint8), connectivity=8)
    peaks, hits = peaks.numpy(), hits.numpy()
    target_interior = target_alpha.detach().numpy() > .9
    gs = gs_alpha.detach().numpy()
    angles = {name: value.detach().numpy().astype(np.float64) for name,value in angle_maps.items()}
    height, width = target_alpha.shape
    components = []
    for ident in range(1, count):
        x, y, w, h, area = (int(value) for value in stats[ident])
        if area > 4:
            continue
        x0, x1 = max(0, x-5), min(width, x+w+5)
        y0, y1 = max(0, y-5), min(height, y+h+5)
        crop = np.s_[y0:y1, x0:x1]
        core = labels[crop] == ident
        covered = gs[crop] > 0
        neighborhood = cv2.dilate(core.astype(np.uint8), np.ones((11,11), dtype=np.uint8)).astype(bool)
        ring = neighborhood & ~peaks[crop] & target_interior[crop]
        matched = int((hits[crop] & core).sum())
        components.append({'label': ident, 'bbox_xywh': [x,y,w,h], 'area_pixels': area,
            'matched_gt_pixels_2px': matched,
            'hit_group': 'missed' if not matched else 'all_hit' if matched == area else 'partial',
            'coverage': {'covered_core_pixels': int((core & covered).sum()),
                'uncovered_core_pixels': int((core & ~covered).sum()),
                'low_alpha_core_pixels': int((core & covered & (gs[crop] <= .9)).sum())},
            'normals': {name: _normal_component(angle[crop], core, ring, covered) for name,angle in angles.items()}})
    counts = {'all_gt_peak_components': count-1, 'all_gt_peak_pixels': int(peaks.sum()),
        'predicted_peak_pixels': int(predicted_peaks.sum()),
        'gt_small_peak_components': len(components), 'gt_small_peak_pixels': sum(row['area_pixels'] for row in components),
        'matched_gt_pixels_2px': sum(row['matched_gt_pixels_2px'] for row in components),
        'components_with_at_least_one_hit': sum(row['matched_gt_pixels_2px'] > 0 for row in components)}
    return {'metadata': _metadata(angle_maps), 'counts': counts, 'components': components,
            'aggregate': _aggregate(components, angle_maps), 'screen': _screen(components)}


def aggregate_angular_cues(frame_reports):
    """Merge raw component reports, preserving optional per-component frame IDs."""
    assert frame_reports, 'At least one frame report is required.'
    metadata = frame_reports[0]['metadata']
    assert all(report['metadata'] == metadata for report in frame_reports)
    components = [dict(row, **({'frame_index': report['frame_index']} if 'frame_index' in report else {}))
                  for report in frame_reports for row in report['components']]
    counts = {key: sum(report['counts'][key] for report in frame_reports) for key in _COUNT_KEYS}
    return {'metadata': metadata, 'frame_count': len(frame_reports), 'counts': counts, 'components': components,
            'aggregate': _aggregate(components, metadata['normal_sources']), 'screen': _screen(components)}
