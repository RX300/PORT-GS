"""Compare saved foundation images against one canonical dataset target."""

import json
import tarfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from data import SceneDataset
from evaluate import (aggregate_neutral_peak_components, aggregate_neutral_peak_metrics,
                      neutral_peak_components, neutral_peak_mask, neutral_peak_metrics,
                      ssim, target_image)


def foundation_paths(model, count):
    """Require a complete indexed PNG export, without resizing or name guessing."""
    if model['layout'] == 'separate':
        names = [f'{i:05}.png' for i in range(count)]
        directories = [Path(model['prediction_dir']), Path(model['target_dir'])]
        pattern = '*.png'
    elif model['layout'] == 'pairs':
        names = [f'pair_{i:03}.png' for i in range(count)]
        directories = [Path(model['directory'])] * 2
        pattern = 'pair_*.png'
    else:
        raise ValueError(f"Unknown foundation layout: {model['layout']}")
    for directory in directories:
        actual = {path.name for path in directory.glob(pattern)}
        if actual != set(names):
            raise ValueError(f'Incomplete or unexpected PNG names in {directory}: '
                             f'missing={sorted(set(names)-actual)}, extra={sorted(actual-set(names))}')
    return [(directories[0]/name, directories[1]/name) for name in names]


def load_foundation_images(model, index, shape):
    height, width = shape
    if model['layout'] == 'pairs':
        path = Path(model['directory'])/f'pair_{index:03}.png'
        with Image.open(path) as image:
            if image.mode != 'RGB' or image.size != (2*width, height):
                raise ValueError(f'Expected RGB GT-left/prediction-right pair at native size: {path}')
            pair = torch.from_numpy(np.array(image))
        return pair[:, width:].clone(), pair[:, :width].clone()
    if model['layout'] != 'separate':
        raise ValueError(f"Unknown foundation layout: {model['layout']}")
    images = []
    for key in ('prediction_dir', 'target_dir'):
        path = Path(model[key])/f'{index:05}.png'
        with Image.open(path) as image:
            if image.mode != 'RGB' or image.size != (width, height):
                raise ValueError(f'Expected native-size RGB PNG: {path}')
            images.append(torch.from_numpy(np.array(image)))
    return tuple(images)


def _comparison_indices(frame_indices, frame_count):
    indices = list(range(frame_count)) if frame_indices is None else list(frame_indices)
    if not indices or indices != sorted(set(indices)):
        raise ValueError('Comparison frame_indices must be nonempty, sorted and unique')
    if indices[0] < 0 or indices[-1] >= frame_count:
        raise ValueError('Comparison frame_indices are outside canonical metadata')
    return indices


def validate_view_mapping(report, frames, frame_indices=None):
    """Match report rows to original metadata IDs, independently of PNG ordinals."""
    indices = _comparison_indices(frame_indices, len(frames))
    count = report['evaluated_frames'] if 'evaluated_frames' in report else report['frames']
    if count != len(indices):
        raise ValueError('Foundation metric frame count differs from canonical metadata')
    if 'views' not in report:
        return {'mode': 'count_only', 'frames': count,
                'limitation': 'Saved metric report has no per-frame mapping; indexed order relies on '
                              'the documented loader/render contract, not independent name verification.'}
    views = report['views']
    if len(views) != count:
        raise ValueError('Foundation metric view count differs from its frame count')
    for row, index in zip(views, indices):
        frame = frames[index]
        if row['frame_index'] != index or row['name'] != Path(frame['file_path']).stem:
            raise ValueError(f'Foundation frame order/name differs at canonical frame {index}')
        if 'file_path' in row and row['file_path'] != frame['file_path']:
            raise ValueError(f'Foundation metadata file_path differs at canonical frame {index}')
    return {'mode': 'indices_and_names_verified', 'frames': count}


def target_difference(saved, canonical):
    difference = (saved.to(torch.int16)-canonical.to(torch.int16)).abs()
    return {'max_abs_byte_difference': int(difference.max()),
            'mean_abs_byte_difference': float(difference.double().mean()),
            'differing_channel_count': int((difference != 0).sum()),
            'differing_pixel_count': int((difference != 0).any(-1).sum()),
            'channel_count': difference.numel(), 'pixel_count': difference.shape[0]*difference.shape[1],
            'absolute_byte_difference_sum': int(difference.long().sum())}


def aggregate_target_differences(rows):
    sums = {key: sum(row[key] for row in rows) for key in (
        'differing_channel_count', 'differing_pixel_count', 'channel_count', 'pixel_count',
        'absolute_byte_difference_sum')}
    sums['max_abs_byte_difference'] = max(row['max_abs_byte_difference'] for row in rows)
    sums['mean_abs_byte_difference'] = sums['absolute_byte_difference_sum']/sums['channel_count']
    sums['all_saved_targets_exact'] = sums['differing_channel_count'] == 0
    sums['target_alignment_review_required'] = sums['max_abs_byte_difference'] > 1
    sums['interpretation'] = ('All predictions scored against canonical GT, without exposure, color, '
                              'pose or pixel alignment. Differences above one byte require explicit review; '
                              'one-byte differences are retained, not silently made equal.')
    return sums


def ring_error(predicted, target, alpha):
    peaks = neutral_peak_mask(target, alpha)
    ring = ((F.max_pool2d(peaks[None, None].float(), 11, 1, 5)[0, 0] > 0)
            & ~peaks & (alpha[..., 0] > .9))
    total = float((predicted.amin(-1)-target.amin(-1)).clamp_min(0)[ring].sum())
    count = int(ring.sum())
    return {'pixels': count, 'neutral_positive_error_sum': total,
            'neutral_positive_error': total/count if count else None}


def aggregate_ring(rows):
    count = sum(row['pixels'] for row in rows)
    total = sum(row['neutral_positive_error_sum'] for row in rows)
    return {'pixels': count, 'neutral_positive_error_sum': total,
            'neutral_positive_error': total/count if count else None}


def tiny_crops(target, alpha, frame_index):
    """First two tiny GT components in connected-component label order."""
    count, labels, stats, centers = cv2.connectedComponentsWithStats(
        neutral_peak_mask(target, alpha).numpy().astype(np.uint8), connectivity=8)
    tiny = [label for label in range(1, count) if 1 <= stats[label, cv2.CC_STAT_AREA] <= 4]
    height, width = target.shape[:2]
    rows = []
    for label in tiny[:2]:
        x, y = centers[label]
        left = max(0, min(width-64, int(round(x))-32))
        top = max(0, min(height-64, int(round(y))-32))
        rows.append({'frame_index': frame_index, 'component_label': label,
                     'component_pixels': int(stats[label, cv2.CC_STAT_AREA]),
                     'box_xyxy': [left, top, left+64, top+64]})
    return rows, {'frame_index': frame_index, 'available_tiny_components': len(tiny),
                  'selected_components': len(rows),
                  'fallback': 'fewer than two tiny components; no substitute selected' if len(rows) < 2 else None}


def comparison_gates(candidate, control, primary_eligible, target_review_required, scope=None):
    a = candidate['neutral_peak_components']['buckets']['1_to_4']['metrics']
    b = control['neutral_peak_components']['buckets']['1_to_4']['metrics']
    checks = {}

    def check(name, left, right, predicate):
        checks[name] = {'candidate': left, 'control': right,
                        'passes': predicate(left, right) if left is not None and right is not None else None}

    check('tiny_contrast_distance_to_one_reduction_at_least_0.03', a['contrast_ratio'], b['contrast_ratio'],
          lambda x, y: abs(y-1)-abs(x-1) >= .03)
    check('tiny_recall_gain_at_least_0.03', a['pixel_recall_2px'], b['pixel_recall_2px'], lambda x, y: x-y >= .03)
    check('tiny_MAE_decreases', a['rgb_mae'], b['rgb_mae'], lambda x, y: x < y)
    check('global_precision_drop_no_more_than_0.01', candidate['neutral_peak_metrics']['precision_2px'],
          control['neutral_peak_metrics']['precision_2px'], lambda x, y: y-x <= .01)
    check('four_fixed_frame_ring_overbrightness_increase_no_more_than_5pct',
          candidate['fixed_frame_ring']['neutral_positive_error'], control['fixed_frame_ring']['neutral_positive_error'],
          lambda x, y: x <= 1.05*y)
    check('PSNR_drop_no_more_than_0.1dB', candidate['PSNR'], control['PSNR'], lambda x, y: y-x <= .1)
    check('LPIPS_increase_no_more_than_0.002', candidate.get('LPIPS'), control.get('LPIPS'), lambda x, y: x-y <= .002)
    check('SSIM_drop_no_more_than_0.002', candidate['SSIM'], control['SSIM'], lambda x, y: y-x <= .002)
    numerical = all(row['passes'] is True for row in checks.values())
    return {'checks': checks, 'all_numeric_checks_passed': numerical,
            'primary_eligible': primary_eligible, 'target_alignment_review_required': target_review_required,
            'eligible_for_manual_foundation_review': numerical and primary_eligible and not target_review_required,
            'manual_fixed_crop_review': {'status': 'pending', 'passes': None},
            'promoted': False,
            'scope': scope if scope is not None else
                     'Observational comparison with different training budgets and implementations; '
                     'practical research filter, not an isolated causal test or untouched blind test.'}


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def _plot_comparison(output, frames, crops, images, model_specs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    columns = [('GT', 'Canonical GT')] + [(label, label + (' [context only]' if not spec['primary_eligible'] else ''))
                                        for label, spec in model_specs.items()]
    fig, axes = plt.subplots(len(frames), len(columns), figsize=(3*len(columns), 3*len(frames)), squeeze=False)
    for row, index in enumerate(frames):
        for col, (label, title) in enumerate(columns):
            axes[row, col].imshow(images[index][label].numpy())
            axes[row, col].set_title(f'{title} | frame {index}')
            axes[row, col].axis('off')
    fig.tight_layout(); fig.savefig(output/'full_images.png', dpi=140); plt.close(fig)
    if not crops:
        return
    fig, axes = plt.subplots(len(crops), len(columns), figsize=(2.5*len(columns), 2.5*len(crops)), squeeze=False)
    for row, crop in enumerate(crops):
        index = crop['frame_index']
        left, top, right, bottom = crop['box_xyxy']
        for col, (label, title) in enumerate(columns):
            axes[row, col].imshow(images[index][label][top:bottom, left:right].numpy(), interpolation='nearest')
            axes[row, col].set_title(f'{title}\nf{index} GT component {crop["component_label"]}')
            axes[row, col].axis('off')
    fig.tight_layout(); fig.savefig(output/'crops.png', dpi=140); plt.close(fig)


@torch.no_grad()
def compare_foundations(manifest_path, output, device='cpu', perceptual=False):
    """Read saved PNGs only; an explicit train subset retains original frame IDs.

    PNG numbering is always the export ordinal, not the original dataset ID.
    The manifest caller supplies subsets and provenance; this function never
    discovers or infers a model's holdout split from its checkpoint or report.
    Device selects LPIPS execution, never model rendering.
    """
    manifest_path, output = Path(manifest_path), Path(output)
    manifest = json.loads(manifest_path.read_text())
    _write_json(output/'manifest.json', manifest)
    _write_json(output/'config.json', {'manifest': str(manifest_path.resolve()),
                                     'lpips': perceptual, 'lpips_device': device,
                                     'other_metrics_device': 'cpu'})
    root = Path(__file__).resolve().parent
    with tarfile.open(output/'source.tar', 'w') as archive:
        for name in ('foundation_comparison.py', 'diagnose_image_errors.py', 'evaluate.py', 'data.py'):
            archive.add(root/name, arcname=name)
    lpips_model = None
    if perceptual:
        import lpips
        lpips_model = lpips.LPIPS(net='vgg').to(device).eval()
    summary = {'scope': manifest.get('scope',
                        'Saved-image foundation audit; no training, checkpoint loading or rendering. '
                        'Previously observed development tests; different training budgets. '
                        'SSD or any primary-ineligible model is context only regardless of its metrics.'),
               'manual_review': 'pending', 'promoted': False, 'scenes': {}}
    for scene in manifest['scenes']:
        scene_output = output/scene['name']
        scene_output.mkdir()
        if scene['split'] not in ('test', 'train'):
            raise ValueError('Saved-image comparison dataset split must be test or train')
        if scene['split'] == 'train' and scene.get('frame_indices') is None:
            raise ValueError('Train-image comparison requires explicit frame_indices')
        dataset = SceneDataset(scene['dataset'], scene['split'], scene['resolution'])
        frame_indices = _comparison_indices(scene.get('frame_indices'), len(dataset))
        if not set(scene['crop_frames']).issubset(frame_indices):
            raise ValueError('A fixed crop frame is outside the selected canonical frame list')
        reports, mappings, paths = {}, {}, {}
        for label, spec in scene['models'].items():
            reports[label] = json.loads(Path(spec['metrics']).read_text())
            mappings[label] = validate_view_mapping(reports[label], dataset.frames, frame_indices)
            if spec['primary_eligible'] and mappings[label]['mode'] == 'count_only':
                raise ValueError(f'Primary model {label} requires verified per-frame metric names')
            paths[label] = foundation_paths(spec, len(frame_indices))
        if not scene['models'][scene['control']]['primary_eligible']:
            raise ValueError('Foundation control must be eligible under the provenance protocol')
        rows = {label: [] for label in scene['models']}
        target_rows = {label: [] for label in scene['models']}
        metadata_rows, crops, crop_selection, images = [], [], [], {}
        for ordinal, index in enumerate(frame_indices):
            sample = dataset[index]
            if sample['alpha'] is None:
                raise ValueError('Foundation highlight audit requires original target alpha')
            target_bytes = (target_image(sample, scene['background'], scene['display_gamma']).clamp(0, 1)*255).round().byte()
            target, alpha = target_bytes.float()/255, sample['alpha']
            metadata_rows.append({'frame_index': index, 'png_ordinal': ordinal, 'name': sample['name'],
                                  'metadata_frame': dataset.frames[index],
                                  'models': {label: {'prediction': str(pair[ordinal][0].resolve()),
                                                     'saved_target': str(pair[ordinal][1].resolve())}
                                             for label, pair in paths.items()}})
            if index in scene['crop_frames']:
                images[index] = {'GT': target_bytes}
                selected, selection = tiny_crops(target, alpha, index)
                crops.extend(selected); crop_selection.append(selection)
            for label, spec in scene['models'].items():
                prediction_bytes, saved_target = load_foundation_images(spec, ordinal, target.shape[:2])
                target_rows[label].append({'frame_index': index, **target_difference(saved_target, target_bytes)})
                predicted = prediction_bytes.float()/255
                peak = neutral_peak_metrics(predicted, target, alpha)
                peak['false_positive_pixels_2px'] = peak['predicted_pixels']-peak['matched_predicted_pixels_2px']
                row = {'frame_index': index, 'name': sample['name'],
                       'PSNR': float(-10*torch.log10((predicted-target).square().mean())),
                       'SSIM': float(ssim(predicted, target)),
                       'neutral_peak_metrics': peak,
                       'neutral_peak_components': neutral_peak_components(predicted, target, alpha),
                       'ring': ring_error(predicted, target, alpha)}
                if lpips_model is not None:
                    row['LPIPS'] = float(lpips_model(predicted.permute(2, 0, 1)[None].to(device)*2-1,
                                                    target.permute(2, 0, 1)[None].to(device)*2-1))
                rows[label].append(row)
                if index in scene['crop_frames']:
                    images[index][label] = prediction_bytes
            if ordinal == 0 or (ordinal+1) % 10 == 0 or ordinal+1 == len(frame_indices):
                print(json.dumps({'scene': scene['name'], 'frames_completed': ordinal+1,
                                  'total_frames': len(frame_indices)}), flush=True)
        if set(images) != set(scene['crop_frames']):
            raise ValueError('A fixed crop frame is outside the canonical frame list')
        metrics, target_checks = {}, {}
        for label, model_rows in rows.items():
            metrics[label] = {key: sum(row[key] for row in model_rows)/len(model_rows)
                              for key in ('PSNR', 'SSIM', 'LPIPS') if key in model_rows[0]}
            peaks = aggregate_neutral_peak_metrics([row['neutral_peak_metrics'] for row in model_rows])
            peaks['false_positive_pixels_2px'] = peaks['predicted_pixels']-peaks['matched_predicted_pixels_2px']
            metrics[label].update(neutral_peak_metrics=peaks,
                                  neutral_peak_components=aggregate_neutral_peak_components(
                                      [row['neutral_peak_components'] for row in model_rows]),
                                  full_frame_ring=aggregate_ring([row['ring'] for row in model_rows]),
                                  fixed_frame_ring=aggregate_ring([row['ring'] for row in model_rows
                                                                   if row['frame_index'] in scene['crop_frames']]))
            target_checks[label] = aggregate_target_differences(target_rows[label])
        control = scene['control']
        gates = {label: comparison_gates(value, metrics[control], scene['models'][label]['primary_eligible'],
                                        target_checks[label]['target_alignment_review_required'] or
                                        target_checks[control]['target_alignment_review_required'],
                                        scope=manifest.get('scope'))
                 for label, value in metrics.items() if label != control}
        _write_json(scene_output/'metrics.json', {'models': metrics, 'views': rows,
                    'evaluated_frames': len(frame_indices), 'protocol': 'Canonical uint8 RGB; CPU float32 /255; '
                    'unit PSNR; existing zero-padded11x11 SSIM; existing peak/component pools; '
                    '11px GT ring excluding all GT peaks and requiring original alpha>.9; '
                    'optional standard VGG LPIPS [-1,1].'})
        _write_json(scene_output/'target_checks.json', {'models': target_checks, 'views': target_rows})
        _write_json(scene_output/'metadata_mapping.json', {'metadata': str(dataset.metadata_path.resolve()),
                    'dataset_split': scene['split'], 'frame_indices': frame_indices,
                    'models': mappings, 'frames': metadata_rows})
        _write_json(scene_output/'crops.json', {'rule': 'First two 1-4px GT component labels per fixed frame; '
                    '64x64 native crops, no prediction-based selection.', 'selection': crop_selection, 'crops': crops})
        _write_json(scene_output/'gates.json', {'control': control, 'models': gates, 'promoted': False})
        _write_json(scene_output/'provenance.json', {'models': scene['models'],
                    'original_metric_reports': reports, 'mapping_checks': mappings})
        _plot_comparison(scene_output, scene['crop_frames'], crops, images, scene['models'])
        summary['scenes'][scene['name']] = {'control': control, 'evaluated_frames': len(frame_indices),
            'metrics': metrics, 'target_checks': target_checks, 'gates': gates,
            'provenance': {label: {'primary_eligible': spec['primary_eligible'], 'provenance': spec['provenance']}
                           for label, spec in scene['models'].items()}}
        _write_json(output/'summary.json', summary)
    return summary
