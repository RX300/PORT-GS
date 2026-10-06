"""Re-evaluate saved predictions against one identical, unaligned uint8 target."""

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from lpips import LPIPS

from evaluate import ssim

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('LiSA-staged', 'GS3', 'SSD-GS', 'RNG', 'OLAT-Gaussians')


def rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB')).copy()


@torch.no_grad()
def evaluate(family, scene, output):
    torch.set_num_threads(8)
    manifest = json.loads((ROOT.parent / 'OLAT-Gaussians/runs/full_dataset/comparison_manifest.json').read_text())
    sources = {method: next(job for job in manifest['jobs']
                           if (job['method'], job['family'], job['scene']) == (method, family, scene))
               for method in METHODS}
    reports = {method: json.loads(Path(job['resultpath']).read_text()) for method, job in sources.items()}
    frames = json.loads((Path(sources['LiSA-staged']['dataset']) / 'transforms_test.json').read_text())['frames']
    for report in reports.values():
        assert [view['frame_index'] for view in report['views']] == list(range(len(frames)))
    rows = {method: [] for method in METHODS}
    differences = {method: dict(differing_channels=0, total_channels=0, maximum_byte_difference=0,
                                absolute_byte_difference=0) for method in METHODS}
    perceptual = LPIPS(net='vgg', version='0.1').cuda().eval()
    reference_directory = Path(sources['LiSA-staged']['resultpath']).parent
    for index, frame in enumerate(frames):
        pair = rgb(reference_directory / f'pair_{index:03d}.png')
        target_bytes, lisa_bytes = np.split(pair, 2, axis=1)
        target = torch.from_numpy(target_bytes.astype(np.float32) / 255).cuda()
        filename = frame['file_path'] if 'file_path' in frame else frame['file_paths'][0]
        for method in METHODS:
            view = reports[method]['views'][index]
            assert view['name'] == Path(filename).stem
            if method == 'LiSA-staged':
                prediction_bytes, old_target = lisa_bytes, target_bytes
            else:
                prediction_bytes = rgb(ROOT.parent / method / view['render'])
                old_target = rgb(ROOT.parent / method / view['gt'])
            assert prediction_bytes.shape == target_bytes.shape == old_target.shape, (method, family, scene, index)
            delta = np.abs(old_target.astype(np.int16) - target_bytes.astype(np.int16))
            audit = differences[method]
            audit['differing_channels'] += int(np.count_nonzero(delta))
            audit['total_channels'] += int(delta.size)
            audit['maximum_byte_difference'] = max(audit['maximum_byte_difference'], int(delta.max()))
            audit['absolute_byte_difference'] += int(delta.sum())
            predicted = torch.from_numpy(prediction_bytes.astype(np.float32) / 255).cuda()
            mse = (predicted - target).square().mean()
            rows[method].append(dict(frame_index=index, name=view['name'], file_path=filename,
                PSNR=(-10 * torch.log10(mse)).item(), SSIM=ssim(predicted, target).item(),
                LPIPS=perceptual(predicted.permute(2, 0, 1)[None] * 2 - 1,
                                 target.permute(2, 0, 1)[None] * 2 - 1).item(), unitRGBMSE=mse.item()))
    output.mkdir(parents=True, exist_ok=False)
    for method in METHODS:
        metrics = {key: statistics.mean(row[key] for row in rows[method])
                   for key in ('PSNR', 'SSIM', 'LPIPS', 'unitRGBMSE')}
        report = dict(method=method, family=family, scene=scene, metrics=metrics, views=rows[method],
                      prediction_source=sources[method]['resultpath'],
                      target_source=str(reference_directory), gt_difference=differences[method],
                      protocol='Common uint8 targets from left panels of the complete LiSA evaluation; '
                               'these are dataset targets, not model predictions. No alignment, fitting, '
                               'resizing, re-rendering or model changes. Unit RGB PSNR, 11x11 zero-pad SSIM, VGG LPIPS.')
        (output / f'{method}.json').write_text(json.dumps(report, indent=2) + '\n')
    (output / 'audit.json').write_text(json.dumps(dict(
        family=family, scene=scene, frames=len(frames), methods=list(METHODS),
        ground_truth_differences=differences), indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', required=True)
    parser.add_argument('--scene', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.family, args.scene, args.output)
