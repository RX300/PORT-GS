"""Rank test frames per scene by LiSA's PSNR margin over the best baseline (common-GT metrics).

Reads only the saved per-frame common-ground-truth reports; no rendering, no test fitting.
Prints the frames at selected percentiles of the margin so qualitative figures can use
representative (not only best-case) views.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2] / 'runs/lisa_supplementary_experiments/common_ground_truth'
METHODS = ('LiSA-staged', 'SSD-GS', 'GS3', 'OLAT-Gaussians', 'RNG')


def per_frame(family, scene):
    reports = {m: json.loads((ROOT / family / scene / f'{m}.json').read_text()) for m in METHODS}
    psnr = {m: np.array([v['PSNR'] for v in r['views']]) for m, r in reports.items()}
    names = [v['name'] for v in reports['LiSA-staged']['views']]
    return names, psnr


if __name__ == '__main__':
    for spec in sys.argv[1:]:
        family, scene = spec.split('/')
        names, psnr = per_frame(family, scene)
        base = np.stack([psnr[m] for m in METHODS[1:4]])  # strongest baselines (exclude RNG)
        margin = psnr['LiSA-staged'] - base.max(0)
        order = np.argsort(margin)
        print(f'== {spec}: frames={len(names)} median margin {np.median(margin):+.2f} dB, '
              f'LiSA wins {np.mean(margin > 0) * 100:.0f}% of frames')
        for q in (5, 25, 50, 75, 90, 97):
            i = order[min(len(order) - 1, int(q / 100 * len(order)))]
            row = ' '.join(f'{m[:5]}={psnr[m][i]:.2f}' for m in METHODS)
            print(f'  p{q:02d} idx {i:4d} ({names[i]}) margin {margin[i]:+.2f}  {row}')
