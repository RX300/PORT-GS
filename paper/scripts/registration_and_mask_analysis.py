"""Evaluation-only diagnostics requested in the mock reviews (no training, no fitting of any model).

1. Foreground-masked PSNR for all five methods on all 18 scenes: MSE over pixels whose ground-truth
   alpha exceeds 0.5, computed against the common 8-bit targets.
2. Registration diagnostic for the seven real captures (experiment E2(b)): for every test view and
   method, the best global integer 2D translation of the prediction within +-24 px (PSNR on the image
   with a 24 px border removed, identical for all shifts and methods).

Inputs are the saved test predictions (sources.py) and the dataset alphas at the evaluation
resolution (PORT-GS SceneDataset). Output: paper/data/registration_mask_analysis.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from sources import frame_images, METHODS

PORT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PORT))
from data import SceneDataset  # noqa: E402

DATA_ROOT = Path('/workspace/datasets/SSD-GS/data')
FAMILIES = {
    'Real_NRHints': ['Cat', 'CatSmall', 'CupFabric', 'Fish', 'FurScene', 'Pikachu', 'Pixiu'],
    'Synthetic_GS3': ['AnisoMetal', 'Drums', 'FurBall', 'Hotdog', 'Lego', 'Translucent'],
    'Synthetic_SSS-GS': ['bunny_small', 'candle_small', 'dragon_small', 'soap_small', 'statue_small'],
}
RADIUS = 24


def psnr(mse):
    return float(-10 * np.log10(max(mse, 1e-12)))


@torch.no_grad()
def best_shift(pred, gt):
    """Best integer translation of `pred` within +-RADIUS; returns (psnr_unaligned, psnr_aligned, dy, dx)."""
    p = torch.from_numpy(pred).float().cuda() / 255
    g = torch.from_numpy(gt).float().cuda() / 255
    h, w, _ = g.shape
    r = RADIUS
    core = g[r:h - r, r:w - r]
    best = (float('inf'), 0, 0)
    for dy in range(-r, r + 1):
        rows = p[r + dy:h - r + dy]
        shifted = torch.stack([rows[:, r + dx:w - r + dx] for dx in range(-r, r + 1)])
        mse = ((shifted - core) ** 2).mean(dim=(1, 2, 3))
        value, index = mse.min(0)
        if float(value) < best[0]:
            best = (float(value), dy, int(index) - r)
    unaligned = float(((p[r:h - r, r:w - r] - core) ** 2).mean())
    return psnr(unaligned), psnr(best[0]), best[1], best[2]


def main():
    out = {}
    for family, scenes in FAMILIES.items():
        resolution = 256 if family == 'Synthetic_SSS-GS' else 512
        for scene in scenes:
            dataset = SceneDataset(DATA_ROOT / family / scene, 'test', resolution,
                                   unit_light_intensity=1.0 if family == 'Synthetic_SSS-GS' else None)
            rows = {m: dict(masked=[], aligned=[], unaligned=[], shift=[]) for m in METHODS}
            background_mismatch = []
            for index in range(len(dataset)):
                alpha = dataset[index]['alpha'][..., 0].numpy()
                images = frame_images(family, scene, index)
                gt = images['GT'].astype(np.float64) / 255
                assert gt.shape[:2] == alpha.shape, (family, scene, index, gt.shape, alpha.shape)
                mask = alpha > 0.5
                bg = 1.0 if family == 'Synthetic_GS3' else 0.0
                background_mismatch.append(float(np.abs(gt[alpha < 0.01] - bg).max()) if (alpha < 0.01).any() else 0.0)
                for method in METHODS:
                    pred = images[method].astype(np.float64) / 255
                    err = ((pred - gt) ** 2).mean(-1)
                    rows[method]['masked'].append(psnr(err[mask].mean()))
                    if family == 'Real_NRHints':
                        a, b, dy, dx = best_shift(images[method], images['GT'])
                        rows[method]['unaligned'].append(a)
                        rows[method]['aligned'].append(b)
                        rows[method]['shift'].append(float(np.hypot(dy, dx)))
            out[f'{family}/{scene}'] = {
                m: {k: (float(np.mean(v)) if v else None) for k, v in rows[m].items()} for m in METHODS}
            out[f'{family}/{scene}']['frames'] = len(dataset)
            out[f'{family}/{scene}']['max_background_mismatch'] = float(np.max(background_mismatch))
            print(family, scene, {m: round(out[f'{family}/{scene}'][m]['masked'], 2) for m in METHODS},
                  {m: (round(out[f'{family}/{scene}'][m]['aligned'], 2) if out[f'{family}/{scene}'][m]['aligned'] else None)
                   for m in METHODS}, flush=True)
    (PORT / 'paper/data/registration_mask_analysis.json').write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
