"""Build every raster panel used in the paper from saved, real model outputs.

Inputs (read-only):
  * final LiSA test pairs and atlas buffers (``paper/data/renders/atlas_*``, exported with
    ``diagnose_image_errors.py --light-atlas-preview`` from the final checkpoints);
  * saved test predictions of the baselines (located through ``sources.py``);
  * the matched 16k loss-domain study (``runs/lisa_radiometric_curriculum_20261004`` and the
    control rendered to ``paper/data/renders/display_loss_control_16k_*``);
  * seed-0 structure-ablation test previews.
Nothing is re-trained or re-fitted. Every panel is written to ``paper/latex/figures/``.
"""
import json
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from PIL import Image, ImageDraw

from sources import frame_images, METHODS

PORT = Path(__file__).resolve().parents[2]
RENDERS = PORT / 'paper/data/renders'
OUT = PORT / 'paper/latex/figures'
GAMMA = 2.2


def save(array, path, quality=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
    if path.suffix == '.jpg':
        image.save(path, quality=quality or 92, subsampling=0)
    else:
        image.save(path)


def encode(linear):
    """Display encoding used by the evaluation (clip, gamma 2.2, 8 bit)."""
    return np.clip(linear, 0, 1) ** (1 / GAMMA) * 255


def colorize(values, mask, cmap, lo, hi):
    unit = np.clip((values - lo) / max(hi - lo, 1e-8), 0, 1)
    rgb = colormaps[cmap](unit)[..., :3] * 255
    rgb[~mask] = 0
    return rgb


def crop(image, box):
    y0, x0, size = box
    return image[y0:y0 + size, x0:x0 + size]


def upscale(image, size):
    return np.asarray(Image.fromarray(image.astype(np.uint8)).resize((size, size), Image.LANCZOS))


def draw_box(image, box, color=(230, 40, 40), width=3):
    y0, x0, size = box
    canvas = Image.fromarray(image.astype(np.uint8))
    ImageDraw.Draw(canvas).rectangle([x0, y0, x0 + size - 1, y0 + size - 1], outline=color, width=width)
    return np.asarray(canvas)


# ---------------------------------------------------------------- teaser / decomposition
def decomposition_panels(scene, frame, name, crop_box=None, background=1.0):
    """Light-space atlas channels and camera-space radiance components of one test frame."""
    buffers = np.load(RENDERS / f'atlas_{scene}' / f'frame_{frame:03d}_buffers.npz')
    level0 = buffers['atlas_level_0']  # [flux (8), M1, M2, M0]
    coverage = level0[-1]
    covered = coverage > 1e-3
    mean = np.where(covered, level0[8] / np.maximum(coverage, 1e-4), 0)
    lo, hi = np.percentile(mean[covered], [1, 99])
    out = OUT / name
    save(colorize(mean, covered, 'viridis', lo, hi), out / 'atlas_depth.png')
    flux = level0[0]
    save(colorize(flux, covered, 'inferno', 0, np.percentile(flux[covered], 99)), out / 'atlas_flux.png')
    # Coarse pyramid level, nearest-upsampled, to show the multi-scale gather.
    coarse = buffers['atlas_level_3']
    cov3 = coarse[-1]
    flux3 = np.where(cov3 > 1e-3, coarse[0], 0)
    big = np.kron(flux3, np.ones((8, 8)))
    save(colorize(big, np.kron(cov3, np.ones((8, 8))) > 1e-3, 'inferno', 0,
                  np.percentile(flux[covered], 99)), out / 'atlas_flux_level3.png')
    alpha = buffers['alpha'][..., 0]
    visibility = buffers['visibility']
    vis = np.repeat((np.clip(visibility, 0, 1) * 255)[..., None], 3, -1)
    vis[alpha < 0.5] = 255 if background else 0
    local = buffers['local_linear'] + buffers['specular_linear']
    transfer = buffers['transfer_linear']
    # Components are additive in linear radiance; show each on black, encoded like the render.
    save(encode(local * alpha[..., None]), out / 'local.png')
    save(encode(transfer * alpha[..., None]), out / 'transfer.png')
    save(vis, out / 'visibility.png')
    pair = np.asarray(Image.open(RENDERS / f'atlas_{scene}' / f'frame_{frame:03d}_pair.png').convert('RGB'))
    gt, ours = np.split(pair, 2, axis=1)
    save(gt, out / 'gt.png')
    save(ours, out / 'ours.png')
    if crop_box is not None:
        for key in ('visibility', 'local', 'transfer', 'gt', 'ours'):
            image = np.asarray(Image.open(out / f'{key}.png').convert('RGB'))
            save(crop(image, crop_box), out / f'{key}_crop.png')
    return out


# ---------------------------------------------------------------- qualitative comparison
def comparison_row(family, scene, frame, box, name, zoom=256):
    images = frame_images(family, scene, frame)
    out = OUT / 'comparison' / name
    full = draw_box(images['GT'], box, width=max(2, images['GT'].shape[0] // 170))
    save(full, out / 'full.png')
    psnr = {}
    gt = images['GT'].astype(np.float64) / 255
    for key in ('GT',) + METHODS:
        save(upscale(crop(images[key], box), zoom), out / f'{key}.png')
        if key != 'GT':
            mse = np.mean((images[key].astype(np.float64) / 255 - gt) ** 2)
            psnr[key] = float(-10 * np.log10(mse))
    (out / 'psnr.json').write_text(json.dumps(psnr, indent=1) + '\n')
    return psnr


# ---------------------------------------------------------------- loss-domain study
def loss_domain(scene, box, name, zoom=256):
    """Frame r_5 of the training-internal holdout: matched 16k display-loss control vs. ours."""
    ours = np.asarray(Image.open(PORT / 'runs/lisa_radiometric_curriculum_20261004/curriculum/Synthetic_GS3'
                                 / scene / 'validation/pair_000.png').convert('RGB'))
    control = np.asarray(Image.open(RENDERS / f'display_loss_control_16k_{scene}/pair_000.png').convert('RGB'))
    gt, ours = np.split(ours, 2, axis=1)
    gt_control, control = np.split(control, 2, axis=1)
    assert np.array_equal(gt, gt_control)
    out = OUT / 'loss_domain' / name
    for key, image in (('gt', gt), ('control', control), ('ours', ours)):
        save(image, out / f'{key}.png')
        save(upscale(crop(image, box), zoom), out / f'{key}_crop.png')
    save(draw_box(gt, box), out / 'gt_box.png')


# ---------------------------------------------------------------- transfer ablation
def ablation(family, scene, frame, box, name, zoom=256):
    root = PORT / 'runs/lisa_supplementary_experiments/structure_ablation'
    full = np.asarray(Image.open(PORT / 'runs/lisa_staged_full_dataset_20261004' / family / scene
                                 / f'appearance/test/pair_{frame:03d}.png').convert('RGB'))
    gt, ours = np.split(full, 2, axis=1)
    out = OUT / 'ablation' / name
    panels = {'gt': gt, 'full': ours}
    for variant in ('no_transfer', 'local_residual'):
        pair = np.asarray(Image.open(root / variant / family / scene
                                     / f'appearance/test/pair_{frame:03d}.png').convert('RGB'))
        gt_variant, prediction = np.split(pair, 2, axis=1)
        assert np.array_equal(gt_variant, gt)
        panels[variant] = prediction
    psnr = {}
    for key, image in panels.items():
        save(upscale(crop(image, box), zoom), out / f'{key}.png')
        if key != 'gt':
            error = np.abs(image.astype(np.float64) - gt.astype(np.float64)).mean(-1) / 255
            heat = colormaps['magma'](np.clip(error / 0.15, 0, 1))[..., :3] * 255
            save(upscale(crop(heat, box), zoom), out / f'{key}_error.png')
        if key != 'gt':
            mse = np.mean((image.astype(np.float64) - gt.astype(np.float64)) ** 2) / 255 ** 2
            psnr[key] = float(-10 * np.log10(mse))
    save(draw_box(gt, box), out / 'gt_box.png')
    (out / 'psnr.json').write_text(json.dumps(psnr, indent=1) + '\n')
    return psnr


if __name__ == '__main__':
    # Boxes are (row, column, size) in pixels of the stored test image.
    decomposition_panels('Translucent', 70, 'teaser')
    decomposition_panels('Pixiu', 60, 'decomp_pixiu', background=0.0)
    decomposition_panels('Hotdog', 59, 'decomp_hotdog')
    rows = {
        'translucent': ('Synthetic_GS3', 'Translucent', 70, (250, 0, 200)),
        'hotdog': ('Synthetic_GS3', 'Hotdog', 59, (250, 250, 200)),
        'fish': ('Real_NRHints', 'Fish', 47, (180, 170, 220)),
        'pixiu': ('Real_NRHints', 'Pixiu', 60, (160, 190, 200)),
        'statue': ('Synthetic_SSS-GS', 'statue_small', 53, (20, 70, 110)),
    }
    rows.update({
        'fail_drums': ('Synthetic_GS3', 'Drums', 247, (30, 150, 120)),
        'fail_anisometal': ('Synthetic_GS3', 'AnisoMetal', 89, (225, 185, 210)),
        'fail_furscene': ('Real_NRHints', 'FurScene', 63, (150, 150, 220)),
        'pixiu_median': ('Real_NRHints', 'Pixiu', 60, (160, 190, 200)),
        'pikachu': ('Real_NRHints', 'Pikachu', 108, (70, 220, 230)),
    })
    summary = {name: comparison_row(*spec[:3], spec[3], name) for name, spec in rows.items()}
    print(json.dumps(summary, indent=1))
    loss_domain('Lego', (120, 140, 230), 'lego')
    loss_domain('Drums', (200, 180, 230), 'drums')
    print(json.dumps({
        'translucent': ablation('Synthetic_GS3', 'Translucent', 0, (150, 150, 260), 'translucent'),
        'soap': ablation('Synthetic_SSS-GS', 'soap_small', 0, (40, 40, 180), 'soap'),
    }, indent=1))
