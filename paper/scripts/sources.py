"""Locate saved test predictions of all compared methods (read-only)."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

PORT = Path(__file__).resolve().parents[2]
WORKSPACE = PORT.parent
MANIFEST = WORKSPACE / 'OLAT-Gaussians/runs/full_dataset/comparison_manifest.json'
METHODS = ('LiSA-staged', 'SSD-GS', 'GS3', 'OLAT-Gaussians', 'RNG')
LABELS = {'LiSA-staged': 'Ours', 'SSD-GS': 'SSD-GS', 'GS3': 'GS$^3$', 'OLAT-Gaussians': 'OLAT-GS',
          'RNG': 'RNG', 'GT': 'Ground truth'}


def _rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB')).copy()


def frame_images(family, scene, index):
    """Return {method: uint8 HxWx3} plus 'GT' (the common target used for all metrics)."""
    manifest = json.loads(MANIFEST.read_text())
    jobs = {job['method']: job for job in manifest['jobs']
            if (job['family'], job['scene']) == (family, scene) and job['method'] in METHODS}
    lisa_dir = Path(jobs['LiSA-staged']['resultpath']).parent
    pair = _rgb(lisa_dir / f'pair_{index:03d}.png')
    gt, ours = np.split(pair, 2, axis=1)
    images = {'GT': gt, 'LiSA-staged': ours}
    for method in METHODS[1:]:
        report = json.loads(Path(jobs[method]['resultpath']).read_text())
        view = report['views'][index]
        images[method] = _rgb(WORKSPACE / method / view['render'])
    return images
