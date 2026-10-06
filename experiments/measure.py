"""Measure a final LiSA checkpoint and stream continuous relighting videos."""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from data import SceneDataset
from evaluate import load_model, render_observation, to_device

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / 'benchmarks'))
from measure_relighting import timed_modes


@torch.no_grad()
def measure(checkpoint_path, output, frame_count, video_frames, ffmpeg):
    torch.set_num_threads(8)
    gaussians, transport, checkpoint = load_model(checkpoint_path)
    transport.eval()
    config = checkpoint['config']
    dataset = SceneDataset(config['scene'], 'test', config['resolution'], config['unit_light_intensity'])
    indices = np.linspace(0, len(dataset) - 1, frame_count, dtype=int).tolist()
    samples = []
    for index in indices:
        source = dataset[index]
        sample = to_device(source, 'cuda')
        sample.update(image=source['image'], alpha=source['alpha'])
        samples.append(sample)

    def render(sample):
        return render_observation(gaussians, transport, sample, config['background'], True, True,
                                  config['display_gamma'], config['shadow_mode'])[0]

    def frame(index, mode):
        sample = dict(samples[index])
        if mode == 'fixed_light':
            sample['light_pos'] = samples[0]['light_pos']
            sample['light_intensity'] = samples[0]['light_intensity']
        return render(sample)

    modes = timed_modes(frame, len(samples))
    output.mkdir(parents=True, exist_ok=False)
    report = dict(method='LiSA-staged', checkpoint=str(checkpoint_path), frame_indices=indices,
                  modes=modes, gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
                  cuda_version=torch.version.cuda, gaussians=len(gaussians.params['means']),
                  image_size=list(samples[0]['image'].shape[:2]),
                  checkpoint_MiB=checkpoint_path.stat().st_size / 2**20,
                  checkpoint_contents='Original loader-required checkpoint, including any training state; not a stripped deployment export.',
                  protocol='Original test cameras; renderer+observation transform; synchronized wall time; '
                           'excludes loading/metrics/disk IO; no added atlas caching; '
                           'fixed_light uses the first selected light.')
    (output / 'performance.json').write_text(json.dumps(report, indent=2) + '\n')
    base = samples[0]
    center = gaussians.center
    height, width = base['image'].shape[:2]
    videos = []
    for mode in ('moving_light', 'moving_camera', 'moving_camera_and_light'):
        path = output / f'{mode}.mp4'
        process = subprocess.Popen([
            str(ffmpeg), '-hide_banner', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
            '-s', f'{width}x{height}', '-r', '24', '-i', '-', '-an', '-c:v', 'libx264',
            '-crf', '18', '-pix_fmt', 'yuv420p', str(path)], stdin=subprocess.PIPE)
        changes = []
        previous = None
        for index in range(video_frames):
            sample = dict(base)
            phase = 2 * math.pi * index / video_frames
            if mode in ('moving_camera', 'moving_camera_and_light'):
                angle = math.sin(phase) * math.pi / 6
                transform = torch.eye(4, device='cuda')
                transform[:3, :3] = torch.tensor([
                    [math.cos(angle), -math.sin(angle), 0],
                    [math.sin(angle), math.cos(angle), 0], [0, 0, 1]], device='cuda')
                transform[:3, 3] = center - transform[:3, :3] @ center
                sample['c2w'] = transform @ base['c2w']
                sample['viewmat'] = base['viewmat'] @ torch.linalg.inv(transform)
            if mode in ('moving_light', 'moving_camera_and_light'):
                angle = math.sin(phase) * math.pi / 4
                rotation = center.new_tensor([[math.cos(angle), -math.sin(angle), 0],
                                              [math.sin(angle), math.cos(angle), 0], [0, 0, 1]])
                sample['light_pos'] = center + rotation @ (base['light_pos'] - center)
            image = render(sample)
            assert torch.isfinite(image).all(), (mode, index)
            encoded = image.clamp(0, 1)
            if previous is not None:
                changes.append((encoded - previous).abs().mean().item())
            previous = encoded
            process.stdin.write((encoded * 255).round().byte().cpu().numpy().tobytes())
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError(f'ffmpeg failed for {path}')
        videos.append(dict(mode=mode, path=str(path), frames=video_frames,
                           mean_frame_change=float(np.mean(changes)), max_frame_change=max(changes)))
    (output / 'continuity.json').write_text(json.dumps(dict(
        checkpoint=str(checkpoint_path), videos=videos,
        trajectory='Sinusoidal world-Z camera orbit ±30 degrees and light orbit ±45 degrees around scene center.',
        interpretation='No GT on these paths. Frame differences include intended motion/lighting changes '
                       'and are diagnostics, not ground-truth flicker error.'), indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=32)
    parser.add_argument('--video-frames', type=int, default=96)
    parser.add_argument('--ffmpeg', type=Path, required=True)
    args = parser.parse_args()
    measure(args.checkpoint, args.output, args.frames, args.video_frames, args.ffmpeg)
