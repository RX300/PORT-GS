"""Real Cat CLI training, checkpoint reload and evaluation for every method.

Run explicitly on one idle GPU; all evidence is written beneath --output.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

from evaluate import load_model, render_observation, to_device
from data import SceneDataset
from methods import METHODS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scene', default='/workspace/datasets/SSD-GS/data/Real_NRHints/Cat')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    report = {}
    for name in METHODS:
        run = output / name
        command = [sys.executable, str(root/'train.py'), '--scene', args.scene,
                   '--output', str(run), '--representation', name, '--steps', '3',
                   '--resolution', '32', '--points', '128', '--max-points', '128',
                   '--rank', '8', '--shadow-start', '1', '--port-start', '1',
                   '--refine-stop', '0', '--validate-every', '0', '--fit-all']
        (output/f'{name}.command.json').write_text(json.dumps(command, indent=2)+'\n')
        with (output/f'{name}.train.log').open('w') as log:
            subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
        g, model, saved = load_model(run/'last.pt')
        assert saved['step'] == 3 and saved['config']['representation'] == name
        assert [p.name for p in run.glob('*.pt')] == ['last.pt']
        for key, parameter in model.named_parameters():
            assert torch.isfinite(parameter).all(), (name, key)
        sample = to_device(SceneDataset(args.scene, 'train', 32)[0], 'cuda')
        with torch.no_grad():
            first = render_observation(g, model, sample, 0., True, True, 2.2, 'deep')[0]
            assert torch.isfinite(first).all()
            second_g, second_model, _ = load_model(run/'last.pt')
            second = render_observation(second_g, second_model, sample, 0., True, True, 2.2, 'deep')[0]
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        eval_command = [sys.executable, str(root/'evaluate.py'), str(run/'last.pt'),
                        '--split', 'fit', '--limit', '1', '--output', str(run/'evaluation')]
        with (output/f'{name}.eval.log').open('w') as log:
            subprocess.run(eval_command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
        metrics = json.loads((run/'evaluation/metrics.json').read_text())
        history = [json.loads(line) for line in (run/'history.jsonl').read_text().splitlines()]
        report[name] = dict(step=saved['step'], final_loss=history[-1]['loss'],
                            reload_pixel_max_abs=float((first-second).abs().max()),
                            metrics=metrics, parameters=sum(p.numel() for p in model.parameters()))
        (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print(f'PASS: {name}: CLI train / final checkpoint / reload / CLI evaluation', flush=True)
        del g, model, second_g, second_model, first, second, sample, saved
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
