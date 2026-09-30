#!/usr/bin/env python3
"""Resolve the six-scene validation JSON into a fresh scheduler manifest."""

import argparse
import json
import subprocess
import tarfile
from pathlib import Path

from methods import DEFAULT_METHOD, METHODS


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs/validation.json"


def train_argv(python, options):
    argv = [python, "-u", str(ROOT / "train.py")]
    for name, value in options.items():
        if isinstance(value, bool):
            if value:
                argv.append(f"--{name}")
            elif name == "absgrad":
                argv.append("--no-absgrad")
        elif isinstance(value, list):
            argv.extend([f"--{name}", *map(str,value)])
        else:
            argv.extend([f"--{name}", str(value)])
    return argv


def build_manifest(config, output_dir):
    """Optional named ablations share the same queue and one source snapshot."""
    if "variants" not in config:
        return _build_single_manifest(config, output_dir)
    jobs = []
    for name, overrides in config["variants"].items():
        child = {k: v for k,v in config.items() if k != "variants"}
        train_overrides = {k:v for k,v in overrides.items() if k != "scenes"}
        child["train"] = {k:v for k,v in {**config["train"],**train_overrides}.items() if v is not None}
        if "scenes" in overrides:
            child["families"] = {
                family: dict(settings, scenes=selected)
                for family,settings in config["families"].items()
                if (selected := [scene for scene in settings["scenes"] if scene in overrides["scenes"]])
            }
            if not child["families"]:
                raise ValueError(f"Variant {name} selects no configured scenes")
        manifest = _build_single_manifest(child, output_dir / name)
        for job in manifest["jobs"]:
            job["id"] += "-" + name
            job["variant"] = name
            job["source_archive"] = str(output_dir / "source.tar")
        jobs.extend(manifest["jobs"])
    manifest.update(status_path=str(output_dir/"status.json"), source_archive=str(output_dir/"source.tar"), jobs=jobs)
    manifest["protocol"]["representation"] = "controlled_variants"
    manifest["protocol"]["train"] = config["train"]
    manifest["protocol"]["variants"] = config["variants"]
    return manifest


def _build_single_manifest(config, output_dir):
    python = config["python"]
    source_archive = output_dir / "source.tar"
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    jobs = []
    eval_split = config.get("eval_split", "test")
    eval_branch = config.get('eval_branch', 'gaussian')
    eval_limit = config.get('eval_limit', 0)
    if eval_branch not in ['gaussian', 'sdf_volume'] or eval_limit < 0:
        raise ValueError('Evaluation branch must be gaussian/sdf_volume and limit nonnegative')
    eval_directory = eval_split+('_sdf' if eval_branch == 'sdf_volume' else '')
    eval_options = ['--sdf-volume'] if eval_branch == 'sdf_volume' else []
    if eval_limit:
        eval_options += ['--limit', str(eval_limit)]
    if config.get('eval_resolution'):
        eval_options += ['--resolution', str(config['eval_resolution'])]
    if config.get('eval_highlights', False):
        eval_options += ['--highlights']
    if config.get('eval_save_all', False):
        eval_options += ['--save-all']
    if config.get('eval_shift_align', 0):
        eval_options += ['--shift-align', str(config['eval_shift_align'])]
    for family, settings in config["families"].items():
        for scene in settings["scenes"]:
            output = output_dir / family / scene
            dataset = Path(config["data_root"]) / family / scene
            options = {**config["train"], **settings["train"],
                       "scene": str(dataset), "output": str(output)}
            representation = options.get("representation", DEFAULT_METHOD)
            if representation not in METHODS:
                raise ValueError(f"Unknown transport method: {representation}")
            for key in ('surface-priors', 'init-checkpoint', 'init-geometry'):
                if key in options:
                    options[key] = options[key].format(family=family, scene=scene)
            preparation = []
            if "surface_preprocessing" in config:
                settings_prior = config["surface_preprocessing"]
                prior_dir = Path(options['surface-priors'])
                for kind in settings_prior.get('kinds', ("normal", "depth")):
                    argv = [settings_prior[f"{kind}_python"], "-u", str(ROOT / "prepare_surface_priors.py"),
                            "--kind", kind, "--scene", str(dataset), "--output", str(prior_dir),
                            "--processing-resolution", str(settings_prior["processing_resolution"])]
                    for key in ("resolution", "background", "display-gamma", "unit-light-intensity", "seed"):
                        if key in options:
                            argv.extend([f"--{key}", str(options[key])])
                    if options.get("fit-all", False):
                        argv.append("--fit-all")
                    preparation.append({
                        "name": f"prepare_{kind}", "state": "pending", "cwd": str(ROOT), "argv": argv,
                        "logpath": str(output_dir / family / f"{scene}.{kind}.log"),
                        "output": str(prior_dir), "resultpath": str(prior_dir / f"{kind}.pt"),
                    })
            checkpoints = [(f"step_{step:06d}.pt",f"{eval_directory}_{step:06d}",f"eval_{step:06d}")
                           for step in options.get("save-steps", [])]
            checkpoints.append(("last.pt",eval_directory,"eval"))
            jobs.append({
                "id": f"PORT-{family}-{scene}", "method": "PORT-GS", "family": family,
                "scene": scene, "dataset": str(dataset), "state": "pending", "mode": "fresh",
                "representation": options.get("representation", DEFAULT_METHOD),
                "source_archive": str(source_archive), "source_revision": revision,
                "output": str(output), "resultpath": str(output / eval_directory / "metrics.json"),
                "steps": preparation + [
                    {"name": "train", "state": "pending", "cwd": str(ROOT),
                     "argv": train_argv(python, options),
                     "logpath": str(output_dir / family / f"{scene}.train.log"),
                     "output": str(output), "resultpath": str(output / "last.pt"),
                     "progress_path": str(output / "history.jsonl")},
                ] + [
                    {"name": phase_name, "state": "pending", "cwd": str(ROOT),
                     "argv": [python, "-u", str(ROOT / "evaluate.py"), str(output / snapshot),
                              "--split", eval_split, "--output", str(output / directory), "--lpips"]+eval_options,
                     "logpath": str(output_dir / family / f"{scene}.{directory}.log"),
                     "output": str(output / directory), "resultpath": str(output / directory / "metrics.json")}
                    for snapshot,directory,phase_name in checkpoints
                ],
            })
            if config.get('eval_test', False):
                if eval_split == 'test':
                    raise ValueError('eval_test is an additional full terminal test after a non-test evaluation')
                directory = 'test'+('_sdf' if eval_branch == 'sdf_volume' else '')
                test_options = ['--sdf-volume'] if eval_branch == 'sdf_volume' else []
                if config.get('eval_resolution'):
                    test_options += ['--resolution', str(config['eval_resolution'])]
                if config.get('eval_highlights', False):
                    test_options += ['--highlights']
                if config.get('eval_save_all', False):
                    test_options += ['--save-all']
                if config.get('eval_shift_align', 0):
                    test_options += ['--shift-align', str(config['eval_shift_align'])]
                jobs[-1]['steps'].append({
                    'name': 'eval_test', 'state': 'pending', 'cwd': str(ROOT),
                    'argv': [python, '-u', str(ROOT/'evaluate.py'), str(output/'last.pt'),
                             '--split', 'test', '--output', str(output/directory), '--lpips']+test_options,
                    'logpath': str(output_dir/family/f'{scene}.{directory}.log'),
                    'output': str(output/directory), 'resultpath': str(output/directory/'metrics.json'),
                })
            if config.get('eval_full_fit', False):
                if eval_split == 'fit' and not eval_limit:
                    raise ValueError('eval_full_fit would repeat the existing complete fit evaluation')
                directory = 'fit_full'+('_sdf' if eval_branch == 'sdf_volume' else '')
                full_options = ['--sdf-volume'] if eval_branch == 'sdf_volume' else []
                if config.get('eval_resolution'):
                    full_options += ['--resolution', str(config['eval_resolution'])]
                if config.get('eval_highlights', False):
                    full_options += ['--highlights']
                if config.get('eval_save_all', False):
                    full_options += ['--save-all']
                if config.get('eval_shift_align', 0):
                    full_options += ['--shift-align', str(config['eval_shift_align'])]
                jobs[-1]['steps'].append({
                    'name': 'eval_full_fit', 'state': 'pending', 'cwd': str(ROOT),
                    'argv': [python, '-u', str(ROOT/'evaluate.py'), str(output/'last.pt'),
                             '--split', 'fit', '--output', str(output/directory), '--lpips']+full_options,
                    'logpath': str(output_dir/family/f'{scene}.{directory}.log'),
                    'output': str(output/directory), 'resultpath': str(output/directory/'metrics.json'),
                })
            if config.get('eval_calibrate_test', 0):
                # Secondary literature-style protocol: per-view camera/light fit on test GT, scene frozen.
                jobs[-1]['steps'].append({
                    'name': 'eval_test_calibrated', 'state': 'pending', 'cwd': str(ROOT),
                    'argv': [python, '-u', str(ROOT/'evaluate.py'), str(output/'last.pt'),
                             '--split', 'test', '--output', str(output/'test_calibrated'), '--lpips',
                             '--resolution', str(config.get('eval_resolution') or options.get('resolution', 512)),
                             '--calibrate-test-views', str(config['eval_calibrate_test'])],
                    'logpath': str(output_dir/family/f'{scene}.test_calibrated.log'),
                    'output': str(output/'test_calibrated'),
                    'resultpath': str(output/'test_calibrated'/'metrics.json'),
                })
            if config.get('eval_validation', False):
                if eval_split == 'validation' and not eval_limit:
                    raise ValueError('eval_validation would repeat the existing complete validation evaluation')
                directory = 'validation'+('_sdf' if eval_branch == 'sdf_volume' else '')
                validation_options = ['--sdf-volume'] if eval_branch == 'sdf_volume' else []
                if config.get('eval_resolution'):
                    validation_options += ['--resolution', str(config['eval_resolution'])]
                if config.get('eval_highlights', False):
                    validation_options += ['--highlights']
                if config.get('eval_save_all', False):
                    validation_options += ['--save-all']
                if config.get('eval_shift_align', 0):
                    validation_options += ['--shift-align', str(config['eval_shift_align'])]
                jobs[-1]['steps'].append({
                    'name': 'eval_validation', 'state': 'pending', 'cwd': str(ROOT),
                    'argv': [python, '-u', str(ROOT/'evaluate.py'), str(output/'last.pt'),
                             '--split', 'validation', '--output', str(output/directory), '--lpips']+validation_options,
                    'logpath': str(output_dir/family/f'{scene}.{directory}.log'),
                    'output': str(output/directory), 'resultpath': str(output/directory/'metrics.json'),
                })
    environment = dict(config["launch_environment"])
    return {
        "name": config["name"], "version": 1, "method": "PORT-GS", "root": str(ROOT),
        "status_path": str(output_dir / "status.json"), "source_archive": str(source_archive),
        "source_revision": revision, "python": python, "data_root": config["data_root"],
        "worker_gpus": config["worker_gpus"], "launch_environment": environment,
        "protocol": {
            "scenes": len(jobs),
            "selection": {name: settings["scenes"] for name, settings in config["families"].items()},
            "representation": config["train"].get("representation", DEFAULT_METHOD),
            "train": config["train"],
            "family_overrides": {name: settings["train"] for name, settings in config["families"].items()},
            "evaluation": {"split": eval_split, "branch":eval_branch, "limit": eval_limit, "checkpoint": "last.pt", "lpips": True,
                           "resolution":config.get('eval_resolution'), "highlights":config.get('eval_highlights', False),
                           "additional_full_terminal_test": config.get('eval_test', False),
                           "additional_full_terminal_fit": config.get('eval_full_fit', False),
                           "additional_full_terminal_validation": config.get('eval_validation', False),
                           "save_all_pairs": config.get('eval_save_all', False),
                           "shift_align_radius": config.get('eval_shift_align', 0),
                           "test_time_calibration_steps": config.get('eval_calibrate_test', 0)},
            "loss_curve": "each scene writes history.jsonl and loss.png after training",
        },
        "jobs": jobs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output_dir = ROOT / "runs" / config["name"]
    manifest = build_manifest(config, output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "validation.json").write_text(json.dumps(config, indent=2) + "\n")
    with tarfile.open(output_dir / "source.tar", "w") as archive:
        for filename in subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT, text=True).splitlines():
            # Historical run archives already contain their own source snapshots.
            # Copying them into each new snapshot grows archives recursively.
            if filename.startswith('docs/experiments/retired/'):
                continue
            source = ROOT / filename
            if source.exists():  # Tracked files may have been deleted in the working tree.
                archive.add(source, arcname=filename)
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
