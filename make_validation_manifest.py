#!/usr/bin/env python3
"""Resolve the canonical experiment JSON into a fresh scheduler manifest."""

import argparse
import hashlib
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


def source_provenance():
    """Relate the archived source to git: HEAD, working-tree state and third_party checkouts.

    source.tar holds the exact project files. HEAD alone is not a revision of
    that code when the tree is dirty, so the status and a diff hash are kept.
    third_party is not archived; its checkouts are identified by revision and
    local-diff hash.
    """
    def git(*argv, cwd=ROOT):
        return subprocess.check_output(["git", *argv], cwd=cwd)

    status = git("status", "--porcelain").decode().splitlines()
    third_party = {}
    for marker in sorted((ROOT / "third_party").glob("*/.git")):
        repo = marker.parent
        third_party[repo.name] = {
            "revision": git("rev-parse", "HEAD", cwd=repo).decode().strip(),
            "dirty_files": len(git("status", "--porcelain", cwd=repo).decode().splitlines()),
            "diff_sha256": hashlib.sha256(git("diff", "HEAD", "--binary", cwd=repo)).hexdigest(),
        }
    return {
        "revision": git("rev-parse", "HEAD").decode().strip(),
        "dirty": bool(status),
        "status": status,
        "diff_sha256": hashlib.sha256(git("diff", "HEAD", "--binary")).hexdigest(),
        "third_party": third_party,
    }


def build_manifest(config, output_dir):
    """Optional named ablations share the same queue and one source snapshot."""
    provenance = source_provenance()
    if "variants" not in config:
        return _build_single_manifest(config, output_dir, provenance)
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
        manifest = _build_single_manifest(child, output_dir / name, provenance)
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


def _evaluation_options(config, branch, limit=0):
    """Shared evaluate.py options; only the primary phase carries a frame limit."""
    options = ['--sdf-volume'] if branch == 'sdf_volume' else []
    if limit:
        options += ['--limit', str(limit)]
    if config.get('eval_resolution'):
        options += ['--resolution', str(config['eval_resolution'])]
    if config.get('eval_highlights', False):
        options += ['--highlights']
    if config.get('eval_save_all', False):
        options += ['--save-all']
    if config.get('eval_shift_align', 0):
        options += ['--shift-align', str(config['eval_shift_align'])]
    return options


def _evaluation_step(python, name, checkpoint, split, output, directory, log_dir, scene, options):
    return {
        'name': name, 'state': 'pending', 'cwd': str(ROOT),
        'argv': [python, '-u', str(ROOT / 'evaluate.py'), str(checkpoint), '--split', split,
                 '--output', str(output / directory), '--lpips'] + options,
        'logpath': str(log_dir / f'{scene}.{directory}.log'),
        'output': str(output / directory), 'resultpath': str(output / directory / 'metrics.json'),
    }


def _training_step(python, name, options, log_dir, scene):
    output = Path(options['output'])
    return {
        'name': name, 'state': 'pending', 'cwd': str(ROOT),
        'argv': train_argv(python, options),
        'logpath': str(log_dir / f'{scene}.{name}.log'),
        'output': str(output), 'resultpath': str(output / 'last.pt'),
        'progress_path': str(output / 'history.jsonl'),
    }


def _surface_preparation(config, options, dataset, output_dir, family, scene):
    settings_prior = config["surface_preprocessing"]
    prior_dir = Path(options['surface-priors'])
    steps = []
    for kind in settings_prior.get('kinds', ("normal", "depth")):
        argv = [settings_prior[f"{kind}_python"], "-u", str(ROOT / "prepare_surface_priors.py"),
                "--kind", kind, "--scene", str(dataset), "--output", str(prior_dir),
                "--processing-resolution", str(settings_prior["processing_resolution"])]
        for key in ("resolution", "background", "display-gamma", "unit-light-intensity", "seed"):
            if key in options:
                argv.extend([f"--{key}", str(options[key])])
        if options.get("fit-all", False):
            argv.append("--fit-all")
        steps.append({
            "name": f"prepare_{kind}", "state": "pending", "cwd": str(ROOT), "argv": argv,
            "logpath": str(output_dir / family / f"{scene}.{kind}.log"),
            "output": str(prior_dir), "resultpath": str(prior_dir / f"{kind}.pt"),
        })
    return steps


def _build_single_manifest(config, output_dir, provenance):
    python = config["python"]
    source_archive = output_dir / "source.tar"
    revision = provenance["revision"]
    jobs = []
    eval_split = config.get("eval_split", "test")
    eval_branch = config.get('eval_branch', 'gaussian')
    eval_limit = config.get('eval_limit', 0)
    if eval_branch not in ['gaussian', 'sdf_volume'] or eval_limit < 0:
        raise ValueError('Evaluation branch must be gaussian/sdf_volume and limit nonnegative')
    suffix = '_sdf' if eval_branch == 'sdf_volume' else ''
    eval_directory = eval_split + suffix
    eval_options = _evaluation_options(config, eval_branch, eval_limit)
    extra_options = _evaluation_options(config, eval_branch)
    if config.get('eval_test', False) and eval_split == 'test':
        raise ValueError('eval_test is an additional full terminal test after a non-test evaluation')
    if config.get('eval_full_fit', False) and eval_split == 'fit' and not eval_limit:
        raise ValueError('eval_full_fit would repeat the existing complete fit evaluation')
    if config.get('eval_validation', False) and eval_split == 'validation' and not eval_limit:
        raise ValueError('eval_validation would repeat the existing complete validation evaluation')
    for family, settings in config["families"].items():
        log_dir = output_dir / family
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
            needs_surface_priors = (
                METHODS[representation].geometry == '2dgs'
                and options.get('init-geometry-format', 'port') != 'gggs'
            )
            preparation = (_surface_preparation(config, options, dataset, output_dir, family, scene)
                           if "surface_preprocessing" in config and needs_surface_priors else [])
            appearance = config.get('appearance_train')
            if appearance:
                geometry_output = output / 'geometry'
                training = [_training_step(python, 'train', dict(options, output=str(geometry_output)),
                                           log_dir, scene)]
                output = output / 'appearance'
                options = {**options, **appearance, 'output': str(output),
                           'init-checkpoint': str(geometry_output / 'last.pt')}
                training.append(_training_step(python, 'appearance', options, log_dir, scene))
            else:
                training = [_training_step(python, 'train', options, log_dir, scene)]
            checkpoints = [(f"step_{step:06d}.pt", f"{eval_directory}_{step:06d}", f"eval_{step:06d}")
                           for step in options.get("save-steps", [])]
            checkpoints.append(("last.pt", eval_directory, "eval"))
            last = output / 'last.pt'
            steps = preparation + training + [
                _evaluation_step(python, phase, output / snapshot, eval_split, output, directory,
                                 log_dir, scene, eval_options)
                for snapshot, directory, phase in checkpoints
            ]
            if config.get('eval_test', False):
                steps.append(_evaluation_step(python, 'eval_test', last, 'test', output, 'test' + suffix,
                                              log_dir, scene, extra_options))
            if config.get('eval_full_fit', False):
                steps.append(_evaluation_step(python, 'eval_full_fit', last, 'fit', output, 'fit_full' + suffix,
                                              log_dir, scene, extra_options))
            if config.get('eval_calibrate_test', 0):
                # Secondary literature-style protocol: per-view camera/light fit on test GT, scene frozen.
                steps.append(_evaluation_step(
                    python, 'eval_test_calibrated', last, 'test', output, 'test_calibrated', log_dir, scene,
                    ['--resolution', str(config.get('eval_resolution') or options.get('resolution', 512)),
                     '--calibrate-test-views', str(config['eval_calibrate_test'])]))
            if config.get('eval_validation', False):
                steps.append(_evaluation_step(python, 'eval_validation', last, 'validation', output,
                                              'validation' + suffix, log_dir, scene, extra_options))
            jobs.append({
                "id": f"PORT-{family}-{scene}", "method": "PORT-GS", "family": family,
                "scene": scene, "dataset": str(dataset), "state": "pending", "mode": "fresh",
                "representation": representation,
                "source_archive": str(source_archive), "source_revision": revision,
                "source_dirty": provenance["dirty"],
                "output": str(output), "resultpath": str(output / eval_directory / "metrics.json"),
                "steps": steps,
            })
    environment = dict(config["launch_environment"])
    return {
        "name": config["name"], "version": 1, "method": "PORT-GS", "root": str(ROOT),
        "status_path": str(output_dir / "status.json"), "source_archive": str(source_archive),
        "source_revision": revision, "source_provenance": provenance,
        "python": python, "data_root": config["data_root"],
        "worker_gpus": config["worker_gpus"], "launch_environment": environment,
        "protocol": {
            "scenes": len(jobs),
            "selection": {name: settings["scenes"] for name, settings in config["families"].items()},
            "representation": config["train"].get("representation", DEFAULT_METHOD),
            "train": config["train"],
            "appearance_train": config.get("appearance_train"),
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
