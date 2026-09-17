#!/usr/bin/env python3
"""Resolve the six-scene validation JSON into a fresh scheduler manifest."""

import argparse
import json
import subprocess
import tarfile
from pathlib import Path

from methods import DEFAULT_METHOD


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
        else:
            argv.extend([f"--{name}", str(value)])
    return argv


def build_manifest(config, output_dir):
    python = config["python"]
    source_archive = output_dir / "source.tar"
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    jobs = []
    for family, settings in config["families"].items():
        for scene in settings["scenes"]:
            output = output_dir / family / scene
            dataset = Path(config["data_root"]) / family / scene
            options = {**config["train"], **settings["train"],
                       "scene": str(dataset), "output": str(output)}
            jobs.append({
                "id": f"PORT-{family}-{scene}", "method": "PORT-GS", "family": family,
                "scene": scene, "dataset": str(dataset), "state": "pending", "mode": "fresh",
                "representation": options.get("representation", DEFAULT_METHOD),
                "source_archive": str(source_archive), "source_revision": revision,
                "output": str(output), "resultpath": str(output / "test/metrics.json"),
                "steps": [
                    {"name": "train", "state": "pending", "cwd": str(ROOT),
                     "argv": train_argv(python, options),
                     "logpath": str(output_dir / family / f"{scene}.train.log"),
                     "output": str(output), "resultpath": str(output / "last.pt"),
                     "progress_path": str(output / "history.jsonl")},
                    {"name": "eval", "state": "pending", "cwd": str(ROOT),
                     "argv": [python, "-u", str(ROOT / "evaluate.py"), str(output / "last.pt"),
                              "--split", "test", "--output", str(output / "test"), "--lpips"],
                     "logpath": str(output_dir / family / f"{scene}.test.log"),
                     "output": str(output / "test"), "resultpath": str(output / "test/metrics.json")},
                ],
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
            "test": "all official test frames, original calibration, fixed last.pt, full LPIPS",
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
            archive.add(ROOT / filename, arcname=filename)
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
