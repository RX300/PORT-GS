#!/usr/bin/env python3
"""Create the fixed six-scene PORT-GS rank-512 validation manifest."""

from __future__ import annotations

import argparse
import copy
import json
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE_MANIFEST = ROOT / "runs/full_benchmark_20260913/manifest.json"
RUN_DIR = ROOT / "runs/rank512_validation_20260914"
SOURCE_FILES = (
    "train.py",
    "transport.py",
    "gaussians.py",
    "renderer.py",
    "refinement.py",
    "evaluate.py",
    "data.py",
    "cameras.py",
    "run_benchmark.py",
    "hourly_monitor.py",
    "make_validation_manifest.py",
    "launch_rank512_validation.sh",
    "README.md",
    "docs/experiments/rank512_validation_20260914.md",
)

# The selection is fixed before training so the subset cannot be chosen from
# outcomes. Each family contributes exactly two scenes with complete metadata
# and a corresponding prior PORT-GS reference where available.
SELECTED_SCENES = {
    "Real_NRHints": ("Cat", "Pixiu"),
    "Synthetic_GS3": ("AnisoMetal", "Translucent"),
    "Synthetic_SSS-GS": ("bunny_small", "dragon_small"),
}


def replace_option(argv: list[str], option: str, value: str) -> list[str]:
    positions = [index for index, item in enumerate(argv) if item == option]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise ValueError(f"expected one value-bearing {option} in {argv!r}")
    result = list(argv)
    result[positions[0] + 1] = value
    return result


def replace_checkpoint(argv: list[str], checkpoint: str) -> list[str]:
    positions = [index for index, item in enumerate(argv) if item.endswith("/last.pt")]
    if len(positions) != 1:
        raise ValueError(f"expected one checkpoint path in {argv!r}")
    result = list(argv)
    result[positions[0]] = checkpoint
    return result


def rewrite_job(source_job: dict, output_root: Path) -> dict:
    family = source_job["family"]
    scene = source_job["scene"]
    output = output_root / family / scene
    train_log = output_root / family / f"{scene}.train.log"
    eval_log = output_root / family / f"{scene}.test.log"
    rewritten = copy.deepcopy(source_job)
    rewritten.update(
        {
            "id": f"PORT-rank512-{family}-{scene}",
            "state": "pending",
            "mode": "fresh",
            "output": str(output),
            "resultpath": str(output / "test/metrics.json"),
            "protocol_source_archive": str(source_job.get("source_archive")),
            "source_provenance": (
                "PORT-GS current worktree after the default spatial-node update; "
                "feature_dim remains 32"
            ),
        }
    )
    rewritten.pop("reused_from", None)
    for step in rewritten["steps"]:
        step["state"] = "pending"
        for key in ("started_utc", "finished_utc", "pid", "last_pid", "error", "startup_confirmed"):
            step.pop(key, None)
        step["argv"] = list(step["argv"])
        if step["name"] == "train":
            step["argv"] = replace_option(step["argv"], "--output", str(output))
            step["argv"] = replace_option(step["argv"], "--rank", "512")
            step["output"] = str(output)
            step["resultpath"] = str(output / "last.pt")
            step["logpath"] = str(train_log)
            step["progress_path"] = str(output / "history.jsonl")
        elif step["name"] == "eval":
            step["argv"] = replace_checkpoint(step["argv"], str(output / "last.pt"))
            step["argv"] = replace_option(step["argv"], "--output", str(output / "test"))
            step["output"] = str(output / "test")
            step["resultpath"] = str(output / "test/metrics.json")
            step["logpath"] = str(eval_log)
        else:
            raise ValueError(f"unexpected PORT phase {step['name']!r}")
        step["cwd"] = str(ROOT)
    return rewritten


def build_manifest(source_path: Path, output_dir: Path) -> dict:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    wanted = {
        (family, scene)
        for family, scenes in SELECTED_SCENES.items()
        for scene in scenes
    }
    candidates = {
        (job["family"], job["scene"]): job
        for job in source["jobs"]
        if job.get("method") == "PORT-GS"
    }
    missing = sorted(wanted - candidates.keys())
    if missing:
        raise ValueError(f"selected scenes missing from source manifest: {missing}")

    jobs = [
        rewrite_job(candidates[(family, scene)], output_dir)
        for family, scenes in SELECTED_SCENES.items()
        for scene in scenes
    ]
    if len(jobs) != 6 or {
        (job["family"], job["scene"]) for job in jobs
    } != wanted:
        raise AssertionError("validation manifest must contain exactly six selected scenes")
    source_archive = output_dir / "port_gs_source_rank512.tar"

    return {
        "name": "rank512_validation_20260914",
        "version": 1,
        "method": "PORT-GS",
        "root": str(ROOT),
        "status_path": str(output_dir / "status.json"),
        "source_archive": str(source_archive),
        "protocol_source_archive": source.get("source_archive"),
        "source_provenance": (
            "Current PORT-GS worktree; transport spatial exchange rank changed "
            "from 32 to 512, while feature_dim remains 32"
        ),
        "python": source["python"],
        "data_root": source["data_root"],
        "worker_gpus": [0, 1],
        "queue": (
            "Fresh rank-512 PORT-GS train→test jobs; exactly two fixed scenes "
            "from each dataset family, with two serial GPU workers"
        ),
        "protocol": {
            "scenes": 6,
            "selection": {family: list(scenes) for family, scenes in SELECTED_SCENES.items()},
            "train": "all official train frames, 30000 steps, seed 0, fixed last.pt",
            "test": "all official test frames, original calibration, one evaluation, full LPIPS",
            "spatial_exchange_nodes": 512,
            "feature_dim": 32,
            "real": "512px, background 0, display gamma 2.2",
            "synthetic_gs3": "512px, background 1, display gamma 2.2",
            "synthetic_sss_gs": "256px, background 0, display gamma 2.2, unit light intensity 1",
        },
        "launch_environment": copy.deepcopy(source["launch_environment"]),
        "jobs": jobs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=RUN_DIR)
    args = parser.parse_args()
    source_path = args.source.resolve()
    output_dir = args.output_dir.resolve()
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to replace existing manifest: {manifest_path}")
    output_dir.mkdir(parents=True, exist_ok=False)
    source_archive = output_dir / "port_gs_source_rank512.tar"
    with tarfile.open(source_archive, "w") as archive:
        for relative in SOURCE_FILES:
            archive.add(ROOT / relative, arcname=relative)
    manifest = build_manifest(source_path, output_dir)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "manifest": str(manifest_path),
        "scenes": manifest["protocol"]["selection"],
        "rank": manifest["protocol"]["spatial_exchange_nodes"],
        "feature_dim": manifest["protocol"]["feature_dim"],
    }, indent=2))


if __name__ == "__main__":
    main()
