# NVIDIA HashGrid validation — 2026-09-14

## Saved baseline

Git commit `9e9596a` preserves the learned-anchor implementation, scripts, and
research documentation before this replacement. `.gitignore` excludes runs,
logs, datasets, checkpoints, build artifacts, and local third-party dependencies.
Existing outputs remain on disk. Historical checkpoints require their saved
source revision; the current model loads HashGrid checkpoints only.

The prior 32 → 512 anchor experiment completed six scenes at 30k, seed 0. Its
scene-mean PSNR changed 28.381270 → 28.424444 (+0.043174 dB). The gains were not
uniform: dragon_small +0.383341 dB, Cat +0.074706 dB, AnisoMetal +0.020403 dB;
Pixiu −0.057180 dB, Translucent −0.036834 dB, bunny_small −0.125395 dB.
This is a single-seed observation, not statistical evidence of improvement.

## Representation and controlled comparison

Replace learned centers/widths with NVIDIA `tinycudann.Encoding(HashGrid)` and a
linear 32 → 512 projection followed by log-softmax. The 512 outputs are exchange
channels, not a spatial grid. Source pooling, material exchange fractions, local
angular/Fourier response, geometry, shadows, and the observation transform retain
their previous definitions. This comparison retains `rank=512` to isolate the
spatial partition change from the preceding run.

`configs/hashgrid.json` is passed directly to NVIDIA's encoding. It specifies 16
levels, 2 features per level, base resolution 16, per-level scale
1.381912879967776, final nominal resolution 2048, and at most 2^19 entries per
level. Linear interpolation, HashGrid parameters, and their gradients are provided
by NVIDIA; no custom hashing or interpolation implementation is used. The
native encoding runs in FP32, with the existing Adam rate/schedule.

World positions normalized by the camera-derived center/radius map as
`u = (xyz + 1) / 2`. Thus the initialization cube maps to [0,1]^3. Positions are
not clamped; out-of-cube queries follow the native grid indexing behavior and
retain coordinate gradients. This finite hashed representation does not provide
an independent parameter for every cell at resolution 2048. Memory is bounded
by the table cap and still includes dense N×512 and Q×512 exchange matrices.

## Dependency setup

The existing `ssd-gs` PyTorch 2.4.1/CUDA 12.1 stack is reused. Its installed
tinycudann 1.7 binary could not import because it required GLIBC_2.33. The
matching NVIDIA source revision was compiled locally and installed only under
`PORT-GS/third_party/python`, leaving the shared environment intact:

```bash
git clone https://github.com/NVlabs/tiny-cuda-nn.git third_party/tiny-cuda-nn
git -C third_party/tiny-cuda-nn checkout 48d6989c95def307a40baf176b2d6015dada19f9
git -C third_party/tiny-cuda-nn submodule update --init --recursive
cd third_party/tiny-cuda-nn/bindings/torch
CUDA_HOME=/usr/local/cuda-12.1 TCNN_CUDA_ARCHITECTURES=89 MAX_JOBS=4 \
  /workspace/ubuntu2004_cuda12_1/utils/conda-envs/ssd-gs/bin/python setup.py --no-networks bdist_wheel
```

Then, from PORT-GS:

```bash
/workspace/ubuntu2004_cuda12_1/utils/conda-envs/ssd-gs/bin/python -m pip install \
  --no-deps --no-compile --target third_party/python \
  third_party/tiny-cuda-nn/bindings/torch/dist/tinycudann-1.7-cp310-cp310-linux_x86_64.whl
export PYTHONPATH="$PWD/third_party/python"
```

The official encoding-only build is sufficient; the small projection head is
an ordinary PyTorch linear layer. Source and wheel are local ignored dependencies.
Upstream: https://github.com/NVlabs/tiny-cuda-nn/tree/48d6989c95def307a40baf176b2d6015dada19f9

## Six-scene protocol and execution

All user-editable experiment settings are in `configs/validation.json`; grid
settings are in `configs/hashgrid.json`. The generator freezes both JSON files
and complete argv in each new run. Change `name` for a new experiment; never
reuse an existing result directory for a fresh run.

| Family | Scenes | Resolution/background |
| --- | --- | --- |
| Real_NRHints | Cat, Pixiu | 512 / black |
| Synthetic_GS3 | AnisoMetal, Translucent | 512 / white |
| Synthetic_SSS-GS | bunny_small, dragon_small | 256 / black, unit light 1 |

Each scene trains from scratch for 30,000 steps with seed 0, all original train
frames, no validation-based selection, fixed `last.pt`, then full original test
evaluation with standard LPIPS. GPU workers 0/1 run two jobs concurrently. The
training budget, scene selection, and frame protocol match the rank512 run.

```bash
bash launch_validation.sh
# Explicit config and resume only after checking the prior scheduler/children:
bash launch_validation.sh configs/validation.json --resume
```

Output: `runs/hashgrid_validation_20260914/<family>/<scene>/`.
Queue: tmux socket `port-validation-hashgrid`, session `validation`, window `queue`.
The manifest records the Git revision, source tar, library revision, GPU
assignments, runtime environment, and exact train/test arguments.

Monitoring uses the existing Luna Max cadence: host checks every 60 seconds,
Luna every 1800 seconds and upon meaningful anomalies/completion. The
`port_validation` profile reads the manifest's HashGrid protocol.

## Validation and results

Prelaunch checks use actual saved Cat geometry and real training images.
`runs/hashgrid_preflight/operator_audit.json` checks conservation, detailed
balance, constant preservation, and finite nonzero gradients/Adam updates through
the NVIDIA table, projection head, and coordinate input. The full receiver audit
is `runs/hashgrid_preflight/receiver_rendering_audit.json`. Both passed.
Two-step training/checkpoint reload is a smoke check, not a scored experiment.
Final six-scene results are pending the new queue.
