# Current system — 2026-09-12

2026-09-14 update: spatial partition generation now uses NVIDIA HashGrid and a
linear 32→512 projection, in place of learned centers/widths. `train.py` resolves
the encoding JSON into the checkpoint config; `evaluate.load_model` restores it
directly. The six-scene launcher reads tracked experiment JSON. Older checkpoint
architectures use their saved Git/source archive. The component boundaries below
remain unchanged.

| File | Responsibility |
| --- | --- |
| `data.py` | Metadata, decoding, camera/light conventions and explicit splits |
| `gaussians.py` | Gaussian state, GPU initialization and parameter optimization |
| `transport.py` | Complete Gaussian-source integral and continuous receiver material response |
| `renderer.py` | Attribute/expected-depth rasterization, pixel reconstruction and shadows |
| `refinement.py` | Opacity-reset scheduling, growth and pruning |
| `cameras.py` | Optional fit-frame camera corrections |
| `train.py` | Optimization, split ownership, checkpoint state and progress |
| `evaluate.py` | Model loading, shared observation transform, rendering and metrics |
| `diagnose_image_errors.py` | Region/detail errors, silhouettes and contribution-weighted support |

[Transport](modules/transport.md), [refinement](modules/refinement.md),
[shadows](modules/shadows.md), and [data/observation](modules/data_observation.md)
describe the component boundaries. The current renderer uses `RGB+ED` to blend
base/features/visibility and expected
camera-Z. It reconstructs covered-pixel receivers and evaluates transport there,
while the source integral always uses all Gaussians. Source-node-only
conservation is distinguished from arbitrary receiver queries. Second-round operator and receiver-rendering GPU audits passed; the
corresponding reports are `receiver_operator_audit.json` and
`receiver_rendering_audit.json` under `runs/research_20260912/`.
Historical global/local port modes, deferred queries, radiance moments and
camera-response experiments are preserved with their original source in
`runs/research_20260912/source_before.tar` and previous outputs.

`evaluate.load_model(path)` reconstructs the current Gaussian and transport
state for evaluation and checkpoint initialization. Checkpoints store config,
step, Gaussian/transport state, radius, fit/validation indices and optional
training-camera offsets. Initialization starts fresh optimizers. Historical
checkpoints are reproduced with their archived code.

Geometry refinement adds screen-radius split candidates above 3% of the image's
long edge, with splitting and duplication mutually exclusive. Screen-radius
pruning is explicitly disabled to avoid applying stale parent radii to children;
opacity/world-size pruning remains. This is geometry budget control, separate
from the exchange operator's research formulation. The old top-level scripts
removed from the production path are listed in
`runs/research_20260912/cleanup_files.json` and preserved in the source archive.

First-round weights and results belong to `runs/research_20260912/cat_r1_source.tar`.
The active second-round network adds 51 world-position channels and a pixel
receiver interface; its checkpoint architecture differs from the first round.

The second-round path is accepted and frozen in `cat_r2_source.tar` for final
full fits. Its validation tradeoffs are recorded in [results](../experiments/results.md).
