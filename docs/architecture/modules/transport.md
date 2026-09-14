# Continuous receiver transport implementation

Current spatial partitions use NVIDIA tiny-cuda-nn HashGrid, configured by
`configs/hashgrid.json`, followed by a linear projection and log-softmax over
512 exchange channels. Learned anchor centers/widths were replaced in place;
their source remains in Git commit `9e9596a`. Grid resolution (16–2048 across 16
levels) and exchange rank are independent. The normalized initialization cube
maps to [0,1]^3 without clamping. HashGrid table/interpolation/backward are native
NVIDIA operations; the quadrature, exchange equations and angular MLP below are
unchanged. Checkpoints embed the complete encoding config and seed.

The second-round `Transport.forward` accepts all Gaussian sources and a separate
receiver dictionary (`means`, `base`, `features`, `visibility`). The renderer
provides one receiver per covered pixel. Source irradiance, mass and source
exchange fractions use the complete Gaussian set. Receiver fractions and spatial
partitions query the same source integral at reconstructed world positions.

`quadrature_mass` remains the detached normalized opacity-weighted projected-area
surrogate. `partition(xyz)` accepts normalized coordinates for either source or
receiver. Factored source normalization uses contiguous `R x N` weights:

```text
w_ri = softmax_i(log(f_ir) + log(m_i))
u_rc = sum_i w_ri a_ic E_ic / sum_i w_ri a_ic
E'_qc = (1-a_qc) E_qc + a_qc sum_r f_qr u_rc.
```

For `Q` receivers and `N` sources, exchange costs `O(NR + QR)` and stores the
source/receiver partition matrices plus small RGB port matrices. Source-node
queries recover the conservative reversible discrete operator. General pixel
queries extend the source field continuously; conservation and reversibility
are claimed only for source-node queries.

The angular MLP uses four width-128 SiLU hidden layers by default. Its input
contains 32 material channels, 82 angular channels and 51 channels from an
eight-band encoding of normalized receiver world position. Its positive output
multiplies queried irradiance to produce foreground RGB. Second-round GPU audits and its fresh 30k run are complete. The accepted path is
frozen in `cat_r2_source.tar`; [results](../../experiments/results.md) records
its perceptual/detail gains and PSNR tradeoff.

## First-round operator performance and numerical evidence

The first-round implementation initially expanded `N x R x RGB` source weights and normalized
along dimension 0. At roughly 400k Gaussians, the main training log showed about
0.5 seconds per step. Isolated CUDA-event measurements on 167,692 real Cat points,
32 ports and an RTX 6000 Ada GPU 0 identified the exchange layout as a bottleneck.
Four warm iterations preceded twelve measured iterations, in float32 with TF32
enabled. Median isolated operator times were:

| Operation | Expanded layout | Factored contiguous layout |
| --- | ---: | ---: |
| Forward | 75.648 ms | 0.246 ms |
| Forward and backward | 126.353 ms | 0.982 ms |
| Forward/backward peak memory increment | 334.90 MiB | 111.21 MiB |

The 128.7x forward/backward ratio measures this operator alone. Shadows,
projection, rasterization, refinement and optimizer work are outside that timing;
end-to-end training throughput requires its own completed-run measurement.

Float64 comparison of both formulas gives maximum relative output difference
3.40e-15 and gradient differences at most 1.76e-13. Against the float64 reference,
maximum relative float32 output error decreases from 1.52e-4 in the archived
layout to 2.13e-6 in the factored layout; each checked gradient also has lower
error. These values describe numerical equivalence and implementation efficiency.
Independent identity and directional-gradient checks are recorded in
[operator audit](../../../runs/research_20260912/operator_audit.json), with complete
layout measurements in
[factorization audit](../../../runs/research_20260912/operator_factorization_audit.json).

See [mathematical principles](../../method/principles.md) for the operator
identities and the scope of physical claims. Validation of the implemented
operator and empirical gains must be recorded separately from those derivations.

The timing table above concerns the first-round source-node operator layout.
The second-round complete 30k run took 1,612.392 seconds, versus 1,470.625 seconds
for round one. These whole-run durations include different rendering paths and
are separate from the isolated factorization timing.
