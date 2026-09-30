# Refinement

`Refinement` derives from the installed gsplat strategy to reuse projected-gradient
statistics and established geometry operations. It directly implements the
post-backward schedule used by PORT-GS.

The installed dependency's condition `step % self.reset_every == 0 & step > 0`
is parsed as a chained comparison ending in `(0 & step) > 0`; it is always false.
PORT uses `step > 0 and step % self.reset_every == 0`. At 3000, 6000, 9000 and
12000 in a 15k refinement phase, `reset_opa` caps opacity at 0.01 and clears its
Adam moments. Explicit `opacity_reset` events record execution.

Growth ranks eligible Gaussians by accumulated image gradient and allocates only
available points. A duplicate adds one Gaussian; a two-child split adds one net
Gaussian. At capacity, pruning and scheduled resets continue. Loading geometry
above the requested budget applies the established low-opacity trimming rule
once before training. The configured refinement end remains fixed.

The September 12 candidate also admits splats with screen radii above 3% of the
image's longest side for splitting throughout refinement. At 512px this is
15.36px. This creates split candidates; point budgets and the refinement end
still determine which candidates split. A split candidate is excluded from
duplication so each selected operation adds exactly one net point.

Pruning uses opacity and world-space scale. The inherited screen-pruning
threshold is explicitly infinite: gsplat copies a parent's radius statistic to
its split children before pruning, so that statistic describes the parent,
rather than the newly reduced child support. This keeps screen splitting and
the pruning decision consistent.

The shared Conda environment is preserved. The repair replaces the erroneous
callback through normal strategy dispatch; it adds no monkey patches or retries.

## 2026-09-30: separate opacity-reset interval

`--opacity-reset-every` sets `Refinement.opacity_reset_every` (default 3000, the
previous behavior; 0 disables resets). `reset_every` stays 3000 because gsplat also
uses it to pause refinement and to start large-Gaussian pruning, so disabling resets
does not move the pruning schedule. With GGGS-imported geometry the default resets
cap every opacity at 0.01 eight times before step 25000 (R2b history); the option
exists to test keeping the imported opacities.
