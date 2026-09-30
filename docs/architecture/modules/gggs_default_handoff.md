# GGGS geometry import for the default/DNA relighting pipelines

`train.py --representation directional_port_v1 --init-geometry-format gggs
--init-geometry SOURCE` loads source GGGS state after registering its enum module.
`gggs_reconstruction.relighting_state` reuses the validated filtered geometry
conversion, changes normalized geometry units to world units, and converts
physical scales/opacity to the default log/logit parameterization. Materials are
fresh. Source metadata must match scene/resolution, and source fit images must
be contained in target material fit. Full official-train fitting is explicit.

The four geometric parameter arrays are copied exactly before training. The
source Mip filter is baked, not reapplied. Default3DGS rendering and the unchanged
DirectionalTransport module own all subsequent shading/training/evaluation.
With trainable geometry, default projection, optimizer and density strategy run;
with `--freeze-geometry`, these updates and per-step parameter clamps are disabled.
No new method registration or new network is introduced.

Source checkpoints are immutable. New checkpoints use the normal PORT schema
and existing save/load/evaluate paths, with geometry provenance in saved config.
See [experiment protocol](../../experiments/gggs_default_joint.md).

The same explicit import now supports `distribution_material`: resolve/save/load
select3DGS only for the GGGS format. DNA gets interpolated face-forward shortest
covariance-axis normal proxies; its existing GGX/mixture networks remain unchanged.
Old2DGS-specific regularizers must be explicitly set tozero in this3D mode.
[DNA protocol and limitations](../../experiments/gggs_dna_joint.md).

Neural-material integration also uses this same3D handoff, with `requires_normals=True` for the covariance-axis proxy. Frozen neural BRDF weights are shared between scenes and embedded in final checkpoints; latent codes, shading-normal residuals and geometry continue to learn. No change to the default or DNA shader path.
