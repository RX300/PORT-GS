## Current rank update — 2026-09-15

The user increased spatial ports from64 to512 for the follow-up experiment.
The active default is R=512, B=4; matrices have shape[512,3,4,4]. Other equations
and parameterizations below are unchanged. The initial64-port implementation
and its completed results remain documented as history.

# Directional spatial ports — 2026-09-15

Implemented from the user-supplied `PORT_GS_Architecture_Change.md` (archived
in the parent architecture directory). Representation is `directional_port_v1`, R=64, B=4.
`transport.py` retains the legacy model and checkpoint path. The pixel renderer,
shadows, geometry/refinement, data and image loss are unchanged.

The shared direction MLP is 32→32→32→12 with SiLU. Each group of four outputs
contains a normalized 3D axis and a softplus concentration. Initial axis biases
are +X/+Y/+Z, concentration 1, final weights Normal(0,.001). The basis is a
constant plus three spherical Gaussians. Source directions point toward the
light; receiver directions point toward the camera.

Source pooling retains stable log-space source weights and a separate RGB
exchange denominator protected at the dtype's smallest positive normal value.
The denominator excludes illumination. Port matrices have shape [64,3,4,4]
and softplus parameterization; initial diagonal .25 and off-diagonal .001 give
nonzero gradients with a modest initial exchange amplitude. This initialization
is an implementation choice; the supplied document did not specify its scale.
All basis/initialization settings and dimensions are saved in the checkpoint.

The nonlocal result is receiver exchange × outgoing basis readout. It is added
to (1-exchange) × direct irradiance × direct response. It is not multiplied by
receiver visibility, incident light, attenuation or local response. When ports
are disabled, the original direct result has no exchange attenuation.
Pooling and reading use matrix multiplications with flattened [B,RGB] channels;
no source/receiver × port × basis × RGB tensor is formed. This directional
operator does not inherit the legacy operator's conservation/reversibility claim.

Validation: `test_directional_transport.py` checks a separately looped float64
formula, disabled-port agreement with legacy Transport, illumination linearity,
nonlocal light at shadowed receivers, every parameter's nonzero finite gradient,
and a real Cat CUDA deep-shadow pixel renderer forward/backward. All passed.
