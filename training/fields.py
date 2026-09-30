"""Auxiliary continuous fields trained beside the Gaussians.

SurfaceFields holds the 2DGS-supervised SDF and its optional light-conditioned
volume radiance head. NormalFieldFit holds the neural-material shading-normal
residual. Loss terms are appended in the original order, so loss sums match.
"""

import math

import torch

from evaluate import observation_image


class SurfaceFields:
    def __init__(self, args, saved, gaussians):
        from sdf import SurfaceSDF

        self.args = args
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            self.sdf = SurfaceSDF(gaussians.center, gaussians.radius, detail=args.sdf_detail)
        self.sdf_steps = 0
        if saved is not None and saved["config"].get("sdf", False):
            if saved['config'].get('sdf_detail', False) and not args.sdf_detail:
                raise ValueError('Restoring this SDF requires --sdf-detail')
            state = saved['sdf']
            if args.sdf_detail and not saved['config'].get('sdf_detail', False):
                # Explicit expansion preserves old keys and initializes only
                # the new zero-output residual; normal reload remains strict.
                state = {**self.sdf.state_dict(), **state}
            self.sdf.load_state_dict(state)
            self.sdf_steps = saved["sdf_steps"]
        self.sdf_optimizer = None
        if args.freeze_sdf:
            self.sdf.requires_grad_(False)
        else:
            self.sdf_optimizer = torch.optim.Adam(self.sdf.parameters(), lr=args.sdf_lr)
        self.sdf_generator = torch.Generator(device="cuda").manual_seed(args.seed + 1)
        self.primitive_generator = torch.Generator(device="cuda").manual_seed(args.seed + 2)
        self.volume = None
        self.volume_steps = 0
        if args.sdf_volume_weight:
            self._build_volume(saved)
        self.sdf_active = self.volume_active = False
        self.primitive_candidates = 0

    def _build_volume(self, saved):
        from sdf_volume import SDFRadiance

        args = self.args
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            self.volume = SDFRadiance('cuda', detail=args.sdf_volume_detail, hint_encoding=args.sdf_volume_hint_encoding)
        if saved is not None and saved['config'].get('sdf_volume_weight', 0):
            if saved['config'].get('sdf_volume_detail', False) and not args.sdf_volume_detail:
                raise ValueError('Restoring this radiance head requires --sdf-volume-detail')
            if saved['config'].get('sdf_volume_hint_encoding', False) and not args.sdf_volume_hint_encoding:
                raise ValueError('Restoring this radiance head requires --sdf-volume-hint-encoding')
            state = saved['sdf_volume']
            if ((args.sdf_volume_detail and not saved['config'].get('sdf_volume_detail', False)) or
                (args.sdf_volume_hint_encoding and not saved['config'].get('sdf_volume_hint_encoding', False))):
                state = {**self.volume.state_dict(), **state}
            self.volume.load_state_dict(state)
            self.volume_steps = saved['sdf_volume_steps']
        if args.sdf_volume_fixed_sharpness is not None:
            with torch.no_grad():
                self.volume.log_sharpness.fill_(math.log(args.sdf_volume_fixed_sharpness))
            self.volume.log_sharpness.requires_grad_(False)
        self.volume_optimizer = torch.optim.Adam(self.volume.parameters(), lr=.001)
        self.volume_generator = torch.Generator(device='cuda').manual_seed(args.seed+3)
        self.peak_rays_total = self.context_rays_total = self.rays_total = 0

    @property
    def shading_field(self):
        return self.sdf if self.args.sdf_shading else None

    def loss_terms(self, step, terms, l1, network_lr, info, alpha, sample, sample_index, target,
                   gaussians, transport, peak_masks, peak_context_masks):
        """Append SDF point, primitive and volume terms; return (terms, l1)."""
        from sdf import sample_surface, visible_primitives, silhouette_rays
        from sdf_volume import select_rays, render_rays, ray_geometry_losses

        args, sdf = self.args, self.sdf
        self.sdf_active = args.sdf_shading and not args.freeze_sdf
        self.volume_active = self.volume is not None and step >= args.sdf_start
        volume_warmup = self.volume_steps < args.sdf_volume_warmup
        self.primitive_candidates = 0
        if self.sdf_optimizer is not None:
            self.sdf_optimizer.zero_grad(set_to_none=True)
        if step >= args.sdf_start and (self.volume is None or (volume_warmup and not args.sdf_volume_only)):
            surface_samples = sample_surface(info, alpha, sample, sdf, args.sdf_samples, self.sdf_generator)
            if surface_samples is not None:
                points, normals, toward_camera = surface_samples
                if not args.freeze_sdf:
                    terms.update(sdf.fit_losses(points, normals, toward_camera))
                ramp = min(1., max(0., (self.sdf_steps-args.sdf_warmup_steps) / max(1, args.sdf_warmup_steps)))
                if ramp > 0 and self.volume is None:
                    terms.update(sdf.geometry_losses(points, normals, ramp*args.sdf_weight,
                                                     ramp*args.sdf_normal_weight))
                    if args.sdf_primitive_weight:
                        selected = visible_primitives(gaussians.params['means'],
                            gaussians.params['opacities'].sigmoid(), info, alpha, sample, gaussians.radius)
                        self.primitive_candidates = len(selected)
                        if len(selected):
                            selected = selected[torch.randint(len(selected), (args.sdf_samples,),
                                device=selected.device, generator=self.primitive_generator)]
                            terms['sdf_primitive'] = ramp*args.sdf_primitive_weight*sdf.primitive_loss(
                                gaussians.params['means'][selected])
                self.sdf_active = not args.freeze_sdf
        if not self.volume_active:
            return terms, l1
        self.volume_optimizer.zero_grad(set_to_none=True)
        self.volume_optimizer.param_groups[0]['lr'] = network_lr
        rays = silhouette_rays(sample, sdf)
        ray_indices = select_rays(rays, args.sdf_volume_rays, self.volume_generator,
                                  peak_masks.get(sample_index), args.sdf_volume_peak_fraction,
                                  peak_context_masks.get(sample_index), args.sdf_volume_peak_context_fraction)
        self.peak_rays_total += int(peak_masks[sample_index].reshape(-1)[ray_indices].sum())
        if args.sdf_volume_peak_context_fraction:
            self.context_rays_total += int(peak_context_masks[sample_index].reshape(-1)[ray_indices].sum())
        self.rays_total += len(ray_indices)
        result = render_rays(sdf, self.volume, sample, rays, ray_indices,
            transport.light_scale.detach(), samples=args.sdf_volume_samples,
            background=args.background, detach_field=volume_warmup or args.freeze_sdf)
        predicted = observation_image(result['linear'], args.display_gamma,
            alpha=None if sample['is_hdr'] else result['alpha'], background=args.background)
        expected = target.reshape(-1, 3)[ray_indices]
        terms['sdf_volume_rgb'] = args.sdf_volume_weight*(predicted-expected).abs().mean()
        terms['sdf_volume_mask'] = args.sdf_volume_weight*.1*(
            result['alpha'][:, 0]-rays['target'][ray_indices]).abs().mean()
        if not volume_warmup:
            if not args.freeze_sdf:
                terms['sdf_volume_eikonal'] = .1*result['eikonal']
                self.sdf_active = True
            ramp = min(1., (self.volume_steps-args.sdf_volume_warmup+1)/max(1, args.sdf_volume_warmup))
            terms.update(ray_geometry_losses(result, ray_indices, info, alpha,
                sample, gaussians.radius, ramp*args.sdf_weight, ramp*args.sdf_normal_weight))
        if args.sdf_volume_only:
            l1 = (predicted-expected).abs().mean()
            terms = {key: value for key, value in terms.items()
                     if key.startswith('sdf_') and not key.endswith('_to_gs')}
        return terms, l1

    def step(self):
        if self.sdf_active:
            self.sdf_optimizer.step()
            self.sdf_steps += 1
        if self.volume_active:
            self.volume_optimizer.step()
            self.volume_steps += 1

    def log_fields(self):
        row = {"sdf_steps": self.sdf_steps}
        if self.volume is not None:
            row.update({'sdf_volume_steps': self.volume_steps,
                        'sdf_volume_sharpness': float(self.volume.log_sharpness.exp()),
                        'sdf_volume_peak_rays_total': self.peak_rays_total,
                        'sdf_volume_rays_total': self.rays_total,
                        'sdf_volume_context_rays_total': self.context_rays_total})
        if self.args.sdf_primitive_weight:
            row["sdf_primitive_candidates"] = self.primitive_candidates
        return row

    def checkpoint_fields(self):
        fields = {"sdf": self.sdf.state_dict(), "sdf_steps": self.sdf_steps}
        if self.volume is not None:
            fields.update(sdf_volume=self.volume.state_dict(), sdf_volume_steps=self.volume_steps)
        return fields


class NormalFieldFit:
    def __init__(self, saved, gaussians, trainable):
        from materials.normal_field import NormalResidualField

        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            self.field = NormalResidualField(gaussians.center, gaussians.radius)
        if saved is not None and saved['config'].get('normal_field', False):
            self.field.load_state_dict(saved['normal_field'])
        self.optimizer = torch.optim.Adam(self.field.parameters(), lr=.001)
        if not trainable:
            self.field.requires_grad_(False)

    def begin(self, network_lr):
        self.optimizer.zero_grad(set_to_none=True)
        self.optimizer.param_groups[0]['lr'] = network_lr

    @staticmethod
    def regularization(info):
        return 1e-4*info['normal_residual'].square().mean()

    def step(self):
        self.optimizer.step()
