"""Radiance-residual-only stage on a frozen neural-material Gaussian teacher."""

import json

import torch


def residual_frame_pool(fit_indices, residual_frames, enabled):
    """Select stage sampling frames without changing source fit membership."""
    if residual_frames is None:
        return list(fit_indices)
    if not enabled:
        raise ValueError('Residual frame selection requires --radiance-residual')
    if len(residual_frames) != len(set(residual_frames)):
        raise ValueError('--residual-frames must contain unique indices')
    if not residual_frames or not set(residual_frames).issubset(fit_indices):
        raise ValueError('--residual-frames must contain saved fit frame indices')
    return list(residual_frames)


def residual_optimization_stats(residual):
    """Post-update weight/pre-update gradient RMS; None for an unqueried group."""
    groups = {
        'interaction_spatial': residual.interaction_spatial.parameters(),
        'interaction_angular': residual.interaction_angular.parameters(),
        'interaction_projection': residual.interaction_projection.parameters(),
        'grid_table': [residual.detail.grid.table],
        'detail_projection': residual.detail.projection.parameters(),
        'network_first': residual.network[0].parameters(),
        'network_last': residual.network[-1].parameters(),
    }
    if residual.angular_bank != 'none':
        groups.update({
            'angular_center_first': residual.center_network[0].parameters(),
            'angular_center_last': residual.center_network[2].parameters(),
            'angular_projection': residual.angular_projection.parameters(),
        })
    stats = {}
    for name, parameters in groups.items():
        parameters = list(parameters)
        count = sum(parameter.numel() for parameter in parameters)
        stats[name+'_parameter_rms'] = (
            sum(parameter.detach().square().sum() for parameter in parameters)/count).sqrt().item()
        stats[name+'_gradient_rms'] = (None if any(parameter.grad is None for parameter in parameters) else
            (sum(parameter.grad.detach().square().sum() for parameter in parameters)/count).sqrt().item())
    if residual.angular_bank != 'none':
        gradient = residual.angular_projection.weight.grad
        for index in range(residual.angular_projection.in_features):
            stats[f'angular_kernel_{index}_projection_gradient_rms'] = (
                None if gradient is None else gradient[:, index].detach().square().mean().sqrt().item())
    return stats


_PAIR_TOTAL_KEYS = ('nominal_pairs', 'sampled_pairs', 'supported_pairs', 'excluded_pairs', 'fallback_steps',
                    'draws', 'pair_peak_draws', 'pair_ring_draws', 'foreground_draws', 'valid_draws')
_PAIR_BUCKETS = ('1_to_4', '5_to_16', 'gt16')


class ResidualStage:
    """Owns the residual head, its ray sampling, optimizer, counters and audit records.

    Gaussians, transport and saved camera corrections stay fixed; only the
    residual optimizer steps.
    """

    def __init__(self, args, config, saved, gaussians, output, sample_pool, peak_masks, peak_context_masks):
        from materials.radiance_residual import RadianceResidual

        self.args = args
        self.peak_masks, self.peak_context_masks = peak_masks, peak_context_masks
        self.paired = args.residual_paired_context
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            self.module = RadianceResidual(gaussians.center, gaussians.radius,
                                           normal_source=args.residual_normal,
                                           interaction=args.residual_interaction,
                                           angular_bank=args.residual_angular_bank)
        self.steps = 0
        if saved['config'].get('radiance_residual', False):
            expected_scales = (self.module.angular_scales.clone()
                               if args.residual_angular_bank != 'none' else None)
            self.module.load_state_dict(saved['radiance_residual'], strict=True)
            if expected_scales is not None and not torch.equal(self.module.angular_scales, expected_scales):
                raise ValueError('Radiance residual angular scales disagree with the saved angular-bank configuration')
            self.steps = saved['residual_steps']
        else:
            torch.save({'config': config, 'radiance_residual': self.module.state_dict(),
                        'residual_steps': 0}, output / 'residual_initial.pt')
        self.optimizer = torch.optim.Adam(self.module.parameters(), lr=.001, eps=1e-15)
        self.generator = torch.Generator(device='cuda').manual_seed(args.seed+3)
        self.sample_pool = sample_pool
        self.frame_counts = {index: 0 for index in sample_pool}
        self.peak_rays_total = self.context_rays_total = self.rays_total = 0
        if self.paired:
            self.pair_pools = {}
            self.pair_eligibility = {}
            self.pair_coverage = {name: {} for name in _PAIR_BUCKETS}
            self.pair_totals = {key: 0 for key in _PAIR_TOTAL_KEYS}
            self.pair_totals['component_buckets'] = {
                name: {'anchor_draws': 0, 'sampled_components': 0} for name in _PAIR_BUCKETS}
            self.pair_stats = None

    def begin(self, network_lr, sample_index, sample, history):
        """Reset the optimizer and draw this step's residual pixel indices."""
        from sdf import silhouette_rays
        from sdf_volume import select_rays

        self.optimizer.zero_grad(set_to_none=True)
        self.optimizer.param_groups[0]['lr'] = network_lr
        rays = silhouette_rays(sample, self.module)
        if self.paired:
            from residual_sampling import build_residual_pair_pool, select_residual_pairs
            if sample_index not in self.pair_pools:
                # Corrected cameras and source geometry stay fixed in this stage;
                # cache GT candidates only, not receivers or render outputs.
                pool = build_residual_pair_pool(rays, self.peak_masks[sample_index])
                self.pair_pools[sample_index] = pool
                self.pair_eligibility[sample_index] = {
                    **pool['stats'], 'valid_pixels': len(rays['valid']),
                    'foreground_pixels': int((rays['target'][rays['valid']] > .9).sum())}
                history.write(json.dumps({'event': 'residual_pair_eligibility',
                    'frame_index': sample_index, **self.pair_eligibility[sample_index]}) + '\n')
            indices, self.pairs = select_residual_pairs(
                rays, self.args.residual_rays, self.generator, self.pair_pools[sample_index])
        else:
            indices = select_rays(rays, self.args.residual_rays, self.generator,
                self.peak_masks.get(sample_index), self.args.residual_peak_fraction,
                self.peak_context_masks.get(sample_index), self.args.residual_context_fraction)
        self.peak_rays_total += int(self.peak_masks[sample_index].reshape(-1)[indices].sum())
        if self.args.residual_context_fraction:
            self.context_rays_total += int(self.peak_context_masks[sample_index].reshape(-1)[indices].sum())
        self.rays_total += len(indices)
        return indices

    def loss_terms(self, predicted, target, alpha, indices, sample_index):
        """Return (terms, l1) on the drawn pixels; paired context adds the pair-difference term."""
        l1 = (predicted.reshape(-1, 3)[indices] - target.reshape(-1, 3)[indices]).abs().mean()
        terms = {'residual_rgb': l1}
        if not self.paired:
            return terms, l1
        from residual_sampling import residual_pair_loss, residual_pair_sample_stats
        rays = self.args.residual_rays
        pairs = self.pairs
        pair_loss, pair_support = residual_pair_loss(predicted, target, alpha, pairs, rays // 4)
        terms['residual_pair'] = self.args.residual_pair_weight * pair_loss
        sampled_pairs = len(pairs)
        supported_pairs = int(pair_support.sum())
        remaining_draws = len(indices) - 2 * sampled_pairs
        foreground_draws = remaining_draws // 2 if self.pair_eligibility[sample_index]['foreground_pixels'] else 0
        pair_counts = {'nominal_pairs': rays // 4, 'sampled_pairs': sampled_pairs,
                       'supported_pairs': supported_pairs, 'excluded_pairs': sampled_pairs - supported_pairs,
                       'fallback_steps': int(sampled_pairs == 0), 'draws': len(indices),
                       'pair_peak_draws': sampled_pairs, 'pair_ring_draws': sampled_pairs,
                       'foreground_draws': foreground_draws, 'valid_draws': remaining_draws - foreground_draws}
        for key, value in pair_counts.items():
            self.pair_totals[key] += value
        sample_stats = residual_pair_sample_stats(self.pair_pools[sample_index], pairs)
        for name, bucket in sample_stats['component_buckets'].items():
            covered = self.pair_coverage[name].setdefault(sample_index, [])
            previous = len(covered)
            covered[:] = sorted(set(covered).union(bucket['component_ids']))
            totals = self.pair_totals['component_buckets'][name]
            totals['anchor_draws'] += bucket['anchor_draws']
            totals['sampled_components'] += len(covered) - previous
        self.pair_stats = {**pair_counts, **sample_stats,
                           'excluded_fraction': (sampled_pairs - supported_pairs) / sampled_pairs if sampled_pairs else 0.0,
                           'pair_loss': pair_loss.item(), 'weighted_pair_loss': terms['residual_pair'].item()}
        return terms, l1

    def step(self, sample_index):
        self.optimizer.step()
        self.steps += 1
        self.frame_counts[sample_index] += 1

    def log_fields(self, info):
        row = {'residual_steps': self.steps, 'residual_rays_total': self.rays_total,
               'residual_frame_counts': self.frame_counts,
               'residual_peak_rays_total': self.peak_rays_total,
               'residual_context_rays_total': self.context_rays_total,
               'residual_lr': self.optimizer.param_groups[0]['lr'],
               'residual_stats': {key: value.item() if isinstance(value, torch.Tensor) else value
                                  for key, value in info['residual_stats'].items()}}
        if self.args.residual_interaction != 'none':
            row['residual_optimization_stats'] = residual_optimization_stats(self.module)
        if self.paired:
            row.update(residual_pair_stats=self.pair_stats, residual_pair_totals=self.pair_totals)
        return row

    def checkpoint_fields(self):
        fields = {'radiance_residual': self.module.state_dict(), 'residual_steps': self.steps,
                  'residual_sample_indices': self.sample_pool, 'residual_frame_counts': self.frame_counts}
        if self.paired:
            fields.update(residual_pair_totals=self.pair_totals, residual_pair_eligibility=self.pair_eligibility,
                          residual_pair_coverage=self.pair_coverage)
        return fields

    def write_audit(self, output, step):
        if self.paired:
            (output / 'residual_pair_audit.json').write_text(json.dumps({
                'stage_steps': step, 'residual_pair_totals': self.pair_totals,
                'residual_pair_eligibility': self.pair_eligibility,
                'residual_pair_coverage': self.pair_coverage}, indent=2) + '\n')
