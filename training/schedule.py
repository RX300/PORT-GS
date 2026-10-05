"""Stage switches and learning-rate schedules shared by the training loop."""

import torch


def radiometric_target(sample, background, display_gamma, step, warmup_steps):
    """Keep the renderer linear; anneal target encoding during geometry fitting.

    The first two thirds fit the ordinary observation. The final third changes
    the target exponent continuously to one. Subsequent joint training stays
    in linear radiance; inference retains the original observation encoding.
    """
    from evaluate import observation_image
    progress = min(1., max(0., 3.*step/warmup_steps-2.))
    exponent = (1-progress)/display_gamma + progress
    foreground = sample['image'] if sample['is_hdr'] else sample['image'].clamp_min(0).pow(display_gamma)
    alpha = sample['alpha']
    linear = foreground if alpha is None else foreground*alpha+background*(1-alpha)
    return observation_image(linear, 1/exponent,
                             alpha=None if sample['is_hdr'] else alpha, background=background)


def set_training_stage(gaussians, transport, optimizers, step, warmup_steps):
    geometry_only = step <= warmup_steps
    transport.requires_grad_(not geometry_only)
    gaussians.params["features"].requires_grad_(
        not geometry_only or getattr(transport, 'normal_model', None) == 'learned')
    if warmup_steps and step == warmup_steps + 1:
        # LiSA includes point-light attenuation in the bootstrap, so its base
        # already represents reflectance. The older 2DGS RGB stage fits observed
        # color and retains its reset contract.
        if not (gaussians.geometry == '3dgs' and getattr(transport,'light_space',False)):
            with torch.no_grad():
                gaussians.params["base"].fill_(-1.5)
            optimizers["base"].state.clear()
    return geometry_only


def learning_rates(args, step, decay_steps, position_scale):
    """Return (means, network) rates; decay holds after decay_steps, relighting decay after warmup."""
    scheduled_step = min(step, decay_steps)
    relight_step = max(0, scheduled_step - args.geometry_warmup_steps)
    decay = 0.1 ** (relight_step / max(1, decay_steps - args.geometry_warmup_steps))
    means = 1.6e-4 * position_scale * args.position_decay ** (scheduled_step / decay_steps)
    return means, 0.001 * (0.2 + 0.8 * decay)


def decayed_rate(initial, final, step, start, steps):
    """Exponential decay from ``initial`` at ``start`` to ``final`` at ``steps``."""
    progress = (step - start) / max(1, steps - start)
    return initial * (final / initial) ** progress
