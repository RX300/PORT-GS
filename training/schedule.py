"""Stage switches and learning-rate schedules shared by the training loop."""

import torch


def set_training_stage(gaussians, transport, optimizers, step, warmup_steps):
    geometry_only = step <= warmup_steps
    transport.requires_grad_(not geometry_only)
    gaussians.params["features"].requires_grad_(not geometry_only)
    if warmup_steps and step == warmup_steps + 1:
        # Warmup colors encode observed lighting, not material reflectance.
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
