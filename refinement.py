"""PORT-GS refinement schedule with explicit opacity resets and a point budget."""

from dataclasses import dataclass

import torch
from gsplat.strategy import DefaultStrategy
from gsplat.strategy.ops import duplicate, reset_opa, split
from gsplat.strategy.ops import _update_param_with_optimizer
from gsplat.utils import normalized_quat_to_rotmat


@torch.no_grad()
def split_surfels(params, optimizers, state, mask):
    """Split in the tangent plane; retain two scales and optimizer alignment."""
    selected, rest = torch.where(mask)[0], torch.where(~mask)[0]
    scales = params["scales"][selected].exp()
    rotation = normalized_quat_to_rotmat(torch.nn.functional.normalize(params["quats"][selected], dim=-1))
    offsets = torch.randn((2, len(selected), 2), device=scales.device, dtype=scales.dtype) * scales
    offsets = torch.einsum("nij,bnj->bni", rotation[..., :2], offsets)

    def param_fn(name, value):
        repeat = [2] + [1] * (value.ndim - 1)
        if name == "means":
            children = (value[selected] + offsets).reshape(-1, 3)
        elif name == "scales":
            children = (scales / 1.6).log().repeat(2, 1)
        else:
            children = value[selected].repeat(repeat)
        return torch.nn.Parameter(torch.cat((value[rest], children)), requires_grad=value.requires_grad)

    def optimizer_fn(key, value):
        return torch.cat((value[rest], value.new_zeros((2 * len(selected), *value.shape[1:]))))

    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
    for key, value in state.items():
        if isinstance(value, torch.Tensor):
            repeat = [2] + [1] * (value.ndim - 1)
            state[key] = torch.cat((value[rest], value[selected].repeat(repeat)))


@dataclass
class Refinement(DefaultStrategy):
    """Reuse gsplat's statistics and geometry operations; own their scheduling.

    ``opacity_reset_every`` controls only opacity resets (None follows
    ``reset_every``, 0 disables them). ``reset_every`` still pauses refinement
    and delays gsplat's large-Gaussian pruning, so that schedule is unchanged.
    """

    max_points: int = 200000
    opacity_reset_every: int | None = None
    # 0 applies max_points from the start. Otherwise the cap rises linearly from
    # initial_points at refine_start_iter to max_points at this step, so points
    # are not all committed while the appearance model is still incomplete.
    budget_ramp: int = 0
    initial_points: int = 0

    def point_cap(self, step):
        if not self.budget_ramp:
            return self.max_points
        progress = min(1.0, max(0.0, (step - self.refine_start_iter) / (self.budget_ramp - self.refine_start_iter)))
        return int(self.initial_points + progress * (self.max_points - self.initial_points))

    @torch.no_grad()
    def step_post_backward(self, params, optimizers, state, step, info, packed=False):
        if step >= self.refine_stop_iter:
            return

        self._update_state(params, state, info, packed=packed)
        if (
            step > self.refine_start_iter
            and step % self.refine_every == 0
            and step % self.reset_every >= self.pause_refine_after_reset
        ):
            self._grow_gs(params, optimizers, state, step)
            self._prune_gs(params, optimizers, state, step)
            state["grad2d"].zero_()
            state["count"].zero_()
            if self.refine_scale2d_stop_iter > 0:
                state["radii"].zero_()

        interval = self.reset_every if self.opacity_reset_every is None else self.opacity_reset_every
        if interval > 0 and step > 0 and step % interval == 0:
            reset_opa(params, optimizers, state, value=2 * self.prune_opa)
            return {
                "event": "opacity_reset",
                "step": step,
                "opacity_max": params["opacities"].sigmoid().max().item(),
                "points": len(params["means"]),
            }

    @torch.no_grad()
    def _grow_gs(self, params, optimizers, state, step):
        """Spend available points on the largest eligible image gradients."""
        capacity = self.point_cap(step) - len(params["means"])
        if capacity <= 0:
            return

        gradients = state["grad2d"] / state["count"].clamp_min(1)
        small = params["scales"].exp().amax(-1) <= self.grow_scale3d * state["scene_scale"]
        duplicate_mask = (gradients > self.grow_grad2d) & small
        split_mask = (gradients > self.grow_grad2d) & ~small
        if step < self.refine_scale2d_stop_iter:
            split_mask |= state["radii"] > self.grow_scale2d
        duplicate_mask &= ~split_mask

        # A duplicate or a two-child split each adds one net Gaussian.
        cost = duplicate_mask.long() + split_mask.long()
        candidates = torch.where(cost > 0)[0]
        order = candidates[gradients[candidates].argsort(descending=True, stable=True)]
        chosen = order[cost[order].cumsum(0) <= capacity]
        selected = torch.zeros_like(small)
        selected[chosen] = True
        duplicate_mask &= selected
        split_mask &= selected

        count = int(duplicate_mask.sum())
        if count:
            duplicate(params, optimizers, state, duplicate_mask)
        split_mask = torch.cat((split_mask, split_mask.new_zeros(count)))
        if split_mask.any():
            if params["scales"].shape[-1] == 2:
                split_surfels(params, optimizers, state, split_mask)
            else:
                split(params, optimizers, state, split_mask, revised_opacity=self.revised_opacity)
