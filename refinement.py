"""PORT-GS refinement schedule with explicit opacity resets and a point budget."""

from dataclasses import dataclass

import torch
from gsplat.strategy import DefaultStrategy
from gsplat.strategy.ops import duplicate, reset_opa, split


@dataclass
class Refinement(DefaultStrategy):
    """Reuse gsplat's statistics and geometry operations; own their scheduling."""

    max_points: int = 200000

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

        if step > 0 and step % self.reset_every == 0:
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
        capacity = self.max_points - len(params["means"])
        if capacity == 0:
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
            split(params, optimizers, state, split_mask, revised_opacity=self.revised_opacity)
