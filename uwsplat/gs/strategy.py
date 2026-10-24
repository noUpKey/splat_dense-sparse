"""Adaptive density control (3DGS clone / split / prune / opacity reset), modality-aware.

Built on gsplat's tested parameter/optimizer surgery ops, with two changes from
gsplat 1.5.3's DefaultStrategy:
  * the periodic opacity reset actually runs (upstream, `step % every == 0 & step > 0`
    parses as a chained comparison ending in `0 > 0` and never fires);
  * Gaussians that originate from the dense water branch must reach
    `water_grad_scale` x the gradient threshold before they are densified:
    the water region is smooth, and splitting there mostly creates floaters,
    whereas wreck Gaussians (SIFT branch) densify normally for precision.
The modality tag lives in the strategy state, so children inherit it.
"""
from dataclasses import dataclass

import torch
from gsplat.strategy.ops import duplicate, remove, reset_opa, split

from .model import DENSE


@dataclass
class DensifyStrategy:
    prune_opa: float = 0.005
    grow_grad2d: float = 0.0002
    water_grad_scale: float = 2.0
    grow_scale3d: float = 0.01
    prune_scale3d: float = 0.1
    refine_start: int = 500
    refine_stop: int = 15_000
    refine_every: int = 100
    reset_every: int = 3000
    reset_value: float = 0.01
    absgrad: bool = False
    max_gaussians: int = 3_000_000

    def initialize_state(self, scene_scale: float, modality: torch.Tensor) -> dict:
        return {"grad2d": None, "count": None, "scene_scale": scene_scale, "modality": modality.to(torch.uint8)}

    def step_pre_backward(self, info: dict):
        info["means2d"].retain_grad()

    @torch.no_grad()
    def step_post_backward(self, params, optimizers, state, step: int, info: dict) -> dict:
        stats = {}
        if step >= self.refine_stop:
            return stats
        self._accumulate(params, state, info)
        if step > self.refine_start and step % self.refine_every == 0 and state["grad2d"] is not None:
            stats["dupli"], stats["split"] = self._grow(params, optimizers, state)
            stats["pruned"] = self._prune(params, optimizers, state, step)
            state["grad2d"].zero_()
            state["count"].zero_()
        if step > 0 and step % self.reset_every == 0:
            reset_opa(params, optimizers, state, value=self.reset_value)
            stats["reset"] = True
        return stats

    def _accumulate(self, params, state, info):
        if self.absgrad and getattr(info["means2d"], "absgrad", None) is None and info["means2d"].grad is not None:
            raise RuntimeError("absgrad=True but the rasteriser did not provide means2d.absgrad "
                               "(render(..., absgrad=strategy.absgrad) must be passed through)")
        g = info["means2d"].absgrad if self.absgrad else info["means2d"].grad
        if g is None:
            return
        g = g.clone()
        g[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
        g[..., 1] *= info["height"] / 2.0 * info["n_cameras"]
        n = len(params["means"])
        if state["grad2d"] is None:
            state["grad2d"] = torch.zeros(n, device=g.device)
            state["count"] = torch.zeros(n, device=g.device)
        sel = (info["radii"] > 0).all(dim=-1)  # (C, N)
        ids = torch.where(sel)[1]
        state["grad2d"].index_add_(0, ids, g[sel].norm(dim=-1))
        state["count"].index_add_(0, ids, torch.ones_like(ids, dtype=torch.float32))

    def _grow(self, params, optimizers, state):
        grads = state["grad2d"] / state["count"].clamp_min(1)
        thr = torch.full_like(grads, self.grow_grad2d)
        thr[state["modality"] == DENSE] *= self.water_grad_scale
        high = grads > thr
        budget = self.max_gaussians - len(grads)
        if budget <= 0:
            return 0, 0
        if high.sum() > budget:
            cut = torch.topk(torch.where(high, grads, torch.zeros_like(grads)), budget).values[-1]
            high &= grads >= cut
        small = torch.exp(params["scales"]).max(-1).values <= self.grow_scale3d * state["scene_scale"]
        is_dupli, is_split = high & small, high & ~small
        n_dupli, n_split = int(is_dupli.sum()), int(is_split.sum())
        if n_dupli:
            duplicate(params, optimizers, state, is_dupli)
        is_split = torch.cat([is_split, torch.zeros(n_dupli, dtype=torch.bool, device=is_split.device)])
        if n_split:
            split(params, optimizers, state, is_split)
        return n_dupli, n_split

    def _prune(self, params, optimizers, state, step):
        prune = torch.sigmoid(params["opacities"]) < self.prune_opa
        if step > self.reset_every:
            prune |= torch.exp(params["scales"]).max(-1).values > self.prune_scale3d * state["scene_scale"]
        n = int(prune.sum())
        if n:
            remove(params, optimizers, state, prune)
        return n
