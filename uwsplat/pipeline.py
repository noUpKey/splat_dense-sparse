"""End-to-end stages: sfm -> dense -> fusion -> mvs -> train -> eval (each reads the previous stages' files)."""
import ast
import os
from dataclasses import fields, replace

from .common.io import list_scene_images, load_rgb, log, save_json
from .dense.run import DenseConfig, run_dense
from .eval.evaluate import evaluate
from .fusion.fuse import FusionConfig, run_fusion
from .gs.trainer import TrainConfig, train
from .mvs.run import MVSConfig, run_mvs
from .sfm.export import export_sfm
from .sfm.reconstruct import SfMConfig, run_sfm

STAGES = ("sfm", "dense", "fusion", "mvs", "train", "eval")

# Ablations: each is a set of TrainConfig overrides on top of the full method.
VARIANTS = {
    "full": {},
    "sparse_init": {"init": "sparse"},                       # SIFT points only (no dense water cloud)
    "dense_init": {"init": "dense"},                         # dense points only
    "no_medium": {"use_medium": False},                      # plain radiance + learned background colour
    "no_depth_priors": {"w_dense_depth": 0.0, "w_sparse_depth": 0.0},
    "no_appearance": {"appearance": False},
    "baseline_3dgs": {"init": "sparse", "use_medium": False, "appearance": False, "w_dense_depth": 0.0,
                      "w_sparse_depth": 0.0, "w_smooth": 0.0, "sh_degree": 3, "densify": {"water_grad_scale": 1.0}},
    # RUSplatting's edge-aware smoothness (rendered image, stopped at rendered depth edges), their code's weight
    "full_esl": {"w_esl": 0.05},
    # Dense cloud v2 (mvs stage): MVS-verified points on the wreck and textured seabed, network points only where
    # nothing could be verified, verified depth maps as the dense prior (also on the wreck). `mvs` changes only the
    # geometry (cloud + prior maps, same loss weights as `full`); the others add one component each.
    "mvs": {"geometry": "mvs"},
    "mvs_verified": {"geometry": "mvs", "init": "verified"},          # without the network-only points
    "mvs_prior": {"geometry": "mvs", "w_dense_depth": 0.1, "depth_decay": 0.3},  # trust verified depth longer
    "mvs_surfel": {"geometry": "mvs", "surfel_ratio": 0.2},           # Gaussians start as discs on the surface
    "mvs_esl": {"geometry": "mvs", "w_esl": 0.05},
    "mvs_all": {"geometry": "mvs", "w_dense_depth": 0.1, "depth_decay": 0.3, "surfel_ratio": 0.2, "w_esl": 0.05},
    # LPIPS-VGG as a training loss (weight 0.05), switched on at step 15000 (lpips_from) with a 1000-step ramp
    "full_lpips": {"w_lpips": 0.05},
    "mvs_lpips": {"geometry": "mvs", "w_lpips": 0.05},
    # Components, meant to be combined with '+' (see resolve_variant): "mvs_lpips+abs" = mvs_lpips with AbsGS
    # densification. On its own each one is a change on top of `full`.
    "abs": {"densify": {"absgrad": True, "grow_grad2d": 0.0008}},  # AbsGS: split by absolute screen-space
                                                                    # gradients (gsplat's advice: 4x threshold)
    "prior": {"w_dense_depth": 0.1, "depth_decay": 0.3},          # = mvs_prior's change
    "surfel": {"surfel_ratio": 0.2},                               # = mvs_surfel's change
    "esl": {"w_esl": 0.05},
    "lpips": {"w_lpips": 0.05},
    "lpips10k": {"w_lpips": 0.05, "lpips_from": 10_000},           # LPIPS before densification ends
    "lpipsx2": {"w_lpips": 0.1},
    "sh3": {"sh_degree": 3},
    "fs": {"forward_scatter": True},                               # forward scattering (gs/scatter.py)
    "fs3k": {"forward_scatter": True, "fs_from": 3000},            # ... switched on after the warm-up
}


def _merge(over: dict, extra: dict) -> dict:
    out = dict(over)
    for k, v in extra.items():
        out[k] = {**out.get(k, {}), **v} if k == "densify" and isinstance(v, dict) else v
    return out


def resolve_variant(tag: str) -> dict:
    """TrainConfig overrides of a variant. 'a+b+c' combines variants from left to right: later keys win and the
    `densify` dicts are merged, e.g. 'mvs_lpips+abs' = mvs_lpips with AbsGS densification. The tag (with its '+')
    is also the run's folder name."""
    over = {}
    for part in tag.split("+"):
        if part not in VARIANTS:
            raise KeyError(f"unknown variant {part!r} in {tag!r}; known: {', '.join(sorted(VARIANTS))}")
        over = _merge(over, VARIANTS[part])
    return over


def apply_overrides(cfg, overrides: dict):
    names = {f.name for f in fields(cfg)}
    bad = set(overrides) - names
    if bad:
        raise KeyError(f"unknown {type(cfg).__name__} fields: {sorted(bad)}")
    return replace(cfg, **overrides)


def parse_sets(items: list) -> dict:
    """['train.iterations=7000', 'dense.k_pairs=3'] -> {'train': {...}, 'dense': {...}}"""
    out = {}
    for it in items or []:
        key, val = it.split("=", 1)
        sec, name = key.split(".", 1)
        try:
            val = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            pass
        out.setdefault(sec, {})[name] = val
    return out


def stage_sfm(scene_dir: str, out_dir: str, cfg: SfMConfig, test_every: int = 8):
    paths = list_scene_images(scene_dir)
    images = [load_rgb(p) for p in paths]
    res = run_sfm(images, [os.path.basename(p) for p in paths], cfg)
    export_sfm(res, images, out_dir, os.path.basename(os.path.normpath(scene_dir)), test_every)
    save_json(os.path.join(out_dir, "sfm", "stats.json"), res.stats)
    return res.stats


def run_scene(scene_dir: str, out_dir: str, stages=STAGES, sets: dict = None, variants=("full",), device="cuda"):
    sets = sets or {}
    os.makedirs(out_dir, exist_ok=True)
    results = {}
    if "sfm" in stages:
        log(f"=== SfM (SIFT, sparse branch): {scene_dir}")
        results["sfm"] = stage_sfm(scene_dir, out_dir, apply_overrides(SfMConfig(device=device), sets.get("sfm", {})))
    if "dense" in stages:
        log("=== dense pointmap branch")
        results["dense"] = run_dense(out_dir, apply_overrides(DenseConfig(device=device), sets.get("dense", {})))
    if "fusion" in stages:
        log("=== multimodal fusion")
        results["fusion"] = run_fusion(out_dir, apply_overrides(FusionConfig(), sets.get("fusion", {})))
    if "mvs" in stages:
        log("=== MVS verification (dense cloud v2)")
        results["mvs"] = run_mvs(out_dir, apply_overrides(MVSConfig(device=device), sets.get("mvs", {})))
    for tag in variants:
        cfg = apply_overrides(TrainConfig(device=device), _merge(resolve_variant(tag), sets.get("train", {})))
        if "train" in stages:
            log(f"=== medium-aware Gaussian splatting [{tag}]")
            results[f"train/{tag}"] = train(out_dir, cfg, tag)
        if "eval" in stages:
            results[f"eval/{tag}"] = evaluate(out_dir, tag, device, cfg.backend)
    return results
