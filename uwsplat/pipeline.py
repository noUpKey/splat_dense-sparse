"""End-to-end stages: sfm -> dense -> fusion -> train -> eval (each reads the previous stage's files)."""
import ast
import os
from dataclasses import fields, replace

from .common.io import list_scene_images, load_rgb, log, save_json
from .dense.run import DenseConfig, run_dense
from .eval.evaluate import evaluate
from .fusion.fuse import FusionConfig, run_fusion
from .gs.trainer import TrainConfig, train
from .sfm.export import export_sfm
from .sfm.reconstruct import SfMConfig, run_sfm

STAGES = ("sfm", "dense", "fusion", "train", "eval")

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
}


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
    for tag in variants:
        cfg = apply_overrides(TrainConfig(device=device), {**VARIANTS[tag], **sets.get("train", {})})
        if "train" in stages:
            log(f"=== medium-aware Gaussian splatting [{tag}]")
            results[f"train/{tag}"] = train(out_dir, cfg, tag)
        if "eval" in stages:
            results[f"eval/{tag}"] = evaluate(out_dir, tag, device, cfg.backend)
    return results
