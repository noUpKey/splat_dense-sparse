"""NaN guard for upstream RUSplatting (theflash987/RUSplatting @ 33e2a62), applied in place.

Upstream has no protection against non-finite values: on the A40 (Oct 2026) Kwaj_base_s0 went NaN before
iteration 7000 and rendered black (PSNR 10.46). This patch changes nothing in a run that stays finite:

1. min-max normalisations and the per-channel max of the medium MLP get a 1e-8 floor on the denominator
   (gaussian_renderer/__init__.py, scene/nerf_model.py, utils/loss_utils.py) -- only active when a range is ~0;
2. train.py: after loss.backward(), if the loss or any gradient is non-finite, the step is skipped
   (all gradients cleared, no optimizer / scheduler step, no densification statistics) and the iteration is
   appended to <model_path>/nan_skipped.txt. Logging, saving and the iteration counter carry on as usual.

Usage: python scripts/patch_rusplatting_nan.py /path/to/RUSplatting
Idempotent (a second run reports "already patched"); every anchor must match exactly, otherwise nothing is
written and the script exits with an error. train.py is compiled after patching.
"""
import argparse
import os
import py_compile
import sys

MARK = "# [uwsplat NaN guard]"

FLOOR = ".clamp_min(1e-8)"

EDITS = {
    "gaussian_renderer/__init__.py": [
        ("depth_norm = (depth - depth.min()) / (depth.max() - depth.min())",
         f"depth_norm = (depth - depth.min()) / (depth.max() - depth.min()){FLOOR}  {MARK}"),
    ],
    "scene/nerf_model.py": [
        ("r = mul_original[:, 0] / mul_original[:, 0].max()",
         f"r = mul_original[:, 0] / mul_original[:, 0].max(){FLOOR}  {MARK}"),
        ("g = mul_original[:, 1] / mul_original[:, 1].max()",
         f"g = mul_original[:, 1] / mul_original[:, 1].max(){FLOOR}  {MARK}"),
        ("b = mul_original[:, 2] / mul_original[:, 2].max()",
         f"b = mul_original[:, 2] / mul_original[:, 2].max(){FLOOR}  {MARK}"),
    ],
    "utils/loss_utils.py": [
        ("depth_D = (depth_D - depth_D.min())/(depth_D.max() - depth_D.min())",
         f"depth_D = (depth_D - depth_D.min())/(depth_D.max() - depth_D.min()){FLOOR}  {MARK}"),
        ("depth_B = (depth_B - depth_B.min()) / (depth_B.max() - depth_B.min())",
         f"depth_B = (depth_B - depth_B.min()) / (depth_B.max() - depth_B.min()){FLOOR}  {MARK}"),
        ("depth = (depth - depth.min()) / (depth.max() - depth.min())\n        depth = depth.repeat(1, 3)",
         f"depth = (depth - depth.min()) / (depth.max() - depth.min()){FLOOR}  {MARK}\n        depth = depth.repeat(1, 3)"),
    ],
    "train.py": [
        # 1. after backward: decide whether this step is finite
        ("        loss.backward()\n        iter_end.record()\n",
         "        loss.backward()\n"
         f"        {MARK} skip the step if the loss or any gradient is non-finite\n"
         "        _params = [p for o in (gaussians.optimizer, gaussians.optimizer_net) for g in o.param_groups for p in g['params']]\n"
         "        _grads = [p.grad for p in _params if p.grad is not None]\n"
         "        if viewspace_point_tensor.grad is not None:\n"
         "            _grads.append(viewspace_point_tensor.grad)\n"
         "        _bad = not bool(torch.stack([torch.isfinite(loss).all()] + [torch.isfinite(g).all() for g in _grads]).all())\n"
         "        if _bad:\n"
         "            for _p in _params:\n"
         "                _p.grad = None\n"
         "            if uncertainty_optimizer is not None:\n"
         "                uncertainty_optimizer.zero_grad(set_to_none=True)\n"
         "            with open(os.path.join(dataset.model_path, 'nan_skipped.txt'), 'a') as _f:\n"
         "                _f.write(f'{iteration}\\n')\n"
         "        iter_end.record()\n"),
        # 2. keep the logged loss finite
        ("            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log\n",
         f"            if not _bad:  {MARK}\n"
         "                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log\n"),
        # 3. no optimizer / scheduler step on a skipped iteration
        ("            if iteration < opt.iterations:\n                gaussians.optimizer.step()\n",
         f"            if iteration < opt.iterations and not _bad:  {MARK}\n                gaussians.optimizer.step()\n"),
        # 4. no densification statistics from a skipped iteration
        ("            if iteration < opt.densify_until_iter:\n",
         f"            if iteration < opt.densify_until_iter and not _bad:  {MARK}\n"),
    ],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("repo", help="RUSplatting checkout")
    a = ap.parse_args()

    new = {}
    for rel, edits in EDITS.items():
        path = os.path.join(a.repo, rel)
        src = open(path).read()
        if MARK in src:
            print(f"{rel}: already patched")
            continue
        for old, rep in edits:
            n = src.count(old)
            if n != 1:
                sys.exit(f"{rel}: anchor found {n} times (expected 1), nothing written:\n{old}")
            src = src.replace(old, rep)
        if rel == "train.py" and "\nimport os" not in src and "\nimport os," not in src:
            src = "import os  " + MARK + "\n" + src
        new[rel] = src

    for rel, src in new.items():
        path = os.path.join(a.repo, rel)
        bak = open(path).read()
        open(path, "w").write(src)
        try:
            py_compile.compile(path, doraise=True)
        except py_compile.PyCompileError as e:
            open(path, "w").write(bak)
            sys.exit(f"{rel}: patched file does not compile, restored the original\n{e}")
        print(f"{rel}: patched")


if __name__ == "__main__":
    main()
