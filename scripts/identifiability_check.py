"""E1: which camera / water parameters can a multi-view underwater fit tell apart? Numerical Jacobian null spaces.

python scripts/identifiability_check.py            (CPU, about a minute; writes identifiability_check.json next to it)

Forward model (linear radiance; channel c; frame k; scene point p seen from frame k at range r = r_kp):

    I_kc(p) = sum_j M_cj g_kj [ J_pj e^(-bD_j r) + B_j (1 - e^(-bB_j r)) ] + o_kc

Parameters: radiance J (N x 3), veiling light B (3), attenuation bD (3) and backscatter bB (3) coefficients, per-frame
gains g_k (diagonal: K x 3; or scalar exposure: K x 1), per-frame offsets o_k (K x 3, optional), shared 3x3 M
(optional; identity if absent). The Jacobian of all observations w.r.t. all free parameters is computed in float64;
its null space = directions no data can see (gauges / confounds).

Predictions, derived by hand and written down BEFORE running (gauge = exact null direction):
  G1  (g_kc, J_pc, B_c) -> (s_c g_kc, J_pc/s_c, B_c/s_c): 3 directions whenever gains are per channel (P0).
  G2  with a shared M and per-channel gains, M -> M D^-1, g_k -> D g_k: 3 more (M's column scale vs the gains).
  G3  M -> M A^-1, (J, B) -> A (J, B) needs A diag(g_k) med_k(r) = diag(g_k) med_k(r) A for all frames and ranges.
        - gains vary in chromaticity across frames (auto white balance): A must be diagonal  -> already counted;
        - gains achromatic (locked WB, exposure only) and bD, bB equal across channels: any A -> 9 (GL(3));
        - gains achromatic and coefficients differ by channel: A diagonal -> 3.
      => The medium shrinks the colour gauge GL(3) -> diagonal ONLY for a locked-white-balance camera; per-frame
         chromatic gains already do it in air. (Refines P3 in "Novel contributions beyond RUSplatting".)
  G4  offsets with bD_c = bB_c: (B_c += e, o_kc -= g_kc e, J_pc += e for all p) is exact: +3 directions.
      With bD != bB the same direction is only weak; its singular value falls as the minimum range grows (P2).
  Expected null dimensions, configurations below:
      D        diag gains, no offset, bD != bB                    3
      D+o      diag gains + offsets, bD != bB                     3  (+ one weak direction per channel)
      D+o eq   diag gains + offsets, bD == bB                     6
      C        shared M + diag gains (chromatic), bD != bB        6
      C eqch   shared M + diag gains (chromatic), coefficients equal across channels      6
      E+M      shared M + scalar exposure, coefficients differ by channel                 3 + 1 = 4
      E+M eqch shared M + scalar exposure, coefficients equal across channels             9 + 1 = 10
      air C    no water (B = 0, coefficients 0, fixed), shared M + chromatic gains        6
      air E+M  no water, shared M + scalar exposure                                       9 + 1 = 10
  (E+M: the "+1" is the scalar exposure scale vs M.)
"""
import json
import math
import os

import numpy as np
import torch

torch.set_default_dtype(torch.float64)


def make_scene(K=8, N=80, altitude=2.0, seed=0):
    """A camera transect over an uneven seabed: every point is seen by 3-6 frames at different ranges."""
    rng = np.random.default_rng(seed)
    xs = np.linspace(0, 6, K)                                         # camera centres along x, looking down
    cams = np.stack([xs, np.zeros(K), np.full(K, altitude)], 1)
    px = rng.uniform(-0.5, 6.5, N)
    py = rng.uniform(-1.5, 1.5, N)
    pz = 0.4 * np.sin(1.3 * px) * np.cos(0.9 * py) + rng.normal(0, 0.1, N)   # bumpy seabed
    pts = np.stack([px, py, pz], 1)
    vis = []
    for k in range(K):
        d = np.linalg.norm(pts - cams[k], axis=1)
        horiz = np.abs(pts[:, 0] - cams[k, 0])
        vis.append(horiz < 2.2)                                       # field of view along the transect
    vis = np.array(vis)                                               # (K, N)
    keep = vis.sum(0) >= 2
    pts, vis = pts[keep], vis[:, keep]
    R = np.linalg.norm(pts[None] - cams[:, None], axis=2)             # (K, N)
    return R, vis


def unpack(theta, spec):
    out, i = {}, 0
    for name, shape in spec:
        n = int(np.prod(shape))
        out[name] = theta[i:i + n].reshape(shape)
        i += n
    return out


def forward(theta, spec, fixed, R, vis):
    p = {**fixed, **unpack(theta, spec)}
    K, N = R.shape
    J, B, bD, bB = p["J"], p["B"], p["bD"], p["bB"]
    if p["g"].shape[1] == 1:
        g = p["g"].expand(K, 3)
    else:
        g = p["g"]
    r = R[..., None]                                                  # (K, N, 1)
    scene = J[None] * torch.exp(-bD * r) + B * (1 - torch.exp(-bB * r))  # (K, N, 3)
    cam = g[:, None, :] * scene                                       # diag gains
    if "M" in p:
        cam = cam @ p["M"].T
    if "o" in p:
        cam = cam + p["o"][:, None, :]
    return cam[vis].reshape(-1)                                       # observed pixels only (flattened)


def null_dim(cfg, R, vis, seed=1, rel_tol=1e-9):
    rng = np.random.default_rng(seed)
    K, N = R.shape
    t = lambda a: torch.tensor(a)
    vals = {
        "J": t(rng.uniform(0.2, 0.9, (N, 3))),
        "B": t(np.array([0.05, 0.25, 0.35])) if cfg["water"] else t(np.zeros(3)),
        "bD": t(np.array([0.45, 0.12, 0.08])) if cfg["water"] else t(np.zeros(3)),
        "bB": t(np.array([0.35, 0.18, 0.11])) if cfg["water"] else t(np.zeros(3)),
    }
    if cfg["water"] and cfg.get("equal_channels"):
        vals["bD"] = t(np.array([0.15, 0.15, 0.15]))
        vals["bB"] = t(np.array([0.22, 0.22, 0.22]))
    if cfg["water"] and cfg.get("bD_eq_bB"):
        vals["bB"] = vals["bD"].clone()
    if cfg["gains"] == "diag":
        vals["g"] = t(np.exp(rng.normal(0, 0.25, (K, 3))))           # chromatic: WB changes per frame
    else:
        vals["g"] = t(np.exp(rng.normal(0, 0.25, (K, 1))))           # exposure only (locked WB)
    if cfg.get("M"):
        vals["M"] = t(np.eye(3) + rng.normal(0, 0.15, (3, 3)))
    if cfg.get("offset"):
        vals["o"] = t(rng.normal(0, 0.02, (K, 3)))
    free = ["J", "g"] + (["B", "bD", "bB"] if cfg["water"] else []) + (["M"] if cfg.get("M") else []) + \
           (["o"] if cfg.get("offset") else [])
    spec = [(n, tuple(vals[n].shape)) for n in free]
    fixed = {n: v for n, v in vals.items() if n not in free}
    theta0 = torch.cat([vals[n].reshape(-1) for n in free])
    Rt, vt = torch.tensor(R), torch.tensor(vis)
    Jac = torch.autograd.functional.jacobian(lambda th: forward(th, spec, fixed, Rt, vt), theta0)
    s = torch.linalg.svdvals(Jac)
    null = int((s < rel_tol * s[0]).sum())
    return {"null_dim": null, "n_params": int(theta0.numel()), "n_obs": int(Jac.shape[0]),
            "smallest_nonnull": float(s[s >= rel_tol * s[0]][-1] / s[0]),
            "largest_null": float(s[s < rel_tol * s[0]][0] / s[0]) if null else None}, (Jac, spec, vals, theta0)


def offset_direction_strength(R, vis, altitude_shift, bD_minus_bB, seed=1):
    """P2: Jacobian norm along (B_c += 1, o_kc -= g_kc, J_pc += 1) for channel c, i.e. the G4 direction, and the
    smallest singular value of the full Jacobian, as the camera rises (minimum range grows)."""
    Rs = R + altitude_shift                                           # same geometry, every range longer
    rng = np.random.default_rng(seed)
    K, N = Rs.shape
    bD = np.array([0.30, 0.30, 0.30])
    vals = {"J": torch.tensor(rng.uniform(0.2, 0.9, (N, 3))), "B": torch.tensor([0.05, 0.25, 0.35]),
            "bD": torch.tensor(bD), "bB": torch.tensor(bD - bD_minus_bB),
            "g": torch.tensor(np.exp(rng.normal(0, 0.25, (K, 3)))), "o": torch.tensor(rng.normal(0, 0.02, (K, 3)))}
    free = ["J", "g", "B", "bD", "bB", "o"]
    spec = [(n, tuple(vals[n].shape)) for n in free]
    theta0 = torch.cat([vals[n].reshape(-1) for n in free])
    Rt, vt = torch.tensor(Rs), torch.tensor(vis)
    Jac = torch.autograd.functional.jacobian(lambda th: forward(th, spec, {}, Rt, vt), theta0)
    # the G4 direction for channel 0 (red), unit-normalised
    d = {n: torch.zeros_like(vals[n]) for n in free}
    c = 0
    d["B"][c] = 1.0
    d["o"][:, c] = -vals["g"][:, c]
    d["J"][:, c] = 1.0
    v = torch.cat([d[n].reshape(-1) for n in free])
    v = v / v.norm()
    s = torch.linalg.svdvals(Jac)
    predicted = float(torch.sqrt((vals["g"][:, c, None] ** 2 * torch.exp(-2 * vals["bB"][c] * Rt) * vt).sum()))
    return {"altitude_shift": altitude_shift, "r_min": float(Rs[vis].min()), "bD_minus_bB": bD_minus_bB,
            "jac_norm_along_G4": float((Jac @ v).norm()), "smallest_sv_rel": float(s[-1] / s[0]),
            "P2_offset_only_strength": predicted}


def main():
    R, vis = make_scene()
    configs = [
        ("D", dict(water=True, gains="diag"), 3),
        ("D+o", dict(water=True, gains="diag", offset=True), 3),
        ("D+o eq", dict(water=True, gains="diag", offset=True, bD_eq_bB=True), 6),
        ("C", dict(water=True, gains="diag", M=True), 6),
        ("C eqch", dict(water=True, gains="diag", M=True, equal_channels=True), 6),
        ("E+M", dict(water=True, gains="scalar", M=True), 4),
        ("E+M eqch", dict(water=True, gains="scalar", M=True, equal_channels=True), 10),
        ("air C", dict(water=False, gains="diag", M=True), 6),
        ("air E+M", dict(water=False, gains="scalar", M=True), 10),
    ]
    res = {"scene": {"frames": int(R.shape[0]), "points": int(R.shape[1]), "views_per_point_mean": float(vis.sum(0).mean()),
                     "range_min": float(R[vis].min()), "range_max": float(R[vis].max())}, "configs": {}}
    print(f"scene: {R.shape[0]} frames, {R.shape[1]} points, {vis.sum(0).mean():.1f} views/point, "
          f"range {R[vis].min():.2f}-{R[vis].max():.2f} m")
    print(f"{'config':10s} {'predicted':>9s} {'measured':>8s}  params  smallest non-null / largest null (relative)")
    ok = True
    for name, cfg, pred in configs:
        r, _ = null_dim(cfg, R, vis)
        r["predicted"] = pred
        res["configs"][name] = r
        ok &= r["null_dim"] == pred
        print(f"{name:10s} {pred:9d} {r['null_dim']:8d}  {r['n_params']:6d}  {r['smallest_nonnull']:.2e} / "
              f"{r['largest_null'] if r['largest_null'] is not None else float('nan'):.2e}"
              f"{'' if r['null_dim'] == pred else '   <-- differs'}")
    # robustness: other random draws of the same configurations
    rob = {}
    for seed in (2, 3):
        for name, cfg, pred in configs:
            r, _ = null_dim(cfg, R, vis, seed=seed)
            rob.setdefault(name, []).append(r["null_dim"])
    res["robustness_other_seeds"] = rob
    print("other seeds:", rob)

    # P2 / G4 strength as ranges grow, for bD - bB in {0.1, 0.03, 0}
    sweep = []
    for dbeta in (0.1, 0.03, 0.0):
        for shift in (0.0, 1.0, 2.0, 4.0, 8.0):
            sweep.append(offset_direction_strength(R, vis, shift, dbeta))
    res["offset_backscatter_sweep"] = sweep
    print("\nG4 direction (B += e, o_k -= g_k e, J += e), red channel: |Jacobian along it| as every range grows")
    print(f"{'bD-bB':>6s} {'r_min':>6s} {'|J v|':>10s} {'offset-only strength (P2)':>26s}")
    for s in sweep:
        print(f"{s['bD_minus_bB']:6.2f} {s['r_min']:6.2f} {s['jac_norm_along_G4']:10.2e} {s['P2_offset_only_strength']:26.3e}")
    res["all_null_dims_as_predicted"] = bool(ok)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "identifiability_check.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print("\nall null dimensions as predicted:", ok, "->", out)


if __name__ == "__main__":
    main()
