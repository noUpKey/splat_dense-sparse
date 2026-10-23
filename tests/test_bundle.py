"""Synthetic check of the Schur-complement bundle adjuster.

python -m tests.test_bundle
"""
import numpy as np

from uwsplat.common.geometry import so3_exp
from uwsplat.sfm.bundle import bundle_adjust, reprojection_errors


def make_problem(rng, n_cam=12, n_pts=3000, noise=0.5):
    true = np.array([548.0, 462.6, 640.0, 360.0, -0.09, 0.03])  # anisotropic pixels like Submerged3D
    X = rng.uniform([-2, -1.2, 3], [2, 1.2, 6], (n_pts, 3))
    R = so3_exp(rng.normal(0, 0.08, (n_cam, 3)))
    C = np.stack([np.linspace(-1.5, 1.5, n_cam), rng.normal(0, 0.2, n_cam), rng.normal(0, 0.3, n_cam)], 1)
    t = -np.einsum("cij,cj->ci", R, C)
    cam = np.repeat(np.arange(n_cam), n_pts)
    pt = np.tile(np.arange(n_pts), n_cam)
    from uwsplat.sfm.bundle import project
    uv, P, *_ = project(true, R, t, X, cam, pt)
    keep = (P[:, 2] > 0) & (uv[:, 0] > 0) & (uv[:, 0] < 1280) & (uv[:, 1] > 0) & (uv[:, 1] < 720)
    keep &= rng.random(len(cam)) < 0.5
    cam, pt, uv = cam[keep], pt[keep], uv[keep] + rng.normal(0, noise, (keep.sum(), 2))
    return true, R, t, X, cam, pt, uv


def main():
    rng = np.random.default_rng(0)
    true, R, t, X, cam, pt, uv = make_problem(rng)
    init = true.copy()
    init[[0, 1]] *= [1.08, 0.94]
    init[[4, 5]] = 0.0
    R0 = so3_exp(rng.normal(0, 0.01, (len(R), 3))) @ R
    R0[0] = R[0]
    t0 = t + rng.normal(0, 0.02, t.shape)
    t0[0] = t[0]
    X0 = X + rng.normal(0, 0.03, X.shape)
    free = np.ones(len(R), bool)
    free[0] = False
    e0 = reprojection_errors(init, R0, t0, X0, cam, pt, uv)
    cp, R1, t1, X1, info = bundle_adjust(init, R0, t0, X0, cam, pt, uv, free, fixed_ref=0, max_iters=60)
    e1 = reprojection_errors(cp, R1, t1, X1, cam, pt, uv)
    print(f"obs {len(cam)}  mean err before {e0.mean():.2f}px  after {e1.mean():.3f}px  iters {info['iters']}")
    print("true   ", np.round(true, 4))
    print("estim. ", np.round(cp, 4))
    # 0.5 px isotropic noise -> expected mean residual norm ~ 0.5*sqrt(pi/2) = 0.63 px
    assert e1.mean() < 0.7, e1.mean()
    assert abs(cp[0] - true[0]) < 3 and abs(cp[1] - true[1]) < 3, cp
    assert abs(cp[4] - true[4]) < 0.01, cp
    print("bundle adjustment test passed")


if __name__ == "__main__":
    main()
