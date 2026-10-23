"""Bundle adjustment: Levenberg-Marquardt with an explicit Schur complement.

Parameters
  camera i  : R_i (left update R <- exp([dw]x) R) and t_i, world-to-camera
  intrinsics: any subset of (fx, fy, k1, k2) shared by all images; the
              principal point stays fixed at the image centre, as in COLMAP
  point j   : X_j
Residual of an observation = projection - measurement (pixels), robustified with
a Cauchy loss through IRLS weights.

With ~20 images the reduced camera system has ~6*20+4 unknowns, so it is
assembled densely as S = U - W V^-1 W^T (V is block-diagonal, 3x3 per point).
Gauge: the reference camera is fixed and one translation coordinate of a
second camera is frozen to remove the scale ambiguity (COLMAP's strategy).
"""
import numpy as np

from ..common.geometry import camera_center, skew, so3_exp

INTRINSIC_INDEX = {"fx": 0, "fy": 1, "k1": 4, "k2": 5}


def project(cam_params, R, t, X, cam, pt):
    fx, fy, cx, cy, k1, k2 = cam_params
    P = np.einsum("oij,oj->oi", R[cam], X[pt]) + t[cam]
    z = np.where(np.abs(P[:, 2]) < 1e-9, 1e-9, P[:, 2])
    x, y = P[:, 0] / z, P[:, 1] / z
    r2 = x * x + y * y
    d = 1.0 + k1 * r2 + k2 * r2 * r2
    uv = np.stack([fx * d * x + cx, fy * d * y + cy], -1)
    return uv, P, x, y, r2, d, z


def reprojection_errors(cam_params, R, t, X, cam, pt, uv):
    uv_hat, P, *_ = project(cam_params, R, t, X, cam, pt)
    err = np.linalg.norm(uv_hat - uv, axis=1)
    err[P[:, 2] <= 1e-6] = np.inf
    return err


def _choose_scale_gauge(R, t, free, fixed_ref):
    """Freeze the translation coordinate of a free camera that is most sensitive to scale."""
    if fixed_ref is None or not np.any(free):
        return None
    C = camera_center(R, t)
    best, best_val = None, 0.0
    for i in np.where(free)[0]:
        v = np.abs(R[i] @ (C[i] - C[fixed_ref]))
        k = int(np.argmax(v))
        if v[k] > best_val:
            best, best_val = (int(i), k), v[k]
    return best


def bundle_adjust(cam_params, R, t, X, cam, pt, uv, free_cams, refine=("fx", "fy", "k1", "k2"),
                  fixed_ref=None, max_iters=30, loss_scale=2.0, lm_init=1e-4, tol=1e-8, verbose=False):
    """Returns (cam_params, R, t, X, info). Inputs are not modified.

    cam/pt/uv: observation arrays (camera index into R/t, point index into X, pixels).
    free_cams: bool mask over cameras whose pose is optimised.
    fixed_ref: index of the fixed reference camera used to pick the scale gauge
               (None when several cameras are fixed, e.g. local BA).
    """
    cam_params = np.asarray(cam_params, np.float64).copy()
    R, t, X = R.copy(), t.copy(), X.copy()
    n_img, n_p, n_o = len(R), len(X), len(cam)
    c2 = loss_scale ** 2
    free = np.asarray(free_cams, bool).copy()
    fi = np.where(free)[0]
    free_idx = np.full(n_img, -1)
    free_idx[fi] = np.arange(len(fi))
    kidx = [INTRINSIC_INDEX[k] for k in refine]
    nk = len(kidx)
    Dc = 6 * len(fi) + nk
    trash = Dc
    Dt = Dc + 1

    base = free_idx[cam]
    cols_pose = np.where(base[:, None] >= 0, 6 * base[:, None] + np.arange(6), trash)
    gauge = _choose_scale_gauge(R, t, free, fixed_ref)
    if gauge is not None:
        gcol = 6 * free_idx[gauge[0]] + 3 + gauge[1]
        cols_pose[cols_pose == gcol] = trash
    cols = np.concatenate([cols_pose, np.broadcast_to(6 * len(fi) + np.arange(nk), (n_o, nk))], 1)
    ar3 = np.arange(3)

    def evaluate(cp, R_, t_, X_):
        uv_hat, P, x, y, r2, d, z = project(cp, R_, t_, X_, cam, pt)
        e = uv_hat - uv
        valid = P[:, 2] > 1e-6
        s = np.where(valid, (e * e).sum(1), 0.0)
        rho = np.where(valid, c2 * np.log1p(s / c2), c2 * np.log1p(1e4))
        return 0.5 * rho.sum(), e, valid, s, (P, x, y, r2, d, z)

    def linearize(cp, R_, t_, e, valid, s, aux):
        P, x, y, r2, d, z = aux
        fx, fy, _, _, k1, k2 = cp
        w = np.where(valid, 1.0 / (1.0 + s / c2), 0.0)
        dp = k1 + 2.0 * k2 * r2
        Jxy = np.empty((n_o, 2, 2))
        Jxy[:, 0, 0] = fx * (d + 2 * x * x * dp)
        Jxy[:, 0, 1] = fx * 2 * x * y * dp
        Jxy[:, 1, 0] = fy * 2 * x * y * dp
        Jxy[:, 1, 1] = fy * (d + 2 * y * y * dp)
        iz = 1.0 / z
        Jpr = np.zeros((n_o, 2, 3))
        Jpr[:, 0, 0] = iz
        Jpr[:, 0, 2] = -x * iz
        Jpr[:, 1, 1] = iz
        Jpr[:, 1, 2] = -y * iz
        JP = Jxy @ Jpr
        Jw = JP @ (-skew(P - t_[cam]))
        JX = JP @ R_[cam]
        Jk = np.zeros((n_o, 2, nk))
        for c, k in enumerate(kidx):
            if k == 0:
                Jk[:, 0, c] = d * x
            elif k == 1:
                Jk[:, 1, c] = d * y
            elif k == 4:
                Jk[:, 0, c], Jk[:, 1, c] = fx * x * r2, fy * y * r2
            else:
                Jk[:, 0, c], Jk[:, 1, c] = fx * x * r2 * r2, fy * y * r2 * r2
        Jc = np.concatenate([Jw, JP, Jk], 2)
        Jcw = Jc * w[:, None, None]
        JXw = JX * w[:, None, None]
        m = Jc.shape[2]
        U = np.bincount((cols[:, :, None] * Dt + cols[:, None, :]).ravel(),
                        np.einsum("oai,oaj->oij", Jcw, Jc).ravel(), Dt * Dt).reshape(Dt, Dt)
        W = np.bincount((cols[:, :, None] * (3 * n_p) + 3 * pt[:, None, None] + ar3).ravel(),
                        np.einsum("oai,oaj->oij", Jcw, JX).ravel(), Dt * 3 * n_p).reshape(Dt, 3 * n_p)
        V = np.bincount((pt[:, None] * 9 + np.arange(9)).ravel(),
                        np.einsum("oai,oaj->oij", JXw, JX).reshape(n_o, 9).ravel(), n_p * 9).reshape(n_p, 3, 3)
        gc = np.bincount(cols.ravel(), np.einsum("oai,oa->oi", Jcw, e).ravel(), Dt)
        gp = np.bincount((pt[:, None] * 3 + ar3).ravel(), np.einsum("oai,oa->oi", JXw, e).ravel(),
                         n_p * 3).reshape(n_p, 3)
        assert m == cols.shape[1]
        return U[:Dc, :Dc], W[:Dc], V, gc[:Dc], gp

    def apply(cp, R_, t_, X_, dc, dpnt):
        cp2, R2, t2 = cp.copy(), R_.copy(), t_.copy()
        if len(fi):
            dpose = dc[:6 * len(fi)].reshape(-1, 6)
            R2[fi] = so3_exp(dpose[:, :3]) @ R_[fi]
            t2[fi] = t_[fi] + dpose[:, 3:]
        if nk:
            cp2[kidx] += dc[6 * len(fi):]
        return cp2, R2, t2, X_ + dpnt

    cost, e, valid, s, aux = evaluate(cam_params, R, t, X)
    cost0, lam, it = cost, lm_init, 0
    for it in range(max_iters):
        U, W, V, gc, gp = linearize(cam_params, R, t, e, valid, s, aux)
        dU = np.maximum(np.diag(U).copy(), 1e-12)
        dV = np.maximum(V[:, ar3, ar3].copy(), 1e-12)
        accepted = False
        for _ in range(12):
            Vd = V.copy()
            Vd[:, ar3, ar3] += lam * dV + 1e-12
            Vinv = np.linalg.inv(Vd)
            WV = np.einsum("cpi,pij->cpj", W.reshape(Dc, n_p, 3), Vinv).reshape(Dc, -1)
            S = U + np.diag(lam * dU) - WV @ W.T
            rhs = -gc + WV @ gp.ravel()
            try:
                L = np.linalg.cholesky(S)
                dc = np.linalg.solve(L.T, np.linalg.solve(L, rhs))
            except np.linalg.LinAlgError:
                dc = np.linalg.lstsq(S, rhs, rcond=None)[0]
            dpnt = np.einsum("pij,pj->pi", Vinv, -gp - (W.T @ dc).reshape(n_p, 3))
            cand = apply(cam_params, R, t, X, dc, dpnt)
            res = evaluate(*cand)
            if res[0] < cost:
                accepted = True
                cam_params, R, t, X = cand
                improvement = cost - res[0]
                cost, e, valid, s, aux = res
                lam = max(lam / 10.0, 1e-10)
                break
            lam *= 10.0
        if verbose:
            print(f"  BA it {it:2d} cost {cost:.4e} lambda {lam:.1e}")
        if not accepted or improvement < tol * cost:
            break
    err = reprojection_errors(cam_params, R, t, X, cam, pt, uv)
    info = {"iters": it + 1, "cost0": cost0, "cost": cost, "mean_err": float(np.mean(err[np.isfinite(err)])) if n_o else 0.0}
    return cam_params, R, t, X, info
