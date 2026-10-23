"""Per-Gaussian water model == volumetric medium integration along a ray.

python -m tests.test_medium_compositing

Along a ray with sorted Gaussians (alpha_i, range r_i), the volumetric image is
  sum_i w_i J_i e^{-bD r_i}  +  sum_{k=0..N} T_k B (e^{-bB r_k} - e^{-bB r_{k+1}}),  r_0 = 0, r_{N+1} = inf
(object transmittance T_k after k Gaussians, w_i = T_{i-1} alpha_i). uwsplat instead
composites per-Gaussian colours J_i e^{-bD r_i} + B (1 - e^{-bB r_i}) over a background B.
"""
import numpy as np


def main(trials=1000, seed=0):
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(trials):
        n = rng.integers(1, 12)
        a = rng.uniform(0, 0.95, n)
        r = np.sort(rng.uniform(0.05, 4.0, n))
        J, B = rng.uniform(0, 1, (n, 3)), rng.uniform(0, 1, 3)
        bD, bB = rng.uniform(0.01, 2, 3), rng.uniform(0.01, 2, 3)
        T = np.concatenate([[1.0], np.cumprod(1 - a)])          # T_0 .. T_N
        w = T[:-1] * a
        per_gauss = (w[:, None] * (J * np.exp(-bD * r[:, None]) + B * (1 - np.exp(-bB * r[:, None])))).sum(0) + T[-1] * B
        edges = np.concatenate([[0.0], r])
        seg = np.exp(-bB * edges[:, None]) - np.concatenate([np.exp(-bB * r[:, None]), np.zeros((1, 3))])
        volumetric = (w[:, None] * J * np.exp(-bD * r[:, None])).sum(0) + (T[:, None] * B * seg).sum(0)
        worst = max(worst, np.abs(per_gauss - volumetric).max())
    print(f"max |per-Gaussian - volumetric| over {trials} random rays: {worst:.2e}")
    assert worst < 1e-12
    print("per-Gaussian medium compositing is exact")


if __name__ == "__main__":
    main()
