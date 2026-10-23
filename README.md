# uwsplat: multimodal underwater Gaussian splatting on Submerged3D

Shipwreck reconstruction and novel-view synthesis from 20 underwater photos per scene. The approach is
multimodal: a **sparse SIFT structure-from-motion branch** gives precise poses and wreck geometry, and a
**dense pointmap branch** gives geometry for the water region where SIFT finds nothing. Both feed a
**medium-aware Gaussian splatting** model, which is validated with **PSNR / SSIM / LPIPS** on held-out views.

```
 images ─┬─▶ SPARSE (wreck, precision)  CLAHE+RootSIFT ─ mutual-ratio matching ─ MAGSAC ─ tracks ─ fx/fy
         │                              self-calibration ─ PnP / DLT ─ Schur-complement bundle adjustment
         │                              ─▶ poses, intrinsics, SIFT points
         └─▶ DENSE (water region)       ViT-L encoder (cached) ─ cross-attending decoders ─ DPT head
                                        ─▶ pointmaps ─ SfM scale ─ fusion ─ multi-view consensus
 FUSION   per-pixel routing:  wreck → SIFT points | water surface → dense points | open water → medium
 SPLAT    3D Gaussians ─▶ restored colour J, depth Z ─▶ per-Gaussian water model (β_D, β_B, B∞(d))
          ─▶ underwater image ◀─ reconstruction + depth + colour + backscatter + opacity losses
 EVAL     PSNR / SSIM / LPIPS(VGG) on every 8th frame (held out), plus wreck / water region PSNR
```

## Run it on the A40

```bash
bash setup_env.sh               # Python 3.10, torch 2.4.1+cu121, prebuilt gsplat (sm_86 kernels), weights
conda activate uwsplat          # or: source .venv/bin/activate
python scripts/run_all.py --data Submerged3D --out outputs                    # full method, 4 scenes
python scripts/run_all.py --data Submerged3D --out outputs \
       --variants sparse_init,dense_init,no_medium,no_depth_priors,baseline_3dgs  # ablations (reuse caches)
cat outputs/results.md          # per-scene + average PSNR / SSIM / LPIPS table
```

- **SLURM**: `sbatch slurm_run_all.sh`. **Jupyter**: `run_pipeline.ipynb`.
- **Single scene**: `python scripts/run_scene.py --scene Submerged3D/Tokai --out outputs/Tokai`.
- **Overrides**: any config field can be changed with `--set section.key=value`. The sections are `sfm`, `dense`, `fusion` and `train`, e.g. `--set train.iterations=20000 --set dense.k_pairs=6`.
- **Older drivers**: if the node's driver predates CUDA 12.1, use `CUDA_TAG=cu118 bash setup_env.sh`.

The protocol is the one used by RUSplatting and Caldeira et al. 2026: 720p, and every 8th frame (sorted by name) is a test view. That gives 3 test and 17 training views per scene. The test frames are used for camera registration only. Dense geometry and depth priors come from training views only, and SIFT points need ≥ 2 training-view observations.

## What is implemented (by task point)

**1. 3D reconstruction → Gaussian splatting.** The `sfm` → `dense` → `fusion` → `train` → `eval` stages, in `uwsplat/pipeline.py`.

**2–3. Multimodal geometry.**
- *Sparse branch, SIFT* (`uwsplat/sfm`). A COLMAP-like incremental SfM written from scratch:
  - CLAHE-equalized RootSIFT, mutual nearest neighbours + ratio test, MAGSAC verification, tracks.
  - Self-calibration of fx and fy from the fundamental matrices (the frames have no EXIF).
  - AP3P-RANSAC registration, multi-view DLT triangulation.
  - Levenberg–Marquardt bundle adjustment with an explicit Schur complement and COLMAP's gauge fixing.
- *Dense branch* (`uwsplat/dense`). My own implementation of the two-view pointmap network from the task figure (ViT-L encoder, 2D RoPE, two cross-attending decoders, DPT head, `v/‖v‖·(e^‖v‖−1)` points, `1+e^c` confidence). It loads the public DUSt3R 512-DPT weights and is pruned to what the pipeline needs.
- *Routing* (`uwsplat/fusion`). Each training-view pixel goes to one of three classes:
  - **wreck**: dense SIFT-track support, covered by SfM points;
  - **water surface**: confident, multi-view consistent dense depth (sand, seabed, hazy far structure), covered by dense points;
  - **open water**: no surface; rendered by the medium model.

  The SIFT points and dense points are merged on a voxel grid, with SIFT taking priority, then outlier-filtered. The result is written as `fusion/fused.ply` with a per-point `source` field.

**4. Medium-aware splatting** (`uwsplat/gs`). SeaSplat-like: 3D Gaussians → restored colour J and depth Z → underwater image formation `I = J·e^(−β_D r) + B∞·(1−e^(−β_B r))` → reconstruction, smooth-depth, colour, backscatter and background-opacity losses. The parts that are redundant here were removed (tables below).

**5. Validation** (`uwsplat/eval`). PSNR, SSIM (3DGS Gaussian-window implementation) and LPIPS-VGG, computed on 8-bit quantized renders exactly as the 3DGS/RUSplatting `metrics.py` does. Results go to `outputs/<scene>/eval/<variant>/metrics.json` and `outputs/results.md`. There are also montages (GT | render | restored J | depth) and region-wise PSNR (wreck vs water).

## How the reference architectures were adapted

### Dense network (task figure 1 → `dense/network.py`)

| Reference | Here | Reason |
|---|---|---|
| 512×288 crop | 512×336, resampled to **square pixels** (21×32 tokens) | The frames are 3:2 photos squeezed into 1280×720: both COLMAP and our SfM find fy/fx = 0.844 = 1.5/1.778. The network was trained on square pixels. |
| Both images re-encoded for every pair (×2 when symmetrized) | Encoder runs **once per image**; tokens are cached | 17 ViT-L passes per scene instead of ≥ 136 |
| Head 2 → `pts3d_in_other_view` | **Not built** (20.2M parameters never loaded) | Known SfM poses make cross-view pointmaps redundant |
| Last block of decoder 2 | **Dropped** (9.5M) | It only feeds head 2 |
| Global alignment (optimizes poses, focal lengths, depths of all pairs) | Closed-form per-pair scale fitted to SIFT depths → confidence-weighted per-view fusion → COLMAP-style multi-view consistency + consensus | Poses are already known and precise |
| Confidence used as a weight | Also used as a **water-column gate** (conf ≥ 1.2, or ≥ 1.05 on textured pixels) | Open water sits at the confidence floor (1.0) and passes plain consistency checks |

My implementation reproduces the official DUSt3R model to **2.0e-6** (pointmaps) and **4.6e-6** relative (confidence) on a real Kwaj pair (`tests/test_dense_equivalence.py`).

### Splatting (task figure 2, SeaSplat → `gs/`)

| SeaSplat | Here | Reason |
|---|---|---|
| Separate colour and depth rasterization passes | **One** 7-channel gsplat pass (underwater I, restored J, depth) | Redundant pass |
| Image formation applied per pixel, on a rendered depth that is **min-max normalized per image** | Applied **per Gaussian**, per camera, on metric range | Per-image normalization gives β a different unit in every view. The per-Gaussian form composited over B∞ is *exactly* the volumetric medium integral (`tests/test_medium_compositing.py`, error 5.6e-16). |
| DeepSeeColor modules: double-exponential attenuation, residual backscatter, white-balance gain | 3 β_D + 3 β_B + B∞ | The extra terms are not identifiable once Gaussian colours are free |
| Constant veiling light | **B∞(d)**: sigmoid of a degree-1 SH of the world-space ray direction | Open water is brighter towards the surface (Isro, Kwaj) |
| Learned background colour, copied into B∞ at a switch iteration | B∞(d) *is* the background from step 0 | Redundant parameter set |
| Alternating medium / Gaussian optimization with freeze/unfreeze steps | Joint Adam, 500-step medium warm-up | Simpler, same physics |
| Depth-weighted L1 term added to the reconstruction loss | Removed | Counts the reconstruction twice |
| Smooth-depth loss weighted by the *signed* image gradient | Edge-aware with \|∇I\| on normalized disparity | Bug fix |
| SH degree 3 (48 floats per Gaussian) | SH degree 1 (12) | The medium explains the range-dependent appearance; 17 views cannot constrain degree-3 terms |
| — | **Multimodal depth priors**: dense depth on the water surface, SIFT keypoint depths on the wreck (both decaying) | Each modality supervises where it is reliable |
| — | Per-image affine colour correction, centred so test views use the average | Submerged3D frames vary in exposure and colour |
| gsplat 1.5.3 `DefaultStrategy` | Own strategy built on gsplat's ops: **opacity reset actually fires** (upstream `step % n == 0 & step > 0` never does), and water-origin Gaussians need 2× the gradient to densify | Fewer floaters in smooth water; precision on the wreck |

The medium is initialized from data. B∞ is the median open-water colour. β_B is fitted, Sea-thru style, to the darkest pixels in each range bin of the SIFT points. Gaussian colours start as the *inverted* medium model of the observed colours, so the very first render already matches the photos.

## Verified locally (CPU laptop, real data)

| Check | Result |
|---|---|
| SfM vs COLMAP 4.2 (same images, OPENCV model, exhaustive matching) | Same registrations (20/20/20 and 19/20 on Tokai, where both fail on the strobe-lit frame 2026). Lower reprojection error: **0.22–0.27 px** vs 0.26–0.39 px. More points. fx/fy within 0.3% (e.g. Kwaj 667.8/562.3 vs 667.4/562.3). |
| Bundle adjustment, synthetic | 20.75 px → 0.53 px (noise floor ≈ 0.6 px); k1/k2 recovered exactly |
| Dense network vs official DUSt3R | max \|Δ\| 2e-6 |
| Per-Gaussian medium == volumetric integral | 5.6e-16 over 1000 random rays |
| Dense depth vs SIFT depth (Isro) | Median 8.4% per pair → **6.3%** after fusion + 2 consensus passes. Inter-view surface dispersion 1.23% → 0.66% of depth. |
| Scale-only vs power-law vs affine alignment (held out, extrapolating across the image) | 12.9% / 13.5% / 17.2% median error → scale-only kept |
| Confidence gate (Isro) | Keeps 94% of SIFT-keypoint pixels, rejects 97% of open-water pixels |
| Colour-stretching the network input | No gain (7.8% vs 8.0%) → off by default |
| CPU smoke test (`tests/test_smoke_train.py`, 80×45, 120 iterations) | Train → densify → prune → opacity reset → evaluate (PSNR/SSIM/LPIPS) runs end-to-end |

No GPU was available here, so **the PSNR/SSIM/LPIPS of the real configuration (720p, 30k iterations) come from your A40 run**. The smoke-test numbers only show that the code path works. Each training run should take roughly 10–15 min on an A40; that is an estimate, not a measurement.

Tests: `python -m tests.test_bundle`, `python -m tests.test_medium_compositing`,
`python -m tests.test_dense_equivalence <dust3r_repo> <ckpt> <img1> <img2>`, and `python -m tests.test_smoke_train outputs/<scene>`.

## Outputs (`outputs/<scene>/`)

```
images/                     undistorted frames (training images + test GT)
sparse/0/                   COLMAP text model → 3DGS / RUSplatting run directly on our SfM (`-s outputs/<scene>`)
sfm/scene.json, points.npz  pinhole camera, poses, split, SIFT points + observations, SfM stats
dense/<frame>.npz, viz/     fused, consensus depth + confidence + consistency per training view
fusion/fused.ply            multimodal cloud (source 0 = SIFT/wreck, 1 = dense/water)
fusion/viz/, masks/         routing overlays (red wreck, green water surface, blue open water)
fusion/medium_init.json     data-driven B∞, β_B, β_D
gs/<variant>/               ckpt.pt, point_cloud.ply (standard 3DGS PLY, restored colours), train_log.jsonl
eval/<variant>/             metrics.json, renders/, gt/, restored/, depth/, montage/
```

## Variants (`--variants`)

| Variant | Initialization | Medium | Depth priors | Notes |
|---|---|---|---|---|
| `full` | SIFT + dense | ✓ | ✓ | the method |
| `sparse_init` | SIFT only | ✓ | ✓ | removes the dense water cloud |
| `dense_init` | dense only | ✓ | ✓ | removes the SIFT wreck cloud |
| `no_medium` | SIFT + dense | ✗ (learned background colour) | ✓ | |
| `no_depth_priors` | SIFT + dense | ✓ | ✗ | |
| `no_appearance` | SIFT + dense | ✓ | ✓ | no per-image colour correction |
| `baseline_3dgs` | SIFT only | ✗ | ✗ | SH degree 3; plain 3DGS on our SfM |

Reference averages on Submerged3D (reported by the papers, not re-run): RUSplatting paper 25.80 / 0.7724 / 0.3008; COLMAP→3DGS 23.48 / 0.7385 / 0.3333 at 720p (Caldeira et al. 2026). Your earlier RUSplatting Colab runs gave, for example, Tokai 24.38 / 0.7347 / 0.3523 and Cormoran 21.16 / 0.6788 / 0.3415 (base config).

## Notes

- The pointmap weights are DUSt3R's (CC BY-NC-SA 4.0): fine for university research, not for commercial use.
- The dense branch is zero-shot underwater (about 6% depth error against SfM). That is why it is gated, aligned and used only where SIFT has no support, rather than trusted everywhere. Fine-tuning it on underwater-degraded data, as Caldeira et al. do with VGGT, is the natural next step.
- Hyper-parameters are 3DGS/SeaSplat defaults plus a few chosen from the measurements above; they have not been tuned on the test views.
#   s p l a t _ d e n s e - s p a r s e  
 