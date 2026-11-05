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

## Dense cloud v2: MVS-verified geometry (`mvs` stage)

**Why.** What the earlier runs say about the initial geometry:

| Test | Result | Consequence for the cloud |
|---|---|---|
| uwsplat v1 vs RUSplatting (A40, Oct 23) | Isro: PSNR +0.84 dB, yet LPIPS 0.360 vs 0.251 and SSIM 0.839 vs 0.870; only 0.29–0.40M Gaussians on Isro / Kwaj | renders are too smooth: too few small Gaussians where the texture is |
| fusion v1 routing | wreck pixels get SIFT points only (13.8k on Isro, 16.1k on Tokai) | the most textured region starts the sparsest |
| DUSt3R depth vs SfM (Isro) | 6.3 % median error after fusion | texture lands several pixels off in neighbouring views |
| VGGT → 3DGS (A40, Oct 24) | 19.45 dB, SSIM 0.45 (Tokai 17.6 dB) | network geometry alone cannot place texture |
| zero-weight interpolated frames (RUSplatting) | +0.85 / +0.89 dB from extra COLMAP tracks only; LPIPS not better | triangulated points help, but they have to be dense too |
| train-vs-test gap, SH truncation | 11–20 dB gap; the degree-3 band memorises; each Gaussian's colour varies 10–14/255 across training views | geometry errors are absorbed as view-dependent colour, so geometry needs a prior that also holds in novel views |
| pruning / water tests | single-view Gaussians cover the never-seen strip; frustum pruning −1.0 dB; learned water +0.42 dB | add no point that two training views cannot verify; open water stays with the medium |

**What it does** (`uwsplat/mvs`, training views only):
1. Every textured pixel of every training view is *re-measured* by a plane sweep with the SfM poses: 4 source views, 7×7 zero-mean NCC, sub-sample peak. The network depth is only the centre of a ±22 % search band (`prior` pass); pixels without a usable prior get a sweep over the view's SIFT depth range (`wide` pass, the only pass without the dense stage). Hypotheses are spaced so the widest-baseline source moves ≤ 0.75 grid px per step.
2. **Water does not break the matching.** Within a window at one range, water acts as a gain (attenuation `e^(−β_D r)`) plus an offset (backscatter `B∞(1−e^(−β_B r))`), and exposure / white balance as another gain; zero-mean NCC cancels both. What water does destroy is contrast, so a texture gate (window std ≥ 2 grey levels) refuses open water.
3. Multi-view geometric consistency (1.5 %) between the independently measured depth maps, with consensus snapping.
4. Classes: **verified** (dense points, wreck included), **completed** (network points, exactly as v1, only where nothing was verified but v1 routing sees a surface), **open water** (no points, medium).
5. Outputs: `mvs/fused.ply` (`source` 0 SIFT / 1 network-only / 2 verified, NCC score, #views, surface normals), per-view depth priors (weight 1.0 verified, 0.3 network-only), wreck masks, medium start (copied from fusion, so v1 and v2 differ only in geometry), `viz/`, `summary.json`.

**Training options** (`TrainConfig`): `geometry="mvs"` (cloud + priors from `mvs/`), `init="verified"` (drop network-only points), `surfel_ratio` (Gaussians with a normal start as discs on the surface), `init_opacity_verified`, `scene_scale` (override of the 1.1 × camera-radius extent that scales the position lr and the densify / prune thresholds; scenes are in median-range units), and RUSplatting's edge-aware smoothness `w_esl` / `esl_beta` / `esl_detach`.

**RUSplatting's "edge smoothness"** (`gs/losses.py: esl_smoothness`) is ported from their code (`EdgeSmoothnessLoss(beta=5)`, weight 0.05; the paper writes λ_s = 0.2 and reports SSIM 0.7306 → 0.7491, PSNR 24.67 → 25.03). It smooths the **rendered image** wherever the **rendered depth** is smooth (weights `exp(−5 |∇D|)`, D = accumulated depth like their rasteriser's), so it denoises surfaces and stops at depth edges. It is a different term from uwsplat's existing edge-aware *depth* smoothness (`w_smooth`), which smooths depth except at image edges; both can run together. By default the weights are detached (`esl_detach=True`): in the original, creating depth jumps at texture edges also lowers the loss. `esl_detach=False` reproduces RUSplatting exactly.

**Training loss** (`gs/trainer.py`: every term goes into the dict `L`, then `loss = sum(L.values())`):

```
L = (0.8 L1 + 0.2 (1 - SSIM))(render, photo)                     render = Gaussians through the medium + water + appearance
  + d(t) (0.05 L_dense + 0.05 L_sparse)                          log-depth priors, d(t) = 0.1^(t/T)
  + 0.01 L_smooth + w_esl L_esl                                   depth smoothness / RUSplatting image smoothness (w_esl 0)
  + [t >= 3000] (0.01 L_grey + 0.1 L_sat)                         restored colour J
  + [t >= 500] (0.05 L_backscatter + 0.01 L_bg_opacity)           water parameters, open-water transparency
  + 0.1 L_appearance
  + [t >= lpips_from] min(1, (t - lpips_from + 1) / lpips_ramp) w_lpips LPIPS(render, photo)
```

The perceptual term is off by default (`w_lpips = 0`). The `*_lpips` variants switch it on with weight 0.05 at step 15000, which is when densification stops: from there it refines the Gaussians that exist without driving splits. It ramps in over 1000 steps, scores one random 512×512 crop per step, and uses a frozen LPIPS-VGG (`lpips_net="alex"` trains on a network other than the evaluation's). Starting it earlier (`--set train.lpips_from=10000`) lets its gradients also steer densification.

**Measured without a GPU** (real Submerged3D frames, our SfM, MVS grid 640×360, *no network prior*: the sandbox could not download the DUSt3R weights, so only the wide pass ran):

| | Isro | Tokai |
|---|---|---|
| verified pixels: all / wreck / off-wreck | 44 % / **87 %** / 19 % | 58 % / **85 %** / 40 % |
| points: SIFT + verified | 13,777 + 591,928 | 16,149 + 562,345 |
| depth vs SIFT triangulations (median / p90; within 1 %) | **0.44 %** / 1.65 %; 79 % | **0.21 %** / 0.60 %; 97 % |
| held-out test views, mean NCC on wreck pixels: SIFT-interpolated / verified | 0.819 / **0.850** | 0.865 / **0.876** |
| held-out NCC, verified with a smooth ±6 % (network-sized) error / exact | 0.627 / 0.847 | 0.274 / 0.859 |
| time on 2 CPU cores | 6.8 min | 10.7 min |

`scripts/heldout_geometry.py` produces the held-out rows: test images are warped into their nearest training views with each candidate depth map and compared by NCC (scoring only, like PSNR). On a synthetic underwater plane with range-dependent attenuation and backscatter, ±15 % exposure changes and noise, the sweep recovers depth to 0.13 % median (p90 0.32 %) and verifies none of the flat water-like region (`tests/test_mvs.py`).

**On the A40** (stage outputs are cached, the `mvs` stage runs once per scene):
```bash
python scripts/run_all.py --data Submerged3D --out outputs --variants full,mvs \
       --set train.scene_scale=0.8                                  # v1 vs v2 geometry, all else equal
python scripts/heldout_geometry.py --out outputs/Tokai                # adds the real network-depth row
python scripts/run_all.py --data Submerged3D --out outputs \
       --variants mvs_prior,mvs_surfel,mvs_esl,full_esl,mvs_all --set train.scene_scale=0.8
```

## Held-out view check: test-view colour correction (`scripts/heldout_appearance.py`)

Training fits an affine colour correction per training image (exposure and white balance change by 6–16 % between
consecutive Submerged3D frames); test views are rendered with the average correction (`identity`), which is what
`metrics.json` reports. The check re-scores a trained run without retraining under other choices, none of which uses
test pixels: `neighbors` (interpolated in time between the training frames before and after the test frame; the
nearest one at a sequence end) and `nearest_pose` (the nearest training camera). `oracle` fits the correction to the
test image itself: an upper bound on what any global colour correction could gain, not a result. The same predictions
are scored leave-one-out on the training views, where the true correction is known. Each test pixel is also marked
seen / never seen (its rendered surface point, or its direction where the render is open water, lands inside some
training image; occlusion ignored), and PSNR is split accordingly.

```bash
python scripts/heldout_appearance.py --out outputs --scenes Tokai,Cormoran,Isro,Kwaj \
       --variants mvs,mvs_lpips --per-view mvs_lpips      # -> eval/<variant>/heldout_appearance.json + table
```

## Interpolated frames as extra geometry views (`scripts/make_interframes.py`)

`python scripts/make_interframes.py --src Submerged3D --out Submerged3D_ifi --method raft` copies the real frames and
adds one optical-flow midpoint frame (`<stem>__ifi.png`) between every two consecutive frames that are both training
frames (never next to a test frame). In the pipeline these frames get the split `aux`: the SfM registers them (more
tracks), the MVS stage may pick them as source views, nothing trains on or scores them, and a point still needs two
real training observations. Run the whole pipeline on `Submerged3D_ifi` into a separate output folder.
CPU check on Isro (wide pass only): 34/34 registered, SIFT points 14.3k vs 13.8k, verified points 646k vs 592k,
verified pixels 43 % vs 44 %, depth vs SIFT 0.47 % vs 0.44 % median: the MVS rarely picks the narrow-baseline frames.

## Forward scattering (`fs` component, `uwsplat/gs/scatter.py`)

Small-angle scattering blurs a surface's image by an angle that grows with the path length r. Each Gaussian keeps its
sharp shape; for every view the renderer widens it in 3D by b = kappa * r * z (so the image blur is kappa * r radians)
and scales its opacity by sqrt(det Sigma_2D / det Sigma_2D') so its integrated footprint is unchanged. kappa is one
learned scalar per scene (in the medium's optimizer and checkpoint). No CUDA change. `--variants mvs_lpips+fs`
(blur from step 0) or `mvs_lpips+fs3k` (from step 3000); `fs_kappa` is logged in train_log.jsonl and final.json.
`tests/test_scatter.py`: no-op at kappa -> 0, summed alpha preserved (0.96 with the 1/255 alpha cut-off), angular
blur proportional to range, gradients reach kappa.

## SeaThru-NeRF (the data SeaSplat evaluates on): `scripts/prepare_seathru.py`

Prepares the SeaThru-NeRF scenes the way 3DGS-based methods receive them (SeaSplat: undistorted images + `sparse/0`):
the images of the dataset's own COLMAP model, from `images_wb/`; undistorted with COLMAP's undistorter (pycolmap,
default options, JPEG quality 100) only if that model has lens distortion; scaled to 1600 px width if wider (3DGS
default); every 8th image by name held out from index 0. `prepare.json` records the camera model, sizes and split.
Our pipeline then runs with `sfm.distortion=False`: its SfM keeps the pinhole model and the images reach training and
evaluation unchanged (bit-identical, checked on 20 real frames).

```bash
pip install pycolmap
python scripts/prepare_seathru.py --src data/SeathruNeRF_dataset --out data/SeaThru
python scripts/run_all.py --data data/SeaThru --out outputs_seathru --variants mvs_lpips \
       --set train.scene_scale=0.8 --set sfm.distortion=False
```

SeaSplat's Table I (PSNR / SSIM / LPIPS): Curaçao 30.30 / 0.90 / 0.19, Japanese Gardens 22.70 / 0.87 / 0.18, Panama
28.76 / 0.90 / 0.15, IUI3 26.67 / 0.87 / 0.21 (mean 27.11 / 0.885 / 0.182); 3DGS there 25.06 / 0.860 / 0.222.

## Outputs (`outputs/<scene>/`)

```
images/                     undistorted frames (training images + test GT)
sparse/0/                   COLMAP text model → 3DGS / RUSplatting run directly on our SfM (`-s outputs/<scene>`)
sfm/scene.json, points.npz  pinhole camera, poses, split, SIFT points + observations, SfM stats
dense/<frame>.npz, viz/     fused, consensus depth + confidence + consistency per training view
fusion/fused.ply            multimodal cloud (source 0 = SIFT/wreck, 1 = dense/water)
fusion/viz/, masks/         routing overlays (red wreck, green water surface, blue open water)
fusion/medium_init.json     data-driven B∞, β_B, β_D
mvs/fused.ply               dense cloud v2 (source 0 = SIFT, 1 = network-only, 2 = MVS-verified; normals)
mvs/priors/, masks/, viz/   verified depth priors, wreck masks, class overlays (red verified, green network, blue water)
gs/<variant>/               ckpt.pt, point_cloud.ply (standard 3DGS PLY, restored colours), train_log.jsonl
eval/<variant>/             metrics.json, renders/, gt/, restored/, depth/, montage/, heldout_appearance.json
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
| `full_esl` | SIFT + dense | ✓ | ✓ | + RUSplatting edge-aware image smoothness (`w_esl` 0.05) |
| `mvs` | SIFT + MVS-verified + network-only (v2) | ✓ | ✓ (verified depth, also on the wreck) | v2 geometry, same loss weights as `full` |
| `mvs_verified` | SIFT + MVS-verified | ✓ | ✓ | v2 without the network-only points |
| `mvs_prior` | v2 | ✓ | ✓ stronger (0.1, decays to 30 %) | trusts the verified depth for longer |
| `mvs_surfel` | v2, discs on the surface | ✓ | ✓ | `surfel_ratio` 0.2 |
| `mvs_esl` | v2 | ✓ | ✓ | + RUSplatting edge-aware image smoothness |
| `mvs_all` | v2, discs | ✓ | ✓ stronger | prior + surfel + ESL |
| `full_lpips` | SIFT + dense | ✓ | ✓ | + LPIPS-VGG loss (0.05) from step 15000 |
| `mvs_lpips` | v2 | ✓ | ✓ | + LPIPS-VGG loss (0.05) from step 15000 |

Components, combined with `+` (left to right; later keys win, `densify` settings merge), e.g.
`--variants mvs_lpips+abs,mvs_lpips+prior`. The tag, `+` included, is the run's folder name; a new combination needs
no code change.

| Component | Change |
|---|---|
| `abs` | AbsGS densification: split / clone by absolute screen-space gradients, threshold 0.0008 (gsplat's advice for absgrad) |
| `prior` | verified depth prior 0.1, decaying to 30 % (as `mvs_prior`) |
| `surfel` | `surfel_ratio` 0.2 (as `mvs_surfel`) |
| `lpips`, `lpipsx2`, `lpips10k` | LPIPS loss 0.05 from 15k / 0.1 from 15k / 0.05 from 10k |
| `esl`, `sh3` | RUSplatting ESL 0.05 / SH degree 3 |

Reference averages on Submerged3D (reported by the papers, not re-run): RUSplatting paper 25.80 / 0.7724 / 0.3008; COLMAP→3DGS 23.48 / 0.7385 / 0.3333 at 720p (Caldeira et al. 2026). Your earlier RUSplatting Colab runs gave, for example, Tokai 24.38 / 0.7347 / 0.3523 and Cormoran 21.16 / 0.6788 / 0.3415 (base config).

## Notes

- The pointmap weights are DUSt3R's (CC BY-NC-SA 4.0): fine for university research, not for commercial use.
- The dense branch is zero-shot underwater (about 6% depth error against SfM). That is why it is gated, aligned and used only where SIFT has no support, rather than trusted everywhere. Fine-tuning it on underwater-degraded data, as Caldeira et al. do with VGGT, is the natural next step.
- Hyper-parameters are 3DGS/SeaSplat defaults plus a few chosen from the measurements above; they have not been tuned on the test views.
#   s p l a t _ d e n s e - s p a r s e  
 
## Selective regularisers (train / test gap)

`mvs_lpips` reaches ~35.6 dB on its training views and ~22.6 dB on held-out views (Tokai): it overfits the
training cameras. Two regularisers act only where that happens (`python -m tests.test_pseudo`):

- `pv` (`w_pv=0.05`, `gs/pseudo.py`): every `pv_every` steps a camera is interpolated between two consecutive
  training cameras (`t ~ U(-pv_extrap, 1 + pv_extrap)`), its depth is rendered at `pv_scale` resolution and compared
  (|log ratio|) with the z-buffered MVS-verified points seen from there. Only verified, front-most points, not hidden
  behind a nearer surface, on pixels the render covers; no photo is compared, so only geometry is constrained.
  Needs `geometry="mvs"`.
- `needle` (`w_needle=0.1`): only Gaussians whose longest axis exceeds `needle_ratio` (10) x their middle axis are
  penalised; spheres and discs get no gradient.

`python scripts/run_all.py ... --variants mvs_lpips+pv,mvs_lpips+needle`. The training log now prints a running
average of the loss next to the single-step value, and `train_log.jsonl` records the view of each logged step.
