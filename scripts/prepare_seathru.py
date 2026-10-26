"""Prepare SeaThru-NeRF scenes the way 3DGS-based methods (SeaSplat) receive them, as input for our pipeline.

python scripts/prepare_seathru.py --src data/SeathruNeRF_dataset --out data/SeaThru

For every scene under --src (a folder with images_wb/ and the dataset's COLMAP model, e.g. sparse/0):
  * the images are the ones in the dataset's COLMAP model (as 3DGS loads them), taken from images_wb/;
  * if that model's camera has lens distortion, the images are undistorted with COLMAP's image undistorter
    (pycolmap, default options, JPEG quality 100): SeaSplat's input format is undistorted images + sparse/0, and
    WaterSplatting's README undistorts SeaThru-NeRF the same way; pinhole models are used as they are (files copied);
  * images wider than 1600 px are scaled to 1600 px width like 3DGS's default (-r -1: int truncation, PIL bicubic).
Writes <out>/<scene>/images/ (the training images and the test ground truth of our pipeline) and
<out>/<scene>/prepare.json (camera model, sizes, images, the every-8th test split by name as in 3DGS).
Run our pipeline on <out> with --set sfm.distortion=False, so that its SfM treats the images as pinhole and the
evaluation scores exactly these pixels.
"""
import argparse
import json
import os
import shutil
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uwsplat.common.colmap_io import PINHOLE_MODELS, find_model, read_cameras, read_image_names  # noqa: E402

MAX_WIDTH = 1600
IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def find_images_dir(scene_dir: str):
    subs = {d.lower(): d for d in os.listdir(scene_dir) if os.path.isdir(os.path.join(scene_dir, d))}
    for key in ("images_wb", "images"):
        if key in subs:
            return os.path.join(scene_dir, subs[key])
    return None


def find_model_dir(scene_dir: str):
    for rel in (("sparse", "0"), ("sparse",), ("colmap", "sparse", "0"), ("colmap", "sparse")):
        d = os.path.join(scene_dir, *rel)
        if find_model(d, "cameras") and find_model(d, "images"):
            return d
    for root, _, files in os.walk(scene_dir):
        if {"cameras.bin", "cameras.txt"} & set(files) and {"images.bin", "images.txt"} & set(files):
            return root
    return None


def discover_scenes(src: str, max_depth: int = 3) -> list:
    """Folders under src (or src itself) that contain images_wb/ (or images/) and a COLMAP model."""
    found = []
    for root, dirs, _ in os.walk(src):
        depth = os.path.relpath(root, src).count(os.sep) + (root != src)
        if depth > max_depth:
            dirs[:] = []
            continue
        if any(d.lower() in ("images_wb", "images") for d in dirs) and find_model_dir(root):
            found.append(root)
            dirs[:] = []  # a scene's own subfolders are not scenes
    return sorted(found)


def threedgs_size(w: int, h: int, max_width: int = MAX_WIDTH):
    """3DGS loadCam, resolution -1: wider than max_width -> scaled to max_width (int truncation)."""
    if w <= max_width:
        return w, h
    scale = float(w / max_width)
    return int(w / scale), int(h / scale)


def threedgs_name(path: str) -> str:
    """3DGS's image_name (the sort key of its train / test split)."""
    return os.path.basename(path).split(".")[0]


def undistort(model_dir: str, image_dir: str, out_dir: str):
    import pycolmap
    kwargs = {"jpeg_quality": 100} if "jpeg_quality" in (pycolmap.undistort_images.__doc__ or "") else {}
    pycolmap.undistort_images(out_dir, model_dir, image_dir, **kwargs)
    return os.path.join(out_dir, "images"), f"pycolmap {pycolmap.__version__} undistort_images {kwargs}"


def prepare_scene(scene_dir: str, out_dir: str, max_width: int = MAX_WIDTH, every: int = 8) -> dict:
    img_dir, model_dir = find_images_dir(scene_dir), find_model_dir(scene_dir)
    if img_dir is None or model_dir is None:
        raise FileNotFoundError(f"{scene_dir}: need images_wb/ (or images/) and a COLMAP model (sparse/0)")
    cams = read_cameras(model_dir)
    names = [os.path.basename(n) for n in read_image_names(model_dir)]
    models = sorted({c["model"] for c in cams.values()})
    distorted = any(m not in PINHOLE_MODELS for m in models)
    on_disk = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(IMAGE_EXTS))
    missing = sorted(set(names) - set(on_disk))
    if missing:
        raise FileNotFoundError(f"{scene_dir}: {len(missing)} model images not in {img_dir}, e.g. {missing[:3]}")
    os.makedirs(out_dir, exist_ok=True)
    src_dir, tool, tmp = img_dir, None, None
    if distorted:
        tmp = os.path.join(out_dir, "_undistorted")
        shutil.rmtree(tmp, ignore_errors=True)
        src_dir, tool = undistort(model_dir, img_dir, tmp)
    dst = os.path.join(out_dir, "images")
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(dst)
    sizes_in, sizes_out, files = set(), set(), []
    for n in sorted(names, key=threedgs_name):
        p = os.path.join(src_dir, n)
        with Image.open(p) as im:
            size_in = im.size
            size = threedgs_size(*size_in, max_width)
            if size != size_in:
                fname = os.path.splitext(n)[0] + ".png"
                im.convert("RGB").resize(size).save(os.path.join(dst, fname))  # PIL default filter, as 3DGS
            else:
                fname = n
                shutil.copy2(p, os.path.join(dst, fname))
        sizes_in.add(size_in)
        sizes_out.add(size)
        files.append(fname)
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    if len(sizes_out) != 1:
        raise ValueError(f"{scene_dir}: images of different sizes {sorted(sizes_out)}; our SfM assumes one camera")
    order = sorted(files, key=threedgs_name)
    test = [threedgs_name(f) for k, f in enumerate(order) if k % every == 0]
    ours = [threedgs_name(f) for k, f in enumerate(sorted(files)) if k % every == 0]  # our pipeline sorts file names
    if ours != test:
        raise ValueError(f"{scene_dir}: file-name order differs from 3DGS's image_name order, the splits would differ")
    info = {
        "scene": os.path.basename(os.path.normpath(scene_dir)), "source_images": img_dir, "colmap_model": model_dir,
        "camera_models": models, "camera_params": {str(k): c["params"] for k, c in cams.items()},
        "distorted": distorted, "undistortion": tool, "size_in": sorted(sizes_in), "size": list(sizes_out.pop()),
        "resized_to_max_width": any(s[0] > max_width for s in sizes_in), "n_images": len(files),
        "not_in_model": sorted(set(on_disk) - set(names)), "test_every": every, "test_images": test,
        "n_train": len(files) - len(test), "n_test": len(test),
    }
    with open(os.path.join(out_dir, "prepare.json"), "w") as f:
        json.dump(info, f, indent=1)
    return info


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="extracted dataset (folder with the scene folders, or one scene)")
    p.add_argument("--out", required=True, help="output root: <out>/<scene>/images/")
    p.add_argument("--scenes", default="", help="comma list of scene folder names (default: all found)")
    p.add_argument("--max-width", type=int, default=MAX_WIDTH)
    a = p.parse_args()
    scenes = discover_scenes(a.src)
    if a.scenes:
        want = a.scenes.split(",")
        scenes = [s for s in scenes if os.path.basename(s) in want]
    if not scenes:
        raise SystemExit(f"no scene with images_wb/ (or images/) and a COLMAP model under {a.src}")
    for s in scenes:
        info = prepare_scene(s, os.path.join(a.out, os.path.basename(s)), a.max_width)
        print(f"{info['scene']:24} {info['n_images']:3d} images ({info['n_train']} train / {info['n_test']} test), "
              f"{info['size'][0]}x{info['size'][1]}, camera {'/'.join(info['camera_models'])}"
              f"{', undistorted' if info['distorted'] else ''}{', resized' if info['resized_to_max_width'] else ''}"
              f"{', not in the model: ' + str(len(info['not_in_model'])) if info['not_in_model'] else ''}; "
              f"test {', '.join(info['test_images'])}", flush=True)


if __name__ == "__main__":
    main()
