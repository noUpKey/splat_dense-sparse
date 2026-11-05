"""SeaThru-NeRF preparation: COLMAP model reading (text + binary), pinhole copy, 3DGS resize rule, split check,
and (if pycolmap is installed) the undistortion path.

python -m tests.test_prepare_seathru      (CPU, a few seconds)
"""
import importlib.util
import os
import struct
import sys
import tempfile

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

from prepare_seathru import discover_scenes, prepare_scene, threedgs_size  # noqa: E402
from uwsplat.common.colmap_io import read_cameras, read_image_names  # noqa: E402


def _write_text_model(d, model, params, w, h, names):
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "cameras.txt"), "w") as f:
        f.write("# Camera list\n")
        f.write(f"1 {model} {w} {h} {' '.join(str(p) for p in params)}\n")
    with open(os.path.join(d, "images.txt"), "w") as f:
        f.write("# Image list\n")
        for i, n in enumerate(names):
            f.write(f"{i + 1} 1 0 0 0 {0.1 * i} 0 0 1 {n}\n\n")   # empty 2D-point line
    open(os.path.join(d, "points3D.txt"), "w").close()


def _write_binary_model(d, w, h, names):
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "cameras.bin"), "wb") as f:
        f.write(struct.pack("<Q", 1) + struct.pack("<iiQQ", 1, 2, w, h) + struct.pack("<4d", 500.0, w / 2, h / 2, 0.01))
    with open(os.path.join(d, "images.bin"), "wb") as f:
        f.write(struct.pack("<Q", len(names)))
        for i, n in enumerate(names):
            f.write(struct.pack("<idddddddi", i + 1, 1, 0, 0, 0, 0.1 * i, 0, 0, 1) + n.encode() + b"\x00")
            f.write(struct.pack("<Q", 2) + struct.pack("<ddq", 1.0, 2.0, -1) * 2)


def _images(d, names, w, h, seed=0):
    os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(seed)
    for n in names:
        Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8)).save(os.path.join(d, n), quality=90)


def main():
    assert threedgs_size(1400, 900) == (1400, 900) and threedgs_size(2000, 1333) == (1600, 1066)
    with tempfile.TemporaryDirectory() as tmp:
        # readers: text and binary give the same answers
        names = [f"_DSC{k:04d}.JPG" for k in range(10)]
        _write_text_model(os.path.join(tmp, "t"), "PINHOLE", [500, 510, 32, 24], 64, 48, names)
        _write_binary_model(os.path.join(tmp, "b"), 64, 48, names)
        ct, cb = read_cameras(os.path.join(tmp, "t")), read_cameras(os.path.join(tmp, "b"))
        assert ct[1]["model"] == "PINHOLE" and ct[1]["params"] == [500, 510, 32, 24]
        assert cb[1] == {"model": "SIMPLE_RADIAL", "width": 64, "height": 48, "params": [500.0, 32.0, 24.0, 0.01]}
        assert read_image_names(os.path.join(tmp, "t")) == names == read_image_names(os.path.join(tmp, "b"))

        # pinhole dataset: model images copied bit for bit, an image outside the model left out, 3DGS split
        src = os.path.join(tmp, "SeathruNeRF_dataset", "Panama")
        _images(os.path.join(src, "images_wb"), names + ["extra.JPG"], 64, 48)
        _write_text_model(os.path.join(src, "sparse", "0"), "PINHOLE", [60, 60, 32, 24], 64, 48, names[::-1])
        assert discover_scenes(os.path.join(tmp, "SeathruNeRF_dataset")) == [src]
        info = prepare_scene(src, os.path.join(tmp, "out", "Panama"))
        out = os.path.join(tmp, "out", "Panama", "images")
        assert sorted(os.listdir(out)) == names and not info["distorted"] and info["not_in_model"] == ["extra.JPG"]
        assert all(open(os.path.join(out, n), "rb").read() == open(os.path.join(src, "images_wb", n), "rb").read()
                   for n in names)
        assert info["test_images"] == ["_DSC0000", "_DSC0008"] and (info["n_train"], info["n_test"]) == (8, 2)

        # wider than 1600 px: PIL resize to 1600 px width, saved losslessly
        big = os.path.join(tmp, "big", "Curasao")
        _images(os.path.join(big, "Images_wb"), names[:3], 2000, 1000, seed=1)
        _write_text_model(os.path.join(big, "colmap", "sparse", "0"), "SIMPLE_PINHOLE", [900, 1000, 500], 2000,
                          1000, names[:3])
        info = prepare_scene(big, os.path.join(tmp, "out", "Curasao"))
        im = np.asarray(Image.open(os.path.join(tmp, "out", "Curasao", "images", "_DSC0001.png")))
        ref = np.asarray(Image.open(os.path.join(big, "Images_wb", "_DSC0001.JPG")).convert("RGB").resize((1600, 800)))
        assert info["size"] == [1600, 800] and info["resized_to_max_width"] and np.array_equal(im, ref)

        # lens distortion: undistorted with COLMAP's undistorter (needs pycolmap)
        if importlib.util.find_spec("pycolmap") is None:
            print("pycolmap not installed: undistortion path not tested")
        else:
            dist = os.path.join(tmp, "dist", "IUI3-RedSea")
            _images(os.path.join(dist, "images_wb"), names[:4], 160, 120, seed=2)
            _write_text_model(os.path.join(dist, "sparse", "0"), "SIMPLE_RADIAL", [150, 80, 60, -0.15], 160, 120,
                              names[:4])
            info = prepare_scene(dist, os.path.join(tmp, "out", "IUI3-RedSea"))
            files = sorted(os.listdir(os.path.join(tmp, "out", "IUI3-RedSea", "images")))
            assert info["distorted"] and info["undistortion"] and files == names[:4], (info, files)
            assert not os.path.exists(os.path.join(tmp, "out", "IUI3-RedSea", "_undistorted"))
            print(f"undistorted 160x120 -> {info['size'][0]}x{info['size'][1]} ({info['undistortion']})")
    _pinhole_export()
    print("SeaThru-NeRF preparation test passed")


def _pinhole_export():
    """sfm.distortion=False -> k1 = k2 = 0 -> export_sfm keeps every image exactly as it is (no half-pixel zoom)."""
    import json

    from uwsplat.common.camera import DistortedCamera
    from uwsplat.sfm.export import export_sfm
    from uwsplat.sfm.reconstruct import SfMResult

    rng = np.random.default_rng(3)
    W, H, n = 40, 30, 3
    imgs = [rng.integers(0, 255, (H, W, 3), dtype=np.uint8) for _ in range(n)]
    X = rng.normal(0, 1, (20, 3)) + np.array([0, 0, 6.0])
    R, t = np.tile(np.eye(3), (n, 1, 1)), np.array([[-0.2 * i, 0, 0] for i in range(n)], float)
    cam = DistortedCamera(W, H, 50.0, 48.0, W / 2, H / 2)
    obs_img, obs_pt = np.repeat(np.arange(n), len(X)), np.tile(np.arange(len(X)), n)
    pc = X[obs_pt] + t[obs_img]
    uv = np.stack([cam.fx * pc[:, 0] / pc[:, 2] + cam.cx, cam.fy * pc[:, 1] / pc[:, 2] + cam.cy], 1)
    res = SfMResult([f"f{i}.png" for i in range(n)], cam, R, t, np.ones(n, bool), X, np.zeros((len(X), 3), np.uint8),
                    obs_img, obs_pt, uv, np.zeros((len(uv), 3), np.uint8))
    with tempfile.TemporaryDirectory() as d:
        meta = export_sfm(res, imgs, d, "pinhole")
        assert meta["max_distortion_px"] == 0.0 and meta["camera"]["fx"] == 50.0 and meta["camera"]["fy"] == 48.0
        for i in range(n):
            assert np.array_equal(np.asarray(Image.open(os.path.join(d, "images", f"f{i}.png"))), imgs[i])
        pts = np.load(os.path.join(d, "sfm", "points.npz"))
        assert np.allclose(pts["obs_uv"], uv, atol=1e-3)
        assert json.load(open(os.path.join(d, "sfm", "scene.json")))["camera"]["cx"] == W / 2


if __name__ == "__main__":
    main()
