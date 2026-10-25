"""Minimal reader for COLMAP sparse models (cameras and image names, text or binary).

Enough to inspect a dataset's own model (camera model, image size, which images it contains) without pycolmap.
Formats as in COLMAP's scripts/python/read_write_model.py.
"""
import os
import struct

# model id -> (name, number of parameters)
CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4), 3: ("RADIAL", 5), 4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8), 6: ("FULL_OPENCV", 12), 7: ("FOV", 5), 8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5), 10: ("THIN_PRISM_FISHEYE", 12),
}
PINHOLE_MODELS = ("SIMPLE_PINHOLE", "PINHOLE")


def find_model(model_dir: str, stem: str):
    """Path of <stem>.bin or <stem>.txt in model_dir (binary preferred), or None."""
    for ext in (".bin", ".txt"):
        p = os.path.join(model_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None


def read_cameras(model_dir: str) -> dict:
    """{camera_id: {"model": name, "width": w, "height": h, "params": [...]}}"""
    path = find_model(model_dir, "cameras")
    if path is None:
        raise FileNotFoundError(f"no cameras.bin / cameras.txt in {model_dir}")
    cams = {}
    if path.endswith(".txt"):
        for line in open(path):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            e = line.split()
            cams[int(e[0])] = {"model": e[1], "width": int(e[2]), "height": int(e[3]),
                               "params": [float(x) for x in e[4:]]}
        return cams
    with open(path, "rb") as f:
        n, = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            cam_id, model_id, w, h = struct.unpack("<iiQQ", f.read(24))
            name, n_params = CAMERA_MODELS[model_id]
            params = struct.unpack("<" + "d" * n_params, f.read(8 * n_params))
            cams[cam_id] = {"model": name, "width": int(w), "height": int(h), "params": list(params)}
    return cams


def read_image_names(model_dir: str) -> list:
    """Names of the images in the model (registered images), in file order."""
    path = find_model(model_dir, "images")
    if path is None:
        raise FileNotFoundError(f"no images.bin / images.txt in {model_dir}")
    names = []
    if path.endswith(".txt"):
        with open(path) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                if line and not line.startswith("#"):
                    names.append(line.split()[9])
                    f.readline()  # the 2D points of this image (may be an empty line)
        return names
    with open(path, "rb") as f:
        n, = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            f.read(64)  # image_id (i), qvec (4d), tvec (3d), camera_id (i)
            name = b""
            while True:
                c = f.read(1)
                if c in (b"\x00", b""):
                    break
                name += c
            names.append(name.decode("utf-8"))
            n_pts, = struct.unpack("<Q", f.read(8))
            f.seek(24 * n_pts, os.SEEK_CUR)  # x (d), y (d), point3D_id (q) per 2D point
    return names
