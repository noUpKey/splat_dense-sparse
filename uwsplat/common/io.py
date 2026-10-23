"""Image, JSON and PLY I/O plus a tiny timestamped logger."""
import json
import os
import sys
import time

import numpy as np
from PIL import Image

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
_T0 = time.time()


def log(*args):
    print(f"[{time.time() - _T0:8.1f}s]", *args, flush=True)
    sys.stdout.flush()


def list_scene_images(scene_dir: str) -> list:
    """Sorted image paths of a Submerged3D-style scene (<scene>/input or <scene>/images)."""
    for sub in ("input", "images", ""):
        d = os.path.join(scene_dir, sub)
        if os.path.isdir(d):
            files = sorted(f for f in os.listdir(d) if f.endswith(IMAGE_EXTS))
            if files:
                return [os.path.join(d, f) for f in files]
    raise FileNotFoundError(f"no images found under {scene_dir}")


def load_rgb(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def save_rgb(path: str, img: np.ndarray):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    Image.fromarray(img).save(path)


def save_json(path: str, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def load_json(path: str):
    with open(path) as f:
        return json.load(f)


_PLY_TYPES = {np.dtype(np.float32): "float", np.dtype(np.uint8): "uchar", np.dtype(np.int32): "int"}


def write_ply(path: str, xyz: np.ndarray, rgb: np.ndarray = None, **fields):
    """Binary little-endian PLY with xyz, optional rgb (uint8) and extra scalar fields."""
    cols = [("x", xyz[:, 0].astype(np.float32)), ("y", xyz[:, 1].astype(np.float32)),
            ("z", xyz[:, 2].astype(np.float32))]
    if rgb is not None:
        rgb = np.asarray(rgb)
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
        cols += [("red", rgb[:, 0]), ("green", rgb[:, 1]), ("blue", rgb[:, 2])]
    for k, v in fields.items():
        v = np.asarray(v)
        v = v.astype(np.float32) if v.dtype.kind == "f" else v.astype(np.uint8 if v.max(initial=0) < 256 and v.min(initial=0) >= 0 else np.int32)
        cols.append((k, v))
    dtype = np.dtype([(n, c.dtype) for n, c in cols])
    data = np.empty(len(xyz), dtype=dtype)
    for n, c in cols:
        data[n] = c
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(xyz)}"]
    header += [f"property {_PLY_TYPES[c.dtype]} {n}" for n, c in cols]
    header.append("end_header")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        f.write(data.tobytes())


def read_ply(path: str) -> dict:
    """Reads PLY files written by `write_ply` (binary little-endian, scalar vertex properties)."""
    inv = {"float": np.float32, "uchar": np.uint8, "int": np.int32, "double": np.float64}
    with open(path, "rb") as f:
        props, n = [], 0
        while True:
            line = f.readline().decode("ascii").strip()
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property"):
                _, t, name = line.split()
                props.append((name, inv[t]))
            elif line == "end_header":
                break
        data = np.frombuffer(f.read(), dtype=np.dtype(props), count=n)
    return {name: np.array(data[name]) for name, _ in props}
