"""Hand uwsplat's geometry to RUSplatting: our cameras + our point cloud, RUSplatting for everything else.

Builds a scene folder that RUSplatting (theflash987/RUSplatting, train.py -s <dst> --eval) reads unchanged:
  <dst>/images/<stem>.png         uwsplat's undistorted frames (symlinks, or copies with --copy)
  <dst>/sparse/0/cameras.txt      PINHOLE camera from our SfM (fx != fy is fine: FovX / FovY are separate)
  <dst>/sparse/0/images.txt       our poses
  <dst>/sparse/0/points3D.txt     our SIFT points (kept for reference)
  <dst>/sparse/0/points3D.ply     the INITIAL CLOUD RUSplatting will use (x y z nx ny nz red green blue):
                                  RUSplatting's dataset reader loads points3D.ply if it exists and only
                                  converts points3D.txt/.bin when it does not
  <dst>/test_list.txt             names of the 3 held-out views (for the --test_list option of our branch)

--cloud chooses the initial cloud:
  sift    our SIFT points only (control run: isolates "our SfM" from "our dense points")
  fusion  fusion/fused.ply  (SIFT wreck + DUSt3R water-surface points, v1)
  mvs     mvs/fused.ply     (SIFT + MVS-verified + completed points, v2)

RUSplatting also needs <dst>/depthmap/<stem>.png for every registered frame (its depth L1 loss reads them and
crashes if one is missing). Generate them on <dst>/images with the same Depth-Anything script used for the
earlier RUSplatting runs, so that only the geometry changes.

python scripts/export_rusplatting.py --scene_out outputs/Tokai --dst rus_scenes/Tokai_mvs --cloud mvs
"""
import argparse
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from uwsplat.common.io import load_json, read_ply  # noqa: E402

CLOUDS = {"fusion": "fusion/fused.ply", "mvs": "mvs/fused.ply"}


def read_points3d_txt(path):
    xyz, rgb = [], []
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            p = line.split()
            xyz.append([float(p[1]), float(p[2]), float(p[3])])
            rgb.append([int(p[4]), int(p[5]), int(p[6])])
    return np.array(xyz, np.float64).reshape(-1, 3), np.array(rgb, np.uint8).reshape(-1, 3)


def write_3dgs_ply(path, xyz, rgb, normals=None):
    """Exactly the layout of RUSplatting's storePly (binary little endian), readable by its fetchPly."""
    n = len(xyz)
    normals = np.zeros((n, 3), np.float32) if normals is None else normals
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
                   ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    v = np.empty(n, dt)
    for k, c in enumerate("xyz"):
        v[c] = xyz[:, k]
        v["n" + c] = normals[:, k]
    for k, c in enumerate(("red", "green", "blue")):
        v[c] = rgb[:, k]
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {n}\n" + "".join(f"property {'float' if dt[f].kind == 'f' else 'uchar'} {f}\n"
                                                 for f in dt.names) + "end_header\n")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(v.tobytes())


def load_cloud(scene_out, cloud):
    if cloud == "sift":
        xyz, rgb = read_points3d_txt(os.path.join(scene_out, "sparse", "0", "points3D.txt"))
        return xyz, rgb, None, {"sift": len(xyz)}
    ply = read_ply(os.path.join(scene_out, CLOUDS[cloud]))
    xyz = np.stack([ply["x"], ply["y"], ply["z"]], 1).astype(np.float64)
    rgb = np.stack([ply["red"], ply["green"], ply["blue"]], 1).astype(np.uint8)
    nrm = np.stack([ply["nx"], ply["ny"], ply["nz"]], 1).astype(np.float32) if "nx" in ply else None
    src = ply["source"]
    names = {"fusion": {0: "sift", 1: "dense"}, "mvs": {0: "sift", 1: "completed", 2: "verified"}}[cloud]
    return xyz, rgb, nrm, {names.get(int(s), str(s)): int((src == s).sum()) for s in np.unique(src)}


def rusplatting_test_names(registered_names, every=8):
    """RUSplatting's split: sort registered frames by name up to the first '.', every 8th from index 0."""
    stems = sorted(os.path.basename(n).split(".")[0] for n in registered_names)
    return [s for i, s in enumerate(stems) if i % every == 0]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene_out", required=True, help="uwsplat output folder of one scene (outputs/<scene>)")
    p.add_argument("--dst", required=True, help="RUSplatting scene folder to create")
    p.add_argument("--cloud", choices=("sift", "fusion", "mvs"), default="mvs")
    p.add_argument("--copy", action="store_true", help="copy the images instead of symlinking them")
    p.add_argument("--force", action="store_true", help="overwrite an existing --dst")
    a = p.parse_args()

    meta = load_json(os.path.join(a.scene_out, "sfm", "scene.json"))
    if os.path.exists(a.dst):
        if not a.force:
            sys.exit(f"{a.dst} exists (use --force)")
        shutil.rmtree(a.dst)
    sp = os.path.join(a.dst, "sparse", "0")
    os.makedirs(sp)
    os.makedirs(os.path.join(a.dst, "images"))

    for im in meta["images"]:
        src = os.path.abspath(os.path.join(a.scene_out, im["file"]))
        dst = os.path.join(a.dst, "images", os.path.basename(im["file"]))
        shutil.copy2(src, dst) if a.copy else os.symlink(src, dst)
    for f in ("cameras.txt", "images.txt", "points3D.txt"):
        shutil.copy2(os.path.join(a.scene_out, "sparse", "0", f), os.path.join(sp, f))

    xyz, rgb, nrm, counts = load_cloud(a.scene_out, a.cloud)
    write_3dgs_ply(os.path.join(sp, "points3D.ply"), xyz, rgb, nrm)

    ours = sorted(os.path.splitext(im["name"])[0] for im in meta["images"] if im["split"] == "test")
    reg = [im["file"] for im in meta["images"] if im["registered"]]
    theirs = rusplatting_test_names(reg, meta.get("test_every", 8))
    with open(os.path.join(a.dst, "test_list.txt"), "w") as f:
        f.write("\n".join(os.path.basename(n) for n in ours) + "\n")

    cam = meta["camera"]
    print(f"{meta['scene']}: {len(reg)}/{len(meta['images'])} registered, PINHOLE {cam['width']}x{cam['height']} "
          f"fx={cam['fx']:.1f} fy={cam['fy']:.1f}")
    print(f"initial cloud '{a.cloud}': {len(xyz)} points {counts}")
    print(f"test views (uwsplat): {ours}")
    if theirs != ours:
        print(f"WARNING: RUSplatting's own every-8th split would pick {theirs}; pass --test_list {a.dst}/test_list.txt")
    else:
        print("RUSplatting's every-8th split picks the same test views")
    print(f"next: depth maps into {a.dst}/depthmap/<stem>.png, then  python train.py -s {a.dst} -m <model> --eval")


if __name__ == "__main__":
    main()
