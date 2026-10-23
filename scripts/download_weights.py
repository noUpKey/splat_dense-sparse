"""Download the public pointmap weights used by the dense branch into weights/."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uwsplat.dense.weights import DEFAULT_NAME, DEFAULT_URL, download  # noqa: E402

if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = download(os.path.join(root, "weights", DEFAULT_NAME), DEFAULT_URL)
    print(path, os.path.getsize(path) // 2 ** 20, "MB")
