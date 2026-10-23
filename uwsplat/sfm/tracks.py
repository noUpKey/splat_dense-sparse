"""Feature tracks = connected components of the verified match graph."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


class Tracks:
    """CSR storage of tracks: observations of track t are obs_*[ptr[t]:ptr[t+1]].

    Components that contain two keypoints of the same image are inconsistent
    (a chain of matches drifted) and are discarded.
    """

    def __init__(self, n_kps: list, two_views: list):
        n_img = len(n_kps)
        offsets = np.concatenate([[0], np.cumsum(n_kps)]).astype(np.int64)
        n_nodes = int(offsets[-1])
        rows = [offsets[tv.i] + tv.idx_i for tv in two_views]
        cols = [offsets[tv.j] + tv.idx_j for tv in two_views]
        rows = np.concatenate(rows) if rows else np.zeros(0, np.int64)
        cols = np.concatenate(cols) if cols else np.zeros(0, np.int64)
        graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_nodes, n_nodes))
        _, labels = connected_components(graph, directed=False)
        img_of = np.repeat(np.arange(n_img), n_kps)
        kp_of = np.arange(n_nodes) - offsets[img_of]

        size = np.bincount(labels)
        keep = size[labels] >= 2
        key = labels.astype(np.int64) * n_img + img_of
        uniq, cnt = np.unique(key[keep], return_counts=True)
        bad = np.unique(uniq[cnt > 1] // n_img)
        self.n_conflicting = len(bad)
        keep &= ~np.isin(labels, bad)

        _, tid = np.unique(labels[keep], return_inverse=True)
        order = np.argsort(tid, kind="stable")
        self.obs_track = tid[order]
        self.obs_img = img_of[keep][order]
        self.obs_kp = kp_of[keep][order]
        self.n = int(tid.max()) + 1 if len(tid) else 0
        self.ptr = np.concatenate([[0], np.cumsum(np.bincount(self.obs_track, minlength=self.n))])
        self.length = np.diff(self.ptr)
        self.kp_track = [np.full(n, -1, np.int64) for n in n_kps]
        for o in range(len(self.obs_img)):
            self.kp_track[self.obs_img[o]][self.obs_kp[o]] = self.obs_track[o]

    def __len__(self):
        return self.n
