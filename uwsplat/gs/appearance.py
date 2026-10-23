"""Per-training-image affine colour correction.

Submerged3D frames vary in exposure and colour (strobes, auto exposure). Each
training image gets I' = (Id + dA_i) I + db_i, with the parameters centred over
the training set so the *mean* correction is exactly the identity; test views,
whose correction is unknown, are rendered with the identity = average appearance.
"""
import torch
import torch.nn as nn


class Appearance(nn.Module):
    def __init__(self, n_views: int):
        super().__init__()
        self.A = nn.Parameter(torch.zeros(n_views, 3, 3))
        self.b = nn.Parameter(torch.zeros(n_views, 3))

    def _centred(self):
        return self.A - self.A.mean(0, keepdim=True), self.b - self.b.mean(0, keepdim=True)

    def forward(self, img: torch.Tensor, i: int) -> torch.Tensor:
        dA, db = self._centred()
        return img @ (torch.eye(3, device=img.device) + dA[i]).T + db[i]

    def regularizer(self) -> torch.Tensor:
        dA, db = self._centred()
        return (dA ** 2).mean() + (db ** 2).mean()
