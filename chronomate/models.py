import math
from typing import Optional

import torch
import torch.nn as nn


class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class FeatureExtractor(nn.Module):
    def __init__(self, in_dim: int, n_ctypes: Optional[int] = None, ct_emb_dim: int = 10):
        super().__init__()
        self.has_ct = n_ctypes is not None and n_ctypes > 0
        self.ct_emb = nn.Embedding(n_ctypes + 1, ct_emb_dim) if self.has_ct else None  # last idx for unknown
        feat_in = in_dim + (ct_emb_dim if self.has_ct else 0)
        self.net = nn.Sequential(
            nn.Linear(feat_in, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(256),
        )

    def forward(self, x: torch.Tensor, ct_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.has_ct:
            if ct_idx is None:
                ct_idx = torch.full((x.size(0),), self.ct_emb.num_embeddings - 1, dtype=torch.long, device=x.device)
            ct = self.ct_emb(ct_idx)
            x = torch.cat([x, ct], dim=1)
        return self.net(x)


class DANN(nn.Module):
    def __init__(self, in_dim: int, n_ctypes: Optional[int] = None, ct_emb_dim: int = 10):
        super().__init__()
        self.fe = FeatureExtractor(in_dim, n_ctypes, ct_emb_dim)
        self.reg_head = nn.Linear(256, 1)
        self.dom_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x, ct_idx=None, alpha: float = 1.0):
        z = self.fe(x, ct_idx)
        y = self.reg_head(z).squeeze(1)
        z_rev = GradientReversalFn.apply(z, alpha if alpha is not None else 0.0)
        d = self.dom_head(z_rev).squeeze(1)
        return y, d


def dann_alpha(progress: float) -> float:
    p = min(max(progress, 0.0), 1.0)
    return 2.0 / (1.0 + math.exp(-10 * p)) - 1.0


class XGBWrapper:
    def __init__(self, **kwargs):
        from xgboost import XGBRegressor
        defaults = dict(n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, n_jobs=8)
        defaults.update(kwargs)
        self.model = XGBRegressor(**defaults)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path: str):
        self.model.save_model(path)

    def load(self, path: str):
        from xgboost import XGBRegressor
        self.model = XGBRegressor()
        self.model.load_model(path)
