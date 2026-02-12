from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
import scanpy as sc


def load_h5ad(path: str) -> sc.AnnData:
    return sc.read_h5ad(path)


def require_obs_cols(adata: sc.AnnData, cols: List[str], name: str):
    missing = [c for c in cols if c not in adata.obs.columns]
    if missing:
        raise ValueError(f"{name} is missing obs columns: {missing}")


def ensure_counts_layer(adata: sc.AnnData, layer: str = "counts", name: str = "adata"):
    if layer not in adata.layers:
        raise ValueError(f"{name} missing layers['{layer}'] (raw counts required for scVI).")
    X = adata.layers[layer]
    mn = X.min()
    if mn < 0:
        raise ValueError(f"{name} layers['{layer}'] has negative values (min={mn}). Not counts.")


def parse_time_to_hours(series: pd.Series) -> np.ndarray:
    """
    Accepts: 48, "48", "48h", etc -> float hours.
    """
    s = series.astype(str).str.replace("h", "", regex=False)
    return pd.to_numeric(s, errors="coerce").astype(float).to_numpy()


def align_genes(train: sc.AnnData, test: sc.AnnData) -> Tuple[sc.AnnData, sc.AnnData, List[str]]:
    common = train.var_names.intersection(test.var_names)
    if len(common) == 0:
        raise ValueError("No overlapping genes between train and test.")
    train2 = train[:, common].copy()
    test2  = test[:, common].copy()
    return train2, test2, list(common)
