import os
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any
from sklearn.preprocessing import StandardScaler
import joblib

try:
    import anndata as ad
except Exception:
    ad = None


def _read_h5ad(path: str) -> Tuple[np.ndarray, pd.DataFrame, pd.Index]:
    if ad is None:
        raise RuntimeError("Install `anndata` to read .h5ad files.")
    A = ad.read_h5ad(path)
    X = A.X.A if hasattr(A.X, "A") else A.X
    if not isinstance(X, np.ndarray):
        X = X.toarray()
    obs = A.obs.copy()
    names = A.obs_names.copy()
    return X.astype(np.float32), obs, names


def _read_csv(path: str) -> Tuple[np.ndarray, pd.DataFrame, pd.Index]:
    df = pd.read_csv(path)
    meta_cols = [c for c in ["time", "cell_type"] if c in df.columns]
    gene_cols = [c for c in df.columns if c not in meta_cols]
    X = df[gene_cols].to_numpy(np.float32)
    obs = pd.DataFrame(index=df.index)
    if "time" in df.columns:
        obs["time"] = df["time"].values
    if "cell_type" in df.columns:
        obs["cell_type"] = df["cell_type"].astype(str).values
    return X, obs, pd.Index(df.index.astype(str))


def load_matrix(
    path: str,
    obs_time_key: Optional[str] = None,
    obs_celltype_key: Optional[str] = None,
) -> Dict[str, Any]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".h5ad":
        X, obs, names = _read_h5ad(path)
        times = obs[obs_time_key].to_numpy(np.float32) if (obs_time_key and obs_time_key in obs) else None
        ctypes = obs[obs_celltype_key].astype(str).to_numpy() if (obs_celltype_key and obs_celltype_key in obs) else None
    elif ext == ".csv":
        X, obs, names = _read_csv(path)
        times = obs["time"].to_numpy(np.float32) if "time" in obs.columns else None
        key = obs_celltype_key if obs_celltype_key else ("cell_type" if "cell_type" in obs.columns else None)
        ctypes = obs[key].astype(str).to_numpy() if key else None
    else:
        raise ValueError(f"Unsupported file: {path}")
    return {"X": X, "times": times, "cell_types": ctypes, "names": names}


def fit_scaler_on_source(X_source: np.ndarray, path: str) -> StandardScaler:
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(X_source)
    joblib.dump(scaler, path)
    return scaler


def load_scaler(path: str) -> StandardScaler:
    return joblib.load(path)


def encode_cell_types(ctypes: Optional[np.ndarray]) -> Tuple[Optional[np.ndarray], Dict[str, int]]:
    if ctypes is None:
        return None, {}
    uniq = sorted(pd.unique(ctypes.astype(str)))
    mapping = {k: i for i, k in enumerate(uniq)}
    enc = np.array([mapping[c] for c in ctypes], dtype=np.int64)
    return enc, mapping


def apply_cell_type_encoder(ctypes: Optional[np.ndarray], mapping: Dict[str, int]) -> Optional[np.ndarray]:
    if ctypes is None:
        return None
    unk = max(mapping.values()) + 1 if mapping else 0
    return np.array([mapping.get(str(c), unk) for c in ctypes], dtype=np.int64)


class SingleCellDataset:
    def __init__(
        self,
        X: np.ndarray,
        times: Optional[np.ndarray],
        ct_encoded: Optional[np.ndarray],
        names: Optional[pd.Index] = None,
    ):
        self.X = X
        self.times = times
        self.ct = ct_encoded
        self.names = names if names is not None else pd.Index([str(i) for i in range(len(X))])

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        x = self.X[idx]
        c = -1 if self.ct is None else int(self.ct[idx])
        if self.times is None:
            t = np.nan
        else:
            t = float(self.times[idx])
        return x.astype(np.float32), c, t, self.names[idx]
