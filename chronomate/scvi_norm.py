from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scvi

from .io import ensure_counts_layer


@dataclass
class SCVIConfig:
    n_latent: int = 16
    max_epochs_train: int = 200
    max_epochs_query: int = 50
    early_stopping: bool = True
    early_stopping_patience_train: int = 20
    early_stopping_patience_query: int = 10
    counts_layer: str = "counts"
    batch_key: Optional[str] = None         # e.g. "sample" or "batch" if you have it
    categorical_covariates: Optional[list[str]] = None


def fit_scvi_reference(
    adata_train: sc.AnnData,
    outdir: str,
    cfg: Optional[SCVIConfig] = None,
) -> Path:
    cfg = cfg or SCVIConfig()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    model_dir = outdir / "scvi_model"

    ensure_counts_layer(adata_train, layer=cfg.counts_layer, name="TRAIN")

    scvi.model.SCVI.setup_anndata(
        adata_train,
        layer=cfg.counts_layer,
        batch_key=cfg.batch_key,
        categorical_covariate_keys=cfg.categorical_covariates,
    )

    model = scvi.model.SCVI(adata_train, n_latent=cfg.n_latent)
    model.train(
        max_epochs=cfg.max_epochs_train,
        early_stopping=cfg.early_stopping,
        early_stopping_patience=cfg.early_stopping_patience_train,
    )

    model.save(model_dir, overwrite=True)
    return model_dir


def map_query_to_reference(
    adata_query: sc.AnnData,
    model_dir: str | Path,
    cfg: Optional[SCVIConfig] = None,
) -> scvi.model.SCVI:
    cfg = cfg or SCVIConfig()
    ensure_counts_layer(adata_query, layer=cfg.counts_layer, name="TEST")
    q = scvi.model.SCVI.load_query_data(adata_query, model_dir)
    if cfg.max_epochs_query and cfg.max_epochs_query > 0:
        q.train(
            max_epochs=cfg.max_epochs_query,
            early_stopping=cfg.early_stopping,
            early_stopping_patience=cfg.early_stopping_patience_query,
        )
    return q


def save_latent_csv(
    Z: np.ndarray,
    obs: pd.DataFrame,
    out_csv: str,
    keep_obs_cols: Optional[list[str]] = None,
    cell_id_col: str = "cell_id",
):
    keep_obs_cols = keep_obs_cols or []
    dfZ = pd.DataFrame(Z, columns=[f"z{i+1}" for i in range(Z.shape[1])])

    if cell_id_col in obs.columns:
        dfZ.insert(0, "cell_id", obs[cell_id_col].astype(str).values)
    else:
        dfZ.insert(0, "cell_id", obs.index.astype(str).values)

    for c in keep_obs_cols:
        if c in obs.columns:
            dfZ.insert(1, c, obs[c].values)

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    dfZ.to_csv(out_csv, index=False)
