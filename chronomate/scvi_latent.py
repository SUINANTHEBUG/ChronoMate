from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import scanpy as sc
import scvi


def _align_by_intersection(ad_tr: sc.AnnData, ad_te: sc.AnnData) -> tuple[sc.AnnData, sc.AnnData]:
    common = ad_tr.var_names.intersection(ad_te.var_names)
    if len(common) == 0:
        raise ValueError("No overlapping genes between TRAIN and TEST.")
    ad_tr = ad_tr[:, common].copy()
    ad_te = ad_te[:, common].copy()
    return ad_tr, ad_te


def export_latents_from_saved_scvi(
    train_h5ad: str,
    test_h5ad: str,
    model_dir: str,
    out_train_csv: str,
    out_test_csv: str,
    *,
    counts_layer: str = "counts",
    query_max_epochs: int = 1,
    keep_train_obs: Sequence[str] = ("time", "rep", "type", "subtype", "class", "set", "genotype"),
    keep_test_obs: Sequence[str] = ("time", "type", "sample", "source_dir", "dataset"),
    accelerator: str = "gpu",
    devices: int = 1,
    precision: str = "16-mixed",
) -> dict:
    """
    Load a *trained* scVI model (reference) from model_dir, then:
      - export train latent (inference only; no training)
      - load query data for test; train query for `query_max_epochs` (usually 1); export test latent

    This avoids re-training the expensive 200-epoch reference model.
    """
    train_h5ad = str(train_h5ad)
    test_h5ad = str(test_h5ad)
    model_dir = str(model_dir)
    out_train_csv = str(out_train_csv)
    out_test_csv = str(out_test_csv)

    ad_tr = sc.read_h5ad(train_h5ad)
    ad_te = sc.read_h5ad(test_h5ad)

    ad_tr.obs_names_make_unique()
    ad_te.obs_names_make_unique()

    # align genes exactly the same way you did in notebooks
    ad_tr, ad_te = _align_by_intersection(ad_tr, ad_te)

    if counts_layer not in ad_tr.layers or counts_layer not in ad_te.layers:
        raise ValueError(f"Expected raw counts in .layers['{counts_layer}'] for both train and test.")

    # ---- Load reference model with train adata
    ref = scvi.model.SCVI.load(model_dir, adata=ad_tr)

    # ---- Train latent (no training)
    Ztr = ref.get_latent_representation()
    df_tr = pd.DataFrame(Ztr, columns=[f"z{i+1}" for i in range(Ztr.shape[1])])
    cell_id_tr = ad_tr.obs["cell_id"].astype(str).values if "cell_id" in ad_tr.obs.columns else ad_tr.obs_names.astype(str)
    df_tr.insert(0, "cell_id", cell_id_tr)

    for col in keep_train_obs:
        if col in ad_tr.obs.columns:
            df_tr.insert(1, col, ad_tr.obs[col].values)

    Path(out_train_csv).parent.mkdir(parents=True, exist_ok=True)
    df_tr.to_csv(out_train_csv, index=False)

    # ---- Query: map test into latent
    # NOTE: pass the *model object*, not the Path (fixes your earlier AttributeError)
    q = scvi.model.SCVI.load_query_data(ad_te, ref)

    if query_max_epochs and query_max_epochs > 0:
        q.train(
            max_epochs=int(query_max_epochs),
            accelerator=accelerator,
            devices=devices,
            precision=precision,
            enable_checkpointing=False,
            log_every_n_steps=50,
        )

    Zte = q.get_latent_representation()
    df_te = pd.DataFrame(Zte, columns=[f"z{i+1}" for i in range(Zte.shape[1])])
    cell_id_te = ad_te.obs["cell_id"].astype(str).values if "cell_id" in ad_te.obs.columns else ad_te.obs_names.astype(str)
    df_te.insert(0, "cell_id", cell_id_te)

    for col in keep_test_obs:
        if col in ad_te.obs.columns:
            df_te.insert(1, col, ad_te.obs[col].values)

    Path(out_test_csv).parent.mkdir(parents=True, exist_ok=True)
    df_te.to_csv(out_test_csv, index=False)

    return {
        "train_latent_csv": out_train_csv,
        "test_latent_csv": out_test_csv,
        "n_train": int(ad_tr.n_obs),
        "n_test": int(ad_te.n_obs),
        "n_genes_common": int(ad_tr.n_vars),
        "n_latent": int(Ztr.shape[1]),
        "query_max_epochs": int(query_max_epochs),
    }
