from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any

import scanpy as sc

from .io import load_h5ad, require_obs_cols, align_genes, parse_time_to_hours, ensure_counts_layer
from .scvi_norm import SCVIConfig, fit_scvi_reference, map_query_to_reference, save_latent_csv
from .train import train_xgb_on_latent
from .eval import eval_and_plot
from .models import XGBConfig


def run_pipeline(
    train_h5ad: str,
    test_h5ad: str,
    outdir: str,
    scvi_cfg: Optional[SCVIConfig] = None,
    xgb_cfg: Optional[XGBConfig] = None,
    batch_key: Optional[str] = None,
    eval_plots: bool = True,
) -> Dict[str, Any]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    scvi_cfg = scvi_cfg or SCVIConfig()
    if batch_key is not None:
        scvi_cfg.batch_key = batch_key

    # Load
    ad_tr = load_h5ad(train_h5ad)
    ad_te = load_h5ad(test_h5ad)

    # Fix duplicate obs names
    ad_tr.obs_names_make_unique()
    ad_te.obs_names_make_unique()

    # Required columns
    require_obs_cols(ad_tr, ["time"], "TRAIN")
    ensure_counts_layer(ad_tr, layer=scvi_cfg.counts_layer, name="TRAIN")
    ensure_counts_layer(ad_te, layer=scvi_cfg.counts_layer, name="TEST")

    # Align genes
    ad_tr, ad_te, _ = align_genes(ad_tr, ad_te)

    # Convert train time to numeric hours into obs["time_hours"]
    ad_tr.obs["time_hours"] = parse_time_to_hours(ad_tr.obs["time"])

    # Tag dataset (optional bookkeeping)
    ad_tr.obs["dataset"] = "train"
    ad_te.obs["dataset"] = "test"

    # ---- scVI stage
    model_dir = fit_scvi_reference(ad_tr, outdir=str(outdir), cfg=scvi_cfg)
    ref = map_query_to_reference(ad_tr, model_dir, cfg=scvi_cfg)  # not strictly needed; keep simple
    q   = map_query_to_reference(ad_te, model_dir, cfg=scvi_cfg)

    # Latents
    Z_train = scvi.model.SCVI.load(model_dir, adata=ad_tr).get_latent_representation()
    Z_test  = q.get_latent_representation()

    train_latent_csv = outdir / "train_latent.csv"
    test_latent_csv  = outdir / "test_latent.csv"

    save_latent_csv(Z_train, ad_tr.obs, str(train_latent_csv), keep_obs_cols=["time_hours", "time"], cell_id_col="cell_id")
    save_latent_csv(Z_test,  ad_te.obs, str(test_latent_csv),  keep_obs_cols=["time"], cell_id_col="cell_id")

    # ---- XGB stage
    res_train = train_xgb_on_latent(
        train_latent_csv=str(train_latent_csv),
        test_latent_csv=str(test_latent_csv),
        outdir=str(outdir),
        time_col="time_hours",
        cell_id_col="cell_id",
        xgb_cfg=xgb_cfg,
    )

    # ---- Eval stage
    metrics = {}
    if eval_plots:
        pred_csv = res_train["predictions_csv"]
        # If your test has numeric times already, good. If it's "70", good.
        metrics = eval_and_plot(pred_csv, outdir=str(outdir))

    return {"train": res_train, "eval": metrics, "outdir": str(outdir)}
