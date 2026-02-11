from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, Dict, Any

import joblib
import numpy as np
import pandas as pd

from .data import load_csv_df, shared_genes_from_train_test, fit_zscore_preproc, require_columns
from .models import make_xgb_regressor, XGBConfig


def train_xgb_zscore(
    train_csv: str,
    test_csv: Optional[str],
    outdir: str,
    time_col: str = "time",
    cell_id_col: str = "cell_id",
    xgb_cfg: Optional[XGBConfig] = None,
) -> Dict[str, Any]:
    """
    Train XGBoost on TRAIN with:
      - shared-gene feature intersection (optionally using TEST)
      - z-score normalization using TRAIN stats only
    Saves:
      - model.joblib
      - preproc.joblib  (genes + mu + sd)
    """
    outdir = str(outdir)
    os.makedirs(outdir, exist_ok=True)

    train_df = load_csv_df(train_csv)
    test_df = load_csv_df(test_csv) if test_csv else None

    require_columns(train_df, [time_col], "TRAIN")

    genes = shared_genes_from_train_test(train_df, test_df)
    if len(genes) == 0:
        raise ValueError("No shared gene columns found between TRAIN and TEST (or TRAIN alone).")

    preproc = fit_zscore_preproc(train_df, genes)

    X_train = preproc.transform(train_df)
    y_train = train_df[time_col].astype(float).to_numpy()

    model = make_xgb_regressor(xgb_cfg)
    model.fit(X_train, y_train)

    model_path = str(Path(outdir) / "xgb_regressor_zscore.joblib")
    preproc_path = str(Path(outdir) / "xgb_regressor_zscore_preproc.joblib")

    joblib.dump(model, model_path)
    joblib.dump({"genes": preproc.genes, "mu": preproc.mu, "sd": preproc.sd}, preproc_path)

    return {
        "model_path": model_path,
        "preproc_path": preproc_path,
        "n_train": int(train_df.shape[0]),
        "n_genes": int(len(genes)),
        "used_test_for_intersection": bool(test_df is not None),
    }
