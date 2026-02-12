from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from .models import XGBConfig, make_xgb_regressor


def train_xgb_on_latent(
    train_latent_csv: str,
    test_latent_csv: str,
    outdir: str,
    time_col: str = "time",
    cell_id_col: str = "cell_id",
    xgb_cfg: Optional[XGBConfig] = None,
) -> Dict[str, Any]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tr = pd.read_csv(train_latent_csv)
    te = pd.read_csv(test_latent_csv)

    if time_col not in tr.columns:
        raise ValueError(f"train_latent_csv must include '{time_col}'.")

    z_cols = [c for c in tr.columns if c.startswith("z")]
    if not z_cols:
        raise ValueError("No latent columns found (expected z1, z2, ...).")

    Xtr = tr[z_cols].to_numpy(np.float32)
    ytr = tr[time_col].astype(float).to_numpy()

    model = make_xgb_regressor(xgb_cfg)
    model.fit(Xtr, ytr)

    Xte = te[z_cols].to_numpy(np.float32)
    pred = model.predict(Xte)

    pred_df = pd.DataFrame({
        cell_id_col: te[cell_id_col].astype(str).values if cell_id_col in te.columns else np.arange(len(te)).astype(str),
        "predicted_time": pred,
    })

    # carry true time if present for evaluation
    if time_col in te.columns:
        pred_df[time_col] = te[time_col].astype(float).values

    model_path = outdir / "xgb_on_scvi_latent.joblib"
    pred_path  = outdir / "predictions.csv"

    joblib.dump(model, model_path)
    pred_df.to_csv(pred_path, index=False)

    return {
        "model_path": str(model_path),
        "predictions_csv": str(pred_path),
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "n_latent": int(len(z_cols)),
    }
