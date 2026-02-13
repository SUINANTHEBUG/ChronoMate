from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor


@dataclass
class XGBConfig:
    n_estimators: int = 2000
    learning_rate: float = 0.03
    max_depth: int = 6
    subsample: float = 0.85
    colsample_bytree: float = 0.85
    reg_lambda: float = 1.0
    objective: str = "reg:squarederror"
    n_jobs: int = -1
    random_state: int = 42

    def to_kwargs(self) -> Dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_lambda": self.reg_lambda,
            "objective": self.objective,
            "n_jobs": self.n_jobs,
            "random_state": self.random_state,
        }


def _infer_zcols(df: pd.DataFrame) -> list[str]:
    zcols = [c for c in df.columns if str(c).startswith("z")]
    if not zcols:
        raise ValueError("No latent columns found (expected z1, z2, ...).")
    # keep stable order z1..zN if possible
    def key(c: str) -> int:
        s = str(c).lstrip("z")
        return int(s) if s.isdigit() else 10**9
    zcols.sort(key=key)
    return zcols


def train_xgb_on_latent(
    train_latent_csv: str,
    test_latent_csv: Optional[str],
    outdir: str,
    *,
    time_col: str = "time",
    cell_id_col: str = "cell_id",
    xgb_cfg: Optional[XGBConfig] = None,
) -> Dict[str, Any]:
    """
    Train XGBoost regressor on scVI latent (TRAIN) and optionally predict TEST.

    Saves:
      - xgb_on_scvi_latent.joblib
      - predictions.csv (if test_latent_csv is provided)
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tr = pd.read_csv(train_latent_csv)
    if time_col not in tr.columns:
        raise ValueError(f"TRAIN latent must include '{time_col}' column.")

    zcols = _infer_zcols(tr)
    Xtr = tr[zcols].to_numpy(np.float32)
    ytr = tr[time_col].astype(float).to_numpy()

    cfg = xgb_cfg or XGBConfig()
    model = XGBRegressor(**cfg.to_kwargs())
    model.fit(Xtr, ytr)

    model_path = outdir / "xgb_on_scvi_latent.joblib"
    joblib.dump(model, model_path)

    out: Dict[str, Any] = {
        "model_path": str(model_path),
        "n_train": int(len(tr)),
        "n_latent": int(len(zcols)),
    }

    if test_latent_csv:
        te = pd.read_csv(test_latent_csv)
        # allow test to omit time (if doing blind prediction); keep if present
        Xte = te[zcols].to_numpy(np.float32)
        pred = model.predict(Xte).astype(float)

        pred_df = pd.DataFrame({
            cell_id_col: te[cell_id_col].astype(str).values if cell_id_col in te.columns else np.arange(len(te)).astype(str),
            "predicted_time": pred,
        })
        if time_col in te.columns:
            pred_df[time_col] = te[time_col].astype(float).values

        pred_path = outdir / "predictions.csv"
        pred_df.to_csv(pred_path, index=False)
        out["predictions_csv"] = str(pred_path)
        out["n_test"] = int(len(te))

    return out
