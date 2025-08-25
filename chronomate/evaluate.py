import os, math
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import joblib

from .data import load_matrix, apply_cell_type_encoder
from .models import DANN


def _metrics(y_true, y_pred) -> Dict[str, float]:
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def predict_dann(checkpoint: str, data_path: str, obs_time_key: Optional[str], obs_celltype_key: Optional[str]) -> Dict[str, Any]:
    ckpt = torch.load(checkpoint, map_location="cpu")
    scaler = joblib.load(ckpt["scaler_path"])
    meta = load_matrix(data_path, obs_time_key, obs_celltype_key)
    X = scaler.transform(meta["X"]).astype(np.float32)
    ct = apply_cell_type_encoder(meta["cell_types"], ckpt["ct_mapping"]) if ckpt.get("ct_mapping") is not None else None

    model = DANN(in_dim=ckpt["in_dim"], n_ctypes=ckpt["n_ctypes"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with torch.no_grad():
        X_t = torch.from_numpy(X)
        ct_t = torch.from_numpy(ct) if ct is not None else torch.full((len(X),), -1, dtype=torch.long)
        yhat, _ = model(X_t, ct_t, alpha=0.0)
        preds = yhat.numpy()

    out = {
        "names": meta["names"].astype(str).to_list(),
        "pred": preds,
        "true": meta["times"] if meta["times"] is not None else None,
    }
    return out


def parity_plot(y_true, y_pred, out_png: str, title: str = "Predicted vs True"):
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, s=6, alpha=0.6)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def eval_checkpoint(checkpoint: str, data_path: str, outdir: str, obs_time_key: Optional[str], obs_celltype_key: Optional[str]) -> Dict[str, float]:
    os.makedirs(outdir, exist_ok=True)
    out = predict_dann(checkpoint, data_path, obs_time_key, obs_celltype_key)
    if out["true"] is None:
        df = pd.DataFrame({"cell": out["names"], "pred_h": out["pred"].reshape(-1)})
        df.to_csv(os.path.join(outdir, "preds.csv"), index=False)
        return {}

    y_true = out["true"].reshape(-1)
    y_pred = out["pred"].reshape(-1)
    metrics = _metrics(y_true, y_pred)
    pd.DataFrame([metrics]).to_csv(os.path.join(outdir, "metrics.csv"), index=False)

    parity_plot(y_true, y_pred, os.path.join(outdir, "parity.png"), title="DANN: Predicted vs True")
    df = pd.DataFrame({"cell": out["names"], "true_h": y_true, "pred_h": y_pred})
    df.to_csv(os.path.join(outdir, "preds_with_truth.csv"), index=False)
    return metrics
