from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from .data import load_csv_df, require_columns


def _snap_to_allowed(pred: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    diffs = np.abs(pred.reshape(-1, 1) - allowed.reshape(1, -1))
    return allowed[np.argmin(diffs, axis=1)]


def eval_xgb_zscore(
    test_csv: str,
    model_path: str,
    preproc_path: str,
    outdir: str,
    time_col: str = "time",
    cell_id_col: str = "cell_id",
) -> Dict[str, Any]:
    """
    Evaluate a saved XGBoost+zscore model on TEST.
    Produces:
      - metrics.json
      - predictions.csv
      - plots (scatter, boxplot, violin Figure 3 style)
    """
    outdir = str(outdir)
    os.makedirs(outdir, exist_ok=True)

    test_df = load_csv_df(test_csv)
    require_columns(test_df, [time_col], "TEST (for metrics)")
    if cell_id_col not in test_df.columns:
        # not fatal; we can synthesize ids
        test_df[cell_id_col] = test_df.index.astype(str)

    model = joblib.load(model_path)
    prep = joblib.load(preproc_path)

    genes = prep["genes"]
    mu = prep["mu"]
    sd = prep["sd"]

    missing_genes = [g for g in genes if g not in test_df.columns]
    if missing_genes:
        raise ValueError(
            "TEST is missing "
            f"{len(missing_genes)} gene columns that the model expects "
            f"(first 20): {missing_genes[:20]}"
        )

    X = test_df[genes].astype(np.float32)
    Xz = ((X - mu) / sd).to_numpy(np.float32)
    y = test_df[time_col].astype(float).to_numpy()

    pred = model.predict(Xz)

    mae = float(mean_absolute_error(y, pred))

    allowed = np.array(sorted(np.unique(y)), dtype=float)
    pred_snap = _snap_to_allowed(pred, allowed)

    snap_acc = float((pred_snap == y).mean())
    snap_mae = float(mean_absolute_error(y, pred_snap))

    # per-time summary
    dfp = pd.DataFrame({"true_time": y, "pred_time": pred, "pred_time_snapped": pred_snap})
    summary = dfp.groupby("true_time").agg(
        n_cells=("pred_time", "size"),
        mean_pred=("pred_time", "mean"),
        median_pred=("pred_time", "median"),
        std_pred=("pred_time", "std"),
        mean_pred_snapped=("pred_time_snapped", "mean"),
    ).reset_index().sort_values("true_time")

    # Save predictions
    pred_out = pd.DataFrame({
        "cell_id": test_df[cell_id_col].astype(str).values,
        "time": y,
        "pred_time": pred,
        "pred_time_snapped": pred_snap,
    })
    pred_csv_path = str(Path(outdir) / "predictions.csv")
    pred_out.to_csv(pred_csv_path, index=False)

    # Save metrics
    metrics = {
        "TEST_MAE_hours": mae,
        "NearestAllowed_accuracy": snap_acc,
        "NearestAllowed_MAE_hours": snap_mae,
        "Allowed_test_timepoints": allowed.tolist(),
        "predictions_csv": pred_csv_path,
    }
    (Path(outdir) / "metrics.json").write_text(
        pd.Series(metrics).to_json(indent=2), encoding="utf-8"
    )
    summary.to_csv(Path(outdir) / "per_time_summary.csv", index=False)

    # ---- Plot 1: scatter true vs pred + mean points ----
    plt.figure()
    plt.scatter(y, pred, s=3, alpha=0.12)
    mean_pts = summary[["true_time", "mean_pred"]].to_numpy()
    plt.scatter(mean_pts[:, 0], mean_pts[:, 1], s=80, marker="x")
    mn = float(min(y.min(), pred.min()))
    mx = float(max(y.max(), pred.max()))
    plt.plot([mn, mx], [mn, mx])
    plt.xlabel("True time")
    plt.ylabel("Predicted time")
    plt.title("Predicted vs True (cell-level) + mean per true time (X)")
    plt.tight_layout()
    plt.savefig(Path(outdir) / "scatter_mean.png", dpi=180)
    plt.close()

    # ---- Plot 2: boxplot predicted distribution per true time ----
    order = sorted(dfp["true_time"].unique())
    data = [dfp.loc[dfp["true_time"] == t, "pred_time"].values for t in order]
    plt.figure()
    plt.boxplot(data, labels=[str(int(t)) for t in order], showfliers=False)
    plt.xlabel("True timepoint")
    plt.ylabel("Predicted time")
    plt.title("Predicted distribution by true timepoint")
    plt.tight_layout()
    plt.savefig(Path(outdir) / "boxplot_by_time.png", dpi=180)
    plt.close()

    # ---- Plot 3 (Figure 3 style): your violin plot ----
    times = sorted(dfp["true_time"].unique())
    violin_data = [dfp.loc[dfp["true_time"] == t, "pred_time"].values for t in times]
    avg_predicted = [float(np.mean(v)) for v in violin_data]

    plt.figure(figsize=(13, 7))
    parts = plt.violinplot(
        violin_data,
        positions=times,
        widths=6,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    for body in parts["bodies"]:
        body.set_facecolor("#BFE0E9")
        body.set_edgecolor("black")
        body.set_linewidth(0.8)
        body.set_alpha(0.9)

    plt.scatter(
        times,
        avg_predicted,
        color="black",
        s=40,
        zorder=3,
        label="Mean predicted time",
    )

    tmin = float(min(list(times) + list(avg_predicted)))
    tmax = float(max(list(times) + list(avg_predicted)))
    plt.plot(
        [tmin, tmax],
        [tmin, tmax],
        linestyle="--",
        color="red",
        linewidth=2,
        label="Perfect Prediction (y = x)",
    )

    plt.xlabel("Actual Time")
    plt.ylabel("Predicted Time")
    plt.title("Predicted Time vs Actual Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(outdir) / "Figure 3.png", dpi=180)  # keeps your Figure 3 output name
    plt.close()

    # Print means to console in the same style you liked
    print("TEST MAE (hours):", mae)
    print("Nearest-allowed-time accuracy:", snap_acc)
    print("Nearest-allowed-time MAE:", snap_mae)
    print("Allowed test timepoints:", allowed.tolist())
    print("\nPer-timepoint summary (true vs predicted):")
    print(summary.to_string(index=False))
    print("\nAverage Predicted Time for each Actual Time:\n")
    print(avg_predicted)
    print("\nSaved predictions:", pred_csv_path)
    print("Saved plots to:", outdir)

    return metrics
