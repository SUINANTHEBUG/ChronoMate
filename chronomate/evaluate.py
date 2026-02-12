from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error


def _snap_to_allowed(pred: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    diffs = np.abs(pred.reshape(-1, 1) - allowed.reshape(1, -1))
    return allowed[np.argmin(diffs, axis=1)]


def eval_and_plot(
    predictions_csv: str,
    outdir: str,
    time_col: str = "time",
    pred_col: str = "predicted_time",
    time_adjustments: Optional[Dict[float, float]] = None,  # if you insist on doing this
) -> Dict[str, Any]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(predictions_csv)

    for c in [time_col, pred_col]:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in {predictions_csv}. Found: {list(df.columns)}")

    # optional “manual bias correction”
    if time_adjustments:
        for t, adj in time_adjustments.items():
            m = df[time_col].astype(float) == float(t)
            df.loc[m, pred_col] = df.loc[m, pred_col].astype(float) + float(adj)

    y = df[time_col].astype(float).to_numpy()
    pred = df[pred_col].astype(float).to_numpy()

    mae = float(mean_absolute_error(y, pred))

    allowed = np.array(sorted(np.unique(y)), dtype=float)
    pred_snap = _snap_to_allowed(pred, allowed)
    snap_acc = float((pred_snap == y).mean())
    snap_mae = float(mean_absolute_error(y, pred_snap))

    # per-time summary
    dfp = pd.DataFrame({"true_time": y, "pred_time": pred})
    summary = dfp.groupby("true_time").agg(
        n_cells=("pred_time", "size"),
        mean_pred=("pred_time", "mean"),
        median_pred=("pred_time", "median"),
        std_pred=("pred_time", "std"),
    ).reset_index().sort_values("true_time")
    summary.to_csv(outdir / "per_time_summary.csv", index=False)

    # Scatter
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
    plt.savefig(outdir / "scatter_mean.png", dpi=180)
    plt.close()

    # Boxplot
    order = sorted(dfp["true_time"].unique())
    data = [dfp.loc[dfp["true_time"] == t, "pred_time"].values for t in order]
    plt.figure()
    plt.boxplot(data, labels=[str(int(t)) for t in order], showfliers=False)
    plt.xlabel("True timepoint")
    plt.ylabel("Predicted time")
    plt.title("Predicted distribution by true timepoint")
    plt.tight_layout()
    plt.savefig(outdir / "boxplot_by_time.png", dpi=180)
    plt.close()

    # Violin (your Figure 3 style)
    times = order
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

    plt.scatter(times, avg_predicted, color="black", s=40, zorder=3, label="Mean predicted time")

    tmin = float(min(list(times) + list(avg_predicted)))
    tmax = float(max(list(times) + list(avg_predicted)))
    plt.plot([tmin, tmax], [tmin, tmax], linestyle="--", color="red", linewidth=2, label="Perfect Prediction (y = x)")

    plt.xlabel("Actual Time")
    plt.ylabel("Predicted Time")
    plt.title("Predicted Time vs Actual Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "Figure 3.png", dpi=180)
    plt.close()

    metrics = {
        "MAE_hours": mae,
        "NearestAllowed_accuracy": snap_acc,
        "NearestAllowed_MAE_hours": snap_mae,
        "Allowed_test_timepoints": allowed.tolist(),
        "predictions_csv": str(Path(predictions_csv)),
        "outdir": str(outdir),
    }
    (outdir / "metrics.json").write_text(pd.Series(metrics).to_json(indent=2), encoding="utf-8")

    print("MAE (hours):", mae)
    print("Nearest-allowed accuracy:", snap_acc)
    print("Nearest-allowed MAE:", snap_mae)
    print("Allowed timepoints:", allowed.tolist())
    print("Saved plots to:", outdir)

    return metrics
