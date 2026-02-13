import argparse
import json
from pathlib import Path

from .scvi_latent import export_latents_from_saved_scvi
from .train import train_xgb_on_latent, XGBConfig
from .evaluate import eval_and_plot


def main():
    p = argparse.ArgumentParser(
        prog="chronomate",
        description="ChronoMate – scVI latent transfer + XGBoost time prediction.",
    )
    sp = p.add_subparsers(dest="cmd", required=True)

    # ------------------------------------------------------------
    # scvi-export-latent
    # ------------------------------------------------------------
    p_lat = sp.add_parser(
        "scvi-export-latent",
        help="Load a trained scVI reference model and export train/test latents.",
    )
    p_lat.add_argument("--train-h5ad", required=True)
    p_lat.add_argument("--test-h5ad", required=True)
    p_lat.add_argument("--model-dir", required=True, help="Folder containing saved scVI model (ref).")
    p_lat.add_argument("--out-train-csv", required=True)
    p_lat.add_argument("--out-test-csv", required=True)
    p_lat.add_argument("--counts-layer", default="counts")
    p_lat.add_argument("--query-max-epochs", type=int, default=1)

    # ------------------------------------------------------------
    # train-xgb-latent
    # ------------------------------------------------------------
    p_tr = sp.add_parser(
        "train-xgb-latent",
        help="Train XGBoost regressor on scVI latent and optionally predict test.",
    )
    p_tr.add_argument("--train-latent", required=True)
    p_tr.add_argument("--test-latent", default=None)
    p_tr.add_argument("--outdir", required=True)
    p_tr.add_argument("--time-col", default="time")
    p_tr.add_argument("--cell-id-col", default="cell_id")

    # optional xgb knobs
    p_tr.add_argument("--n-estimators", type=int, default=2000)
    p_tr.add_argument("--learning-rate", type=float, default=0.03)
    p_tr.add_argument("--max-depth", type=int, default=6)

    # ------------------------------------------------------------
    # eval
    # ------------------------------------------------------------
    p_ev = sp.add_parser(
        "eval",
        help="Evaluate predictions.csv and generate plots.",
    )
    p_ev.add_argument("--predictions", required=True, help="predictions.csv with columns time + predicted_time")
    p_ev.add_argument("--outdir", required=True)
    p_ev.add_argument("--time-col", default="time")
    p_ev.add_argument("--pred-col", default="predicted_time")

    args = p.parse_args()

    if args.cmd == "scvi-export-latent":
        out = export_latents_from_saved_scvi(
            train_h5ad=args.train_h5ad,
            test_h5ad=args.test_h5ad,
            model_dir=args.model_dir,
            out_train_csv=args.out_train_csv,
            out_test_csv=args.out_test_csv,
            counts_layer=args.counts_layer,
            query_max_epochs=args.query_max_epochs,
        )
        print(json.dumps(out, indent=2))

    elif args.cmd == "train-xgb-latent":
        cfg = XGBConfig(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
        )
        out = train_xgb_on_latent(
            train_latent_csv=args.train_latent,
            test_latent_csv=args.test_latent,
            outdir=args.outdir,
            time_col=args.time_col,
            cell_id_col=args.cell_id_col,
            xgb_cfg=cfg,
        )
        print(json.dumps(out, indent=2))

    elif args.cmd == "eval":
        m = eval_and_plot(
            predictions_csv=args.predictions,
            outdir=args.outdir,
            time_col=args.time_col,
            pred_col=args.pred_col,
        )
        print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
