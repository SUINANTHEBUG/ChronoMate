import argparse
import json
from .train import train_xgb_zscore
from .evaluate import eval_xgb_zscore


def main():
    p = argparse.ArgumentParser(
        prog="chronomate",
        description="ChronoMate – cross-dataset developmental time prediction (XGBoost transfer).",
    )
    sp = p.add_subparsers(dest="cmd", required=True)

    # train-xgb-zscore
    p_train = sp.add_parser("train-xgb-zscore", help="Train XGBoost with shared-gene alignment + train-based z-scoring.")
    p_train.add_argument("--train", required=True, help="TRAIN csv (must include time column).")
    p_train.add_argument("--test", default=None, help="Optional TEST csv used only to define shared genes.")
    p_train.add_argument("--outdir", required=True)
    p_train.add_argument("--time-col", default="time")
    p_train.add_argument("--cell-id-col", default="cell_id")
    p_train.add_argument("--type-col", default=None, help="Optional type column (not used by this trainer; reserved).")

    # eval-xgb-zscore
    p_eval = sp.add_parser("eval-xgb-zscore", help="Evaluate XGBoost transfer model and generate plots.")
    p_eval.add_argument("--test", required=True, help="TEST csv (time column required for metrics).")
    p_eval.add_argument("--model", required=True, help="Path to model.joblib saved by train-xgb-zscore.")
    p_eval.add_argument("--preproc", required=True, help="Path to preproc.joblib saved by train-xgb-zscore.")
    p_eval.add_argument("--outdir", required=True)
    p_eval.add_argument("--time-col", default="time")
    p_eval.add_argument("--cell-id-col", default="cell_id")

    args = p.parse_args()

    if args.cmd == "train-xgb-zscore":
        out = train_xgb_zscore(
            train_csv=args.train,
            test_csv=args.test,
            outdir=args.outdir,
            time_col=args.time_col,
            cell_id_col=args.cell_id_col,
        )
        print(json.dumps(out, indent=2))

    elif args.cmd == "eval-xgb-zscore":
        m = eval_xgb_zscore(
            test_csv=args.test,
            model_path=args.model,
            preproc_path=args.preproc,
            outdir=args.outdir,
            time_col=args.time_col,
            cell_id_col=args.cell_id_col,
        )
        print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
