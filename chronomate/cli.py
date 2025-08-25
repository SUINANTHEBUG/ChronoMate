import argparse, json
from .train import train_dann, train_xgb, DANNConfig
from .evaluate import eval_checkpoint, predict_dann


def main():
    p = argparse.ArgumentParser(prog="chronomate", description="ChronoMate – DANN pipeline for developmental time.")
    sp = p.add_subparsers(dest="cmd", required=True)

    # train-dann
    p_train = sp.add_parser("train-dann", help="Train a DANN model.")
    p_train.add_argument("--source", required=True)
    p_train.add_argument("--target", required=True)
    p_train.add_argument("--obs-time-key", default=None)
    p_train.add_argument("--obs-celltype-key", default=None)
    p_train.add_argument("--outdir", required=True)
    p_train.add_argument("--epochs", type=int, default=150)
    p_train.add_argument("--batch-size", type=int, default=256)
    p_train.add_argument("--lr", type=float, default=1e-4)
    p_train.add_argument("--wd", type=float, default=1e-4)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--no-amp", action="store_true")

    # train-xgb
    p_xgb = sp.add_parser("train-xgb", help="Train an XGBoost baseline.")
    p_xgb.add_argument("--source", required=True)
    p_xgb.add_argument("--target", default=None)
    p_xgb.add_argument("--obs-time-key", default=None)
    p_xgb.add_argument("--obs-celltype-key", default=None)
    p_xgb.add_argument("--outdir", required=True)

    # eval
    p_eval = sp.add_parser("eval", help="Evaluate a DANN checkpoint on a dataset (writes metrics if labels exist).")
    p_eval.add_argument("--checkpoint", required=True)
    p_eval.add_argument("--data", required=True)
    p_eval.add_argument("--obs-time-key", default=None)
    p_eval.add_argument("--obs-celltype-key", default=None)
    p_eval.add_argument("--outdir", required=True)

    # predict
    p_pred = sp.add_parser("predict", help="Predict only (no metrics).")
    p_pred.add_argument("--checkpoint", required=True)
    p_pred.add_argument("--data", required=True)
    p_pred.add_argument("--obs-time-key", default=None)
    p_pred.add_argument("--obs-celltype-key", default=None)
    p_pred.add_argument("--out", required=True)

    args = p.parse_args()
    if args.cmd == "train-dann":
        cfg = DANNConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, wd=args.wd, seed=args.seed, amp=not args.no_amp)
        ckpt = train_dann(args.source, args.target, args.outdir, args.obs_time_key, args.obs_celltype_key, cfg)
        print(ckpt)
    elif args.cmd == "train-xgb":
        path = train_xgb(args.source, args.target, args.outdir, args.obs_time_key, args.obs_celltype_key)
        print(path)
    elif args.cmd == "eval":
        m = eval_checkpoint(args.checkpoint, args.data, args.outdir, args.obs_time_key, args.obs_celltype_key)
        print(json.dumps(m, indent=2) if m else "Predictions written (no labels).")
    elif args.cmd == "predict":
        out = predict_dann(args.checkpoint, args.data, args.obs_time_key, args.obs_celltype_key)
        import pandas as pd
        df = pd.DataFrame({"cell": out["names"], "pred_h": out["pred"].reshape(-1)})
        df.to_csv(args.out, index=False)
        print(args.out)


if __name__ == "__main__":
    main()
