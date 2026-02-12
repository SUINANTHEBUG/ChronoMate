from chronomate.pipeline import run_pipeline
from chronomate.scvi_norm import SCVIConfig
from chronomate.models import XGBConfig

if __name__ == "__main__":
    train_h5ad = r"C:\2024 Fall\chronocell\GSE156455\processed_train_counts.h5ad"
    test_h5ad  = r"C:\2024 Fall\chronocell\GSE142787\processed_test_counts.h5ad"
    outdir     = r"C:\2024 Fall\chronocell\scvi_xgb_run"

    scvi_cfg = SCVIConfig(
        n_latent=16,
        max_epochs_train=200,
        max_epochs_query=50,
        early_stopping=True,
        counts_layer="counts",
        batch_key=None,  # set to "sample" later if you add it
    )

    xgb_cfg = XGBConfig(
        n_estimators=2000,
        learning_rate=0.02,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=0,
        n_jobs=-1,
    )

    run_pipeline(train_h5ad, test_h5ad, outdir, scvi_cfg=scvi_cfg, xgb_cfg=xgb_cfg, eval_plots=True)
