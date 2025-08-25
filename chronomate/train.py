import os, math, json
from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import mean_squared_error
import joblib
from tqdm import trange

from .data import load_matrix, fit_scaler_on_source, encode_cell_types, apply_cell_type_encoder, SingleCellDataset
from .models import DANN, dann_alpha, XGBWrapper


@dataclass
class DANNConfig:
    epochs: int = 150
    batch_size: int = 256
    lr: float = 1e-4
    wd: float = 1e-4
    seed: int = 42
    amp: bool = True


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _make_loader(ds: SingleCellDataset, bs: int, shuffle: bool = True) -> DataLoader:
    X = torch.from_numpy(ds.X)
    ct = torch.from_numpy(ds.ct) if ds.ct is not None else torch.full((len(ds),), -1, dtype=torch.long)
    t = torch.from_numpy(np.nan_to_num(ds.times, nan=-1.0)) if ds.times is not None else torch.full((len(ds),), -1.0)
    return DataLoader(TensorDataset(X, ct, t), batch_size=bs, shuffle=shuffle, drop_last=False)


def _mix_iter(src_loader: DataLoader, tgt_loader: DataLoader):
    src_iter = iter(src_loader)
    tgt_iter = iter(tgt_loader)
    while True:
        try:
            s = next(src_iter)
        except StopIteration:
            src_iter = iter(src_loader)
            s = next(src_iter)
        try:
            t = next(tgt_iter)
        except StopIteration:
            tgt_iter = iter(tgt_loader)
            t = next(tgt_iter)
        yield s, t


def train_dann(
    source_path: str,
    target_path: str,
    outdir: str,
    obs_time_key: Optional[str] = None,
    obs_celltype_key: Optional[str] = None,
    cfg: Optional[DANNConfig] = None,
) -> str:
    os.makedirs(outdir, exist_ok=True)
    cfg = cfg or DANNConfig()
    set_seed(cfg.seed)

    src = load_matrix(source_path, obs_time_key, obs_celltype_key)
    tgt = load_matrix(target_path, obs_time_key, obs_celltype_key)
    if src["times"] is None:
        raise ValueError("Source must include times/labels.")

    scaler_path = os.path.join(outdir, "scaler.joblib")
    scaler = fit_scaler_on_source(src["X"], scaler_path)
    Xs = scaler.transform(src["X"]).astype(np.float32)
    Xt = scaler.transform(tgt["X"]).astype(np.float32)

    ct_src_enc, ct_mapping = encode_cell_types(src["cell_types"])
    ct_tgt_enc = apply_cell_type_encoder(tgt["cell_types"], ct_mapping)

    src_ds = SingleCellDataset(Xs, src["times"].astype(np.float32), ct_src_enc, src["names"])
    tgt_ds = SingleCellDataset(Xt, tgt["times"].astype(np.float32) if tgt["times"] is not None else None, ct_tgt_enc, tgt["names"])

    if np.unique(src["times"]).size < 2:
        print("WARNING: Source has only one time point; DANN may underperform.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_dim = Xs.shape[1]
    n_ctypes = len(ct_mapping) if ct_mapping else 0
    model = DANN(in_dim=in_dim, n_ctypes=n_ctypes).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=1e-6)

    # ✅ Guard AMP on CUDA availability
    use_amp = cfg.amp and torch.cuda.is_available()
    scaler_amp = torch.cuda.amp.GradScaler(enabled=use_amp)

    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()

    src_loader = _make_loader(src_ds, bs=max(2, cfg.batch_size // 2), shuffle=True)
    tgt_loader = _make_loader(tgt_ds, bs=max(2, cfg.batch_size // 2), shuffle=True)
    mixer = _mix_iter(src_loader, tgt_loader)

    best_path = os.path.join(outdir, "best.pt")
    best_rmse = float("inf")

    steps_per_epoch = max(len(src_loader), len(tgt_loader))
    for epoch in trange(cfg.epochs, desc="Training DANN"):
        for step in range(steps_per_epoch):
            (xs, cts, ys), (xt, ctt, _yt) = next(mixer)
            xs, xt = xs.to(device), xt.to(device)
            cts, ctt = cts.to(device), ctt.to(device)
            ys = ys.to(device)
            dom_s = torch.ones(xs.size(0), device=device)
            dom_t = torch.zeros(xt.size(0), device=device)

            progress = (epoch * steps_per_epoch + step) / (cfg.epochs * steps_per_epoch)
            alpha = dann_alpha(progress)

            opt.zero_grad(set_to_none=True)
            # ✅ Use AMP only if CUDA is available
            with torch.cuda.amp.autocast(enabled=use_amp):
                y_pred_s, d_logits_s = model(xs, cts, alpha=alpha)
                _y_pred_t, d_logits_t = model(xt, ctt, alpha=alpha)

                loss_reg = mse(y_pred_s, ys)
                loss_dom = 0.5 * (bce(d_logits_s, dom_s) + bce(d_logits_t, dom_t))
                loss = loss_reg + loss_dom

            scaler_amp.scale(loss).backward()
            scaler_amp.step(opt)
            scaler_amp.update()

        sched.step()

        # quick validation on 10% of source
        with torch.no_grad():
            model.eval()
            nval = max(1, int(0.1 * len(src_ds)))
            idx = torch.randperm(len(src_ds))[:nval]
            xv = torch.from_numpy(src_ds.X[idx.numpy()]).to(device)
            if src_ds.ct is not None:
                ctv = torch.from_numpy(src_ds.ct[idx.numpy()]).to(device)
            else:
                ctv = torch.full((nval,), -1, dtype=torch.long, device=device)
            yv = torch.from_numpy(src_ds.times[idx.numpy()]).to(device)
            yhat, _ = model(xv, ctv, alpha=0.0)
            rmse = math.sqrt(mean_squared_error(yv.cpu().numpy(), yhat.cpu().numpy()))
            model.train()

        if rmse < best_rmse:
            best_rmse = rmse
            torch.save({
                "model_state": model.state_dict(),
                "opt_state": opt.state_dict(),
                "sched_state": sched.state_dict(),
                "scaler_state": scaler_amp.state_dict(),
                "in_dim": in_dim,
                "n_ctypes": n_ctypes,
                "ct_mapping": ct_mapping,
                "scaler_path": scaler_path,
                "config": vars(cfg),
                "epoch": epoch,
            }, best_path)

    return best_path


def train_xgb(
    source_path: str,
    target_path: Optional[str],
    outdir: str,
    obs_time_key: Optional[str] = None,
    obs_celltype_key: Optional[str] = None,
    xgb_kwargs: Optional[Dict[str, Any]] = None,
) -> str:
    os.makedirs(outdir, exist_ok=True)
    src = load_matrix(source_path, obs_time_key, obs_celltype_key)
    if src["times"] is None:
        raise ValueError("Source must include times/labels.")

    scaler_path = os.path.join(outdir, "scaler.joblib")
    scaler = fit_scaler_on_source(src["X"], scaler_path)
    Xs = scaler.transform(src["X"]).astype(np.float32)

    model = XGBWrapper(**(xgb_kwargs or {}))
    model.fit(Xs, src["times"].astype(np.float32))

    model_path = os.path.join(outdir, "xgb.json")
    model.save(model_path)

    meta = {"scaler_path": scaler_path, "ct_mapping": None}
    json.dump(meta, open(os.path.join(outdir, "meta.json"), "w"))
    return model_path
