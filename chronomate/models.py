from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any

from xgboost import XGBRegressor


@dataclass
class XGBConfig:
    n_estimators: int = 1500
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


def make_xgb_regressor(cfg: XGBConfig | None = None) -> XGBRegressor:
    cfg = cfg or XGBConfig()
    return XGBRegressor(**cfg.to_kwargs())
