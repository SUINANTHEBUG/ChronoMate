from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import numpy as np
import pandas as pd


DEFAULT_META_COLS = {
    "cell_id", "time", "sample", "type",
    "cell_type",  # legacy
}


def infer_gene_columns(df: pd.DataFrame, meta_cols: Optional[set] = None) -> List[str]:
    meta_cols = meta_cols or DEFAULT_META_COLS
    return [c for c in df.columns if c not in meta_cols]


def load_csv_df(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def shared_genes_from_train_test(train_df: pd.DataFrame, test_df: Optional[pd.DataFrame]) -> List[str]:
    train_genes = set(infer_gene_columns(train_df))
    if test_df is None:
        return sorted(train_genes)
    test_genes = set(infer_gene_columns(test_df))
    return sorted(train_genes.intersection(test_genes))


@dataclass
class ZScorePreproc:
    genes: List[str]
    mu: pd.Series
    sd: pd.Series

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.genes].astype(np.float32)
        return ((X - self.mu) / self.sd).to_numpy(np.float32)


def fit_zscore_preproc(train_df: pd.DataFrame, genes: List[str]) -> ZScorePreproc:
    X = train_df[genes].astype(np.float32)
    mu = X.mean(axis=0)
    sd = X.std(axis=0).replace(0, 1.0)
    return ZScorePreproc(genes=genes, mu=mu, sd=sd)


def require_columns(df: pd.DataFrame, cols: List[str], name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")
