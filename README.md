# ChronoMate

ChronoMate predicts ***Drosophila* developmental state (hours)** from a labeled **scRNA-seq** dataset
and transfers that predictor across unlabeled datasets and samples from different experiments. 

**Pipeline: Raw counts → scVI latent → XGBoost regression → evaluation + plots**


---

## What it does

Given a labeled **TRAIN** dataset and a unlabeled **TEST** dataset for developmental state marked in hours, ChronocMate determines the **TEST** sample developmetal state with accuracy.  

## Steps
1) Learn a batch-corrected representation with **scVI (VAE)** on raw counts  
2) Map TEST into the same latent space  
3) Train **XGBoost** on TRAIN latent to predict time (hours)  
4) Predict + evaluate on TEST and generate plots  

---

## Data sources (GEO)

**TRAIN (source):** Kurmangaliyev et al., *Neuron* 2020  
- This is a time-series *Drosophila* single-cell dataset with samples from pupa (P0) to adult (P96) in 12h increments
- GEO: GSE156455  
- https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE156455

**TEST (target):** Özel et al., *Nature* 2021  
- similar to train data with time points P15, P30, P40, P50, P70. 
- GEO: GSE142787  
- https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE142787

---

## Batch-correction

ChronoMate uses **scVI (scvi-tools)** to learn a **batch-aware latent embedding** from **raw counts**.

- Input: nonnegative raw counts in `adata.layers["counts"]`
- Features for regression: scVI latent `Z` (cells × latent_dim)

The regressor does **not** train on gene-level raw counts directly.  
It trains on the **scVI latent representation**.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\Activate on Windows
pip install -r requirements.txt
```

Verify installation:

```bash
python -m chronomate.cli --help
```

---

## Quick start (CLI)

### 1) Export scVI latents

```bash
python -m chronomate.cli scvi-export-latent   --train-h5ad path/to/train_counts.h5ad   --test-h5ad  path/to/test_counts.h5ad   --model-dir  path/to/scvi_model   --out-train-csv train_latent.csv   --out-test-csv  test_latent.csv   --query-max-epochs 1
```

### 2) Train XGBoost + predict

```bash
python -m chronomate.cli train-xgb-latent   --train-latent train_latent.csv   --test-latent  test_latent.csv   --outdir runs/xgb_on_scvi_latent
```

### 3) Evaluate

```bash
python -m chronomate.cli eval   --predictions runs/xgb_on_scvi_latent/predictions.csv   --outdir runs/xgb_on_scvi_latent/eval
```

---

## Example results (scVI → XGBoost)

| Sample | True Time (h) | Predicted (h) | Error (h) | Abs Error (h) | # Cells |
|--------|---------------|---------------|-----------|---------------|--------:|
| P15_1  | 15 | 15.92 | +0.92 | 0.92 | 6959 |
| P15_2  | 15 | 15.82 | +0.82 | 0.82 | 6905 |
| P15_3  | 15 | 15.91 | +0.91 | 0.91 | 6708 |
| P15_5  | 15 | 15.63 | +0.63 | 0.63 | 5087 |
| P15_6  | 15 | 15.63 | +0.63 | 0.63 | 5359 |
| P30_1  | 30 | 27.31 | -2.69 | 2.69 | 6717 |
| P30_2  | 30 | 27.63 | -2.37 | 2.37 | 6595 |
| P30_3  | 30 | 27.61 | -2.39 | 2.39 | 6638 |
| P30_4  | 30 | 28.14 | -1.86 | 1.86 | 7846 |
| P30_5  | 30 | 28.28 | -1.72 | 1.72 | 7692 |
| P40_1  | 40 | 38.59 | -1.41 | 1.41 | 6138 |
| P40_2  | 40 | 38.63 | -1.37 | 1.37 | 6221 |
| P40_3  | 40 | 38.77 | -1.23 | 1.23 | 6188 |
| P40_4  | 40 | 39.17 | -0.83 | 0.83 | 5164 |
| P50_1  | 50 | 49.77 | -0.23 | 0.23 | 6334 |
| P50_2  | 50 | 49.01 | -0.99 | 0.99 | 7895 |
| P50_3  | 50 | 48.91 | -1.09 | 1.09 | 8342 |
| P50_4  | 50 | 48.93 | -1.07 | 1.07 | 8012 |
| P70_5  | 70 | 74.20 | +4.20 | 4.20 | 10547 |
| P70_6  | 70 | 73.72 | +3.72 | 3.72 | 8774 |
| P70_7  | 70 | 73.68 | +3.68 | 3.68 | 8891 |
| P70_8  | 70 | 73.25 | +3.25 | 3.25 | 8166 |
| P70_9  | 70 | 73.36 | +3.36 | 3.36 | 7101 |

### Overall performance

**Sample-level**
- MAE: **1.80 hours**
- RMSE: **2.13 hours**
- R²: **0.988**
- n = **23 samples**

**Cell-level**
- MAE: **3.38 hours**
- RMSE: **4.66 hours**
- R²: **0.944**
- n = **164,279 cells**

> For reporting performance, metrics can be computed over **all cells**.

![Figure: Predicted time vs actual time](./scVI_XGBoost.png)
Figure 1: example of prediction performance using Chronocell

---

## Notes 

### On organisms outside of *Drosophila*
This tool is only guaranteed to work with *Drosophila* neuron data, performance may drop for higher organisms like mouse or human due to higher variation between individuals in transcription profiles. 

### scVI requires raw counts
If you see negative values or floats, those are likely scaled/z-scored features and not counts.
scVI expects nonnegative count-like input. Refer to the Seurat/Scanpy object and find the ['counts'] layer to proceed. 
### Query mapping
Depending on scvi-tools version, query models may require:

```python
q.train(max_epochs=1)
```

before calling `get_latent_representation()`.

---

## License

MIT
