# ChronoMate

ChronoMate predicts Drosophila developmental time from single-cell RNA-seq across datasets/labs.

Current stable workflow:
- Train an XGBoost regressor on a labeled source dataset (TRAIN)
- Transfer to a target dataset (TEST) using shared-gene alignment + train-based standardization
- Evaluate with MAE and “nearest-allowed-time” accuracy
- Visualize performance with the violin/mean/y=x plot (see Figure)

Note: Some older parts of this repo may reference domain-adversarial training (DANN). The current stable pipeline is the XGBoost transfer method described below.

---

## Data sources (GEO)

ChronoMate aligns predicted and actual developmental time with high accuracy.

TRAIN (source): Kurmangaliyev et al., Neuron 2020
- GEO: GSE156455
- Link: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE156455

TEST (target): Özel et al., Nature 2021
- GEO: GSE142787
- Link: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE142787

---
## Installation

### 1. Create a virtual environment
<pre><code>
# Windows (PowerShell)
python -m venv .venv

# macOS/Linux
python3 -m venv .venv
</code></pre>

### 2. Activate the virtual environment
<pre><code>
# Windows (PowerShell)
\.venv\Scripts\Activate

# macOS/Linux
source .venv/bin/activate
</code></pre>

### 3. Install dependencies
<pre><code>
pip install -r requirements.txt
</code></pre>

### 4. (Optional) Install the package itself
<pre><code>
pip install .
</code></pre>

### 5. Verify installation
<pre><code>
# Should print a help message with available commands
python -m chronomate.cli --help
</code></pre>

---

## Method overview (XGBoost transfer)

Inputs:
- TRAIN CSV: cells x genes with labels
  Required columns: cell_id, time
  Optional columns: sample, type
  Remaining columns: gene features (floats)

- TEST CSV: cells x genes (time optional if you are predicting; required if you are evaluating)

Training procedure:
1) Feature alignment
   - Compute the intersection of gene columns between TRAIN and TEST.
   - Use only the shared gene set for modeling.

2) Standardization (train-based)
   - Compute mean and std for each gene using TRAIN only.
   - Z-score TRAIN and TEST using those TRAIN statistics:
       Xz = (X - mu_train) / sd_train

3) Model fit
   - Train XGBRegressor to predict time (in hours) from standardized gene features.
   - Save:
       - trained model (joblib)
       - preprocessing bundle (genes + mu_train + sd_train)

Evaluation procedure:
- Predict continuous time for each TEST cell.
- Report:
  - MAE (hours)
  - Nearest-allowed-time accuracy:
      snap each prediction to the closest time in the TEST time grid (e.g., [15,30,40,50,70])
      then compute accuracy and MAE on the snapped values.

---

## Quick start

Train:
  python -m chronomate.cli train-xgb-zscore ^
    --train "C:\2024 Fall\chronocell\dann_data\train.csv" ^
    --test  "C:\2024 Fall\chronocell\GSE142787\test_gene_normalized.csv" ^
    --outdir "C:\2024 Fall\chronocell\runs\xgb_zscore"

Evaluate + plots:
  python -m chronomate.cli eval-xgb-zscore ^
    --test "C:\2024 Fall\chronocell\GSE142787\test_gene_normalized.csv" ^
    --model  "C:\2024 Fall\chronocell\runs\xgb_zscore\xgb_regressor_zscore.joblib" ^
    --preproc "C:\2024 Fall\chronocell\runs\xgb_zscore\xgb_regressor_zscore_preproc.joblib" ^
    --outdir "C:\2024 Fall\chronocell\runs\xgb_zscore\eval"

Outputs:
- metrics printed to console
- predictions CSV saved to disk
- plots displayed and/or saved (depending on CLI flags)

---

## Figure

![Figure 3: Predicted time vs actual time](./output_DAN_scVI.png)

---

## References

- Kurmangaliyev YZ et al. Transcriptional Programs of Circuit Assembly in the Drosophila Visual System. Neuron (2020). GEO: GSE156455
- Özel MN et al. Neuronal diversity and convergence in a visual system developmental atlas. Nature (2021). GEO: GSE142787

---

## License

MIT
