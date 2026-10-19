# Data

The dataset is not included in this repository. Download it and place the CSV in `data/raw/`.

**Source:** Sepsis Survival Minimal Clinical Records
* UCI: https://archive.ics.uci.edu/dataset/827
* Kaggle: https://www.kaggle.com/datasets/mathurinache/sepsis-survival-minimal-clinical-records

**File used:** `s41598-020-73558-3_sepsis_survival_primary_cohort.csv` (110,204 rows)

Expected layout:

```
data/raw/s41598-020-73558-3_sepsis_survival_primary_cohort.csv
```

The study cohort and Korean validation cohort from the same release can be used for external validation by passing `--csv` to `scripts/run_pipeline.py`.

Reference: Chicco D., Jurman G. *Scientific Reports* 10, 17156 (2020). doi:10.1038/s41598-020-73558-3
