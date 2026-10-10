# Medical Cost Prediction Lab: Linear vs. Polynomial Regression

> **Portfolio focus:** controlled regression experiments, leakage-safe preprocessing, cross-validation, residual diagnostics, permutation importance, and subgroup error auditing.

## Research question
**Does polynomial regression improve prediction of medical insurance charges over linear regression, and what trade-offs arise as model complexity increases?**

This project uses the [Medical Cost Personal Dataset on Kaggle](https://www.kaggle.com/datasets/mirichoi0218/insurance). It is designed as a transparent, reproducible machine-learning investigation—not merely a notebook that calls `.fit()` and prints a score.

## Why this project is more than a basic tutorial

- Compares a mean baseline, age-only linear regression, multiple linear regression, polynomial degrees 2 and 3, and Ridge-regularized polynomial regression.
- Chooses a candidate using **5-fold cross-validation on training data**, then uses a held-out test split as a final generalization check.
- Keeps imputers, scaling, one-hot encoding, and polynomial expansion inside scikit-learn pipelines to prevent preprocessing leakage.
- Reports MAE, RMSE, R², train/test error gaps, residual diagnostics, and actual-versus-predicted plots.
- Adds permutation importance and an exploratory error audit by smoking status and broad age bands, including group sample counts.
- Exports reproducible metrics, figures, a run summary, and a fitted pipeline from the notebook.

## Dataset

Download `insurance.csv` from the original [Kaggle dataset page](https://www.kaggle.com/datasets/mirichoi0218/insurance). The expected columns are:

| Column | Meaning in this experiment | Role |
|---|---|---|
| `age` | Age | Predictor |
| `sex` | Recorded sex category | Predictor |
| `bmi` | Body mass index | Predictor |
| `children` | Number of children/dependents as recorded in the data | Predictor |
| `smoker` | Smoking-status category | Predictor |
| `region` | Region category in source dataset | Predictor |
| `charges` | Medical insurance charges | Target |

The CSV is not bundled here. Obtain it from Kaggle and follow the source's applicable terms. This repository does not claim ownership of the dataset.

## Quick start: Google Colab (recommended)

1. Open [`notebooks/medical_cost_regression.ipynb`](notebooks/medical_cost_regression.ipynb) in GitHub.
2. Click **Open in Colab** (or upload the notebook to [Google Colab](https://colab.research.google.com/)).
3. Run cells from top to bottom and upload `insurance.csv` when prompted.
4. Review the actual metrics, charts, and subgroup results. Add your own observations to the notebook and this README.
5. The last notebook cell bundles the generated metrics, run summary, and figures into `medical_cost_regression_results.zip` and starts a download in Colab. Extract those reviewed outputs and upload them to this repository if you want the figures/results displayed on GitHub.

No GPU is needed: this dataset is small and these regression models train quickly on a CPU. A Colab GPU is not expected to make this particular project faster in a meaningful way.

## Run locally (optional)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Place `insurance.csv` at the repository root or in `data/`, then open the notebook with Jupyter. The first data-loading cell looks in both locations.

## Methodology

1. **Inspect and validate:** column checks, missing-value summary, duplicate reporting, numeric summaries.
2. **Explore:** target distribution, age/charges and BMI/charges plots, smoker-group comparison, numeric correlation heatmap.
3. **Split once:** fixed 80/20 train/test split with seed 42.
4. **Preprocess inside pipelines:** median imputation and scaling for numeric features; most-frequent imputation and one-hot encoding for categorical features; polynomial features are applied to numeric predictors only.
5. **Select on training data:** 5-fold shuffled cross-validation, using mean absolute error (MAE). The test set is not used to choose the candidate.
6. **Evaluate:** compare MAE, RMSE and R²; inspect residuals and train/test gaps.
7. **Interpret cautiously:** permutation importance is model-dependent; subgroup errors are exploratory and include counts.

### Candidate models

- Mean-only `DummyRegressor` baseline
- Simple linear regression using `age`
- Multiple linear regression using all six predictors
- Polynomial regression (degree 2)
- Polynomial regression (degree 3)
- Ridge regression with degree-2 polynomial features

## Results

**Run the notebook before writing performance claims.** The repository intentionally contains no fabricated scores. After running it, copy the actual selected model and metrics here and include figures exported from `reports/figures/`.

| Item | Result |
|---|---|
| Selected candidate (lowest training CV MAE, excluding dummy baseline) | *Fill after running* |
| Held-out test MAE | *Fill after running* |
| Held-out test RMSE | *Fill after running* |
| Held-out test R² | *Fill after running* |
| Main diagnostic observation | *Fill after reviewing plots* |
| Subgroup audit observation | *Fill after reviewing counts and errors* |

### Figures to include after execution

The notebook creates these files:

- `reports/figures/01_target_distribution.png`
- `reports/figures/02_age_vs_charges.png`
- `reports/figures/03_bmi_vs_charges.png`
- `reports/figures/04_charges_by_smoker.png`
- `reports/figures/05_numeric_correlation.png`
- `reports/figures/06_test_mae_comparison.png`
- `reports/figures/07_train_cv_test_mae.png`
- `reports/figures/08_residuals_vs_predicted.png`
- `reports/figures/09_residual_distribution.png`
- `reports/figures/10_actual_vs_predicted.png`
- `reports/figures/11_permutation_importance.png`
- `reports/figures/12_subgroup_mae_smoker.png`
- `reports/figures/13_subgroup_mae_age_band.png`

## Repository layout

```text
medical-cost-regression/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── docs/
│   └── DISSECTION_GUIDE.md
├── notebooks/
│   └── medical_cost_regression.ipynb
├── reports/
│   └── figures/                 # generated by the notebook
└── models/                      # selected pipeline generated by the notebook
```

## Limitations and responsible interpretation

- This is a small, frequently used educational dataset; it is not sufficient evidence for a production underwriting model.
- `charges` may reflect insurer-specific and historical policies not represented in the features.
- Observed relationships are not causal effects.
- Overall scores can conceal different error rates across groups. The subgroup audit is descriptive, not a fairness certification; always read sample counts with group metrics.
- A single holdout split has uncertainty. Follow-up work should examine repeated validation, independent data, and uncertainty intervals before making stronger claims.

## Next experiments

1. Tune Ridge's regularization strength using training-only cross-validation.
2. Compare repeated cross-validation or repeated holdout splits for stability.
3. Investigate whether log-transforming the target is justified, with predictions transformed back before comparing errors on the original charge scale.
4. Add uncertainty intervals and validation on independent data.
5. Package the fitted pipeline in a small prediction interface only after the experimental workflow is understood.

## Disclaimer
Educational use only. Not medical advice, an authoritative insurance quote, or a basis for real pricing or eligibility decisions.
