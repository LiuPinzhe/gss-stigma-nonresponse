# GSS Stigma-Related Nonresponse Analysis

## Overview
This pipeline analyzes stigma-related nonresponse patterns in the General Social Survey (GSS) data using predictive modeling and inverse probability weighting (IPW) for bias adjustment.

## Requirements
- Python 3.x
- pandas, numpy, scikit-learn, matplotlib

## Data
- GSS data file: `GSS.xlsx` (1972-2024) or `gss7224_r1.dta`
- Supports both Excel (.xlsx) and Stata (.dta) formats
- Note: Column names in the data are lowercase

## Usage

### Basic Commands

**Single Item Mode (original approach):**
```bash
python gss_stigma_starter.py --data "data/GSS.xlsx" --out outputs_single --predictors age educ sex race region year relig attend income marital --items sexornt premarsx xmarsex --mode single
```

**Composite Mode (any item nonresponse):**
```bash
python gss_stigma_starter.py --data "data/GSS.xlsx" --out outputs_composite --predictors age educ sex race region year relig attend income marital --items sexornt premarsx xmarsex --mode composite
```

**AUC-Optimized Version:**
```bash
python gss_stigma_starter11.py --data "data/GSS.xlsx" --out outputs_auc_optimized --mode composite
```

### Parameters
- `--data`: Path to GSS data file (.xlsx or .dta)
- `--out`: Output directory
- `--predictors`: Predictor variables (use lowercase names)
- `--items`: Stigma-related items to analyze (use lowercase names)
- `--mode`: Modeling approach - "single" (first item only) or "composite" (any item nonresponse, default)
- `--bootstrap`: Number of bootstrap iterations (default: 500)

### Available Variables
**Predictors:** age, educ, sex, race, region, year, relig, attend, income, marital

**Stigma Items:** sexornt, premarsx, xmarsex

### File Versions
- **gss_stigma_starter.py**: Basic version with logistic regression + gradient boosting
- **gss_stigma_starter11.py**: AUC-optimized version with advanced feature engineering and multiple models

## Output Files
- `model_metrics.json`: Detailed model performance metrics
- `adjustment_comparison.csv`: IPW bias adjustment results (categorical proportions)
- `mds_nonresponse.csv`: MDS clustering coordinates
- `mds_nonresponse.png`: MDS visualization
- `run_log.json`: Complete run parameters and results
- `feature_importance.csv`: Feature importance rankings (starter11 only)
- `roc_curves.png`: ROC curve comparison (starter11 only)

## Key Results

### Updated Results (Post-Correction)
**Data Statistics:**
- sexornt: Valid: 17,739, Refused: 458, Not asked: 57,502
- premarsx: Valid: 45,697, Refused: 1,391, Not asked: 28,611
- xmarsex: Valid: 46,266, Refused: 763, Not asked: 28,670

**Model Performance:**
- AUC: ~0.65 (moderate predictive ability)
- Class imbalance: 96.5% respond vs 3.5% refuse
- Optimal threshold: ~0.032 (very low due to imbalance)
- F1-Score: ~0.10 (challenging prediction task)

### Key Improvements
- Corrected nonresponse coding: `.i (Inapplicable)` now properly excluded
- Added support for Excel (.xlsx) format
- Enhanced model evaluation with threshold optimization
- Categorical IPW analysis for proportions rather than means

## Important Notes

### Nonresponse Coding
- `.i (Inapplicable)`: Not asked the question (excluded from analysis)
- `.d (Do not Know)`, `.s (Skipped)`, `.n (No answer)`: Actual nonresponse (included as refusal)
- This correction significantly reduces apparent refusal rates to realistic levels

### Model Interpretation
- High default accuracy (96%+) reflects extreme class imbalance, not model quality
- AUC (~0.65) indicates moderate but limited predictive ability
- Low optimal thresholds reflect the rarity of refusal behavior
- This is typical for social science nonresponse prediction

## Troubleshooting
- If you get "No predictor variables found" error, ensure you're using lowercase column names
- For file format issues, ensure data is in .xlsx or .dta format
- Missing variables will be automatically excluded with warnings