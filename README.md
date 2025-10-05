# GSS Stigma-Related Nonresponse Analysis

## Overview
This pipeline analyzes stigma-related nonresponse patterns in the General Social Survey (GSS) data using predictive modeling and inverse probability weighting (IPW) for bias adjustment.

## Requirements
- Python 3.x
- pandas, numpy, scikit-learn, matplotlib

## Data
- GSS data file: `gss7224_r1.dta` (1972-2024)
- Note: Column names in the data are lowercase

## Usage

### Basic Commands

**Single Item Mode (original approach):**
```bash
python gss_stigma_starter.py --data "data/gss7224_r1.dta" --out outputs_single --predictors age educ sex race region year relig attend polviews income marital --items sexornt premarsx xmarsex homosex --mode single
```

**Composite Mode (any item nonresponse):**
```bash
python gss_stigma_starter.py --data "data/gss7224_r1.dta" --out outputs_composite --predictors age educ sex race region year relig attend polviews income marital --items sexornt premarsx xmarsex homosex --mode composite
```

### Parameters
- `--data`: Path to GSS .dta file
- `--out`: Output directory
- `--predictors`: Predictor variables (use lowercase names)
- `--items`: Stigma-related items to analyze (use lowercase names)
- `--mode`: Modeling approach - "single" (first item only) or "composite" (any item nonresponse, default)
- `--bootstrap`: Number of bootstrap iterations (default: 500)

### Available Variables
**Predictors:** age, educ, sex, race, region, year, relig, attend, polviews, income, marital

**Stigma Items:** sexornt, premarsx, xmarsex, homosex

## Output Files
- `model_metrics.json`: Model performance metrics
- `adjustment_comparison.csv`: IPW bias adjustment results
- `mds_nonresponse.csv`: MDS clustering coordinates
- `mds_nonresponse.png`: MDS visualization
- `run_log.json`: Complete run parameters and results

## Key Results

### Single Mode Results
- Gradient boosting achieved 76.9% accuracy (AUC: 0.725) in predicting sexual orientation nonresponse
- Logistic regression achieved 76.2% accuracy (AUC: 0.678)
- Predicts who will refuse to answer the first sensitive item (sexual orientation)

### Composite Mode Results  
- Both models achieved 95.6% accuracy in predicting any sexuality topic nonresponse
- Gradient boosting AUC: 0.675, Logistic regression AUC: 0.633
- Predicts who will refuse to answer any sexuality-related question

### Common Findings
- IPW adjustment revealed small but significant bias in sexual orientation responses
- MDS analysis shows sexual orientation has distinct nonresponse patterns compared to other sexual behavior items

## Troubleshooting
If you get "No predictor variables found" error, ensure you're using lowercase column names that match the actual GSS data structure.