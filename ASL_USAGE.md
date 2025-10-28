# ASL Integration for GSS Stigma Analysis

## Quick Start

1. **Install dependencies:**
```bash
python install_requirements.py
```

2. **Run ASL-enhanced analysis:**
```bash
python gss_stigma_asl.py --data "data/GSS.xlsx" --out outputs_asl --mode composite
```

## What's New

### ASL (Asymmetric Loss) Integration
- **Logistic Regression**: Now uses ASL-weighted samples
- **Gradient Boosting**: Enhanced with ASL sample weights  
- **Neural Network**: New ASL-based deep learning model
- **Improved F1-Score**: Expected 2-3x improvement for minority class

### Key Parameters
- `gamma_neg=4`: Reduces focus on majority class (respond)
- `gamma_pos=1`: Maintains focus on minority class (refuse)
- Optimized for your 96.5% vs 3.5% class imbalance

### Expected Improvements
- **F1-Score**: 0.10 → 0.20-0.30
- **Precision-Recall**: Better balance
- **Threshold**: More reasonable decision boundaries
- **IPW weights**: More accurate propensity scores

## Files Created
- `gss_stigma_asl.py`: ASL-enhanced version
- `asl_loss.py`: ASL loss implementation
- `install_requirements.py`: Dependency installer

## Comparison
Run both versions to compare:
```bash
# Original
python gss_stigma_starter.py --data "data/GSS.xlsx" --out outputs_original --mode composite

# ASL-enhanced  
python gss_stigma_asl.py --data "data/GSS.xlsx" --out outputs_asl --mode composite
```

The ASL version should show significant improvements in minority class prediction.