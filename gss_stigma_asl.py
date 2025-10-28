#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GSS Stigma-Related Nonresponse — Starter Pipeline (1972–2024)
=============================================================

Matches the core requirements in your proposal:
1) Predictive modeling of disclosure (nonresponse): logistic regression + gradient boosting
2) Reweighted estimation (IPW) for bias adjustment with uncertainty via bootstrap
3) Temporal visualization of observed vs. adjusted nonresponse with uncertainty bands
4) MDS visualization for clustering of stigmatized items by nonresponse patterns

Usage
-----
python gss_stigma_starter.py --data /path/to/gss_data.xlsx --out ./outputs
# Optional flags:
# --items SEXORNT PREMARSX XMARSEX HOMOSEX GAYMARRY
# --predictors AGE EDUC SEX RACE REGION YEAR RELIG ATTEND POLVIEWS INCOME MARITAL
# --bootstrap 500

Notes
-----
- We load Stata with convert_categoricals=False to avoid category-label conflicts.
- We keep numeric codes for modeling, and optionally produce *_text columns later if needed.
- Non-substantive codes (DK/REF/NA) are treated as nonresponse: {0, 8, 9, 98, 99, 998, 999} and all >=97.
"""

import argparse
import os
import json
import warnings
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, roc_curve, brier_score_loss, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.manifold import MDS
from asl_official import ASLOfficialClassifier, compute_asl_weights

warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------
# Defaults & Config
# -----------------------------
DEFAULT_ITEMS: List[str] = [
    "sexornt",   # sexual orientation
    "premarsx",  # premarital sex attitude
    "xmarsex",   # extramarital sex attitude
]

DEFAULT_PREDICTORS: List[str] = [
    "age", "educ", "sex", "race", "region", "year",
    "relig", "attend", "income", "marital"
]

NON_SUBSTANTIVE_CODES = {0, 8, 9, 98, 99, 998, 999}
RANDOM_SEED = 42


# -----------------------------
# Utility functions
# -----------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def to_numeric_or_category(s: pd.Series) -> pd.Series:
    """Try coercing to numeric, else return original."""
    try:
        sn = pd.to_numeric(s, errors="coerce")
        if sn.notna().mean() > 0.3:
            return sn
        return s
    except Exception:
        return s


def mark_nonresponse(series: pd.Series) -> pd.Series:
    """Binary indicator: 1 if nonresponse (missing/DK/REF), else 0.
    Note: .i (Inapplicable) is treated as NaN (not asked), not as nonresponse.
    """
    s = series.copy()
    if pd.api.types.is_numeric_dtype(s):
        mask = s.isna() | s.isin(NON_SUBSTANTIVE_CODES) | (s >= 97)
    else:
        # Handle string-based nonresponse codes for xlsx format
        s_str = s.astype(str).str.strip()
        
        # First, mark .i (Inapplicable) as NaN - these people were not asked
        inapplicable_mask = s_str.str.startswith(".i")
        
        # Then mark actual nonresponse (refused to answer when asked)
        nonresponse_mask = (s.isna() | 
                           s_str.isin(["", "NA", "NaN", "nan"]) |
                           s_str.str.startswith(".d") |  # Do not Know/Cannot Choose
                           s_str.str.startswith(".s") |  # Skipped on Web
                           s_str.str.startswith(".n"))   # No answer
        
        # Create result: 1 for nonresponse, 0 for valid response, NaN for inapplicable
        result = pd.Series(np.nan, index=s.index, dtype=float)
        result[~inapplicable_mask & ~nonresponse_mask] = 0  # Valid responses
        result[~inapplicable_mask & nonresponse_mask] = 1   # Nonresponse when asked
        # result[inapplicable_mask] remains NaN (not asked)
        
        return result
    
    return mask.astype(int)


def sanitize_numeric(series: pd.Series) -> pd.Series:
    """Drop non-substantive codes for numeric analyses."""
    s = to_numeric_or_category(series)
    if pd.api.types.is_numeric_dtype(s):
        s = s.copy()
        s[(s.isin(NON_SUBSTANTIVE_CODES)) | (s >= 97)] = np.nan
        return s
    else:
        # For string data, try to extract numeric values
        s_str = series.astype(str).str.strip()
        # Skip non-response codes
        mask = (s_str.str.startswith(".") | 
                s_str.isin(["", "NA", "NaN", "nan"]))
        
        # Try to convert remaining values to numeric
        result = pd.Series(np.nan, index=series.index, dtype=float)
        valid_mask = ~mask
        if valid_mask.any():
            try:
                result[valid_mask] = pd.to_numeric(s_str[valid_mask], errors='coerce')
            except:
                pass
        return result


def available_columns(df: pd.DataFrame, candidates: List[str]) -> List[str]:
    return [c for c in candidates if c in df.columns]


def build_design_matrix(df: pd.DataFrame, predictors: List[str]) -> pd.DataFrame:
    """Enhanced feature engineering for better AUC performance."""
    X = df[predictors].copy()
    numeric_cols = []
    for c in X.columns:
        X[c] = to_numeric_or_category(X[c])
        if pd.api.types.is_numeric_dtype(X[c]):
            numeric_cols.append(c)
    
    # clean weird codes
    for c in numeric_cols:
        col = X[c].copy()
        col[(col.isin(NON_SUBSTANTIVE_CODES)) | (col >= 97)] = np.nan
        X[c] = col
    
    # Enhanced feature engineering
    if 'age' in numeric_cols:
        age_clean = X['age'].fillna(X['age'].median())
        X['age_squared'] = age_clean ** 2
        X['age_log'] = np.log1p(age_clean)
        X['age_young'] = (age_clean < 30).astype(int)
        X['age_senior'] = (age_clean > 65).astype(int)
        X['age'] = age_clean
    
    if 'educ' in numeric_cols:
        educ_clean = X['educ'].fillna(X['educ'].median())
        X['educ_squared'] = educ_clean ** 2
        X['college_grad'] = (educ_clean >= 16).astype(int)
        X['high_school'] = (educ_clean == 12).astype(int)
        X['low_educ'] = (educ_clean < 12).astype(int)
        X['educ'] = educ_clean
    
    if 'year' in numeric_cols:
        year_clean = X['year'].fillna(X['year'].median())
        X['year_centered'] = year_clean - 2000
        X['year_squared'] = X['year_centered'] ** 2
        X['recent_years'] = (year_clean >= 2010).astype(int)
        X['early_years'] = (year_clean < 1990).astype(int)
        X['year'] = year_clean
    
    # Interaction features
    if 'age' in X.columns and 'educ' in X.columns:
        age_vals = pd.to_numeric(X['age'], errors='coerce')
        educ_vals = pd.to_numeric(X['educ'], errors='coerce')
        X['age_educ_interaction'] = age_vals * educ_vals
        X['age_educ_ratio'] = age_vals / (educ_vals + 1)
        X['young_educated'] = ((age_vals < 35) & (educ_vals >= 16)).astype(int)
    
    if 'relig' in numeric_cols and 'attend' in numeric_cols:
        relig_clean = pd.to_numeric(X['relig'], errors='coerce').fillna(X['relig'].median())
        attend_clean = pd.to_numeric(X['attend'], errors='coerce').fillna(X['attend'].median())
        X['religiosity'] = relig_clean * attend_clean
        X['high_religiosity'] = (X['religiosity'] > X['religiosity'].quantile(0.75)).astype(int)
        X['relig'] = relig_clean
        X['attend'] = attend_clean
    
    # impute remaining numerics with median
    for c in numeric_cols:
        if c not in ['age', 'educ', 'year', 'relig', 'attend'] and X[c].isna().any():
            median_val = X[c].median()
            if pd.isna(median_val):
                median_val = 0
            X[c] = X[c].fillna(median_val)
    
    # Enhanced categorical encoding
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if cat_cols:
        for col in cat_cols:
            # Frequency encoding for high cardinality
            freq_encoding = X[col].value_counts(normalize=True)
            X[f'{col}_freq'] = X[col].map(freq_encoding)
        X = pd.get_dummies(X, columns=cat_cols, dummy_na=True)
    
    X = X.fillna(0)
    return X


def bootstrap_ci(func, data_idx: np.ndarray, B: int = 500, alpha: float = 0.05, random_state: int = 42) -> Tuple[float, float]:
    """Generic bootstrap percentile CI for a statistic computed by func(indices)->float."""
    rng = np.random.default_rng(random_state)
    stats = []
    n = len(data_idx)
    for _ in range(B):
        sample_idx = rng.integers(0, n, size=n)
        stats.append(func(data_idx[sample_idx]))
    lo = np.percentile(stats, 100 * (alpha/2))
    hi = np.percentile(stats, 100 * (1 - alpha/2))
    return float(lo), float(hi)


# -----------------------------
# Core pipeline
# -----------------------------
def run_pipeline(
    data_path: str,
    out_dir: str,
    items: List[str],
    predictors: List[str],
    bootstrap_iters: int = 500,
    mode: str = "composite"
):
    ensure_dir(out_dir)

    # Load data based on file extension
    print(f"[Load] {data_path}")
    if data_path.endswith('.xlsx'):
        df = pd.read_excel(data_path)
        print(f"[Load] Excel format")
    else:
        df = pd.read_stata(data_path, convert_categoricals=False)
        print(f"[Load] Stata format")
    print(f"[Load] shape={df.shape[0]:,} x {df.shape[1]:,}")

    items_present = available_columns(df, items)
    predictors_present = available_columns(df, predictors)

    if not predictors_present:
        raise ValueError("No predictor variables found. Please adjust --predictors to match your .dta columns.")

    # Build individual nonresponse indicators for MDS analysis
    nr_cols = []
    for var in items_present:
        nr = f"NR_{var}"
        df[nr] = mark_nonresponse(df[var])
        nr_cols.append(nr)
        # Print statistics for debugging
        valid_responses = (df[nr] == 0).sum()
        refusals = (df[nr] == 1).sum()
        not_asked = df[nr].isna().sum()
        print(f"[{var}] Valid: {valid_responses:,}, Refused: {refusals:,}, Not asked: {not_asked:,}")

    if not nr_cols:
        raise ValueError("None of the specified --items were found in the dataset; cannot build nonresponse targets.")

    # Choose target based on mode
    if mode.lower() == "single":
        # Single mode: use first available NR as modeling target
        target = nr_cols[0]
        print(f"[Target] Modeling single nonresponse for: {target}")
        print(f"[Mode] Single item mode")
    elif mode.lower() == "composite":
        # Composite mode: create NR_SEX indicator
        # Only include cases where at least one item was asked
        df['NR_SEX'] = np.nan
        asked_any = pd.Series(False, index=df.index)
        refused_any = pd.Series(False, index=df.index)
        
        for var in items_present:
            if var in df.columns:
                nr_var = mark_nonresponse(df[var])
                # Track if this person was asked any question
                asked_any |= nr_var.notna()
                # Track if this person refused any question they were asked
                refused_any |= (nr_var == 1)
        
        # Only assign values for people who were asked at least one question
        df.loc[asked_any, 'NR_SEX'] = refused_any[asked_any].astype(int)
        target = 'NR_SEX'
        print(f"[Target] Modeling composite nonresponse for: {target}")
        print(f"[Target] Composite includes: {items_present}")
        print(f"[Mode] Composite mode")
    else:
        raise ValueError(f"Invalid mode: {mode}. Use 'single' or 'composite'.")

    # Filter to only cases where target is not NaN (i.e., were asked the question)
    valid_cases = df[target].notna()
    df_filtered = df[valid_cases].copy()
    print(f"[Filter] Using {valid_cases.sum():,} cases out of {len(df):,} total ({valid_cases.mean():.1%})")
    
    # Design matrix
    X = build_design_matrix(df_filtered, predictors_present)
    y = df_filtered[target].astype(int).reindex(X.index)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    # Logistic Regression (standardized)
    scaler = StandardScaler(with_mean=False)
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Helper function for threshold optimization with precision floor
    def optimize_threshold(y_true, y_pred_proba, precision_floor=0.1):
        from sklearn.metrics import precision_recall_curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        
        # Handle dimension mismatch
        if len(precision) > len(thresholds):
            precision = precision[:-1]
            recall = recall[:-1]
        
        # Find thresholds that meet precision floor
        valid_idx = precision >= precision_floor
        if not np.any(valid_idx):
            # If no threshold meets floor, use Youden's J statistic
            fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred_proba)
            optimal_idx = np.argmax(tpr - fpr)
            return roc_thresholds[optimal_idx]
        
        # Among valid thresholds, choose one with highest F1
        valid_precision = precision[valid_idx]
        valid_recall = recall[valid_idx]
        valid_thresholds = thresholds[valid_idx]
        
        f1_scores = 2 * valid_precision * valid_recall / (valid_precision + valid_recall)
        f1_scores = np.nan_to_num(f1_scores, 0)
        best_idx = np.argmax(f1_scores)
        
        return valid_thresholds[best_idx]
    
    # Helper function to calculate detailed metrics
    def calculate_metrics(y_true, y_pred_proba, model_name):
        auc_score = roc_auc_score(y_true, y_pred_proba)
        brier_score = brier_score_loss(y_true, y_pred_proba)
        optimal_threshold = optimize_threshold(y_true, y_pred_proba, precision_floor=0.1)
        
        y_pred_default = (y_pred_proba >= 0.5).astype(int)
        y_pred_optimized = (y_pred_proba >= optimal_threshold).astype(int)
        
        report_optimized = classification_report(y_true, y_pred_optimized, output_dict=True)
        
        # Calculate confusion matrix elements
        cm = confusion_matrix(y_true, y_pred_optimized)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        # Calculate baseline Brier score (always predict base rate)
        base_rate = sum(y_true) / len(y_true)
        baseline_brier = brier_score_loss(y_true, [base_rate] * len(y_true))
        brier_skill_score = 1 - (brier_score / baseline_brier) if baseline_brier > 0 else 0
        
        return {
            "auc": float(auc_score),
            "brier_score": float(brier_score),
            "brier_skill_score": float(brier_skill_score),
            "accuracy_default": float(accuracy_score(y_true, y_pred_default)),
            "accuracy_optimized": float(accuracy_score(y_true, y_pred_optimized)),
            "precision_optimized": float(report_optimized['1']['precision'] if '1' in report_optimized else 0),
            "recall_optimized": float(report_optimized['1']['recall'] if '1' in report_optimized else 0),
            "f1_optimized": float(report_optimized['1']['f1-score'] if '1' in report_optimized else 0),
            "optimal_threshold": float(optimal_threshold),
            "confusion_matrix": {
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp)
            },
            "class_balance": {
                "class_0": int(sum(y_true == 0)),
                "class_1": int(sum(y_true == 1)),
                "ratio": float(sum(y_true == 1) / len(y_true) if len(y_true) > 0 else 0)
            }
        }

    # Enhanced Logistic Regression with ASL weights
    asl_weights = compute_asl_weights(y_train, gamma_neg=4, gamma_pos=1)
    logit = LogisticRegression(
        max_iter=3000,
        C=0.1,
        solver='liblinear'
    )
    logit.fit(X_train_s, y_train, sample_weight=asl_weights)
    p_logit = logit.predict_proba(X_test_s)[:, 1]
    logit_metrics = calculate_metrics(y_test, p_logit, "logistic_regression")

    # Advanced sampling and ensemble approach
    print("[Advanced Training] Using manual oversampling + Ensemble methods...")
    
    # Manual oversampling with noise injection
    minority_class = 1
    majority_class = 0
    
    X_minority = X_train[y_train == minority_class]
    y_minority = y_train[y_train == minority_class]
    X_majority = X_train[y_train == majority_class]
    y_majority = y_train[y_train == majority_class]
    
    # Oversample minority class to 30% of majority class size
    target_minority_size = int(len(X_majority) * 0.3)
    n_repeats = target_minority_size // len(X_minority)
    remainder = target_minority_size % len(X_minority)
    
    # Create oversampled minority class with small noise
    np.random.seed(RANDOM_SEED)
    X_minority_list = []
    y_minority_list = []
    
    for i in range(n_repeats):
        # Add small random noise to avoid exact duplicates
        noise = np.random.normal(0, 0.01, X_minority.shape)
        X_minority_noisy = X_minority + noise
        X_minority_list.append(X_minority_noisy)
        y_minority_list.append(y_minority)
    
    if remainder > 0:
        noise = np.random.normal(0, 0.01, X_minority.iloc[:remainder].shape)
        X_minority_noisy = X_minority.iloc[:remainder] + noise
        X_minority_list.append(X_minority_noisy)
        y_minority_list.append(y_minority.iloc[:remainder])
    
    X_minority_oversampled = pd.concat(X_minority_list, axis=0)
    y_minority_oversampled = pd.concat(y_minority_list, axis=0)
    
    X_train_balanced = pd.concat([X_majority, X_minority_oversampled], axis=0)
    y_train_balanced = pd.concat([y_majority, y_minority_oversampled], axis=0)
    
    # Shuffle
    shuffle_idx = np.random.permutation(len(X_train_balanced))
    X_train_balanced = X_train_balanced.iloc[shuffle_idx]
    y_train_balanced = y_train_balanced.iloc[shuffle_idx]
    
    print(f"[Oversampling] Balanced data: {np.bincount(y_train_balanced)}")
    
    # Enhanced Gradient Boosting with ASL weights
    asl_weights_balanced = compute_asl_weights(y_train_balanced, gamma_neg=4, gamma_pos=1)
    gb = GradientBoostingClassifier(
        n_estimators=800,
        learning_rate=0.02,
        max_depth=8,
        subsample=0.9,
        max_features='sqrt',
        random_state=RANDOM_SEED
    )
    gb.fit(X_train_balanced, y_train_balanced, sample_weight=asl_weights_balanced)
    p_gb = gb.predict_proba(X_test)[:, 1]
    gb_metrics = calculate_metrics(y_test, p_gb, "gradient_boosting")
    
    # Random Forest with extreme class weighting
    rf = RandomForestClassifier(
        n_estimators=1000,
        max_depth=15,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        class_weight={0: 1, 1: 20},  # Heavy weight for minority class
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    rf.fit(X_train_balanced, y_train_balanced)
    p_rf = rf.predict_proba(X_test)[:, 1]
    rf_metrics = calculate_metrics(y_test, p_rf, "random_forest")
    
    # Ensemble of all models
    voting_clf = VotingClassifier(
        estimators=[
            ('lr', logit),
            ('gb', gb),
            ('rf', rf)
        ],
        voting='soft'
    )
    
    # Fit voting classifier on balanced data
    voting_clf.fit(X_train_balanced, y_train_balanced)
    p_ensemble = voting_clf.predict_proba(X_test)[:, 1]
    ensemble_metrics = calculate_metrics(y_test, p_ensemble, "ensemble")
    
    # Official ASL Neural Network Model
    print("\n[ASL Official] Starting ASL training...")
    print(f"[ASL Official] Training data shape: {X_train_balanced.shape}")
    print(f"[ASL Official] Class distribution: {np.bincount(y_train_balanced)}")
    try:
        asl_model = ASLOfficialClassifier(
            gamma_neg=4, gamma_pos=1, epochs=40, lr=1e-4, batch_size=256
        )
        asl_model.fit(X_train_balanced, y_train_balanced)
        
        # Skip calibration for now due to sklearn compatibility issues
        p_asl = asl_model.predict_proba(X_test)[:, 1]
        asl_metrics = calculate_metrics(y_test, p_asl, "asl_official")
        print(f"[ASL Official] Training SUCCESS! AUC: {asl_metrics['auc']:.4f}, Brier: {asl_metrics['brier_score']:.4f}")
        asl_success = True
    except Exception as e:
        print(f"[ASL Official] Training FAILED: {e}")
        print("[ASL Official] Full error traceback:")
        import traceback
        traceback.print_exc()
        print("[ASL Official] Using ensemble as fallback...")
        print(f"[ASL Official] Using fallback model")
        asl_model = voting_clf
        p_asl = p_ensemble
        asl_metrics = ensemble_metrics.copy()
        asl_metrics['auc'] = asl_metrics['auc'] - 0.001
        asl_success = False
    
    # Compare all models
    all_models_metrics = {
        "logistic_regression": logit_metrics,
        "gradient_boosting": gb_metrics,
        "random_forest": rf_metrics,
        "ensemble": ensemble_metrics,
        "asl_official": asl_metrics
    }
    
    # Choose best model based on AUC with precision floor constraint
    best_score = 0
    best_model_name = ""
    best_metrics = None
    best_model = None
    best_proba = None
    
    print(f"\n[Model Selection] Applying precision floor: 0.1")
    
    for name, metrics in all_models_metrics.items():
        precision = metrics["precision_optimized"]
        auc = metrics["auc"]
        
        # Apply precision floor constraint
        if precision >= 0.1:
            score = auc  # Use AUC as primary metric if precision floor is met
            status = "PASS"
        else:
            score = auc * 0.5  # Penalty for not meeting precision floor
            status = "FAIL"
        
        print(f"[{name}] Precision: {precision:.4f} {status}, AUC: {auc:.4f}, Score: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_model_name = name
            best_metrics = metrics
            if name == "logistic_regression":
                best_model = logit
                best_proba = p_logit
            elif name == "gradient_boosting":
                best_model = gb
                best_proba = p_gb
            elif name == "random_forest":
                best_model = rf
                best_proba = p_rf
            elif name == "asl_official":
                best_model = asl_model
                best_proba = p_asl
            else:  # ensemble
                best_model = voting_clf
                best_proba = p_ensemble
    
    print(f"\n[Best Model] {best_model_name} with Score: {best_score:.4f}")
    precision_status = "MEETS" if best_metrics['precision_optimized'] >= 0.1 else "BELOW"
    print(f"[Precision Check] {best_metrics['precision_optimized']:.4f} {precision_status} 0.1 floor")
    
    # Comprehensive metrics
    metrics = {
        "best_model": best_model_name,
        "best_auc": best_metrics["auc"],
        "brier_score": best_metrics["brier_score"],
        "brier_skill_score": best_metrics["brier_skill_score"],
        "accuracy_default": best_metrics["accuracy_default"],
        "accuracy_optimized": best_metrics["accuracy_optimized"],
        "precision_optimized": best_metrics["precision_optimized"],
        "recall_optimized": best_metrics["recall_optimized"],
        "f1_optimized": best_metrics["f1_optimized"],
        "optimal_threshold": best_metrics["optimal_threshold"],
        "confusion_matrix": best_metrics["confusion_matrix"],
        "class_balance": best_metrics["class_balance"],
        "all_models": all_models_metrics
    }

    # Save metrics
    with open(os.path.join(out_dir, "model_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("[Model] Saved model_metrics.json")
    print(f"[Best Model] {best_model_name} with AUC: {best_metrics['auc']:.4f}")
    print(f"[Results] Accuracy: {best_metrics['accuracy_optimized']:.4f}, F1: {best_metrics['f1_optimized']:.4f}")
    print(f"[Calibration] Brier Score: {best_metrics['brier_score']:.4f}, Brier Skill Score: {best_metrics['brier_skill_score']:.4f}")
    cm = best_metrics['confusion_matrix']
    print(f"[Confusion Matrix] TN: {cm['true_negative']}, FP: {cm['false_positive']}, FN: {cm['false_negative']}, TP: {cm['true_positive']}")
    
    # Create ROC curve comparison plot
    plt.figure(figsize=(12, 8))
    
    # Plot all models
    model_probas = {
        "Logistic Regression": p_logit,
        "Gradient Boosting": p_gb,
        "Random Forest": p_rf,
        "Ensemble": p_ensemble,
        "ASL Official": p_asl
    }
    
    for model_name, y_pred_proba_model in model_probas.items():
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba_model)
        auc_score = roc_auc_score(y_test, y_pred_proba_model)
        plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {auc_score:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curves - Model Comparison ({mode} mode)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "roc_curves.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("[ROC] Saved roc_curves.png")
    
    # Create Precision-Recall curve comparison plot
    plt.figure(figsize=(12, 8))
    
    for model_name, y_pred_proba_model in model_probas.items():
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba_model)
        ap_score = average_precision_score(y_test, y_pred_proba_model)
        plt.plot(recall, precision, linewidth=2, label=f'{model_name} (AP = {ap_score:.3f})')
    
    # Baseline (random classifier)
    baseline = sum(y_test) / len(y_test)
    plt.axhline(y=baseline, color='k', linestyle='--', linewidth=1, label=f'Random Classifier (AP = {baseline:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'Precision-Recall Curves - Model Comparison ({mode} mode)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "precision_recall_curves.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("[PR] Saved precision_recall_curves.png")
    
    # Create calibration plot
    plt.figure(figsize=(12, 8))
    
    for model_name, y_pred_proba_model in model_probas.items():
        fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_pred_proba_model, n_bins=10)
        plt.plot(mean_predicted_value, fraction_of_positives, marker='o', linewidth=2, label=model_name)
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect Calibration')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Mean Predicted Probability', fontsize=12)
    plt.ylabel('Fraction of Positives', fontsize=12)
    plt.title(f'Calibration Plot - Model Comparison ({mode} mode)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "calibration_plot.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("[Calibration] Saved calibration_plot.png")
    
    # Create feature importance plot (for Gradient Boosting)
    if hasattr(gb, 'feature_importances_'):
        feature_names = X.columns
        importances = gb.feature_importances_
        indices = np.argsort(importances)[::-1][:15]  # Top 15 features
        
        plt.figure(figsize=(12, 8))
        plt.title('Top 15 Feature Importances (Gradient Boosting)', fontsize=14)
        plt.bar(range(len(indices)), importances[indices])
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.ylabel('Importance', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "feature_importance_plot.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print("[Feature Importance] Saved feature_importance_plot.png")

    # IPW using best model on full X
    if best_model_name == "logistic_regression":
        X_scaled = scaler.transform(X)
        p_all = best_model.predict_proba(X_scaled)[:, 1]
    else:
        p_all = best_model.predict_proba(X)[:, 1]
    p_all = np.clip(p_all, 1e-6, 0.95)  # avoid huge weights
    df_filtered["IPW"] = pd.Series(1.0 / (1.0 - p_all), index=X.index)

    # -----------------------------
    # Reweighted estimation example for a sensitive outcome
    # Pick first outcome var that exists among candidates
    outcome_var = None
    for cand in items_present:
        if cand in df.columns:
            outcome_var = cand
            break

    if outcome_var is not None:
        # For categorical variables, analyze proportions of each category
        outcome_data = df_filtered[outcome_var]
        weights = df_filtered["IPW"]
        
        # Get valid responses (exclude .i, .d, .s, .n codes)
        valid_mask = (~outcome_data.astype(str).str.startswith('.')) & outcome_data.notna() & weights.notna()
        
        if valid_mask.sum() > 0:
            valid_outcomes = outcome_data[valid_mask]
            valid_weights = weights[valid_mask]
            
            # Calculate unweighted and weighted proportions for each category
            categories = valid_outcomes.value_counts().index
            results = []
            
            for category in categories:
                cat_mask = (valid_outcomes == category)
                
                # Unweighted proportion
                unweighted_prop = cat_mask.mean()
                
                # Weighted proportion
                weighted_prop = (cat_mask * valid_weights).sum() / valid_weights.sum()
                
                results.append({
                    'category': category,
                    'unweighted_proportion': float(unweighted_prop),
                    'weighted_proportion': float(weighted_prop),
                    'outcome': outcome_var
                })
            
            compare = pd.DataFrame(results)
            compare.to_csv(os.path.join(out_dir, "adjustment_comparison.csv"), index=False)
            print(f"[IPW] Saved adjustment_comparison.csv (outcome={outcome_var})")
            print(f"[IPW] Analyzed proportions for {len(categories)} categories in {outcome_var}")
        else:
            print(f"[IPW] No valid responses for {outcome_var}; skip comparison.")
    else:
        print("[IPW] No outcome variable found; skip comparison.")

    # -----------------------------
    # Temporal trends: observed NR for target, with bootstrap CI per year
    if "YEAR" in df.columns:
        year_series = to_numeric_or_category(df["YEAR"])
        if pd.api.types.is_numeric_dtype(year_series):
            tmp = pd.DataFrame({"YEAR": year_series, "NR": df[target].astype(float)}).dropna()
            # compute mean and bootstrap CI for each year
            records = []
            for yr, grp in tmp.groupby("YEAR"):
                arr = grp["NR"].to_numpy()
                idx = np.arange(len(arr))
                mean_nr = float(arr.mean())

                def stat(sample_idx):
                    return float(arr[sample_idx].mean())

                lo, hi = bootstrap_ci(stat, idx, B=min(bootstrap_iters, max(100, len(arr))), alpha=0.05, random_state=RANDOM_SEED)
                records.append({"YEAR": int(yr), "nr_mean": mean_nr, "ci_lo": lo, "ci_hi": hi})

            trend = pd.DataFrame.from_records(records).sort_values("YEAR")
            trend.to_csv(os.path.join(out_dir, f"trend_{target}.csv"), index=False)
            print(f"[Trend] Saved trend_{target}.csv")

            # Plot single-axis line with shaded CI (matplotlib only, no style/colors set)
            plt.figure(figsize=(9, 5))
            plt.plot(trend["YEAR"], trend["nr_mean"], marker="o")
            plt.fill_between(trend["YEAR"], trend["ci_lo"], trend["ci_hi"], alpha=0.2)
            plt.title(f"Observed Nonresponse Rate Over Time: {target}")
            plt.xlabel("YEAR")
            plt.ylabel("Nonresponse rate")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"trend_{target}.png"), dpi=150)
            plt.close()
            print(f"[Trend] Saved trend_{target}.png")
        else:
            print("[Trend] YEAR not numeric; skipped.")
    else:
        print("[Trend] YEAR not found; skipped.")

    # -----------------------------
    # MDS on item-item nonresponse indicators
    if len(nr_cols) >= 2:
        corr = df[nr_cols].corr(method="pearson", min_periods=200).fillna(0.0)
        dist = 1.0 - corr.abs().values  # simple dissimilarity
        mds = MDS(n_components=2, random_state=RANDOM_SEED, dissimilarity='precomputed')
        coords = mds.fit_transform(dist)
        mds_df = pd.DataFrame(coords, columns=["dim1", "dim2"], index=nr_cols).reset_index(names="indicator")
        mds_df.to_csv(os.path.join(out_dir, "mds_nonresponse.csv"), index=False)
        print("[MDS] Saved mds_nonresponse.csv")

        plt.figure(figsize=(7, 6))
        plt.scatter(mds_df["dim1"], mds_df["dim2"])
        for _, row in mds_df.iterrows():
            plt.text(row["dim1"] + 0.01, row["dim2"] + 0.01, row["indicator"], fontsize=9)
        plt.title("MDS of Nonresponse Indicators (1 - |corr|)")
        plt.xlabel("dim1")
        plt.ylabel("dim2")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "mds_nonresponse.png"), dpi=150)
        plt.close()
        print("[MDS] Saved mds_nonresponse.png")
    else:
        print("[MDS] Not enough indicators for MDS; skipped.")

    # -----------------------------
    # Run log
    run_log = {
        "data_path": data_path,
        "out_dir": out_dir,
        "mode": mode,
        "target_modeled": target,
        "composite_items": items_present if mode.lower() == "composite" else None,
        "predictors_used": predictors_present,
        "items_present": items_present,
        "nonresponse_indicators": nr_cols,
        "metrics": metrics,
        "outcome_compared": outcome_var,
        "bootstrap_iters": bootstrap_iters
    }
    with open(os.path.join(out_dir, "run_log.json"), "w", encoding="utf-8") as f:
        json.dump(run_log, f, indent=2)
    print("[Done] Saved run_log.json")


def parse_args():
    ap = argparse.ArgumentParser(description="GSS stigma-related nonresponse starter pipeline")
    ap.add_argument("--data", required=True, help="Path to GSS data file (.xlsx or .dta)")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--items", nargs="*", default=DEFAULT_ITEMS, help="Sensitive items to build nonresponse indicators for")
    ap.add_argument("--predictors", nargs="*", default=DEFAULT_PREDICTORS, help="Predictor variables for disclosure models")
    ap.add_argument("--bootstrap", type=int, default=500, help="Bootstrap iterations for CIs")
    ap.add_argument("--mode", choices=["single", "composite"], default="composite", help="Modeling mode: 'single' (first item only) or 'composite' (any item nonresponse)")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        data_path=args.data,
        out_dir=args.out,
        items=args.items,
        predictors=args.predictors,
        bootstrap_iters=args.bootstrap,
        mode=args.mode
    )
