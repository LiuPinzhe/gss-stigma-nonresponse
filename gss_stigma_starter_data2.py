#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GSS Stigma-Related Nonresponse — Starter Pipeline with Enhanced Variables (Data2)
================================================================================

Based on gss_stigma_starter.py but using expanded variable set from starter13
and targeting GSS2.xlsx data file.

Usage
-----
python gss_stigma_starter_data2.py --data "data/GSS2.xlsx" --out outputs_data2 --mode composite
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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, roc_curve, brier_score_loss, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve
from sklearn.manifold import MDS

warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------
# Defaults & Config (Enhanced from starter13)
# -----------------------------
DEFAULT_ITEMS: List[str] = [
    "homosex"   # homosexual attitude (from starter13)
]

DEFAULT_PREDICTORS: List[str] = [
    "age", "educ", "sex", "race", "region", "year",
    "relig", "attend", "income", "marital", "divorce", 
    "wrkstat", "spwrksta", "childs", "speduc", "res16", "reg16", "rincome", "xnorcsiz", 
    "partyid", "polviews", "natenvir", "natheal", "natcity", "natcrime", "natdrug", 
    "nateduc", "natrace", "cappun", "gunlaw", "happy", "hapmar", "health", "life", 
    "sexeduc", "class_", "finrela", "owngun", "ethnic", "hispanic", "spaneng"
]

# Engineered numeric features we will force to numeric
CONTINUOUS_PREDICTORS: List[str] = [
    "age", "childs", "year", "year_centered", "year_squared",
    "age_clean", "age_squared", "age_log",
    "year_clean", "year_scaled", "decade"
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


def find_matching_columns(df: pd.DataFrame, candidates: List[str]) -> List[str]:
    """Find columns that match case-insensitively."""
    available_cols = set(df.columns.str.lower())
    matched = []
    for candidate in candidates:
        cl = candidate.lower()
        if cl in available_cols:
            actual = df.columns[df.columns.str.lower() == cl][0]
            matched.append(actual)
    return matched


def available_columns(df: pd.DataFrame, candidates: List[str]) -> List[str]:
    return find_matching_columns(df, candidates)


def column_category(s: pd.Series) -> str:
    """Return 'numeric' or 'categorical' for a column."""
    if s.name in CONTINUOUS_PREDICTORS:
        return "numeric"
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    return "categorical"


def create_advanced_features(df: pd.DataFrame, base_predictors: List[str]) -> pd.DataFrame:
    """Create engineered features from starter13."""
    features = df[base_predictors].copy()

    # age
    if "age" in features:
        age_clean = pd.to_numeric(features["age"], errors="coerce")
        features["age_clean"] = age_clean
        features["age_binned"] = pd.cut(age_clean, bins=[0, 30, 45, 60, 120], labels=[1, 2, 3, 4])
        features["age_squared"] = age_clean ** 2
        features["age_log"] = np.log1p(age_clean)

    # year
    if "year" in features:
        year_clean = pd.to_numeric(features["year"], errors="coerce")
        features["year_clean"] = year_clean
        # center around 2000
        features["year_centered"] = year_clean - 2000
        features["year_squared"] = (features["year_centered"]) ** 2
        features["decade"] = (year_clean // 10) * 10

    # Basic interaction example (only if both numeric-like present)
    if "age_clean" in features and "year_clean" in features:
        # age-year interaction sometimes helps for cohort/time effects
        features["age_year_interaction"] = features["age_clean"] * features["year_clean"]

    return features


def build_design_matrix(df: pd.DataFrame, predictors: List[str]) -> pd.DataFrame:
    """Enhanced feature engineering combining both approaches."""
    actual_predictors = find_matching_columns(df, predictors)
    X_advanced = create_advanced_features(df, actual_predictors)

    numeric_cols = []
    categorical_cols = []

    # Force engineered numeric features to numeric if present
    for feat in CONTINUOUS_PREDICTORS:
        if feat in X_advanced.columns:
            X_advanced[feat] = pd.to_numeric(X_advanced[feat], errors="coerce")

    # Classify and clean
    for c in X_advanced.columns:
        t = column_category(X_advanced[c])
        if t == "numeric":
            numeric_cols.append(c)
            # Coerce numeric; set obvious non-substantive numeric codes to NaN
            X_advanced[c] = pd.to_numeric(X_advanced[c], errors="coerce")
            mask = (X_advanced[c].isin(NON_SUBSTANTIVE_CODES)) | (X_advanced[c] >= 97) | X_advanced[c].isna()
            X_advanced.loc[mask, c] = np.nan
            # Fill numeric: median
            if X_advanced[c].notna().any():
                X_advanced[c] = X_advanced[c].fillna(X_advanced[c].median())
            else:
                X_advanced[c] = 0.0
        else:
            # Categorical: preserve refusal as 'NR', set inapplicable (.i...) to NaN
            categorical_cols.append(c)
            col = X_advanced[c].astype(str).str.strip()
            inapp_mask = col.str.startswith(".i")
            nr_mask = (
                col.isna() |
                col.isin(["", "NA", "NaN", "nan"]) |
                col.str.startswith(".d") |
                col.str.startswith(".s") |
                col.str.startswith(".n") |
                col.str.startswith(".r") |
                col.isin(map(str, NON_SUBSTANTIVE_CODES))
            )
            col[inapp_mask] = np.nan
            col[nr_mask] = "NR"
            X_advanced[c] = col

    # Missingness features from raw categorical values (before dummies)
    if categorical_cols:
        nr_hits = X_advanced[categorical_cols].apply(lambda s: s.astype(str).str.strip().eq("NR")).astype(int)
        X_advanced["total_missing_predictors"] = nr_hits.sum(axis=1)
        X_advanced["prop_missing_predictors"] = nr_hits.mean(axis=1)

    # Categorical encoding: keep top categories
    if categorical_cols:
        for col in categorical_cols:
            top_categories = X_advanced[col].value_counts(dropna=True).head(50).index
            X_advanced[col] = X_advanced[col].where(X_advanced[col].isin(top_categories), "OTHER")
            # frequency encoding companion
            freq_encoding = X_advanced[col].value_counts(normalize=True)
            X_advanced[f"{col}_freq"] = X_advanced[col].map(freq_encoding)

        X_advanced = pd.get_dummies(X_advanced, columns=categorical_cols, drop_first=True, prefix=categorical_cols)

    # Low variance prune
    nunique = X_advanced.nunique()
    X_advanced = X_advanced.loc[:, nunique > 1]

    return X_advanced


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

    # Load data - default to GSS2.xlsx
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
        raise ValueError("No predictor variables found. Please adjust --predictors to match your data columns.")

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
    
    # Design matrix with enhanced features
    X = build_design_matrix(df_filtered, predictors_present)
    y = df_filtered[target].astype(int).reindex(X.index)
    print(f"[Features] X: {X.shape}, y: {y.shape}, nonresponse rate={y.mean():.4f}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    # Logistic Regression (standardized)
    scaler = StandardScaler(with_mean=False)
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Helper function for threshold optimization
    def optimize_threshold(y_true, y_pred_proba):
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        optimal_idx = np.argmax(tpr - fpr)
        return thresholds[optimal_idx]
    
    # Helper function to calculate detailed metrics
    def calculate_metrics(y_true, y_pred_proba, model_name):
        auc_score = roc_auc_score(y_true, y_pred_proba)
        brier_score = brier_score_loss(y_true, y_pred_proba)
        optimal_threshold = optimize_threshold(y_true, y_pred_proba)
        
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

    # Enhanced Logistic Regression
    logit = LogisticRegression(
        max_iter=3000,
        C=0.1,
        class_weight='balanced',
        solver='liblinear'
    )
    logit.fit(X_train_s, y_train)
    p_logit = logit.predict_proba(X_test_s)[:, 1]
    logit_metrics = calculate_metrics(y_test, p_logit, "logistic_regression")

    # Enhanced Gradient Boosting
    gb = GradientBoostingClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        max_features='sqrt',
        random_state=RANDOM_SEED
    )
    gb.fit(X_train, y_train)
    p_gb = gb.predict_proba(X_test)[:, 1]
    gb_metrics = calculate_metrics(y_test, p_gb, "gradient_boosting")
    
    # Choose best model based on AUC
    if gb_metrics["auc"] > logit_metrics["auc"]:
        best_model = gb
        best_metrics = gb_metrics
        best_model_name = "gradient_boosting"
        best_proba = p_gb
    else:
        best_model = logit
        best_metrics = logit_metrics
        best_model_name = "logistic_regression"
        best_proba = p_logit
    
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
        "all_models": {
            "logistic_regression": logit_metrics,
            "gradient_boosting": gb_metrics
        }
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
    
    # Create plots (ROC, PR, Calibration, Feature Importance)
    # ROC curve comparison plot
    plt.figure(figsize=(10, 8))
    for model_name, model_obj in [("Logistic Regression", logit), ("Gradient Boosting", gb)]:
        if model_name == "Logistic Regression":
            y_pred_proba_model = model_obj.predict_proba(X_test_s)[:, 1]
        else:
            y_pred_proba_model = model_obj.predict_proba(X_test)[:, 1]
        
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
    
    # Feature importance plot (for Gradient Boosting)
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

    # IPW analysis (simplified)
    if best_model_name == "logistic_regression":
        X_scaled = scaler.transform(X)
        p_all = best_model.predict_proba(X_scaled)[:, 1]
    else:
        p_all = best_model.predict_proba(X)[:, 1]
    p_all = np.clip(p_all, 1e-6, 0.95)  # avoid huge weights
    df_filtered["IPW"] = pd.Series(1.0 / (1.0 - p_all), index=X.index)

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
        "bootstrap_iters": bootstrap_iters
    }
    with open(os.path.join(out_dir, "run_log.json"), "w", encoding="utf-8") as f:
        json.dump(run_log, f, indent=2)
    print("[Done] Saved run_log.json")


def parse_args():
    ap = argparse.ArgumentParser(description="GSS stigma-related nonresponse starter pipeline with enhanced variables (Data2)")
    ap.add_argument("--data", default="data/GSS2.xlsx", help="Path to GSS data file (.xlsx or .dta)")
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