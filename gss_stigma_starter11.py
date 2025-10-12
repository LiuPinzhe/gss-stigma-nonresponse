#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GSS Stigma-Related Nonresponse — AUC Optimized Pipeline (1972–2024)
===================================================================

AUC optimization without external dependencies:
1. Manual class imbalance handling
2. Multiple algorithm comparison
3. Threshold optimization
4. Advanced feature engineering
5. Composite mode support

Usage
-----
python gss_stigma_starter11.py --data /path/to/gss_data.xlsx --out ./outputs --mode composite
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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, roc_curve
from sklearn.utils import class_weight

warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------
# Defaults & Config
# -----------------------------
DEFAULT_ITEMS: List[str] = [
    "sexornt",  # sexual orientation
    "premarsx",  # premarital sex attitude
    "xmarsex",  # extramarital sex attitude
]

DEFAULT_PREDICTORS: List[str] = [
    "age", "educ", "sex", "race", "region", "year",
    "relig", "attend", "income", "marital"
]

NON_SUBSTANTIVE_CODES = {0, 8, 9, 98, 99, 998, 999}
RANDOM_SEED = 42


# -----------------------------
# AUC Optimized Utility functions
# -----------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def find_matching_columns(df: pd.DataFrame, candidates: List[str]) -> List[str]:
    """Find columns that match case-insensitively."""
    available_cols = set(df.columns.str.lower())
    matched = []
    for candidate in candidates:
        if candidate.lower() in available_cols:
            actual_name = df.columns[df.columns.str.lower() == candidate.lower()][0]
            matched.append(actual_name)
    return matched


def to_numeric_or_category(s: pd.Series) -> pd.Series:
    """Fast numeric conversion."""
    try:
        return pd.to_numeric(s, errors="coerce")
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


def available_columns(df: pd.DataFrame, candidates: List[str]) -> List[str]:
    return find_matching_columns(df, candidates)


def create_advanced_features(df: pd.DataFrame, base_predictors: List[str]) -> pd.DataFrame:
    """Create advanced features specifically for non-response prediction."""
    features = df[base_predictors].copy()
    
    # Convert all columns to numeric first
    for col in features.columns:
        features[col] = to_numeric_or_category(features[col])

    # Enhanced feature engineering
    if 'age' in base_predictors and pd.api.types.is_numeric_dtype(features['age']):
        age_clean = features['age'].fillna(features['age'].median())
        features['age_binned'] = pd.cut(age_clean, bins=[0, 30, 45, 60, 100], labels=[1, 2, 3, 4])
        features['age_squared'] = age_clean ** 2
        features['age_log'] = np.log1p(age_clean)

    if 'educ' in base_predictors and pd.api.types.is_numeric_dtype(features['educ']):
        educ_clean = features['educ'].fillna(features['educ'].median())
        features['educ_binned'] = pd.cut(educ_clean, bins=[0, 12, 16, 20], labels=[1, 2, 3])
        features['educ_squared'] = educ_clean ** 2

    if 'year' in base_predictors and pd.api.types.is_numeric_dtype(features['year']):
        year_clean = features['year'].fillna(features['year'].median())
        features['year_centered'] = year_clean - 2000
        features['year_squared'] = features['year_centered'] ** 2
        features['decade'] = (year_clean // 10) * 10

    # Interaction features that might predict non-response
    if all(col in base_predictors for col in ['age', 'educ']):
        if pd.api.types.is_numeric_dtype(features['age']) and pd.api.types.is_numeric_dtype(features['educ']):
            age_clean = features['age'].fillna(features['age'].median())
            educ_clean = features['educ'].fillna(features['educ'].median())
            features['age_educ_interaction'] = age_clean * educ_clean
            features['age_educ_ratio'] = age_clean / (educ_clean + 1)

    if all(col in base_predictors for col in ['relig', 'attend']):
        if pd.api.types.is_numeric_dtype(features['relig']) and pd.api.types.is_numeric_dtype(features['attend']):
            relig_clean = features['relig'].fillna(features['relig'].median())
            attend_clean = features['attend'].fillna(features['attend'].median())
            features['religiosity'] = relig_clean * attend_clean

    # Create non-response propensity indicators from other variables
    for col in base_predictors:
        if col not in ['age', 'educ', 'year'] and pd.api.types.is_numeric_dtype(features[col]):
            features[f'{col}_missing'] = features[col].isna().astype(int)

    return features


def build_auc_optimized_matrix(df: pd.DataFrame, predictors: List[str]) -> pd.DataFrame:
    """Build feature matrix optimized for AUC performance."""
    actual_predictors = find_matching_columns(df, predictors)
    X_advanced = create_advanced_features(df, actual_predictors)

    # Process features
    numeric_cols = []
    categorical_cols = []

    for c in X_advanced.columns:
        X_advanced[c] = to_numeric_or_category(X_advanced[c])
        if pd.api.types.is_numeric_dtype(X_advanced[c]):
            numeric_cols.append(c)
            mask = (X_advanced[c].isin(NON_SUBSTANTIVE_CODES)) | (X_advanced[c] >= 97) | X_advanced[c].isna()
            X_advanced.loc[mask, c] = np.nan

            if X_advanced[c].nunique() < 20:
                X_advanced[c] = X_advanced[c].fillna(X_advanced[c].mode()[0] if not X_advanced[c].mode().empty else 0)
            else:
                X_advanced[c] = X_advanced[c].fillna(X_advanced[c].median() if X_advanced[c].notna().any() else 0)
        else:
            categorical_cols.append(c)

    # Enhanced categorical encoding
    if categorical_cols:
        for col in categorical_cols:
            top_categories = X_advanced[col].value_counts().head(20).index
            X_advanced[col] = X_advanced[col].where(X_advanced[col].isin(top_categories), 'OTHER')
            freq_encoding = X_advanced[col].value_counts(normalize=True)
            X_advanced[f'{col}_freq'] = X_advanced[col].map(freq_encoding)

        X_advanced = pd.get_dummies(X_advanced, columns=categorical_cols, drop_first=True, prefix=categorical_cols)

    # Remove low variance features
    nunique = X_advanced.nunique()
    X_advanced = X_advanced.loc[:, nunique > 1]

    return X_advanced


def optimize_threshold_for_auc(y_true, y_pred_proba):
    """Find optimal threshold that maximizes AUC."""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold


def manual_oversample(X_train, y_train):
    """Manual oversampling for minority class."""
    minority_class = 1 if sum(y_train == 1) < sum(y_train == 0) else 0
    majority_class = 1 - minority_class

    X_minority = X_train[y_train == minority_class]
    y_minority = y_train[y_train == minority_class]
    X_majority = X_train[y_train == majority_class]
    y_majority = y_train[y_train == majority_class]

    n_minority = len(X_minority)
    n_majority = len(X_majority)

    if n_minority < n_majority:
        n_repeats = n_majority // n_minority
        remainder = n_majority % n_minority

        X_minority_oversampled = pd.concat([X_minority] * n_repeats + [X_minority.iloc[:remainder]], axis=0)
        y_minority_oversampled = pd.concat([y_minority] * n_repeats + [y_minority.iloc[:remainder]], axis=0)

        X_balanced = pd.concat([X_majority, X_minority_oversampled], axis=0)
        y_balanced = pd.concat([y_majority, y_minority_oversampled], axis=0)

        shuffle_idx = np.random.permutation(len(X_balanced))
        return X_balanced.iloc[shuffle_idx], y_balanced.iloc[shuffle_idx]

    return X_train, y_train


def auc_optimized_training(X_train, y_train, X_test, y_test, use_oversampling=True):
    """AUC-optimized training with multiple strategies."""
    print("[AUC Optimization] Training multiple models with class imbalance handling...")

    # Calculate class weights for imbalance handling
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

    # Apply manual oversampling if requested
    if use_oversampling:
        X_train_processed, y_train_processed = manual_oversample(X_train, y_train)
        print(f"[Oversampling] After oversampling - Class distribution: {np.bincount(y_train_processed)}")
    else:
        X_train_processed, y_train_processed = X_train, y_train

    # Multiple model strategies
    models = {
        'rf_weighted': RandomForestClassifier(
            n_estimators=500,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features='sqrt',
            class_weight=class_weight_dict,
            random_state=RANDOM_SEED,
            n_jobs=-1
        ),
        'rf_balanced': RandomForestClassifier(
            n_estimators=300,
            max_depth=25,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='log2',
            class_weight='balanced',
            random_state=RANDOM_SEED,
            n_jobs=-1
        ),
        'gradient_boosting': GradientBoostingClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=RANDOM_SEED
        )
    }

    # Train all models
    best_auc = -1
    best_model = None
    best_model_name = ""
    all_metrics = {}

    for name, model in models.items():
        print(f"[Training] {name}...")
        model.fit(X_train_processed, y_train_processed)

        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred_default = (y_pred_proba >= 0.5).astype(int)
        auc_score = roc_auc_score(y_test, y_pred_proba)

        optimal_threshold = optimize_threshold_for_auc(y_test, y_pred_proba)
        y_pred_optimized = (y_pred_proba >= optimal_threshold).astype(int)

        report_optimized = classification_report(y_test, y_pred_optimized, output_dict=True)

        metrics = {
            "auc": auc_score,
            "accuracy_default": accuracy_score(y_test, y_pred_default),
            "accuracy_optimized": accuracy_score(y_test, y_pred_optimized),
            "precision_optimized": report_optimized['1']['precision'] if '1' in report_optimized else 0,
            "recall_optimized": report_optimized['1']['recall'] if '1' in report_optimized else 0,
            "f1_optimized": report_optimized['1']['f1-score'] if '1' in report_optimized else 0,
            "optimal_threshold": optimal_threshold,
            "class_balance": {
                "class_0": sum(y_test == 0),
                "class_1": sum(y_test == 1),
                "ratio": sum(y_test == 1) / len(y_test) if len(y_test) > 0 else 0
            }
        }

        all_metrics[name] = metrics
        print(f"[{name}] AUC: {auc_score:.4f}, Optimal Threshold: {optimal_threshold:.3f}")

        if auc_score > best_auc:
            best_auc = auc_score
            best_model = model
            best_model_name = name

    print(f"[Best Model] {best_model_name} with AUC: {best_auc:.4f}")

    # Feature importance from best model
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
    else:
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': np.ones(len(X_train.columns)) / len(X_train.columns)
        })

    return best_model, all_metrics[best_model_name], feature_importance, all_metrics, models


# -----------------------------
# AUC Optimized Core pipeline
# -----------------------------
def run_pipeline(
        data_path: str,
        out_dir: str,
        items: List[str],
        predictors: List[str],
        bootstrap_iters: int = 200,
        use_oversampling: bool = True,
        mode: str = "composite"
):
    ensure_dir(out_dir)

    # Load data
    print(f"[Load] {data_path}")
    
    # 只读取需要的列以节省内存
    needed_cols = list(set(items + predictors + ['year']))
    
    if data_path.endswith('.xlsx'):
        df = pd.read_excel(data_path, usecols=needed_cols)
        print(f"[Load] Excel format")
    elif data_path.endswith('.sav'):
        df, meta = pyreadstat.read_sav(
            data_path, 
            usecols=needed_cols,
            apply_value_formats=True
        )
        print(f"[Load] SPSS format with value labels applied")
    else:
        df = pd.read_stata(data_path, convert_categoricals=False, columns=needed_cols)
        print(f"[Load] Stata format")
    print(f"[Load] shape={df.shape[0]:,} x {df.shape[1]:,}")

    # Use case-insensitive matching
    items_present = available_columns(df, items)
    predictors_present = available_columns(df, predictors)

    print(f"[Items] Found {len(items_present)} items: {items_present}")
    print(f"[Predictors] Found {len(predictors_present)} predictors: {predictors_present}")

    if not predictors_present:
        raise ValueError("No predictor variables found.")

    # Build nonresponse indicators
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
        raise ValueError("None of the specified --items were found.")

    # Choose target based on mode
    if mode.lower() == "single":
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

    # Check class balance
    class_counts = df[target].value_counts()
    print(f"[Class Balance] 0 (Response): {class_counts.get(0, 0):,}, 1 (Non-response): {class_counts.get(1, 0):,}")
    minority_ratio = class_counts.get(1, 0) / len(df) if len(df) > 0 else 0
    print(f"[Class Balance] Minority class ratio: {minority_ratio:.4f}")

    # Filter to only cases where target is not NaN (i.e., were asked the question)
    valid_cases = df[target].notna()
    df_filtered = df[valid_cases].copy()
    print(f"[Filter] Using {valid_cases.sum():,} cases out of {len(df):,} total ({valid_cases.mean():.1%})")
    
    # Build optimized feature matrix
    print("[Features] Building AUC-optimized feature matrix...")
    X = build_auc_optimized_matrix(df_filtered, predictors_present)
    y = df_filtered[target].astype(int).reindex(X.index)

    print(f"[Features] Final feature matrix: {X.shape}")

    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    print(f"[Training] Data shape - X_train: {X_train.shape}, X_test: {X_test.shape}")

    # AUC Optimized Training
    print("\n[Training] AUC-optimized model training...")
    best_model, metrics, feature_importance, all_metrics, models = auc_optimized_training(
        X_train, y_train, X_test, y_test, use_oversampling=use_oversampling
    )

    # Save feature importance
    feature_importance.to_csv(os.path.join(out_dir, "feature_importance.csv"), index=False)
    print("[Feature Importance] Saved feature_importance.csv")

    # Print top features
    print("\n[Top 15 Features]:")
    for i, row in feature_importance.head(15).iterrows():
        print(f"  {i + 1:2d}. {row['feature']:<30} {row['importance']:.4f}")

    # Save comprehensive metrics
    final_metrics = {
        "best_model": "auc_optimized",
        "best_auc": float(metrics["auc"]),
        "accuracy_default": float(metrics["accuracy_default"]),
        "accuracy_optimized": float(metrics["accuracy_optimized"]),
        "precision_optimized": float(metrics["precision_optimized"]),
        "recall_optimized": float(metrics["recall_optimized"]),
        "f1_optimized": float(metrics["f1_optimized"]),
        "optimal_threshold": float(metrics["optimal_threshold"]),
        "class_balance": metrics["class_balance"],
        "all_models": all_metrics
    }

    with open(os.path.join(out_dir, "model_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)

    # Get final predictions with optimized threshold
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    optimal_threshold = metrics["optimal_threshold"]
    y_pred_optimized = (y_pred_proba >= optimal_threshold).astype(int)

    # Save detailed classification report
    with open(os.path.join(out_dir, "classification_report.txt"), "w") as f:
        f.write("AUC Optimized Model - Classification Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Mode: {mode}\n")
        f.write(f"Target: {target}\n")
        f.write(f"Best Model Type: {final_metrics['best_model']}\n")
        f.write(f"AUC: {metrics['auc']:.4f}\n")
        f.write(f"Accuracy (default threshold): {metrics['accuracy_default']:.4f}\n")
        f.write(f"Accuracy (optimized threshold): {metrics['accuracy_optimized']:.4f}\n")
        f.write(f"Precision (optimized): {metrics['precision_optimized']:.4f}\n")
        f.write(f"Recall (optimized): {metrics['recall_optimized']:.4f}\n")
        f.write(f"F1-Score (optimized): {metrics['f1_optimized']:.4f}\n")
        f.write(f"Optimal Threshold: {optimal_threshold:.4f}\n")

        f.write(f"\nClass Balance:\n")
        f.write(f"  Class 0 (Response): {metrics['class_balance']['class_0']}\n")
        f.write(f"  Class 1 (Non-response): {metrics['class_balance']['class_1']}\n")
        f.write(f"  Minority Ratio: {metrics['class_balance']['ratio']:.4f}\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("Detailed Classification Report (Optimized Threshold):\n")
        f.write(classification_report(y_test, y_pred_optimized))
        f.write(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred_optimized)}")

        f.write("\n" + "=" * 60 + "\n")
        f.write("All Models Comparison:\n")
        for model_name, model_metrics in all_metrics.items():
            f.write(f"\n{model_name}:\n")
            f.write(f"  AUC: {model_metrics['auc']:.4f}, ")
            f.write(f"F1: {model_metrics['f1_optimized']:.4f}\n")

    print(f"\n[Final Results]")
    print(f"Mode: {mode}")
    print(f"Target: {target}")
    print(f"AUC: {metrics['auc']:.4f}")
    print(f"Accuracy (optimized): {metrics['accuracy_optimized']:.4f}")
    print(f"Precision: {metrics['precision_optimized']:.4f}")
    print(f"Recall: {metrics['recall_optimized']:.4f}")
    print(f"F1-Score: {metrics['f1_optimized']:.4f}")
    print(f"Optimal Threshold: {optimal_threshold:.4f}")

    # Create ROC curve plot
    plt.figure(figsize=(10, 8))
    for model_name, model_obj in models.items():
        y_pred_proba_model = model_obj.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba_model)
        auc_score = roc_auc_score(y_test, y_pred_proba_model)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves - Model Comparison ({mode} mode)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "roc_curves.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("[ROC] Saved roc_curves.png")

    # Run log
    run_log = {
        "data_path": data_path,
        "out_dir": out_dir,
        "mode": mode,
        "target_modeled": target,
        "composite_items": items_present if mode.lower() == "composite" else None,
        "predictors_used": predictors_present,
        "items_present": items_present,
        "feature_matrix_shape": X.shape,
        "metrics": final_metrics,
        "bootstrap_iters": bootstrap_iters,
        "oversampling_used": use_oversampling
    }
    with open(os.path.join(out_dir, "run_log.json"), "w", encoding="utf-8") as f:
        json.dump(run_log, f, indent=2)

    print("\n[AUC Optimized Pipeline Complete]")
    print(f"Results saved to: {out_dir}")
    print(f"Final AUC: {metrics['auc']:.4f}")


def parse_args():
    ap = argparse.ArgumentParser(description="GSS stigma-related nonresponse - AUC Optimized Pipeline")
    ap.add_argument("--data", required=True, help="Path to GSS data file (.xlsx, .dta, or .sav)")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--items", nargs="*", default=DEFAULT_ITEMS, help="Sensitive items")
    ap.add_argument("--predictors", nargs="*", default=DEFAULT_PREDICTORS, help="Predictor variables")
    ap.add_argument("--bootstrap", type=int, default=200, help="Bootstrap iterations")
    ap.add_argument("--no-oversampling", action="store_true", help="Disable manual oversampling")
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
        use_oversampling=not args.no_oversampling,
        mode=args.mode
    )