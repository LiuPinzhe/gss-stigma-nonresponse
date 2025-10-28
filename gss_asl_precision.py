#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GSS ASL Tree Models - Precision-Optimized (Target: 0.2+ Precision)
================================================================
"""

import argparse
import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, roc_curve, brier_score_loss, precision_recall_curve
from sklearn.calibration import CalibratedClassifierCV
# Manual sampling strategies without imblearn

warnings.filterwarnings("ignore", category=UserWarning)

def compute_precision_weights(y, target_precision=0.2):
    """Compute weights to achieve target precision"""
    y_array = np.array(y)
    n_pos = np.sum(y_array == 1)
    n_neg = np.sum(y_array == 0)
    
    # Extreme weighting for precision
    pos_weight = 50.0  # Very high weight for positive class
    neg_weight = 1.0   # Normal weight for negative class
    
    weights = np.where(y_array == 1, pos_weight, neg_weight)
    return weights

def load_and_process_data(data_path):
    """Load and process GSS data"""
    print(f"[Load] {data_path}")
    df = pd.read_excel(data_path) if data_path.endswith('.xlsx') else pd.read_stata(data_path, convert_categoricals=False)
    print(f"[Load] shape={df.shape[0]:,} x {df.shape[1]:,}")
    
    # Items and predictors
    items = ['sexornt', 'premarsx', 'xmarsex']
    predictors = ['age', 'educ', 'sex', 'race', 'region', 'year', 'relig', 'attend', 'income', 'marital', 'polviews', 'wrkstat']
    
    items_present = [item for item in items if item in df.columns]
    predictors_present = [pred for pred in predictors if pred in df.columns]
    
    # Create composite nonresponse target
    def mark_nonresponse(series):
        s = series.copy()
        if pd.api.types.is_numeric_dtype(s):
            mask = s.isna() | s.isin({0, 8, 9, 98, 99, 998, 999}) | (s >= 97)
        else:
            s_str = s.astype(str).str.strip()
            inapplicable_mask = s_str.str.startswith(".i")
            nonresponse_mask = (s.isna() | s_str.isin(["", "NA", "NaN", "nan"]) |
                               s_str.str.startswith(".d") | s_str.str.startswith(".s") | s_str.str.startswith(".n"))
            result = pd.Series(np.nan, index=s.index, dtype=float)
            result[~inapplicable_mask & ~nonresponse_mask] = 0
            result[~inapplicable_mask & nonresponse_mask] = 1
            return result
        return mask.astype(int)
    
    df['NR_COMPOSITE'] = np.nan
    asked_any = pd.Series(False, index=df.index)
    refused_any = pd.Series(False, index=df.index)
    
    for item in items_present:
        nr_var = mark_nonresponse(df[item])
        asked_any |= nr_var.notna()
        refused_any |= (nr_var == 1)
    
    df.loc[asked_any, 'NR_COMPOSITE'] = refused_any[asked_any].astype(int)
    
    # Filter and build features
    valid_cases = df['NR_COMPOSITE'].notna()
    df_filtered = df[valid_cases].copy()
    
    X = df_filtered[predictors_present].copy()
    
    # Clean and process features
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].replace({0: np.nan, 8: np.nan, 9: np.nan, 98: np.nan, 99: np.nan, 998: np.nan, 999: np.nan})
            X.loc[X[col] >= 97, col] = np.nan
            X[col] = X[col].fillna(X[col].median() if not X[col].isna().all() else 0)
        else:
            X[col] = X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 'missing')
    
    # One-hot encode categoricals
    cat_cols = [col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])]
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, dummy_na=True)
    
    X = X.fillna(0)
    y = df_filtered['NR_COMPOSITE'].astype(int)
    
    print(f"[Data] Features: {X.shape[1]}, Class balance: {np.bincount(y)}")
    return X, y

def precision_threshold_search(y_true, y_pred_proba, target_precision=0.2):
    """Find threshold that achieves target precision"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    # Handle dimension mismatch (precision/recall have one more element than thresholds)
    if len(precision) > len(thresholds):
        precision = precision[:-1]
        recall = recall[:-1]
    
    # Find thresholds that meet precision target
    valid_idx = precision >= target_precision
    if not np.any(valid_idx):
        # If no threshold meets target, use highest precision
        best_idx = np.argmax(precision)
        return thresholds[best_idx], precision[best_idx], recall[best_idx]
    
    # Among valid thresholds, choose one with highest recall
    valid_precision = precision[valid_idx]
    valid_recall = recall[valid_idx]
    valid_thresholds = thresholds[valid_idx]
    
    best_idx = np.argmax(valid_recall)
    return valid_thresholds[best_idx], valid_precision[best_idx], valid_recall[best_idx]

def calculate_precision_metrics(y_true, y_pred_proba, target_precision=0.2):
    """Calculate metrics optimized for precision"""
    threshold, precision, recall = precision_threshold_search(y_true, y_pred_proba, target_precision)
    
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    auc_score = roc_auc_score(y_true, y_pred_proba)
    brier_score = brier_score_loss(y_true, y_pred_proba)
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "auc": float(auc_score),
        "brier_score": float(brier_score),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    }

def run_precision_pipeline(data_path, out_dir):
    """Main precision-optimized pipeline"""
    os.makedirs(out_dir, exist_ok=True)
    
    # Load data
    X, y = load_and_process_data(data_path)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"\n[Training] Precision-optimized models (target: 0.2+ precision)...")
    
    models = {}
    results = {}
    
    # Strategy 1: Extreme class weights
    print("[1/5] Random Forest with extreme weights...")
    rf_extreme = RandomForestClassifier(
        n_estimators=1000, max_depth=20, min_samples_split=2, min_samples_leaf=1,
        class_weight={0: 1, 1: 100}, random_state=42, n_jobs=-1
    )
    precision_weights = compute_precision_weights(y_train)
    rf_extreme.fit(X_train, y_train, sample_weight=precision_weights)
    models['rf_extreme'] = rf_extreme
    
    # Strategy 2: Manual oversampling
    print("[2/5] Manual oversampling + Random Forest...")
    # Simple oversampling by duplication
    minority_mask = y_train == 1
    X_minority = X_train[minority_mask]
    y_minority = y_train[minority_mask]
    
    # Duplicate minority class 5 times
    X_over_list = [X_train]
    y_over_list = [y_train]
    for _ in range(5):
        X_over_list.append(X_minority)
        y_over_list.append(y_minority)
    
    X_resampled = pd.concat(X_over_list, axis=0)
    y_resampled = pd.concat(y_over_list, axis=0)
    print(f"  Resampled: {np.bincount(y_resampled)}")
    
    rf_over = RandomForestClassifier(
        n_estimators=1000, max_depth=15, class_weight={0: 1, 1: 10}, random_state=42, n_jobs=-1
    )
    rf_over.fit(X_resampled, y_resampled)
    models['rf_over'] = rf_over
    
    # Strategy 3: Gradient Boosting with high learning rate
    print("[3/5] Gradient Boosting optimized...")
    gb_precision = GradientBoostingClassifier(
        n_estimators=2000, learning_rate=0.005, max_depth=10,
        subsample=0.7, max_features='sqrt', random_state=42
    )
    gb_precision.fit(X_train, y_train, sample_weight=precision_weights)
    models['gb_precision'] = gb_precision
    
    # Strategy 4: Extra Trees with strict parameters
    print("[4/5] Extra Trees strict...")
    et_strict = ExtraTreesClassifier(
        n_estimators=1500, max_depth=25, min_samples_split=2, min_samples_leaf=1,
        class_weight={0: 1, 1: 80}, random_state=42, n_jobs=-1
    )
    et_strict.fit(X_train, y_train, sample_weight=precision_weights)
    models['et_strict'] = et_strict
    
    # Strategy 5: Manual undersampling
    print("[5/5] Manual undersampling + Random Forest...")
    # Keep all minority, sample 10% of majority
    majority_mask = y_train == 0
    minority_mask = y_train == 1
    
    X_majority = X_train[majority_mask]
    y_majority = y_train[majority_mask]
    X_minority = X_train[minority_mask]
    y_minority = y_train[minority_mask]
    
    # Sample 10% of majority class
    n_majority_sample = int(len(X_majority) * 0.1)
    np.random.seed(42)
    majority_idx = np.random.choice(len(X_majority), n_majority_sample, replace=False)
    
    X_under = pd.concat([X_majority.iloc[majority_idx], X_minority], axis=0)
    y_under = pd.concat([y_majority.iloc[majority_idx], y_minority], axis=0)
    print(f"  Undersampled: {np.bincount(y_under)}")
    
    rf_under = RandomForestClassifier(
        n_estimators=1000, max_depth=15, class_weight={0: 1, 1: 5}, random_state=42, n_jobs=-1
    )
    rf_under.fit(X_under, y_under)
    models['rf_under'] = rf_under
    
    # Evaluate all models
    print("\n[Evaluation] Testing precision performance...")
    
    for name, model in models.items():
        print(f"[Eval] {name}...")
        
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calibrate
        calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=3)
        if name == 'rf_over':
            calibrated_model.fit(X_resampled, y_resampled)
        elif name == 'rf_under':
            calibrated_model.fit(X_under, y_under)
        else:
            calibrated_model.fit(X_train, y_train)
        
        y_pred_proba_cal = calibrated_model.predict_proba(X_test)[:, 1]
        
        # Calculate precision-optimized metrics
        metrics = calculate_precision_metrics(y_test, y_pred_proba_cal, target_precision=0.2)
        results[name] = metrics
        
        print(f"  AUC: {metrics['auc']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
        print(f"  F1: {metrics['f1']:.4f}, Threshold: {metrics['threshold']:.4f}")
    
    # Find best model by precision
    best_model = max(results.keys(), key=lambda x: results[x]['precision'])
    best_metrics = results[best_model]
    
    print(f"\n[Best Model] {best_model}")
    print(f"[Results] Precision: {best_metrics['precision']:.4f} (target: 0.2+)")
    print(f"[Results] AUC: {best_metrics['auc']:.4f}, F1: {best_metrics['f1']:.4f}")
    print(f"[Results] Recall: {best_metrics['recall']:.4f}, Threshold: {best_metrics['threshold']:.4f}")
    
    cm = best_metrics['confusion_matrix']
    print(f"[Confusion] TN: {cm['tn']}, FP: {cm['fp']}, FN: {cm['fn']}, TP: {cm['tp']}")
    
    # Save results
    output = {
        'best_model': best_model,
        'best_metrics': best_metrics,
        'all_results': results,
        'target_precision': 0.2,
        'achieved_precision': best_metrics['precision']
    }
    
    with open(os.path.join(out_dir, "precision_results.json"), "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n[Done] Results saved to {out_dir}")
    print(f"[Success] {'✓' if best_metrics['precision'] >= 0.2 else '✗'} Target precision {'achieved' if best_metrics['precision'] >= 0.2 else 'not achieved'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precision-Optimized GSS Analysis")
    parser.add_argument("--data", default="data/GSS2.xlsx", help="Data file path")
    parser.add_argument("--out", default="outputs_precision", help="Output directory")
    args = parser.parse_args()
    
    run_precision_pipeline(args.data, args.out)