#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GSS ASL Tree-Based Models with Dataset 2 and Starter13 Variables
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
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, roc_curve, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore", category=UserWarning)

# ASL-inspired sample weighting for tree models
def compute_asl_tree_weights(y, gamma_neg=4, gamma_pos=1):
    """Compute ASL-inspired weights optimized for tree models"""
    y_array = np.array(y)
    n_samples = len(y_array)
    n_pos = np.sum(y_array == 1)
    n_neg = np.sum(y_array == 0)
    
    if n_pos == 0 or n_neg == 0:
        return np.ones(n_samples)
    
    # Tree-optimized ASL weighting
    pos_weight = (n_samples / (2 * n_pos)) * gamma_pos
    neg_weight = (n_samples / (2 * n_neg)) / gamma_neg
    
    weights = np.where(y_array == 1, pos_weight, neg_weight)
    weights = weights / np.mean(weights)  # Normalize
    
    return weights

def load_data(data_path):
    """Load GSS dataset 2"""
    print(f"[Load] {data_path}")
    if data_path.endswith('.xlsx'):
        df = pd.read_excel(data_path)
        print(f"[Load] Excel format")
    else:
        df = pd.read_stata(data_path, convert_categoricals=False)
        print(f"[Load] Stata format")
    print(f"[Load] shape={df.shape[0]:,} x {df.shape[1]:,}")
    return df

def mark_nonresponse(series):
    """Mark nonresponse patterns"""
    s = series.copy()
    if pd.api.types.is_numeric_dtype(s):
        mask = s.isna() | s.isin({0, 8, 9, 98, 99, 998, 999}) | (s >= 97)
    else:
        s_str = s.astype(str).str.strip()
        inapplicable_mask = s_str.str.startswith(".i")
        nonresponse_mask = (s.isna() | 
                           s_str.isin(["", "NA", "NaN", "nan"]) |
                           s_str.str.startswith(".d") |
                           s_str.str.startswith(".s") |
                           s_str.str.startswith(".n"))
        
        result = pd.Series(np.nan, index=s.index, dtype=float)
        result[~inapplicable_mask & ~nonresponse_mask] = 0
        result[~inapplicable_mask & nonresponse_mask] = 1
        return result
    
    return mask.astype(int)

def build_features(df, predictors):
    """Build feature matrix with starter13 variables"""
    X = df[predictors].copy()
    
    # Convert to numeric where possible
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            try:
                X[col] = pd.to_numeric(X[col], errors='coerce')
            except:
                pass
    
    # Clean non-substantive codes first
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            # Remove non-substantive codes
            X[col] = X[col].replace({0: np.nan, 8: np.nan, 9: np.nan, 98: np.nan, 99: np.nan, 998: np.nan, 999: np.nan})
            X.loc[X[col] >= 97, col] = np.nan
    
    # Fill missing values
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            median_val = X[col].median()
            if pd.isna(median_val):
                median_val = 0
            X[col] = X[col].fillna(median_val)
        else:
            mode_val = X[col].mode()
            fill_val = mode_val[0] if len(mode_val) > 0 else 'missing'
            X[col] = X[col].fillna(fill_val)
    
    # One-hot encode categorical variables
    cat_cols = [col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])]
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, dummy_na=True)
    
    # Final check for any remaining NaNs
    X = X.fillna(0)
    
    return X

def calculate_metrics(y_true, y_pred_proba):
    """Calculate comprehensive metrics"""
    auc_score = roc_auc_score(y_true, y_pred_proba)
    brier_score = brier_score_loss(y_true, y_pred_proba)
    
    # Optimal threshold
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    y_pred_opt = (y_pred_proba >= optimal_threshold).astype(int)
    
    # Classification metrics
    report = classification_report(y_true, y_pred_opt, output_dict=True)
    cm = confusion_matrix(y_true, y_pred_opt)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    # Brier skill score
    base_rate = sum(y_true) / len(y_true)
    baseline_brier = brier_score_loss(y_true, [base_rate] * len(y_true))
    brier_skill_score = 1 - (brier_score / baseline_brier) if baseline_brier > 0 else 0
    
    return {
        "auc": float(auc_score),
        "brier_score": float(brier_score),
        "brier_skill_score": float(brier_skill_score),
        "accuracy": float(accuracy_score(y_true, y_pred_opt)),
        "precision": float(report['1']['precision'] if '1' in report else 0),
        "recall": float(report['1']['recall'] if '1' in report else 0),
        "f1": float(report['1']['f1-score'] if '1' in report else 0),
        "optimal_threshold": float(optimal_threshold),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    }

def run_asl_tree_pipeline(data_path, out_dir, items, predictors):
    """Main ASL tree-based pipeline"""
    os.makedirs(out_dir, exist_ok=True)
    
    # Load data
    df = load_data(data_path)
    
    # Check available columns
    items_present = [item for item in items if item in df.columns]
    predictors_present = [pred for pred in predictors if pred in df.columns]
    
    print(f"[Variables] Items: {items_present}")
    print(f"[Variables] Predictors: {predictors_present}")
    
    if not items_present or not predictors_present:
        raise ValueError("Missing required variables")
    
    # Create composite nonresponse target
    df['NR_COMPOSITE'] = np.nan
    asked_any = pd.Series(False, index=df.index)
    refused_any = pd.Series(False, index=df.index)
    
    for item in items_present:
        nr_var = mark_nonresponse(df[item])
        asked_any |= nr_var.notna()
        refused_any |= (nr_var == 1)
    
    df.loc[asked_any, 'NR_COMPOSITE'] = refused_any[asked_any].astype(int)
    
    # Filter valid cases
    valid_cases = df['NR_COMPOSITE'].notna()
    df_filtered = df[valid_cases].copy()
    print(f"[Filter] Using {valid_cases.sum():,} cases ({valid_cases.mean():.1%})")
    
    # Build features
    X = build_features(df_filtered, predictors_present)
    y = df_filtered['NR_COMPOSITE'].astype(int)
    
    print(f"[Data] Features: {X.shape[1]}, Class balance: {np.bincount(y)}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # ASL weights
    asl_weights = compute_asl_tree_weights(y_train, gamma_neg=4, gamma_pos=1)
    print(f"[ASL] Weight ratio (pos/neg): {asl_weights[y_train==1].mean()/asl_weights[y_train==0].mean():.1f}")
    
    # Tree-based models with ASL weighting
    models = {}
    
    print("\n[Training] ASL-weighted tree models...")
    
    # 1. Gradient Boosting
    print("[1/5] Gradient Boosting...")
    gb = GradientBoostingClassifier(
        n_estimators=1000, learning_rate=0.01, max_depth=8,
        subsample=0.8, max_features='sqrt', random_state=42
    )
    gb.fit(X_train, y_train, sample_weight=asl_weights)
    models['gradient_boosting'] = gb
    
    # 2. Random Forest
    print("[2/5] Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=1000, max_depth=15, min_samples_split=5,
        max_features='sqrt', class_weight={0: 1, 1: 20}, random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train, sample_weight=asl_weights)
    models['random_forest'] = rf
    
    # 3. Extra Trees
    print("[3/5] Extra Trees...")
    et = ExtraTreesClassifier(
        n_estimators=1000, max_depth=15, min_samples_split=5,
        max_features='sqrt', class_weight={0: 1, 1: 20}, random_state=42, n_jobs=-1
    )
    et.fit(X_train, y_train, sample_weight=asl_weights)
    models['extra_trees'] = et
    
    # 4. AdaBoost
    print("[4/5] AdaBoost...")
    ada = AdaBoostClassifier(
        n_estimators=500, learning_rate=0.1, random_state=42
    )
    ada.fit(X_train, y_train, sample_weight=asl_weights)
    models['adaboost'] = ada
    
    # 5. Decision Tree (Deep)
    print("[5/5] Deep Decision Tree...")
    dt = DecisionTreeClassifier(
        max_depth=20, min_samples_split=10, min_samples_leaf=5,
        class_weight={0: 1, 1: 20}, random_state=42
    )
    dt.fit(X_train, y_train, sample_weight=asl_weights)
    models['decision_tree'] = dt
    
    # Evaluate all models
    print("\n[Evaluation] Testing models...")
    results = {}
    predictions = {}
    
    for name, model in models.items():
        print(f"[Eval] {name}...")
        
        # Get predictions
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba = model.predict(X_test)
        
        # Calibrate probabilities
        calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=3)
        calibrated_model.fit(X_train, y_train)
        y_pred_proba_cal = calibrated_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics_raw = calculate_metrics(y_test, y_pred_proba)
        metrics_cal = calculate_metrics(y_test, y_pred_proba_cal)
        
        results[name] = {
            'raw': metrics_raw,
            'calibrated': metrics_cal
        }
        predictions[name] = y_pred_proba_cal
        
        print(f"  Raw AUC: {metrics_raw['auc']:.4f}, Calibrated AUC: {metrics_cal['auc']:.4f}")
        print(f"  Raw Brier: {metrics_raw['brier_score']:.4f}, Calibrated Brier: {metrics_cal['brier_score']:.4f}")
    
    # Find best model
    best_model = max(results.keys(), key=lambda x: results[x]['calibrated']['auc'])
    best_metrics = results[best_model]['calibrated']
    
    print(f"\n[Best Model] {best_model}")
    print(f"[Results] AUC: {best_metrics['auc']:.4f}, F1: {best_metrics['f1']:.4f}")
    print(f"[Results] Brier: {best_metrics['brier_score']:.4f}, BSS: {best_metrics['brier_skill_score']:.4f}")
    
    # Save results
    output = {
        'best_model': best_model,
        'best_metrics': best_metrics,
        'all_results': results,
        'data_info': {
            'total_samples': len(df_filtered),
            'features': X.shape[1],
            'class_balance': np.bincount(y).tolist()
        }
    }
    
    with open(os.path.join(out_dir, "asl_tree_results.json"), "w") as f:
        json.dump(output, f, indent=2)
    
    # Plot ROC curves
    plt.figure(figsize=(12, 8))
    for name, y_pred in predictions.items():
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred)
        plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {auc_score:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ASL Tree Models - ROC Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "asl_tree_roc.png"), dpi=150)
    plt.close()
    
    print(f"[Done] Results saved to {out_dir}")

if __name__ == "__main__":
    # Starter13 variables
    PREDICTORS = [
        'age', 'educ', 'sex', 'race', 'region', 'year', 'relig', 'attend', 
        'income', 'marital', 'polviews', 'class', 'degree', 'wrkstat'
    ]
    
    ITEMS = ['sexornt', 'premarsx', 'xmarsex']
    
    parser = argparse.ArgumentParser(description="ASL Tree-based GSS Analysis")
    parser.add_argument("--data", default="data/GSS2.xlsx", help="Data file path")
    parser.add_argument("--out", default="outputs_asl_tree", help="Output directory")
    args = parser.parse_args()
    
    run_asl_tree_pipeline(args.data, args.out, ITEMS, PREDICTORS)