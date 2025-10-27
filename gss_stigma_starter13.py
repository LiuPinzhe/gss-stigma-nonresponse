#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GSS Stigma-Related Nonresponse — Recall-First Pipeline (Precision Floor)
=======================================================================

Goal
----
Identify likely refusers with high recall while keeping precision acceptable.
Selection rule: maximize recall among thresholds with precision >= precision_floor (default 0.20).

Outputs
-------
- feature_importance.csv
- model_metrics.json
- classification_report.txt
- roc_curves.png
- pr_curve.png (marked chosen threshold)
- calibration_plot.png
- run_log.json
"""

import argparse
import os
import json
import warnings
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    roc_curve, precision_recall_curve, average_precision_score,
    brier_score_loss
)
from sklearn.calibration import calibration_curve
from sklearn.utils import class_weight
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------
# Defaults & Config
# -----------------------------
DEFAULT_ITEMS: List[str] = [
    "homosex"   # homosexual attitude (example single item; add more if needed)
]

DEFAULT_PREDICTORS: List[str] = [
    "age", "educ", "sex", "race", "region", "year",
    "relig", "attend", "income", "marital", "divorce", 
    "wrkstat", "spwrksta", "childs", "speduc", "res16", "reg16", "rincome", "xnorcsiz", "partyid", "polviews", "natenvir", "natheal", "natcity", "natcrime", "natdrug", "nateduc", "natrace", "cappun", "gunlaw", "happy", "hapmar", "health", "life", "sexeduc", "class_", "finrela", "owngun", "ethnic", "hispanic", "spaneng"
]

# Engineered numeric features we will force to numeric
CONTINUOUS_PREDICTORS: List[str] = [
    "age", "childs", "year", "year_centered", "year_squared",
    "age_clean", "age_squared", "age_log",
    "year_clean", "year_scaled", "decade"  # decade may be treated numeric or categorical; we keep numeric here
]

NON_SUBSTANTIVE_CODES = {0, 8, 9, 98, 99, 998, 999}
RANDOM_SEED = 42

# Precision floor for recall-first selection
PRECISION_FLOOR = 0.20


# -----------------------------
# Utils
# -----------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


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
    # If the series dtype is numeric and not just small code categories,
    # we still will handle in the numeric path.
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    return "categorical"


def mark_nonresponse(series: pd.Series) -> pd.Series:
    """
    Binary indicator: 1 if nonresponse (missing/DK/REF), else 0.
    .i (Inapplicable) -> NaN (not asked)
    """
    s = series.copy()
    if pd.api.types.is_numeric_dtype(s):
        mask = s.isna() | s.isin(NON_SUBSTANTIVE_CODES) | (s >= 97)
        return mask.astype(int)
    else:
        s_str = s.astype(str).str.strip()
        inapplicable_mask = s_str.str.startswith(".i")
        nonresponse_mask = (
            s.isna() |
            s_str.isin(["", "NA", "NaN", "nan"]) |
            s_str.str.startswith(".d") |  # DK/Cannot choose
            s_str.str.startswith(".s") |  # Skipped
            s_str.str.startswith(".n")    # No answer
        )
        result = pd.Series(np.nan, index=s.index, dtype=float)
        result[~inapplicable_mask & ~nonresponse_mask] = 0
        result[~inapplicable_mask & nonresponse_mask] = 1
        return result


def create_advanced_features(df: pd.DataFrame, base_predictors: List[str]) -> pd.DataFrame:
    """
    Create engineered features. Do NOT coerce everything to numeric here.
    Numeric transforms only where appropriate.
    """
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


def build_feature_matrix(df: pd.DataFrame, predictors: List[str]) -> pd.DataFrame:
    """
    Build model-ready X with:
     - Engineered features
     - Numeric handling (median fill)
     - Categorical handling with 'NR' as explicit level and dummies
     - Low variance pruning
    """
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

    # ---- Missingness features from raw categorical values (before dummies)
    if categorical_cols:
# BEFORE one-hot encoding
        nr_hits = X_advanced[categorical_cols].apply(lambda s: s.astype(str).str.strip().eq("NR")).astype(int)
        print(nr_hits)
        X_advanced["total_missing_predictors"] = nr_hits.sum(axis=1)
        X_advanced["prop_missing_predictors"]  = nr_hits.mean(axis=1)

    # Categorical encoding: keep top categories (broaden to 50 to preserve subgroups)
    if categorical_cols:
        for col in categorical_cols:
            top_categories = X_advanced[col].value_counts(dropna=True).head(50).index
            X_advanced[col] = X_advanced[col].where(X_advanced[col].isin(top_categories), "OTHER")
            # frequency encoding companion (can help trees)
            freq_encoding = X_advanced[col].value_counts(normalize=True)
            X_advanced[f"{col}_freq"] = X_advanced[col].map(freq_encoding)

        X_advanced = pd.get_dummies(X_advanced, columns=categorical_cols, drop_first=True, prefix=categorical_cols)

    # Add missingness-derived features BEFORE one-hot encoding
    missing_counts = df[actual_predictors].isna().sum(axis=1)
    X_advanced["total_missing_predictors"] = missing_counts
    X_advanced["prop_missing_predictors"] = missing_counts / len(actual_predictors)

    # Low variance prune
    nunique = X_advanced.nunique()
    X_advanced = X_advanced.loc[:, nunique > 1]

    return X_advanced


# -----------------------------
# Threshold optimization (Recall-first with Precision floor)
# -----------------------------
def optimize_threshold_with_precision_floor(y_true, y_proba, precision_floor=PRECISION_FLOOR):
    """
    Maximize recall subject to precision >= precision_floor.
    Returns: best_threshold, precision_at_best, recall_at_best
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # Align indices: thresholds length is len(precision)-1
    valid = precision >= precision_floor
    if valid.any():
        idxs = np.where(valid)[0]
        best_k = idxs[np.argmax(recall[idxs])]
        thr_idx = max(best_k - 1, 0)
        best_thr = thresholds[thr_idx] if len(thresholds) > 0 else 0.5
        return float(best_thr), float(precision[best_k]), float(recall[best_k])

    # Fallback: if nothing meets floor, choose highest precision point (safest)
    best_k = int(np.argmax(precision))
    thr_idx = max(best_k - 1, 0)
    best_thr = thresholds[thr_idx] if len(thresholds) > 0 else 0.5
    return float(best_thr), float(precision[best_k]), float(recall[best_k])


# -----------------------------
# Class imbalance helper
# -----------------------------
def manual_oversample(X_train, y_train):
    """Manual oversampling for the minority class."""
    minority_class = 1 if (y_train == 1).sum() < (y_train == 0).sum() else 0
    X_min = X_train[y_train == minority_class]
    y_min = y_train[y_train == minority_class]
    X_maj = X_train[y_train != minority_class]
    y_maj = y_train[y_train != minority_class]

    n_min, n_maj = len(X_min), len(X_maj)
    if n_min == 0 or n_maj == 0:
        return X_train, y_train

    if n_min < n_maj:
        reps = n_maj // n_min
        rem = n_maj % n_min
        X_min_over = pd.concat([X_min] * reps + [X_min.iloc[:rem]], axis=0)
        y_min_over = pd.concat([y_min] * reps + [y_min.iloc[:rem]], axis=0)
        X_bal = pd.concat([X_maj, X_min_over], axis=0)
        y_bal = pd.concat([y_maj, y_min_over], axis=0)
        idx = np.random.permutation(len(X_bal))
        return X_bal.iloc[idx], y_bal.iloc[idx]

    return X_train, y_train


# -----------------------------
# Training (Recall-first selection with precision floor)
# -----------------------------
def train_models_recall_priority(X_train, y_train, X_test, y_test, use_oversampling=True, precision_floor=PRECISION_FLOOR):
    print("[Training] Recall-first with precision floor...")

    # Compute class weights (used for RF)
    cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    cw_dict = {0: cw[0], 1: cw[1]}

    # Optionally oversample minority
    if use_oversampling:
        X_train_proc, y_train_proc = manual_oversample(X_train, y_train)
        print(f"[Oversampling] Class distribution after: {np.bincount(y_train_proc)}")
    else:
        X_train_proc, y_train_proc = X_train, y_train

    models = {
        "rf_balanced": RandomForestClassifier(
            n_estimators=300, max_depth=25, min_samples_split=5, min_samples_leaf=2,
            max_features='log2', class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8, random_state=RANDOM_SEED
        ),
        "catboost": CatBoostClassifier(
            depth=6, learning_rate=0.05, iterations=500,
            loss_function="Logloss",
            random_state=RANDOM_SEED,
            verbose=False,
            class_weights=[cw_dict[0], cw_dict[1]]
        )
    }

    best_model = None
    best_model_name = None
    best_metrics = None
    all_metrics = {}

    def selector_key(m):
        # maximize recall; break ties by higher precision then PR-AUC
        return (m["recall_at_floor"], m["precision_at_floor"], m["pr_auc"])

    for name, model in models.items():
        print(f"[Fit] {name}")
        model.fit(X_train_proc, y_train_proc)

        y_proba = model.predict_proba(X_test)[:, 1]

        roc = roc_auc_score(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)

        thr, p_at, r_at = optimize_threshold_with_precision_floor(y_test, y_proba, precision_floor)

        m = {
            "roc_auc": float(roc),
            "pr_auc": float(pr_auc),
            "precision_floor": float(precision_floor),
            "threshold_floor": float(thr),
            "precision_at_floor": float(p_at),
            "recall_at_floor": float(r_at),
        }
        all_metrics[name] = m

        print(f"[{name}] ROC-AUC={roc:.3f}, PR-AUC={pr_auc:.3f}, "
              f"Prec@{precision_floor:.2f}={p_at:.3f}, Rec@floor={r_at:.3f}, Thr={thr:.3f}")

        if best_metrics is None or selector_key(m) > selector_key(best_metrics):
            best_metrics = m
            best_model = model
            best_model_name = name

    print(f"[Best Model] {best_model_name} | Rec@floor={best_metrics['recall_at_floor']:.3f}, "
          f"Prec@floor={best_metrics['precision_at_floor']:.3f}, PR-AUC={best_metrics['pr_auc']:.3f}")

    # Feature importance
    if hasattr(best_model, "feature_importances_"):
        feature_importance = pd.DataFrame({
            "feature": X_train.columns,
            "importance": best_model.feature_importances_
        }).sort_values("importance", ascending=False)
    else:
        feature_importance = pd.DataFrame({
            "feature": X_train.columns,
            "importance": np.ones(len(X_train.columns)) / len(X_train.columns)
        })

    return best_model, best_metrics, feature_importance, all_metrics, models


# -----------------------------
# Core pipeline
# -----------------------------
def run_pipeline(data_path: str, out_dir: str, items: List[str], predictors: List[str],
                 bootstrap_iters: int = 200, use_oversampling: bool = True, mode: str = "composite"):
    ensure_dir(out_dir)

    # Load data
    print(f"[Load] {data_path}")
    needed_cols = list(set(items + predictors + ["year"]))

    df = None
    if data_path.endswith(".xlsx"):
        df = pd.read_excel(data_path, usecols=needed_cols)
        print("[Load] Excel")
    elif data_path.endswith(".sav"):
        try:
            import pyreadstat
            df, meta = pyreadstat.read_sav(data_path, usecols=needed_cols, apply_value_formats=True)
            print("[Load] SPSS")
        except Exception as e:
            raise RuntimeError("Reading .sav requires pyreadstat. Install it or provide .xlsx/.dta.") from e
    else:
        df = pd.read_stata(data_path, convert_categoricals=False, columns=needed_cols)
        print("[Load] Stata")
    print(f"[Load] shape={df.shape[0]:,} x {df.shape[1]:,}")

    # Items/predictors present
    items_present = available_columns(df, items)
    predictors_present = available_columns(df, predictors)
    print(f"[Items] {items_present}")
    print(f"[Predictors] {predictors_present}")
    if not predictors_present:
        raise ValueError("No predictor variables found.")

    # Build nonresponse indicators
    nr_cols = []
    for var in items_present:
        nr = f"NR_{var}"
        df[nr] = mark_nonresponse(df[var])
        nr_cols.append(nr)
        valid = (df[nr] == 0).sum()
        refused = (df[nr] == 1).sum()
        not_asked = df[nr].isna().sum()
        print(f"[{var}] Valid={valid:,}, Refused={refused:,}, Not asked={not_asked:,}")

    if not nr_cols:
        raise ValueError("None of the specified --items were found.")

    # Target
    if mode.lower() == "single":
        target = nr_cols[0]
        print(f"[Target] Single: {target}")
    elif mode.lower() == "composite":
        # Composite: 1 if refused any item among the items_present (only among those asked any)
        df["NR_SEX"] = np.nan
        asked_any = pd.Series(False, index=df.index)
        refused_any = pd.Series(False, index=df.index)
        for var in items_present:
            nr_var = mark_nonresponse(df[var])
            asked_any |= nr_var.notna()
            refused_any |= (nr_var == 1)
        df.loc[asked_any, "NR_SEX"] = refused_any[asked_any].astype(int)
        target = "NR_SEX"
        print(f"[Target] Composite: {target} from {items_present}")
    else:
        raise ValueError("Invalid mode. Use 'single' or 'composite'.")

    # Filter to asked cases
    valid_cases = df[target].notna()
    df_filtered = df[valid_cases].copy()
    print(f"[Filter] Using {valid_cases.sum():,} / {len(df):,} ({valid_cases.mean():.1%})")

    # Build X and y
    print("[Features] Building feature matrix...")
    X = build_feature_matrix(df_filtered, predictors_present)
    y = df_filtered[target].astype(int).reindex(X.index)
    print(f"[Features] X: {X.shape}, y: {y.shape}, nonresponse rate={y.mean():.4f}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    print(f"[Split] X_train={X_train.shape}, X_test={X_test.shape}")

    # Train
    best_model, best_metrics, feature_importance, all_metrics, models = train_models_recall_priority(
        X_train, y_train, X_test, y_test, use_oversampling=use_oversampling, precision_floor=PRECISION_FLOOR
    )

    # Save feature importance
    feature_importance.to_csv(os.path.join(out_dir, "feature_importance.csv"), index=False)
    print("[Feature Importance] Saved feature_importance.csv")

    # Final predictions at chosen threshold
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    optimal_threshold = best_metrics["threshold_floor"]
    y_pred_optimized = (y_pred_proba >= optimal_threshold).astype(int)

    # Diagnostics
    pr_auc = average_precision_score(y_test, y_pred_proba)
    brier = brier_score_loss(y_test, y_pred_proba)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    prec, rec, thr = precision_recall_curve(y_test, y_pred_proba)

    # Plots
    # ROC
    plt.figure(figsize=(9, 7))
    for name, mdl in models.items():
        yp = mdl.predict_proba(X_test)[:, 1]
        f, t, _ = roc_curve(y_test, yp)
        auc = roc_auc_score(y_test, yp)
        plt.plot(f, t, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC Curves - Model Comparison ({mode} mode)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "roc_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("[ROC] Saved roc_curves.png")

    # PR curve (mark chosen point)
    plt.figure(figsize=(9, 7))
    plt.plot(rec, prec, label=f"Best model PR (AP={pr_auc:.3f})")
    # chosen point
    # Align index for chosen threshold
    # Find the closest threshold index to optimal_threshold
    if len(thr) > 0:
        idx_thr = np.argmin(np.abs(thr - optimal_threshold))
        plt.scatter(rec[idx_thr], prec[idx_thr], s=80, zorder=3, label="Chosen threshold")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (chosen point marked)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pr_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("[PR] Saved pr_curve.png")

    # Calibration
    frac_pos, mean_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)
    plt.figure(figsize=(7, 6))
    plt.plot(mean_pred, frac_pos, marker="o", label="Calibration")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Calibration Plot")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "calibration_plot.png"), dpi=150)
    plt.close()
    print("[Calibration] Saved calibration_plot.png")

    # Save metrics JSON
    final_metrics = {
        "best_model": best_model.__class__.__name__,
        "best_model_key": best_metrics,  # includes pr_auc, threshold, precision/recall at floor
        "threshold_best": float(optimal_threshold),
        "roc_auc": float(best_metrics["roc_auc"]),
        "pr_auc": float(pr_auc),
        "precision_floor": float(best_metrics["precision_floor"]),
        "precision_best": float(best_metrics["precision_at_floor"]),
        "recall_best": float(best_metrics["recall_at_floor"]),
        "brier_score": float(brier),
        "all_models": all_metrics
    }
    with open(os.path.join(out_dir, "model_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)
    print("[Metrics] Saved model_metrics.json")

    # Save detailed classification report
    with open(os.path.join(out_dir, "classification_report.txt"), "w") as f:
        f.write("Recall-First Model (Precision Floor) - Classification Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Mode: {mode}\nTarget: {target}\n")
        f.write(f"Best Model: {best_model.__class__.__name__}\n")
        f.write(f"ROC-AUC: {best_metrics['roc_auc']:.4f}\n")
        f.write(f"PR-AUC: {pr_auc:.4f}\n")
        f.write(f"Brier Score: {brier:.4f}\n")
        f.write(f"Precision Floor: {best_metrics['precision_floor']:.2f}\n")
        f.write(f"Decision Threshold: {optimal_threshold:.4f}\n")
        f.write(f"Precision@floor: {best_metrics['precision_at_floor']:.4f}\n")
        f.write(f"Recall@floor: {best_metrics['recall_at_floor']:.4f}\n\n")
        f.write("Detailed Classification Report (at chosen threshold):\n")
        f.write(classification_report(y_test, y_pred_optimized))
        f.write(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred_optimized)}\n")
    print("[Report] Saved classification_report.txt")

    # Run log (compact)
    run_log = {
        "data_path": data_path,
        "out_dir": out_dir,
        "mode": mode,
        "target_modeled": target,
        "composite_items": items_present if mode.lower() == "composite" else None,
        "predictors_used": predictors_present,
        "items_present": items_present,
        "feature_matrix_shape": list(X.shape),
        "precision_floor": PRECISION_FLOOR,
        "metrics_summary": {
            "roc_auc": final_metrics["roc_auc"],
            "pr_auc": final_metrics["pr_auc"],
            "precision_best": final_metrics["precision_best"],
            "recall_best": final_metrics["recall_best"],
            "threshold_best": final_metrics["threshold_best"],
            "brier_score": final_metrics["brier_score"]
        }
    }
    with open(os.path.join(out_dir, "run_log.json"), "w", encoding="utf-8") as f:
        json.dump(run_log, f, indent=2)
    print("[Run Log] Saved run_log.json")

    # Console summary
    print("\n[Final Results]")
    print(f"Mode: {mode}")
    print(f"Target: {target}")
    print(f"ROC-AUC: {final_metrics['roc_auc']:.4f}")
    print(f"PR-AUC: {final_metrics['pr_auc']:.4f}")
    print(f"Brier Score: {final_metrics['brier_score']:.4f}")
    print(f"Precision Floor: {final_metrics['precision_floor']:.2f}")
    print(f"Precision@floor: {final_metrics['precision_best']:.4f}")
    print(f"Recall@floor: {final_metrics['recall_best']:.4f}")
    print(f"Decision Threshold: {final_metrics['threshold_best']:.4f}")
    print(f"[Done] Results saved to: {out_dir}")


def parse_args():
    ap = argparse.ArgumentParser(description="GSS stigma-related nonresponse - Recall-first pipeline (precision floor)")
    ap.add_argument("--data", required=True, help="Path to GSS data file (.xlsx, .dta, or .sav)")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--items", nargs="*", default=DEFAULT_ITEMS, help="Sensitive items")
    ap.add_argument("--predictors", nargs="*", default=DEFAULT_PREDICTORS, help="Predictor variables")
    ap.add_argument("--bootstrap", type=int, default=200, help="Bootstrap iterations (placeholder)")
    ap.add_argument("--no-oversampling", action="store_true", help="Disable manual oversampling")
    ap.add_argument("--mode", choices=["single", "composite"], default="composite",
                    help="Modeling mode: 'single' (first item only) or 'composite' (any item nonresponse)")
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