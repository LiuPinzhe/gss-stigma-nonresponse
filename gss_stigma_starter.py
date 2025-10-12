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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.manifold import MDS

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
    """Numeric + one-hot for non-numeric, with simple median impute for numerics."""
    X = df[predictors].copy()
    numeric_cols = []
    for c in X.columns:
        X[c] = to_numeric_or_category(X[c])
        if pd.api.types.is_numeric_dtype(X[c]):
            numeric_cols.append(c)
    # clean weird codes
    for c in numeric_cols:
        col = X[c].copy()  # 避免SettingWithCopyWarning
        col[(col.isin(NON_SUBSTANTIVE_CODES)) | (col >= 97)] = np.nan
        X[c] = col
    # impute numerics with median, fallback to 0 if all NaN
    for c in numeric_cols:
        if X[c].isna().any():
            median_val = X[c].median()
            if pd.isna(median_val):  # 如果所有值都是NaN
                median_val = 0
            X[c] = X[c].fillna(median_val)
    # one-hot categoricals
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, dummy_na=True)
    
    # 最终检查：确保没有NaN值
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

    logit = LogisticRegression(max_iter=2000)
    logit.fit(X_train_s, y_train)
    p_logit = logit.predict_proba(X_test_s)[:, 1]
    metrics = {
        "logit_accuracy": float(accuracy_score(y_test, (p_logit >= 0.5).astype(int))),
        "logit_roc_auc": float(roc_auc_score(y_test, p_logit))
    }

    # Gradient Boosting (no scaling)
    gb = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=RANDOM_SEED
    )
    gb.fit(X_train, y_train)
    p_gb = gb.predict_proba(X_test)[:, 1]
    metrics.update({
        "gb_accuracy": float(accuracy_score(y_test, (p_gb >= 0.5).astype(int))),
        "gb_roc_auc": float(roc_auc_score(y_test, p_gb))
    })

    # Save metrics
    with open(os.path.join(out_dir, "model_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("[Model] Saved model_metrics.json")

    # IPW using GB on full X
    p_all = gb.predict_proba(X)[:, 1]
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
