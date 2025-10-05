#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GSS Stigma-Related Nonresponse — Starter Pipeline (1972–2024)
v1.1 — case-insensitive variable matching for GSS columns

Changes in v1.1:
- Match item and predictor variable names case-insensitively (GSS often stores in lowercase)
- Print diagnostic suggestions if variables aren't found
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

DEFAULT_ITEMS: List[str] = [
    "SEXORNT", "PREMARSX", "XMARSEX", "HOMOSEX", "GAYMARRY", "GAYMAR"
]

DEFAULT_PREDICTORS: List[str] = [
    "AGE", "EDUC", "DEGREE", "SEX", "RACE", "REGION", "YEAR",
    "RELIG", "ATTEND", "POLVIEWS", "INCOME", "REALINC", "MARITAL", "HISPANIC"
]

NON_SUBSTANTIVE_CODES = {0, 8, 9, 98, 99, 998, 999}
RANDOM_SEED = 42


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def to_numeric_or_category(s: pd.Series) -> pd.Series:
    try:
        sn = pd.to_numeric(s, errors="coerce")
        if sn.notna().mean() > 0.3:
            return sn
        return s
    except Exception:
        return s


def mark_nonresponse(series: pd.Series) -> pd.Series:
    s = to_numeric_or_category(series)
    if pd.api.types.is_numeric_dtype(s):
        mask = s.isna() | s.isin(NON_SUBSTANTIVE_CODES) | (s >= 97)
    else:
        mask = s.isna() | (s.astype(str).str.strip().isin(["", "NA", "NaN", "nan"]))
    return mask.astype(int)


def sanitize_numeric(series: pd.Series) -> pd.Series:
    s = to_numeric_or_category(series)
    if pd.api.types.is_numeric_dtype(s):
        s = s.copy()
        s[(s.isin(NON_SUBSTANTIVE_CODES)) | (s >= 97)] = np.nan
        return s
    return series


def case_insensitive_match(df_cols: List[str], candidates: List[str]) -> Dict[str, str]:
    """
    Return a mapping {canonical_candidate: actual_df_col} using case-insensitive lookup.
    """
    lower_map = {c.lower(): c for c in df_cols}
    out = {}
    for cand in candidates:
        key = cand.lower()
        if key in lower_map:
            out[cand] = lower_map[key]
    return out


def build_design_matrix(df: pd.DataFrame, predictors_actual: List[str]) -> pd.DataFrame:
    X = df[predictors_actual].copy()
    numeric_cols = []
    for c in X.columns:
        X[c] = to_numeric_or_category(X[c])
        if pd.api.types.is_numeric_dtype(X[c]):
            numeric_cols.append(c)
    for c in numeric_cols:
        col = X[c]
        col[(col.isin(NON_SUBSTANTIVE_CODES)) | (col >= 97)] = np.nan
        X[c] = col
    for c in numeric_cols:
        if X[c].isna().any():
            X[c] = X[c].fillna(X[c].median())
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    X = pd.get_dummies(X, columns=cat_cols, dummy_na=True)
    return X


def bootstrap_ci(func, data_idx: np.ndarray, B: int = 500, alpha: float = 0.05, random_state: int = 42) -> Tuple[float, float]:
    rng = np.random.default_rng(random_state)
    stats = []
    n = len(data_idx)
    for _ in range(B):
        sample_idx = rng.integers(0, n, size=n)
        stats.append(func(data_idx[sample_idx]))
    lo = np.percentile(stats, 100 * (alpha/2))
    hi = np.percentile(stats, 100 * (1 - alpha/2))
    return float(lo), float(hi)


def run_pipeline(
    data_path: str,
    out_dir: str,
    items: List[str],
    predictors: List[str],
    bootstrap_iters: int = 500
):
    ensure_dir(out_dir)

    print(f"[Load] {data_path}")
    df = pd.read_stata(data_path, convert_categoricals=False)
    print(f"[Load] shape={df.shape[0]:,} x {df.shape[1]:,}")

    # Case-insensitive resolve
    item_map = case_insensitive_match(list(df.columns), items)
    pred_map = case_insensitive_match(list(df.columns), predictors)

    items_present = list(item_map.values())
    predictors_present = list(pred_map.values())

    if not predictors_present:
        # Print diagnostics: show top likely basic predictors if present in lowercase
        basics = ["year", "age", "sex", "race", "educ", "degree", "marital", "relig", "attend", "polviews", "realinc", "income", "region", "hispanic"]
        have = [c for c in basics if c in [col.lower() for col in df.columns]]
        print("[Diag] Could not find any of the requested predictors (case-insensitive).")
        print(f"[Diag] However, I do see these typical basics in your file: {have}")
        raise ValueError("No predictor variables found after case-insensitive matching. "
                         "Try passing --predictors with actual column names (likely lowercase in your file).")

    if not items_present:
        basics_items = ["sexornt", "premarsx", "xmarsex", "homosex", "gaymar", "gaymarry"]
        have_items = [c for c in basics_items if c in [col.lower() for col in df.columns]]
        print("[Diag] Could not find any sensitive items from your list.")
        print(f"[Diag] I do see these possible candidates: {have_items}")
        raise ValueError("No sensitive items found after case-insensitive matching. "
                         "Try passing --items with actual column names (likely lowercase).")

    # Build nonresponse indicators
    nr_cols = []
    for canon, actual in item_map.items():
        nr = f"NR_{actual}"
        df[nr] = mark_nonresponse(df[actual])
        nr_cols.append(nr)

    target = nr_cols[0]
    print(f"[Target] Modeling nonresponse for: {target}")

    # Design matrix
    X = build_design_matrix(df, predictors_present)
    y = df[target].astype(int).reindex(X.index)

    # Train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    # Logistic Regression
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

    # Gradient Boosting
    gb = GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=3, random_state=RANDOM_SEED
    )
    gb.fit(X_train, y_train)
    p_gb = gb.predict_proba(X_test)[:, 1]
    metrics.update({
        "gb_accuracy": float(accuracy_score(y_test, (p_gb >= 0.5).astype(int))),
        "gb_roc_auc": float(roc_auc_score(y_test, p_gb))
    })

    with open(os.path.join(out_dir, "model_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("[Model] Saved model_metrics.json")

    # IPW on full X
    p_all = gb.predict_proba(X)[:, 1]
    p_all = np.clip(p_all, 1e-6, 0.95)
    df["IPW"] = pd.Series(1.0 / (1.0 - p_all), index=X.index)

    # Reweighted estimation: use first available *numeric* item as outcome
    outcome_var = None
    for v in items_present:
        if v in df.columns:
            cand = sanitize_numeric(df[v])
            if pd.api.types.is_numeric_dtype(cand):
                outcome_var = v
                break

    if outcome_var is not None:
        y_out = sanitize_numeric(df[outcome_var])
        mask = y_out.notna() & df["IPW"].notna()
        yvals = y_out[mask].to_numpy()
        wvals = df.loc[mask, "IPW"].to_numpy()
        unadj_mean = float(np.nanmean(yvals))
        ipw_mean = float(np.average(yvals, weights=wvals))

        idx = np.arange(len(yvals))

        def stat_unadj(sample_idx):
            return float(np.nanmean(yvals[sample_idx]))

        def stat_ipw(sample_idx):
            return float(np.average(yvals[sample_idx], weights=wvals[sample_idx]))

        lo_u, hi_u = bootstrap_ci(stat_unadj, idx, B=500, alpha=0.05, random_state=RANDOM_SEED)
        lo_w, hi_w = bootstrap_ci(stat_ipw, idx, B=500, alpha=0.05, random_state=RANDOM_SEED)

        compare = pd.DataFrame({
            "metric": ["mean_unadjusted", "mean_ipw_adjusted"],
            "value": [unadj_mean, ipw_mean],
            "ci_lo": [lo_u, lo_w],
            "ci_hi": [hi_u, hi_w],
            "outcome": [outcome_var, outcome_var]
        })
        compare.to_csv(os.path.join(out_dir, "adjustment_comparison.csv"), index=False)
        print(f"[IPW] Saved adjustment_comparison.csv (outcome={outcome_var})")
    else:
        print("[IPW] No numeric outcome among items; skipped comparison.")

    # Temporal trends per YEAR
    if "year" in [c.lower() for c in df.columns]:
        # get actual column name for YEAR
        year_actual = [c for c in df.columns if c.lower() == "year"][0]
        year_series = to_numeric_or_category(df[year_actual])
        if pd.api.types.is_numeric_dtype(year_series):
            tmp = pd.DataFrame({"YEAR": year_series, "NR": df[target].astype(float)}).dropna()
            records = []
            for yr, grp in tmp.groupby("YEAR"):
                arr = grp["NR"].to_numpy()
                idx = np.arange(len(arr))
                mean_nr = float(arr.mean())

                def stat(sample_idx):
                    return float(arr[sample_idx].mean())

                lo, hi = bootstrap_ci(stat, idx, B=min(500, max(100, len(arr))), alpha=0.05, random_state=RANDOM_SEED)
                records.append({"YEAR": int(yr), "nr_mean": mean_nr, "ci_lo": lo, "ci_hi": hi})

            trend = pd.DataFrame.from_records(records).sort_values("YEAR")
            trend.to_csv(os.path.join(out_dir, f"trend_{target}.csv"), index=False)
            print(f"[Trend] Saved trend_{target}.csv")

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

    # MDS
    if len(nr_cols) >= 2:
        corr = df[nr_cols].corr(method="pearson", min_periods=200).fillna(0.0)
        dist = 1.0 - corr.abs().values
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

    run_log = {
        "data_path": data_path,
        "out_dir": out_dir,
        "target_modeled": target,
        "predictors_used": predictors_present,
        "items_present": items_present,
        "nonresponse_indicators": nr_cols,
        "metrics": metrics
    }
    with open(os.path.join(out_dir, "run_log.json"), "w", encoding="utf-8") as f:
        json.dump(run_log, f, indent=2)
    print("[Done] Saved run_log.json")


def parse_args():
    ap = argparse.ArgumentParser(description="GSS stigma-related nonresponse starter pipeline (v1.1)")
    ap.add_argument("--data", required=True, help="Path to GSS .dta (e.g., gss7224_r1.dta)")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--items", nargs="*", default=DEFAULT_ITEMS, help="Sensitive items list (case-insensitive)")
    ap.add_argument("--predictors", nargs="*", default=DEFAULT_PREDICTORS, help="Predictors list (case-insensitive)")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        data_path=args.data,
        out_dir=args.out,
        items=args.items,
        predictors=args.predictors,
        bootstrap_iters=500
    )
