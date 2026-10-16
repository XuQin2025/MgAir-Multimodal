# -*- coding: utf-8 -*-
"""Build ML supplementary figure/source tables from the paper's saved OOF results."""

from __future__ import annotations

import ast
import json
import math
import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "supplementary_materials" / "ML"
OOF_PATH = PROJECT_ROOT / "outputs_figure3_8models" / "Figure3_oof_predictions_8models.csv"
RAWDATA_PATH = PROJECT_ROOT / "rawdata.xlsx"
HPO_DIR = PROJECT_ROOT / "outputs_GBstack_hpo"

MODEL_ORDER = [
    "GB",
    "MLP",
    "DT",
    "ANN",
    "BERT-Text",
    "BatteryBERT-Text",
    "BERT-Fusion",
    "BatteryBERT-Fusion",
]
TARGET_UNITS = {"Voltage": "mV", "Efficiency": "%"}
SEED = 42


def save_csv(df: pd.DataFrame, name: str) -> None:
    df.to_csv(OUTPUT_DIR / name, index=False, encoding="utf-8-sig")


def table_s1() -> pd.DataFrame:
    rows = [
        ["GB", "Numeric", "39 structured descriptors", "GradientBoostingRegressor", "No", "Fixed baseline configuration", "Composition, processing, and current density"],
        ["MLP", "Numeric", "39 structured descriptors", "StandardScaler + MLP (100, 50)", "No", "Fixed baseline configuration", "Composition, processing, and current density"],
        ["DT", "Numeric", "39 structured descriptors", "DecisionTreeRegressor", "No", "Fixed baseline configuration", "Composition, processing, and current density"],
        ["ANN", "Numeric", "39 structured descriptors", "StandardScaler + MLP (128, 64, 32)", "No", "Fixed baseline configuration", "Composition, processing, and current density"],
        ["BERT-Text", "Text", "Text with numeric values", "BERT mean pooling + Ridge", "Frozen", "Nested Ridge alpha selection", "Composition, processing, and current-density values written into text"],
        ["BatteryBERT-Text", "Text", "Text with numeric values", "BatteryBERT mean pooling + Ridge", "Frozen", "Nested Ridge alpha selection", "Composition, processing, and current-density values written into text"],
        ["BERT-Fusion", "Multimodal", "Numeric + semantic text", "GB numeric + BERT/Ridge text + GB meta-learner", "Frozen", "Nested HPO for base and meta learners", "Fusion text contains element identities and process sequence, without numeric values"],
        ["BatteryBERT-Fusion", "Multimodal", "Numeric + semantic text", "GB numeric + BatteryBERT/Ridge text + GB meta-learner", "Frozen", "Nested HPO for base and meta learners", "Final surrogate; fusion text contains element identities and process sequence, without numeric values"],
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "Model",
            "Input modality",
            "Input representation",
            "Predictor",
            "Encoder fine-tuning",
            "Model selection / HPO",
            "Input-definition note",
        ],
    )


def table_s2_search_space() -> pd.DataFrame:
    rows = [
        ["Numeric GB", "n_estimators", "200, 300, 500, 800, 1200", "RandomizedSearchCV", 25],
        ["Numeric GB", "learning_rate", "0.01, 0.02, 0.03, 0.05, 0.08, 0.10", "RandomizedSearchCV", 25],
        ["Numeric GB", "max_depth", "2, 3, 4, 5", "RandomizedSearchCV", 25],
        ["Numeric GB", "subsample", "0.6, 0.7, 0.8, 0.9, 1.0", "RandomizedSearchCV", 25],
        ["Numeric GB", "min_samples_split", "2, 4, 6, 8, 10", "RandomizedSearchCV", 25],
        ["Numeric GB", "min_samples_leaf", "1, 2, 3, 4, 5", "RandomizedSearchCV", 25],
        ["Numeric GB", "max_features", "None, sqrt, log2", "RandomizedSearchCV", 25],
        ["Text Ridge", "alpha", "0.1, 0.2, 0.5, 1, 2, 5, 10, 30, 80", "RandomizedSearchCV", 9],
        ["Meta GB", "n_estimators", "200, 300, 500, 800, 1200", "RandomizedSearchCV", 12],
        ["Meta GB", "learning_rate", "0.01, 0.02, 0.03, 0.05, 0.08, 0.10", "RandomizedSearchCV", 12],
        ["Meta GB", "max_depth", "2, 3, 4, 5", "RandomizedSearchCV", 12],
        ["Meta GB", "subsample", "0.6, 0.7, 0.8, 0.9, 1.0", "RandomizedSearchCV", 12],
        ["Meta GB", "min_samples_split", "2, 4, 6, 8, 10", "RandomizedSearchCV", 12],
        ["Meta GB", "min_samples_leaf", "1, 2, 3, 4, 5", "RandomizedSearchCV", 12],
        ["Meta GB", "max_features", "None, sqrt, log2", "RandomizedSearchCV", 12],
    ]
    df = pd.DataFrame(rows, columns=["Component", "Hyperparameter", "Candidate values", "Search strategy", "Trials per outer fold"])
    df["Inner validation"] = "5-fold CV, scoring = R2"
    return df


def table_s2_selected() -> pd.DataFrame:
    rows = []
    for encoder_tag, encoder_name in [("BERT", "BERT"), ("BatteryBERT", "BatteryBERT")]:
        for target in ["Voltage", "Efficiency"]:
            path = HPO_DIR / f"BestParams_{encoder_tag}_GBstack_HPO_{target}.csv"
            source = pd.read_csv(path)
            for _, row in source.iterrows():
                num = ast.literal_eval(row["best_gb_num"])
                meta = ast.literal_eval(row["best_meta"])
                rows.append({
                    "Encoder": encoder_name,
                    "Target": target,
                    "Outer fold": int(row["Fold"]),
                    "Numeric GB parameters": json.dumps(num, ensure_ascii=False, sort_keys=True),
                    "Ridge alpha": float(row["best_ridge_alpha"]),
                    "Meta GB parameters": json.dumps(meta, ensure_ascii=False, sort_keys=True),
                })
    return pd.DataFrame(rows)


def bootstrap_metrics(y: np.ndarray, pred: np.ndarray, ref_pred: np.ndarray, n_boot: int = 2000) -> dict:
    rng = np.random.default_rng(SEED)
    n = len(y)
    values = np.empty((n_boot, 4), dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp, yr = y[idx], pred[idx], ref_pred[idx]
        values[b, 0] = r2_score(yt, yp)
        values[b, 1] = math.sqrt(mean_squared_error(yt, yp))
        values[b, 2] = mean_absolute_error(yt, yp)
        values[b, 3] = values[b, 1] - math.sqrt(mean_squared_error(yt, yr))
    result = {}
    for j, label in enumerate(["R2", "RMSE", "MAE", "Delta_RMSE_vs_GB"]):
        result[f"{label}_CI_low"], result[f"{label}_CI_high"] = np.quantile(values[:, j], [0.025, 0.975])
    return result


def table_s3(oof: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    fold_rows = []
    for target in ["Voltage", "Efficiency"]:
        target_df = oof[oof["Target"] == target]
        reference = target_df[target_df["Model"] == "GB"].sort_values("Sample")
        ref_pred = reference["OOF predicted"].to_numpy(float)
        for model in MODEL_ORDER:
            model_df = target_df[target_df["Model"] == model].sort_values("Sample")
            y = model_df["Measured"].to_numpy(float)
            pred = model_df["OOF predicted"].to_numpy(float)
            folds = model_df["Fold"].to_numpy(int)
            per_fold = []
            for fold in sorted(np.unique(folds)):
                mask = folds == fold
                row = {
                    "Target": target,
                    "Model": model,
                    "Fold": int(fold),
                    "N": int(mask.sum()),
                    "R2": r2_score(y[mask], pred[mask]),
                    "RMSE": math.sqrt(mean_squared_error(y[mask], pred[mask])),
                    "MAE": mean_absolute_error(y[mask], pred[mask]),
                }
                per_fold.append(row)
                fold_rows.append(row)
            fold_df = pd.DataFrame(per_fold)
            pooled_rmse = math.sqrt(mean_squared_error(y, pred))
            summary = {
                "Target": target,
                "Unit": TARGET_UNITS[target],
                "Model": model,
                "Input group": "Numeric" if model in MODEL_ORDER[:4] else "Text" if model in MODEL_ORDER[4:6] else "Multimodal fusion",
                "N": len(y),
                "Pooled R2": r2_score(y, pred),
                "Pooled RMSE": pooled_rmse,
                "Pooled MAE": mean_absolute_error(y, pred),
                "Fold R2 mean": fold_df["R2"].mean(),
                "Fold R2 SD": fold_df["R2"].std(ddof=1),
                "Fold RMSE mean": fold_df["RMSE"].mean(),
                "Fold RMSE SD": fold_df["RMSE"].std(ddof=1),
                "Fold MAE mean": fold_df["MAE"].mean(),
                "Fold MAE SD": fold_df["MAE"].std(ddof=1),
                "Delta RMSE vs GB": pooled_rmse - math.sqrt(mean_squared_error(y, ref_pred)),
            }
            summary.update(bootstrap_metrics(y, pred, ref_pred))
            summaries.append(summary)
    return pd.DataFrame(summaries), pd.DataFrame(fold_rows)


def compute_ad_scores(raw: pd.DataFrame, folds: np.ndarray) -> np.ndarray:
    numeric = raw.iloc[:, :39].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float)
    scores = np.zeros(len(raw), dtype=float)
    for fold in sorted(np.unique(folds)):
        test_mask = folds == fold
        train_mask = ~test_mask
        scaler = StandardScaler().fit(numeric[train_mask])
        train = scaler.transform(numeric[train_mask])
        test = scaler.transform(numeric[test_mask])
        k = 5
        train_nn = NearestNeighbors(n_neighbors=k + 1).fit(train)
        train_dist = train_nn.kneighbors(train, return_distance=True)[0][:, 1:].mean(axis=1)
        threshold = np.quantile(train_dist, 0.95)
        test_nn = NearestNeighbors(n_neighbors=k).fit(train)
        test_dist = test_nn.kneighbors(test, return_distance=True)[0].mean(axis=1)
        scores[test_mask] = test_dist / max(threshold, 1e-12)
    return scores


def coverage_table(oof: pd.DataFrame) -> pd.DataFrame:
    levels = np.array([0.50, 0.60, 0.70, 0.80, 0.90, 0.95])
    rows = []
    for target in ["Voltage", "Efficiency"]:
        d = oof[(oof["Target"] == target) & (oof["Model"] == "BatteryBERT-Fusion")].sort_values("Sample")
        errors = np.abs(d["Measured"].to_numpy(float) - d["OOF predicted"].to_numpy(float))
        folds = d["Fold"].to_numpy(int)
        for level in levels:
            covered = np.zeros(len(d), dtype=bool)
            half_width = np.zeros(len(d), dtype=float)
            for fold in sorted(np.unique(folds)):
                calibration = np.sort(errors[folds != fold])
                rank = int(np.ceil((len(calibration) + 1) * level)) - 1
                q = calibration[np.clip(rank, 0, len(calibration) - 1)]
                mask = folds == fold
                covered[mask] = errors[mask] <= q
                half_width[mask] = q
            rows.append({
                "Target": target,
                "Nominal coverage": level,
                "Empirical coverage": covered.mean(),
                "Mean interval half-width": half_width.mean(),
                "Unit": TARGET_UNITS[target],
            })
    return pd.DataFrame(rows)


def figure_s3(oof: pd.DataFrame, raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    best = oof[oof["Model"] == "BatteryBERT-Fusion"].copy()
    voltage = best[best["Target"] == "Voltage"].sort_values("Sample").reset_index(drop=True)
    efficiency = best[best["Target"] == "Efficiency"].sort_values("Sample").reset_index(drop=True)
    folds = voltage["Fold"].to_numpy(int)
    ad_score = compute_ad_scores(raw, folds)
    current_density = pd.to_numeric(raw.iloc[:, 38], errors="coerce").to_numpy(float)
    source = pd.DataFrame({
        "Sample": voltage["Sample"].to_numpy(int),
        "Fold": folds,
        "Current density (mA cm^-2)": current_density,
        "AD score": ad_score,
        "Within AD (score <= 1)": ad_score <= 1.0,
        "Measured voltage (mV)": voltage["Measured"].to_numpy(float),
        "Predicted voltage (mV)": voltage["OOF predicted"].to_numpy(float),
        "Absolute voltage error (mV)": np.abs(voltage["Residual"].to_numpy(float)),
        "Measured efficiency (%)": efficiency["Measured"].to_numpy(float),
        "Predicted efficiency (%)": efficiency["OOF predicted"].to_numpy(float),
        "Absolute efficiency error (%)": np.abs(efficiency["Residual"].to_numpy(float)),
    })
    coverage = coverage_table(oof)

    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10.5,
        "axes.linewidth": 1.4,
    })
    blue, orange, navy, gray = "#69AED3", "#F3A261", "#1F5A85", "#6B7280"
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 9.2), constrained_layout=True)

    ax = axes[0, 0]
    bins = np.linspace(0, max(1.5, float(ad_score.max()) * 1.02), 25)
    ax.hist(ad_score, bins=bins, color=blue, edgecolor="white", linewidth=0.8)
    ax.axvline(1.0, color="#C94747", linestyle="--", linewidth=2.0)
    ax.text(0.98, 0.94, f"Within AD: {(ad_score <= 1).mean() * 100:.1f}%", transform=ax.transAxes,
            ha="right", va="top", fontweight="bold")
    ax.set_xlabel("Applicability-domain score")
    ax.set_ylabel("Count")
    ax.text(-0.14, 1.04, "(a)", transform=ax.transAxes, fontsize=16, fontweight="bold")

    def error_panel(ax, error_col: str, ylabel: str, color: str, tag: str) -> None:
        within = source["Within AD (score <= 1)"].to_numpy(bool)
        ax.scatter(ad_score[within], source.loc[within, error_col], s=18, alpha=0.28, color=color,
                   edgecolors="none", label="Within AD")
        ax.scatter(ad_score[~within], source.loc[~within, error_col], s=25, alpha=0.55, color="#C94747",
                   edgecolors="none", label="Outside AD")
        q = pd.qcut(ad_score, q=10, duplicates="drop")
        grouped = source.assign(_bin=q).groupby("_bin", observed=True).agg(
            x=("AD score", "median"), y=(error_col, "median")
        )
        ax.plot(grouped["x"], grouped["y"], color=navy, marker="o", linewidth=2.2,
                markersize=5, label="Binned median")
        rho, p = spearmanr(ad_score, source[error_col])
        ax.text(0.04, 0.94, f"Spearman $\\rho$ = {rho:.2f}\n$p$ = {p:.2g}", transform=ax.transAxes,
                ha="left", va="top", fontweight="bold")
        ax.axvline(1.0, color="#C94747", linestyle="--", linewidth=1.5)
        ax.set_xlabel("Applicability-domain score")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, loc="upper right")
        ax.text(-0.14, 1.04, tag, transform=ax.transAxes, fontsize=16, fontweight="bold")

    error_panel(axes[0, 1], "Absolute voltage error (mV)", "Absolute voltage error (mV)", blue, "(b)")
    error_panel(axes[1, 0], "Absolute efficiency error (%)", "Absolute efficiency error (%)", orange, "(c)")

    ax = axes[1, 1]
    ax.plot([0.45, 1.0], [0.45, 1.0], color=gray, linestyle="--", linewidth=1.8, label="Ideal")
    for target, color, marker in [("Voltage", blue, "o"), ("Efficiency", orange, "s")]:
        d = coverage[coverage["Target"] == target]
        ax.plot(d["Nominal coverage"], d["Empirical coverage"], color=color, marker=marker,
                linewidth=2.3, markersize=6, label=target)
    ax.set_xlim(0.47, 0.98)
    ax.set_ylim(0.47, 0.98)
    ax.set_xlabel("Nominal residual-interval coverage")
    ax.set_ylabel("Empirical OOF coverage")
    ax.legend(frameon=False, loc="lower right")
    ax.text(-0.14, 1.04, "(d)", transform=ax.transAxes, fontsize=16, fontweight="bold")

    for ax in axes.flat:
        ax.grid(True, color="#D5DADF", linewidth=0.7, alpha=0.65)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_linewidth(1.4)

    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(OUTPUT_DIR / f"FigureS3_applicability_uncertainty.{suffix}", dpi=600 if suffix == "png" else None,
                    bbox_inches="tight")
    plt.close(fig)
    return source, coverage


def write_caption(source: pd.DataFrame, coverage: pd.DataFrame) -> None:
    coverage90 = coverage[np.isclose(coverage["Nominal coverage"], 0.90)].set_index("Target")
    text = f"""# Supplementary machine-learning materials

## Figure S3 caption

**Figure S3. Applicability-domain and uncertainty diagnostics for the BatteryBERT-Fusion model.**
(a) Distribution of the cross-validated applicability-domain (AD) score for the 1040 experimental records. For each outer test fold, structured descriptors were standardized using the corresponding training fold, and the mean distance to the five nearest training samples was divided by the 95th percentile of the training-set leave-one-out five-neighbor distances. The dashed line at AD score = 1 denotes the operational domain threshold. (b, c) Relationships between AD score and absolute OOF prediction error for average discharge voltage and anodic efficiency, respectively. Points outside the operational AD are highlighted in red, and the solid line connects decile-wise median errors. (d) Empirical coverage of cross-fold residual intervals relative to their nominal coverage. At a nominal coverage of 90%, the empirical coverages are {coverage90.loc['Voltage', 'Empirical coverage']:.3f} for voltage and {coverage90.loc['Efficiency', 'Empirical coverage']:.3f} for anodic efficiency.

## Table captions

**Table S1.** Input modalities, representations, and learning configurations of the eight regression models evaluated using the common five-fold OOF split.

**Table S2.** Hyperparameter search spaces and fold-specific optimal parameters for the BERT- and BatteryBERT-based late-fusion stacking models.

**Table S3.** Pooled and fold-wise OOF performance of the eight models for voltage and anodic-efficiency prediction. Confidence intervals were obtained using 2000 paired bootstrap resamples. Negative delta RMSE values indicate improvement relative to the numeric GB reference.

## Interpretation note

The AD score is a distance-based diagnostic rather than a formal guarantee of model validity. The residual intervals quantify empirical OOF error calibration within the experimental dataset and should not be interpreted as uncertainty bounds for virtual candidates outside the represented descriptor domain.
"""
    (OUTPUT_DIR / "Supplementary_ML_captions.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    oof = pd.read_csv(OOF_PATH)
    raw = pd.read_excel(RAWDATA_PATH, sheet_name="Sheet1")
    if len(raw) != oof["Sample"].nunique():
        raise ValueError("Raw-data and OOF sample counts do not match.")

    s1 = table_s1()
    s2_search = table_s2_search_space()
    s2_selected = table_s2_selected()
    s3_summary, s3_folds = table_s3(oof)
    fig_source, coverage = figure_s3(oof, raw)

    save_csv(s1, "TableS1_model_input_definitions.csv")
    save_csv(s2_search, "TableS2_hyperparameter_search_space.csv")
    save_csv(s2_selected, "TableS2_fold_selected_parameters.csv")
    save_csv(s3_summary, "TableS3_OOF_performance_summary.csv")
    save_csv(s3_folds, "TableS3_fold_metrics.csv")
    save_csv(fig_source, "FigureS3_sample_source_data.csv")
    save_csv(coverage, "FigureS3_interval_coverage.csv")
    write_caption(fig_source, coverage)
    print(f"Supplementary materials written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
