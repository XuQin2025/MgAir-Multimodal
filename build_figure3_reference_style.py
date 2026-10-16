# -*- coding: utf-8 -*-
"""Build the six-panel Figure 3 and the eight-model OOF benchmark table."""

from __future__ import annotations

import argparse
import gc
import hashlib
import math
import os
import warnings
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.transforms import ScaledTranslation
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import gaussian_kde
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from transformers import AutoModel, AutoTokenizer


ROOT = Path(__file__).resolve().parent
RAWDATA_PATH = ROOT / "rawdata.xlsx"
FIGURE_DIR = ROOT / "paper_figures"
OUTPUT_DIR = ROOT / "outputs_figure3_8models"
EMBED_CACHE_DIR = OUTPUT_DIR / "text_embeddings_with_values"

SEED = 42
N_SPLITS = 5
MAX_LENGTH = 160
EMBED_BATCH_SIZE = 24

MODEL_BERT = "bert-base-uncased"
MODEL_BATTERYBERT = "batterydata/batterybert-uncased"

ELEMENT_COLS = [
    "Mg", "Al", "Ca", "Mn", "Zn", "Sn", "La", "Ce", "Er", "In", "Y",
    "Ga", "Gd", "Sm", "Ge", "Nd", "Li", "Sr", "Zr", "Fe", "Cu", "Ni",
    "Pb", "Bi", "Ag", "Si", "Co", "Ti", "Cd",
]

PROCESS_COLS = {
    "homogenization_temperature": "均匀化温度（℃）",
    "homogenization_time": "均匀化时间",
    "extrusion_temperature": "挤压温度",
    "extrusion_speed": "挤压速度（mm/s）",
    "extrusion_ratio": "挤压比",
    "rolling_temperature": "轧制温度",
    "rolling_reduction": "总轧制压下量%",
    "annealing_temperature": "退火温度",
    "annealing_time": "退火时间",
}

TARGETS = {
    "Voltage": "Average discharge voltage/mV",
    "Efficiency": "Anodic efficiency/%",
}

MODEL_ORDER = [
    "GB", "MLP", "DT", "ANN",
    "BERT-Text", "BatteryBERT-Text",
    "BERT-Fusion", "BatteryBERT-Fusion",
]

MODEL_LABELS = [
    "GB\n(num.)", "MLP\n(num.)", "DT\n(num.)", "ANN\n(num.)",
    "BERT\n(text)", "BatteryBERT\n(text)",
    "BERT\n(fusion)", "BatteryBERT\n(fusion)",
]

GROUP_COLORS = {
    "numeric": "#67B7DC",
    "text": "#F2B56B",
    "fusion": "#E87B78",
}

BLUE = "#56A9D8"
BLUE_DARK = "#2678A6"
CORAL = "#EE7D78"
CORAL_DARK = "#C95B57"
NAVY = "#263746"
GRID = "#C9D1D8"
LAVENDER = "#D9C5EC"


def configure_matplotlib() -> None:
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.sans-serif": ["Arial"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "axes.unicode_minus": True,
        "axes.linewidth": 1.6,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "xtick.labelsize": 13.5,
        "ytick.labelsize": 13.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 1.4,
        "ytick.major.width": 1.4,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(col).strip() for col in out.columns]
    return out


def find_column(df: pd.DataFrame, preferred: str) -> str:
    preferred_clean = preferred.strip().lower()
    for col in df.columns:
        if str(col).strip().lower() == preferred_clean:
            return col
    for col in df.columns:
        if preferred_clean in str(col).strip().lower():
            return col
    raise KeyError(f"Column not found: {preferred}")


def compact_number(value: float) -> str:
    if not np.isfinite(value):
        return "0"
    if abs(value - round(value)) < 1e-8:
        return str(int(round(value)))
    return f"{value:.3f}".rstrip("0").rstrip(".")


def positive_value(row: pd.Series, column: str) -> float:
    value = pd.to_numeric(pd.Series([row.get(column, 0)]), errors="coerce").iloc[0]
    return float(value) if np.isfinite(value) and float(value) > 0 else 0.0


def build_numeric_text(row: pd.Series, current_col: str) -> str:
    alloy_parts = []
    for element in ELEMENT_COLS:
        if element == "Mg" or element not in row.index:
            continue
        value = positive_value(row, element)
        if value > 0:
            alloy_parts.append(f"{compact_number(value)} wt% {element}")

    if alloy_parts:
        sentences = ["Mg-based alloy with " + ", ".join(alloy_parts) + "."]
    else:
        sentences = ["Mg-based alloy."]

    ht = positive_value(row, PROCESS_COLS["homogenization_temperature"])
    hh = positive_value(row, PROCESS_COLS["homogenization_time"])
    if ht or hh:
        parts = []
        if ht:
            parts.append(f"at {compact_number(ht)} degrees Celsius")
        if hh:
            parts.append(f"for {compact_number(hh)} hours")
        sentences.append("Homogenization " + " ".join(parts) + ".")

    et = positive_value(row, PROCESS_COLS["extrusion_temperature"])
    es = positive_value(row, PROCESS_COLS["extrusion_speed"])
    er = positive_value(row, PROCESS_COLS["extrusion_ratio"])
    if et or es or er:
        parts = []
        if et:
            parts.append(f"at {compact_number(et)} degrees Celsius")
        if es:
            parts.append(f"at {compact_number(es)} millimeters per second")
        if er:
            parts.append(f"with an extrusion ratio of {compact_number(er)}")
        sentences.append("Extrusion " + ", ".join(parts) + ".")

    rt = positive_value(row, PROCESS_COLS["rolling_temperature"])
    rr = positive_value(row, PROCESS_COLS["rolling_reduction"])
    if rt or rr:
        parts = []
        if rt:
            parts.append(f"at {compact_number(rt)} degrees Celsius")
        if rr:
            parts.append(f"with {compact_number(rr)} percent total reduction")
        sentences.append("Rolling " + ", ".join(parts) + ".")

    at = positive_value(row, PROCESS_COLS["annealing_temperature"])
    ah = positive_value(row, PROCESS_COLS["annealing_time"])
    if at or ah:
        parts = []
        if at:
            parts.append(f"at {compact_number(at)} degrees Celsius")
        if ah:
            parts.append(f"for {compact_number(ah)} hours")
        sentences.append("Annealing " + " ".join(parts) + ".")

    current_density = positive_value(row, current_col)
    sentences.append(
        "Tested at a current density of "
        f"{compact_number(current_density)} milliamperes per square centimeter."
    )
    return " ".join(sentences)


def load_data() -> Tuple[pd.DataFrame, List[str], str, Dict[str, str]]:
    df = clean_column_names(pd.read_excel(RAWDATA_PATH, sheet_name="Sheet1"))
    current_col = find_column(df, "current density/mA/cm2")
    target_cols = {name: find_column(df, col) for name, col in TARGETS.items()}

    numeric_cols = [col for col in ELEMENT_COLS if col in df.columns]
    numeric_cols.extend(col for col in PROCESS_COLS.values() if col in df.columns)
    numeric_cols.append(current_col)

    required = numeric_cols + list(target_cols.values())
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=required).reset_index(drop=True)
    df["text_with_values"] = [build_numeric_text(row, current_col) for _, row in df.iterrows()]
    return df, numeric_cols, current_col, target_cols


def embedding_cache_path(model_name: str, texts: Iterable[str]) -> Path:
    digest = hashlib.sha256()
    digest.update(model_name.encode("utf-8"))
    digest.update(f"|L{MAX_LENGTH}|".encode("ascii"))
    for text in texts:
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")
    safe_name = model_name.replace("/", "--")
    return EMBED_CACHE_DIR / f"{safe_name}_{digest.hexdigest()[:16]}.npy"


def mean_pool(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(hidden_state)
    return (hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def embed_texts(model_name: str, texts: List[str]) -> np.ndarray:
    EMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = embedding_cache_path(model_name, texts)
    if cache_path.exists():
        print(f"Using cached embeddings: {cache_path.name}", flush=True)
        return np.load(cache_path)

    print(f"Encoding {len(texts)} value-aware texts with {model_name}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModel.from_pretrained(model_name, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    batches = []
    total_batches = math.ceil(len(texts) / EMBED_BATCH_SIZE)
    with torch.inference_mode():
        for batch_no, start in enumerate(range(0, len(texts), EMBED_BATCH_SIZE), start=1):
            batch = texts[start:start + EMBED_BATCH_SIZE]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            output = model(**encoded)
            pooled = mean_pool(output.last_hidden_state, encoded["attention_mask"])
            batches.append(pooled.cpu().numpy().astype(np.float32))
            if batch_no == 1 or batch_no % 10 == 0 or batch_no == total_batches:
                print(f"  embedding batch {batch_no}/{total_batches}", flush=True)

    embeddings = np.vstack(batches)
    np.save(cache_path, embeddings)
    del model, tokenizer, batches
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return embeddings


def outer_folds(n_samples: int) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], np.ndarray]:
    folds = list(KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED).split(np.arange(n_samples)))
    fold_ids = np.zeros(n_samples, dtype=int)
    for fold_no, (_, test_idx) in enumerate(folds, start=1):
        fold_ids[test_idx] = fold_no
    return folds, fold_ids


def numeric_model_factories() -> Dict[str, Callable[[], object]]:
    return {
        "GB": lambda: GradientBoostingRegressor(
            n_estimators=350,
            learning_rate=0.035,
            max_depth=3,
            min_samples_leaf=2,
            subsample=0.85,
            random_state=SEED,
        ),
        "MLP": lambda: Pipeline([
            ("scale", StandardScaler()),
            ("model", MLPRegressor(
                hidden_layer_sizes=(100, 50),
                activation="relu",
                alpha=1e-3,
                learning_rate_init=8e-4,
                max_iter=2200,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=60,
                random_state=SEED,
            )),
        ]),
        "DT": lambda: DecisionTreeRegressor(
            min_samples_leaf=2,
            random_state=SEED,
        ),
        # A deeper feed-forward network is retained as a distinct ANN baseline.
        "ANN": lambda: Pipeline([
            ("scale", StandardScaler()),
            ("model", MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                activation="relu",
                alpha=5e-4,
                learning_rate_init=5e-4,
                max_iter=2600,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=80,
                random_state=SEED + 101,
            )),
        ]),
    }


def fit_oof_model(factory: Callable[[], object], X: np.ndarray, y: np.ndarray, folds) -> np.ndarray:
    prediction = np.zeros(len(y), dtype=float)
    for fold_no, (train_idx, test_idx) in enumerate(folds, start=1):
        model = factory()
        model.fit(X[train_idx], y[train_idx])
        prediction[test_idx] = model.predict(X[test_idx])
        print(f"    fold {fold_no}/{N_SPLITS}", flush=True)
    return prediction


def fit_oof_text_ridge(X: np.ndarray, y: np.ndarray, folds) -> np.ndarray:
    prediction = np.zeros(len(y), dtype=float)
    alphas = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
    for fold_no, (train_idx, test_idx) in enumerate(folds, start=1):
        inner_cv = KFold(n_splits=5, shuffle=True, random_state=SEED + fold_no)
        estimator = Pipeline([
            ("scale", StandardScaler()),
            ("ridge", Ridge()),
        ])
        search = GridSearchCV(
            estimator,
            param_grid={"ridge__alpha": alphas},
            cv=inner_cv,
            scoring="r2",
            n_jobs=1,
        )
        search.fit(X[train_idx], y[train_idx])
        prediction[test_idx] = search.predict(X[test_idx])
        print(
            f"    fold {fold_no}/{N_SPLITS}, alpha={search.best_params_['ridge__alpha']}",
            flush=True,
        )
    return prediction


def load_multimodal_oof(target_tag: str, model_tag: str) -> np.ndarray:
    import build_paper_package as legacy

    bundle = legacy.get_oof_bundle(model_tag, target_tag)
    return np.asarray(bundle["y_pred"], dtype=float)


def pooled_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    return (
        float(r2_score(y_true, y_pred)),
        float(np.sqrt(mean_squared_error(y_true, y_pred))),
    )


def fold_metric_rows(
    target_tag: str,
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fold_ids: np.ndarray,
) -> List[Dict[str, float]]:
    rows = []
    for fold_no in range(1, N_SPLITS + 1):
        mask = fold_ids == fold_no
        r2, rmse = pooled_metrics(y_true[mask], y_pred[mask])
        rows.append({
            "Target": target_tag,
            "Model": model_name,
            "Fold": fold_no,
            "R2": r2,
            "RMSE": rmse,
        })
    return rows


def run_benchmarks(
    df: pd.DataFrame,
    numeric_cols: List[str],
    target_cols: Dict[str, str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[Tuple[str, str], np.ndarray]]:
    X_num = df[numeric_cols].astype(float).to_numpy()
    texts = df["text_with_values"].tolist()
    folds, fold_ids = outer_folds(len(df))

    emb_bert = embed_texts(MODEL_BERT, texts)
    emb_battery = embed_texts(MODEL_BATTERYBERT, texts)

    all_predictions: Dict[Tuple[str, str], np.ndarray] = {}
    fold_rows: List[Dict[str, float]] = []
    metric_rows: List[Dict[str, float]] = []
    numeric_factories = numeric_model_factories()

    for target_tag, target_col in target_cols.items():
        y = df[target_col].astype(float).to_numpy()
        print(f"\nTarget: {target_tag}", flush=True)

        for model_name, factory in numeric_factories.items():
            print(f"  Numeric OOF: {model_name}", flush=True)
            pred = fit_oof_model(factory, X_num, y, folds)
            all_predictions[(target_tag, model_name)] = pred

        for model_name, embeddings in [
            ("BERT-Text", emb_bert),
            ("BatteryBERT-Text", emb_battery),
        ]:
            print(f"  Text-only OOF: {model_name}", flush=True)
            pred = fit_oof_text_ridge(embeddings, y, folds)
            all_predictions[(target_tag, model_name)] = pred

        print("  Reconstructing OOF: BERT-Fusion", flush=True)
        all_predictions[(target_tag, "BERT-Fusion")] = load_multimodal_oof(
            target_tag,
            "BERT_GBstack_HPO",
        )
        print("  Reconstructing OOF: BatteryBERT-Fusion", flush=True)
        all_predictions[(target_tag, "BatteryBERT-Fusion")] = load_multimodal_oof(
            target_tag,
            "BatteryBERT_GBstack_HPO",
        )

        for model_name in MODEL_ORDER:
            pred = all_predictions[(target_tag, model_name)]
            r2, rmse = pooled_metrics(y, pred)
            current_fold_rows = fold_metric_rows(target_tag, model_name, y, pred, fold_ids)
            fold_rows.extend(current_fold_rows)
            metric_rows.append({
                "Target": target_tag,
                "Model": model_name,
                "Input group": (
                    "Numeric" if model_name in MODEL_ORDER[:4]
                    else "Text" if model_name in MODEL_ORDER[4:6]
                    else "Multimodal fusion"
                ),
                "R2": r2,
                "R2 fold SD": float(np.std([row["R2"] for row in current_fold_rows], ddof=1)),
                "RMSE": rmse,
                "RMSE fold SD": float(np.std([row["RMSE"] for row in current_fold_rows], ddof=1)),
            })

    metrics_df = pd.DataFrame(metric_rows)
    fold_metrics_df = pd.DataFrame(fold_rows)

    oof_rows = []
    for target_tag, target_col in target_cols.items():
        y = df[target_col].astype(float).to_numpy()
        for model_name in MODEL_ORDER:
            pred = all_predictions[(target_tag, model_name)]
            oof_rows.extend({
                "Target": target_tag,
                "Model": model_name,
                "Sample": idx + 1,
                "Fold": int(fold_ids[idx]),
                "Measured": float(y[idx]),
                "OOF predicted": float(pred[idx]),
                "Residual": float(pred[idx] - y[idx]),
            } for idx in range(len(y)))
    oof_df = pd.DataFrame(oof_rows)
    return metrics_df, fold_metrics_df, oof_df, all_predictions


def kde_count_curve(values: np.ndarray, bins: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    if len(np.unique(values)) < 2:
        return np.array([]), np.array([])
    grid = np.linspace(bins[0], bins[-1], 250)
    density = gaussian_kde(values)(grid)
    bin_width = float(np.mean(np.diff(bins)))
    return grid, density * len(values) * bin_width


def panel_label(ax: plt.Axes, label: str, x: float = -0.18, y: float = 1.08) -> None:
    ax.text(
        x,
        y,
        f"({label})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=24,
        fontweight="bold",
        color="black",
        clip_on=False,
    )


def style_full_frame(ax: plt.Axes, linewidth: float = 1.65) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(linewidth)
        spine.set_color("#252C32")


def bold_tick_labels(ax: plt.Axes) -> None:
    for tick_label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        tick_label.set_fontweight("bold")


def joint_parity_panel(
    fig: plt.Figure,
    spec,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_title: str,
    unit: str,
    label: str,
) -> None:
    inner = GridSpecFromSubplotSpec(
        2,
        2,
        subplot_spec=spec,
        width_ratios=[4.7, 1.05],
        height_ratios=[1.02, 4.7],
        wspace=0.055,
        hspace=0.055,
    )
    ax_top = fig.add_subplot(inner[0, 0])
    ax_main = fig.add_subplot(inner[1, 0])
    ax_right = fig.add_subplot(inner[1, 1], sharey=ax_main)
    fig.add_subplot(inner[0, 1]).axis("off")

    lower = min(float(np.min(y_true)), float(np.min(y_pred)))
    upper = max(float(np.max(y_true)), float(np.max(y_pred)))
    padding = 0.04 * (upper - lower)
    bounds = (lower - padding, upper + padding)
    bins = np.linspace(bounds[0], bounds[1], 18)

    ax_top.hist(y_true, bins=bins, color=BLUE, alpha=0.62, edgecolor=NAVY, linewidth=0.55, label="Measured")
    ax_top.hist(y_pred, bins=bins, color=CORAL, alpha=0.55, edgecolor=CORAL_DARK, linewidth=0.55, label="Predicted")
    for values, color in [(y_true, BLUE_DARK), (y_pred, CORAL_DARK)]:
        grid, count = kde_count_curve(values, bins)
        ax_top.plot(grid, count, color=color, linewidth=1.5)
    ax_top.set_xlim(bounds)
    ax_top.set_ylabel("Count", fontsize=14.5, fontweight="bold")
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.tick_params(labelsize=12.5)
    ax_top.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    style_full_frame(ax_top)
    bold_tick_labels(ax_top)
    ax_top.legend(
        loc="upper left",
        frameon=False,
        prop={"family": "Arial", "size": 15.5, "weight": "bold"},
        ncol=1,
        handlelength=1.2,
        labelspacing=0.28,
        borderaxespad=0.15,
    )

    ax_main.scatter(
        y_true,
        y_pred,
        s=34,
        c=BLUE,
        alpha=0.76,
        edgecolors="white",
        linewidths=0.45,
        rasterized=True,
    )
    line = np.linspace(bounds[0], bounds[1], 250)
    ax_main.plot(line, line, linestyle="--", color=NAVY, linewidth=1.35, label="Ideal")
    coefficient = np.polyfit(y_true, y_pred, 1)
    fitted = coefficient[0] * line + coefficient[1]
    residuals = y_pred - (coefficient[0] * y_true + coefficient[1])
    band = 0.30 * float(np.std(residuals))
    ax_main.plot(line, fitted, color=CORAL, linewidth=2.0, label="Linear fit")
    ax_main.fill_between(line, fitted - band, fitted + band, color=CORAL, alpha=0.14, linewidth=0)
    ax_main.set_xlim(bounds)
    ax_main.set_ylim(bounds)
    ax_main.set_xlabel(f"Measured {target_title.lower()} ({unit})", fontsize=16, fontweight="bold")
    ax_main.set_ylabel(f"Predicted {target_title.lower()} ({unit})", fontsize=16, fontweight="bold")
    ax_main.tick_params(labelsize=13.5, pad=5)
    ax_main.xaxis.set_major_locator(MaxNLocator(nbins=6, prune="upper"))
    ax_main.yaxis.set_major_locator(MaxNLocator(nbins=6, prune="upper"))
    ax_main.grid(True, linestyle=":", color=GRID, linewidth=0.8, alpha=0.9)
    ax_main.set_axisbelow(True)
    style_full_frame(ax_main, linewidth=1.8)
    bold_tick_labels(ax_main)

    r2, rmse = pooled_metrics(y_true, y_pred)
    ax_main.text(
        0.045,
        0.955,
        f"$R^2$: {r2:.3f}\nRMSE: {rmse:.2f} {unit}",
        transform=ax_main.transAxes,
        ha="left",
        va="top",
        fontsize=16,
        fontweight="bold",
        color=BLUE_DARK,
    )
    inset = inset_axes(
        ax_main,
        width="44%",
        height="38%",
        loc="lower right",
        bbox_to_anchor=(0, 0.06, 1, 1),
        bbox_transform=ax_main.transAxes,
        borderpad=1.15,
    )
    raw_error = y_pred - y_true
    error_bins = np.linspace(float(np.min(raw_error)), float(np.max(raw_error)), 14)
    inset.hist(raw_error, bins=error_bins, color="#F4B3AF", alpha=0.72, edgecolor=CORAL_DARK, linewidth=0.5)
    error_grid, error_count = kde_count_curve(raw_error, error_bins)
    inset.plot(error_grid, error_count, color=CORAL_DARK, linewidth=1.1)
    inset.axvline(0, color=NAVY, linestyle=":", linewidth=0.9)
    inset.set_xlabel("Error", fontsize=12.5, labelpad=0)
    inset.set_ylabel("Count", fontsize=12.5, labelpad=0)
    inset.tick_params(labelsize=11.5, length=2)
    style_full_frame(inset, linewidth=1.35)
    bold_tick_labels(inset)

    ax_right.hist(y_true, bins=bins, orientation="horizontal", color=BLUE, alpha=0.62, edgecolor=NAVY, linewidth=0.55)
    ax_right.hist(y_pred, bins=bins, orientation="horizontal", color=CORAL, alpha=0.55, edgecolor=CORAL_DARK, linewidth=0.55)
    for values, color in [(y_true, BLUE_DARK), (y_pred, CORAL_DARK)]:
        grid, count = kde_count_curve(values, bins)
        ax_right.plot(count, grid, color=color, linewidth=1.5)
    ax_right.set_xlabel("Count", fontsize=14.5, fontweight="bold")
    ax_right.tick_params(axis="y", labelleft=False)
    ax_right.tick_params(labelsize=12.5, pad=5)
    ax_right.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    style_full_frame(ax_right)
    bold_tick_labels(ax_right)
    ax_right.set_ylim(bounds)
    if label == "c":
        for tick_value, tick_label in zip(ax_right.get_xticks(), ax_right.get_xticklabels()):
            if np.isclose(tick_value, 0.0):
                tick_label.set_horizontalalignment("left")
                tick_label.set_transform(
                    tick_label.get_transform()
                    + ScaledTranslation(9.0 / 72.0, 0.0, fig.dpi_scale_trans)
                )

    panel_label(ax_top, label, x=-0.20, y=1.13)


def signed_symmetric_relative_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    denominator = np.abs(y_true) + np.abs(y_pred)
    return 200.0 * (y_pred - y_true) / np.maximum(denominator, 1e-12)


def mre_panel(
    ax: plt.Axes,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fold_ids: np.ndarray,
    label: str,
    point_color: str,
) -> None:
    folds = [np.flatnonzero(fold_ids == fold_no) for fold_no in range(1, N_SPLITS + 1)]
    order = np.concatenate(folds)
    mre = signed_symmetric_relative_error(y_true, y_pred)[order]
    x = np.arange(1, len(mre) + 1)

    q995 = float(np.quantile(np.abs(mre), 0.995))
    y_limit = max(20.0, min(200.0, math.ceil(q995 / 10.0) * 10.0))
    clipped = np.clip(mre, -y_limit, y_limit)

    boundaries = np.cumsum([len(indices) for indices in folds])
    starts = np.r_[0, boundaries[:-1]]
    for fold_no, (start, end) in enumerate(zip(starts, boundaries), start=1):
        if fold_no % 2 == 1:
            ax.axvspan(start + 0.5, end + 0.5, color=point_color, alpha=0.045, linewidth=0)
        ax.text(
            (start + end + 1) / 2,
            0.965,
            f"Fold {fold_no}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=16,
            color="#68727A",
        )
        if fold_no < N_SPLITS:
            ax.axvline(end + 0.5, color="#B8C0C7", linewidth=0.75, linestyle=":")

    ax.vlines(x, 0, clipped, color=point_color, alpha=0.32, linewidth=0.55, rasterized=True)
    ax.scatter(x, clipped, s=9, color=point_color, alpha=0.76, edgecolor="none", rasterized=True)
    outlier_mask = np.abs(mre) > y_limit
    if np.any(outlier_mask):
        marker = np.where(mre[outlier_mask] > 0, y_limit, -y_limit)
        ax.scatter(x[outlier_mask], marker, s=17, marker="^", color=CORAL_DARK, zorder=3)

    mean_mre = float(np.mean(mre))
    ax.axhline(0, color=NAVY, linewidth=1.0)
    ax.axhline(mean_mre, color=CORAL_DARK, linewidth=1.25, linestyle="--")
    ax.text(
        0.012,
        0.07 if mean_mre >= 0 else 0.93,
        f"Mean = {mean_mre:.2f}%",
        transform=ax.transAxes,
        ha="left",
        va="bottom" if mean_mre >= 0 else "top",
        fontsize=16,
        color=CORAL_DARK,
        fontweight="bold",
    )
    ax.set_xlim(0, len(mre) + 1)
    ax.set_ylim(-y_limit, y_limit)
    ax.set_xlabel("Number of sample", fontsize=16, fontweight="bold")
    ax.set_ylabel("MRE (%)", fontsize=16, fontweight="bold")
    ax.tick_params(labelsize=13.5)
    ax.grid(True, axis="y", linestyle=":", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    style_full_frame(ax, linewidth=1.8)
    bold_tick_labels(ax)
    panel_label(ax, label, x=-0.13, y=1.10)


def model_colors() -> List[str]:
    return (
        [GROUP_COLORS["numeric"]] * 4
        + [GROUP_COLORS["text"]] * 2
        + ["#F09B98", "#D95854"]
    )


def model_comparison_panel(
    fig: plt.Figure,
    spec,
    metrics_df: pd.DataFrame,
    target_tag: str,
    rmse_unit: str,
    label: str,
) -> None:
    inner = GridSpecFromSubplotSpec(
        2,
        1,
        subplot_spec=spec,
        height_ratios=[1.05, 0.95],
        hspace=0.035,
    )
    ax_r2 = fig.add_subplot(inner[0, 0])
    ax_rmse = fig.add_subplot(inner[1, 0], sharex=ax_r2)

    target_df = metrics_df[metrics_df["Target"] == target_tag].set_index("Model").loc[MODEL_ORDER]
    x = np.arange(len(MODEL_ORDER))
    colors = model_colors()

    for ax in (ax_r2, ax_rmse):
        ax.axvspan(-0.48, 3.48, color=GROUP_COLORS["numeric"], alpha=0.055, linewidth=0)
        ax.axvspan(3.52, 5.48, color=GROUP_COLORS["text"], alpha=0.070, linewidth=0)
        ax.axvspan(5.52, 7.48, color=GROUP_COLORS["fusion"], alpha=0.060, linewidth=0)
        ax.axvspan(6.56, 7.44, color=LAVENDER, alpha=0.36, linewidth=0, zorder=0)
        ax.axvline(3.5, color="#AEB7BE", linewidth=0.8, linestyle=":")
        ax.axvline(5.5, color="#AEB7BE", linewidth=0.8, linestyle=":")
        ax.grid(True, axis="y", color=GRID, linestyle="--", linewidth=0.75, alpha=0.9)
        ax.set_axisbelow(True)

    r2_values = target_df["R2"].to_numpy(dtype=float)
    r2_error = target_df["R2 fold SD"].to_numpy(dtype=float)
    rmse_values = target_df["RMSE"].to_numpy(dtype=float)
    rmse_error = target_df["RMSE fold SD"].to_numpy(dtype=float)

    ax_r2.bar(
        x,
        r2_values,
        width=0.68,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
        yerr=r2_error,
        capsize=2.3,
        error_kw={"elinewidth": 0.8, "ecolor": "#65717A"},
        zorder=2,
    )
    ax_rmse.bar(
        x,
        -rmse_values,
        width=0.68,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
        yerr=rmse_error,
        capsize=2.3,
        error_kw={"elinewidth": 0.8, "ecolor": "#65717A"},
        zorder=2,
    )

    r2_low = min(-0.08, float(np.min(r2_values - r2_error)) - 0.07)
    ax_r2.set_ylim(r2_low, 1.05)
    ax_rmse.set_ylim(-float(np.max(rmse_values + rmse_error)) * 1.25, 0)
    ax_r2.axhline(0, color=NAVY, linewidth=0.9)
    ax_rmse.axhline(0, color=NAVY, linewidth=0.9)

    for idx, value in enumerate(r2_values):
        ax_r2.text(
            idx,
            value + max(0.025, r2_error[idx] + 0.012),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=16.5,
            color=NAVY,
            fontweight="bold",
        )
    for idx, value in enumerate(rmse_values):
        ax_rmse.text(
            idx,
            -value - max(0.025 * float(np.max(rmse_values)), rmse_error[idx] + 0.01),
            f"{value:.2f}",
            ha="center",
            va="top",
            fontsize=16.5,
            color=NAVY,
            fontweight="bold",
        )

    ax_r2.set_ylabel("$R^2$", fontsize=18, fontweight="bold")
    ax_rmse.set_ylabel(f"RMSE ({rmse_unit})", fontsize=18, fontweight="bold")
    ax_rmse.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{abs(value):g}"))
    ax_rmse.set_xticks(x)
    ax_rmse.set_xticklabels(MODEL_LABELS, fontsize=12.0, fontweight="bold")
    ax_rmse.tick_params(axis="x", pad=7)
    ax_r2.tick_params(axis="x", labelbottom=False)
    ax_r2.tick_params(axis="y", labelsize=15.0, width=1.4)
    ax_rmse.tick_params(axis="y", labelsize=15.0, width=1.4)
    ax_rmse.set_xlim(-0.55, 7.55)

    header_groups = [
        (-0.5, 4.0, 1.5, "Numeric input", GROUP_COLORS["numeric"]),
        (3.5, 2.0, 4.5, "Text input", GROUP_COLORS["text"]),
        (5.5, 2.0, 6.5, "Multimodal\nfusion", "#EA8D8B"),
    ]
    for start, width, center, group_name, group_color in header_groups:
        ax_r2.add_patch(
            mpl.patches.Rectangle(
                (start, 1.015),
                width,
                0.205,
                transform=ax_r2.get_xaxis_transform(),
                facecolor=group_color,
                edgecolor="#69737B",
                linewidth=1.25,
                clip_on=False,
                zorder=4,
            )
        )
        ax_r2.text(
            center,
            1.117,
            group_name,
            transform=ax_r2.get_xaxis_transform(),
            ha="center",
            va="center",
            fontsize=16.0,
            fontweight="bold",
            color=NAVY,
            clip_on=False,
            linespacing=0.95,
            zorder=5,
        )

    for axis in (ax_r2, ax_rmse):
        style_full_frame(axis, linewidth=1.8)
        bold_tick_labels(axis)
    panel_label(ax_r2, label, x=-0.13, y=1.25)


def build_figure(
    df: pd.DataFrame,
    target_cols: Dict[str, str],
    metrics_df: pd.DataFrame,
    predictions: Dict[Tuple[str, str], np.ndarray],
) -> None:
    _, fold_ids = outer_folds(len(df))
    y_voltage = df[target_cols["Voltage"]].astype(float).to_numpy()
    y_efficiency = df[target_cols["Efficiency"]].astype(float).to_numpy()
    p_voltage = predictions[("Voltage", "BatteryBERT-Fusion")]
    p_efficiency = predictions[("Efficiency", "BatteryBERT-Fusion")]

    fig = plt.figure(figsize=(18.2, 16.5), constrained_layout=False)
    outer = GridSpec(
        3,
        2,
        figure=fig,
        height_ratios=[1.04, 1.26, 0.70],
        hspace=0.30,
        wspace=0.27,
        left=0.072,
        right=0.982,
        bottom=0.055,
        top=0.985,
    )

    model_comparison_panel(fig, outer[0, 0], metrics_df, "Voltage", "mV", "a")
    model_comparison_panel(fig, outer[0, 1], metrics_df, "Efficiency", "%", "b")

    joint_parity_panel(
        fig,
        outer[1, 0],
        y_voltage,
        p_voltage,
        "Average discharge voltage",
        "mV",
        "c",
    )
    joint_parity_panel(
        fig,
        outer[1, 1],
        y_efficiency,
        p_efficiency,
        "Anodic efficiency",
        "%",
        "d",
    )

    ax_e = fig.add_subplot(outer[2, 0])
    ax_f = fig.add_subplot(outer[2, 1])
    mre_panel(
        ax_e,
        y_voltage,
        p_voltage,
        fold_ids,
        "e",
        BLUE,
    )
    mre_panel(
        ax_f,
        y_efficiency,
        p_efficiency,
        fold_ids,
        "f",
        CORAL,
    )

    output_stems = [
        FIGURE_DIR / "Figure3_model_performance_reference_style",
        FIGURE_DIR / "Figure3_model_performance_composite",
    ]
    for output_stem in output_stems:
        fig.savefig(output_stem.with_suffix(".png"), dpi=450, bbox_inches="tight")
        fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def export_source_data(
    df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    fold_metrics_df: pd.DataFrame,
    oof_df: pd.DataFrame,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    metrics_df.to_csv(OUTPUT_DIR / "Figure3_model_metrics_8models.csv", index=False, encoding="utf-8-sig")
    fold_metrics_df.to_csv(OUTPUT_DIR / "Figure3_fold_metrics_8models.csv", index=False, encoding="utf-8-sig")
    oof_df.to_csv(OUTPUT_DIR / "Figure3_oof_predictions_8models.csv", index=False, encoding="utf-8-sig")

    examples = df[["text_with_values"]].head(12).copy()
    examples.insert(0, "Sample", np.arange(1, len(examples) + 1))
    workbook_path = FIGURE_DIR / "Figure3_model_metrics_8models.xlsx"
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        metrics_df.to_excel(writer, sheet_name="Pooled OOF metrics", index=False)
        fold_metrics_df.to_excel(writer, sheet_name="Fold metrics", index=False)
        examples.to_excel(writer, sheet_name="Text input examples", index=False)


def load_cached_benchmarks() -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    Dict[Tuple[str, str], np.ndarray],
]:
    metrics_df = pd.read_csv(OUTPUT_DIR / "Figure3_model_metrics_8models.csv")
    fold_metrics_df = pd.read_csv(OUTPUT_DIR / "Figure3_fold_metrics_8models.csv")
    oof_df = pd.read_csv(OUTPUT_DIR / "Figure3_oof_predictions_8models.csv")
    predictions = {}
    for target_tag in TARGETS:
        for model_name in MODEL_ORDER:
            subset = oof_df[
                (oof_df["Target"] == target_tag)
                & (oof_df["Model"] == model_name)
            ].sort_values("Sample")
            predictions[(target_tag, model_name)] = subset["OOF predicted"].to_numpy(dtype=float)
    return metrics_df, fold_metrics_df, oof_df, predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Reuse cached OOF predictions and only rebuild the figure.",
    )
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    configure_matplotlib()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cpu_count = os.cpu_count() or 1
    torch.set_num_threads(max(1, min(12, cpu_count)))

    print("Loading dataset and constructing value-aware text...", flush=True)
    df, numeric_cols, _, target_cols = load_data()
    print(f"Samples: {len(df)} | Numeric descriptors: {len(numeric_cols)}", flush=True)
    print("Example text:", df.loc[0, "text_with_values"], flush=True)

    if args.plot_only:
        print("Loading cached OOF predictions for layout-only rendering...", flush=True)
        metrics_df, fold_metrics_df, oof_df, predictions = load_cached_benchmarks()
    else:
        metrics_df, fold_metrics_df, oof_df, predictions = run_benchmarks(
            df,
            numeric_cols,
            target_cols,
        )
        print("\nPooled five-fold OOF metrics:", flush=True)
        print(metrics_df[["Target", "Model", "R2", "RMSE"]].to_string(index=False), flush=True)
        export_source_data(df, metrics_df, fold_metrics_df, oof_df)

    build_figure(df, target_cols, metrics_df, predictions)
    print("\nFigure and source tables completed.", flush=True)


if __name__ == "__main__":
    main()
