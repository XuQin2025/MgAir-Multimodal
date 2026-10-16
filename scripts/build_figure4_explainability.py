# -*- coding: utf-8 -*-
"""Build Figure 4 with global and current-density-conditioned SHAP explanations."""

from __future__ import annotations

import gc
import hashlib
import math
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Rectangle
from transformers import AutoModel, AutoTokenizer


ROOT = Path(__file__).resolve().parent.parent
RAWDATA_PATH = ROOT / "rawdata.xlsx"
MODEL_DIR = ROOT / "outputs_design_search_multiCD_selected"
FIGURE_DIR = ROOT / "paper_figures"
OUTPUT_DIR = ROOT / "outputs_figure4_explainability"
CACHE_DIR = OUTPUT_DIR / "cache"

MODEL_NAME = "batterydata/batterybert-uncased"
MAX_LENGTH = 80
BATCH_SIZE = 32
SEED = 42

ELEMENTS = [
    "Mg", "Al", "Ca", "Mn", "Zn", "Sn", "La", "Ce", "Er", "In", "Y",
    "Ga", "Gd", "Sm", "Ge", "Nd", "Li", "Sr", "Zr", "Fe", "Cu", "Ni",
    "Pb", "Bi", "Ag", "Si", "Co", "Ti", "Cd",
]

PROCESS_LABELS = {
    "均匀化温度（℃）": "Homogenization temperature",
    "均匀化时间": "Homogenization time",
    "挤压温度": "Extrusion temperature",
    "挤压速度（mm/s）": "Extrusion speed",
    "挤压比": "Extrusion ratio",
}

CURRENT_LABEL = "Current density"
TEXT_FEATURE = "Text semantics"

BLUE = "#2C7FB8"
LIGHT_BLUE = "#A6CEE3"
RED = "#D34A43"
LIGHT_RED = "#F6B0A8"
NAVY = "#263746"
GRID = "#CDD4DA"

PDP_SPEC = {
    "current": 40.0,
    "x_feature": "Er",
    "y_feature": "Gd",
    "x_range": (0.0, 3.2),
    "y_range": (0.0, 3.2),
    "sum_limit": 11.97,
    "fixed_elements": {"Al": 8.0},
    "candidate": (0.8, 2.4),
    "system": "Mg-Al-Er-Gd",
}


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.sans-serif": ["Arial"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "axes.linewidth": 1.35,
        "axes.labelsize": 19,
        "axes.titlesize": 18,
        "xtick.labelsize": 17,
        "ytick.labelsize": 17,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 1.25,
        "ytick.major.width": 1.25,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(col).strip() for col in out.columns]
    return out


def find_column(df: pd.DataFrame, fragment: str) -> str:
    fragment = fragment.strip().lower()
    for col in df.columns:
        if str(col).strip().lower() == fragment:
            return col
    for col in df.columns:
        if fragment in str(col).strip().lower():
            return col
    raise KeyError(fragment)


def pretty_feature(feature: str) -> str:
    if feature in PROCESS_LABELS:
        return PROCESS_LABELS[feature]
    if "current density" in feature.lower():
        return CURRENT_LABEL
    return feature


def abbreviated_feature(feature: str) -> str:
    abbreviations = {
        "均匀化温度（℃）": "HT",
        "均匀化时间": "Ht",
        "挤压温度": "ET",
        "挤压速度（mm/s）": "ES",
        "挤压比": "ER",
    }
    return abbreviations.get(feature, pretty_feature(feature))


def wrapped_feature(feature: str) -> str:
    label = pretty_feature(feature)
    replacements = {
        "Homogenization temperature": "Homogenization\ntemperature",
        "Homogenization time": "Homogenization\ntime",
        "Extrusion temperature": "Extrusion\ntemperature",
        "Extrusion speed": "Extrusion\nspeed",
        "Extrusion ratio": "Extrusion\nratio",
    }
    return replacements.get(label, label)


def beeswarm_feature_label(feature: str) -> str:
    return abbreviated_feature(feature)


def feature_category(feature: str) -> str:
    if feature == TEXT_FEATURE:
        return "Text semantics"
    if feature in ELEMENTS:
        return "Composition"
    if "current density" in feature.lower():
        return "Operating condition"
    return "Processing"


def mean_pool(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(hidden_state)
    return (hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def text_cache_path(texts: Iterable[str]) -> Path:
    digest = hashlib.sha256()
    digest.update(MODEL_NAME.encode("utf-8"))
    digest.update(f"|{MAX_LENGTH}|".encode("ascii"))
    for text in texts:
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")
    return CACHE_DIR / f"embeddings_{digest.hexdigest()[:18]}.npy"


def embed_texts(
    texts: List[str],
    tokenizer,
    encoder,
    device: torch.device,
) -> np.ndarray:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = text_cache_path(texts)
    if path.exists():
        return np.load(path)

    chunks = []
    with torch.inference_mode():
        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start:start + BATCH_SIZE]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            output = encoder(**encoded)
            pooled = mean_pool(output.last_hidden_state, encoded["attention_mask"])
            chunks.append(pooled.cpu().numpy().astype(np.float32))
    embeddings = np.vstack(chunks)
    np.save(path, embeddings)
    return embeddings


def build_text_from_values(values: Dict[str, float], numeric_cols: List[str]) -> str:
    active = []
    for element in ELEMENTS:
        if element != "Mg" and element in numeric_cols and abs(float(values.get(element, 0.0))) > 1e-12:
            active.append(element)

    if active:
        composition = "Mg-based alloy with " + ", ".join(sorted(active)) + "."
    else:
        composition = "Mg-based alloy."

    steps = []
    ht_col = next((col for col in numeric_cols if "均匀化温度" in col), None)
    et_col = next((col for col in numeric_cols if col == "挤压温度"), None)
    if ht_col and abs(float(values.get(ht_col, 0.0))) > 1e-12:
        steps.append("homogenization")
    if et_col and abs(float(values.get(et_col, 0.0))) > 1e-12:
        steps.append("extrusion")
    if steps:
        return composition + " After " + " and ".join(steps) + "."
    return composition


def load_inputs():
    df = clean_columns(pd.read_excel(RAWDATA_PATH, sheet_name="Sheet1"))
    numeric_cols = joblib.load(MODEL_DIR / "numeric_cols_used.joblib")
    numeric_cols = [str(col).strip() for col in numeric_cols]
    current_col = next(col for col in numeric_cols if "current density" in col.lower())

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=numeric_cols).reset_index(drop=True)

    model_v = joblib.load(MODEL_DIR / "stack_voltage.joblib")
    model_e = joblib.load(MODEL_DIR / "stack_efficiency.joblib")
    return df, numeric_cols, current_col, model_v, model_e


def load_encoder():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    try:
        encoder = AutoModel.from_pretrained(
            MODEL_NAME,
            add_pooling_layer=False,
            local_files_only=True,
        )
    except TypeError:
        encoder = AutoModel.from_pretrained(MODEL_NAME, local_files_only=True)
    encoder.to(device).eval()
    return tokenizer, encoder, device


def predict_pack(model_pack, X_num: np.ndarray, X_emb: np.ndarray):
    base_num, base_text, meta = model_pack
    pred_num = base_num.predict(X_num)
    pred_text = base_text.predict(X_emb)
    pred_final = meta.predict(np.column_stack([pred_num, pred_text]))
    return pred_final, pred_num, pred_text


def expected_scalar(value) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def hierarchical_explanation(
    model_pack,
    X_num: np.ndarray,
    X_emb: np.ndarray,
    numeric_cols: List[str],
):
    base_num, base_text, meta = model_pack
    pred_final, pred_num, pred_text = predict_pack(model_pack, X_num, X_emb)

    num_explainer = shap.TreeExplainer(base_num)
    meta_explainer = shap.TreeExplainer(meta)
    num_shap = np.asarray(num_explainer.shap_values(X_num), dtype=float)
    meta_input = np.column_stack([pred_num, pred_text])
    meta_shap = np.asarray(meta_explainer.shap_values(meta_input), dtype=float)

    abs_num = np.abs(num_shap)
    numeric_weights = abs_num / np.maximum(abs_num.sum(axis=1, keepdims=True), 1e-12)
    allocated_global = numeric_weights * np.abs(meta_shap[:, [0]])
    numeric_importance = allocated_global.mean(axis=0)
    text_importance = float(np.mean(np.abs(meta_shap[:, 1])))

    global_df = pd.DataFrame({
        "Feature": [abbreviated_feature(col) for col in numeric_cols] + [TEXT_FEATURE],
        "Raw feature": numeric_cols + [TEXT_FEATURE],
        "Mean absolute SHAP value": np.r_[numeric_importance, text_importance],
    })
    global_df["Category"] = global_df["Raw feature"].map(feature_category)
    global_df = global_df.sort_values("Mean absolute SHAP value", ascending=False).reset_index(drop=True)

    state = {
        "pred_final": pred_final,
        "pred_num": pred_num,
        "pred_text": pred_text,
        "num_explainer": num_explainer,
        "meta_explainer": meta_explainer,
        "num_shap": num_shap,
        "meta_shap": meta_shap,
        "num_expected": expected_scalar(num_explainer.expected_value),
        "meta_expected": expected_scalar(meta_explainer.expected_value),
    }
    return global_df, state


def candidate_row(numeric_cols: List[str], current_col: str) -> np.ndarray:
    values = {col: 0.0 for col in numeric_cols}
    values.update({
        "Al": 8.0,
        "Er": 0.8,
        "Gd": 2.4,
        "均匀化温度（℃）": 450.0,
        "均匀化时间": 24.0,
        "挤压温度": 400.0,
        "挤压速度（mm/s）": 0.5,
        "挤压比": 45.0,
        current_col: 40.0,
    })
    return np.array([[float(values[col]) for col in numeric_cols]], dtype=float)


def local_hierarchical_contributions(
    model_pack,
    state: Dict[str, object],
    X_candidate: np.ndarray,
    E_candidate: np.ndarray,
    numeric_cols: List[str],
) -> Tuple[pd.DataFrame, float, float]:
    final_pred, numeric_pred, text_pred = predict_pack(model_pack, X_candidate, E_candidate)
    num_shap = np.asarray(state["num_explainer"].shap_values(X_candidate), dtype=float)[0]
    meta_input = np.column_stack([numeric_pred, text_pred])
    meta_shap = np.asarray(state["meta_explainer"].shap_values(meta_input), dtype=float)[0]

    numeric_delta = float(np.sum(num_shap))
    if abs(numeric_delta) < 1e-10:
        allocated = np.zeros_like(num_shap)
    else:
        allocated = num_shap * (float(meta_shap[0]) / numeric_delta)

    rows = []
    for idx, col in enumerate(numeric_cols):
        rows.append({
            "Feature": pretty_feature(col),
            "Raw feature": col,
            "Value": float(X_candidate[0, idx]),
            "Contribution": float(allocated[idx]),
        })
    rows.append({
        "Feature": TEXT_FEATURE,
        "Raw feature": TEXT_FEATURE,
        "Value": np.nan,
        "Contribution": float(meta_shap[1]),
    })
    local = pd.DataFrame(rows)
    local["Absolute contribution"] = local["Contribution"].abs()

    text_row = local[local["Raw feature"] == TEXT_FEATURE]
    numeric = local[local["Raw feature"] != TEXT_FEATURE].sort_values("Absolute contribution", ascending=False)
    keep_numeric = numeric.head(6)
    other = numeric.iloc[6:]
    condensed = pd.concat([keep_numeric, text_row], ignore_index=True)
    if not other.empty:
        condensed = pd.concat([
            condensed,
            pd.DataFrame([{
                "Feature": "Other numeric features",
                "Raw feature": "Other numeric features",
                "Value": np.nan,
                "Contribution": float(other["Contribution"].sum()),
                "Absolute contribution": float(abs(other["Contribution"].sum())),
            }]),
        ], ignore_index=True)
    condensed = condensed.sort_values("Absolute contribution", ascending=False).reset_index(drop=True)
    return condensed, float(state["meta_expected"]), float(final_pred[0])


def choose_background(df: pd.DataFrame, numeric_cols: List[str], n: int = 36) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    if len(df) <= n:
        return df[numeric_cols].copy().reset_index(drop=True)
    indices = np.sort(rng.choice(len(df), size=n, replace=False))
    return df.loc[indices, numeric_cols].reset_index(drop=True)


def conditional_pdp(
    spec: Dict[str, object],
    background: pd.DataFrame,
    numeric_cols: List[str],
    current_col: str,
    model_v,
    model_e,
    tokenizer,
    encoder,
    device,
    resolution: int = 29,
) -> pd.DataFrame:
    x_grid = np.linspace(*spec["x_range"], resolution)
    y_grid = np.linspace(*spec["y_range"], resolution)
    process_cols = [col for col in numeric_cols if col not in ELEMENTS and col != current_col]

    rows = []
    texts = []
    point_ids = []
    grid_records = []
    point_id = 0

    for y_value in y_grid:
        for x_value in x_grid:
            fixed_sum = sum(float(value) for value in spec.get("fixed_elements", {}).values())
            valid = (x_value + y_value + fixed_sum) <= float(spec["sum_limit"]) + 1e-9
            grid_records.append({
                "Point": point_id,
                spec["x_feature"]: x_value,
                spec["y_feature"]: y_value,
                "Valid": valid,
            })
            if valid:
                for _, process_row in background.iterrows():
                    values = {col: 0.0 for col in numeric_cols}
                    for element, amount in spec.get("fixed_elements", {}).items():
                        values[element] = float(amount)
                    values[spec["x_feature"]] = float(x_value)
                    values[spec["y_feature"]] = float(y_value)
                    values[current_col] = float(spec["current"])
                    for col in process_cols:
                        values[col] = float(process_row[col])
                    rows.append([values[col] for col in numeric_cols])
                    texts.append(build_text_from_values(values, numeric_cols))
                    point_ids.append(point_id)
            point_id += 1

    X_num = np.asarray(rows, dtype=float)
    unique_texts = list(dict.fromkeys(texts))
    unique_embeddings = embed_texts(unique_texts, tokenizer, encoder, device)
    emb_lookup = {text: unique_embeddings[idx] for idx, text in enumerate(unique_texts)}
    X_emb = np.vstack([emb_lookup[text] for text in texts])
    pred_v, _, _ = predict_pack(model_v, X_num, X_emb)
    pred_e, _, _ = predict_pack(model_e, X_num, X_emb)

    point_ids_array = np.asarray(point_ids, dtype=int)
    counts = np.bincount(point_ids_array, minlength=len(grid_records))
    sum_v = np.bincount(point_ids_array, weights=pred_v, minlength=len(grid_records))
    sum_e = np.bincount(point_ids_array, weights=pred_e, minlength=len(grid_records))
    mean_v = np.divide(sum_v, counts, out=np.full_like(sum_v, np.nan), where=counts > 0)
    mean_e = np.divide(sum_e, counts, out=np.full_like(sum_e, np.nan), where=counts > 0)

    result = pd.DataFrame(grid_records)
    result["Partial dependence voltage (mV)"] = mean_v
    result["Partial dependence efficiency (%)"] = mean_e
    valid_mask = result["Valid"] & result["Partial dependence voltage (mV)"].notna()
    for source, destination in [
        ("Partial dependence voltage (mV)", "Normalized voltage"),
        ("Partial dependence efficiency (%)", "Normalized efficiency"),
    ]:
        values = result.loc[valid_mask, source]
        low, high = float(values.min()), float(values.max())
        result[destination] = (result[source] - low) / max(high - low, 1e-12)
    result["Joint desirability"] = 0.5 * result["Normalized voltage"] + 0.5 * result["Normalized efficiency"]
    result["Current density"] = float(spec["current"])
    result["Alloy system"] = spec["system"]
    return result


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.08,
        f"({label})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=32,
        fontweight="bold",
        clip_on=False,
    )


def draw_gradient_bar(ax, y: float, value: float, height: float, max_value: float) -> None:
    cmap = LinearSegmentedColormap.from_list("shap_gradient", ["#DCEAF4", "#F3D7CF", RED])
    gradient = np.linspace(0, min(1.0, value / max(max_value, 1e-12)), 256).reshape(1, -1)
    ax.imshow(
        gradient,
        extent=[0, value, y - height / 2, y + height / 2],
        cmap=cmap,
        aspect="auto",
        interpolation="bicubic",
        vmin=0,
        vmax=1,
        zorder=2,
    )
    ax.add_patch(Rectangle((0, y - height / 2), value, height, fill=False, edgecolor=NAVY, linewidth=0.8, zorder=3))


def plot_global_importance(
    ax: plt.Axes,
    global_df: pd.DataFrame,
    unit: str,
    label: str,
) -> None:
    display_df = global_df[global_df["Raw feature"] != "Mg"].copy()
    top = display_df.head(8).sort_values("Mean absolute SHAP value", ascending=True)
    values = top["Mean absolute SHAP value"].to_numpy(dtype=float)
    y = np.arange(len(top))
    max_value = float(values.max()) * 1.28
    for y_pos, value in zip(y, values):
        draw_gradient_bar(ax, y_pos, value, 0.68, max_value)
        ax.text(value + max_value * 0.018, y_pos, f"{value:.2f}", va="center", fontsize=20.5, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(top["Feature"], fontsize=22.0, fontweight="bold")
    ax.set_ylim(-0.82, len(top) - 0.30)
    ax.set_xlim(0, max_value)
    ax.set_xlabel(f"Mean |SHAP value| ({unit})", fontsize=22.0, fontweight="bold")
    ax.grid(True, axis="x", linestyle=":", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=19.0, width=1.35)
    for tick_label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        tick_label.set_fontweight("bold")

    panel_label(ax, label)


def plot_pdp_target(
    ax: plt.Axes,
    pdp_df: pd.DataFrame,
    spec: Dict[str, object],
    target: str,
    label: str,
):
    resolution = int(round(math.sqrt(len(pdp_df))))
    x = pdp_df[spec["x_feature"]].to_numpy().reshape(resolution, resolution)
    y = pdp_df[spec["y_feature"]].to_numpy().reshape(resolution, resolution)
    if target == "voltage":
        column = "Partial dependence voltage (mV)"
        values = pdp_df[column].to_numpy().reshape(resolution, resolution)
        unit = "mV"
        title = "Voltage response"
        cmap = "Blues"
        fmt = "%.0f"
    else:
        column = "Partial dependence efficiency (%)"
        values = pdp_df[column].to_numpy().reshape(resolution, resolution)
        unit = "%"
        title = "Efficiency response"
        cmap = "OrRd"
        fmt = "%.1f"

    valid_values = values[np.isfinite(values)]
    levels = np.linspace(float(valid_values.min()), float(valid_values.max()), 13)
    filled = ax.contourf(x, y, values, levels=levels, cmap=cmap)
    contours = ax.contour(x, y, values, levels=levels[2:-1:2], colors=NAVY, linewidths=1.0, alpha=0.72)
    ax.clabel(contours, inline=True, fontsize=10.5, fmt=fmt)
    ax.scatter(*spec["candidate"], marker="*", s=260, color="#FFD447", edgecolor=NAVY, linewidth=1.4, zorder=5)
    ax.annotate(
        "Selected\n0.8Er-2.4Gd",
        xy=spec["candidate"],
        xytext=(18, -27),
        textcoords="offset points",
        fontsize=11.5,
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": NAVY, "lw": 1.1},
    )

    ax.set_xlabel(f"{spec['x_feature']} content (wt%)", fontsize=12.5, fontweight="bold")
    ax.set_ylabel(f"{spec['y_feature']} content (wt%)", fontsize=12.5, fontweight="bold")
    ax.set_title(
        f"{title} at {spec['current']:g} mA cm$^{{-2}}$\n{spec['system']} | Al = 8 wt%, selected process fixed",
        fontsize=15.5,
        fontweight="bold",
        pad=8,
    )
    ax.tick_params(labelsize=11.5)
    colorbar = plt.colorbar(filled, ax=ax, orientation="horizontal", fraction=0.055, pad=0.16, aspect=34)
    colorbar.set_label(f"Partial dependence ({unit})", fontsize=11.8, fontweight="bold")
    colorbar.ax.tick_params(labelsize=10)
    panel_label(ax, label)
    return filled


def format_local_label(row: pd.Series) -> str:
    feature = row["Raw feature"]
    value = row["Value"]
    if feature in ELEMENTS and np.isfinite(value):
        return f"{row['Feature']} = {value:.2f} wt%"
    if feature == "均匀化温度（℃）" or feature == "挤压温度":
        return f"{row['Feature']} = {value:.0f} °C"
    if feature == "均匀化时间":
        return f"{row['Feature']} = {value:.1f} h"
    if feature == "挤压速度（mm/s）":
        return f"{row['Feature']} = {value:.1f} mm s$^{{-1}}$"
    if feature == "挤压比":
        return f"{row['Feature']} = {value:.0f}"
    if "current density" in str(feature).lower():
        return f"{CURRENT_LABEL} = {value:.1f} mA cm$^{{-2}}$"
    return str(row["Feature"])


def plot_local_waterfall(
    ax: plt.Axes,
    local_df: pd.DataFrame,
    baseline: float,
    final_value: float,
    title: str,
    unit: str,
    label: str,
) -> None:
    plot_df = local_df.copy()
    plot_df["Display"] = plot_df.apply(format_local_label, axis=1)
    n = len(plot_df)
    y_positions = np.arange(n)[::-1]
    current = baseline

    x_values = [baseline, final_value]
    for y_pos, (_, row) in zip(y_positions, plot_df.iterrows()):
        contribution = float(row["Contribution"])
        next_value = current + contribution
        left = min(current, next_value)
        width = abs(contribution)
        color = RED if contribution >= 0 else BLUE
        ax.barh(y_pos, width, left=left, height=0.62, color=color, alpha=0.88, edgecolor="white", linewidth=0.8)
        ax.plot([next_value, next_value], [y_pos - 0.31, y_pos - 0.69], color="#9AA3AA", linewidth=0.8)
        text_x = next_value + (0.012 if contribution >= 0 else -0.012) * max(abs(final_value - baseline), 1.0)
        ax.text(
            text_x,
            y_pos,
            f"{contribution:+.2f}",
            ha="left" if contribution >= 0 else "right",
            va="center",
            fontsize=10.5,
            fontweight="bold",
            color=color,
        )
        x_values.extend([current, next_value, text_x])
        current = next_value

    ax.axvline(baseline, color=BLUE, linestyle="--", linewidth=1.2)
    ax.axvline(final_value, color=RED, linestyle="--", linewidth=1.2)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_df["Display"], fontsize=11.2)
    ax.set_xlabel(f"Model output ({unit})", fontsize=13.5, fontweight="bold")
    ax.set_title(title, fontsize=16.5, fontweight="bold", pad=9)
    ax.grid(True, axis="x", linestyle=":", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    span = max(max(x_values) - min(x_values), 1.0)
    ax.set_xlim(min(x_values) - 0.12 * span, max(x_values) + 0.15 * span)
    labels_are_close = abs(final_value - baseline) < 0.18 * span
    ax.text(
        baseline,
        0.985,
        f"E[f(X)] = {baseline:.2f}",
        transform=ax.get_xaxis_transform(),
        ha="right" if labels_are_close else "center",
        va="top",
        fontsize=10.5,
        color=BLUE,
        fontweight="bold",
    )
    ax.text(
        final_value,
        0.985,
        f"f(x) = {final_value:.2f}",
        transform=ax.get_xaxis_transform(),
        ha="left" if labels_are_close else "center",
        va="top",
        fontsize=10.5,
        color=RED,
        fontweight="bold",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    panel_label(ax, label)


def signed_hierarchical_numeric_shap(state: Dict[str, object]) -> np.ndarray:
    """Project numeric-branch SHAP directions onto the final fusion output."""
    numeric_shap = np.asarray(state["num_shap"], dtype=float)
    meta_numeric = np.asarray(state["meta_shap"], dtype=float)[:, [0]]
    scale = np.maximum(np.abs(numeric_shap).sum(axis=1, keepdims=True), 1e-12)
    return (numeric_shap / scale) * meta_numeric


def select_composition_processing_features(
    shap_values: np.ndarray,
    numeric_cols: List[str],
    n_composition: int = 6,
    n_processing: int = 4,
    required_composition: Iterable[str] = (),
) -> List[int]:
    importance = np.mean(np.abs(shap_values), axis=0)
    composition_candidates = [idx for idx, feature in enumerate(numeric_cols) if feature in ELEMENTS and feature != "Mg"]
    processing = [idx for idx, feature in enumerate(numeric_cols) if feature_category(feature) == "Processing"]
    ranked_composition = sorted(composition_candidates, key=lambda idx: importance[idx], reverse=True)
    required_indices = [
        numeric_cols.index(feature)
        for feature in required_composition
        if feature in numeric_cols and feature != "Mg"
    ]
    composition = required_indices + [idx for idx in ranked_composition if idx not in required_indices]
    composition = composition[:n_composition]
    processing = sorted(processing, key=lambda idx: importance[idx], reverse=True)[:n_processing]
    return sorted(composition + processing, key=lambda idx: importance[idx])


def plot_shap_beeswarm(
    ax: plt.Axes,
    X_num: np.ndarray,
    shap_values: np.ndarray,
    numeric_cols: List[str],
    unit: str,
    label: str,
    current_density: float,
) -> pd.DataFrame:
    required_composition = ("Er", "Gd") if np.isclose(current_density, 40.0) else ()
    selected = select_composition_processing_features(
        shap_values,
        numeric_cols,
        required_composition=required_composition,
    )
    rng = np.random.default_rng(SEED)
    rows = []
    cmap = mpl.colormaps["coolwarm"]

    for y_pos, feature_idx in enumerate(selected):
        values = X_num[:, feature_idx].astype(float)
        impacts = shap_values[:, feature_idx].astype(float)
        low, high = np.quantile(values, [0.02, 0.98])
        if high - low < 1e-12:
            normalized = np.full_like(values, 0.5)
        else:
            normalized = np.clip((values - low) / (high - low), 0.0, 1.0)
        jitter = np.clip(rng.normal(0.0, 0.115, len(values)), -0.28, 0.28)
        ax.scatter(
            impacts,
            y_pos + jitter,
            c=normalized,
            cmap=cmap,
            vmin=0,
            vmax=1,
            s=42,
            alpha=0.68,
            edgecolors="none",
            rasterized=True,
        )
        rows.extend({
            "Sample": sample_idx + 1,
            "Feature": pretty_feature(numeric_cols[feature_idx]),
            "Raw feature": numeric_cols[feature_idx],
            "Feature value": float(values[sample_idx]),
            "SHAP impact": float(impacts[sample_idx]),
            "Current density": current_density,
        } for sample_idx in range(len(values)))

    labels = [beeswarm_feature_label(numeric_cols[idx]) for idx in selected]
    ax.set_yticks(np.arange(len(selected)))
    ax.set_yticklabels(labels, fontsize=22.0, fontweight="bold")
    ax.axvline(0.0, color=NAVY, linewidth=1.25)
    ax.set_xlabel(f"SHAP impact ({unit})", fontsize=22.5, fontweight="bold")
    ax.grid(True, axis="x", linestyle=":", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=20.0, width=1.45)
    for tick_label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        tick_label.set_fontweight("bold")
    ax.text(
        0.985,
        0.035,
        f"{current_density:g} mA cm$^{{-2}}$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=28.0,
        fontweight="bold",
        color=NAVY,
        zorder=6,
    )

    colorbar = plt.colorbar(
        mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap=cmap),
        ax=ax,
        orientation="vertical",
        fraction=0.035,
        pad=0.025,
    )
    colorbar.set_ticks([0, 1])
    colorbar.set_ticklabels(["Low", "High"])
    colorbar.set_label("Feature value", fontsize=21.0, fontweight="bold")
    colorbar.ax.tick_params(labelsize=19.0, width=1.3)
    for tick_label in colorbar.ax.get_yticklabels():
        tick_label.set_fontweight("bold")
    panel_label(ax, label)
    return pd.DataFrame(rows)


def top_feature_index(
    shap_values: np.ndarray,
    numeric_cols: List[str],
    category: str,
) -> int:
    importance = np.mean(np.abs(shap_values), axis=0)
    if category == "composition":
        candidates = [idx for idx, feature in enumerate(numeric_cols) if feature in ELEMENTS and feature != "Mg"]
    else:
        candidates = [idx for idx, feature in enumerate(numeric_cols) if feature_category(feature) == "Processing"]
    return max(candidates, key=lambda idx: importance[idx])


def plot_dependence_pair(
    fig: plt.Figure,
    subplot_spec,
    X_num: np.ndarray,
    shap_values: np.ndarray,
    numeric_cols: List[str],
    target_title: str,
    unit: str,
    label: str,
) -> pd.DataFrame:
    inner = GridSpecFromSubplotSpec(1, 2, subplot_spec=subplot_spec, wspace=0.48)
    ax_element = fig.add_subplot(inner[0, 0])
    ax_process = fig.add_subplot(inner[0, 1])
    element_idx = top_feature_index(shap_values, numeric_cols, "composition")
    process_idx = top_feature_index(shap_values, numeric_cols, "processing")

    pairs = [
        (ax_element, element_idx, process_idx, "Element dependence"),
        (ax_process, process_idx, element_idx, "Process dependence"),
    ]
    for ax, feature_idx, color_idx, subtitle in pairs:
        x = X_num[:, feature_idx].astype(float)
        y = shap_values[:, feature_idx].astype(float)
        color_values = X_num[:, color_idx].astype(float)
        scatter = ax.scatter(
            x,
            y,
            c=color_values,
            cmap="viridis",
            s=32,
            alpha=0.68,
            edgecolors="white",
            linewidths=0.25,
            rasterized=True,
        )
        ax.axhline(0.0, color=NAVY, linewidth=0.95, linestyle="--")
        feature = numeric_cols[feature_idx]
        if feature in ELEMENTS:
            xlabel = f"{pretty_feature(feature)} content (wt%)"
        else:
            xlabel = pretty_feature(feature)
        ax.set_xlabel(xlabel, fontsize=14.0, fontweight="bold")
        ax.set_ylabel(f"SHAP impact ({unit})", fontsize=14.0, fontweight="bold")
        ax.set_title(f"{target_title} | {subtitle}", fontsize=16.0, fontweight="bold", pad=9)
        ax.grid(True, linestyle=":", color=GRID, linewidth=0.75)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=12.3)
        colorbar = fig.colorbar(scatter, ax=ax, fraction=0.048, pad=0.025)
        colorbar.set_label(pretty_feature(numeric_cols[color_idx]), fontsize=12.0, fontweight="bold")
        colorbar.ax.tick_params(labelsize=11.0)

    ax_element.text(
        -0.30,
        1.10,
        f"({label})",
        transform=ax_element.transAxes,
        ha="left",
        va="top",
        fontsize=22,
        fontweight="bold",
        clip_on=False,
    )
    return pd.DataFrame({
        "Sample": np.arange(1, len(X_num) + 1),
        "Top element": pretty_feature(numeric_cols[element_idx]),
        "Element value": X_num[:, element_idx],
        "Element SHAP impact": shap_values[:, element_idx],
        "Top process": pretty_feature(numeric_cols[process_idx]),
        "Process value": X_num[:, process_idx],
        "Process SHAP impact": shap_values[:, process_idx],
    })


def export_source_data(
    global_v: pd.DataFrame,
    global_e: pd.DataFrame,
    conditional_beeswarms: Dict[Tuple[float, str], pd.DataFrame],
) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    workbook = FIGURE_DIR / "Figure4_explainability_source_data.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        global_v.to_excel(writer, sheet_name="Global_SHAP_voltage", index=False)
        global_e.to_excel(writer, sheet_name="Global_SHAP_efficiency", index=False)
        for (current_density, target), frame in conditional_beeswarms.items():
            frame.to_excel(
                writer,
                sheet_name=f"CD{current_density:g}_{target}",
                index=False,
            )


def main() -> None:
    warnings.filterwarnings("ignore")
    configure_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading final BatteryBERT fusion models...", flush=True)
    df, numeric_cols, current_col, model_v, model_e = load_inputs()
    tokenizer, encoder, device = load_encoder()

    X_num = df[numeric_cols].astype(float).to_numpy()
    raw_texts = [
        build_text_from_values({col: float(row[col]) for col in numeric_cols}, numeric_cols)
        for _, row in df.iterrows()
    ]
    print("Encoding experimental records...", flush=True)
    X_emb = embed_texts(raw_texts, tokenizer, encoder, device)

    print("Computing SHAP explanations...", flush=True)
    global_v, state_v = hierarchical_explanation(model_v, X_num, X_emb, numeric_cols)
    global_e, state_e = hierarchical_explanation(model_e, X_num, X_emb, numeric_cols)
    signed_v = signed_hierarchical_numeric_shap(state_v)
    signed_e = signed_hierarchical_numeric_shap(state_e)

    display_global_v = global_v[global_v["Raw feature"] != "Mg"].reset_index(drop=True)
    display_global_e = global_e[global_e["Raw feature"] != "Mg"].reset_index(drop=True)

    fig = plt.figure(figsize=(21.0, 29.0))
    grid = GridSpec(
        4,
        2,
        figure=fig,
        height_ratios=[1.12, 1.0, 1.0, 1.0],
        hspace=0.28,
        wspace=0.38,
        left=0.095,
        right=0.975,
        bottom=0.040,
        top=0.965,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    plot_global_importance(ax_a, display_global_v, "mV", "a")
    plot_global_importance(ax_b, display_global_e, "%", "b")

    current_idx = numeric_cols.index(current_col)
    conditional_beeswarms: Dict[Tuple[float, str], pd.DataFrame] = {}
    panel_pairs = [(2.5, "c", "d"), (10.0, "e", "f"), (40.0, "g", "h")]
    for row_idx, (current_density, voltage_label, efficiency_label) in enumerate(panel_pairs, start=1):
        mask = np.isclose(X_num[:, current_idx], current_density)
        if not np.any(mask):
            raise RuntimeError(f"No records found at {current_density:g} mA cm^-2.")
        ax_voltage = fig.add_subplot(grid[row_idx, 0])
        ax_efficiency = fig.add_subplot(grid[row_idx, 1])
        conditional_beeswarms[(current_density, "voltage")] = plot_shap_beeswarm(
            ax_voltage,
            X_num[mask],
            signed_v[mask],
            numeric_cols,
            "mV",
            voltage_label,
            current_density,
        )
        conditional_beeswarms[(current_density, "efficiency")] = plot_shap_beeswarm(
            ax_efficiency,
            X_num[mask],
            signed_e[mask],
            numeric_cols,
            "%",
            efficiency_label,
            current_density,
        )

    export_source_data(display_global_v, display_global_e, conditional_beeswarms)

    stem = FIGURE_DIR / "Figure4_model_explainability"
    fig.savefig(stem.with_suffix(".png"), dpi=450, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)

    del encoder
    gc.collect()
    print("Figure 4 completed.", flush=True)


if __name__ == "__main__":
    main()
