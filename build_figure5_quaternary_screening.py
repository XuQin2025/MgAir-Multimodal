# -*- coding: utf-8 -*-
"""Build Figure 5 for current-specific quaternary Mg-alloy searches."""

from __future__ import annotations

import itertools
import re
import warnings
from pathlib import Path

import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, Normalize

from build_figure4_explainability import embed_texts, load_encoder, predict_pack


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "outputs_design_search_multiCD_selected"
FIGURE_DIR = ROOT / "paper_figures"

SEED = 42
N_SAMPLE = 36000
N_FAMILY_SAMPLE = 1200
SEARCH_CONFIGS = {
    2.5: {
        "element_pool": ["Al", "Ca", "Zn", "Sn", "Y", "Sr"],
        "composition_step": 0.1,
        "maximum_additions": 2.06,
    },
    10.0: {
        "element_pool": ["Al", "Ca", "Mn", "Zn", "La", "Gd", "In"],
        "composition_step": 0.5,
        "maximum_additions": 10.43,
    },
    40.0: {
        "element_pool": ["Al", "Ca", "Mn", "Zn", "Er", "Y", "La", "Sm", "Gd", "Li"],
        "composition_step": 0.8,
        "maximum_additions": 11.97,
    },
}
PROCESS_GRID = {
    "HT": [400.0, 450.0],
    "Ht": [6.0, 12.0, 24.0],
    "ET": [250.0, 300.0, 350.0, 400.0],
    "ES": [0.5, 1.5, 2.5],
    "ER": [30.0, 45.0],
}

BLUE = "#557FA7"
MID_BLUE = "#7FA8CA"
PALE_BLUE = "#D8E5EE"
NAVY = "#23384D"
MAGENTA = "#D92F6A"
YELLOW = "#F2C94C"
GRID = "#CDD4DA"


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.sans-serif": ["Arial"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "axes.linewidth": 1.15,
        "axes.labelsize": 25,
        "axes.titlesize": 25,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })


def panel_label(
    ax: plt.Axes,
    label: str,
    *,
    x: float = -0.17,
    y: float = 1.11,
) -> None:
    ax.text(
        x,
        y,
        f"({label})",
        transform=ax.transAxes,
        fontsize=30,
        fontweight="bold",
        ha="left",
        va="top",
        clip_on=False,
    )


def strict_quaternary(row: pd.Series) -> bool:
    parts = str(row["AlloySystem"]).split("-")
    return len(parts) == 3 and all(float(row.get(element, 0.0)) > 1e-12 for element in parts)


def load_retained_results(current_density: float) -> tuple[pd.DataFrame, pd.Series, float, float, float, float]:
    top_path = MODEL_DIR / f"grid_top_CD{current_density:g}_Ternary1.xlsx"
    element_pool = SEARCH_CONFIGS[current_density]["element_pool"]
    frames = []
    score_candidates = None
    for sheet in ["top_by_voltage", "top_by_efficiency", "top_by_score"]:
        frame = pd.read_excel(top_path, sheet_name=sheet)
        frame["Retained source"] = sheet
        frames.append(frame)
        if sheet == "top_by_score":
            score_candidates = frame[frame.apply(strict_quaternary, axis=1)].copy()
    retained = pd.concat(frames, ignore_index=True)
    retained = retained[retained.apply(strict_quaternary, axis=1)].copy()

    design_cols = [
        "AlloySystem", *element_pool, "HT", "Ht", "ET", "ES", "ER",
        "Pred_AvgDischargeVoltage_mV", "Pred_AnodicEfficiency_pct",
    ]
    retained = retained.sort_values(
        ["Pred_AvgDischargeVoltage_mV", "Pred_AnodicEfficiency_pct"],
        ascending=False,
    ).drop_duplicates(subset=design_cols).reset_index(drop=True)

    settings = pd.read_excel(top_path, sheet_name="settings").iloc[0]
    numbers = [float(value) for value in re.findall(r"[-+]?\d+(?:\.\d+)?", str(settings["score_stats"]))]
    mean_v, std_v, mean_e, std_e = numbers[-4:]

    if score_candidates is None:
        raise RuntimeError(f"The top_by_score sheet is missing at {current_density:g} mA cm^-2.")
    score_candidates = score_candidates.sort_values(
        ["Score_zV_plus_zE", "Pred_AnodicEfficiency_pct", "Pred_AvgDischargeVoltage_mV"],
        ascending=False,
    )
    if score_candidates.empty:
        raise RuntimeError(f"No score-ranked quaternary candidate was found at {current_density:g} mA cm^-2.")
    target = score_candidates.iloc[0].copy()
    return retained, target, mean_v, std_v, mean_e, std_e


def build_representative_grid(
    current_density: float,
    family_system: tuple[str, ...],
    priority_designs: pd.DataFrame,
) -> pd.DataFrame:
    config = SEARCH_CONFIGS[current_density]
    element_pool = config["element_pool"]
    composition_step = float(config["composition_step"])
    maximum_additions = float(config["maximum_additions"])
    rng = np.random.default_rng(SEED + int(round(current_density * 10)))
    systems = list(itertools.combinations(element_pool, 3))
    values = np.round(np.arange(composition_step, maximum_additions + 1e-9, composition_step), 3)
    compositions = np.asarray(
        [triple for triple in itertools.product(values, repeat=3) if sum(triple) <= maximum_additions + 1e-9],
        dtype=float,
    )
    processes = list(itertools.product(*PROCESS_GRID.values()))

    system_idx = rng.integers(0, len(systems), size=N_SAMPLE)
    comp_idx = rng.integers(0, len(compositions), size=N_SAMPLE)
    process_idx = rng.integers(0, len(processes), size=N_SAMPLE)

    rows = []
    for s_idx, c_idx, p_idx in zip(system_idx, comp_idx, process_idx):
        system = systems[int(s_idx)]
        composition = compositions[int(c_idx)]
        process = processes[int(p_idx)]
        row = {element: 0.0 for element in element_pool}
        row.update(dict(zip(system, composition)))
        row.update(dict(zip(PROCESS_GRID, process)))
        row["AlloySystem"] = "-".join(system)
        row["current_density"] = current_density
        row["sum_selected_additions_wt%"] = float(composition.sum())
        rows.append(row)

    family_space_size = len(compositions) * len(processes)
    family_indices = rng.choice(
        family_space_size,
        size=min(N_FAMILY_SAMPLE, family_space_size),
        replace=False,
    )
    for flat_idx in family_indices:
        composition = compositions[int(flat_idx) // len(processes)]
        process = processes[int(flat_idx) % len(processes)]
        row = {element: 0.0 for element in element_pool}
        row.update(dict(zip(family_system, composition)))
        row.update(dict(zip(PROCESS_GRID, process)))
        row["AlloySystem"] = "-".join(family_system)
        row["current_density"] = current_density
        row["sum_selected_additions_wt%"] = float(composition.sum())
        rows.append(row)

    # Resolve the sparse high-performance tail with complete composition grids
    # evaluated at the best process route of each top-ranked alloy system.
    for _, design in priority_designs.iterrows():
        system = tuple(str(design["AlloySystem"]).split("-"))
        process = tuple(float(design[column]) for column in PROCESS_GRID)
        for composition in compositions:
            row = {element: 0.0 for element in element_pool}
            row.update(dict(zip(system, composition)))
            row.update(dict(zip(PROCESS_GRID, process)))
            row["AlloySystem"] = "-".join(system)
            row["current_density"] = current_density
            row["sum_selected_additions_wt%"] = float(composition.sum())
            rows.append(row)

    grid = pd.DataFrame(rows)
    keys = ["AlloySystem", *element_pool, "HT", "Ht", "ET", "ES", "ER"]
    return grid.drop_duplicates(subset=keys).reset_index(drop=True)


def predict_grid(
    grid: pd.DataFrame,
    current_density: float,
    element_pool: list[str],
    prediction_assets: tuple[list[str], object, object, object, object, object],
) -> pd.DataFrame:
    numeric_cols, model_v, model_e, tokenizer, encoder, device = prediction_assets
    X_num = np.zeros((len(grid), len(numeric_cols)), dtype=float)
    index = {column: idx for idx, column in enumerate(numeric_cols)}
    for element in element_pool:
        X_num[:, index[element]] = grid[element].to_numpy(dtype=float)
    process_map = {
        "HT": "均匀化温度（℃）",
        "Ht": "均匀化时间",
        "ET": "挤压温度",
        "ES": "挤压速度（mm/s）",
        "ER": "挤压比",
    }
    for short, column in process_map.items():
        X_num[:, index[column]] = grid[short].to_numpy(dtype=float)
    current_col = next(column for column in numeric_cols if "current density" in column.lower())
    X_num[:, index[current_col]] = current_density

    unique_systems = sorted(grid["AlloySystem"].unique())
    texts = [
        "Mg-based alloy with " + ", ".join(sorted(system.split("-"))) + ". After homogenization and extrusion."
        for system in unique_systems
    ]
    unique_embeddings = embed_texts(texts, tokenizer, encoder, device)
    embedding_lookup = {system: unique_embeddings[idx] for idx, system in enumerate(unique_systems)}
    X_emb = np.vstack([embedding_lookup[system] for system in grid["AlloySystem"]])

    result = grid.copy()
    result["Pred_AvgDischargeVoltage_mV"] = predict_pack(model_v, X_num, X_emb)[0]
    result["Pred_AnodicEfficiency_pct"] = predict_pack(model_e, X_num, X_emb)[0]
    result["Retained source"] = "stratified grid sample"
    return result


def add_score(df: pd.DataFrame, mean_v: float, std_v: float, mean_e: float, std_e: float) -> pd.DataFrame:
    out = df.copy()
    z_v = (out["Pred_AvgDischargeVoltage_mV"] - mean_v) / std_v
    z_e = (out["Pred_AnodicEfficiency_pct"] - mean_e) / std_e
    computed = 0.5 * z_v + 0.5 * z_e
    if "Score_zV_plus_zE" in out.columns:
        out["Score_zV_plus_zE"] = out["Score_zV_plus_zE"].where(out["Score_zV_plus_zE"].notna(), computed)
    else:
        out["Score_zV_plus_zE"] = computed
    return out


def pareto_front(df: pd.DataFrame) -> pd.DataFrame:
    x_col = "Pred_AvgDischargeVoltage_mV"
    y_col = "Pred_AnodicEfficiency_pct"
    ordered = df.sort_values([x_col, y_col], ascending=[False, False])
    best_y = -np.inf
    keep = []
    for idx, row in ordered.iterrows():
        value = float(row[y_col])
        if value > best_y + 1e-10:
            keep.append(idx)
            best_y = value
    return df.loc[keep].sort_values(x_col).reset_index(drop=True)


def minmax_to_50(values: pd.Series) -> pd.Series:
    values = values.astype(float)
    low, high = float(values.min()), float(values.max())
    if high - low < 1e-12:
        return pd.Series(np.full(len(values), 25.0), index=values.index)
    return 50.0 * (values - low) / (high - low)


def compact_composition_value(value: float) -> str:
    value = float(value)
    if np.isclose(value, round(value)):
        return str(int(round(value)))
    return f"{value:.3f}".rstrip("0").rstrip(".")


def rank_display_label(row: pd.Series) -> str:
    elements = str(row["AlloySystem"]).split("-")
    indexed = [(element, float(row.get(element, 0.0)), idx) for idx, element in enumerate(elements)]
    indexed.sort(key=lambda item: (-item[1], item[2]))
    additions = [f"{compact_composition_value(value)}{element}" for element, value, _ in indexed]
    return "Mg-" + "-".join(additions)


def prepare_ranking_panel(current_density: float, top_n: int = 5) -> pd.DataFrame:
    file_path = MODEL_DIR / f"grid_top_CD{current_density:g}_Ternary1.xlsx"
    frames = []
    for sheet in ["top_by_score", "top_by_voltage", "top_by_efficiency"]:
        frame = pd.read_excel(file_path, sheet_name=sheet)
        frame["Ranking source"] = sheet
        frame["Stored score"] = frame["Score_zV_plus_zE"].notna() if "Score_zV_plus_zE" in frame.columns else False
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    if "Score_zV_plus_zE" not in data.columns:
        data["Score_zV_plus_zE"] = np.nan
    design_columns = [column for column in data.columns if column not in {"Ranking source", "Stored score", "Score_zV_plus_zE"}]
    data = data.sort_values("Stored score", ascending=False).drop_duplicates(subset=design_columns).reset_index(drop=True)
    data = data[data.apply(strict_quaternary, axis=1)].copy()

    settings = pd.read_excel(file_path, sheet_name="settings").iloc[0]
    numbers = [float(value) for value in re.findall(r"[-+]?\d+(?:\.\d+)?", str(settings["score_stats"]))]
    mean_v, std_v, mean_e, std_e = numbers[-4:]
    computed_score = (
        0.5 * (data["Pred_AvgDischargeVoltage_mV"] - mean_v) / std_v
        + 0.5 * (data["Pred_AnodicEfficiency_pct"] - mean_e) / std_e
    )
    data["Normalized score"] = data["Score_zV_plus_zE"].where(data["Score_zV_plus_zE"].notna(), computed_score)
    data["Normalized voltage contribution"] = 0.5 * (data["Pred_AvgDischargeVoltage_mV"] - mean_v) / std_v
    data["Normalized efficiency contribution"] = data["Normalized score"] - data["Normalized voltage contribution"]
    data = data.sort_values(
        ["Normalized score", "Pred_AnodicEfficiency_pct", "Pred_AvgDischargeVoltage_mV"],
        ascending=False,
    ).drop_duplicates(subset=["AlloySystem"]).head(top_n).copy()
    data = data.sort_values("Normalized score", ascending=False).reset_index(drop=True)
    data["Display label"] = data.apply(rank_display_label, axis=1)
    data["Current density"] = current_density
    return data


def plot_ranking_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    current_density: float,
    label: str,
    show_ylabel: bool,
) -> None:
    x = np.arange(len(data))
    width = 0.70
    voltage_color = "#7DB7D8"
    efficiency_color = "#F4A261"
    ax.bar(
        x,
        data["Normalized voltage contribution"],
        width=width,
        color=voltage_color,
        edgecolor="white",
        linewidth=1.35,
        zorder=2,
    )
    ax.bar(
        x,
        data["Normalized efficiency contribution"],
        width=width,
        bottom=data["Normalized voltage contribution"],
        color=efficiency_color,
        edgecolor="white",
        linewidth=1.35,
        zorder=2,
    )
    for xpos, score in zip(x, data["Normalized score"]):
        ax.text(xpos, float(score) + 0.06, f"{score:.2f}", ha="center", va="bottom", fontsize=22.0, fontweight="bold", color="#3F454B")
    ax.set_xticks(x)
    ax.set_xticklabels(
        data["Display label"],
        fontsize=21.0,
        fontweight="bold",
        rotation=18,
        ha="center",
        rotation_mode="anchor",
    )
    ax.tick_params(axis="x", pad=30)
    ax.set_xlim(-0.55, len(data) - 0.45)
    ymax = max(3.55, float(data["Normalized score"].max()) + 0.48)
    minimum_contribution = float(data[["Normalized voltage contribution", "Normalized efficiency contribution"]].min().min())
    ymin = minimum_contribution - 0.10 if minimum_contribution < 0 else 0.0
    ax.set_ylim(ymin, ymax)
    ax.axhline(0.0, color=NAVY, linewidth=1.0)
    ax.grid(True, axis="y", linestyle=":", color=GRID, linewidth=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=22.0, width=1.5, length=6)
    for tick_label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        tick_label.set_fontweight("bold")
    ax.set_ylabel("Normalized score" if show_ylabel else "", fontsize=25.0, fontweight="bold")
    ax.text(
        0.018,
        0.980,
        f"{current_density:g} mA cm$^{{-2}}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=23.5,
        fontweight="bold",
        color="#232A30",
        zorder=5,
        clip_on=False,
    )
    panel_label(ax, label)


def plot_candidate_scatter(
    ax: plt.Axes,
    feasible: pd.DataFrame,
    target: pd.Series,
    current_density: float,
    label: str,
) -> None:
    x_col = "Pred_AvgDischargeVoltage_mV"
    y_col = "Pred_AnodicEfficiency_pct"

    plot_data = (
        feasible.groupby([x_col, y_col], as_index=False)
        .agg(
            Score_zV_plus_zE=("Score_zV_plus_zE", "mean"),
            Candidate_count=("Score_zV_plus_zE", "size"),
        )
    )
    score = plot_data["Score_zV_plus_zE"].to_numpy(dtype=float)
    finite_score = score[np.isfinite(score)]
    score_norm = Normalize(
        vmin=float(np.nanpercentile(finite_score, 2)),
        vmax=float(np.nanpercentile(finite_score, 98)),
        clip=True,
    )
    scatter_cmap = LinearSegmentedColormap.from_list(
        "candidate_score",
        ["#DDEAF1", "#92BDD3", "#4A88B3", "#24496E"],
    )
    scatter = ax.scatter(
        plot_data[x_col],
        plot_data[y_col],
        s=22.0 + 15.0 * np.log1p(plot_data["Candidate_count"].to_numpy(dtype=float)),
        c=score,
        cmap=scatter_cmap,
        norm=score_norm,
        edgecolors="white",
        linewidths=0.30,
        alpha=0.48,
        rasterized=True,
        zorder=2,
    )

    target_x = float(target[x_col])
    target_y = float(target[y_col])
    ax.scatter(target_x, target_y, marker="*", s=980, color="white", edgecolor="white", linewidth=4.0, zorder=7)
    ax.scatter(target_x, target_y, marker="*", s=720, color=MAGENTA, edgecolor="#8F1745", linewidth=1.8, zorder=8)

    ax.set_xlabel("Predicted average discharge voltage (mV)", fontweight="bold")
    ax.set_ylabel("Predicted anodic efficiency (%)", fontweight="bold")
    ax.tick_params(axis="both", labelsize=22.0, width=1.5, length=6)
    for tick_label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        tick_label.set_fontweight("bold")
    ax.grid(True, linestyle=":", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    colorbar = plt.colorbar(scatter, ax=ax, fraction=0.047, pad=0.025)
    colorbar.set_label("Normalized score", fontsize=23.0, fontweight="bold")
    colorbar.ax.tick_params(labelsize=21.0, width=1.4)
    for tick_label in colorbar.ax.get_yticklabels():
        tick_label.set_fontweight("bold")
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="", markerfacecolor="#6FA5C4", markeredgecolor="white", alpha=0.65, markersize=11, label="Screened candidates"),
            Line2D([0], [0], marker="*", linestyle="", markerfacecolor=MAGENTA, markeredgecolor="#A3194B", markersize=19, label="Selected candidate"),
        ],
        loc="lower left",
        prop={"family": "Arial", "size": 24.0, "weight": "bold"},
        frameon=False,
    )
    ax.text(
        0.025,
        0.975,
        f"{current_density:g} mA cm$^{{-2}}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=24.0,
        fontweight="bold",
        color=NAVY,
        zorder=6,
    )
    panel_label(ax, label, x=-0.17, y=1.08)


def plot_parallel(
    ax: plt.Axes,
    family: pd.DataFrame,
    target: pd.Series,
    current_density: float,
    label: str,
) -> None:
    family_elements = str(target["AlloySystem"]).split("-")
    axes = [
        *[(element, f"{element}\n(wt%)") for element in family_elements],
        ("sum_selected_additions_wt%", "Total additions\n(wt%)"),
        ("HT", "HT\n(°C)"),
        ("Ht", "Ht\n(h)"),
        ("ET", "ET\n(°C)"),
        ("ES", "ES\n(mm s$^{-1}$)"),
        ("ER", "ER"),
        ("Score_zV_plus_zE", "Score"),
    ]
    columns = [column for column, _ in axes]
    labels = [label for _, label in axes]
    x = np.arange(len(columns), dtype=float)

    ranked = family.sort_values("Score_zV_plus_zE", ascending=False).copy()
    ranked["_score_key"] = ranked["Score_zV_plus_zE"].round(6)
    ranked = ranked.drop_duplicates(subset=["_score_key"]).head(100).drop(columns=["_score_key"])
    combined = pd.concat([ranked, pd.DataFrame([target])], ignore_index=True)
    combined = combined.drop_duplicates(subset=columns)

    lower = combined[columns].min(axis=0).astype(float)
    upper = combined[columns].max(axis=0).astype(float)
    for column, floor, ceiling in [
        ("HT", 400.0, 450.0),
        ("Ht", 6.0, 24.0),
        ("ET", 250.0, 400.0),
        ("ES", 0.5, 2.5),
        ("ER", 30.0, 45.0),
    ]:
        lower[column] = floor
        upper[column] = ceiling
    span = (upper - lower).replace(0.0, 1.0)

    normalized = (combined[columns] - lower) / span
    display_values = normalized.to_numpy(dtype=float, copy=True)
    jitter_rng = np.random.default_rng(SEED + 17 + int(round(current_density * 10)))
    jitter_scale = np.array([0.010, 0.010, 0.010, 0.012, 0.020, 0.020, 0.020, 0.020, 0.020, 0.007])
    jitter = jitter_rng.uniform(-1.0, 1.0, size=display_values.shape) * jitter_scale
    jitter = np.where(display_values <= 1e-12, np.abs(jitter), jitter)
    jitter = np.where(display_values >= 1.0 - 1e-12, -np.abs(jitter), jitter)
    display_values = np.clip(display_values + jitter, 0.0, 1.0)
    pathway_cmap = LinearSegmentedColormap.from_list(
        "pathway_score",
        ["#DDEAF2", "#9FC1D6", "#5D91B5", "#244E72"],
    )
    pathway_scores = combined["Score_zV_plus_zE"].to_numpy(dtype=float)
    pathway_norm = Normalize(
        vmin=float(np.min(pathway_scores)),
        vmax=float(np.max(pathway_scores)),
        clip=True,
    )
    for row_idx in np.argsort(pathway_scores):
        score = pathway_scores[int(row_idx)]
        ax.plot(
            x,
            display_values[int(row_idx)],
            color=pathway_cmap(pathway_norm(score)),
            alpha=0.72,
            linewidth=1.20,
            zorder=1,
        )

    target_norm = ((target[columns].astype(float) - lower) / span).to_numpy(dtype=float)
    ax.plot(x, target_norm, color=YELLOW, linewidth=6.5, linestyle="--", marker="o", markersize=14.5, markeredgecolor=NAVY, markeredgewidth=1.5, zorder=5)

    selected_labels = []
    for column in columns:
        value = float(target[column])
        if column == "Score_zV_plus_zE":
            selected_labels.append(f"{value:.2f}")
        else:
            selected_labels.append(compact_composition_value(value))

    for idx, (value, text_value) in enumerate(zip(target_norm, selected_labels)):
        if value >= 0.72:
            offset_y, va = -23, "top"
        elif value <= 0.28:
            offset_y, va = 23, "bottom"
        else:
            offset_y, va = (23, "bottom") if idx % 2 == 0 else (-23, "top")
        offset_x = -5 if idx % 2 == 0 else 5
        ax.annotate(
            text_value,
            xy=(x[idx], value),
            xytext=(offset_x, offset_y),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=18.5,
            fontweight="bold",
            color=NAVY,
            bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": "none", "alpha": 0.90},
            zorder=7,
            clip_on=False,
        )

    for idx, column in enumerate(columns):
        ax.axvline(idx, color="#5E6770", linewidth=1.35, zorder=0)
        low_label = f"{lower[column]:.2f}" if column == "Score_zV_plus_zE" else f"{lower[column]:g}"
        high_label = f"{upper[column]:.2f}" if column == "Score_zV_plus_zE" else f"{upper[column]:g}"
        ax.text(idx, -0.075, low_label, ha="center", va="top", fontsize=20.5, fontweight="bold", color="#4E5963")
        ax.text(idx, 1.025, high_label, ha="center", va="bottom", fontsize=20.5, fontweight="bold", color="#4E5963")

    ax.set_xlim(-0.25, len(columns) - 0.75)
    ax.set_ylim(-0.14, 1.14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=22.5, fontweight="bold")
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(
        handles=[
            Line2D([0], [0], color=pathway_cmap(0.10), linewidth=3.0, label="Lower composite score"),
            Line2D([0], [0], color=pathway_cmap(0.95), linewidth=3.0, label="Higher composite score"),
            Line2D([0], [0], color=YELLOW, linewidth=6.2, linestyle="--", marker="o", markeredgecolor=NAVY, label="Selected pathway"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.055),
        ncol=3,
        prop={"family": "Arial", "size": 24.0, "weight": "bold"},
        frameon=False,
    )
    panel_label(ax, label, x=-0.08, y=1.08)


def export_source_data(
    screening_results: dict[float, dict[str, object]],
    ranking_panels: dict[float, pd.DataFrame],
) -> None:
    path = FIGURE_DIR / "Figure5_screening_source_data.xlsx"
    settings_rows = []
    for current_density, result in screening_results.items():
        config = SEARCH_CONFIGS[current_density]
        settings_rows.append({
            "Current density": current_density,
            "Alloy class": "Mg + three nonzero additions",
            "Element pool": ", ".join(config["element_pool"]),
            "Composition step (wt%)": config["composition_step"],
            "Maximum additions (wt%)": config["maximum_additions"],
            "Representative sample size": len(result["sampled"]),
            "Selected system": result["target"]["AlloySystem"],
        })
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for current_density, result in screening_results.items():
            tag = f"{current_density:g}mA"
            result["feasible"].to_excel(writer, sheet_name=f"Feasible_{tag}", index=False)
            result["retained"].to_excel(writer, sheet_name=f"Retained_{tag}", index=False)
            pd.DataFrame([result["target"]]).to_excel(writer, sheet_name=f"Selected_{tag}", index=False)
            result["family"].to_excel(writer, sheet_name=f"Family_{tag}", index=False)
        for current_density, panel_data in ranking_panels.items():
            panel_data.to_excel(writer, sheet_name=f"Ranking_{current_density:g}mA", index=False)
        pd.DataFrame(settings_rows).to_excel(writer, sheet_name="Search_space", index=False)


def main() -> None:
    warnings.filterwarnings("ignore")
    configure_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    ranking_panels = {
        2.5: prepare_ranking_panel(2.5, top_n=4),
        10.0: prepare_ranking_panel(10.0, top_n=4),
        40.0: prepare_ranking_panel(40.0, top_n=4),
    }

    numeric_cols = [str(column).strip() for column in joblib.load(MODEL_DIR / "numeric_cols_used.joblib")]
    model_v = joblib.load(MODEL_DIR / "stack_voltage.joblib")
    model_e = joblib.load(MODEL_DIR / "stack_efficiency.joblib")
    tokenizer, encoder, device = load_encoder()
    prediction_assets = (numeric_cols, model_v, model_e, tokenizer, encoder, device)

    screening_results: dict[float, dict[str, object]] = {}
    for current_density in [2.5, 10.0, 40.0]:
        config = SEARCH_CONFIGS[current_density]
        retained, target, mean_v, std_v, mean_e, std_e = load_retained_results(current_density)
        family_system = tuple(str(target["AlloySystem"]).split("-"))
        print(
            f"Sampling strict quaternary design space at {current_density:g} mA cm^-2...",
            flush=True,
        )
        sampled = build_representative_grid(
            current_density,
            family_system,
            ranking_panels[current_density],
        )
        sampled = predict_grid(sampled, current_density, config["element_pool"], prediction_assets)
        sampled = add_score(sampled, mean_v, std_v, mean_e, std_e)
        retained = add_score(retained, mean_v, std_v, mean_e, std_e)

        design_keys = ["AlloySystem", *config["element_pool"], "HT", "Ht", "ET", "ES", "ER"]
        feasible = pd.concat([sampled, retained], ignore_index=True, sort=False)
        feasible = feasible.drop_duplicates(subset=design_keys).reset_index(drop=True)

        target_mask = np.ones(len(feasible), dtype=bool)
        for column in ["AlloySystem", *config["element_pool"], "HT", "Ht", "ET", "ES", "ER"]:
            if column == "AlloySystem":
                target_mask &= feasible[column].eq(target[column]).to_numpy()
            else:
                target_mask &= np.isclose(feasible[column].astype(float), float(target[column]))
        if not np.any(target_mask):
            raise RuntimeError(f"Selected candidate was not retained at {current_density:g} mA cm^-2.")
        target = feasible.loc[target_mask].iloc[0].copy()
        family = feasible[feasible["AlloySystem"].eq(target["AlloySystem"])].copy()
        screening_results[current_density] = {
            "sampled": sampled,
            "retained": retained,
            "feasible": feasible,
            "target": target,
            "family": family,
        }

    export_source_data(screening_results, ranking_panels)

    fig = plt.figure(figsize=(26.0, 35.5))
    grid = GridSpec(
        4,
        6,
        figure=fig,
        height_ratios=[0.88, 1.08, 1.08, 1.08],
        left=0.055,
        right=0.985,
        bottom=0.035,
        top=0.970,
        hspace=0.48,
        wspace=0.38,
    )
    rank_axes = [
        fig.add_subplot(grid[0, 0:2]),
        fig.add_subplot(grid[0, 2:4]),
        fig.add_subplot(grid[0, 4:6]),
    ]
    for ax, current, label in zip(rank_axes, [2.5, 10.0, 40.0], ["a", "b", "c"]):
        plot_ranking_panel(ax, ranking_panels[current], current, label, show_ylabel=(label == "a"))

    panel_rows = [
        (1, 2.5, "d", "e"),
        (2, 10.0, "f", "g"),
        (3, 40.0, "h", "i"),
    ]
    for row_idx, current_density, scatter_label, parallel_label in panel_rows:
        result = screening_results[current_density]
        ax_scatter = fig.add_subplot(grid[row_idx, 0:2])
        ax_parallel = fig.add_subplot(grid[row_idx, 2:6])
        plot_candidate_scatter(
            ax_scatter,
            result["feasible"],
            result["target"],
            current_density,
            scatter_label,
        )
        plot_parallel(
            ax_parallel,
            result["family"],
            result["target"],
            current_density,
            parallel_label,
        )

    stem = FIGURE_DIR / "Figure5_screening_rankings"
    fig.savefig(stem.with_suffix(".png"), dpi=450, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    total_candidates = sum(len(result["feasible"]) for result in screening_results.values())
    print(f"Figure 5 completed with {total_candidates:,} screened candidates.", flush=True)


if __name__ == "__main__":
    main()
