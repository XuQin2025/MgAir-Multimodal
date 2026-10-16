# -*- coding: utf-8 -*-
"""Build the revised dataset-distribution composite for Figure 2."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, LogNorm, to_rgb


ROOT = Path(__file__).resolve().parent
RAWDATA_PATH = ROOT / "rawdata.xlsx"
FIGURE_DIR = ROOT / "paper_figures"

ELEMENTS = [
    "Al", "Ca", "Mn", "Zn", "Sn", "La", "Ce", "Er", "In", "Y", "Ga", "Gd",
    "Sm", "Ge", "Nd", "Li", "Sr", "Zr", "Fe", "Cu", "Ni", "Pb", "Bi", "Ag",
    "Si", "Co", "Ti", "Cd",
]

PROCESS_FEATURES = [
    ("Homogenization\nT", "均匀化温度（℃）"),
    ("Homogenization\ntime", "均匀化时间"),
    ("Extrusion\nT", "挤压温度"),
    ("Extrusion\nspeed", "挤压速度（mm/s）"),
    ("Extrusion\nratio", "挤压比"),
    ("Rolling\nT", "轧制温度"),
    ("Rolling\nreduction", "总轧制压下量%"),
    ("Annealing\nT", "退火温度"),
    ("Annealing\ntime", "退火时间"),
]

NAVY = "#24496E"
BLUE = "#5E9FC9"
TEAL = "#3FA7A0"
GOLD = "#EABF62"
ORANGE = "#EB8E52"
RED = "#C9574F"
GRID = "#D5DADF"


def configure_style() -> None:
    sns.set_theme(style="whitegrid", rc={"grid.color": GRID, "grid.linewidth": 0.8})
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.sans-serif": ["Arial"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "font.size": 18,
        "axes.titlesize": 25.5,
        "axes.labelsize": 25.0,
        "xtick.labelsize": 22.0,
        "ytick.labelsize": 22.0,
        "legend.fontsize": 18,
        "axes.linewidth": 1.15,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


def find_column(frame: pd.DataFrame, token: str) -> str:
    token_lower = token.lower()
    for column in frame.columns:
        if token_lower in str(column).lower():
            return str(column)
    raise KeyError(f"Column containing {token!r} was not found.")


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.08,
        f"({label})",
        transform=ax.transAxes,
        fontsize=28,
        fontweight="bold",
        ha="left",
        va="top",
        clip_on=False,
    )


def style_cartesian(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.9)
        spine.set_color("#2B3137")
    ax.grid(True, linestyle=":", linewidth=0.85, color=GRID)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", which="major", labelsize=22.0, width=1.6, length=6)
    ax.title.set_fontsize(25.5)
    ax.title.set_fontweight("bold")
    ax.xaxis.label.set_size(25.0)
    ax.xaxis.label.set_weight("bold")
    ax.yaxis.label.set_size(25.0)
    ax.yaxis.label.set_weight("bold")
    for tick_label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        tick_label.set_fontweight("bold")


def plot_grouped_annular_rose(
    ax: plt.Axes,
    value_map: dict[str, float],
    groups: list[tuple[str, list[str], str]],
    panel: str,
    center_label: str,
    center_color: str,
) -> None:
    values = np.asarray([value_map[label] for _, labels, _ in groups for label in labels], dtype=float)
    maximum = max(float(values.max()), 1.0)
    group_gap = 0.18
    n_bars = len(values)
    bar_width = (2 * np.pi - group_gap * len(groups)) / n_bars
    inner_radius = 0.27
    inner_band = 0.13
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    # Keep only a narrow outer margin so the rose charts fill their panels.
    ax.set_ylim(0, 1.10)
    ax.set_xticks([])
    ax.set_yticks([0.45, 0.65, 0.85])
    ax.set_yticklabels([])
    ax.grid(True, axis="y", color="#AEB8BF", linestyle=(0, (4, 4)), linewidth=0.8)
    ax.spines["polar"].set_visible(False)

    cursor = 0.0
    for group_name, labels, group_color in groups:
        group_angles = []
        for feature_index, feature in enumerate(labels):
            value = float(value_map[feature])
            angle = cursor + bar_width / 2
            group_angles.append(angle)
            normalized_height = 0.19 + 0.54 * np.sqrt(value / maximum)
            outer_height = max(normalized_height - inner_band, 0.02)
            base_rgb = np.asarray(to_rgb(group_color), dtype=float)
            color_strength = 0.42 + 0.58 * np.sqrt(value / maximum)
            value_color = np.ones(3) * (1.0 - color_strength) + base_rgb * color_strength
            ax.bar(
                angle,
                inner_band,
                width=bar_width * 0.88,
                bottom=inner_radius,
                color=TEAL,
                edgecolor="white",
                linewidth=0.9,
                zorder=3,
            )
            ax.bar(
                angle,
                outer_height,
                width=bar_width * 0.88,
                bottom=inner_radius + inner_band,
                color=value_color,
                edgecolor="white",
                linewidth=1.0,
                zorder=3,
            )
            tip_radius = inner_radius + normalized_height
            rotation = 90.0 - np.degrees(angle)
            if rotation < -90:
                rotation += 180
            if rotation > 90:
                rotation -= 180
            short_feature_labels = {
                "Homogenization\nT": "HT",
                "Homogenization\ntime": "Ht",
                "Extrusion\nT": "ET",
                "Extrusion\nspeed": "ES",
                "Extrusion\nratio": "ER",
                "Rolling\nT": "RT",
                "Rolling\nreduction": "RR",
                "Annealing\nT": "AT",
                "Annealing\ntime": "At",
            }
            clean_feature = short_feature_labels.get(feature, feature.replace("\n", " "))
            if feature == "Others":
                label_radius = 1.03
            else:
                label_radius = min(tip_radius + 0.055, 0.97) + (0.060 if feature_index % 2 else 0.0)
            ax.text(
                angle,
                label_radius,
                clean_feature,
                ha="center",
                va="center",
                rotation=rotation,
                rotation_mode="anchor",
                fontsize=21.0,
                fontweight="bold",
                color="#28333C",
                zorder=5,
            )
            ax.text(
                angle,
                inner_radius + inner_band + outer_height * 0.50,
                f"{int(value)}",
                ha="center",
                va="center",
                fontsize=19.5,
                fontweight="bold",
                color="#5B3A24" if group_color in {GOLD, ORANGE} else "#263947",
                zorder=5,
            )
            cursor += bar_width

        cursor += group_gap

    center_theta = np.linspace(0, 2 * np.pi, 720)
    ax.fill_between(center_theta, 0, inner_radius, color=center_color, edgecolor="none", zorder=6)
    ax.plot(center_theta, np.full_like(center_theta, inner_radius), color="#25313A", linewidth=1.0, zorder=7)
    ax.text(0, 0, center_label, ha="center", va="center", fontsize=26.0, fontweight="bold", color=RED, zorder=7)
    panel_label(ax, panel)


def bubble_sizes(total_additions: pd.Series) -> np.ndarray:
    clipped = total_additions.clip(lower=0)
    ceiling = max(float(clipped.quantile(0.97)), 1.0)
    return 34.0 + 175.0 * np.sqrt(np.minimum(clipped, ceiling) / ceiling)


def main() -> None:
    configure_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    raw = pd.read_excel(RAWDATA_PATH)
    current_col = find_column(raw, "current density")
    voltage_col = find_column(raw, "average discharge voltage")
    efficiency_col = find_column(raw, "anodic efficiency")
    element_cols = [column for column in ELEMENTS if column in raw.columns]

    work = raw.dropna(subset=[current_col, voltage_col, efficiency_col]).copy()
    work[element_cols] = work[element_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    work[current_col] = pd.to_numeric(work[current_col], errors="coerce")
    work[voltage_col] = pd.to_numeric(work[voltage_col], errors="coerce")
    work[efficiency_col] = pd.to_numeric(work[efficiency_col], errors="coerce")
    work = work.dropna(subset=[current_col, voltage_col, efficiency_col])
    work["Total additions (wt%)"] = work[element_cols].sum(axis=1)

    current_counts = work[current_col].round(3).value_counts().sort_index()
    all_element_frequency = (work[element_cols] > 1e-12).sum().sort_values(ascending=False)
    element_frequency = all_element_frequency.head(12).copy()
    element_frequency.loc["Others"] = int(all_element_frequency.iloc[12:].sum())
    process_coverage = pd.Series({
        label: int((pd.to_numeric(work[column], errors="coerce").fillna(0.0) > 0).sum())
        for label, column in PROCESS_FEATURES
    })
    process_columns = dict(PROCESS_FEATURES)
    rolling_route = (
        pd.to_numeric(work[process_columns["Rolling\nT"]], errors="coerce").fillna(0.0).gt(0)
        | pd.to_numeric(work[process_columns["Rolling\nreduction"]], errors="coerce").fillna(0.0).gt(0)
    )
    rolling_route_count = int(rolling_route.sum())
    process_coverage.loc["Rolling\nT"] = rolling_route_count
    process_coverage.loc["Rolling\nreduction"] = rolling_route_count

    current_cmap = LinearSegmentedColormap.from_list(
        "current_density",
        [NAVY, BLUE, TEAL, GOLD, ORANGE, RED],
    )
    positive_current = work.loc[work[current_col] > 0, current_col]
    current_norm = LogNorm(vmin=float(positive_current.min()), vmax=float(positive_current.max()))

    fig = plt.figure(figsize=(25.5, 17.5))
    grid = fig.add_gridspec(
        2,
        3,
        left=0.055,
        right=0.965,
        bottom=0.065,
        top=0.915,
        hspace=0.56,
        wspace=0.29,
        height_ratios=[0.90, 1.10],
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[0, 2])
    ax_d = fig.add_subplot(grid[1, 0], projection="polar")
    ax_e = fig.add_subplot(grid[1, 1], projection="polar")
    ax_f = fig.add_subplot(grid[1, 2])

    x = np.arange(len(current_counts))
    ax_a.bar(
        x,
        current_counts.values,
        width=0.78,
        color=TEAL,
        edgecolor="white",
        linewidth=1.2,
        alpha=0.96,
    )
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(
        [f"{value:g}" for value in current_counts.index],
        rotation=55,
        ha="right",
        fontsize=19.5,
        fontweight="bold",
    )
    ax_a.set_xlabel("Current density (mA cm$^{-2}$)", fontweight="bold")
    ax_a.xaxis.set_label_coords(0.5, -0.20)
    ax_a.set_ylabel("Count", fontweight="bold")
    style_cartesian(ax_a)
    panel_label(ax_a, "a")

    sns.histplot(work[voltage_col], bins=22, color=BLUE, edgecolor="white", linewidth=0.9, kde=False, ax=ax_b)
    ax_b.set_xlabel("Average discharge voltage (mV)", fontweight="bold")
    ax_b.xaxis.set_label_coords(0.5, -0.20)
    ax_b.set_ylabel("Count", fontweight="bold")
    style_cartesian(ax_b)
    panel_label(ax_b, "b")

    sns.histplot(work[efficiency_col], bins=22, color=ORANGE, edgecolor="white", linewidth=0.9, kde=False, ax=ax_c)
    ax_c.set_xlabel("Anodic efficiency (%)", fontweight="bold")
    ax_c.xaxis.set_label_coords(0.5, -0.20)
    ax_c.set_ylabel("Count", fontweight="bold")
    style_cartesian(ax_c)
    panel_label(ax_c, "c")

    composition_groups = [
        ("Elements", element_frequency.index.tolist(), BLUE),
    ]
    plot_grouped_annular_rose(
        ax_d,
        element_frequency.astype(float).to_dict(),
        composition_groups,
        "d",
        "Elements",
        "#F4FAF9",
    )
    process_groups = [
        ("Homogenization", ["Homogenization\nT", "Homogenization\ntime"], BLUE),
        ("Extrusion", ["Extrusion\nT", "Extrusion\nspeed", "Extrusion\nratio"], GOLD),
        ("Rolling", ["Rolling\nT", "Rolling\nreduction"], ORANGE),
        ("Annealing", ["Annealing\nT", "Annealing\ntime"], RED),
    ]
    plot_grouped_annular_rose(
        ax_e,
        process_coverage.astype(float).to_dict(),
        process_groups,
        "e",
        "Process",
        "#FFF0DD",
    )

    sizes = bubble_sizes(work["Total additions (wt%)"])
    scatter = ax_f.scatter(
        work[voltage_col],
        work[efficiency_col],
        s=sizes,
        c=work[current_col],
        cmap=current_cmap,
        norm=current_norm,
        alpha=0.66,
        edgecolor="white",
        linewidth=0.45,
        rasterized=True,
    )
    ax_f.set_xlabel("Average discharge voltage (mV)", fontweight="bold")
    ax_f.set_ylabel("Anodic efficiency (%)", fontweight="bold")
    style_cartesian(ax_f)
    colorbar = fig.colorbar(scatter, ax=ax_f, fraction=0.050, pad=0.025)
    colorbar.set_label("Current density (mA cm$^{-2}$)", fontsize=22.5, fontweight="bold")
    colorbar.ax.tick_params(labelsize=20.0)
    for tick_label in colorbar.ax.get_yticklabels():
        tick_label.set_fontweight("bold")

    panel_label(ax_f, "f")

    fig.suptitle(
        "Distribution of the Mg-air battery alloy dataset",
        fontsize=30,
        fontweight="bold",
        y=0.973,
    )

    stem = FIGURE_DIR / "Figure2_dataset_distribution"
    fig.savefig(stem.with_suffix(".png"), dpi=450, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)

    with pd.ExcelWriter(FIGURE_DIR / "Figure2_dataset_distribution_source_data.xlsx") as writer:
        current_counts.rename_axis("Current density (mA cm^-2)").reset_index(name="Count").to_excel(
            writer, sheet_name="Current_density", index=False
        )
        element_frequency.rename_axis("Element").reset_index(name="Occurrence count").to_excel(
            writer, sheet_name="Composition_rose", index=False
        )
        process_coverage.rename_axis("Process descriptor").reset_index(name="Record count").to_excel(
            writer, sheet_name="Processing_rose", index=False
        )
        work[[current_col, voltage_col, efficiency_col, "Total additions (wt%)"]].to_excel(
            writer, sheet_name="All_current_bubbles", index=False
        )

    print(f"Figure 2 completed with {len(work):,} records and {current_counts.size} current-density levels.")


if __name__ == "__main__":
    main()
