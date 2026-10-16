# -*- coding: utf-8 -*-
"""Export Figure 5 panels (a-i) as independent publication files."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from build_figure5_quaternary_screening import (
    configure_style,
    plot_candidate_scatter,
    plot_parallel,
    plot_ranking_panel,
)


ROOT = Path(__file__).resolve().parent
SOURCE_PATH = ROOT / "paper_figures" / "Figure5_screening_source_data.xlsx"
OUTPUT_DIR = ROOT / "paper_figures" / "Figure5_subfigures"


def save_panel(fig: plt.Figure, stem: str) -> None:
    output_stem = OUTPUT_DIR / stem
    fig.savefig(Path(f"{output_stem}.png"), dpi=450, bbox_inches="tight", facecolor="white")
    fig.savefig(Path(f"{output_stem}.pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(Path(f"{output_stem}.svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def load_panel_data(current_density: float) -> dict[str, object]:
    tag = f"{current_density:g}mA"
    return {
        "ranking": pd.read_excel(SOURCE_PATH, sheet_name=f"Ranking_{tag}"),
        "feasible": pd.read_excel(SOURCE_PATH, sheet_name=f"Feasible_{tag}"),
        "target": pd.read_excel(SOURCE_PATH, sheet_name=f"Selected_{tag}").iloc[0],
        "family": pd.read_excel(SOURCE_PATH, sheet_name=f"Family_{tag}"),
    }


def export_ranking_panel(current_density: float, label: str, data: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.4, 7.2))
    plot_ranking_panel(ax, data, current_density, label, show_ylabel=True)
    fig.subplots_adjust(left=0.16, right=0.97, bottom=0.24, top=0.92)
    save_panel(fig, f"Figure5{label}_ranking_{current_density:g}mA")


def export_scatter_panel(current_density: float, label: str, data: dict[str, object]) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 7.4))
    plot_candidate_scatter(ax, data["feasible"], data["target"], current_density, label)
    fig.subplots_adjust(left=0.16, right=0.92, bottom=0.15, top=0.92)
    save_panel(fig, f"Figure5{label}_candidates_{current_density:g}mA")


def export_parallel_panel(current_density: float, label: str, data: dict[str, object]) -> None:
    fig, ax = plt.subplots(figsize=(17.6, 7.4))
    plot_parallel(ax, data["family"], data["target"], current_density, label)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.16, top=0.86)
    save_panel(fig, f"Figure5{label}_pathways_{current_density:g}mA")


def main() -> None:
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(f"Figure 5 source data not found: {SOURCE_PATH}")

    configure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    panel_specs = [
        (2.5, "a", "d", "e"),
        (10.0, "b", "f", "g"),
        (40.0, "c", "h", "i"),
    ]
    for current_density, ranking_label, scatter_label, parallel_label in panel_specs:
        data = load_panel_data(current_density)
        export_ranking_panel(current_density, ranking_label, data["ranking"])
        export_scatter_panel(current_density, scatter_label, data)
        export_parallel_panel(current_density, parallel_label, data)
        print(f"Exported Figure 5 panels at {current_density:g} mA cm^-2.", flush=True)

    print(f"Figure 5 subfigures saved to: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
