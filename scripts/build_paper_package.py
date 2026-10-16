# -*- coding: utf-8 -*-
from __future__ import annotations

import ast
import hashlib
import math
import shutil
import textwrap
import zipfile
from html import escape as xml_escape
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent.parent
RAWDATA_PATH = ROOT / "rawdata.xlsx"
CLASSICAL_METRICS_PATH = ROOT / "model_evaluation_cv_results.xlsx"
GBSTACK_DIR = ROOT / "outputs_GBstack_hpo"
GBSTACK_METRICS_PATH = GBSTACK_DIR / "GBstack_HPO_metrics.xlsx"
TOP5_SUMMARY_PATH = ROOT / "top5_alloys_summary.xlsx"
SHAP_SUMMARY_PATH = ROOT / "shap_rankings_summary.xlsx"
SEARCH_DIR = ROOT / "outputs_design_search_multiCD_selected"
FIGURE_DIR = ROOT / "paper_figures"
TEXT_DIR = ROOT / "paper_manuscript"

SEED = 42
OUTER_SPLITS = 5
INNER_SPLITS = 5

ELEMENT_ORDER = [
    "Al", "Ca", "Mn", "Zn", "Sn", "La", "Ce", "Er", "In", "Y", "Ga", "Gd",
    "Sm", "Ge", "Nd", "Li", "Sr", "Zr", "Fe", "Cu", "Ni", "Pb", "Bi", "Ag",
    "Si", "Co", "Ti", "Cd",
]

SELECTED_POOLS = {
    2.5: ["Al", "Ca", "Zn", "Sn", "Y", "Sr"],
    10.0: ["Al", "Ca", "Mn", "Zn", "La", "Gd", "In"],
    40.0: ["Al", "Ca", "Mn", "Zn", "Er", "Y", "La", "Sm", "Gd", "Li"],
}

MAX_CONTENT_LIMITS = {
    2.5: 2.06,
    10.0: 10.43,
    40.0: 11.97,
}

VALIDATION_SHORTLIST = [
    {
        "Current density (mA cm^-2)": 2.5,
        "Alloy system": "Y-Sr",
        "Composition (wt%)": "Sr 0.3, Y 1.2",
        "HT (degC)": 450,
        "Ht (h)": 12,
        "ET (degC)": 400,
        "ES (mm s^-1)": 0.5,
        "ER": 45,
        "Predicted voltage (mV)": 1533.75,
        "Predicted efficiency (%)": 52.47,
        "Score_zV_plus_zE": 2.534,
        "Validation priority": "Primary ternary candidate under ultra-low current density",
    },
    {
        "Current density (mA cm^-2)": 10.0,
        "Alloy system": "Zn-In",
        "Composition (wt%)": "Zn 1.5, In 1.5",
        "HT (degC)": 450,
        "Ht (h)": 12,
        "ET (degC)": 400,
        "ES (mm s^-1)": 0.5,
        "ER": 45,
        "Predicted voltage (mV)": 1349.84,
        "Predicted efficiency (%)": 68.38,
        "Score_zV_plus_zE": 2.499,
        "Validation priority": "Balanced ternary candidate under medium current density",
    },
    {
        "Current density (mA cm^-2)": 40.0,
        "Alloy system": "Al-Li",
        "Composition (wt%)": "Al 8.0, Li 0.8",
        "HT (degC)": 450,
        "Ht (h)": 12,
        "ET (degC)": 400,
        "ES (mm s^-1)": 2.5,
        "ER": 45,
        "Predicted voltage (mV)": 1133.71,
        "Predicted efficiency (%)": 71.45,
        "Score_zV_plus_zE": 2.456,
        "Validation priority": "High-rate ternary candidate with strong efficiency retention",
    },
]

MODEL_NAME_MAP = {
    "GB": "GB",
    "DT": "DT",
    "MLP": "MLP",
    "KNN": "KNN",
    "BERT_GBstack_HPO": "BERT-GBStack-HPO",
    "BatteryBERT_GBStack_HPO": "BatteryBERT-GBStack-HPO",
    "BatteryBERT_GBstack_HPO": "BatteryBERT-GBStack-HPO",
}

EMBEDDING_SOURCE_MAP = {
    "BERT_GBstack_HPO": "bert-base-uncased",
    "BatteryBERT_GBstack_HPO": "batterydata/batterybert-uncased",
}

MODEL_COLORS = {
    "GB": "#2C5F2D",
    "DT": "#9B5DE5",
    "MLP": "#00A6A6",
    "KNN": "#F77F00",
    "BERT-GBStack-HPO": "#3A86FF",
    "BatteryBERT-GBStack-HPO": "#D62828",
}

MODEL_SHORT_LABELS = {
    "GB": "GB",
    "DT": "DT",
    "MLP": "MLP",
    "KNN": "KNN",
    "BERT-GBStack-HPO": "BERT",
    "BatteryBERT-GBStack-HPO": "Battery\nBERT",
}

OOF_CACHE: Dict[Tuple[str, str], Dict[str, object]] = {}

PRETTY_FEATURE_MAP = {
    "Current Density": "Current density",
    "Homogenization Temp": "Homogenization temp",
    "Homogenization Time": "Homogenization time",
    "Extrusion Temp": "Extrusion temp",
    "Extrusion Speed": "Extrusion speed",
    "Extrusion Ratio": "Extrusion ratio",
    "Rolling Temp": "Rolling temp",
    "Rolling Reduction": "Rolling reduction",
    "Annealing Temp": "Annealing temp",
    "Annealing Time": "Annealing time",
}

TERNARY_TARGET_SELECTION = {
    2.5: {"AlloySystem": "Y-Sr", "Sr": 0.3, "Y": 1.2, "HT": 450.0, "Ht": 12.0, "ET": 400.0, "ES": 0.5, "ER": 45.0},
    10.0: {"AlloySystem": "Zn-In", "Zn": 1.5, "In": 1.5, "HT": 450.0, "Ht": 12.0, "ET": 400.0, "ES": 0.5, "ER": 45.0},
    40.0: {"AlloySystem": "Al-Li", "Al": 8.0, "Li": 0.8, "HT": 450.0, "Ht": 12.0, "ET": 400.0, "ES": 2.5, "ER": 45.0},
}

FIGURE_THEME = {
    "navy": "#1D3557",
    "teal": "#2A9D8F",
    "gold": "#E9C46A",
    "orange": "#F4A261",
    "red": "#E76F51",
    "slate": "#6D6875",
    "light": "#F8F5F0",
}

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})
sns.set_theme(style="whitegrid")

TERNARY_FULL_FILE_MAP = {
    2.5: "grid_predictions_CD2.5_Ternary_full.csv",
    10.0: "grid_predictions_CD10_Ternary_full.csv",
    40.0: "grid_predictions_CD40_Ternary_full.csv",
}

QUATERNARY_TOP_FILE_MAP = {
    2.5: "grid_top_CD2.5_Ternary1.xlsx",
    10.0: "grid_top_CD10_Ternary1.xlsx",
    40.0: "grid_top_CD40_Ternary1.xlsx",
}


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def hash_key(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:10]


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def find_column(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    columns = list(df.columns)
    lower_map = {str(col).lower(): col for col in columns}
    for candidate in candidates:
        if candidate in columns:
            return candidate
    for candidate in candidates:
        target = candidate.lower()
        for lower_name, original in lower_map.items():
            if target in lower_name:
                return original
    raise KeyError(f"Cannot find any of {list(candidates)} in columns: {columns}")


def format_value(value: float) -> str:
    if math.isclose(value, round(value), rel_tol=0.0, abs_tol=1e-12):
        return str(int(round(value)))
    return f"{value:.3f}".rstrip("0").rstrip(".")


def candidate_signature(row: pd.Series, element_cols: List[str]) -> str:
    parts = []
    for element in element_cols:
        value = float(row.get(element, 0) or 0)
        if value > 1e-12:
            parts.append(f"{element} {format_value(value)}")
    return ", ".join(parts)


def compact_composition_label(row: pd.Series, element_cols: List[str]) -> str:
    parts = []
    for element in element_cols:
        value = float(row.get(element, 0) or 0)
        if value > 1e-12:
            parts.append(f"{element}{format_value(value)}")
    return " | ".join(parts)


def wrap_bar_text(text: str, parts_per_line: int = 2) -> str:
    parts = [part.strip() for part in text.split("|") if part.strip()]
    if not parts:
        return text
    lines = []
    for i in range(0, len(parts), parts_per_line):
        lines.append(" | ".join(parts[i:i + parts_per_line]))
    return "\n".join(lines)


def row_matches_criteria(row: pd.Series, criteria: Dict[str, float]) -> bool:
    for key, expected in criteria.items():
        value = row.get(key)
        if isinstance(expected, str):
            if str(value) != expected:
                return False
        else:
            if not math.isclose(float(value), float(expected), rel_tol=0.0, abs_tol=1e-9):
                return False
    return True


def current_label(value: float) -> str:
    if math.isclose(value, 2.5, abs_tol=1e-9):
        return "2.5 mA cm$^{-2}$"
    if math.isclose(value, 10.0, abs_tol=1e-9):
        return "10 mA cm$^{-2}$"
    if math.isclose(value, 40.0, abs_tol=1e-9):
        return "40 mA cm$^{-2}$"
    return f"{format_value(value)} mA cm$^{{-2}}$"


def prettify_feature_name(name: str) -> str:
    return PRETTY_FEATURE_MAP.get(name, name)


def get_selected_workbook() -> Path:
    candidates = [
        path for path in SEARCH_DIR.glob("*.xlsx")
        if not path.name.startswith("grid_top_") and not path.name.startswith("~$")
    ]
    if not candidates:
        raise FileNotFoundError("Cannot find selected screening workbook in outputs_design_search_multiCD_selected.")
    return sorted(candidates)[0]


def build_basic_docx(docx_path: Path, paragraphs: List[str]) -> None:
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>
"""
    rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>
"""
    paragraph_xml = []
    for para in paragraphs:
        safe_para = xml_escape(para)
        paragraph_xml.append("<w:p><w:r><w:t xml:space=\"preserve\">" + safe_para + "</w:t></w:r></w:p>")
    document = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    {''.join(paragraph_xml)}
    <w:sectPr>
      <w:pgSz w:w="11906" w:h="16838"/>
      <w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="708" w:footer="708" w:gutter="0"/>
    </w:sectPr>
  </w:body>
</w:document>
"""
    with zipfile.ZipFile(docx_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", rels)
        archive.writestr("word/document.xml", document)


def load_clean_training_data() -> Tuple[pd.DataFrame, List[str], str, str]:
    clean_df = pd.read_excel(GBSTACK_METRICS_PATH, sheet_name="clean_data_with_text")
    numeric_cols = pd.read_excel(GBSTACK_METRICS_PATH, sheet_name="numeric_cols")["numeric_cols_used"].dropna().tolist()
    voltage_col = find_column(clean_df, ["Average discharge voltage/mV"])
    efficiency_col = find_column(clean_df, ["Anodic efficiency/%"])
    return clean_df, numeric_cols, voltage_col, efficiency_col


def get_cached_embedding(model_name: str) -> np.ndarray:
    source_name = EMBEDDING_SOURCE_MAP.get(model_name, model_name)
    model_hash = hash_key(source_name)
    matches = sorted((GBSTACK_DIR / "cache_embeddings").glob(f"emb_{model_hash}_*.npy"))
    if not matches:
        raise FileNotFoundError(f"Cannot find cached embedding for {model_name}.")
    return np.load(matches[0])


def reconstruct_oof_predictions(model_name: str, target_tag: str) -> Dict[str, float]:
    cache_key = (model_name, target_tag)
    if cache_key in OOF_CACHE:
        return dict(OOF_CACHE[cache_key]["metrics"])

    clean_df, numeric_cols, voltage_col, efficiency_col = load_clean_training_data()
    X_num = clean_df[numeric_cols].astype(float).to_numpy()
    X_emb = get_cached_embedding(model_name)

    if target_tag == "Voltage":
        target_col = voltage_col
    elif target_tag == "Efficiency":
        target_col = efficiency_col
    else:
        raise ValueError(f"Unsupported target tag: {target_tag}")

    y = clean_df[target_col].astype(float).to_numpy()
    params_path = GBSTACK_DIR / f"BestParams_{model_name}_{target_tag}.csv"
    params_df = pd.read_csv(params_path)

    outer_cv = KFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=SEED)
    oof_pred = np.zeros(len(y), dtype=float)

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_num), start=1):
        fold_params = params_df.loc[params_df["Fold"] == fold_idx].iloc[0]
        gb_params = ast.literal_eval(fold_params["best_gb_num"])
        meta_params = ast.literal_eval(fold_params["best_meta"])
        ridge_alpha = float(fold_params["best_ridge_alpha"])

        Xn_train, Xn_test = X_num[train_idx], X_num[test_idx]
        Xe_train, Xe_test = X_emb[train_idx], X_emb[test_idx]
        y_train = y[train_idx]

        inner_cv = KFold(n_splits=INNER_SPLITS, shuffle=True, random_state=SEED + fold_idx)
        meta_train = np.zeros((len(train_idx), 2), dtype=float)

        for inner_train_idx, inner_valid_idx in inner_cv.split(Xn_train):
            Xn_a, Xn_b = Xn_train[inner_train_idx], Xn_train[inner_valid_idx]
            Xe_a, Xe_b = Xe_train[inner_train_idx], Xe_train[inner_valid_idx]
            y_a = y_train[inner_train_idx]

            base_num = GradientBoostingRegressor(random_state=SEED + fold_idx, **gb_params)
            base_num.fit(Xn_a, y_a)

            base_txt = Pipeline([
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=ridge_alpha, random_state=SEED + fold_idx)),
            ])
            base_txt.fit(Xe_a, y_a)

            meta_train[inner_valid_idx, 0] = base_num.predict(Xn_b)
            meta_train[inner_valid_idx, 1] = base_txt.predict(Xe_b)

        meta_model = GradientBoostingRegressor(random_state=SEED + 200 + fold_idx, **meta_params)
        meta_model.fit(meta_train, y_train)

        full_num = GradientBoostingRegressor(random_state=SEED + fold_idx, **gb_params)
        full_num.fit(Xn_train, y_train)
        full_txt = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=ridge_alpha, random_state=SEED + fold_idx)),
        ])
        full_txt.fit(Xe_train, y_train)

        test_meta = np.column_stack([full_num.predict(Xn_test), full_txt.predict(Xe_test)])
        oof_pred[test_idx] = meta_model.predict(test_meta)

    metrics = {
        "R2": float(r2_score(y, oof_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y, oof_pred))),
        "MAE": float(mean_absolute_error(y, oof_pred)),
    }
    OOF_CACHE[cache_key] = {"metrics": metrics, "y_true": y.copy(), "y_pred": oof_pred.copy()}
    return dict(metrics)


def get_oof_bundle(model_name: str, target_tag: str) -> Dict[str, object]:
    cache_key = (model_name, target_tag)
    if cache_key not in OOF_CACHE:
        reconstruct_oof_predictions(model_name, target_tag)
    return OOF_CACHE[cache_key]


def compile_model_metrics() -> pd.DataFrame:
    classical_df = clean_columns(pd.read_excel(CLASSICAL_METRICS_PATH))
    classical_df["ModelName"] = classical_df["Model"].map(MODEL_NAME_MAP)
    classical_df = classical_df.rename(columns={"Avg R2": "R2", "Avg RMSE": "RMSE", "Avg MAE": "MAE"})
    classical_df = classical_df[["Target", "ModelName", "R2", "RMSE", "MAE"]]

    stored_bert_df = pd.read_excel(GBSTACK_METRICS_PATH, sheet_name="metrics_oof")
    reconstructed_rows = []
    for source_name in ["BERT_GBstack_HPO", "BatteryBERT_GBstack_HPO"]:
        for target_tag in ["Voltage", "Efficiency"]:
            metrics = reconstruct_oof_predictions(source_name, target_tag)
            reconstructed_rows.append({
                "Target": target_tag,
                "ModelName": MODEL_NAME_MAP[source_name],
                "R2": metrics["R2"],
                "RMSE": metrics["RMSE"],
                "MAE": metrics["MAE"],
            })

    bert_df = pd.DataFrame(reconstructed_rows)
    metrics_df = pd.concat([classical_df, bert_df], ignore_index=True)
    metrics_df["ModelName"] = pd.Categorical(
        metrics_df["ModelName"],
        categories=["GB", "DT", "MLP", "KNN", "BERT-GBStack-HPO", "BatteryBERT-GBStack-HPO"],
        ordered=True,
    )
    metrics_df = metrics_df.sort_values(["Target", "ModelName"]).reset_index(drop=True)

    stored_bert_df["ModelName"] = stored_bert_df["EmbeddingModel"].map(MODEL_NAME_MAP)
    stored_bert_df["Target"] = stored_bert_df["Target"].apply(lambda x: "Voltage" if "voltage" in x.lower() else "Efficiency")
    stored_bert_df = stored_bert_df[["Target", "ModelName", "R2", "RMSE"]]
    metrics_df = metrics_df.merge(stored_bert_df, on=["Target", "ModelName"], how="left", suffixes=("", "_stored"))
    metrics_df["R2"] = metrics_df["R2_stored"].fillna(metrics_df["R2"])
    metrics_df["RMSE"] = metrics_df["RMSE_stored"].fillna(metrics_df["RMSE"])
    metrics_df = metrics_df.drop(columns=["R2_stored", "RMSE_stored"])
    return metrics_df


def load_cached_model_metrics() -> pd.DataFrame:
    cached_table = TEXT_DIR / "Paper_source_tables.xlsx"
    if cached_table.exists():
        try:
            return pd.read_excel(cached_table, sheet_name="Figure3_model_metrics")
        except Exception:
            pass
    return compile_model_metrics()


def plot_dataset_distribution(raw_df: pd.DataFrame) -> pd.DataFrame:
    current_col = find_column(raw_df, ["current density/mA/cm2", "current density/mA/cm2,", "current density"])
    voltage_col = find_column(raw_df, ["Average discharge voltage/mV"])
    efficiency_col = find_column(raw_df, ["Anodic efficiency/%"])

    work_df = raw_df.copy()
    element_cols = [col for col in ELEMENT_ORDER if col in work_df.columns]
    work_df = work_df.dropna(subset=[current_col, voltage_col, efficiency_col]).copy()
    work_df["Total additions (wt%)"] = work_df[element_cols].fillna(0).sum(axis=1)

    main_currents = [2.5, 10.0, 40.0]
    palette = {2.5: FIGURE_THEME["navy"], 10.0: FIGURE_THEME["teal"], 40.0: FIGURE_THEME["red"]}

    fig = plt.figure(figsize=(17, 11), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, hspace=0.18, wspace=0.12)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])

    current_counts = work_df[current_col].round(2).value_counts().sort_index().reset_index()
    current_counts.columns = ["Current density", "Count"]
    ax1.bar(current_counts["Current density"].astype(str), current_counts["Count"], color=FIGURE_THEME["teal"], edgecolor="white")
    ax1.set_title("a  Current-density distribution")
    ax1.set_xlabel("Current density (mA cm$^{-2}$)")
    ax1.set_ylabel("Number of records")
    ax1.tick_params(axis="x", rotation=55)

    sns.histplot(work_df[voltage_col], bins=22, color=FIGURE_THEME["navy"], kde=True, ax=ax2)
    ax2.set_title("b  Voltage distribution")
    ax2.set_xlabel("Average discharge voltage (mV)")
    ax2.set_ylabel("Count")

    sns.histplot(work_df[efficiency_col], bins=22, color=FIGURE_THEME["orange"], kde=True, ax=ax3)
    ax3.set_title("c  Efficiency distribution")
    ax3.set_xlabel("Anodic efficiency (%)")
    ax3.set_ylabel("Count")

    sns.histplot(work_df["Total additions (wt%)"], bins=22, color=FIGURE_THEME["gold"], kde=True, ax=ax4)
    ax4.set_title("d  Total alloying-addition distribution")
    ax4.set_xlabel("Total alloy additions (wt%)")
    ax4.set_ylabel("Count")

    element_frequency = (work_df[element_cols].fillna(0) > 1e-12).sum().sort_values(ascending=False).head(12)
    ax5.barh(element_frequency.index[::-1], element_frequency.values[::-1], color=sns.color_palette("crest", n_colors=len(element_frequency)), edgecolor="white")
    ax5.set_title("e  Most frequent alloying elements")
    ax5.set_xlabel("Occurrence count")
    ax5.set_ylabel("")

    main_df = work_df[work_df[current_col].isin(main_currents)].copy()
    ax6.scatter(
        main_df[voltage_col],
        main_df[efficiency_col],
        s=35 + main_df["Total additions (wt%)"] * 12,
        c=main_df[current_col].map(palette),
        alpha=0.72,
        linewidth=0.35,
        edgecolor="white",
    )
    ax6.set_title("f  Voltage-efficiency landscape at key current densities")
    ax6.set_xlabel("Average discharge voltage (mV)")
    ax6.set_ylabel("Anodic efficiency (%)")
    handles = [mpl.lines.Line2D([0], [0], marker="o", linestyle="", markersize=8, markerfacecolor=palette[current], markeredgecolor="white", label=current_label(current)) for current in main_currents]
    ax6.legend(handles=handles, frameon=True, loc="lower left")

    fig.suptitle("Figure 2. Distribution of the Mg-air battery alloy dataset", fontsize=18, y=1.02)
    fig.savefig(FIGURE_DIR / "Figure2_dataset_distribution.png", dpi=350, bbox_inches="tight")
    plt.close(fig)

    return pd.DataFrame({
        "Statistic": [
            "Total valid records",
            "Distinct current-density levels",
            "Median voltage (mV)",
            "Median efficiency (%)",
            "Median total additions (wt%)",
        ],
        "Value": [
            len(work_df),
            work_df[current_col].nunique(),
            round(work_df[voltage_col].median(), 2),
            round(work_df[efficiency_col].median(), 2),
            round(work_df["Total additions (wt%)"].median(), 3),
        ],
    })


def make_single_metric_radar(
    target_df: pd.DataFrame,
    metric_col: str,
    title: str,
    output_path: Path,
    face_color: str,
    edge_color: str,
    tick_values: List[float],
    higher_is_better: bool = True,
    radial_power: float = 1.35,
) -> None:
    plot_df = target_df.copy().sort_values("ModelName")
    model_labels = [MODEL_SHORT_LABELS[str(name)] for name in plot_df["ModelName"]]
    raw_values = plot_df[metric_col].astype(float).to_numpy()
    tick_values = list(tick_values)

    raw_min = min(tick_values)
    raw_max = max(tick_values)

    if higher_is_better:
        transform = lambda v: (v - raw_min) / max(raw_max - raw_min, 1e-9)
    else:
        transform = lambda v: (raw_max - v) / max(raw_max - raw_min, 1e-9)

    normalized_values = np.clip(np.array([transform(v) for v in raw_values], dtype=float), 0.0, 1.0)
    radial_values = normalized_values ** radial_power

    tick_pairs = []
    for tick in tick_values:
        tick_norm = float(np.clip(transform(tick), 0.0, 1.0))
        tick_pairs.append((tick_norm ** radial_power, tick))
    tick_pairs = sorted(tick_pairs, key=lambda item: item[0])
    radial_ticks = [item[0] for item in tick_pairs]
    radial_tick_labels = [f"{item[1]:.3f}".rstrip("0").rstrip(".") for item in tick_pairs]

    angles = np.linspace(0, 2 * np.pi, len(model_labels), endpoint=False).tolist()
    angles += angles[:1]
    radar_curve = radial_values.tolist() + [radial_values[0]]

    fig, ax = plt.subplots(figsize=(7.4, 7.0), subplot_kw={"polar": True})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(model_labels, fontsize=18, fontweight="bold")
    ax.set_rlabel_position(0)
    ax.set_yticks(radial_ticks)
    ax.set_yticklabels(radial_tick_labels, fontsize=14, fontweight="bold")
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.grid(color="#D0D0D0", linewidth=2.0)
    ax.xaxis.grid(color="#D0D0D0", linewidth=2.0)
    ax.spines["polar"].set_color("#D0D0D0")
    ax.spines["polar"].set_linewidth(2.5)

    ax.plot(angles, radar_curve, color=edge_color, linewidth=4.3, solid_capstyle="round")
    ax.fill(angles, radar_curve, color=face_color, alpha=0.28)

    ax.set_title(title, fontsize=24, fontweight="bold", pad=22)
    fig.savefig(output_path, dpi=350, bbox_inches="tight")
    plt.close(fig)


def build_top10_shap_bar(target_tag: str, output_path: Path, bar_color: str) -> None:
    shap_df = clean_columns(pd.read_excel(SHAP_SUMMARY_PATH))
    target_df = shap_df[shap_df["Target"].astype(str).str.lower() == target_tag.lower()].copy()
    if target_df.empty:
        raise ValueError(f"No SHAP rows found for target {target_tag}.")

    top_df = (
        target_df.groupby("Feature", as_index=False)["Mean_Abs_SHAP"]
        .mean()
        .sort_values("Mean_Abs_SHAP", ascending=False)
        .head(10)
        .sort_values("Mean_Abs_SHAP", ascending=True)
    )
    top_df["Feature"] = top_df["Feature"].map(prettify_feature_name)

    fig, ax = plt.subplots(figsize=(8.2, 6.3))
    ax.set_facecolor("#FBFBFB")
    ax.grid(True, axis="x", linestyle=":", linewidth=1.0, color="#CDD3DB")
    ax.grid(False, axis="y")
    ax.barh(
        top_df["Feature"],
        top_df["Mean_Abs_SHAP"],
        color=bar_color,
        edgecolor="white",
        alpha=0.92,
        height=0.72,
    )
    ax.set_title(f"{target_tag} | SHAP top 10", fontsize=22, fontweight="bold", pad=10)
    ax.set_xlabel("Mean |SHAP value|", fontsize=15, fontweight="bold")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=13)
    fig.savefig(output_path, dpi=340, bbox_inches="tight")
    plt.close(fig)


def create_styled_parity_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    x_label: str,
    y_label: str,
    output_path: Path,
    point_color: str,
    fit_color: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    ax.set_facecolor("#FBFBFB")
    ax.grid(True, linestyle=":", linewidth=1.0, color="#CDD3DB")

    ax.scatter(
        y_true,
        y_pred,
        s=48,
        c=point_color,
        alpha=0.72,
        edgecolors="white",
        linewidths=0.8,
    )

    min_v = min(float(np.min(y_true)), float(np.min(y_pred)))
    max_v = max(float(np.max(y_true)), float(np.max(y_pred)))
    padding = (max_v - min_v) * 0.06
    x_line = np.linspace(min_v, max_v, 300)
    coef = np.polyfit(y_true, y_pred, 1)
    fit_line = coef[0] * x_line + coef[1]
    residual_std = float(np.std(y_pred - (coef[0] * y_true + coef[1])))

    ax.plot(x_line, x_line, linestyle="--", color="#333333", linewidth=2.0)
    ax.plot(x_line, fit_line, color=fit_color, linewidth=3.2)
    ax.fill_between(x_line, fit_line - 0.35 * residual_std, fit_line + 0.35 * residual_std, color=fit_color, alpha=0.14)

    r2_value = r2_score(y_true, y_pred)
    rmse_value = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    ax.text(
        0.055,
        0.92,
        f"$R^2 = {r2_value:.3f}$\n$RMSE = {rmse_value:.2f}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=15,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#BBBBBB", "alpha": 0.96},
    )

    ax.set_title(title, fontsize=22, fontweight="bold", pad=10)
    ax.set_xlabel(x_label, fontsize=15, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=15, fontweight="bold")
    ax.set_xlim(min_v - padding, max_v + padding)
    ax.set_ylim(min_v - padding, max_v + padding)
    fig.savefig(output_path, dpi=340, bbox_inches="tight")
    plt.close(fig)


def build_model_performance_figures(metrics_df: pd.DataFrame) -> pd.DataFrame:
    voltage_df = metrics_df[metrics_df["Target"] == "Voltage"].copy()
    efficiency_df = metrics_df[metrics_df["Target"] == "Efficiency"].copy()

    make_single_metric_radar(
        voltage_df,
        metric_col="R2",
        title="Voltage | $R^2$",
        output_path=FIGURE_DIR / "Figure3a_voltage_r2_radar.png",
        face_color="#B9D8F0",
        edge_color="#417CB8",
        tick_values=[0.55, 0.63, 0.72, 0.80, 0.85, 0.88],
        higher_is_better=True,
        radial_power=1.42,
    )
    make_single_metric_radar(
        efficiency_df,
        metric_col="R2",
        title="Efficiency | $R^2$",
        output_path=FIGURE_DIR / "Figure3b_efficiency_r2_radar.png",
        face_color="#F7D2C8",
        edge_color="#C75A43",
        tick_values=[0.40, 0.48, 0.58, 0.68, 0.76, 0.82],
        higher_is_better=True,
        radial_power=1.42,
    )
    make_single_metric_radar(
        voltage_df,
        metric_col="RMSE",
        title="Voltage | RMSE",
        output_path=FIGURE_DIR / "Figure3c_voltage_rmse_radar.png",
        face_color="#D7EAD9",
        edge_color="#4C956C",
        tick_values=[100, 110, 125, 145, 170, 190],
        higher_is_better=False,
        radial_power=1.35,
    )
    make_single_metric_radar(
        efficiency_df,
        metric_col="RMSE",
        title="Efficiency | RMSE",
        output_path=FIGURE_DIR / "Figure3d_efficiency_rmse_radar.png",
        face_color="#F7E8C2",
        edge_color="#C99A1A",
        tick_values=[5.5, 6.0, 6.8, 7.6, 8.6, 9.5],
        higher_is_better=False,
        radial_power=1.35,
    )

    voltage_bundle = get_oof_bundle("BatteryBERT_GBstack_HPO", "Voltage")
    efficiency_bundle = get_oof_bundle("BatteryBERT_GBstack_HPO", "Efficiency")
    create_styled_parity_plot(
        voltage_bundle["y_true"],
        voltage_bundle["y_pred"],
        "Voltage - BatteryBERT-GBStack-HPO",
        "Measured Voltage (mV)",
        "Predicted Voltage (mV)",
        FIGURE_DIR / "Figure3e_batterybert_voltage_parity.png",
        point_color="#8DBBE5",
        fit_color="#FF6B6B",
    )
    create_styled_parity_plot(
        efficiency_bundle["y_true"],
        efficiency_bundle["y_pred"],
        "Efficiency - BatteryBERT-GBStack-HPO",
        "Measured Efficiency (%)",
        "Predicted Efficiency (%)",
        FIGURE_DIR / "Figure3f_batterybert_efficiency_parity.png",
        point_color="#7FC8C9",
        fit_color="#F28E2B",
    )
    build_top10_shap_bar("Voltage", FIGURE_DIR / "Figure3g_voltage_shap_top10.png", "#6FA8DC")
    build_top10_shap_bar("Efficiency", FIGURE_DIR / "Figure3h_efficiency_shap_top10.png", "#86C99E")

    composite_paths = [
        FIGURE_DIR / "Figure3a_voltage_r2_radar.png",
        FIGURE_DIR / "Figure3b_efficiency_r2_radar.png",
        FIGURE_DIR / "Figure3c_voltage_rmse_radar.png",
        FIGURE_DIR / "Figure3d_efficiency_rmse_radar.png",
        FIGURE_DIR / "Figure3e_batterybert_voltage_parity.png",
        FIGURE_DIR / "Figure3f_batterybert_efficiency_parity.png",
        FIGURE_DIR / "Figure3g_voltage_shap_top10.png",
        FIGURE_DIR / "Figure3h_efficiency_shap_top10.png",
    ]
    labels = list("abcdefgh")

    fig, axes = plt.subplots(2, 4, figsize=(29, 14))
    for ax, path, label in zip(axes.flat, composite_paths, labels):
        image = plt.imread(path)
        ax.imshow(image)
        ax.axis("off")
        ax.text(
            0.015,
            0.975,
            f"({label})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=18,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.88},
        )
    fig.suptitle("Figure 3. Model comparison, parity performance, and SHAP interpretation", fontsize=20, y=0.99)
    fig.savefig(FIGURE_DIR / "Figure3_model_performance_composite.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return metrics_df.copy()


def build_element_selection_figure(top5_df: pd.DataFrame) -> pd.DataFrame:
    top5_df = top5_df.copy()
    top5_df["Screened_Current_Density"] = top5_df["Screened_Current_Density"].astype(float)
    relevant_elements = []
    for pool in SELECTED_POOLS.values():
        relevant_elements.extend(pool)
    relevant_elements = list(dict.fromkeys(relevant_elements))

    rows = []
    for current, pool in SELECTED_POOLS.items():
        sub_df = top5_df[np.isclose(top5_df["Screened_Current_Density"], current)].copy()
        for element in relevant_elements:
            presence = sub_df[element].fillna(0).astype(float) > 1e-12
            rows.append({
                "Current density": current,
                "Element": element,
                "Frequency": int(presence.sum()),
                "Mean content (wt%)": float(sub_df.loc[presence, element].mean()) if presence.any() else 0.0,
                "Mean score": float(sub_df.loc[presence, "Combined_Score"].mean()) if presence.any() else 0.0,
                "Selected pool": element in pool,
            })
    support_df = pd.DataFrame(rows)

    x_positions = {element: idx for idx, element in enumerate(relevant_elements)}
    y_positions = {2.5: 0, 10.0: 1, 40.0: 2}

    fig, ax = plt.subplots(figsize=(16, 7))
    support_df["x"] = support_df["Element"].map(x_positions)
    support_df["y"] = support_df["Current density"].map(y_positions)
    scatter = ax.scatter(
        support_df["x"],
        support_df["y"],
        s=140 + support_df["Frequency"] * 250,
        c=support_df["Mean content (wt%)"],
        cmap="YlOrRd",
        alpha=np.where(support_df["Frequency"] > 0, 0.92, 0.18),
        edgecolors=np.where(support_df["Selected pool"], FIGURE_THEME["navy"], "white"),
        linewidths=np.where(support_df["Selected pool"], 2.2, 0.8),
    )

    ax.set_xticks(range(len(relevant_elements)))
    ax.set_xticklabels(relevant_elements)
    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels([current_label(current) for current in y_positions])
    ax.set_xlim(-0.75, len(relevant_elements) - 0.25)
    ax.set_ylim(-0.6, len(y_positions) - 0.4)
    ax.grid(axis="x", alpha=0.18)
    ax.grid(axis="y", alpha=0.12)
    ax.set_title("Figure 4. Element-pool determination from the experimental top-5 alloys", fontsize=17, pad=16)
    ax.set_xlabel("Alloying element")
    ax.set_ylabel("Current density")

    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Mean content among top-5 alloys (wt%)")
    fig.savefig(FIGURE_DIR / "Figure4_element_selection_map.png", dpi=350, bbox_inches="tight")
    plt.close(fig)
    return support_df.drop(columns=["x", "y"])


def select_distinct_score_rows(
    df: pd.DataFrame,
    top_n: int,
    preferred_criteria: Dict[str, float] | None = None,
) -> pd.DataFrame:
    selected_indices: List[int] = []
    used_scores = set()

    if preferred_criteria is not None:
        preferred_df = df[df.apply(lambda row: row_matches_criteria(row, preferred_criteria), axis=1)]
        if not preferred_df.empty:
            preferred_row = preferred_df.iloc[0]
            selected_indices.append(int(preferred_row.name))
            used_scores.add(round(float(preferred_row["Score_zV_plus_zE"]), 12))

    for idx, row in df.iterrows():
        score_key = round(float(row["Score_zV_plus_zE"]), 12)
        if idx in selected_indices or score_key in used_scores:
            continue
        selected_indices.append(int(idx))
        used_scores.add(score_key)
        if len(selected_indices) >= top_n:
            break

    selected_df = df.loc[selected_indices].copy()
    return selected_df.sort_values("Score_zV_plus_zE", ascending=False).head(top_n)


def minmax_scale_to_50(values: pd.Series) -> pd.Series:
    series = pd.Series(values, dtype=float)
    min_value = float(series.min())
    max_value = float(series.max())
    if math.isclose(min_value, max_value, rel_tol=0.0, abs_tol=1e-12):
        return pd.Series(np.full(len(series), 25.0), index=series.index, dtype=float)
    return 50.0 * (series - min_value) / (max_value - min_value)


def format_alloy_system_label(alloy_system: str) -> str:
    parts = ["Mg"] + [part for part in str(alloy_system).split("-") if part]
    if len(parts) <= 3:
        return "-".join(parts)
    split_idx = math.ceil(len(parts) / 2)
    return "-".join(parts[:split_idx]) + "\n" + "-".join(parts[split_idx:])


def load_screening_panel_source(family: str, current_density: float) -> pd.DataFrame:
    if family == "Ternary":
        file_path = SEARCH_DIR / TERNARY_FULL_FILE_MAP[current_density]
        return clean_columns(pd.read_csv(file_path))

    file_path = SEARCH_DIR / QUATERNARY_TOP_FILE_MAP[current_density]
    frames = [
        clean_columns(pd.read_excel(file_path, sheet_name=sheet_name))
        for sheet_name in ["top_by_score", "top_by_voltage", "top_by_efficiency"]
    ]
    return pd.concat(frames, ignore_index=True).drop_duplicates().reset_index(drop=True)


def prepare_screening_panel(family: str, current_density: float, top_n: int = 5) -> pd.DataFrame:
    df = load_screening_panel_source(family, current_density)
    meta_cols = {
        "AlloySystem",
        "HT",
        "Ht",
        "ET",
        "ES",
        "ER",
        "current_density",
        "sum_selected_additions_wt%",
        "Pred_AvgDischargeVoltage_mV",
        "Pred_AnodicEfficiency_pct",
        "Score_zV_plus_zE",
    }
    element_cols = [col for col in df.columns if col not in meta_cols]
    required_positive = 2 if family == "Ternary" else 3
    positive_counts = (df[element_cols].fillna(0).astype(float) > 1e-12).sum(axis=1)

    panel_df = df.loc[positive_counts == required_positive].copy()
    panel_df = panel_df.dropna(subset=["AlloySystem", "Pred_AvgDischargeVoltage_mV", "Pred_AnodicEfficiency_pct"]).copy()
    panel_df["sum_selected_additions_wt%"] = panel_df["sum_selected_additions_wt%"].astype(float)
    panel_df = panel_df[panel_df["sum_selected_additions_wt%"] > 0].copy()

    panel_df["Normalized voltage"] = minmax_scale_to_50(panel_df["Pred_AvgDischargeVoltage_mV"])
    panel_df["Normalized efficiency"] = minmax_scale_to_50(panel_df["Pred_AnodicEfficiency_pct"])
    panel_df["Display score"] = panel_df["Normalized voltage"] + panel_df["Normalized efficiency"]
    panel_df["Composition label"] = panel_df.apply(lambda row: candidate_signature(row, element_cols), axis=1)
    panel_df = panel_df.sort_values(
        ["Display score", "Pred_AnodicEfficiency_pct", "Pred_AvgDischargeVoltage_mV"],
        ascending=False,
    )
    panel_df = panel_df.drop_duplicates(subset=["AlloySystem"], keep="first").head(top_n).copy()
    panel_df = panel_df.sort_values("Display score", ascending=False).reset_index(drop=True)
    panel_df["Display label"] = panel_df["AlloySystem"].map(format_alloy_system_label)
    panel_df["Family"] = family
    panel_df["Current density (mA cm^-2)"] = current_density
    return panel_df


def draw_screening_score_panel(
    ax: mpl.axes.Axes,
    panel_df: pd.DataFrame,
    title: str,
    panel_label: str,
    show_ylabel: bool,
) -> None:
    x = np.arange(len(panel_df))
    width = 0.66
    voltage_color = "#7DB7D8"
    efficiency_color = "#F4A261"

    ax.bar(
        x,
        panel_df["Normalized voltage"],
        width=width,
        color=voltage_color,
        edgecolor="white",
        linewidth=1.2,
        label="Normalized voltage",
    )
    ax.bar(
        x,
        panel_df["Normalized efficiency"],
        width=width,
        bottom=panel_df["Normalized voltage"],
        color=efficiency_color,
        edgecolor="white",
        linewidth=1.2,
        label="Normalized efficiency",
    )

    for xpos, total_score in zip(x, panel_df["Display score"]):
        ax.text(
            xpos,
            float(total_score) + 1.2,
            f"{float(total_score):.1f}",
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
            color="#444444",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(panel_df["Display label"], fontsize=11, fontweight="bold")
    ax.tick_params(axis="y", labelsize=11)
    ax.set_ylim(0, 103)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=8)
    ax.grid(axis="y", alpha=0.24)
    ax.set_axisbelow(True)
    ax.text(0.02, 0.97, f"({panel_label})", transform=ax.transAxes, ha="left", va="top", fontsize=15, fontweight="bold")
    if show_ylabel:
        ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    else:
        ax.set_ylabel("")


def build_screening_ranking_figure(selected_workbook: Path | None = None) -> pd.DataFrame:
    panel_specs = [
        ("Ternary", 2.5, "a"),
        ("Ternary", 10.0, "b"),
        ("Ternary", 40.0, "c"),
        ("Quaternary", 2.5, "d"),
        ("Quaternary", 10.0, "e"),
        ("Quaternary", 40.0, "f"),
    ]

    ranking_rows = []
    fig, axes = plt.subplots(2, 3, figsize=(22, 13.5), sharey=True)

    for ax, (family, current_density, panel_label) in zip(axes.flat, panel_specs):
        panel_df = prepare_screening_panel(family, current_density, top_n=5)
        draw_screening_score_panel(
            ax,
            panel_df,
            title=f"{family} | {current_label(current_density)}",
            panel_label=panel_label,
            show_ylabel=ax in {axes[0, 0], axes[1, 0]},
        )
        for _, row in panel_df.iterrows():
            ranking_rows.append({
                "Family": family,
                "Current density (mA cm^-2)": float(current_density),
                "Alloy system": str(row["AlloySystem"]),
                "Best composition": str(row["Composition label"]),
                "Predicted voltage (mV)": float(row["Pred_AvgDischargeVoltage_mV"]),
                "Predicted efficiency (%)": float(row["Pred_AnodicEfficiency_pct"]),
                "Normalized voltage": float(row["Normalized voltage"]),
                "Normalized efficiency": float(row["Normalized efficiency"]),
                "Display score": float(row["Display score"]),
            })

    legend_handles = [
        mpl.patches.Patch(facecolor="#7DB7D8", edgecolor="white", label="Normalized voltage"),
        mpl.patches.Patch(facecolor="#F4A261", edgecolor="white", label="Normalized efficiency"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=2, frameon=False, fontsize=13)
    fig.suptitle("Figure 5. Top ranked ternary and quaternary alloy systems under three current densities", fontsize=19, y=0.995)
    fig.savefig(FIGURE_DIR / "Figure5_screening_rankings.png", dpi=340, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(ranking_rows)


def write_nano_banana_prompt() -> None:
    paragraphs = [
        "Figure 1 prompt for Nano Banana architecture drawing",
        "",
        "Prompt (English):",
        "Create a publication-ready scientific workflow figure for a Chemical Engineering Journal paper on Mg-air battery alloy screening.",
        "The workflow must read from left to right as: Data collection -> Multimodal modeling -> Candidate alloy screening -> Experimental validation.",
        "Use a horizontal layout, white background, refined scientific infographic style, soft shadows, clean arrows, and CEJ-like editorial colors such as deep navy, teal, muted orange, and warm gray.",
        "Panel 1: Data collection. Keep the left-side database concept. Show structured experimental data including alloy composition, homogenization, extrusion, and current density. On the right side, replace generic 'Element set / process route' wording with one explicit example sentence: 'Mg-based alloy with Al, Ca, Mn. After homogenization and extrusion.'",
        "Panel 2: Multimodal modeling. Emphasize the fusion model. Show one structured branch and one text branch merging into one fusion block. Mention Gradient Boosting only once in the whole panel. Highlight late fusion / stacking as the visual center. Represent the two outputs using icons or small schematic symbols rather than long text labels, to indicate average discharge voltage and anodic efficiency.",
        "Panel 3: High-throughput search space. Rename this panel exactly as 'High-throughput search space'. Show a generic screening process from a large candidate pool through filtering/funneling into a few final candidate alloys. Do not label specific alloy names or compositions in this panel. The emphasis should be on search, filtering, and final recommendation.",
        "Panel 4: Experimental validation. Show the downstream validation route as casting, heat treatment, extrusion, and final battery-performance testing. Use clear process icons or mini schematics, and end with a battery-discharge test illustration.",
        "Typography should be clean and publication-grade. Avoid cartoons, dark backgrounds, excessive 3D rendering, and cluttered text. All labels should be in English.",
    ]
    build_basic_docx(FIGURE_DIR / "Figure1_Nano_Banana_prompt.docx", paragraphs)
    (FIGURE_DIR / "Figure1_Nano_Banana_prompt.txt").write_text("\n".join(paragraphs), encoding="utf-8")


def build_manuscript(model_metrics: pd.DataFrame, dataset_stats: pd.DataFrame) -> None:
    voltage_best = model_metrics[(model_metrics["Target"] == "Voltage") & (model_metrics["ModelName"] == "BatteryBERT-GBStack-HPO")].iloc[0]
    efficiency_best = model_metrics[(model_metrics["Target"] == "Efficiency") & (model_metrics["ModelName"] == "BatteryBERT-GBStack-HPO")].iloc[0]
    n_records = int(dataset_stats.loc[dataset_stats["Statistic"] == "Total valid records", "Value"].iloc[0])
    raw_df = clean_columns(pd.read_excel(RAWDATA_PATH))
    current_col = find_column(raw_df, ["current density/mA/cm2", "current density/mA/cm2,", "current density"])
    voltage_col = find_column(raw_df, ["Average discharge voltage/mV"])
    efficiency_col = find_column(raw_df, ["Anodic efficiency/%"])
    valid_df = raw_df.dropna(subset=[current_col, voltage_col, efficiency_col]).copy()
    current_min = float(valid_df[current_col].min())
    current_max = float(valid_df[current_col].max())
    voltage_min = float(valid_df[voltage_col].min())
    voltage_max = float(valid_df[voltage_col].max())
    efficiency_min = float(valid_df[efficiency_col].min())
    efficiency_max = float(valid_df[efficiency_col].max())

    manuscript = textwrap.dedent(f"""
    2. Materials and methods
    2.1 Overall workflow
    Fig. 1 illustrates the proposed BatteryBERT-guided Mg-alloy design workflow for Mg-air battery anodes. The whole framework contains four linked stages: data collection, multimodal fusion modeling, high-throughput screening, and experimental validation. First, a structured Mg-air battery database was established from literature-derived alloy records containing composition, processing, current density, average discharge voltage, and anodic efficiency. At the same time, each record was converted into a standardized process-route sentence, for example, "Mg-based alloy with Al, Ca, Mn. After homogenization and extrusion." Second, a multimodal fusion surrogate model was built by combining structured numeric descriptors with BatteryBERT text embeddings. In this design, the numeric branch captures the direct quantitative mapping from composition/process variables to electrochemical response, whereas the text branch encodes alloy-system and process-route priors in semantic space. The two branches were then integrated by a stacking meta-learner to output the final predictions of average discharge voltage and anodic efficiency. Third, the trained surrogate was used to perform current-density-specific high-throughput screening in constrained ternary and quaternary Mg-alloy spaces. Candidate compositions and process windows were generated on discrete grids and ranked by a dual-objective score that balances voltage and efficiency. Finally, the top recommended alloys were forwarded to experimental verification through casting, heat treatment, extrusion, and Mg-air battery testing. In this way, Fig. 1 summarizes a complete data-driven loop that connects representation, prediction, screening, and validation.

    2.2 Dataset construction
    This study systematically searched the Web of Science Core Collection database using the keyword "Mg-air battery" and compiled a literature dataset covering the period from 2010 to 2025. To guarantee data consistency, the following criteria were applied during data extraction and cleaning: (1) alloy records from as-cast, homogenized, extruded, rolled, or annealed conditions were retained, whereas catalyst-only activation strategies and records lacking alloy descriptors were excluded; (2) each entry had to report the alloy-element type/content together with electrochemical outputs, especially average discharge voltage and anodic efficiency, under an identifiable current density; and (3) duplicated, incomplete, or internally inconsistent records were removed. After cleaning, {n_records} qualified Mg-alloy entries were retained for model construction. The final table contains elemental compositions, process variables, current density, average discharge voltage, and anodic efficiency. Figure 2 summarizes the dataset distribution: the current density spans {current_min:.3f} to {current_max:.0f} mA cm^-2, the discharge voltage spans {voltage_min:.0f} to {voltage_max:.0f} mV, and the anodic efficiency spans {efficiency_min:.1f} to {efficiency_max:.1f}%. These broad distributions provide a suitable basis for both surrogate-model training and current-density-specific screening.

    2.3 Descriptor definition
    Two complementary descriptor types were constructed for each alloy record. The first is the structured numeric descriptor x_num, which contains 39 variables. Among them, 29 variables describe the elemental composition, namely Mg, Al, Ca, Mn, Zn, Sn, La, Ce, Er, In, Y, Ga, Gd, Sm, Ge, Nd, Li, Sr, Zr, Fe, Cu, Ni, Pb, Bi, Ag, Si, Co, Ti, and Cd. The remaining 10 variables describe the processing and operating conditions, including homogenization temperature, homogenization time, extrusion temperature, extrusion speed, extrusion ratio, rolling temperature, total rolling reduction, annealing temperature, annealing time, and current density. The model outputs are the two target properties y = [V, eta], where V denotes the average discharge voltage and eta denotes the anodic efficiency.

    The second descriptor is a text feature x_text that summarizes the alloy system and process route in natural language. Different from the numeric branch, the text branch intentionally excludes explicit numerical values and retains only element identity and process order, so that semantic prior information can be learned without duplicating continuous variables already present in x_num. A representative template is "Mg-based alloy with Al, Ca, Mn. After homogenization and extrusion." In this work, BatteryBERT was directly used as a pretrained battery-domain language encoder rather than retrained from scratch. Therefore, the text descriptor mainly serves as a domain-informed semantic supplement to the structured descriptors.

    2.4 Multimodal fusion model
    2.4.1 Numeric branch
    For each target k in [V, eta], the structured input x_num was mapped to a numeric prediction by a Gradient Boosting regressor:
    yhat_num^(k) = f_num^(k)(x_num).
    The numeric branch uses the full set of composition, processing, and current-density descriptors as inputs. Gradient boosting was selected because it can flexibly model nonlinear interactions between alloy chemistry and processing variables while remaining stable on medium-sized tabular datasets. During inner-loop optimization, the main hyperparameters of the numeric branch, including n_estimators, learning_rate, max_depth, subsample, min_samples_split, min_samples_leaf, and max_features, were tuned by randomized cross-validation search.

    2.4.2 Text branch
    The text description x_text was tokenized by the pretrained BatteryBERT tokenizer and encoded into contextual hidden states H = [h_1, ..., h_L], where L is the token length. Given the corresponding attention mask a_i, a mean-pooling operator was used to obtain the sentence embedding:
    e_text = (sum_i a_i h_i) / (sum_i a_i).
    The pooled embedding e_text was then passed to a Ridge regressor to generate the text-branch prediction for each target:
    yhat_text^(k) = w_k^T e_text + b_k.
    This branch captures correlations between alloy-system terminology, process-route wording, and electrochemical behavior that may not be fully represented by the numeric vector alone. In parallel, a generic BERT-based text branch was also retained as a benchmark, allowing direct comparison between general-language and battery-domain semantic encoders.

    2.4.3 Fusion branch
    A late-fusion stacking strategy was adopted to combine the two branches. For each target, the numeric-branch output and text-branch output were first concatenated into a two-dimensional meta-feature:
    z^(k) = [yhat_num^(k), yhat_text^(k)].
    A Gradient Boosting meta-learner then produced the final prediction:
    yhat^(k) = g^(k)(z^(k)).
    Separate stacked models were trained for voltage and efficiency. Compared with early fusion of the raw embedding vector and structured descriptors, this late-fusion design reduces the risk of high-dimensional noise dominating the structured signal, while keeping the contribution of each branch transparent and easy to interpret.

    2.4.4 Model evaluation and validation
    Model development followed a leakage-safe nested cross-validation framework. The outer loop used five folds to obtain out-of-fold generalization performance, and the inner loop was used for hyperparameter optimization of the numeric branch, the Ridge regularization parameter in the text branch, and the meta-learner. For each outer fold m, the dataset D was split into a training subset D_train^(m) and a held-out test subset D_test^(m). Inner-fold predictions from the two base learners were used to train the stacking model, and the fully retrained base learners were then applied to the outer-fold test set. This procedure prevents the meta-learner from seeing target information from the held-out fold during training.

    The predictive performance was quantified by the coefficient of determination (R2), root-mean-square error (RMSE), and mean absolute error (MAE):
    R2 = 1 - sum_i (y_i - yhat_i)^2 / sum_i (y_i - ybar)^2,
    RMSE = sqrt[(1/n) sum_i (y_i - yhat_i)^2],
    MAE = (1/n) sum_i |y_i - yhat_i|.
    Here, y_i is the measured value, yhat_i is the predicted value, ybar is the mean of the measured values, and n is the number of evaluated samples. In addition to the two multimodal stacking models (BERT-GBStack-HPO and BatteryBERT-GBStack-HPO), four conventional machine-learning baselines, namely Gradient Boosting, Decision Tree, K-nearest neighbors, and multilayer perceptron, were trained for comparison using the same cleaned dataset.

    2.5 High-throughput screening space
    The screening stage was performed separately at 2.5, 10, and 40 mA cm^-2, corresponding to low-current, medium-current, and high-current discharge regimes, respectively. This separation is necessary because the dominant degradation mode and the preferred alloying strategy vary strongly with discharge rate. Candidate element pools were first determined from experimentally strong alloys at the same current density, and then two search families were constructed: a ternary search (Mg + 2 alloying elements) and a quaternary search (Mg + 3 alloying elements). The current-density-specific element pools were [{", ".join(SELECTED_POOLS[2.5])}] for 2.5 mA cm^-2, [{", ".join(SELECTED_POOLS[10.0])}] for 10 mA cm^-2, and [{", ".join(SELECTED_POOLS[40.0])}] for 40 mA cm^-2. The total addition limits were constrained to {MAX_CONTENT_LIMITS[2.5]:.2f}, {MAX_CONTENT_LIMITS[10.0]:.2f}, and {MAX_CONTENT_LIMITS[40.0]:.2f} wt%, respectively. The composition step sizes used in the screening scripts were 0.1 wt% at 2.5 mA cm^-2, 0.5 wt% at 10 mA cm^-2, and 0.8 wt% at 40 mA cm^-2. For a given current density cd, the composition constraint can be written as:
    sum_j c_j <= Cmax,cd,
    where c_j is the content of each selected alloying element and Cmax,cd is the corresponding current-density-specific upper bound.

    The process space was discretized into a fixed grid shared by all current densities:
    HT in [400, 450] degC,
    Ht in [6, 12, 24] h,
    ET in [250, 300, 350, 400] degC,
    ES in [0.5, 1.5, 2.5] mm s^-1,
    ER in [30, 45].
    This yields 2 x 3 x 4 x 3 x 2 = 144 process combinations for each composition candidate. Every generated candidate was converted into both a structured numeric vector and a standardized text sentence, and then evaluated by the trained BatteryBERT-GBStack-HPO surrogate. To keep the screening computationally tractable at the million-scale level, streaming prediction was used and the candidate tables were processed in chunks.

    Because voltage and efficiency have different physical units and numerical scales, current-density-specific target normalization was introduced before final ranking. For candidate j in one screening set:
    zV_j = (Vhat_j - mu_V) / sigma_V,
    zE_j = (etahat_j - mu_E) / sigma_E,
    Score_j = 0.5 zV_j + 0.5 zE_j,
    where Vhat_j and etahat_j are the predicted voltage and efficiency, and mu_V, sigma_V, mu_E, and sigma_E are the mean and standard deviation of the predicted values within the same current-density search space. A larger Score_j indicates a better simultaneous balance between voltage and efficiency. For visualization in Figure 5, an additional min-max normalization was used to decompose the display score into voltage and efficiency contributions:
    Vnorm_j = (Vhat_j - Vmin) / (Vmax - Vmin),
    Enorm_j = (etahat_j - Emin) / (Emax - Emin),
    Sdisplay_j = 50 Vnorm_j + 50 Enorm_j.
    The min-max normalized score was only used for graphical display, whereas the z-score-based composite score was used for candidate ranking in the screening scripts. After ranking, only the highest-scoring composition/process route was retained within each alloy system for system-level comparison.

    2.6 Experimental validation
    The final shortlisted alloys will be validated experimentally following the route shown in Fig. 1. The selected Mg alloys are first prepared by casting or alloy melting, followed by homogenization/heat treatment and extrusion under the recommended processing window. Electrochemical tests will then be carried out in Mg-air battery cells at the corresponding target current densities to obtain the measured average discharge voltage and anodic efficiency. The measured results will be compared directly with the model predictions to evaluate the reliability of the screening framework. When needed, supplementary microstructural and phase analyses can be added to interpret the origin of the observed electrochemical behavior.

    3. Results and discussion
    3.1 Dataset landscape
    Figure 2 summarizes the cleaned Mg-air alloy database. The dataset spans multiple current densities and covers a broad distribution of discharge voltage, anodic efficiency, and total alloy additions. The most frequent alloying elements are concentrated in Al-, Zn-, rare-earth-, and alkaline-earth-containing systems, supporting the use of chemically informed element-pool selection before large-scale candidate generation.

    3.2 Predictive performance of conventional ML and multimodal stacking
    The predictive comparison between the four conventional machine-learning models and the two multimodal stacking models is shown in Figure 3. For voltage prediction, BatteryBERT-GBStack-HPO delivered the best overall performance with R^2 = {voltage_best['R2']:.3f}, RMSE = {voltage_best['RMSE']:.1f} mV, and MAE = {voltage_best['MAE']:.1f} mV. For efficiency prediction, BatteryBERT-GBStack-HPO likewise achieved the strongest generalization with R^2 = {efficiency_best['R2']:.3f}, RMSE = {efficiency_best['RMSE']:.2f}%, and MAE = {efficiency_best['MAE']:.2f}%. Compared with the best conventional baseline (GB), the multimodal approach clearly improved the efficiency task and provided a consistent gain for voltage, demonstrating that semantic alloy-system information is complementary to structured descriptors.

    The parity plots in Figure 3e and Figure 3f further confirm that the BatteryBERT-based model reproduces the experimental trends with limited systematic deviation across the main range of both targets. The benefit of the BatteryBERT branch is particularly relevant for data-sparse alloy systems, because the text encoder captures cross-system priors that are difficult to express through raw numeric composition vectors alone.

    3.3 Element-pool determination from experimentally strong alloys
    Figure 4 visualizes the occurrence frequency and mean content of alloying elements among the top-5 experimental alloys at each current density. At 2.5 mA cm^-2, the strongest signals concentrate on Zn, Sn, Y, and Sr, while Al and Ca remain chemically relevant pool members for low-content exploration. At 10 mA cm^-2, the pool broadens to Al, Ca, Mn, Zn, La, Gd, and In, reflecting a stronger contribution from medium-content rare-earth and post-transition-metal alloying. At 40 mA cm^-2, the pool further expands toward Al-, Li-, Er-, La-, Sm-, and Gd-containing systems, consistent with the need to stabilize high-rate discharge behavior.

    3.4 High-throughput screening and candidate ranking
    Figure 5 compares the ranked ternary and quaternary candidates after screening within the current-density-specific search spaces. Quaternary systems can achieve higher predicted composite scores in several work points, especially at 2.5 and 40 mA cm^-2, because the extra alloying degree of freedom enables simultaneous improvement of voltage and efficiency. Nevertheless, ternary systems remain highly attractive for the next experimental stage owing to their lower compositional complexity, lower synthesis burden, and more transparent mechanism analysis.

    At 2.5 mA cm^-2, the Y-Sr ternary family dominated the screened ternary candidates, while Zn-Sn-Sr led the quaternary list. At 10 mA cm^-2, Zn-In emerged as the most compelling ternary system, whereas Ca-Zn-In reached the top of the quaternary ranking. At 40 mA cm^-2, Al-Li represented the most practical ternary choice, while Al-Er-Gd and Al-La-Gd delivered the strongest quaternary predictions. These trends indicate that low-current operation favors fine rare-earth/alkaline-earth tuning, the medium-current regime benefits from Zn-In-based synergy, and high-current operation requires stronger alloying for rate capability and efficiency retention.

    3.5 Recommended alloys for experimental validation
    To keep the first validation stage focused and experimentally tractable, one ternary candidate was selected for each current density:
    - 2.5 mA cm^-2: Y-Sr (Sr 0.3 wt%, Y 1.2 wt%; HT 450 degC, Ht 12 h, ET 400 degC, ES 0.5 mm s^-1, ER 45), with a predicted voltage of 1533.75 mV and an anodic efficiency of 52.47%.
    - 10 mA cm^-2: Zn-In (Zn 1.5 wt%, In 1.5 wt%; HT 450 degC, Ht 12 h, ET 400 degC, ES 0.5 mm s^-1, ER 45), with a predicted voltage of 1349.84 mV and an anodic efficiency of 68.38%.
    - 40 mA cm^-2: Al-Li (Al 8.0 wt%, Li 0.8 wt%; HT 450 degC, Ht 12 h, ET 400 degC, ES 2.5 mm s^-1, ER 45), with a predicted voltage of 1133.71 mV and an anodic efficiency of 71.45%.

    These three alloys balance predictive performance, compositional simplicity, and the practical value of covering low-, medium-, and high-rate discharge regimes. Once electrochemical validation is completed, the corresponding experimental section can be appended to compare predicted and observed voltage/efficiency outcomes, together with microstructural analysis of the discharge morphology and corrosion products.
    """).strip()

    (TEXT_DIR / "CEJ_style_manuscript.md").write_text(manuscript, encoding="utf-8")
    build_basic_docx(TEXT_DIR / "CEJ_style_manuscript.docx", manuscript.splitlines())
    (TEXT_DIR / "Methods_Results_only.md").write_text(manuscript, encoding="utf-8")
    build_basic_docx(TEXT_DIR / "Methods_Results_only.docx", manuscript.splitlines())

    captions = textwrap.dedent("""
    Figure 2. Distribution of the Mg-air battery alloy dataset. (a) Record counts at different current densities. (b) Histogram of average discharge voltage. (c) Histogram of anodic efficiency. (d) Distribution of total alloying additions. (e) Frequency of the most common alloying elements. (f) Joint landscape of voltage and efficiency at the three key current densities used for screening.

    Figure 3. Predictive performance and interpretation of the six benchmarked models. (a) R2 radar chart for voltage prediction. (b) R2 radar chart for efficiency prediction. (c) RMSE radar chart for voltage prediction. (d) RMSE radar chart for efficiency prediction. (e) Out-of-fold parity plot of the BatteryBERT-GBStack-HPO model for voltage. (f) Out-of-fold parity plot of the BatteryBERT-GBStack-HPO model for efficiency. (g) SHAP top-10 feature ranking for voltage prediction. (h) SHAP top-10 feature ranking for efficiency prediction.

    Figure 4. Element-pool determination from the experimentally strongest alloys. Bubble size denotes the occurrence frequency among the top-5 experimental alloys at each current density, while bubble color denotes the mean content of the corresponding element. Dark outlines indicate the final candidate element pools used in the current-density-specific screening.

    Figure 5. Top ranked ternary and quaternary alloy systems under the three target current densities. Each panel shows the top five distinct alloy systems after selecting the highest-scoring composition within the same system. The stacked bars decompose the display score into normalized voltage and normalized efficiency contributions, and the total bar height represents the final composite score.
    """).strip()
    (TEXT_DIR / "Figure_captions.md").write_text(captions, encoding="utf-8")
    pd.DataFrame(VALIDATION_SHORTLIST).to_excel(TEXT_DIR / "Experimental_validation_shortlist.xlsx", index=False)


def build_chinese_results_section(model_metrics: pd.DataFrame, element_support: pd.DataFrame) -> None:
    voltage_metrics = model_metrics[model_metrics["Target"] == "Voltage"].copy()
    efficiency_metrics = model_metrics[model_metrics["Target"] == "Efficiency"].copy()

    battery_v = voltage_metrics[voltage_metrics["ModelName"] == "BatteryBERT-GBStack-HPO"].iloc[0]
    bert_v = voltage_metrics[voltage_metrics["ModelName"] == "BERT-GBStack-HPO"].iloc[0]
    gb_v = voltage_metrics[voltage_metrics["ModelName"] == "GB"].iloc[0]
    dt_v = voltage_metrics[voltage_metrics["ModelName"] == "DT"].iloc[0]
    mlp_v = voltage_metrics[voltage_metrics["ModelName"] == "MLP"].iloc[0]
    knn_v = voltage_metrics[voltage_metrics["ModelName"] == "KNN"].iloc[0]

    battery_e = efficiency_metrics[efficiency_metrics["ModelName"] == "BatteryBERT-GBStack-HPO"].iloc[0]
    bert_e = efficiency_metrics[efficiency_metrics["ModelName"] == "BERT-GBStack-HPO"].iloc[0]
    gb_e = efficiency_metrics[efficiency_metrics["ModelName"] == "GB"].iloc[0]
    dt_e = efficiency_metrics[efficiency_metrics["ModelName"] == "DT"].iloc[0]
    mlp_e = efficiency_metrics[efficiency_metrics["ModelName"] == "MLP"].iloc[0]
    knn_e = efficiency_metrics[efficiency_metrics["ModelName"] == "KNN"].iloc[0]

    voltage_rmse_gain = (float(gb_v["RMSE"]) - float(battery_v["RMSE"])) / float(gb_v["RMSE"]) * 100.0
    efficiency_rmse_gain = (float(gb_e["RMSE"]) - float(battery_e["RMSE"])) / float(gb_e["RMSE"]) * 100.0

    shap_df = clean_columns(pd.read_excel(SHAP_SUMMARY_PATH))
    voltage_shap = (
        shap_df[shap_df["Target"] == "Voltage"]
        .groupby("Feature", as_index=False)["Mean_Abs_SHAP"]
        .mean()
        .sort_values("Mean_Abs_SHAP", ascending=False)
        .head(10)
    )
    efficiency_shap = (
        shap_df[shap_df["Target"] == "Efficiency"]
        .groupby("Feature", as_index=False)["Mean_Abs_SHAP"]
        .mean()
        .sort_values("Mean_Abs_SHAP", ascending=False)
        .head(10)
    )

    def shap_sentence(df: pd.DataFrame) -> str:
        parts = []
        for _, row in df.iterrows():
            parts.append(f"{prettify_feature_name(str(row['Feature']))}（{float(row['Mean_Abs_SHAP']):.2f}）")
        return "、".join(parts)

    figure4_2p5 = element_support[element_support["Current density"] == 2.5].sort_values(["Frequency", "Mean content (wt%)"], ascending=[False, False])
    figure4_10 = element_support[element_support["Current density"] == 10.0].sort_values(["Frequency", "Mean content (wt%)"], ascending=[False, False])
    figure4_40 = element_support[element_support["Current density"] == 40.0].sort_values(["Frequency", "Mean content (wt%)"], ascending=[False, False])

    ternary_2p5 = prepare_screening_panel("Ternary", 2.5, top_n=5)
    ternary_10 = prepare_screening_panel("Ternary", 10.0, top_n=5)
    ternary_40 = prepare_screening_panel("Ternary", 40.0, top_n=5)
    quaternary_2p5 = prepare_screening_panel("Quaternary", 2.5, top_n=5)
    quaternary_10 = prepare_screening_panel("Quaternary", 10.0, top_n=5)
    quaternary_40 = prepare_screening_panel("Quaternary", 40.0, top_n=5)

    shortlist_2p5 = VALIDATION_SHORTLIST[0]
    shortlist_10 = VALIDATION_SHORTLIST[1]
    shortlist_40 = VALIDATION_SHORTLIST[2]

    results_cn = textwrap.dedent(f"""
    3. 结果与讨论
    3.1 模型预测性能与关键特征解释
    图3系统比较了四种传统机器学习模型（GB、DT、MLP、KNN）以及两种多模态堆叠模型（BERT-GBStack-HPO 和 BatteryBERT-GBStack-HPO）在平均放电电压和阳极效率预测中的表现。整体来看，多模态模型在两个目标上均表现出最优或次优性能，说明在结构化成分/工艺描述符之外，引入电池领域文本语义信息能够为镁空气电池负极合金预测提供稳定增益。

    对平均放电电压而言，图3a和图3c显示 BatteryBERT-GBStack-HPO 在六个模型中取得了最佳综合表现，其 R² 为 {battery_v['R2']:.3f}，RMSE 为 {battery_v['RMSE']:.2f} mV，MAE 为 {battery_v['MAE']:.2f} mV。BERT-GBStack-HPO 紧随其后，R² 为 {bert_v['R2']:.3f}，RMSE 为 {bert_v['RMSE']:.2f} mV。传统模型中，GB 和 MLP 的电压预测能力较为接近，分别达到 R² = {gb_v['R2']:.3f} 和 {mlp_v['R2']:.3f}，但其误差仍高于 BatteryBERT 融合模型；DT 的表现进一步下降，而 KNN 的电压预测最弱，仅有 R² = {knn_v['R2']:.3f}，RMSE 达 {knn_v['RMSE']:.2f} mV。与最佳传统基线 GB 相比，BatteryBERT-GBStack-HPO 的电压 RMSE 下降了 {voltage_rmse_gain:.1f}%，表明电池领域语义嵌入对于刻画放电电压的复杂非线性关系具有明确贡献。

    对阳极效率而言，图3b和图3d呈现出更明显的模型区分度。BatteryBERT-GBStack-HPO 取得 R² = {battery_e['R2']:.3f}、RMSE = {battery_e['RMSE']:.3f}% 和 MAE = {battery_e['MAE']:.3f}% 的最优结果，BERT-GBStack-HPO 以 R² = {bert_e['R2']:.3f}、RMSE = {bert_e['RMSE']:.3f}% 略次之。相比之下，传统 GB、MLP 和 DT 在效率预测中的性能明显下降，其中 GB 的 R² 为 {gb_e['R2']:.3f}，DT 和 KNN 分别仅为 {dt_e['R2']:.3f} 和 {knn_e['R2']:.3f}。这说明与电压相比，阳极效率对合金体系语义信息和跨体系知识迁移更为敏感。相较于 GB 基线，BatteryBERT-GBStack-HPO 的效率 RMSE 下降了 {efficiency_rmse_gain:.1f}%，进一步验证了多模态晚期融合策略在双目标预测中的有效性。

    图3e和图3f给出了 BatteryBERT-GBStack-HPO 在两个目标上的预测-实测一致性图（parity plot）。在电压任务中，大多数样本点沿 y = x 对角线分布，说明模型能够较好再现实验数据的整体变化趋势；仅在低电压区和部分极端高电压区出现略大的离散，反映出少数边界样本仍然存在一定误差。在效率任务中，散点同样集中分布在对角线附近，且拟合直线与理想线之间偏差较小，表明模型不仅在整体统计指标上占优，也具备较好的样本级一致性。结合图3a-d的指标雷达图和图3e-f的一致性图，可以认为 BatteryBERT-GBStack-HPO 是后续高通量筛选最可靠的代理模型。

    图3g和图3h进一步给出了模型的 SHAP 解释结果。从电压预测看，电流密度是绝对主导因素，其平均 |SHAP| 值达到 {float(voltage_shap.iloc[0]['Mean_Abs_SHAP']):.2f}，远高于其他变量；其后依次为 {shap_sentence(voltage_shap.iloc[1:])}。这一结果说明放电电压首先受工作电流密度控制，同时还受到 Al、Zn、Mn、Ca、Sm 等合金元素以及挤压比、均质化温度/时间、总轧制压下量等工艺参数的共同影响。从阳极效率预测看，电流密度同样排名第一，平均 |SHAP| 值为 {float(efficiency_shap.iloc[0]['Mean_Abs_SHAP']):.2f}；其后主要特征依次为 {shap_sentence(efficiency_shap.iloc[1:])}。与电压任务相比，效率任务中合金元素的重要性更加集中，尤其是 Zn、Sn、Al、Ca、In 和 Mn，对阳极利用行为具有更直接影响。整体来看，SHAP 分析与方法部分中“成分-工艺-电流密度”协同建模的设定是一致的，也为后续元素池筛选提供了机理依据。

    3.2 基于实验优异合金的元素池确定
    图4展示了在 2.5、10 和 40 mA cm^-2 三种工作条件下，实验 top-5 优势合金中各元素的出现频次与平均含量分布。其中，气泡大小表示该元素在对应电流密度 top-5 合金中的出现次数，颜色深浅表示该元素在相关合金中的平均添加量。通过这种“频次 + 含量”的联合筛选方式，可以从已有实验优异合金中提炼出更加稳健的候选元素池，并与高通量搜索空间建立直接联系。

    在 2.5 mA cm^-2 低电流条件下，Y、Sn 和 Sr 在 top-5 合金中均出现 2 次，分别对应平均含量 {figure4_2p5.loc[figure4_2p5['Element']=='Y', 'Mean content (wt%)'].iloc[0]:.3f}、{figure4_2p5.loc[figure4_2p5['Element']=='Sn', 'Mean content (wt%)'].iloc[0]:.3f} 和 {figure4_2p5.loc[figure4_2p5['Element']=='Sr', 'Mean content (wt%)'].iloc[0]:.3f} wt%，说明稀土/碱土微量合金化在低倍率放电下具有较强优势；Ca 和 Zn 也分别出现 1 次，具有一定辅助作用。尽管 Al 在该条件下的出现频次和平均含量都较低，但考虑到其在镁合金体系中的基础调控作用，最终仍与 Ca、Zn、Sn、Y 和 Sr 一同构成低电流筛选元素池。

    在 10 mA cm^-2 中等电流条件下，Al、La 和 Mn 的出现频次最高，均达到 3 次，其中 Al 的平均含量为 {figure4_10.loc[figure4_10['Element']=='Al', 'Mean content (wt%)'].iloc[0]:.3f} wt%，明显高于其他元素；Zn 出现 2 次，而 In、Gd 和 Ca 各出现 1 次。该结果说明中等电流密度下，合金性能开始更加依赖多元素协同调控，既需要 Al、Mn 这类对组织演化敏感的元素，也需要 La、Gd、In 等能改善电化学行为的元素。因此，最终构建的元素池为 Al、Ca、Mn、Zn、La、Gd 和 In。

    在 40 mA cm^-2 高电流条件下，Al 和 Mn 在 top-5 合金中均出现 4 次，Zn 出现 3 次，Y 出现 2 次，而 Li、Sm、La、Gd、Er 和 Ca 各出现 1 次。其中 Al 和 Li 的平均含量分别达到 {figure4_40.loc[figure4_40['Element']=='Al', 'Mean content (wt%)'].iloc[0]:.3f} wt% 和 {figure4_40.loc[figure4_40['Element']=='Li', 'Mean content (wt%)'].iloc[0]:.3f} wt%，显示出高倍率放电条件下较高含量强化元素的重要性。综合频次与平均含量之后，40 mA cm^-2 的最终元素池被扩展为 Al、Ca、Mn、Zn、Er、Y、La、Sm、Gd 和 Li。总体上，图4表明随着电流密度升高，候选元素池由低含量微合金化逐渐过渡到更复杂、含量更高的多元协同调控体系，这与方法部分中分层设置筛选空间的策略保持一致。

    3.3 高通量筛选结果与候选体系排序
    基于图4确定的元素池和方法部分给出的离散成分-工艺网格，分别在 2.5、10 和 40 mA cm^-2 下开展了三元与四元体系的高通量筛选。图5展示的是在每一个电流密度与体系类别中，对同一合金体系仅保留最高得分成分/工艺组合后的前五名结果。图中蓝色柱表示归一化后的电压贡献，橙色柱表示归一化后的阳极效率贡献，两者叠加形成用于可视化比较的系统级综合得分；而实际候选排序仍然以方法部分定义的 z-score 复合指标为准。

    在 2.5 mA cm^-2 条件下，三元体系中 Mg-Y-Sr 以 83.4 的可视化综合得分位居第一，对应的最佳候选为 {ternary_2p5.iloc[0]['Composition label']}，预测平均放电电压和阳极效率分别为 {ternary_2p5.iloc[0]['Pred_AvgDischargeVoltage_mV']:.2f} mV 和 {ternary_2p5.iloc[0]['Pred_AnodicEfficiency_pct']:.2f}% 。其后依次为 Mg-Sn-Sr、Mg-Ca-Sr、Mg-Ca-Y 和 Mg-Sn-Y，说明低电流密度下 Y/Sr 与 Sn/Sr 相关体系最具潜力。四元体系中，Mg-Zn-Sn-Sr、Mg-Zn-Y-Sr 和 Mg-Ca-Sn-Sr 的得分较高，其中 Mg-Zn-Sn-Sr 对应的最佳候选可达到 {quaternary_2p5.iloc[0]['Pred_AvgDischargeVoltage_mV']:.2f} mV 和 {quaternary_2p5.iloc[0]['Pred_AnodicEfficiency_pct']:.2f}% 。这说明在低倍率条件下，引入第三种合金化元素虽然可以进一步拉高部分电压指标，但以 Y-Sr 为代表的简洁三元体系已能够在电压和效率之间取得较优平衡。

    在 10 mA cm^-2 条件下，三元体系的头部结果集中在 In 参与的合金系统。Mg-Zn-In 以 88.2 的综合得分居首，其最佳组合为 {ternary_10.iloc[0]['Composition label']}，预测电压和效率分别为 {ternary_10.iloc[0]['Pred_AvgDischargeVoltage_mV']:.2f} mV 和 {ternary_10.iloc[0]['Pred_AnodicEfficiency_pct']:.2f}% 。Mg-Ca-In 和 Mg-Gd-In 分列第二、第三位，而 Mg-Mn-In 与 Mg-Mn-Gd 也进入前五，说明 In 在中等电流密度下具有较强的正向调控作用。四元体系方面，Mg-Ca-Zn-In 明显领先，其后为 Mg-Zn-La-In、Mg-Al-Mn-La、Mg-Al-Zn-La 和 Mg-Ca-Zn-La。由此可见，中电流密度下候选体系呈现出“Zn/In 主导、Ca/La/Mn 协同”的特点，这与图4中 Al、La、Mn、Zn、In 等元素的高频出现形成了良好呼应。

    在 40 mA cm^-2 高电流条件下，筛选结果进一步向高含量 Al 和 Li/稀土协同体系集中。三元体系中，Mg-Al-Li 以 85.0 的综合得分排名第一，其后为 Mg-Al-Er、Mg-Er-Li、Mg-Al-Y 和 Mg-Al-La；对应最佳候选的预测结果显示，Al-Li 体系在高倍率下能够同时保持较高的放电电压和阳极效率。四元体系中，Mg-Al-La-Gd 和 Mg-Al-Er-Gd 排名前两位，随后是 Mg-Er-Sm-Li、Mg-Zn-Y-Li 和 Mg-Al-Er-Sm。总体而言，40 mA cm^-2 下头部候选基本都包含 Al，并与 Li 或稀土元素形成协同，说明高倍率放电更依赖较强的合金化调控来稳定放电过程和提升阳极利用率。

    从图5整体可以看出，在当前的系统级去重与可视化归一化规则下，三元体系在三个电流密度下都给出了较高的综合得分，并且具有更低的成分复杂度；四元体系虽然提供了更丰富的设计空间，但其优势更多体现在局部性能增强和更多元的元素协同路径上。因此，在首轮实验验证阶段，优先从排名靠前且组成更简洁的三元体系中选取代表性合金，更有利于兼顾可制造性、机理分析清晰度和实验成本。

    3.4 推荐用于实验验证的合金
    在图5给出的体系级排序基础上，进一步结合 z-score 综合排名、合金组成简洁性、工艺可行性以及低/中/高电流密度的代表性，最终从三元体系中各选取一种合金作为首轮实验验证对象。对于 2.5 mA cm^-2，推荐合金为 {shortlist_2p5['Alloy system']}，成分为 {shortlist_2p5['Composition (wt%)']}，推荐工艺参数为 HT = {shortlist_2p5['HT (degC)']:.0f} degC、Ht = {shortlist_2p5['Ht (h)']:.0f} h、ET = {shortlist_2p5['ET (degC)']:.0f} degC、ES = {shortlist_2p5['ES (mm s^-1)']:.1f} mm s^-1、ER = {shortlist_2p5['ER']:.0f}；其预测平均放电电压为 {shortlist_2p5['Predicted voltage (mV)']:.2f} mV，阳极效率为 {shortlist_2p5['Predicted efficiency (%)']:.2f}% 。该体系代表了低电流密度下 Y-Sr 微合金化路线的最佳平衡方案。

    对于 10 mA cm^-2，推荐合金为 {shortlist_10['Alloy system']}，成分为 {shortlist_10['Composition (wt%)']}，其推荐工艺参数同样集中在较高均质化/挤压温度与较低挤压速度区域；模型预测该合金的平均放电电压可达 {shortlist_10['Predicted voltage (mV)']:.2f} mV，阳极效率达到 {shortlist_10['Predicted efficiency (%)']:.2f}% 。结合图5b的体系排序可知，Zn-In 体系不仅在中电流密度下综合得分最高，而且具备较好的成分简洁性，因此适合作为中倍率工况的优先验证对象。

    对于 40 mA cm^-2，推荐合金为 {shortlist_40['Alloy system']}，成分为 {shortlist_40['Composition (wt%)']}，推荐工艺为 HT = {shortlist_40['HT (degC)']:.0f} degC、Ht = {shortlist_40['Ht (h)']:.0f} h、ET = {shortlist_40['ET (degC)']:.0f} degC、ES = {shortlist_40['ES (mm s^-1)']:.1f} mm s^-1、ER = {shortlist_40['ER']:.0f}。其预测平均放电电压为 {shortlist_40['Predicted voltage (mV)']:.2f} mV，阳极效率为 {shortlist_40['Predicted efficiency (%)']:.2f}% 。虽然高电流条件下也出现了多种高分四元候选，但 Al-Li 体系具有更清晰的成分设计逻辑和更高的实验可实施性，因此被选为高倍率放电的代表性验证合金。

    综上，图3-图5共同构成了一个完整的结果链条：图3证明了 BatteryBERT-GBStack-HPO 在双目标预测上的可靠性；图4将实验优势元素信息转化为可筛选的候选元素池；图5则在此基础上完成了体系级排序与代表性候选筛选。最终推荐的 Y-Sr、Zn-In 和 Al-Li 三种合金分别对应低、中、高三种电流密度工作场景，可作为下一步实验制备与电化学验证的核心对象。
    """).strip()

    (TEXT_DIR / "结果部分_中文.md").write_text(results_cn, encoding="utf-8")
    build_basic_docx(TEXT_DIR / "结果部分_中文.docx", results_cn.splitlines())


def export_source_tables(dataset_stats: pd.DataFrame, model_metrics: pd.DataFrame, element_support: pd.DataFrame, ranking_df: pd.DataFrame) -> None:
    with pd.ExcelWriter(TEXT_DIR / "Paper_source_tables.xlsx", engine="openpyxl") as writer:
        dataset_stats.to_excel(writer, sheet_name="Figure2_dataset_stats", index=False)
        model_metrics.to_excel(writer, sheet_name="Figure3_model_metrics", index=False)
        element_support.to_excel(writer, sheet_name="Figure4_element_support", index=False)
        ranking_df.to_excel(writer, sheet_name="Figure5_rankings", index=False)
        pd.DataFrame(VALIDATION_SHORTLIST).to_excel(writer, sheet_name="Validation_shortlist", index=False)


def main() -> None:
    safe_mkdir(FIGURE_DIR)
    safe_mkdir(TEXT_DIR)

    raw_df = clean_columns(pd.read_excel(RAWDATA_PATH))
    top5_raw_df = clean_columns(pd.read_excel(TOP5_SUMMARY_PATH, sheet_name="Top 5 Raw Data"))
    selected_workbook = get_selected_workbook()

    dataset_stats = plot_dataset_distribution(raw_df)
    model_metrics = load_cached_model_metrics()
    build_model_performance_figures(model_metrics)
    element_support = build_element_selection_figure(top5_raw_df)
    ranking_df = build_screening_ranking_figure(selected_workbook)
    write_nano_banana_prompt()
    build_manuscript(model_metrics, dataset_stats)
    build_chinese_results_section(model_metrics, element_support)
    export_source_tables(dataset_stats, model_metrics, element_support, ranking_df)

    print("Generated figure folder:", FIGURE_DIR)
    print("Generated manuscript folder:", TEXT_DIR)


if __name__ == "__main__":
    main()
