# -*- coding: utf-8 -*-
import os, math, hashlib
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from transformers import AutoTokenizer, AutoModel

from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error

# =========================
# Config
# =========================
INPUT_XLSX = "rawdata.xlsx"
SHEET_NAME = "Sheet1"
OUTPUT_DIR = "outputs_GBstack_hpo"
SEED = 42
OUTER_SPLITS = 5
INNER_SPLITS = 5

MODEL_BERT = "bert-base-uncased"
MODEL_BATTERYBERT = "batterydata/batterybert-uncased"
EMB_BATCH_SIZE = 32
MAX_LENGTH = 96

# HPO budget (reduce if too slow)
N_ITER = 25

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


# =========================
# Helpers
# =========================
def safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)

def hash_key(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:10]

def is_valid_number(x) -> bool:
    if x is None:
        return False
    try:
        if isinstance(x, str) and x.strip() == "":
            return False
        v = float(x)
        return not math.isnan(v)
    except Exception:
        return False

def find_col(df: pd.DataFrame, candidates: List[str], required: bool=True) -> Optional[str]:
    cols = list(df.columns)
    for c in candidates:
        if c in cols:
            return c
    low_cols = {c.lower(): c for c in cols}
    for c in candidates:
        cl = c.lower()
        for k, orig in low_cols.items():
            if cl in k:
                return orig
    if required:
        raise KeyError(f"Cannot find column. Tried {candidates}\nAvailable: {cols}")
    return None

def r2_rmse(y_true, y_pred) -> Dict[str, float]:
    r2 = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {"R2": r2, "RMSE": rmse}

def parity_plot_style(y_true, y_pred, title, xlabel, ylabel, outpath, r2=None, rmse=None):
    plt.figure(figsize=(7, 6))
    plt.scatter(
        y_true, y_pred,
        s=55, alpha=0.7,
        edgecolors="k", linewidths=0.6,
        label="CV Prediction"
    )
    minv = float(np.nanmin([np.nanmin(y_true), np.nanmin(y_pred)]))
    maxv = float(np.nanmax([np.nanmax(y_true), np.nanmax(y_pred)]))
    plt.plot([minv, maxv], [minv, maxv], linestyle="--", linewidth=2.5, label="Perfect Prediction")

    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    if r2 is not None and rmse is not None:
        plt.title(f"{title}\n$R^2$={r2:.3f}  RMSE={rmse:.3f}", fontsize=14)
    else:
        plt.title(title, fontsize=14)

    plt.legend(frameon=True, fontsize=12)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


# =========================
# Columns / Text builder (no numbers)
# =========================
ELEMENT_COLS = [
    "Mg","Al","Ca","Mn","Zn","Sn","La","Ce","Er","In","Y","Ga","Gd","Sm","Ge","Nd",
    "Li","Sr","Zr","Fe","Cu","Ni","Pb","Bi","Ag","Si","Co","Ti","Cd"
]

PROCESS_STEPS = [
    ("homogenization", ["均匀化温度（℃）", "均匀化时间", "Homogenization temperature", "Homogenization time"]),
    ("extrusion",      ["挤压温度", "挤压速度（mm/s）", "挤压比", "Extrusion temperature", "Extrusion speed", "Extrusion ratio"]),
    ("rolling",        ["轧制温度", "总轧制压下量%", "Rolling temperature", "Total rolling reduction"]),
    ("annealing",      ["退火温度", "退火时间", "Annealing temperature", "Annealing time"]),
]

def build_text_no_numbers(row: pd.Series,
                          element_cols_present: List[str],
                          proc_cols_present: Dict[str, List[str]]) -> str:
    elems = []
    for el in element_cols_present:
        if el == "Mg":
            continue
        v = row.get(el, 0)
        if is_valid_number(v) and abs(float(v)) > 1e-12:
            elems.append(el)
    comp_str = "Mg-based alloy." if len(elems) == 0 else "Mg-based alloy with " + ", ".join(elems) + "."

    steps = []
    for step_name, _ in PROCESS_STEPS:
        cols_found = proc_cols_present.get(step_name, [])
        present = False
        for c in cols_found:
            v = row.get(c, np.nan)
            if is_valid_number(v) and abs(float(v)) > 1e-12:
                present = True
                break
        if present:
            steps.append(step_name)
    proc_str = "" if len(steps) == 0 else " After " + " and ".join(steps) + "."
    return (comp_str + proc_str).strip()


# =========================
# Embedding
# =========================
@torch.no_grad()
def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts

def embed_texts(texts: List[str], model_name: str, cache_dir: str,
                batch_size: int = 32, max_length: int = 128) -> np.ndarray:
    safe_mkdir(cache_dir)
    fp = hash_key("\n".join(texts[:200]) + f"__N{len(texts)}")
    cache_path = os.path.join(cache_dir, f"emb_{hash_key(model_name)}_L{max_length}_{fp}.npy")
    if os.path.exists(cache_path):
        return np.load(cache_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name)
    mdl.to(device)
    mdl.eval()

    all_emb = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tok(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = mdl(**enc)
        emb = mean_pool(out.last_hidden_state, enc["attention_mask"])
        all_emb.append(emb.detach().cpu().numpy().astype(np.float32))

    embs = np.vstack(all_emb)
    np.save(cache_path, embs)
    return embs


# =========================
# HPO spaces
# =========================
def make_gb_param_dist(prefix: str = "") -> Dict[str, list]:
    # use list-based distributions (RandomizedSearchCV accepts lists)
    # You can broaden if needed.
    dist = {
        f"{prefix}n_estimators": [200, 300, 500, 800, 1200],
        f"{prefix}learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
        f"{prefix}max_depth": [2, 3, 4, 5],
        f"{prefix}subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        f"{prefix}min_samples_split": [2, 4, 6, 8, 10],
        f"{prefix}min_samples_leaf": [1, 2, 3, 4, 5],
        f"{prefix}max_features": [None, "sqrt", "log2"],
    }
    return dist

RIDGE_ALPHA_LIST = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 80.0]


# =========================
# Leakage-safe nested stacking with HPO (per target)
# =========================
def fit_best_gb(X, y, cv, n_iter, random_state) -> GradientBoostingRegressor:
    gb = GradientBoostingRegressor(random_state=random_state)
    search = RandomizedSearchCV(
        gb,
        param_distributions=make_gb_param_dist(),
        n_iter=n_iter,
        scoring="r2",
        cv=cv,
        random_state=random_state,
        n_jobs=-1
    )
    search.fit(X, y)
    return search.best_estimator_

def fit_best_ridge(X, y, cv, n_iter, random_state) -> Pipeline:
    # Ridge alpha search (simple)
    pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(random_state=random_state))])
    param_dist = {"ridge__alpha": RIDGE_ALPHA_LIST}
    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=min(n_iter, len(RIDGE_ALPHA_LIST)),
        scoring="r2",
        cv=cv,
        random_state=random_state,
        n_jobs=-1
    )
    search.fit(X, y)
    return search.best_estimator_

def stacking_oof_hpo_one_target(X_num, X_emb, y, outer_splits=5, inner_splits=5, seed=42, n_iter=25):
    outer = KFold(n_splits=outer_splits, shuffle=True, random_state=seed)
    oof_pred = np.zeros(len(y), dtype=float)
    best_params_rows = []

    for fold, (tr_idx, te_idx) in enumerate(outer.split(X_num), start=1):
        Xn_tr, Xn_te = X_num[tr_idx], X_num[te_idx]
        Xe_tr, Xe_te = X_emb[tr_idx], X_emb[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        inner_cv = KFold(n_splits=inner_splits, shuffle=True, random_state=seed + fold)

        # ---- 1) HPO for base models on outer-train ----
        best_gb_num = fit_best_gb(Xn_tr, y_tr, cv=inner_cv, n_iter=n_iter, random_state=seed + fold)
        best_ridge = fit_best_ridge(Xe_tr, y_tr, cv=inner_cv, n_iter=n_iter, random_state=seed + fold)

        # ---- 2) Build meta-train features via inner OOF predictions (no leakage) ----
        meta_X_tr = np.zeros((len(tr_idx), 2), dtype=float)
        for itr, (itr_tr, itr_va) in enumerate(inner_cv.split(Xn_tr), start=1):
            Xn_a, Xn_b = Xn_tr[itr_tr], Xn_tr[itr_va]
            Xe_a, Xe_b = Xe_tr[itr_tr], Xe_tr[itr_va]
            y_a = y_tr[itr_tr]

            gb_tmp = GradientBoostingRegressor(random_state=seed + fold)
            gb_tmp.set_params(**best_gb_num.get_params())
            gb_tmp.fit(Xn_a, y_a)
            p_num = gb_tmp.predict(Xn_b)

            ridge_tmp = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(random_state=seed + fold))])
            ridge_tmp.set_params(**best_ridge.get_params())
            ridge_tmp.fit(Xe_a, y_a)
            p_txt = ridge_tmp.predict(Xe_b)

            meta_X_tr[itr_va, 0] = p_num
            meta_X_tr[itr_va, 1] = p_txt

        # ---- 3) HPO for meta model on meta-train features ----
        # (we can tune meta too; use a small search for speed)
        meta_cv = KFold(n_splits=inner_splits, shuffle=True, random_state=seed + 100 + fold)
        meta_search = RandomizedSearchCV(
            GradientBoostingRegressor(random_state=seed + 200 + fold),
            param_distributions=make_gb_param_dist(),
            n_iter=max(10, n_iter // 2),
            scoring="r2",
            cv=meta_cv,
            random_state=seed + 300 + fold,
            n_jobs=-1
        )
        meta_search.fit(meta_X_tr, y_tr)
        best_meta = meta_search.best_estimator_

        # ---- 4) Fit best bases on full outer-train and predict outer-test ----
        best_gb_num.fit(Xn_tr, y_tr)
        best_ridge.fit(Xe_tr, y_tr)
        p_num_te = best_gb_num.predict(Xn_te)
        p_txt_te = best_ridge.predict(Xe_te)

        meta_X_te = np.vstack([p_num_te, p_txt_te]).T
        p_final = best_meta.predict(meta_X_te)
        oof_pred[te_idx] = p_final

        # record best params
        best_params_rows.append({
            "Fold": fold,
            "best_gb_num": str({k: best_gb_num.get_params()[k] for k in ["n_estimators","learning_rate","max_depth","subsample","min_samples_split","min_samples_leaf","max_features"]}),
            "best_ridge_alpha": float(best_ridge.get_params()["ridge__alpha"]),
            "best_meta": str({k: best_meta.get_params()[k] for k in ["n_estimators","learning_rate","max_depth","subsample","min_samples_split","min_samples_leaf","max_features"]}),
        })

    return oof_pred, pd.DataFrame(best_params_rows)


# =========================
# Main
# =========================
def main():
    safe_mkdir(OUTPUT_DIR)
    cache_dir = os.path.join(OUTPUT_DIR, "cache_embeddings")
    safe_mkdir(cache_dir)

    df = pd.read_excel(INPUT_XLSX, sheet_name=SHEET_NAME).copy()

    cd_col = find_col(df, ["current density/mA/cm2", "current density", "Current density"], required=True)
    yV_col = find_col(df, ["Average discharge voltage/mV", "Average discharge voltage", "discharge voltage"], required=True)
    yE_col = find_col(df, ["Anodic efficiency/%", "Anodic efficiency", "anodic efficiency"], required=True)

    element_cols_present = [c for c in ELEMENT_COLS if c in df.columns]

    proc_cols_present = {}
    for step_name, candidates in PROCESS_STEPS:
        cols_found = []
        for cand in candidates:
            col = find_col(df, [cand], required=False)
            if col is not None:
                cols_found.append(col)
        proc_cols_present[step_name] = cols_found

    # numeric cols: all elements + all process cols found + current density
    numeric_cols = []
    numeric_cols += [c for c in element_cols_present if c in df.columns]
    for cols in proc_cols_present.values():
        for c in cols:
            if c not in numeric_cols:
                numeric_cols.append(c)
    if cd_col not in numeric_cols:
        numeric_cols.append(cd_col)

    df["text_input"] = [build_text_no_numbers(row, element_cols_present, proc_cols_present)
                        for _, row in df.iterrows()]

    needed = [yV_col, yE_col] + numeric_cols
    mask = df[needed].applymap(is_valid_number).all(axis=1)
    df2 = df.loc[mask].reset_index(drop=True)

    texts = df2["text_input"].tolist()
    X_num = df2[numeric_cols].astype(float).values
    yV = df2[yV_col].astype(float).values
    yE = df2[yE_col].astype(float).values

    emb_bert = embed_texts(texts, MODEL_BERT, cache_dir, batch_size=EMB_BATCH_SIZE, max_length=MAX_LENGTH)
    emb_batt = embed_texts(texts, MODEL_BATTERYBERT, cache_dir, batch_size=EMB_BATCH_SIZE, max_length=MAX_LENGTH)

    outputs = []

    for name, emb in [("BERT_GBstack_HPO", emb_bert), ("BatteryBERT_GBstack_HPO", emb_batt)]:
        # Voltage
        oofV, bestV = stacking_oof_hpo_one_target(
            X_num, emb, yV, OUTER_SPLITS, INNER_SPLITS, SEED, N_ITER
        )
        mV = r2_rmse(yV, oofV)
        parity_plot_style(
            yV, oofV,
            title=f"Voltage - {name}",
            xlabel="Experimental Voltage (mV)",
            ylabel="Predicted Voltage (mV)",
            outpath=os.path.join(OUTPUT_DIR, f"Parity_{name}_Voltage.png"),
            r2=mV["R2"], rmse=mV["RMSE"]
        )
        outputs.append({"EmbeddingModel": name, "Target": "Average discharge voltage (mV)", **mV, "CV": "5-fold OOF"})

        # Efficiency
        oofE, bestE = stacking_oof_hpo_one_target(
            X_num, emb, yE, OUTER_SPLITS, INNER_SPLITS, SEED, N_ITER
        )
        mE = r2_rmse(yE, oofE)
        parity_plot_style(
            yE, oofE,
            title=f"Efficiency - {name}",
            xlabel="Experimental Efficiency (%)",
            ylabel="Predicted Efficiency (%)",
            outpath=os.path.join(OUTPUT_DIR, f"Parity_{name}_Efficiency.png"),
            r2=mE["R2"], rmse=mE["RMSE"]
        )
        outputs.append({"EmbeddingModel": name, "Target": "Anodic efficiency (%)", **mE, "CV": "5-fold OOF"})

        # save fold best params
        bestV.to_csv(os.path.join(OUTPUT_DIR, f"BestParams_{name}_Voltage.csv"), index=False, encoding="utf-8-sig")
        bestE.to_csv(os.path.join(OUTPUT_DIR, f"BestParams_{name}_Efficiency.csv"), index=False, encoding="utf-8-sig")

    metrics_df = pd.DataFrame(outputs)
    metrics_df["R2"] = metrics_df["R2"].round(3)
    metrics_df["RMSE"] = metrics_df["RMSE"].round(3)

    out_xlsx = os.path.join(OUTPUT_DIR, "GBstack_HPO_metrics.xlsx")
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        df2.to_excel(writer, index=False, sheet_name="clean_data_with_text")
        pd.DataFrame({"numeric_cols_used": numeric_cols}).to_excel(writer, index=False, sheet_name="numeric_cols")
        metrics_df.to_excel(writer, index=False, sheet_name="metrics_oof")

    print("Done:", out_xlsx)


if __name__ == "__main__":
    main()
