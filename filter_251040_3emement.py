# -*- coding: utf-8 -*-
import os, math
import numpy as np
import pandas as pd

import torch
from transformers import AutoTokenizer, AutoModel

from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import joblib
from itertools import combinations

# =========================
# CONFIG
# =========================
INPUT_XLSX = "rawdata.xlsx"
SHEET_NAME = "Sheet1"
OUTPUT_DIR = "outputs_design_search_multiCD_selected"

MODEL_NAME = "batterydata/batterybert-uncased"   # or "bert-base-uncased"

# ---- Multi-CD search plans (your latest) ----
# NOTE: This script performs TRUE TERNARY (Mg + 2 elements) search.
SEARCH_PLANS = [
    {
        "cd": 2.5,
        "element_pool": ["Al", "Ca", "Zn", "Sn", "Y", "Sr"],
        "max_sum": 2.06,
        "step": 0.1,
        "tag": "CD2.5_Ternary"
    },
    {
        "cd": 10.0,
        "element_pool": ["Al", "Ca", "Mn", "Zn", "La", "Gd", "In"],
        "max_sum": 10.43,
        "step": 0.5,
        "tag": "CD10_Ternary"
    },
    {
        "cd": 40.0,
        "element_pool": ["Al", "Ca", "Mn", "Zn", "Er", "Y", "La", "Sm", "Gd", "Li"],
        "max_sum": 11.97,
        "step": 0.8,
        "tag": "CD40_Ternary"
    },
]

# ---- Process Grid (144 combos) ----
HT_vals = [400.0, 450.0]                  # HT (2)
Ht_vals = [6.0, 12.0, 24.0]               # Ht (3)
ET_vals = [250.0, 300.0, 350.0, 400.0]    # ET (4)
ES_vals = [0.5, 1.5, 2.5]                 # ES (3)
ER_vals = [30.0, 45.0]                    # ER (2)

# training / HPO
SEED = 42
OUTER_SPLITS = 5
INNER_SPLITS = 5
N_ITER_BASE = 25
N_ITER_META = 12

# embedding
EMB_BATCH_SIZE = 64
MAX_LENGTH = 80

# streaming prediction
CHUNK_SIZE = 50000
TOPK = 500

# cache trained models
SAVE_MODELS = True
SKIP_TRAIN_IF_EXISTS = True

MODEL_CACHE_DIR_CANDIDATES = [
    "outputs_design_search_multiCD",
    OUTPUT_DIR,
]

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


# =========================
# HELPERS
# =========================
def safe_mkdir(p):
    os.makedirs(p, exist_ok=True)

def is_valid_number(x):
    try:
        if x is None:
            return False
        if isinstance(x, str) and x.strip() == "":
            return False
        v = float(x)
        return not math.isnan(v)
    except:
        return False

def find_col(df, candidates, required=True):
    cols = list(df.columns)
    for c in candidates:
        if c in cols:
            return c
    low = {c.lower(): c for c in cols}
    for cand in candidates:
        cl = cand.lower()
        for k, orig in low.items():
            if cl in k:
                return orig
    if required:
        raise KeyError(f"Cannot find column. Tried {candidates}\nAvailable columns:\n{cols}")
    return None

def make_gb_param_dist():
    return {
        "n_estimators": [200, 300, 500, 800, 1200],
        "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
        "max_depth": [2, 3, 4, 5],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "min_samples_split": [2, 4, 6, 8, 10],
        "min_samples_leaf": [1, 2, 3, 4, 5],
        "max_features": [None, "sqrt", "log2"],
    }

RIDGE_ALPHA_LIST = [1, 2, 5, 10, 30, 80, 200, 500]

def fit_best_gb(X, y, cv, n_iter, rs):
    search = RandomizedSearchCV(
        GradientBoostingRegressor(random_state=rs),
        param_distributions=make_gb_param_dist(),
        n_iter=n_iter,
        scoring="r2",
        cv=cv,
        random_state=rs,
        n_jobs=-1
    )
    search.fit(X, y)
    return search.best_estimator_

def fit_best_ridge(X, y, cv, rs):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(random_state=rs, solver="lsqr"))
    ])
    search = RandomizedSearchCV(
        pipe,
        param_distributions={"ridge__alpha": RIDGE_ALPHA_LIST},
        n_iter=min(len(RIDGE_ALPHA_LIST), 12),
        scoring="r2",
        cv=cv,
        random_state=rs,
        n_jobs=-1
    )
    search.fit(X, y)
    return search.best_estimator_

@torch.no_grad()
def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts

@torch.no_grad()
def embed_texts(texts, tokenizer, model, batch_size=64, max_length=80, device="cpu"):
    embs = []
    model.eval()
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            emb = mean_pool(out.last_hidden_state, enc["attention_mask"])
            embs.append(emb.detach().cpu().numpy().astype(np.float32))
    return np.vstack(embs)

def predict_stacking(base_num, base_txt, meta, X_num, X_emb):
    p_num = base_num.predict(X_num)
    p_txt = base_txt.predict(X_emb)
    meta_X = np.vstack([p_num, p_txt]).T
    return meta.predict(meta_X)

def metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return r2, rmse, mae

def build_grid_text_from_elements(active_elements):
    active_elements = sorted(list(active_elements))
    if len(active_elements) == 0:
        return "Mg-based alloy. After homogenization and extrusion."
    return f"Mg-based alloy with {', '.join(active_elements)}. After homogenization and extrusion."

class RunningStats:
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
    def update(self, x):
        x = np.asarray(x, dtype=float)
        for v in x:
            self.n += 1
            delta = v - self.mean
            self.mean += delta / self.n
            delta2 = v - self.mean
            self.M2 += delta * delta2
    def finalize(self):
        if self.n < 2:
            return self.mean, 1.0
        var = self.M2 / (self.n - 1)
        return self.mean, float(np.sqrt(var) + 1e-9)

def z_from_stats(x, mean, std):
    return (x - mean) / (std + 1e-9)

def keep_top(df, key, k):
    if df.empty:
        return df
    return df.nlargest(k, key)

def arange_inclusive(a, b, step, ndigits=3):
    if step <= 0:
        raise ValueError("step must be > 0")
    n = int(np.floor((b - a) / step + 1e-12))
    vals = [a + i * step for i in range(n + 1)]
    return np.array([round(v, ndigits) for v in vals], dtype=float)

def pick_existing_model_paths(output_dir):
    for d in MODEL_CACHE_DIR_CANDIDATES:
        v = os.path.join(d, "stack_voltage.joblib")
        e = os.path.join(d, "stack_efficiency.joblib")
        n = os.path.join(d, "numeric_cols_used.joblib")
        info = os.path.join(d, "encoder_info.joblib")
        if os.path.exists(v) and os.path.exists(e) and os.path.exists(n):
            return v, e, n, info, d
    return (
        os.path.join(output_dir, "stack_voltage.joblib"),
        os.path.join(output_dir, "stack_efficiency.joblib"),
        os.path.join(output_dir, "numeric_cols_used.joblib"),
        os.path.join(output_dir, "encoder_info.joblib"),
        output_dir
    )

# =========================
# NESTED OOF METRICS
# =========================
def nested_oof_metrics(X_num, X_emb, y, seed=42, outer_splits=5, inner_splits=5,
                       n_iter_base=25, n_iter_meta=12):
    outer = KFold(n_splits=outer_splits, shuffle=True, random_state=seed)
    oof_pred = np.zeros(len(y), dtype=float)

    for fold, (tr_idx, te_idx) in enumerate(outer.split(X_num), start=1):
        Xn_tr, Xn_te = X_num[tr_idx], X_num[te_idx]
        Xe_tr, Xe_te = X_emb[tr_idx], X_emb[te_idx]
        y_tr = y[tr_idx]

        inner_cv = KFold(n_splits=inner_splits, shuffle=True, random_state=seed + fold)

        base_num = fit_best_gb(Xn_tr, y_tr, cv=inner_cv, n_iter=n_iter_base, rs=seed + 100 + fold)
        base_txt = fit_best_ridge(Xe_tr, y_tr, cv=inner_cv, rs=seed + 200 + fold)

        metaX = np.zeros((len(y_tr), 2), dtype=float)
        for a, b in inner_cv.split(Xn_tr):
            gb_tmp = GradientBoostingRegressor(random_state=seed + 300 + fold).set_params(**base_num.get_params())
            rg_tmp = Pipeline([("scaler", StandardScaler()),
                               ("ridge", Ridge(random_state=seed + 400 + fold, solver="lsqr"))]).set_params(**base_txt.get_params())
            gb_tmp.fit(Xn_tr[a], y_tr[a])
            rg_tmp.fit(Xe_tr[a], y_tr[a])
            metaX[b, 0] = gb_tmp.predict(Xn_tr[b])
            metaX[b, 1] = rg_tmp.predict(Xe_tr[b])

        meta_search = RandomizedSearchCV(
            GradientBoostingRegressor(random_state=seed + 500 + fold),
            param_distributions=make_gb_param_dist(),
            n_iter=n_iter_meta,
            scoring="r2",
            cv=inner_cv,
            random_state=seed + 600 + fold,
            n_jobs=-1
        )
        meta_search.fit(metaX, y_tr)
        meta = meta_search.best_estimator_

        base_num.fit(Xn_tr, y_tr)
        base_txt.fit(Xe_tr, y_tr)

        oof_pred[te_idx] = predict_stacking(base_num, base_txt, meta, Xn_te, Xe_te)

    r2v, rmsev, maev = metrics(y, oof_pred)
    return r2v, rmsev, maev


# =========================
# TRAINING / LOADING (shared once)
# =========================
def train_or_load_models(df, numeric_cols, yV_col, yE_col,
                         cd_col, htT_col, ht_t_col, etT_col, es_col, er_col):

    needed = [yV_col, yE_col] + numeric_cols
    mask = df[needed].applymap(is_valid_number).all(axis=1)
    df2 = df.loc[mask].reset_index(drop=True)

    X_num_train = df2[numeric_cols].astype(float).values
    yV_train = df2[yV_col].astype(float).values
    yE_train = df2[yE_col].astype(float).values

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    try:
        encoder_model = AutoModel.from_pretrained(MODEL_NAME, add_pooling_layer=False).to(device).eval()
    except TypeError:
        encoder_model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

    ELEMENT_COLS_ALL = [
        "Mg","Al","Ca","Mn","Zn","Sn","La","Ce","Er","In","Y","Ga","Gd","Sm","Ge","Nd",
        "Li","Sr","Zr","Fe","Cu","Ni","Pb","Bi","Ag","Si","Co","Ti","Cd"
    ]
    element_names_all = [c for c in ELEMENT_COLS_ALL if c in df2.columns]

    train_texts = []
    for _, r in df2.iterrows():
        active_elements = []
        for el in element_names_all:
            try:
                if abs(float(r.get(el, 0.0))) > 1e-12:
                    active_elements.append(el)
            except:
                pass

        steps = []
        if htT_col and is_valid_number(r.get(htT_col, np.nan)) and abs(float(r.get(htT_col))) > 1e-12:
            steps.append("homogenization")
        if etT_col and is_valid_number(r.get(etT_col, np.nan)) and abs(float(r.get(etT_col))) > 1e-12:
            steps.append("extrusion")

        if len(active_elements) == 0:
            comp = "Mg-based alloy."
        else:
            comp = f"Mg-based alloy with {', '.join(sorted(active_elements))}."

        txt = comp if len(steps) == 0 else (comp + " After " + " and ".join(steps) + ".")
        train_texts.append(txt)

    X_emb_train = embed_texts(train_texts, tokenizer, encoder_model, EMB_BATCH_SIZE, MAX_LENGTH, device)

    MODEL_V_PATH, MODEL_E_PATH, NUMCOLS_PATH, ENCODER_INFO_PATH, MODEL_SOURCE_DIR = pick_existing_model_paths(OUTPUT_DIR)

    r2V, rmseV, maeV = nested_oof_metrics(
        X_num_train, X_emb_train, yV_train, seed=SEED,
        outer_splits=OUTER_SPLITS, inner_splits=INNER_SPLITS,
        n_iter_base=N_ITER_BASE, n_iter_meta=N_ITER_META
    )
    r2E, rmseE, maeE = nested_oof_metrics(
        X_num_train, X_emb_train, yE_train, seed=SEED,
        outer_splits=OUTER_SPLITS, inner_splits=INNER_SPLITS,
        n_iter_base=N_ITER_BASE, n_iter_meta=N_ITER_META
    )

    metrics_df = pd.DataFrame([
        {"EmbeddingModel": "BatteryBERT_GBstack_HPO", "Target": "Average discharge voltage (mV)",
         "R2": round(r2V, 3), "RMSE": round(rmseV, 3), "MAE": round(maeV, 3), "CV": "5-fold OOF (nested HPO)", "N": len(df2)},
        {"EmbeddingModel": "BatteryBERT_GBstack_HPO", "Target": "Anodic efficiency (%)",
         "R2": round(r2E, 3), "RMSE": round(rmseE, 3), "MAE": round(maeE, 3), "CV": "5-fold OOF (nested HPO)", "N": len(df2)},
    ])

    have_models = (os.path.exists(MODEL_V_PATH) and os.path.exists(MODEL_E_PATH)
                   and os.path.exists(NUMCOLS_PATH))

    if SKIP_TRAIN_IF_EXISTS and have_models:
        saved_numeric_cols = joblib.load(NUMCOLS_PATH)
        if saved_numeric_cols != numeric_cols:
            print("[WARN] Cached models numeric_cols mismatch -> retrain.")
            have_models = False
        else:
            if os.path.exists(ENCODER_INFO_PATH):
                saved_encoder_info = joblib.load(ENCODER_INFO_PATH)
                if saved_encoder_info.get("MODEL_NAME") != MODEL_NAME:
                    print("[WARN] Cached MODEL_NAME mismatch -> retrain.")
                    have_models = False

    if have_models:
        base_num_V, base_txt_V, meta_V = joblib.load(MODEL_V_PATH)
        base_num_E, base_txt_E, meta_E = joblib.load(MODEL_E_PATH)
        print(f"[OK] Loaded saved final models from: {MODEL_SOURCE_DIR}")
    else:
        print("[INFO] Training final models...")
        MODEL_V_PATH = os.path.join(OUTPUT_DIR, "stack_voltage.joblib")
        MODEL_E_PATH = os.path.join(OUTPUT_DIR, "stack_efficiency.joblib")
        NUMCOLS_PATH = os.path.join(OUTPUT_DIR, "numeric_cols_used.joblib")
        ENCODER_INFO_PATH = os.path.join(OUTPUT_DIR, "encoder_info.joblib")

        inner_cv = KFold(n_splits=INNER_SPLITS, shuffle=True, random_state=SEED)

        base_num_V = fit_best_gb(X_num_train, yV_train, cv=inner_cv, n_iter=N_ITER_BASE, rs=SEED+1)
        base_txt_V = fit_best_ridge(X_emb_train, yV_train, cv=inner_cv, rs=SEED+2)

        metaX_V = np.zeros((len(yV_train), 2), dtype=float)
        for a, b in inner_cv.split(X_num_train):
            gb_tmp = GradientBoostingRegressor(random_state=SEED+11).set_params(**base_num_V.get_params())
            rg_tmp = Pipeline([("scaler", StandardScaler()),
                               ("ridge", Ridge(random_state=SEED+12, solver="lsqr"))]).set_params(**base_txt_V.get_params())
            gb_tmp.fit(X_num_train[a], yV_train[a])
            rg_tmp.fit(X_emb_train[a], yV_train[a])
            metaX_V[b, 0] = gb_tmp.predict(X_num_train[b])
            metaX_V[b, 1] = rg_tmp.predict(X_emb_train[b])

        meta_search_V = RandomizedSearchCV(
            GradientBoostingRegressor(random_state=SEED+21),
            param_distributions=make_gb_param_dist(),
            n_iter=N_ITER_META,
            scoring="r2",
            cv=inner_cv,
            random_state=SEED+22,
            n_jobs=-1
        )
        meta_search_V.fit(metaX_V, yV_train)
        meta_V = meta_search_V.best_estimator_
        base_num_V.fit(X_num_train, yV_train)
        base_txt_V.fit(X_emb_train, yV_train)

        base_num_E = fit_best_gb(X_num_train, yE_train, cv=inner_cv, n_iter=N_ITER_BASE, rs=SEED+3)
        base_txt_E = fit_best_ridge(X_emb_train, yE_train, cv=inner_cv, rs=SEED+4)

        metaX_E = np.zeros((len(yE_train), 2), dtype=float)
        for a, b in inner_cv.split(X_num_train):
            gb_tmp = GradientBoostingRegressor(random_state=SEED+31).set_params(**base_num_E.get_params())
            rg_tmp = Pipeline([("scaler", StandardScaler()),
                               ("ridge", Ridge(random_state=SEED+32, solver="lsqr"))]).set_params(**base_txt_E.get_params())
            gb_tmp.fit(X_num_train[a], yE_train[a])
            rg_tmp.fit(X_emb_train[a], yE_train[a])
            metaX_E[b, 0] = gb_tmp.predict(X_num_train[b])
            metaX_E[b, 1] = rg_tmp.predict(X_emb_train[b])

        meta_search_E = RandomizedSearchCV(
            GradientBoostingRegressor(random_state=SEED+41),
            param_distributions=make_gb_param_dist(),
            n_iter=N_ITER_META,
            scoring="r2",
            cv=inner_cv,
            random_state=SEED+42,
            n_jobs=-1
        )
        meta_search_E.fit(metaX_E, yE_train)
        meta_E = meta_search_E.best_estimator_
        base_num_E.fit(X_num_train, yE_train)
        base_txt_E.fit(X_emb_train, yE_train)

        if SAVE_MODELS:
            joblib.dump((base_num_V, base_txt_V, meta_V), MODEL_V_PATH)
            joblib.dump((base_num_E, base_txt_E, meta_E), MODEL_E_PATH)
            joblib.dump(numeric_cols, NUMCOLS_PATH)
            joblib.dump({"MODEL_NAME": MODEL_NAME, "MAX_LENGTH": MAX_LENGTH}, ENCODER_INFO_PATH)
            print(f"[OK] Trained and saved final models to: {OUTPUT_DIR}")

    model_pack = {
        "tokenizer": tokenizer,
        "encoder_model": encoder_model,
        "device": device,
        "base_num_V": base_num_V, "base_txt_V": base_txt_V, "meta_V": meta_V,
        "base_num_E": base_num_E, "base_txt_E": meta_E, "meta_E": meta_E,
        "metrics_df": metrics_df,
    }
    model_pack["base_txt_E"] = base_txt_E

    return model_pack, df2


# =========================
# RUN ONE SEARCH PLAN
#   TRUE TERNARY SEARCH:
#     - loop over all 2-element systems (Mg + A + B) from element_pool
#     - composition grid for 2 elements under sum <= max_sum
# =========================
def run_search_plan(plan, numeric_cols, colmap, model_pack, element_cols_present):
    cd_fixed = float(plan["cd"])
    element_pool = list(plan["element_pool"])
    max_sum = float(plan["max_sum"])
    step = float(plan["step"])
    tag = plan["tag"]

    element_pool = [e for e in element_pool if e in element_cols_present]
    if len(element_pool) < 2:
        print(f"[SKIP] {tag}: element_pool after filtering < 2. Available={element_pool}")
        return

    pool_cols = sorted(element_pool)
    BASE_OUT_COLS = ["AlloySystem"] + pool_cols + ["HT","Ht","ET","ES","ER","current_density","sum_selected_additions_wt%"]
    PRED_COLS = ["Pred_AvgDischargeVoltage_mV", "Pred_AnodicEfficiency_pct"]
    CSV_COLS = BASE_OUT_COLS + PRED_COLS
    CSV_COLS_WITH_SCORE = CSV_COLS + ["Score_zV_plus_zE"]

    cd_col = colmap["cd_col"]
    htT_col = colmap["htT_col"]
    ht_t_col = colmap["ht_t_col"]
    etT_col = colmap["etT_col"]
    es_col  = colmap["es_col"]
    er_col  = colmap["er_col"]

    tokenizer = model_pack["tokenizer"]
    encoder_model = model_pack["encoder_model"]
    device = model_pack["device"]

    base_num_V = model_pack["base_num_V"]
    base_txt_V = model_pack["base_txt_V"]
    meta_V     = model_pack["meta_V"]
    base_num_E = model_pack["base_num_E"]
    base_txt_E = model_pack["base_txt_E"]
    meta_E     = model_pack["meta_E"]
    metrics_df = model_pack["metrics_df"]

    EL_vals = arange_inclusive(0.0, max_sum, step, ndigits=3)

    n_proc = len(HT_vals) * len(Ht_vals) * len(ET_vals) * len(ES_vals) * len(ER_vals)
    # 组合数从 3 改为 2
    n_sys = math.comb(len(element_pool), 2)
    print(f"\n[INFO] Running {tag} | fixed CD={cd_fixed} | systems={n_sys} | proc={n_proc}")

    csv_path = os.path.join(OUTPUT_DIR, f"grid_predictions_{tag}_full.csv")
    xlsx_path = os.path.join(OUTPUT_DIR, f"grid_top_{tag}.xlsx")
    if os.path.exists(csv_path):
        os.remove(csv_path)

    header_written = False
    statsV, statsE = RunningStats(), RunningStats()
    top_by_V, top_by_E, top_by_S = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    def flush_buffer(rows, combo_emb, var_elements):
        nonlocal header_written, top_by_V, top_by_E
        if not rows:
            return
        chunk = pd.DataFrame(rows)

        Xnum = pd.DataFrame(0.0, index=np.arange(len(chunk)), columns=numeric_cols)

        # fill elements (with NaN filling as fixed previously)
        for el in var_elements:
            if el in Xnum.columns:
                Xnum[el] = chunk[el].fillna(0.0).values

        if htT_col and htT_col in Xnum.columns: Xnum[htT_col] = chunk["HT"].values
        if ht_t_col and ht_t_col in Xnum.columns: Xnum[ht_t_col] = chunk["Ht"].values
        if etT_col and etT_col in Xnum.columns: Xnum[etT_col] = chunk["ET"].values
        if es_col  and es_col  in Xnum.columns: Xnum[es_col]  = chunk["ES"].values
        if er_col  and er_col  in Xnum.columns: Xnum[er_col]  = chunk["ER"].values
        Xnum[cd_col] = cd_fixed

        X_num = Xnum.values.astype(float)

        # embeddings via 4 masks (since 2 elements: 2^2 = 4 combinations)
        mask_id = np.zeros(len(chunk), dtype=int)
        for i, el in enumerate(var_elements):
            # shift bit by 0 or 1
            mask_id += ((chunk[el].values > 1e-12).astype(int) << i)
        X_emb = np.vstack([combo_emb[m] for m in mask_id])

        predV = predict_stacking(base_num_V, base_txt_V, meta_V, X_num, X_emb)
        predE = predict_stacking(base_num_E, base_txt_E, meta_E, X_num, X_emb)

        chunk["Pred_AvgDischargeVoltage_mV"] = predV
        chunk["Pred_AnodicEfficiency_pct"] = predE

        statsV.update(predV)
        statsE.update(predE)

        top_by_V = keep_top(pd.concat([top_by_V, chunk], ignore_index=True), "Pred_AvgDischargeVoltage_mV", TOPK)
        top_by_E = keep_top(pd.concat([top_by_E, chunk], ignore_index=True), "Pred_AnodicEfficiency_pct", TOPK)

        for c in CSV_COLS:
            if c not in chunk.columns:
                chunk[c] = 0.0
        chunk = chunk.reindex(columns=CSV_COLS)

        chunk.to_csv(csv_path, mode="a", index=False, header=(not header_written), encoding="utf-8-sig")
        header_written = True

    # 循环从 combinations(pool, 3) 改为了 combinations(pool, 2)
    for var_elements in combinations(element_pool, 2):
        var_elements = list(var_elements)
        e1, e2 = var_elements # 提取2个元素

        # 4 embeddings for this binary pair (2^2 = 4)
        combo_texts = []
        for m in range(4):
            active = []
            for i, el in enumerate(var_elements):
                if (m >> i) & 1:
                    active.append(el)
            combo_texts.append(build_grid_text_from_elements(active))

        combo_emb = embed_texts(
            combo_texts, tokenizer, encoder_model,
            batch_size=min(EMB_BATCH_SIZE, len(combo_texts)),
            max_length=MAX_LENGTH, device=device
        ).astype(np.float32)

        buf = []
        # 去掉 v3 的嵌套循环
        for v1 in EL_vals:
            for v2 in EL_vals:
                csum = v1 + v2
                if csum > max_sum + 1e-12:
                    continue

                for HT in HT_vals:
                    for Ht in Ht_vals:
                        for ET in ET_vals:
                            for ES in ES_vals:
                                for ER in ER_vals:
                                    row = {el: 0.0 for el in pool_cols}
                                    row.update({
                                        "AlloySystem": f"{e1}-{e2}",
                                        e1: float(v1), e2: float(v2),
                                        "HT": float(HT), "Ht": float(Ht),
                                        "ET": float(ET), "ES": float(ES), "ER": float(ER),
                                        "current_density": cd_fixed,
                                        "sum_selected_additions_wt%": float(round(csum, 3))
                                    })
                                    buf.append(row)
                                    if len(buf) >= CHUNK_SIZE:
                                        flush_buffer(buf, combo_emb, var_elements)
                                        buf = []
        flush_buffer(buf, combo_emb, var_elements)

    meanV, stdV = statsV.finalize()
    meanE, stdE = statsE.finalize()

    # second pass for score
    for chunk in pd.read_csv(csv_path, chunksize=CHUNK_SIZE):
        zV = z_from_stats(chunk["Pred_AvgDischargeVoltage_mV"].values, meanV, stdV)
        zE = z_from_stats(chunk["Pred_AnodicEfficiency_pct"].values, meanE, stdE)
        chunk["Score_zV_plus_zE"] = 0.5*zV + 0.5*zE

        for c in CSV_COLS_WITH_SCORE:
            if c not in chunk.columns:
                chunk[c] = 0.0
        chunk = chunk.reindex(columns=CSV_COLS_WITH_SCORE)

        top_by_S = keep_top(pd.concat([top_by_S, chunk], ignore_index=True), "Score_zV_plus_zE", TOPK)

    settings_row = {
        "plan_tag": tag,
        "element_pool": ", ".join(element_pool),
        "pool_cols_sorted": ", ".join(pool_cols),
        "ternary_systems": math.comb(len(element_pool), 2),
        "comp_grid": f"0-{max_sum} step{step}",
        "sum_max_wt%": max_sum,
        "HT": str(HT_vals), "Ht": str(Ht_vals), "ET": str(ET_vals),
        "ES": str(ES_vals), "ER": str(ER_vals),
        "current_density_fixed": cd_fixed,
        "score_definition": "Score_zV_plus_zE = 0.5*z(Voltage) + 0.5*z(Efficiency)",
        "score_stats": f"Voltage mean={meanV:.3f}, std={stdV:.3f}; Eff mean={meanE:.3f}, std={stdE:.3f}",
        "encoder": MODEL_NAME,
        "TOPK": TOPK
    }

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        top_by_V.reindex(columns=CSV_COLS).to_excel(writer, index=False, sheet_name="top_by_voltage")
        top_by_E.reindex(columns=CSV_COLS).to_excel(writer, index=False, sheet_name="top_by_efficiency")
        top_by_S.reindex(columns=CSV_COLS_WITH_SCORE).to_excel(writer, index=False, sheet_name="top_by_score")
        metrics_df.to_excel(writer, index=False, sheet_name="metrics_oof")
        pd.DataFrame({"numeric_cols_used": numeric_cols}).to_excel(writer, index=False, sheet_name="numeric_cols_used")
        pd.DataFrame([settings_row]).to_excel(writer, index=False, sheet_name="settings")

    print("[OK] Full predictions CSV:", csv_path)
    print("[OK] Top results + metrics Excel:", xlsx_path)


# =========================
# MAIN
# =========================
def main():
    safe_mkdir(OUTPUT_DIR)
    df = pd.read_excel(INPUT_XLSX, sheet_name=SHEET_NAME).copy()

    cd_col = find_col(df, ["current density/mA/cm2", "current density", "Current density"], required=True)
    yV_col = find_col(df, ["Average discharge voltage/mV", "Average discharge voltage", "discharge voltage"], required=True)
    yE_col = find_col(df, ["Anodic efficiency/%", "Anodic efficiency", "anodic efficiency"], required=True)

    htT_col = find_col(df, ["均匀化温度（℃）", "均匀化温度", "均质化温度"], required=False)
    ht_t_col = find_col(df, ["均匀化时间", "均质化时间"], required=False)
    etT_col = find_col(df, ["挤压温度"], required=False)
    es_col  = find_col(df, ["挤压速度（mm/s）", "挤压速度", "Extrusion speed"], required=False)
    er_col  = find_col(df, ["挤压比", "Extrusion ratio"], required=False)

    ELEMENT_COLS_ALL = [
        "Mg","Al","Ca","Mn","Zn","Sn","La","Ce","Er","In","Y","Ga","Gd","Sm","Ge","Nd",
        "Li","Sr","Zr","Fe","Cu","Ni","Pb","Bi","Ag","Si","Co","Ti","Cd"
    ]
    element_cols_present = [c for c in ELEMENT_COLS_ALL if c in df.columns]

    numeric_cols = []
    numeric_cols += element_cols_present
    for c in [htT_col, ht_t_col, etT_col, es_col, er_col]:
        if c is not None and c not in numeric_cols:
            numeric_cols.append(c)
    if cd_col not in numeric_cols:
        numeric_cols.append(cd_col)

    colmap = {
        "cd_col": cd_col,
        "yV_col": yV_col,
        "yE_col": yE_col,
        "htT_col": htT_col,
        "ht_t_col": ht_t_col,
        "etT_col": etT_col,
        "es_col": es_col,
        "er_col": er_col,
    }

    model_pack, _ = train_or_load_models(
        df=df,
        numeric_cols=numeric_cols,
        yV_col=yV_col,
        yE_col=yE_col,
        cd_col=cd_col,
        htT_col=htT_col,
        ht_t_col=ht_t_col,
        etT_col=etT_col,
        es_col=es_col,
        er_col=er_col
    )

    for plan in SEARCH_PLANS:
        run_search_plan(
            plan,
            numeric_cols=numeric_cols,
            colmap=colmap,
            model_pack=model_pack,
            element_cols_present=element_cols_present
        )

    print("\n[ALL DONE] Finished all CD-specific ternary searches.")

if __name__ == "__main__":
    main()