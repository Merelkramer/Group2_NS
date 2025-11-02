# app.py
import json
from pathlib import Path
from datetime import datetime, time as dtime

import numpy as np
import pandas as pd
import streamlit as st

# ========= Paths / Artifacts =========
DATA_PATH = Path("data/complete_dataset_clean_24oct.csv")
ART_DIR = Path("artifacts_for demo")
MODEL_CANDIDATES = [
    ART_DIR / "mlp_model.keras",   # primary
    ART_DIR / "mlp_model.h5",      # fallback
]
SCALER_PATH = ART_DIR / "scaler.pkl"
FEATCOLS_PATH = ART_DIR / "feature_columns.json"

# ========= Exact raw features your MLP was trained on =========
REQUIRED_RAW_FEATURES = [
    "DAGDEELTREIN", "TRAIN_TYPE", "PROGNOSE_REIZEN",
    "Cancelled", "ExtraTrain",
    "DELAY_CATEGORY", "DISRUPTION_CATEGORY",
    "Previous train canceled", "Previous train delayed (min)",
    "disrupt_any", "Extra train added before departure",
]

# ========= UI Colors (NS palette) =========
NS_YELLOW = "#FFC917"
NS_BLUE = "#003082"
BG_GREY = "#F7F9FC"

st.set_page_config(page_title="NS Passenger Forecast Demo", page_icon="🚆", layout="wide")

st.markdown(
    f"""
    <style>
    .main {{ background-color: {BG_GREY}; }}
    .ns-hero {{
        background: linear-gradient(90deg, {NS_BLUE} 0%, {NS_BLUE} 60%, #001c4d 100%);
        color: white; padding: 16px 20px; border-radius: 14px; margin-bottom: 14px;
        box-shadow: 0 6px 16px rgba(0,0,0,0.15);
    }}
    .ns-badge {{ background: {NS_YELLOW}; color: #111; padding: 4px 10px; border-radius: 999px; font-weight: 700; display: inline-block; margin-left: 8px; }}
    .card {{ background: white; border-radius: 14px; padding: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border: 1px solid #eef1f6; }}
    .metric-title {{ font-size: 0.9rem; color: #334155; margin-bottom: 6px; }}
    .metric-value {{ font-size: 2rem; font-weight: 800; color: {NS_BLUE}; }}
    .metric-sub {{ font-size: 0.85rem; color: #475569; }}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <div class="ns-hero">
      <h2 style="margin:0;display:flex;align-items:center;gap:12px;">
        🚆 NS Passenger Forecast Demo
        <span class="ns-badge">MLP</span>
      </h2>
      <div>Compare NS baseline (PROGNOSE_REIZEN) with our trained MLP — with optional delay/disruption inputs.</div>
    </div>
    """,
    unsafe_allow_html=True
)

# ========= Load data =========
if not DATA_PATH.exists():
    st.error(f"Missing data file: {DATA_PATH}")
    st.stop()

if DATA_PATH.suffix.lower() == ".csv":
    df = pd.read_csv(DATA_PATH, sep=";")
else:
    df = pd.read_parquet(DATA_PATH)

# Normalize a few names if needed
def _norm(s): return s.strip().upper()
rename_map = {}
canonical = ["TRAJECT","DAGDEELTREIN","TRAIN_TYPE","PROGNOSE_REIZEN",
             "DELAY_CATEGORY","DISRUPTION_CATEGORY","REALISATIE"]
for c in df.columns:
    cu = _norm(c)
    if cu in canonical and cu != c:
        rename_map[c] = cu
df = df.rename(columns=rename_map)

# origin/dest from TRAJECT like "Shl_Ledn" if not provided
if "origin" not in df.columns or "dest" not in df.columns:
    if "TRAJECT" in df.columns and df["TRAJECT"].notna().any():
        def split_traj(x):
            x = str(x)
            if "_" in x:
                a, b = x.split("_", 1)
                return pd.Series([a, b])
            return pd.Series([np.nan, np.nan])
        od = df["TRAJECT"].apply(split_traj)
        df["origin"] = od[0].fillna("UNK")
        df["dest"] = od[1].fillna("UNK")
    else:
        df["origin"] = "UNK"
        df["dest"] = "UNK"

# -------- Sidebar: user inputs (now inside a form) --------
with st.sidebar:
    st.markdown("### Select trip and time")
    with st.form("ns_input_form", clear_on_submit=False):
        origins = sorted(df["origin"].dropna().unique().tolist())
        dests = sorted(df["dest"].dropna().unique().tolist())
        origin = st.selectbox("From (origin)", origins, index=0)
        dest = st.selectbox("To (destination)", dests, index=min(1, len(dests)-1))

        manual_time = st.checkbox("Enter a specific time?")
        if manual_time:
            date_sel = st.date_input("Date", value=datetime(2024,6,10).date())
            time_sel = st.time_input("Time", value=dtime(8, 0))
            peak_flag = None
        else:
            date_sel = st.date_input("Date", value=datetime(2024,6,10).date())
            time_sel = None
            peak_flag = st.selectbox("Time of day", ["Ochtendspits","Avondspits","Daluren","Weekend"], index=0)

        st.markdown("---")
        st.markdown("### Train & Disruptions")

        train_types_known = sorted([x for x in df.get("TRAIN_TYPE", pd.Series(dtype=str)).dropna().unique().tolist()])
        train_type = st.selectbox("Train type", ["(keep data value)"] + train_types_known, index=0)

        delay_cats = [
            "No delay (0-1 min)", "Small (1–5 min)", "Medium (5–10 min)",
            "Large (10–30 min)", "Very Large (+30min)"
        ]
        delay_cat_sel = st.selectbox("Delay category", ["(keep data value)"] + delay_cats, index=0)

        disruption_cats = ["No Disruption", "Small (0–1 h)", "Medium (1–2 h)", "Large (>2 h)"]
        disruption_cat_sel = st.selectbox("Disruption category", ["(keep data value)"] + disruption_cats, index=0)

        extra_train_sel = st.selectbox("Is this an extra train?", ["(keep data value)", "No", "Yes"], index=0)

        prev_status = st.selectbox(
            "Previous train status",
            ["(keep data value)", "None", "Cancelled", "Delayed", "Cancelled & Delayed"],
            index=0
        )
        prev_delay_cat = None
        if prev_status in {"Delayed", "Cancelled & Delayed"}:
            prev_delay_cat = st.selectbox("Previous train delay category", delay_cats, index=1)

        submitted = st.form_submit_button("▶️ Run prediction", use_container_width=True)

# Gate the rest of the app on the button
if not submitted:
    st.info("Fill the inputs on the left and click **Run prediction**.")
    st.stop()

# ========= Helpers =========
def time_to_dagdeel(weekday: int, tt: dtime):
    if weekday >= 5:  # Sat/Sun
        return "Weekend"
    if dtime(7,0) <= tt < dtime(10,0):
        return "Ochtendspits"
    if dtime(16,0) <= tt < dtime(19,0):
        return "Avondspits"
    return "Daluren"

REP_MIN = {  # representative minutes for "Previous train delayed (min)"
    "No delay (0-1 min)": 0.5, "Small (1–5 min)": 3.0, "Medium (5–10 min)": 7.5,
    "Large (10–30 min)": 20.0, "Very Large (+30min)": 45.0,
}

# ========= Build anchor row OR historical median fallback =========
if time_sel is not None:
    dagdeel = time_to_dagdeel(datetime.combine(date_sel, time_sel).weekday(), time_sel)
else:
    dagdeel = peak_flag or "Daluren"

def historical_median_row(df, origin, dest, dagdeel=None, train_type=None):
    scope = df[(df["origin"] == origin) & (df["dest"] == dest)].copy()
    if dagdeel and "DAGDEELTREIN" in scope.columns:
        scope = scope[scope["DAGDEELTREIN"].astype(str).str.lower() == dagdeel.lower()]
    if train_type and "TRAIN_TYPE" in scope.columns:
        scope = scope[scope["TRAIN_TYPE"] == train_type]
    if scope.empty:
        return None
    synth = {}
    synth["DAGDEELTREIN"] = dagdeel or (scope["DAGDEELTREIN"].mode().iloc[0] if "DAGDEELTREIN" in scope else "Daluren")
    synth["TRAIN_TYPE"]   = train_type or (scope["TRAIN_TYPE"].mode().iloc[0] if "TRAIN_TYPE" in scope else None)
    for k in ["PROGNOSE_REIZEN","Previous train delayed (min)"]:
        if k in scope: synth[k] = float(scope[k].median())
    for k in ["Cancelled","ExtraTrain","Previous train canceled","disrupt_any","Extra train added before departure"]:
        if k in scope: synth[k] = int(round(scope[k].median()))
    for k_src, k_dst in [("delay_category","DELAY_CATEGORY"), ("disruption_category","DISRUPTION_CATEGORY")]:
        if k_src in scope and scope[k_src].notna().any():
            synth[k_dst] = scope[k_src].mode().iloc[0]
    synth["_ACTUAL_MEDIAN"] = float(scope["REALISATIE"].median()) if "REALISATIE" in scope else np.nan
    return pd.Series(synth)

# Filter for chosen OD/time bucket
df_choice = df[(df["origin"] == origin) & (df["dest"] == dest)].copy()
if "DAGDEELTREIN" in df_choice.columns:
    df_choice = df_choice[df_choice["DAGDEELTREIN"].astype(str).str.lower() == dagdeel.lower()]
if train_type != "(keep data value)" and "TRAIN_TYPE" in df_choice.columns:
    df_choice = df_choice[df_choice["TRAIN_TYPE"] == train_type]

used_synthetic = False
if df_choice.empty:
    s = historical_median_row(df, origin, dest, dagdeel=dagdeel,
                              train_type=(train_type if train_type!="(keep data value)" else None))
    if s is None:
        st.error("No historical data for this route/time. Try another selection.")
        st.stop()
    row = s
    ns_forecast = float(row.get("PROGNOSE_REIZEN", np.nan))
    actual = float(row.get("_ACTUAL_MEDIAN", np.nan))
    used_synthetic = True
else:
    row = df_choice.sort_index().iloc[-1]  # deterministic anchor
    ns_forecast = float(row.get("PROGNOSE_REIZEN", np.nan))
    actual = float(row.get("REALISATIE", np.nan)) if "REALISATIE" in df.columns else np.nan

if used_synthetic:
    st.caption("ⓘ No exact historical run matched your selection. Using a **historical median** row for this OD/time bucket.")

# ========= Build one feature row (+ overrides), LIMITED to trained features =========
raw = {}
for c in REQUIRED_RAW_FEATURES:
    if c in df.columns:
        raw[c] = row.get(c)
    elif c == "DELAY_CATEGORY" and "delay_category" in df.columns:
        raw["DELAY_CATEGORY"] = row.get("delay_category")
    elif c == "DISRUPTION_CATEGORY" and "disruption_category" in df.columns:
        raw["DISRUPTION_CATEGORY"] = row.get("disruption_category")

raw["DAGDEELTREIN"] = dagdeel
raw.setdefault("TRAIN_TYPE", row.get("TRAIN_TYPE", None))
raw.setdefault("PROGNOSE_REIZEN", 0.0 if np.isnan(ns_forecast) else ns_forecast)

for k, default in [
    ("Cancelled", 0), ("ExtraTrain", 0), ("Previous train canceled", 0),
    ("Previous train delayed (min)", 0.0), ("disrupt_any", 0), ("Extra train added before departure", 0),
]:
    raw.setdefault(k, default)

# UI overrides (only features your MLP knows)
if train_type != "(keep data value)":
    raw["TRAIN_TYPE"] = train_type
if delay_cat_sel != "(keep data value)":
    raw["DELAY_CATEGORY"] = delay_cat_sel
if disruption_cat_sel != "(keep data value)":
    raw["DISRUPTION_CATEGORY"] = disruption_cat_sel
if extra_train_sel != "(keep data value)":
    raw["ExtraTrain"] = 1 if extra_train_sel == "Yes" else 0

if prev_status != "(keep data value)":
    if prev_status == "None":
        raw["Previous train canceled"] = 0
        raw["Previous train delayed (min)"] = 0.0
    elif prev_status == "Cancelled":
        raw["Previous train canceled"] = 1
        raw["Previous train delayed (min)"] = 0.0
    elif prev_status == "Delayed":
        raw["Previous train canceled"] = 0
        raw["Previous train delayed (min)"] = float(REP_MIN.get(prev_delay_cat, 3.0))
    elif prev_status == "Cancelled & Delayed":
        raw["Previous train canceled"] = 1
        raw["Previous train delayed (min)"] = float(REP_MIN.get(prev_delay_cat, 3.0))

raw = {k: raw.get(k, 0 if k not in {"DAGDEELTREIN","TRAIN_TYPE","DELAY_CATEGORY","DISRUPTION_CATEGORY"} else None)
       for k in REQUIRED_RAW_FEATURES}

X_raw = pd.DataFrame([raw])

# ========= Load artifacts + predict (NumPy-only scaling preferred) =========
missing = []
model_path = next((p for p in MODEL_CANDIDATES if p.exists()), None)
if model_path is None:
    missing.append("mlp_model.h5 / mlp_model.keras")
if not FEATCOLS_PATH.exists():
    missing.append("feature_columns.json")

scale_npz = ART_DIR / "scaler_arrays.npz"
scaler_pkl = ART_DIR / "scaler.pkl"
use_numpy_scaler = scale_npz.exists()
if not use_numpy_scaler and not scaler_pkl.exists():
    missing.append("scaler_arrays.npz or scaler.pkl")

mlp_pred = None
mlp_msg = ""
mlp_ok = False

if missing:
    mlp_msg = "Missing in 'artifacts_for demo': " + ", ".join(missing)
else:
    try:
        # Prefer standalone keras if present; else use tf.keras (works in Geospatial)
        try:
            import keras as kapi
        except Exception:
            from tensorflow import keras as kapi

        model = kapi.models.load_model(model_path, compile=False)

        feat_cols = json.loads(FEATCOLS_PATH.read_text())
        cat_cols = [c for c in ["DAGDEELTREIN","TRAIN_TYPE","DELAY_CATEGORY","DISRUPTION_CATEGORY"] if c in X_raw.columns]
        X_enc = pd.get_dummies(X_raw, columns=cat_cols, drop_first=True)
        X_enc = X_enc.reindex(columns=feat_cols, fill_value=0.0).astype("float32")

        if use_numpy_scaler:
            data = np.load(scale_npz)
            scale_vec = data["scale"].astype("float32")
            scale_vec = np.where(scale_vec == 0.0, 1.0, scale_vec)
            X_scaled = X_enc.values / scale_vec
        else:
            import joblib
            scaler = joblib.load(scaler_pkl)
            X_scaled = scaler.transform(X_enc.values)

        y_hat = model.predict(X_scaled, verbose=0).ravel()
        mlp_pred = float(y_hat[0])
        mlp_ok = True
    except Exception as e:
        mlp_msg = f"Could not run MLP prediction: {type(e).__name__}: {e}"

# ========= Error metrics vs actual =========
def _ae(pred, truth): return abs(float(pred) - float(truth))
def _mape(pred, truth):
    denom = max(1.0, abs(float(truth)))
    return 100.0 * abs(float(pred) - float(truth)) / denom

err_ns = err_mlp = mape_ns = mape_mlp = None
if not np.isnan(actual):
    if not np.isnan(ns_forecast):
        err_ns = _ae(ns_forecast, actual); mape_ns = _mape(ns_forecast, actual)
    if mlp_pred is not None:
        err_mlp = _ae(mlp_pred, actual); mape_mlp = _mape(mlp_pred, actual)

# ========= UI: results =========
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Selected trip")
    st.markdown(
        f"""
        <div class="card">
            <div style="font-weight:800;font-size:1.1rem;color:{NS_BLUE};">{origin} ➜ {dest}</div>
            <div style="margin-top:6px;">
                <span class="metric-sub">Time of day:</span> <b>{raw.get('DAGDEELTREIN','(n/a)')}</b><br/>
                <span class="metric-sub">Train type:</span> <b>{raw.get('TRAIN_TYPE','(n/a)')}</b><br/>
                <span class="metric-sub">Delay category:</span> <b>{raw.get('DELAY_CATEGORY','(n/a)')}</b><br/>
                <span class="metric-sub">Disruption category:</span> <b>{raw.get('DISRUPTION_CATEGORY','(n/a)')}</b><br/>
                <span class="metric-sub">Extra train:</span> <b>{'Yes' if int(raw.get('ExtraTrain',0) or 0)==1 else 'No'}</b><br/>
                <span class="metric-sub">Prev. cancelled:</span> <b>{int(raw.get('Previous train canceled',0) or 0)}</b><br/>
                <span class="metric-sub">Prev. delay (min):</span> <b>{float(raw.get('Previous train delayed (min)',0) or 0):.1f}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown("#### Prediction comparison")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""
            <div class="card">
              <div class="metric-title">NS baseline (PROGNOSE_REIZEN)</div>
              <div class="metric-value">{'—' if np.isnan(ns_forecast) else int(round(ns_forecast))}</div>
              <div class="metric-sub">passengers</div>
              {("" if err_ns is None else f"<div class='metric-sub' style='margin-top:6px;'>AE: {int(round(err_ns))} | MAPE: {round(mape_ns,1)}%</div>")}
            </div>
            """,
            unsafe_allow_html=True
        )
    with c2:
        if mlp_ok and mlp_pred is not None:
            st.markdown(
                f"""
                <div class="card">
                  <div class="metric-title">Our MLP prediction</div>
                  <div class="metric-value">{int(round(mlp_pred))}</div>
                  <div class="metric-sub">passengers</div>
                  {("" if err_mlp is None else f"<div class='metric-sub' style='margin-top:6px;'>AE: {int(round(err_mlp))} | MAPE: {round(mape_mlp,1)}%</div>")}
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="card">
                  <div class="metric-title">Our MLP prediction</div>
                  <div class="metric-value">—</div>
                  <div class="metric-sub">{mlp_msg}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
    with c3:
        st.markdown(
            f"""
            <div class="card">
              <div class="metric-title">Actual (REALISATIE)</div>
              <div class="metric-value">{'—' if np.isnan(actual) else int(round(actual))}</div>
              <div class="metric-sub">passengers</div>
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown("### Feature row used for prediction")
st.dataframe(X_raw.T)

st.markdown("---")
st.caption("Notes: If no exact historical run matched your selection, we use a historical median row for this OD/time bucket. Categories you pick here override the data row for the MLP; NS baseline always reads PROGNOSE_REIZEN from the data.")
