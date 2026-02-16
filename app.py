import json
import re
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st  # type: ignore
import plotly.graph_objects as go  # type: ignore


# =========================================================
# Config APP (dark background)
# =========================================================
st.set_page_config(
    page_title="ETH Price Up Predictor",
    page_icon=None,
    layout="wide",
)

st.markdown(
    """
    <style>
      /* =========================
         Base dark
         ========================= */
      .stApp { background-color: #0e1117 !important; color: #ffffff !important; }
      section.main > div { background-color: #0e1117 !important; }

      section[data-testid="stSidebar"] { background-color: #111827 !important; }
      section[data-testid="stSidebar"] * { color: #ffffff !important; }

      h1,h2,h3,h4,h5,h6,p,span,label,div { color: #ffffff; }

      /* =========================
         EXPANDER: hard override (header + open state)
         ========================= */
      div[data-testid="stExpander"] {
        background: #0b1220 !important;
        border: 1px solid rgba(255,255,255,0.10) !important;
        border-radius: 12px !important;
        overflow: hidden !important;
      }

      div[data-testid="stExpander"] details { background: #0b1220 !important; }

      div[data-testid="stExpander"] summary {
        background: #0b1220 !important;
        color: #ffffff !important;
        padding: 12px 14px !important;
        font-weight: 700 !important;
        border: none !important;
        outline: none !important;
      }

      div[data-testid="stExpander"] summary,
      div[data-testid="stExpander"] summary * ,
      div[data-testid="stExpander"] summary div,
      div[data-testid="stExpander"] summary span,
      div[data-testid="stExpander"] summary p {
        background-color: #0b1220 !important;
        color: #ffffff !important;
      }

      div[data-testid="stExpander"] summary:hover,
      div[data-testid="stExpander"] summary:focus,
      div[data-testid="stExpander"] summary:active {
        background-color: #0b1220 !important;
        color: #ffffff !important;
      }

      div[data-testid="stExpander"] details[open] > summary {
        background-color: #0b1220 !important;
        color: #ffffff !important;
        border-bottom: 1px solid rgba(255,255,255,0.08) !important;
      }

      div[data-testid="stExpander"] details > div { background: #0b1220 !important; }

      /* =========================
         File uploader + download button
         ========================= */
      div[data-testid="stFileUploader"] section {
        background: #0b1220 !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 10px !important;
      }
      div[data-testid="stFileUploader"] section * { color: #ffffff !important; }
      div[data-testid="stFileUploader"] button {
        background: #111827 !important;
        color: #ffffff !important;
        border: 1px solid rgba(255,255,255,0.20) !important;
        border-radius: 10px !important;
      }

      [data-testid="stFileUploaderFile"] {
        background: #0b1220 !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 10px !important;
        padding: 8px !important;
      }
      [data-testid="stFileUploaderFile"] * { color: #ffffff !important; }
      [data-testid="stFileUploaderDeleteBtn"] svg { fill: #ffffff !important; }

      div[data-testid="stDownloadButton"] button {
        background: #111827 !important;
        color: #ffffff !important;
        border: 1px solid rgba(255,255,255,0.20) !important;
        border-radius: 10px !important;
      }

      /* =========================
         Hide Streamlit chrome
         ========================= */
      #MainMenu { visibility: hidden; }
      footer { visibility: hidden; }

            /* =========================
    Hide / Theme Streamlit chrome (fix white bar)
    ========================= */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    /* Quita la franja superior (la que suele verse blanca o con color) */
    div[data-testid="stDecoration"] {
    display: none !important;
    }

    /* Mantén el header para que exista el botón de sidebar,
    pero con fondo oscuro (sin blanco) */
    header[data-testid="stHeader"] {
    visibility: visible !important;
    background: #0e1117 !important;
    box-shadow: none !important;
    }

    /* Asegura que lo interno del header no pinte blanco */
    header[data-testid="stHeader"] * {
    background: transparent !important;
    }

    /* Toolbar superior (a veces queda clara) */
    div[data-testid="stToolbar"] {
    background: #0e1117 !important;
    }

    /* Contenedores base (por si alguno queda blanco) */
    html, body,
    div[data-testid="stAppViewContainer"],
    div[data-testid="stMain"],
    section.main {
    background: #0e1117 !important;
    }


    /* TOP HEADER (fix white bar) */
    header[data-testid="stHeader"]{
    visibility: visible !important;
    background: #0e1117 !important;
    border-bottom: 1px solid rgba(255,255,255,0.08) !important;
    }
    div[data-testid="stToolbar"]{
    background: #0e1117 !important;
    }

.block-container { padding-top: 1.0rem; }
html, body { background: #0e1117 !important; }


      .block-container { padding-top: 1.0rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ====================
# Paths
# ====================
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH  = BASE_DIR / "models" / "final_pipeline_naive_bayes.pkl"
META_PATH   = BASE_DIR / "models" / "final_pipeline_naive_bayes_metadata.pkl"
PRICES_PATH = BASE_DIR / "data" / "eth_prices_binance_1h.pkl"


# ====================
# Loaders
# ====================
@st.cache_resource
def load_pipeline(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_metadata(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_prices_1h(path: Path) -> pd.DataFrame:
    df = pd.read_pickle(path)
    df["open_time"] = pd.to_datetime(df["open_time"], errors="coerce", utc=True)
    df["date_hour"] = df["open_time"].dt.floor("h")

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return (
        df.dropna(subset=["date_hour", "close"])
        .sort_values("date_hour")
        .reset_index(drop=True)
    )


# =========================================================
# Parse helpers
# =========================================================
def parse_amount(amount_str):
    if amount_str is None:
        return (np.nan, None)

    s = str(amount_str).strip()
    if not s:
        return (np.nan, None)

    parts = s.split()
    unit = parts[-1]
    num = " ".join(parts[:-1])
    num = re.sub(r"[,\s]", "", num)

    try:
        return (float(num), unit)
    except ValueError:
        return (np.nan, unit)


def parse_usd(value_str):
    if value_str is None:
        return np.nan
    s = str(value_str).replace("$", "").replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


def load_uploaded_json_files(uploaded_files) -> pd.DataFrame:
    rows = []
    for uf in uploaded_files:
        raw = uf.read()
        data = json.loads(raw.decode("utf-8"))

        if not isinstance(data, list):
            raise ValueError(f"El archivo {uf.name} no contiene una lista JSON (array).")

        for record in data:
            r = dict(record)
            r["_source_file"] = uf.name

            if "sender_lable" in r and "sender_label" not in r:
                r["sender_label"] = r.pop("sender_lable")
            if "receiver_lable" in r and "receiver_label" not in r:
                r["receiver_label"] = r.pop("receiver_lable")

            rows.append(r)

    df = pd.DataFrame(rows)
    df["date_time"] = pd.to_datetime(df.get("date_time"), errors="coerce", utc=True)

    if "amount" in df.columns:
        parsed = df["amount"].apply(parse_amount)
        df["amount_value"] = parsed.apply(lambda x: x[0])
    else:
        df["amount_value"] = np.nan

    if "value" in df.columns:
        df["value_usd"] = df["value"].apply(parse_usd)
    else:
        if "value_usd" in df.columns:
            df["value_usd"] = pd.to_numeric(df["value_usd"], errors="coerce")
        else:
            df["value_usd"] = np.nan

    df["date_hour"] = df["date_time"].dt.floor("h")

    subset_cols = [c for c in ["tx_hash", "date_time", "sender", "receiver"] if c in df.columns]
    if subset_cols:
        df = df.drop_duplicates(subset=subset_cols, keep="first").reset_index(drop=True)

    return df


def aggregate_onchain_1h(df_trx: pd.DataFrame) -> pd.DataFrame:
    if "date_hour" not in df_trx.columns:
        raise ValueError("Falta columna date_hour en transacciones.")

    tx_col = "tx_hash" if "tx_hash" in df_trx.columns else None
    sender_col = "sender" if "sender" in df_trx.columns else None
    receiver_col = "receiver" if "receiver" in df_trx.columns else None

    agg_dict = {}
    agg_dict["trx_count"] = (tx_col, "count") if tx_col else ("date_hour", "count")
    agg_dict["unique_senders"] = (sender_col, "nunique") if sender_col else ("date_hour", "size")
    agg_dict["unique_receivers"] = (receiver_col, "nunique") if receiver_col else ("date_hour", "size")
    agg_dict["total_amount"] = ("amount_value", "sum")
    agg_dict["total_value_usd"] = ("value_usd", "sum")

    trx_1h = (
        df_trx.groupby("date_hour")
        .agg(**agg_dict)
        .reset_index()
        .sort_values("date_hour")
        .reset_index(drop=True)
    )

    trx_1h["total_amount"] = trx_1h["total_amount"].fillna(0)
    trx_1h["total_value_usd"] = trx_1h["total_value_usd"].fillna(0)
    trx_1h["log_total_amount"] = np.log1p(trx_1h["total_amount"])
    trx_1h["log_total_value_usd"] = np.log1p(trx_1h["total_value_usd"])

    return trx_1h


def build_model_dataset(trx_1h: pd.DataFrame, prices_1h: pd.DataFrame) -> pd.DataFrame:
    price_cols = ["date_hour", "open", "high", "low", "close", "volume"]
    price_1h = prices_1h[price_cols].copy()

    dfm = trx_1h.merge(price_1h, on="date_hour", how="inner")
    dfm = dfm.sort_values("date_hour").reset_index(drop=True)

    needed = [
        "trx_count",
        "unique_senders",
        "unique_receivers",
        "log_total_amount",
        "log_total_value_usd",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for c in needed:
        dfm[c] = pd.to_numeric(dfm[c], errors="coerce")

    dfm = dfm.dropna(subset=needed).reset_index(drop=True)
    return dfm


# =========================================================
# Dark table renderer (NO st.dataframe) + SCROLL interno
# =========================================================
def render_dark_table(
    df: pd.DataFrame,
    max_rows: int = 200,
    max_height_px: int = 420,
    min_col_width_px: int = 110,
):
    df_show = df.head(max_rows).copy()

    styler = (
        df_show.style
        .set_table_styles([
            {"selector": "table", "props": [
                ("width", "100%"),
                ("border-collapse", "collapse"),
                ("background-color", "#0b1220"),
            ]},
            {"selector": "thead th", "props": [
                ("background-color", "#111827"),
                ("color", "white"),
                ("border", "1px solid rgba(255,255,255,0.12)"),
                ("padding", "10px"),
                ("font-weight", "700"),
                ("text-align", "left"),
                ("white-space", "nowrap"),
                ("position", "sticky"),
                ("top", "0"),
                ("z-index", "2"),
                ("min-width", f"{min_col_width_px}px"),
            ]},
            {"selector": "tbody td", "props": [
                ("background-color", "#0b1220"),
                ("color", "white"),
                ("border", "1px solid rgba(255,255,255,0.08)"),
                ("padding", "8px"),
                ("white-space", "nowrap"),
                ("min-width", f"{min_col_width_px}px"),
            ]},
            {"selector": "tbody tr:hover td", "props": [
                ("background-color", "#0f1a2e"),
            ]},
        ])
        .hide(axis="index")
    )

    table_html = styler.to_html()

    st.markdown(
        f"""
        <div style="
            max-height:{max_height_px}px;
            overflow-y:auto;
            overflow-x:auto;
            border:1px solid rgba(255,255,255,0.12);
            border-radius:12px;
            background:#0b1220;
            padding:6px;
        ">
          {table_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# Plotly chart (ONLY PRICE)
# =========================================================
def plotly_price_line(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["date_hour"],
            y=df["close"],
            mode="lines",
            name="ETH Close",
            line=dict(color="#00e676", width=4),
        )
    )
    fig.update_layout(
        template="plotly_dark",
        title="ETH Price (Close) vs tiempo",
        xaxis_title="date_hour",
        yaxis_title="close (USD)",
        paper_bgcolor="#0b1220",
        plot_bgcolor="#0b1220",
        height=480,
        margin=dict(l=40, r=20, t=60, b=40),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
    )
    return fig


# ======================
# App
# ======================
st.title("ETH Price Up Predictor (offline)")
st.caption("Sube JSONs (Apify), se agregan por hora, se unen con precios 1H y se predice P(UP).")

st.sidebar.header("Indice")
section = st.sidebar.radio(
    "Ir a",
    options=[
        "Informacion del modelo",
        "Vista previa (features)",
        "Tabla de predicciones",
        "Descargar CSV",
        "Grafico",
    ],
    index=0,
)

st.sidebar.header("Carga de datos (Apify JSON)")
uploaded = st.sidebar.file_uploader(
    "Sube uno o varios JSON (array de items)",
    type=["json"],
    accept_multiple_files=True,
)

try:
    pipeline = load_pipeline(MODEL_PATH)
    meta = load_metadata(META_PATH)
except Exception as e:
    st.error(f"No pude cargar modelo/metadata desde Downloads: {e}")
    st.stop()

FEATURES = meta["features"]

try:
    df_prices = load_prices_1h(PRICES_PATH)
except Exception as e:
    st.error(f"No pude cargar precios 1H desde {PRICES_PATH}: {e}")
    st.stop()

if not uploaded:
    with st.expander("Informacion del modelo", expanded=True):
        st.write(f"Modelo: {meta.get('model')}")
        st.write(f"Features: {FEATURES}")
        st.write(f"Target: {meta.get('target')}")
        st.write(f"Rango entrenamiento: {meta.get('date_min')} → {meta.get('date_max')}")
        st.write(f"Precios 1H disponibles: {len(df_prices):,} filas")
    st.info("Sube tus JSONs para construir el dataset y predecir.")
    st.stop()

try:
    with st.spinner("Leyendo JSONs..."):
        df_trx = load_uploaded_json_files(uploaded)

    with st.spinner("Agregando on-chain por hora (1H)..."):
        trx_1h = aggregate_onchain_1h(df_trx)

    with st.spinner("Uniendo con precios 1H y preparando features..."):
        df_model = build_model_dataset(trx_1h, df_prices)

except Exception as e:
    st.error(f"Error procesando JSONs: {e}")
    st.stop()

X_pred = df_model[FEATURES].copy()
pred = pipeline.predict(X_pred).astype(int)
proba = pipeline.predict_proba(X_pred)[:, 1].astype(float)

df_out = df_model[["date_hour", "close"]].copy()
df_out["pred_price_up"] = pred
df_out["pred_proba_up"] = proba
df_out = df_out.sort_values("date_hour").reset_index(drop=True)
df_out["date_hour"] = df_out["date_hour"].dt.tz_convert(None)

c1, c2, c3 = st.columns(3)
c1.metric("Transacciones crudas", f"{len(df_trx):,}")
c2.metric("Horas agregadas (on-chain)", f"{len(trx_1h):,}")
c3.metric("Filas para modelo (merge)", f"{len(df_model):,}")

with st.expander("Informacion del modelo", expanded=(section == "Informacion del modelo")):
    st.write(f"Modelo: {meta.get('model')}")
    st.write(f"Features: {FEATURES}")
    st.write(f"Target: {meta.get('target')}")
    st.write(f"Rango entrenamiento: {meta.get('date_min')} → {meta.get('date_max')}")
    st.write(f"Precios 1H disponibles: {len(df_prices):,} filas")

with st.expander("Vista previa (features para el modelo)", expanded=(section == "Vista previa (features)")):
    render_dark_table(df_model, max_rows=30, max_height_px=320)

with st.expander("Predicciones (todas las filas)", expanded=(section == "Tabla de predicciones")):
    render_dark_table(
        df_out[["date_hour", "close", "pred_price_up", "pred_proba_up"]].tail(200),
        max_rows=200,
        max_height_px=520,
    )

with st.expander("Descargar CSV", expanded=(section == "Descargar CSV")):
    csv_bytes = df_out.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Descargar predicciones CSV",
        data=csv_bytes,
        file_name="predicciones_price_up_offline.csv",
        mime="text/csv",
    )

with st.expander("Grafico (Plotly)", expanded=(section == "Grafico")):
    if len(df_out) < 2:
        st.warning("No hay suficientes puntos para graficar.")
    else:
        st.plotly_chart(plotly_price_line(df_out), use_container_width=True)














