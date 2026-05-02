"""
╔══════════════════════════════════════════════════════════════════╗
║          DATAFORGE — Autonomous Data Analysis Platform           ║
║          Production-grade single-file Streamlit app              ║
╚══════════════════════════════════════════════════════════════════╝
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io, base64, json, re, textwrap
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DataForge · Autonomous Analysis",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap');

:root {
  --primary:   #0f2d5e;
  --primary-l: #1a4480;
  --teal:      #0d9488;
  --teal-l:    #14b8a6;
  --orange:    #f97316;
  --orange-l:  #fb923c;
  --bg:        #050d1a;
  --bg2:       #0a1628;
  --bg3:       #0f2040;
  --border:    #1e3a5f;
  --text:      #e2e8f0;
  --text2:     #94a3b8;
  --success:   #22c55e;
  --warning:   #eab308;
  --danger:    #ef4444;
  --radius:    10px;
}

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}
.stApp { background: var(--bg); }
.block-container { padding: 1.5rem 2rem; max-width: 1600px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'Syne', sans-serif;
    color: var(--teal-l);
}

/* ── Headers ── */
h1, h2, h3, h4 { font-family: 'Syne', sans-serif !important; }
h1 { font-size: 2.2rem !important; font-weight: 800 !important; letter-spacing: -0.5px; }
h2 { font-size: 1.5rem !important; font-weight: 700 !important; color: var(--teal-l) !important; }
h3 { font-size: 1.15rem !important; font-weight: 600 !important; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: var(--bg2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1rem !important;
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] { color: var(--text2) !important; font-size: .75rem !important; text-transform: uppercase; letter-spacing: .08em; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { font-family: 'Syne', sans-serif !important; font-size: 1.8rem !important; color: var(--teal-l) !important; }

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: var(--bg2);
    border-radius: var(--radius);
    gap: 4px;
    padding: 4px;
    border: 1px solid var(--border);
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    font-family: 'Syne', sans-serif;
    font-size: .82rem;
    font-weight: 600;
    letter-spacing: .05em;
    color: var(--text2);
    border-radius: 7px;
    padding: 6px 14px;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: var(--primary-l) !important;
    color: #fff !important;
}

/* ── Buttons ── */
.stButton > button {
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    border: none !important;
    background: var(--primary-l) !important;
    color: #fff !important;
    transition: all .2s ease !important;
}
.stButton > button:hover { background: var(--teal) !important; transform: translateY(-1px); box-shadow: 0 4px 15px rgba(13,148,136,.3) !important; }

/* Primary CTA buttons */
.cta-btn > button {
    background: linear-gradient(135deg, var(--teal), var(--teal-l)) !important;
    font-size: 1rem !important;
    padding: .65rem 1.5rem !important;
}
.cta-btn-orange > button {
    background: linear-gradient(135deg, var(--orange), var(--orange-l)) !important;
    font-size: 1rem !important;
    padding: .65rem 1.5rem !important;
}

/* ── Cards ── */
.df-card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}
.df-card-accent { border-left: 3px solid var(--teal-l); }
.df-card-warn   { border-left: 3px solid var(--orange); }
.df-card-danger { border-left: 3px solid var(--danger); }
.df-card-ok     { border-left: 3px solid var(--success); }

/* ── Badge ── */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 99px;
    font-size: .72rem;
    font-weight: 600;
    font-family: 'DM Mono', monospace;
    letter-spacing: .04em;
}
.badge-teal   { background: rgba(20,184,166,.15); color: var(--teal-l); border: 1px solid rgba(20,184,166,.3); }
.badge-orange { background: rgba(249,115,22,.15);  color: var(--orange-l); border: 1px solid rgba(249,115,22,.3); }
.badge-red    { background: rgba(239,68,68,.15);   color: #f87171; border: 1px solid rgba(239,68,68,.3); }
.badge-green  { background: rgba(34,197,94,.15);   color: #4ade80; border: 1px solid rgba(34,197,94,.3); }
.badge-gray   { background: rgba(148,163,184,.1);  color: var(--text2); border: 1px solid rgba(148,163,184,.2); }

/* ── Insight box ── */
.insight-item {
    background: var(--bg3);
    border-radius: 8px;
    padding: .6rem 1rem;
    margin: .3rem 0;
    font-size: .88rem;
    border-left: 3px solid var(--teal-l);
    color: var(--text);
}
.insight-item.warn { border-left-color: var(--orange); }
.insight-item.danger { border-left-color: var(--danger); }
.insight-item.ok   { border-left-color: var(--success); }

/* ── Health score ring ── */
.health-ring {
    text-align: center;
    padding: 1.5rem;
    background: var(--bg2);
    border-radius: var(--radius);
    border: 1px solid var(--border);
}
.health-score {
    font-family: 'Syne', sans-serif;
    font-size: 4rem;
    font-weight: 800;
    background: linear-gradient(135deg, var(--teal-l), var(--orange-l));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* ── DataFrame ── */
[data-testid="stDataFrame"] { border-radius: var(--radius); overflow: hidden; }
.stDataFrame { background: var(--bg2); }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: var(--bg2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}

/* ── Selectbox / Input ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div,
[data-testid="stTextInput"] > div > div > input {
    background: var(--bg3) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    border-radius: 7px !important;
}

/* ── Code block ── */
pre, code { font-family: 'DM Mono', monospace !important; font-size: .82rem !important; }
.stCodeBlock { border-radius: var(--radius) !important; }

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

/* ── Logo ── */
.logo-wrap {
    display: flex;
    align-items: center;
    gap: .6rem;
    padding: 1rem 0 1.5rem;
}
.logo-icon {
    font-size: 1.8rem;
    background: linear-gradient(135deg,var(--teal),var(--orange));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.logo-text {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    color: #fff;
}
.logo-sub { font-size: .65rem; color: var(--text2); letter-spacing: .12em; text-transform: uppercase; }

/* ── Progress bar override ── */
[data-testid="stProgressBar"] > div { background: linear-gradient(90deg, var(--teal), var(--orange)) !important; border-radius: 99px !important; }

/* ── Model card ── */
.model-card {
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin: .4rem 0;
    transition: border-color .2s;
}
.model-card:hover { border-color: var(--teal); }
.model-name { font-family: 'Syne', sans-serif; font-weight: 700; font-size: 1rem; color: #fff; }
.model-desc { font-size: .82rem; color: var(--text2); margin-top: .2rem; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg2); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--primary-l); }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────

def init_state():
    defaults = {
        "df": None,
        "df_clean": None,
        "df_ml": None,
        "filename": None,
        "cleaning_log": [],
        "insights": [],
        "health_score": None,
        "health_details": {},
        "auto_ran": False,
        "label_col": None,
        "task_type": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ─────────────────────────────────────────────────────────────────────────────
# UTILITY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def badge(text, kind="gray"):
    return f'<span class="badge badge-{kind}">{text}</span>'

def insight_html(items):
    html = ""
    for it in items:
        cls = it.get("cls", "")
        html += f'<div class="insight-item {cls}">{"⚠️" if cls=="warn" else "🔴" if cls=="danger" else "✅" if cls=="ok" else "ℹ️"} {it["msg"]}</div>'
    return html

def df_mem(df):
    mem = df.memory_usage(deep=True).sum()
    for unit in ["B","KB","MB","GB"]:
        if mem < 1024: return f"{mem:.1f} {unit}"
        mem /= 1024
    return f"{mem:.1f} TB"

def to_csv_bytes(df):
    return df.to_csv(index=False).encode()

def to_download_link(data: bytes, filename: str, label: str, mime="text/csv"):
    b64 = base64.b64encode(data).decode()
    return f'<a href="data:{mime};base64,{b64}" download="{filename}" style="text-decoration:none;"><button style="font-family:\'Syne\',sans-serif;font-weight:600;border-radius:8px;border:none;background:#1a4480;color:#fff;padding:8px 18px;cursor:pointer;font-size:.85rem;transition:all .2s;">{label}</button></a>'

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 1 — DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_data(uploaded_file):
    name = uploaded_file.name
    ext = name.rsplit(".", 1)[-1].lower()
    try:
        if ext == "csv":
            df = pd.read_csv(uploaded_file)
        elif ext in ("xlsx", "xls"):
            df = pd.read_excel(uploaded_file)
        elif ext == "json":
            df = pd.read_json(uploaded_file)
        else:
            st.error("Unsupported file format.")
            return None
        st.session_state.filename = name
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def display_dataset_overview(df):
    st.markdown("## 📋 Dataset Overview")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Missing", f"{df.isnull().sum().sum():,}")
    c4.metric("Duplicates", f"{df.duplicated().sum():,}")
    c5.metric("Memory", df_mem(df))

    with st.expander("🔍 Preview & Column Info", expanded=True):
        st.dataframe(df.head(20), use_container_width=True)
        info_rows = []
        for col in df.columns:
            info_rows.append({
                "Column": col,
                "Dtype": str(df[col].dtype),
                "Non-Null": df[col].notna().sum(),
                "Null%": f"{df[col].isnull().mean()*100:.1f}%",
                "Unique": df[col].nunique(),
                "Sample": str(df[col].dropna().iloc[0]) if df[col].notna().any() else "—",
            })
        st.dataframe(pd.DataFrame(info_rows), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 2 — AUTOMATIC ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def detect_column_types(df):
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=["object","category","bool"]).columns.tolist()
    dt_cols  = df.select_dtypes(include=["datetime64"]).columns.tolist()
    # Try to detect hidden datetime
    for c in cat_cols[:]:
        try:
            parsed = pd.to_datetime(df[c], infer_datetime_format=True, errors="coerce")
            if parsed.notna().mean() > .7:
                dt_cols.append(c)
                cat_cols.remove(c)
        except: pass
    return num_cols, cat_cols, dt_cols

def detect_outliers_iqr(df, cols):
    out = {}
    for c in cols:
        q1, q3 = df[c].quantile(.25), df[c].quantile(.75)
        iqr = q3 - q1
        mask = (df[c] < q1 - 1.5*iqr) | (df[c] > q3 + 1.5*iqr)
        out[c] = int(mask.sum())
    return out

def detect_skew(df, cols, thresh=1.0):
    return {c: df[c].skew() for c in cols if abs(df[c].skew()) > thresh and df[c].notna().sum() > 5}

def detect_high_corr(df, cols, thresh=0.85):
    if len(cols) < 2: return []
    corr = df[cols].corr().abs()
    pairs = []
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            if corr.iloc[i,j] >= thresh:
                pairs.append((cols[i], cols[j], round(corr.iloc[i,j],3)))
    return pairs

def detect_imbalance(df, col, thresh=3.0):
    vc = df[col].value_counts()
    if len(vc) < 2: return False, 0
    ratio = vc.iloc[0] / vc.iloc[-1] if vc.iloc[-1] > 0 else float("inf")
    return ratio > thresh, round(ratio, 2)

def generate_insights(df):
    num_cols, cat_cols, dt_cols = detect_column_types(df)
    insights = []

    # Missing
    miss = df.isnull().mean()
    for c, v in miss[miss > 0].items():
        cls = "danger" if v > .4 else "warn" if v > .15 else ""
        insights.append({"msg": f"<b>{c}</b> has {v*100:.1f}% missing values.", "cls": cls})

    # Duplicates
    d = df.duplicated().sum()
    if d > 0:
        insights.append({"msg": f"{d} duplicate rows detected ({d/len(df)*100:.1f}% of data).", "cls": "warn"})

    # Outliers
    out = detect_outliers_iqr(df, num_cols)
    for c, n in out.items():
        if n > 0:
            cls = "warn" if n/len(df) < .1 else "danger"
            insights.append({"msg": f"<b>{c}</b>: {n} outliers detected via IQR.", "cls": cls})

    # Skew
    sk = detect_skew(df, num_cols)
    for c, v in sk.items():
        insights.append({"msg": f"<b>{c}</b> is skewed (skewness={v:.2f}). Log transform may help.", "cls": "warn"})

    # Correlation
    hc = detect_high_corr(df, num_cols)
    for a, b, r in hc:
        insights.append({"msg": f"High correlation between <b>{a}</b> and <b>{b}</b> (r={r}). Consider dropping one.", "cls": "warn"})

    # Imbalance
    for c in cat_cols[:3]:
        if df[c].nunique() == 2:
            imb, ratio = detect_imbalance(df, c)
            if imb:
                insights.append({"msg": f"<b>{c}</b> appears imbalanced (majority:minority ratio = {ratio}:1).", "cls": "warn"})

    # Positives
    if d == 0:
        insights.append({"msg": "No duplicate rows found. ✓", "cls": "ok"})
    if not hc:
        insights.append({"msg": "No high-correlation feature pairs detected. ✓", "cls": "ok"})
    if miss[miss > .4].empty:
        insights.append({"msg": "No columns with >40% missing values. ✓", "cls": "ok"})

    return insights, num_cols, cat_cols, dt_cols

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 3 — EDA ENGINE
# ─────────────────────────────────────────────────────────────────────────────

PLOT_THEME = dict(
    plot_bgcolor="#0a1628",
    paper_bgcolor="#0a1628",
    font=dict(color="#e2e8f0", family="DM Sans"),
    margin=dict(l=40, r=30, t=45, b=40),
    colorway=["#14b8a6","#f97316","#6366f1","#22c55e","#eab308","#ec4899","#8b5cf6"],
)

def style_fig(fig, h=400):
    fig.update_layout(height=h, **PLOT_THEME)
    fig.update_xaxes(gridcolor="#1e3a5f", showline=False, tickfont=dict(size=11))
    fig.update_yaxes(gridcolor="#1e3a5f", showline=False, tickfont=dict(size=11))
    return fig

def perform_eda(df):
    num_cols, cat_cols, dt_cols = detect_column_types(df)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Summary", "🔥 Correlations", "🕳️ Missing", "📈 Distributions", "📦 Boxplots", "🗂️ Categorical"
    ])

    with tab1:
        st.markdown("### Summary Statistics")
        if num_cols:
            st.dataframe(df[num_cols].describe().T.round(4), use_container_width=True)
        else:
            st.info("No numerical columns found.")

        if cat_cols:
            st.markdown("### Categorical Summary")
            cat_info = []
            for c in cat_cols:
                vc = df[c].value_counts()
                cat_info.append({"Column": c, "Unique": df[c].nunique(), "Top Value": vc.index[0], "Top Count": vc.iloc[0], "Missing%": f"{df[c].isnull().mean()*100:.1f}%"})
            st.dataframe(pd.DataFrame(cat_info), use_container_width=True)

    with tab2:
        if len(num_cols) >= 2:
            corr = df[num_cols].corr()
            fig = px.imshow(
                corr, text_auto=".2f", aspect="auto",
                color_continuous_scale=[[0,"#ef4444"],[.5,"#0f2040"],[1,"#14b8a6"]],
                zmin=-1, zmax=1, title="Correlation Heatmap"
            )
            style_fig(fig, 500)
            fig.update_traces(textfont_size=10)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need at least 2 numerical columns for correlation heatmap.")

    with tab3:
        miss_s = df.isnull().mean().sort_values(ascending=False)
        miss_s = miss_s[miss_s > 0]
        if not miss_s.empty:
            fig = go.Figure(go.Bar(
                x=miss_s.values * 100, y=miss_s.index,
                orientation="h", marker_color="#f97316",
                text=[f"{v*100:.1f}%" for v in miss_s.values],
                textposition="outside",
            ))
            fig.update_layout(title="Missing Values by Column (%)", xaxis_title="Missing %", **PLOT_THEME, height=max(300, len(miss_s)*30+100))
            fig.update_xaxes(gridcolor="#1e3a5f"); fig.update_yaxes(gridcolor="#1e3a5f")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown('<div class="df-card df-card-ok">✅ <b>No missing values found in this dataset.</b></div>', unsafe_allow_html=True)

        # Missing heatmap (sample)
        if not miss_s.empty and len(df) <= 5000:
            miss_mat = df[miss_s.index].isnull().astype(int)
            fig2 = px.imshow(miss_mat.T, color_continuous_scale=[[0,"#0f2040"],[1,"#ef4444"]],
                             aspect="auto", title="Missing Value Heatmap (row sample)")
            style_fig(fig2, 300)
            st.plotly_chart(fig2, use_container_width=True)

    with tab4:
        if num_cols:
            sel = st.selectbox("Select column", num_cols, key="hist_col")
            c1, c2 = st.columns(2)
            with c1:
                fig = px.histogram(df, x=sel, nbins=40, title=f"Distribution: {sel}",
                                   color_discrete_sequence=["#14b8a6"])
                style_fig(fig, 350)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.ecdf(df, x=sel, title=f"ECDF: {sel}", color_discrete_sequence=["#f97316"])
                style_fig(fig, 350)
                st.plotly_chart(fig, use_container_width=True)

            if len(num_cols) >= 2:
                st.markdown("### Scatterplot")
                c1, c2, c3 = st.columns(3)
                sx = c1.selectbox("X axis", num_cols, key="sc_x")
                sy = c2.selectbox("Y axis", num_cols, index=min(1, len(num_cols)-1), key="sc_y")
                sc = c3.selectbox("Color by", ["None"] + cat_cols, key="sc_c")
                color_arg = None if sc == "None" else sc
                fig = px.scatter(df, x=sx, y=sy, color=color_arg,
                                 opacity=.7, title=f"{sx} vs {sy}",
                                 color_discrete_sequence=["#14b8a6","#f97316","#6366f1","#22c55e"])
                style_fig(fig, 420)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numerical columns for distribution plots.")

    with tab5:
        if num_cols:
            sel = st.multiselect("Select columns for boxplots", num_cols, default=num_cols[:min(5,len(num_cols))], key="box_cols")
            if sel:
                fig = go.Figure()
                colors = ["#14b8a6","#f97316","#6366f1","#22c55e","#eab308","#ec4899","#8b5cf6","#0ea5e9"]
                for i, c in enumerate(sel):
                    fig.add_trace(go.Box(y=df[c].dropna(), name=c, marker_color=colors[i % len(colors)], boxmean=True))
                fig.update_layout(title="Boxplots", showlegend=False, **PLOT_THEME, height=420)
                fig.update_xaxes(gridcolor="#1e3a5f"); fig.update_yaxes(gridcolor="#1e3a5f")
                st.plotly_chart(fig, use_container_width=True)

    with tab6:
        if cat_cols:
            sel = st.selectbox("Select categorical column", cat_cols, key="bar_col")
            n_top = st.slider("Top N categories", 5, 30, 15, key="bar_n")
            vc = df[sel].value_counts().head(n_top)
            fig = go.Figure(go.Bar(
                x=vc.values, y=vc.index, orientation="h",
                marker_color="#14b8a6", text=vc.values, textposition="outside",
            ))
            fig.update_layout(title=f"Value Counts: {sel}", xaxis_title="Count", **PLOT_THEME, height=max(300, n_top*25+100))
            fig.update_xaxes(gridcolor="#1e3a5f"); fig.update_yaxes(gridcolor="#1e3a5f", autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)

            if len(cat_cols) >= 2:
                st.markdown("### Cross-tab Heatmap")
                c1, c2 = st.columns(2)
                cx = c1.selectbox("Row", cat_cols, key="ct_x")
                cy = c2.selectbox("Column", cat_cols, index=min(1, len(cat_cols)-1), key="ct_y")
                ct = pd.crosstab(df[cx], df[cy])
                fig = px.imshow(ct, text_auto=True, color_continuous_scale=[[0,"#0f2040"],[1,"#14b8a6"]],
                                title=f"Cross-tab: {cx} × {cy}")
                style_fig(fig, 400)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No categorical columns found.")

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 4 — DATA CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def clean_data_ui(df):
    df_work = df.copy()
    log = st.session_state.cleaning_log

    num_cols, cat_cols, _ = detect_column_types(df_work)

    st.markdown("## 🧹 Data Cleaning")
    st.caption("Changes apply sequentially. Dataset updates live.")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Drop Columns")
        drop_cols = st.multiselect("Select columns to drop", df_work.columns.tolist(), key="drop_cols")
        if st.button("Drop Selected Columns", key="btn_drop"):
            if drop_cols:
                df_work.drop(columns=drop_cols, inplace=True)
                log.append(f"Dropped columns: {drop_cols}")
                st.success(f"Dropped {len(drop_cols)} column(s).")

        st.markdown("### Drop Duplicates")
        if st.button("Drop Duplicate Rows", key="btn_dup"):
            before = len(df_work)
            df_work.drop_duplicates(inplace=True)
            df_work.reset_index(drop=True, inplace=True)
            removed = before - len(df_work)
            log.append(f"Dropped {removed} duplicate rows.")
            st.success(f"Removed {removed} duplicate row(s).")

        st.markdown("### Rename Column")
        r_old = st.selectbox("Column to rename", df_work.columns.tolist(), key="ren_old")
        r_new = st.text_input("New name", key="ren_new")
        if st.button("Rename", key="btn_ren") and r_new.strip():
            df_work.rename(columns={r_old: r_new.strip()}, inplace=True)
            log.append(f"Renamed '{r_old}' → '{r_new.strip()}'")
            st.success(f"Renamed '{r_old}' to '{r_new.strip()}'.")

    with c2:
        st.markdown("### Fill Missing Values")
        fill_col = st.selectbox("Column", df_work.columns.tolist(), key="fill_col")
        fill_method = st.selectbox("Strategy", ["mean","median","mode","custom","forward fill","backward fill"], key="fill_meth")
        custom_val = ""
        if fill_method == "custom":
            custom_val = st.text_input("Custom value", key="fill_custom")
        if st.button("Fill Missing", key="btn_fill"):
            col_data = df_work[fill_col]
            missing_before = col_data.isnull().sum()
            if missing_before == 0:
                st.info("No missing values in this column.")
            else:
                if fill_method == "mean" and fill_col in num_cols:
                    df_work[fill_col].fillna(col_data.mean(), inplace=True)
                elif fill_method == "median" and fill_col in num_cols:
                    df_work[fill_col].fillna(col_data.median(), inplace=True)
                elif fill_method == "mode":
                    df_work[fill_col].fillna(col_data.mode()[0], inplace=True)
                elif fill_method == "forward fill":
                    df_work[fill_col].fillna(method="ffill", inplace=True)
                elif fill_method == "backward fill":
                    df_work[fill_col].fillna(method="bfill", inplace=True)
                elif fill_method == "custom" and custom_val != "":
                    try: df_work[fill_col].fillna(float(custom_val) if fill_col in num_cols else custom_val, inplace=True)
                    except: df_work[fill_col].fillna(custom_val, inplace=True)
                log.append(f"Filled missing in '{fill_col}' using {fill_method}.")
                st.success(f"Filled {missing_before} missing value(s) in '{fill_col}'.")

        st.markdown("### Handle Outliers (IQR)")
        out_col = st.selectbox("Column", num_cols if num_cols else df_work.columns.tolist(), key="out_col")
        out_action = st.selectbox("Action", ["Remove rows", "Cap (clip) to bounds"], key="out_act")
        if st.button("Handle Outliers", key="btn_out") and out_col in df_work.columns:
            q1, q3 = df_work[out_col].quantile(.25), df_work[out_col].quantile(.75)
            iqr = q3 - q1
            lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
            mask = (df_work[out_col] < lo) | (df_work[out_col] > hi)
            n = mask.sum()
            if out_action == "Remove rows":
                df_work = df_work[~mask].reset_index(drop=True)
                log.append(f"Removed {n} outlier rows from '{out_col}'.")
            else:
                df_work[out_col] = df_work[out_col].clip(lo, hi)
                log.append(f"Clipped outliers in '{out_col}' to [{lo:.2f}, {hi:.2f}].")
            st.success(f"Handled {n} outlier(s) in '{out_col}'.")

        st.markdown("### Convert Data Type")
        conv_col = st.selectbox("Column", df_work.columns.tolist(), key="conv_col")
        conv_type = st.selectbox("Target type", ["int","float","str","datetime","bool"], key="conv_type")
        if st.button("Convert", key="btn_conv"):
            try:
                if conv_type == "datetime":
                    df_work[conv_col] = pd.to_datetime(df_work[conv_col], errors="coerce")
                elif conv_type == "bool":
                    df_work[conv_col] = df_work[conv_col].astype(bool)
                else:
                    df_work[conv_col] = df_work[conv_col].astype(conv_type)
                log.append(f"Converted '{conv_col}' to {conv_type}.")
                st.success(f"Converted '{conv_col}' to {conv_type}.")
            except Exception as e:
                st.error(f"Conversion failed: {e}")

    st.divider()
    st.markdown("### Cleaned Dataset Preview")
    st.dataframe(df_work.head(20), use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{df_work.shape[0]:,}", delta=f"{df_work.shape[0]-df.shape[0]}")
    c2.metric("Columns", f"{df_work.shape[1]:,}", delta=f"{df_work.shape[1]-df.shape[1]}")
    c3.metric("Missing", f"{df_work.isnull().sum().sum():,}")

    if log:
        with st.expander("🪵 Cleaning Log"):
            for i, entry in enumerate(log, 1):
                st.markdown(f"`{i}.` {entry}")

    st.session_state.df_clean = df_work
    return df_work

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 5 — FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(df):
    df_work = df.copy()
    num_cols, cat_cols, dt_cols = detect_column_types(df_work)

    st.markdown("## ⚙️ Feature Engineering")

    tab1, tab2, tab3, tab4 = st.tabs(["🏷️ Encoding","📏 Scaling","➕ Transforms","📅 Datetime"])

    with tab1:
        st.markdown("### Label Encoding")
        le_cols = st.multiselect("Columns to label-encode", cat_cols, key="le_cols")
        if st.button("Apply Label Encoding", key="btn_le"):
            le = LabelEncoder()
            for c in le_cols:
                df_work[c + "_le"] = le.fit_transform(df_work[c].astype(str))
            st.success(f"Created {len(le_cols)} label-encoded column(s).")

        st.markdown("### One-Hot Encoding")
        ohe_cols = st.multiselect("Columns to one-hot encode", cat_cols, key="ohe_cols")
        ohe_maxcat = st.slider("Max unique categories", 2, 30, 10, key="ohe_max")
        if st.button("Apply One-Hot Encoding", key="btn_ohe"):
            valid = [c for c in ohe_cols if df_work[c].nunique() <= ohe_maxcat]
            if valid:
                df_work = pd.get_dummies(df_work, columns=valid, drop_first=False)
                st.success(f"One-hot encoded: {valid}")
            else:
                st.warning("No columns with ≤ max categories selected.")

    with tab2:
        st.markdown("### Standard Scaling")
        ss_cols = st.multiselect("Columns to standardize", num_cols, key="ss_cols")
        if st.button("Apply Standard Scaler", key="btn_ss"):
            sc = StandardScaler()
            df_work[ss_cols] = sc.fit_transform(df_work[ss_cols])
            st.success(f"Standardized {len(ss_cols)} column(s).")

        st.markdown("### Min-Max Normalization")
        mm_cols = st.multiselect("Columns to normalize (0–1)", num_cols, key="mm_cols")
        if st.button("Apply Min-Max Scaler", key="btn_mm"):
            sc = MinMaxScaler()
            df_work[mm_cols] = sc.fit_transform(df_work[mm_cols])
            st.success(f"Normalized {len(mm_cols)} column(s).")

    with tab3:
        st.markdown("### Log Transform")
        log_cols = st.multiselect("Columns to log-transform (x > 0)", num_cols, key="log_cols")
        if st.button("Apply Log Transform", key="btn_log"):
            for c in log_cols:
                min_v = df_work[c].min()
                offset = abs(min_v) + 1 if min_v <= 0 else 0
                df_work[c + "_log"] = np.log1p(df_work[c] + offset)
            st.success(f"Log-transformed {len(log_cols)} column(s).")

        st.markdown("### Polynomial Features")
        poly_cols = st.multiselect("Columns for polynomial features", num_cols[:5] if len(num_cols) > 5 else num_cols, key="poly_cols")
        poly_deg = st.slider("Degree", 2, 3, 2, key="poly_deg")
        if st.button("Generate Polynomial Features", key="btn_poly"):
            if poly_cols:
                pf = PolynomialFeatures(degree=poly_deg, include_bias=False)
                arr = pf.fit_transform(df_work[poly_cols].fillna(0))
                pnames = pf.get_feature_names_out(poly_cols)
                new_df = pd.DataFrame(arr, columns=pnames, index=df_work.index)
                df_work = pd.concat([df_work, new_df[[c for c in pnames if c not in df_work.columns]]], axis=1)
                st.success(f"Added {len(pnames)-len(poly_cols)} polynomial feature(s).")

    with tab4:
        st.markdown("### Datetime Feature Extraction")
        dt_target = st.selectbox("Datetime column", dt_cols + cat_cols, key="dt_target")
        if st.button("Extract Datetime Features", key="btn_dt"):
            try:
                series = pd.to_datetime(df_work[dt_target], errors="coerce")
                df_work[dt_target + "_year"]    = series.dt.year
                df_work[dt_target + "_month"]   = series.dt.month
                df_work[dt_target + "_day"]     = series.dt.day
                df_work[dt_target + "_dayofwk"] = series.dt.dayofweek
                df_work[dt_target + "_quarter"] = series.dt.quarter
                st.success(f"Extracted 5 datetime features from '{dt_target}'.")
            except Exception as e:
                st.error(f"Error: {e}")

    st.divider()
    st.dataframe(df_work.head(15), use_container_width=True)
    st.caption(f"Shape after engineering: {df_work.shape}")
    st.session_state.df_clean = df_work
    return df_work

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 6 — ML PREPROCESSING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def prepare_ml_dataset(df):
    st.markdown("## 🤖 ML Preprocessing Pipeline")

    df_work = df.copy()
    num_cols, cat_cols, dt_cols = detect_column_types(df_work)

    target = st.selectbox("Select target column (label)", ["— None —"] + df_work.columns.tolist(), key="ml_target")
    test_size = st.slider("Test set size (%)", 10, 40, 20, key="ml_test") / 100

    if st.button("🚀 Prepare Dataset for Machine Learning", key="btn_ml"):
        progress = st.progress(0, text="Starting pipeline…")
        log_items = []

        # 1. Drop datetime cols
        df_work.drop(columns=dt_cols, errors="ignore", inplace=True)
        if dt_cols: log_items.append(f"Dropped datetime columns: {dt_cols}")
        progress.progress(10, "Handling missing values…")

        # 2. Fill missing
        for c in df_work.select_dtypes(include=np.number).columns:
            if df_work[c].isnull().any():
                df_work[c].fillna(df_work[c].median(), inplace=True)
        for c in df_work.select_dtypes(exclude=np.number).columns:
            if df_work[c].isnull().any():
                df_work[c].fillna(df_work[c].mode()[0] if not df_work[c].mode().empty else "Unknown", inplace=True)
        log_items.append("Filled missing values (num=median, cat=mode).")
        progress.progress(30, "Encoding categorical features…")

        # 3. Encode categoricals (exclude target if categorical)
        cat_now = df_work.select_dtypes(include=["object","category","bool"]).columns.tolist()
        target_col = None if target == "— None —" else target
        encode_cols = [c for c in cat_now if c != target_col]
        high_card = [c for c in encode_cols if df_work[c].nunique() > 15]
        low_card  = [c for c in encode_cols if df_work[c].nunique() <= 15]

        le = LabelEncoder()
        for c in high_card:
            df_work[c] = le.fit_transform(df_work[c].astype(str))
        if low_card:
            df_work = pd.get_dummies(df_work, columns=low_card, drop_first=True)
        if encode_cols: log_items.append(f"Label-encoded {len(high_card)} high-cardinality; OHE {len(low_card)} low-cardinality cols.")
        progress.progress(55, "Removing high-correlation features…")

        # 4. Remove high-corr features (keep target safe)
        num_now = df_work.select_dtypes(include=np.number).columns.tolist()
        if target_col and target_col in num_now: num_now.remove(target_col)
        if len(num_now) >= 2:
            corr_matrix = df_work[num_now].corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            drop_corr = [c for c in upper.columns if any(upper[c] > 0.9)]
            df_work.drop(columns=drop_corr, inplace=True, errors="ignore")
            if drop_corr: log_items.append(f"Dropped high-corr features: {drop_corr}")
        progress.progress(70, "Scaling numeric features…")

        # 5. Scale numeric (exclude target)
        num_final = df_work.select_dtypes(include=np.number).columns.tolist()
        if target_col and target_col in num_final: num_final.remove(target_col)
        if num_final:
            sc = StandardScaler()
            df_work[num_final] = sc.fit_transform(df_work[num_final])
            log_items.append(f"Standard-scaled {len(num_final)} numeric features.")
        progress.progress(85, "Splitting train/test…")

        # 6. Train-test split
        if target_col and target_col in df_work.columns:
            X = df_work.drop(columns=[target_col])
            y = df_work[target_col]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            st.session_state.label_col = target_col
            # Guess task type
            if y.dtype == object or y.nunique() <= 20:
                st.session_state.task_type = "classification"
            else:
                st.session_state.task_type = "regression"
            log_items.append(f"Split: train={len(X_train)}, test={len(X_test)} ({int(test_size*100)}% test).")
            st.success(f"✅ Train: {len(X_train)} rows | Test: {len(X_test)} rows")

            # Feature importance (mutual info)
            try:
                mi_fn = mutual_info_classif if st.session_state.task_type == "classification" else mutual_info_regression
                mi = mi_fn(X_train.fillna(0), y_train)
                mi_s = pd.Series(mi, index=X_train.columns).sort_values(ascending=False).head(10)
                fig = go.Figure(go.Bar(y=mi_s.index[::-1], x=mi_s.values[::-1], orientation="h",
                                       marker_color="#14b8a6"))
                fig.update_layout(title="Top Feature Importance (Mutual Info)", **PLOT_THEME, height=350)
                fig.update_xaxes(gridcolor="#1e3a5f"); fig.update_yaxes(gridcolor="#1e3a5f")
                st.plotly_chart(fig, use_container_width=True)
                log_items.append(f"Top features: {', '.join(mi_s.index[:3].tolist())}")
            except: pass
        progress.progress(100, "Done!")

        st.session_state.df_ml = df_work

        st.markdown("### Pipeline Log")
        for i, item in enumerate(log_items, 1):
            st.markdown(f'<div class="insight-item ok">✅ {i}. {item}</div>', unsafe_allow_html=True)

        st.markdown("### ML-Ready Dataset Preview")
        st.dataframe(df_work.head(15), use_container_width=True)
        st.caption(f"Final shape: {df_work.shape}")

    return df_work

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 7 — DATASET HEALTH SCORE
# ─────────────────────────────────────────────────────────────────────────────

def dataset_health_score(df):
    score = 100
    details = {}
    suggestions = []

    # Missing
    miss_pct = df.isnull().mean().mean() * 100
    miss_pen = min(30, miss_pct * 1.5)
    score -= miss_pen
    details["Missing Values"] = {"score": round(30 - miss_pen), "max": 30, "value": f"{miss_pct:.1f}%"}
    if miss_pct > 5: suggestions.append(f"Impute or drop columns with high missing rates (avg {miss_pct:.1f}%).")

    # Duplicates
    dup_pct = df.duplicated().sum() / len(df) * 100
    dup_pen = min(15, dup_pct * 2)
    score -= dup_pen
    details["Duplicates"] = {"score": round(15 - dup_pen), "max": 15, "value": f"{dup_pct:.1f}%"}
    if dup_pct > 1: suggestions.append(f"Remove {df.duplicated().sum()} duplicate rows.")

    # Outliers
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    out_pct = 0
    if num_cols:
        outlier_flags = []
        for c in num_cols:
            q1, q3 = df[c].quantile(.25), df[c].quantile(.75)
            iqr = q3 - q1
            outlier_flags.append(((df[c] < q1-1.5*iqr)|(df[c] > q3+1.5*iqr)).sum())
        out_pct = sum(outlier_flags) / (len(df) * len(num_cols)) * 100 if num_cols else 0
    out_pen = min(20, out_pct * 2)
    score -= out_pen
    details["Outliers"] = {"score": round(20 - out_pen), "max": 20, "value": f"{out_pct:.1f}%"}
    if out_pct > 5: suggestions.append("Handle outliers via IQR clipping or removal.")

    # High correlation
    hc = detect_high_corr(df, num_cols)
    hc_pen = min(20, len(hc) * 4)
    score -= hc_pen
    details["High Correlation"] = {"score": round(20 - hc_pen), "max": 20, "value": f"{len(hc)} pairs"}
    if hc: suggestions.append(f"Remove or merge {len(hc)} highly correlated feature pair(s).")

    # Class imbalance (categorical cols with 2–10 unique)
    cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()
    imb_pen = 0
    for c in cat_cols[:3]:
        if 2 <= df[c].nunique() <= 10:
            imb, ratio = detect_imbalance(df, c)
            if imb: imb_pen += min(5, ratio)
    imb_pen = min(15, imb_pen)
    score -= imb_pen
    details["Class Balance"] = {"score": round(15 - imb_pen), "max": 15, "value": f"-{imb_pen:.0f}pts"}
    if imb_pen > 3: suggestions.append("Address class imbalance via oversampling (SMOTE) or class weights.")

    score = max(0, min(100, round(score)))
    return score, details, suggestions

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 8 — MODEL RECOMMENDATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

MODEL_DB = {
    "classification": [
        {"name":"Logistic Regression","pkg":"sklearn.linear_model","desc":"Fast, interpretable baseline. Works best with linearly separable data.","params":"max_iter=1000, C=1.0"},
        {"name":"Random Forest","pkg":"sklearn.ensemble","desc":"Robust ensemble model, handles non-linearity and noisy features well.","params":"n_estimators=200, max_depth=None"},
        {"name":"XGBoost","pkg":"xgboost","desc":"State-of-the-art gradient boosting. High accuracy with proper tuning.","params":"n_estimators=300, learning_rate=0.1, max_depth=6"},
        {"name":"LightGBM","pkg":"lightgbm","desc":"Faster training than XGBoost; great for large datasets.","params":"num_leaves=31, learning_rate=0.05"},
    ],
    "regression": [
        {"name":"Linear Regression","pkg":"sklearn.linear_model","desc":"Simple baseline for continuous targets. Assumes linearity.","params":"fit_intercept=True"},
        {"name":"Ridge Regression","pkg":"sklearn.linear_model","desc":"L2 regularization to prevent overfitting in high-dimensional data.","params":"alpha=1.0"},
        {"name":"Lasso Regression","pkg":"sklearn.linear_model","desc":"L1 regularization; performs automatic feature selection.","params":"alpha=0.1"},
        {"name":"Gradient Boosting","pkg":"sklearn.ensemble","desc":"Captures non-linear patterns; often outperforms linear models.","params":"n_estimators=200, learning_rate=0.1"},
    ],
    "clustering": [
        {"name":"KMeans","pkg":"sklearn.cluster","desc":"Partitions data into k clusters. Fast and scalable.","params":"n_clusters=3, random_state=42"},
        {"name":"DBSCAN","pkg":"sklearn.cluster","desc":"Density-based; finds arbitrary shapes and handles noise.","params":"eps=0.5, min_samples=5"},
        {"name":"Agglomerative","pkg":"sklearn.cluster","desc":"Hierarchical clustering; no need to specify k beforehand.","params":"n_clusters=3, linkage='ward'"},
    ],
}

def model_recommendations(df, task_type=None):
    st.markdown("## 🎯 Model Suggestions")

    if task_type is None:
        task_type = st.selectbox("Task type", ["classification","regression","clustering"], key="model_task")
    else:
        st.markdown(f'Auto-detected task: {badge(task_type, "teal")}', unsafe_allow_html=True)
        task_type = st.selectbox("Override task type", ["classification","regression","clustering"],
                                  index=["classification","regression","clustering"].index(task_type), key="model_task_ov")

    models = MODEL_DB.get(task_type, [])
    c1, c2 = st.columns(2)
    for i, m in enumerate(models):
        col = c1 if i % 2 == 0 else c2
        with col:
            st.markdown(f"""
<div class="model-card">
  <div class="model-name">🔷 {m['name']}</div>
  <div class="model-desc">{m['desc']}</div>
  <div style="margin-top:.6rem;">
    <span class="badge badge-gray">{m['pkg']}</span>
    <code style="font-size:.75rem;margin-left:.4rem;color:#94a3b8;">{m['params']}</code>
  </div>
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 9 — REPORTS & CODE EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def generate_eda_report(df, insights, health_score, health_details, suggestions):
    num_cols, cat_cols, _ = detect_column_types(df)
    desc_html = df.describe().to_html(classes="tbl", border=0)
    insight_rows = "".join(f"<li>{i['msg']}</li>" for i in insights)
    health_rows = "".join(f"<tr><td>{k}</td><td>{v['score']}/{v['max']}</td><td>{v['value']}</td></tr>"
                          for k, v in health_details.items())
    sug_rows = "".join(f"<li>{s}</li>" for s in suggestions)
    color = "#22c55e" if health_score >= 80 else "#eab308" if health_score >= 60 else "#ef4444"

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>DataForge EDA Report</title>
<style>
body{{font-family:'Segoe UI',sans-serif;background:#050d1a;color:#e2e8f0;margin:0;padding:2rem;}}
h1{{font-size:2rem;color:#14b8a6;}}h2{{color:#14b8a6;border-bottom:1px solid #1e3a5f;padding-bottom:.4rem;}}
.score{{font-size:4rem;font-weight:900;color:{color};}}
.tbl{{border-collapse:collapse;width:100%;font-size:.85rem;background:#0a1628;}}
.tbl th{{background:#1a4480;padding:.4rem .8rem;text-align:left;}}
.tbl td{{padding:.4rem .8rem;border-bottom:1px solid #1e3a5f;}}
.badge{{display:inline-block;padding:2px 10px;border-radius:99px;font-size:.72rem;background:rgba(20,184,166,.15);color:#14b8a6;border:1px solid rgba(20,184,166,.3);margin:2px;}}
ul{{padding-left:1.2rem;}}li{{margin:.4rem 0;}}
</style></head><body>
<h1>⚡ DataForge EDA Report</h1>
<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} &nbsp;|&nbsp; Shape: {df.shape[0]:,} rows × {df.shape[1]:,} cols</p>
<h2>Dataset Health Score</h2>
<div class="score">{health_score} / 100</div>
<table class="tbl"><tr><th>Metric</th><th>Score</th><th>Value</th></tr>{health_rows}</table>
<h2>Improvement Suggestions</h2><ul>{sug_rows}</ul>
<h2>Automated Insights</h2><ul>{insight_rows}</ul>
<h2>Summary Statistics</h2>{desc_html}
<h2>Column Types</h2>
<b>Numerical:</b> {''.join(f'<span class="badge">{c}</span>' for c in num_cols)}<br>
<b>Categorical:</b> {''.join(f'<span class="badge">{c}</span>' for c in cat_cols)}
</body></html>"""
    return html.encode()

def generate_pipeline_script(df, cleaning_log, label_col=None, task_type=None):
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()
    if label_col and label_col in num_cols: num_cols.remove(label_col)
    if label_col and label_col in cat_cols: cat_cols.remove(label_col)

    script = f'''"""
DataForge — Auto-Generated ML Pipeline
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# ── Load Data ──────────────────────────────────────────────
df = pd.read_csv("your_data.csv")

# ── Cleaning Steps ─────────────────────────────────────────
# {chr(10)+"# ".join(cleaning_log) if cleaning_log else "# (no cleaning steps recorded)"}
df.drop_duplicates(inplace=True)

# ── Fill Missing Values ────────────────────────────────────
num_cols = {num_cols}
cat_cols  = {cat_cols}
for c in num_cols:
    df[c].fillna(df[c].median(), inplace=True)
for c in cat_cols:
    df[c].fillna(df[c].mode()[0] if not df[c].mode().empty else "Unknown", inplace=True)

# ── Encode Categoricals ────────────────────────────────────
le = LabelEncoder()
for c in cat_cols:
    if df[c].nunique() > 15:
        df[c] = le.fit_transform(df[c].astype(str))
low_card = [c for c in cat_cols if df[c].nunique() <= 15]
if low_card:
    df = pd.get_dummies(df, columns=low_card, drop_first=True)

# ── Scale Numeric ──────────────────────────────────────────
sc = StandardScaler()
scale_cols = df.select_dtypes(include=np.number).columns.tolist()
{"if '"+label_col+"' in scale_cols: scale_cols.remove('"+label_col+"')" if label_col else "# (no target column set)"}
df[scale_cols] = sc.fit_transform(df[scale_cols])

# ── Train-Test Split ───────────────────────────────────────
{"target = '"+label_col+"'" if label_col else "target = 'your_target_column'"}
X = df.drop(columns=[target])
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Train: {{len(X_train)}} | Test: {{len(X_test)}}")
print(f"Features: {{X_train.shape[1]}}")

# ── Suggested Model ────────────────────────────────────────
{"# Task: "+task_type if task_type else "# Set task type: classification / regression / clustering"}
# from sklearn.ensemble import RandomForestClassifier
# model = RandomForestClassifier(n_estimators=200, random_state=42)
# model.fit(X_train, y_train)
# print(model.score(X_test, y_test))
'''
    return script.encode()

def generate_reports(df, insights, health_score, health_details, suggestions, cleaning_log):
    st.markdown("## 💾 Download Center")

    c1, c2, c3, c4 = st.columns(4)
    df_clean = st.session_state.df_clean if st.session_state.df_clean is not None else df
    df_ml    = st.session_state.df_ml    if st.session_state.df_ml    is not None else df

    with c1:
        st.markdown("**🧼 Cleaned Dataset**")
        csv_bytes = to_csv_bytes(df_clean)
        st.markdown(to_download_link(csv_bytes, "cleaned_dataset.csv", "⬇ Download CSV"), unsafe_allow_html=True)

    with c2:
        st.markdown("**🤖 ML-Ready Dataset**")
        ml_bytes = to_csv_bytes(df_ml)
        st.markdown(to_download_link(ml_bytes, "ml_ready_dataset.csv", "⬇ Download CSV"), unsafe_allow_html=True)

    with c3:
        st.markdown("**📄 EDA Report HTML**")
        html_bytes = generate_eda_report(df, insights, health_score, health_details, suggestions)
        st.markdown(to_download_link(html_bytes, "eda_report.html", "⬇ Download HTML", "text/html"), unsafe_allow_html=True)

    with c4:
        st.markdown("**🐍 Pipeline Script**")
        py_bytes = generate_pipeline_script(df_ml, cleaning_log,
                                            st.session_state.label_col, st.session_state.task_type)
        st.markdown(to_download_link(py_bytes, "ml_pipeline.py", "⬇ Download .py", "text/x-python"), unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 10 — AUTONOMOUS FULL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def run_full_auto_analysis(df):
    st.markdown("## ⚡ Autonomous Analysis Report")
    progress = st.progress(0, "Initialising…")

    # Step 1: insights
    progress.progress(10, "Generating insights…")
    insights, num_cols, cat_cols, dt_cols = generate_insights(df)
    st.session_state.insights = insights

    # Step 2: health score
    progress.progress(25, "Calculating health score…")
    score, details, suggestions = dataset_health_score(df)
    st.session_state.health_score    = score
    st.session_state.health_details  = details

    # Step 3: auto clean
    progress.progress(40, "Auto-cleaning dataset…")
    df_c = df.copy()
    df_c.drop_duplicates(inplace=True)
    for c in df_c.select_dtypes(include=np.number).columns:
        df_c[c].fillna(df_c[c].median(), inplace=True)
    for c in df_c.select_dtypes(exclude=np.number).columns:
        df_c[c].fillna(df_c[c].mode()[0] if not df_c[c].mode().empty else "Unknown", inplace=True)
    st.session_state.df_clean = df_c
    st.session_state.cleaning_log.append("Auto-clean: filled missing, dropped duplicates.")

    # Step 4: auto ML prep
    progress.progress(65, "Preparing ML dataset…")
    df_m = df_c.copy()
    cat_now = df_m.select_dtypes(include=["object","category","bool"]).columns.tolist()
    le = LabelEncoder()
    for c in cat_now:
        if df_m[c].nunique() <= 15:
            df_m = pd.get_dummies(df_m, columns=[c], drop_first=True)
        else:
            df_m[c] = le.fit_transform(df_m[c].astype(str))
    num_now = df_m.select_dtypes(include=np.number).columns.tolist()
    hc_pairs = detect_high_corr(df_m, num_now, .9)
    drop_hc = list(set([p[1] for p in hc_pairs]))
    df_m.drop(columns=drop_hc, inplace=True, errors="ignore")
    if drop_hc: st.session_state.cleaning_log.append(f"Dropped high-corr: {drop_hc}")
    st.session_state.df_ml = df_m
    progress.progress(85, "Generating charts…")

    # ── Render report ──────────────────────────────────────────────────────

    # Health score
    color = "#22c55e" if score >= 80 else "#eab308" if score >= 60 else "#ef4444"
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"""
<div class="health-ring">
  <div style="font-family:'Syne',sans-serif;font-size:.75rem;text-transform:uppercase;letter-spacing:.12em;color:#94a3b8;">Dataset Health</div>
  <div class="health-score" style="color:{color};">{score}</div>
  <div style="font-size:1.2rem;color:#94a3b8;">/100</div>
  <div style="margin-top:.8rem;">
    {'<span class="badge badge-green">HEALTHY</span>' if score>=80 else '<span class="badge badge-orange">NEEDS WORK</span>' if score>=60 else '<span class="badge badge-red">CRITICAL</span>'}
  </div>
</div>""", unsafe_allow_html=True)

    with col2:
        fig = go.Figure()
        cats = list(details.keys())
        scores = [v["score"] for v in details.values()]
        maxes  = [v["max"]   for v in details.values()]
        fig.add_trace(go.Bar(name="Score", x=cats, y=scores, marker_color="#14b8a6"))
        fig.add_trace(go.Bar(name="Max",   x=cats, y=maxes,  marker_color="#1e3a5f"))
        fig.update_layout(barmode="overlay", title="Health Score Breakdown", **PLOT_THEME, height=280, showlegend=False)
        fig.update_xaxes(gridcolor="#1e3a5f"); fig.update_yaxes(gridcolor="#1e3a5f")
        st.plotly_chart(fig, use_container_width=True)

    progress.progress(90, "Finalising…")

    # Insights
    st.markdown("### 🔍 Automated Insights")
    st.markdown(insight_html(insights), unsafe_allow_html=True)

    # Suggestions
    if suggestions:
        st.markdown("### 💡 Improvement Suggestions")
        for s in suggestions:
            st.markdown(f'<div class="insight-item warn">💡 {s}</div>', unsafe_allow_html=True)

    # Quick EDA charts
    st.markdown("### 📊 Quick EDA Snapshot")
    if num_cols:
        c1, c2 = st.columns(2)
        with c1:
            miss_s = df.isnull().mean().sort_values(ascending=False).head(10)
            miss_s = miss_s[miss_s > 0]
            if not miss_s.empty:
                fig = go.Figure(go.Bar(x=miss_s.values*100, y=miss_s.index, orientation="h",
                                       marker_color="#f97316"))
                fig.update_layout(title="Missing % (top 10)", **PLOT_THEME, height=280)
                fig.update_xaxes(gridcolor="#1e3a5f"); fig.update_yaxes(gridcolor="#1e3a5f")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.markdown('<div class="df-card df-card-ok">✅ No missing values!</div>', unsafe_allow_html=True)
        with c2:
            if len(num_cols) >= 2:
                corr = df[num_cols[:12]].corr()
                fig = px.imshow(corr, text_auto=".1f", aspect="auto",
                                color_continuous_scale=[[0,"#ef4444"],[.5,"#0f2040"],[1,"#14b8a6"]],
                                zmin=-1, zmax=1, title="Correlation Heatmap")
                style_fig(fig, 280)
                st.plotly_chart(fig, use_container_width=True)

    # Distributions for top 6 numeric
    if num_cols:
        st.markdown("### 📈 Distributions")
        cols_show = num_cols[:6]
        rows = (len(cols_show) + 2) // 3
        fig = make_subplots(rows=rows, cols=3, subplot_titles=cols_show)
        colors = ["#14b8a6","#f97316","#6366f1","#22c55e","#eab308","#ec4899"]
        for idx, c in enumerate(cols_show):
            r, col_ = idx // 3 + 1, idx % 3 + 1
            vals = df[c].dropna()
            fig.add_trace(go.Histogram(x=vals, nbinsx=30, name=c,
                                       marker_color=colors[idx % len(colors)], showlegend=False), row=r, col=col_)
        fig.update_layout(height=280*rows, **PLOT_THEME)
        fig.update_xaxes(gridcolor="#1e3a5f"); fig.update_yaxes(gridcolor="#1e3a5f")
        st.plotly_chart(fig, use_container_width=True)

    # Model recommendation
    st.markdown("### 🎯 Recommended Models")
    # Guess task type from label column presence
    task = st.session_state.task_type or "clustering"
    models = MODEL_DB.get(task, MODEL_DB["clustering"])
    c1, c2 = st.columns(2)
    for i, m in enumerate(models[:4]):
        col = c1 if i % 2 == 0 else c2
        with col:
            st.markdown(f"""
<div class="model-card">
  <div class="model-name">🔷 {m['name']}</div>
  <div class="model-desc">{m['desc']}</div>
  <span class="badge badge-teal">{task}</span>
</div>""", unsafe_allow_html=True)

    progress.progress(100, "Analysis complete!")
    st.session_state.auto_ran = True
    st.success("✅ Full Autonomous Analysis complete!")

    return insights, score, details, suggestions

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("""
<div class="logo-wrap">
  <span class="logo-icon">⚡</span>
  <div>
    <div class="logo-text">DataForge</div>
    <div class="logo-sub">Autonomous Analytics</div>
  </div>
</div>""", unsafe_allow_html=True)

        st.divider()

        uploaded = st.file_uploader(
            "Upload Dataset",
            type=["csv","xlsx","xls","json"],
            help="CSV, Excel, or JSON",
        )

        if uploaded:
            df = load_data(uploaded)
            if df is not None:
                st.session_state.df = df
                if st.session_state.df_clean is None:
                    st.session_state.df_clean = df.copy()
                st.success(f"✅ Loaded: {uploaded.name}")
                st.markdown(f"""
<div class="df-card df-card-accent" style="margin-top:.5rem;">
  <div style="font-size:.75rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.08em;">Dataset</div>
  <div style="font-family:'Syne',sans-serif;font-weight:700;margin:.2rem 0;">{uploaded.name}</div>
  {badge(f'{df.shape[0]:,} rows', 'teal')} {badge(f'{df.shape[1]} cols', 'gray')} {badge(df_mem(df), 'gray')}
</div>""", unsafe_allow_html=True)

        st.divider()
        st.markdown("### Navigation")
        pages = {
            "🏠 Overview":          "overview",
            "🔍 Auto Insights":     "insights",
            "📊 EDA Engine":        "eda",
            "🧹 Data Cleaning":     "clean",
            "⚙️ Feature Eng.":      "features",
            "🤖 ML Pipeline":       "ml",
            "🎯 Model Suggestions": "models",
            "💾 Download Center":   "download",
            "⚡ Autonomous Mode":   "auto",
        }
        page = st.radio("", list(pages.keys()), label_visibility="collapsed", key="nav")
        st.divider()
        if st.session_state.health_score is not None:
            s = st.session_state.health_score
            color = "#22c55e" if s>=80 else "#eab308" if s>=60 else "#ef4444"
            st.markdown(f"""
<div style="text-align:center;padding:.8rem;">
  <div style="font-size:.7rem;text-transform:uppercase;letter-spacing:.1em;color:#94a3b8;">Health Score</div>
  <div style="font-family:'Syne',sans-serif;font-size:2.5rem;font-weight:800;color:{color};">{s}</div>
  <div style="color:#94a3b8;font-size:.8rem;">/100</div>
</div>""", unsafe_allow_html=True)
        st.markdown('<div style="color:#94a3b8;font-size:.7rem;text-align:center;margin-top:2rem;">DataForge v1.0 · Built with Streamlit</div>', unsafe_allow_html=True)

    return pages[page]

# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    page = render_sidebar()
    df = st.session_state.df

    # ── Hero when no data ───────────────────────────────────────────────────
    if df is None:
        st.markdown("""
<div style="text-align:center;padding:4rem 2rem;">
  <div style="font-size:4rem;margin-bottom:.5rem;">⚡</div>
  <h1 style="font-family:'Syne',sans-serif;font-size:3rem;font-weight:800;background:linear-gradient(135deg,#14b8a6,#f97316);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">DataForge</h1>
  <p style="font-size:1.2rem;color:#94a3b8;max-width:600px;margin:1rem auto;">Autonomous data analysis, cleaning, feature engineering, and ML preparation — all in one platform.</p>
  <div style="display:flex;gap:1rem;justify-content:center;flex-wrap:wrap;margin-top:2rem;">
    <div class="df-card" style="max-width:180px;text-align:center;"><div style="font-size:1.5rem;">📊</div><div style="font-weight:600;">Auto EDA</div></div>
    <div class="df-card" style="max-width:180px;text-align:center;"><div style="font-size:1.5rem;">🧹</div><div style="font-weight:600;">Smart Cleaning</div></div>
    <div class="df-card" style="max-width:180px;text-align:center;"><div style="font-size:1.5rem;">⚙️</div><div style="font-weight:600;">Feature Eng.</div></div>
    <div class="df-card" style="max-width:180px;text-align:center;"><div style="font-size:1.5rem;">🤖</div><div style="font-weight:600;">ML Pipeline</div></div>
    <div class="df-card" style="max-width:180px;text-align:center;"><div style="font-size:1.5rem;">💾</div><div style="font-weight:600;">Export</div></div>
  </div>
  <p style="color:#94a3b8;margin-top:2rem;font-size:.95rem;">👈 Upload your dataset (CSV, Excel, JSON) in the sidebar to begin.</p>
</div>""", unsafe_allow_html=True)
        return

    df_work = st.session_state.df_clean if st.session_state.df_clean is not None else df

    # ── Pages ───────────────────────────────────────────────────────────────

    if page == "overview":
        display_dataset_overview(df)

    elif page == "insights":
        st.markdown("## 🔍 Automated Dataset Analysis")
        with st.spinner("Analysing dataset…"):
            insights, num_cols, cat_cols, dt_cols = generate_insights(df)
        st.session_state.insights = insights

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Numerical Cols",   len(num_cols))
        c2.metric("Categorical Cols", len(cat_cols))
        c3.metric("Datetime Cols",    len(dt_cols))
        c4.metric("Total Insights",   len(insights))

        st.markdown("### 💡 Insights")
        st.markdown(insight_html(insights), unsafe_allow_html=True)

        st.divider()
        score, details, suggestions = dataset_health_score(df)
        st.session_state.health_score   = score
        st.session_state.health_details = details

        st.markdown("### 📊 Health Score Breakdown")
        c1, c2 = st.columns([1, 2])
        color = "#22c55e" if score>=80 else "#eab308" if score>=60 else "#ef4444"
        with c1:
            st.markdown(f"""
<div class="health-ring">
  <div style="font-size:.75rem;text-transform:uppercase;letter-spacing:.12em;color:#94a3b8;">Health Score</div>
  <div class="health-score" style="color:{color};">{score}</div>
  <div style="color:#94a3b8;">/100</div>
</div>""", unsafe_allow_html=True)
        with c2:
            for k, v in details.items():
                pct = v["score"] / v["max"] if v["max"] else 0
                st.markdown(f'<div style="display:flex;justify-content:space-between;font-size:.85rem;margin:.2rem 0;"><span>{k}</span><span style="color:#94a3b8;">{v["score"]}/{v["max"]} ({v["value"]})</span></div>', unsafe_allow_html=True)
                st.progress(pct)

        if suggestions:
            st.markdown("### 💡 Suggestions")
            for s in suggestions:
                st.markdown(f'<div class="insight-item warn">💡 {s}</div>', unsafe_allow_html=True)

    elif page == "eda":
        perform_eda(df_work)

    elif page == "clean":
        df_result = clean_data_ui(df_work)
        st.session_state.df_clean = df_result

    elif page == "features":
        df_result = engineer_features(df_work)
        st.session_state.df_clean = df_result

    elif page == "ml":
        prepare_ml_dataset(df_work)

    elif page == "models":
        model_recommendations(df_work, st.session_state.task_type)

    elif page == "download":
        insights = st.session_state.insights or []
        health_score = st.session_state.health_score or 0
        health_details = st.session_state.health_details or {}
        if not insights:
            insights, *_ = generate_insights(df)
        if not health_score:
            health_score, health_details, suggestions = dataset_health_score(df)
        else:
            _, _, suggestions = dataset_health_score(df)
        generate_reports(df_work, insights, health_score, health_details, suggestions,
                         st.session_state.cleaning_log)

    elif page == "auto":
        st.markdown("""
<div class="df-card df-card-accent" style="margin-bottom:1.5rem;">
  <b>⚡ Autonomous Analysis Mode</b><br>
  <span style="color:#94a3b8;font-size:.88rem;">Click the button below to run a complete end-to-end analysis automatically — cleaning, EDA, ML prep, health scoring, and model suggestions.</span>
</div>""", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 3])
        with col1:
            run_btn = st.button("⚡ Run Full Automatic Data Analysis", key="btn_auto_run")
        if run_btn or st.session_state.auto_ran:
            if run_btn:
                st.session_state.auto_ran = False  # reset to re-run
            with st.spinner("Running autonomous analysis…"):
                insights, score, details, suggestions = run_full_auto_analysis(df)
            st.session_state.insights      = insights
            st.session_state.health_score  = score
            st.session_state.health_details = details

if __name__ == "__main__":
    main()
