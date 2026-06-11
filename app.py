
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings, io

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

# ─── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Résistance du Béton — ML",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ---------- fond général ---------- */
[data-testid="stAppViewContainer"] { background: #F0F4F8; }
[data-testid="stHeader"]           { background: transparent; }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1F2937 0%, #111827 100%);
    border-right: 1px solid #374151;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSlider label { color: #D1D5DB !important; font-size: 13px; }
[data-testid="stSidebar"] h2             { color: #2DD4BF !important; font-size: 18px; letter-spacing: 0.3px; }
[data-testid="stSidebar"] hr             { border-color: #374151; margin: 12px 0; }

/* ---------- section labels in sidebar ---------- */
.sb-section {
    background: #374151;
    color: #9CA3AF !important;
    font-size: 11px !important;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 4px 10px;
    border-radius: 6px;
    margin: 10px 0 6px 0;
}

/* ---------- carte principale ---------- */
.pred-card {
    background: linear-gradient(135deg, #0D9488 0%, #0F766E 100%);
    border-radius: 20px;
    padding: 32px 24px;
    color: white;
    text-align: center;
    box-shadow: 0 8px 32px rgba(13,148,136,0.35);
    position: relative;
    overflow: hidden;
}
.pred-card::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 160px; height: 160px;
    border-radius: 50%;
    background: rgba(255,255,255,0.07);
}
.pred-mpa  { font-size: 72px; font-weight: 800; line-height: 1; margin: 4px 0; }
.pred-unit { font-size: 22px; opacity: 0.8; }
.pred-sub  { font-size: 13px; opacity: 0.75; margin-top: 4px; }

/* ---------- badge classe béton ---------- */
.badge {
    display: inline-block;
    padding: 6px 20px;
    border-radius: 30px;
    font-size: 13px;
    font-weight: 700;
    margin-top: 14px;
    letter-spacing: 0.3px;
}
.bg-low   { background:#FEE2E2; color:#B91C1C; }
.bg-med   { background:#FEF3C7; color:#B45309; }
.bg-good  { background:#D1FAE5; color:#065F46; }
.bg-high  { background:#DBEAFE; color:#1D4ED8; }

/* ---------- KPI mini-cartes ---------- */
[data-testid="stMetric"] {
    background: white;
    border-radius: 12px;
    padding: 14px 16px 10px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.06);
    border-top: 3px solid #0D9488;
}
[data-testid="stMetricLabel"] { font-size: 12px !important; color: #6B7280 !important; }
[data-testid="stMetricValue"] { font-size: 22px !important; color: #1F2937 !important; font-weight: 700 !important; }

/* ---------- chart containers ---------- */
.chart-card {
    background: white;
    border-radius: 16px;
    padding: 16px;
    box-shadow: 0 1px 8px rgba(0,0,0,0.06);
}

/* ---------- ratio pill ---------- */
.ratio-pill {
    background: #F9FAFB;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    padding: 10px 14px;
    margin-top: 4px;
}
.ratio-val { font-size: 28px; font-weight: 700; }
.ratio-lbl { font-size: 11px; color: #9CA3AF; margin-left: 6px; }

/* ---------- titre page ---------- */
.page-title {
    font-size: 26px;
    font-weight: 800;
    color: #1F2937;
    margin-bottom: 0;
}
.page-sub {
    font-size: 14px;
    color: #6B7280;
    margin-top: 2px;
    margin-bottom: 18px;
}

/* ---------- footer ---------- */
.footer {
    text-align: center;
    color: #9CA3AF;
    font-size: 11px;
    margin-top: 40px;
    padding-top: 16px;
    border-top: 1px solid #E2E8F0;
}

/* ---------- divider ---------- */
hr { border-color: #E5E7EB !important; }
</style>
""", unsafe_allow_html=True)


# ─── Entraînement & Cache ────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Entraînement des modèles ML…")
def train_models(csv_bytes: bytes):
    df = pd.read_csv(io.BytesIO(csv_bytes))
    df.columns = df.columns.str.strip()

    TARGET = "concrete_compressive_strength"
    X = df.drop(TARGET, axis=1)
    y = df[TARGET]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    # ── Régression Linéaire (avec standardisation)
    sc = StandardScaler()
    Xtr_sc = sc.fit_transform(X_tr)
    Xte_sc = sc.transform(X_te)
    lr = LinearRegression().fit(Xtr_sc, y_tr)
    yh_lr = lr.predict(Xte_sc)

    # ── Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    yh_rf = rf.predict(X_te)

    perf = {
        "lr": dict(r2=r2_score(y_te, yh_lr),
                   rmse=np.sqrt(mean_squared_error(y_te, yh_lr)),
                   mae=mean_absolute_error(y_te, yh_lr)),
        "rf": dict(r2=r2_score(y_te, yh_rf),
                   rmse=np.sqrt(mean_squared_error(y_te, yh_rf)),
                   mae=mean_absolute_error(y_te, yh_rf)),
    }
    fi = pd.Series(rf.feature_importances_, index=X.columns)

    return rf, lr, sc, X, fi, perf, df


# ─── Classification Béton ────────────────────────────────────────
def classify(mpa: float):
    if mpa < 20:
        return "Faible résistance",  "C16",  "#DC2626", "bg-low"
    elif mpa < 30:
        return "Résistance ordinaire", "C25", "#D97706", "bg-med"
    elif mpa < 45:
        return "Normale / Structurale", "C35", "#059669", "bg-good"
    else:
        return "Haute résistance",   "C45+", "#2563EB", "bg-high"


# ─── Graphique Jauge ─────────────────────────────────────────────
def make_gauge(value: float, max_val: float = 90.0) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": " MPa", "font": {"size": 34, "color": "#1F2937"}},
        gauge={
            "axis": {"range": [0, max_val], "tickwidth": 1,
                     "tickcolor": "#9CA3AF", "tickfont": {"size": 10}},
            "bar": {"color": "#0D9488", "thickness": 0.28},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0,   20], "color": "#FEE2E2"},
                {"range": [20,  30], "color": "#FEF3C7"},
                {"range": [30,  45], "color": "#D1FAE5"},
                {"range": [45, max_val], "color": "#DBEAFE"},
            ],
        },
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=20, b=10),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


# ─── Graphique Feature Importance ───────────────────────────────
def make_fi_chart(fi: pd.Series) -> go.Figure:
    fi_s = fi.sort_values(ascending=True)
    top2 = set(fi.nlargest(2).index)
    colors = ["#F97316" if n in top2 else "#0D9488" for n in fi_s.index]
    labels = {
        "cement": "Ciment", "blast_furnace_slag": "Laitier H.F.",
        "fly_ash": "Cendres vol.", "water": "Eau",
        "superplasticizer": "Superplast.", "coarse_aggregate": "Agrégats G.",
        "fine_aggregate": "Agrégats F.", "age": "Âge",
    }
    fig = go.Figure(go.Bar(
        x=fi_s.values * 100,
        y=[labels.get(n, n) for n in fi_s.index],
        orientation="h",
        marker_color=colors,
        text=[f"{v*100:.1f}%" for v in fi_s.values],
        textposition="outside",
        textfont=dict(size=11, color="#374151"),
    ))
    fig.update_layout(
        xaxis=dict(title="Importance (%)", range=[0, 42],
                   gridcolor="#F1F5F9", tickfont=dict(size=10)),
        yaxis=dict(gridcolor="#F1F5F9", tickfont=dict(size=11)),
        height=300, margin=dict(l=0, r=50, t=10, b=30),
        paper_bgcolor="white", plot_bgcolor="white",
    )
    return fig


# ─── Graphique Comparaison mélange vs dataset ───────────────────
def make_compare_chart(your_vals: list, mean_vals: list, labels: list) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Votre mélange", x=labels, y=your_vals,
                         marker_color="#0D9488", opacity=0.9))
    fig.add_trace(go.Bar(name="Moyenne dataset", x=labels, y=mean_vals,
                         marker_color="#CBD5E1", opacity=0.85))
    fig.update_layout(
        barmode="group",
        height=300,
        legend=dict(orientation="h", yanchor="bottom", y=1.0,
                    xanchor="right", x=1, font=dict(size=11)),
        margin=dict(l=0, r=0, t=30, b=30),
        paper_bgcolor="white", plot_bgcolor="white",
        xaxis=dict(tickfont=dict(size=10)),
        yaxis=dict(gridcolor="#F1F5F9"),
    )
    return fig


# ─── Application principale ──────────────────────────────────────
def main():

    # ── 1. Chargement des données ────────────────────────────────
    csv_bytes = None

    try:
        with open("concrete_data.csv", "rb") as f:
            csv_bytes = f.read()
    except FileNotFoundError:
        pass

    if csv_bytes is None:
        st.markdown('<p class="page-title">🏗️ Résistance du Béton — Dashboard ML</p>', unsafe_allow_html=True)
        st.warning("📁 `concrete_data.csv` introuvable dans le répertoire courant.")
        uploaded = st.file_uploader("Importer le fichier CSV du dataset :", type=["csv"])
        if uploaded is None:
            st.info("ℹ️ Téléversez le fichier `concrete_data.csv` pour lancer l'application.")
            st.stop()
        csv_bytes = uploaded.read()

    # ── 2. Entraînement (mis en cache) ───────────────────────────
    rf, lr, sc, X, fi, perf, df = train_models(csv_bytes)
    cols = X.columns.tolist()
    COL_LABELS = ["Ciment", "Laitier H.F.", "Cendres vol.",
                  "Eau", "Superplast.", "Agrég. G.", "Agrég. F.", "Âge"]

    # ── 3. Sidebar — saisie du mélange ───────────────────────────
    with st.sidebar:
        st.markdown("## 🏗️ Composition du mélange")
        st.markdown("---")

        st.markdown('<p class="sb-section">Liants</p>', unsafe_allow_html=True)
        cement = st.slider("Ciment (kg/m³)",             102.0, 540.0, 350.0, 5.0)
        slag   = st.slider("Laitier haut-fourneau (kg/m³)", 0.0, 360.0, 100.0, 5.0)
        ash    = st.slider("Cendres volantes (kg/m³)",      0.0, 200.0,   0.0, 5.0)

        st.markdown('<p class="sb-section">Eau &amp; Adjuvant</p>', unsafe_allow_html=True)
        water  = st.slider("Eau (kg/m³)",                 122.0, 247.0, 182.0, 1.0)
        superp = st.slider("Superplastifiant (kg/m³)",      0.0,  32.0,   6.0, 0.5)

        st.markdown('<p class="sb-section">Granulats</p>', unsafe_allow_html=True)
        coarse = st.slider("Agrégats grossiers (kg/m³)",  801.0, 1145.0, 950.0, 10.0)
        fine   = st.slider("Agrégats fins (kg/m³)",       594.0,  993.0, 780.0, 10.0)

        st.markdown('<p class="sb-section">Maturité</p>', unsafe_allow_html=True)
        age    = st.slider("Âge de cure (jours)",             1,    365,    28,    1)

        # Ratio E/C
        st.markdown("---")
        ec = water / cement if cement > 0 else 0
        if ec < 0.45:
            ec_color, ec_label = "#059669", "✅ Excellent"
        elif ec < 0.55:
            ec_color, ec_label = "#D97706", "⚠️ Acceptable"
        else:
            ec_color, ec_label = "#DC2626", "❌ Trop élevé"

        st.markdown(
            f'<div class="ratio-pill">'
            f'  <span style="color:#6B7280;font-size:11px;font-weight:600;letter-spacing:1px">RAPPORT E/C</span><br>'
            f'  <span class="ratio-val" style="color:{ec_color}">{ec:.3f}</span>'
            f'  <span class="ratio-lbl">{ec_label}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

    # ── 4. Prédiction ────────────────────────────────────────────
    input_vec = np.array([[cement, slag, ash, water, superp, coarse, fine, float(age)]])
    pred_rf = float(rf.predict(input_vec)[0])
    pred_lr = max(0.0, float(lr.predict(sc.transform(input_vec))[0]))

    lbl, cls_str, cls_color, badge_css = classify(pred_rf)

    # ── 5. En-tête page ──────────────────────────────────────────
    st.markdown(
        '<p class="page-title">🏗️ Résistance du Béton — Dashboard ML</p>'
        '<p class="page-sub">Entrez la composition du mélange dans le panneau gauche pour obtenir une prédiction en temps réel.</p>',
        unsafe_allow_html=True,
    )

    # ── 6. Bloc résultat + jauge ─────────────────────────────────
    col_card, col_gauge = st.columns([1, 1.5], gap="large")

    with col_card:
        st.markdown(
            f'<div class="pred-card">'
            f'  <div class="pred-sub">🌲 Random Forest — Résistance prédite</div>'
            f'  <div class="pred-mpa">{pred_rf:.1f}</div>'
            f'  <div class="pred-unit">MPa</div>'
            f'  <span class="badge {badge_css}">{lbl} — Classe {cls_str}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col_gauge:
        st.plotly_chart(make_gauge(pred_rf), use_container_width=True)

    # ── 7. KPIs ──────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("📈 Rég. Linéaire",    f"{pred_lr:.1f} MPa")
    k2.metric("📊 R² – RF",          f"{perf['rf']['r2']*100:.1f}%",    delta="meilleur ✅")
    k3.metric("📉 RMSE – RF",        f"{perf['rf']['rmse']:.2f} MPa")
    k4.metric("📊 R² – Rég. Lin.",   f"{perf['lr']['r2']*100:.1f}%")
    k5.metric("📉 RMSE – Rég. Lin.", f"{perf['lr']['rmse']:.2f} MPa")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 8. Charts ────────────────────────────────────────────────
    c_fi, c_cmp = st.columns(2, gap="large")

    with c_fi:
        st.markdown("### 📊 Importance des variables")
        st.plotly_chart(make_fi_chart(fi), use_container_width=True)

    with c_cmp:
        st.markdown("### 🔢 Votre mélange vs Moyenne dataset")
        your_vals = [cement, slag, ash, water, superp, coarse, fine, float(age)]
        mean_vals = [round(df[c].mean(), 1) for c in cols]
        st.plotly_chart(
            make_compare_chart(your_vals, mean_vals, COL_LABELS),
            use_container_width=True,
        )

    # ── 9. Guide interprétation ──────────────────────────────────
    with st.expander("📚 Guide d'interprétation des classes de béton"):
        st.markdown("""
| Classe | Résistance | Usage typique |
|--------|-----------|---------------|
| **C16** (< 20 MPa) | Faible | Béton non structurel, remblai |
| **C25** (20–30 MPa) | Ordinaire | Fondations, dalles résidentielles |
| **C35** (30–45 MPa) | Normal/Structurel | Poteaux, poutres, ponts |
| **C45+** (> 45 MPa) | Haute résistance | Gratte-ciel, ouvrages d'art |

> **Rapport E/C** : plus il est faible (< 0.45), plus la résistance est élevée.  
> **Âge** : la résistance croît jusqu'à ~28 jours puis se stabilise progressivement.  
> **Ciment** : plus la teneur est élevée, plus la résistance augmente.
        """)

    # ── 10. Footer ───────────────────────────────────────────────
    st.markdown(
        '<div class="footer">'
        "Génie Civil · Projet 01/5 · Science des Matériaux<br>"
        "Dataset : <a href='https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength' "
        "style='color:#0D9488'>UCI Concrete Compressive Strength</a> — 1 030 échantillons · 8 features"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
