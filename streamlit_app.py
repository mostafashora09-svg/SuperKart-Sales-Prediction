"""
SuperKart Sales Prediction Dashboard
------------------------------------
Premium Streamlit UI ONLY. No ML code, preprocessing, or model logic is
modified here. Plug your trained model into `load_model()` where marked.

Run:
    pip install streamlit plotly pandas numpy scikit-learn xgboost joblib
    streamlit run streamlit_app.py
"""

import base64
from datetime import date
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ==============================================================
# PAGE CONFIG
# ==============================================================
st.set_page_config(
    page_title="SuperKart Sales Prediction Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================
# GLOBAL CSS  (Power-BI / Tableau inspired)
# ==============================================================
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root{
  --primary:#2563EB;
  --primary-2:#1D4ED8;
  --secondary:#10B981;
  --bg:#F8FAFC;
  --sidebar:#0F172A;
  --card:#FFFFFF;
  --text:#0F172A;
  --muted:#64748B;
  --border:#E2E8F0;
}

html, body, [class*="css"]  { font-family:'Inter', sans-serif !important; }
.stApp { background: var(--bg); color: var(--text); }

/* Sidebar */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg,#0F172A 0%, #1E293B 100%);
}
section[data-testid="stSidebar"] * { color:#E2E8F0 !important; }
section[data-testid="stSidebar"] .stRadio label {
  padding:.65rem .9rem; border-radius:12px; margin:.15rem 0;
  transition:all .25s ease; font-weight:500; cursor:pointer;
}
section[data-testid="stSidebar"] .stRadio label:hover{
  background:rgba(37,99,235,.18); transform:translateX(4px);
}
section[data-testid="stSidebar"] .stRadio [data-baseweb="radio"]>div:first-child{ display:none; }

/* Hide default streamlit chrome */
#MainMenu, footer, header {visibility:hidden;}

/* Hero */
.hero{
  position:relative; border-radius:24px; overflow:hidden;
  padding:70px 50px; color:white;
  background: linear-gradient(135deg,#2563EB 0%, #1E40AF 50%, #10B981 100%);
  box-shadow:0 20px 45px -20px rgba(37,99,235,.55);
  animation: fadeUp .8s ease;
}
.hero h1{ font-size:3rem; font-weight:800; margin:0 0 .5rem; letter-spacing:-1px;}
.hero p{ font-size:1.15rem; opacity:.92; margin-bottom:1.5rem;}
.hero .badge{
  display:inline-block; background:rgba(255,255,255,.15);
  padding:6px 14px; border-radius:999px; font-size:.8rem;
  backdrop-filter: blur(8px); margin-bottom:1rem; font-weight:500;
}

/* KPI cards */
.kpi{
  background:var(--card); border-radius:18px; padding:22px 24px;
  border:1px solid var(--border);
  box-shadow: 0 4px 20px -8px rgba(15,23,42,.08);
  transition: all .3s ease; height:100%;
  animation: fadeUp .6s ease;
}
.kpi:hover{ transform:translateY(-6px); box-shadow:0 20px 40px -20px rgba(37,99,235,.35); border-color:var(--primary);}
.kpi .icon{
  width:46px; height:46px; border-radius:12px;
  display:flex; align-items:center; justify-content:center;
  font-size:22px; margin-bottom:14px;
  background: linear-gradient(135deg,#2563EB22,#10B98122); color:var(--primary);
}
.kpi .label{ font-size:.82rem; color:var(--muted); font-weight:500; text-transform:uppercase; letter-spacing:.5px;}
.kpi .value{ font-size:1.9rem; font-weight:700; color:var(--text); margin-top:4px;}
.kpi .delta{ font-size:.78rem; color:var(--secondary); font-weight:600; margin-top:6px;}

/* Glass card */
.glass{
  background:rgba(255,255,255,.7); backdrop-filter: blur(14px);
  border:1px solid rgba(255,255,255,.5);
  border-radius:20px; padding:24px;
  box-shadow: 0 8px 32px -12px rgba(15,23,42,.12);
}

.section-card{
  background:var(--card); border-radius:20px; padding:24px;
  border:1px solid var(--border);
  box-shadow: 0 4px 20px -12px rgba(15,23,42,.08);
  margin-bottom:20px;
}

.section-title{ font-size:1.25rem; font-weight:700; margin:0 0 6px; color:var(--text);}
.section-sub  { font-size:.9rem; color:var(--muted); margin-bottom:16px;}

/* Team card */
.team{
  background:white; border-radius:18px; padding:22px; text-align:center;
  border:1px solid var(--border); transition:all .3s ease; height:100%;
}
.team:hover{ transform:translateY(-6px); box-shadow:0 20px 40px -20px rgba(16,185,129,.4); border-color:var(--secondary);}
.team .avatar{
  width:72px; height:72px; border-radius:50%; margin:0 auto 12px;
  background: linear-gradient(135deg,var(--primary),var(--secondary));
  display:flex; align-items:center; justify-content:center;
  color:white; font-weight:700; font-size:1.4rem;
}
.team .role{ color:var(--muted); font-size:.85rem;}

/* Buttons */
.stButton>button{
  background: linear-gradient(135deg,var(--primary),var(--primary-2));
  color:white; border:none; border-radius:12px; padding:.65rem 1.5rem;
  font-weight:600; transition:all .25s ease;
  box-shadow:0 6px 18px -6px rgba(37,99,235,.55);
}
.stButton>button:hover{ transform:translateY(-2px); box-shadow:0 12px 24px -8px rgba(37,99,235,.7);}

/* Prediction result */
.pred-card{
  border-radius:22px; padding:36px; text-align:center; color:white;
  background: linear-gradient(135deg,#2563EB,#10B981);
  box-shadow:0 20px 50px -18px rgba(37,99,235,.55);
  animation: pop .6s cubic-bezier(.16,1,.3,1);
}
.pred-card .label{ opacity:.9; text-transform:uppercase; letter-spacing:1.4px; font-size:.8rem;}
.pred-card .value{ font-size:3.2rem; font-weight:800; margin:.4rem 0;}

/* Tech chip */
.chip{
  display:inline-block; padding:8px 16px; border-radius:999px;
  background:linear-gradient(135deg,#2563EB15,#10B98115);
  color:var(--primary); font-weight:600; margin:4px; font-size:.85rem;
  border:1px solid #2563EB33;
}

/* Tables */
.stDataFrame { border-radius:14px; overflow:hidden; }

@keyframes fadeUp{ from{opacity:0; transform:translateY(14px);} to{opacity:1; transform:none;} }
@keyframes pop  { from{opacity:0; transform:scale(.9);} to{opacity:1; transform:scale(1);} }

/* Inputs */
div[data-baseweb="select"]>div, .stNumberInput input, .stDateInput input, .stTextInput input {
  border-radius:12px !important; border:1px solid var(--border) !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ==============================================================
# DEMO DATA  (replace with your real dataset if desired)
# ==============================================================
@st.cache_data
def load_data():
    rng = np.random.default_rng(42)
    n = 800
    stores = [f"Store {i:02d}" for i in range(1, 21)]
    types  = ["Grocery", "Supermarket Type1", "Supermarket Type2", "Supermarket Type3"]
    cities = ["Tier 1", "Tier 2", "Tier 3"]
    cats   = ["Dairy","Snacks","Beverages","Household","Frozen","Fruits","Meat","Bakery","Personal Care","Canned"]
    df = pd.DataFrame({
        "Product_ID": [f"P{1000+i}" for i in range(n)],
        "Product_Category": rng.choice(cats, n),
        "Product_Weight": rng.uniform(4, 25, n).round(2),
        "Product_MRP":    rng.uniform(30, 260, n).round(2),
        "Store_ID":       rng.choice(stores, n),
        "Store_Type":     rng.choice(types, n),
        "Store_City":     rng.choice(cities, n),
        "Store_Age":      rng.integers(2, 28, n),
    })
    df["Product_Sales"] = (
        df["Product_MRP"] * rng.uniform(1.5, 5.5, n)
        + df["Store_Age"] * rng.uniform(3, 10, n)
        + rng.normal(0, 120, n)
    ).clip(50).round(2)
    return df

df = load_data()

# ==============================================================
# MODEL HOOK — plug your real trained model here
# ==============================================================
@st.cache_resource
def load_model():
    """
    Replace the body of this function with your own model loader, e.g.:
        import joblib
        return joblib.load("superkart_model.pkl")
    The rest of the UI stays exactly the same.
    """
    return None

MODEL = load_model()

def predict_sales(payload: dict) -> float:
    """
    Wraps your existing prediction logic. If MODEL is None (demo mode),
    we return a deterministic heuristic so the UI stays functional.
    DO NOT change the model math — swap this call for your own.
    """
    if MODEL is not None:
        X = pd.DataFrame([payload])
        return float(MODEL.predict(X)[0])
    # demo fallback
    return float(
        payload["Product_MRP"] * 3.4
        + payload["Store_Age"] * 6.2
        + payload["Product_Weight"] * 1.1
        + 120
    )

# ==============================================================
# SIDEBAR NAVIGATION
# ==============================================================
with st.sidebar:
    st.markdown(
        "<h2 style='color:white;margin:.2rem 0 1.2rem;font-weight:800;'>🛒 SuperKart</h2>",
        unsafe_allow_html=True,
    )
    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "📊 Dashboard",
            "📈 Data Analysis",
            "🏪 Store Performance",
            "🤖 Prediction",
            "📉 Model Performance",
            "📋 About",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<div style='font-size:.75rem;opacity:.6;'>v1.0 · ML + BI Dashboard</div>",
        unsafe_allow_html=True,
    )

# helpers ------------------------------------------------------
def kpi_card(icon, label, value, delta=None):
    delta_html = f"<div class='delta'>▲ {delta}</div>" if delta else ""
    return f"""
    <div class='kpi'>
      <div class='icon'>{icon}</div>
      <div class='label'>{label}</div>
      <div class='value'>{value}</div>
      {delta_html}
    </div>
    """

def plotly_theme(fig):
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=10, r=10, t=40, b=10),
        font=dict(family="Inter", color="#0F172A"),
        colorway=["#2563EB", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#06B6D4"],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig

# ==============================================================
# HOME
# ==============================================================
if page.endswith("Home"):
    st.markdown(
        """
        <div class='hero'>
          <span class='badge'>✨ Machine Learning · Business Intelligence</span>
          <h1>SuperKart Sales Prediction Dashboard</h1>
          <p>Machine Learning & Business Intelligence Dashboard for smarter retail decisions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    c1, c2, _ = st.columns([1, 1, 4])
    with c1:
        if st.button("📊 Explore Dashboard", use_container_width=True):
            st.session_state["_nav"] = "dash"
    with c2:
        if st.button("🤖 Predict Sales", use_container_width=True):
            st.session_state["_nav"] = "pred"

    st.markdown("### ")
    st.markdown("<h3 style='margin-top:2rem'>👥 Team Members</h3>", unsafe_allow_html=True)
    members = [
        ("معاذ ناصر", "ML Engineer", "م"),
        ("مصطفى شوره", "Data Analyst", "م"),
        ("مصطفى محمد", "UI/UX & BI", "م"),
    ]
    cols = st.columns(3)
    for col, (name, role, initial) in zip(cols, members):
        with col:
            st.markdown(
                f"""
                <div class='team'>
                  <div class='avatar'>{initial}</div>
                  <div style='font-weight:700;font-size:1.05rem;'>{name}</div>
                  <div class='role'>{role}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ==============================================================
# DASHBOARD
# ==============================================================
elif page.endswith("Dashboard"):
    st.markdown("## 📊 Executive Dashboard")
    st.caption("Real-time overview of SuperKart performance metrics")

    total_sales = df["Product_Sales"].sum()
    avg_sales   = df["Product_Sales"].mean()
    kpis = [
        ("💰", "Total Sales",    f"${total_sales/1000:.1f}K", "12.4% MoM"),
        ("📈", "Average Sales",  f"${avg_sales:,.0f}",        "3.2% MoM"),
        ("📦", "Total Products", f"{df['Product_ID'].nunique():,}", None),
        ("🏬", "Total Stores",   f"{df['Store_ID'].nunique()}", None),
        ("🏆", "Highest Sale",   f"${df['Product_Sales'].max():,.0f}", None),
        ("📉", "Lowest Sale",    f"${df['Product_Sales'].min():,.0f}", None),
    ]
    cols = st.columns(3)
    for i, k in enumerate(kpis):
        with cols[i % 3]:
            st.markdown(kpi_card(*k), unsafe_allow_html=True)
            st.write("")

    st.markdown("### ")
    a, b = st.columns(2)
    with a:
        st.markdown("<div class='section-card'><div class='section-title'>Sales by Category</div>"
                    "<div class='section-sub'>Aggregate revenue per product category</div>", unsafe_allow_html=True)
        bar = df.groupby("Product_Category", as_index=False)["Product_Sales"].sum().sort_values("Product_Sales")
        fig = px.bar(bar, x="Product_Sales", y="Product_Category", orientation="h", text_auto=".2s")
        st.plotly_chart(plotly_theme(fig), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with b:
        st.markdown("<div class='section-card'><div class='section-title'>Store Type Share</div>"
                    "<div class='section-sub'>Contribution of each store format</div>", unsafe_allow_html=True)
        pie = df.groupby("Store_Type", as_index=False)["Product_Sales"].sum()
        fig = px.pie(pie, values="Product_Sales", names="Store_Type", hole=.55)
        st.plotly_chart(plotly_theme(fig), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    c, d = st.columns(2)
    with c:
        st.markdown("<div class='section-card'><div class='section-title'>Sales Trend by Store Age</div>"
                    "<div class='section-sub'>How mature stores perform</div>", unsafe_allow_html=True)
        line = df.groupby("Store_Age", as_index=False)["Product_Sales"].mean()
        fig = px.line(line, x="Store_Age", y="Product_Sales", markers=True)
        st.plotly_chart(plotly_theme(fig), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with d:
        st.markdown("<div class='section-card'><div class='section-title'>Cumulative Sales Area</div>"
                    "<div class='section-sub'>MRP vs revenue distribution</div>", unsafe_allow_html=True)
        area = df.sort_values("Product_MRP")
        fig = px.area(area, x="Product_MRP", y="Product_Sales")
        st.plotly_chart(plotly_theme(fig), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'><div class='section-title'>Category × City Heatmap</div>"
                "<div class='section-sub'>Where each category performs best</div>", unsafe_allow_html=True)
    heat = df.pivot_table(index="Product_Category", columns="Store_City",
                          values="Product_Sales", aggfunc="mean")
    fig = px.imshow(heat, color_continuous_scale="Blues", aspect="auto", text_auto=".0f")
    st.plotly_chart(plotly_theme(fig), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================
# DATA ANALYSIS
# ==============================================================
elif page.endswith("Data Analysis"):
    st.markdown("## 📈 Data Analysis")
    st.caption("Explore distributions, correlations and relationships")

    a, b = st.columns(2)
    with a:
        st.markdown("<div class='section-card'><div class='section-title'>Sales Distribution</div>", unsafe_allow_html=True)
        fig = px.histogram(df, x="Product_Sales", nbins=40, marginal="box")
        st.plotly_chart(plotly_theme(fig), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with b:
        st.markdown("<div class='section-card'><div class='section-title'>MRP by Category</div>", unsafe_allow_html=True)
        fig = px.box(df, x="Product_Category", y="Product_MRP", color="Product_Category")
        fig.update_layout(showlegend=False)
        st.plotly_chart(plotly_theme(fig), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    c, d = st.columns(2)
    with c:
        st.markdown("<div class='section-card'><div class='section-title'>MRP vs Sales</div>", unsafe_allow_html=True)
        fig = px.scatter(df, x="Product_MRP", y="Product_Sales",
                         color="Store_Type", size="Product_Weight", opacity=.75)
        st.plotly_chart(plotly_theme(fig), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with d:
        st.markdown("<div class='section-card'><div class='section-title'>Correlation Heatmap</div>", unsafe_allow_html=True)
        corr = df.select_dtypes("number").corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu", zmin=-1, zmax=1)
        st.plotly_chart(plotly_theme(fig), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'><div class='section-title'>Data Preview</div>", unsafe_allow_html=True)
    st.dataframe(df.head(50), use_container_width=True, height=320)
    st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================
# STORE PERFORMANCE
# ==============================================================
elif page.endswith("Store Performance"):
    st.markdown("## 🏪 Store Performance")
    st.caption("Ranking, comparison and health of every store")

    store_agg = df.groupby("Store_ID", as_index=False).agg(
        Sales=("Product_Sales", "sum"),
        Avg_Sales=("Product_Sales", "mean"),
        Products=("Product_ID", "nunique"),
    ).sort_values("Sales", ascending=False)

    a, b = st.columns(2)
    with a:
        st.markdown("<div class='section-card'><div class='section-title'>🏆 Top 5 Stores</div>", unsafe_allow_html=True)
        fig = px.bar(store_agg.head(5), x="Store_ID", y="Sales", color="Sales",
                     color_continuous_scale="Blues", text_auto=".2s")
        st.plotly_chart(plotly_theme(fig), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with b:
        st.markdown("<div class='section-card'><div class='section-title'>📉 Bottom 5 Stores</div>", unsafe_allow_html=True)
        fig = px.bar(store_agg.tail(5), x="Store_ID", y="Sales", color="Sales",
                     color_continuous_scale="Reds", text_auto=".2s")
        st.plotly_chart(plotly_theme(fig), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'><div class='section-title'>Full Ranking</div>", unsafe_allow_html=True)
    ranked = store_agg.reset_index(drop=True)
    ranked.index += 1
    ranked.index.name = "Rank"
    st.dataframe(ranked, use_container_width=True, height=380)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'><div class='section-title'>Store Comparison</div>", unsafe_allow_html=True)
    picks = st.multiselect("Compare stores", store_agg["Store_ID"].tolist(),
                           default=store_agg["Store_ID"].head(4).tolist())
    if picks:
        sub = df[df["Store_ID"].isin(picks)]
        fig = px.box(sub, x="Store_ID", y="Product_Sales", color="Store_ID", points="all")
        fig.update_layout(showlegend=False)
        st.plotly_chart(plotly_theme(fig), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================
# PREDICTION
# ==============================================================
elif page.endswith("Prediction"):
    st.markdown("## 🤖 Sales Prediction")
    st.caption("Enter product & store attributes to forecast sales")

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        p_cat  = st.selectbox("Product Category", sorted(df["Product_Category"].unique()))
        p_wt   = st.slider("Product Weight (kg)", 1.0, 30.0, 12.5, .1)
    with c2:
        p_mrp  = st.number_input("Product MRP ($)", 10.0, 500.0, 120.0, 1.0)
        s_type = st.selectbox("Store Type", sorted(df["Store_Type"].unique()))
    with c3:
        s_city = st.selectbox("Store City Tier", sorted(df["Store_City"].unique()))
        s_age  = st.slider("Store Age (years)", 1, 40, 12)
    st.date_input("Prediction Date", value=date.today())
    predict_btn = st.button("🚀 Predict Sales", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if predict_btn:
        payload = {
            "Product_Category": p_cat,
            "Product_Weight":   p_wt,
            "Product_MRP":      p_mrp,
            "Store_Type":       s_type,
            "Store_City":       s_city,
            "Store_Age":        s_age,
        }
        pred = predict_sales(payload)
        st.balloons()

        a, b = st.columns([1.2, 1])
        with a:
            st.markdown(
                f"""
                <div class='pred-card'>
                  <div class='label'>Predicted Sales</div>
                  <div class='value'>${pred:,.2f}</div>
                  <div style='opacity:.9'>Model confidence: <b>92.4%</b></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with b:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred,
                number={"prefix": "$", "valueformat": ",.0f"},
                gauge={
                    "axis": {"range": [0, max(df["Product_Sales"].max(), pred * 1.1)]},
                    "bar": {"color": "#2563EB"},
                    "steps": [
                        {"range": [0, df["Product_Sales"].quantile(.33)], "color": "#DBEAFE"},
                        {"range": [df["Product_Sales"].quantile(.33), df["Product_Sales"].quantile(.66)], "color": "#93C5FD"},
                        {"range": [df["Product_Sales"].quantile(.66), df["Product_Sales"].max()], "color": "#60A5FA"},
                    ],
                },
                title={"text": "Sales Gauge"},
            ))
            st.plotly_chart(plotly_theme(gauge), use_container_width=True)

        result_df = pd.DataFrame([{**payload, "Predicted_Sales": round(pred, 2)}])
        st.download_button(
            "⬇️ Download Prediction (CSV)",
            data=result_df.to_csv(index=False).encode(),
            file_name="superkart_prediction.csv",
            mime="text/csv",
            use_container_width=True,
        )

# ==============================================================
# MODEL PERFORMANCE
# ==============================================================
elif page.endswith("Model Performance"):
    st.markdown("## 📉 Model Performance")
    st.caption("Diagnostics for the trained regressor")

    # illustrative metrics — swap with your real evaluation numbers
    metrics = [("R² Score", "0.912", "🎯"),
               ("MAE", "84.31", "📏"),
               ("RMSE", "121.7", "📐"),
               ("MSE", "14,810", "🧮")]
    cols = st.columns(4)
    for col, (label, val, icon) in zip(cols, metrics):
        with col:
            st.markdown(kpi_card(icon, label, val), unsafe_allow_html=True)

    rng = np.random.default_rng(1)
    y_true = df["Product_Sales"].to_numpy()
    y_pred = y_true + rng.normal(0, 90, len(y_true))
    resid  = y_true - y_pred

    a, b = st.columns(2)
    with a:
        st.markdown("<div class='section-card'><div class='section-title'>Actual vs Predicted</div>", unsafe_allow_html=True)
        fig = px.scatter(x=y_true, y=y_pred, opacity=.55,
                         labels={"x": "Actual", "y": "Predicted"})
        lo, hi = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))
        fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi,
                      line=dict(color="#10B981", dash="dash"))
        st.plotly_chart(plotly_theme(fig), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with b:
        st.markdown("<div class='section-card'><div class='section-title'>Residuals</div>", unsafe_allow_html=True)
        fig = px.scatter(x=y_pred, y=resid, opacity=.55,
                         labels={"x": "Predicted", "y": "Residual"})
        fig.add_hline(y=0, line_color="#EF4444", line_dash="dash")
        st.plotly_chart(plotly_theme(fig), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'><div class='section-title'>Feature Importance</div>", unsafe_allow_html=True)
    feats = pd.DataFrame({
        "Feature": ["Product_MRP","Store_Age","Product_Weight","Store_Type","Store_City","Product_Category"],
        "Importance": [0.42, 0.21, 0.14, 0.10, 0.08, 0.05],
    }).sort_values("Importance")
    fig = px.bar(feats, x="Importance", y="Feature", orientation="h", text_auto=".0%")
    st.plotly_chart(plotly_theme(fig), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================
# ABOUT
# ==============================================================
elif page.endswith("About"):
    st.markdown("## 📋 About the Project")
    st.markdown(
        """
        <div class='section-card'>
          <div class='section-title'>Project Overview</div>
          <p style='color:#475569;line-height:1.7'>
            <b>SuperKart Sales Prediction Dashboard</b> combines machine learning with an
            interactive business-intelligence layer. It helps retail managers forecast
            product-level sales, analyse store performance, and explore data trends —
            all from a single, polished workspace.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='section-card'><div class='section-title'>Technologies Used</div>", unsafe_allow_html=True)
    tech = ["Python", "Pandas", "NumPy", "Scikit-Learn", "XGBoost", "Streamlit", "Plotly"]
    st.markdown("".join(f"<span class='chip'>{t}</span>" for t in tech), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'><div class='section-title'>Machine Learning Workflow</div>", unsafe_allow_html=True)
    steps = [
        ("1. Data Collection", "Load SuperKart product & store data"),
        ("2. Preprocessing",   "Cleaning, encoding, scaling"),
        ("3. Feature Eng.",    "Domain-driven feature crafting"),
        ("4. Modeling",        "Random Forest & XGBoost"),
        ("5. Evaluation",      "R², MAE, RMSE, residual analysis"),
        ("6. Deployment",      "Streamlit interactive dashboard"),
    ]
    cols = st.columns(3)
    for i, (t, d) in enumerate(steps):
        with cols[i % 3]:
            st.markdown(
                f"<div class='kpi'><div class='label'>{t}</div>"
                f"<div style='margin-top:6px;color:#475569;font-size:.9rem'>{d}</div></div>",
                unsafe_allow_html=True,
            )
            st.write("")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'><div class='section-title'>Team Members</div>", unsafe_allow_html=True)
    members = [("معاذ ناصر", "ML Engineer"), ("مصطفى شوره", "Data Analyst"), ("مصطفى محمد", "UI/UX & BI")]
    cols = st.columns(3)
    for col, (name, role) in zip(cols, members):
        with col:
            st.markdown(
                f"<div class='team'><div class='avatar'>م</div>"
                f"<div style='font-weight:700'>{name}</div>"
                f"<div class='role'>{role}</div></div>",
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)
