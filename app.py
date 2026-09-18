"""A polished Streamlit interface for the saved restaurant-revenue model."""
from pathlib import Path
import json
import pickle
import pandas as pd
import streamlit as st

from restaurant_pipeline import RestaurantRevenuePipeline

ROOT = Path(__file__).resolve().parent
CUISINE_IMAGES = {
    "Chinese": "https://images.unsplash.com/photo-1559314809-0d155014e29e?auto=format&fit=crop&w=1000&q=85",
    "North Indian": "https://images.unsplash.com/photo-1585937421612-70a008356fbe?auto=format&fit=crop&w=1000&q=85",
    "Continental": "https://images.unsplash.com/photo-1547592180-85f173990554?auto=format&fit=crop&w=1000&q=85",
    "Italian": "https://images.unsplash.com/photo-1473093295043-cdd812d0e601?auto=format&fit=crop&w=1000&q=85",
    "Fast Food": "https://images.unsplash.com/photo-1568901346375-23c9450c58cd?auto=format&fit=crop&w=1000&q=85",
}

@st.cache_resource
def load_model():
    with open(ROOT / "model.pkl", "rb") as file:
        return pickle.load(file)

model = load_model()
st.set_page_config(page_title="Restaurant Revenue Predictor", page_icon="✦", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');
:root { --ink:#0d0d0d; --panel:#161616; --muted:#aaa49e; --cream:#f7f1e8; --red:#e8403d; --line:rgba(255,255,255,.12); }
.stApp { background: radial-gradient(circle at 82% 8%, rgba(232,64,61,.16), transparent 23%), radial-gradient(circle at 10% 36%, rgba(232,64,61,.09), transparent 26%), #0b0b0b; color:var(--cream); font-family:'DM Sans',sans-serif; }
header[data-testid="stHeader"] { background:transparent; }
[data-testid="stToolbar"] { right:1rem; }
.block-container { max-width:1180px; padding-top:1.3rem; padding-bottom:3rem; }
.eyebrow { color:var(--red); text-transform:uppercase; letter-spacing:.17em; font-weight:700; font-size:.72rem; margin-bottom:.45rem; }
.hero-title { font-family:'Playfair Display',serif; font-size:clamp(2.7rem,6vw,5.4rem); font-weight:700; line-height:.92; letter-spacing:-.04em; color:var(--cream); margin:0; }
.hero-title span { color:var(--red); }
.hero-copy { color:var(--muted); max-width:31rem; line-height:1.65; margin:1.2rem 0 1.7rem; }
.brand { display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid var(--line); padding:0 0 1rem; margin-bottom:3rem; }
.brand-name { font-family:'Playfair Display',serif; font-size:1.3rem; color:var(--cream); }.brand-dot{color:var(--red)}
.tag { color:var(--muted); border:1px solid var(--line); border-radius:100px; padding:.4rem .8rem; font-size:.75rem; }
.hero-art { min-height:330px; border:1px solid var(--line); border-radius:26px; background: radial-gradient(circle at 49% 46%, #ee6a4e 0 11%, #451913 11.5% 13%, #e8d6aa 13.5% 20%, #111 20.5% 43%, transparent 43.5%), radial-gradient(circle at 66% 30%, #e8403d 0 6%, transparent 6.5%), radial-gradient(circle at 28% 78%, #a83228 0 4%, transparent 4.5%), #171717; position:relative; overflow:hidden; }
.hero-art:before { content:'REVENUE / PLANNING'; position:absolute; bottom:24px; left:26px; color:var(--cream); letter-spacing:.14em; font-size:.68rem; font-weight:700; }.hero-art:after { content:'01'; position:absolute; right:24px; top:20px; color:var(--red); font:700 3rem 'Playfair Display',serif; }
.section-title { font-family:'Playfair Display',serif; color:var(--cream); font-size:2rem; margin:3.2rem 0 .25rem; }.section-copy { color:var(--muted); margin-bottom:1.2rem; }
.metric-card { background:linear-gradient(135deg,rgba(255,255,255,.075),rgba(255,255,255,.025)); border:1px solid var(--line); border-radius:18px; padding:1.05rem 1.1rem; }.metric-label{color:var(--muted);font-size:.78rem;text-transform:uppercase;letter-spacing:.1em}.metric-value{color:var(--cream);font:700 1.45rem 'Playfair Display',serif;margin-top:.25rem}
[data-testid="stForm"] { background:rgba(22,22,22,.9); border:1px solid var(--line); border-radius:22px; padding:1.35rem; }
label, [data-testid="stWidgetLabel"] p { color:var(--cream)!important; font-weight:600!important; }
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div { background:#202020!important; border-color:#353535!important; color:var(--cream)!important; border-radius:10px!important; }
input { color:var(--cream)!important; }.stSlider [data-baseweb="slider"] div[role="slider"] { background:var(--red)!important; }
.stButton button, [data-testid="stFormSubmitButton"] button { background:var(--red)!important; color:white!important; border:0!important; border-radius:100px!important; padding:.65rem 1.25rem!important; font-weight:700!important; width:100%; transition:transform .15s ease; }.stButton button:hover, [data-testid="stFormSubmitButton"] button:hover { transform:translateY(-2px); background:#f1504c!important; }
.result { background:linear-gradient(125deg,#e8403d,#a91f25); border-radius:22px; padding:1.5rem 1.6rem; color:white; margin-top:1.1rem; }.result-kicker{opacity:.8;text-transform:uppercase;letter-spacing:.14em;font-size:.72rem;font-weight:700}.result-value{font:700 clamp(2.5rem,5vw,4.3rem) 'Playfair Display',serif;line-height:1.1;margin:.25rem 0}.result-copy{opacity:.88;max-width:38rem}
.note { color:var(--muted); font-size:.83rem; line-height:1.55; margin-top:.9rem; } .stAlert { border-radius:12px!important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""<div class="brand"><div class="brand-name">Restaurant Revenue<span class="brand-dot">.</span> Predictor</div><div class="tag">Restaurant intelligence, made simple</div></div>""", unsafe_allow_html=True)
hero_text, hero_art = st.columns([1.05, .95], gap="large")
with hero_text:
    st.markdown('<div class="eyebrow">Monthly revenue planner</div><h1 class="hero-title">Restaurant<br>revenue <span>clarity.</span></h1><p class="hero-copy">Build a restaurant scenario, then turn operating choices into a clear, data-driven monthly revenue estimate.</p>', unsafe_allow_html=True)
    a, b, c = st.columns(3)
    for col, label, value in [(a, "Model fit", "97.45%"), (b, "Test MAE", "INR 8.7k"), (c, "Ready", "Instant")]:
        with col: st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)
with hero_art:
    st.markdown('<div class="hero-art"></div>', unsafe_allow_html=True)

planner_tab, performance_tab, guide_tab = st.tabs(["Revenue planner", "Model performance", "How to use it"])

with planner_tab:
    planner_intro, food_preview = st.columns([1.1, .9], gap="large")
    with planner_intro:
        st.markdown('<h2 class="section-title">Design your outlet</h2><p class="section-copy">Choose a cuisine to see a dish preview, then adjust the operating details below.</p>', unsafe_allow_html=True)
        cuisine = st.selectbox("Cuisine type", ["Chinese", "North Indian", "Continental", "Italian", "Fast Food"], key="cuisine_preview")
    with food_preview:
        st.image(CUISINE_IMAGES[cuisine], caption=f"{cuisine} menu inspiration", use_container_width=True)
    with st.form("prediction_form"):
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown('<div class="eyebrow">Outlet profile</div>', unsafe_allow_html=True)
            location = st.selectbox("Location type", ["High Street", "Food Court", "Airport", "Standalone", "Mall"])
            customers = st.slider("Average daily customers", 1, 750, 250, 5)
            bill = st.slider("Average bill value (INR)", 50, 10000, 500, 50)
            seating = st.slider("Seating capacity", 1, 300, 80, 5)
        with right:
            st.markdown('<div class="eyebrow">Service & growth</div>', unsafe_allow_html=True)
            delivery = st.slider("Delivery orders (%)", 0, 100, 35, 1)
            employees = st.slider("Employee count", 1, 80, 15, 1)
            promotion = st.slider("Monthly promotion spend (INR)", 0, 150000, 30000, 1000)
            rating = st.slider("Customer rating", 1.0, 5.0, 4.0, .1)
        submitted = st.form_submit_button("Reveal revenue estimate")
    if submitted:
        row = pd.DataFrame([{"Average_Daily_Customers": customers, "Average_Bill_Value": bill, "Delivery_Order_Percentage": delivery, "Seating_Capacity": seating, "Employee_Count": employees, "Promotion_Spend": promotion, "Customer_Rating": rating, "Cuisine_Type": cuisine, "Location_Type": location}])
        prediction = max(0, float(model.predict(row)[0])); daily = prediction / 30
        st.markdown(f'<div class="result"><div class="result-kicker">Estimated monthly revenue</div><div class="result-value">INR {prediction:,.0f}</div><div class="result-copy">That is approximately INR {daily:,.0f} per day for your {cuisine.lower()} outlet in a {location.lower()} setting.</div></div>', unsafe_allow_html=True)
        r1, r2, r3 = st.columns(3); r1.metric("Estimated daily revenue", f"INR {daily:,.0f}"); r2.metric("Guests per seat", f"{customers / seating:.1f}"); r3.metric("Promotion intensity", f"INR {promotion / max(customers * 30, 1):.0f}/guest")
        st.markdown('<p class="note">Planning estimate only: use this to compare scenarios alongside local demand, seasonality, and operational judgment.</p>', unsafe_allow_html=True)

with performance_tab:
    metrics = json.loads((ROOT / "model_metrics.json").read_text(encoding="utf-8"))
    st.markdown('<h2 class="section-title">Trust the estimate</h2><p class="section-copy">A quick view of the model evaluation on unseen restaurant records.</p>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test R-squared", metrics["test_r_squared"]); m2.metric("Test MAE", f"INR {metrics['test_mae_in_INR']:,.0f}"); m3.metric("Test RMSE", f"INR {metrics['test_rmse_in_INR']:,.0f}"); m4.metric("Test outlets", metrics["test_rows"])
    st.markdown('<div class="metric-card"><div class="metric-label">What this means</div><div class="metric-value">Strong scenario guidance</div><p class="note">The model explains about 97% of the variation in the held-out test data. It is designed for planning and comparisons, not as a guarantee of actual sales.</p></div>', unsafe_allow_html=True)

with guide_tab:
    st.markdown('<h2 class="section-title">Use it like a manager</h2><p class="section-copy">Create a baseline, then change one lever at a time to compare decisions.</p>', unsafe_allow_html=True)
    g1, g2, g3 = st.columns(3)
    for col, number, heading, copy in [(g1, "01", "Set the outlet", "Choose the cuisine and the location type."), (g2, "02", "Shape the service", "Tune traffic, seating, delivery, staffing, and promotion."), (g3, "03", "Compare scenarios", "Record the revenue estimate before choosing a plan.")]:
        with col: st.markdown(f'<div class="metric-card"><div class="metric-label">{number}</div><div class="metric-value">{heading}</div><p class="note">{copy}</p></div>', unsafe_allow_html=True)
