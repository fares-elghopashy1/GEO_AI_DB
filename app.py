import streamlit as st
import pandas as pd
import folium
import numpy as np
import shap
import matplotlib.pyplot as plt
import json

from streamlit_folium import st_folium

# Models
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

MODEL_NAMES = [
    "Random Forest",
    "XGBoost",
    "LightGBM",
    "Linear Regression",
]


def build_model(model_name: str):
    if model_name == "Random Forest":
        return RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)
    if model_name == "XGBoost":
        return XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
        )
    if model_name == "LightGBM":
        return LGBMRegressor(
            n_estimators=120,
            learning_rate=0.05,
            num_leaves=10,
            max_depth=4,
            min_child_samples=8,
            subsample=0.85,
            random_state=42,
            verbose=-1,
        )
    return LinearRegression()


def regression_hit_accuracy(y_true, y_pred, relative_tol: float = 0.15) -> float:
    """Share of points where absolute relative error is within ``relative_tol`` (regression 'accuracy')."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    denom = np.maximum(np.abs(y_true), 1e-6)
    return float(np.mean(np.abs(y_true - y_pred) / denom <= relative_tol))


@st.cache_data
def benchmark_models_cv(X: pd.DataFrame, y: pd.Series, n_splits: int = 5, random_state: int = 42):
    """Mean / std metrics per model across K-fold CV."""
    n_samples = len(X)
    if n_samples < 3:
        raise ValueError("Dataset too small for cross-validation (need ≥3 rows).")
    n_splits = int(min(max(2, n_splits), n_samples - 1))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    Xv = X.values
    yv = y.values.astype(np.float64)

    rows = []
    for name in MODEL_NAMES:
        rmse_f, r2_f, acc_f = [], [], []
        for train_i, val_i in kf.split(Xv):
            m = build_model(name)
            m.fit(Xv[train_i], yv[train_i])
            pred = m.predict(Xv[val_i])
            rmse_f.append(np.sqrt(mean_squared_error(yv[val_i], pred)))
            r2_f.append(r2_score(yv[val_i], pred))
            acc_f.append(regression_hit_accuracy(yv[val_i], pred))
        rows.append(
            {
                "Model": name,
                "RMSE": float(np.mean(rmse_f)),
                "RMSE_std": float(np.std(rmse_f)),
                "R²": float(np.mean(r2_f)),
                "R²_std": float(np.std(r2_f)),
                "Accuracy": float(np.mean(acc_f)),
                "Accuracy_std": float(np.std(acc_f)),
            }
        )
    return pd.DataFrame(rows)

# ---------------------
# Page Config
# ---------------------
st.set_page_config(
    page_title="Geo AI Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

_UI_CSS = """
<style>
    /* Root layout */
    .block-container {
        padding-top: 1.25rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {
        background: rgba(10, 12, 16, 0.85);
        backdrop-filter: blur(10px);
    }
    /* Typography */
    html, body, [data-testid="stAppViewContainer"] {
        font-feature-settings: "cv02", "cv03", "cv04", "cv11";
    }
    /* Hero */
    .geo-hero {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.12) 0%, rgba(10, 12, 16, 0) 55%),
                    linear-gradient(180deg, #12151c 0%, #0a0c10 100%);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 16px;
        padding: 1.75rem 1.85rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.35);
    }
    .geo-hero h1 {
        font-size: 1.65rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        margin: 0 0 0.35rem 0;
        color: #f8fafc;
        line-height: 1.2;
    }
    .geo-hero p {
        margin: 0;
        color: #94a3b8;
        font-size: 0.98rem;
        line-height: 1.5;
        max-width: 52rem;
    }
    .geo-badge {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #38bdf8;
        background: rgba(56, 189, 248, 0.12);
        border: 1px solid rgba(56, 189, 248, 0.25);
        padding: 0.25rem 0.55rem;
        border-radius: 6px;
        margin-bottom: 0.65rem;
    }
    /* Section title */
    .geo-section-title {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748b;
        margin: 0 0 0.75rem 0;
    }
    /* Metric cards */
    div[data-testid="column"] .metric-card-wrap {
        background: linear-gradient(180deg, #151a24 0%, #12151c 100%);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 14px;
        padding: 1.15rem 1.25rem;
        min-height: 5.5rem;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.2);
    }
    .metric-card-wrap .mc-label {
        font-size: 0.78rem;
        font-weight: 500;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.35rem;
    }
    .metric-card-wrap .mc-value {
        font-size: 1.65rem;
        font-weight: 700;
        color: #f1f5f9;
        letter-spacing: -0.02em;
        font-variant-numeric: tabular-nums;
    }
    .metric-card-wrap .mc-accent {
        color: #38bdf8;
    }
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: rgba(20, 24, 33, 0.6);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(148, 163, 184, 0.1);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.55rem 1rem;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(56, 189, 248, 0.15) !important;
        color: #38bdf8 !important;
    }
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1218 0%, #0a0c10 100%);
        border-right: 1px solid rgba(148, 163, 184, 0.08);
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }
    /* Dataframe / chart container */
    .stDataFrame, [data-testid="stVegaLiteChart"] {
        border-radius: 12px !important;
    }
    /* Map color ↔ cell id legend */
    .map-legend-bar {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 0.75rem 1.25rem;
        padding: 0.65rem 1rem;
        margin: 0.35rem 0 0.85rem 0;
        background: rgba(18, 21, 28, 0.95);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 10px;
    }
    .map-legend-item {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
    }
    .map-legend-swatch {
        display: inline-block;
        width: 16px;
        height: 16px;
        border-radius: 4px;
        border: 1px solid rgba(248, 250, 252, 0.35);
        flex-shrink: 0;
    }
    .map-legend-id {
        font-weight: 600;
        font-size: 0.95rem;
        color: #f1f5f9;
        font-variant-numeric: tabular-nums;
    }
    .map-legend-rank {
        font-size: 0.78rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
</style>
"""


def _inject_ui_css():
    st.markdown(_UI_CSS, unsafe_allow_html=True)


def _hero():
    st.markdown(
        """
        <div class="geo-hero">
            <div class="geo-badge">Spatial intelligence</div>
            <h1>Geo AI recommendation</h1>
            <p>Rank grid cells by modeled opportunity, compare algorithms, and review map-based
            recommendations with transparent model signals.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _metric_card(label: str, value: str, accent: bool = False):
    cls = "mc-value mc-accent" if accent else "mc-value"
    st.markdown(
        f'<div class="metric-card-wrap"><div class="mc-label">{label}</div><div class="{cls}">{value}</div></div>',
        unsafe_allow_html=True,
    )


def _iter_xy_coords(geometry_obj):
    """Yield (lon, lat) pairs from GeoJSON Polygon/MultiPolygon geometry."""
    if not isinstance(geometry_obj, dict):
        return
    gtype = geometry_obj.get("type")
    coords = geometry_obj.get("coordinates", [])
    if gtype == "Polygon":
        for ring in coords:
            for pt in ring:
                if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    yield float(pt[0]), float(pt[1])
    elif gtype == "MultiPolygon":
        for poly in coords:
            for ring in poly:
                for pt in ring:
                    if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                        yield float(pt[0]), float(pt[1])


def _geometry_bounds(geometry_series: pd.Series):
    """Return [min_lon, min_lat, max_lon, max_lat] for a series of GeoJSON geometries."""
    xs, ys = [], []
    for geom in geometry_series:
        for lon, lat in _iter_xy_coords(geom):
            xs.append(lon)
            ys.append(lat)
    if not xs or not ys:
        return [31.44, 31.42, 31.68, 31.46]
    return [min(xs), min(ys), max(xs), max(ys)]


def _geometry_center(geometry_series: pd.Series):
    b = _geometry_bounds(geometry_series)
    return ((b[1] + b[3]) / 2.0, (b[0] + b[2]) / 2.0)


# ---------------------
# Load Data
# ---------------------
@st.cache_data
def load_data():
    df = pd.read_csv("final_df.csv")
    with open("grid_features.geojson", "r", encoding="utf-8") as f:
        geojson = json.load(f)
    geo = pd.DataFrame(
        [
            {"cell_id": feat["properties"]["cell_id"], "geometry": feat["geometry"]}
            for feat in geojson.get("features", [])
            if "properties" in feat and "geometry" in feat and "cell_id" in feat["properties"]
        ]
    )

    if "geometry" in df.columns:
        df = df.drop(columns=["geometry"])

    svc_cols = ["cafe", "pharmacy", "restaurant", "supermarket"]
    for s in svc_cols:
        tgt = f"{s}_target_score"
        if tgt not in df.columns:
            qg = df["quality_gap"] if "quality_gap" in df.columns else np.where(
                df["avg_reviews"] > 0,
                (5 - df["avg_rating"]) * np.log1p(df["avg_reviews"]),
                2.0,
            )
            df[tgt] = (
                np.log1p(df["population_sum"].astype(float))
                * (pd.Series(qg).astype(float) + 0.1)
                / (1.0 + df[s].astype(float))
            )

    df = df.merge(
        geo[["cell_id", "geometry"]],
        on="cell_id",
        how="left",
    )

    return df


_inject_ui_css()
_hero()

gdf = load_data()

# ---------------------
# Sidebar
# ---------------------
with st.sidebar:
    st.markdown("### Controls")
    st.caption("Pick a vertical. Models are compared with CV; AutoML keeps the best RMSE.")

    service = st.selectbox(
        "Service vertical",
        ["cafe", "pharmacy", "restaurant", "supermarket"],
        label_visibility="visible",
    )

    model_mode = st.radio(
        "Model selection",
        ["AutoML (best RMSE)", "Manual"],
        horizontal=False,
    )

    model_name = None
    if model_mode == "Manual":
        model_name = st.selectbox("Model", MODEL_NAMES)
    else:
        st.caption("The app scores every candidate with 5-fold CV and picks the lowest mean RMSE.")

    st.divider()

score_col = f"{service}_target_score"

# ---------------------
# Feature Engineering
# ---------------------
gdf["activity_vibrancy"] = np.log1p(gdf["avg_reviews"])

gdf["quality_gap"] = np.where(
    gdf["avg_reviews"] > 0,
    (5 - gdf["avg_rating"]) * np.log1p(gdf["avg_reviews"]),
    2,
)

features = [
    "population_sum",
    service,
    "cafe",
    "pharmacy",
    "restaurant",
    "supermarket",
    "activity_vibrancy",
    "quality_gap",
]

features = list(dict.fromkeys(features))

X = gdf[features].fillna(0)
y = gdf[score_col]

# ---------------------
# CV benchmark + AutoML or manual model
# ---------------------
cv_results = benchmark_models_cv(X, y, n_splits=5, random_state=42)
best_row = cv_results.sort_values("RMSE", ascending=True).iloc[0]
if model_mode == "AutoML (best RMSE)":
    model_name = str(best_row["Model"])
else:
    model_name = str(model_name)

model = build_model(model_name)
model.fit(X, y)

with st.sidebar:
    auto_note = " (AutoML)" if model_mode == "AutoML (best RMSE)" else ""
    st.markdown(
        f'<p style="font-size:0.85rem;color:#64748b;margin:0;">Active learner{auto_note}</p>'
        f'<p style="font-size:1rem;font-weight:600;color:#38bdf8;margin:0.2rem 0 0 0;">{model_name}</p>',
        unsafe_allow_html=True,
    )
    if model_mode == "AutoML (best RMSE)":
        st.caption(f"Best mean CV RMSE: {best_row['RMSE']:.4f}")

gdf["predicted_score"] = model.predict(X)

top3 = gdf.sort_values(by="predicted_score", ascending=False).head(3)

# ---------------------
# KPIs
# ---------------------
st.markdown('<p class="geo-section-title">Overview</p>', unsafe_allow_html=True)

k1, k2, k3 = st.columns(3)
with k1:
    _metric_card("Best cell ID", str(int(top3.iloc[0]["cell_id"])), accent=True)
with k2:
    _metric_card("Top predicted score", f"{top3.iloc[0]['predicted_score']:.2f}")
with k3:
    _metric_card("Highlighted zones", "3")

st.markdown("<div style='height:1.25rem'></div>", unsafe_allow_html=True)

# ---------------------
# Layout Tabs
# ---------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["Map", "Rankings", "Model benchmark", "Explainability"]
)

# ---------------------
# MAP
# ---------------------
with tab1:
    st.markdown('<p class="geo-section-title">Geographic view</p>', unsafe_allow_html=True)
    st.caption("Top three cells by predicted score — styled for quick comparison on the basemap.")

    map_colors = ["#fbbf24", "#38bdf8", "#34d399"]

    _legend_items = []
    for _i, (_, _row) in enumerate(top3.iterrows()):
        _cid = int(_row["cell_id"])
        _c = map_colors[_i]
        _legend_items.append(
            f'<div class="map-legend-item">'
            f'<span class="map-legend-swatch" style="background:{_c};"></span>'
            f'<span class="map-legend-id">Cell {_cid}</span>'
            f'<span class="map-legend-rank">Rank {_i + 1}</span>'
            f"</div>"
        )
    st.markdown(
        '<div class="map-legend-bar">' + "".join(_legend_items) + "</div>",
        unsafe_allow_html=True,
    )

    _center_lat, _center_lon = _geometry_center(top3["geometry"])
    m = folium.Map(
        location=[_center_lat, _center_lon],
        zoom_start=13,
        tiles="CartoDB dark_matter",
        attr="CartoDB",
    )

    colors = map_colors
    fg = folium.FeatureGroup(name="Top cells")
    for i, (_, row) in enumerate(top3.iterrows()):
        folium.GeoJson(
            row["geometry"],
            style_function=lambda x, c=colors[i]: {
                "fillColor": c,
                "color": "#f8fafc",
                "weight": 2,
                "fillOpacity": 0.55,
            },
            tooltip=folium.Tooltip(
                f"<b>Rank {i + 1}</b> · cell <b>{int(row['cell_id'])}</b><br/>"
                f"Score: {row['predicted_score']:.2f}",
            ),
        ).add_to(fg)
    fg.add_to(m)

    b = _geometry_bounds(top3["geometry"])
    m.fit_bounds([[b[1], b[0]], [b[3], b[2]]])

    folium.LayerControl(collapsed=True).add_to(m)

    st_folium(m, width=1200, height=520)

# ---------------------
# DATA
# ---------------------
with tab2:
    st.markdown('<p class="geo-section-title">Leaderboard</p>', unsafe_allow_html=True)
    st.caption("Numeric summary of the same three cells shown on the map.")

    rank_tbl = top3[["cell_id", "predicted_score"]].copy()
    rank_tbl["predicted_score"] = rank_tbl["predicted_score"].round(3)
    st.dataframe(rank_tbl, width="stretch", hide_index=True)

    st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)
    chart_df = top3.set_index("cell_id")[["predicted_score"]].rename(
        columns={"predicted_score": "score"}
    )
    st.bar_chart(chart_df)

# ---------------------
# Model benchmark (CV + AutoML)
# ---------------------
with tab3:
    st.markdown('<p class="geo-section-title">Cross-validation leaderboard</p>', unsafe_allow_html=True)
    st.caption(
        "RMSE / R² are fold-wise means (± std). **Accuracy** = fraction of validation rows where "
        "|error| ≤ 15% of |actual| (a regression-friendly hit rate, not classification accuracy)."
    )

    display_df = cv_results.assign(
        RMSE_fmt=cv_results["RMSE"].map(lambda v: f"{v:.4f}") + " ± " + cv_results["RMSE_std"].map(lambda v: f"{v:.4f}"),
        R2_fmt=cv_results["R²"].map(lambda v: f"{v:.3f}") + " ± " + cv_results["R²_std"].map(lambda v: f"{v:.3f}"),
        Accuracy_pct=(cv_results["Accuracy"] * 100).map(lambda v: f"{v:.1f}%")
        + " ± "
        + (cv_results["Accuracy_std"] * 100).map(lambda v: f"{v:.1f}%"),
    )[["Model", "RMSE_fmt", "R2_fmt", "Accuracy_pct"]]
    display_df = display_df.rename(
        columns={
            "RMSE_fmt": "RMSE (mean ± std)",
            "R2_fmt": "R² (mean ± std)",
            "Accuracy_pct": "Accuracy (mean ± std)",
        }
    )

    st.dataframe(display_df, width="stretch", hide_index=True)

    chart_cmp = cv_results.set_index("Model")[["RMSE"]].sort_values("RMSE", ascending=True)
    st.markdown('<p class="geo-section-title">RMSE (lower is better)</p>', unsafe_allow_html=True)
    st.bar_chart(chart_cmp)

    if model_mode == "AutoML (best RMSE)":
        st.success(
            f"**AutoML choice:** `{model_name}` — lowest mean CV RMSE among {len(MODEL_NAMES)} candidates."
        )
    else:
        st.info(f"Manual mode: scoring uses `{model_name}` (CV table is still shown for comparison).")

# ---------------------
# Insights
# ---------------------
with tab4:
    st.markdown('<p class="geo-section-title">Model narrative</p>', unsafe_allow_html=True)

    if model_name == "Linear Regression":
        st.info(
            "SHAP tree explanations are not available for ordinary least squares. "
            "Switch to a tree-based model to see a waterfall plot for the top cell."
        )
    else:
        plt.style.use("dark_background")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        top_idx = top3.index[0]

        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor("#0a0c10")
        ax.set_facecolor("#0a0c10")

        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[top_idx],
                base_values=explainer.expected_value,
                data=X.iloc[top_idx],
                feature_names=X.columns,
            ),
            show=False,
        )
        fig.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)
        plt.style.use("default")

    with st.expander("Interpretation cues", expanded=False):
        st.markdown(
            """
            - **Population** — larger demand pools lift the score.  
            - **Quality gap** — weaker incumbents imply room for a stronger offer.  
            - **Competition counts** — saturation in the chosen vertical pushes the score down.  
            """
        )

st.toast("Dashboard ready")
