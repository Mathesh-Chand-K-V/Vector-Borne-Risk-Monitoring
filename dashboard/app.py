import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
MAX_MAP_POINTS = 1500
FAST_TABLE_LIMIT = 100


@st.cache_data(show_spinner=True)
def load_dashboard_inputs():
    records = pd.read_csv(DATA_DIR / "dashboard_records.csv", low_memory=False)
    metadata = json.loads((DATA_DIR / "dashboard_metadata.json").read_text(encoding="utf-8"))
    return records, metadata


@st.cache_data(show_spinner=False)
def prepare_dashboard_records(records):
    records = records.copy()
    for column in ["actual_acquisition_date", "requested_date"]:
        if column in records.columns:
            records[column] = pd.to_datetime(records[column], dayfirst=True, errors="coerce")

    if "final_eri_score" in records.columns:
        records["score_0_100"] = (records["final_eri_score"] * 100).clip(lower=0, upper=100)
    else:
        records["score_0_100"] = np.nan

    if "risk_band" not in records.columns:
        records["risk_band"] = np.select(
            [records["score_0_100"] <= 33.33, records["score_0_100"] <= 66.66],
            ["Low", "Medium"],
            default="High",
        )

    records["suggested_action"] = records["risk_band"].map(
        {
            "Low": "Routine environmental monitoring.",
            "Medium": "Schedule targeted environmental inspection.",
            "High": "Prioritize preventive inspection of water-holding and drainage-prone areas.",
        }
    )

    required_columns = [
        "sample_id",
        "split",
        "latitude",
        "longitude",
        "actual_acquisition_date",
        "requested_date",
        "ERI_target",
        "rf1_prediction",
        "final_eri_score",
        "score_0_100",
        "risk_band",
        "suggested_action",
    ]
    existing = [column for column in required_columns if column in records.columns]
    return records[existing].copy()


@st.cache_data(show_spinner=False)
def build_filtered_dataset(records, split_option, risk_option, score_start, score_end, date_start, date_end):
    filtered = records.copy()
    if split_option != "All":
        filtered = filtered[filtered["split"] == split_option]
    if risk_option != "All":
        filtered = filtered[filtered["risk_band"] == risk_option]
    if score_start is not None and score_end is not None:
        filtered = filtered[
            (filtered["score_0_100"] >= score_start) & (filtered["score_0_100"] <= score_end)
        ]
    if date_start is not None and date_end is not None:
        filtered = filtered[
            filtered["actual_acquisition_date"].between(
                pd.Timestamp(date_start), pd.Timestamp(date_end), inclusive="both"
            )
        ]
    filtered = filtered.dropna(subset=["latitude", "longitude"]).copy()
    return filtered


st.set_page_config(
    page_title="VectorSafe AI – Environmental Mosquito Suitability Dashboard",
    page_icon="🦟",
    layout="wide",
)

records, metadata = load_dashboard_inputs()
records = prepare_dashboard_records(records)

st.title("VectorSafe AI – Environmental Mosquito Suitability Dashboard")
st.caption("Bhopal-area satellite and environmental suitability analysis")

st.warning(
    "This dashboard displays a formula-derived Environmental Mosquito Suitability Score (EMSS). "
    "It is intended for preventive environmental inspection prioritization only. It is not a dengue diagnosis, "
    "confirmed outbreak prediction, confirmed mosquito-presence result, or confirmed breeding-ground detection."
)

sidebar = st.sidebar
sidebar.header("Filters")
fast_mode = st.sidebar.checkbox("Fast Exhibition Mode", value=True)

split_option = sidebar.selectbox("Split", ["All", "Validation", "Test"])
risk_option = sidebar.selectbox("Risk band", ["All", "Low", "Medium", "High"])
score_threshold = sidebar.slider("EMSS score threshold", 0.0, 100.0, (0.0, 100.0), step=0.1)

if records["actual_acquisition_date"].notna().any():
    filter_date = sidebar.checkbox("Filter by acquisition date")
    if filter_date:
        valid_dates = records["actual_acquisition_date"].dropna()
        min_date = valid_dates.min().date()
        max_date = valid_dates.max().date()
        start_date, end_date = sidebar.date_input(
            "Acquisition date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
    else:
        start_date = None
        end_date = None
else:
    filter_date = False
    start_date = None
    end_date = None

filtered = build_filtered_dataset(
    records,
    split_option=split_option,
    risk_option=risk_option,
    score_start=score_threshold[0],
    score_end=score_threshold[1],
    date_start=start_date,
    date_end=end_date,
)

if filtered.empty:
    st.info("No records match the selected filters.")
    st.stop()

selected_split = split_option if split_option != "All" else "All splits"
selected_date_text = f"{start_date} to {end_date}" if filter_date and start_date is not None and end_date is not None else "All dates"

col_total, col_avg, col_high, col_split = st.columns(4)
col_total.metric("Total filtered samples", f"{len(filtered):,}")
col_avg.metric("Average final EMSS score", f"{filtered['score_0_100'].mean():.2f}")
col_high.metric("High suitability samples", int((filtered["risk_band"] == "High").sum()))
col_split.metric("Selected split / date range", f"{selected_split} | {selected_date_text}")

map_df = filtered[["sample_id", "split", "actual_acquisition_date", "latitude", "longitude", "final_eri_score", "risk_band", "suggested_action"]].copy()
map_df = map_df.dropna(subset=["latitude", "longitude"]).head(MAX_MAP_POINTS)

st.subheader("Spatial view")
with st.spinner("Preparing map..."):
    low_risk_layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df[map_df["risk_band"] == "Low"],
        get_position="[longitude, latitude]",
        get_radius=120,
        get_fill_color="[34, 197, 94, 180]",
        get_line_color="[255, 255, 255, 180]",
        line_width_min_pixels=0.5,
        pickable=True,
        opacity=0.9,
    )
    medium_risk_layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df[map_df["risk_band"] == "Medium"],
        get_position="[longitude, latitude]",
        get_radius=120,
        get_fill_color="[255, 165, 0, 180]",
        get_line_color="[255, 255, 255, 180]",
        line_width_min_pixels=0.5,
        pickable=True,
        opacity=0.9,
    )
    high_risk_layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df[map_df["risk_band"] == "High"],
        get_position="[longitude, latitude]",
        get_radius=120,
        get_fill_color="[255, 0, 0, 180]",
        get_line_color="[255, 255, 255, 180]",
        line_width_min_pixels=0.5,
        pickable=True,
        opacity=0.9,
    )
    view_state = pdk.ViewState(latitude=23.2599, longitude=77.4126, zoom=9.5, pitch=0)
    tooltip = {
        "html": (
            "<b>sample_id</b>: {sample_id}<br/>"
            "<b>split</b>: {split}<br/>"
            "<b>final_eri_score</b>: {final_eri_score}<br/>"
            "<b>risk band</b>: {risk_band}<br/>"
            "<b>acquisition date</b>: {actual_acquisition_date}<br/>"
            "<b>suggested action</b>: {suggested_action}"
        ),
        "style": {"color": "black"},
    }
    st.pydeck_chart(
        pdk.Deck(
            map_style="light",
            initial_view_state=view_state,
            layers=[low_risk_layer, medium_risk_layer, high_risk_layer],
            tooltip=tooltip,
            description="Environmental suitability map",
        ),
        width="stretch",
    )

legend_html = """
<div style='padding:8px 10px; background:#ffffff; border:1px solid #e5e7eb; border-radius:6px; font-size:12px; margin-top:8px;'>
  <div style='display:flex; align-items:center; gap:8px; margin-bottom:4px;'><span style='display:inline-block; width:12px; height:12px; background:#22c55e; border-radius:50%;'></span>Low</div>
  <div style='display:flex; align-items:center; gap:8px; margin-bottom:4px;'><span style='display:inline-block; width:12px; height:12px; background:#f59e0b; border-radius:50%;'></span>Medium</div>
  <div style='display:flex; align-items:center; gap:8px;'><span style='display:inline-block; width:12px; height:12px; background:#ef4444; border-radius:50%;'></span>High</div>
</div>
"""
st.markdown(legend_html, unsafe_allow_html=True)

risk_counts = filtered["risk_band"].value_counts().reindex(["Low", "Medium", "High"], fill_value=0).reset_index()
risk_counts.columns = ["risk_band", "count"]
bar_chart = px.bar(
    risk_counts,
    x="risk_band",
    y="count",
    color="risk_band",
    color_discrete_map={"Low": "#22c55e", "Medium": "#f59e0b", "High": "#ef4444"},
    title="Risk-band count",
    category_orders={"risk_band": ["Low", "Medium", "High"]},
)

hist_chart = px.histogram(
    filtered,
    x="score_0_100",
    nbins=30,
    title="Final EMSS distribution",
    color_discrete_sequence=["#3b82f6"],
)

chart_col_1, chart_col_2 = st.columns(2)
with chart_col_1:
    st.plotly_chart(bar_chart, width="stretch")
with chart_col_2:
    st.plotly_chart(hist_chart, width="stretch")

if not fast_mode and len(filtered) < 1500 and "rf1_prediction" in filtered.columns:
    with st.expander("RF1 vs RF2 comparison"):
        scatter_df = filtered[["rf1_prediction", "final_eri_score", "risk_band"]].copy()
        scatter_df["rf1_prediction_0_100"] = scatter_df["rf1_prediction"] * 100
        scatter_df["final_eri_score_0_100"] = scatter_df["final_eri_score"] * 100
        scatter_plot = px.scatter(
            scatter_df,
            x="rf1_prediction_0_100",
            y="final_eri_score_0_100",
            color="risk_band",
            color_discrete_map={"Low": "#22c55e", "Medium": "#f59e0b", "High": "#ef4444"},
            title="RF1 versus RF2 suitability comparison",
            labels={"rf1_prediction_0_100": "RF1 prediction (0-100)", "final_eri_score_0_100": "RF2 final EMSS (0-100)"},
        )
        st.plotly_chart(scatter_plot, width="stretch")

output_columns = [
    "sample_id",
    "split",
    "latitude",
    "longitude",
    "actual_acquisition_date",
    "final_eri_score",
    "risk_band",
    "suggested_action",
]
if "rf1_prediction" in filtered.columns:
    output_columns.insert(5, "rf1_prediction")

table_df = filtered[output_columns].copy()
if "actual_acquisition_date" in table_df.columns:
    table_df["actual_acquisition_date"] = table_df["actual_acquisition_date"].dt.strftime("%Y-%m-%d")
if "rf1_prediction" in table_df.columns:
    table_df["rf1_prediction"] = table_df["rf1_prediction"].map(lambda x: round(float(x), 6) if pd.notna(x) else x)

table_df["final_eri_score"] = table_df["final_eri_score"].map(lambda x: round(float(x), 6) if pd.notna(x) else x)

if fast_mode:
    table_df = table_df.head(FAST_TABLE_LIMIT)

st.subheader("Filtered records")
st.dataframe(table_df, width="stretch", hide_index=True)

with st.expander("Methodology"):
    st.markdown(
        "- Bhopal environmental variables feed RF1.\n"
        "- Bhopal satellite images feed ResNet-18.\n"
        "- ResNet-18 generates 512 image features.\n"
        "- RF2 fuses RF1 prediction and 512 ResNet features.\n"
        "- sample_id alignment passed for:\n"
        "  - Train: 3321\n"
        "  - Validation: 712\n"
        "  - Test: 712\n"
        "- Only Validation and Test outputs are shown in this dashboard."
    )

with st.expander("Data Sources and Limitations"):
    st.markdown(
        "- Bhopal, India, 50 km collection radius.\n"
        "- Date range: 2024-01-01 to 2024-06-30.\n"
        "- Copernicus Sentinel-2 L2A through Microsoft Planetary Computer.\n"
        "- Open-Meteo Historical Weather API, using ECMWF ERA5 / ERA5-Land where applicable.\n"
        "- Formula-derived EMSS limitation.\n"
        "- Risk-band visualization limitation.\n"
        "- Final licensing, attribution wording, and provider terms must be reviewed before public deployment."
    )

st.caption(
    "This dashboard uses already generated Bhopal validation and test prediction outputs. "
    "It does not train models, generate targets, call external providers, or modify original project files."
)
