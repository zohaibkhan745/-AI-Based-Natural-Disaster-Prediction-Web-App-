"""
AI-Based Natural Disaster Prediction Web App
Streamlit application for real-time flood prediction in KP, Pakistan

Features:
- Real-time weather data integration
- Flood risk prediction using trained ML models
- Interactive maps and visualizations
- Historical data analysis
- Alert system for high-risk predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime, timedelta
from pathlib import Path
import os

# Color palette for consistent UI accents
PRIMARY_COLOR = "#1f77b4"
ACCENT_COLOR = "#0f4c81"
DANGER_COLOR = "#e74c3c"
WARNING_COLOR = "#f39c12"
SUCCESS_COLOR = "#2ecc71"
CARD_BG = "#f7f9fb"

# Try to import plotly, provide fallback
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Flood Prediction System - KP Pakistan",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define paths
PROJECT_ROOT = Path(__file__).parent
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = RESULTS_DIR

# Location configurations
LOCATIONS = {
    "swat": {
        "name": "Swat District, KP",
        "latitude": 34.8091,
        "longitude": 72.3617,
        "elevation": 980,
        "location_id": 0
    },
    "upper_dir": {
        "name": "Upper Dir District, KP", 
        "latitude": 35.3350,
        "longitude": 71.8760,
        "elevation": 1420,
        "location_id": 1
    }
}

# OpenWeatherMap API
try:
    OPENWEATHER_API_KEY = st.secrets.get("OPENWEATHER_API_KEY", "demo")
except:
    OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "demo")

# Networking settings for API calls (always defined)
WEATHER_TIMEOUT = int(os.environ.get("WEATHER_TIMEOUT", "15"))  # seconds
RETRY_STRATEGY = Retry(
    total=3,
    connect=3,
    read=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
)


def get_http_session():
    """Create a requests session with retries for more resilient API calls"""
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=RETRY_STRATEGY)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


HTTP_SESSION = get_http_session()


@st.cache_resource
def load_model():
    """Load the trained flood prediction model"""
    model_paths = [
        MODELS_DIR / "best_flood_model.pkl",
        MODELS_DIR / "random_forest_model.pkl",
        MODELS_DIR / "logistic_regression_model.pkl"
    ]
    
    for path in model_paths:
        if path.exists():
            try:
                with open(path, 'rb') as f:
                    model_data = pickle.load(f)
                return model_data
            except Exception as e:
                st.warning(f"Could not load {path.name}: {e}")
    
    return None


def get_model_recall():
    """Retrieve model recall from metrics file if available"""
    metrics_path = RESULTS_DIR / "improved_model_metrics.csv"
    if metrics_path.exists():
        try:
            df = pd.read_csv(metrics_path)
            # prefer recall if present, fallback to sensitivity/recall_score column names
            for col in ["recall", "recall_score", "sensitivity"]:
                if col in df.columns:
                    val = df[col].iloc[0]
                    return float(val)
        except Exception:
            return None
    return None


def get_data_last_updated():
    """Return the latest timestamp in the processed dataset"""
    data_path = DATA_DIR / "flood_weather_dataset.csv"
    if data_path.exists():
        try:
            df = pd.read_csv(data_path, usecols=["date"], low_memory=False)
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            latest = df["date"].max()
            if pd.notna(latest):
                return latest
        except Exception:
            return None
    return None


def get_precip_trend(location_key: str, days: int = 7):
    """Return recent precipitation trend for the selected location"""
    data_path = DATA_DIR / "flood_weather_dataset.csv"
    if not data_path.exists():
        return None
    try:
        df = pd.read_csv(data_path, usecols=["date", "prcp", "location_key"], low_memory=False)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        loc_df = df[df["location_key"].str.contains(location_key, case=False, na=False)]
        loc_df = loc_df.sort_values("date").tail(days)
        return loc_df if not loc_df.empty else None
    except Exception:
        return None


def risk_badge(probability: float):
    """Map probability to a label, color, and message"""
    if probability >= 0.7:
        return "HIGH", DANGER_COLOR, "Take immediate precautions"
    if probability >= 0.4:
        return "MODERATE", WARNING_COLOR, "Stay alert and monitor"
    if probability >= 0.2:
        return "LOW", "#f1c40f", "Normal but watch forecasts"
    return "VERY LOW", SUCCESS_COLOR, "Normal conditions"


@st.cache_data(ttl=1800)
def fetch_weather_data(lat, lon, api_key):
    """Fetch current weather from OpenWeatherMap API"""
    if api_key == "demo":
        # Return demo data for testing
        return generate_demo_weather()
    
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather"
        params = {
            "lat": lat,
            "lon": lon,
            "appid": api_key,
            "units": "metric"
        }
        response = HTTP_SESSION.get(url, params=params, timeout=WEATHER_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        return {
            "tavg": data["main"]["temp"],
            "tmin": data["main"]["temp_min"],
            "tmax": data["main"]["temp_max"],
            "humidity": data["main"]["humidity"],
            "pres": data["main"]["pressure"],
            "wspd": data["wind"]["speed"] * 3.6,  # m/s to km/h
            "prcp": data.get("rain", {}).get("1h", 0),
            "description": data["weather"][0]["description"],
            "icon": data["weather"][0]["icon"]
        }
    except Exception as e:
        st.warning(f"API error: {e}. Using demo data.")
    
    return generate_demo_weather()


@st.cache_data(ttl=3600)
def fetch_weather_forecast(lat, lon, api_key, target_date):
    """Fetch weather forecast for a specific future date from OpenWeatherMap API"""
    if api_key == "demo":
        return generate_demo_weather_for_date(target_date), "Demo Mode (No API Key)"
    
    try:
        # OpenWeatherMap 5-day/3-hour forecast API
        url = "https://api.openweathermap.org/data/2.5/forecast"
        params = {
            "lat": lat,
            "lon": lon,
            "appid": api_key,
            "units": "metric"
        }
        response = HTTP_SESSION.get(url, params=params, timeout=WEATHER_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        forecasts = data.get("list", [])
        
        # Find forecasts for the target date
        target_str = target_date.strftime("%Y-%m-%d")
        day_forecasts = [f for f in forecasts if target_str in f["dt_txt"]]
        
        if day_forecasts:
            # Aggregate data from all forecasts for that day
            temps = [f["main"]["temp"] for f in day_forecasts]
            temp_mins = [f["main"]["temp_min"] for f in day_forecasts]
            temp_maxs = [f["main"]["temp_max"] for f in day_forecasts]
            humidities = [f["main"]["humidity"] for f in day_forecasts]
            pressures = [f["main"]["pressure"] for f in day_forecasts]
            wind_speeds = [f["wind"]["speed"] * 3.6 for f in day_forecasts]  # m/s to km/h
            
            # Sum precipitation (rain in last 3h)
            total_prcp = sum(f.get("rain", {}).get("3h", 0) for f in day_forecasts)
            
            # Get the most common weather description
            descriptions = [f["weather"][0]["description"] for f in day_forecasts]
            most_common_desc = max(set(descriptions), key=descriptions.count) if descriptions else ""
            
            return {
                "tavg": np.mean(temps),
                "tmin": min(temp_mins),
                "tmax": max(temp_maxs),
                "humidity": np.mean(humidities),
                "pres": np.mean(pressures),
                "wspd": np.mean(wind_speeds),
                "prcp": total_prcp,
                "description": most_common_desc,
                "icon": day_forecasts[0]["weather"][0]["icon"],
                "forecast_count": len(day_forecasts)
            }, f"OpenWeatherMap Forecast ({len(day_forecasts)} readings)"
        else:
            # Date is beyond 5-day forecast range
            return None, "Date beyond 5-day forecast range"
            
    except Exception as e:
        return generate_demo_weather_for_date(target_date), f"API Error: {str(e)[:50]} - Using Demo"


def generate_demo_weather_for_date(target_date):
    """Generate realistic demo weather data for a specific date"""
    month = target_date.month
    
    # Seasonal variations for KP Pakistan
    if month in [6, 7, 8]:  # Monsoon - HIGH RISK
        temp_base, prcp_base, humidity_base = 28, 25, 75
    elif month in [9]:  # Post monsoon
        temp_base, prcp_base, humidity_base = 24, 15, 65
    elif month in [12, 1, 2]:  # Winter
        temp_base, prcp_base, humidity_base = 5, 5, 50
    elif month in [3, 4, 5]:  # Spring
        temp_base, prcp_base, humidity_base = 18, 8, 55
    else:  # Fall
        temp_base, prcp_base, humidity_base = 20, 10, 60
    
    # Add some randomness
    np.random.seed(target_date.toordinal())  # Consistent for same date
    
    return {
        "tavg": temp_base + np.random.uniform(-3, 3),
        "tmin": temp_base - 5 + np.random.uniform(-2, 2),
        "tmax": temp_base + 5 + np.random.uniform(-2, 2),
        "humidity": humidity_base + np.random.uniform(-10, 15),
        "pres": 1010 + np.random.uniform(-15, 15),
        "wspd": 10 + np.random.uniform(-5, 15),
        "prcp": prcp_base + np.random.uniform(0, 20),
        "description": "Demo forecast - simulated data",
        "icon": "03d"
    }


def generate_demo_weather():
    """Generate realistic demo weather data"""
    month = datetime.now().month
    # Seasonal variations for KP Pakistan
    if month in [6, 7, 8]:  # Monsoon
        temp_base, prcp_base = 28, 25
    elif month in [12, 1, 2]:  # Winter
        temp_base, prcp_base = 5, 5
    else:
        temp_base, prcp_base = 18, 10
    
    return {
        "tavg": temp_base + np.random.uniform(-3, 3),
        "tmin": temp_base - 5 + np.random.uniform(-2, 2),
        "tmax": temp_base + 5 + np.random.uniform(-2, 2),
        "humidity": 60 + np.random.uniform(-20, 30),
        "pres": 1010 + np.random.uniform(-15, 15),
        "wspd": 10 + np.random.uniform(-5, 15),
        "prcp": prcp_base + np.random.uniform(0, 20),
        "description": "Demo mode - scattered clouds",
        "icon": "03d"
    }


def prepare_features(weather_data, location_id):
    """Prepare features for model prediction"""
    now = datetime.now()
    prcp = weather_data.get('prcp', 0)
    humidity = weather_data.get('humidity', 60)
    
    features = {
        'tavg': weather_data.get('tavg', 20),
        'tmin': weather_data.get('tmin', 15),
        'tmax': weather_data.get('tmax', 25),
        'prcp': prcp,
        'wspd': weather_data.get('wspd', 10),
        'wpgt': weather_data.get('wspd', 10) * 1.5,  # Estimated gust
        'pres': weather_data.get('pres', 1010),
        'humidity': humidity,
        'solar_radiation': 15 + np.random.uniform(-5, 10),
        'month': now.month,
        'day_of_year': now.timetuple().tm_yday,
        'quarter': (now.month - 1) // 3 + 1,
        'is_monsoon': 1 if now.month in [6, 7, 8, 9] else 0,
        'temp_range': weather_data.get('tmax', 25) - weather_data.get('tmin', 15),
        'high_humidity': 1 if humidity > 70 else 0,
        'pressure_anomaly': weather_data.get('pres', 1010) - 1013,
        'prcp_7day_avg': prcp * 0.8,
        'prcp_3day_sum': prcp * 2.5,
        'prcp_7day_sum': prcp * 5,
        'heavy_rain': 1 if prcp > 10 else 0,
        'extreme_rain': 1 if prcp > 50 else 0,
        'tavg_7day_avg': weather_data.get('tavg', 20),
        'wspd_7day_avg': weather_data.get('wspd', 10),
        'location_encoded': location_id
    }
    
    return pd.DataFrame([features])


def predict_flood_risk(model_data, features):
    """Make flood prediction using trained model"""
    if model_data is None:
        return 0.1, "No model available"
    
    try:
        # Handle both dict format and direct model object
        if isinstance(model_data, dict):
            model = model_data.get('model', model_data)
            threshold = model_data.get('threshold', 0.5)
        else:
            # model_data is the model itself
            model = model_data
            threshold = 0.5
        
        # Get prediction probability
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features)[0][1]
        else:
            proba = float(model.predict(features)[0])
        
        # Determine risk level
        if proba >= 0.7:
            risk_level = "HIGH"
        elif proba >= 0.4:
            risk_level = "MODERATE"
        elif proba >= 0.2:
            risk_level = "LOW"
        else:
            risk_level = "VERY LOW"
        
        return proba, risk_level
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return 0.1, "Error"


def create_gauge_chart(probability, risk_level):
    """Create a gauge chart for flood risk"""
    if not PLOTLY_AVAILABLE:
        return None
    
    colors = {
        "VERY LOW": "#2ecc71",
        "LOW": "#f39c12",
        "MODERATE": "#e67e22",
        "HIGH": "#e74c3c"
    }
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Flood Risk: {risk_level}", 'font': {'size': 24}},
        delta={'reference': 30, 'increasing': {'color': "red"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': colors.get(risk_level, "#3498db")},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 20], 'color': '#2ecc71'},
                {'range': [20, 40], 'color': '#f39c12'},
                {'range': [40, 70], 'color': '#e67e22'},
                {'range': [70, 100], 'color': '#e74c3c'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': probability * 100
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig


def main():
    """Main application function orchestrating the dashboard navigation"""

    st.sidebar.title("🌊 Flood Safety Dashboard")
    st.sidebar.markdown("Manage predictions, analysis, and AI labs from the left sidebar.")
    st.sidebar.markdown("---")

    selected_location_key = st.sidebar.selectbox(
        "Location",
        options=list(LOCATIONS.keys()),
        format_func=lambda x: LOCATIONS[x]["name"],
    )
    location = LOCATIONS[selected_location_key]

    st.sidebar.markdown("### Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "🏠 Dashboard (Home)",
            "⚡ Live Flood Prediction",
            "📊 Historical Analysis",
            "🛣️ Evacuation Route Planning",
            "🧩 Resource Allocation (CSP)",
            "🔬 Model Explainability",
            "ℹ️ About Project",
        ],
    )

    st.sidebar.markdown("### AI Lab (advanced)")
    ai_page = st.sidebar.selectbox(
        "Optional modules",
        ["None", "🧬 Neural Network (LSTM)", "📈 K-Means Clustering", "🎮 Reinforcement Learning"],
    )

    model_data = load_model()

    # Prefer AI lab selection when chosen, otherwise follow main navigation
    if ai_page != "None":
        if ai_page == "🧬 Neural Network (LSTM)":
            show_neural_network_page()
        elif ai_page == "📈 K-Means Clustering":
            show_clustering_page()
        elif ai_page == "🎮 Reinforcement Learning":
            show_reinforcement_learning_page()
        return

    if page == "🏠 Dashboard (Home)":
        show_dashboard(location, model_data)
    elif page == "⚡ Live Flood Prediction":
        show_custom_prediction(location, model_data)
    elif page == "📊 Historical Analysis":
        show_historical_data(location)
    elif page == "🛣️ Evacuation Route Planning":
        show_search_algorithms()
    elif page == "🧩 Resource Allocation (CSP)":
        show_csp_page()
    elif page == "🔬 Model Explainability":
        show_explainability_page(model_data)
    else:
        show_about()


def show_dashboard(location, model_data):
    """Display the upgraded home dashboard with KPIs, map, and trends"""

    st.title(f"🌊 Flood Risk Dashboard — {location['name']}")

    weather = fetch_weather_data(location['latitude'], location['longitude'], OPENWEATHER_API_KEY)
    features = prepare_features(weather, location['location_id'])
    probability, risk_level = predict_flood_risk(model_data, features)
    risk_label, risk_color, risk_msg = risk_badge(probability)
    recall = get_model_recall()
    last_updated = get_data_last_updated()

    # KPI strip
    kpi1, kpi2, kpi3 = st.columns(3)
    with kpi1:
        st.markdown(f"<div style='background:{CARD_BG};padding:14px;border-radius:12px;border-left:6px solid {risk_color};'>"
                    f"<h4 style='margin:0;'>Current Flood Risk</h4>"
                    f"<p style='font-size:28px;margin:4px 0;color:{risk_color};font-weight:700;'>{risk_label}</p>"
                    f"<small>{risk_msg}</small></div>", unsafe_allow_html=True)
    with kpi2:
        recall_txt = f"{recall*100:.1f}%" if recall is not None else "—"
        st.markdown(f"<div style='background:{CARD_BG};padding:14px;border-radius:12px;border-left:6px solid {PRIMARY_COLOR};'>"
                    "<h4 style='margin:0;'>Model Recall</h4>"
                    f"<p style='font-size:28px;margin:4px 0;color:{PRIMARY_COLOR};font-weight:700;'>{recall_txt}</p>"
                    "<small>Prioritizing safety over precision</small></div>", unsafe_allow_html=True)
    with kpi3:
        ts = last_updated.strftime("%Y-%m-%d") if last_updated else "Not available"
        st.markdown(f"<div style='background:{CARD_BG};padding:14px;border-radius:12px;border-left:6px solid {ACCENT_COLOR};'>"
                    "<h4 style='margin:0;'>Data Last Updated</h4>"
                    f"<p style='font-size:24px;margin:4px 0;color:{ACCENT_COLOR};font-weight:700;'>{ts}</p>"
                    "<small>Latest processed observation</small></div>", unsafe_allow_html=True)

    st.markdown("---")

    # Map + risk gauge row
    map_col, gauge_col = st.columns([1.2, 1])
    with map_col:
        st.subheader("🗺️ KP Regions Overview")
        map_df = pd.DataFrame([
            {"lat": info["latitude"], "lon": info["longitude"], "name": info["name"]}
            for info in LOCATIONS.values()
        ])
        st.map(map_df)
    with gauge_col:
        st.subheader("🎯 Current Flood Probability")
        gauge = create_gauge_chart(probability, risk_label)
        if gauge:
            st.plotly_chart(gauge, use_container_width=True)
        else:
            st.metric("Flood Risk", f"{probability*100:.1f}%", risk_label)
        st.caption(f"Live weather source: {'API' if OPENWEATHER_API_KEY != 'demo' else 'Demo mode'}")

    # Weather snapshot + trend
    st.markdown("---")
    snap_col, trend_col = st.columns([1, 1])
    with snap_col:
        st.subheader("📡 Live Weather Snapshot")
        c1, c2, c3 = st.columns(3)
        c1.metric("Temp (°C)", f"{weather['tavg']:.1f}", help="Average temperature")
        c2.metric("Humidity (%)", f"{weather['humidity']:.0f}")
        c3.metric("Precip (mm)", f"{weather['prcp']:.1f}")
        c1.metric("Wind (km/h)", f"{weather['wspd']:.1f}")
        c2.metric("Pressure (hPa)", f"{weather['pres']:.0f}")
        c3.metric("Temp Range", f"{weather['tmax']-weather['tmin']:.1f}")

    with trend_col:
        st.subheader("📈 7-Day Precipitation Trend")
        loc_key = "swat" if "swat" in location["name"].lower() else "upper_dir"
        trend_df = get_precip_trend(loc_key, days=7)
        if trend_df is not None and PLOTLY_AVAILABLE:
            fig = px.line(trend_df, x="date", y="prcp", markers=True, title="Recent Precipitation")
            fig.update_layout(showlegend=False, height=260)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No recent precipitation data available.")

    # Risk status card
    st.markdown("---")
    st.subheader("✅ Risk Status")
    st.markdown(
        f"<div style='background:{risk_color}1A;border:1px solid {risk_color};padding:16px;border-radius:12px;'>"
        f"<h3 style='margin:0;color:{risk_color};'>Current Status: {risk_label}</h3>"
        f"<p style='margin:6px 0;color:#1e1e1e;'>{risk_msg}</p>"
        "<ul style='margin:0 0 0 18px;'>"
        "<li>Monitor official NDMA/PMD alerts</li>"
        "<li>Keep evacuation routes handy</li>"
        "<li>Prepare emergency supplies during monsoon</li>"
        "</ul></div>",
        unsafe_allow_html=True,
    )

    if OPENWEATHER_API_KEY == "demo":
        st.info("🔧 Running in demo mode. Set OPENWEATHER_API_KEY for live weather data.")


def show_historical_data(location):
    """Show historical flood data analysis"""
    st.title("📊 Historical Analysis")

    data_path = DATA_DIR / "flood_weather_dataset.csv"
    if not data_path.exists():
        st.warning("Historical data not found. Please run the data pipeline first.")
        return

    df = pd.read_csv(data_path, low_memory=False)
    df['date'] = pd.to_datetime(df['date'])
    loc_key = "swat" if "swat" in location['name'].lower() else "upper_dir"
    loc_df = df[df['location_key'] == loc_key].copy()

    if loc_df.empty:
        st.info("No historical records for this location.")
        return

    st.subheader(f"Data for {location['name']}")
    st.write(f"📅 Date Range: {loc_df['date'].min().date()} to {loc_df['date'].max().date()}")
    st.write(f"📊 Total Records: {len(loc_df)}")

    # Date range selector
    st.markdown("### Date Range")
    min_date, max_date = loc_df['date'].min().date(), loc_df['date'].max().date()
    start_date, end_date = st.date_input(
        "Select range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    filtered = loc_df[(loc_df['date'].dt.date >= start_date) & (loc_df['date'].dt.date <= end_date)]

    flood_count = filtered['flood_event'].sum()
    percent = (flood_count/len(filtered))*100 if len(filtered) else 0
    st.write(f"🌊 Flood Events: {flood_count} ({percent:.2f}% of selected range)")

    if PLOTLY_AVAILABLE and len(filtered) > 0:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.line(filtered, x='date', y='prcp', title='Precipitation Trend (mm)')
            fig.update_layout(xaxis_title="Date", yaxis_title="Precipitation (mm)")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = px.line(filtered, x='date', y=['tmin', 'tavg', 'tmax'],
                          title='Temperature Trend (°C)',
                          labels={'value': 'Temperature (°C)', 'variable': 'Type'})
            st.plotly_chart(fig2, use_container_width=True)

        flood_df = filtered[filtered['flood_event'] == 1]
        fig3 = px.scatter(flood_df, x='date', y='prcp',
                          title='Flood Events (precip on flood days)',
                          color_discrete_sequence=['red']) if len(flood_df) > 0 else None
        if fig3:
            fig3.update_traces(marker=dict(size=10))
            st.plotly_chart(fig3, use_container_width=True)

    # Cluster visualization (K-Means risk groups)
    st.markdown("### 🧭 Cluster Visualization (Risk Groups)")
    try:
        from code.clustering import FloodPatternAnalyzer
        sample_cols = ['tavg', 'prcp', 'humidity', 'pres', 'wspd']
        cluster_df = filtered[sample_cols].dropna().sample(min(400, len(filtered))) if len(filtered) > 0 else pd.DataFrame()
        if not cluster_df.empty:
            analyzer = FloodPatternAnalyzer(n_clusters=4)
            analyzer.fit(cluster_df.values, sample_cols)
            labels = analyzer.predict(cluster_df.values)
            cluster_df['cluster'] = labels
            if PLOTLY_AVAILABLE:
                fig = px.scatter(cluster_df, x='prcp', y='humidity', color='cluster',
                                 title='Risk Clusters by Precipitation vs Humidity')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data to render clusters.")
    except Exception as e:
        st.info(f"Cluster view unavailable: {e}")

    with st.expander("View Raw Data (First 100 rows)"):
        st.dataframe(filtered.head(100))


def show_model_info(model_data):
    """Display model information and metrics"""
    st.title("🤖 Model Information")
    
    # Load metrics
    metrics_path = RESULTS_DIR / "improved_model_metrics.csv"
    
    if metrics_path.exists():
        metrics_df = pd.read_csv(metrics_path)
        
        st.subheader("📊 Model Performance Metrics")
        st.dataframe(metrics_df)
        
        # Best model info
        if model_data:
            st.subheader("🏆 Currently Loaded Model")
            if isinstance(model_data, dict):
                st.write(f"**Model:** {model_data.get('model_name', 'Unknown')}")
                st.write(f"**Threshold:** {model_data.get('threshold', 0.5):.4f}")
                
                if 'metrics' in model_data:
                    st.json(model_data['metrics'])
    else:
        st.warning("Metrics file not found. Please train the models first.")
    
    # Feature importance
    st.subheader("📈 Feature Importance")
    fi_path = RESULTS_DIR / "feature_importance.json"
    if fi_path.exists():
        with open(fi_path) as f:
            importance = json.load(f)
        st.json(importance)


def show_custom_prediction(location, model_data):
    """Live flood prediction page with manual or auto-fetched weather"""
    st.title("⚡ Live Flood Prediction")
    st.caption("Fetch live weather for KP regions or enter custom values to estimate flood probability.")
    st.markdown("---")
    
    # Get today and tomorrow dates
    today = datetime.now().date()
    tomorrow = today + timedelta(days=1)
    
    # Quick date selection buttons
    st.markdown("### ⚡ Quick Date Selection")
    col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
    
    with col_btn1:
        if st.button("📅 Today", use_container_width=True):
            st.session_state['selected_date'] = today
    with col_btn2:
        if st.button("🔮 Tomorrow", use_container_width=True, type="primary"):
            st.session_state['selected_date'] = tomorrow
    with col_btn3:
        if st.button("📆 Day After", use_container_width=True):
            st.session_state['selected_date'] = today + timedelta(days=2)
    with col_btn4:
        if st.button("📅 In 3 Days", use_container_width=True):
            st.session_state['selected_date'] = today + timedelta(days=3)
    
    st.markdown("---")
    
    # Date and location selection
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        # Use session state if set, otherwise default to today
        default_date = st.session_state.get('selected_date', today)
        selected_date = st.date_input(
            "📅 Select Date",
            value=default_date,
            min_value=datetime(2000, 1, 1),
            max_value=datetime(2030, 12, 31)
        )
    with col2:
        selected_loc = st.selectbox(
            "📍 Location",
            options=["Swat District", "Upper Dir District"],
            index=0 if "swat" in location['name'].lower() else 1
        )
    with col3:
        auto_fetch = st.checkbox("Auto-fetch", value=True, help="Automatically load weather data")

    fetch_now = False
    if st.button("📡 Fetch Live Weather", type="primary", use_container_width=True):
        fetch_now = True
        selected_date = today
    
    # Determine if date is future or past
    is_future_date = selected_date > today
    is_within_forecast = selected_date <= today + timedelta(days=5)
    
    # Get location coordinates
    loc_key = "swat" if "swat" in selected_loc.lower() else "upper_dir"
    loc_info = LOCATIONS[loc_key]
    
    # Default values
    tavg, tmin, tmax, prcp, humidity, pres, wspd, solar = 20.0, 15.0, 25.0, 5.0, 60.0, 1010.0, 10.0, 18.0
    weather_source = "Manual Input"
    actual_flood = None
    forecast_info = None
    
    # Auto-fetch weather data
    if auto_fetch or fetch_now:
        if is_future_date and is_within_forecast:
            # FUTURE DATE: Fetch weather forecast from API
            st.info(f"🔮 **Future Date Selected** - Fetching weather forecast for {selected_date}...")
            
            forecast_data, source_msg = fetch_weather_forecast(
                loc_info['latitude'],
                loc_info['longitude'],
                OPENWEATHER_API_KEY,
                datetime.combine(selected_date, datetime.min.time())
            )
            
            if forecast_data:
                tavg = float(forecast_data.get('tavg', tavg))
                tmin = float(forecast_data.get('tmin', tmin))
                tmax = float(forecast_data.get('tmax', tmax))
                prcp = float(forecast_data.get('prcp', prcp))
                humidity = float(forecast_data.get('humidity', humidity))
                pres = float(forecast_data.get('pres', pres))
                wspd = float(forecast_data.get('wspd', wspd))
                weather_source = source_msg
                forecast_info = forecast_data.get('description', '')
                
                if selected_date == tomorrow:
                    st.success(f"✅ **Tomorrow's Forecast Loaded!** Weather: {forecast_info}")
                else:
                    st.success(f"✅ Forecast loaded for {selected_date}. Weather: {forecast_info}")
            else:
                st.warning(f"⚠️ {source_msg}. Using estimated values based on historical patterns.")
                weather_source = "Estimated (No Forecast)"
                
        elif is_future_date and not is_within_forecast:
            # Date too far in future - use seasonal estimates
            st.warning(f"⚠️ {selected_date} is beyond 5-day forecast. Using seasonal estimates.")
            demo_data = generate_demo_weather_for_date(datetime.combine(selected_date, datetime.min.time()))
            tavg = demo_data['tavg']
            tmin = demo_data['tmin']
            tmax = demo_data['tmax']
            prcp = demo_data['prcp']
            humidity = demo_data['humidity']
            pres = demo_data['pres']
            wspd = demo_data['wspd']
            weather_source = f"Seasonal Estimate for {selected_date.strftime('%B')}"
            
        else:
            # PAST/TODAY: Load from historical data
            data_path = DATA_DIR / "flood_weather_dataset.csv"
            if data_path.exists():
                df = pd.read_csv(data_path, low_memory=False)
                df['date'] = pd.to_datetime(df['date'])
                
                # Filter by date and location
                matches = df[(df['date'].dt.date == selected_date) & (df['location_key'].str.contains(loc_key, case=False, na=False))]
                
                if len(matches) > 0:
                    row = matches.iloc[0]
                    tavg = float(row['tavg']) if pd.notna(row['tavg']) else tavg
                    tmin = float(row['tmin']) if pd.notna(row['tmin']) else tmin
                    tmax = float(row['tmax']) if pd.notna(row['tmax']) else tmax
                    prcp = float(row['prcp']) if pd.notna(row['prcp']) else prcp
                    humidity = float(row['humidity']) if pd.notna(row['humidity']) else humidity
                    pres = float(row['pres']) if pd.notna(row['pres']) else pres
                    wspd = float(row['wspd']) if pd.notna(row['wspd']) else wspd
                    solar = float(row['solar_radiation']) if pd.notna(row['solar_radiation']) else solar
                    actual_flood = int(row['flood_event']) if pd.notna(row['flood_event']) else None
                    weather_source = f"Historical Data ({selected_date})"
                    
                    st.success(f"✅ Historical weather loaded for {selected_date}")
                    if actual_flood == 1:
                        st.error("🌊 **Note: This date had an actual flood event!**")
                elif selected_date == today:
                    # Today - fetch current weather
                    current_weather = fetch_weather_data(
                        loc_info['latitude'],
                        loc_info['longitude'],
                        OPENWEATHER_API_KEY
                    )
                    tavg = current_weather['tavg']
                    tmin = current_weather['tmin']
                    tmax = current_weather['tmax']
                    prcp = current_weather['prcp']
                    humidity = current_weather['humidity']
                    pres = current_weather['pres']
                    wspd = current_weather['wspd']
                    weather_source = "Current Weather (Live)"
                    st.success("✅ Today's live weather data loaded!")
                else:
                    st.warning(f"⚠️ No historical data for {selected_date}. Enter values manually.")
                    weather_source = "Default Values (no data)"
    
    # Display date type indicator
    if is_future_date:
        st.markdown(f"### 🔮 **FORECAST MODE** - Predicting for {selected_date}")
        if selected_date == tomorrow:
            st.markdown("#### 🌅 Tomorrow's Flood Risk Prediction")
    else:
        st.markdown(f"### 📊 Weather Parameters for {selected_date}")
    
    st.caption(f"📡 Source: {weather_source}")
    
    # Weather inputs in columns (editable even when auto-fetched)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        tavg = st.number_input("Average Temp (°C)", value=tavg, min_value=-20.0, max_value=50.0, step=0.5, key="tavg")
        tmin = st.number_input("Min Temp (°C)", value=tmin, min_value=-30.0, max_value=45.0, step=0.5, key="tmin")
        tmax = st.number_input("Max Temp (°C)", value=tmax, min_value=-10.0, max_value=55.0, step=0.5, key="tmax")
        
    with col2:
        prcp = st.number_input("Precipitation (mm)", value=prcp, min_value=0.0, max_value=500.0, step=1.0, key="prcp")
        humidity = st.number_input("Humidity (%)", value=humidity, min_value=0.0, max_value=100.0, step=1.0, key="humidity")
        pres = st.number_input("Pressure (hPa)", value=pres, min_value=70.0, max_value=1100.0, step=1.0, key="pres")
        
    with col3:
        wspd = st.number_input("Wind Speed (km/h)", value=wspd, min_value=0.0, max_value=200.0, step=1.0, key="wspd")
        solar = st.number_input("Solar Radiation (MJ/m²)", value=solar, min_value=0.0, max_value=40.0, step=0.5, key="solar")
    
    st.markdown("---")
    
    # Predict button - different text for future dates
    button_text = "🔮 Predict Tomorrow's Flood Risk" if selected_date == tomorrow else "🔮 Predict Flood Risk"
    
    if st.button(button_text, type="primary", use_container_width=True):
        # Prepare features
        location_id = 0 if "swat" in selected_loc.lower() else 1
        
        custom_weather = {
            'tavg': tavg,
            'tmin': tmin,
            'tmax': tmax,
            'prcp': prcp,
            'humidity': humidity,
            'pres': pres,
            'wspd': wspd
        }
        
        # Create feature dataframe
        features = {
            'tavg': tavg,
            'tmin': tmin,
            'tmax': tmax,
            'prcp': prcp,
            'wspd': wspd,
            'wpgt': wspd * 1.5,
            'pres': pres,
            'humidity': humidity,
            'solar_radiation': solar,
            'month': selected_date.month,
            'day_of_year': selected_date.timetuple().tm_yday,
            'quarter': (selected_date.month - 1) // 3 + 1,
            'is_monsoon': 1 if selected_date.month in [6, 7, 8, 9] else 0,
            'temp_range': tmax - tmin,
            'high_humidity': 1 if humidity > 70 else 0,
            'pressure_anomaly': pres - 1013,
            'prcp_7day_avg': prcp * 0.8,
            'prcp_3day_sum': prcp * 2.5,  # Estimate 3-day accumulation
            'prcp_7day_sum': prcp * 5,    # Estimate 7-day accumulation
            'heavy_rain': 1 if prcp > 10 else 0,
            'extreme_rain': 1 if prcp > 50 else 0,
            'tavg_7day_avg': tavg,
            'wspd_7day_avg': wspd,
            'location_encoded': location_id
        }
        
        features_df = pd.DataFrame([features])
        
        # Make prediction
        probability, risk_level = predict_flood_risk(model_data, features_df)
        
        # Display results
        st.markdown("### 📊 Prediction Results")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            gauge = create_gauge_chart(probability, risk_level)
            if gauge:
                st.plotly_chart(gauge, use_container_width=True)
            else:
                st.metric("Flood Risk Probability", f"{probability*100:.1f}%")
        
        with col2:
            st.markdown(f"### Risk Level: **{risk_level}**")
            st.markdown(f"**Date:** {selected_date}")
            st.markdown(f"**Location:** {selected_loc}")
            st.markdown(f"**Probability:** {probability*100:.2f}%")
            st.metric("Confidence", f"{max(probability, 1-probability)*100:.1f}%")
            
            if risk_level == "HIGH":
                st.error("⚠️ High flood risk! Take precautions.")
            elif risk_level == "MODERATE":
                st.warning("⚡ Moderate risk. Stay alert.")
            else:
                st.success("✅ Low risk conditions.")
            
            # Show actual vs predicted if we have historical data
            if actual_flood is not None:
                st.markdown("---")
                st.markdown("**Model Validation:**")
                predicted_flood = 1 if risk_level in ["HIGH", "MODERATE"] else 0
                if actual_flood == 1 and predicted_flood == 1:
                    st.success("✅ Correct! Model predicted flood, and flood occurred.")
                elif actual_flood == 0 and predicted_flood == 0:
                    st.success("✅ Correct! Model predicted no flood, and none occurred.")
                elif actual_flood == 1 and predicted_flood == 0:
                    st.error("❌ Missed! Flood occurred but model predicted low risk.")
                else:
                    st.warning("⚠️ False alarm. Model predicted risk but no flood occurred.")
        
        # Special message for tomorrow's prediction
        if selected_date == tomorrow:
            st.markdown("---")
            st.markdown("### 🌅 Tomorrow's Forecast Summary")
            if risk_level == "HIGH":
                st.error(f"""
                ⚠️ **HIGH FLOOD RISK TOMORROW ({tomorrow})**
                
                Based on weather forecast:
                - Expected precipitation: **{prcp:.1f} mm**
                - Temperature: **{tavg:.1f}°C**
                - Humidity: **{humidity:.0f}%**
                
                **RECOMMENDED ACTIONS:**
                - Monitor official alerts closely
                - Prepare emergency supplies
                - Know your evacuation routes
                - Stay away from rivers and streams
                """)
            elif risk_level == "MODERATE":
                st.warning(f"""
                ⚡ **MODERATE FLOOD RISK TOMORROW ({tomorrow})**
                
                Based on weather forecast:
                - Expected precipitation: **{prcp:.1f} mm**
                - Temperature: **{tavg:.1f}°C**
                
                **RECOMMENDED ACTIONS:**
                - Stay alert to weather updates
                - Review emergency plans
                - Secure outdoor items
                """)
            else:
                st.success(f"""
                ✅ **LOW FLOOD RISK TOMORROW ({tomorrow})**
                
                Weather forecast shows normal conditions:
                - Expected precipitation: **{prcp:.1f} mm**
                - Temperature: **{tavg:.1f}°C**
                
                No special precautions needed, but stay informed.
                """)
        
        # Show input summary
        with st.expander("📋 Input Summary"):
            input_df = pd.DataFrame({
                'Parameter': ['Temperature (Avg)', 'Temperature (Min)', 'Temperature (Max)', 
                             'Precipitation', 'Humidity', 'Pressure', 'Wind Speed', 'Solar Radiation'],
                'Value': [f"{tavg}°C", f"{tmin}°C", f"{tmax}°C", 
                         f"{prcp} mm", f"{humidity}%", f"{pres} hPa", f"{wspd} km/h", f"{solar} MJ/m²"]
            })
            st.table(input_df)
    
    # Historical lookup section
    st.markdown("---")
    st.markdown("### 📜 Lookup Historical Date")
    st.markdown("Check if a specific date in history had a flood event.")
    
    lookup_date = st.date_input(
        "Select a historical date",
        value=datetime(2010, 8, 1),
        key="lookup_date"
    )
    
    if st.button("🔍 Lookup Date"):
        # Load historical data
        data_path = DATA_DIR / "flood_weather_dataset.csv"
        if data_path.exists():
            df = pd.read_csv(data_path)
            df['date'] = pd.to_datetime(df['date'])
            
            # Find matching records
            matches = df[df['date'].dt.date == lookup_date]
            
            if len(matches) > 0:
                st.success(f"Found {len(matches)} record(s) for {lookup_date}")
                
                for _, row in matches.iterrows():
                    loc_name = row.get('location_name', 'Unknown')
                    flood = row.get('flood_event', 0)
                    
                    if flood == 1:
                        st.error(f"🌊 **FLOOD EVENT** at {loc_name}")
                        if 'flood_severity' in row and pd.notna(row['flood_severity']):
                            st.write(f"Severity: {row['flood_severity']}")
                        if 'flood_notes' in row and pd.notna(row['flood_notes']):
                            st.write(f"Notes: {row['flood_notes']}")
                    else:
                        st.info(f"✅ No flood at {loc_name}")
                    
                    # Show weather conditions
                    with st.expander(f"Weather at {loc_name}"):
                        weather_cols = ['tavg', 'tmin', 'tmax', 'prcp', 'humidity', 'wspd', 'pres']
                        weather_data = {col: row.get(col, 'N/A') for col in weather_cols if col in row}
                        st.json(weather_data)
            else:
                st.warning(f"No data found for {lookup_date}")
        else:
            st.error("Historical data not available.")


# ============================================================================
# AI TECHNIQUE PAGES - Week 8-12 Requirements
# ============================================================================

def show_search_algorithms():
    """Display Search Algorithms page (Week 8)"""
    st.title("🛣️ Evacuation Route Planning (A* Search)")
    st.caption("Grid-based route planning comparing A*, BFS, and DFS for flood evacuation.")
    
    try:
        from code.search_algorithms import FloodEvacuationGrid
        
        st.subheader("🗺️ Flood Evacuation Grid Simulation")

        col1, col2, col3 = st.columns(3)
        with col1:
            grid_size = st.slider("Grid Size", 10, 30, 15)
            flood_prob = st.slider("Flood Probability", 0.1, 0.5, 0.25)
        with col2:
            start_x = st.number_input("Start X", 0, grid_size - 1, 0)
            start_y = st.number_input("Start Y", 0, grid_size - 1, 0)
        with col3:
            dest_x = st.number_input("Destination X", 0, grid_size - 1, grid_size - 1)
            dest_y = st.number_input("Destination Y", 0, grid_size - 1, grid_size - 1)
        seed = st.number_input("Random Seed", 0, 1000, 42)

        if st.button("🚀 Run A* & Compare", use_container_width=True):
            grid = FloodEvacuationGrid(grid_size, grid_size)
            grid.generate_flood_scenario(flood_probability=flood_prob,
                                        n_safe_zones=1,
                                        seed=seed)
            grid.start = (start_x, start_y)
            grid.safe_zones = [(dest_x, dest_y)]

            # Visual grid
            st.markdown("#### Grid Legend")
            st.markdown("- 🟦 Normal terrain | 🟥 Flooded | 🟩 Destination | 🟨 Start")
            visual_grid = []
            for i in range(grid_size):
                row = []
                for j in range(grid_size):
                    if (i, j) == grid.start:
                        row.append("🟨")
                    elif (i, j) in grid.safe_zones:
                        row.append("🟩")
                    elif grid.grid[i][j] == 2:
                        row.append("🟥")
                    else:
                        row.append("🟦")
                visual_grid.append("".join(row))
            st.text("\n".join(visual_grid))
            st.info(f"Start: {grid.start} | Destination: {grid.safe_zones[0]}")

            results = grid.compare_algorithms()

            st.subheader("📊 Algorithm Comparison")
            results_df = pd.DataFrame(results["comparison"])
            st.dataframe(results_df, hide_index=True)

            details = results.get("details", {})
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**A***")
                a_star = details.get("A*", {})
                if a_star.get("success"):
                    st.success(f"Path: {a_star.get('path_length', 0)} steps | Cost {a_star.get('cost', 0):.1f}")
                else:
                    st.error("No path found")
            with col2:
                st.markdown("**BFS**")
                bfs = details.get("BFS", {})
                st.success(f"Path: {bfs.get('path_length', '—')} steps" if bfs.get("success") else "No path found")
            with col3:
                st.markdown("**DFS**")
                dfs = details.get("DFS", {})
                if dfs.get("success"):
                    st.warning(f"Path: {dfs.get('path_length', 0)} steps (may be non-optimal)")
                else:
                    st.error("No path found")

            st.info("💡 A* typically yields the shortest safe route for evacuation.")
            
    except ImportError as e:
        st.error(f"Could not load search algorithms module: {e}")


def show_csp_page():
    """Display CSP Resource Allocation page (Week 9)"""
    st.title("🧩 CSP: Emergency Resource Allocation")
    
    st.markdown("""
    ### Week 9: Constraint Satisfaction Problems
    
    This module uses **CSP techniques** to allocate emergency resources during flood disasters.
    
    **Problem Formulation:**
    - **Variables**: Evacuation shelters
    - **Domains**: Available resources (medical teams, rescue boats, supplies)
    - **Constraints**: Resource limits, minimum requirements, distance limits
    
    **Techniques Used:**
    - AC-3 Arc Consistency
    - Backtracking Search
    - MRV (Minimum Remaining Values) heuristic
    - LCV (Least Constraining Value) heuristic
    """)
    
    try:
        from code.csp_resource_allocation import FloodResourceAllocationCSP
        
        st.subheader("⚙️ Configure Scenario")
        
        col1, col2 = st.columns(2)
        with col1:
            num_shelters = st.slider("Number of Shelters", 2, 8, 4)
        with col2:
            num_resources = st.slider("Number of Resources", 4, 15, 8)
        
        if st.button("🔧 Solve Resource Allocation"):
            csp = FloodResourceAllocationCSP()
            csp.generate_scenario(num_shelters=num_shelters, 
                                 num_resources=num_resources, 
                                 seed=42)
            
            # Show shelters
            st.subheader("🏠 Evacuation Shelters")
            shelter_data = []
            for s_id, s in csp.shelters.items():
                shelter_data.append({
                    "ID": s_id,
                    "Name": s["name"],
                    "Population": s["population"],
                    "Priority": s["priority"],
                    "Min Medical": s["min_medical"],
                    "Min Rescue": s["min_rescue"],
                    "Min Supplies": s["min_supplies"]
                })
            st.dataframe(pd.DataFrame(shelter_data))
            
            # Show resources
            st.subheader("📦 Available Resources")
            resource_data = []
            for r_id, r in csp.resources.items():
                resource_data.append({
                    "ID": r_id,
                    "Type": r["type"].capitalize(),
                    "Quantity": r["quantity"]
                })
            st.dataframe(pd.DataFrame(resource_data))
            
            # Solve
            with st.spinner("Solving CSP with backtracking..."):
                result = csp.solve()
            
            if result["success"]:
                st.success("✅ Solution Found!")
                
                st.subheader("📋 Resource Assignments")
                for entry in result["summary"]:
                    with st.expander(f"📍 {entry['shelter']} (Pop: {entry['population']})"):
                        if entry["resources_assigned"]:
                            for r in entry["resources_assigned"]:
                                icon = {"medical": "🏥", "rescue": "🚤", "supplies": "📦"}.get(r["type"], "•")
                                st.write(f"{icon} {r['id']}: {r['type']} (Qty: {r['quantity']})")
                        else:
                            st.write("No resources assigned")
            else:
                st.error(f"❌ No solution found: {result.get('error')}")
                
    except ImportError as e:
        st.error(f"Could not load CSP module: {e}")


def show_neural_network_page():
    """Display Neural Network page (Week 11)"""
    st.title("🧬 LSTM Neural Network for Flood Prediction")
    
    st.markdown("""
    ### Week 11: Neural Networks
    
    This module implements an **LSTM (Long Short-Term Memory)** neural network
    for time-series flood prediction.
    
    **Architecture:**
    - Input Layer: Sequential weather data (7 days lookback)
    - LSTM Layer: 64 hidden units with tanh activation
    - Output Layer: Sigmoid for flood probability
    
    **Why LSTM?**
    - Captures temporal patterns in weather data
    - Handles long-term dependencies (monsoon buildup)
    - Better than simple feedforward networks for sequences
    """)
    
    try:
        from code.neural_network import SimpleLSTMWrapper, FloodLSTM
        
        st.subheader("🎯 Train LSTM on Synthetic Data")
        
        col1, col2 = st.columns(2)
        with col1:
            n_samples = st.slider("Training Samples", 200, 1000, 500)
            epochs = st.slider("Training Epochs", 10, 50, 20)
        with col2:
            hidden_size = st.slider("Hidden Units", 16, 64, 32)
            seq_length = st.slider("Sequence Length (days)", 3, 14, 7)
        
        if st.button("🚀 Train LSTM Model"):
            with st.spinner("Generating data and training..."):
                # Generate synthetic data
                np.random.seed(42)
                X = np.random.randn(n_samples, 5)  # 5 weather features
                
                # Create flood labels based on precipitation pattern
                flood_prob = 0.3 * X[:, 1] + 0.2 * X[:, 2]
                y = (flood_prob > np.percentile(flood_prob, 90)).astype(int)
                
                st.info(f"Dataset: {n_samples} samples, {sum(y)} flood events ({100*sum(y)/len(y):.1f}%)")
                
                # Split data
                split = int(0.8 * n_samples)
                X_train, X_test = X[:split], X[split:]
                y_train, y_test = y[:split], y[split:]
                
                # Train
                lstm = SimpleLSTMWrapper(sequence_length=seq_length, hidden_size=hidden_size)
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                history = lstm.fit(X_train, y_train, epochs=epochs)
                progress_bar.progress(100)
                
                # Evaluate
                metrics = lstm.evaluate(X_test, y_test)
                
            st.success("Training Complete!")
            
            # Show metrics
            st.subheader("📊 Model Performance")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
            col2.metric("Precision", f"{metrics['precision']:.2%}")
            col3.metric("Recall", f"{metrics['recall']:.2%}")
            col4.metric("F1 Score", f"{metrics['f1']:.2%}")
            
            # Training curve
            if PLOTLY_AVAILABLE:
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=history['loss'], name='Loss', mode='lines'))
                fig.update_layout(title='Training Loss', xaxis_title='Epoch', yaxis_title='Loss')
                st.plotly_chart(fig, use_container_width=True)
            
            # Confusion matrix
            st.subheader("Confusion Matrix")
            cm = metrics['confusion_matrix']
            st.write(f"TN: {cm[0][0]} | FP: {cm[0][1]}")
            st.write(f"FN: {cm[1][0]} | TP: {cm[1][1]}")
            
    except ImportError as e:
        st.error(f"Could not load neural network module: {e}")


def show_clustering_page():
    """Display K-Means Clustering page (Week 12)"""
    st.title("📈 K-Means Clustering for Flood Patterns")
    
    st.markdown("""
    ### Week 12: Clustering Analysis
    
    This module uses **K-Means clustering** to identify patterns in flood conditions.
    
    **Applications:**
    - Identify different types of flood conditions (monsoon, flash flood, riverine)
    - Group similar weather patterns
    - Discover regional risk profiles
    
    **Techniques:**
    - K-Means++ initialization
    - Elbow method for optimal K
    - Cluster interpretation based on centroids
    """)
    
    try:
        from code.clustering import FloodPatternKMeans, FloodPatternAnalyzer, find_optimal_k
        
        st.subheader("⚙️ Clustering Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            n_samples = st.slider("Number of Samples", 200, 800, 400)
        with col2:
            n_clusters = st.slider("Number of Clusters (K)", 2, 8, 5)
        
        if st.button("🔬 Run Clustering Analysis"):
            with st.spinner("Clustering weather patterns..."):
                # Generate synthetic weather data with patterns
                np.random.seed(42)
                
                # Create different weather patterns
                p1 = np.random.randn(n_samples//4, 5) + np.array([25, 50, 85, 1000, 10])  # Monsoon
                p2 = np.random.randn(n_samples//4, 5) + np.array([30, 80, 70, 995, 15])   # Flash flood
                p3 = np.random.randn(n_samples//4, 5) + np.array([35, 5, 40, 1015, 5])    # Dry
                p4 = np.random.randn(n_samples//4, 5) + np.array([28, 20, 60, 1008, 8])   # Moderate
                
                X = np.vstack([p1, p2, p3, p4])
                feature_names = ['Temperature', 'Precipitation', 'Humidity', 'Pressure', 'Wind Speed']
                
                # Fit clustering
                analyzer = FloodPatternAnalyzer(n_clusters=n_clusters)
                analyzer.fit(X, feature_names)
                
                labels = analyzer.predict(X)
                
            st.success("Clustering Complete!")
            
            # Cluster distribution
            st.subheader("📊 Cluster Distribution")
            cluster_counts = pd.Series(labels).value_counts().sort_index()
            
            if PLOTLY_AVAILABLE:
                fig = px.pie(values=cluster_counts.values, 
                           names=[f"Cluster {i}" for i in cluster_counts.index],
                           title="Weather Pattern Distribution")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(cluster_counts)
            
            # Cluster interpretations
            st.subheader("🔍 Cluster Interpretations")
            for k, interp in analyzer.cluster_interpretations.items():
                with st.expander(f"Cluster {k}: {interp['name']}"):
                    risk_color = {"HIGH": "🔴", "MODERATE": "🟡", "LOW": "🟢"}.get(interp['risk_level'], "⚪")
                    st.write(f"**Risk Level**: {risk_color} {interp['risk_level']}")
                    st.write(f"**Description**: {interp['description']}")
            
            # Test sample analysis
            st.subheader("🧪 Analyze Sample")
            col1, col2, col3, col4, col5 = st.columns(5)
            temp = col1.number_input("Temp", value=25.0)
            prcp = col2.number_input("Precip", value=40.0)
            humid = col3.number_input("Humidity", value=70.0)
            pres = col4.number_input("Pressure", value=1000.0)
            wind = col5.number_input("Wind", value=10.0)
            
            if st.button("Classify Sample"):
                sample = np.array([[temp, prcp, humid, pres, wind]])
                analysis = analyzer.analyze_sample(sample)
                
                st.info(f"**Pattern**: {analysis['pattern_name']}")
                st.info(f"**Risk Level**: {analysis['risk_level']}")
                
    except ImportError as e:
        st.error(f"Could not load clustering module: {e}")


def show_reinforcement_learning_page():
    """Display Reinforcement Learning page (Week 12)"""
    st.title("🎮 Q-Learning for Evacuation Decisions")
    
    st.markdown("""
    ### Week 12: Reinforcement Learning
    
    This module implements **Q-Learning** for optimal flood evacuation decisions.
    
    **Environment:**
    - **States**: (flood_level, population_at_risk, resources_deployed, time_remaining)
    - **Actions**: Wait, Issue Warning, Voluntary Evacuation, Mandatory Evacuation, Deploy Resources
    - **Rewards**: +100 per person saved, -500 per casualty, -30 for false alarms
    
    **Q-Learning Features:**
    - Epsilon-greedy exploration
    - Bellman equation updates
    - Learned policy for optimal decisions
    """)
    
    try:
        from code.reinforcement_learning import FloodEvacuationRL
        
        st.subheader("🎯 Train Q-Learning Agent")
        
        col1, col2 = st.columns(2)
        with col1:
            episodes = st.slider("Training Episodes", 100, 1000, 300)
        with col2:
            eval_episodes = st.slider("Evaluation Episodes", 20, 100, 50)
        
        if st.button("🚀 Train Agent"):
            rl_system = FloodEvacuationRL()
            
            with st.spinner("Training Q-Learning agent..."):
                progress = st.progress(0)
                history = rl_system.train(episodes=episodes)
                progress.progress(100)
                
            st.success("Training Complete!")
            
            # Training curve
            if PLOTLY_AVAILABLE:
                fig = go.Figure()
                # Smooth rewards
                window = min(50, len(history['rewards']) // 5)
                smoothed = pd.Series(history['rewards']).rolling(window).mean()
                fig.add_trace(go.Scatter(y=smoothed, name='Avg Reward', mode='lines'))
                fig.update_layout(title='Training Progress', xaxis_title='Episode', yaxis_title='Reward')
                st.plotly_chart(fig, use_container_width=True)
            
            # Evaluate
            st.subheader("📊 Evaluation Results")
            with st.spinner("Evaluating policy..."):
                results = rl_system.evaluate(n_episodes=eval_episodes)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Avg Reward", f"{results['avg_reward']:.1f}")
            col2.metric("Avg Evacuated", f"{results['avg_evacuated']:.0f}")
            col3.metric("Avg Casualties", f"{results['avg_casualties']:.1f}")
            col4.metric("Success Rate", f"{results['success_rate']*100:.1f}%")
            
            # Get recommendation
            st.subheader("🎯 Get Recommendation")
            col1, col2, col3, col4 = st.columns(4)
            flood_level = col1.selectbox("Flood Level", [0, 1, 2, 3, 4], index=2,
                                        format_func=lambda x: ["None", "Low", "Moderate", "High", "Severe"][x])
            population = col2.number_input("Population at Risk", 100, 1000, 500)
            resources = col3.number_input("Resources Deployed", 0, 10, 2)
            time_left = col4.number_input("Hours Remaining", 1, 24, 12)
            
            if st.button("Get Recommendation"):
                rec = rl_system.get_recommendation(flood_level, population, resources, time_left)
                
                action_icons = {
                    "Wait & Monitor": "⏳",
                    "Issue Warning": "⚠️",
                    "Voluntary Evacuation": "🚶",
                    "Mandatory Evacuation": "🚨",
                    "Deploy Resources": "🚁"
                }
                icon = action_icons.get(rec['action_name'], "•")
                
                st.success(f"{icon} **Recommended Action**: {rec['action_name']}")
                st.info(rec['explanation'])
                
                with st.expander("Q-Values for All Actions"):
                    for action, value in rec['all_q_values'].items():
                        st.write(f"{action}: {value:.2f}")
                        
    except ImportError as e:
        st.error(f"Could not load reinforcement learning module: {e}")


def show_explainability_page(model_data):
    """Display SHAP/LIME Explainability page"""
    st.title("🔬 Model Explainability (SHAP & LIME)")
    
    st.markdown("""
    ### Explainability: Understanding Model Decisions
    
    This module explains **why** the model makes specific predictions.
    
    **SHAP (SHapley Additive exPlanations):**
    - Based on game theory (Shapley values)
    - Shows contribution of each feature to prediction
    - Global and local explanations
    
    **LIME (Local Interpretable Model-agnostic Explanations):**
    - Fits simple model locally
    - Provides interpretable feature weights
    - Works with any black-box model
    """)
    
    try:
        from code.explainability import FloodPredictionExplainer, SHAPExplainer, LIMEExplainer
        
        # Create demo model if no model loaded
        if model_data is None:
            st.warning("No trained model found. Using demo model for illustration.")
            
            class DemoModel:
                def predict_proba(self, X):
                    X = np.atleast_2d(X)
                    probs = 0.3 * (X[:, 3] / 100) + 0.2 * (X[:, 7] / 100)
                    probs = np.clip(probs, 0, 1)
                    return np.column_stack([1 - probs, probs])
                
                def predict(self, X):
                    return (self.predict_proba(X)[:, 1] > 0.5).astype(int)
            
            model = DemoModel()
        else:
            if isinstance(model_data, dict):
                model = model_data.get('model', model_data)
            else:
                model = model_data
        
        feature_names = ['tavg', 'tmin', 'tmax', 'prcp', 'wspd', 'wpgt', 'pres', 'humidity',
                        'solar_radiation', 'month', 'day_of_year', 'quarter']
        
        st.subheader("🎯 Explain a Prediction")
        
        st.markdown("**Enter Weather Conditions:**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            tavg = st.number_input("Temperature (°C)", value=25.0)
            tmin = st.number_input("Min Temp", value=20.0)
            tmax = st.number_input("Max Temp", value=30.0)
        with col2:
            prcp = st.number_input("Precipitation (mm)", value=50.0)
            wspd = st.number_input("Wind Speed (km/h)", value=15.0)
            wpgt = st.number_input("Wind Gust", value=25.0)
        with col3:
            pres = st.number_input("Pressure (hPa)", value=1005.0)
            humidity = st.number_input("Humidity (%)", value=80.0)
            solar = st.number_input("Solar Radiation", value=15.0)
        with col4:
            month = st.selectbox("Month", range(1, 13), index=6)
            day_of_year = st.number_input("Day of Year", 1, 366, 180)
            quarter = (month - 1) // 3 + 1
            st.write(f"Quarter: {quarter}")
        
        sample = np.array([tavg, tmin, tmax, prcp, wspd, wpgt, pres, humidity, 
                          solar, month, day_of_year, quarter])
        
        if st.button("🔍 Explain Prediction"):
            # Generate background data
            np.random.seed(42)
            background = np.random.randn(100, len(feature_names)) * 10 + sample
            
            explainer = FloodPredictionExplainer(model, feature_names)
            explainer.fit(background)
            
            with st.spinner("Generating explanations..."):
                explanation = explainer.explain_prediction(sample, method='both')
            
            # Show prediction
            pred = explanation.get('shap', {}).get('prediction', 0)
            risk_level = "HIGH" if pred > 0.6 else "MODERATE" if pred > 0.3 else "LOW"
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Flood Probability", f"{pred*100:.1f}%")
            with col2:
                risk_colors = {"HIGH": "🔴", "MODERATE": "🟡", "LOW": "🟢"}
                st.metric("Risk Level", f"{risk_colors.get(risk_level, '')} {risk_level}")
            
            st.markdown("---")
            
            # SHAP explanations
            if 'shap' in explanation:
                st.subheader("📊 SHAP Feature Contributions")
                
                shap_vals = explanation['shap']['shap_values']
                contrib_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Value': sample,
                    'SHAP Value': shap_vals
                }).sort_values('SHAP Value', key=abs, ascending=False)
                
                if PLOTLY_AVAILABLE:
                    fig = px.bar(contrib_df.head(10), x='SHAP Value', y='Feature', 
                               orientation='h', color='SHAP Value',
                               color_continuous_scale=['green', 'gray', 'red'])
                    fig.update_layout(title='Top 10 Feature Contributions')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.dataframe(contrib_df)
                
                st.markdown("**Interpretation:**")
                st.markdown(explanation['shap']['prediction_explanation'])
            
            # Flood interpretation
            st.subheader("🌊 Flood Risk Assessment")
            st.markdown(explanation.get('flood_interpretation', ''))
                
    except ImportError as e:
        st.error(f"Could not load explainability module: {e}")


def show_about():
    """Display about page"""
    st.title("ℹ️ About This Project")
    
    st.markdown("""
    ## AI-Based Natural Disaster Prediction Web App
    
    This application predicts flood risk for districts in Khyber Pakhtunkhwa, Pakistan
    using machine learning models trained on historical weather and flood data.
    
    ### 🎯 Main Features
    - **Real-time Weather Integration**: Live weather data from OpenWeatherMap API
    - **ML-Based Predictions**: Trained models predict flood probability
    - **Custom Prediction**: Enter manual weather parameters for prediction
    - **Historical Analysis**: View past weather patterns and flood events (2000-2025)
    - **Alert System**: Color-coded risk levels for quick assessment
    
    ### 🧠 AI Techniques Implemented
    
    | Week | Technique | Application |
    |------|-----------|-------------|
    | 8 | **Search Algorithms** | A*, BFS, DFS for evacuation route planning |
    | 9 | **CSP** | Constraint satisfaction for emergency resource allocation |
    | 11 | **Neural Networks** | LSTM for time-series flood prediction |
    | 12 | **Clustering** | K-Means for flood pattern analysis |
    | 12 | **Reinforcement Learning** | Q-Learning for evacuation decisions |
    | Bonus | **Explainability** | SHAP & LIME for model interpretation |
    
    ### 📍 Covered Locations
    - Swat District, KP
    - Upper Dir District, KP
    
    ### 🔬 Technology Stack
    - **Frontend**: Streamlit with Plotly visualizations
    - **ML Models**: Logistic Regression, Random Forest, Gradient Boosting
    - **Neural Network**: Custom LSTM implementation
    - **Data Sources**: Meteostat, NASA POWER, NDMA Reports
    - **Deployment**: Docker, GitHub Actions CI/CD
    
    ### 📊 Dataset Statistics
    - **Total Records**: 18,902 weather observations
    - **Time Range**: January 2000 - November 2025
    - **Flood Events**: 517 labeled events (2.74%)
    - **Features**: 24 engineered features
    
    ### 🏆 Model Performance
    - **Best Model**: Logistic Regression with class weights
    - **Recall**: 60% (prioritized for safety)
    - **Features**: Temperature, precipitation, humidity, pressure, wind speed, 
      monsoon indicators, cumulative rainfall, and more
    
    ### 👨‍💻 Developer
    **CS351 - Artificial Intelligence Project**
    Semester 5
    
    ### ⚠️ Disclaimer
    This is an educational project demonstrating AI techniques for disaster prediction.
    For actual emergency situations, please refer to official government sources and NDMA alerts.
    """)
    
    # Show project structure
    with st.expander("📁 Project Structure"):
        st.code("""
AI-Based Natural Disaster/
├── app.py                    # Main Streamlit application
├── code/
│   ├── search_algorithms.py  # A*, BFS, DFS (Week 8)
│   ├── csp_resource_allocation.py  # CSP (Week 9)
│   ├── neural_network.py     # LSTM (Week 11)
│   ├── clustering.py         # K-Means (Week 12)
│   ├── reinforcement_learning.py  # Q-Learning (Week 12)
│   ├── explainability.py     # SHAP/LIME (Bonus)
│   ├── improved_models.py    # ML model training
│   ├── preprocessing.py      # Data preprocessing
│   └── ...
├── data/
│   ├── processed/            # Cleaned datasets
│   └── raw/                  # Raw data files
├── results/                  # Model outputs
├── Dockerfile               # Docker deployment
├── docker-compose.yml       # Docker Compose config
└── requirements.txt         # Python dependencies
        """, language="")


if __name__ == "__main__":
    main()
