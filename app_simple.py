"""
AI-Based Natural Disaster Prediction Web App
SIMPLIFIED VERSION FOR PRESENTATION
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import requests
from datetime import datetime, timedelta
from pathlib import Path
import os

# Try to import plotly
try:
    import plotly.express as px
    import plotly.graph_objects as go
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

# API Key
try:
    OPENWEATHER_API_KEY = st.secrets.get("OPENWEATHER_API_KEY", "demo")
except:
    OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "demo")


@st.cache_resource
def load_model():
    """Load the trained flood prediction model"""
    model_paths = [
        RESULTS_DIR / "best_flood_model.pkl",
        RESULTS_DIR / "random_forest_model.pkl",
        RESULTS_DIR / "logistic_regression_model.pkl"
    ]
    
    for path in model_paths:
        if path.exists():
            try:
                with open(path, 'rb') as f:
                    return pickle.load(f)
            except:
                pass
    return None


def generate_demo_weather():
    """Generate realistic demo weather data"""
    month = datetime.now().month
    if month in [6, 7, 8]:  # Monsoon
        temp_base, prcp_base, humidity_base = 28, 25, 75
    elif month in [12, 1, 2]:  # Winter
        temp_base, prcp_base, humidity_base = 8, 5, 55
    else:  # Spring/Autumn
        temp_base, prcp_base, humidity_base = 20, 10, 60
    
    return {
        "tavg": temp_base + np.random.uniform(-3, 3),
        "tmin": temp_base - 5 + np.random.uniform(-2, 2),
        "tmax": temp_base + 8 + np.random.uniform(-2, 2),
        "humidity": humidity_base + np.random.uniform(-10, 10),
        "pres": 1010 + np.random.uniform(-10, 10),
        "wspd": 8 + np.random.uniform(-3, 5),
        "prcp": prcp_base + np.random.uniform(-5, 15),
        "description": "Demo weather data",
        "icon": "10d"
    }


def generate_demo_weather_for_date(target_date):
    """Generate demo weather for a specific date"""
    month = target_date.month
    if month in [6, 7, 8]:  # Monsoon - HIGH RISK
        temp_base, prcp_base, humidity_base = 28, 25, 75
    elif month in [12, 1, 2]:  # Winter
        temp_base, prcp_base, humidity_base = 8, 5, 55
    else:
        temp_base, prcp_base, humidity_base = 20, 10, 60
    
    return {
        "tavg": temp_base + np.random.uniform(-3, 3),
        "tmin": temp_base - 5 + np.random.uniform(-2, 2),
        "tmax": temp_base + 8 + np.random.uniform(-2, 2),
        "humidity": humidity_base + np.random.uniform(-10, 10),
        "pres": 1010 + np.random.uniform(-10, 10),
        "wspd": 8 + np.random.uniform(-3, 5),
        "prcp": prcp_base + np.random.uniform(-5, 15),
        "description": f"Forecast for {target_date.strftime('%B %d')}",
        "icon": "10d"
    }


@st.cache_data(ttl=1800)
def fetch_weather_data(lat, lon, api_key):
    """Fetch current weather from OpenWeatherMap"""
    if api_key == "demo":
        return generate_demo_weather()
    
    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"lat": lat, "lon": lon, "appid": api_key, "units": "metric"}
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            return {
                "tavg": data["main"]["temp"],
                "tmin": data["main"]["temp_min"],
                "tmax": data["main"]["temp_max"],
                "humidity": data["main"]["humidity"],
                "pres": data["main"]["pressure"],
                "wspd": data["wind"]["speed"] * 3.6,
                "prcp": data.get("rain", {}).get("1h", 0),
                "description": data["weather"][0]["description"],
                "icon": data["weather"][0]["icon"]
            }
    except:
        pass
    return generate_demo_weather()


@st.cache_data(ttl=3600)
def fetch_weather_forecast(lat, lon, api_key, target_date):
    """Fetch weather forecast for future date"""
    if api_key == "demo":
        return generate_demo_weather_for_date(target_date), "Demo Mode"
    
    try:
        url = "https://api.openweathermap.org/data/2.5/forecast"
        params = {"lat": lat, "lon": lon, "appid": api_key, "units": "metric"}
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            target_str = target_date.strftime("%Y-%m-%d")
            day_forecasts = [f for f in data.get("list", []) if target_str in f["dt_txt"]]
            
            if day_forecasts:
                return {
                    "tavg": np.mean([f["main"]["temp"] for f in day_forecasts]),
                    "tmin": min([f["main"]["temp_min"] for f in day_forecasts]),
                    "tmax": max([f["main"]["temp_max"] for f in day_forecasts]),
                    "humidity": np.mean([f["main"]["humidity"] for f in day_forecasts]),
                    "pres": np.mean([f["main"]["pressure"] for f in day_forecasts]),
                    "wspd": np.mean([f["wind"]["speed"] * 3.6 for f in day_forecasts]),
                    "prcp": sum(f.get("rain", {}).get("3h", 0) for f in day_forecasts),
                    "description": day_forecasts[0]["weather"][0]["description"],
                    "icon": day_forecasts[0]["weather"][0]["icon"]
                }, "Live Forecast"
    except:
        pass
    return generate_demo_weather_for_date(target_date), "Demo Mode"


def create_features(weather, location, date):
    """Create feature vector matching the model's training features (24 features)"""
    month = date.month
    day_of_year = date.timetuple().tm_yday
    
    tavg = weather.get('tavg', 20)
    tmin = weather.get('tmin', 15)
    tmax = weather.get('tmax', 25)
    prcp = weather.get('prcp', 0)
    wspd = weather.get('wspd', 10)
    pres = weather.get('pres', 1010)
    humidity = weather.get('humidity', 60)
    
    # These 24 features must match best_flood_model.pkl feature_names exactly
    features = {
        'tavg': tavg,
        'tmin': tmin,
        'tmax': tmax,
        'prcp': prcp,
        'wspd': wspd,
        'wpgt': wspd * 1.5,  # Estimate wind gust
        'pres': pres,
        'humidity': humidity,
        'solar_radiation': 200 if month in [5,6,7,8] else 150,
        'month': month,
        'day_of_year': day_of_year,
        'quarter': (month - 1) // 3 + 1,
        'is_monsoon': 1 if month in [7, 8, 9] else 0,
        'temp_range': tmax - tmin,
        'high_humidity': 1 if humidity > 70 else 0,
        'pressure_anomaly': pres - 1013.25,
        'prcp_7day_avg': prcp,  # Use current as estimate
        'prcp_3day_sum': prcp * 3,
        'prcp_7day_sum': prcp * 7,
        'heavy_rain': 1 if prcp > 20 else 0,
        'extreme_rain': 1 if prcp > 50 else 0,
        'tavg_7day_avg': tavg,
        'wspd_7day_avg': wspd,
        'location_encoded': location.get('location_id', 0)
    }
    return features


def make_prediction(model_data, features):
    """Make flood prediction using correct feature order and scaler"""
    if model_data is None:
        return None, None
    
    try:
        model = model_data.get('model') if isinstance(model_data, dict) else model_data
        threshold = model_data.get('threshold', 0.5) if isinstance(model_data, dict) else 0.5
        scaler = model_data.get('scaler') if isinstance(model_data, dict) else None
        
        # Get feature order from model_data or use default 24 features
        if isinstance(model_data, dict) and 'feature_names' in model_data:
            feature_order = model_data['feature_names']
        else:
            # Default 24 features for best_flood_model
            feature_order = [
                'tavg', 'tmin', 'tmax', 'prcp', 'wspd', 'wpgt', 'pres', 'humidity',
                'solar_radiation', 'month', 'day_of_year', 'quarter', 'is_monsoon',
                'temp_range', 'high_humidity', 'pressure_anomaly', 'prcp_7day_avg',
                'prcp_3day_sum', 'prcp_7day_sum', 'heavy_rain', 'extreme_rain',
                'tavg_7day_avg', 'wspd_7day_avg', 'location_encoded'
            ]
        
        X = pd.DataFrame([features])[feature_order].values
        
        # CRITICAL: Apply scaler if available (model was trained on scaled data!)
        if scaler is not None:
            X = scaler.transform(X)
        
        if hasattr(model, 'predict_proba'):
            prob = model.predict_proba(X)[0][1]
        else:
            prob = model.predict(X)[0]
        
        return prob, prob >= threshold
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None


# ============== MAIN PAGES ==============

def show_dashboard(location, model_data):
    """Main Dashboard - Real-time prediction"""
    st.title("🌊 Flood Prediction Dashboard")
    st.markdown(f"### 📍 {location['name']}")
    
    # Fetch weather
    weather = fetch_weather_data(location['latitude'], location['longitude'], OPENWEATHER_API_KEY)
    
    # Weather display
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🌡️ Temperature", f"{weather['tavg']:.1f}°C")
    col2.metric("💧 Humidity", f"{weather['humidity']:.0f}%")
    col3.metric("🌧️ Precipitation", f"{weather['prcp']:.1f} mm")
    col4.metric("💨 Wind Speed", f"{weather['wspd']:.1f} km/h")
    
    # Prediction
    st.markdown("---")
    features = create_features(weather, location, datetime.now())
    prob, is_flood = make_prediction(model_data, features)
    
    if prob is not None:
        st.subheader("🎯 Flood Risk Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            risk_color = "🔴" if is_flood else "🟢"
            risk_text = "HIGH RISK" if is_flood else "LOW RISK"
            st.markdown(f"## {risk_color} {risk_text}")
            st.metric("Probability", f"{prob*100:.1f}%")
        
        with col2:
            if PLOTLY_AVAILABLE:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkred" if is_flood else "darkgreen"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 60], 'color': "yellow"},
                            {'range': [60, 100], 'color': "lightcoral"}
                        ],
                        'threshold': {'value': 50, 'line': {'color': "red", 'width': 2}}
                    }
                ))
                fig.update_layout(height=300, margin=dict(t=20, b=20))
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Model not loaded. Please train the model first.")


def show_custom_prediction(location, model_data):
    """Custom prediction with date selection"""
    st.title("🔮 Flood Prediction")
    st.markdown(f"### 📍 {location['name']}")
    
    # Quick date buttons
    st.subheader("📅 Quick Date Selection")
    col1, col2, col3, col4 = st.columns(4)
    
    today = datetime.now().date()
    tomorrow = today + timedelta(days=1)
    day_after = today + timedelta(days=2)
    in_3_days = today + timedelta(days=3)
    
    with col1:
        if st.button("📅 Today", use_container_width=True):
            st.session_state['selected_date'] = today
    with col2:
        if st.button("📅 Tomorrow", use_container_width=True, type="primary"):
            st.session_state['selected_date'] = tomorrow
    with col3:
        if st.button("📅 Day After", use_container_width=True):
            st.session_state['selected_date'] = day_after
    with col4:
        if st.button("📅 In 3 Days", use_container_width=True):
            st.session_state['selected_date'] = in_3_days
    
    # Date picker
    selected_date = st.date_input(
        "Or select a specific date:",
        value=st.session_state.get('selected_date', today)
    )
    st.session_state['selected_date'] = selected_date
    
    # Fetch weather
    is_future = selected_date > today
    
    if is_future:
        weather, source = fetch_weather_forecast(
            location['latitude'], location['longitude'], 
            OPENWEATHER_API_KEY, datetime.combine(selected_date, datetime.min.time())
        )
        st.info(f"📡 Using forecast data ({source})")
    else:
        weather = fetch_weather_data(location['latitude'], location['longitude'], OPENWEATHER_API_KEY)
        source = "Current Weather"
    
    # Show weather
    st.markdown("---")
    st.subheader(f"🌤️ Weather for {selected_date.strftime('%B %d, %Y')}")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🌡️ Temperature", f"{weather['tavg']:.1f}°C")
    col2.metric("💧 Humidity", f"{weather['humidity']:.0f}%")
    col3.metric("🌧️ Precipitation", f"{weather['prcp']:.1f} mm")
    col4.metric("💨 Wind Speed", f"{weather['wspd']:.1f} km/h")
    
    # Manual adjustments
    with st.expander("⚙️ Adjust Weather Parameters"):
        adj_col1, adj_col2 = st.columns(2)
        with adj_col1:
            weather['prcp'] = st.slider("Precipitation (mm)", 0.0, 100.0, float(weather['prcp']))
            weather['tavg'] = st.slider("Temperature (°C)", -5.0, 45.0, float(weather['tavg']))
        with adj_col2:
            weather['humidity'] = st.slider("Humidity (%)", 20.0, 100.0, float(weather['humidity']))
            weather['wspd'] = st.slider("Wind Speed (km/h)", 0.0, 60.0, float(weather['wspd']))
    
    # Prediction
    st.markdown("---")
    if st.button("🎯 Predict Flood Risk", type="primary", use_container_width=True):
        features = create_features(weather, location, datetime.combine(selected_date, datetime.min.time()))
        prob, is_flood = make_prediction(model_data, features)
        
        if prob is not None:
            st.subheader("🎯 Prediction Result")
            
            col1, col2 = st.columns(2)
            with col1:
                if is_flood:
                    st.error(f"## 🔴 HIGH FLOOD RISK")
                    st.warning("⚠️ **Recommendation:** Monitor weather closely and prepare for potential flooding.")
                else:
                    st.success(f"## 🟢 LOW FLOOD RISK")
                    st.info("✅ Conditions appear safe, but stay informed.")
                
                st.metric("Risk Probability", f"{prob*100:.1f}%")
            
            with col2:
                if PLOTLY_AVAILABLE:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prob * 100,
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkred" if is_flood else "darkgreen"},
                            'steps': [
                                {'range': [0, 30], 'color': "lightgreen"},
                                {'range': [30, 60], 'color': "yellow"},
                                {'range': [60, 100], 'color': "lightcoral"}
                            ]
                        }
                    ))
                    fig.update_layout(height=300, margin=dict(t=20, b=20))
                    st.plotly_chart(fig, use_container_width=True)
            
            # Tomorrow's summary
            if selected_date == tomorrow:
                st.markdown("---")
                st.subheader("📋 Tomorrow's Flood Prediction Summary")
                summary = f"""
                **Date:** {tomorrow.strftime('%A, %B %d, %Y')}  
                **Location:** {location['name']}  
                **Risk Level:** {"🔴 HIGH" if is_flood else "🟢 LOW"}  
                **Probability:** {prob*100:.1f}%
                
                **Weather Conditions:**
                - Temperature: {weather['tavg']:.1f}°C
                - Precipitation: {weather['prcp']:.1f} mm
                - Humidity: {weather['humidity']:.0f}%
                - Wind Speed: {weather['wspd']:.1f} km/h
                """
                st.markdown(summary)
        else:
            st.error("Model not available")


def show_model_info(model_data):
    """Display model information"""
    st.title("🤖 Model Information")
    
    # Model metrics
    metrics_path = RESULTS_DIR / "improved_model_metrics.csv"
    if metrics_path.exists():
        st.subheader("📊 Model Performance")
        metrics_df = pd.read_csv(metrics_path)
        st.dataframe(metrics_df, use_container_width=True)
    
    # Current model
    if model_data:
        st.subheader("🏆 Current Model")
        if isinstance(model_data, dict):
            st.write(f"**Model:** {model_data.get('model_name', 'Unknown')}")
            st.write(f"**Threshold:** {model_data.get('threshold', 0.5):.4f}")
    
    # AI Techniques used
    st.subheader("🧠 AI Techniques Implemented")
    techniques = {
        "Search Algorithms (Week 8)": "A*, BFS, DFS for evacuation route planning",
        "CSP (Week 9)": "Constraint Satisfaction for resource allocation",
        "Neural Networks (Week 11)": "LSTM for time-series prediction",
        "K-Means (Week 12)": "Clustering for flood pattern analysis",
        "Reinforcement Learning (Week 12)": "Q-Learning for evacuation decisions",
        "Explainability (Bonus)": "SHAP & LIME for model interpretation"
    }
    
    for tech, desc in techniques.items():
        st.markdown(f"- **{tech}**: {desc}")


def show_about():
    """About page"""
    st.title("ℹ️ About This Project")
    
    st.markdown("""
    ## 🌊 AI-Based Flood Prediction System
    
    This application predicts flood risk in **Khyber Pakhtunkhwa (KP), Pakistan** 
    using machine learning and multiple AI techniques.
    
    ### 📊 Key Statistics
    - **Records:** 18,902 weather observations
    - **Time Range:** 2000 - 2025
    - **Flood Events:** 517 labeled events (2.74%)
    - **Features:** 24 engineered features
    
    ### 🏆 Model Performance
    - **Best Model:** Logistic Regression
    - **Recall:** 60% (optimized for safety)
    
    ### 🔬 Technologies
    - **Frontend:** Streamlit
    - **ML:** scikit-learn, Random Forest, Gradient Boosting
    - **Data:** NASA POWER, Meteostat, NDMA Reports
    
    ---
    **CS351 - Artificial Intelligence | Semester 5**
    """)


# ============== MAIN ==============

def main():
    """Main application"""
    st.sidebar.title("🌊 Flood Prediction")
    st.sidebar.markdown("---")
    
    # Location selection
    selected_location = st.sidebar.selectbox(
        "📍 Select Location",
        options=list(LOCATIONS.keys()),
        format_func=lambda x: LOCATIONS[x]["name"]
    )
    location = LOCATIONS[selected_location]
    
    # Navigation
    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Dashboard", "🔮 Predict Flood", "🤖 Model Info", "ℹ️ About"]
    )
    
    # Load model
    model_data = load_model()
    
    # Route pages
    if page == "🏠 Dashboard":
        show_dashboard(location, model_data)
    elif page == "🔮 Predict Flood":
        show_custom_prediction(location, model_data)
    elif page == "🤖 Model Info":
        show_model_info(model_data)
    else:
        show_about()


if __name__ == "__main__":
    main()
