# 🌊 Streamlit Web Application Guide

## Quick Start

### Option 1: Using the Startup Script (Recommended)

```bash
# Navigate to project directory
cd /Users/hussain/Documents/Projects/-AI-Based-Natural-Disaster-Prediction-Web-App-

# Make script executable
chmod +x run_app.sh

# Run the app
./run_app.sh
```

### Option 2: Direct Command

```bash
# Activate virtual environment
source .venv/bin/activate

# Run streamlit app
streamlit run app.py
```

### Option 3: Using Python Virtual Environment

```bash
# Navigate to project directory
cd /Users/hussain/Documents/Projects/-AI-Based-Natural-Disaster-Prediction-Web-App-

# Run directly with venv Python
.venv/bin/python3 -m streamlit run app.py
```

---

## 🌐 Access the Application

Once the app is running, open your browser and go to:

```
http://localhost:8501
```

You should see the Streamlit dashboard load automatically.

---

## 📊 Application Features

### 1. 🏠 Home Page
- Quick start guide
- System statistics
- Available models
- Coverage areas

### 2. 🎯 Make Prediction
Three tabs for different input methods:

#### Tab 1: Manual Input
- Adjust 19 weather features using sliders
- Set location (Swat or Upper Dir)
- Get real-time predictions from both models
- See probabilities and risk levels

#### Tab 2: Sample Data
- Choose from 1,151 test samples
- View actual results
- Compare model predictions with actual outcomes
- See prediction accuracy

#### Tab 3: Random Data
- Generate random weather data
- Test model robustness
- Get instant predictions

### 3. 📈 Model Performance
- View performance metrics (Accuracy, AUC-ROC, etc.)
- Compare Logistic Regression vs Random Forest
- View visualizations:
  - Model Performance Comparison
  - ROC Curves
  - Confusion Matrices
  - Feature Importance

### 4. 📊 Data Analysis
- View dataset statistics
- Select individual features
- See distribution analysis
- Correlation matrix visualization

### 5. ℹ️ About
- Project overview
- Technology stack
- Project structure
- Developer information
- License and support

---

## 🎨 Features of the Web App

### ✨ Interactive UI
- Clean, modern interface with gradient backgrounds
- Color-coded risk levels (Green/Yellow/Red)
- Responsive layout for different screen sizes
- Easy-to-use navigation

### 📊 Real-time Predictions
- Instant flood risk predictions
- Probability scores for each outcome
- Confidence levels displayed
- Dual model comparison

### 💾 Data Caching
- Models cached for fast predictions
- Test data loaded once
- Smooth user experience

### 🎯 Multiple Input Methods
- Manual weather data entry
- Pre-loaded sample data from test set
- Random data generation for testing

### 📈 Comprehensive Analytics
- Performance metrics dashboard
- Visual comparisons
- Distribution analysis
- Feature correlations

---

## 🔑 Key Inputs Explained

### Weather Features (19 Total)

**Temperature Metrics:**
- `tavg` - Average Temperature (°C)
- `tmin` - Minimum Temperature (°C)
- `tmax` - Maximum Temperature (°C)
- `temp_range` - Max - Min (Auto-calculated)

**Precipitation & Wind:**
- `prcp` - Precipitation (mm)
- `wspd` - Wind Speed (m/s)
- `wpgt` - Wind Gust (m/s)

**Atmospheric Conditions:**
- `pres` - Atmospheric Pressure (hPa)
- `humidity` - Humidity (%)
- `pressure_anomaly` - Deviation from mean pressure
- `high_humidity` - Flag if humidity above average

**Solar & Temporal:**
- `solar_radiation` - Solar Radiation (MJ/m²)
- `month` - Month (1-12)
- `day_of_year` - Day of year (1-365)
- `quarter` - Quarter (1-4)

**Rolling Averages (7-day):**
- `prcp_7day_avg` - 7-day Precipitation average
- `tavg_7day_avg` - 7-day Temperature average
- `wspd_7day_avg` - 7-day Wind Speed average

**Location:**
- `location_encoded` - Swat (-1.0) or Upper Dir (1.0)

---

## 📊 Understanding Predictions

### Risk Levels

```
🟢 LOW RISK (< 30% probability)
   - Safe conditions
   - No immediate flood threat
   - Normal weather patterns

🟡 MEDIUM RISK (30-60% probability)
   - Moderate rainfall/weather
   - Monitor weather conditions
   - Prepare for possible flooding

🔴 HIGH RISK (> 60% probability)
   - Severe weather conditions
   - Significant flood threat
   - Take precautionary measures
```

### Model Recommendations

**Logistic Regression:**
- Good baseline model
- Fast predictions
- Lower AUC-ROC (0.8243)

**Random Forest** ⭐ (Recommended)
- More accurate predictions
- Better discrimination ability
- Higher AUC-ROC (0.8643)
- Use this for critical decisions

---

## ⚙️ Installation & Requirements

### Prerequisites

```bash
# Python 3.9+
python3 --version

# Ensure virtual environment exists
source .venv/bin/activate

# Check if streamlit is installed
pip list | grep streamlit
```

### Install Streamlit

```bash
# If not already installed
pip install streamlit

# Or install from requirements
pip install -r requirements.txt
```

### Verify Installation

```bash
streamlit --version
```

---

## 🚀 Running the Application

### Step 1: Navigate to Project Directory
```bash
cd /Users/hussain/Documents/Projects/-AI-Based-Natural-Disaster-Prediction-Web-App-
```

### Step 2: Activate Virtual Environment
```bash
source .venv/bin/activate
```

### Step 3: Run Streamlit App
```bash
streamlit run app.py
```

### Step 4: Access in Browser
```
http://localhost:8501
```

---

## 🛠️ Troubleshooting

### Issue: "streamlit command not found"
**Solution:**
```bash
source .venv/bin/activate
pip install streamlit
```

### Issue: "Models not found"
**Solution:**
Make sure you've run the training pipeline first:
```bash
.venv/bin/python3 run_pipeline.py
```

### Issue: "Port 8501 already in use"
**Solution:**
```bash
# Run on different port
streamlit run app.py --server.port 8502
```

### Issue: "ModuleNotFoundError: No module named 'streamlit'"
**Solution:**
```bash
source .venv/bin/activate
pip install streamlit pandas matplotlib seaborn
```

---

## 📱 Deployment Options

### Local Deployment
```bash
streamlit run app.py
```

### Remote Deployment (Streamlit Cloud)
1. Push code to GitHub
2. Go to https://share.streamlit.io/
3. Deploy directly from GitHub repository

### Docker Deployment
Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["streamlit", "run", "app.py"]
```

Run:
```bash
docker build -t flood-predictor .
docker run -p 8501:8501 flood-predictor
```

---

## 📊 Application Structure

```
app.py (Main Streamlit App)
├── Configuration
├── Models & Data Loading
├── Utility Functions
├── Navigation (Sidebar)
├── Pages:
│   ├── 🏠 Home
│   ├── 🎯 Make Prediction
│   │   ├── Manual Input
│   │   ├── Sample Data
│   │   └── Random Data
│   ├── 📈 Model Performance
│   ├── 📊 Data Analysis
│   └── ℹ️ About
└── Footer
```

---

## 🎯 Use Cases

### For Users
- Check flood risk before travel planning
- Monitor weather conditions
- Make informed decisions about safety

### For Administrators
- Real-time flood risk monitoring
- Historical pattern analysis
- Emergency preparedness planning

### For Researchers
- Model performance analysis
- Data exploration
- Feature importance understanding

---

## 📚 Integration with Existing Code

✅ **No Changes to Existing Code**
- Original preprocessing.py - Unchanged
- Original baseline_models.py - Unchanged
- Original model_evaluation.py - Unchanged
- Trained models (.pkl files) - Used as-is

✅ **New Additions**
- app.py - Complete Streamlit interface
- run_app.sh - Startup script
- STREAMLIT_GUIDE.md - This documentation

---

## 🔒 Security Notes

- Keep sensitive data (API keys, credentials) in environment variables
- Use HTTPS in production
- Validate all user inputs
- Monitor application logs
- Set up proper authentication if needed

---

## 📞 Support & Documentation

For more information, refer to:
- `README.md` - Project overview
- `ENVIRONMENT_SETUP.md` - Environment configuration
- `ML_PIPELINE_README.md` - ML pipeline details
- `XGBOOST_ERROR_RESOLUTION.md` - Error handling

---

## ✨ Tips & Best Practices

1. **Model Performance**: Random Forest is more accurate - use for critical decisions
2. **Data Input**: Provide realistic weather values for accurate predictions
3. **Sample Testing**: Use the Sample Data tab to validate model accuracy
4. **Feature Analysis**: Check Data Analysis tab to understand feature distributions
5. **Regular Updates**: Retrain models with new weather data periodically

---

**Last Updated:** November 16, 2025  
**Version:** 1.0.0  
**Status:** ✅ Ready for Deployment
