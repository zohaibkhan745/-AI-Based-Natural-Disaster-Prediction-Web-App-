# 🚀 Quick Start - Streamlit Web App

## Start the Application in 3 Steps

### Step 1: Navigate to Project
```bash
cd /Users/hussain/Documents/Projects/-AI-Based-Natural-Disaster-Prediction-Web-App-
```

### Step 2: Run the App (Choose One)

**Option A - Using the Startup Script (Easiest):**
```bash
./run_app.sh
```

**Option B - Direct Command:**
```bash
.venv/bin/python3 -m streamlit run app.py
```

**Option C - With Virtual Environment:**
```bash
source .venv/bin/activate
streamlit run app.py
```

### Step 3: Open in Browser
```
http://localhost:8501
```

---

## 📱 What You'll See

✅ **Home Page** - Project overview and statistics  
✅ **Make Prediction** - 3 ways to get flood risk predictions  
✅ **Model Performance** - View metrics and visualizations  
✅ **Data Analysis** - Explore dataset statistics  
✅ **About** - Project details and documentation  

---

## 🎯 Key Features

### 1️⃣ Manual Weather Input
- Adjust 19 weather features with sliders
- Select location (Swat/Upper Dir)
- Get instant predictions from both models
- See probability scores

### 2️⃣ Test with Sample Data
- Pick from 1,151 test samples
- See actual vs predicted results
- Compare model accuracy

### 3️⃣ Generate Random Data
- Test models with random weather
- Check robustness

### 4️⃣ View Metrics & Charts
- Performance comparison
- ROC curves
- Confusion matrices
- Feature importance

---

## 📊 Risk Levels

| Level | Probability | Color |
|-------|-------------|-------|
| 🟢 LOW | < 30% | Green |
| 🟡 MEDIUM | 30-60% | Yellow |
| 🔴 HIGH | > 60% | Red |

---

## ❌ Troubleshooting

**"Port already in use?"**
```bash
streamlit run app.py --server.port 8502
```

**"Streamlit not found?"**
```bash
source .venv/bin/activate
pip install streamlit
```

**"Models not found?"**
```bash
# Run training first
.venv/bin/python3 run_pipeline.py
```

---

## 📖 Full Documentation

For complete details, see:
- `STREAMLIT_GUIDE.md` - Full guide with all features
- `README.md` - Project overview
- `ML_PIPELINE_README.md` - Model details

---

**Ready? Start now with:** `./run_app.sh`
