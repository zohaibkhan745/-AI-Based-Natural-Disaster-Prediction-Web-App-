"""
Train Improved Models for Flood Prediction
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from code.preprocessing import DataPreprocessor
from code.improved_models import run_improved_training

def main():
    print("=" * 60)
    print("🌊 FLOOD PREDICTION MODEL TRAINING")
    print("=" * 60)
    
    data_path = PROJECT_ROOT / "data" / "processed" / "flood_weather_dataset.csv"
    
    if not data_path.exists():
        print(f"❌ Dataset not found at: {data_path}")
        return None
    
    # Initialize and run preprocessing
    preprocessor = DataPreprocessor(str(data_path))
    preprocessor.load_data()
    preprocessor.explore_data()
    preprocessor.handle_missing_values()
    preprocessor.feature_engineering()
    preprocessor.select_features()
    preprocessor.prepare_data(test_size=0.2)
    preprocessor.scale_features()
    
    # Get data
    X_train_scaled = preprocessor.X_train
    X_test_scaled = preprocessor.X_test
    y_train = preprocessor.y_train
    y_test = preprocessor.y_test
    
    print(f"\n📊 Training: {len(y_train)} | Test: {len(y_test)} | Features: {len(preprocessor.feature_names)}")
    
    # Train improved models - PASS SCALER for saving!
    trainer = run_improved_training(
        X_train_scaled, X_test_scaled, 
        y_train.values, y_test.values, 
        preprocessor.feature_names,
        scaler=preprocessor.scaler  # CRITICAL: Pass scaler to save!
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TRAINING COMPLETE")
    print("=" * 60)
    
    for name, res in trainer.results.items():
        print(f"{name}: Recall={res['recall']:.2%} | F1={res['f1']:.4f}")
    
    return trainer


if __name__ == "__main__":
    main()
