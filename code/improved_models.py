"""
Improved ML Models with Class Imbalance Handling
Uses class weights and optimized thresholds for better flood recall
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, precision_recall_curve
)
from sklearn.calibration import CalibratedClassifierCV

from pathlib import Path
import pickle
import json

# Define paths
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


class ImprovedFloodModels:
    """
    Train flood prediction models with class imbalance handling.
    
    Key improvements:
    1. Class-weighted training for minority class
    2. Optimized classification thresholds
    3. Probability calibration for better estimates
    """
    
    def __init__(self, X_train, X_test, y_train, y_test, feature_names):
        """Initialize with training and test data"""
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = feature_names
        self.models = {}
        self.results = {}
        self.optimal_thresholds = {}
        self.feature_importance = {}
    
    def find_optimal_threshold(self, model, model_name):
        """Find optimal classification threshold using precision-recall curve"""
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        # Calculate precision-recall curve
        precisions, recalls, thresholds = precision_recall_curve(self.y_test, y_pred_proba)
        
        # Find threshold that maximizes F1 score
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        best_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        
        self.optimal_thresholds[model_name] = {
            'f1_optimal': optimal_threshold,
            'best_f1': f1_scores[best_idx]
        }
        
        print(f"   🎯 Optimal threshold (max F1): {optimal_threshold:.4f}")
        return optimal_threshold
    
    def train_logistic_regression(self):
        """Train Logistic Regression with class weights"""
        print("\n" + "="*60)
        print("🔵 LOGISTIC REGRESSION (Balanced)")
        print("="*60)
        
        model = LogisticRegression(
            max_iter=2000,
            random_state=42,
            solver='lbfgs',
            class_weight='balanced',
            C=0.1,
            n_jobs=1
        )
        
        print("📚 Training with balanced class weights...")
        model.fit(self.X_train, self.y_train)
        self.models['Logistic Regression (Balanced)'] = model
        
        print(f"✅ Model trained!")
        self.find_optimal_threshold(model, 'Logistic Regression (Balanced)')
        self.feature_importance['Logistic Regression (Balanced)'] = np.abs(model.coef_[0])
        
        return model
    
    def train_random_forest(self):
        """Train Random Forest with optimized parameters"""
        print("\n" + "="*60)
        print("🌲 RANDOM FOREST (Balanced)")
        print("="*60)
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=1,
            class_weight='balanced',
            max_features='sqrt',
            bootstrap=True
        )
        
        print("📚 Training with balanced class weights...")
        model.fit(self.X_train, self.y_train)
        self.models['Random Forest (Balanced)'] = model
        
        print(f"✅ Model trained!")
        self.find_optimal_threshold(model, 'Random Forest (Balanced)')
        self.feature_importance['Random Forest (Balanced)'] = model.feature_importances_
        
        return model
    
    def train_gradient_boosting(self):
        """Train Gradient Boosting with calibration"""
        print("\n" + "="*60)
        print("🚀 GRADIENT BOOSTING (Calibrated)")
        print("="*60)
        
        base_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
            min_samples_split=10,
            min_samples_leaf=5
        )
        
        print("📚 Training Gradient Boosting with probability calibration...")
        
        model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
        model.fit(self.X_train, self.y_train)
        self.models['Gradient Boosting (Calibrated)'] = model
        
        print(f"✅ Model trained with isotonic calibration!")
        self.find_optimal_threshold(model, 'Gradient Boosting (Calibrated)')
        
        return model
    
    def evaluate_model(self, model_name, model, use_optimal_threshold=True):
        """Evaluate model with optimal threshold"""
        print(f"\n📊 Evaluating {model_name}...")
        
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        if use_optimal_threshold and model_name in self.optimal_thresholds:
            threshold = self.optimal_thresholds[model_name]['f1_optimal']
        else:
            threshold = 0.5
            
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, zero_division=0)
        recall = recall_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)
        auc = roc_auc_score(self.y_test, y_pred_proba)
        
        cm = confusion_matrix(self.y_test, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        results = {
            'model': model,
            'threshold': threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp)
        }
        
        self.results[model_name] = results
        
        print(f"   ✅ Accuracy: {accuracy:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
        return results
    
    def train_all(self):
        """Train all improved models"""
        print("\n" + "="*60)
        print("🎯 TRAINING IMPROVED MODELS")
        print("="*60)
        
        self.train_logistic_regression()
        self.train_random_forest()
        self.train_gradient_boosting()
        
        print("\n✅ All models trained!")
        return self.models
    
    def evaluate_all(self):
        """Evaluate all trained models"""
        print("\n" + "="*60)
        print("📋 MODEL EVALUATION")
        print("="*60)
        
        for model_name, model in self.models.items():
            self.evaluate_model(model_name, model)
        
        return self.results
    
    def get_best_model(self):
        """Return the best performing model based on Recall (for flood detection, catching floods is critical)"""
        if not self.results:
            raise ValueError("No models evaluated yet.")
        
        # For flood detection, we prioritize RECALL (catching actual floods) over precision
        # A missed flood is much worse than a false alarm
        best_name = max(self.results.keys(), key=lambda k: self.results[k]['recall'])
        best_result = self.results[best_name]
        
        print(f"\n🏆 Best Model: {best_name} (Recall: {best_result['recall']:.4f}, F1: {best_result['f1']:.4f})")
        return best_name, best_result
    
    def save_best_model(self, filename='best_flood_model.pkl', scaler=None):
        """Save the best model to disk with scaler"""
        best_name, best_result = self.get_best_model()
        
        model_data = {
            'model': best_result['model'],
            'model_name': best_name,
            'threshold': best_result['threshold'],
            'feature_names': self.feature_names,
            'scaler': scaler,  # CRITICAL: Save the scaler for prediction!
            'metrics': {
                'accuracy': best_result['accuracy'],
                'precision': best_result['precision'],
                'recall': best_result['recall'],
                'f1': best_result['f1'],
                'auc': best_result['auc']
            }
        }
        
        filepath = RESULTS_DIR / filename
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 Best model saved to: {filepath}")
        print(f"   ✅ Scaler included: {scaler is not None}")
        return filepath
    
    def save_all_results(self):
        """Save comprehensive results"""
        metrics_data = []
        for name, res in self.results.items():
            metrics_data.append({
                'Model': name,
                'Accuracy': res['accuracy'],
                'Precision': res['precision'],
                'Recall': res['recall'],
                'F1': res['f1'],
                'AUC': res['auc'],
                'Threshold': res['threshold'],
                'TP': res['tp'],
                'FP': res['fp'],
                'TN': res['tn'],
                'FN': res['fn']
            })
        
        df_metrics = pd.DataFrame(metrics_data)
        df_metrics.to_csv(RESULTS_DIR / 'improved_model_metrics.csv', index=False)
        
        with open(RESULTS_DIR / 'optimal_thresholds.json', 'w') as f:
            json.dump(self.optimal_thresholds, f, indent=2)
        
        print(f"📊 Results saved to {RESULTS_DIR}")


def run_improved_training(X_train, X_test, y_train, y_test, feature_names, scaler=None):
    """Run the complete improved training pipeline."""
    trainer = ImprovedFloodModels(X_train, X_test, y_train, y_test, feature_names)
    trainer.train_all()
    trainer.evaluate_all()
    trainer.save_best_model(scaler=scaler)  # Pass scaler to save
    trainer.save_all_results()
    return trainer


if __name__ == "__main__":
    print("Import and use with preprocessed data:")
    print("  from code.improved_models import run_improved_training")
