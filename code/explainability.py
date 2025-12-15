"""
Explainability Module - SHAP/LIME for Model Interpretation

"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class SHAPExplainer:
    """
    SHAP (SHapley Additive exPlanations) implementation for model explainability.
    
    Computes feature importance using Shapley values from cooperative game theory.
    Each feature's contribution is calculated by comparing model output with and
    without that feature across all possible combinations.
    
    For computational efficiency, this implementation uses:
    - Kernel SHAP approximation for complex models
    - Sampling-based estimation for large feature sets
    """
    
    def __init__(self, model: Any, feature_names: List[str] = None):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained model with predict/predict_proba method
            feature_names: Names of features
        """
        self.model = model
        self.feature_names = feature_names
        self.background_data = None
        self.expected_value = None
        
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Get model predictions"""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)[:, 1]
        else:
            return self.model.predict(X)
    
    def fit(self, background_data: np.ndarray, n_samples: int = 100):
        """
        Fit explainer with background data.
        
        Args:
            background_data: Reference data for computing expected values
            n_samples: Number of samples to use from background
        """
        # Sample background data if too large
        if len(background_data) > n_samples:
            indices = np.random.choice(len(background_data), n_samples, replace=False)
            self.background_data = background_data[indices]
        else:
            self.background_data = background_data
        
        # Compute expected value (average prediction on background)
        self.expected_value = np.mean(self._predict(self.background_data))
        
        if self.feature_names is None:
            self.feature_names = [f"Feature_{i}" for i in range(background_data.shape[1])]
    
    def _kernel_shap(self, x: np.ndarray, n_samples: int = 100) -> np.ndarray:
        """
        Kernel SHAP approximation for single instance.
        
        Uses weighted linear regression on sampled coalitions to estimate
        Shapley values efficiently.
        
        Args:
            x: Single instance to explain (1, n_features)
            n_samples: Number of coalition samples
            
        Returns:
            shap_values: Shapley values for each feature
        """
        n_features = len(x)
        
        # Generate coalition samples
        # Each row is a binary mask indicating which features are included
        coalitions = np.random.randint(0, 2, size=(n_samples, n_features))
        
        # Ensure we have some diversity
        coalitions[0] = np.zeros(n_features)  # Empty coalition
        coalitions[1] = np.ones(n_features)   # Full coalition
        
        # Compute predictions for each coalition
        predictions = []
        weights = []
        
        for coalition in coalitions:
            n_included = int(coalition.sum())
            
            # Create masked instance
            masked_x = np.zeros_like(x)
            for i in range(n_features):
                if coalition[i] == 1:
                    masked_x[i] = x[i]
                else:
                    # Use background mean for missing features
                    masked_x[i] = self.background_data[:, i].mean()
            
            # Get prediction
            pred = self._predict(masked_x.reshape(1, -1))[0]
            predictions.append(pred)
            
            # Compute SHAP kernel weight
            if n_included == 0 or n_included == n_features:
                weight = 1e6  # Large weight for boundary cases
            else:
                # SHAP kernel weight
                weight = (n_features - 1) / (
                    np.math.comb(n_features, n_included) * 
                    n_included * (n_features - n_included)
                )
            weights.append(weight)
        
        predictions = np.array(predictions)
        weights = np.array(weights)
        
        # Weighted least squares to estimate Shapley values
        # y = expected_value + sum(shap_i * z_i)
        # Where z_i indicates if feature i is in coalition
        
        y = predictions - self.expected_value
        
        # Add regularization for numerical stability
        reg = 1e-6
        
        # Solve weighted least squares
        W = np.diag(weights)
        X = coalitions
        
        try:
            # (X'WX + λI)^(-1) X'Wy
            XtWX = X.T @ W @ X + reg * np.eye(n_features)
            XtWy = X.T @ W @ y
            shap_values = np.linalg.solve(XtWX, XtWy)
        except np.linalg.LinAlgError:
            # Fallback to simple gradient-based approximation
            shap_values = self._gradient_approximation(x)
        
        return shap_values
    
    def _gradient_approximation(self, x: np.ndarray) -> np.ndarray:
        """
        Simple gradient-based approximation for Shapley values.
        
        Computes feature importance by measuring prediction change
        when each feature is removed.
        """
        n_features = len(x)
        shap_values = np.zeros(n_features)
        
        base_pred = self._predict(x.reshape(1, -1))[0]
        
        for i in range(n_features):
            # Create masked version
            x_masked = x.copy()
            x_masked[i] = self.background_data[:, i].mean()
            
            masked_pred = self._predict(x_masked.reshape(1, -1))[0]
            shap_values[i] = base_pred - masked_pred
        
        return shap_values
    
    def explain_instance(self, x: np.ndarray, n_samples: int = 200) -> Dict:
        """
        Explain prediction for single instance.
        
        Args:
            x: Instance to explain (n_features,)
            n_samples: Number of samples for approximation
            
        Returns:
            explanation: Dictionary with SHAP values and analysis
        """
        if self.background_data is None:
            raise ValueError("Explainer not fitted. Call fit() first.")
        
        x = np.asarray(x).flatten()
        shap_values = self._kernel_shap(x, n_samples)
        
        # Get prediction
        prediction = self._predict(x.reshape(1, -1))[0]
        
        # Create feature importance ranking
        importance_order = np.argsort(np.abs(shap_values))[::-1]
        
        feature_contributions = []
        for idx in importance_order:
            feature_contributions.append({
                'feature': self.feature_names[idx],
                'value': float(x[idx]),
                'shap_value': float(shap_values[idx]),
                'contribution': 'positive' if shap_values[idx] > 0 else 'negative'
            })
        
        return {
            'prediction': float(prediction),
            'expected_value': float(self.expected_value),
            'shap_values': shap_values.tolist(),
            'feature_names': self.feature_names,
            'feature_contributions': feature_contributions,
            'prediction_explanation': self._generate_explanation(x, shap_values, prediction)
        }
    
    def _generate_explanation(self, x: np.ndarray, shap_values: np.ndarray, 
                             prediction: float) -> str:
        """Generate human-readable explanation"""
        
        # Get top contributors
        top_positive_idx = np.argsort(shap_values)[-3:][::-1]
        top_negative_idx = np.argsort(shap_values)[:3]
        
        explanation = f"Prediction: {prediction:.3f} (Base: {self.expected_value:.3f})\n\n"
        
        explanation += "Top factors INCREASING prediction:\n"
        for idx in top_positive_idx:
            if shap_values[idx] > 0:
                explanation += f"  • {self.feature_names[idx]}: {x[idx]:.2f} (+{shap_values[idx]:.3f})\n"
        
        explanation += "\nTop factors DECREASING prediction:\n"
        for idx in top_negative_idx:
            if shap_values[idx] < 0:
                explanation += f"  • {self.feature_names[idx]}: {x[idx]:.2f} ({shap_values[idx]:.3f})\n"
        
        return explanation
    
    def explain_batch(self, X: np.ndarray, n_samples: int = 100) -> Dict:
        """
        Explain predictions for multiple instances.
        
        Args:
            X: Instances to explain (n_instances, n_features)
            n_samples: Samples per instance
            
        Returns:
            explanations: Aggregated SHAP analysis
        """
        all_shap_values = []
        
        for i, x in enumerate(X):
            shap_values = self._kernel_shap(x, n_samples)
            all_shap_values.append(shap_values)
            
            if (i + 1) % 10 == 0:
                print(f"Explained {i + 1}/{len(X)} instances")
        
        all_shap_values = np.array(all_shap_values)
        
        # Compute global feature importance (mean absolute SHAP)
        global_importance = np.mean(np.abs(all_shap_values), axis=0)
        
        feature_importance = []
        for idx in np.argsort(global_importance)[::-1]:
            feature_importance.append({
                'feature': self.feature_names[idx],
                'importance': float(global_importance[idx]),
                'mean_shap': float(np.mean(all_shap_values[:, idx])),
                'std_shap': float(np.std(all_shap_values[:, idx]))
            })
        
        return {
            'shap_values': all_shap_values.tolist(),
            'feature_importance': feature_importance,
            'expected_value': float(self.expected_value)
        }


class LIMEExplainer:
    """
    LIME (Local Interpretable Model-agnostic Explanations) implementation.
    
    Explains predictions by fitting a simple interpretable model (linear)
    locally around the instance being explained.
    """
    
    def __init__(self, model: Any, feature_names: List[str] = None):
        """
        Initialize LIME explainer.
        
        Args:
            model: Trained model with predict method
            feature_names: Names of features
        """
        self.model = model
        self.feature_names = feature_names
        self.kernel_width = None
        
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Get model predictions"""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)[:, 1]
        else:
            return self.model.predict(X)
    
    def _generate_perturbations(self, x: np.ndarray, n_samples: int, 
                               std_multiplier: float = 0.1) -> np.ndarray:
        """
        Generate perturbed samples around instance.
        
        Args:
            x: Original instance
            n_samples: Number of perturbations
            std_multiplier: Controls perturbation magnitude
            
        Returns:
            perturbations: Perturbed samples
        """
        n_features = len(x)
        
        # Generate Gaussian perturbations
        noise = np.random.randn(n_samples, n_features) * std_multiplier
        perturbations = x + noise * np.abs(x + 1e-10)  # Scale by feature magnitude
        
        # Include original instance
        perturbations[0] = x
        
        return perturbations
    
    def _exponential_kernel(self, d: np.ndarray, width: float) -> np.ndarray:
        """Exponential kernel for weighting perturbations"""
        return np.sqrt(np.exp(-(d ** 2) / (width ** 2)))
    
    def _compute_distances(self, x: np.ndarray, 
                          perturbations: np.ndarray) -> np.ndarray:
        """Compute distances from original to perturbations"""
        return np.sqrt(np.sum((perturbations - x) ** 2, axis=1))
    
    def explain_instance(self, x: np.ndarray, n_samples: int = 500,
                        num_features: int = 10) -> Dict:
        """
        Explain prediction for single instance using LIME.
        
        Args:
            x: Instance to explain
            n_samples: Number of perturbations to generate
            num_features: Number of top features to include
            
        Returns:
            explanation: Dictionary with LIME explanation
        """
        x = np.asarray(x).flatten()
        n_features = len(x)
        
        if self.feature_names is None:
            self.feature_names = [f"Feature_{i}" for i in range(n_features)]
        
        # Set kernel width based on feature dimension
        if self.kernel_width is None:
            self.kernel_width = np.sqrt(n_features) * 0.75
        
        # Generate perturbations
        perturbations = self._generate_perturbations(x, n_samples)
        
        # Get model predictions for perturbations
        predictions = self._predict(perturbations)
        
        # Compute distances and weights
        distances = self._compute_distances(x, perturbations)
        weights = self._exponential_kernel(distances, self.kernel_width)
        
        # Fit weighted linear model (interpretable surrogate)
        # Using weighted least squares: (X'WX)^(-1) X'Wy
        
        # Normalize perturbations for stability
        X_mean = perturbations.mean(axis=0)
        X_std = perturbations.std(axis=0) + 1e-10
        X_norm = (perturbations - X_mean) / X_std
        
        W = np.diag(weights)
        
        try:
            # Add regularization
            reg = 1e-6 * np.eye(n_features)
            XtWX = X_norm.T @ W @ X_norm + reg
            XtWy = X_norm.T @ W @ predictions
            coefficients = np.linalg.solve(XtWX, XtWy)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse
            coefficients = np.linalg.lstsq(X_norm, predictions, rcond=None)[0]
        
        # Scale coefficients back
        coefficients = coefficients / X_std
        
        # Get original prediction
        original_pred = self._predict(x.reshape(1, -1))[0]
        
        # Rank features by absolute coefficient
        importance_order = np.argsort(np.abs(coefficients))[::-1][:num_features]
        
        feature_contributions = []
        for idx in importance_order:
            feature_contributions.append({
                'feature': self.feature_names[idx],
                'coefficient': float(coefficients[idx]),
                'value': float(x[idx]),
                'contribution': float(coefficients[idx] * x[idx]),
                'direction': 'positive' if coefficients[idx] > 0 else 'negative'
            })
        
        # Calculate local accuracy (how well surrogate fits locally)
        surrogate_preds = X_norm @ coefficients * X_std + X_mean @ coefficients
        weighted_mse = np.average((predictions - surrogate_preds) ** 2, weights=weights)
        local_accuracy = 1 - weighted_mse / (np.var(predictions) + 1e-10)
        
        return {
            'prediction': float(original_pred),
            'coefficients': coefficients.tolist(),
            'feature_contributions': feature_contributions,
            'local_accuracy': float(max(0, local_accuracy)),
            'explanation_text': self._generate_explanation(x, coefficients, original_pred)
        }
    
    def _generate_explanation(self, x: np.ndarray, coefficients: np.ndarray,
                             prediction: float) -> str:
        """Generate human-readable LIME explanation"""
        
        explanation = f"LIME Local Explanation (Prediction: {prediction:.3f})\n"
        explanation += "=" * 50 + "\n\n"
        
        # Get top features
        top_idx = np.argsort(np.abs(coefficients))[-5:][::-1]
        
        explanation += "Most important features locally:\n"
        for rank, idx in enumerate(top_idx, 1):
            direction = "↑" if coefficients[idx] > 0 else "↓"
            explanation += f"  {rank}. {self.feature_names[idx]}: {direction} {np.abs(coefficients[idx]):.4f}\n"
            explanation += f"     Current value: {x[idx]:.2f}\n"
        
        return explanation


class FloodPredictionExplainer:
    """
    Combined explainability system for flood prediction models.
    Provides both SHAP and LIME explanations with flood-specific interpretations.
    """
    
    def __init__(self, model: Any, feature_names: List[str]):
        """
        Initialize flood prediction explainer.
        
        Args:
            model: Trained flood prediction model
            feature_names: Names of weather features
        """
        self.model = model
        self.feature_names = feature_names
        self.shap_explainer = SHAPExplainer(model, feature_names)
        self.lime_explainer = LIMEExplainer(model, feature_names)
        
    def fit(self, training_data: np.ndarray):
        """Fit explainers with training data"""
        self.shap_explainer.fit(training_data)
        
    def explain_prediction(self, x: np.ndarray, method: str = 'both') -> Dict:
        """
        Explain flood prediction.
        
        Args:
            x: Weather features for prediction
            method: 'shap', 'lime', or 'both'
            
        Returns:
            explanation: Comprehensive explanation
        """
        x = np.asarray(x).flatten()
        
        result = {
            'input_features': dict(zip(self.feature_names, x.tolist()))
        }
        
        if method in ['shap', 'both']:
            result['shap'] = self.shap_explainer.explain_instance(x)
        
        if method in ['lime', 'both']:
            result['lime'] = self.lime_explainer.explain_instance(x)
        
        # Add flood-specific interpretation
        result['flood_interpretation'] = self._interpret_for_flood(x, result)
        
        return result
    
    def _interpret_for_flood(self, x: np.ndarray, explanations: Dict) -> str:
        """Generate flood-specific interpretation"""
        
        interpretation = "FLOOD RISK ASSESSMENT\n"
        interpretation += "=" * 40 + "\n\n"
        
        # Get prediction
        if hasattr(self.model, 'predict_proba'):
            prob = self.model.predict_proba(x.reshape(1, -1))[0, 1]
        else:
            prob = self.model.predict(x.reshape(1, -1))[0]
        
        # Risk level
        if prob >= 0.7:
            risk_level = "HIGH"
            interpretation += f"⚠️ HIGH FLOOD RISK ({prob*100:.1f}%)\n\n"
        elif prob >= 0.4:
            risk_level = "MODERATE"
            interpretation += f"⚡ MODERATE FLOOD RISK ({prob*100:.1f}%)\n\n"
        else:
            risk_level = "LOW"
            interpretation += f"✓ LOW FLOOD RISK ({prob*100:.1f}%)\n\n"
        
        # Explain key factors
        if 'shap' in explanations:
            shap_values = explanations['shap']['shap_values']
            
            # Find precipitation-related features
            high_risk_factors = []
            protective_factors = []
            
            for i, name in enumerate(self.feature_names):
                if shap_values[i] > 0.05:
                    high_risk_factors.append((name, x[i], shap_values[i]))
                elif shap_values[i] < -0.05:
                    protective_factors.append((name, x[i], shap_values[i]))
            
            if high_risk_factors:
                interpretation += "Risk Increasing Factors:\n"
                for name, value, contrib in sorted(high_risk_factors, key=lambda x: -x[2])[:3]:
                    interpretation += f"  • {name}: {value:.2f} (+{contrib:.3f})\n"
            
            if protective_factors:
                interpretation += "\nRisk Decreasing Factors:\n"
                for name, value, contrib in sorted(protective_factors, key=lambda x: x[2])[:3]:
                    interpretation += f"  • {name}: {value:.2f} ({contrib:.3f})\n"
        
        # Recommendations
        interpretation += "\nRecommendations:\n"
        if risk_level == "HIGH":
            interpretation += "  • Monitor weather updates closely\n"
            interpretation += "  • Prepare emergency supplies\n"
            interpretation += "  • Know evacuation routes\n"
        elif risk_level == "MODERATE":
            interpretation += "  • Stay informed about weather conditions\n"
            interpretation += "  • Review emergency plans\n"
        else:
            interpretation += "  • Continue normal activities\n"
            interpretation += "  • Stay aware of weather forecasts\n"
        
        return interpretation
    
    def get_global_importance(self, X: np.ndarray, n_samples: int = 100) -> Dict:
        """
        Compute global feature importance across dataset.
        
        Args:
            X: Dataset to explain
            n_samples: Number of samples to use
            
        Returns:
            importance: Global feature importance rankings
        """
        # Sample if dataset is large
        if len(X) > n_samples:
            indices = np.random.choice(len(X), n_samples, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
        
        return self.shap_explainer.explain_batch(X_sample)


def demo_explainability():
    """Demonstrate SHAP and LIME explainability"""
    print("=" * 60)
    print("SHAP & LIME EXPLAINABILITY FOR FLOOD PREDICTION - Demo")
    print("=" * 60)
    
    # Create simple mock model for demonstration
    class MockFloodModel:
        """Simple model that predicts flood based on precipitation and humidity"""
        def predict_proba(self, X):
            X = np.atleast_2d(X)
            # Higher precipitation and humidity = higher flood risk
            probs = 0.3 * (X[:, 1] / 100) + 0.2 * (X[:, 2] / 100) + 0.1 * (X[:, 4] / 20)
            probs = np.clip(probs, 0, 1)
            return np.column_stack([1 - probs, probs])
        
        def predict(self, X):
            return (self.predict_proba(X)[:, 1] > 0.5).astype(int)
    
    model = MockFloodModel()
    feature_names = ['Temperature', 'Precipitation', 'Humidity', 'Pressure', 'Wind Speed']
    
    # Create training data
    np.random.seed(42)
    n_samples = 200
    X_train = np.column_stack([
        np.random.uniform(15, 40, n_samples),   # Temperature
        np.random.uniform(0, 100, n_samples),   # Precipitation
        np.random.uniform(30, 100, n_samples),  # Humidity
        np.random.uniform(980, 1030, n_samples),# Pressure
        np.random.uniform(0, 30, n_samples)     # Wind Speed
    ])
    
    # Create explainer
    explainer = FloodPredictionExplainer(model, feature_names)
    explainer.fit(X_train)
    
    # Explain a high-risk sample
    print("\n" + "=" * 60)
    print("HIGH RISK SAMPLE EXPLANATION")
    print("=" * 60)
    
    high_risk_sample = np.array([28, 80, 95, 995, 15])  # High rain, high humidity
    explanation = explainer.explain_prediction(high_risk_sample)
    
    print(f"\nSample: {dict(zip(feature_names, high_risk_sample))}")
    print("\n" + explanation['flood_interpretation'])
    
    # Explain a low-risk sample
    print("\n" + "=" * 60)
    print("LOW RISK SAMPLE EXPLANATION")
    print("=" * 60)
    
    low_risk_sample = np.array([25, 5, 40, 1015, 8])  # Low rain, low humidity
    explanation = explainer.explain_prediction(low_risk_sample)
    
    print(f"\nSample: {dict(zip(feature_names, low_risk_sample))}")
    print("\n" + explanation['flood_interpretation'])
    
    # Global importance
    print("\n" + "=" * 60)
    print("GLOBAL FEATURE IMPORTANCE")
    print("=" * 60)
    
    global_importance = explainer.get_global_importance(X_train, n_samples=50)
    
    print("\nFeature Rankings (by mean |SHAP|):")
    for i, feat in enumerate(global_importance['feature_importance'][:5], 1):
        print(f"  {i}. {feat['feature']}: {feat['importance']:.4f}")
    
    return explainer


if __name__ == "__main__":
    demo_explainability()
