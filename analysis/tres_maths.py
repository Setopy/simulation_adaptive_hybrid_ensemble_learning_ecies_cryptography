import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.metrics import r2_score

class IDSThresholdPredictor:
    def __init__(self):
        # Experimental data with exact values
        self.thresholds = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        
        # Actual experimental results
        self.detection_rates = np.array([57.59, 30.41, 40.60, 31.82, 26.13])
        self.total_events = np.array([6458, 26801, 11408, 16026, 20513])
        self.mean_probs = np.array([0.6589, 0.6700, 0.6680, 0.6709, 0.6699])
        
        # Model contributions from experiments
        self.model_contributions = {
            'CNN': np.array([0.1241, 0.1237, 0.1229, 0.1226, 0.1216]),
            'LSTM': np.array([0.0559, 0.0635, 0.0629, 0.0649, 0.0646]),
            'DNN': np.array([0.1500, 0.1500, 0.1500, 0.1500, 0.1500]),
            'SVM': np.array([0.0150, 0.0188, 0.0183, 0.0194, 0.0197]),
            'XGBoost': np.array([0.1860, 0.1860, 0.1860, 0.1860, 0.1860]),
            'RandomForest': np.array([0.1279, 0.1280, 0.1280, 0.1280, 0.1281])
        }
        
        self._fit_advanced_models()

    def _sigmoid(self, x, a, b, c, d):
        """Enhanced sigmoid function for better fitting"""
        return a / (1 + np.exp(-b * (x - c))) + d

    def _fit_advanced_models(self):
        """Fit sophisticated models to the experimental data"""
        # Detection rate fitting using optimization
        def rate_objective(params):
            a, b, c, d = params
            predicted = self._sigmoid(self.thresholds, a, b, c, d)
            return np.sum((predicted - self.detection_rates)**2)
        
        # Initial guess for sigmoid parameters
        initial_guess = [60, -5, 0.6, 20]
        self.rate_params = minimize(rate_objective, initial_guess).x
        
        # Event count fitting with polynomial and correction factor
        self.event_coeffs = np.polyfit(self.thresholds, self.total_events, 3)
        
        # Probability fitting with weighted polynomial
        self.prob_coeffs = np.polyfit(self.thresholds, self.mean_probs, 3)
        
        # Model contribution fitting
        self.contribution_models = {}
        for model in self.model_contributions:
            if np.all(self.model_contributions[model] == self.model_contributions[model][0]):
                # Constant models (DNN, XGBoost)
                self.contribution_models[model] = ('constant', self.model_contributions[model][0])
            else:
                # Variable models (CNN, LSTM, SVM, RandomForest)
                coeffs = np.polyfit(self.thresholds, self.model_contributions[model], 2)
                self.contribution_models[model] = ('polynomial', coeffs)

    def predict_threshold_performance(self, threshold, include_confidence=True):
        """Enhanced prediction with confidence intervals"""
        if not 0.5 <= threshold <= 0.9:
            print("Warning: Predictions may be less accurate outside [0.5, 0.9] range")
        
        # Detection rate prediction using sigmoid
        pred_rate = self._sigmoid(threshold, *self.rate_params)
        
        # Event count prediction with correction
        pred_events = np.polyval(self.event_coeffs, threshold)
        
        # Ensemble probability prediction
        pred_prob = np.polyval(self.prob_coeffs, threshold)
        
        # Model contribution predictions
        model_contributions = {}
        for model, (model_type, params) in self.contribution_models.items():
            if model_type == 'constant':
                model_contributions[model] = params
            else:
                model_contributions[model] = np.polyval(params, threshold)
        
        # Calculate confidence intervals if requested
        confidence_intervals = None
        if include_confidence:
            confidence_intervals = self._calculate_confidence_intervals(threshold)
        
        result = {
            'threshold': threshold,
            'detection_rate': max(0, min(100, pred_rate)),
            'total_events': max(0, int(pred_events)),
            'ensemble_probability': min(1, max(0, pred_prob)),
            'model_contributions': model_contributions,
        }
        
        if confidence_intervals:
            result['confidence_intervals'] = confidence_intervals
            
        return result

    def _calculate_confidence_intervals(self, threshold):
        """Calculate confidence intervals for predictions"""
        # Use local polynomial fitting for uncertainty estimation
        window = 0.1
        mask = np.abs(self.thresholds - threshold) <= window
        
        if sum(mask) < 2:
            return None
            
        local_thresholds = self.thresholds[mask]
        intervals = {}
        
        # Detection rate CI
        local_rates = self.detection_rates[mask]
        rate_std = np.std(local_rates)
        intervals['detection_rate'] = (
            max(0, self._sigmoid(threshold, *self.rate_params) - 1.96 * rate_std),
            min(100, self._sigmoid(threshold, *self.rate_params) + 1.96 * rate_std)
        )
        
        return intervals

    def evaluate_model_accuracy(self):
        """Evaluate model accuracy using R² scores"""
        accuracies = {}
        
        # Detection rate accuracy
        pred_rates = [self._sigmoid(t, *self.rate_params) for t in self.thresholds]
        accuracies['detection_rate'] = r2_score(self.detection_rates, pred_rates)
        
        # Event count accuracy
        pred_events = [np.polyval(self.event_coeffs, t) for t in self.thresholds]
        accuracies['total_events'] = r2_score(self.total_events, pred_events)
        
        # Probability accuracy
        pred_probs = [np.polyval(self.prob_coeffs, t) for t in self.thresholds]
        accuracies['ensemble_probability'] = r2_score(self.mean_probs, pred_probs)
        
        return accuracies

def main():
    predictor = IDSThresholdPredictor()
    
    # Evaluate model accuracy
    accuracies = predictor.evaluate_model_accuracy()
    print("\nModel Accuracy (R² scores):")
    for metric, accuracy in accuracies.items():
        print(f"{metric}: {accuracy:.4f}")
    
    # Predict at specific threshold
    threshold = 0.9
    prediction = predictor.predict_threshold_performance(threshold)
    
    print(f"\nPredicted Performance at threshold {threshold}:")
    print(f"Detection Rate: {prediction['detection_rate']:.2f}%")
    print(f"Total Events: {prediction['total_events']}")
    print(f"Ensemble Probability: {prediction['ensemble_probability']:.4f}")
    
    print("\nModel Contributions:")
    for model, contrib in prediction['model_contributions'].items():
        print(f"{model}: {contrib:.4f}")
    
    if 'confidence_intervals' in prediction:
        print("\nConfidence Intervals:")
        for metric, (lower, upper) in prediction['confidence_intervals'].items():
            print(f"{metric}: [{lower:.2f}, {upper:.2f}]")

if __name__ == "__main__":
    main()