import os
import logging
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from sim_config import CONFIG
from sim_network_simulator import NetworkSimulator, NetworkMonitor
from sim_crypto_manager import CryptoManager
from sim_ids_monitor import EnhancedIDSMonitor

def setup_pid_logging():
    """Create PID-based logging with automatic file organization"""
    pid = os.getpid()
    sim_dir = f"sim_{pid}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(sim_dir, exist_ok=True)
    
    log_file = os.path.join(sim_dir, f'simulation_{pid}.log')
    
    # Clear existing handlers
    logging.getLogger().handlers = []
    
    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s [PID-{pid}] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return sim_dir, pid

def create_simulation_directory():
    """Create a unique directory for each simulation"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    sim_dir = os.path.join(
        CONFIG['PATHS']['SIMULATION'],
        f"simulation_{timestamp}"
    )
    os.makedirs(sim_dir, exist_ok=True)
    return sim_dir

def setup_logging(simulation_dir):
    """Configure logging to use simulation-specific log file"""
    log_file = os.path.join(simulation_dir, 'simulation.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Simulation directory created at: {simulation_dir}")

def save_simulation_data(traffic_data, alerts, simulation_dir):
    """Save all simulation data to the designated directory"""
    # Save traffic data
    if traffic_data:
        traffic_path = os.path.join(simulation_dir, 'traffic_data.csv')
        pd.DataFrame(traffic_data).to_csv(traffic_path, index=False)
        logging.info(f"Traffic data saved to {traffic_path}")
    
    # Save alerts
    if alerts:
        alerts_path = os.path.join(simulation_dir, 'alerts.json')
        with open(alerts_path, 'w') as f:
            json.dump(alerts, f, indent=4)
        logging.info(f"Alerts saved to {alerts_path}")
    
    return simulation_dir

def log_crypto_metrics(crypto_manager, simulation_dir):
    """Log and save detailed crypto metrics"""
    # Get metrics from the CryptoManager
    metrics = crypto_manager.get_metrics()
    
    # Log key metrics to console and log file
    logging.info("=== Crypto Performance Metrics ===")
    
    # Encryption metrics
    enc_metrics = metrics['encryption']
    logging.info(f"Encryption avg time: {enc_metrics['average_time']:.4f} ms")
    logging.info(f"Encryption success rate: {enc_metrics['success_rate']:.2%}")
    logging.info(f"Encryption avg overhead: {enc_metrics['average_overhead']} bytes")
    
    # Decryption metrics
    dec_metrics = metrics['decryption']
    logging.info(f"Decryption avg time: {dec_metrics['average_time']:.4f} ms")
    logging.info(f"Decryption success rate: {dec_metrics['success_rate']:.2%}")
    
    # Save full metrics to JSON file
    metrics_path = os.path.join(simulation_dir, 'crypto_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    logging.info(f"Detailed crypto metrics saved to {metrics_path}")
    
    return metrics

def extract_and_log_feature_importances(ids_monitor, feature_names, simulation_dir):
    """Extract and log feature importances from IDS models"""
    if not hasattr(ids_monitor, 'ensemble') or not hasattr(ids_monitor.ensemble, 'models'):
        logging.warning("Cannot extract feature importances: IDS monitor does not have ensemble models")
        return {}
    
    feature_importances = {}
    
    for model_name, model in ids_monitor.ensemble.models.items():
        # Skip if model doesn't support feature importances
        if not (hasattr(model, 'feature_importances_') or hasattr(model, 'coef_')):
            continue
        
        # Extract feature importances
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # For linear models, use absolute coefficient values
            importances = np.abs(model.coef_[0] if model.coef_.ndim > 1 else model.coef_)
        else:
            continue
        
        # Create a dict mapping feature names to importance values
        model_importances = {}
        for i, importance in enumerate(importances):
            feature_name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
            model_importances[feature_name] = float(importance)
        
        # Sort by importance (descending)
        sorted_importances = sorted(
            model_importances.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Log the top 5 features
        logging.info(f"{model_name.upper()} top feature importances:")
        for i, (feature, importance) in enumerate(sorted_importances[:5], 1):
            logging.info(f"  {i}. {feature}: {importance:.6f}")
        
        # Store in the result dict
        feature_importances[model_name] = sorted_importances
    
    # Save feature importances to JSON
    if feature_importances:
        importances_path = os.path.join(simulation_dir, 'feature_importances.json')
        with open(importances_path, 'w') as f:
            json.dump(feature_importances, f, indent=4)
        logging.info(f"Feature importances saved to {importances_path}")
    
    return feature_importances

def collect_ensemble_metrics(ids_monitor, model_predictions=None, ensemble_probabilities=None):
    """Collect comprehensive metrics about the ensemble models with model-specific predictions"""
    # Get base metrics from the IDS monitor
    base_metrics = ids_monitor.get_ensemble_metrics()
    
    # Add additional metrics
    extended_metrics = {
        'model_performance': {},
        'ensemble': {
            'values': ensemble_probabilities if ensemble_probabilities else [],
            'mean': np.mean(ensemble_probabilities) if ensemble_probabilities and len(ensemble_probabilities) > 0 else 0,
            'std': np.std(ensemble_probabilities) if ensemble_probabilities and len(ensemble_probabilities) > 0 else 0
        },
        'threshold': ids_monitor.threshold if hasattr(ids_monitor, 'threshold') else 0.5
    }
    
    # Extract model predictions and contributions
    if hasattr(ids_monitor, 'ensemble') and hasattr(ids_monitor.ensemble, 'models'):
        for model_name, model in ids_monitor.ensemble.models.items():
            model_metrics = {
                'predictions': {},
                'contributions': {}
            }
            
            # Use tracked model predictions if available
            if model_predictions and model_name in model_predictions and model_predictions[model_name]:
                predictions = model_predictions[model_name]
                model_metrics['predictions'] = {
                    'mean': float(np.mean(predictions)),
                    'std': float(np.std(predictions)),
                    'min': float(np.min(predictions)),
                    'max': float(np.max(predictions)),
                    'count': len(predictions)
                }
            else:
                # Add empty predictions if we haven't tracked any
                model_metrics['predictions'] = {
                    'mean': 0,
                    'std': 0,
                    'min': 0,
                    'max': 0,
                    'count': 0
                }
            
            # Get contribution metrics if available
            if hasattr(ids_monitor.ensemble, 'weights') and model_name in ids_monitor.ensemble.weights:
                # For static weights, just use the current weight
                weight = ids_monitor.ensemble.weights[model_name]
                model_metrics['contributions'] = {
                    'mean': weight,
                    'std': 0,
                    'min': weight,
                    'max': weight,
                    'count': 1
                }
            
            extended_metrics['model_performance'][model_name] = model_metrics
    
    # Merge with base metrics
    merged_metrics = {**base_metrics, **extended_metrics}
    
    return merged_metrics

def main():
    sim_dir, pid = setup_pid_logging()
    logging.info(f"All simulation files for this run in: {sim_dir}")
    
    try:
        # Create simulation directory and setup logging
        simulation_dir = create_simulation_directory()
        setup_logging(simulation_dir)
        
        # Initialize components
        network = NetworkSimulator(CONFIG)
        crypto = CryptoManager()
        ids = EnhancedIDSMonitor(CONFIG['MODEL']['MODELS_DIR'])
        monitor = NetworkMonitor(crypto, ids)
        
        # Track metrics for models
        model_predictions = defaultdict(list)
        ensemble_probabilities = []
        
        # Store original predict method to capture metrics
        original_detect_intrusion = ids.detect_intrusion
        
        # Define the enhanced predict method that tracks metrics
        def detect_intrusion_with_tracking(traffic_data):
            # Call original predict method
            result = original_detect_intrusion(traffic_data)
            
            # Track individual model predictions
            if hasattr(ids, 'ensemble') and hasattr(ids.ensemble, 'models'):
                for model_name, model in ids.ensemble.models.items():
                    if hasattr(model, 'last_prediction'):
                        model_predictions[model_name].append(model.last_prediction)
            
            # Track ensemble probability
            if isinstance(result, tuple) and len(result) > 1:
                _, probability = result
                ensemble_probabilities.append(probability)
            
            return result
        
        # Replace predict method with enhanced version
        ids.detect_intrusion = detect_intrusion_with_tracking
        
        # Simulation parameters
        collected_traffic = []
        start_time = time.time()
        simulation_duration =  11700
        
        # Get feature names if available
        feature_names = CONFIG.get('FEATURES', {}).get('NAMES', [])
        if not feature_names:
            feature_names = [f"feature_{i}" for i in range(100)]  # Default placeholders
        
        # Run simulation
        network.start()
        try:
            while time.time() - start_time < simulation_duration:
                if (traffic := network.get_traffic()):
                    monitor.process_traffic(traffic)
                    collected_traffic.append(traffic)
                
                # Log progress every 30 seconds
                if (time.time() - start_time) % 30 < 0.1:
                    progress = (time.time() - start_time) / simulation_duration * 100
                    logging.info(f"Simulation progress: {progress:.1f}%")
                
                time.sleep(0.1)
        finally:
            network.stop()
        
        # Save results
        save_simulation_data(collected_traffic, monitor.get_alerts(), simulation_dir)
        
        # Extract and log feature importances
        feature_importances = extract_and_log_feature_importances(ids, feature_names, simulation_dir)
        
        # Log and save crypto metrics
        crypto_metrics = log_crypto_metrics(crypto, simulation_dir)
        
        # Collect and save ensemble metrics with extended data
        ids_metrics = collect_ensemble_metrics(ids, model_predictions, ensemble_probabilities)
        metrics_path = os.path.join(simulation_dir, 'ensemble_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(ids_metrics, f, indent=4)
        
        # Final report
        logging.info("\n=== Simulation Summary ===")
        logging.info(f"Total traffic processed: {len(collected_traffic)}")
        logging.info(f"Alerts generated: {len(monitor.get_alerts())}")
        
        # Add model summary
        if model_predictions:
            logging.info("\nModel Performance Summary:")
            for model, predictions in model_predictions.items():
                if predictions:
                    mean_pred = np.mean(predictions)
                    logging.info(f"  {model.upper()} mean prediction: {mean_pred:.4f}")
        
        logging.info(f"All results stored in: {simulation_dir}")
        
    except Exception as e:
        logging.error(f"Simulation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
