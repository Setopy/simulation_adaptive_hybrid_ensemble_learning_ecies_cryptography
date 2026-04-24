import os
import torch
import logging
import warnings
from datetime import datetime
from config import CONFIG
from models.cnn_model import CNN_IDS
from models.lstm_model import LSTM_IDS
from models.dnn_model import DNN_IDS
from models.traditional_models import TraditionalModels
from trainers.neural_trainer import NeuralTrainer
from trainers.traditional_trainer import TraditionalTrainer
from utils.data_processor import DataProcessor

def setup_logging():
    """Setup logging configuration"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = CONFIG['PATHS']['LOGS']
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    warnings.filterwarnings('ignore')

def setup_device() -> torch.device:
    """Setup and return the appropriate device"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        logging.info("Using CPU")
    return device

def train_neural_models(X_train, X_test, y_train, y_test, device):
    """Train all neural network models"""
    neural_models = {
        'CNN': CNN_IDS(input_size=X_train.shape[1]),
        'LSTM': LSTM_IDS(input_size=X_train.shape[1]),
        'DNN': DNN_IDS(input_size=X_train.shape[1])
    }
    
    results = {}
    for model_name, model in neural_models.items():
        logging.info(f"\nTraining {model_name}...")
        trainer = NeuralTrainer(model_name, device)
        
        try:
            trained_model, metrics = trainer.train(
                model, X_train, y_train, X_test, y_test,
                gradient_accumulation_steps=CONFIG['TRAINING']['gradient_accumulation_steps']
            )
            results[model_name] = {
                'model': trained_model,
                'metrics': metrics
            }
            
            # Save model
            save_path = os.path.join(CONFIG['PATHS']['MODELS'], f'{model_name.lower()}_model.pth')
            trainer.save_model(trained_model, save_path)
            
        except Exception as e:
            logging.error(f"Error training {model_name}: {str(e)}")
            continue
    
    return results

def train_traditional_models(X_train, X_test, y_train, y_test):
    """Train all traditional models"""
    trad_models = TraditionalModels()
    models_to_train = {
        'XGBoost': trad_models.create_xgboost(),
        'RandomForest': trad_models.create_random_forest(),
        'SVM': trad_models.create_svm()
    }
    
    results = {}
    for model_name, model in models_to_train.items():
        logging.info(f"\nTraining {model_name}...")
        trainer = TraditionalTrainer(model_name)
        
        try:
            trained_model, metrics = trainer.train(
                model, X_train, y_train, X_test, y_test
            )
            results[model_name] = {
                'model': trained_model,
                'metrics': metrics
            }
            
            # Save model
            trad_models.save_model(model_name, trained_model)
            
        except Exception as e:
            logging.error(f"Error training {model_name}: {str(e)}")
            continue
    
    return results

def main():
    try:
        # Setup
        setup_logging()
        device = setup_device()
        
        # Load and preprocess data
        logging.info("Loading and preprocessing data...")
        data_processor = DataProcessor()
        X_train, X_test, y_train, y_test, input_size = data_processor.load_and_preprocess_data(
            CONFIG['PATHS']['TRAIN_FILE'],
            CONFIG['PATHS']['TEST_FILE']
        )
        
        logging.info(f"Data loaded successfully. Training set shape: {X_train.shape}")
        
        # Train neural models
        logging.info("\nStarting neural model training...")
        neural_results = train_neural_models(X_train, X_test, y_train, y_test, device)
        
        # Train traditional models
        logging.info("\nStarting traditional model training...")
        traditional_results = train_traditional_models(X_train, X_test, y_train, y_test)
        
        # Log final results
        logging.info("\nTraining completed. Final results:")
        for model_type, results in [("Neural", neural_results), ("Traditional", traditional_results)]:
            logging.info(f"\n{model_type} Models:")
            for model_name, result in results.items():
                metrics = result['metrics']
                logging.info(f"{model_name}:")
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        logging.info(f"  {metric_name}: {value:.4f}")
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
