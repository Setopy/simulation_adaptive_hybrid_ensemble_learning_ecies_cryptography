import os
from pathlib import Path


# Create base directory structure
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = {
    'DATA': os.path.join(BASE_DIR, 'data'),
    'RESULTS': os.path.join(BASE_DIR, 'results'),
    'MODELS': os.path.join(BASE_DIR, 'results', 'models'),
    'METRICS': os.path.join(BASE_DIR, 'results', 'metrics'),
    'LOGS': os.path.join(BASE_DIR, 'results', 'logs')
}

# Create directories if they don't exist
for dir_path in PROJECT_DIR.values():
    os.makedirs(dir_path, exist_ok=True)

# Configuration settings
CONFIG = {
    # Training parameters
    'BATCH_SIZE': 32,
    'NUM_EPOCHS': 50,
    'LEARNING_RATE': 0.001,
    'EARLY_STOPPING_PATIENCE': 5,
    'MAX_TRAINING_TIME': 3600,  # 1 hour
    'RANDOM_SEED': 42,
    
    # Paths
    'PATHS': {
        'DATA': PROJECT_DIR['DATA'],
        'RESULTS': PROJECT_DIR['RESULTS'],
        'MODELS': PROJECT_DIR['MODELS'],
        'METRICS': PROJECT_DIR['METRICS'],
        'LOGS': PROJECT_DIR['LOGS'],
        'TRAIN_FILE': os.path.join(PROJECT_DIR['DATA'], 'UNSW_NB15_training-set.csv'),
        'TEST_FILE': os.path.join(PROJECT_DIR['DATA'], 'UNSW_NB15_testing-set.csv')
    },
    
    # Model parameters
    'MODEL_PARAMS': {
        'CNN': {
            'conv_layers': [64, 128, 256],
            'kernel_size': 3,
            'dropout': 0.2,
            'attention_heads': 4,
            'sequence_length': 14,
            'use_batch_norm': True,
            'activation': 'leaky_relu',
            'leaky_slope': 0.1
        },
        
        'LSTM': {
            'hidden_size': 256,
            'num_layers': 2,
            'dropout': 0.3,
            'bidirectional': True,
            'attention_size': 64,
            'feature_size': 128,
            'use_batch_norm': True
        },
        
        'DNN': {
            'layers': [512, 256, 128, 64],
            'dropout_rates': [0.2, 0.3, 0.3, 0.3],
            'use_batch_norm': True,
            'activation': 'leaky_relu',
            'leaky_slope': 0.1,
            'use_residual': True
        },
        
        'XGBoost': {
            'max_depth': 8,
            'learning_rate': 0.08,
            'n_estimators': 150,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'binary:logistic',
            'tree_method': 'hist',
            'scale_pos_weight': 1,
            'random_state': 42
        },
        
        'RandomForest': {
            'n_estimators': 250,
            'max_depth': 25,
            'min_samples_split': 6,
            'min_samples_leaf': 3,
            'max_features': 'sqrt',
            'bootstrap': True,
            'class_weight': 'balanced_subsample',
            'criterion': 'entropy',
            'n_jobs': -1,
            'random_state': 42,
            'max_samples': 0.8
        },
        
        'SVM': {
            'dual': 'auto',
            'max_iter': 2500,
            'random_state': 42,
            'C': 1.0,
            'class_weight': 'balanced',
            'tol': 1e-4
        }
    },
    
    # Training settings
    'TRAINING': {
        'gradient_accumulation_steps': 4,
        'mixed_precision': True,
        'max_grad_norm': 1.0,
        'clip_value': 5.0,
        'validation_interval': 5,
        'save_best_only': True,
        'scheduler': {
            'type': 'one_cycle',
            'pct_start': 0.3,
            'anneal_strategy': 'cos',
            'div_factor': 25.0,
            'final_div_factor': 10000.0
        }
    },
    
    # Metrics settings
    'METRICS': {
        'save_training_history': True,
        'save_predictions': True,
        'detailed_logging': True,
        'evaluation_metrics': [
            'accuracy',
            'precision',
            'recall',
            'f1_score',
            'loss'
        ]
    },
    
    # Data preprocessing
    'PREPROCESSING': {
        'normalize_method': 'standard',
        'handle_missing': 'zero',
        'categorical_columns': ['proto', 'service', 'state'],
        'test_size': 0.2,
        'stratify': True,
        'shuffle': True
    }
}
