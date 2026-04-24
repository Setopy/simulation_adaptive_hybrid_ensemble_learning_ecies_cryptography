import os
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple
from config import CONFIG

class DataProcessor:
    """Handle data loading and preprocessing"""
    def __init__(self, random_state: int = CONFIG['RANDOM_SEED']):
        self.scaler = StandardScaler()
        self.random_state = random_state
        
    def load_and_preprocess_data(self, training_file: str, testing_file: str) -> Tuple:
        """Load and preprocess the UNSW-NB15 dataset"""
        try:
            logging.info(f"Loading training data from {training_file}")
            logging.info(f"Loading testing data from {testing_file}")
            
            # Load datasets
            df_train = pd.read_csv(training_file)
            df_test = pd.read_csv(testing_file)
            
            # Combine for preprocessing
            df_combined = pd.concat([df_train, df_test], axis=0)
            
            # Clean column names
            if 'ï»¿id' in df_combined.columns:
                df_combined = df_combined.rename(columns={'ï»¿id': 'id'})

            # Extract features
            numeric_features = [col for col in df_combined.columns 
                              if df_combined[col].dtype in ['int64', 'float64']
                              and col not in ['id', 'label']]
            categorical_features = ['proto', 'service', 'state']
            
            columns_to_keep = numeric_features + categorical_features + ['label']
            df_combined = df_combined[columns_to_keep]
            
            # Prepare features and target
            X = df_combined.drop(columns=['label'])
            y = df_combined['label']
            
            # One-hot encoding
            X = pd.get_dummies(X, columns=categorical_features)
            X = X.fillna(0)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=self.random_state, stratify=y
            )

            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            logging.info(f"Data preprocessing completed. Training set shape: {X_train_scaled.shape}")
            
            return X_train_scaled, X_test_scaled, y_train, y_test, X_train.shape[1]

        except Exception as e:
            logging.error(f"Error in data preprocessing: {str(e)}")
            raise
