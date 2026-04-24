from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
import joblib
import os
from typing import Dict, Any
from config import CONFIG


class TraditionalModels:
    """Class containing all traditional ML model implementations"""
    
    def __init__(self):
        self.config = CONFIG['MODEL_PARAMS']
        self.models = {}
        self.trained_models = {}
    
    def create_xgboost(self) -> XGBClassifier:
        """Create XGBoost model with configured parameters"""
        params = self.config['XGBoost']
        model = XGBClassifier(
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            n_estimators=params['n_estimators'],
            min_child_weight=params['min_child_weight'],
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            objective=params['objective'],
            tree_method=params['tree_method'],
            scale_pos_weight=params['scale_pos_weight'],
            random_state=params['random_state']
        )
        self.models['XGBoost'] = model
        return model
    
    def create_random_forest(self) -> RandomForestClassifier:
        """Create Random Forest model with configured parameters"""
        params = self.config['RandomForest']
        model = RandomForestClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf'],
            max_features=params['max_features'],
            bootstrap=params['bootstrap'],
            class_weight=params['class_weight'],
            criterion=params['criterion'],
            n_jobs=params['n_jobs'],
            random_state=params['random_state']
        )
        self.models['RandomForest'] = model
        return model
    
    def create_svm(self) -> LinearSVC:
        """Create SVM model with configured parameters"""
        params = self.config['SVM']
        model = LinearSVC(
            dual=params['dual'],
            max_iter=params['max_iter'],
            random_state=params['random_state'],
            C=params['C'],
            class_weight=params['class_weight'],
            tol=params['tol']
        )
        self.models['SVM'] = model
        return model

    def save_model(self, model_name: str, model: Any):
        """Save trained model"""
        save_dir = os.path.join(CONFIG['PATHS']['MODELS'], model_name)
        os.makedirs(save_dir, exist_ok=True)
        
        model_path = os.path.join(save_dir, f'{model_name.lower()}_model.joblib')
        joblib.dump(model, model_path)
        
        if hasattr(model, 'get_params'):
            params = model.get_params()
            params_path = os.path.join(save_dir, 'model_params.json')
            import json
            with open(params_path, 'w') as f:
                json.dump(params, f, indent=4)

    def load_model(self, model_name: str) -> Any:
        """Load trained model"""
        model_path = os.path.join(
            CONFIG['PATHS']['MODELS'],
            model_name,
            f'{model_name.lower()}_model.joblib'
        )
        
        if os.path.exists(model_path):
            return joblib.load(model_path)
        else:
            raise FileNotFoundError(f"No saved model found at {model_path}")
