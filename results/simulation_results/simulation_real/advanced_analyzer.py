import os
import torch
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torch import nn

# Define the neural network architectures
class LSTM_IDS(nn.Module):
    def __init__(self, input_size=196, hidden_size=128):
        super(LSTM_IDS, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return self.sigmoid(out)

class CNN_IDS(nn.Module):
    def __init__(self, input_size=196):
        super(CNN_IDS, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(6144, 1),  # Adjust size based on your architecture
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        return self.cnn(x)

class DNN_IDS(nn.Module):
    def __init__(self, input_size=196):
        super(DNN_IDS, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

class ComprehensiveModelAnalyzer:
    def __init__(self, models_dir: str):
        self.models_dir = Path(models_dir)
        self.feature_importance = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network_features = self.get_network_features()
        
        plt.style.use('default')
        sns.set_theme(style='whitegrid')
        plt.rcParams.update({
            'font.size': 12,
            'font.weight': 'bold',
            'axes.labelweight': 'bold',
            'axes.titleweight': 'bold',
            'figure.dpi': 300
        })

    def get_network_features(self):
        """Get actual network feature names"""
        # Basic network features
        basic_features = [
            'dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl',
            'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit',
            'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 'smean',
            'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src', 'ct_state_ttl',
            'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm',
            'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst'
        ]
        
        # Protocol features
        protocols = ['tcp', 'udp', 'icmp']
        proto_features = [f'proto_{p}' for p in protocols]
        
        # Service features
        services = ['http', 'ftp', 'smtp', 'ssh', 'dns', 'ftp-data', 
                   'irc', 'pop3', 'ssl', 'radius']
        service_features = [f'service_{s}' for s in services]
        
        # State features
        states = ['FIN', 'CON', 'REQ', 'RST', 'PAR', 'URN', 'INT', 'ECO']
        state_features = [f'state_{s}' for s in states]
        
        # Combine all features
        all_features = (basic_features + proto_features + 
                       service_features + state_features)
        
        # Add additional features if needed
        if len(all_features) < 196:
            remaining = 196 - len(all_features)
            all_features.extend([f'additional_feature_{i}' for i in range(remaining)])
        
        return all_features[:196]

    def plot_model_type_comparison(self, output_dir: str):
        """Compare traditional ML vs neural network approaches"""
        traditional = ['RandomForest', 'XGBoost', 'SVM']
        neural = ['CNN', 'DNN', 'LSTM']
        
        # Get available models
        trad_models = [m for m in traditional if m in self.feature_importance]
        neural_models = [m for m in neural if m in self.feature_importance]
        
        if trad_models:
            trad_importance = np.stack([self.feature_importance[m] for m in trad_models]).mean(axis=0)
        else:
            trad_importance = np.zeros(196)
            
        if neural_models:
            neural_importance = np.stack([self.feature_importance[m] for m in neural_models]).mean(axis=0)
        else:
            neural_importance = np.zeros(196)
        
        # Create comparison plot
        plt.figure(figsize=(15, 8))
        
        # Traditional ML plot
        plt.subplot(1, 2, 1)
        plt.title('Traditional ML Top Features')
        if np.any(trad_importance):
            top_idx = np.argsort(trad_importance)[-10:]
            plt.barh(range(10), trad_importance[top_idx])
            plt.yticks(range(10), [self.network_features[i] for i in top_idx])
        else:
            plt.text(0.5, 0.5, 'No traditional model data', 
                    ha='center', va='center')
        
        # Neural Network plot
        plt.subplot(1, 2, 2)
        plt.title('Neural Network Top Features')
        if np.any(neural_importance):
            top_idx = np.argsort(neural_importance)[-10:]
            plt.barh(range(10), neural_importance[top_idx])
            plt.yticks(range(10), [self.network_features[i] for i in top_idx])
        else:
            plt.text(0.5, 0.5, 'No neural network data', 
                    ha='center', va='center')
        
        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'model_type_comparison.png')
        plt.close()

    def create_comprehensive_analysis(self, output_dir: str):
        """Create comprehensive analysis visualizations"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create individual model plots
        for model_name, importance in self.feature_importance.items():
            plt.figure(figsize=(12, 8))
            top_idx = np.argsort(importance)[-10:]
            plt.barh(range(10), importance[top_idx])
            plt.yticks(range(10), [self.network_features[i] for i in top_idx])
            plt.title(f'{model_name} Top Features')
            plt.tight_layout()
            plt.savefig(Path(output_dir) / f'{model_name.lower()}_features.png')
            plt.close()
        
        # Create model type comparison
        self.plot_model_type_comparison(output_dir)
        
        # Create correlation matrix
        if len(self.feature_importance) > 1:
            corr_matrix = np.zeros((len(self.feature_importance), len(self.feature_importance)))
            models = list(self.feature_importance.keys())
            
            for i, model1 in enumerate(models):
                for j, model2 in enumerate(models):
                    corr = np.corrcoef(self.feature_importance[model1], 
                                     self.feature_importance[model2])[0, 1]
                    corr_matrix[i, j] = corr
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, xticklabels=models, 
                       yticklabels=models, cmap='coolwarm')
            plt.title('Model Correlation Matrix')
            plt.tight_layout()
            plt.savefig(Path(output_dir) / 'model_correlations.png')
            plt.close()
        
        # Create and return DataFrame
        return pd.DataFrame(self.feature_importance, index=self.network_features)

    def load_all_models(self):
        """Load and analyze both traditional and neural models"""
        # Load traditional models
        trad_models = {
            'RandomForest': 'randomforest_model.joblib',
            'XGBoost': 'xgboost_model.joblib',
            'SVM': 'svm_model.joblib'
        }
        
        for model_name, model_file in trad_models.items():
            try:
                model_path = self.models_dir / model_name / model_file
                print(f"Loading {model_name} from: {model_path}")
                
                model = joblib.load(model_path)
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[model_name] = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    coef = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
                    self.feature_importance[model_name] = coef / np.sum(np.abs(coef))
                
                print(f"Extracted importance for {model_name}")
                
            except Exception as e:
                print(f"Error loading {model_name}: {str(e)}")

def main():
    try:
        models_dir = '/home/seyitope/recent_ids_model/results/models'
        output_dir = '/home/seyitope/recent_ids_model/results/comprehensive_analysis'
        
        analyzer = ComprehensiveModelAnalyzer(models_dir)
        analyzer.load_all_models()
        importance_df = analyzer.create_comprehensive_analysis(output_dir)
        
        print("\nResults saved to:", output_dir)
        print("\nTop 10 Important Features by Model:")
        for model in importance_df.columns:
            print(f"\n{model}:")
            top_features = importance_df[model].sort_values(ascending=False)[:10]
            for feat, imp in top_features.items():
                print(f"{feat}: {imp:.4f}")

    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()