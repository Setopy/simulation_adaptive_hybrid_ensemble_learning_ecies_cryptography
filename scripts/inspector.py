import os
import pandas as pd
import os
import json
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime
class DataInspector:
    """Inspect data files and their contents"""
    
    def __init__(self, traffic_path, alerts_path):
        self.traffic_path = os.path.expanduser(traffic_path)
        self.alerts_path = os.path.expanduser(alerts_path)

    def inspect_traffic_data(self):
        """Analyze traffic data file structure and content"""
        try:
            print("\n=== Traffic Data Analysis ===")
            df = pd.read_csv(self.traffic_path)
            
            print("\nBasic Information:")
            print(f"Number of records: {len(df)}")
            print(f"Number of columns: {len(df.columns)}")
            
            print("\nColumns available:")
            for col in df.columns:
                print(f"- {col} ({df[col].dtype})")
            
            print("\nSample numeric statistics:")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols[:5]:  # Show first 5 numeric columns
                stats = df[col].describe()
                print(f"\n{col}:")
                print(f"  Mean: {stats['mean']:.2f}")
                print(f"  Std: {stats['std']:.2f}")
                print(f"  Min: {stats['min']:.2f}")
                print(f"  Max: {stats['max']:.2f}")
            
            print("\nTimestamp information:")
            if 'timestamp' in df.columns:
                print(f"Start: {df['timestamp'].min()}")
                print(f"End: {df['timestamp'].max()}")
            else:
                print("No timestamp column found")
                
            print("\nEncryption related columns:")
            enc_cols = [col for col in df.columns if any(x in col.lower() 
                      for x in ['encrypt', 'decrypt', 'overhead'])]
            if enc_cols:
                print("Found columns:", enc_cols)
                for col in enc_cols:
                    print(f"\n{col} statistics:")
                    print(df[col].describe())
            else:
                print("No encryption-related columns found")
                
        except Exception as e:
            print(f"Error analyzing traffic data: {str(e)}")

    def inspect_alerts(self):
        """Analyze alerts file structure and content"""
        try:
            print("\n=== Alerts Data Analysis ===")
            with open(self.alerts_path) as f:
                alerts = json.load(f)
            
            print(f"\nNumber of alerts: {len(alerts)}")
            
            if alerts:
                print("\nAlert structure (first alert):")
                self._print_dict_structure(alerts[0])
                
                print("\nChecking for specific fields across all alerts:")
                fields = {
                    'probability': 0,
                    'model_contributions': 0,
                    'traffic_data': 0,
                    'encryption_overhead': 0,
                    'timestamp': 0
                }
                
                for alert in alerts:
                    for field in fields:
                        if field in alert:
                            fields[field] += 1
                
                print("\nField availability:")
                for field, count in fields.items():
                    print(f"- {field}: {count}/{len(alerts)} alerts")
                
                # Check model-specific information
                if any('model_contributions' in alert for alert in alerts):
                    print("\nModel information found:")
                    models = set()
                    features = set()
                    for alert in alerts:
                        if 'model_contributions' in alert:
                            models.update(alert['model_contributions'].keys())
                            for model_data in alert['model_contributions'].values():
                                if isinstance(model_data, dict):
                                    features.update(model_data.keys())
                    
                    print("Models present:", sorted(models))
                    print("\nFeatures tracked:", sorted(features)[:10], "...")  # Show first 10
                
            else:
                print("No alerts found in file")
                
        except Exception as e:
            print(f"Error analyzing alerts: {str(e)}")

    def _print_dict_structure(self, d, indent=0, max_depth=3):
        """Helper to print nested dictionary structure"""
        if indent >= max_depth:
            print(" " * indent + "...")
            return
            
        for key, value in d.items():
            if isinstance(value, dict):
                print(" " * indent + f"{key}:")
                self._print_dict_structure(value, indent + 2, max_depth)
            elif isinstance(value, list):
                print(" " * indent + f"{key}: list[{len(value)}]")
                if value and indent < max_depth - 1:
                    print(" " * (indent + 2) + f"First item type: {type(value[0]).__name__}")
            else:
                print(" " * indent + f"{key}: {type(value).__name__}")

def main():
    """Main execution flow"""
    base_dir = os.path.expanduser("~/recent_ids_modell/results/simulation_results")
    print(f"Looking for simulation results in: {base_dir}")
    
    try:
        simulation_dirs = [d for d in os.listdir(base_dir) if d.startswith('simulation_')]
        if not simulation_dirs:
            print("No simulation results found!")
            return
            
        latest_sim = max(simulation_dirs)
        sim_dir = os.path.join(base_dir, latest_sim)
        print(f"\nUsing simulation results from: {sim_dir}")
        
        # Setup paths
        traffic_path = os.path.join(sim_dir, "traffic_data.csv")
        alerts_path = os.path.join(sim_dir, "alerts.json")
        
        # Inspect data files
        inspector = DataInspector(traffic_path, alerts_path)
        inspector.inspect_traffic_data()
        inspector.inspect_alerts()
        
    except Exception as e:
        print(f"\nError in inspection: {str(e)}")

if __name__ == "__main__":
    main()
