import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
from pathlib import Path
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker

# Set up visualization style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.family'] = 'serif'

# Define paths to your data files
sim_dir = Path('/home/seyitope/recent_ids_modell/results/simulation_results/simulation_20250228_125040')
log_dir = Path('/home/seyitope/recent_ids_modell/sim_983038_20250228_125040')

# Function to explore JSON files
def explore_json_file(file_path):
    """Explore the structure and content of a JSON file."""
    print(f"\n{'='*80}\nExploring {file_path.name}\n{'='*80}")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        print(f"Top-level keys: {list(data.keys())}")
        for key, value in data.items():
            value_type = type(value).__name__
            sample = str(value)[:100] + '...' if len(str(value)) > 100 else str(value)
            print(f"  - {key} ({value_type}): {sample}")
    elif isinstance(data, list):
        print(f"Data is a list with {len(data)} items")
        if data:
            print(f"Sample first item type: {type(data[0]).__name__}")
            sample = str(data[0])[:100] + '...' if len(str(data[0])) > 100 else str(data[0])
            print(f"Sample first item: {sample}")
    
    return data

# Function to explore CSV files
def explore_csv_file(file_path):
    """Explore the structure and content of a CSV file."""
    print(f"\n{'='*80}\nExploring {file_path.name}\n{'='*80}")
    
    df = pd.read_csv(file_path)
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nSample data (first 5 rows):")
    print(df.head())
    
    print("\nNumeric columns summary:")
    print(df.describe())
    
    return df

# Function to explore log files (and extract key metrics)
def explore_log_file(file_path):
    """Explore log file and extract important metrics."""
    print(f"\n{'='*80}\nExploring {file_path.name}\n{'='*80}")
    
    with open(file_path, 'r') as f:
        log_content = f.read()
    
    print(f"Log file size: {len(log_content)} characters")
    
    # Find key patterns in logs
    intrusion_patterns = re.findall(r"Intrusion confirmed! Confidence: ([0-9.]+)", log_content)
    adaptation_patterns = re.findall(r"Successfully adapted ensemble weights:", log_content)
    
    print(f"Found {len(intrusion_patterns)} intrusion confirmations")
    print(f"Found {len(adaptation_patterns)} ensemble weight adaptations")
    
    # Print sample of log content
    print("\nLog sample (first 500 characters):")
    print(log_content[:500] + "...")
    
    return log_content

# Explore the data files
try:
    ensemble_metrics = explore_json_file(sim_dir / 'ensemble_metrics.json')
    crypto_metrics = explore_json_file(sim_dir / 'crypto_metrics.json')
    feature_importances = explore_json_file(sim_dir / 'feature_importances.json')
    alerts = explore_json_file(sim_dir / 'alerts.json')
    traffic_data = explore_csv_file(sim_dir / 'traffic_data.csv')
    log_content = None
    
    log_file = log_dir / 'simulation_983038.log'
    if log_file.exists():
        log_content = explore_log_file(log_file)
    else:
        print(f"Log file not found at {log_file}")
        
except Exception as e:
    print(f"Error exploring data files: {str(e)}")