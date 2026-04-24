import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mticker
from datetime import datetime
import matplotlib as mpl

# Set high-quality visualization defaults
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.family'] = 'serif'
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.autolayout'] = True

# Create a custom color palette for the models
model_colors = {
    'cnn': '#1f77b4',    # Blue
    'lstm': '#ff7f0e',   # Orange
    'dnn': '#2ca02c',    # Green
    'svm': '#d62728',    # Red
    'xgboost': '#9467bd', # Purple
    'randomforest': '#8c564b' # Brown
}

# Set paths to your simulation data
base_dir = Path('/home/seyitope/recent_ids_modell/results/simulation_results/simulation_20250227_210515')
viz_dir = base_dir / 'professional_visualizations'
viz_dir.mkdir(exist_ok=True)

# Load the data
with open(base_dir / 'ensemble_metrics.json', 'r') as f:
    ensemble_metrics = json.load(f)
    
with open(base_dir / 'crypto_metrics.json', 'r') as f:
    crypto_metrics = json.load(f)
    
with open(base_dir / 'feature_importances.json', 'r') as f:
    feature_importances = json.load(f)
    
with open(base_dir / 'alerts.json', 'r') as f:
    alerts = json.load(f)
    
traffic_data = pd.read_csv(base_dir / 'traffic_data.csv')

# 1. Enhanced Model Performance and Contribution Visualization
def visualize_model_performance():
    """Create enhanced visualization of model performance and contributions"""
    fig = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1], figure=fig)
    
    # Extract model data
    models = list(ensemble_metrics['model_performance'].keys())
    
    # Upper plot: Model prediction performance
    ax1 = fig.add_subplot(gs[0])
    
    # Prepare data
    means = []
    stds = []
    for model in models:
        data = ensemble_metrics['model_performance'][model]['predictions']
        means.append(data['mean'])
        stds.append(data['std'])
    
    # Create bar chart with error bars
    bars = ax1.bar(
        [m.upper() for m in models], 
        means, 
        yerr=stds, 
        capsize=10,
        color=[model_colors[m] for m in models],
        alpha=0.8
    )
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        label = f"{means[i]:.4f}\n±{stds[i]:.4f}"
        ax1.annotate(
            label,
            xy=(bar.get_x() + bar.get_width()/2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center', va='bottom',
            fontsize=12, fontweight='bold'
        )
    
    ax1.set_title('Model Prediction Performance', fontsize=18, fontweight='bold')
    ax1.set_ylabel('Prediction Value', fontsize=14)
    ax1.set_ylim(0, 1.2)
    ax1.grid(axis='y', alpha=0.3)
    
    # Lower plot: Model contributions to ensemble
    ax2 = fig.add_subplot(gs[1])
    
    # Prepare contribution data from top-level 'weights'
    contributions = []
    for model in models:
        contributions.append(ensemble_metrics['weights'][model])
    
    # Create bar chart for contributions
    bars = ax2.bar(
        [m.upper() for m in models], 
        contributions,
        color=[model_colors[m] for m in models],
        alpha=0.8
    )
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax2.annotate(
            f"{contributions[i]:.4f}",
            xy=(bar.get_x() + bar.get_width()/2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center', va='bottom',
            fontsize=12, fontweight='bold'
        )
    
    ax2.set_title('Model Contribution to Ensemble', fontsize=18, fontweight='bold')
    ax2.set_ylabel('Contribution Weight', fontsize=14)
    max_contrib = max(contributions) if contributions else 0.2
    ax2.set_ylim(0, max_contrib * 1.2)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'model_performance_and_contributions.png')
    print(f"Created model performance visualization")
    
# 2. Normalized Feature Importance Visualization
def visualize_feature_importance():
    """Create normalized feature importance visualizations across models"""
    # Build a feature name mapping
    feature_mapping = {}
    
    # Process raw feature importances to get feature names and create mapping
    for model, importances in feature_importances.items():
        for feat_imp in importances:
            feature_name = feat_imp[0]
            if feature_name.startswith('feature_'):
                feature_index = feature_name.split('_')[1]
                # Find feature in traffic data columns by index
                index = int(feature_index)
                if index < len(traffic_data.columns):
                    feature_mapping[feature_name] = traffic_data.columns[index]
                else:
                    feature_mapping[feature_name] = feature_name
            else:
                feature_mapping[feature_name] = feature_name
    
    # Create figure
    fig = plt.figure(figsize=(16, 14))
    
    # Create subplots for each model
    models = list(feature_importances.keys())
    for i, model in enumerate(models):
        ax = fig.add_subplot(len(models), 1, i+1)
        
        # Get model's feature importances
        importances = feature_importances[model]
        
        # Get top 10 features (or fewer if less available)
        n_features = min(10, len(importances))
        
        # Extract feature names and values
        features = []
        values = []
        for j in range(n_features):
            feature_name = importances[j][0]
            # Use mapped name if available
            if feature_name in feature_mapping:
                features.append(feature_mapping[feature_name])
            else:
                features.append(feature_name)
            values.append(importances[j][1])
        
        # Normalize values for fair comparison
        max_val = max(values)
        norm_values = [v/max_val for v in values]
        
        # Reverse for bottom-to-top display
        features.reverse()
        values.reverse()
        norm_values.reverse()
        
        # Choose color based on model
        color = model_colors.get(model, '#333333')
        
        # Create horizontal bar chart
        bars = ax.barh(features, norm_values, color=color, alpha=0.8)
        
        # Add value labels
        for j, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(
                width + 0.02,
                bar.get_y() + bar.get_height()/2,
                f"{values[j]:.6f}",
                va='center',
                fontsize=10,
                fontweight='bold'
            )
        
        ax.set_title(f"{model.upper()} - Top Features (Normalized)", fontsize=16, fontweight='bold')
        ax.set_xlim(0, 1.1)  # Add space for labels
        ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'normalized_feature_importance.png')
    print(f"Created feature importance visualization")

# 3. Cryptographic Performance Visualization
def visualize_crypto_performance():
    """Create comprehensive cryptographic performance visualization"""
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], figure=fig)
    
    # 1. Operation times comparison
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Extract crypto metrics
    ops = ['Encryption', 'Decryption', 'Key Operations']
    times = [
        crypto_metrics['encryption']['average_time'],
        crypto_metrics['decryption']['average_time'],
        crypto_metrics['key_operations']['average_time']
    ]
    
    # Create bar chart
    colors = ['#3274A1', '#E1812C', '#3A923A']
    bars = ax1.bar(ops, times, color=colors)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(
            f'{height:.6f} ms',
            xy=(bar.get_x() + bar.get_width()/2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center', va='bottom',
            fontsize=12, fontweight='bold'
        )
    
    ax1.set_title('Cryptographic Operation Performance', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Time (ms)', fontsize=14)
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Success rates pie chart
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Extract success rate data
    labels = ['Encryption Success', 'Decryption Success']
    sizes = [
        crypto_metrics['encryption']['success_rate'] * 100,
        crypto_metrics['decryption']['success_rate'] * 100
    ]
    
    # Add authentication failures if any
    if crypto_metrics['decryption']['auth_failure_rate'] > 0:
        labels.append('Auth Failures')
        sizes.append(crypto_metrics['decryption']['auth_failure_rate'] * 100)
    
    # Create pie chart
    colors = ['#4CAF50', '#2196F3', '#F44336']
    explode = tuple(0.1 if label == 'Auth Failures' else 0.05 for label in labels)
    
    ax2.pie(
        sizes, 
        explode=explode, 
        labels=labels,
        colors=colors[:len(labels)],
        autopct='%1.1f%%',
        shadow=True,
        startangle=90,
        textprops={'fontsize': 12, 'fontweight': 'bold'}
    )
    ax2.axis('equal')
    ax2.set_title('Cryptographic Operation Success Rates', fontsize=16, fontweight='bold')
    
    # 3. Component sizes visualization
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Extract component sizes
    components = list(crypto_metrics['component_sizes'].keys())
    sizes = list(crypto_metrics['component_sizes'].values())
    
    # Calculate total overhead
    total_overhead = sum(sizes)
    
    # Create horizontal bar chart
    colors = sns.color_palette("Blues", len(components))
    bars = ax3.barh(components, sizes, color=colors)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax3.annotate(
            f'{width} bytes',
            xy=(width, bar.get_y() + bar.get_height()/2),
            xytext=(3, 0),
            textcoords="offset points",
            ha='left', va='center',
            fontsize=12, fontweight='bold'
        )
    
    ax3.set_title('Cryptographic Component Sizes', fontsize=16, fontweight='bold')
    ax3.set_xlabel('Size (bytes)', fontsize=14)
    ax3.grid(axis='x', alpha=0.3)
    
    # Add annotation with total overhead
    ax3.text(
        0.95, 0.05,
        f'Total Overhead: {total_overhead} bytes',
        horizontalalignment='right',
        verticalalignment='bottom',
        transform=ax3.transAxes,
        fontsize=14,
        fontweight='bold',
        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5')
    )
    
    # 4. Performance metrics table
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Hide axes
    ax4.axis('off')
    
    # Extract key metrics
    metrics_data = [
        ['Metric', 'Encryption', 'Decryption'],
        ['Operations', 
         str(crypto_metrics['encryption']['total_operations']),
         str(crypto_metrics['decryption']['total_operations'])],
        ['Bytes Processed', 
         str(crypto_metrics['encryption']['total_bytes_processed']),
         str(crypto_metrics['decryption']['total_bytes_processed'])],
        ['Avg Time (ms)', 
         f"{crypto_metrics['encryption']['average_time']:.6f}",
         f"{crypto_metrics['decryption']['average_time']:.6f}"],
        ['Success Rate', 
         f"{crypto_metrics['encryption']['success_rate'] * 100:.2f}%",
         f"{crypto_metrics['decryption']['success_rate'] * 100:.2f}%"],
        ['Failure Rate', 
         f"{crypto_metrics['encryption']['failure_rate'] * 100:.2f}%",
         f"{crypto_metrics['decryption']['failure_rate'] * 100:.2f}%"],
        ['Auth Failure Rate', 
         'N/A',
         f"{crypto_metrics['decryption']['auth_failure_rate'] * 100:.2f}%"]
    ]
    
    # Create table
    table = ax4.table(
        cellText=metrics_data,
        loc='center',
        cellLoc='center',
        colWidths=[0.3, 0.35, 0.35]
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    
    # Style header row
    for i in range(3):
        table[(0, i)].set_text_props(weight='bold', color='white')
        table[(0, i)].set_facecolor('#4472C4')
    
    # Style metric names column
    for i in range(1, len(metrics_data)):
        table[(i, 0)].set_text_props(weight='bold')
        table[(i, 0)].set_facecolor('#D9E1F2')
    
    ax4.set_title('Cryptographic Performance Metrics', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'crypto_performance.png')
    print(f"Created crypto performance visualization")

# 4. Enhanced Ensemble Probability Distribution
# 4. Enhanced Ensemble Probability Distribution
def visualize_ensemble_probability():
    """Create enhanced visualization of ensemble probability distribution"""
    plt.figure(figsize=(14, 8))
    
    # Extract data
    probabilities = ensemble_metrics['ensemble']['values']
    threshold = ensemble_metrics['threshold']
    
    if probabilities:
        # Choose appropriate binning
        n_bins = min(30, max(10, len(probabilities) // 10))
        
        # Create custom colormap - below threshold yellow, above threshold red
        cmap = LinearSegmentedColormap.from_list(
            'threshold_cmap', 
            [(0, '#FFE0B2'), (threshold-0.01, '#FFE0B2'), 
             (threshold, '#FFAB91'), (1, '#E64A19')]
        )
        
        # Get histogram data for coloring
        hist, bin_edges = np.histogram(probabilities, bins=n_bins, range=(0, 1))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_colors = [cmap(x) for x in bin_centers]
        
        # Create histogram using bar plot for individual bin coloring
        for i in range(len(hist)):
            plt.bar(
                bin_edges[i], 
                height=hist[i], 
                width=bin_edges[i+1]-bin_edges[i], 
                color=bin_colors[i], 
                edgecolor='black', 
                align='edge', 
                alpha=0.8
            )
        
        # Add threshold line
        plt.axvline(x=threshold, color='red', linestyle='--', linewidth=2)
        
        # Rest of the original code remains the same...
        # Add threshold annotation
        plt.annotate(
            f'Threshold: {threshold:.2f}',
            xy=(threshold, plt.gca().get_ylim()[1] * 0.9),
            xytext=(threshold + 0.1, plt.gca().get_ylim()[1] * 0.9),
            arrowprops=dict(facecolor='red', shrink=0.05, width=2),
            fontsize=14,
            fontweight='bold'
        )
        
        # Add statistics box
        mean_val = np.mean(probabilities)
        std_val = np.std(probabilities)
        intrusions = sum(1 for p in probabilities if p >= threshold)
        intrusion_pct = intrusions / len(probabilities) * 100
        
        stats_text = (
            f"Total samples: {len(probabilities)}\n"
            f"Mean: {mean_val:.4f}\n"
            f"Std Dev: {std_val:.4f}\n"
            f"Min: {min(probabilities):.4f}\n"
            f"Max: {max(probabilities):.4f}\n"
            f"Intrusions: {intrusions} ({intrusion_pct:.1f}%)"
        )
        
        plt.text(
            0.02, 0.95, 
            stats_text,
            transform=plt.gca().transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        # Add legend
        plt.legend(['Decision Threshold'], loc='upper right')
        
        # Add mean and std lines
        plt.axvline(x=mean_val, color='blue', linestyle='-', linewidth=1)
        plt.axvline(x=mean_val - std_val, color='blue', linestyle=':', linewidth=1)
        plt.axvline(x=mean_val + std_val, color='blue', linestyle=':', linewidth=1)
        
    # Set labels and title
    plt.title('Ensemble Probability Distribution', fontsize=18, fontweight='bold')
    plt.xlabel('Ensemble Probability', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'ensemble_probability.png')
    print(f"Created ensemble probability visualization")

# 5. Intrusion Timeline Analysis
def visualize_intrusion_timeline():
    """Create timeline visualization of detected intrusions"""
    plt.figure(figsize=(14, 8))
    
    # Extract data from alerts
    timestamps = []
    probabilities = []
    
    for alert in alerts:
        if 'timestamp' in alert and 'probability' in alert:
            timestamps.append(alert['timestamp'])
            probabilities.append(alert['probability'])
    
    if timestamps and probabilities:
        # Convert timestamps to relative seconds from start
        start_time = min(timestamps)
        rel_times = [(ts - start_time) for ts in timestamps]
        
        # Create a colormap for probability values
        cmap = plt.cm.plasma
        colors = [cmap(p) for p in probabilities]
        
        # Create scatter plot
        plt.scatter(rel_times, probabilities, c=colors, s=100, alpha=0.8, edgecolors='black')
        
        # Add threshold line
        threshold = ensemble_metrics['threshold']
        plt.axhline(y=threshold, color='red', linestyle='--', linewidth=2)
        
        # Add trend line
        z = np.polyfit(rel_times, probabilities, 1)
        p = np.poly1d(z)
        plt.plot(rel_times, p(rel_times), "r--", alpha=0.5)
        
        # Add annotations
        plt.annotate(
            f'Detection Threshold ({threshold:.1f})',
            xy=(rel_times[0], threshold),
            xytext=(rel_times[0], threshold - 0.05),
            fontsize=12,
            fontweight='bold'
        )
        
        # Add statistics
        detected = sum(1 for p in probabilities if p >= threshold)
        detection_rate = detected / len(probabilities) * 100
        
        plt.text(
            0.02, 0.05,
            f"Total alerts: {len(probabilities)}\n"
            f"Above threshold: {detected} ({detection_rate:.1f}%)\n"
            f"Average confidence: {np.mean(probabilities):.4f}",
            transform=plt.gca().transAxes,
            fontsize=12,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        # Format x-axis as minutes:seconds
        def format_time(x, pos):
            minutes = int(x // 60)
            seconds = int(x % 60)
            return f'{minutes:02d}:{seconds:02d}'
        
        plt.gca().xaxis.set_major_formatter(mticker.FuncFormatter(format_time))
        
        # Create colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm)
        cbar.set_label('Confidence', fontsize=12, fontweight='bold')
    
    # Set labels and title
    plt.title('Intrusion Detection Timeline', fontsize=18, fontweight='bold')
    plt.xlabel('Time (MM:SS) from Start', fontsize=14)
    plt.ylabel('Detection Confidence', fontsize=14)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'intrusion_timeline.png')
    print(f"Created intrusion timeline visualization")

# 6. Model Weight Adaptation Visualization
def visualize_weight_adaptation():
    """Visualize how model weights adapted over time"""
    if 'weights' in ensemble_metrics['metrics_history']:
        plt.figure(figsize=(14, 8))
        
        # Extract weight history
        weight_history = ensemble_metrics['metrics_history']['weights']
        
        if not weight_history:
            print("No weight adaptation history found")
            return
            
        # Get all model names
        models = list(weight_history[0].keys())
        
        # Extract weight values over time
        weight_values = {model: [] for model in models}
        for weights in weight_history:
            for model in models:
                if model in weights:
                    weight_values[model].append(weights[model])
                else:
                    weight_values[model].append(0)
        
        # Plot weight evolution for each model
        for model in models:
            plt.plot(
                weight_values[model], 
                label=model.upper(),
                color=model_colors.get(model, 'gray'),
                linewidth=2
            )
        
        # Set labels and title
        plt.title('Ensemble Weight Adaptation Over Time', fontsize=18, fontweight='bold')
        plt.xlabel('Adaptation Steps', fontsize=14)
        plt.ylabel('Model Weight', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'weight_adaptation.png')
        print(f"Created weight adaptation visualization")
    else:
        print("No weight adaptation history found")

# 7. Traffic Analysis Dashboard
def visualize_traffic_analysis():
    """Create a traffic analysis dashboard with key metrics"""
    plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], figure=plt)
    
    # 1. Traffic Label Distribution
    ax1 = plt.subplot(gs[0, 0])
    
    label_counts = traffic_data['label'].value_counts()
    labels = ['Normal', 'Intrusion']
    sizes = [label_counts.get(0, 0), label_counts.get(1, 0)]
    colors = ['#4CAF50', '#F44336']
    
    ax1.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        shadow=True,
        explode=(0.05, 0.1),
        textprops={'fontsize': 12, 'fontweight': 'bold'}
    )
    ax1.axis('equal')
    ax1.set_title('Traffic Distribution', fontsize=16, fontweight='bold')
    
    # 2. Protocol Distribution
    ax2 = plt.subplot(gs[0, 1])
    
    # Extract protocol columns
    proto_cols = [col for col in traffic_data.columns if col.startswith('proto_')]
    
    # Sum up instances of each protocol
    proto_sums = traffic_data[proto_cols].sum().sort_values(ascending=False)
    
    # Get top 10 protocols
    top_protos = proto_sums.head(10)
    
    # Clean up protocol names
    clean_names = [p.replace('proto_', '') for p in top_protos.index]
    
    # Plot horizontal bar chart
    bars = ax2.barh(clean_names, top_protos.values, color=sns.color_palette("viridis", 10))
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax2.annotate(
            f'{width:.0f}',
            xy=(width, bar.get_y() + bar.get_height()/2),
            xytext=(3, 0),
            textcoords="offset points",
            ha='left', va='center',
            fontsize=10,
            fontweight='bold'
        )
    
    ax2.set_title('Top 10 Protocols', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Count', fontsize=14)
    ax2.grid(axis='x', alpha=0.3)
    
    # 3. Connection State Distribution
    ax3 = plt.subplot(gs[1, 0])
    
    # Extract state columns
    state_cols = [col for col in traffic_data.columns if col.startswith('state_')]
    
    # Sum up instances of each state
    state_sums = traffic_data[state_cols].sum().sort_values(ascending=False)
    
    # Clean up state names
    clean_state_names = [s.replace('state_', '') for s in state_sums.index]
    
    # Plot horizontal bar chart
    bars = ax3.barh(clean_state_names, state_sums.values, color=sns.color_palette("mako", len(state_sums)))
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax3.annotate(
            f'{width:.0f}',
            xy=(width, bar.get_y() + bar.get_height()/2),
            xytext=(3, 0),
            textcoords="offset points",
            ha='left', va='center',
            fontsize=10,
            fontweight='bold'
        )
    
    ax3.set_title('Connection State Distribution', fontsize=16, fontweight='bold')
    ax3.set_xlabel('Count', fontsize=14)
    ax3.grid(axis='x', alpha=0.3)
    
    # 4. Traffic Volume Analysis
    ax4 = plt.subplot(gs[1, 1])
    
    # Extract key volume metrics
    if 'sbytes' in traffic_data.columns and 'dbytes' in traffic_data.columns:
        # Group by label
        normal = traffic_data[traffic_data['label'] == 0]
        intrusion = traffic_data[traffic_data['label'] == 1]
        
        # Calculate statistics
        metrics = ['sbytes', 'dbytes']
        normal_means = [normal[m].mean() for m in metrics]
        intrusion_means = [intrusion[m].mean() for m in metrics]
        
        # Create grouped bar chart
        x = np.arange(len(metrics))
        width = 0.35
        
        ax4.bar(x - width/2, normal_means, width, label='Normal', color='#4CAF50')
        ax4.bar(x + width/2, intrusion_means, width, label='Intrusion', color='#F44336')
        
        # Add value labels
        for i, v in enumerate(normal_means):
            ax4.text(i - width/2, v + 100, f'{v:.0f}', ha='center', fontweight='bold')
        for i, v in enumerate(intrusion_means):
            ax4.text(i + width/2, v + 100, f'{v:.0f}', ha='center', fontweight='bold')
        
        # Customize axes
        ax4.set_ylabel('Average Bytes', fontsize=14)
        ax4.set_title('Traffic Volume by Type', fontsize=16, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(['Source Bytes', 'Destination Bytes'])
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'traffic_analysis.png')
    print(f"Created traffic analysis visualization")

# Run all visualizations
visualize_model_performance()
visualize_feature_importance()
visualize_crypto_performance()
visualize_ensemble_probability()
visualize_intrusion_timeline()
visualize_weight_adaptation()
visualize_traffic_analysis()

print(f"\nAll visualizations have been saved to: {viz_dir}")