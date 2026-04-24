import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import squarify
from matplotlib.gridspec import GridSpec
from pathlib import Path

# Set professional styling
plt.style.use('seaborn-whitegrid')
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'font.weight': 'semibold',
    'axes.titleweight': 'bold'
})

COLORS = {
    'cnn': '#1f77b4', 'lstm': '#ff7f0e', 'dnn': '#2ca02c',
    'svm': '#d62728', 'xgboost': '#9467bd', 'randomforest': '#8c564b',
    'primary': '#0072B2', 'highlight': '#D55E00', 'success': '#009E73'
}

def load_data():
    """Load required data files"""
    data = {}
    with open('ensemble_metrics.json') as f:
        data['ensemble'] = json.load(f)
    with open('crypto_metrics.json') as f:
        data['crypto'] = json.load(f)
    data['traffic'] = pd.read_csv('traffic_data.csv')
    return data

def plot_system_architecture():
    """Network flow diagram of the hybrid system"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # System components
    components = {
        'Network\nSensor': (0.1, 0.5),
        'Feature\nProcessor': (0.3, 0.5),
        'AI\nEnsemble': (0.5, 0.7),
        'Cryptographic\nEngine': (0.5, 0.3),
        'Threat\nDashboard': (0.7, 0.5),
        'Secure\nStorage': (0.9, 0.5)
    }

    # Draw components
    for label, (x, y) in components.items():
        ax.add_patch(plt.Rectangle((x-0.1, y-0.05), 0.2, 0.1,
                     facecolor=COLORS['primary'], alpha=0.8))
        ax.text(x, y, label, ha='center', va='center',
                color='white', fontweight='bold')

    # Data flows
    connections = [
        ('Network\nSensor', 'Feature\nProcessor'),
        ('Feature\nProcessor', 'AI\nEnsemble'),
        ('Feature\nProcessor', 'Cryptographic\nEngine'),
        ('AI\nEnsemble', 'Threat\nDashboard'),
        ('Cryptographic\nEngine', 'Threat\nDashboard'),
        ('Threat\nDashboard', 'Secure\nStorage')
    ]

    for start, end in connections:
        xs, ys = components[start]
        xe, ye = components[end]
        ax.annotate("", xy=(xe-0.1, ye), xytext=(xs+0.1, ys),
                    arrowprops=dict(arrowstyle="->", color=COLORS['highlight'],
                                  lw=2, shrinkA=5, shrinkB=5))

    plt.title('Hybrid AI-Cryptography System Architecture', fontsize=16)
    plt.savefig('visualizations/system_architecture.png', bbox_inches='tight')
    plt.close()

def plot_performance_radar(data):
    """Radar chart comparing model performance"""
    metrics = ['precision', 'recall', 'f1', 'accuracy']
    model_data = {}
    
    for model in data['ensemble']['model_performance']:
        perf = data['ensemble']['model_performance'][model]
        model_data[model] = [perf.get(m, 0) for m in metrics]
    
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    
    for model, values in model_data.items():
        values += values[:1]
        ax.plot(angles, values, label=model.upper(), 
                color=COLORS.get(model, 'gray'))
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels(metrics)
    plt.yticks([0.2, 0.4, 0.6, 0.8], ["20%", "40%", "60%", "80%"])
    plt.legend(loc='lower left', bbox_to_anchor=(1.1, 0))
    plt.title('Model Performance Comparison', pad=30)
    plt.savefig('visualizations/performance_radar.png', bbox_inches='tight')
    plt.close()

def plot_feature_heatmap(data):
    """Heatmap of feature importance across models"""
    features = set()
    for model in data['ensemble']['feature_importances']:
        features.update([f[0] for f in data['ensemble']['feature_importances'][model]])
    
    df = pd.DataFrame(index=data['ensemble']['feature_importances'].keys(),
                     columns=list(features))
    for model, importances in data['ensemble']['feature_importances'].items():
        for feature, score in importances:
            df.loc[model, feature] = score
    
    plt.figure(figsize=(15, 10))
    sns.heatmap(df.T.fillna(0), cmap="YlGnBu", cbar_kws={'label': 'Importance Score'})
    plt.title('Cross-Model Feature Importance', fontsize=14)
    plt.xlabel('Models', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.xticks(rotation=45)
    plt.savefig('visualizations/feature_heatmap.png', bbox_inches='tight')
    plt.close()

def plot_crypto_components(data):
    """Treemap of cryptographic component sizes"""
    sizes = data['crypto']['component_sizes']
    labels = [f"{k}\n({v} B)" for k, v in sizes.items()]
    
    plt.figure(figsize=(10, 6))
    squarify.plot(sizes=sizes.values(), label=labels,
                 color=sns.color_palette("Blues", len(sizes)),
                 alpha=0.7, text_kwargs={'fontsize':10})
    plt.title('Cryptographic Component Size Distribution', fontsize=14)
    plt.axis('off')
    plt.savefig('visualizations/crypto_components.png', bbox_inches='tight')
    plt.close()

def create_main_dashboard(data):
    """Comprehensive performance dashboard"""
    fig = plt.figure(figsize=(20, 15))
    gs = GridSpec(3, 3, figure=fig)
    
    # Model Weights
    ax1 = fig.add_subplot(gs[0, 0])
    models = list(data['ensemble']['weights'].keys())
    weights = list(data['ensemble']['weights'].values()))
    ax1.bar(models, weights, color=[COLORS[m] for m in models])
    ax1.set_title('Ensemble Model Weights')
    
    # Crypto Performance
    ax2 = fig.add_subplot(gs[0, 1])
    operations = ['Encryption', 'Decryption']
    times = [data['crypto']['encryption']['average_time'],
             data['crypto']['decryption']['average_time']]
    ax2.bar(operations, times, color=COLORS['primary'])
    ax2.set_title('Cryptographic Operation Times')
    
    # Traffic Distribution
    ax3 = fig.add_subplot(gs[0, 2])
    traffic = data['traffic']['label'].value_counts()
    ax3.pie(traffic, labels=['Normal', 'Anomaly'],
           colors=[COLORS['success'], COLORS['highlight']],
           autopct='%1.1f%%')
    ax3.set_title('Traffic Classification')
    
    # Feature Importance
    ax4 = fig.add_subplot(gs[1:, :2])
    combined_importances = {}
    for model in data['ensemble']['feature_importances']:
        for feature, score in data['ensemble']['feature_importances'][model]:
            combined_importances[feature] = combined_importances.get(feature, 0) + score
    top_features = dict(sorted(combined_importances.items(),
                             key=lambda x: x[1], reverse=True)[:10])
    ax4.barh(list(top_features.keys()), list(top_features.values()),
            color=COLORS['primary'])
    ax4.set_title('Top 10 Important Features')
    
    # Timeline
    ax5 = fig.add_subplot(gs[1:, 2])
    alerts = data['crypto'].get('alerts', [])
    if alerts:
        times = [a['timestamp'] for a in alerts]
        confidences = [a['probability'] for a in alerts]
        ax5.plot(times, confidences, color=COLORS['highlight'])
        ax5.set_title('Detection Confidence Timeline')
    
    plt.tight_layout()
    plt.savefig('visualizations/main_dashboard.png', bbox_inches='tight')
    plt.close()

def main():
    Path('visualizations').mkdir(exist_ok=True)
    data = load_data()
    
    plot_system_architecture()
    plot_performance_radar(data)
    plot_feature_heatmap(data)
    plot_crypto_components(data)
    create_main_dashboard(data)
    
    print("Visualization generation complete. Check 'visualizations' directory.")

if __name__ == "__main__":
    main()