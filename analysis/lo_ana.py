import re
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from datetime import datetime
from matplotlib.gridspec import GridSpec

# Configure matplotlib without explicit style
plt.rcParams.update({
    'figure.dpi': 300,
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'font.family': 'sans-serif',
    'font.weight': 'bold',
    'axes.grid': True,
    'grid.alpha': 0.3
})

class IDSLogAnalyzer:
    """Enhanced log analyzer with integrated data processing"""
    
    def __init__(self, log_path, traffic_path, alerts_path):
        self.log_path = os.path.expanduser(log_path)
        self.traffic_path = os.path.expanduser(traffic_path)
        self.alerts_path = os.path.expanduser(alerts_path)
        self._initialize_containers()
        
    def _initialize_containers(self):
        """Initialize data storage containers"""
        self.model_data = defaultdict(lambda: {'predictions': [], 'contributions': []})
        self.ensemble_probs = []
        self.timestamps = []
        self.log_stats = {'levels': defaultdict(int)}
        self.traffic_df = None
        self.alerts_df = None
        self.feature_importance = defaultdict(int)

    def analyze(self):
        """Main analysis workflow"""
        try:
            print(f"Analyzing log file: {self.log_path}")
            print(f"Traffic data path: {self.traffic_path}")
            print(f"Alerts data path: {self.alerts_path}")
            
            self._analyze_log_file()
            self._analyze_traffic_data()
            self._analyze_alerts()
            return self._compile_results()
        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            return None

    def _analyze_log_file(self):
        """Parse and analyze IDS log file"""
        if not os.path.exists(self.log_path):
            print(f"Log file not found: {self.log_path}")
            return

        log_pattern = re.compile(r'(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2},\d{3}).*?-\s(\w+).*?-\s(.*)')
        
        try:
            with open(self.log_path) as f:
                for line in f:
                    match = log_pattern.match(line)
                    if not match:
                        continue

                    ts, level, msg = match.groups()
                    self.log_stats['levels'][level] += 1
                    self._process_log_message(ts, msg)
        except Exception as e:
            print(f"Error reading log file: {str(e)}")

    def _process_log_message(self, timestamp, message):
        """Extract meaningful data from log messages"""
        try:
            # Model predictions
            if 'prediction:' in message:
                model_match = re.search(r'(\w+)\s+prediction:\s+([\d.]+)', message)
                if model_match:
                    model, val = model_match.groups()
                    self.model_data[model.lower()]['predictions'].append(float(val))

            # Model contributions
            if 'contribution:' in message:
                contrib_match = re.search(r'(\w+)\s+contribution:\s+([\d.]+)', message)
                if contrib_match:
                    model, val = contrib_match.groups()
                    self.model_data[model.lower()]['contributions'].append(float(val))

            # Ensemble probabilities
            if 'Final ensemble probability:' in message:
                prob_match = re.search(r'([\d.]+)$', message)
                if prob_match:
                    self.ensemble_probs.append(float(prob_match.group(1)))
                    self.timestamps.append(datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S,%f'))
        except Exception as e:
            print(f"Error processing log message: {str(e)}")

    def _analyze_traffic_data(self):
        """Load and process traffic data"""
        if not os.path.exists(self.traffic_path):
            print(f"Traffic data file not found: {self.traffic_path}")
            return

        try:
            self.traffic_df = pd.read_csv(self.traffic_path)
            if 'timestamp' in self.traffic_df.columns:
                self.traffic_df['timestamp'] = pd.to_datetime(self.traffic_df['timestamp'])
                self.traffic_df['hour'] = self.traffic_df['timestamp'].dt.floor('H')
        except Exception as e:
            print(f"Error loading traffic data: {str(e)}")

    def _analyze_alerts(self):
        """Process alert data and feature importance"""
        if not os.path.exists(self.alerts_path):
            print(f"Alerts file not found: {self.alerts_path}")
            return

        try:
            with open(self.alerts_path) as f:
                alerts = json.load(f)
                for alert in alerts:
                    if isinstance(alert, dict) and 'traffic_data' in alert:
                        for feat, val in alert['traffic_data'].items():
                            if isinstance(val, (int, float)) and val > 0:
                                self.feature_importance[feat] += 1
        except Exception as e:
            print(f"Error loading alerts: {str(e)}")

    def _compile_results(self):
        """Compile comprehensive analysis results"""
        try:
            return {
                'model_stats': self._model_statistics(),
                'ensemble_stats': self._ensemble_statistics(),
                'traffic_data': self.traffic_df,
                'feature_importance': dict(self.feature_importance),
                'log_stats': dict(self.log_stats['levels'])
            }
        except Exception as e:
            print(f"Error compiling results: {str(e)}")
            return None

    def _model_statistics(self):
        """Calculate model performance metrics"""
        try:
            return {
                model: {
                    'predictions': self._calc_stats(data['predictions']),
                    'contributions': self._calc_stats(data['contributions'])
                }
                for model, data in self.model_data.items()
            }
        except Exception as e:
            print(f"Error calculating model statistics: {str(e)}")
            return {}

    def _ensemble_statistics(self):
        """Calculate ensemble performance metrics"""
        try:
            if self.ensemble_probs:
                return self._calc_stats(self.ensemble_probs)
            return {}
        except Exception as e:
            print(f"Error calculating ensemble statistics: {str(e)}")
            return {}

    @staticmethod
    def _calc_stats(values):
        """Calculate basic statistics for a list of values"""
        if not values:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
        
        try:
            return {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        except Exception as e:
            print(f"Error calculating statistics: {str(e)}")
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0}

class IDSVisualizer:
    """Advanced visualization generator"""
    
    def __init__(self, analysis_results):
        self.results = analysis_results
        self.palette = sns.color_palette("husl", 10)

    def _model_performance_plot(self, save_path):
        """Model comparison visualization"""
        try:
            if not self.results.get('model_stats'):
                print("No model statistics available for plotting")
                return
                
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
            
            # Performance plot
            models = list(self.results['model_stats'].keys())
            means = [v['predictions']['mean'] for v in self.results['model_stats'].values()]
            stds = [v['predictions']['std'] for v in self.results['model_stats'].values()]
            
            performance_data = pd.DataFrame({
                'Model': models,
                'Mean': means,
                'Std': stds
            })
            
            # Updated barplot with hue parameter
            sns.barplot(data=performance_data, x='Model', y='Mean', 
                       hue='Model', legend=False, ax=ax1)
            ax1.errorbar(x=range(len(models)), y=means, yerr=stds, 
                        fmt='none', color='black', capsize=5)
            ax1.set_ylabel('Prediction Value', weight='bold')
            ax1.set_title('Model Prediction Performance', weight='bold')

            # Contribution plot
            contribs = [v['contributions']['mean'] for v in self.results['model_stats'].values()]
            contrib_data = pd.DataFrame({
                'Model': models,
                'Contribution': contribs
            })
            
            # Updated barplot with hue parameter
            sns.barplot(data=contrib_data, x='Model', y='Contribution',
                       hue='Model', legend=False, ax=ax2)
            ax2.set_ylabel('Contribution Weight', weight='bold')
            ax2.set_title('Model Contribution to Ensemble', weight='bold')

            plt.tight_layout()
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error generating model performance plot: {str(e)}")

    def _temporal_analysis_plot(self, save_path):
        """Temporal pattern visualization"""
        try:
            if self.results['traffic_data'] is None:
                print("No traffic data available for temporal analysis")
                return

            plt.figure(figsize=(14, 7))
            if 'timestamp' in self.results['traffic_data'].columns and 'label' in self.results['traffic_data'].columns:
                data = self.results['traffic_data'].copy()
                data['hour'] = pd.to_datetime(data['timestamp']).dt.floor('H')
                hourly_counts = data.groupby('hour')['label'].sum()
                
                plt.fill_between(hourly_counts.index, hourly_counts.values, 
                               alpha=0.3, color='red')
                plt.plot(hourly_counts.index, hourly_counts.values, 
                        color='red', alpha=0.7)
                plt.ylabel('Intrusion Count', weight='bold')
                plt.title('Hourly Intrusion Patterns', weight='bold')
                plt.grid(True, alpha=0.3)
                plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error generating temporal analysis plot: {str(e)}")

    def _feature_importance_plot(self, save_path):
        """Feature importance visualization"""
        try:
            if not self.results.get('feature_importance'):
                print("No feature importance data available")
                return
                
            features = sorted(self.results['feature_importance'].items(), 
                            key=lambda x: x[1], reverse=True)[:10]
            
            plt.figure(figsize=(12, 6))
            if features:
                plot_data = pd.DataFrame({
                    'Feature': [v[0] for v in features],
                    'Frequency': [v[1] for v in features]
                })
                # Updated barplot with explicit hue
                sns.barplot(data=plot_data, x='Frequency', y='Feature', 
                          hue='Feature', legend=False)
                plt.xlabel('Frequency in Alerts', weight='bold')
                plt.title('Top 10 Important Features in Intrusions', weight='bold')
                plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error generating feature importance plot: {str(e)}")

    def _traffic_comparison_plot(self, save_path):
        """Normal vs malicious traffic comparison"""
        try:
            if self.results['traffic_data'] is None:
                print("No traffic data available for comparison")
                return

            metrics = ['sbytes', 'dbytes', 'dur', 'sload']
            valid_metrics = [m for m in metrics if m in self.results['traffic_data'].columns]
            
            if valid_metrics:
                fig, axs = plt.subplots(2, 2, figsize=(14, 10))
                
                for ax, metric in zip(axs.flatten(), valid_metrics):
                    if 'label' in self.results['traffic_data'].columns:
                        data = self.results['traffic_data']
                        data['Type'] = data['label'].map({0: 'Normal', 1: 'Malicious'})
                        # Updated violinplot with hue
                        sns.violinplot(data=data, x='Type', y=metric,
                                     hue='Type', legend=False, ax=ax)
                        ax.set_title(f'{metric.upper()} Distribution', weight='bold')
                        ax.set_xlabel('')
                
                plt.tight_layout()
                plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error generating traffic comparison plot: {str(e)}")

    def _probability_distribution_plot(self, save_path):
        """Probability distribution analysis"""
        try:
            if not self.results.get('ensemble_stats'):
                print("No ensemble statistics available for distribution plot")
                return
                
            plt.figure(figsize=(12, 6))
            probs = self.results['ensemble_stats']
            if isinstance(probs, dict) and 'mean' in probs:
                data = pd.DataFrame({'Probability': [probs['mean']]})
                sns.histplot(data=data, x='Probability', bins=30, 
                           kde=True, color=self.palette[2])
            plt.axvline(0.5, color='red', linestyle='--', label='Decision Threshold')
            plt.xlabel('Ensemble Probability', weight='bold')
            plt.ylabel('Frequency', weight='bold')
            plt.title('Ensemble Probability Distribution', weight='bold')
            plt.legend(prop={'weight': 'bold'})
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error generating probability distribution plot: {str(e)}")

    def create_all_visuals(self, output_dir='visualization_results'):
        """Generate complete visualization suite"""
        os.makedirs(output_dir, exist_ok=True)
        print(f"Generating visualizations in: {output_dir}")
        
        if self.results is None:
            print("No results to visualize")
            return
        
        try:
            for plot_func, plot_name in [
                (self._model_performance_plot, 'model_performance.png'),
                (self._temporal_analysis_plot, 'temporal_analysis.png'),
                (self._feature_importance_plot, 'feature_importance.png'),
                (self._traffic_comparison_plot, 'traffic_comparison.png'),
                (self._probability_distribution_plot, 'probability_distribution.png')
            ]:
                try:
                    plot_func(os.path.join(output_dir, plot_name))
                except Exception as e:
                    print(f"Error generating {plot_name}: {str(e)}")
            
            print("Visualizations completed successfully")
        except Exception as e:
            print(f"Error in visualization generation: {str(e)}")

def main():
    """Main execution flow"""
    # Use expanduser to handle the ~ in the path
    base_dir = os.path.expanduser("~/recent_ids_modell/results/simulation_results")
    current_dir = os.path.expanduser("~/recent_ids_modell")
    
    print(f"Looking for simulation results in: {base_dir}")
    
    # Check if directories exist
    if not os.path.exists(base_dir):
        print(f"Simulation results directory not found: {base_dir}")
        return
        
    if not os.path.exists(current_dir):
        print(f"Project directory not found: {current_dir}")
        return
        
    # Get the latest simulation directory
    try:
        simulation_dirs = [d for d in os.listdir(base_dir) if d.startswith('simulation_')]
        if not simulation_dirs:
            print("No simulation results found!")
            return
            
        latest_sim = max(simulation_dirs)
        sim_dir = os.path.join(base_dir, latest_sim)
        print(f"Using simulation results from: {sim_dir}")
        
        # Setup paths
        log_path = os.path.join(current_dir, "ids_monitor.log")
        traffic_path = os.path.join(sim_dir, "traffic_data.csv")
        alerts_path = os.path.join(sim_dir, "alerts.json")
        
        # Verify all required files exist
        required_files = {
            'Log file': log_path,
            'Traffic data': traffic_path,
            'Alerts data': alerts_path
        }
        
        missing_files = []
        for file_desc, file_path in required_files.items():
            if not os.path.exists(file_path):
                missing_files.append(f"{file_desc}: {file_path}")
        
        if missing_files:
            print("Missing required files:")
            for missing in missing_files:
                print(f"- {missing}")
            return
            
        print("\nInitializing analysis with paths:")
        print(f"Log file: {log_path}")
        print(f"Traffic data: {traffic_path}")
        print(f"Alerts data: {alerts_path}\n")
        
        # Initialize analysis
        analyzer = IDSLogAnalyzer(
            log_path=log_path,
            traffic_path=traffic_path,
            alerts_path=alerts_path
        )
        
        # Run analysis
        print("Starting analysis...")
        results = analyzer.analyze()
        
        if results:
            # Create visualization directory
            viz_dir = os.path.join(sim_dir, 'visualizations')
            os.makedirs(viz_dir, exist_ok=True)
            
            print(f"\nGenerating visualizations in: {viz_dir}")
            
            # Generate visualizations
            visualizer = IDSVisualizer(results)
            visualizer.create_all_visuals(output_dir=viz_dir)
            print(f"\nAnalysis complete. Visualizations saved to: {viz_dir}")
        else:
            print("\nAnalysis failed to produce results")
            
    except Exception as e:
        print(f"\nError in main execution: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()