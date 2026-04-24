import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import matplotlib.dates as mdates
from datetime import datetime
import matplotlib.gridspec as gridspec

def convert_to_serializable(obj):
    """Convert numpy types to Python native types"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.complexfloating):
        return str(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif pd.isna(obj):
        return None
    return obj

class SimulationAnalyzer:
    def __init__(self, simulation_dir):
        """Initialize analyzer with simulation directory path"""
        self.sim_dir = Path(simulation_dir)
        self.traffic_data = pd.read_csv(self.sim_dir / 'traffic_data.csv')
        with open(self.sim_dir / 'alerts.json', 'r') as f:
            self.alerts = json.load(f)
            
        # Set up modern plotting style
        sns.set_theme(style="darkgrid")
        plt.rcParams.update({
            'font.size': 24,
            'font.weight': 'bold',
            'axes.labelweight': 'bold',
            'axes.titleweight': 'bold',
            'figure.dpi': 300,
            'figure.figsize': (15, 10),
            'lines.linewidth': 3,
            'axes.grid': True,
            'grid.linestyle': '--',
            'grid.alpha': 0.7
        })

    def analyze_traffic_patterns(self):
        """Generate comprehensive traffic pattern analysis plots"""
        try:
            # Create figure with subplots
            fig = plt.figure(figsize=(20, 25))
            gs = gridspec.GridSpec(3, 2, figure=fig)
            
            # 1. Traffic Volume Over Time
            ax1 = fig.add_subplot(gs[0, :])
            self.traffic_data.plot(x='id', y=['sbytes', 'dbytes'], 
                                 ax=ax1)
            ax1.set_title('Network Traffic Volume Over Time')
            ax1.set_xlabel('Packet ID')
            ax1.set_ylabel('Bytes')
            ax1.legend(['Source Bytes', 'Destination Bytes'])
            
            # 2. Protocol Distribution
            ax2 = fig.add_subplot(gs[1, 0])
            proto_cols = [col for col in self.traffic_data.columns if col.startswith('proto_')]
            if proto_cols:
                proto_usage = self.traffic_data[proto_cols].sum().sort_values(ascending=False)[:10]
                sns.barplot(x=proto_usage.index, y=proto_usage.values, ax=ax2)
                ax2.set_title('Top 10 Protocol Distribution')
                ax2.set_xlabel('Protocol')
                ax2.set_ylabel('Frequency')
                plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
            
            # 3. Service Distribution
            ax3 = fig.add_subplot(gs[1, 1])
            service_cols = [col for col in self.traffic_data.columns if col.startswith('service_')]
            if service_cols:
                service_usage = self.traffic_data[service_cols].sum().sort_values(ascending=False)
                sns.barplot(x=service_usage.index, y=service_usage.values, ax=ax3)
                ax3.set_title('Service Distribution')
                ax3.set_xlabel('Service')
                ax3.set_ylabel('Frequency')
                plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
            
            # 4. State Transition Analysis
            ax4 = fig.add_subplot(gs[2, 0])
            state_cols = [col for col in self.traffic_data.columns if col.startswith('state_')]
            if state_cols:
                state_transitions = self.traffic_data[state_cols].sum().sort_values(ascending=False)
                sns.barplot(x=state_transitions.index, y=state_transitions.values, ax=ax4)
                ax4.set_title('State Transitions')
                ax4.set_xlabel('State')
                ax4.set_ylabel('Frequency')
                plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
            
            # 5. Error Rate Analysis
            ax5 = fig.add_subplot(gs[2, 1])
            error_metrics = ['serror_rate', 'rerror_rate'] if 'serror_rate' in self.traffic_data.columns else []
            if error_metrics:
                self.traffic_data.plot(x='id', y=error_metrics, ax=ax5)
                ax5.set_title('Error Rates Over Time')
                ax5.set_xlabel('Packet ID')
                ax5.set_ylabel('Error Rate')
            
            plt.tight_layout()
            plt.savefig(self.sim_dir / 'traffic_analysis.png', bbox_inches='tight')
            plt.close()
            print("Traffic patterns analysis completed successfully")
        except Exception as e:
            print(f"Error in traffic patterns analysis: {str(e)}")

    def analyze_intrusion_detection(self):
        """Analyze intrusion detection results"""
        try:
            if self.alerts:
                # Convert alerts timestamps to datetime
                alert_times = [datetime.fromtimestamp(alert['timestamp']) for alert in self.alerts]
                probabilities = [alert['probability'] for alert in self.alerts]
                
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 20))
                
                # Alert Timeline
                ax1.plot(alert_times, probabilities, 'ro-')
                ax1.set_title('Intrusion Detection Timeline')
                ax1.set_xlabel('Time')
                ax1.set_ylabel('Detection Probability')
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                
                # Probability Distribution
                sns.histplot(probabilities, bins=20, ax=ax2)
                ax2.set_title('Distribution of Detection Probabilities')
                ax2.set_xlabel('Probability')
                ax2.set_ylabel('Frequency')
                
                plt.tight_layout()
                plt.savefig(self.sim_dir / 'intrusion_analysis.png', bbox_inches='tight')
                plt.close()
                print("Intrusion detection analysis completed successfully")
        except Exception as e:
            print(f"Error in intrusion detection analysis: {str(e)}")

    def analyze_encryption_overhead(self):
        """Analyze encryption performance metrics"""
        try:
            if 'encryption_time' in self.traffic_data.columns:
                fig, ax = plt.subplots(figsize=(15, 10))
                
                total_bytes = self.traffic_data['sbytes'] + self.traffic_data['dbytes']
                ax.scatter(total_bytes, self.traffic_data['encryption_time'],
                          alpha=0.5)
                ax.set_title('Encryption Overhead Analysis')
                ax.set_xlabel('Total Bytes')
                ax.set_ylabel('Encryption Time (ms)')
                
                plt.tight_layout()
                plt.savefig(self.sim_dir / 'encryption_analysis.png', bbox_inches='tight')
                plt.close()
                print("Encryption overhead analysis completed successfully")
        except Exception as e:
            print(f"Error in encryption overhead analysis: {str(e)}")

    def generate_summary_statistics(self):
        """Generate summary statistics"""
        try:
            raw_summary = {
                'total_packets': len(self.traffic_data),
                'total_alerts': len(self.alerts),
                'alert_rate': float(len(self.alerts) / len(self.traffic_data) * 100) if len(self.traffic_data) > 0 else 0,
                'avg_detection_probability': float(np.mean([alert['probability'] for alert in self.alerts])) if self.alerts else 0,
                'total_traffic_volume': int(self.traffic_data['sbytes'].sum() + self.traffic_data['dbytes'].sum())
            }
            
            # Add additional traffic statistics
            if 'label' in self.traffic_data.columns:
                raw_summary['intrusion_packets'] = int(self.traffic_data['label'].sum())
                raw_summary['normal_packets'] = int(len(self.traffic_data) - raw_summary['intrusion_packets'])
            
            # Convert all values to JSON serializable types
            summary = {k: convert_to_serializable(v) for k, v in raw_summary.items()}
            
            with open(self.sim_dir / 'analysis_summary.json', 'w') as f:
                json.dump(summary, f, indent=4)
            
            print("Summary statistics generated successfully")
            return summary
            
        except Exception as e:
            print(f"Error generating summary statistics: {str(e)}")
            return {}

def main():
    try:
        # Initialize analyzer with the correct path
        analyzer = SimulationAnalyzer('/home/seyitope/recent_ids_model/results/simulation_results/simulation_real')
        
        # Run analysis
        analyzer.analyze_traffic_patterns()
        analyzer.analyze_intrusion_detection()
        analyzer.analyze_encryption_overhead()
        summary = analyzer.generate_summary_statistics()
        
        # Print summary
        if summary:
            print("\nAnalysis Summary:")
            for key, value in summary.items():
                print(f"{key}: {value}")
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main()