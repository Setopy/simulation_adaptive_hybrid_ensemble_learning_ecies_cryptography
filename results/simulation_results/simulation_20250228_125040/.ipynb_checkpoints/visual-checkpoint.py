import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
from pathlib import Path

# Set global plotting parameters for professional publication
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 20
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['figure.titlesize'] = 24
plt.rcParams['figure.titleweight'] = 'bold'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['lines.linewidth'] = 2.5

# Define a professional color palette
colors = {
    'cnn': '#1f77b4',      # Blue
    'lstm': '#ff7f0e',     # Orange
    'dnn': '#2ca02c',      # Green
    'svm': '#d62728',      # Red
    'xgboost': '#9467bd',  # Purple
    'randomforest': '#8c564b',  # Brown
    'primary': '#0072B2',
    'secondary': '#E69F00',
    'highlight': '#D55E00',
    'neutral': '#56B4E9',
    'background': '#F0F0F0',
    'success': '#009E73',
    'alert': '#CC79A7'
}

print("Loading data files...")

# Load the data from JSON files
with open('ensemble_metrics.json', 'r') as f:
    ensemble_metrics = json.load(f)

with open('crypto_metrics.json', 'r') as f:
    crypto_metrics = json.load(f)

with open('feature_importances.json', 'r') as f:
    feature_importances = json.load(f)

with open('alerts.json', 'r') as f:
    alerts = json.load(f)

# Load traffic data from CSV
traffic_data = pd.read_csv('traffic_data.csv')

# Create output directory for visualizations
output_dir = Path('visualizations')
output_dir.mkdir(exist_ok=True)

# Extract key metrics
model_names = list(ensemble_metrics['weights'].keys())
model_weights = list(ensemble_metrics['weights'].values())
encrypted_times = crypto_metrics['encryption']['average_time']
decrypted_times = crypto_metrics['decryption']['average_time']
component_sizes = crypto_metrics['component_sizes']

# Feature mapping based on CONFIG
feature_mapping = {
    'feature_0': 'id',
    'feature_1': 'dur',
    'feature_2': 'spkts',
    'feature_3': 'dpkts',
    'feature_4': 'sbytes',
    'feature_5': 'dbytes',
    'feature_6': 'rate',
    'feature_7': 'sttl',
    'feature_8': 'dttl',
    'feature_9': 'sload',
    'feature_10': 'dload',
    'feature_11': 'sloss',
    'feature_12': 'dloss',
    'feature_13': 'sinpkt',
    'feature_14': 'dinpkt',
    'feature_15': 'sjit',
    'feature_16': 'djit',
    'feature_17': 'swin',
    'feature_18': 'stcpb',
    'feature_19': 'dtcpb',
    'feature_20': 'dwin',
    'feature_21': 'tcprtt',
    'feature_22': 'synack',
    'feature_23': 'ackdat',
    'feature_24': 'smean',
    'feature_25': 'dmean',
    'feature_26': 'trans_depth',
    'feature_27': 'response_body_len',
    'feature_28': 'ct_srv_src',
    'feature_29': 'ct_state_ttl',
    'feature_30': 'ct_dst_ltm',
    'feature_31': 'ct_src_dport_ltm',
    'feature_32': 'ct_dst_sport_ltm',
    'feature_33': 'ct_dst_src_ltm',
    'feature_34': 'is_ftp_login',
    'feature_35': 'ct_ftp_cmd',
    'feature_36': 'ct_flw_http_mthd',
    'feature_37': 'ct_src_ltm',
    'feature_38': 'ct_srv_dst',
    'feature_39': 'is_sm_ips_ports',
    'feature_40': 'label',
    'feature_41': 'proto_tcp',
    'feature_42': 'proto_udp',
    'feature_43': 'proto_arp',
    'feature_44': 'proto_ospf',
    'feature_45': 'proto_icmp',
    'feature_46': 'proto_igmp',
    'feature_47': 'proto_rtp',
    'feature_48': 'proto_ddp',
    'feature_49': 'proto_ipv6-frag',
    'feature_50': 'proto_cftp',
    'feature_51': 'proto_wsn',
    'feature_52': 'proto_pvp',
    'feature_53': 'proto_wb-expak',
    'feature_54': 'proto_mtp',
    'feature_55': 'proto_pri-enc',
    'feature_56': 'proto_sat-mon',
    'feature_57': 'proto_cphb',
    'feature_58': 'proto_sun-nd',
    'feature_59': 'proto_iso-ip',
    'feature_60': 'proto_xtp',
    'feature_61': 'proto_il',
    'feature_62': 'proto_unas',
    'feature_63': 'proto_mfe-nsp',
    'feature_64': 'proto_3pc',
    'feature_65': 'proto_ipv6-route',
    'feature_66': 'proto_idrp',
    'feature_67': 'proto_bna',
    'feature_68': 'proto_swipe',
    'feature_69': 'proto_kryptolan',
    'feature_70': 'proto_cpnx',
    'feature_71': 'proto_rsvp',
    'feature_72': 'proto_wb-mon',
    'feature_73': 'proto_vmtp',
    'feature_74': 'proto_ib',
    'feature_75': 'proto_dgp',
    'feature_76': 'proto_eigrp',
    'feature_77': 'proto_ax.25',
    'feature_78': 'proto_gmtp',
    'feature_79': 'proto_pnni',
    'feature_80': 'proto_sep',
    'feature_81': 'proto_pgm',
    'feature_82': 'proto_idpr-cmtp',
    'feature_83': 'proto_zero',
    'feature_84': 'proto_rvd',
    'feature_85': 'proto_mobile',
    'feature_86': 'proto_narp',
    'feature_87': 'proto_fc',
    'feature_88': 'proto_pipe',
    'feature_89': 'proto_ipcomp',
    'feature_90': 'proto_ipv6-no',
    'feature_91': 'proto_sat-expak',
    'feature_92': 'proto_ipv6-opts',
    'feature_93': 'proto_snp',
    'feature_94': 'proto_ipcv',
    'feature_95': 'proto_br-sat-mon',
    'feature_96': 'proto_ttp',
    'feature_97': 'proto_tcf',
    'feature_98': 'proto_nsfnet-igp',
    'feature_99': 'proto_sprite-rpc',
    'feature_100': 'proto_aes-sp3-d',
    'feature_101': 'proto_sccopmce',
    'feature_102': 'proto_sctp',
    'feature_103': 'proto_qnx',
    'feature_104': 'proto_scps',
    'feature_105': 'proto_etherip',
    'feature_106': 'proto_aris',
    'feature_107': 'proto_pim',
    'feature_108': 'proto_compaq-peer',
    'feature_109': 'proto_vrrp',
    'feature_110': 'proto_iatp',
    'feature_111': 'proto_stp',
    'feature_112': 'proto_l2tp',
    'feature_113': 'proto_srp',
    'feature_114': 'proto_sm',
    'feature_115': 'proto_isis',
    'feature_116': 'proto_smp',
    'feature_117': 'proto_fire',
    'feature_118': 'proto_ptp',
    'feature_119': 'proto_crtp',
    'feature_120': 'proto_sps',
    'feature_121': 'proto_merit-inp',
    'feature_122': 'proto_idpr',
    'feature_123': 'proto_skip',
    'feature_124': 'proto_any',
    'feature_125': 'proto_larp',
    'feature_126': 'proto_ipip',
    'feature_127': 'proto_micp',
    'feature_128': 'proto_encap',
    'feature_129': 'proto_ifmp',
    'feature_130': 'proto_tp++',
    'feature_131': 'proto_a/n',
    'feature_132': 'proto_ipv6',
    'feature_133': 'proto_i-nlsp',
    'feature_134': 'proto_ipx-n-ip',
    'feature_135': 'proto_sdrp',
    'feature_136': 'proto_tlsp',
    'feature_137': 'proto_gre',
    'feature_138': 'proto_mhrp',
    'feature_139': 'proto_ddx',
    'feature_140': 'proto_ippc',
    'feature_141': 'proto_visa',
    'feature_142': 'proto_secure-vmtp',
    'feature_143': 'proto_uti',
    'feature_144': 'proto_vines',
    'feature_145': 'proto_crudp',
    'feature_146': 'proto_iplt',
    'feature_147': 'proto_ggp',
    'feature_148': 'proto_ip',
    'feature_149': 'proto_ipnip',
    'feature_150': 'proto_st2',
    'feature_151': 'proto_argus',
    'feature_152': 'proto_bbn-rcc',
    'feature_153': 'proto_egp',
    'feature_154': 'proto_emcon',
    'feature_155': 'proto_igp',
    'feature_156': 'proto_nvp',
    'feature_157': 'proto_pup',
    'feature_158': 'proto_xnet',
    'feature_159': 'proto_chaos',
    'feature_160': 'proto_mux',
    'feature_161': 'proto_dcn',
    'feature_162': 'proto_hmp',
    'feature_163': 'proto_prm',
    'feature_164': 'proto_trunk-1',
    'feature_165': 'proto_xns-idp',
    'feature_166': 'proto_leaf-1',
    'feature_167': 'proto_leaf-2',
    'feature_168': 'proto_rdp',
    'feature_169': 'proto_irtp',
    'feature_170': 'proto_iso-tp4',
    'feature_171': 'proto_netblt',
    'feature_172': 'proto_trunk-2',
    'feature_173': 'proto_cbt',
    'feature_174': 'service_-',
    'feature_175': 'service_ftp',
    'feature_176': 'service_smtp',
    'feature_177': 'service_snmp',
    'feature_178': 'service_http',
    'feature_179': 'service_ftp-data',
    'feature_180': 'service_dns',
    'feature_181': 'service_ssh',
    'feature_182': 'service_radius',
    'feature_183': 'service_pop3',
    'feature_184': 'service_dhcp',
    'feature_185': 'service_ssl',
    'feature_186': 'service_irc',
    'feature_187': 'state_FIN',
    'feature_188': 'state_INT',
    'feature_189': 'state_CON',
    'feature_190': 'state_ECO',
    'feature_191': 'state_REQ',
    'feature_192': 'state_RST',
    'feature_193': 'state_PAR',
    'feature_194': 'state_URN',
    'feature_195': 'state_no'
}

# Create human-readable feature descriptions
feature_descriptions={
    'id': 'Identifier',
    'dur': 'Duration',
    'spkts': 'Source Packets',
    'dpkts': 'Destination Packets',
    'sbytes': 'Source Bytes',
    'dbytes': 'Destination Bytes',
    'rate': 'Rate',
    'sttl': 'Source TTL',
    'dttl': 'Destination TTL',
    'sload': 'Source Load',
    'dload': 'Destination Load',
    'sloss': 'Source Loss',
    'dloss': 'Destination Loss',
    'sinpkt': 'Source Inter-packet Time',
    'dinpkt': 'Destination Inter-packet Time',
    'sjit': 'Source Jitter',
    'djit': 'Destination Jitter',
    'swin': 'Source Window',
    'stcpb': 'Source TCP Base Sequence Number',
    'dtcpb': 'Destination TCP Base Sequence Number',
    'dwin': 'Destination Window',
    'tcprtt': 'TCP Round Trip Time',
    'synack': 'SYN-ACK Time',
    'ackdat': 'ACK-Data Time',
    'smean': 'Source Mean Packet Size',
    'dmean': 'Destination Mean Packet Size',
    'trans_depth': 'Transaction Depth',
    'response_body_len': 'Response Body Length',
    'ct_srv_src': 'Count of Connections to Same Service from Source',
    'ct_state_ttl': 'Count of Connections with Same State and TTL',
    'ct_dst_ltm': 'Count of Connections to Destination in Last Time Window',
    'ct_src_dport_ltm': 'Count of Connections from Source to Destination Port in Last Time Window',
    'ct_dst_sport_ltm': 'Count of Connections from Destination to Source Port in Last Time Window',
    'ct_dst_src_ltm': 'Count of Connections from Destination to Source in Last Time Window',
    'is_ftp_login': 'Is FTP Login',
    'ct_ftp_cmd': 'Count of FTP Commands',
    'ct_flw_http_mthd': 'Count of Flows with HTTP Method',
    'ct_src_ltm': 'Count of Connections from Source in Last Time Window',
    'ct_srv_dst': 'Count of Connections to Same Service from Destination',
    'is_sm_ips_ports': 'Is Source and Destination IP and Ports Same',
    'label': 'Label',
    'proto_tcp': 'TCP Protocol',
    'proto_udp': 'UDP Protocol',
    'proto_arp': 'ARP Protocol',
    'proto_ospf': 'OSPF Protocol',
    'proto_icmp': 'ICMP Protocol',
    'proto_igmp': 'IGMP Protocol',
    'proto_rtp': 'RTP Protocol',
    'proto_ddp': 'DDP Protocol',
    'proto_ipv6-frag': 'IPv6 Fragment Protocol',
    'proto_cftp': 'CFTP Protocol',
    'proto_wsn': 'WSN Protocol',
    'proto_pvp': 'PVP Protocol',
    'proto_wb-expak': 'WB-EXPAK Protocol',
    'proto_mtp': 'MTP Protocol',
    'proto_pri-enc': 'PRI-ENC Protocol',
    'proto_sat-mon': 'SAT-MON Protocol',
    'proto_cphb': 'CPHB Protocol',
    'proto_sun-nd': 'SUN-ND Protocol',
    'proto_iso-ip': 'ISO-IP Protocol',
    'proto_xtp': 'XTP Protocol',
    'proto_il': 'IL Protocol',
    'proto_unas': 'UNAS Protocol',
    'proto_mfe-nsp': 'MFE-NSP Protocol',
    'proto_3pc': '3PC Protocol',
    'proto_ipv6-route': 'IPv6 Route Protocol',
    'proto_idrp': 'IDRP Protocol',
    'proto_bna': 'BNA Protocol',
    'proto_swipe': 'SWIPE Protocol',
    'proto_kryptolan': 'KRYPTOLAN Protocol',
    'proto_cpnx': 'CPNX Protocol',
    'proto_rsvp': 'RSVP Protocol',
    'proto_wb-mon': 'WB-MON Protocol',
    'proto_vmtp': 'VMTP Protocol',
    'proto_ib': 'IB Protocol',
    'proto_dgp': 'DGP Protocol',
    'proto_eigrp': 'EIGRP Protocol',
    'proto_ax.25': 'AX.25 Protocol',
    'proto_gmtp': 'GMTP Protocol',
    'proto_pnni': 'PNNI Protocol',
    'proto_sep': 'SEP Protocol',
    'proto_pgm': 'PGM Protocol',
    'proto_idpr-cmtp': 'IDPR-CMTP Protocol',
    'proto_zero': 'ZERO Protocol',
    'proto_rvd': 'RVD Protocol',
    'proto_mobile': 'MOBILE Protocol',
    'proto_narp': 'NARP Protocol',
    'proto_fc': 'FC Protocol',
    'proto_pipe': 'PIPE Protocol',
    'proto_ipcomp': 'IPCOMP Protocol',
    'proto_ipv6-no': 'IPv6-NO Protocol',
    'proto_sat-expak': 'SAT-EXPAK Protocol',
    'proto_ipv6-opts': 'IPv6 Options Protocol',
    'proto_snp': 'SNP Protocol',
    'proto_ipcv': 'IPCV Protocol',
    'proto_br-sat-mon': 'BR-SAT-MON Protocol',
    'proto_ttp': 'TTP Protocol',
    'proto_tcf': 'TCF Protocol',
    'proto_nsfnet-igp': 'NSFNET-IGP Protocol',
    'proto_sprite-rpc': 'SPRITE-RPC Protocol',
    'proto_aes-sp3-d': 'AES-SP3-D Protocol',
    'proto_sccopmce': 'SCCOPMCE Protocol',
    'proto_sctp': 'SCTP Protocol',
    'proto_qnx': 'QNX Protocol',
    'proto_scps': 'SCPS Protocol',
    'proto_etherip': 'ETHERIP Protocol',
    'proto_aris': 'ARIS Protocol',
    'proto_pim': 'PIM Protocol',
    'proto_compaq-peer': 'COMPAQ-PEER Protocol',
    'proto_vrrp': 'VRRP Protocol',
    'proto_iatp': 'IATP Protocol',
    'proto_stp': 'STP Protocol',
    'proto_l2tp': 'L2TP Protocol',
    'proto_srp': 'SRP Protocol',
    'proto_sm': 'SM Protocol',
    'proto_isis': 'ISIS Protocol',
    'proto_smp': 'SMP Protocol',
    'proto_fire': 'FIRE Protocol',
    'proto_ptp': 'PTP Protocol',
    'proto_crtp': 'CRTP Protocol',
    'proto_sps': 'SPS Protocol',
    'proto_merit-inp': 'MERIT-INP Protocol',
    'proto_idpr': 'IDPR Protocol',
    'proto_skip': 'SKIP Protocol',
    'proto_any': 'ANY Protocol',
    'proto_larp': 'LARP Protocol',
    'proto_ipip': 'IPIP Protocol',
    'proto_micp': 'MICP Protocol',
    'proto_encap': 'ENCAP Protocol',
    'proto_ifmp': 'IFMP Protocol',
    'proto_tp++': 'TP++ Protocol',
    'proto_a/n': 'A/N Protocol',
    'proto_ipv6': 'IPv6 Protocol',
    'proto_i-nlsp': 'I-NLSP Protocol',
    'proto_ipx-n-ip': 'IPX-N-IP Protocol',
    'proto_sdrp': 'SDRP Protocol',
    'proto_tlsp': 'TLSP Protocol',
    'proto_gre': 'GRE Protocol',
    'proto_mhrp': 'MHRP Protocol',
    'proto_ddx': 'DDX Protocol',
    'proto_ippc': 'IPPC Protocol',
    'proto_visa': 'VISA Protocol',
    'proto_secure-vmtp': 'SECURE-VMTP Protocol',
    'proto_uti': 'UTI Protocol',
    'proto_vines': 'VINES Protocol',
    'proto_crudp': 'CRUDP Protocol',
    'proto_iplt': 'IPLT Protocol',
    'proto_ggp': 'GGP Protocol',
    'proto_ip': 'IP Protocol',
    'proto_ipnip': 'IPNIP Protocol',
    'proto_st2': 'ST2 Protocol',
    'proto_argus': 'ARGUS Protocol',
    'proto_bbn-rcc': 'BBN-RCC Protocol',
    'proto_egp': 'EGP Protocol',
    'proto_emcon': 'EMCON Protocol',
    'proto_igp': 'IGP Protocol',
    'proto_nvp': 'NVP Protocol',
    'proto_pup': 'PUP Protocol',
    'proto_xnet': 'XNET Protocol',
    'proto_chaos': 'CHAOS Protocol',
    'proto_mux': 'MUX Protocol',
    'proto_dcn': 'DCN Protocol',
    'proto_hmp': 'HMP Protocol',
    'proto_prm': 'PRM Protocol',
    'proto_trunk-1': 'TRUNK-1 Protocol',
    'proto_xns-idp': 'XNS-IDP Protocol',
    'proto_leaf-1': 'LEAF-1 Protocol',
    'proto_leaf-2': 'LEAF-2 Protocol',
    'proto_rdp': 'RDP Protocol',
    'proto_irtp': 'IRTP Protocol',
    'proto_iso-tp4': 'ISO-TP4 Protocol',
    'proto_netblt': 'NETBLT Protocol',
    'proto_trunk-2': 'TRUNK-2 Protocol',
    'proto_cbt': 'CBT Protocol',
    'service_-': 'No Service',
    'service_ftp': 'File Transfer Protocol Service',
    'service_smtp': 'Simple Mail Transfer Protocol Service',
    'service_snmp': 'Simple Network Management Protocol Service',
    'service_http': 'Hypertext Transfer Protocol Service',
    'service_ftp-data': 'FTP Data Service',
    'service_dns': 'Domain Name System Service',
    'service_ssh': 'Secure Shell Service',
    'service_radius': 'Remote Authentication Dial-In User Service',
    'service_pop3': 'Post Office Protocol 3 Service',
    'service_dhcp': 'Dynamic Host Configuration Protocol Service',
    'service_ssl': 'Secure Sockets Layer Service',
    'service_irc': 'Internet Relay Chat Service',
    'state_FIN': 'Finished State',
    'state_INT': 'Intermediate State',
    'state_CON': 'Connection State',
    'state_ECO': 'Echo State',
    'state_REQ': 'Request State',
    'state_RST': 'Reset State',
    'state_PAR': 'Partial State',
    'state_URN': 'Urgent State',
    'state_no': 'No State'
}

# 1. Ensemble Weight Distribution and Model Performance Visualization
def plot_model_performance():
    print("Creating model performance visualization...")
    fig = plt.figure(figsize=(20, 15))
    
    # Use GridSpec for complex layout
    gs = GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[3, 2])
    
    # Extract model performance data
    model_perfs = {}
    for model in model_names:
        if model in ensemble_metrics['model_performance']:
            perf = ensemble_metrics['model_performance'][model]
            if 'predictions' in perf and 'mean' in perf['predictions']:
                model_perfs[model] = {
                    'mean': perf['predictions']['mean'],
                    'std': perf['predictions']['std'],
                    'weight': ensemble_metrics['weights'][model]
                }
    
    # Plot 1: Model Weights
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(model_names, 
                  [ensemble_metrics['weights'][m] for m in model_names],
                  color=[colors[m] for m in model_names],
                  alpha=0.8)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    ax1.set_title('Ensemble Model Weight Distribution')
    ax1.set_ylabel('Weight')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    ax1.set_ylim(0, max([ensemble_metrics['weights'][m] for m in model_names]) * 1.2)
    
    # Plot 2: Mean Predictions with Error Bars
    ax2 = fig.add_subplot(gs[1, 0])
    
    # Extract model means and stds
    means = []
    stds = []
    for model in model_names:
        if model in model_perfs:
            means.append(model_perfs[model]['mean'])
            stds.append(model_perfs[model]['std'])
        else:
            # Use average if no specific data
            means.append(ensemble_metrics.get('average_confidence', 0.5))
            stds.append(ensemble_metrics.get('confidence_std', 0.1))
    
    bars = ax2.bar(model_names, means, 
                  yerr=stds, 
                  capsize=10, 
                  color=[colors[m] for m in model_names],
                  alpha=0.8)
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax2.annotate(f'{means[i]:.3f}±{stds[i]:.3f}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    ax2.set_title('Model Prediction Confidence')
    ax2.set_ylabel('Mean Prediction Value')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    ax2.set_ylim(0, 1.2)
    
    # Plot 3: Pie chart of distribution
    ax3 = fig.add_subplot(gs[0, 1])
    
    # Use weights for the pie chart
    weights = [ensemble_metrics['weights'][m] for m in model_names]
    
    wedges, texts, autotexts = ax3.pie(
        weights,
        labels=model_names,
        autopct='%1.1f%%',
        startangle=90,
        colors=[colors[m] for m in model_names],
        wedgeprops={'edgecolor': 'w', 'linewidth': 1},
        textprops={'fontweight': 'bold'}
    )
    
    # Make text bold and visible
    for text in texts:
        text.set_fontweight('bold')
    for autotext in autotexts:
        autotext.set_fontweight('bold')
        autotext.set_color('white')
    
    ax3.set_title('Relative Contribution to Ensemble')
    
    # Plot 4: F1/Precision/Recall Metrics History
    ax4 = fig.add_subplot(gs[1, 1])
    metrics_history = ensemble_metrics['metrics_history']
    
    f1_values = metrics_history.get('f1', [])
    precision_values = metrics_history.get('precision', [])
    recall_values = metrics_history.get('recall', [])
    
    # Only plot if we have data
    if f1_values and precision_values and recall_values:
        x = range(len(f1_values))
        ax4.plot(x, f1_values, 'o-', label='F1', color='#ff9a00')
        ax4.plot(x, precision_values, 's-', label='Precision', color='#00a3e0')
        ax4.plot(x, recall_values, '^-', label='Recall', color='#b80058')
        
        ax4.set_title('Performance Metrics Evolution')
        ax4.set_xlabel('Adaptation Step')
        ax4.set_ylabel('Metric Value')
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, "No metrics history available", 
                ha='center', va='center', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ensemble_performance.png', bbox_inches='tight')
    plt.close()

# 2. Cryptographic Performance Metrics
# Fix for the crypto_performance function
def plot_crypto_performance():
    print("Creating cryptographic performance visualization...")
    fig = plt.figure(figsize=(20, 15))
    gs = GridSpec(2, 2)
    
    # Plot 1: Operation Time Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    operations = ['Encryption', 'Decryption', 'Key Operations']
    times = [
        crypto_metrics['encryption']['average_time'],
        crypto_metrics['decryption']['average_time'],
        crypto_metrics['key_operations']['average_time']
    ]
    
    # Calculate standard deviations if available
    stds = [
        crypto_metrics['encryption'].get('std_time', 0),
        crypto_metrics['decryption'].get('std_time', 0),
        0  # Key operations might not have std
    ]
    
    bars = ax1.bar(operations, times, yerr=stds, capsize=10, 
                  color=['#3274A1', '#E1812C', '#3A923A'])
    
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height:.6f} ms',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    ax1.set_title('Cryptographic Operation Time')
    ax1.set_ylabel('Time (ms)')
    
    # Plot 2: Component Size Analysis
    ax2 = fig.add_subplot(gs[0, 1])
    components = list(component_sizes.keys())
    sizes = list(component_sizes.values())
    
    # Calculate total overhead
    total_overhead = sum(sizes)
    
    bars = ax2.barh(components, sizes, color=sns.color_palette("Blues", len(components)))
    
    for bar in bars:
        width = bar.get_width()
        ax2.annotate(f'{width} bytes',
                    xy=(width, bar.get_y() + bar.get_height()/2),
                    xytext=(3, 0),
                    textcoords="offset points",
                    ha='left', va='center')
    
    ax2.set_title('Cryptographic Component Sizes')
    ax2.set_xlabel('Size (bytes)')
    
    # Add annotation for total overhead
    ax2.text(0.95, 0.05, f'Total Overhead: {total_overhead} bytes',
             transform=ax2.transAxes,
             fontsize=16, fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Plot 3: Encryption Overhead Distribution
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Check if we can extract overhead values directly
    overhead_bytes = []
    if 'overhead_bytes' in crypto_metrics['encryption']:
        overhead_bytes = crypto_metrics['encryption']['overhead_bytes']
    
    # Fix: Check if overhead_bytes exists and is non-empty using appropriate array check
    if isinstance(overhead_bytes, list) and len(overhead_bytes) > 0:
        # Create histogram from the list data
        n, bins, patches = ax3.hist(overhead_bytes, 
                                  bins=30, 
                                  color='#3274A1', 
                                  alpha=0.7, 
                                  edgecolor='black')
        
        # Add vertical line for the average
        avg = np.mean(overhead_bytes)
        ax3.axvline(x=avg, color='red', linestyle='--', linewidth=2)
        ax3.text(avg*1.05, max(n)*0.95, f'Mean: {avg:.1f} bytes',
                fontweight='bold', color='red')
    else:
        # Use average_overhead to create a synthetic distribution
        avg_overhead = crypto_metrics['encryption'].get('average_overhead', 125)
        if avg_overhead > 0:
            x = np.linspace(avg_overhead*0.5, avg_overhead*1.5, 1000)
            std = avg_overhead * 0.1  # Assuming 10% standard deviation
            y = np.exp(-0.5 * ((x - avg_overhead) / std) ** 2) / (std * np.sqrt(2 * np.pi))
            
            ax3.plot(x, y, color='#3274A1', linewidth=3)
            ax3.fill_between(x, y, color='#3274A1', alpha=0.3)
            
            # Add vertical line for the average
            ax3.axvline(x=avg_overhead, color='red', linestyle='--', linewidth=2)
            ax3.text(avg_overhead*1.05, max(y)*0.95, f'Mean: {avg_overhead:.1f} bytes',
                    fontweight='bold', color='red')
        else:
            ax3.text(0.5, 0.5, "No overhead data available", 
                    ha='center', va='center', fontsize=16)
    
    ax3.set_title('Encryption Overhead Distribution')
    ax3.set_xlabel('Overhead (bytes)')
    ax3.set_ylabel('Frequency')
    
    # Plot 4: Success Rate Metrics
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Get success rates
    e_success = crypto_metrics['encryption'].get('success_rate', 1.0) * 100
    d_success = crypto_metrics['decryption'].get('success_rate', 1.0) * 100
    auth_failure = crypto_metrics['decryption'].get('auth_failure_rate', 0.0) * 100
    
    # Create data with appropriate labels
    labels = ['Encryption\nSuccess', 'Decryption\nSuccess']
    sizes = [e_success, d_success]
    colors = ['#4CAF50', '#2196F3']
    
    # Add authentication failures if they exist
    if auth_failure > 0.01:
        labels.append('Auth\nFailures')
        sizes.append(auth_failure)
        colors.append('#F44336')
    
    explode = tuple(0.05 for _ in range(len(labels)))
    
    ax4.pie(
        sizes,
        explode=explode,
        labels=labels,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        shadow=True,
        textprops={'fontweight': 'bold'}
    )
    
    ax4.set_title('Cryptographic Operation Success Rates')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'crypto_performance.png', bbox_inches='tight')
    plt.close()

# 3. Feature Importance Analysis with proper mapping
def plot_feature_importance():
    print("Creating feature importance visualization...")
    fig = plt.figure(figsize=(20, 20))
    
    # Get all models with feature importance data
    models = list(feature_importances.keys())
    
    # Number of top features to display per model
    n_top = 8
    
    # Calculate how many models we have to create subplot grid
    n_models = len(models)
    
    for i, model in enumerate(models):
        ax = fig.add_subplot(n_models, 1, i+1)
        
        # Get the feature importances for this model
        if len(feature_importances[model]) >= n_top:
            model_importances = feature_importances[model][:n_top]
        else:
            model_importances = feature_importances[model]
            
        # Extract feature names and importance scores
        features = []
        scores = []
        
        for feat, score in model_importances:
            # Map feature to meaningful name
            display_name = feat
            
            # Check if it's a feature index that needs to be mapped
            if feat in feature_mapping:
                raw_name = feature_mapping[feat]
                # Use human-readable description if available
                if raw_name in feature_descriptions:
                    display_name = feature_descriptions[raw_name]
                else:
                    display_name = raw_name
            
            features.append(display_name)
            scores.append(score)
            
        # Reverse for bottom-to-top display
        features.reverse()
        scores.reverse()
        
        # Create horizontal bar chart
        bars = ax.barh(features, scores, color=colors.get(model, 'gray'), alpha=0.8)
        
        # Add importance values
        for bar in bars:
            width = bar.get_width()
            ax.text(width * 1.01, bar.get_y() + bar.get_height()/2,
                   f'{width:.4f}', va='center', fontweight='bold')
        
        ax.set_title(f'{model.upper()} - Top {len(features)} Important Features')
        ax.set_xlabel('Importance Score')
        
        # Set x-axis limit based on max score with padding for labels
        ax.set_xlim(0, max(scores) * 1.15)
        
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance.png', bbox_inches='tight')
    plt.close()
    
    # Also create a combined feature importance visualization
    plot_combined_feature_importance()

# Create a combined feature importance chart
def plot_combined_feature_importance():
    print("Creating combined feature importance visualization...")
    
    # Get all feature importances and combine
    all_features = {}
    for model, importances in feature_importances.items():
        weight = ensemble_metrics['weights'].get(model, 1.0/len(feature_importances))
        for feature, score in importances:
            if feature not in all_features:
                all_features[feature] = 0
            # Weight the importance by model weight
            all_features[feature] += score * weight
    
    # Get top features
    top_features = sorted(all_features.items(), key=lambda x: x[1], reverse=True)[:15]
    
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111)
    
    # Extract feature names and scores
    features = []
    scores = []
    
    for feat, score in top_features:
        # Map feature to meaningful name
        display_name = feat
        
        # Check if it's a feature index that needs to be mapped
        if feat in feature_mapping:
            raw_name = feature_mapping[feat]
            # Use human-readable description if available
            if raw_name in feature_descriptions:
                display_name = feature_descriptions[raw_name]
            else:
                display_name = raw_name
        
        features.append(display_name)
        scores.append(score)
    
    # Reverse for bottom-to-top display
    features.reverse()
    scores.reverse()
    
    # Create a custom colormap
    cmap = plt.cm.viridis
    norm = plt.Normalize(min(scores), max(scores))
    
    # Create horizontal bar chart with color gradient
    bars = ax.barh(features, scores, color=cmap(norm(scores)), alpha=0.8)
    
    # Add importance values
    for bar in bars:
        width = bar.get_width()
        ax.text(width * 1.01, bar.get_y() + bar.get_height()/2,
               f'{width:.4f}', va='center', fontweight='bold')
    
    ax.set_title(f'Combined Feature Importance (Weighted by Ensemble)')
    ax.set_xlabel('Weighted Importance Score')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'combined_feature_importance.png', bbox_inches='tight')
    plt.close()

# 4. Intrusion Detection Timeline
def plot_intrusion_timeline():
    print("Creating intrusion detection timeline visualization...")
    fig = plt.figure(figsize=(20, 15))
    gs = GridSpec(2, 1, height_ratios=[2, 1])
    
    # Extract timestamps and probabilities from alerts
    timestamps = []
    probabilities = []
    detected = []
    
    for alert in alerts:
        if 'timestamp' in alert and 'probability' in alert:
            timestamps.append(alert['timestamp'])
            probabilities.append(alert['probability'])
            detected.append(True)  # All alerts are detected intrusions
    
    # Convert timestamps to relative time in minutes
    if timestamps:
        start_time = min(timestamps)
        rel_times = [(ts - start_time) / 60.0 for ts in timestamps]
        
        # Plot 1: Detection Timeline with Confidence Heatmap
        ax1 = fig.add_subplot(gs[0])
        
        # Create a colormap for the probabilities
        cmap = plt.cm.plasma
        norm = plt.Normalize(min(probabilities), max(probabilities))
        
        # Create scatter plot with color based on probability
        scatter = ax1.scatter(rel_times, probabilities, 
                             c=probabilities, cmap=cmap, norm=norm,
                             s=150, alpha=0.8, edgecolors='black')
        
        # Add a horizontal line at the threshold
        threshold = ensemble_metrics.get('threshold', 0.6)
        ax1.axhline(y=threshold, color='red', linestyle='--', linewidth=2)
        
        # Add an annotation for the threshold
        if rel_times:
            ax1.text(rel_times[0], threshold + 0.02, 
                    f'Detection Threshold ({threshold:.1f})',
                    fontweight='bold', color='red', va='bottom')
        
        # Add a colorbar
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Detection Confidence')
        
        # Set labels and title
        ax1.set_title('Intrusion Detection Timeline')
        ax1.set_xlabel('Time (minutes)')
        ax1.set_ylabel('Detection Confidence')
        ax1.set_ylim(0, 1)
        
        # Add stats annotation
        stats_text = (
            f"Total Alerts: {len(probabilities)}\n"
            f"Average Confidence: {sum(probabilities)/len(probabilities):.4f}\n"
            f"Max Confidence: {max(probabilities):.4f}"
        )
        ax1.text(0.02, 0.05, stats_text,
                transform=ax1.transAxes,
                fontsize=16, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
        # Plot 2: Confidence Distribution
        ax2 = fig.add_subplot(gs[1])
        
        # Create custom bins centered on threshold
        bin_edges = np.linspace(0.5, 1.0, 21)
        
        # Create histogram with custom colormap based on threshold
        hist, bins, patches = ax2.hist(probabilities, bins=bin_edges, alpha=0.8, 
                                      edgecolor='black')
        
        # Color the bars based on threshold
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        for i, patch in enumerate(patches):
            if bin_centers[i] >= threshold:
                patch.set_facecolor('#d62728')  # Red above threshold
            else:
                patch.set_facecolor('#2ca02c')  # Green below threshold
        
        # Add threshold line
        ax2.axvline(x=threshold, color='red', linestyle='--', linewidth=2)
        
        # Set labels and title
        ax2.set_title('Detection Confidence Distribution')
        ax2.set_xlabel('Confidence')
        ax2.set_ylabel('Frequency')
        
        # Add legend
        threshold_patch = mpatches.Patch(color='#d62728', label='Above Threshold')
        below_patch = mpatches.Patch(color='#2ca02c', label='Below Threshold')
        ax2.legend(handles=[threshold_patch, below_patch], loc='upper left')
    
    else:
        # Handle case with no alerts
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No alerts data available", 
               ha='center', va='center', fontsize=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'intrusion_timeline.png', bbox_inches='tight')
    plt.close()

# 5. Traffic Analysis Dashboard
def plot_traffic_analysis():
    print("Creating traffic analysis dashboard...")
    fig = plt.figure(figsize=(20, 15))
    gs = GridSpec(2, 2)
    
    # Plot 1: Label Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    
    if 'label' in traffic_data.columns:
        label_counts = traffic_data['label'].value_counts()
        labels = ['Normal', 'Intrusion']
        sizes = [label_counts.get(0, 0), label_counts.get(1, 0)]
        explode = (0.05, 0.1)
        colors = ['#4CAF50', '#F44336']
        
        wedges, texts, autotexts = ax1.pie(
            sizes,
            explode=explode,
            labels=labels,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            shadow=True,
            textprops={'fontweight': 'bold'}
        )
        
        # Make text bold and visible
        for text in texts:
            text.set_fontweight('bold')
        for autotext in autotexts:
            autotext.set_fontweight('bold')
            autotext.set_color('white')
            
        # Add details in annotation box
        total = sum(sizes)
        intrusion_pct = sizes[1] / total * 100 if total > 0 else 0
        
        # Only add annotation if we have meaningful data
        if total > 10:
            ax1.text(0.05, 0.05, 
                    f"Total Traffic: {total}\n"
                    f"Normal: {sizes[0]} ({100-intrusion_pct:.1f}%)\n"
                    f"Intrusion: {sizes[1]} ({intrusion_pct:.1f}%)",
                    transform=ax1.transAxes,
                    fontsize=16, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
            
        ax1.set_title('Traffic Classification Distribution')
    else:
        ax1.text(0.5, 0.5, "No label data available", 
                ha='center', va='center', fontsize=20)
    
    # Plot 2: Protocol Distribution
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Find protocol columns
    proto_cols = [col for col in traffic_data.columns if col.startswith('proto_')]
    
    if proto_cols:
        # Sum up protocol occurrences
        proto_sums = traffic_data[proto_cols].sum().sort_values(ascending=False)
        
        # Get top protocols
        top_n = 10
        top_protos = proto_sums.head(top_n)
        
        # Clean up protocol names
        clean_names = [p.replace('proto_', '') for p in top_protos.index]
        
        # Plot horizontal bar chart
        bars = ax2.barh(clean_names, top_protos.values, 
                        color=sns.color_palette("viridis", top_n))
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax2.annotate(f'{width:.0f}',
                         xy=(width, bar.get_y() + bar.get_height()/2),
                         xytext=(3, 0),
                         textcoords="offset points",
                         ha='left', va='center')
        
        ax2.set_title('Top 10 Protocols')
        ax2.set_xlabel('Count')
    else:
        ax2.text(0.5, 0.5, "No protocol data available", 
                ha='center', va='center', fontsize=20)
    
    # Plot 3: Connection State Distribution
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Find state columns
    state_cols = [col for col in traffic_data.columns if col.startswith('state_')]
    
    if state_cols:
        # Sum up state occurrences
        state_sums = traffic_data[state_cols].sum().sort_values(ascending=False)
        
        # Clean up state names
        clean_names = [s.replace('state_', '') for s in state_sums.index]
        
        # Plot horizontal bar chart
        bars = ax3.barh(clean_names, state_sums.values, 
                       color=sns.color_palette("mako", len(state_sums)))
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax3.annotate(f'{width:.0f}',
                        xy=(width, bar.get_y() + bar.get_height()/2),
                        xytext=(3, 0),
                        textcoords="offset points",
                        ha='left', va='center')
        
        ax3.set_title('Connection State Distribution')
        ax3.set_xlabel('Count')
    else:
        ax3.text(0.5, 0.5, "No state data available", 
                ha='center', va='center', fontsize=20)
    
    # Plot 4: Traffic Volume by Label
    ax4 = fig.add_subplot(gs[1, 1])
    
    if 'label' in traffic_data.columns and 'sbytes' in traffic_data.columns and 'dbytes' in traffic_data.columns:
        # Group by label
        normal = traffic_data[traffic_data['label'] == 0]
        intrusion = traffic_data[traffic_data['label'] == 1]
        
        # Calculate metrics
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
        
        # Set labels and title
        ax4.set_title('Traffic Volume by Classification')
        ax4.set_ylabel('Average Bytes')
        ax4.set_xticks(x)
        ax4.set_xticklabels(['Source Bytes', 'Destination Bytes'])
        ax4.legend()
        
        # Add a percentage comparison annotation
        if normal_means[0] > 0:
            sbyte_diff_pct = (intrusion_means[0] / normal_means[0] - 1) * 100
            dbyte_diff_pct = (intrusion_means[1] / normal_means[1] - 1) * 100 if normal_means[1] > 0 else 0
            
            ax4.text(0.05, 0.05,
                    f"Intrusion vs Normal:\n"
                    f"Source: {sbyte_diff_pct:.1f}% {'higher' if sbyte_diff_pct > 0 else 'lower'}\n"
                    f"Dest: {dbyte_diff_pct:.1f}% {'higher' if dbyte_diff_pct > 0 else 'lower'}",
                    transform=ax4.transAxes,
                    fontsize=16, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    else:
        ax4.text(0.5, 0.5, "Missing data for volume analysis", 
                ha='center', va='center', fontsize=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'traffic_analysis.png', bbox_inches='tight')
    plt.close()

# 6. Encryption Overhead Analysis
def plot_encryption_overhead():
    print("Creating encryption overhead visualization...")
    fig = plt.figure(figsize=(20, 15))
    gs = GridSpec(2, 2)
    
    # Plot 1: Component size breakdown
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Extract component sizes
    components = list(component_sizes.keys())
    sizes = list(component_sizes.values())
    
    # Calculate total and percentage
    total = sum(sizes)
    percentages = [size/total*100 for size in sizes]
    
    # Create horizontal bar chart
    bars = ax1.barh(components, sizes, color=sns.color_palette("Blues", len(components)))
    
    # Add value and percentage labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax1.annotate(f'{width} bytes ({percentages[i]:.1f}%)',
                   xy=(width, bar.get_y() + bar.get_height()/2),
                   xytext=(3, 0),
                   textcoords="offset points",
                   ha='left', va='center')
    
    ax1.set_title('ECIES Component Size Distribution')
    ax1.set_xlabel('Size (bytes)')
    
    # Add annotation with total
    ax1.text(0.95, 0.05, f'Total Overhead: {total} bytes',
             transform=ax1.transAxes,
             fontsize=16, fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    
    
    plt.tight_layout()
    plt.savefig(output_dir / 'encryption_overhead.png', bbox_inches='tight')
    plt.close()

# 7. Connection State Transition Analysis
def plot_connection_state_analysis():
    print("Creating connection state analysis visualization...")
    fig = plt.figure(figsize=(20, 15))
    gs = GridSpec(2, 2)
    
    # Plot 1: Connection State Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Extract state columns and their sums
    state_cols = [col for col in traffic_data.columns if col.startswith('state_')]
    
    if state_cols:
        # Sum up state occurrences
        state_sums = traffic_data[state_cols].sum().sort_values(ascending=False)
        
        # Clean up state names
        states = [s.replace('state_', '') for s in state_sums.index]
        counts = state_sums.values
        
        # Create bar chart
        bars = ax1.bar(states, counts, color=sns.color_palette("mako", len(states)))
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.0f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom')
        
        # Calculate percentages
        total = sum(counts)
        percentages = [count/total*100 for count in counts]
        
        # Add percentage text box
        percentage_text = "State Distribution:\n"
        for i in range(min(5, len(states))):
            percentage_text += f"{states[i]}: {percentages[i]:.1f}%\n"
        
        ax1.text(0.05, 0.95, percentage_text,
                transform=ax1.transAxes,
                fontsize=16, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
        ax1.set_title('Connection State Distribution')
        ax1.set_ylabel('Count')
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    else:
        ax1.text(0.5, 0.5, "No state data available", 
                ha='center', va='center', fontsize=20)
    
    # Plot 2: State distribution by traffic type
    ax2 = fig.add_subplot(gs[0, 1])
    
    if 'label' in traffic_data.columns and state_cols:
        # Group by label
        normal = traffic_data[traffic_data['label'] == 0]
        intrusion = traffic_data[traffic_data['label'] == 1]
        
        # Get top states
        top_n = min(5, len(state_cols))
        top_states = state_sums.head(top_n).index
        
        # Calculate percentages for each label
        normal_pcts = [(normal[state].sum() / len(normal) * 100) if len(normal) > 0 else 0 
                      for state in top_states]
        intrusion_pcts = [(intrusion[state].sum() / len(intrusion) * 100) if len(intrusion) > 0 else 0 
                         for state in top_states]
        
        # Clean state names
        state_names = [s.replace('state_', '') for s in top_states]
        
        # Create grouped bar chart
        x = np.arange(len(state_names))
        width = 0.35
        
        ax2.bar(x - width/2, normal_pcts, width, label='Normal', color='#4CAF50')
        ax2.bar(x + width/2, intrusion_pcts, width, label='Intrusion', color='#F44336')
        
        # Add value labels
        for i, v in enumerate(normal_pcts):
            ax2.text(i - width/2, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')
        for i, v in enumerate(intrusion_pcts):
            ax2.text(i + width/2, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')
        
        ax2.set_title('Connection States by Traffic Type')
        ax2.set_ylabel('Percentage')
        ax2.set_xticks(x)
        ax2.set_xticklabels(state_names)
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, "Missing data for state comparison", 
                ha='center', va='center', fontsize=20)
    
    # Plot 3: State transitions simulation (conceptual)
    ax3 = fig.add_subplot(gs[1, :])
    
    # This is a conceptual visualization of state transitions
    # Based on typical network state transitions
    
    # Define state positions for a graph
    states = ['FIN', 'INT', 'CON', 'ECO', 'REQ', 'RST', 'PAR', 'URN', 'no']
    
    # Create a grid layout for states
    positions = {
        'INT': (0, 0.5),
        'CON': (0.2, 0.8),
        'ECO': (0.4, 0.6),
        'REQ': (0.6, 0.8),
        'PAR': (0.8, 0.6),
        'URN': (0.9, 0.3),
        'RST': (0.7, 0.2),
        'FIN': (0.3, 0.2),
        'no':  (0.5, 0.4)
    }
    
    # Define state transitions
    transitions = [
        ('INT', 'CON', 0.8),
        ('CON', 'ECO', 0.6),
        ('ECO', 'REQ', 0.5),
        ('REQ', 'PAR', 0.7),
        ('PAR', 'URN', 0.3),
        ('CON', 'RST', 0.2),
        ('ECO', 'RST', 0.1),
        ('REQ', 'FIN', 0.2),
        ('PAR', 'FIN', 0.3),
        ('no', 'INT', 0.5),
        ('no', 'RST', 0.3),
        ('no', 'FIN', 0.1),
        ('RST', 'no', 0.4),
        ('FIN', 'no', 0.5)
    ]
    
    # Create node colors based on state frequencies
    node_sizes = {}
    node_colors = {}
    
    if state_cols:
        for state in states:
            state_col = f'state_{state}'
            if state_col in state_sums.index:
                freq = state_sums[state_col]
                # Size proportional to frequency
                node_sizes[state] = 2000 * (freq / state_sums.max())
                # Color based on whether it's related to normal or intrusion traffic
                if 'label' in traffic_data.columns:
                    normal_freq = normal[state_col].sum() if len(normal) > 0 else 0
                    intrusion_freq = intrusion[state_col].sum() if len(intrusion) > 0 else 0
                    
                    if normal_freq > intrusion_freq:
                        intensity = min(1.0, normal_freq / (normal_freq + intrusion_freq + 1))
                        node_colors[state] = (0, intensity, 0)  # Green for normal
                    else:
                        intensity = min(1.0, intrusion_freq / (normal_freq + intrusion_freq + 1))
                        node_colors[state] = (intensity, 0, 0)  # Red for intrusion
                else:
                    node_colors[state] = (0.5, 0.5, 0.5)  # Gray if no label info
            else:
                node_sizes[state] = 500  # Default size
                node_colors[state] = (0.5, 0.5, 0.5)  # Default color
    else:
        # Default values if no state data
        for state in states:
            node_sizes[state] = 1000
            node_colors[state] = (0.5, 0.5, 0.5)
    
    # Draw nodes
    for state in states:
        x, y = positions[state]
        ax3.scatter(x, y, s=node_sizes[state], color=node_colors[state], alpha=0.7, edgecolors='black')
        ax3.text(x, y, state, fontsize=16, ha='center', va='center', fontweight='bold')
    
    # Draw edges with arrows
    for source, target, weight in transitions:
        sx, sy = positions[source]
        tx, ty = positions[target]
        
        # Calculate arrow positions
        dx = tx - sx
        dy = ty - sy
        
        # Draw arrow
        ax3.annotate("", 
                    xy=(tx, ty), xycoords='data',
                    xytext=(sx, sy), textcoords='data',
                    arrowprops=dict(arrowstyle="->", 
                                    connectionstyle="arc3,rad=0.2",
                                    linewidth=max(1, 5*weight),
                                    color='gray', alpha=0.6))
    
    ax3.set_title('Connection State Transition Model')
    ax3.set_xlim(-0.1, 1.1)
    ax3.set_ylim(-0.1, 1.1)
    ax3.axis('off')
    
    # Add legend
    normal_patch = mpatches.Patch(color='#4CAF50', label='Predominant in Normal Traffic')
    intrusion_patch = mpatches.Patch(color='#F44336', label='Predominant in Intrusion Traffic')
    ax3.legend(handles=[normal_patch, intrusion_patch], loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'connection_state_analysis.png', bbox_inches='tight')
    plt.close()

# 8. Create a comprehensive summary dashboard
def create_summary_dashboard():
    print("Creating summary dashboard...")
    fig = plt.figure(figsize=(20, 20))
    fig.suptitle('AI-Enhanced Cryptography Hybrid System: Performance Summary', fontsize=30, fontweight='bold', y=0.98)
    
    gs = GridSpec(4, 2, height_ratios=[1, 1, 1, 1])
    
    # Plot 1: Model weight distribution
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(model_names, 
                  [ensemble_metrics['weights'][m] for m in model_names],
                  color=[colors[m] for m in model_names],
                  alpha=0.8)
    
    ax1.set_title('Ensemble Model Weights')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Plot 2: Cryptographic performance
    ax2 = fig.add_subplot(gs[0, 1])
    operations = ['Encryption', 'Decryption', 'Key\nOps']
    times = [
        crypto_metrics['encryption']['average_time'],
        crypto_metrics['decryption']['average_time'],
        crypto_metrics['key_operations']['average_time']
    ]
    
    bars = ax2.bar(operations, times, color=['#3274A1', '#E1812C', '#3A923A'])
    
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{height:.6f} ms',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    ax2.set_title('Cryptographic Performance')
    ax2.set_ylabel('Time (ms)')
    
    # Plot 3: Top features (combined)
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Get all feature importances and combine
    all_features = {}
    for model, importances in feature_importances.items():
        weight = ensemble_metrics['weights'].get(model, 1.0/len(feature_importances))
        for feature, score in importances:
            if feature not in all_features:
                all_features[feature] = 0
            # Weight the importance by model weight
            all_features[feature] += score * weight
    
    # Get top features
    top_features = sorted(all_features.items(), key=lambda x: x[1], reverse=True)[:8]
    
    # Extract feature names and scores
    features = []
    scores = []
    
    for feat, score in top_features:
        # Map feature to meaningful name
        display_name = feat
        
        # Check if it's a feature index that needs to be mapped
        if feat in feature_mapping:
            raw_name = feature_mapping[feat]
            # Use human-readable description if available
            if raw_name in feature_descriptions:
                display_name = feature_descriptions[raw_name]
            else:
                display_name = raw_name
        
        features.append(display_name)
        scores.append(score)
    
    # Reverse for bottom-to-top display
    features.reverse()
    scores.reverse()
    
    # Create horizontal bar chart
    bars = ax3.barh(features, scores, color=plt.cm.viridis(np.linspace(0, 1, len(features))), alpha=0.8)
    
    ax3.set_title('Top Features (Weighted by Ensemble)')
    ax3.set_xlabel('Importance Score')
    
    # Plot 4: ECIES Overhead components
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Calculate component percentages
    total_size = sum(component_sizes.values())
    component_pcts = [size/total_size*100 for size in component_sizes.values()]
    
    # Create pie chart
    wedges, texts, autotexts = ax4.pie(
        component_pcts,
        labels=list(component_sizes.keys()),
        autopct='%1.1f%%',
        startangle=90,
        colors=plt.cm.Blues(np.linspace(0.3, 0.7, len(component_sizes))),
        wedgeprops={'edgecolor': 'w', 'linewidth': 1},
        textprops={'fontweight': 'bold'}
    )
    
    # Make text bold and visible
    for text in texts:
        text.set_fontweight('bold')
    for autotext in autotexts:
        autotext.set_fontweight('bold')
        autotext.set_color('white')
    
    ax4.set_title('ECIES Component Size Distribution')
    
    # Add annotation with total overhead
    ax4.text(-0.1, -0.15, f"Total Overhead: {total_size} bytes",
             fontsize=16, fontweight='bold')
    
    # Plot 5: Detection probability distribution
    ax5 = fig.add_subplot(gs[2, 0])
    
    # Extract probabilities from alerts
    probabilities = []
    for alert in alerts:
        if 'probability' in alert:
            probabilities.append(alert['probability'])
    
    if probabilities:
        # Create custom bins
        bins = np.linspace(0.5, 1.0, 21)
        
        # Create histogram
        hist, bins, patches = ax5.hist(probabilities, bins=bins, alpha=0.8, edgecolor='black')
        
        # Color the bars based on threshold
        threshold = ensemble_metrics.get('threshold', 0.6)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        for i, patch in enumerate(patches):
            if bin_centers[i] >= threshold:
                patch.set_facecolor('#d62728')  # Red above threshold
            else:
                patch.set_facecolor('#2ca02c')  # Green below threshold
        
        # Add threshold line
        ax5.axvline(x=threshold, color='red', linestyle='--', linewidth=2)
        
        # Calculate stats
        mean_prob = np.mean(probabilities)
        
        # Add stats annotation
        ax5.text(0.05, 0.95, 
                f"Detections: {len(probabilities)}\n"
                f"Mean: {mean_prob:.4f}\n"
                f"Above Threshold: {sum(p >= threshold for p in probabilities)}",
                transform=ax5.transAxes,
                fontsize=16, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    else:
        ax5.text(0.5, 0.5, "No probability data available", 
               ha='center', va='center', fontsize=20)
    
    ax5.set_title('Detection Confidence Distribution')
    ax5.set_xlabel('Confidence')
    ax5.set_ylabel('Frequency')
    
    # Plot 6: Traffic classification
    ax6 = fig.add_subplot(gs[2, 1])
    
    if 'label' in traffic_data.columns:
        label_counts = traffic_data['label'].value_counts()
        labels = ['Normal', 'Intrusion']
        sizes = [label_counts.get(0, 0), label_counts.get(1, 0)]
        
        wedges, texts, autotexts = ax6.pie(
            sizes,
            labels=labels,
            autopct='%1.1f%%',
            startangle=90,
            colors=['#4CAF50', '#F44336'],
            wedgeprops={'edgecolor': 'w', 'linewidth': 1},
            textprops={'fontweight': 'bold'}
        )
        
        # Make text bold and visible
        for text in texts:
            text.set_fontweight('bold')
        for autotext in autotexts:
            autotext.set_fontweight('bold')
            autotext.set_color('white')
    else:
        ax6.text(0.5, 0.5, "No label data available", 
                ha='center', va='center', fontsize=20)
    
    ax6.set_title('Traffic Classification Distribution')
    
    # Plot 7: Connection State Distribution
    ax7 = fig.add_subplot(gs[3, 0])
    
    # Extract state columns and their sums
    state_cols = [col for col in traffic_data.columns if col.startswith('state_')]
    
    if state_cols:
        # Sum up state occurrences
        state_sums = traffic_data[state_cols].sum().sort_values(ascending=False)
        
        # Clean up state names
        states = [s.replace('state_', '') for s in state_sums.index]
        counts = state_sums.values
        
        # Create bar chart
        bars = ax7.bar(states, counts, color=sns.color_palette("mako", len(states)))
        
        ax7.set_title('Connection State Distribution')
        ax7.set_ylabel('Count')
        plt.setp(ax7.get_xticklabels(), rotation=45, ha='right')
    else:
        ax7.text(0.5, 0.5, "No state data available", 
                ha='center', va='center', fontsize=20)
    
    # Plot 8: Traffic Volume Comparison
    ax8 = fig.add_subplot(gs[3, 1])
    
    if 'label' in traffic_data.columns and 'sbytes' in traffic_data.columns and 'dbytes' in traffic_data.columns:
        # Group by label
        normal = traffic_data[traffic_data['label'] == 0]
        intrusion = traffic_data[traffic_data['label'] == 1]
        
        # Calculate metrics (more comprehensive)
        metrics = ['sbytes', 'dbytes', 'spkts', 'dpkts'] if 'spkts' in traffic_data.columns and 'dpkts' in traffic_data.columns else ['sbytes', 'dbytes']
        metric_labels = ['Source\nBytes', 'Dest\nBytes', 'Source\nPackets', 'Dest\nPackets'] if len(metrics) == 4 else ['Source\nBytes', 'Dest\nBytes']
        
        normal_means = [normal[m].mean() for m in metrics]
        intrusion_means = [intrusion[m].mean() for m in metrics]
        
        # Normalize to make bars comparable
        max_values = [max(n, i) for n, i in zip(normal_means, intrusion_means)]
        normal_norm = [n/m*100 if m > 0 else 0 for n, m in zip(normal_means, max_values)]
        intrusion_norm = [i/m*100 if m > 0 else 0 for i, m in zip(intrusion_means, max_values)]
        
        # Create grouped bar chart
        x = np.arange(len(metrics))
        width = 0.35
        
        ax8.bar(x - width/2, normal_norm, width, label='Normal', color='#4CAF50')
        ax8.bar(x + width/2, intrusion_norm, width, label='Intrusion', color='#F44336')
        
        # Add value labels with actual values
        for i, v in enumerate(normal_means):
            ax8.text(i - width/2, normal_norm[i] + 2, f'{v:.0f}', ha='center', fontweight='bold')
        for i, v in enumerate(intrusion_means):
            ax8.text(i + width/2, intrusion_norm[i] + 2, f'{v:.0f}', ha='center', fontweight='bold')
        
        ax8.set_title('Traffic Volume Comparison')
        ax8.set_ylabel('Normalized Value (%)')
        ax8.set_xticks(x)
        ax8.set_xticklabels(metric_labels)
        ax8.legend()
    else:
        ax8.text(0.5, 0.5, "Missing data for volume analysis", 
                ha='center', va='center', fontsize=20)
    
    # Add watermark with summary statistics
    if 'average_confidence' in ensemble_metrics:
        avg_conf = ensemble_metrics['average_confidence']
        fig.text(0.5, 0.01, 
                f"Average Detection Confidence: {avg_conf:.4f} | "
                f"Encryption Time: {crypto_metrics['encryption']['average_time']:.6f} ms | "
                f"Overhead: {total_size} bytes",
                ha='center', fontsize=16, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_dir / 'summary_dashboard.png', bbox_inches='tight')
    plt.close()

# Execute all visualization functions
def main():
    print(f"Creating visualizations in directory: {output_dir}")
    
    # Run all visualization functions
    plot_model_performance()
    plot_crypto_performance()
    plot_feature_importance()
    plot_intrusion_timeline()
    plot_traffic_analysis()
    plot_encryption_overhead()
    plot_connection_state_analysis()
    create_summary_dashboard()
    
    print("Visualization complete! All images saved to the 'visualizations' directory.")

if __name__ == "__main__":
    main()