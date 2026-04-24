# config.py
import os
import logging
from pathlib import Path


CONFIG = {
    'NETWORK': {
        'NORMAL_TRAFFIC_RATE': 20,
        'BUFFER_SIZE': 1000,
        'INTRUSION_PROBABILITY': 0.05
    },
    'MODEL': {
        'BASE_PATH': '/home/seyitope/recent_ids_modell/results',
        'MODELS_DIR': '/home/seyitope/recent_ids_modell/results/models',
        'MODEL_PATHS': {
            # Neural models are directly in models directory
            'CNN': 'cnn_model.pth',
            'LSTM': 'lstm_model.pth',
            'DNN': 'dnn_model.pth',
            # Traditional models in their respective directories
            'XGBoost': 'XGBoost/model.joblib',
            'RandomForest': 'RandomForest/model.joblib',
            'SVM': 'SVM/svm_model.joblib'  # Note the specific name for SVM
        },
        'WEIGHTS': {
            'cnn': 0.12,    # Based on F1: 0.9495
            'lstm': 0.13,   # Based on F1: 0.9514
            'dnn': 0.15,    # Based on F1: 0.9500
            'xgboost': 0.25,  # Based on F1: 0.9591
            'randomforest': 0.25,  # Based on F1: 0.9567
            'svm': 0.10    # Based on F1: 0.9228
        }

    },
    'FEATURES': {
        'NUMERIC': [
            'id', 'dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl',
            'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit',
            'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 'smean',
            'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src', 'ct_state_ttl',
            'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm',
            'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm',
            'ct_srv_dst', 'is_sm_ips_ports', 'label'
        ],
        'PROTOCOLS': [
            'tcp', 'udp', 'arp', 'ospf', 'icmp', 'igmp', 'rtp', 'ddp', 'ipv6-frag', 'cftp',
            'wsn', 'pvp', 'wb-expak', 'mtp', 'pri-enc', 'sat-mon', 'cphb', 'sun-nd', 'iso-ip',
            'xtp', 'il', 'unas', 'mfe-nsp', '3pc', 'ipv6-route', 'idrp', 'bna', 'swipe',
            'kryptolan', 'cpnx', 'rsvp', 'wb-mon', 'vmtp', 'ib', 'dgp', 'eigrp', 'ax.25',
            'gmtp', 'pnni', 'sep', 'pgm', 'idpr-cmtp', 'zero', 'rvd', 'mobile', 'narp', 'fc',
            'pipe', 'ipcomp', 'ipv6-no', 'sat-expak', 'ipv6-opts', 'snp', 'ipcv',
            'br-sat-mon', 'ttp', 'tcf', 'nsfnet-igp', 'sprite-rpc', 'aes-sp3-d', 'sccopmce',
            'sctp', 'qnx', 'scps', 'etherip', 'aris', 'pim', 'compaq-peer', 'vrrp', 'iatp',
            'stp', 'l2tp', 'srp', 'sm', 'isis', 'smp', 'fire', 'ptp', 'crtp', 'sps',
            'merit-inp', 'idpr', 'skip', 'any', 'larp', 'ipip', 'micp', 'encap', 'ifmp',
            'tp++', 'a/n', 'ipv6', 'i-nlsp', 'ipx-n-ip', 'sdrp', 'tlsp', 'gre', 'mhrp', 'ddx',
            'ippc', 'visa', 'secure-vmtp', 'uti', 'vines', 'crudp', 'iplt', 'ggp', 'ip',
            'ipnip', 'st2', 'argus', 'bbn-rcc', 'egp', 'emcon', 'igp', 'nvp', 'pup', 'xnet',
            'chaos', 'mux', 'dcn', 'hmp', 'prm', 'trunk-1', 'xns-idp', 'leaf-1', 'leaf-2',
            'rdp', 'irtp', 'iso-tp4', 'netblt', 'trunk-2', 'cbt'
        ],
        'SERVICES': [
            '-', 'ftp', 'smtp', 'snmp', 'http', 'ftp-data', 'dns', 'ssh', 
            'radius', 'pop3', 'dhcp', 'ssl', 'irc'
        ],
        'STATES': [
            'FIN', 'INT', 'CON', 'ECO', 'REQ', 'RST', 'PAR', 'URN', 'no'
        ]
    },
    'THRESHOLDS': {
        'INTRUSION_PROBABILITY': 0.6,
        'HIGH_VOLUME_BYTES': 10000,
        'HIGH_ERROR_RATE': 0.7,
        'SUSPICIOUS_LOAD': 100
    },
     'PATHS': {
        'BASE': '/home/seyitope/recent_ids_modell',
        'RESULTS': '/home/seyitope/recent_ids_modell/results',
        'LOGS': '/home/seyitope/recent_ids_modell/results/logs',
        'METRICS': '/home/seyitope/recent_ids_modell/results/metrics',
        'SIMULATION': '/home/seyitope/recent_ids_modell/results/simulation_results'
    },
    
    'LOGGING': {
        'LEVEL': logging.INFO,
        'FORMAT': '%(asctime)s - %(levelname)s - %(message)s',
        'DEBUG': True,
        'LOG_FILE': os.path.join('/home/seyitope/recent_ids_modell/results/logs', 'ids_debug.log')
    },
    
    'DEBUG': {
        'PRINT_FEATURES': True,        # Print feature extraction details
        'PRINT_PREDICTIONS': True,     # Print model predictions
        'PRINT_MODEL_LOADING': True,   # Print model loading details
        'SAVE_TRAFFIC_DATA': True,     # Save processed traffic data
        'VERBOSE_OUTPUT': True,        # Enable detailed output
        'LOG_PREPROCESSING': True,     # Log preprocessing steps
        'LOG_MODEL_DECISIONS': True,   # Log individual model decisions
        'LOG_ENSEMBLE_WEIGHTS': True,   # Log how ensemble weights are applied
        'SAVE_VISUALIZATIONS': True 
    }
}
