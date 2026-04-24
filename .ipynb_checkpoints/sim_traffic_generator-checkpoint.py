# traffic_generator.py

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List

@dataclass
class TrafficFeatures:
    numeric: Dict[str, float] = field(default_factory=dict)
    protocols: Dict[str, int] = field(default_factory=dict)
    services: Dict[str, int] = field(default_factory=dict)
    states: Dict[str, int] = field(default_factory=dict)

class TrafficGenerator:
    def __init__(self, config: Dict):
        self.config = config
        self.packet_id = 0
        self.normal_ranges = {
            'dur': (0, 60),
            'spkts': (1, 100),
            'dpkts': (1, 100),
            'sbytes': (100, 10000),
            'dbytes': (100, 10000),
            'rate': (0, 1000),
            'sttl': (1, 255),
            'dttl': (1, 255),
            'sload': (0, 100),
            'dload': (0, 100),
            'sloss': (0, 10),
            'dloss': (0, 10),
            'sinpkt': (0, 1),
            'dinpkt': (0, 1),
            'sjit': (0, 10),
            'djit': (0, 10),
            'swin': (1000, 65535),
            'stcpb': (1000, 65535),
            'dtcpb': (1000, 65535),
            'dwin': (1000, 65535),
            'tcprtt': (0, 1),
            'synack': (0, 1),
            'ackdat': (0, 1),
            'smean': (100, 1000),
            'dmean': (100, 1000),
            'trans_depth': (0, 5),
            'response_body_len': (0, 1000),
            'ct_srv_src': (1, 10),
            'ct_state_ttl': (1, 10),
            'ct_dst_ltm': (1, 10),
            'ct_src_dport_ltm': (1, 10),
            'ct_dst_sport_ltm': (1, 10),
            'ct_dst_src_ltm': (1, 10),
            'ct_flw_http_mthd': (1, 5),
            'ct_src_ltm': (1, 10),
            'ct_srv_dst': (1, 10)
        }
        
        self.intrusion_ranges = {
            'dur': (60, 120),
            'sbytes': (10000, 50000),
            'dbytes': (10000, 50000),
            'sload': (100, 500),
            'dload': (100, 500),
            'sloss': (5, 20),
            'dloss': (5, 20),
            'ct_srv_src': (10, 30),
            'ct_state_ttl': (10, 30),
            'ct_dst_ltm': (10, 30)
        }

    def _generate_numeric_value(self, feature: str, is_intrusion: bool = False) -> float:
        """Generate a numeric value based on feature type and traffic class"""
        ranges = self.intrusion_ranges if is_intrusion else self.normal_ranges
        
        if feature in ranges:
            min_val, max_val = ranges[feature]
            if feature in ['dur']:
                return np.random.exponential(max_val - min_val) + min_val
            elif feature in ['sbytes', 'dbytes']:
                return np.random.randint(min_val, max_val)
            elif feature in ['sload', 'dload']:
                return np.random.uniform(min_val, max_val)
            else:
                return np.random.randint(min_val, max_val)
        return 0

    def generate_traffic(self, is_intrusion: bool = False) -> Dict[str, Any]:
        """Generate complete traffic data"""
        self.packet_id += 1
        traffic = {}
        
        # Generate numeric features
        for feature in self.config['FEATURES']['NUMERIC']:
            if feature == 'id':
                traffic[feature] = self.packet_id
            elif feature == 'label':
                traffic[feature] = 1 if is_intrusion else 0
            elif feature in ['is_ftp_login', 'ct_ftp_cmd', 'is_sm_ips_ports']:
                traffic[feature] = 0
            else:
                traffic[feature] = self._generate_numeric_value(feature, is_intrusion)

        # Generate protocol features
        selected_proto = np.random.choice(self.config['FEATURES']['PROTOCOLS'])
        for proto in self.config['FEATURES']['PROTOCOLS']:
            traffic[f'proto_{proto}'] = 1 if proto == selected_proto else 0

        # Generate service features
        selected_service = np.random.choice(self.config['FEATURES']['SERVICES'])
        for service in self.config['FEATURES']['SERVICES']:
            traffic[f'service_{service}'] = 1 if service == selected_service else 0

        # Generate state features
        selected_state = np.random.choice(self.config['FEATURES']['STATES'])
        for state in self.config['FEATURES']['STATES']:
            traffic[f'state_{state}'] = 1 if state == selected_state else 0

        # Log feature counts
        self._log_feature_counts(traffic)
        
        return traffic

    def _log_feature_counts(self, traffic: Dict[str, Any]):
        """Log feature counts for debugging"""
        numeric_count = len([k for k in traffic.keys() 
                           if not k.startswith(('proto_', 'service_', 'state_'))])
        proto_count = len([k for k in traffic.keys() if k.startswith('proto_')])
        service_count = len([k for k in traffic.keys() if k.startswith('service_')])
        state_count = len([k for k in traffic.keys() if k.startswith('state_')])
        
        logging.debug("\nGenerated traffic feature counts:")
        logging.debug(f"  Numeric features: {numeric_count}")
        logging.debug(f"  Protocol features: {proto_count}")
        logging.debug(f"  Service features: {service_count}")
        logging.debug(f"  State features: {state_count}")
        logging.debug(f"  Total features: {len(traffic)}")