# network_simulator.py
import logging
from queue import Queue
from threading import Thread, Lock
from typing import Optional, Dict, Any
import time
import numpy as np
from sim_traffic_generator import TrafficGenerator

class NetworkSimulator:
    def __init__(self, config: Dict[str, Any]):
        """Initialize network simulator"""
        self.config = config
        self.buffer = Queue(maxsize=config['NETWORK']['BUFFER_SIZE'])
        self.lock = Lock()
        self.running = False
        self.traffic_generator = TrafficGenerator(config)
    
    def start(self):
        """Start the network simulation"""
        self.running = True
        Thread(target=self._generate_traffic).start()
        logging.info("Network simulation started")
        
    def stop(self):
        """Stop the network simulation"""
        self.running = False
        logging.info("Network simulation stopped")
        
    def _generate_traffic(self):
        """Generate network traffic"""
        while self.running:
            if not self.buffer.full():
                with self.lock:
                    # Generate traffic with configured intrusion probability
                    is_intrusion = np.random.random() < self.config['NETWORK']['INTRUSION_PROBABILITY']
                    traffic = self.traffic_generator.generate_traffic(is_intrusion)
                    self.buffer.put(traffic)
                    
            time.sleep(1/self.config['NETWORK']['NORMAL_TRAFFIC_RATE'])
            
    def get_traffic(self) -> Optional[Dict[str, Any]]:
        """Get next traffic item from buffer"""
        return self.buffer.get() if not self.buffer.empty() else None

class NetworkMonitor:
    def __init__(self, crypto_manager, ids_monitor):
        """Initialize network monitor"""
        self.crypto = crypto_manager
        self.ids = ids_monitor
        self.alerts = []
        
    def process_traffic(self, traffic: Dict[str, Any]) -> bool:
        """Process and analyze network traffic"""
        try:
            # Encrypt traffic
            encrypted_traffic = self.crypto.encrypt_traffic(traffic)
            logging.info("Traffic encrypted")
            
            # Initial detection on metadata
            is_intrusion, probability = self.ids.detect_intrusion(traffic)
            
            if is_intrusion:
                logging.warning(f"Potential intrusion detected! (confidence: {probability:.2f})")
                
                # Decrypt for detailed analysis
                decrypted_traffic = self.crypto.decrypt_traffic(encrypted_traffic)
                logging.info("Traffic decrypted for analysis")
                
                # Pattern analysis
                pattern_analysis = self.ids.analyze_traffic_pattern(decrypted_traffic)
                
                # Detailed detection
                confirmed_intrusion, final_probability = self.ids.detect_intrusion(decrypted_traffic)
                
                if confirmed_intrusion:
                    self._handle_intrusion(decrypted_traffic, pattern_analysis, final_probability)
                    return True
                    
            return False
            
        except Exception as e:
            logging.error(f"Error processing traffic: {str(e)}")
            return False
            
    def _handle_intrusion(self, traffic: Dict[str, Any], 
                         analysis: Dict[str, bool], 
                         probability: float):
        """Handle confirmed intrusion"""
        alert = {
            'timestamp': time.time(),
            'probability': probability,
            'traffic_data': traffic,
            'analysis': analysis
        }
        
        self.alerts.append(alert)
        
        logging.error("Intrusion confirmed! Alert triggered!")
        logging.error(f"Confidence: {probability:.2f}")
        logging.error("Analysis results:")
        for key, value in analysis.items():
            if value:
                logging.error(f"  {key}: {value}")
        
        # Log important traffic features
        for key, value in traffic.items():
            if isinstance(value, (int, float)) and value > 0:
                logging.error(f"  {key}: {value}")
                
    def get_alerts(self):
        """Get all recorded alerts"""
        return self.alerts