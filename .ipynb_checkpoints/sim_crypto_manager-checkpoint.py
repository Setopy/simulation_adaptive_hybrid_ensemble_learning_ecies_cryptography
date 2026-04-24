import os
import json
import time
import logging
import secrets
import sys
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidKey, InvalidTag

@dataclass
class CryptoMetrics:
    """Dataclass for tracking cryptographic metrics"""
    times: List[float] = field(default_factory=list)
    overhead_bytes: List[int] = field(default_factory=list)
    memory_usage: List[int] = field(default_factory=list)
    successes: int = 0
    failures: int = 0
    total_bytes: int = 0
    auth_failures: int = 0

class CryptoConfig:
    """Configuration for CryptoManager"""
    MAX_INPUT_SIZE = 1024 * 1024  # 1MB
    MIN_INPUT_SIZE = 1  # 1 byte
    NONCE_SIZE = 12
    TAG_SIZE = 16
    KEY_SIZE = 32
    SALT_SIZE = 32
    MAX_MEMORY_USAGE = 1024 * 1024 * 10  # 10MB
    METRICS_HISTORY_SIZE = 1000

class CryptoManager:
    """Enhanced secure cryptographic operations manager"""
    
    def __init__(self):
        """Initialize with secure defaults and metrics tracking"""
        try:
            # Initialize metrics
            self.metrics = {
                'encryption': CryptoMetrics(),
                'decryption': CryptoMetrics(),
                'key_operations': CryptoMetrics()
            }
            
            # Generate static salt for HKDF
            self.static_salt = secrets.token_bytes(CryptoConfig.SALT_SIZE)
            
            # Track component sizes
            self.component_sizes = {
                'public_key': 0,
                'nonce': CryptoConfig.NONCE_SIZE,
                'tag': CryptoConfig.TAG_SIZE,
                'salt': CryptoConfig.SALT_SIZE
            }
            
            start_time = time.perf_counter()
            
            # Generate key pair
            self.private_key = ec.generate_private_key(
                ec.SECP256K1(),
                default_backend()
            )
            self.public_key = self.private_key.public_key()
            
            # Record key generation metrics
            key_gen_time = time.perf_counter() - start_time
            self.metrics['key_operations'].times.append(key_gen_time)
            
            # Record public key size
            self.component_sizes['public_key'] = len(
                self.public_key.public_bytes(
                    encoding=serialization.Encoding.X962,
                    format=serialization.PublicFormat.UncompressedPoint
                )
            )
            
            logging.info(
                f"CryptoManager initialized. Public key size: "
                f"{self.component_sizes['public_key']} bytes"
            )
            
        except Exception as e:
            logging.error(f"Failed to initialize CryptoManager: {str(e)}")
            raise

    def _derive_key(self, shared_secret: bytes, salt: Optional[bytes] = None) -> bytes:
        """Derive encryption key with enhanced security"""
        try:
            start_time = time.perf_counter()
            
            # Use provided salt or static salt
            effective_salt = salt if salt else self.static_salt
            
            # Derive key with salt
            derived_key = HKDF(
                algorithm=hashes.SHA256(),
                length=CryptoConfig.KEY_SIZE,
                salt=effective_salt,
                info=b'ecies-encryption',
                backend=default_backend()
            ).derive(shared_secret)
            
            # Record metrics
            derive_time = time.perf_counter() - start_time
            self.metrics['key_operations'].times.append(derive_time)
            
            return derived_key
            
        except Exception as e:
            logging.error(f"Key derivation failed: {str(e)}")
            raise

    def _validate_input_size(self, data: bytes, is_encrypted: bool = False):
        """Validate input sizes"""
        size = len(data)
        
        if is_encrypted:
            min_size = (
                self.component_sizes['public_key'] +
                CryptoConfig.NONCE_SIZE +
                CryptoConfig.TAG_SIZE
            )
            if size < min_size:
                raise ValueError(
                    f"Encrypted data too short: {size} bytes "
                    f"(minimum {min_size} bytes)"
                )
        else:
            if not CryptoConfig.MIN_INPUT_SIZE <= size <= CryptoConfig.MAX_INPUT_SIZE:
                raise ValueError(
                    f"Invalid input size: {size} bytes "
                    f"(must be between {CryptoConfig.MIN_INPUT_SIZE} "
                    f"and {CryptoConfig.MAX_INPUT_SIZE} bytes)"
                )

    def _track_memory(self, data: Any, operation: str):
        """Track memory usage"""
        try:
            memory = sys.getsizeof(data)
            if memory > CryptoConfig.MAX_MEMORY_USAGE:
                raise MemoryError(f"Memory usage exceeded: {memory} bytes")
            
            metrics = self.metrics[operation]
            metrics.memory_usage.append(memory)
            
            # Maintain history size
            if len(metrics.memory_usage) > CryptoConfig.METRICS_HISTORY_SIZE:
                metrics.memory_usage.pop(0)
                
        except Exception as e:
            logging.error(f"Memory tracking failed: {str(e)}")

    def encrypt_traffic(self, traffic_data: Dict[str, Any]) -> bytes:
        """Encrypt traffic with enhanced security and metrics"""
        encryption_start = time.perf_counter()
        try:
            # Validate input
            if not isinstance(traffic_data, dict):
                raise ValueError("Input must be a dictionary")
            
            # Serialize safely
            plaintext = json.dumps(traffic_data).encode()
            self._validate_input_size(plaintext)
            self._track_memory(plaintext, 'encryption')
            
            # Generate ephemeral key pair
            ephemeral_private_key = ec.generate_private_key(
                ec.SECP256K1(),
                default_backend()
            )
            ephemeral_public_key = ephemeral_private_key.public_key()
            
            # Perform ECDH
            shared_secret = ephemeral_private_key.exchange(
                ec.ECDH(),
                self.public_key
            )
            
            # Generate fresh salt
            salt = secrets.token_bytes(CryptoConfig.SALT_SIZE)
            
            # Derive key with salt
            derived_key = self._derive_key(shared_secret, salt)
            
            # Generate nonce
            nonce = secrets.token_bytes(CryptoConfig.NONCE_SIZE)
            
            # Create cipher
            cipher = Cipher(
                algorithms.AES(derived_key),
                modes.GCM(nonce),
                backend=default_backend()
            ).encryptor()
            
            # Encrypt
            ciphertext = cipher.update(plaintext) + cipher.finalize()
            
            # Combine components
            encrypted_data = (
                ephemeral_public_key.public_bytes(
                    encoding=serialization.Encoding.X962,
                    format=serialization.PublicFormat.UncompressedPoint
                ) +
                salt +
                nonce +
                ciphertext +
                cipher.tag
            )
            
            # Update metrics
            metrics = self.metrics['encryption']
            metrics.times.append(time.perf_counter() - encryption_start)
            metrics.overhead_bytes.append(len(encrypted_data) - len(plaintext))
            metrics.total_bytes += len(plaintext)
            metrics.successes += 1
            self._track_memory(encrypted_data, 'encryption')
            
            # Trim metrics history
            if len(metrics.times) > CryptoConfig.METRICS_HISTORY_SIZE:
                metrics.times.pop(0)
                metrics.overhead_bytes.pop(0)
            
            return encrypted_data
            
        except Exception as e:
            self.metrics['encryption'].failures += 1
            logging.error(f"Encryption failed: {str(e)}")
            raise

    def decrypt_traffic(self, encrypted_data: bytes) -> Dict[str, Any]:
        """Decrypt traffic with enhanced security and metrics"""
        decryption_start = time.perf_counter()
        try:
            # Validate input
            if not isinstance(encrypted_data, bytes):
                raise ValueError("Input must be bytes")
            
            self._validate_input_size(encrypted_data, is_encrypted=True)
            self._track_memory(encrypted_data, 'decryption')
            
            # Extract components
            pubkey_size = self.component_sizes['public_key']
            salt_size = CryptoConfig.SALT_SIZE
            nonce_size = CryptoConfig.NONCE_SIZE
            tag_size = CryptoConfig.TAG_SIZE
            
            # Validate lengths and extract components
            current_pos = 0
            
            ephemeral_public_key_bytes = encrypted_data[
                current_pos:current_pos + pubkey_size
            ]
            current_pos += pubkey_size
            
            salt = encrypted_data[
                current_pos:current_pos + salt_size
            ]
            current_pos += salt_size
            
            nonce = encrypted_data[
                current_pos:current_pos + nonce_size
            ]
            current_pos += nonce_size
            
            tag = encrypted_data[-tag_size:]
            ciphertext = encrypted_data[current_pos:-tag_size]
            
            # Reconstruct ephemeral public key
            ephemeral_public_key = ec.EllipticCurvePublicKey.from_encoded_point(
                ec.SECP256K1(),
                ephemeral_public_key_bytes
            )
            
            # Perform ECDH
            shared_secret = self.private_key.exchange(
                ec.ECDH(),
                ephemeral_public_key
            )
            
            # Derive key with received salt
            derived_key = self._derive_key(shared_secret, salt)
            
            # Decrypt
            cipher = Cipher(
                algorithms.AES(derived_key),
                modes.GCM(nonce, tag),
                backend=default_backend()
            ).decryptor()
            
            plaintext = cipher.update(ciphertext) + cipher.finalize()
            
            # Safely deserialize
            decrypted_data = json.loads(plaintext.decode())
            
            # Update metrics
            metrics = self.metrics['decryption']
            metrics.times.append(time.perf_counter() - decryption_start)
            metrics.total_bytes += len(ciphertext)
            metrics.successes += 1
            self._track_memory(decrypted_data, 'decryption')
            
            # Trim metrics history
            if len(metrics.times) > CryptoConfig.METRICS_HISTORY_SIZE:
                metrics.times.pop(0)
            
            return decrypted_data
            
        except InvalidTag:
            self.metrics['decryption'].auth_failures += 1
            logging.error("Authentication failed")
            raise
        except Exception as e:
            self.metrics['decryption'].failures += 1
            logging.error(f"Decryption failed: {str(e)}")
            raise

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        try:
            def safe_mean(values: List[float]) -> float:
                return sum(values) / len(values) if values else 0.0
                
            def safe_std(values: List[float]) -> float:
                if not values:
                    return 0.0
                mean = safe_mean(values)
                variance = sum((x - mean) ** 2 for x in values) / len(values)
                return variance ** 0.5
                
            def get_operation_metrics(metrics: CryptoMetrics) -> Dict[str, Any]:
                total = metrics.successes + metrics.failures
                return {
                    'average_time': safe_mean(metrics.times),
                    'std_time': safe_std(metrics.times),
                    'success_rate': metrics.successes / total if total > 0 else 0.0,
                    'failure_rate': metrics.failures / total if total > 0 else 0.0,
                    'auth_failure_rate': (
                        metrics.auth_failures / total if total > 0 else 0.0
                    ),
                    'average_overhead': safe_mean(metrics.overhead_bytes),
                    'average_memory': safe_mean(metrics.memory_usage),
                    'peak_memory': max(metrics.memory_usage) if metrics.memory_usage else 0,
                    'total_operations': total,
                    'total_bytes_processed': metrics.total_bytes
                }
            
            return {
                'encryption': get_operation_metrics(self.metrics['encryption']),
                'decryption': get_operation_metrics(self.metrics['decryption']),
                'key_operations': {
                    'average_time': safe_mean(self.metrics['key_operations'].times)
                },
                'component_sizes': self.component_sizes
            }
            
        except Exception as e:
            logging.error(f"Error calculating metrics: {str(e)}")
            return {}