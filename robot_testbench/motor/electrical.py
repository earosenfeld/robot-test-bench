"""
Electrical interface simulation including power supply, communication protocols,
and fault injection capabilities.
"""

import random
import time
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

@dataclass
class PowerSupplyConfig:
    """Configuration for power supply simulation."""
    nominal_voltage: float = 24.0  # V
    voltage_noise_std: float = 0.1  # V
    current_limit: float = 10.0  # A
    voltage_drop_per_amp: float = 0.1  # V/A
    brownout_threshold: float = 20.0  # V
    overvoltage_threshold: float = 28.0  # V

class PowerSupply:
    """Simulates a power supply with voltage regulation, noise, and protection."""
    
    def __init__(self, config: PowerSupplyConfig):
        self.config = config
        self._current_draw = 0.0
        self._fault_active = False
        
    def step(self, current_draw: float, dt: float) -> Tuple[float, bool]:
        """
        Simulate power supply behavior for one time step.
        
        Args:
            current_draw: Current draw in amperes
            dt: Time step in seconds
            
        Returns:
            Tuple of (voltage, fault_active)
        """
        self._current_draw = current_draw
        
        # Check current limit
        if current_draw > self.config.current_limit:
            self._fault_active = True
            return 0.0, True
            
        # Calculate voltage drop due to current draw
        voltage_drop = current_draw * self.config.voltage_drop_per_amp
        
        # Add noise
        noise = np.random.normal(0, self.config.voltage_noise_std)
        
        # Calculate output voltage
        voltage = self.config.nominal_voltage - voltage_drop + noise
        
        # Check voltage limits
        if voltage < self.config.brownout_threshold:
            self._fault_active = True
            return 0.0, True
        elif voltage > self.config.overvoltage_threshold:
            self._fault_active = True
            return 0.0, True
            
        return voltage, False

@dataclass
class I2CConfig:
    """Configuration for I2C interface simulation."""
    clock_frequency: float = 100e3  # Hz
    bit_error_rate: float = 1e-6
    timeout_ms: float = 10.0
    address: int = 0x50

class I2CInterface:
    """Simulates an I2C interface with timing, errors, and faults."""
    
    def __init__(self, config: I2CConfig):
        self.config = config
        self._fault_active = False
        
    def write(self, data: bytes) -> bool:
        """
        Simulate I2C write operation.
        
        Args:
            data: Bytes to write
            
        Returns:
            True if successful, False if failed
        """
        if self._fault_active:
            return False
            
        # Simulate bit errors
        for byte in data:
            if random.random() < self.config.bit_error_rate:
                return False
                
        return True
        
    def read(self, length: int) -> Optional[bytes]:
        """
        Simulate I2C read operation.
        
        Args:
            length: Number of bytes to read
            
        Returns:
            Bytes read or None if failed
        """
        if self._fault_active:
            return None
            
        # Simulate bit errors
        data = bytearray()
        for _ in range(length):
            if random.random() < self.config.bit_error_rate:
                return None
            data.append(random.randint(0, 255))
            
        return bytes(data)

@dataclass
class FaultConfig:
    """Configuration for fault injection."""
    short_circuit_prob: float = 0.0
    open_circuit_prob: float = 0.0
    high_impedance_prob: float = 0.0
    fault_duration_ms: float = 100.0

class FaultInjector:
    """Simulates electrical faults like short circuits, open circuits, etc."""
    
    def __init__(self, config: FaultConfig):
        self.config = config
        self._active_faults: Dict[str, float] = {}  # fault_name -> end_time
        self._sim_time = 0.0  # accumulated simulation time [s] (deterministic)

    def step(self, dt: float) -> List[str]:
        """
        Update fault states and inject new faults.

        Args:
            dt: Time step in seconds

        Returns:
            List of active fault names
        """
        # Advance deterministic simulation clock (no wall-clock time.time()).
        self._sim_time += dt
        current_time = self._sim_time
        self._active_faults = {
            fault: end_time 
            for fault, end_time in self._active_faults.items()
            if end_time > current_time
        }
        
        # Inject new faults
        if random.random() < self.config.short_circuit_prob:
            self._active_faults['short_circuit'] = current_time + self.config.fault_duration_ms/1000
        if random.random() < self.config.open_circuit_prob:
            self._active_faults['open_circuit'] = current_time + self.config.fault_duration_ms/1000
        if random.random() < self.config.high_impedance_prob:
            self._active_faults['high_impedance'] = current_time + self.config.fault_duration_ms/1000
            
        return list(self._active_faults.keys())
        
    def get_fault_impedance(self) -> float:
        """
        Get the current fault impedance.
        
        Returns:
            Impedance in ohms (0 for short, inf for open, high value for high impedance)
        """
        if 'short_circuit' in self._active_faults:
            return 0.0
        elif 'open_circuit' in self._active_faults:
            return float('inf')
        elif 'high_impedance' in self._active_faults:
            return 1e6  # 1 MOhm
        return 0.0  # No fault 