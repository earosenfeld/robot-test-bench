# Copied from robot_testbench/sensors.py
"""
Enhanced sensor models for RobotTestBench.

Includes:
- Physical sensor models (QuadratureEncoder, ForceTorqueSensor, JointAngleSensor)
- Generic noisy signal simulation and filtering (SensorSimulator)
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List
import time
from scipy import signal
import random

@dataclass
class EncoderConfig:
    """Configuration for quadrature encoder."""
    counts_per_rev: int = 1024  # Resolution
    edge_trigger_noise: float = 0.0001  # 100 μs
    max_frequency: float = 1000.0  # Hz
    temp_coeff: float = 0.0001  # counts/°C
    vibration_immunity: float = 5.0  # g
    redundancy_mode: str = 'dual'  # single/dual/triple

class EncoderSimulator:
    """Enhanced quadrature encoder with advanced features."""
    
    def __init__(self, config: EncoderConfig):
        self.config = config
        self.reset()
    
    def reset(self):
        self.position = 0.0  # rad
        self.velocity = 0.0  # rad/s
        self.count = 0
        self.last_time = time.time()
        self.temperature = 25.0  # °C
        self.channel_a = False
        self.channel_b = False
    
    def update(self, true_position: float, true_velocity: float, dt: float) -> Tuple[bool, bool]:
        expected_count = int(true_position * self.config.counts_per_rev / (2 * np.pi))
        temp_offset = self.temperature * self.config.temp_coeff
        expected_count += int(temp_offset)
        if np.random.random() < self.config.edge_trigger_noise / dt:
            expected_count += np.random.choice([-1, 1])
        self.count = expected_count
        self.position = 2 * np.pi * self.count / self.config.counts_per_rev
        self.velocity = true_velocity
        self._update_channels()
        return self.channel_a, self.channel_b
    
    def _update_channels(self):
        count_mod = self.count % 4
        self.channel_a = count_mod in [1, 2]
        self.channel_b = count_mod in [2, 3]
    
    def get_position(self) -> float:
        return self.position
    def get_velocity(self) -> float:
        return self.velocity
    def get_count(self) -> int:
        return self.count
    def set_temperature(self, temperature: float):
        self.temperature = temperature
    def get_temperature(self) -> float:
        return self.temperature

@dataclass
class ForceTorqueSensorConfig:
    """Configuration for force/torque sensor."""
    sensitivity: float = 1.0  # N⋅m/V
    noise_std: float = 0.05  # V
    drift_rate: float = 0.005  # V/s
    hysteresis: float = 0.05  # N⋅m
    cross_coupling: float = 0.01  # N⋅m/N⋅m
    bandwidth: float = 1000.0  # Hz
    overload_limit: float = 10.0  # N⋅m
    temp_coeff: float = 0.001  # V/°C
    calibration_matrix: Optional[np.ndarray] = None

class ForceTorqueSensor:
    """Enhanced force/torque sensor with advanced features."""
    def __init__(self, config: ForceTorqueSensorConfig):
        self.config = config
        self.reset()
        if self.config.calibration_matrix is None:
            self.config.calibration_matrix = np.eye(6)
    def reset(self):
        self.last_reading = 0.0
        self.last_time = time.time()
        self.temperature = 25.0  # °C
        self.history = []
    def update(self, true_torque: float, dt: float) -> float:
        calibrated_torque = self._apply_calibration(true_torque)
        temp_drift = self.temperature * self.config.temp_coeff
        time_drift = self.config.drift_rate * dt
        hysteresis = self._calculate_hysteresis(calibrated_torque)
        noise = np.random.normal(0, self.config.noise_std)
        reading = (calibrated_torque + temp_drift + time_drift + hysteresis + noise)
        reading = self._apply_bandwidth(reading, dt)
        if abs(reading) > self.config.overload_limit:
            reading = np.sign(reading) * self.config.overload_limit
        self.last_reading = reading
        return reading
    def _apply_calibration(self, torque: float) -> float:
        return torque * self.config.sensitivity
    def _calculate_hysteresis(self, torque: float) -> float:
        if not self.history:
            self.history.append(torque)
            return 0.0
        last_torque = self.history[-1]
        delta = torque - last_torque
        if abs(delta) < self.config.hysteresis:
            return 0.0
        else:
            return np.sign(delta) * self.config.hysteresis
    def _apply_bandwidth(self, reading: float, dt: float) -> float:
        # Deterministic first-order low-pass: alpha set by the simulation dt,
        # not wall-clock time, so repeated runs are reproducible.
        if dt <= 0.0:
            return reading
        alpha = 1.0 - np.exp(-2 * np.pi * self.config.bandwidth * dt)
        return alpha * reading + (1 - alpha) * self.last_reading
    def set_temperature(self, temperature: float):
        self.temperature = temperature
    def get_temperature(self) -> float:
        return self.temperature

@dataclass
class JointAngleSensorConfig:
    """Configuration for joint angle sensor."""
    resolution: float = 0.001  # rad
    noise_std: float = 0.0005  # rad
    backlash: float = 0.01  # rad
    temp_coeff: float = 0.0001  # rad/°C
    limit_switches: bool = True
    limit_pos: float = np.pi  # rad
    limit_neg: float = -np.pi  # rad

class JointAngleSensor:
    """Enhanced joint angle sensor with advanced features."""
    def __init__(self, config: JointAngleSensorConfig):
        self.config = config
        self.reset()
    def reset(self):
        self.position = 0.0  # rad
        self.velocity = 0.0  # rad/s
        self.temperature = 25.0  # °C
        self.last_direction = 0
        self.limit_triggered = False
    def update(self, true_position: float, true_velocity: float) -> float:
        quantized_pos = np.round(true_position / self.config.resolution) * self.config.resolution
        temp_offset = self.temperature * self.config.temp_coeff
        quantized_pos += temp_offset
        noise = np.random.normal(0, self.config.noise_std)
        quantized_pos += noise
        direction = np.sign(true_velocity)
        if direction != 0 and direction != self.last_direction:
            quantized_pos += direction * self.config.backlash
        self.last_direction = direction
        if self.config.limit_switches:
            if quantized_pos >= self.config.limit_pos:
                quantized_pos = self.config.limit_pos
                self.limit_triggered = True
            elif quantized_pos <= self.config.limit_neg:
                quantized_pos = self.config.limit_neg
                self.limit_triggered = True
            else:
                self.limit_triggered = False
        self.position = quantized_pos
        self.velocity = true_velocity
        return quantized_pos
    def get_position(self) -> float:
        return self.position
    def get_velocity(self) -> float:
        return self.velocity
    def is_limit_triggered(self) -> bool:
        return self.limit_triggered
    def set_temperature(self, temperature: float):
        self.temperature = temperature
    def get_temperature(self) -> float:
        return self.temperature

# === Generic Noisy Signal Simulation and Filtering ===
"""
The following classes provide generic noisy signal simulation and filtering (lowpass, Kalman).
These are useful for simulating generic sensor noise and post-processing, and are migrated from simulation/sensors.py.
"""

@dataclass
class SensorParameters:
    """Parameters for generic sensor simulation and filtering."""
    position_noise_std: float = 0.001  # rad
    velocity_noise_std: float = 0.01   # rad/s
    current_noise_std: float = 0.1     # A
    sampling_rate: float = 1000.0      # Hz
    filter_type: str = 'lowpass'       # 'lowpass' or 'kalman'
    filter_cutoff: float = 50.0        # Hz (for lowpass)
    kalman_process_noise: float = 0.01  # Reduced process noise
    kalman_measurement_noise: float = 0.1  # Reduced measurement noise

class SensorSimulator:
    """Simulates noisy sensor signals with filtering options (lowpass, Kalman)."""
    def __init__(self, params: SensorParameters):
        self.params = params
        self._setup_filters()
    def _setup_filters(self):
        if self.params.filter_type == 'lowpass':
            self.window_size = int(self.params.sampling_rate / (4 * self.params.filter_cutoff))
            self.position_window = []
            self.velocity_window = []
            self.current_window = []
        elif self.params.filter_type == 'kalman':
            self.kalman_state = np.zeros(2)  # [position, velocity]
            self.kalman_cov = np.eye(2)
            self.dt = 1.0 / self.params.sampling_rate
    def _apply_lowpass_filter(self, x: np.ndarray, window: List[float]) -> np.ndarray:
        if x.size == 1:
            if len(window) >= self.window_size:
                window.pop(0)
            window.append(x[0])
            return np.array([np.mean(window)])
        else:
            filtered = np.zeros_like(x)
            for i in range(len(x)):
                start = max(0, i - self.window_size + 1)
                filtered[i] = np.mean(x[start:i+1])
            return filtered
    def _apply_kalman_filter(self, measurement: np.ndarray) -> np.ndarray:
        F = np.array([[1, self.dt], [0, 1]])
        self.kalman_state = F @ self.kalman_state
        self.kalman_cov = F @ self.kalman_cov @ F.T + np.eye(2) * self.params.kalman_process_noise
        H = np.array([[1, 0]])
        K = self.kalman_cov @ H.T @ np.linalg.inv(H @ self.kalman_cov @ H.T + self.params.kalman_measurement_noise)
        self.kalman_state = self.kalman_state + K @ (measurement - H @ self.kalman_state)
        self.kalman_cov = (np.eye(2) - K @ H) @ self.kalman_cov
        return self.kalman_state
    def add_noise(self, position: float, velocity: float, current: float) -> Tuple[float, float, float]:
        noisy_position = position + np.random.normal(0, self.params.position_noise_std)
        noisy_velocity = velocity + np.random.normal(0, self.params.velocity_noise_std)
        noisy_current = current + np.random.normal(0, self.params.current_noise_std)
        return noisy_position, noisy_velocity, noisy_current
    def filter_signals(self, position: float, velocity: float, current: float) -> Tuple[float, float, float]:
        if self.params.filter_type == 'lowpass':
            pos_array = np.array([position])
            vel_array = np.array([velocity])
            curr_array = np.array([current])
            filtered_position = self._apply_lowpass_filter(pos_array, self.position_window)[0]
            filtered_velocity = self._apply_lowpass_filter(vel_array, self.velocity_window)[0]
            filtered_current = self._apply_lowpass_filter(curr_array, self.current_window)[0]
        elif self.params.filter_type == 'kalman':
            measurement = np.array([position])
            filtered_state = self._apply_kalman_filter(measurement)
            filtered_position = filtered_state[0]
            filtered_velocity = filtered_state[1]
            filtered_current = current
        return filtered_position, filtered_velocity, filtered_current
    def process_signals(self, position: float, velocity: float, current: float) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        raw_position, raw_velocity, raw_current = self.add_noise(position, velocity, current)
        filtered_position, filtered_velocity, filtered_current = self.filter_signals(
            raw_position, raw_velocity, raw_current
        )
        return (
            (raw_position, raw_velocity, raw_current),
            (filtered_position, filtered_velocity, filtered_current)
        )

class DAQSimulator:
    """
    Simulates a data acquisition system with multiple synchronized sensors,
    sampling delay, aliasing, and signal dropout/fault injection.
    """
    def __init__(self, sensors, sample_rate=1000.0, delay=0.0, dropout_prob=0.0, fault_pattern=None):
        """
        Args:
            sensors: dict of {name: sensor_instance}
            sample_rate: DAQ sample rate in Hz
            delay: fixed sampling delay in seconds
            dropout_prob: probability of missing a sample (0.0-1.0)
            fault_pattern: list of (start_idx, end_idx, sensor_name) for forced dropouts
        """
        self.sensors = sensors
        self.sample_rate = sample_rate
        self.delay = delay
        self.dropout_prob = dropout_prob
        self.fault_pattern = fault_pattern or []
        self._sample_idx = 0
        self._fault_active = {name: False for name in sensors}

    def step(self, true_values, dt):
        """
        Simulate a DAQ step.
        Args:
            true_values: dict of {name: (position, velocity, torque, ...)}
            dt: simulation time step
        Returns:
            dict of {name: sensor_reading or None}
        """
        readings = {}
        # Check for forced faults
        for (start, end, name) in self.fault_pattern:
            if start <= self._sample_idx < end:
                self._fault_active[name] = True
            else:
                self._fault_active[name] = False
        for name, sensor in self.sensors.items():
            # Simulate dropout
            if random.random() < self.dropout_prob or self._fault_active.get(name, False):
                readings[name] = None
                continue
            # Simulate delay (for now, just a fixed delay in reporting)
            # In a real system, you would buffer and return delayed values
            # Here, we just call the sensor's update method
            vals = true_values.get(name, (0.0, 0.0, 0.0))
            if hasattr(sensor, 'update'):
                # Try to call update with as many args as possible
                try:
                    reading = sensor.update(*vals[:2], dt)
                except TypeError:
                    try:
                        reading = sensor.update(*vals[:2])
                    except TypeError:
                        reading = sensor.update(vals[0])
                readings[name] = reading
            else:
                readings[name] = None
        self._sample_idx += 1
        return readings 