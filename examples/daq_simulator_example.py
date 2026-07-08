"""
Example usage of the DAQSimulator.
"""

import time
from robot_testbench.sensors import EncoderSimulator, EncoderConfig, ForceTorqueSensor, ForceTorqueSensorConfig, DAQSimulator

def main():
    # Create sensor instances
    encoder_config = EncoderConfig(counts_per_rev=1000)
    encoder = EncoderSimulator(encoder_config)
    torque_config = ForceTorqueSensorConfig(sensitivity=1.0, noise_std=0.05)
    torque_sensor = ForceTorqueSensor(torque_config)

    # Create DAQ simulator with multiple sensors
    daq = DAQSimulator(
        sensors={
            'encoder': encoder,
            'torque': torque_sensor
        },
        sample_rate=1000.0,
        delay=0.001,  # 1ms delay
        dropout_prob=0.1,  # 10% dropout
        fault_pattern=[
            (100, 200, 'encoder'),  # Force encoder dropout from sample 100 to 200
            (300, 400, 'torque')    # Force torque dropout from sample 300 to 400
        ]
    )

    # Simulate a few steps
    for i in range(500):
        # Simulate true values (e.g., from motor simulation)
        true_values = {
            'encoder': (i * 0.1, 0.1, 0.0),  # (position, velocity, torque)
            'torque': (i * 0.1, 0.1, 0.5)    # (position, velocity, torque)
        }
        dt = 0.001  # 1ms time step
        readings = daq.step(true_values, dt)
        print(f"Step {i}: {readings}")
        time.sleep(0.001)  # Simulate real-time delay

if __name__ == '__main__':
    main() 