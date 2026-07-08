import numpy as np
import matplotlib.pyplot as plt
from robot_testbench.sensors import (
    EncoderSimulator, EncoderConfig,
    ForceTorqueSensor, ForceTorqueSensorConfig,
    JointAngleSensor, JointAngleSensorConfig
)

def demo_quadrature_encoder():
    """Demonstrate quadrature encoder behavior."""
    # Create encoder with realistic parameters
    config = EncoderConfig(
        counts_per_rev=1000,        # 1000 counts per revolution
        edge_trigger_noise=0.0001,  # 100 us timing noise
        max_frequency=1000.0        # 1 kHz max frequency
    )
    encoder = EncoderSimulator(config)
    
    # Generate test trajectory
    t = np.linspace(0, 2, 1000)
    position = np.sin(2 * np.pi * t)  # 1 Hz sine wave
    velocity = 2 * np.pi * np.cos(2 * np.pi * t)
    
    # Collect encoder readings
    a_signals = []
    b_signals = []
    
    for i in range(len(t)):
        a, b = encoder.update(position[i], velocity[i], 0.002)  # 500 Hz sampling
        a_signals.append(a)
        b_signals.append(b)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    plt.subplot(311)
    plt.plot(t, position)
    plt.title('Position')
    plt.grid(True)
    
    plt.subplot(312)
    plt.plot(t, a_signals)
    plt.title('Channel A')
    plt.grid(True)
    
    plt.subplot(313)
    plt.plot(t, b_signals)
    plt.title('Channel B')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('quadrature_encoder_demo.png')
    plt.close()

def demo_force_torque_sensor():
    """Demonstrate force/torque sensor behavior."""
    # Create sensor with realistic parameters
    config = ForceTorqueSensorConfig(
        sensitivity=1.0,  # 1 N⋅m/V
        noise_std=0.1,    # 100 mV noise
        drift_rate=0.01,  # 10 mV/s drift
        hysteresis=0.1,   # 100 mN⋅m hysteresis
        temp_coeff=0.001  # 1 mV/°C
    )
    sensor = ForceTorqueSensor(config)
    
    # Generate test trajectory
    t = np.linspace(0, 10, 1000)
    force = np.sin(2 * np.pi * 0.1 * t)  # 0.1 Hz sine wave
    
    # Collect sensor readings
    readings = []
    temperatures = np.linspace(20, 30, len(t))  # Temperature sweep
    
    for i in range(len(t)):
        sensor.set_temperature(temperatures[i])
        reading = sensor.update(force[i], 0.01)
        readings.append(reading)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    plt.subplot(211)
    plt.plot(t, force)
    plt.title('Actual Force')
    plt.grid(True)
    
    plt.subplot(212)
    plt.plot(t, readings)
    plt.title('Sensor Reading')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('force_torque_sensor_demo.png')
    plt.close()

def demo_joint_angle_sensor():
    """Demonstrate joint angle sensor behavior."""
    # Create sensor with realistic parameters
    config = JointAngleSensorConfig(
        resolution=0.001,  # 1 mrad resolution
        noise_std=0.0005,  # 0.5 mrad noise
        backlash=0.01,     # 10 mrad backlash
        limit_pos=np.pi,
        limit_neg=-np.pi,
        temp_coeff=0.0001  # 0.1 mrad/°C
    )
    sensor = JointAngleSensor(config)
    
    # Generate test trajectory
    t = np.linspace(0, 10, 1000)
    angle = np.sin(2 * np.pi * 0.1 * t)  # 0.1 Hz sine wave
    velocity = 2 * np.pi * 0.1 * np.cos(2 * np.pi * 0.1 * t)
    
    # Collect sensor readings
    readings = []
    temperatures = np.linspace(20, 30, len(t))  # Temperature sweep
    
    for i in range(len(t)):
        sensor.set_temperature(temperatures[i])
        reading = sensor.update(angle[i], velocity[i])
        readings.append(reading)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    plt.subplot(211)
    plt.plot(t, angle)
    plt.title('Actual Angle')
    plt.grid(True)
    
    plt.subplot(212)
    plt.plot(t, readings)
    plt.title('Sensor Reading')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('joint_angle_sensor_demo.png')
    plt.close()

if __name__ == '__main__':
    print("Running sensor demonstrations...")
    
    print("\nDemonstrating Quadrature Encoder...")
    demo_quadrature_encoder()
    
    print("\nDemonstrating Force/Torque Sensor...")
    demo_force_torque_sensor()
    
    print("\nDemonstrating Joint Angle Sensor...")
    demo_joint_angle_sensor()
    
    print("\nDemonstrations complete. Check the generated PNG files for results.") 