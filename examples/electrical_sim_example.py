"""
Example demonstrating electrical interface simulation including power supply,
I2C communication, and fault injection.
"""

import time
from robot_testbench.motor.electrical import (
    PowerSupply, PowerSupplyConfig,
    I2CInterface, I2CConfig,
    FaultInjector, FaultConfig
)

def main():
    # Create power supply
    ps_config = PowerSupplyConfig(
        nominal_voltage=24.0,
        voltage_noise_std=0.1,
        current_limit=10.0,
        voltage_drop_per_amp=0.1
    )
    power_supply = PowerSupply(ps_config)
    
    # Create I2C interface
    i2c_config = I2CConfig(
        clock_frequency=100e3,
        bit_error_rate=1e-6,
        address=0x50
    )
    i2c = I2CInterface(i2c_config)
    
    # Create fault injector
    fault_config = FaultConfig(
        short_circuit_prob=0.01,
        open_circuit_prob=0.01,
        high_impedance_prob=0.01,
        fault_duration_ms=100.0
    )
    fault_injector = FaultInjector(fault_config)
    
    # Simulate for a few seconds
    dt = 0.001  # 1ms time step
    for i in range(5000):
        # Simulate power supply
        current_draw = 5.0 + 2.0 * (i % 100) / 100  # Oscillating current draw
        voltage, ps_fault = power_supply.step(current_draw, dt)
        
        # Simulate I2C communication
        if i % 100 == 0:  # Every 100ms
            data = bytes([0x01, 0x02, 0x03])
            write_success = i2c.write(data)
            if write_success:
                response = i2c.read(4)
                print(f"I2C Read: {response}")
        
        # Simulate fault injection
        active_faults = fault_injector.step(dt)
        if active_faults:
            print(f"Active faults: {active_faults}")
            print(f"Fault impedance: {fault_injector.get_fault_impedance()}")
        
        # Print status every second
        if i % 1000 == 0:
            print(f"\nTime: {i*dt:.1f}s")
            print(f"Voltage: {voltage:.2f}V")
            print(f"Current: {current_draw:.2f}A")
            print(f"Power Supply Fault: {ps_fault}")
        
        time.sleep(dt)

if __name__ == '__main__':
    main() 