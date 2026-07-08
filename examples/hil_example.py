"""
Example demonstrating Hardware-in-the-Loop (HIL) simulation with real-time
performance monitoring and fault injection.
"""

import time
from robot_testbench.hil import HILInterface, HILConfig, HILMode

def main():
    # Create HIL interface
    config = HILConfig(
        mode=HILMode.SIMULATION_ONLY,  # Start in simulation-only mode
        target_frequency=1000.0,  # 1kHz target frequency
        sync_tolerance_ms=1.0,
        buffer_size=1000,
        timeout_ms=100.0
    )
    hil = HILInterface(config)
    
    # Start the HIL interface
    hil.start()
    
    try:
        # Simulate for a few seconds
        start_time = time.time()
        while time.time() - start_time < 5.0:
            # Write some test data
            data = {
                'position': 1.0,
                'velocity': 0.5,
                'torque': 2.0
            }
            hil.write_data(data)
            
            # Read any available data
            response = hil.read_data()
            if response:
                print(f"Received data: {response}")
            
            # Get performance metrics
            metrics = hil.get_performance_metrics()
            print(f"\nPerformance metrics:")
            print(f"Average latency: {metrics['avg_latency']*1000:.2f}ms")
            print(f"Maximum latency: {metrics['max_latency']*1000:.2f}ms")
            print(f"Average jitter: {metrics['avg_jitter']*1000:.2f}ms")
            print(f"Maximum jitter: {metrics['max_jitter']*1000:.2f}ms")
            
            # Inject a fault every second
            if int(time.time() - start_time) % 2 == 0:
                print("\nInjecting hardware fault...")
                hil.inject_hardware_fault("communication_error", 100.0)
            
            time.sleep(0.1)  # Print metrics every 100ms
            
    finally:
        # Stop the HIL interface
        hil.stop()

if __name__ == '__main__':
    main() 