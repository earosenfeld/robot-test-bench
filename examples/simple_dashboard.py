"""
Enhanced visualization dashboard for RobotTestBench.
"""

import dash
from dash import dcc, html
import plotly.graph_objects as go
import numpy as np
from robot_testbench.motor import MotorParameters, MotorSimulator
from robot_testbench.sensors import (
    ForceTorqueSensor, ForceTorqueSensorConfig,
    EncoderSimulator, EncoderConfig,
    JointAngleSensor, JointAngleSensorConfig
)

# Create motor and sensors
motor_params = MotorParameters(
    inertia=0.05,
    damping=1.0,
    torque_constant=0.1,
    max_torque=4.0,
    max_speed=20.0,
    resistance=1.0,
    inductance=0.1,
    thermal_mass=0.1,
    thermal_resistance=0.5,
    gear_ratio=10.0,
    gear_efficiency=0.95
)
motor = MotorSimulator(motor_params)

ft_sensor = ForceTorqueSensor(ForceTorqueSensorConfig())
encoder = EncoderSimulator(EncoderConfig())
ja_sensor = JointAngleSensor(JointAngleSensorConfig())

# Generate time points
t = np.linspace(0, 10, 1000)
dt = t[1] - t[0]

# Generate setpoints
position_setpoint = np.sin(t) * np.pi
velocity_setpoint = np.cos(t) * np.pi
torque_setpoint = -np.sin(t) * 2.0

# Initialize arrays for data
motor_positions = []
motor_velocities = []
motor_currents = []
motor_torques = []
motor_temperatures = []
motor_efficiencies = []
motor_power_losses = []

sensor_positions = []
sensor_velocities = []
sensor_torques = []
encoder_a = []
encoder_b = []
ja_angles = []
ft_voltages = []

# Run simulation
for i in range(len(t)):
    # Step motor
    position, velocity, current = motor.step(dt, voltage=12.0)
    
    # Update sensors
    torque = ft_sensor.update(motor.get_state()['torque'], dt)
    a, b = encoder.update(position, velocity, dt)
    angle = ja_sensor.update(position, velocity)
    
    # Store data
    motor_positions.append(position)
    motor_velocities.append(velocity)
    motor_currents.append(current)
    motor_torques.append(motor.get_state()['torque'])
    motor_temperatures.append(motor.get_temperature())
    motor_efficiencies.append(motor.get_efficiency())
    motor_power_losses.append(motor.get_power_loss())
    
    sensor_positions.append(encoder.get_position())
    sensor_velocities.append(encoder.get_velocity())
    sensor_torques.append(torque)
    encoder_a.append(1 if a else 0)
    encoder_b.append(1 if b else 0)
    ja_angles.append(angle)
    ft_voltages.append(torque / ft_sensor.config.sensitivity)

# Common layout settings
common_layout = {
    'template': 'plotly_white',
    'font': {'family': 'Arial', 'size': 12},
    'margin': {'l': 50, 'r': 50, 't': 50, 'b': 50},
    'showlegend': True,
    'legend': {'x': 0, 'y': 1},
    'hovermode': 'x unified'
}

# Common x-axis settings
xaxis_settings = {
    'title': 'Time [s]',
    'gridcolor': '#e0e0e0',
    'showgrid': True,
    'zeroline': True,
    'zerolinecolor': '#969696',
    'zerolinewidth': 1
}

# Common y-axis settings
yaxis_common = {
    'gridcolor': '#e0e0e0',
    'showgrid': True,
    'zeroline': True,
    'zerolinecolor': '#969696',
    'zerolinewidth': 1,
    'title': {
        'font': {'size': 16},
        'standoff': 15
    }
}

# Create the layout
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("RobotTestBench Enhanced Test Dashboard", 
            style={'textAlign': 'center', 'color': '#2c3e50', 'margin': '20px'}),
    
    # Position plot
    dcc.Graph(
        id='position-plot',
        figure={
            'data': [
                {
                    'x': t,
                    'y': position_setpoint,
                    'type': 'scatter',
                    'mode': 'lines',
                    'name': 'Position Setpoint',
                    'line': {'width': 2, 'color': '#1f77b4'}
                },
                {
                    'x': t,
                    'y': motor_positions,
                    'type': 'scatter',
                    'mode': 'lines',
                    'name': 'Motor Position',
                    'line': {'width': 2, 'color': '#ff7f0e'}
                },
                {
                    'x': t,
                    'y': sensor_positions,
                    'type': 'scatter',
                    'mode': 'lines',
                    'name': 'Sensor Position',
                    'line': {'width': 2, 'color': '#2ca02c'}
                }
            ],
            'layout': {
                **common_layout,
                'title': {
                    'text': 'Motor Position vs Time',
                    'font': {'size': 24, 'color': '#2c3e50'},
                    'y': 0.95
                },
                'xaxis': xaxis_settings,
                'yaxis': {
                    **yaxis_common,
                    'title': {
                        'text': 'Position [rad]',
                        'font': {'size': 14},
                        'standoff': 10
                    }
                }
            }
        },
        style={'height': '400px', 'margin': '20px'}
    ),
    
    # Velocity plot
    dcc.Graph(
        id='velocity-plot',
        figure={
            'data': [
                {
                    'x': t,
                    'y': velocity_setpoint,
                    'type': 'scatter',
                    'mode': 'lines',
                    'name': 'Velocity Setpoint',
                    'line': {'width': 2, 'color': '#1f77b4'}
                },
                {
                    'x': t,
                    'y': motor_velocities,
                    'type': 'scatter',
                    'mode': 'lines',
                    'name': 'Motor Velocity',
                    'line': {'width': 2, 'color': '#ff7f0e'}
                },
                {
                    'x': t,
                    'y': sensor_velocities,
                    'type': 'scatter',
                    'mode': 'lines',
                    'name': 'Sensor Velocity',
                    'line': {'width': 2, 'color': '#2ca02c'}
                }
            ],
            'layout': {
                **common_layout,
                'title': {
                    'text': 'Motor Velocity vs Time',
                    'font': {'size': 24, 'color': '#2c3e50'},
                    'y': 0.95
                },
                'xaxis': xaxis_settings,
                'yaxis': {
                    **yaxis_common,
                    'title': {
                        'text': 'Velocity [rad/s]',
                        'font': {'size': 14},
                        'standoff': 10
                    }
                }
            }
        },
        style={'height': '400px', 'margin': '20px'}
    ),
    
    # Torque/Current plot
    dcc.Graph(
        id='torque-plot',
        figure={
            'data': [
                {
                    'x': t,
                    'y': torque_setpoint,
                    'type': 'scatter',
                    'mode': 'lines',
                    'name': 'Torque Setpoint',
                    'line': {'width': 2, 'color': '#1f77b4'}
                },
                {
                    'x': t,
                    'y': motor_torques,
                    'type': 'scatter',
                    'mode': 'lines',
                    'name': 'Motor Torque',
                    'line': {'width': 2, 'color': '#ff7f0e'}
                },
                {
                    'x': t,
                    'y': sensor_torques,
                    'type': 'scatter',
                    'mode': 'lines',
                    'name': 'Sensor Torque',
                    'line': {'width': 2, 'color': '#2ca02c'}
                }
            ],
            'layout': {
                **common_layout,
                'title': {
                    'text': 'Motor Torque and Current vs Time',
                    'font': {'size': 24, 'color': '#2c3e50'},
                    'y': 0.95
                },
                'xaxis': xaxis_settings,
                'yaxis': {
                    **yaxis_common,
                    'title': {
                        'text': 'Torque [N⋅m] / Current [A]',
                        'font': {'size': 14},
                        'standoff': 10
                    }
                }
            }
        },
        style={'height': '400px', 'margin': '20px'}
    ),
    
    # Thermal and Efficiency plot
    dcc.Graph(
        id='thermal-plot',
        figure={
            'data': [
                {
                    'x': t,
                    'y': motor_temperatures,
                    'type': 'scatter',
                    'mode': 'lines',
                    'name': 'Motor Temperature',
                    'line': {'width': 2, 'color': '#d62728'}
                },
                {
                    'x': t,
                    'y': motor_efficiencies,
                    'type': 'scatter',
                    'mode': 'lines',
                    'name': 'Motor Efficiency',
                    'line': {'width': 2, 'color': '#9467bd'},
                    'yaxis': 'y2'
                },
                {
                    'x': t,
                    'y': motor_power_losses,
                    'type': 'scatter',
                    'mode': 'lines',
                    'name': 'Power Loss',
                    'line': {'width': 2, 'color': '#8c564b'},
                    'yaxis': 'y3'
                }
            ],
            'layout': {
                **common_layout,
                'title': {
                    'text': 'Motor Thermal and Efficiency Performance',
                    'font': {'size': 24, 'color': '#2c3e50'},
                    'y': 0.95
                },
                'xaxis': xaxis_settings,
                'yaxis': {
                    **yaxis_common,
                    'title': {
                        'text': 'Temperature [°C]',
                        'font': {'size': 14},
                        'standoff': 10
                    }
                },
                'yaxis2': {
                    **yaxis_common,
                    'title': {
                        'text': 'Efficiency [-]',
                        'font': {'size': 14},
                        'standoff': 10
                    },
                    'overlaying': 'y',
                    'side': 'right'
                },
                'yaxis3': {
                    **yaxis_common,
                    'title': {
                        'text': 'Power Loss [W]',
                        'font': {'size': 14},
                        'standoff': 10
                    },
                    'overlaying': 'y',
                    'side': 'right',
                    'anchor': 'free',
                    'position': 1.0
                }
            }
        },
        style={'height': '400px', 'margin': '20px'}
    ),
    
    # Quadrature Encoder Channels
    dcc.Graph(
        id='encoder-plot',
        figure={
            'data': [
                {'x': t, 'y': encoder_a, 'type': 'scatter', 'mode': 'lines+markers', 
                 'name': 'Encoder A (Logic High/Low)', 'line': {'width': 2, 'color': '#d62728', 'shape': 'hv'}},
                {'x': t, 'y': encoder_b, 'type': 'scatter', 'mode': 'lines+markers', 
                 'name': 'Encoder B (Logic High/Low, offset)', 'line': {'width': 2, 'color': '#9467bd', 'shape': 'hv'}},
            ],
            'layout': {
                **common_layout,
                'title': {'text': 'Quadrature Encoder Channels vs Time', 'font': {'size': 22, 'color': '#2c3e50'}, 'y': 0.95},
                'xaxis': xaxis_settings,
                'yaxis': {
                    **yaxis_common,
                    'title': {
                        'text': 'Logic Level (0 = Low, 1 = High)\nB channel offset for clarity',
                        'font': {'size': 14},
                        'standoff': 10
                    },
                    'dtick': 0.2,
                    'range': [-0.3, 1.2]
                }
            }
        },
        style={'height': '250px', 'margin': '20px'}
    ),
    
    # Force/Torque Sensor Output
    dcc.Graph(
        id='ft-sensor-plot',
        figure={
            'data': [
                {'x': t, 'y': ft_voltages, 'type': 'scatter', 'mode': 'lines', 
                 'name': 'Force/Torque Sensor Voltage', 'line': {'width': 2, 'color': '#8c564b'}},
            ],
            'layout': {
                **common_layout,
                'title': {'text': 'Force/Torque Sensor Output vs Time', 'font': {'size': 22, 'color': '#2c3e50'}, 'y': 0.95},
                'xaxis': xaxis_settings,
                'yaxis': {
                    **yaxis_common,
                    'title': {
                        'text': 'Sensor Output [V]\n(Includes noise, drift, hysteresis)',
                        'font': {'size': 14},
                        'standoff': 10
                    }
                }
            }
        },
        style={'height': '250px', 'margin': '20px'}
    ),
    
    # Joint Angle Sensor Output
    dcc.Graph(
        id='ja-sensor-plot',
        figure={
            'data': [
                {'x': t, 'y': ja_angles, 'type': 'scatter', 'mode': 'lines', 
                 'name': 'Joint Angle Sensor', 'line': {'width': 2, 'color': '#e377c2'}},
            ],
            'layout': {
                **common_layout,
                'title': {'text': 'Joint Angle Sensor Output vs Time', 'font': {'size': 22, 'color': '#2c3e50'}, 'y': 0.95},
                'xaxis': xaxis_settings,
                'yaxis': {
                    **yaxis_common,
                    'title': {
                        'text': 'Angle [rad]\n(Quantized, noisy, with backlash)',
                        'font': {'size': 14},
                        'standoff': 10
                    }
                }
            }
        },
        style={'height': '250px', 'margin': '20px'}
    ),
], style={'backgroundColor': '#f8f9fa', 'padding': '20px'})

if __name__ == '__main__':
    app.run_server(debug=True) 