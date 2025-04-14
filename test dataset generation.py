import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import csv
import os

class ApplianceSimulator:
    def __init__(self, name, min_power, max_power, daily_usage_patterns, startup_power=None):
        """
        Enhanced appliance simulator with more realistic behavior
        
        :param name: Name of the appliance
        :param min_power: Minimum power consumption (watts)
        :param max_power: Maximum power consumption (watts)
        :param daily_usage_patterns: Detailed usage patterns
        :param startup_power: Optional additional power during startup
        """
        self.name = name
        self.min_power = min_power
        self.max_power = max_power
        self.daily_usage_patterns = daily_usage_patterns
        self.startup_power = startup_power or (max_power * 1.2)
        
        # Track state for more realistic simulation
        self.current_state = 'off'
        self.state_duration = 0
        self.next_state_change = 0

def generate_realistic_nilm_dataset(duration_days=7, sample_rate=1, output_file='nilm_synthetic_dataset.csv'):
    """
    Generate an enhanced synthetic NILM dataset with more realistic scenarios
    
    :param duration_days: Number of days to simulate
    :param sample_rate: Number of samples per second
    :param output_file: Output CSV file path
    :return: None (directly writes to CSV)
    """
    # Enhanced appliance definitions with more complex usage patterns
    appliances = [
        ApplianceSimulator('Table_Fan', 50, 60, {
            'night': {'on_prob': 0.1, 'min_duration': 300, 'max_duration': 3600},
            'morning': {'on_prob': 0.4, 'min_duration': 600, 'max_duration': 7200},
            'afternoon': {'on_prob': 0.6, 'min_duration': 900, 'max_duration': 5400},
            'evening': {'on_prob': 0.8, 'min_duration': 1200, 'max_duration': 10800}
        }),
        ApplianceSimulator('Mobile_Charger', 20, 40, {
            'night': {'on_prob': 0.7, 'min_duration': 3600, 'max_duration': 28800},
            'morning': {'on_prob': 0.5, 'min_duration': 1800, 'max_duration': 7200},
            'afternoon': {'on_prob': 0.3, 'min_duration': 1200, 'max_duration': 5400},
            'evening': {'on_prob': 0.6, 'min_duration': 2700, 'max_duration': 10800}
        }, startup_power=50),
        ApplianceSimulator('Laptop_Charger', 150, 180, {
            'night': {'on_prob': 0.6, 'min_duration': 7200, 'max_duration': 28800},
            'morning': {'on_prob': 0.7, 'min_duration': 3600, 'max_duration': 10800},
            'afternoon': {'on_prob': 0.5, 'min_duration': 2700, 'max_duration': 7200},
            'evening': {'on_prob': 0.4, 'min_duration': 3600, 'max_duration': 14400}
        }, startup_power=200),
        ApplianceSimulator('LED_Bulb', 10, 20, {
            'night': {'on_prob': 0.9, 'min_duration': 3600, 'max_duration': 28800},
            'morning': {'on_prob': 0.1, 'min_duration': 300, 'max_duration': 1800},
            'afternoon': {'on_prob': 0.2, 'min_duration': 300, 'max_duration': 3600},
            'evening': {'on_prob': 0.9, 'min_duration': 7200, 'max_duration': 14400}
        }),
        ApplianceSimulator('Refrigerator', 350, 500, {
            'night': {'on_prob': 1.0, 'min_duration': 28800, 'max_duration': 86400},
            'morning': {'on_prob': 1.0, 'min_duration': 28800, 'max_duration': 86400},
            'afternoon': {'on_prob': 1.0, 'min_duration': 28800, 'max_duration': 86400},
            'evening': {'on_prob': 1.0, 'min_duration': 28800, 'max_duration': 86400}
        }, startup_power=600),
        ApplianceSimulator('Television', 60, 150, {
            'night': {'on_prob': 0.2, 'min_duration': 1800, 'max_duration': 7200},
            'morning': {'on_prob': 0.1, 'min_duration': 600, 'max_duration': 3600},
            'afternoon': {'on_prob': 0.3, 'min_duration': 1200, 'max_duration': 5400},
            'evening': {'on_prob': 0.8, 'min_duration': 5400, 'max_duration': 18000}
        }, startup_power=200)
    ]

    # Prepare data storage
    total_samples = duration_days * 24 * 3600 * sample_rate
    print(f"Generating dataset with {total_samples} samples...")
    
    # Random seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    # Prepare CSV file for writing
    try:
        with open(output_file, 'w', newline='') as csvfile:
            # Prepare CSV writer
            csv_writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
            
            # Prepare header
            header = ['timestamp', 'total_power', 'voltage', 'current', 'active_power']
            for appliance in appliances:
                header.extend([
                    f'{appliance.name}_status', 
                    f'{appliance.name}_power'
                ])
            csv_writer.writerow(header)

            # Generate data
            for i in range(total_samples):
                # Create timestamp
                timestamp = (datetime(2024, 1, 1) + timedelta(seconds=i/sample_rate)).isoformat()
                
                # Initialize power and status data
                total_power = 0
                appliance_data = {}

                # Simulate each appliance
                for appliance in appliances:
                    # Determine time of day
                    hour = (datetime(2024, 1, 1) + timedelta(seconds=i/sample_rate)).hour
                    if 0 <= hour < 6:
                        time_category = 'night'
                    elif 6 <= hour < 12:
                        time_category = 'morning'
                    elif 12 <= hour < 18:
                        time_category = 'afternoon'
                    else:
                        time_category = 'evening'
                    
                    # More sophisticated state management
                    usage_pattern = appliance.daily_usage_patterns[time_category]
                    
                    # Manage appliance state
                    if appliance.state_duration <= 0:
                        # Decide next state
                        if random.random() < usage_pattern['on_prob']:
                            appliance.current_state = 'on'
                            appliance.state_duration = random.randint(
                                usage_pattern['min_duration'], 
                                usage_pattern['max_duration']
                            )
                        else:
                            appliance.current_state = 'off'
                            appliance.state_duration = random.randint(
                                usage_pattern['min_duration'], 
                                usage_pattern['max_duration']
                            )
                    
                    # Decrement state duration
                    appliance.state_duration -= 1
                    
                    # Set status and power
                    if appliance.current_state == 'on':
                        status = 1
                        # Simulate startup and steady-state power
                        if appliance.state_duration > usage_pattern['max_duration'] - 10:
                            # Startup phase
                            base_power = np.random.uniform(
                                appliance.startup_power * 0.8, 
                                appliance.startup_power * 1.2
                            )
                        else:
                            # Steady-state
                            base_power = np.random.uniform(appliance.min_power, appliance.max_power)
                        
                        # Add noise
                        noise = np.random.normal(0, base_power * 0.03)  # 3% noise
                        appliance_power = max(0, base_power + noise)
                    else:
                        status = 0
                        appliance_power = 0
                    
                    # Store appliance data
                    appliance_data[appliance.name] = {
                        'status': status,
                        'power': appliance_power
                    }
                    
                    # Update total power
                    total_power += appliance_power

                # Simulate voltage and current
                base_voltage = 230
                voltage = base_voltage + np.random.normal(0, 0.5)
                current = total_power / voltage

                # Prepare row data
                row_data = [
                    timestamp, 
                    round(total_power, 2), 
                    round(voltage, 2), 
                    round(current, 2), 
                    round(total_power, 2)
                ]

                # Add appliance status and power
                for appliance in appliances:
                    row_data.extend([
                        appliance_data[appliance.name]['status'],
                        round(appliance_data[appliance.name]['power'], 2)
                    ])

                # Write row to CSV
                csv_writer.writerow(row_data)

                # Progress tracking
                if i % (total_samples // 10) == 0:
                    print(f"Progress: {i/total_samples*100:.2f}%")

    except Exception as e:
        print(f"Error generating dataset: {e}")
        raise

    print(f"\nDataset generated successfully. Saved to {output_file}")
    print(f"Total samples: {total_samples}")

def main():
    # Output file path
    output_path = 'nilm_synthetic_dataset.csv'
    
    # Generate 7 days of data with 1 sample per second
    generate_realistic_nilm_dataset(
        duration_days=7, 
        sample_rate=1, 
        output_file=output_path
    )

    # Optional: Verify file and print first few lines
    print("\nFirst few lines of the dataset:")
    with open(output_path, 'r') as f:
        for _ in range(5):
            print(f.readline().strip())

if __name__ == "__main__":
    main()