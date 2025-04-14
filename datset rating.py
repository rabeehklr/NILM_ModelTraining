import pandas as pd
import numpy as np
from datetime import timedelta, datetime
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
DATASET_PATH = r'C:\Users\ASUS\Desktop\S8 PROJECT\1\nilm_synthetic_dataset_2.csv'  # Update with your path
BASE_DATETIME = datetime(2025, 2, 26, 0, 0, 0)  # Arbitrary start for absolute timestamps
OUTPUT_DIR = r'C:\Users\ASUS\Desktop\S8 PROJECT\1\visualizations'  # Directory to save graphs

# Specific appliances to focus on
FOCUS_APPLIANCES = ['laptop charger', 'bulb', 'mobile charger']

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_timestamp(ts):
    """Convert MM:SS.s format to timedelta."""
    try:
        minutes, seconds = ts.split(':')
        total_seconds = int(minutes) * 60 + float(seconds)
        return timedelta(seconds=total_seconds)
    except ValueError as e:
        raise ValueError(f"Invalid timestamp format: {ts}. Expected MM:SS.s (e.g., 53:25.5)")

def load_dataset(path):
    """Load the dataset and validate its structure."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found at {path}")
    
    df = pd.read_csv(path)
    
    # Parse timestamp as duration and convert to absolute datetime
    df['timedelta'] = df['timestamp'].apply(parse_timestamp)
    df['timestamp'] = BASE_DATETIME + df['timedelta']
    return df

def plot_power_timeseries(df):
    """Plot total power and individual appliance power over time."""
    # Sample the dataframe if it's very large to improve plot readability
    if len(df) > 5000:
        df_sample = df.iloc[::len(df)//5000].copy()
    else:
        df_sample = df.copy()
    
    plt.figure(figsize=(15, 8))
    
    # Plot total power
    plt.subplot(2, 1, 1)
    plt.plot(df_sample['timestamp'], df_sample['total_power'], label='Total Power', color='blue')
    plt.plot(df_sample['timestamp'], df_sample['active_power'], label='Active Power', color='green', alpha=0.7)
    plt.title('Aggregate Power Consumption Over Time', fontsize=14)
    plt.xlabel('Time')
    plt.ylabel('Power (W)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot focused appliances
    plt.subplot(2, 1, 2)
    colors = ['red', 'purple', 'orange']  # Distinct colors for each appliance
    markers = ['o', 's', '^']  # Different markers for ON status
    
    for i, app in enumerate(FOCUS_APPLIANCES):
        power_col = f'{app}_power'
        status_col = f'{app}_status'
        
        # Use status to color the points - ON vs OFF
        on_mask = df_sample[status_col] == 1
        
        # Plot points when appliance is ON with solid color
        plt.scatter(df_sample.loc[on_mask, 'timestamp'], 
                   df_sample.loc[on_mask, power_col],
                   label=f'{app} (ON)', 
                   color=colors[i], 
                   marker=markers[i],
                   s=30, 
                   alpha=0.7)
        
        # Plot points when appliance is OFF with faded color
        plt.scatter(df_sample.loc[~on_mask, 'timestamp'], 
                   df_sample.loc[~on_mask, power_col],
                   label=f'{app} (OFF)', 
                   color=colors[i], 
                   marker='.',
                   s=10, 
                   alpha=0.2)
    
    plt.title('Power Consumption of Selected Appliances', fontsize=14)
    plt.xlabel('Time')
    plt.ylabel('Power (W)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'power_timeseries.png'), dpi=300)
    plt.close()
    
    print(f"Power time series plot saved to {os.path.join(OUTPUT_DIR, 'power_timeseries.png')}")

def plot_appliance_status_probability(df):
    """Visualize ON/OFF status probability for selected appliances."""
    # Calculate ON/OFF probabilities
    status_data = []
    for app in FOCUS_APPLIANCES:
        status_col = f'{app}_status'
        on_prob = df[status_col].mean()
        off_prob = 1 - on_prob
        status_data.append({
            'Appliance': app,
            'ON': on_prob,
            'OFF': off_prob,
            'ON Percentage': f"{on_prob*100:.1f}%",
            'OFF Percentage': f"{off_prob*100:.1f}%"
        })
    status_df = pd.DataFrame(status_data)

    # Define the layout using GridSpec
    num_appliances = len(FOCUS_APPLIANCES)
    fig = plt.figure(figsize=(15, 6))
    gs = plt.GridSpec(1, num_appliances + 1, width_ratios=[1.5] + [1] * num_appliances)  # Bar chart gets more width

    # Bar chart showing ON/OFF probability
    ax_bar = fig.add_subplot(gs[0, 0])
    status_df[['Appliance', 'ON', 'OFF']].set_index('Appliance').plot(
        kind='barh', 
        stacked=True, 
        color=['forestgreen', 'lightgray'],
        ax=ax_bar
    )
    
    # Add percentage labels on the bars
    for i, app in enumerate(status_df['Appliance']):
        on_pct = status_df.loc[i, 'ON Percentage']
        ax_bar.text(
            status_df.loc[i, 'ON'] / 2,
            i,
            on_pct,
            ha='center',
            va='center',
            color='white',
            fontweight='bold'
        )
        off_pct = status_df.loc[i, 'OFF Percentage']
        ax_bar.text(
            status_df.loc[i, 'ON'] + status_df.loc[i, 'OFF'] / 2,
            i,
            off_pct,
            ha='center',
            va='center',
            color='black',
            fontweight='bold'
        )
    
    ax_bar.set_title('Appliance ON/OFF Probability Distribution', fontsize=14)
    ax_bar.set_xlabel('Probability')
    ax_bar.set_xlim(0, 1)
    ax_bar.grid(axis='x', alpha=0.3)

    # Pie charts for each appliance
    colors = ['forestgreen', 'lightgray']
    for i, app in enumerate(FOCUS_APPLIANCES):
        ax_pie = fig.add_subplot(gs[0, i + 1])
        values = [status_df.loc[i, 'ON'], status_df.loc[i, 'OFF']]
        
        wedges, texts, autotexts = ax_pie.pie(
            values,
            labels=['ON', 'OFF'],
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            wedgeprops={'linewidth': 1, 'edgecolor': 'white'}
        )
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax_pie.set_title(app)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'appliance_status_probability.png'), dpi=300)
    plt.close()
    
    print(f"Appliance status probability plot saved to {os.path.join(OUTPUT_DIR, 'appliance_status_probability.png')}")

def main():
    try:
        print(f"Loading dataset from {DATASET_PATH}")
        df = load_dataset(DATASET_PATH)
        
        # Generate the requested visualizations
        print("Generating power time series plot...")
        plot_power_timeseries(df)
        
        print("Generating appliance status probability plots...")
        plot_appliance_status_probability(df)
        
        print(f"All visualizations saved to {OUTPUT_DIR}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()