import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data
df = pd.read_csv('data/solar_power_predictions_comparison.csv')
df['DATETIME'] = pd.to_datetime(df['DATETIME'])

def create_realistic_consumption(df, target_correlation=0.3):
    """
    Create realistic energy consumption data with proper energy metrics
    """
    np.random.seed(42)  # For reproducibility
    
    # Extract time features
    df['hour'] = df['DATETIME'].dt.hour
    df['day_of_week'] = df['DATETIME'].dt.dayofweek
    df['month'] = df['DATETIME'].dt.month
    df['is_weekend'] = df['day_of_week'] >= 5
    
    # Base consumption patterns
    # 1. Daily pattern: higher during waking hours, lower at night
    daily_pattern = np.zeros(24)
    
    # Typical residential consumption pattern (normalized)
    # Night (12AM-6AM): low consumption
    daily_pattern[0:6] = [0.2, 0.15, 0.1, 0.1, 0.15, 0.3]
    # Morning (6AM-9AM): rising consumption
    daily_pattern[6:9] = [0.5, 0.8, 0.7]
    # Daytime (9AM-5PM): moderate consumption
    daily_pattern[9:17] = [0.6, 0.6, 0.5, 0.5, 0.5, 0.5, 0.6, 0.7]
    # Evening (5PM-12AM): high consumption
    daily_pattern[17:24] = [0.9, 1.0, 0.9, 0.8, 0.7, 0.5, 0.4]
    
    # 2. Seasonal pattern with HVAC loads
    # Base seasonal variation
    seasonal_multiplier = 1 + 0.2 * np.sin((df['month'] - 1) * np.pi / 6)
    
    # HVAC loads: higher in summer (cooling) and winter (heating)
    hvac_load = np.zeros(len(df))
    # Summer cooling (June, July, August) - peaks during afternoon
    summer_mask = df['month'].isin([6, 7, 8])
    hvac_load[summer_mask] = 0.3 * (1 + 0.3 * np.sin((df.loc[summer_mask, 'hour'] - 6) * np.pi / 12))
    
    # Winter heating (December, January, February) - more constant
    winter_mask = df['month'].isin([12, 1, 2])
    hvac_load[winter_mask] = 0.4 * (1 + 0.2 * np.sin((df.loc[winter_mask, 'hour'] - 12) * np.pi / 12))
    
    # 3. Weekend pattern: different consumption behavior
    weekend_multiplier = np.where(df['is_weekend'], 1.1, 1.0)  # Slightly higher on weekends
    
    # Create base consumption using daily pattern
    base_consumption = np.array([daily_pattern[h] for h in df['hour']])
    
    # 4. Solar correlation component (FIXED: negative for lighting reduction)
    # Normalize solar generation
    solar_normalized = (df['XGB_Predicted'] - df['XGB_Predicted'].mean()) / (df['XGB_Predicted'].std() + 1e-6)
    
    # When sun is shining, less lighting needed (negative correlation)
    lighting_reduction = -0.15 * solar_normalized * (df['hour'] >= 6) * (df['hour'] <= 20)
    
    # But more cooling in summer during sunny hours (positive correlation)
    solar_cooling = 0.1 * solar_normalized * summer_mask * (df['hour'] >= 10) * (df['hour'] <= 18)
    
    solar_component = lighting_reduction + solar_cooling
    
    # 5. Add stochastic appliance usage
    # Simulate occasional high-draw appliances (washer, dryer, dishwasher, oven)
    appliance_probability = 0.05  # 5% chance per hour
    appliance_events = np.random.random(len(df)) < appliance_probability
    appliance_load = np.where(appliance_events, np.random.uniform(0.3, 0.8, len(df)), 0)
    # Appliances more likely during certain hours
    appliance_hours_mask = ((df['hour'] >= 7) & (df['hour'] <= 10)) | ((df['hour'] >= 18) & (df['hour'] <= 22))
    appliance_load = appliance_load * appliance_hours_mask
    
    # Combine all components
    consumption = (base_consumption * 0.6 +                    # Base pattern (60%)
                  hvac_load * 0.3 +                            # HVAC loads (30%)
                  solar_component * 0.1 +                      # Solar interaction (10%)
                  appliance_load * 0.2 +                       # Appliance spikes
                  np.random.normal(0, 0.15, len(df)) * 0.2)   # Random noise
    
    # Apply seasonal and weekend patterns
    consumption = consumption * seasonal_multiplier * weekend_multiplier
    
    # Add general random noise for realism
    noise = np.random.normal(0, 0.08, len(df))
    consumption = consumption + noise
    
    # Scale to realistic kWh values for a typical household
    # Typical household: 15-25 kWh per day, so ~0.6-1.0 kWh per hour average
    consumption = (consumption - consumption.min()) / (consumption.max() - consumption.min())
    consumption = consumption * 0.9 + 0.3  # Scale to 0.3-1.2 kWh range
    
    return consumption

def calculate_energy_metrics(df):
    """
    Calculate various energy metrics to validate consumption logic
    """
    metrics = {}
    
    # Basic statistics
    metrics['total_consumption_kwh'] = df['Consumption'].sum()
    metrics['avg_hourly_consumption'] = df['Consumption'].mean()
    metrics['max_hourly_consumption'] = df['Consumption'].max()
    metrics['min_hourly_consumption'] = df['Consumption'].min()
    
    # Daily consumption patterns
    daily_consumption = df.groupby(df['DATETIME'].dt.hour)['Consumption'].mean()
    metrics['peak_hour'] = daily_consumption.idxmax()
    metrics['peak_consumption'] = daily_consumption.max()
    metrics['off_peak_consumption'] = daily_consumption.loc[2:5].mean()  # Early morning hours
    
    # Correlation analysis
    metrics['correlation_with_solar'] = pearsonr(df['XGB_Predicted'], df['Consumption'])[0]
    metrics['correlation_with_actual'] = pearsonr(df['Actual'], df['Consumption'])[0]
    
    # Energy efficiency metrics
    metrics['solar_self_consumption_ratio'] = np.minimum(df['Actual'], df['Consumption']).sum() / df['Consumption'].sum()
    
    # Load factor (average load / peak load)
    metrics['load_factor'] = metrics['avg_hourly_consumption'] / metrics['peak_consumption']
    
    # Daily total consumption
    daily_totals = df.groupby(df['DATETIME'].dt.date)['Consumption'].sum()
    metrics['avg_daily_consumption'] = daily_totals.mean()
    metrics['max_daily_consumption'] = daily_totals.max()
    
    return metrics

def plot_consumption_analysis(df, metrics):
    """
    Create plots to visualize consumption patterns
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Daily consumption pattern
    hourly_avg = df.groupby(df['DATETIME'].dt.hour)['Consumption'].mean()
    axes[0,0].plot(hourly_avg.index, hourly_avg.values, marker='o', linewidth=2)
    axes[0,0].set_title('Average Hourly Consumption Pattern')
    axes[0,0].set_xlabel('Hour of Day')
    axes[0,0].set_ylabel('Consumption (kWh)')
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Consumption vs Solar Generation
    axes[0,1].scatter(df['XGB_Predicted'], df['Consumption'], alpha=0.5)
    axes[0,1].set_title(f'Consumption vs Solar Generation (r={metrics["correlation_with_solar"]:.3f})')
    axes[0,1].set_xlabel('Solar Generation (normalized)')
    axes[0,1].set_ylabel('Consumption (kWh)')
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Correlation Matrix Heatmap
    correlation_matrix = df[['Actual', 'RF_Predicted', 'XGB_Predicted', 'Consumption']].corr()
    im = axes[1,0].imshow(correlation_matrix.values, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    axes[1,0].set_title('Correlation Matrix')
    axes[1,0].set_xticks(range(len(correlation_matrix.columns)))
    axes[1,0].set_yticks(range(len(correlation_matrix.columns)))
    axes[1,0].set_xticklabels(correlation_matrix.columns, rotation=45, ha='right')
    axes[1,0].set_yticklabels(correlation_matrix.columns)
    
    # Add correlation values as text annotations
    for i in range(len(correlation_matrix.columns)):
        for j in range(len(correlation_matrix.columns)):
            text = axes[1,0].text(j, i, f'{correlation_matrix.iloc[i, j]:.3f}',
                               ha="center", va="center", 
                               color="white" if abs(correlation_matrix.iloc[i, j]) > 0.5 else "black",
                               fontweight='bold')
    
    # Add colorbar
    plt.colorbar(im, ax=axes[1,0], label='Correlation Coefficient')
    
    # 4. Monthly consumption pattern
    monthly_avg = df.groupby(df['DATETIME'].dt.month)['Consumption'].mean()
    axes[1,1].bar(monthly_avg.index, monthly_avg.values, alpha=0.7)
    axes[1,1].set_title('Average Monthly Consumption')
    axes[1,1].set_xlabel('Month')
    axes[1,1].set_ylabel('Average Hourly Consumption (kWh)')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Create realistic consumption data
df['Consumption'] = create_realistic_consumption(df, target_correlation=0.3)

# Calculate energy metrics
metrics = calculate_energy_metrics(df)

# Print comprehensive energy metrics report
print("=" * 60)
print("ENERGY CONSUMPTION METRICS REPORT")
print("=" * 60)

print(f"\nBASIC CONSUMPTION STATISTICS:")
print(f"  Total consumption period: {len(df)} hours ({len(df)/24:.1f} days)")
print(f"  Total energy consumed: {metrics['total_consumption_kwh']:.1f} kWh")
print(f"  Average daily consumption: {metrics['avg_daily_consumption']:.1f} kWh/day")
print(f"  Average hourly consumption: {metrics['avg_hourly_consumption']:.3f} kWh")
print(f"  Peak hourly consumption: {metrics['max_hourly_consumption']:.3f} kWh")
print(f"  Minimum hourly consumption: {metrics['min_hourly_consumption']:.3f} kWh")

print(f"\nDAILY PATTERNS:")
print(f"  Peak consumption hour: {metrics['peak_hour']}:00")
print(f"  Peak consumption level: {metrics['peak_consumption']:.3f} kWh")
print(f"  Off-peak consumption: {metrics['off_peak_consumption']:.3f} kWh")
print(f"  Load factor: {metrics['load_factor']:.3f}")

print(f"\nCORRELATION ANALYSIS:")
print(f"  Correlation with solar prediction: {metrics['correlation_with_solar']:.3f}")
print(f"  Correlation with actual solar: {metrics['correlation_with_actual']:.3f}")

print(f"\nENERGY EFFICIENCY METRICS:")
print(f"  Solar self-consumption ratio: {metrics['solar_self_consumption_ratio']:.3f}")

print(f"\nREALISM VALIDATION:")
# Check if metrics are within realistic ranges
realism_checks = {
    "Average daily consumption (10-30 kWh typical)": 10 <= metrics['avg_daily_consumption'] <= 30,
    "Load factor (0.3-0.7 typical)": 0.3 <= metrics['load_factor'] <= 0.7,
    "Peak hour during daytime (8-20)": 8 <= metrics['peak_hour'] <= 20,
    "Reasonable correlation with solar": abs(metrics['correlation_with_solar']) <= 0.5
}

for check, passed in realism_checks.items():
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {status} {check}")

# Generate plots
plot_consumption_analysis(df, metrics)

# Save the enhanced dataset
output_columns = ['DATETIME', 'Actual', 'RF_Predicted', 'XGB_Predicted', 'Consumption']
df[output_columns].to_csv('data/solar_power_predictions_with_realistic_consumption.csv', index=False)

print(f"\nEnhanced dataset saved with realistic consumption data")
print(f"Final correlation with XGB_Predicted: {metrics['correlation_with_solar']:.3f}")