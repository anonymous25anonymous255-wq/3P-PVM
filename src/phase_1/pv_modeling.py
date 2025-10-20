import numpy as np
import pandas as pd
def calculate_pv_power(df, system_params=None):
    """
    Calculate pv production in kwh output from NASA POWER solar data
    
    Parameters:
    -----------
    df : pandas DataFrame
        NASA POWER data with required columns
    system_params : dict
        PV system parameters (optional, uses defaults if not provided)
    
    Returns:
    --------
    pandas DataFrame with added pv production in kwh column
    """
    
    # Default PV system parameters
    if system_params is None:
        system_params = {
            'capacity_kw': 1,          # System capacity in kW (DC)
            'dc_ac_ratio': 1.2,          # DC to AC ratio
            'inverter_eff': 0.96,        # Inverter efficiency
            'temp_coeff': -0.0045,       # Temperature coefficient (%/°C)
            'system_losses': 0.14,        # Other system losses (14%)
            'tilt': 30,                   # Panel tilt angle (degrees)
            'azimuth': 180,               # Panel azimuth (180 = south-facing)
            'soiling_loss': 0.02,        # Soiling losses (2%)
            'ref_temp': 25,               # Reference temperature (°C)
            'noct': 45                    # Nominal Operating Cell Temperature (°C)
        }
    
    # Create a copy of the dataframe
    df_result = df.copy()
    
    # Extract required columns
    ghi = df['ALLSKY_SFC_SW_DWN'].values  # Global Horizontal Irradiance (W/m²)
    dhi = df['ALLSKY_SFC_SW_DIFF'].values # Diffuse Horizontal Irradiance (W/m²)
    temp_ambient = df['T2M'].values       # Ambient temperature (°C)
    wind_speed = df['WS10M'].values       # Wind speed (m/s)
    
    # Calculate cell temperature using NOCT model
    # Tcell = Tamb + (NOCT - 20) * (G / 800)
    irradiance_ratio = ghi / 800
    temp_cell = temp_ambient + (system_params['noct'] - 20) * irradiance_ratio
    
    # Wind cooling effect (simplified)
    wind_cooling = 1.5 * wind_speed
    temp_cell = temp_cell - wind_cooling
    
    # Calculate temperature derating factor
    temp_derate = 1 + system_params['temp_coeff'] * (temp_cell - system_params['ref_temp'])
    temp_derate = np.clip(temp_derate, 0, 1.2)  # Limit to reasonable range
    
    # Calculate POA (Plane of Array) irradiance - simplified method
    # For a more accurate calculation, you would need solar position calculations
    
    # Simple approximation for tilted surface
    tilt_factor = 1.0  # Simplified - in reality this varies with sun position
    
    # During daylight hours, use a weighted combination
    # Use default 30° tilt if not specified (for compatibility)
    tilt_angle = system_params.get('tilt', 30)
    poa_irradiance = np.where(ghi > 0,
                              ghi * tilt_factor + dhi * (1 + np.cos(np.radians(tilt_angle)))/2,
                              0)
    
    # Normalize POA irradiance (1000 W/m² is standard test condition)
    poa_normalized = poa_irradiance / 1000
    
    # Calculate DC power before losses
    dc_power = system_params['capacity_kw'] * poa_normalized * temp_derate
    
    # Apply soiling losses
    dc_power = dc_power * (1 - system_params['soiling_loss'])
    
    # Apply other system losses (wiring, mismatch, etc.)
    dc_power = dc_power * (1 - system_params['system_losses'])
    
    # Calculate pv production in kwh (considering inverter efficiency and clipping)
    ac_capacity = system_params['capacity_kw'] / system_params['dc_ac_ratio']
    ac_power = dc_power * system_params['inverter_eff']
    
    # Inverter clipping (pv production in kwh cannot exceed inverter capacity)
    ac_power = np.minimum(ac_power, ac_capacity)
    
    # Set negative or very small values to 0
    ac_power = np.where(ac_power < 0.001, 0, ac_power)
    
    # Add pv production in kwh to dataframe
    df_result['pv_production_kwh'] = np.round(ac_power, 3)
    
    # Add additional calculated columns for analysis
    df_result['CELL_TEMP_C'] = np.round(temp_cell, 2)
    df_result['TEMP_DERATE'] = np.round(temp_derate, 4)
    df_result['POA_IRRADIANCE'] = np.round(poa_irradiance, 2)
    
    return df_result


def calculate_performance_metrics(df, system_params):
    """
    Calculate comprehensive PV system performance metrics
    """
    metrics = {}
    
    # Basic production metrics
    total_production = df['pv_production_kwh'].sum()
    num_days = df.index.normalize().nunique()
    daily_avg = total_production / num_days if num_days > 0 else float('nan')
    
    metrics['total_production_kwh'] = total_production
    metrics['daily_avg_kwh'] = daily_avg
    metrics['num_days'] = num_days
    
    # Capacity factor
    system_capacity_ac_kw = system_params['capacity_kw'] / system_params['dc_ac_ratio']
    hours_in_period = len(df)
    max_possible_production = system_capacity_ac_kw * hours_in_period
    capacity_factor = (total_production / max_possible_production) * 100
    metrics['capacity_factor_percent'] = capacity_factor
    
    # Weather impact analysis
    clear_sky_ratio = df['ALLSKY_SFC_SW_DWN'] / df['CLRSKY_SFC_SW_DWN']
    clear_sky_ratio = clear_sky_ratio.replace([np.inf, -np.inf], np.nan).dropna()
    metrics['avg_sky_clearness'] = clear_sky_ratio.mean()
    
    # System efficiency during production hours
    production_hours = df['pv_production_kwh'] > 0.01
    if production_hours.sum() > 0:
        avg_efficiency = (df.loc[production_hours, 'pv_production_kwh'] * 1000 / 
                         (df.loc[production_hours, 'POA_IRRADIANCE'] * system_params['capacity_kw'])).mean()
        metrics['avg_system_efficiency_percent'] = avg_efficiency * 100
    
    # Daily statistics
    daily_energy = df['pv_production_kwh'].resample('D').sum()
    metrics['daily_production_stats'] = {
        'mean': daily_energy.mean(),
        'max': daily_energy.max(),
        'min': daily_energy.min(),
        'std': daily_energy.std()
    }
    
    return metrics
def save_results(df, output_filename):
    """
    Save results to CSV file
    """
    df.to_csv(output_filename)
    print(f"Results saved to {output_filename}")

def calculate_daily_summary(df):
    """
    Calculate daily energy production summary
    """
    # Create a copy for daily aggregation
    daily_df = df.copy()
    
    # Ensure datetime column is datetime type
    daily_df['datetime'] = pd.to_datetime(daily_df['datetime'])
    
    # Group by date and calculate daily totals
    daily_summary = daily_df.groupby(daily_df['datetime'].dt.date).agg({
        'pv_production_kwh': 'sum',  # Daily energy in kWh
        'ALLSKY_SFC_SW_DWN': 'sum',  # Daily insolation
        'T2M': 'mean',  # Average temperature
        'WS10M': 'mean'  # Average wind speed
    }).round(2)
    
    daily_summary.columns = ['Daily_Energy_kWh', 'Daily_Insolation_Wh/m2',
                             'Avg_Temp_C', 'Avg_Wind_m/s']
    
    return daily_summary

# Main execution
# Main execution
# Main execution
if __name__ == "__main__":
    # Define PV system parameters (customize as needed)
    system_params = {
        'capacity_kw': 6.56,          # 6.56 kW system
        'dc_ac_ratio': 1.2,          
        'inverter_eff': 0.96,        
        'temp_coeff': -0.0045,       
        'system_losses': 0.14,        
        'tilt': 30,                   
        'azimuth': 180,               
        'soiling_loss': 0.02,        
        'ref_temp': 25,               
        'noct': 45                    
    }
    
    # Load data
    print("Loading NASA POWER data from processed_pv_data_2023_2024.csv...")
    df = pd.read_csv('./data/processed_pv_data_2023_2024.csv', index_col=0)
    
    # Remove any remaining Unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    # Reset index to remove the ID column
    df = df.reset_index(drop=True)
    
    # Parse datetime column properly
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Reorder columns to put datetime first
    cols = ['datetime'] + [col for col in df.columns if col != 'datetime']
    df = df[cols]
    
    print(f"\nData shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Calculate pv production in kwh
    print("\nCalculating pv production in kwh output...")
    df_with_power = calculate_pv_power(df, system_params)
    
    # Reorder columns: datetime first, pv_production_kwh last
    cols = ['datetime'] + [col for col in df_with_power.columns if col not in ['datetime', 'pv_production_kwh']] + ['pv_production_kwh']
    df_with_power = df_with_power[cols]
    
    # Display first few rows with key columns
    print("\nFirst 10 rows of data:")
    print(df_with_power.head(10))
    
    # Calculate daily summary
    print("\nDaily energy summary:")
    daily_summary = calculate_daily_summary(df_with_power)
    print(daily_summary)
    
    # Save results (without index)
    df_with_power.to_csv('./data/processed_pv_data_2023_2024.csv', index=False)
    print("Results saved to ./data/processed_pv_data_2023_2024.csv")
    
    # Print total statistics
    print("\n=== SUMMARY STATISTICS ===")
    print(f"Total AC Energy Generated: {df_with_power['pv_production_kwh'].sum():.2f} kWh")
    print(f"Peak pv production in kwh: {df_with_power['pv_production_kwh'].max():.2f} kW")
    print(f"Average Cell Temperature: {df_with_power['CELL_TEMP_C'].mean():.2f} °C")
    print(f"Capacity Factor: {(df_with_power['pv_production_kwh'].sum() / (len(df_with_power) * system_params['capacity_kw'] / system_params['dc_ac_ratio']) * 100):.2f}%")