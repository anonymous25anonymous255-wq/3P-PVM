import pandas as pd
import numpy as np


def load_nasa_power_data(filename):
    """
    Load NASA POWER data from CSV file
    """
    # Read the CSV file
    df = pd.read_csv(filename)
    
    # Create datetime column
    df['DATETIME'] = pd.to_datetime(df[['YEAR', 'MO', 'DY', 'HR']].rename(columns={
        'YEAR': 'year', 'MO': 'month', 'DY': 'day', 'HR': 'hour'
    }))
    
    # Set datetime as index with explicit hourly frequency
    df.set_index('DATETIME', inplace=True)
    df.index.freq = 'H'  # Set explicit hourly frequency to avoid warnings
    
    return df


def prepare_data_for_analysis(df):
    """
    Convert to datetime and clean data
    """
    # Convert to datetime if not already done
    if 'datetime' not in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df['datetime'] = pd.to_datetime(dict(year=df['YEAR'], month=df['MO'], day=df['DY'], hour=df['HR']))
        df = df.set_index('datetime')
        df = df.drop(['YEAR', 'MO', 'DY', 'HR'], axis=1)
    
    # Ensure frequency is set to avoid statsmodels warnings
    if hasattr(df.index, 'freq') and df.index.freq is None:
        df.index.freq = 'H'
    
    # Convert MJ/hr to Watts (1 MJ/hr = 277.778 W)
    irradiance_columns = ['ALLSKY_SFC_SW_DWN', 'CLRSKY_SFC_SW_DWN', 
                         'ALLSKY_SFC_SW_DIFF', 'ALLSKY_SFC_SW_DNI']

    for col in irradiance_columns:
        if col in df.columns and f'{col}_W' not in df.columns:
            df[f'{col}_W'] = df[col] * 277.778  # Convert to Watts/m²
    
    return df


def calculate_daily_summary(df):
    """
    Calculate daily energy production summary
    """
    # Create a copy for daily aggregation
    daily_df = df.copy()
    daily_df['DATE'] = daily_df.index.date
    
    # Group by date and calculate daily totals
    daily_summary = daily_df.groupby('DATE').agg({
        'AC_POWER_KW': 'sum',  # Daily energy in kWh (since hourly data)
        'ALLSKY_SFC_SW_DWN': 'sum',  # Daily insolation
        'T2M': 'mean',  # Average temperature
        'WS10M': 'mean'  # Average wind speed
    }).round(2)
    
    daily_summary.columns = ['Daily_Energy_kWh', 'Daily_Insolation_Wh/m2', 
                             'Avg_Temp_C', 'Avg_Wind_m/s']
    
    return daily_summary


def prepare_features_for_ml(df):
    """
    Prepare features for ML models using the advanced PV calculations
    """
    df_ml = df.copy()
    
    # Ensure we have a datetime index for temporal features
    if not isinstance(df_ml.index, pd.DatetimeIndex):
        print("Warning: Index is not datetime. Creating temporal features from available columns.")
        if all(col in df_ml.columns for col in ['YEAR', 'MO', 'DY', 'HR']):
            df_ml['datetime'] = pd.to_datetime({
                'year': df_ml['YEAR'],
                'month': df_ml['MO'], 
                'day': df_ml['DY'],
                'hour': df_ml['HR']
            })
            df_ml = df_ml.set_index('datetime')
        else:
            df_ml.index = pd.date_range(start='2023-01-01', periods=len(df_ml), freq='H')
    
    # Ensure frequency is explicitly set
    if hasattr(df_ml.index, 'freq') and df_ml.index.freq is None:
        df_ml.index.freq = 'H'
    
    # Basic temporal features
    df_ml['hour'] = df_ml.index.hour
    df_ml['day_of_year'] = df_ml.index.dayofyear
    df_ml['month'] = df_ml.index.month
    df_ml['day_of_week'] = df_ml.index.dayofweek
    df_ml['is_weekend'] = (df_ml.index.dayofweek >= 5).astype(int)
    
    # Solar position features
    df_ml['hour_sin'] = np.sin(2 * np.pi * df_ml['hour'] / 24)
    df_ml['hour_cos'] = np.cos(2 * np.pi * df_ml['hour'] / 24)
    df_ml['doy_sin'] = np.sin(2 * np.pi * df_ml['day_of_year'] / 365)
    df_ml['doy_cos'] = np.cos(2 * np.pi * df_ml['day_of_year'] / 365)
    
    # Weather and system features
    df_ml['irradiance_wm2'] = df_ml['ALLSKY_SFC_SW_DWN'] * 277.778
    df_ml['clearness_ratio'] = (df_ml['ALLSKY_SFC_SW_DWN'] / 
                               df_ml['CLRSKY_SFC_SW_DWN']).replace([np.inf, -np.inf], 0).clip(0, 1)
    
    # Lag features
    for lag in [1, 2, 3, 24]:  # 1h, 2h, 3h, and 24h lags
        df_ml[f'ac_power_lag_{lag}'] = df_ml['AC_POWER_KW'].shift(lag)
        df_ml[f'poa_irradiance_lag_{lag}'] = df_ml['POA_IRRADIANCE'].shift(lag)
        df_ml[f'temp_lag_{lag}'] = df_ml['T2M'].shift(lag)
    
    # Rolling features
    df_ml['ac_power_rolling_3h_mean'] = df_ml['AC_POWER_KW'].rolling(3).mean()
    df_ml['ac_power_rolling_6h_std'] = df_ml['AC_POWER_KW'].rolling(6).std()
    df_ml['irradiance_rolling_3h_mean'] = df_ml['irradiance_wm2'].rolling(3).mean()
    
    return df_ml


def get_feature_columns():
    """
    Define the feature columns for the models
    """
    return [
        # Temporal features
        'hour', 'month', 'day_of_week', 'is_weekend',
        'hour_sin', 'hour_cos', 'doy_sin', 'doy_cos',
        
        # Advanced PV system metrics
        'POA_IRRADIANCE', 'CELL_TEMP_C', 'TEMP_DERATE',
        
        # Weather features
        'irradiance_wm2', 'clearness_ratio', 'T2M', 'WS10M', 'RH2M',
        
        # Lag features
        'ac_power_lag_1', 'ac_power_lag_2', 'ac_power_lag_3', 'ac_power_lag_24',
        'poa_irradiance_lag_1', 'temp_lag_1',
        
        # Rolling features
        'ac_power_rolling_3h_mean', 'ac_power_rolling_6h_std', 'irradiance_rolling_3h_mean'
    ]


def get_common_train_test_split(df, test_start_date='2023-05-25'):
    """
    *** SINGLE SOURCE OF TRUTH FOR TRAIN/TEST SPLIT ***
    
    Define a common train/test split that all models should use for fair comparison.
    This function ensures DRY principle and consistency across all models:
    - ML models (Random Forest, XGBoost)  
    - SARIMA models
    - LSTM models
    
    Args:
        df: DataFrame with datetime index
        test_start_date: Start date for test period (everything after this is test)
    
    Returns:
        dict: Contains train_mask, test_mask, train_end_date, test_start_date
    """
    test_start = pd.to_datetime(test_start_date)
    
    # Create masks for train and test periods
    train_mask = df.index < test_start
    test_mask = df.index >= test_start
    
    train_end_date = df.index[train_mask][-1] if train_mask.any() else None
    actual_test_start = df.index[test_mask][0] if test_mask.any() else None
    
    print("🎯 COMMON TRAIN/TEST SPLIT DEFINED:")
    print(f"   Training period: {df.index[0]} to {train_end_date}")
    print(f"   Test period: {actual_test_start} to {df.index[-1]}")
    print(f"   Train samples: {train_mask.sum():,}")
    print(f"   Test samples: {test_mask.sum():,}")
    print(f"   Test ratio: {test_mask.sum() / len(df) * 100:.1f}%")
    
    return {
        'train_mask': train_mask,
        'test_mask': test_mask,
        'train_end_date': train_end_date,
        'test_start_date': actual_test_start,
        'test_start_param': test_start_date
    }


def save_results(df, output_filename):
    """
    Save results to CSV file
    """
    df.to_csv(output_filename)
    print(f"Results saved to {output_filename}")