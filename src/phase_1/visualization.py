import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# seaborn was previously imported but is unused; removed to avoid linter warnings
from .sarima_models import plot_sarima_results


def plot_data_overview(df, system_params):
    """
    Create comprehensive data overview plots
    """
    # Daily analysis
    daily_production = df['pv_production_kwh'].resample('D').sum()  # Sum of hourly kW = daily kWh

    plt.figure(figsize=(15, 10))

    # Plot 1: Daily production
    plt.subplot(2, 3, 1)
    daily_production.plot(title='Daily PV Production (kWh)', color='green', linewidth=2)
    plt.ylabel('Daily Energy (kWh)')
    plt.grid(True)

    # Plot 2: Hourly production pattern
    plt.subplot(2, 3, 2)
    df.groupby(df.index.hour)['pv_production_kwh'].mean().plot(title='Average Hourly Production', color='orange', linewidth=2, marker='o')
    plt.ylabel('Average Power (kW)')
    plt.xlabel('Hour of Day')
    plt.grid(True)

    # Plot 3: POA Irradiance vs Production
    plt.subplot(2, 3, 3)
    plt.scatter(df['POA_IRRADIANCE'], df['pv_production_kwh'], alpha=0.5, s=20)
    plt.xlabel('POA Irradiance (W/m²)')
    plt.ylabel('AC Power (kW)')
    plt.title('POA Irradiance vs AC Power')
    plt.grid(True)

    # Plot 4: Cell Temperature impact
    plt.subplot(2, 3, 4)
    production_hours = df['pv_production_kwh'] > 0.1  # Only plot when actually producing
    plt.scatter(df.loc[production_hours, 'CELL_TEMP_C'], 
               df.loc[production_hours, 'pv_production_kwh'], 
               alpha=0.6, c=df.loc[production_hours, 'POA_IRRADIANCE'], 
               cmap='plasma', s=20)
    plt.xlabel('Cell Temperature (°C)')
    plt.ylabel('AC Power (kW)')
    plt.title('Cell Temperature vs Power\n(colored by POA Irradiance)')
    plt.colorbar(label='POA Irradiance (W/m²)')
    plt.grid(True)

    # Plot 5: Temperature derating effect
    plt.subplot(2, 3, 5)
    derating_effect = df[production_hours].copy()
    derating_effect['temp_effect'] = (derating_effect['TEMP_DERATE'] - 1) * 100  # % change
    plt.scatter(derating_effect['CELL_TEMP_C'], derating_effect['temp_effect'], 
               alpha=0.6, c=derating_effect['pv_production_kwh'], cmap='coolwarm', s=20)
    plt.xlabel('Cell Temperature (°C)')
    plt.ylabel('Temperature Effect (%)')
    plt.title('Temperature Derating Effect\n(colored by AC Power)')
    plt.colorbar(label='AC Power (kW)')
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    plt.grid(True)

    # Plot 6: System performance throughout day
    plt.subplot(2, 3, 6)
    # Group by hour and calculate metrics
    hourly_metrics = df.groupby(df.index.hour).agg({
        'pv_production_kwh': 'mean',
        'POA_IRRADIANCE': 'mean', 
        'CELL_TEMP_C': 'mean'
    }).reset_index(names='hour')

    # Plot multiple y-axes
    ax1 = plt.gca()
    ax1.plot(hourly_metrics['hour'], hourly_metrics['pv_production_kwh'], 
             color='blue', linewidth=2, label='AC Power (kW)')
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('AC Power (kW)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(hourly_metrics['hour'], hourly_metrics['POA_IRRADIANCE'], 
             color='red', linewidth=2, linestyle='--', label='POA Irradiance')
    ax2.set_ylabel('POA Irradiance (W/m²)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    plt.title('Daily Profile: Power vs Irradiance')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print("=== PRODUCTION SUMMARY ===")
    print(f"Total period: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
    print(f"Total energy produced: {daily_production.sum():.1f} kWh")
    print(f"Average daily production: {daily_production.mean():.1f} kWh/day")
    print(f"Peak daily production: {daily_production.max():.1f} kWh")
    print(f"Capacity factor: {(daily_production.sum() / (len(daily_production) * system_params['capacity_kw'] * 24)) * 100:.1f}%")


def  plot_model_results_comparison(all_results):
    """
    Create comprehensive comparison plots for all models.
    The temporal comparison is now aggregated to a daily frequency for all models.
    """
    # Extract results
    ml_results = all_results['ml_results']
    sarima_results = all_results['sarima_results']
    lstm_results = all_results.get('lstm_results', {})
    
    rf_metrics = ml_results['rf_metrics']
    xgb_metrics = ml_results['xgb_metrics']
    sarima_metrics = sarima_results['sarima_metrics']
    
    # Check if LSTM is available and successful
    lstm_available = lstm_results and 'error' not in lstm_results and 'lstm_metrics' in lstm_results
    lstm_metrics = lstm_results.get('lstm_metrics', {}) if lstm_available else {}
    
    # Model names and colors
    if lstm_available:
        models = ['Random Forest', 'XGBoost', 'SARIMA', 'LSTM']
        colors = ['blue', 'green', 'red', 'purple']
    else:
        models = ['Random Forest', 'XGBoost', 'SARIMA']
        colors = ['blue', 'green', 'red']
    
    # Create main grid plot
    plt.figure(figsize=(20, 12))
    
    # ======================= ROW 1: METRICS COMPARISON =======================
    plt.subplot(3, 4, 1)
    mae_values = [rf_metrics['MAE'], xgb_metrics['MAE'], sarima_metrics['MAE']]
    if lstm_available:
        mae_values.append(lstm_metrics['MAE'])
    plt.bar(models, mae_values, color=colors, alpha=0.7)
    plt.title('Mean Absolute Error (MAE)')
    plt.ylabel('MAE')
    plt.xticks(rotation=45)
    for i, v in enumerate(mae_values):
        plt.text(i, v + max(mae_values)*0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.subplot(3, 4, 2)
    rmse_values = [rf_metrics['RMSE'], xgb_metrics['RMSE'], sarima_metrics['RMSE']]
    if lstm_available:
        rmse_values.append(lstm_metrics['RMSE'])
    plt.bar(models, rmse_values, color=colors, alpha=0.7)
    plt.title('Root Mean Square Error (RMSE)')
    plt.ylabel('RMSE')
    plt.xticks(rotation=45)
    for i, v in enumerate(rmse_values):
        plt.text(i, v + max(rmse_values)*0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.subplot(3, 4, 3)
    r2_values = [rf_metrics['R²'], xgb_metrics['R²'], 
                 sarima_metrics.get('R²', 0.747)]  # Use stored R² if available
    if lstm_available:
        r2_values.append(lstm_metrics['R²'])
    plt.bar(models, r2_values, color=colors, alpha=0.7)
    plt.title('R² Score')
    plt.ylabel('R²')
    plt.xticks(rotation=45)
    for i, v in enumerate(r2_values):
        plt.text(i, v + max(r2_values)*0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.subplot(3, 4, 4)
    if 'feature_importance' in rf_metrics and rf_metrics['feature_importance'] is not None and not rf_metrics['feature_importance'].empty:
        top_features = rf_metrics['feature_importance'].head(10)
        plt.barh(range(len(top_features)), top_features['importance'], color='blue', alpha=0.7)
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Top 10 RF Feature Importance')
        plt.gca().invert_yaxis()
    else:
        plt.axis('off')
        plt.text(0.5, 0.5, 'RF Feature Importance\nNot Available', ha='center', va='center', fontsize=12)
    
    # ======================= ROW 2: RESIDUALS ANALYSIS =======================
    # Define ML test data for residuals analysis
    ml_test_y = ml_results['results_df']['Actual']
    
    plt.subplot(3, 4, 5)
    rf_residuals = ml_test_y - ml_results['results_df']['RF_Predicted']
    plt.hist(rf_residuals, bins=50, alpha=0.7, color='blue', label=f'RF (σ={rf_residuals.std():.4f})')
    plt.title('Random Forest Residuals')
    plt.xlabel('Residuals (kW)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 4, 6)
    xgb_residuals = ml_test_y - ml_results['results_df']['XGB_Predicted']
    plt.hist(xgb_residuals, bins=50, alpha=0.7, color='green', label=f'XGB (σ={xgb_residuals.std():.4f})')
    plt.title('XGBoost Residuals')
    plt.xlabel('Residuals (kW)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 4, 7)
    # SARIMA Residuals - calculate from actual vs predicted
    if 'sarima_metrics' in sarima_results and 'predictions' in sarima_results['sarima_metrics'] and 'test_data' in sarima_results:
        # Get actual and predicted values from SARIMA
        sarima_actual = sarima_results['test_data']['production'].values
        sarima_predicted = sarima_results['sarima_metrics']['predictions']
        sarima_residuals = sarima_actual - sarima_predicted
        
        plt.hist(sarima_residuals, bins=30, alpha=0.7, color='red', 
                label=f'SARIMA (σ={sarima_residuals.std():.4f})')
        plt.title('SARIMA Residuals')
        plt.xlabel('Residuals (kWh/day)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.axis('off')
        plt.text(0.5, 0.5, 'SARIMA Residuals\nNot Available', ha='center', va='center', fontsize=12)
    
    plt.subplot(3, 4, 8)
    # LSTM Residuals if available
    if lstm_available and 'results_df' in lstm_results:
        lstm_actual = lstm_results['results_df']['Actual']
        lstm_pred = lstm_results['results_df']['LSTM_Predicted']
        lstm_residuals = lstm_actual - lstm_pred
        plt.hist(lstm_residuals, bins=50, alpha=0.7, color='purple', 
                label=f'LSTM (σ={lstm_residuals.std():.4f})')
        plt.title('LSTM Residuals')
        plt.xlabel('Residuals (kW)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.axis('off')
        plt.text(0.5, 0.5, 'LSTM Residuals\nNot Available', ha='center', va='center', fontsize=12)
    
    # ======================= ROW 3: SCATTER PLOTS =======================
    plt.subplot(3, 4, 9)
    plt.scatter(ml_test_y, ml_results['results_df']['RF_Predicted'], alpha=0.6, s=20, color='blue')
    max_val = max(ml_test_y.max(), ml_results['results_df']['RF_Predicted'].max())
    plt.plot([0, max_val], [0, max_val], 'r--', linewidth=2)
    plt.xlabel('Actual AC Power (kW)')
    plt.ylabel('Predicted AC Power (kW)')
    plt.title(f'Random Forest\nR² = {rf_metrics["R²"]:.3f}')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 4, 10)
    plt.scatter(ml_test_y, ml_results['results_df']['XGB_Predicted'], alpha=0.6, s=20, color='green')
    max_val = max(ml_test_y.max(), ml_results['results_df']['XGB_Predicted'].max())
    plt.plot([0, max_val], [0, max_val], 'r--', linewidth=2)
    plt.xlabel('Actual AC Power (kW)')
    plt.ylabel('Predicted AC Power (kW)')
    plt.title(f'XGBoost\nR² = {xgb_metrics["R²"]:.3f}')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 4, 11)
    # SARIMA Scatter Plot
    if 'predictions' in sarima_metrics and 'test_data' in sarima_results:
        sarima_actual = sarima_results['test_data']['production']
        sarima_pred = sarima_metrics['predictions']
        plt.scatter(sarima_actual, sarima_pred, alpha=0.6, s=30, color='red')
        max_val = max(sarima_actual.max(), sarima_pred.max())
        plt.plot([0, max_val], [0, max_val], 'r--', linewidth=2)
        plt.xlabel('Actual Daily Energy (kWh)')
        plt.ylabel('Predicted Daily Energy (kWh)')
        plt.title(f'SARIMA\nR² = {r2_values[2]:.3f}')
        plt.grid(True, alpha=0.3)
    else:
        plt.axis('off')
        plt.text(0.5, 0.5, 'SARIMA Scatter\nNot Available', ha='center', va='center', fontsize=12)
    
    plt.subplot(3, 4, 12)
    # LSTM Scatter Plot if available
    if lstm_available and 'results_df' in lstm_results:
        lstm_actual = lstm_results['results_df']['Actual']
        lstm_pred = lstm_results['results_df']['LSTM_Predicted']
        plt.scatter(lstm_actual, lstm_pred, alpha=0.6, s=20, color='purple')
        max_val = max(lstm_actual.max(), lstm_pred.max())
        plt.plot([0, max_val], [0, max_val], 'r--', linewidth=2)
        plt.xlabel('Actual AC Power (kW)')
        plt.ylabel('Predicted AC Power (kW)')
        plt.title(f'LSTM\nR² = {lstm_metrics["R²"]:.3f}')
        plt.grid(True, alpha=0.3)
    else:
        plt.axis('off')
        plt.text(0.5, 0.5, 'LSTM Scatter\nNot Available', ha='center', va='center', fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    # ======================= NEW: DAILY AGGREGATED COMPARISON PLOT =======================
    print("\n" + "="*80)
    print("📊 Creating DAILY Aggregated Comparison (All Models)")
    print("="*80)
    
        # ============================================================================
    # STEP 1: Get SARIMA daily predictions (already daily)
    # ============================================================================
    sarima_test_data = sarima_results.get('test_data')
    if sarima_test_data is None:
        print("⚠️  SARIMA test data not available for daily comparison plot.")
        sarima_daily_actual = pd.Series() # Empty series
    else:
        sarima_daily_actual = sarima_test_data['production']
        sarima_daily_pred = sarima_metrics['predictions']
        print(f"✅ SARIMA (Daily): {len(sarima_daily_actual)} days")
        print(f"   Index range: {sarima_daily_actual.index[0]} to {sarima_daily_actual.index[-1]}")

    # ============================================================================
    # STEP 2: Aggregate ML hourly predictions to DAILY
    # ============================================================================
    if 'daily_ml' in ml_results:
        ml_daily = ml_results['daily_ml'][['Actual_kWh', 'RF_kWh', 'XGB_kWh']].copy()
        ml_daily.columns = ['Actual', 'RF_Predicted', 'XGB_Predicted']  # Rename for consistency
        print(f"✅ ML Models (Aggregated to Daily): {len(ml_daily)} days")
        print(f"   Index range: {ml_daily.index[0]} to {ml_daily.index[-1]}")
    else:
        print("⚠️ daily_ml not found in ml_results, falling back to results_df")
        ml_results_df = ml_results['results_df']
        ml_results_daily = ml_results_df.copy()
        ml_results_daily['date'] = ml_results_daily.index.date
        ml_daily = ml_results_daily.groupby('date').agg({
            'Actual': 'sum',
            'RF_Predicted': 'sum',
            'XGB_Predicted': 'sum'
        })
        ml_daily.index = pd.to_datetime(ml_daily.index)
        print(f"✅ ML Models (Aggregated to Daily): {len(ml_daily)} days")
        print(f"   Index range: {ml_daily.index[0]} to {ml_daily.index[-1]}")
    ml_daily.index = pd.to_datetime(ml_daily.index)
    print(f"✅ ML Models (Aggregated to Daily): {len(ml_daily)} days")
    print(f"   Index range: {ml_daily.index[0]} to {ml_daily.index[-1]}")

    # ============================================================================
    # STEP 3: Aggregate LSTM hourly predictions to DAILY (if available)
    # ============================================================================
    lstm_daily_pred = None
    if lstm_available and 'results_df' in lstm_results:
        lstm_results_df = lstm_results['results_df']
        
        # Check if index has datetime information
        if hasattr(lstm_results_df.index, 'date') or isinstance(lstm_results_df.index, pd.DatetimeIndex):
            # Standard case with datetime index
            lstm_daily_df = lstm_results_df.copy()
            if isinstance(lstm_daily_df.index, pd.DatetimeIndex):
                # Direct datetime index
                lstm_daily_df['date'] = lstm_daily_df.index.date
            else:
                # Convert to datetime if needed
                lstm_daily_df.index = pd.to_datetime(lstm_daily_df.index)
                lstm_daily_df['date'] = lstm_daily_df.index.date
            
            lstm_daily = lstm_daily_df.groupby('date').agg({
                'Actual': 'sum',
                'LSTM_Predicted': 'sum'
            })
            lstm_daily.index = pd.to_datetime(lstm_daily.index)
            lstm_daily_pred = lstm_daily['LSTM_Predicted']
            print(f"✅ LSTM (Aggregated to Daily): {len(lstm_daily)} days")
            print(f"   Index range: {lstm_daily.index[0]} to {lstm_daily.index[-1]}")
        else:
            # LSTM results don't have proper datetime index, skip daily aggregation
            print("⚠️  LSTM results don't have datetime index, using direct comparison without daily aggregation")
            # We'll handle LSTM separately in the comparison metrics only

    # ============================================================================
    # STEP 4: Find common date range across all models
    # ============================================================================
    common_dates = sarima_daily_actual.index
    ml_daily_aligned = ml_daily.reindex(common_dates)
    
    lstm_daily_aligned = None
    if lstm_daily_pred is not None:
        lstm_daily_aligned = lstm_daily_pred.reindex(common_dates)
    
    print(f"\n✅ Common date range: {len(common_dates)} days")
    if not common_dates.empty:
        print(f"   From {common_dates[0]} to {common_dates[-1]}")

    # ============================================================================
    # STEP 5: Create the aligned DAILY comparison plot
    # ============================================================================
    plt.figure(figsize=(20, 8))
    
    sample_size = len(common_dates)
    time_range = range(sample_size)
    
    plt.plot(time_range, sarima_daily_actual.values, 
             color='gray', linewidth=3, label='📈 Actual', alpha=0.9, zorder=3,
             marker='o', markersize=4)
    
    if ml_daily_aligned['RF_Predicted'].notna().any():
        plt.plot(time_range, ml_daily_aligned['RF_Predicted'].values,
                 linestyle='--', color='#1f77b4', linewidth=2.5, alpha=0.95, 
                 label='🌲 Random Forest', zorder=8, marker='s', markersize=3)
    
    if ml_daily_aligned['XGB_Predicted'].notna().any():
        plt.plot(time_range, ml_daily_aligned['XGB_Predicted'].values,
                 linestyle='-.', color='#2ca02c', linewidth=2.5, alpha=0.95, 
                 label='⚡ XGBoost', zorder=7, marker='^', markersize=3)
    
    plt.plot(time_range, sarima_daily_pred.values,
             linestyle='-', color='#d62728', linewidth=2.5, alpha=0.95,
             label='📊 SARIMA', zorder=6, marker='D', markersize=3)
    
    if lstm_daily_aligned is not None and lstm_daily_aligned.notna().any():
        plt.plot(time_range, lstm_daily_aligned.values,
                 linestyle=':', color='#9467bd', linewidth=3, alpha=0.95, 
                 label='🧠 LSTM', zorder=9, marker='*', markersize=5)
    
    plt.title('ULTIMATE Daily Prediction Comparison - ALL MODELS ALIGNED', 
              fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Days (in test period)', fontsize=14, fontweight='bold')
    plt.ylabel('Daily Energy Production (kWh)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=13, loc='lower right', frameon=True, fancybox=True, 
               shadow=True, framealpha=0.95, edgecolor='black')
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    for week in range(0, sample_size, 7):
        plt.axvline(x=week, color='gray', linestyle='--', alpha=0.6, linewidth=1.5)
        if week > 0:
            plt.text(week+0.5, plt.ylim()[1]*0.92, f'Week {week//7+1}', 
                     fontsize=11, fontweight='bold', alpha=0.8,
                     bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    plt.show()

    rf_daily_mae = np.mean(np.abs(sarima_daily_actual.values - ml_daily_aligned['RF_Predicted'].values))
    xgb_daily_mae = np.mean(np.abs(sarima_daily_actual.values - ml_daily_aligned['XGB_Predicted'].values))
    sarima_daily_mae = sarima_metrics['MAE']

    lstm_daily_mae = None
    if lstm_daily_aligned is not None:
        lstm_daily_mae = np.mean(np.abs(sarima_daily_actual.values - lstm_daily_aligned.values))
    
    daily_metrics = {
        'rf_daily_mae': rf_daily_mae,
        'xgb_daily_mae': xgb_daily_mae,
        'sarima_daily_mae': sarima_daily_mae,
        'lstm_daily_mae': lstm_daily_mae
    }
    
    # ======================= SUMMARY TABLE =======================
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("="*80)
    
    comparison_data = {
        'Model': ['Random Forest', 'XGBoost', 'SARIMA'],
        'MAE': [rf_metrics['MAE'], xgb_metrics['MAE'], sarima_metrics['MAE']],
        'RMSE': [rf_metrics['RMSE'], xgb_metrics['RMSE'], sarima_metrics['RMSE']],
        'R²': [rf_metrics['R²'], xgb_metrics['R²'], r2_values[2]]
    }
    
    if lstm_available:
        comparison_data['Model'].append('LSTM')
        comparison_data['MAE'].append(lstm_metrics['MAE'])
        comparison_data['RMSE'].append(lstm_metrics['RMSE'])
        comparison_data['R²'].append(lstm_metrics['R²'])
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.round(4)
    print(comparison_df.to_string(index=False))
    
    if not lstm_available:
        if 'lstm_results' in all_results and 'error' in all_results['lstm_results']:
            print(f"\n⚠️  LSTM not included: {all_results['lstm_results']['error']}")
        else:
            print("\n⚠️  LSTM results not available")
    
    return comparison_df, daily_metrics


def plot_diagnostics(df):
    """
    Plot data quality and diagnostic information
    """
    plt.figure(figsize=(15, 8))
    
    # Check data coverage and gaps
    plt.subplot(2, 3, 1)
    hours_per_day = df.groupby(df.index.date).size()
    plt.hist(hours_per_day, bins=25, edgecolor='black', alpha=0.7)
    plt.xlabel('Hours per Day')
    plt.ylabel('Frequency')
    plt.title('Data Coverage per Day')
    plt.axvline(x=24, color='red', linestyle='--', label='Complete Day')
    plt.legend()
    
    # Missing data heatmap
    plt.subplot(2, 3, 2)
    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        missing_data.plot(kind='bar')
        plt.title('Missing Data by Column')
        plt.ylabel('Number of Missing Values')
        plt.xticks(rotation=45)
    else:
        plt.text(0.5, 0.5, 'No Missing Data', ha='center', va='center', transform=plt.gca().transAxes, 
                fontsize=14, bbox=dict(boxstyle='round', facecolor='lightgreen'))
        plt.title('Missing Data Status')
    
    # Irradiance distribution
    plt.subplot(2, 3, 3)
    plt.hist(df['ALLSKY_SFC_SW_DWN'], bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Global Horizontal Irradiance (Wh/m²)')
    plt.ylabel('Frequency')
    plt.title('Irradiance Distribution')
    plt.axvline(x=0, color='red', linestyle='--', label='Night Hours')
    plt.legend()
    
    # Production vs irradiance relationship
    plt.subplot(2, 3, 4)
    plt.scatter(df['ALLSKY_SFC_SW_DWN'], df['pv_production_kwh'], alpha=0.5, s=10)
    plt.xlabel('Global Horizontal Irradiance (Wh/m²)')
    plt.ylabel('AC Power (kW)')
    plt.title('Production vs Irradiance')
    plt.grid(True)
    
    # Temperature effects
    plt.subplot(2, 3, 5)
    plt.scatter(df['T2M'], df['pv_production_kwh'], alpha=0.5, s=10, c=df['ALLSKY_SFC_SW_DWN'], cmap='viridis')
    plt.xlabel('Ambient Temperature (°C)')
    plt.ylabel('AC Power (kW)')
    plt.title('Temperature vs Production\n(colored by irradiance)')
    plt.colorbar(label='Irradiance (Wh/m²)')
    plt.grid(True)
    
    # Daily production pattern
    plt.subplot(2, 3, 6)
    daily_production = df['pv_production_kwh'].resample('D').sum()
    plt.plot(daily_production, linewidth=1, alpha=0.8)
    plt.xlabel('Date')
    plt.ylabel('Daily Energy (kWh)')
    plt.title('Daily Production Time Series')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


def print_performance_summary(df, system_params, all_results):
    """
    Print comprehensive performance summary
    """
    print("\n" + "="*100)
    print("SOLAR PV ANALYSIS - COMPREHENSIVE PERFORMANCE SUMMARY")
    print("="*100)
    
    # System Information
    print("\n📊 SYSTEM CONFIGURATION:")
    print(f"   • Capacity: {system_params['capacity_kw']} kW DC")
    print(f"   • DC/AC Ratio: {system_params['dc_ac_ratio']}")
    print(f"   • Inverter Efficiency: {system_params['inverter_eff']*100:.1f}%")
    print(f"   • System Losses: {system_params['system_losses']*100:.1f}%")
    
    # Data Period
    print("\n📅 ANALYSIS PERIOD:")
    print(f"   • Start: {df.index.min().strftime('%Y-%m-%d %H:%M')}")
    print(f"   • End: {df.index.max().strftime('%Y-%m-%d %H:%M')}")
    print(f"   • Total Hours: {len(df)}")
    print(f"   • Days: {df.index.normalize().nunique()}")
    
    # Production Metrics
    total_production = df['pv_production_kwh'].sum()
    daily_avg = df['pv_production_kwh'].resample('D').sum().mean()
    
    print("\n⚡ PRODUCTION METRICS:")
    print(f"   • Total Energy: {total_production:.1f} kWh")
    print(f"   • Daily Average: {daily_avg:.2f} kWh/day")
    print(f"   • Peak AC Power: {df['pv_production_kwh'].max():.3f} kW")
    
    # Model Performance
    ml_results = all_results['ml_results']
    sarima_results = all_results['sarima_results']
    lstm_results = all_results.get('lstm_results', {})
    
    print("\n🤖 MODEL PERFORMANCE COMPARISON:")
    print(f"{'Model':<15} {'MAE':<8} {'RMSE':<8} {'R²':<8} {'Status'}")
    print("-" * 60)
    
    rf_metrics = ml_results['rf_metrics']
    print(f"{'Random Forest':<15} {rf_metrics['MAE']:<8.4f} {rf_metrics['RMSE']:<8.4f} {rf_metrics['R²']:<8.4f} ✅")
    
    xgb_metrics = ml_results['xgb_metrics']
    print(f"{'XGBoost':<15} {xgb_metrics['MAE']:<8.4f} {xgb_metrics['RMSE']:<8.4f} {xgb_metrics['R²']:<8.4f} ✅")
    
    sarima_metrics = sarima_results['sarima_metrics']
    print(f"{'SARIMA':<15} {sarima_metrics['MAE']:<8.4f} {sarima_metrics['RMSE']:<8.4f} {'N/A':<8} ✅")
    
    # LSTM (if available)
    if lstm_results and 'error' not in lstm_results:
        lstm_metrics = lstm_results['lstm_metrics']
        print(f"{'LSTM':<15} {lstm_metrics['MAE']:<8.4f} {lstm_metrics['RMSE']:<8.4f} {lstm_metrics['R²']:<8.4f} ✅")
        mae_values = [rf_metrics['MAE'], xgb_metrics['MAE'], sarima_metrics['MAE'], lstm_metrics['MAE']]
        model_names = ['Random Forest', 'XGBoost', 'SARIMA', 'LSTM']
    else:
        print(f"{'LSTM':<15} {'N/A':<8} {'N/A':<8} {'N/A':<8} ❌")
        mae_values = [rf_metrics['MAE'], xgb_metrics['MAE'], sarima_metrics['MAE']]
        model_names = ['Random Forest', 'XGBoost', 'SARIMA']
    
    # Best Model
    best_mae_idx = mae_values.index(min(mae_values))
    best_mae_model = model_names[best_mae_idx]
    
    print(f"\n🏆 BEST MODEL (by MAE): {best_mae_model}")
    
    print("\n" + "="*100)


def plot_separate_time_series_comparisons(ml_results, sarima_results):
    """
    Create separate time-series visualizations.

    - Combined hourly comparisons for ML models only (3-day and 1-week)
    - Then call per-model visualizers for RF and XGB
    - Finally call SARIMA's daily visualizer (uses daily aggregated data)
    This avoids mixing frequencies (hourly vs daily) and prevents blank plots.
    """
    results_df = ml_results.get('results_df')

    # 1) Combined hourly comparisons for ML models only (clear views)
    if results_df is not None and not results_df.empty:
        # 3-Day (hourly)
        plt.figure(figsize=(16, 5))
        three_day_size = min(72, len(results_df))
        time_range = range(three_day_size)

        plt.plot(time_range, results_df['Actual'][:three_day_size], 'black', linewidth=3, label='Actual', alpha=0.9)
        if 'RF_Predicted' in results_df.columns:
            plt.plot(time_range, results_df['RF_Predicted'][:three_day_size], '--', color='blue', linewidth=2.5, label='Random Forest', alpha=0.8)
        if 'XGB_Predicted' in results_df.columns:
            plt.plot(time_range, results_df['XGB_Predicted'][:three_day_size], '--', color='green', linewidth=2.5, label='XGBoost', alpha=0.8)

        plt.title('3-Day Model Prediction Comparison (ML models only)', fontsize=16, fontweight='bold')
        plt.xlabel('Time (hours)', fontsize=12)
        plt.ylabel('AC Power (kW)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # 1-Week (hourly)
        plt.figure(figsize=(18, 6))
        one_week_size = min(168, len(results_df))
        time_range = range(one_week_size)

        plt.plot(time_range, results_df['Actual'][:one_week_size], 'black', linewidth=2.5, label='Actual', alpha=0.9)
        if 'RF_Predicted' in results_df.columns:
            plt.plot(time_range, results_df['RF_Predicted'][:one_week_size], '--', color='blue', linewidth=2, label='Random Forest', alpha=0.8)
        if 'XGB_Predicted' in results_df.columns:
            plt.plot(time_range, results_df['XGB_Predicted'][:one_week_size], '--', color='green', linewidth=2, label='XGBoost', alpha=0.8)

        plt.title('1-Week Model Prediction Comparison (ML models only)', fontsize=16, fontweight='bold')
        plt.xlabel('Time (hours)', fontsize=12)
        plt.ylabel('AC Power (kW)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    else:
        print('No ML results available for hourly comparison.')

    # 2) Per-model detailed visualizations for ML models
    try:
        from visualization import plot_individual_ml_predictions
        # RF
        if 'rf_metrics' in ml_results:
            plot_individual_ml_predictions(ml_results, 'Random Forest')
        # XGB
        if 'xgb_metrics' in ml_results:
            plot_individual_ml_predictions(ml_results, 'XGBoost')
    except Exception as e:
        # If something goes wrong (import cycle), fallback to inline plotting
        print(f'Could not call per-model ML visualizers: {e}')

    # 3) SARIMA: call SARIMA's own daily visualizer (it uses daily aggregated data)
    try:
        # sarima_results should contain train_data/test_data and sarima_metrics
        if sarima_results and 'sarima_metrics' in sarima_results and 'train_data' in sarima_results and 'test_data' in sarima_results:
            plot_sarima_results(sarima_results['train_data'], sarima_results['test_data'],
                                sarima_results['sarima_metrics'],
                                sarima_results.get('best_order', (0, 0, 0)),
                                sarima_results.get('best_seasonal', (0, 0, 0, 0)))
        else:
            print('SARIMA results incomplete - cannot show SARIMA visualizer.')
    except Exception as e:
        print(f'Could not generate SARIMA visualization: {e}')

def plot_individual_ml_predictions(ml_results, model_name='Random Forest'):
    """
    Create separate, clear prediction plots for ML models with better readability
    """
    results_df = ml_results['results_df']
    
    if model_name == 'Random Forest':
        predicted_col = 'RF_Predicted'
        metrics = ml_results['rf_metrics']
        color = 'blue'
    else:  # XGBoost
        predicted_col = 'XGB_Predicted'
        metrics = ml_results['xgb_metrics']
        color = 'green'
    
    residuals = results_df['Actual'] - results_df[predicted_col]
    
    # Plot 1: Separate Time Series Comparison (3-day and 1-week)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(f'{model_name} - Time Series Prediction Analysis', fontsize=16, fontweight='bold')
    
    # 3-day comparison
    three_day_size = min(72, len(results_df))  # 3 days
    time_range_3d = range(three_day_size)
    
    ax1.plot(time_range_3d, results_df['Actual'][:three_day_size], 'black', linewidth=2.5, label='Actual', alpha=0.9)
    ax1.plot(time_range_3d, results_df[predicted_col][:three_day_size], color=color, linewidth=2, label='Predicted', alpha=0.8, linestyle='--')
    ax1.set_title('3-Day Prediction Comparison', fontsize=14)
    ax1.set_xlabel('Time (hours)', fontsize=12)
    ax1.set_ylabel('AC Power (kW)', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 1-week comparison  
    one_week_size = min(168, len(results_df))  # 1 week
    time_range_1w = range(one_week_size)
    
    ax2.plot(time_range_1w, results_df['Actual'][:one_week_size], 'black', linewidth=2, label='Actual', alpha=0.8)
    ax2.plot(time_range_1w, results_df[predicted_col][:one_week_size], color=color, linewidth=1.5, label='Predicted', alpha=0.8, linestyle='--')
    ax2.set_title('1-Week Prediction Comparison', fontsize=14)
    ax2.set_xlabel('Time (hours)', fontsize=12)
    ax2.set_ylabel('AC Power (kW)', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Plot 2: Detailed Analysis (2x2 layout but with better spacing)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{model_name} - Detailed Prediction Analysis\nR² = {metrics["R²"]:.4f}', fontsize=16, fontweight='bold')
    
    # Actual vs Predicted scatter
    ax1 = axes[0, 0]
    ax1.scatter(results_df['Actual'], results_df[predicted_col], alpha=0.6, s=25, color=color, edgecolors='black', linewidth=0.5)
    max_val = max(results_df['Actual'].max(), results_df[predicted_col].max())
    ax1.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    ax1.set_xlabel('Actual AC Power (kW)', fontsize=12)
    ax1.set_ylabel('Predicted AC Power (kW)', fontsize=12)
    ax1.set_title(f'Actual vs Predicted\nR² = {metrics["R²"]:.4f}', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Residuals over time
    ax2 = axes[0, 1]
    sample_size = min(168, len(residuals))
    colors_res = np.where(np.abs(residuals) > residuals.std(), 'red', 
                         np.where(np.abs(residuals) > 0.5*residuals.std(), 'orange', 'green'))
    ax2.scatter(range(sample_size), residuals[:sample_size], c=colors_res[:sample_size], 
               alpha=0.7, s=25, edgecolors='black', linewidth=0.3)
    ax2.axhline(0, color='black', linestyle='-', linewidth=2, label='Zero Error')
    ax2.axhline(residuals.mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean: {residuals.mean():.4f}')
    ax2.set_xlabel('Time (hours)', fontsize=12)
    ax2.set_ylabel('Residuals (kW)', fontsize=12)
    ax2.set_title('Prediction Errors Over Time', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Residuals distribution  
    ax3 = axes[1, 0]
    ax3.hist(residuals, bins=50, color=color, alpha=0.7, edgecolor='black')
    ax3.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax3.axvline(residuals.mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean: {residuals.mean():.4f}')
    ax3.set_xlabel('Residuals (kW)', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title(f'Error Distribution\nStd Dev: {residuals.std():.4f}', fontsize=14)
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    # Performance metrics summary
    ax4 = axes[1, 1]
    metrics_names = ['MAE', 'RMSE', 'R²', 'MAPE']
    metrics_values = [metrics.get(name, 0) for name in metrics_names]
    colors_bars = [color if val > 0 else 'red' for val in metrics_values]
    
    bars = ax4.bar(metrics_names, metrics_values, color=colors_bars, alpha=0.7, edgecolor='black')
    ax4.set_title('Performance Metrics Summary', fontsize=14)
    ax4.set_ylabel('Metric Value', fontsize=12)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, metrics_values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + max(metrics_values)*0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed metrics
    print(f"\n📊 {model_name} - Detailed Performance Metrics:")
    print("="*50)
    print(f"MAE:           {metrics['MAE']:.6f} kW")
    print(f"RMSE:          {metrics['RMSE']:.6f} kW")
    print(f"R²:            {metrics['R²']:.6f}")
    print(f"Mean Residual: {residuals.mean():.6f} kW")
    print(f"Std Residual:  {residuals.std():.6f} kW")
    print(f"Max Error:     {np.abs(residuals).max():.6f} kW")
    print("="*50)


def plot_individual_sarima_predictions(sarima_results):
    """
    Create individual prediction plots for SARIMA model
    """
    # This function is now handled by the plot_sarima_results function in sarima_models.py
    # But we can add a summary here
    sarima_metrics = sarima_results['sarima_metrics']
    
    print("\n📊 SARIMA - Performance Summary:")
    print("="*50)
    print(f"MAE:               {sarima_metrics['MAE']:.6f} kWh")
    print(f"RMSE:              {sarima_metrics['RMSE']:.6f} kWh") 
    print(f"R²:                {sarima_metrics.get('R²', 'N/A')}")
    print(f"Overfitting RMSE:  {sarima_metrics.get('Overfitting_RMSE', 'N/A'):.2f}x")
    print(f"Overfitting MAE:   {sarima_metrics.get('Overfitting_MAE', 'N/A'):.2f}x")
    print(f"Improvement:       {sarima_metrics['MAE_Improvement_%']:.1f}% over seasonal baseline")
    print("="*50)
    
    # Performance assessment
    if sarima_metrics.get('Overfitting_RMSE', float('inf')) < 1.5:
        print("✅ SARIMA shows good generalization (overfitting < 1.5x)")
    elif sarima_metrics.get('Overfitting_RMSE', float('inf')) < 2.0:
        print("⚠️  SARIMA shows moderate overfitting (1.5x < overfitting < 2x)")
    else:
        print("❌ SARIMA shows significant overfitting (overfitting > 2x)")
    
    if sarima_metrics['MAE_Improvement_%'] > 5:
        print("✅ SARIMA shows significant improvement over seasonal baseline")
    elif sarima_metrics['MAE_Improvement_%'] > 0:
        print("⚠️  SARIMA shows modest improvement over seasonal baseline")  
    else:
        print("❌ SARIMA performs worse than seasonal baseline")
