from src.phase_1.data_processing import get_common_train_test_split

import pandas as pd
import numpy as np
import itertools
from tqdm import tqdm
import warnings
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LinearRegression, RidgeCV
import sys
import os

proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)


def create_regression_baseline(df):
    """
    Create linear regression baseline using weather variables
    Works with daily aggregated data for faster SARIMA processing
    Uses get_common_train_test_split() for consistent data splitting across all models
    
    Data split: 2023-01-01 to 2023-05-24 (training) | 2023-05-25 to 2023-06-30 (test)
    """
    print("Converting hourly data to daily aggregation for SARIMA analysis...")
    
    # Aggregate hourly data to daily for faster processing
    daily_production = df['AC_POWER_KW'].resample('D').sum()
    daily_irradiance_mean = df['ALLSKY_SFC_SW_DWN'].resample('D').mean()
    daily_temp_mean = df['T2M'].resample('D').mean()
    
    # Create daily dataframe
    daily_df = pd.DataFrame({
        'production': daily_production,
        'irradiance': daily_irradiance_mean,
        'temperature': daily_temp_mean
    })
    
    # Filter out days with very low production and handle missing values
    daily_df = daily_df[daily_df['production'] > 0.1].copy()
    daily_df = daily_df.fillna(method='ffill').fillna(method='bfill')
    
    # Ensure frequency is set to avoid statsmodels warnings
    daily_df.index.freq = 'D'
    
    print(f"Daily samples: {len(daily_df)} (reduced from {len(df)} hourly samples)")
    
    # *** USE COMMON TRAIN/TEST SPLIT FOR FAIR COMPARISON ***
    # Use the centralized function to ensure consistency
    split_info = get_common_train_test_split(daily_df, test_start_date='2023-05-25')
    train_mask = split_info['train_mask']
    test_mask = split_info['test_mask']
    
    # Define feature columns for baseline model (daily aggregated)
    available_cols = ['irradiance', 'temperature']
    
    # Create features matrix
    X = daily_df[available_cols].fillna(0)
    y = daily_df['production']
    
    # Apply common split
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    
    # Fit linear regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    # Get predictions and residuals
    y_pred_train = lr_model.predict(X_train)
    y_pred_test = lr_model.predict(X_test)
    
    train_residual = y_train - y_pred_train
    test_residual = y_test - y_pred_test
    
    # Ensure residuals have proper frequency for SARIMA
    train_residual.index.freq = 'D'
    test_residual.index.freq = 'D'
    
    # Create train and test dataframes with residuals (daily data)
    train_data = daily_df[train_mask].copy()
    test_data = daily_df[test_mask].copy()
    
    train_data['baseline_pred'] = y_pred_train
    train_data['residual'] = train_residual
    
    test_data['baseline_pred'] = y_pred_test
    test_data['residual'] = test_residual
    
    # Calculate baseline metrics
    baseline_mae = mean_absolute_error(y_test, y_pred_test)
    baseline_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    print(f"Baseline Linear Regression - MAE: {baseline_mae:.4f}, RMSE: {baseline_rmse:.4f}")
    
    # Add SIMPLIFIED seasonal modeling appropriate for 6-month data
    print("Adding simplified seasonal components (optimized for 6-month data)...")
    
    # For 6 months of data, use simpler seasonal modeling
    train_data['doy'] = train_data.index.dayofyear
    test_data['doy'] = test_data.index.dayofyear
    
    # Only use primary annual cycle - remove semi-annual for shorter series
    train_data['sin1'] = np.sin(2 * np.pi * train_data['doy'] / 365.25)
    train_data['cos1'] = np.cos(2 * np.pi * train_data['doy'] / 365.25)
    test_data['sin1'] = np.sin(2 * np.pi * test_data['doy'] / 365.25)
    test_data['cos1'] = np.cos(2 * np.pi * test_data['doy'] / 365.25)
    
    # Add a simple linear trend for the 6-month period
    train_trend = np.arange(len(train_data)) / len(train_data)
    test_trend = np.arange(len(train_data), len(train_data) + len(test_data)) / len(train_data)
    
    train_data['trend'] = train_trend
    test_data['trend'] = test_trend
    
    # Use Ridge regression with STRONG regularization for short series
    seasonal_features = ['sin1', 'cos1', 'trend']  # Simplified features
    X_seasonal_train = train_data[seasonal_features].values
    
    # Much stronger regularization for 6-month data
    seasonal_model = RidgeCV(alphas=[1.0, 10.0, 100.0, 1000.0])  # Higher alphas
    seasonal_model.fit(X_seasonal_train, train_data['production'])
    
    # Get seasonal predictions
    train_data['seasonal'] = seasonal_model.predict(X_seasonal_train)
    X_seasonal_test = test_data[seasonal_features].values
    test_data['seasonal'] = seasonal_model.predict(X_seasonal_test)
    
    # Update residuals to remove seasonal component
    train_data['residual'] = train_data['production'] - train_data['seasonal']
    test_data['residual'] = test_data['production'] - test_data['seasonal']
    
    print(f"Simplified seasonal model R²: {seasonal_model.score(X_seasonal_train, train_data['production']):.4f}")
    print(f"Regularization alpha used: {seasonal_model.alpha_}")
    print(f"Residual range: [{train_data['residual'].min():.2f}, {train_data['residual'].max():.2f}]")
    print(f"Residual std: {train_data['residual'].std():.4f} (smaller is better for SARIMA)")
    
    return train_data, test_data, lr_model, available_cols


def prepare_exogenous_features(train_data, test_data, feature_cols):
    """
    Prepare minimal and highly stable exogenous features for 6-month SARIMA
    Focus on preventing overfitting with very simple transformations
    """
    # Ultra-conservative approach: minimal feature engineering
    
    # Use simple median-based robust normalization
    overall_irrad_median = train_data['irradiance'].median()
    overall_irrad_mad = (train_data['irradiance'] - overall_irrad_median).abs().median()
    
    # Robust standardization using median and MAD (more stable than mean/std)
    if overall_irrad_mad > 0:
        train_data['irrad_normalized'] = (train_data['irradiance'] - overall_irrad_median) / overall_irrad_mad
        test_data['irrad_normalized'] = (test_data['irradiance'] - overall_irrad_median) / overall_irrad_mad
    else:
        # Fallback if no variation
        train_data['irrad_normalized'] = 0
        test_data['irrad_normalized'] = 0
    
    # Aggressive clipping to prevent outlier influence
    train_data['irrad_normalized'] = np.clip(train_data['irrad_normalized'], -2, 2)
    test_data['irrad_normalized'] = np.clip(test_data['irrad_normalized'], -2, 2)
    
    # Prepare exogenous arrays 
    exog_train = train_data[['irrad_normalized']].values
    exog_test = test_data[['irrad_normalized']].values
    
    # Skip additional scaling since we already normalized
    print(f"Exogenous variable stats - Train mean: {exog_train.mean():.3f}, std: {exog_train.std():.3f}")
    
    return exog_train, exog_test, None


def find_best_sarima_model(train_residual, exog_train_scaled):
    """
    Find the best SARIMA model using simplified grid search focused on generalization
    """
    print("\n=== Finding Best SARIMA Model (Simplified for 6-month data) ===")
    
    # Simplified parameter ranges for better generalization on short data
    # Focus on simple, robust models
    p_range = range(0, 3)  # Reduced from 4
    d_range = [0, 1]       # Keep simple differencing
    q_range = range(0, 3)  # Reduced from 4
    
    # Generate conservative combinations prioritizing generalization
    all_combinations = []
    
    # Priority 1: Simple non-seasonal models (most robust for short series)
    for p, d, q in itertools.product(p_range, d_range, q_range):
        if p == 0 and d == 0 and q == 0:
            continue
        # Very conservative complexity limit for better generalization
        if p + d + q <= 3:  # Reduced from 4
            all_combinations.append(((p, d, q), (0, 0, 0, 0)))
    
    # Priority 2: Only the simplest seasonal models for 6-month data
    # These are manually selected as most appropriate for short seasonal series
    simple_seasonal_models = [
        ((0, 1, 1), (1, 0, 0, 7)),  # ARIMA(0,1,1) with seasonal AR(1)
        ((1, 0, 1), (0, 0, 1, 7)),  # ARIMA(1,0,1) with seasonal MA(1)  
        ((0, 0, 1), (1, 0, 0, 7)),  # Simple MA with seasonal AR
    ]
    
    # Only add seasonal models if they're very conservative
    for seasonal_model in simple_seasonal_models:
        all_combinations.append(seasonal_model)
    
    print(f"Total configurations to test: {len(all_combinations)}")
    
    # Prepare cross-validation
    tscv = TimeSeriesSplit(n_splits=3)
    
    # Evaluate all models sequentially with progress bar
    results_list = []
    
    print("\nEvaluating models with 3-fold CV...")
    for order, seasonal_order in tqdm(all_combinations, desc="Model search"):
        try:
            cv_scores = []
            
            for train_idx, val_idx in tscv.split(train_residual):
                try:
                    y_cv_train = train_residual.iloc[train_idx]
                    y_cv_val = train_residual.iloc[val_idx]
                    exog_cv_train = exog_train_scaled[train_idx]
                    exog_cv_val = exog_train_scaled[val_idx]
                    
                    # Fit model with conservative settings for stability
                    model = SARIMAX(
                        y_cv_train,
                        exog=exog_cv_train,
                        order=order,
                        seasonal_order=seasonal_order,
                        enforce_stationarity=True,   # Enforce for stability
                        enforce_invertibility=True   # Enforce for stability
                    )
                    
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        result = model.fit(
                            disp=False,
                            maxiter=50,  # Reduced for faster, more stable fitting
                            warn_convergence=False,
                            method='lbfgs',
                            low_memory=True  # More conservative memory usage
                        )
                    
                    # Forecast
                    pred = result.forecast(steps=len(val_idx), exog=exog_cv_val)
                    
                    # Calculate RMSE
                    if not np.isnan(pred).any() and not np.isinf(pred).any():
                        rmse = np.sqrt(mean_squared_error(y_cv_val, pred))
                        if rmse < 10:  # Sanity check
                            cv_scores.append(rmse)
                            
                except Exception:
                    continue
            
            # Store result if we have valid CV scores
            if len(cv_scores) >= 2:  # At least 2 of 3 folds succeeded
                avg_score = np.mean(cv_scores)
                n_params = sum(order) + sum(seasonal_order[:3])
                results_list.append((order, seasonal_order, avg_score, n_params))
                
        except Exception:
            continue
    
    print(f"\nCompleted: {len(results_list)} valid models found")
    
    # Find best models with strong overfitting penalty
    if len(results_list) == 0:
        print("No valid models found, using fallback ARIMA(1,0,0)")
        best_order = (1, 0, 0)  # Even simpler fallback
        best_seasonal = (0, 0, 0, 0)
        best_score = np.inf
        best_params = 1
    else:
        # Apply HEAVY complexity penalty to prevent overfitting on short series
        adjusted_results = []
        for order, seasonal_order, score, n_params in results_list:
            # Much stronger penalty for complexity (increased from 0.01 to 0.05)
            complexity_penalty = 1 + 0.05 * n_params
            
            # Additional penalty for seasonal models on short data
            seasonal_penalty = 1.1 if seasonal_order != (0, 0, 0, 0) else 1.0
            
            # Combined penalty strongly favors simple models
            total_penalty = complexity_penalty * seasonal_penalty
            adjusted_score = score * total_penalty
            adjusted_results.append((order, seasonal_order, score, adjusted_score, n_params))
        
        # Sort by adjusted score (heavily penalized for complexity)
        adjusted_results.sort(key=lambda x: x[3])
        
        # Show top 15 models
        print("\n" + "="*80)
        print("TOP 15 MODELS (sorted by adjusted CV-RMSE)")
        print("="*80)
        print(f"{'Rank':<5} {'Order':<15} {'Seasonal':<15} {'CV-RMSE':<10} {'Adjusted':<10} {'Params':<7}")
        print("-"*80)
        
        for i, (order, seasonal, cv_score, adj_score, n_params) in enumerate(adjusted_results[:15], 1):
            seasonal_str = str(seasonal) if seasonal != (0,0,0,0) else "None"
            print(f"{i:<5} {str(order):<15} {seasonal_str:<15} {cv_score:<10.4f} {adj_score:<10.4f} {n_params:<7}")
        
        # Select best model
        best_order, best_seasonal, best_score, best_adj_score, best_params = adjusted_results[0]
        
        print("\n" + "="*80)
        print(f"SELECTED MODEL: ARIMA{best_order}")
        if best_seasonal != (0, 0, 0, 0):
            print(f"                x {best_seasonal}")
        print(f"CV-RMSE: {best_score:.4f}")
        print(f"Adjusted Score: {best_adj_score:.4f}")
        print(f"Parameters: {best_params}")
        print("="*80)
    
    return best_order, best_seasonal


def train_final_sarima_model(train_residual, test_residual, exog_train_scaled, exog_test_scaled, 
                           best_order, best_seasonal):
    """
    Train the final SARIMA model and make predictions
    """
    print("\n=== Training Final SARIMA Model ===")
    print(f"Model: ARIMA{best_order}")
    if best_seasonal != (0, 0, 0, 0):
        print(f"Seasonal: {best_seasonal}")
    
    # Train final model with conservative settings to prevent overfitting
    final_model = SARIMAX(
        train_residual,
        exog=exog_train_scaled,
        order=best_order,
        seasonal_order=best_seasonal,
        enforce_stationarity=True,   # Enforce for stability on short series
        enforce_invertibility=True   # Enforce for stability on short series
    )
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_result = final_model.fit(
            disp=False,
            maxiter=100,  # Reduced iterations to prevent overfitting
            warn_convergence=False,
            method='lbfgs',
            low_memory=True
        )
    
    # Make predictions
    sarima_pred_residual = final_result.forecast(steps=len(test_residual), exog=exog_test_scaled)
    train_pred_residual = final_result.fittedvalues
    
    # Calculate residual prediction metrics
    residual_mae = mean_absolute_error(test_residual, sarima_pred_residual)
    residual_rmse = np.sqrt(mean_squared_error(test_residual, sarima_pred_residual))
    
    print(f"SARIMA Residual Prediction - MAE: {residual_mae:.4f}, RMSE: {residual_rmse:.4f}")
    
    return final_result, sarima_pred_residual, train_pred_residual


def evaluate_sarima_performance(test_data, train_data, sarima_pred_residual, train_pred_residual):
    """
    Evaluate the complete SARIMA model performance (seasonal + residual predictions)
    """
    # Combined prediction = seasonal + SARIMA residual prediction
    sarima_combined_pred = test_data['seasonal'] + sarima_pred_residual
    # Clip to avoid negative values
    sarima_combined_pred = np.clip(sarima_combined_pred, 0, None)
    
    # Training predictions for overfitting analysis
    train_combined_pred = train_data['seasonal'] + train_pred_residual
    train_combined_pred = np.clip(train_combined_pred, 0, None)
    
    # Calculate final metrics
    actual = test_data['production']
    train_actual = train_data['production']
    
    sarima_mae = mean_absolute_error(actual, sarima_combined_pred)
    sarima_rmse = np.sqrt(mean_squared_error(actual, sarima_combined_pred))
    
    # Training metrics for overfitting analysis
    train_mae = mean_absolute_error(train_actual, train_combined_pred)
    train_rmse = np.sqrt(mean_squared_error(train_actual, train_combined_pred))
    
    # R² scores
    from sklearn.metrics import r2_score
    test_r2 = r2_score(actual, sarima_combined_pred)
    train_r2 = r2_score(train_actual, train_combined_pred)
    
    # Also calculate seasonal-only metrics for comparison
    seasonal_mae = mean_absolute_error(actual, test_data['seasonal'])
    seasonal_rmse = np.sqrt(mean_squared_error(actual, test_data['seasonal']))
    
    # Overfitting ratios
    overfitting_mae = sarima_mae / train_mae if train_mae > 0 else float('inf')
    overfitting_rmse = sarima_rmse / train_rmse if train_rmse > 0 else float('inf')
    
    print("\n" + "="*80)
    print("FINAL SARIMA RESULTS")
    print("="*80)
    print("Overall Performance:")
    print(f"  Test RMSE:        {sarima_rmse:.4f} kWh")
    print(f"  Test MAE:         {sarima_mae:.4f} kWh")
    print(f"  Test R²:          {test_r2:.4f}")
    print("\nTraining Performance:")
    print(f"  Train RMSE:       {train_rmse:.4f} kWh")
    print(f"  Train MAE:        {train_mae:.4f} kWh") 
    print(f"  Train R²:         {train_r2:.4f}")
    print("\nGeneralization:")
    print(f"  Overfitting:      {overfitting_rmse:.2f}x {'✓' if overfitting_rmse < 2 else '⚠️'}")
    print(f"  MAE Overfitting:  {overfitting_mae:.2f}x")
    print("\nComparison to Seasonal Baseline:")
    print(f"  Seasonal Only    - MAE: {seasonal_mae:.4f}, RMSE: {seasonal_rmse:.4f}")
    print(f"  SARIMA Combined  - MAE: {sarima_mae:.4f}, RMSE: {sarima_rmse:.4f}")
    print(f"  Improvement      - MAE: {((seasonal_mae - sarima_mae) / seasonal_mae * 100):+.1f}%, "
          f"RMSE: {((seasonal_rmse - sarima_rmse) / seasonal_rmse * 100):+.1f}%")
    print("\n  Target: < 1.5x for good generalization")
    print("="*80)
    
    sarima_metrics = {
        'MAE': sarima_mae,
        'RMSE': sarima_rmse,
        'R²': test_r2,
        'Train_MAE': train_mae,
        'Train_RMSE': train_rmse,
        'Train_R²': train_r2,
        'Overfitting_RMSE': overfitting_rmse,
        'Overfitting_MAE': overfitting_mae,
        'Baseline_MAE': seasonal_mae,
        'Baseline_RMSE': seasonal_rmse,
        'MAE_Improvement_%': (seasonal_mae - sarima_mae) / seasonal_mae * 100,
        'RMSE_Improvement_%': (seasonal_rmse - sarima_rmse) / seasonal_rmse * 100,
        'predictions': sarima_combined_pred,
        'train_predictions': train_combined_pred,
        'baseline_predictions': test_data['seasonal'],
        'residual_predictions': sarima_pred_residual
    }
    
    return sarima_metrics


def plot_sarima_results(train_data, test_data, sarima_metrics, best_order, best_seasonal):
    """
    Create comprehensive SARIMA visualization like the original notebook
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    
    # Extract data for plotting
    train_actual = train_data['production'].values
    test_actual = test_data['production'].values
    
    forecast_series = pd.Series(sarima_metrics['predictions'], index=test_data.index)
    train_pred_series = pd.Series(sarima_metrics['train_predictions'], index=train_data.index)
    
    # Create the comprehensive visualization
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: Full time series (top, full width)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(train_data.index, train_actual, 
             label='Train Actual', alpha=0.5, linewidth=1, color='steelblue')
    ax1.plot(train_data.index, train_pred_series,
             label='Train Fit', alpha=0.6, linewidth=1, color='navy')
    ax1.plot(test_data.index, test_actual, 
             label='Test Actual', color='darkorange', linewidth=2, alpha=0.8)
    ax1.plot(forecast_series.index, forecast_series, 
             label='Forecast', color='darkgreen', linewidth=2.5)
    
    seasonal_label = f"x{best_seasonal}" if best_seasonal != (0,0,0,0) else ""
    ax1.set_title(f'Daily PV Production | Model: ARIMA{best_order}{seasonal_label}', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Production (kWh)', fontsize=11)
    ax1.legend(loc='upper left', fontsize=10, ncol=2)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Test period zoom
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(test_data.index, test_actual, 
             label='Actual', color='darkorange', linewidth=2.5, marker='o', markersize=4)
    ax2.plot(forecast_series.index, forecast_series, 
             label='Forecast', color='darkgreen', linewidth=2.5, marker='s', markersize=3)
    ax2.plot(test_data.index, test_data['seasonal'],
             label='Seasonal Component', color='purple', linewidth=2, linestyle='--', alpha=0.7)
    ax2.set_title('Test Period: Forecast Tracking (Seasonal + Residual)', 
                  fontsize=13, fontweight='bold')
    ax2.set_ylabel('Production (kWh)', fontsize=11)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Residuals over time
    ax3 = fig.add_subplot(gs[2, 0])
    errors = test_actual - forecast_series.values
    colors = np.where(np.abs(errors) > 1, 'red', np.where(np.abs(errors) > 0.5, 'orange', 'green'))
    ax3.scatter(test_data.index, errors, c=colors, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    ax3.axhline(0, color='black', linestyle='--', linewidth=2)
    ax3.axhline(errors.mean(), color='blue', linestyle='--', linewidth=1.5, 
                label=f'Mean: {errors.mean():.3f}')
    ax3.fill_between(test_data.index, -0.5, 0.5, alpha=0.15, color='green', label='±0.5 kWh')
    ax3.set_title('Forecast Errors Over Time', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Date', fontsize=10)
    ax3.set_ylabel('Error (kWh)', fontsize=10)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Error distribution
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.hist(errors, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax4.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
    ax4.axvline(errors.mean(), color='green', linestyle='--', linewidth=2,
                label=f'Mean: {errors.mean():.3f}')
    ax4.set_xlabel('Error (kWh)', fontsize=10)
    ax4.set_ylabel('Frequency', fontsize=10)
    ax4.set_title('Error Distribution', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Actual vs Predicted
    ax5 = fig.add_subplot(gs[3, 0])
    ax5.scatter(test_actual, forecast_series, alpha=0.6, s=80, 
               edgecolors='black', linewidth=0.5, color='steelblue')
    min_val = min(test_actual.min(), forecast_series.min())
    max_val = max(test_actual.max(), forecast_series.max())
    ax5.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
    ax5.set_xlabel('Actual (kWh)', fontsize=10)
    ax5.set_ylabel('Predicted (kWh)', fontsize=10)
    ax5.set_title('Actual vs Predicted', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Residual component analysis
    ax6 = fig.add_subplot(gs[3, 1])
    ax6.plot(test_data.index, test_data['residual'], 
             label='Actual Residual', color='orange', linewidth=1.5, alpha=0.7)
    ax6.plot(test_data.index, sarima_metrics['residual_predictions'], 
             label='Predicted Residual', color='green', linewidth=1.5)
    ax6.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax6.fill_between(test_data.index, 0, test_data['residual'], alpha=0.2, color='orange')
    ax6.set_title('Residual Component (After Seasonal Removal)', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Date', fontsize=10)
    ax6.set_ylabel('Residual (kWh)', fontsize=10)
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle(f'SARIMA Model Analysis | Test RMSE: {sarima_metrics["RMSE"]:.3f} | Overfitting: {sarima_metrics["Overfitting_RMSE"]:.2f}x', 
                 fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.show()


def plot_sarima_diagnostics(sarima_model):
    """
    Plot SARIMA model diagnostics
    """
    import matplotlib.pyplot as plt
    
    try:
        sarima_model.plot_diagnostics(figsize=(14, 10))
        plt.suptitle('SARIMA Diagnostics (Residual Component)', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Could not generate diagnostics: {e}")


def run_sarima_experiment(df):
    """
    Main function to run the complete SARIMA experiment
    Uses get_common_train_test_split() for consistent data splitting across all models
    
    Data split: 2023-01-01 to 2023-05-24 (training) | 2023-05-25 to 2023-06-30 (test)
    """
    print("=== STARTING SARIMA EXPERIMENT ===")
    
    # Step 1: Create regression baseline with common split  
    train_data, test_data, lr_model, feature_cols = create_regression_baseline(df)
    
    # Step 2: Prepare simplified exogenous features  
    exog_train_scaled, exog_test_scaled, _ = prepare_exogenous_features(
        train_data, test_data, feature_cols
    )
    
    # Step 3: Find best SARIMA model
    best_order, best_seasonal = find_best_sarima_model(
        train_data['residual'], exog_train_scaled
    )
    
    # Step 4: Train final model
    final_result, sarima_pred_residual, train_pred_residual = train_final_sarima_model(
        train_data['residual'], test_data['residual'], 
        exog_train_scaled, exog_test_scaled, 
        best_order, best_seasonal
    )
    
    # Step 5: Evaluate performance
    sarima_metrics = evaluate_sarima_performance(test_data, train_data, sarima_pred_residual, train_pred_residual)
    
    # Step 6: Create comprehensive visualizations
    print("\n🎨 Creating SARIMA visualization plots...")
    plot_sarima_results(train_data, test_data, sarima_metrics, best_order, best_seasonal)
    
    # Step 7: Plot model diagnostics
    print("📊 Creating SARIMA diagnostic plots...")
    plot_sarima_diagnostics(final_result)
    
    return {
        'sarima_metrics': sarima_metrics,
        'train_data': train_data,
        'test_data': test_data,
        'lr_model': lr_model,
        'sarima_model': final_result,
        'feature_cols': feature_cols,
        'best_order': best_order,
        'best_seasonal': best_seasonal
    }