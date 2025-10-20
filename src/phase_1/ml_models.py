from src.phase_1.visualization import plot_individual_ml_predictions
from src.phase_1.data_processing import get_common_train_test_split
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import sys
import os

proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)


def train_random_forest(X_train, X_test, y_train, y_test):
    """
    Train and evaluate Random Forest with regularization to combat overfitting
    """
    print("Training HIGHLY Regularized Random Forest with Cross-Validation...")
    
    # Aggressive regularization
    rf_model = RandomForestRegressor(
        n_estimators=30,
        max_depth=3,
        min_samples_split=100,
        min_samples_leaf=50,
        max_features=0.2,
        max_samples=0.5,
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    cv_scores = cross_val_score(
        rf_model, X_train, y_train, 
        cv=tscv, 
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )
    
    cv_mae_scores = -cv_scores
    mean_cv_mae = cv_mae_scores.mean()
    std_cv_mae = cv_mae_scores.std()
    
    print(f"Cross-Validation MAE: {mean_cv_mae:.4f} (+/- {std_cv_mae:.4f})")
    
    # Train final model
    rf_model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = rf_model.predict(X_train)
    y_pred_rf = rf_model.predict(X_test)
    
    # Calculate metrics
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_rf)
    
    overfit_gap_mae = train_mae - mae_rf
    overfit_gap_r2 = train_r2 - test_r2
    
    # Feature importance analysis
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    rf_metrics = {
        'MAE': mae_rf,
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
        'R²': test_r2,
        'Train_R²': train_r2,
        'R²_Overfit_Gap': overfit_gap_r2,
        'MAPE': np.mean(np.abs((y_test - y_pred_rf) / np.maximum(y_test, 0.001))) * 100,
        'Energy Error %': np.abs(y_test.sum() - y_pred_rf.sum()) / y_test.sum() * 100,
        'Train MAE': train_mae,
        'Overfit Gap': overfit_gap_mae,
        'CV_MAE_Mean': mean_cv_mae,
        'CV_MAE_Std': std_cv_mae,
        'model': rf_model,
        'predictions': y_pred_rf,
        'feature_importance': feature_importance
    }
    
    print("\n=== RANDOM FOREST RESULTS ===")
    print(f"Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
    print(f"R² Overfit Gap: {overfit_gap_r2:.4f} (target: < 0.05)")
    print(f"Train MAE: {train_mae:.4f}, Test MAE: {mae_rf:.4f}")
    print(f"MAE Overfit Gap: {overfit_gap_mae:.4f} (ideal: close to 0)")
    
    # Performance assessment
    if overfit_gap_r2 > 0.1:
        print("🚨 SIGNIFICANT OVERFITTING - R² gap too large!")
    elif overfit_gap_r2 > 0.05:
        print("⚠️  MODERATE OVERFITTING - Consider stronger regularization")
    else:
        print("✅ GOOD GENERALIZATION - Overfitting is controlled")
    
    return rf_metrics


def train_xgboost(X_train, X_test, y_train, y_test):
    """
    Train and evaluate XGBoost with proper configuration for solar data
    """
    print("Training Properly Configured XGBoost...")
    
    # First, let's understand our data scale
    target_mean = y_train.mean()
    target_std = y_train.std()
    print(f"Target stats - Mean: {target_mean:.4f}, Std: {target_std:.4f}")
    
    # Configure XGBoost for small-scale solar data
    xgb_model = xgb.XGBRegressor(
        n_estimators=2000,
        max_depth=2,
        learning_rate=0.01,
        subsample=0.7,
        colsample_bytree=0.5,
        reg_alpha=1.0,
        reg_lambda=2.0,
        gamma=0.5,
        min_child_weight=5,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=100
    )
    
    # Train with early stopping
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50
    )
    
    # Predictions
    y_pred_xgb = xgb_model.predict(X_test)
    
    # Calculate metrics
    mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
    train_score = mean_absolute_error(y_train, xgb_model.predict(X_train))
    overfit_gap = train_score - mae_xgb
    
    xgb_metrics = {
        'MAE': mae_xgb,
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_xgb)),
        'R²': r2_score(y_test, y_pred_xgb),
        'MAPE': np.mean(np.abs((y_test - y_pred_xgb) / np.maximum(y_test, 0.001))) * 100,
        'Energy Error %': np.abs(y_test.sum() - y_pred_xgb.sum()) / y_test.sum() * 100,
        'Train MAE': train_score,
        'Overfit Gap': overfit_gap,
        'Best Iteration': getattr(xgb_model, 'best_iteration', 'N/A'),
        'model': xgb_model,
        'predictions': y_pred_xgb
    }
    
    print("XGBoost completed!")
    print(f"Train MAE: {train_score:.4f}, Test MAE: {mae_xgb:.4f}")
    print(f"Overfit Gap: {overfit_gap:.4f}, Best Iteration: {xgb_metrics['Best Iteration']}")
    
    return xgb_metrics


def compare_models(rf_metrics, xgb_metrics):
    """
    Compare the performance of both models
    """
    print("\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)
    
    comparison_data = {
        'Random Forest': {
            'MAE': rf_metrics['MAE'],
            'RMSE': rf_metrics['RMSE'],
            'R²': rf_metrics['R²'],
            'MAPE': rf_metrics['MAPE'],
            'Energy Error %': rf_metrics['Energy Error %']
        },
        'XGBoost': {
            'MAE': xgb_metrics['MAE'],
            'RMSE': xgb_metrics['RMSE'],
            'R²': xgb_metrics['R²'],
            'MAPE': xgb_metrics['MAPE'],
            'Energy Error %': xgb_metrics['Energy Error %']
        }
    }
    
    comparison_df = pd.DataFrame(comparison_data).T
    comparison_df = comparison_df.round(4)
    comparison_df['Energy Error %'] = comparison_df['Energy Error %'].round(2)
    
    print(comparison_df)
    
    # Determine best model for each metric
    print("\nBEST MODEL FOR EACH METRIC:")
    for metric in ['MAE', 'RMSE', 'R²', 'MAPE', 'Energy Error %']:
        if metric == 'R²':  # Higher is better
            best_model = comparison_df[metric].idxmax()
            best_value = comparison_df.loc[best_model, metric]
        else:  # Lower is better
            best_model = comparison_df[metric].idxmin()
            best_value = comparison_df.loc[best_model, metric]
        print(f"{metric}: {best_model} ({best_value})")
    
    return comparison_df


def aggregate_ml_predictions_to_daily(df, df_ml, feature_columns, ml_results):
    """
    Aggregate hourly ML predictions to daily kWh for fair comparison with SARIMA/LSTM.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Original dataframe with AC_POWER_KW column
    df_ml : pd.DataFrame
        ML-ready dataframe with features
    feature_columns : list
        List of feature column names
    ml_results : dict
        Dictionary containing 'rf_model' and 'xgb_model'
    
    Returns:
    --------
    pd.DataFrame
        Daily aggregated predictions with columns: Actual_kWh, RF_kWh, XGB_kWh
    """
    print("\n" + "="*60)
    print("AGGREGATING ML PREDICTIONS TO DAILY kWh")
    print("="*60)
    
    # Build full-hour ML predictions
    X_full = df_ml[feature_columns].copy()
    X_full = X_full[~X_full.isna().any(axis=1)]
    
    print(f"Total valid samples for prediction: {len(X_full)}")
    
    # Get models
    rf = ml_results.get('rf_model')
    xgb = ml_results.get('xgb_model')
    
    if rf is None or xgb is None:
        raise RuntimeError(
            "RF/XGB models missing in ml_results. "
            "Ensure ml_models.py returns models in the results dictionary."
        )
    
    # Generate predictions
    print("Generating Random Forest predictions...")
    rf_preds = rf.predict(X_full)
    
    print("Generating XGBoost predictions...")
    xgb_preds = xgb.predict(X_full)
    
    # Create hourly dataframe
    hourly = pd.DataFrame({
        'Actual_kW': df.loc[X_full.index, 'AC_POWER_KW'],
        'RF_kW': rf_preds,
        'XGB_kW': xgb_preds
    }, index=X_full.index)
    
    print(f"\nHourly predictions shape: {hourly.shape}")
    print(f"Date range: {hourly.index.min()} to {hourly.index.max()}")
    
    # Detect time resolution
    time_diffs = hourly.index.to_series().diff().dropna()
    median_interval = time_diffs.median()
    hours_per_reading = median_interval.total_seconds() / 3600
    
    print(f"\nDetected time interval: {median_interval}")
    print(f"Hours per reading: {hours_per_reading:.3f}")
    
    # Convert power (kW) to energy (kWh) and aggregate to daily
    # Energy (kWh) = Power (kW) × Time (hours)
    print(f"\nConverting kW to kWh using factor: {hours_per_reading}")
    daily_ml = hourly.resample('D').sum() * hours_per_reading
    
    # Rename columns for clarity
    daily_ml.columns = ['Actual_kWh', 'RF_kWh', 'XGB_kWh']
    
    print(f"\nDaily aggregated shape: {daily_ml.shape}")
    print(f"Days covered: {len(daily_ml)}")
    
    # Display sample and statistics
    print("\n--- Sample Daily Predictions (first 5 days) ---")
    print(daily_ml.head().to_string())
    
    print("\n--- Daily Statistics (kWh) ---")
    print(daily_ml.describe().round(2).to_string())
    
    # Calculate daily errors
    daily_ml['RF_Error_kWh'] = daily_ml['Actual_kWh'] - daily_ml['RF_kWh']
    daily_ml['XGB_Error_kWh'] = daily_ml['Actual_kWh'] - daily_ml['XGB_kWh']
    daily_ml['RF_Error_%'] = (daily_ml['RF_Error_kWh'] / daily_ml['Actual_kWh'] * 100).abs()
    daily_ml['XGB_Error_%'] = (daily_ml['XGB_Error_kWh'] / daily_ml['Actual_kWh'] * 100).abs()
    
    # Summary metrics
    print("\n--- Daily Prediction Accuracy ---")
    print(f"RF  - Mean Absolute Error: {daily_ml['RF_Error_kWh'].abs().mean():.2f} kWh")
    print(f"RF  - Mean Percentage Error: {daily_ml['RF_Error_%'].mean():.2f}%")
    print(f"XGB - Mean Absolute Error: {daily_ml['XGB_Error_kWh'].abs().mean():.2f} kWh")
    print(f"XGB - Mean Percentage Error: {daily_ml['XGB_Error_%'].mean():.2f}%")
    
    # Total energy comparison
    total_actual = daily_ml['Actual_kWh'].sum()
    total_rf = daily_ml['RF_kWh'].sum()
    total_xgb = daily_ml['XGB_kWh'].sum()
    
    print("\n--- Total Energy Over Period ---")
    print(f"Actual: {total_actual:.2f} kWh")
    print(f"RF:     {total_rf:.2f} kWh (error: {(total_rf - total_actual)/total_actual*100:+.2f}%)")
    print(f"XGB:    {total_xgb:.2f} kWh (error: {(total_xgb - total_actual)/total_actual*100:+.2f}%)")
    
    print("\n" + "="*60)
    
    return daily_ml


def plot_model_comparison(rf_metrics, xgb_metrics, y_test, feature_columns, rf_model, xgb_model):
    """
    Create comprehensive comparison plots
    """
    y_pred_rf = rf_metrics['predictions']
    y_pred_xgb = xgb_metrics['predictions']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Actual vs Predicted (RF)
    axes[0, 0].scatter(y_test, y_pred_rf, alpha=0.6, s=20, color='blue')
    max_val = max(y_test.max(), y_pred_rf.max())
    axes[0, 0].plot([0, max_val], [0, max_val], 'r--', linewidth=2)
    axes[0, 0].set_xlabel('Actual AC Power (kW)')
    axes[0, 0].set_ylabel('Predicted AC Power (kW)')
    axes[0, 0].set_title(f'Random Forest\nR² = {rf_metrics["R²"]:.3f}')
    axes[0, 0].grid(True)
    
    # Plot 2: Actual vs Predicted (XGBoost)
    axes[0, 1].scatter(y_test, y_pred_xgb, alpha=0.6, s=20, color='green')
    max_val = max(y_test.max(), y_pred_xgb.max())
    axes[0, 1].plot([0, max_val], [0, max_val], 'r--', linewidth=2)
    axes[0, 1].set_xlabel('Actual AC Power (kW)')
    axes[0, 1].set_ylabel('Predicted AC Power (kW)')
    axes[0, 1].set_title(f'XGBoost\nR² = {xgb_metrics["R²"]:.3f}')
    axes[0, 1].grid(True)
    
    # Plot 3: Time series comparison
    sample_size = min(72, len(y_test))
    time_index = range(sample_size)
    
    axes[0, 2].plot(time_index, y_test.values[:sample_size], label='Actual', linewidth=2, color='black')
    axes[0, 2].plot(time_index, y_pred_rf[:sample_size], label='RF', linestyle='--', alpha=0.8)
    axes[0, 2].plot(time_index, y_pred_xgb[:sample_size], label='XGB', linestyle='--', alpha=0.8)
    axes[0, 2].set_xlabel('Time (hours)')
    axes[0, 2].set_ylabel('AC Power (kW)')
    axes[0, 2].set_title('3-Day Prediction Comparison')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # Plot 4: Residuals comparison
    rf_residuals = y_test - y_pred_rf
    xgb_residuals = y_test - y_pred_xgb
    
    axes[1, 0].hist(rf_residuals, bins=50, alpha=0.7, color='blue', label=f'RF (std: {rf_residuals.std():.3f})')
    axes[1, 0].hist(xgb_residuals, bins=50, alpha=0.7, color='green', label=f'XGB (std: {xgb_residuals.std():.3f})')
    axes[1, 0].set_xlabel('Residuals (kW)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Residuals Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot 5: Feature importance comparison (top 10)
    top_n = 10
    
    rf_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': rf_model.feature_importances_
    }).nlargest(top_n, 'importance')
    
    xgb_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': xgb_model.feature_importances_
    }).nlargest(top_n, 'importance')
    
    # Get union of top features from both models
    all_top_features = set(rf_importance['feature'].tolist() + xgb_importance['feature'].tolist())
    
    # Create aligned comparison - get importance for each feature from both models
    comparison_data = []
    for feature in all_top_features:
        rf_imp = rf_importance[rf_importance['feature'] == feature]['importance'].iloc[0] if feature in rf_importance['feature'].values else 0
        xgb_imp = xgb_importance[xgb_importance['feature'] == feature]['importance'].iloc[0] if feature in xgb_importance['feature'].values else 0
        comparison_data.append({
            'feature': feature,
            'rf_importance': rf_imp,
            'xgb_importance': xgb_imp,
            'max_importance': max(rf_imp, xgb_imp)
        })
    
    # Sort by maximum importance and take top N
    comparison_df = pd.DataFrame(comparison_data).sort_values('max_importance', ascending=False).head(top_n)
    
    # Create side-by-side bars
    y_pos = np.arange(len(comparison_df))
    width = 0.35
    
    axes[1, 1].barh(y_pos - width/2, comparison_df['rf_importance'], width, 
                   alpha=0.7, color='blue', label='Random Forest')
    axes[1, 1].barh(y_pos + width/2, comparison_df['xgb_importance'], width, 
                   alpha=0.7, color='green', label='XGBoost')
    
    axes[1, 1].set_yticks(y_pos)
    axes[1, 1].set_yticklabels(comparison_df['feature'])
    axes[1, 1].set_xlabel('Feature Importance')
    axes[1, 1].set_title('Top 10 Feature Importance Comparison')
    axes[1, 1].legend()
    axes[1, 1].invert_yaxis()  # Highest importance at the top
    
    # Plot 6: Metrics comparison
    metrics = ['MAE', 'RMSE', 'MAPE']
    rf_values = [rf_metrics[m] for m in metrics]
    xgb_values = [xgb_metrics[m] for m in metrics]
    
    x_pos = np.arange(len(metrics))
    axes[1, 2].bar(x_pos - 0.2, rf_values, 0.4, label='RF', color='blue', alpha=0.7)
    axes[1, 2].bar(x_pos + 0.2, xgb_values, 0.4, label='XGB', color='green', alpha=0.7)
    axes[1, 2].set_xticks(x_pos)
    axes[1, 2].set_xticklabels(metrics)
    axes[1, 2].set_ylabel('Error Value')
    axes[1, 2].set_title('Error Metrics Comparison')
    axes[1, 2].legend()
    axes[1, 2].grid(True, axis='y')
    
    plt.tight_layout()
    plt.show()


def run_ml_experiments(df, df_ml, feature_columns):
    """
    Main function to run the complete ML model training and comparison
    Uses get_common_train_test_split() for consistent data splitting across all models
    
    Data split: 2023-01-01 to 2023-05-24 (training) | 2023-05-25 to 2023-06-30 (test)
    """
    # Get feature columns and target
    X = df_ml[feature_columns].copy()
    y = df_ml['AC_POWER_KW'].copy()
    
    # Remove any remaining NaN values
    mask = ~X.isna().any(axis=1) & ~y.isna()
    X = X[mask]
    y = y[mask]
    
    print(f"Final dataset size: {len(X)} samples")
    print(f"Number of features: {len(feature_columns)}")
    
    # *** USE COMMON TRAIN/TEST SPLIT FOR FAIR COMPARISON ***
    # Use the centralized function to ensure consistency
    split_info = get_common_train_test_split(df_ml, test_start_date='2023-05-25')
    
    # Apply the common split to our ML data
    train_mask = split_info['train_mask'][mask]  # Align with cleaned data
    test_mask = split_info['test_mask'][mask]    # Align with cleaned data
    
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    
    print(f"Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    print(f"Test period: {y_test.index[0]} to {y_test.index[-1]}")
    
    # Train models separately
    rf_metrics = train_random_forest(X_train, X_test, y_train, y_test)
    xgb_metrics = train_xgboost(X_train, X_test, y_train, y_test)
    
    # Compare models
    comparison_df = compare_models(rf_metrics, xgb_metrics)
    
    # Plot results
    plot_model_comparison(
        rf_metrics, xgb_metrics, y_test, feature_columns,
        rf_metrics['model'], xgb_metrics['model']
    )
    
    # Save predictions (for individual plotting)
    results_df = pd.DataFrame({
        'Actual': y_test,
        'RF_Predicted': rf_metrics['predictions'],
        'XGB_Predicted': xgb_metrics['predictions']
    }, index=y_test.index)
    
    # Create results for individual plotting
    ml_results_for_plotting = {
        'rf_metrics': rf_metrics,
        'xgb_metrics': xgb_metrics,
        'results_df': results_df
    }
    
    # Plot individual model predictions
    print("\n🎨 Creating individual Random Forest prediction plots...")
    plot_individual_ml_predictions(ml_results_for_plotting, 'Random Forest')
    
    print("\n🎨 Creating individual XGBoost prediction plots...")
    plot_individual_ml_predictions(ml_results_for_plotting, 'XGBoost')
    
    results_df.to_csv('data/solar_power_predictions_comparison.csv')
    print("\nPredictions saved to 'data/solar_power_predictions_comparison.csv'")
    
    # Create ml_results dictionary FIRST with models included
    ml_results = {
        'rf_metrics': rf_metrics,
        'xgb_metrics': xgb_metrics,
        'comparison_df': comparison_df,
        'results_df': results_df,
        'rf_model': rf_metrics['model'],
        'xgb_model': xgb_metrics['model'],
        'split_info': split_info  # Add split info for reference
    }
    
    # THEN call aggregation with the complete ml_results
    daily_ml = aggregate_ml_predictions_to_daily(df, df_ml, feature_columns, ml_results)
    ml_results['daily_ml'] = daily_ml
    
    return ml_results