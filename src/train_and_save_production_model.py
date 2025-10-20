"""
Production Model Training Script
Trains XGBoost on Jan 2023 - Sep 2024, tests on Oct-Dec 2024
Saves model as .pkl for Phase 2 deployment
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import json
from pathlib import Path
from datetime import datetime


def load_and_split_data(data_path='./data/processed_pv_data_2023_2024.csv', 
                        train_end='2024-09-30', 
                        test_start='2024-10-01'):
    """
    Load data and split chronologically for production training
    """
    print("=" * 80)
    print("📂 LOADING AND SPLITTING DATA")
    print("=" * 80)
    
    # Load data
    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    print(f"✅ Loaded {len(df):,} samples")
    print(f"   Date range: {df.index.min()} to {df.index.max()}")
    print(f"   Columns: {len(df.columns)}")
    
    # Check for target column
    if 'pv_production_kwh' not in df.columns:
        raise ValueError("pv_production_kwh column not found! Check your data.")
    
    # Split chronologically
    train_data = df[df.index <= train_end].copy()
    test_data = df[df.index >= test_start].copy()
    
    print(f"\n📅 CHRONOLOGICAL SPLIT:")
    print(f"   Training:  {train_data.index.min().date()} to {train_data.index.max().date()}")
    print(f"              {len(train_data):,} samples ({len(train_data)/len(df)*100:.1f}%)")
    print(f"   Testing:   {test_data.index.min().date()} to {test_data.index.max().date()}")
    print(f"              {len(test_data):,} samples ({len(test_data)/len(df)*100:.1f}%)")
    
    # Get feature columns (exclude target)
    feature_columns = [col for col in df.columns if col != 'pv_production_kwh']
    print(f"\n🔧 Features: {len(feature_columns)} columns")
    
    # Prepare X and y
    X_train = train_data[feature_columns].copy()
    y_train = train_data['pv_production_kwh'].copy()
    X_test = test_data[feature_columns].copy()
    y_test = test_data['pv_production_kwh'].copy()
    
    # Handle missing values
    print(f"\n🧹 Cleaning data...")
    train_na_before = X_train.isna().any(axis=1).sum()
    test_na_before = X_test.isna().any(axis=1).sum()
    
    train_mask = ~X_train.isna().any(axis=1) & ~y_train.isna()
    test_mask = ~X_test.isna().any(axis=1) & ~y_test.isna()
    
    X_train = X_train[train_mask]
    y_train = y_train[train_mask]
    X_test = X_test[test_mask]
    y_test = y_test[test_mask]
    
    print(f"   Train: Removed {train_na_before} rows with NaN → {len(X_train):,} samples")
    print(f"   Test:  Removed {test_na_before} rows with NaN → {len(X_test):,} samples")
    
    return X_train, X_test, y_train, y_test, feature_columns


def train_production_xgboost(X_train, X_test, y_train, y_test, feature_columns):
    """
    Train XGBoost using your existing configuration
    (Same hyperparameters as your working model)
    """
    print("\n" + "=" * 80)
    print("🚀 TRAINING PRODUCTION XGBOOST")
    print("=" * 80)
    
    # Target statistics
    target_mean = y_train.mean()
    target_std = y_train.std()
    print(f"\n📊 Target Statistics:")
    print(f"   Mean: {target_mean:.4f} kW")
    print(f"   Std:  {target_std:.4f} kW")
    print(f"   Min:  {y_train.min():.4f} kW")
    print(f"   Max:  {y_train.max():.4f} kW")
    
    # Configure XGBoost (YOUR EXACT SETTINGS)
    print(f"\n⚙️  Configuring XGBoost...")
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
    print(f"\n🏋️  Training model (max 2000 trees, early stopping after 100)...")
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=100
    )
    
    best_iteration = getattr(xgb_model, 'best_iteration', xgb_model.n_estimators)
    print(f"\n✅ Training complete!")
    print(f"   Best iteration: {best_iteration}")
    
    # Predictions
    y_pred_train = xgb_model.predict(X_train)
    y_pred_test = xgb_model.predict(X_test)
    
    # Calculate metrics
    train_mae = mean_absolute_error(y_train, y_pred_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    train_r2 = r2_score(y_train, y_pred_train)
    
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_r2 = r2_score(y_test, y_pred_test)
    test_mape = np.mean(np.abs((y_test - y_pred_test) / np.maximum(y_test, 0.001))) * 100
    energy_error = np.abs(y_test.sum() - y_pred_test.sum()) / y_test.sum() * 100
    
    overfit_gap_mae = train_mae - test_mae
    overfit_gap_r2 = train_r2 - test_r2
    
    # Print results
    print("\n" + "=" * 80)
    print("📊 MODEL PERFORMANCE")
    print("=" * 80)
    
    print("\n🎯 TRAINING SET:")
    print(f"   MAE:  {train_mae:.4f} kW")
    print(f"   RMSE: {train_rmse:.4f} kW")
    print(f"   R²:   {train_r2:.4f}")
    
    print("\n🎯 TEST SET (Oct-Dec 2024):")
    print(f"   MAE:  {test_mae:.4f} kW")
    print(f"   RMSE: {test_rmse:.4f} kW")
    print(f"   R²:   {test_r2:.4f}")
    print(f"   MAPE: {test_mape:.2f}%")
    print(f"   Total Energy Error: {energy_error:.2f}%")
    
    print("\n🔍 OVERFITTING CHECK:")
    print(f"   MAE Gap:  {overfit_gap_mae:+.4f} kW")
    print(f"   R² Gap:   {overfit_gap_r2:+.4f}")
    
    if overfit_gap_r2 > 0.1:
        print("   Status: 🚨 SIGNIFICANT OVERFITTING")
    elif overfit_gap_r2 > 0.05:
        print("   Status: ⚠️  MODERATE OVERFITTING")
    else:
        print("   Status: ✅ GOOD GENERALIZATION")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n🏆 TOP 10 FEATURES:")
    for i, row in feature_importance.head(10).iterrows():
        print(f"   {i+1:2d}. {row['feature']:35s} {row['importance']:.4f}")
    
    return xgb_model, {
        'train': {'mae': train_mae, 'rmse': train_rmse, 'r2': train_r2},
        'test': {'mae': test_mae, 'rmse': test_rmse, 'r2': test_r2, 
                 'mape': test_mape, 'energy_error': energy_error},
        'overfit': {'mae_gap': overfit_gap_mae, 'r2_gap': overfit_gap_r2},
        'best_iteration': best_iteration,
        'target_stats': {'mean': target_mean, 'std': target_std},
        'predictions': {'train': y_pred_train, 'test': y_pred_test},
        'feature_importance': feature_importance
    }


def save_production_model(model, metrics, feature_columns, X_test, y_test):
    """
    Save the trained model (.pkl) and the list of feature columns (.json)
    """
    print("\n" + "=" * 80)
    print("💾 SAVING PRODUCTION MODEL")
    print("=" * 80)

    Path("models").mkdir(exist_ok=True)

    # Save model
    model_path = "models/production_xgboost.pkl"
    joblib.dump(model, model_path)

    # Save feature columns
    feature_path = "models/feature_columns.json"
    with open(feature_path, "w") as f:
        json.dump(feature_columns, f, indent=4)

    model_size = Path(model_path).stat().st_size / (1024 * 1024)
    print(f"\n✅ Model saved: {model_path} ({model_size:.2f} MB)")
    print(f"✅ Feature columns saved: {feature_path}")

def create_validation_plot(y_test, y_pred_test, feature_importance, r2, mae):
    """
    Create comprehensive validation plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Actual vs Predicted
    axes[0, 0].scatter(y_test, y_pred_test, alpha=0.5, s=15, color='green', edgecolor='none')
    max_val = max(y_test.max(), y_pred_test.max())
    axes[0, 0].plot([0, max_val], [0, max_val], 'r--', linewidth=2.5, label='Perfect Prediction')
    axes[0, 0].set_xlabel('Actual AC Power (kW)', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Predicted AC Power (kW)', fontsize=12, fontweight='bold')
    axes[0, 0].set_title(f'Test Set Performance\nR² = {r2:.4f} | MAE = {mae:.4f} kW', 
                         fontsize=13, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Time Series (First week)
    sample_size = min(7 * 24, len(y_test))
    time_hours = range(sample_size)
    axes[0, 1].plot(time_hours, y_test.values[:sample_size], 
                    label='Actual', linewidth=2.5, color='black', alpha=0.8)
    axes[0, 1].plot(time_hours, y_pred_test[:sample_size], 
                    label='Predicted', linestyle='--', linewidth=2, color='green', alpha=0.9)
    axes[0, 1].set_xlabel('Hours from Oct 1, 2024', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('AC Power (kW)', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('First Week Predictions', fontsize=13, fontweight='bold')
    axes[0, 1].legend(fontsize=11, loc='upper right')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Residuals
    residuals = y_test.values - y_pred_test
    axes[1, 0].hist(residuals, bins=60, alpha=0.75, color='green', edgecolor='black', linewidth=0.8)
    axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=2.5, label='Zero Error')
    axes[1, 0].axvline(residuals.mean(), color='blue', linestyle=':', linewidth=2, 
                       label=f'Mean: {residuals.mean():.4f}')
    axes[1, 0].set_xlabel('Residuals (kW)', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[1, 0].set_title(f'Residuals Distribution\nStd: {residuals.std():.4f} kW', 
                         fontsize=13, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Top Features
    top_n = 12
    top_features = feature_importance.head(top_n)
    colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(top_features)))
    
    axes[1, 1].barh(range(len(top_features)), top_features['importance'], 
                    color=colors, edgecolor='black', linewidth=0.8)
    axes[1, 1].set_yticks(range(len(top_features)))
    axes[1, 1].set_yticklabels(top_features['feature'], fontsize=10)
    axes[1, 1].set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    axes[1, 1].set_title(f'Top {top_n} Most Important Features', fontsize=13, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    axes[1, 1].invert_yaxis()
    
    plt.tight_layout()
    


def main():
    """
    Main execution function
    """
    print("\n" + "=" * 80)
    print("🎯 PRODUCTION MODEL TRAINING PIPELINE")
    print("=" * 80)
    print("\nThis script will:")
    print("  1. Load data from Jan 2023 - Dec 2024")
    print("  2. Split: Train (Jan 2023 - Sep 2024) | Test (Oct - Dec 2024)")
    print("  3. Train XGBoost with your proven hyperparameters")
    print("  4. Evaluate performance on future data (Oct-Dec 2024)")
    print("  5. Save model as .pkl for Phase 2 deployment")
    print("\n" + "=" * 80)
    
    # Step 1: Load and split data
    X_train, X_test, y_train, y_test, feature_columns = load_and_split_data()
    
    # Step 2: Train model
    model, metrics = train_production_xgboost(X_train, X_test, y_train, y_test, feature_columns)
    
    # Step 3: Save everything
    save_production_model(model, metrics, feature_columns, X_test, y_test)
    
    print("\n" + "🎉" * 40)
    print("SUCCESS! Your production model is ready for Phase 2!")
    print("🎉" * 40)
    
    return model, metrics


if __name__ == "__main__":
    model, metrics = main()