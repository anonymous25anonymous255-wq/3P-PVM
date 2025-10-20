"""
LSTM Models for Solar PV Power Prediction

This module provides LSTM-based deep learning models for solar PV power prediction,
following the same pattern as other models in the project.
"""
from src.phase_1.data_processing import get_common_train_test_split

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import sys
import os
warnings.filterwarnings('ignore')

proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    import tensorflow as tf
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("⚠️  TensorFlow not available. LSTM functionality will be limited.")
    TENSORFLOW_AVAILABLE = False


def prepare_lstm_data(df, time_steps=24):
    """
    Prepare data for LSTM training with sequences
    
    Args:
        df: DataFrame with PV data and features
        time_steps: Number of time steps to look back (default 24 hours)
    
    Returns:
        dict: Contains X_seq, y_seq, scalers, and feature info
    """
    print(f"🔧 Preparing LSTM data with {time_steps} time steps...")
    
    # Define features for LSTM (similar to ML models but focused on key variables)
    lstm_features = [
        'ALLSKY_SFC_SW_DWN',    # Solar irradiance
        'POA_IRRADIANCE',       # Plane of Array irradiance  
        'T2M',                  # Temperature
        'CELL_TEMP_C',          # Cell temperature
        'TEMP_DERATE',          # Temperature derating
        'WS10M',                # Wind speed
        'RH2M',                 # Humidity
        'hour',                 # Hour of day
        'month',                # Month
        'clearness_ratio'       # Clearness index
    ]
    
    # Add basic features if not present
    if 'hour' not in df.columns:
        df['hour'] = df.index.hour
    if 'month' not in df.columns:
        df['month'] = df.index.month
    if 'clearness_ratio' not in df.columns:
        df['clearness_ratio'] = df['ALLSKY_SFC_SW_DWN'] / (df['CLRSKY_SFC_SW_DWN'] + 0.001)
    
    # Select available features
    available_features = [f for f in lstm_features if f in df.columns]
    print(f"   Using {len(available_features)} features: {available_features}")
    
    # Prepare feature matrix and target
    X = df[available_features].values
    y = df['AC_POWER_KW'].values.reshape(-1, 1)
    
    # Scale the data
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    # Create sequences
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, time_steps)
    
    print(f"   Sequences created: X shape {X_seq.shape}, y shape {y_seq.shape}")
    
    return {
        'X_seq': X_seq,
        'y_seq': y_seq,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'features': available_features,
        'time_steps': time_steps
    }


def create_sequences(X, y, time_steps):
    """
    Create sequences for LSTM input
    """
    X_seq, y_seq = [], []
    
    for i in range(time_steps, len(X)):
        X_seq.append(X[i-time_steps:i])
        y_seq.append(y[i])
    
    return np.array(X_seq), np.array(y_seq)


def split_lstm_data_with_common_split(lstm_data, df):
    """
    Split LSTM data using get_common_train_test_split() for consistent data splitting
    Uses the same centralized function as other models for fair comparison
    
    Args:
        lstm_data: Dict from prepare_lstm_data
        df: Original dataframe with datetime index for split reference
    
    Returns:
        dict: Contains train, validation, and test splits
    """
    X_seq = lstm_data['X_seq']
    y_seq = lstm_data['y_seq']
    time_steps = lstm_data['time_steps']
    
    # We need to map the original dataframe indices to sequence indices
    # LSTM sequences start from time_steps index in the original data
    original_indices = df.index[time_steps:]  # Skip first time_steps due to lookback
    
    # Apply the common split to the sequence-compatible indices
    # Use the centralized function to ensure consistency
    split_info = get_common_train_test_split(df, test_start_date='2023-05-25')
    
    # Map the original split to sequence indices
    sequence_train_mask = split_info['train_mask'][time_steps:]  # Skip first time_steps
    sequence_test_mask = split_info['test_mask'][time_steps:]    # Skip first time_steps
    
    # Find sequence indices for train and test
    train_size = sequence_train_mask.sum()
    test_size = sequence_test_mask.sum()
    
    # Validation set: use last 15% of training data for validation
    val_size = int(train_size * 0.15)
    actual_train_size = train_size - val_size
    
    print("📊 LSTM Data Split (Common Test Period):")
    print(f"   Train: {actual_train_size:,} sequences")
    print(f"   Validation: {val_size:,} sequences")
    print(f"   Test: {test_size:,} sequences")
    if test_size > 0:
        test_start_idx = train_size
        test_end_idx = train_size + test_size
        print(f"   Test period: {original_indices[test_start_idx]} to {original_indices[test_end_idx-1]}")
    
    splits = {
        'X_train': X_seq[:actual_train_size],
        'y_train': y_seq[:actual_train_size],
        'X_val': X_seq[actual_train_size:train_size],
        'y_val': y_seq[actual_train_size:train_size],
        'X_test': X_seq[train_size:train_size + test_size],
        'y_test': y_seq[train_size:train_size + test_size],
        'test_indices': original_indices[train_size:train_size + test_size] if test_size > 0 else pd.Index([])
    }
    
    return splits


def build_lstm_model(input_shape):
    """
    Build LSTM model architecture optimized for PV prediction
    
    Args:
        input_shape: Tuple (time_steps, n_features)
    
    Returns:
        Compiled Keras model
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow is required for LSTM models")
    
    model = Sequential([
        # First LSTM layer with return sequences
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        
        # Second LSTM layer with return sequences  
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        
        # Third LSTM layer without return sequences
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        
        # Dense layers
        Dense(16, activation='relu'),
        Dense(1)  # Output layer
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model


def train_lstm_model(model, splits, epochs=100, batch_size=32, patience=15):
    """
    Train the LSTM model with early stopping and checkpoints
    
    Args:
        model: Compiled Keras model
        splits: Dict with training data splits
        epochs: Maximum number of epochs
        batch_size: Training batch size
        patience: Early stopping patience
    
    Returns:
        Training history
    """
    print("🚀 Training LSTM model...")
    print(f"   Epochs: {epochs}, Batch size: {batch_size}, Patience: {patience}")
    
    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=0
    )
    
    # Train the model
    history = model.fit(
        splits['X_train'], splits['y_train'],
        validation_data=(splits['X_val'], splits['y_val']),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=0
    )
    
    print(f"   Training completed after {len(history.history['loss'])} epochs")
    return history


def evaluate_lstm_model(model, splits, lstm_data, original_df=None):
    """
    Evaluate LSTM model and calculate metrics
    
    Args:
        model: Trained Keras model
        splits: Data splits dict
        lstm_data: Original LSTM data dict
        original_df: Original DataFrame with datetime index for alignment
    
    Returns:
        dict: Evaluation metrics and predictions
    """
    print("📊 Evaluating LSTM model...")
    
    scaler_y = lstm_data['scaler_y']
    
    # Make predictions on test set
    y_pred_scaled = model.predict(splits['X_test'], verbose=0)
    
    # Inverse transform to original scale
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = scaler_y.inverse_transform(splits['y_test'])
    
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Additional metrics
    max_error = np.max(np.abs(y_true - y_pred))
    mean_actual = np.mean(y_true)
    
    # Training metrics for overfitting analysis
    y_train_pred_scaled = model.predict(splits['X_train'], verbose=0)
    y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled)
    y_train_true = scaler_y.inverse_transform(splits['y_train'])
    
    train_mae = mean_absolute_error(y_train_true, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train_true, y_train_pred))
    
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2,
        'Train_MAE': train_mae,
        'Train_RMSE': train_rmse,
        'Max_Error': max_error,
        'Mean_Actual': mean_actual,
        'Overfitting_MAE': train_mae / mae if mae > 0 else float('inf'),
        'Overfitting_RMSE': train_rmse / rmse if rmse > 0 else float('inf')
    }
    
    # Create results DataFrame with proper datetime index alignment
    # Use the test_indices from splits which were correctly calculated
    if 'test_indices' in splits and len(splits['test_indices']) > 0:
        test_indices = splits['test_indices']
        
        # Ensure we have the right number of predictions to match test indices
        if len(test_indices) == len(y_true):
            results_df = pd.DataFrame({
                'Actual': y_true.flatten(),
                'LSTM_Predicted': y_pred.flatten()
            }, index=test_indices)
            
            print(f"   ✅ LSTM results aligned with datetime index: {len(results_df)} test samples")
            print(f"   Test period: {test_indices[0]} to {test_indices[-1]}")
        else:
            print(f"   ⚠️ Length mismatch: test_indices={len(test_indices)}, predictions={len(y_true)}")
            # Fallback to sequential index
            results_df = pd.DataFrame({
                'Actual': y_true.flatten(),
                'LSTM_Predicted': y_pred.flatten()
            })
    else:
        # Fallback to sequential index
        results_df = pd.DataFrame({
            'Actual': y_true.flatten(),
            'LSTM_Predicted': y_pred.flatten()
        })
    
    print("   LSTM Test Metrics:")
    print(f"     MAE: {mae:.4f}")
    print(f"     RMSE: {rmse:.4f}")
    print(f"     R²: {r2:.4f}")
    print(f"     Overfitting (MAE): {metrics['Overfitting_MAE']:.2f}x")
    
    return {
        'metrics': metrics,
        'results_df': results_df,
        'predictions': y_pred,
        'actual': y_true
    }


def plot_lstm_results(history, evaluation, lstm_data):
    """
    Create comprehensive LSTM visualization plots
    
    Args:
        history: Training history
        evaluation: Evaluation results dict
        lstm_data: LSTM data dict
    """
    print("📈 Creating LSTM visualization plots...")
    
    plt.figure(figsize=(18, 12))
    
    # Plot 1: Training history - Loss
    plt.subplot(3, 4, 1)
    plt.plot(history.history['loss'], label='Train Loss', linewidth=2, color='blue')
    plt.plot(history.history['val_loss'], label='Val Loss', linewidth=2, color='red')
    plt.title('LSTM Training Loss', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Training history - MAE
    plt.subplot(3, 4, 2)
    plt.plot(history.history['mae'], label='Train MAE', linewidth=2, color='blue')
    plt.plot(history.history['val_mae'], label='Val MAE', linewidth=2, color='red')
    plt.title('LSTM Training MAE', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Metrics summary
    plt.subplot(3, 4, 3)
    plt.axis('off')
    metrics = evaluation['metrics']
    metrics_text = f"""LSTM Test Metrics
    
MAE: {metrics['MAE']:.4f} kW
RMSE: {metrics['RMSE']:.4f} kW  
R²: {metrics['R²']:.4f}
Overfitting: {metrics['Overfitting_MAE']:.2f}x
Max Error: {metrics['Max_Error']:.4f} kW"""
    
    plt.text(0.1, 0.5, metrics_text, fontsize=11, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # Plot 4: Full time series comparison
    plt.subplot(3, 4, 4)
    results_df = evaluation['results_df']
    sample_size = min(168, len(results_df))  # 1 week max
    plt.plot(results_df['Actual'][:sample_size], label='Actual', alpha=0.8, linewidth=2)
    plt.plot(results_df['LSTM_Predicted'][:sample_size], label='LSTM Predicted', alpha=0.8, linewidth=2)
    plt.title('LSTM Predictions (First Week)', fontsize=12, fontweight='bold')
    plt.xlabel('Hour')
    plt.ylabel('AC Power (kW)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Scatter plot - Actual vs Predicted
    plt.subplot(3, 4, 5)
    plt.scatter(evaluation['actual'], evaluation['predictions'], alpha=0.6, s=20, color='green')
    max_val = max(evaluation['actual'].max(), evaluation['predictions'].max())
    plt.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    plt.xlabel('Actual AC Power (kW)')
    plt.ylabel('Predicted AC Power (kW)')
    plt.title(f'LSTM: Actual vs Predicted\nR² = {metrics["R²"]:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Residuals histogram
    plt.subplot(3, 4, 6)
    residuals = evaluation['actual'].flatten() - evaluation['predictions'].flatten()
    plt.hist(residuals, bins=50, alpha=0.7, color='purple', edgecolor='black')
    plt.title('LSTM Residuals Distribution', fontsize=12, fontweight='bold')
    plt.xlabel('Residuals (kW)')
    plt.ylabel('Frequency')
    plt.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    plt.axvline(x=np.mean(residuals), color='g', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(residuals):.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 7: Residual scatter plot
    plt.subplot(3, 4, 7)
    plt.scatter(evaluation['predictions'], residuals, alpha=0.6, s=20, color='orange')
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Predicted AC Power (kW)')
    plt.ylabel('Residuals (kW)')
    plt.title('LSTM Residual Plot', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Plot 8: Error over time
    plt.subplot(3, 4, 8)
    abs_errors = np.abs(residuals)
    plt.plot(abs_errors[:sample_size], alpha=0.7, linewidth=1.5, color='red')
    plt.axhline(y=np.mean(abs_errors), color='blue', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(abs_errors):.3f}')
    plt.title('LSTM Absolute Error Over Time', fontsize=12, fontweight='bold')
    plt.xlabel('Time Step')
    plt.ylabel('Absolute Error (kW)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 9-12: Additional analysis
    # Model architecture summary
    plt.subplot(3, 4, 9)
    plt.axis('off')
    arch_text = f"""LSTM Architecture
    
Time Steps: {lstm_data['time_steps']}
Features: {len(lstm_data['features'])}
Layers: LSTM(128) → LSTM(64) → LSTM(32)
Dense: 16 → 1
Dropout: 0.2 each layer"""
    
    plt.text(0.1, 0.5, arch_text, fontsize=10, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Feature importance (show top features used)
    plt.subplot(3, 4, 10)
    plt.axis('off')
    features_text = "LSTM Features Used:\n\n" + "\n".join([f"• {f}" for f in lstm_data['features'][:8]])
    if len(lstm_data['features']) > 8:
        features_text += f"\n... and {len(lstm_data['features'])-8} more"
    
    plt.text(0.1, 0.9, features_text, fontsize=9, verticalalignment='top',
             family='monospace')
    
    # Training summary
    plt.subplot(3, 4, 11)
    plt.axis('off')
    train_epochs = len(history.history['loss'])
    min_val_loss = min(history.history['val_loss'])
    final_train_loss = history.history['loss'][-1]
    
    train_text = f"""Training Summary
    
Epochs: {train_epochs}
Final Train Loss: {final_train_loss:.6f}
Best Val Loss: {min_val_loss:.6f}
Generalization: {'Good' if metrics['Overfitting_MAE'] < 1.5 else 'Fair' if metrics['Overfitting_MAE'] < 2.0 else 'Poor'}"""
    
    plt.text(0.1, 0.5, train_text, fontsize=10, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    # Performance comparison context
    plt.subplot(3, 4, 12)
    plt.axis('off')
    perf_text = f"""Performance Context
    
Mean Actual: {metrics['Mean_Actual']:.3f} kW
MAE %: {(metrics['MAE']/metrics['Mean_Actual']*100):.1f}%
RMSE %: {(metrics['RMSE']/metrics['Mean_Actual']*100):.1f}%
Model: Deep Learning (LSTM)
Suitable for: Sequential patterns"""
    
    plt.text(0.1, 0.5, perf_text, fontsize=10, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))
    
    plt.tight_layout()
    plt.show()
    
    print("✅ LSTM visualization completed!")


def run_lstm_experiment(df, time_steps=24, epochs=100, batch_size=32):
    """
    Run complete LSTM experiment following the same pattern as other models
    Uses get_common_train_test_split() for consistent data splitting across all models
    
    Data split: 2023-01-01 to 2023-05-24 (training) | 2023-05-25 to 2023-06-30 (test)
    
    Args:
        df: DataFrame with PV data 
        time_steps: Number of time steps for LSTM (default 24)
        epochs: Maximum training epochs (default 100)
        batch_size: Training batch size (default 32)
    
    Returns:
        dict: Complete LSTM experiment results
    """
    print("🔬 Running LSTM Deep Learning Experiment")
    print("="*60)

    if not TENSORFLOW_AVAILABLE:
        print("❌ TensorFlow not available. Skipping LSTM experiment.")
        return {
            'lstm_metrics': {'MAE': float('inf'), 'RMSE': float('inf'), 'R²': 0.0},
            'error': 'TensorFlow not available'
        }

    try:
        # 1. Prepare LSTM data
        lstm_data = prepare_lstm_data(df, time_steps=time_steps)
        
        # 2. Split data using common test period (fixed date)
        splits = split_lstm_data_with_common_split(lstm_data, df)
        
        # 3. Build model
        input_shape = (lstm_data['time_steps'], len(lstm_data['features']))
        model = build_lstm_model(input_shape)
        
        print(f"   LSTM Model built with input shape: {input_shape}")
        
        # 4. Train model
        history = train_lstm_model(model, splits, epochs=epochs, batch_size=batch_size)
        
        # 5. Evaluate model (pass original df for datetime alignment)
        evaluation = evaluate_lstm_model(model, splits, lstm_data, original_df=df)
        
        # 6. Create visualizations
        plot_lstm_results(history, evaluation, lstm_data)
        
        # 7. Return results in standard format
        results = {
            'lstm_metrics': evaluation['metrics'],
            'results_df': evaluation['results_df'],
            'history': history.history,
            'lstm_data': lstm_data,
            'model_summary': {
                'architecture': 'LSTM',
                'time_steps': time_steps,
                'features': len(lstm_data['features']),
                'epochs_trained': len(history.history['loss'])
            }
        }
        
        print("\n✅ LSTM experiment completed successfully!")
        return results
        
    except Exception as e:
        print(f"❌ Error in LSTM experiment: {str(e)}")
        return {
            'lstm_metrics': {'MAE': float('inf'), 'RMSE': float('inf'), 'R²': 0.0},
            'error': str(e)
        }
def print_lstm_performance_summary(lstm_results):
    """
    Print LSTM performance summary
    
    Args:
        lstm_results: Results from run_lstm_experiment
    """
    if 'error' in lstm_results:
        print(f"❌ LSTM Error: {lstm_results['error']}")
        return
    
    metrics = lstm_results['lstm_metrics']
    
    print("\n" + "="*50)
    print("LSTM DEEP LEARNING - PERFORMANCE SUMMARY")
    print("="*50)
    print(f"MAE:              {metrics['MAE']:.6f} kW")
    print(f"RMSE:             {metrics['RMSE']:.6f} kW") 
    print(f"R²:               {metrics['R²']:.6f}")
    print(f"Overfitting MAE:  {metrics.get('Overfitting_MAE', 'N/A'):.2f}x")
    print(f"Overfitting RMSE: {metrics.get('Overfitting_RMSE', 'N/A'):.2f}x")
    print("="*50)
    
    # Performance assessment
    if metrics.get('Overfitting_MAE', float('inf')) < 1.5:
        print("✅ LSTM shows good generalization (overfitting < 1.5x)")
    elif metrics.get('Overfitting_MAE', float('inf')) < 2.0:
        print("⚠️  LSTM shows moderate overfitting (1.5x < overfitting < 2x)")
    else:
        print("❌ LSTM shows significant overfitting (overfitting > 2x)")
    
    if metrics['R²'] > 0.9:
        print("✅ LSTM shows excellent predictive performance (R² > 0.9)")
    elif metrics['R²'] > 0.7:
        print("✅ LSTM shows good predictive performance (R² > 0.7)")
    else:
        print("⚠️  LSTM shows moderate predictive performance")