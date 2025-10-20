# Solar PV Power Prediction Analysis

This project provides a comprehensive analysis framework for solar photovoltaic (PV) power prediction using NASA POWER meteorological data. The analysis includes advanced PV modeling, machine learning approaches, and time series forecasting.

## Project Structure

```
├── solar_pv_analysis.ipynb    # Main analysis notebook (centralized interface)
├── data/                      # NASA POWER data files
├── src/                       # Organized codebase modules
│   ├── __init__.py           # Package initialization
│   ├── data_processing.py    # Data loading, cleaning, feature engineering
│   ├── pv_modeling.py        # Physical PV system modeling
│   ├── ml_models.py          # Machine learning models (RF, XGBoost)
│   ├── lstm_models.py        # LSTM time series modeling
│   ├── sarima_models.py      # SARIMA time series modeling
│   └── visualization.py      # Plotting and analysis functions
├── results/                  # Output files and results
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Features

### 1. Advanced PV System Modeling
- Physical-based power calculation with temperature effects
- System losses modeling (inverter, wiring, soiling, etc.)
- Cell temperature calculation using NOCT model
- Plane-of-array (POA) irradiance calculation

### 2. Machine Learning Models
- **Random Forest**: Hyperparameter-tuned with overfitting control
- **XGBoost**: Advanced gradient boosting with early stopping
- Feature engineering: temporal, lag, rolling, and weather features
- Cross-validation with time series splits

### 3. LSTM Neural Networks
- Sequence modeling with sliding windows
- Long Short-Term Memory (LSTM) cells for capturing temporal dependencies
- Early stopping to prevent overfitting

### 4. Time Series Analysis
- **SARIMA**: Seasonal AutoRegressive Integrated Moving Average
- Automated model selection with grid search
- Residual modeling approach with weather baseline
- Cross-validation for model selection

### 5. Comprehensive Evaluation
- Multiple performance metrics (MAE, RMSE, R², MAPE)
- Overfitting analysis and generalization assessment
- Feature importance analysis
- Comprehensive visualizations

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Data**: 
   - Place NASA POWER CSV files in the `data/` directory
   - Update the data file path in the notebook if needed

3. **Run Analysis**:
   - Open `solar_pv_analysis.ipynb` in Jupyter
   - Execute all cells to run the complete analysis
   - Results will be saved in the `results/` directory

## Key Functions

### Data Processing (`src/data_processing.py`)
- `load_nasa_power_data()`: Load and parse NASA POWER data
- `prepare_features_for_ml()`: Create ML features with lags and rolling statistics
- `get_feature_columns()`: Get standardized feature column names

### PV Modeling (`src/pv_modeling.py`) 
- `calculate_pv_power()`: Advanced PV power calculation
- `calculate_performance_metrics()`: System performance analysis

### ML Models (`src/ml_models.py`)
- `run_ml_experiments()`: Complete ML pipeline with RF and XGBoost

### LSTM Models (`src/lstm_models.py`)
- `run_lstm_experiment()`: Complete LSTM analysis pipeline

### SARIMA Models (`src/sarima_models.py`)
- `run_sarima_experiment()`: Complete SARIMA analysis pipeline

### Visualization (`src/visualization.py`)
- `plot_data_overview()`: Comprehensive data visualization
- `plot_model_results_comparison()`: Model comparison plots
- `print_performance_summary()`: Detailed performance metrics

## Output Files

- `results/processed_solar_data_complete.csv`: Complete processed dataset
- `results/model_comparison_summary.csv`: Model performance comparison
- `results/analysis_summary.json`: Key metrics and results summary
- `solar_power_predictions_comparison.csv`: ML model predictions

## Model Performance

The framework evaluates three different modeling approaches:

1. **Random Forest**: Ensemble method with strong regularization
2. **XGBoost**: Gradient boosting with advanced hyperparameter tuning  
3. **SARIMA**: Time series model with seasonal patterns and weather integration
4. **LSTM**: Long Short-Term Memory model for capturing temporal dependencies

Each model is evaluated using:
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE) 
- R² Score
- Mean Absolute Percentage Error (MAPE)
- Energy prediction accuracy

## Key Innovations

- **Physical-First Approach**: Combines physical PV modeling with data-driven methods
- **Overfitting Control**: Extensive regularization and validation for ML models
- **Temporal Features**: Advanced time-based feature engineering
- **Comprehensive Evaluation**: Multiple metrics and visualization approaches
- **Modular Design**: Clean separation of concerns for maintainability

## Usage Notes

- The notebook serves as a **façade** to the organized codebase in `src/`
- All major functionality is implemented in the source modules
- The analysis is designed for NASA POWER hourly meteorological data
- System parameters can be easily modified for different PV configurations
- Results include both technical metrics and business-relevant insights

## Requirements

- Python 3.7+
- See `requirements.txt` for specific package versions
- NASA POWER meteorological data in CSV format

This framework provides a complete solution for solar PV power prediction analysis, from raw meteorological data to production-ready forecasting models.