# Smart Grid Energy Management System

This project implements a comprehensive energy management solution that combines solar PV power prediction with intelligent battery management and real-time decision making. The system is divided into two main phases:

## Phase 1: Solar PV Power Prediction

An advanced solar photovoltaic (PV) power prediction system using NASA POWER meteorological data, combining physical modeling with machine learning approaches.

### Key Features

- Physical PV system modeling with temperature effects and losses
- Multiple prediction models:
  - Random Forest and XGBoost ML models
  - SARIMA time series analysis
  - LSTM deep learning models
- Comprehensive feature engineering and evaluation
- Detailed performance visualization and analysis

## Phase 2: Energy Management System

A real-time energy management system that optimizes battery usage and power distribution based on predicted solar production and consumption patterns.

### Key Components

- Battery Management System
  - State of Charge (SOC) monitoring
  - Charging/discharging optimization
  - Operating range: 20-80% SOC
- Decision Engine
  - Real-time power flow optimization
  - Day-ahead scheduling
  - Dynamic strategy adjustment
- Consumption Modeling
  - Load profile analysis
  - Usage pattern prediction
  - Demand response capabilities

## Project Structure

```
├── solar_pv_analysis.ipynb    # Main Phase 1 analysis notebook
├── data/                      # Data files
│   ├── NASA POWER data       # Solar radiation and weather
│   └── consumption/          # Power usage data
├── src/                      # Source code
│   ├── phase_1/             # Solar PV prediction
│   │   ├── data_processing.py
│   │   ├── pv_modeling.py
│   │   ├── ml_models.py
│   │   ├── lstm_models.py
│   │   ├── sarima_models.py
│   │   └── visualization.py
│   └── phase_2/             # Energy management
│       ├── battery_manager.py
│       ├── decision_engine.py
│       ├── realtime_simulator.py
│       └── system_config.yaml
├── models/                   # Saved model files
├── results/                  # Analysis outputs
└── requirements.txt          # Dependencies
```

## Features

### 1. Solar Power Prediction (Phase 1)
- Advanced PV system modeling with temperature effects
- Multiple prediction approaches:
  - Random Forest with hyperparameter tuning
  - XGBoost with early stopping
  - SARIMA for time series analysis
  - LSTM for deep learning
- Comprehensive feature engineering
- Detailed performance evaluation

### 2. Energy Management (Phase 2)
- Real-time power flow optimization
- Battery state monitoring and management
- Intelligent charging/discharging strategies
- Consumption pattern analysis
- System performance visualization

## System Components

### Battery Management
- Optimal SOC range: 20-80%
- Dynamic charging/discharging based on:
  - Current SOC
  - Predicted solar production
  - Expected consumption
  - Grid conditions

### Decision Engine
- Real-time optimization strategies
- Day-ahead scheduling
- Power flow management
- Grid interaction control

### Simulation Framework
- Real-time system simulation
- Performance analysis
- Strategy validation
- System optimization

## Installation & Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Data Preparation**: 
   - Place NASA POWER data in `data/`
   - Configure consumption data in `data/consumption/`
   - Update paths in configuration files if needed

3. **Run Analysis**:
   - Phase 1: Execute `solar_pv_analysis.ipynb`
   - Phase 2: Configure `system_config.yaml` and run simulator

## Results and Analysis

The system provides comprehensive analysis through:
- Power prediction accuracy metrics
- Battery usage optimization results
- System performance visualization
- Energy efficiency metrics
- Cost-benefit analysis

## Key Outputs

- Solar power predictions
- Battery state monitoring
- System performance metrics
- Energy optimization results
- Real-time decision logs

## Performance Metrics

The system is evaluated on:
- Solar prediction accuracy
- Battery utilization efficiency
- Energy self-consumption rate
- Grid interaction optimization
- Overall system efficiency

## Requirements

- Python 3.7+
- Key packages: pandas, numpy, sklearn, tensorflow
- See `requirements.txt` for complete list

## Contributing

Contributions welcome! Please read the contribution guidelines first.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

This comprehensive energy management system provides a complete solution for solar PV integration with battery storage, offering both predictive capabilities and real-time optimization for maximum energy efficiency.