import sys
sys.path.append('..')

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict
from datetime import timedelta
import warnings
import joblib
import json
warnings.filterwarnings('ignore')

from _01_config_models import SystemConfig
from _03_battery_manager import BatteryManager
from _05_decision_engine import DecisionEngine


class RealtimeEnergyManager:
    
    def __init__(self, 
                 config_path: str = './src/phase_2/system_config.yaml',
                 data_path: str = './data/processed_pv_data_2023_2024.csv',
                 production_model_path: str = 'models/production_xgboost.pkl',
                 consumption_model_path: str = 'models/consumption_xgboost.pkl'):
        
        print("\n" + "="*80)
        print(" "*15 + "PHASE 2: OFF-GRID ENERGY MANAGEMENT SYSTEM")
        print("="*80)
        
        print("\n📋 Step 1: Loading System Configuration...")
        self.config = SystemConfig.from_yaml(config_path)
        print(f"   ✅ Config loaded: {self.config.system_name}")
        print(f"   Location: {self.config.location}")
        print(f"   Battery: {self.config.battery.capacity_kwh} kWh")
        print(f"   System type: OFF-GRID (No grid connection)")
        
        # Initialize simulation tracking
        self.day_ahead_simulation = None
        self.daily_strategy = None
        
        print("\n🤖 Step 2: Loading Trained Production Model...")
        self.production_model = joblib.load(production_model_path)
        features_path = Path(production_model_path).parent / 'feature_columns.json'
        with open(features_path, 'r') as f:
            self.production_feature_columns = json.load(f)
        print(f"   ✅ Production XGBoost model loaded")
        print(f"   Features: {len(self.production_feature_columns)}")
        
        print("\n⚡ Step 2b: Loading Trained Consumption Model...")
        self.consumption_model = joblib.load(consumption_model_path)
        self.consumption_feature_columns = ['year', 'month', 'day', 'hour', 'day_of_week']
        print(f"   ✅ Consumption XGBoost model loaded")
        print(f"   Features: {len(self.consumption_feature_columns)}")
        
        self.daily_event_tracker = {
            'date': None,
            'min_soc': 100,
            'blackout_count': 0,
            'total_blackout_kw': 0,
            'total_wasted_kw': 0,
            'severe_weather_detected': False
        }
        
        print("\n📊 Step 3: Loading Historical Data...")
        self.df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        print(f"   ✅ Data loaded: {self.df.index.min()} to {self.df.index.max()}")
        print(f"   Total records: {len(self.df):,}")
        
        test_start = pd.Timestamp('2024-10-01')
        test_end = pd.Timestamp('2024-12-31')
        test_data = self.df[(self.df.index >= test_start) & (self.df.index <= test_end)]
        print(f"   Test period (Oct-Dec 2024): {len(test_data):,} hours")
        
        print("\n⚙️  Step 4: Initializing System Components...")
        self.battery_manager = BatteryManager(self.config.battery)
        self.decision_engine = DecisionEngine(self.config)
        print("   ✅ All components initialized")
        
        self.simulation_log = []
        self._consumption_data = None
        self._consumption_col = None
        self.day_ahead_simulation = None
        self.daily_strategy = None
        
        print("\n" + "="*80)
        print("✅ OFF-GRID SYSTEM READY FOR SIMULATION")
        print("="*80)
    
    def reset_daily_tracker(self, current_date):
        self.daily_event_tracker = {
            'date': current_date,
            'min_soc': 100,
            'blackout_count': 0,
            'total_blackout_kw': 0,
            'total_wasted_kw': 0,
            'severe_weather_detected': False
        }
    
    def update_daily_tracker(self, timestamp, battery_soc, blackout_kw, wasted_kw, weather_severity):
        current_date = timestamp.date()
        
        if self.daily_event_tracker['date'] != current_date:
            self.reset_daily_tracker(current_date)
        
        if battery_soc < self.daily_event_tracker['min_soc']:
            self.daily_event_tracker['min_soc'] = battery_soc
        
        if blackout_kw > 0.1:
            self.daily_event_tracker['blackout_count'] += 1
            self.daily_event_tracker['total_blackout_kw'] += blackout_kw
        
        if wasted_kw > 0.1:
            self.daily_event_tracker['total_wasted_kw'] += wasted_kw
        
        if weather_severity == 'critical':
            self.daily_event_tracker['severe_weather_detected'] = True
    
    def print_daily_summary(self, timestamp):
        if timestamp.hour != 23:
            return None
        
        tracker = self.daily_event_tracker
        alerts = []
        
        if tracker['blackout_count'] > 0:
            alerts.append(f"Blackout: {tracker['blackout_count']} hours, {tracker['total_blackout_kw']:.1f} kWh lost")
        
        if tracker['min_soc'] < 20:
            alerts.append(f"Battery critical: dropped to {tracker['min_soc']:.0f}%")
        
        if tracker['total_wasted_kw'] > 5.0:
            alerts.append(f"Wasted energy: {tracker['total_wasted_kw']:.1f} kW")
        
        if tracker['severe_weather_detected']:
            alerts.append(f"Severe weather detected")
        
        return " | ".join(alerts) if alerts else "No issues"
    
    def forecast_solar_production_horizon(self, start_timestamp: pd.Timestamp, hours: int = 24) -> list:
        """Generate solar production forecast for specified number of hours starting from given timestamp"""
        predictions = []
        current_time = start_timestamp
        
        for _ in range(hours):
            if current_time not in self.df.index:
                predictions.append(0.0)
            else:
                features = self.df.loc[current_time, self.production_feature_columns].values.reshape(1, -1)
                prediction = self.production_model.predict(features)[0]
                predictions.append(max(0.0, prediction))
            current_time += pd.Timedelta(hours=1)
        
        return predictions

    def predict_solar_production_for_hour(self, timestamp: pd.Timestamp) -> float:
        """Predict solar PV production for a specific hour using the trained model"""
        if timestamp not in self.df.index:
            print(f"⚠️  Warning: {timestamp} not in dataset")
            return 0.0
        
        features = self.df.loc[timestamp, self.production_feature_columns].values.reshape(1, -1)
        prediction = self.production_model.predict(features)[0]
        return max(0.0, prediction)
    
    def get_historical_power_data(self, timestamp: pd.Timestamp) -> tuple:
        """Retrieve historical solar production and load consumption data for analysis"""
        production = 0.0
        if timestamp in self.df.index and 'pv_production_kwh' in self.df.columns:
            production = float(self.df.loc[timestamp, 'pv_production_kwh'])
        
        consumption = 0.0
        consumption_path = './data/consumption/processed_power_usage_2023_2024.csv'
        
        # Load consumption data once and cache it
        if self._consumption_data is None:
            try:
                self._consumption_data = pd.read_csv(consumption_path, index_col=0, parse_dates=True)
                possible_cols = ['consumption_kwh', 'power_usage_kwh', 'load_kwh', 'consumption', 'power_usage']
                self._consumption_col = None
                for col in possible_cols:
                    if col in self._consumption_data.columns:
                        self._consumption_col = col
                        break
            except FileNotFoundError:
                print(f"⚠️  Warning: Consumption data not found at {consumption_path}")
                self._consumption_data = pd.DataFrame()  # Empty dataframe to avoid repeated attempts
        
        if self._consumption_data is not None and not self._consumption_data.empty:
            if timestamp in self._consumption_data.index and self._consumption_col:
                consumption = float(self._consumption_data.loc[timestamp, self._consumption_col])
        
        return production, consumption
    
    def forecast_load_consumption_horizon(self, start_timestamp: pd.Timestamp, hours: int = 24) -> list:
        """Generate load consumption forecast for specified number of hours starting from given timestamp"""
        predictions = []
        current_time = start_timestamp
        
        for _ in range(hours):
            features = pd.DataFrame({
                'year': [current_time.year],
                'month': [current_time.month],
                'day': [current_time.day],
                'hour': [current_time.hour],
                'day_of_week': [current_time.dayofweek]
            })
            prediction = self.consumption_model.predict(features[self.consumption_feature_columns])[0]
            predictions.append(max(0.0, prediction))
            current_time += pd.Timedelta(hours=1)
        
        return predictions

    def predict_load_consumption_for_hour(self, timestamp: pd.Timestamp) -> float:
        """Predict load consumption for a specific hour using the trained model"""
        features = pd.DataFrame({
            'year': [timestamp.year],
            'month': [timestamp.month],
            'day': [timestamp.day],
            'hour': [timestamp.hour],
            'day_of_week': [timestamp.dayofweek]
        })
        
        prediction = self.consumption_model.predict(features[self.consumption_feature_columns])[0]
        return max(0.0, prediction)

    def simulate_day_ahead_battery_profile(self, start_timestamp: pd.Timestamp) -> dict:
        """Simulate the battery behavior for the next 24 hours and determine if SOC bounds will be exceeded"""
        production_forecast = self.forecast_solar_production_horizon(start_timestamp, hours=24)
        consumption_forecast = self.forecast_load_consumption_horizon(start_timestamp, hours=24)
        
        current_soc = self.battery_manager.get_soc()
        hourly_net_power = []
        simulated_soc = current_soc
        soc_profile = [current_soc]
        will_exceed_bounds = False
        
        # Simulate the day's battery behavior
        for hour in range(24):
            net_power = production_forecast[hour] - consumption_forecast[hour]
            hourly_net_power.append(net_power)
            
            # Simulate battery state change
            if net_power > 0:  # Charging
                # Fixed: Multiply by efficiency, not divide
                max_energy_to_add = (95 - simulated_soc) * self.config.battery.capacity_kwh / 100
                energy_to_add = min(
                    net_power * 0.9,  # 90% charging efficiency
                    max_energy_to_add
                )
                soc_change = (energy_to_add / self.config.battery.capacity_kwh) * 100
                simulated_soc = min(95, simulated_soc + soc_change)
            else:  # Discharging
                # Fixed: Multiply by efficiency for available energy
                max_energy_to_remove = (simulated_soc - 15) * self.config.battery.capacity_kwh / 100
                energy_to_remove = min(
                    abs(net_power) / 0.9,  # Account for discharge efficiency
                    max_energy_to_remove
                )
                soc_change = (energy_to_remove / self.config.battery.capacity_kwh) * 100
                simulated_soc = max(15, simulated_soc - soc_change)
            
            soc_profile.append(simulated_soc)
            
            # Check if we'll exceed normal bounds (20-80%)
            if simulated_soc < 20 or simulated_soc > 80:
                will_exceed_bounds = True
        
        return {
            'total_production': sum(production_forecast),
            'total_consumption': sum(consumption_forecast),
            'hourly_production': production_forecast,
            'hourly_consumption': consumption_forecast,
            'hourly_net_power': hourly_net_power,
            'soc_profile': soc_profile,
            'will_exceed_bounds': will_exceed_bounds,
            'final_soc': simulated_soc,
        }
        
    def execute_energy_management_step(self, timestamp: pd.Timestamp) -> Dict:
        """Execute one time step of the energy management system with daily or hourly strategy"""
        
        # At midnight, perform day-ahead simulation for next 24 hours
        if timestamp.hour == 0:
            self.day_ahead_simulation = self.simulate_day_ahead_battery_profile(timestamp)
            # Calculate optimal SOC bounds based on the simulation
            current_soc = self.battery_manager.get_soc()
            soc_profile = self.day_ahead_simulation['soc_profile']
            soc_min = min(soc_profile)
            soc_max = max(soc_profile)
            
            # Add safety margins to the bounds
            optimal_min = max(20, soc_min - 5)  # Add 5% buffer below predicted minimum
            optimal_max = min(80, soc_max + 5)  # Add 5% buffer above predicted maximum
            
            self.daily_strategy = {
                'hourly_net_power': self.day_ahead_simulation['hourly_net_power'],
                'soc_profile': soc_profile,
                'soc_min': optimal_min,
                'soc_max': optimal_max,
                'will_exceed_bounds': self.day_ahead_simulation['will_exceed_bounds']
            }
        
        # Get predictions for current hour
        predicted_production_kw = self.predict_solar_production_for_hour(timestamp)
        predicted_consumption_kw = self.predict_load_consumption_for_hour(timestamp)
        predicted_net_power_kw = predicted_production_kw - predicted_consumption_kw
        current_battery_soc = self.battery_manager.get_soc()
        
        # Get weather forecast data
        future_data = self.df[
            (self.df.index >= timestamp) & 
            (self.df.index <= timestamp + timedelta(hours=72))
        ]
        
        # Check if we need hourly strategy based on day-ahead simulation
        if self.daily_strategy is not None and self.daily_strategy.get('will_exceed_bounds', True):
            # Use complex hourly decision making when SOC bounds will be exceeded
            strategic_decision = self.decision_engine.decide(
                production_kw=predicted_production_kw,
                consumption_kw=predicted_consumption_kw,
                battery_soc=current_battery_soc,
                weather_forecast=future_data,
                timestamp=timestamp,
                daily_strategy=self.daily_strategy
            )
        else:
            # Use simple daily strategy - maintain battery between 20-80%
            strategic_decision = {
                'action': 'idle',
                'power_kw': 0.0,
                'reason': 'Normal operation within bounds',
                'weather_severity': 'normal',
                'weather_status': 'clear',
                'bad_weather_detected': False,
                'blackout_deficit_kw': 0.0,
                'wasted_energy_kw': 0.0
            }
            
            # Calculate available battery capacity for charging/discharging
            if predicted_net_power_kw > 0:  # Surplus power available
                # Fixed: Multiply by efficiency for charging capacity
                remaining_capacity_kwh = ((80 - current_battery_soc) * self.config.battery.capacity_kwh / 100)
                max_charging_power = min(
                    predicted_net_power_kw,
                    self.config.battery.max_charge_kw,
                    remaining_capacity_kwh * 0.9  # Account for efficiency
                )
                
                if max_charging_power > 0.1 and current_battery_soc < 80:
                    strategic_decision.update({
                        'action': 'charge_battery',
                        'power_kw': max_charging_power,
                        'reason': 'Normal charging within daily bounds (20-80%)'
                    })
                else:
                    strategic_decision.update({
                        'action': 'waste_energy',
                        'wasted_energy_kw': predicted_net_power_kw,
                        'reason': 'Battery near upper bound, excess power wasted'
                    })
                    
            elif predicted_net_power_kw < 0:  # Power deficit
                required_power = abs(predicted_net_power_kw)
                # Fixed: Multiply by efficiency for available discharge energy
                available_energy_kwh = ((current_battery_soc - 20) * self.config.battery.capacity_kwh / 100) * 0.9
                max_discharge_power = min(
                    required_power,
                    self.config.battery.max_discharge_kw,
                    available_energy_kwh
                )
                
                if max_discharge_power > 0.1 and current_battery_soc > 20:
                    strategic_decision.update({
                        'action': 'discharge_battery',
                        'power_kw': max_discharge_power,
                        'reason': 'Normal discharging within daily bounds (20-80%)'
                    })
                    
                    if max_discharge_power < required_power:
                        strategic_decision.update({
                            'blackout_deficit_kw': required_power - max_discharge_power,
                            'reason': f'{strategic_decision["reason"]} | Partial blackout: insufficient discharge capacity'
                        })
                else:
                    strategic_decision.update({
                        'action': 'blackout',
                        'blackout_deficit_kw': required_power,
                        'reason': 'Battery at lower bound, cannot discharge further'
                    })
        
        # Initialize battery action variables
        battery_result = {'success': True, 'energy_change_kwh': 0.0, 'action': 'idle'}
        wasted_energy_kw = strategic_decision.get('wasted_energy_kw', 0.0)
        blackout_deficit_kw = strategic_decision.get('blackout_deficit_kw', 0.0)
        
        # Execute battery actions based on decision
        decision_action = strategic_decision['action']
        action_reason = strategic_decision.get('reason', '')
        
        if decision_action == 'charge_battery':
            charge_power = strategic_decision['power_kw']
            if charge_power > 0.1:
                try:
                    charge_result = self.battery_manager.charge(
                        charge_power,
                        duration_hours=1.0,
                        timestamp=timestamp
                    )
                    battery_result.update({
                        'success': True,
                        'energy_change_kwh': charge_result['energy_added_kwh'],
                        'action': 'charge'
                    })
                except Exception as e:
                    battery_result.update({
                        'success': False,
                        'error': str(e),
                        'action': 'charge_failed'
                    })
                    wasted_energy_kw = charge_power  # Energy that couldn't be stored is wasted
                
        elif decision_action == 'discharge_battery':
            discharge_power = strategic_decision['power_kw']
            if discharge_power > 0.1:
                try:
                    discharge_result = self.battery_manager.discharge(
                        discharge_power,
                        duration_hours=1.0,
                        timestamp=timestamp
                    )
                    battery_result.update({
                        'success': True,
                        'energy_change_kwh': -discharge_result['energy_removed_kwh'],
                        'action': 'discharge'
                    })
                    
                    # Update blackout deficit if discharge was insufficient
                    if strategic_decision.get('blackout_deficit_kw', 0) > 0:
                        blackout_deficit_kw = strategic_decision['blackout_deficit_kw']
                except Exception as e:
                    battery_result.update({
                        'success': False,
                        'error': str(e),
                        'action': 'discharge_failed'
                    })
                    blackout_deficit_kw = discharge_power  # Failed discharge leads to blackout
        
        # Get final battery state
        final_battery_soc = self.battery_manager.get_soc()
        
        # Update daily tracking metrics
        self.update_daily_tracker(
            timestamp=timestamp,
            battery_soc=final_battery_soc,
            blackout_kw=blackout_deficit_kw,
            wasted_kw=wasted_energy_kw,
            weather_severity=strategic_decision.get('weather_severity', 'normal')
        )
        
        # Generate daily summary if needed
        daily_summary = self.print_daily_summary(timestamp)
        
        # Issue critical alerts for severe blackouts
        if blackout_deficit_kw > 2.0:
            critical_message = (
                f"🚨 CRITICAL BLACKOUT ALERT at {timestamp.strftime('%Y-%m-%d %H:%M')}\n"
                f"   Predicted deficit: {blackout_deficit_kw:.1f} kW\n"
                f"   Battery SOC: {final_battery_soc:.1f}%\n"
                f"   Weather severity: {strategic_decision.get('weather_severity', 'unknown')}"
            )
            print(critical_message)
        
        # Get historical data for performance analysis (only for logging/metrics)
        historical_production_kw, historical_consumption_kw = self.get_historical_power_data(timestamp)
        historical_net_power_kw = historical_production_kw - historical_consumption_kw
        
        # Calculate prediction errors
        production_error = predicted_production_kw - historical_production_kw
        consumption_error = predicted_consumption_kw - historical_consumption_kw
        net_power_error = predicted_net_power_kw - historical_net_power_kw
        
        # Fixed: Better prediction accuracy calculation
        if abs(historical_net_power_kw) > 0.01:
            prediction_accuracy_pct = (1 - abs(net_power_error) / abs(historical_net_power_kw)) * 100
            prediction_accuracy_pct = max(0, min(100, prediction_accuracy_pct))  # Clamp to 0-100%
        else:
            # When actual net power is near zero, use absolute error instead
            prediction_accuracy_pct = 100 if abs(net_power_error) < 0.1 else 0
        
        # Prepare comprehensive timestep results
        return {
            # Timestamp information
            'timestamp': timestamp,
            'hour_of_day': timestamp.hour,
            'day_of_week': timestamp.dayofweek,
            
            # Predictions vs Actuals
            'predicted_production_kw': predicted_production_kw,
            'predicted_consumption_kw': predicted_consumption_kw,
            'predicted_net_power_kw': predicted_net_power_kw,
            'actual_production_kw': historical_production_kw,
            'actual_consumption_kw': historical_consumption_kw,
            'actual_net_power_kw': historical_net_power_kw,
            
            # Prediction Errors
            'production_error_kw': production_error,
            'consumption_error_kw': consumption_error,
            'net_power_error_kw': net_power_error,
            'prediction_accuracy_pct': prediction_accuracy_pct,
            
            # Battery Status
            'battery_soc_percent': final_battery_soc,
            'battery_energy_kwh': self.battery_manager.get_energy(),
            'battery_mode': self.battery_manager.mode.value,
            'battery_action': battery_result.get('action', 'idle'),
            'battery_action_success': battery_result['success'],
            'battery_energy_change_kwh': battery_result.get('energy_change_kwh', 0.0),
            'battery_error': battery_result.get('error', None),
            
            # Decision Making
            'decision_action': decision_action,
            'action_reason': action_reason,
            'strategy_type': 'hourly' if (self.daily_strategy and self.daily_strategy.get('will_exceed_bounds', True)) else 'daily',
            
            # System Performance
            'wasted_energy_kw': wasted_energy_kw,
            'blackout_deficit_kw': blackout_deficit_kw,
            'is_blackout': blackout_deficit_kw > 0.1,
            'blackout_severity': 'critical' if blackout_deficit_kw > 2.0 else 'minor' if blackout_deficit_kw > 0.1 else 'none',
            
            # Weather and Environmental
            'weather_status': strategic_decision.get('weather_status', 'unknown'),
            'weather_severity': strategic_decision.get('weather_severity', 'normal'),
            'bad_weather_ahead': strategic_decision.get('bad_weather_detected', False),
            
            # Alerts and Notifications
            'notifications_count': 1 if blackout_deficit_kw > 2.0 else 0,
            'has_critical_alert': blackout_deficit_kw > 2.0,
            'daily_summary': daily_summary,
            
            # System Efficiency
            'energy_efficiency_pct': (
                (predicted_consumption_kw / (predicted_production_kw + 1e-6)) * 100 
                if predicted_production_kw > 0 else 0.0
            )
        }
    
    def run_simulation(self, 
                      start_date: str = '2024-10-01',
                      end_date: str = '2024-12-31',
                      save_results: bool = True) -> pd.DataFrame:
        
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)
        
        timestamps = self.df[
            (self.df.index >= start_dt) & 
            (self.df.index <= end_dt)
        ].index
        
        total_hours = len(timestamps)
        total_days = total_hours / 24
        
        print("\n" + "="*80)
        print("🎬 STARTING OFF-GRID SIMULATION")
        print("="*80)
        print(f"Period: {start_date} to {end_date}")
        print(f"Duration: {total_hours:,} hours ({total_days:.1f} days)")
        print(f"Initial Battery SOC: {self.battery_manager.get_soc():.1f}%")
        print(f"Initial Battery Energy: {self.battery_manager.get_energy():.2f} kWh")
        print(f"Usable Battery Range: 15% - 95% ({self.config.battery.capacity_kwh * 0.80:.2f} kWh usable)")
        print("="*80 + "\n")
        
        self.simulation_log = []
        
        for idx, timestamp in enumerate(timestamps):
            # Fixed: Use correct method name
            result = self.execute_energy_management_step(timestamp)
            self.simulation_log.append(result)
            
            if (idx + 1) % 24 == 0:
                day_num = (idx + 1) // 24
                today_results = self.simulation_log[-24:]
                today_blackouts = sum(1 for r in today_results if r['is_blackout'])
                today_wasted = sum(r['wasted_energy_kw'] for r in today_results)
                
                print(f"📅 Day {day_num:3d}/{int(total_days)} | "
                      f"Date:{timestamp.date()} | "
                      f"Batt:{self.battery_manager.get_soc():5.1f}% | "
                      f"Blackout:{today_blackouts}x | "
                      f"Waste:{today_wasted:.1f}kW")
        
        print("\n" + "="*80)
        print("✅ OFF-GRID SIMULATION COMPLETE!")
        print("="*80)
        
        df_results = pd.DataFrame(self.simulation_log)
        self.print_summary(df_results)
        
        if save_results:
            self.save_results(df_results)
        
        return df_results
    
    def print_summary(self, df_results: pd.DataFrame):
        print("\n📊 OFF-GRID SIMULATION SUMMARY")
        print("="*80)
        
        total_predicted_prod = df_results['predicted_production_kw'].sum()
        total_actual_prod = df_results['actual_production_kw'].sum()
        prod_mae = np.abs(df_results['production_error_kw']).mean()
        prod_rmse = np.sqrt((df_results['production_error_kw']**2).mean())
        
        print("\n🌞 PV PRODUCTION:")
        print(f"   Predicted total: {total_predicted_prod:,.2f} kWh")
        print(f"   Actual total:    {total_actual_prod:,.2f} kWh")
        print(f"   Error:           {total_predicted_prod - total_actual_prod:+,.2f} kWh")
        print(f"   MAE:             {prod_mae:.4f} kW")
        print(f"   RMSE:            {prod_rmse:.4f} kW")
        
        if 'actual_consumption_kw' in df_results.columns:
            total_predicted_cons = df_results['predicted_consumption_kw'].sum()
            total_actual_cons = df_results['actual_consumption_kw'].sum()
            cons_mae = np.abs(df_results['consumption_error_kw']).mean()
            cons_rmse = np.sqrt((df_results['consumption_error_kw']**2).mean())
            
            print("\n⚡ CONSUMPTION PREDICTION:")
            print(f"   Predicted total: {total_predicted_cons:,.2f} kWh")
            print(f"   Actual total:    {total_actual_cons:,.2f} kWh")
            print(f"   Error:           {total_predicted_cons - total_actual_cons:+,.2f} kWh")
            print(f"   MAE:             {cons_mae:.4f} kW")
            print(f"   RMSE:            {cons_rmse:.4f} kW")
        
        total_consumption = df_results['predicted_consumption_kw'].sum()
        avg_consumption = df_results['predicted_consumption_kw'].mean()
        daily_avg = total_consumption / (len(df_results) / 24)
        
        print("\n⚡ CONSUMPTION USAGE:")
        print(f"   Total:       {total_consumption:,.2f} kWh")
        print(f"   Average:     {avg_consumption:.2f} kW")
        print(f"   Daily avg:   {daily_avg:.2f} kWh/day")
        
        net_energy = total_predicted_prod - total_consumption
        self_sufficiency = (total_predicted_prod / total_consumption * 100) if total_consumption > 0 else 0
        
        print("\n⚖️  ENERGY BALANCE:")
        print(f"   Production - Consumption: {net_energy:+,.2f} kWh")
        print(f"   Self-sufficiency ratio:   {self_sufficiency:.1f}%")
        
        max_soc = df_results['battery_soc_percent'].max()
        min_soc = df_results['battery_soc_percent'].min()
        avg_soc = df_results['battery_soc_percent'].mean()
        final_soc = df_results['battery_soc_percent'].iloc[-1]
        
        print("\n🔋 BATTERY:")
        print(f"   Max SOC:     {max_soc:.1f}%")
        print(f"   Min SOC:     {min_soc:.1f}%")
        print(f"   Avg SOC:     {avg_soc:.1f}%")
        print(f"   Final SOC:   {final_soc:.1f}%")
        print(f"   Final Energy: {df_results['battery_energy_kwh'].iloc[-1]:.2f} kWh")
        
        total_wasted = df_results['wasted_energy_kw'].sum()
        wasted_hours = (df_results['wasted_energy_kw'] > 0.1).sum()
        wasted_pct = (total_wasted / total_predicted_prod * 100) if total_predicted_prod > 0 else 0
        
        total_blackout_deficit = df_results['blackout_deficit_kw'].sum()
        blackout_hours = df_results['is_blackout'].sum()
        blackout_pct = (blackout_hours / len(df_results) * 100)
        
        print("\n⚠️  OFF-GRID ENERGY LOSSES:")
        print(f"   Wasted energy (battery full): {total_wasted:,.2f} kWh ({wasted_pct:.1f}% of production)")
        print(f"   Hours with waste:             {wasted_hours} hours")
        print(f"   Avg waste per hour:           {total_wasted/wasted_hours if wasted_hours > 0 else 0:.2f} kW")
        
        print("\n🔌 BLACKOUT ANALYSIS:")
        print(f"   Total blackout deficit:  {total_blackout_deficit:,.2f} kWh")
        print(f"   Blackout hours:          {blackout_hours} hours ({blackout_pct:.1f}%)")
        print(f"   Avg deficit per blackout: {total_blackout_deficit/blackout_hours if blackout_hours > 0 else 0:.2f} kW")
        
        action_counts = df_results['decision_action'].value_counts()
        print("\n🧠 DECISIONS BREAKDOWN:")
        for action, count in action_counts.items():
            pct = count / len(df_results) * 100
            print(f"   {action:20s}: {count:5d} times ({pct:5.1f}%)")
        
        bad_weather_count = df_results['bad_weather_ahead'].sum()
        print(f"\n🌦️  WEATHER EVENTS:")
        print(f"   Bad weather detected: {bad_weather_count} hours")
        
        total_notifications = df_results['notifications_count'].sum()
        critical_alerts = df_results['has_critical_alert'].sum()
        print(f"\n🔔 NOTIFICATIONS:")
        print(f"   Total notifications: {total_notifications}")
        print(f"   Critical alerts:     {critical_alerts}")
        
        reliability = ((len(df_results) - blackout_hours) / len(df_results) * 100)
        print(f"\n📈 SYSTEM RELIABILITY:")
        print(f"   Uptime: {reliability:.2f}%")
        print(f"   Downtime: {100 - reliability:.2f}%")
        
        print("\n" + "="*80)
    
    def save_results(self, df_results: pd.DataFrame) -> Path:
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        
        filename = 'phase_2_offgrid_simulation.csv'
        filepath = results_dir / filename
        
        df_results.to_csv(filepath, index=False)
        
        print(f"\n💾 Results saved to: {filepath}")
        print(f"   File size: {filepath.stat().st_size / 1024:.1f} KB")
        print(f"   Rows: {len(df_results):,}")
        
        return filepath


def main():
    print("\n" + "🚀"*40)
    print("\n      SOLAR PV ENERGY MANAGEMENT SYSTEM - PHASE 2 (OFF-GRID)")
    print("\n" + "🚀"*40)
    
    manager = RealtimeEnergyManager(
        config_path='./src/phase_2/system_config.yaml',
        data_path='./data/processed_pv_data_2023_2024.csv',
        production_model_path='models/production_xgboost.pkl',
        consumption_model_path='models/consumption_xgboost.pkl'
    )
    
    results = manager.run_simulation(
        start_date='2024-10-01',
        end_date='2024-12-31',
        save_results=True   
    )
    
    print("\n" + "🎉"*40)
    print("\n          PHASE 2 OFF-GRID SIMULATION COMPLETE!")
    print("\n" + "🎉"*40)
    print("\nCheck the 'results/' folder for detailed CSV output.")
    print("\n")


if __name__ == "__main__":
    main()