import pandas as pd
import numpy as np
from typing import Dict, Optional
from enum import Enum


class DecisionAction(Enum):
    CHARGE_BATTERY = "charge_battery"
    DISCHARGE_BATTERY = "discharge_battery"
    SELF_CONSUME = "self_consume"
    IDLE = "idle"
    WASTE_ENERGY = "waste_energy"
    BLACKOUT = "blackout"


class DecisionEngine:
    
    def __init__(self, system_config):
        self.config = system_config
        self.decision_config = system_config.decision
        self.battery_config = system_config.battery
        
        self.critical_soc = 15.0
        self.low_soc = 25.0
        self.max_soc = 95.0
        
        self.bad_weather_clearness_threshold = 0.3
        self.critical_weather_clearness = 0.2
        self.consumption_adjustment_factor = 0.90
        
        self.decision_history = []
        self.total_wasted_kwh = 0.0
        self.total_blackout_hours = 0
    
    def decide(self,
               production_kw: float,
               consumption_kw: float,
               battery_soc: float,
               weather_forecast: pd.DataFrame,
               timestamp: Optional[pd.Timestamp] = None,
               daily_strategy: Optional[dict] = None) -> Dict:
        
        weather_status = self._analyze_weather_forecast(weather_forecast)
        adjusted_consumption_kw = consumption_kw * self.consumption_adjustment_factor
        net_power_kw = production_kw - adjusted_consumption_kw
        
        decision = {
            'timestamp': timestamp,
            'production_kw': production_kw,
            'consumption_kw': consumption_kw,
            'adjusted_consumption_kw': adjusted_consumption_kw,
            'net_power_kw': net_power_kw,
            'battery_soc': battery_soc,
            'action': 'idle',
            'power_kw': 0.0,
            'reason': '',
            'wasted_energy_kw': 0.0,
            'blackout_deficit_kw': 0.0,
            'weather_status': weather_status['status'],
            'bad_weather_detected': weather_status['bad_weather_detected'],
            'weather_severity': weather_status['severity']
        }
        
        # Use daily strategy if available
        if daily_strategy and timestamp:
            hour = timestamp.hour
            soc_min = daily_strategy['soc_min']
            soc_max = daily_strategy['soc_max']
            hourly_net_power = daily_strategy['hourly_net_power']
            
            # Look ahead at the next few hours' net power
            upcoming_balance = sum(hourly_net_power[hour:min(hour + 4, 24)])
            
            if net_power_kw > 0.1:  # We have surplus power
                if upcoming_balance < 0:  # We'll need power soon
                    # Charge more aggressively
                    return self._handle_surplus(net_power_kw, battery_soc, decision, 
                                             max_soc=soc_max)
                else:
                    # Normal charging within daily limits
                    return self._handle_surplus(net_power_kw, battery_soc, decision, 
                                             max_soc=min(80, soc_max))
            elif net_power_kw < -0.1:  # We have a deficit
                if upcoming_balance > 0:  # We'll have surplus soon
                    # Discharge more aggressively
                    return self._handle_deficit(net_power_kw, battery_soc, weather_status, 
                                             decision, min_soc=soc_min)
                else:
                    # Normal discharging within daily limits
                    return self._handle_deficit(net_power_kw, battery_soc, weather_status, 
                                             decision, min_soc=max(20, soc_min))
        
        if net_power_kw > 0.1:
            return self._handle_surplus(net_power_kw, battery_soc, decision)
        elif net_power_kw < -0.1:
            return self._handle_deficit(net_power_kw, battery_soc, weather_status, decision)
        else:
            decision['action'] = 'self_consume'
            decision['reason'] = 'Production matches consumption'
            return decision
    
    def _handle_surplus(self, surplus_kw: float, battery_soc: float, decision: Dict, max_soc: float = 80.0) -> Dict:
        if battery_soc < max_soc:
            remaining_capacity_pct = max_soc - battery_soc
            remaining_capacity_kwh = self.battery_config.capacity_kwh * (remaining_capacity_pct / 100)
            
            charge_power = min(
                surplus_kw,
                self.battery_config.max_charge_kw,
                remaining_capacity_kwh / 0.9
            )
            
            decision['action'] = 'charge_battery'
            decision['power_kw'] = charge_power
            decision['reason'] = f'Charging battery to {max_soc:.1f}% (current: {battery_soc:.1f}%)'
            return decision
        
        decision['action'] = 'waste_energy'
        decision['power_kw'] = 0.0
        decision['wasted_energy_kw'] = surplus_kw
        decision['reason'] = f'Battery FULL ({battery_soc:.1f}%), wasting {surplus_kw:.2f} kW'
        self.total_wasted_kwh += surplus_kw
        return decision
    
    def _handle_deficit(self, deficit_kw: float, battery_soc: float,
                        weather_status: Dict, decision: Dict, min_soc: float = 20.0) -> Dict:
        deficit_kw = abs(deficit_kw)
        
        should_preserve = (
            weather_status['severity'] == 'critical' and 
            battery_soc < self.low_soc and
            battery_soc > min_soc
        )
        
        if should_preserve:
            decision['action'] = 'blackout'
            decision['power_kw'] = 0.0
            decision['blackout_deficit_kw'] = deficit_kw
            decision['reason'] = f'Critical weather + low battery ({battery_soc:.1f}%), preserving battery'
            self.total_blackout_hours += 1
            return decision
        
        if battery_soc > self.critical_soc:
            available_capacity_pct = battery_soc - self.critical_soc
            available_capacity_kwh = self.battery_config.capacity_kwh * (available_capacity_pct / 100) * 0.9
            
            discharge_power = min(
                deficit_kw,
                self.battery_config.max_discharge_kw,
                available_capacity_kwh
            )
            
            if discharge_power > 0.1:
                decision['action'] = 'discharge_battery'
                decision['power_kw'] = discharge_power
                decision['reason'] = f'Discharging battery (SOC: {battery_soc:.1f}%)'
                
                unmet_deficit = deficit_kw - discharge_power
                if unmet_deficit > 0.1:
                    decision['blackout_deficit_kw'] = unmet_deficit
                    decision['reason'] += f' | Blackout: {unmet_deficit:.2f} kW unmet (battery limit)'
                    self.total_blackout_hours += 1
                
                return decision
        
        decision['action'] = 'blackout'
        decision['power_kw'] = 0.0
        decision['blackout_deficit_kw'] = deficit_kw
        decision['reason'] = f'Battery empty ({battery_soc:.1f}%), blackout: {deficit_kw:.2f} kW'
        self.total_blackout_hours += 1
        return decision
    
    def _analyze_weather_forecast(self, weather_forecast):
        if 'ALLSKY_SFC_SW_DWN' in weather_forecast.columns and 'CLRSKY_SFC_SW_DWN' in weather_forecast.columns:
            clearness = weather_forecast['ALLSKY_SFC_SW_DWN'] / (weather_forecast['CLRSKY_SFC_SW_DWN'] + 0.001)
            avg_clearness = clearness.clip(0, 1).mean()
        else:
            avg_clearness = 0.7
        
        is_bad = avg_clearness < self.bad_weather_clearness_threshold
        
        if avg_clearness < self.critical_weather_clearness:
            severity = 'critical'
        elif avg_clearness < self.bad_weather_clearness_threshold:
            severity = 'warning'
        elif avg_clearness < 0.5:
            severity = 'caution'
        else:
            severity = 'normal'
        
        return {
            'bad_weather_detected': is_bad,
            'severity': severity,
            'status': 'bad' if is_bad else 'good'
        }
