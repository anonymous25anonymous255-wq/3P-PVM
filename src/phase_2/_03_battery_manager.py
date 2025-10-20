"""
Battery Manager - State of Charge tracking and protection

Manages battery charging/discharging with protection logic:
- Normal operation: 20-80% SOC
- Emergency mode: 0-100% SOC (bad weather, grid outage)
- Respects charge/discharge rate limits
- Tracks efficiency losses
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Literal
from enum import Enum


class BatteryMode(Enum):
    """Battery operating modes"""
    NORMAL = "normal"           # 20-80% SOC
    EMERGENCY_CHARGE = "emergency_charge"  # Charge to 100% (bad weather coming)
    EMERGENCY_DISCHARGE = "emergency_discharge"  # Use all capacity (grid outage)
    PROTECTION = "protection"   # Minimal operation (fault detected)


class BatteryManager:
    """
    Manages battery state of charge and operations
    
    Usage:
        battery = BatteryManager(config.battery)
        battery.charge(2.5, duration_hours=1)  # Charge at 2.5 kW for 1 hour
        soc = battery.get_soc()
    """
    
    def __init__(self, battery_config):
        """
        Initialize battery manager
        
        Args:
            battery_config: BatteryConfig from config_models
        """
        self.config = battery_config
        
        # Current state
        self.current_soc_percent = battery_config.initial_soc_percent
        self.current_energy_kwh = (battery_config.initial_soc_percent / 100) * battery_config.capacity_kwh
        self.mode = BatteryMode.NORMAL
        
        # Operation tracking
        self.total_charged_kwh = 0.0
        self.total_discharged_kwh = 0.0
        self.cycle_count = 0.0  # Full equivalent cycles
        self.current_capacity_kwh = battery_config.capacity_kwh  # Degrades over time
        
        # History for logging
        self.history = []
        
        print(f"🔋 Battery Manager initialized")
        print(f"   Capacity: {self.config.capacity_kwh} kWh")
        print(f"   Initial SOC: {self.current_soc_percent:.1f}%")
        print(f"   Operating range: {self.config.min_soc_percent:.0f}-{self.config.max_soc_percent:.0f}%")
    
    def get_soc(self) -> float:
        """Get current state of charge (%)"""
        return self.current_soc_percent
    
    def get_energy(self) -> float:
        """Get current stored energy (kWh)"""
        return self.current_energy_kwh
    
    def get_available_capacity_to_charge(self) -> float:
        """
        Get how much energy can be stored (kWh)
        Respects current operating mode limits
        """
        if self.mode == BatteryMode.EMERGENCY_CHARGE:
            max_soc = self.config.emergency_max_soc
        else:
            max_soc = self.config.max_soc_percent
        
        max_energy = (max_soc / 100) * self.current_capacity_kwh
        available = max_energy - self.current_energy_kwh
        
        return max(0, available)
    
    def get_available_capacity_to_discharge(self) -> float:
        """
        Get how much energy can be extracted (kWh)
        Respects current operating mode limits
        """
        if self.mode == BatteryMode.EMERGENCY_DISCHARGE:
            min_soc = self.config.emergency_min_soc
        else:
            min_soc = self.config.min_soc_percent
        
        min_energy = (min_soc / 100) * self.current_capacity_kwh
        available = self.current_energy_kwh - min_energy
        
        return max(0, available)
    
    def can_charge(self, power_kw: float, duration_hours: float = 1.0) -> bool:
        """
        Check if battery can accept charge
        
        Args:
            power_kw: Charging power (kW)
            duration_hours: Charging duration (hours)
        
        Returns:
            True if charge is possible
        """
        # Check power limit
        if power_kw > self.config.max_charge_kw:
            return False
        
        # Check if we have space
        energy_to_add = power_kw * duration_hours * self.config.round_trip_efficiency
        available = self.get_available_capacity_to_charge()
        
        return energy_to_add <= available
    
    def can_discharge(self, power_kw: float, duration_hours: float = 1.0) -> bool:
        """
        Check if battery can provide discharge
        
        Args:
            power_kw: Discharge power (kW)
            duration_hours: Discharge duration (hours)
        
        Returns:
            True if discharge is possible
        """
        # Check power limit
        if power_kw > self.config.max_discharge_kw:
            return False
        
        # Check if we have energy
        energy_to_remove = power_kw * duration_hours / self.config.round_trip_efficiency
        available = self.get_available_capacity_to_discharge()
        
        return energy_to_remove <= available
    
    def charge(self, power_kw: float, duration_hours: float = 1.0, timestamp: Optional[pd.Timestamp] = None) -> Dict:
        """
        Charge the battery
        
        Args:
            power_kw: Charging power (kW)
            duration_hours: Charging duration (hours)
            timestamp: Current time (for logging)
        
        Returns:
            Dict with charge results
        """
        # Limit to max charge rate
        actual_power = min(power_kw, self.config.max_charge_kw)
        
        # Calculate energy to add (accounting for efficiency)
        energy_before_loss = actual_power * duration_hours
        energy_to_add = energy_before_loss * self.config.round_trip_efficiency
        
        # Limit by available capacity
        available = self.get_available_capacity_to_charge()
        energy_to_add = min(energy_to_add, available)
        
        # Update state
        self.current_energy_kwh += energy_to_add
        self.current_soc_percent = (self.current_energy_kwh / self.current_capacity_kwh) * 100
        
        # Track for cycle counting
        self.total_charged_kwh += energy_to_add
        self._update_cycles()
        
        # Log operation
        result = {
            'action': 'charge',
            'requested_power_kw': power_kw,
            'actual_power_kw': actual_power,
            'duration_hours': duration_hours,
            'energy_added_kwh': energy_to_add,
            'energy_loss_kwh': energy_before_loss - energy_to_add,
            'new_soc_percent': self.current_soc_percent,
            'new_energy_kwh': self.current_energy_kwh,
            'timestamp': timestamp
        }
        
        self.history.append(result)
        
        return result
    
    def discharge(self, power_kw: float, duration_hours: float = 1.0, timestamp: Optional[pd.Timestamp] = None) -> Dict:
        """
        Discharge the battery
        
        Args:
            power_kw: Discharge power (kW)
            duration_hours: Discharge duration (hours)
            timestamp: Current time (for logging)
        
        Returns:
            Dict with discharge results
        """
        # Limit to max discharge rate
        actual_power = min(power_kw, self.config.max_discharge_kw)
        
        # Calculate energy to provide (accounting for efficiency)
        energy_delivered = actual_power * duration_hours
        energy_to_remove = energy_delivered / self.config.round_trip_efficiency
        
        # Limit by available capacity
        available = self.get_available_capacity_to_discharge()
        energy_to_remove = min(energy_to_remove, available)
        energy_delivered = energy_to_remove * self.config.round_trip_efficiency
        
        # Update state
        self.current_energy_kwh -= energy_to_remove
        self.current_soc_percent = (self.current_energy_kwh / self.current_capacity_kwh) * 100
        
        # Track for cycle counting
        self.total_discharged_kwh += energy_to_remove
        self._update_cycles()
        
        # Log operation
        result = {
            'action': 'discharge',
            'requested_power_kw': power_kw,
            'actual_power_kw': actual_power,
            'duration_hours': duration_hours,
            'energy_removed_kwh': energy_to_remove,
            'energy_delivered_kwh': energy_delivered,
            'energy_loss_kwh': energy_to_remove - energy_delivered,
            'new_soc_percent': self.current_soc_percent,
            'new_energy_kwh': self.current_energy_kwh,
            'timestamp': timestamp
        }
        
        self.history.append(result)
        
        return result
    
    def _update_cycles(self):
        """Update cycle count (full equivalent cycles)"""
        # One full cycle = charge + discharge of full capacity
        throughput = self.total_charged_kwh + self.total_discharged_kwh
        self.cycle_count = throughput / (2 * self.config.capacity_kwh)
        
        # Apply degradation if enabled
        if self.config.enable_degradation:
            self._apply_degradation()
    
    def _apply_degradation(self):
        """Apply capacity degradation based on cycles and calendar aging"""
        # Cycle-based degradation
        cycle_loss = self.cycle_count * self.config.degradation_per_cycle
        
        # Calendar aging (simplified - would need install date for real calc)
        # For now, just apply a small factor
        calendar_loss = 0.0  # Would calculate based on time since install
        
        # Total degradation
        total_degradation = cycle_loss + calendar_loss
        
        # Update capacity (minimum 70% of original)
        self.current_capacity_kwh = self.config.capacity_kwh * (1 - total_degradation)
        self.current_capacity_kwh = max(self.current_capacity_kwh, self.config.capacity_kwh * 0.7)
    
    def set_mode(self, mode: BatteryMode, reason: str = ""):
        """
        Change battery operating mode
        
        Args:
            mode: New operating mode
            reason: Reason for mode change (for logging)
        """
        old_mode = self.mode
        self.mode = mode
        
        print(f"🔋 Battery mode changed: {old_mode.value} → {mode.value}")
        if reason:
            print(f"   Reason: {reason}")
        
        # Log mode change
        self.history.append({
            'action': 'mode_change',
            'old_mode': old_mode.value,
            'new_mode': mode.value,
            'reason': reason,
            'soc_percent': self.current_soc_percent
        })
    
    def get_status(self) -> Dict:
        """Get comprehensive battery status"""
        return {
            'soc_percent': self.current_soc_percent,
            'energy_kwh': self.current_energy_kwh,
            'capacity_kwh': self.current_capacity_kwh,
            'mode': self.mode.value,
            'available_to_charge_kwh': self.get_available_capacity_to_charge(),
            'available_to_discharge_kwh': self.get_available_capacity_to_discharge(),
            'total_charged_kwh': self.total_charged_kwh,
            'total_discharged_kwh': self.total_discharged_kwh,
            'cycle_count': self.cycle_count,
            'health_percent': (self.current_capacity_kwh / self.config.capacity_kwh) * 100
        }
    
    def check_protection_limits(self) -> Dict:
        """
        Check if battery is within safe operating limits
        
        Returns:
            Dict with status and any warnings
        """
        warnings = []
        status = 'ok'
        
        # Check SOC limits
        if self.current_soc_percent < self.config.emergency_min_soc:
            warnings.append(f"⚠️ SOC critically low: {self.current_soc_percent:.1f}%")
            status = 'critical'
        elif self.current_soc_percent < self.config.min_soc_percent and self.mode == BatteryMode.NORMAL:
            warnings.append(f"⚠️ SOC below normal minimum: {self.current_soc_percent:.1f}%")
            status = 'warning'
        
        if self.current_soc_percent > self.config.emergency_max_soc:
            warnings.append(f"⚠️ SOC critically high: {self.current_soc_percent:.1f}%")
            status = 'critical'
        elif self.current_soc_percent > self.config.max_soc_percent and self.mode == BatteryMode.NORMAL:
            warnings.append(f"⚠️ SOC above normal maximum: {self.current_soc_percent:.1f}%")
            status = 'warning'
        
        # Check capacity degradation
        health = (self.current_capacity_kwh / self.config.capacity_kwh) * 100
        if health < 80:
            warnings.append(f"⚠️ Battery health degraded: {health:.1f}%")
            if status == 'ok':
                status = 'warning'
        
        return {
            'status': status,
            'warnings': warnings,
            'soc_percent': self.current_soc_percent,
            'health_percent': health
        }
    def execute_action(self, action: str, power_kw: float, duration_hours: float = 1.0, timestamp: Optional[pd.Timestamp] = None) -> Dict:
        """
        Execute a decision engine action
        
        Args:
            action: Action to execute ('charge_battery', 'discharge_battery', 'idle', etc.)
            power_kw: Power level for action (kW)
            duration_hours: Duration (hours)
            timestamp: Current timestamp
            
        Returns:
            Dict with execution result
        """
        if action == 'charge_battery' or action == 'emergency_charge':
            if power_kw > 0:
                result = self.charge(power_kw, duration_hours, timestamp)
                result['success'] = True
                result['energy_change_kwh'] = result['energy_added_kwh']
                return result
            else:
                return {'success': True, 'action': action, 'energy_change_kwh': 0.0, 'message': 'No charge needed'}
        
        elif action == 'discharge_battery':
            if power_kw > 0:
                result = self.discharge(power_kw, duration_hours, timestamp)
                result['success'] = True
                result['energy_change_kwh'] = -result['energy_removed_kwh']
                return result
            else:
                return {'success': True, 'action': action, 'energy_change_kwh': 0.0, 'message': 'No discharge needed'}
        
        else:
            # idle, self_consume, sell_to_grid, buy_from_grid - no battery action needed
            return {
                'success': True,
                'action': action,
                'energy_change_kwh': 0.0,
                'message': f'No battery action for {action}'
            }
    def simulate_self_discharge(self, hours: float):
        """
        Simulate battery self-discharge over time
        
        Args:
            hours: Number of hours elapsed
        """
        if self.config.self_discharge_per_day > 0:
            days = hours / 24
            loss_fraction = self.config.self_discharge_per_day * days
            energy_loss = self.current_energy_kwh * loss_fraction
            
            self.current_energy_kwh -= energy_loss
            self.current_soc_percent = (self.current_energy_kwh / self.current_capacity_kwh) * 100
    
    def get_power_capability(self) -> Dict:
        """
        Get current charge/discharge power capabilities
        
        Returns:
            Dict with max charge/discharge power available
        """
        # Max charge power is limited by:
        # 1. Battery max charge rate
        # 2. Available capacity
        available_charge_kwh = self.get_available_capacity_to_charge()
        # Assume 1 hour charging window for this calculation
        max_charge_power = min(
            self.config.max_charge_kw,
            available_charge_kwh / self.config.round_trip_efficiency
        )
        
        # Max discharge power is limited by:
        # 1. Battery max discharge rate
        # 2. Available energy
        available_discharge_kwh = self.get_available_capacity_to_discharge()
        max_discharge_power = min(
            self.config.max_discharge_kw,
            available_discharge_kwh * self.config.round_trip_efficiency
        )
        
        return {
            'max_charge_kw': max(0, max_charge_power),
            'max_discharge_kw': max(0, max_discharge_power),
            'can_charge': max_charge_power > 0.1,  # Minimum 100W
            'can_discharge': max_discharge_power > 0.1
        }
    
    def reset_to_soc(self, soc_percent: float):
        """
        Reset battery to specific SOC (for testing/simulation)
        
        Args:
            soc_percent: Target SOC (0-100)
        """
        soc_percent = np.clip(soc_percent, 0, 100)
        self.current_soc_percent = soc_percent
        self.current_energy_kwh = (soc_percent / 100) * self.current_capacity_kwh
        
        print(f"🔋 Battery reset to {soc_percent:.1f}% SOC ({self.current_energy_kwh:.2f} kWh)")
    
    def get_history_dataframe(self) -> pd.DataFrame:
        """Get battery operation history as DataFrame"""
        if not self.history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.history)


# ============================================================================
# TESTING
# ============================================================================
if __name__ == "__main__":
    from _01_config_models import BatteryConfig
    
    print("\n" + "="*60)
    print("TESTING BATTERY MANAGER")
    print("="*60)
    
    # Create test battery
    config = BatteryConfig(
        capacity_kwh=10.0,
        max_charge_kw=5.0,
        max_discharge_kw=5.0,
        min_soc_percent=20.0,
        max_soc_percent=80.0,
        initial_soc_percent=50.0,
        round_trip_efficiency=0.9,
        enable_degradation=False,
        degradation_per_cycle=0.0001,
        self_discharge_per_day=0.0001,  # Missing parameter
        emergency_min_soc=0.0,          # Missing parameter
        emergency_max_soc=100.0         # Missing parameter
    )
    
    battery = BatteryManager(config)
    
    # Test 1: Charge operation
    print("\n📊 Test 1: Charging")
    print(f"   Initial SOC: {battery.get_soc():.1f}%")
    result = battery.charge(3.0, duration_hours=1.0)
    print(f"   Charged {result['energy_added_kwh']:.2f} kWh at {result['actual_power_kw']:.1f} kW")
    print(f"   New SOC: {battery.get_soc():.1f}%")
    print(f"   Energy loss: {result['energy_loss_kwh']:.2f} kWh")
    
    # Test 2: Discharge operation
    print("\n📊 Test 2: Discharging")
    print(f"   Current SOC: {battery.get_soc():.1f}%")
    result = battery.discharge(2.0, duration_hours=2.0)
    print(f"   Discharged {result['energy_removed_kwh']:.2f} kWh at {result['actual_power_kw']:.1f} kW")
    print(f"   Energy delivered: {result['energy_delivered_kwh']:.2f} kWh")
    print(f"   New SOC: {battery.get_soc():.1f}%")
    
    # Test 3: Check limits
    print("\n📊 Test 3: Testing capacity limits")
    available_charge = battery.get_available_capacity_to_charge()
    available_discharge = battery.get_available_capacity_to_discharge()
    print(f"   Available to charge: {available_charge:.2f} kWh")
    print(f"   Available to discharge: {available_discharge:.2f} kWh")
    
    # Test 4: Power capability
    print("\n📊 Test 4: Power capabilities")
    capability = battery.get_power_capability()
    print(f"   Max charge power: {capability['max_charge_kw']:.2f} kW")
    print(f"   Max discharge power: {capability['max_discharge_kw']:.2f} kW")
    
    # Test 5: Emergency mode
    print("\n📊 Test 5: Emergency mode")
    battery.set_mode(BatteryMode.EMERGENCY_CHARGE, reason="Testing emergency mode")
    print(f"   Available to charge in emergency: {battery.get_available_capacity_to_charge():.2f} kWh")
    
    # Test 6: Protection check
    print("\n📊 Test 6: Protection status")
    status = battery.check_protection_limits()
    print(f"   Status: {status['status']}")
    print(f"   Health: {status['health_percent']:.1f}%")