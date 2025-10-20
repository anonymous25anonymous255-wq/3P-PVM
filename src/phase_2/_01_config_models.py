"""
Configuration Models - Reads system_config.yaml
Numbered: 01_ (first to use)
"""

from dataclasses import dataclass
from typing import Optional, List
from abc import ABC,abstractmethod
import yaml


@dataclass
class BatteryConfig:
    """Battery specifications and limits"""
    capacity_kwh: float
    max_charge_kw: float
    max_discharge_kw: float
    round_trip_efficiency: float
    self_discharge_per_day: float
    min_soc_percent: float
    max_soc_percent: float
    emergency_min_soc: float
    emergency_max_soc: float
    initial_soc_percent: float
    
    # Optional fields
    enable_temp_derating: bool = False
    optimal_temp_c: float = 25.0
    temp_derating_coeff: float = 0.005
    enable_degradation: bool = False
    degradation_per_cycle: float = 0.0001
    calendar_aging_per_year: float = 0.02
    manufacturer: str = 'Generic'
    model: str = 'Generic Battery'
    warranty_years: int = 10
    install_date: Optional[str] = None
    def get_usable_capacity(self) -> float:
        return self.capacity_kwh * (self.max_soc_percent - self.min_soc_percent) / 100



@dataclass
class GridConfig:
    """Grid connection and pricing"""
    connected: bool
    can_export: bool
    max_import_kw: Optional[float]
    max_export_kw: Optional[float]
    buy_price_kwh: float
    sell_price_kwh: float
    enable_tou_pricing: bool = False
    peak_hours: List[int] = None
    peak_buy_price: Optional[float] = None
    voltage_min: float = 200.0
    voltage_max: float = 240.0
    frequency_nominal: float = 50.0
    def get_buy_price(self, hour: int) -> float:
        if self.enable_tou_pricing and self.peak_hours and hour in self.peak_hours and self.peak_buy_price:
            return self.peak_buy_price
        return self.buy_price_kwh
    
    def get_export_credit(self, kwh_exported: float) -> float:
        if not self.net_metering_enabled:
            return 0.0
        return kwh_exported * self.buy_price_kwh

@dataclass
class ConsumptionConfig:
    """Consumption patterns"""
    avg_daily_kwh: float
    peak_morning_hours: List[int]
    peak_evening_hours: List[int]
    summer_multiplier: float = 1.3
    winter_multiplier: float = 1.1
    major_appliances: Optional[dict] = None
    def get_hourly_average(self) -> float:
        return self.avg_daily_kwh / 24.0


@dataclass
class WeatherStrategyConfig:
    """Weather-based strategy settings"""
    bad_weather_threshold: float
    forecast_horizon_hours: int
    emergency_charge_threshold: float
    emergency_charge_target: float
    strategy_mode: str = 'balanced'


@dataclass
class DecisionConfig:
    """Decision engine thresholds"""
    charge_threshold_kw: float
    discharge_threshold_kw: float
    sell_threshold_kw: float
    buy_threshold_kw: float
    priority_mode: str = 'cost_optimization'
    decision_interval_minutes: int = 60


@dataclass
class NotificationConfig:
    """Notification settings"""
    enable_info: bool = True
    enable_warnings: bool = True
    enable_critical: bool = True
    channels: List[str] = None
    max_notifications_per_hour: int = 10
    chatbot_enabled: bool = False
    chatbot_endpoint: Optional[str] = None


@dataclass
class SystemConfig:
    """Complete system configuration"""
    config_version: str
    system_name: str
    location: str
    battery: BatteryConfig
    grid: GridConfig
    consumption: ConsumptionConfig
    weather_strategy: WeatherStrategyConfig
    decision: DecisionConfig
    notification: NotificationConfig
    enable_logging: bool = True
    log_level: str = 'INFO'
    data_retention_days: int = 365
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'SystemConfig':
        """
        Load configuration from system_config.yaml
        """
        with open(yaml_path, 'r') as f:
            d = yaml.safe_load(f)
        return cls(
            config_version=d.get('config_version', '1.0.0'),
            system_name=d.get('system_name', 'Energy System'),
            location=d.get('location', 'Unknown'),
            battery=BatteryConfig(**d['battery']),
            grid=GridConfig(**d['grid']),
            consumption=ConsumptionConfig(**d['consumption']),
            weather_strategy=WeatherStrategyConfig(**d['weather_strategy']),
            decision=DecisionConfig(**d['decision']),
            notification=NotificationConfig(**d['notification']),
            enable_logging=d.get('enable_logging', True),
            log_level=d.get('log_level', 'INFO'),
            data_retention_days=d.get('data_retention_days', 365))
    
    def __repr__(self):
        return f"SystemConfig(system='{self.system_name}', battery={self.battery.capacity_kwh}kWh)"


# Quick test
if __name__ == "__main__":
    # Test loading config
    config = SystemConfig.from_yaml('./src/phase_2/system_config.yaml')
    print("✅ Configuration loaded successfully!")
    print(f"\nSystem: {config.system_name}")
    print(f"Battery: {config.battery.capacity_kwh} kWh")
    print(f"Location: {config.location}")
    print(f"Daily consumption: {config.consumption.avg_daily_kwh} kWh")