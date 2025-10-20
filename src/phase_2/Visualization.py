"""
OFF-GRID ENERGY SYSTEM ANALYSIS & VISUALIZATION
================================================
Analyzes the 91-day simulation results with solar production and load consumption
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 8)
plt.rcParams['font.size'] = 10

# ============================================================================
# LOAD DATA
# ============================================================================
print("=" * 80)
print("LOADING SIMULATION RESULTS")
print("=" * 80)

results_path = './results/phase_2_offgrid_simulation.csv'
df = pd.read_csv(results_path)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['day'] = (df.index // 24) + 1
df['date'] = df['timestamp'].dt.date

print(f"✅ Loaded {len(df):,} hours of simulation data")
print(f"   Period: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"   Total days: {df['day'].max()}")
print(f"\n📋 Available columns: {list(df.columns)}")

# ============================================================================
# FIND CORRECT COLUMN NAMES
# ============================================================================
print("\n" + "=" * 80)
print("IDENTIFYING DATA COLUMNS")
print("=" * 80)

pv_col = None
load_col = None

print(f"\n🔍 Searching for correct columns...")
print(f"   All columns: {list(df.columns)}\n")

for col in df.columns:
    col_lower = col.lower()
    if 'actual' in col_lower and 'production' in col_lower:
        pv_col = col
        print(f"   ✅ Found ACTUAL PV column: {pv_col}")
    elif 'actual' in col_lower and 'consumption' in col_lower:
        load_col = col
        print(f"   ✅ Found ACTUAL Load column: {load_col}")

# If we didn't find "actual" columns, look for any production/consumption
if pv_col is None:
    for col in df.columns:
        if 'production' in col.lower() or 'pv' in col.lower():
            pv_col = col
            print(f"   ⚠️ Using PV column: {pv_col}")
            break

if load_col is None:
    for col in df.columns:
        if 'consumption' in col.lower() or 'load' in col.lower():
            load_col = col
            print(f"   ⚠️ Using Load column: {load_col}")
            break

if pv_col is None or load_col is None:
    print(f"   ❌ ERROR: Could not find PV or Load columns!")
    print(f"   Available columns: {list(df.columns)}")
    exit()

# Show sample values
print(f"\n📊 Sample values from data:")
print(f"   {pv_col}: {df[pv_col].head(10).values}")
print(f"   {load_col}: {df[load_col].head(10).values}")
print(f"   {pv_col} sum: {df[pv_col].sum():.2f}")
print(f"   {load_col} sum: {df[load_col].sum():.2f}")

# ============================================================================
# PRINT STATISTICS
# ============================================================================
print(f"\n📊 BATTERY STATISTICS:")
print(f"   Min SOC reached: {df['battery_soc_percent'].min():.1f}%")
print(f"   Max SOC reached: {df['battery_soc_percent'].max():.1f}%")
print(f"   Average SOC: {df['battery_soc_percent'].mean():.1f}%")

print(f"\n☀️ PV PRODUCTION STATISTICS:")
print(f"   Total PV Energy: {df[pv_col].sum():.1f} kWh")
print(f"   Average PV Power: {df[pv_col].mean():.2f} kWh")
print(f"   Peak PV Power: {df[pv_col].max():.2f} kWh")

print(f"\n⚡ LOAD CONSUMPTION STATISTICS:")
print(f"   Total Energy Consumed: {df[load_col].sum():.1f} kWh")
print(f"   Average Load: {df[load_col].mean():.2f} kW")
print(f"   Peak Load: {df[load_col].max():.2f} kW")

print(f"\n🔋 ENERGY BALANCE:")
energy_balance = df[pv_col].sum() - df[load_col].sum()
print(f"   Net Energy Balance: {energy_balance:.1f} kWh")
print(f"   PV Coverage: {(df[pv_col].sum() / df[load_col].sum() * 100):.1f}%")

# ============================================================================
# CALCULATE DAILY STATISTICS
# ============================================================================
print("\n" + "=" * 80)
print("CALCULATING DAILY STATISTICS")
print("=" * 80)

daily_stats = df.groupby('day').agg({
    'battery_soc_percent': ['min', 'max', 'mean'],
    pv_col: 'sum',
    load_col: 'sum'
}).reset_index()
daily_stats.columns = ['day', 'min_soc', 'max_soc', 'avg_soc', 'daily_pv', 'daily_load']

print(f"\n   Days with SOC < 20%: {(daily_stats['min_soc'] < 20).sum()}")
print(f"   Days with negative energy balance: {(daily_stats['daily_pv'] < daily_stats['daily_load']).sum()}")
print(f"   Average daily PV production: {daily_stats['daily_pv'].mean():.1f} kWh")
print(f"   Average daily load consumption: {daily_stats['daily_load'].mean():.1f} kWh")

print(f"\n📊 Scale Info:")
print(f"   Max PV Production: {daily_stats['daily_pv'].max():.2f} kWh/day")
print(f"   Max Load Consumption: {daily_stats['daily_load'].max():.2f} kWh/day")
print(f"   Average PV Production: {daily_stats['daily_pv'].mean():.2f} kWh/day")
print(f"   Average Load Consumption: {daily_stats['daily_load'].mean():.2f} kWh/day")

# ============================================================================
# CREATE FINAL PLOT ONLY
# ============================================================================
print("\n" + "=" * 80)
print("GENERATING FINAL VISUALIZATION")
print("=" * 80)

fig, ax2 = plt.subplots(figsize=(16, 8))

# Create twin axis for energy (right side)
ax2_energy = ax2.twinx()

# Plot SOC range and average on left axis
ax2.fill_between(daily_stats['day'], daily_stats['min_soc'], daily_stats['max_soc'], 
                  alpha=0.3, color='blue', label='Daily SOC Range')
ax2.plot(daily_stats['day'], daily_stats['avg_soc'], linewidth=2.5, color='darkblue', 
         label='Daily Average SOC', zorder=5)
ax2.axhline(y=50, color='gray', linestyle=':', alpha=0.5, label='50% SOC')

# Plot PV production and Load consumption as LINES on right axis
ax2_energy.plot(daily_stats['day'], daily_stats['daily_pv'], linewidth=3, 
                color='#FF8C00', label='PV Production', marker='o', markersize=4, zorder=4)
ax2_energy.plot(daily_stats['day'], daily_stats['daily_load'], linewidth=2.5, 
                color='#E63946', label='Load Consumption', marker='s', markersize=3, zorder=4, alpha=0.7)
ax2_energy.fill_between(daily_stats['day'], 0, daily_stats['daily_pv'], 
                        alpha=0.4, color='#FFB627', zorder=3)
ax2_energy.fill_between(daily_stats['day'], 0, daily_stats['daily_load'], 
                        alpha=0.25, color='#E63946', zorder=2)

# Configure left axis (SOC)
ax2.set_xlabel('Day', fontsize=12, fontweight='bold')
ax2.set_ylabel('Battery SOC (%)', fontsize=12, fontweight='bold', color='darkblue')
ax2.tick_params(axis='y', labelcolor='darkblue')
ax2.set_xlim(1, 91)
ax2.set_ylim(0, 100)
ax2.grid(True, alpha=0.3, zorder=0)

# Configure right axis (Energy)
max_load = daily_stats['daily_load'].max()
ax2_energy.set_ylabel('Energy (kWh/day)', fontsize=12, fontweight='bold')
ax2_energy.set_ylim(0, max_load * 1.15)

# Title
ax2.set_title('Daily Battery SOC & Energy Production/Consumption', fontsize=14, fontweight='bold')

# Combine legends from both axes
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_energy.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

plt.tight_layout()
print("\n✅ Plot ready to display")
plt.show()