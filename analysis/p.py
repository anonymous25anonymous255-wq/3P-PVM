# Import required libraries
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Load simulation results
df = pd.read_csv('results/phase_2_offgrid_simulation.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Set timestamp as index for time series analysis
df.set_index('timestamp', inplace=True)
print("Data loaded successfully:")
print(f"Period: {df.index.min()} to {df.index.max()}")
print(f"Total hours: {len(df):,}")
print(f"Total days: {len(df)/24:.1f}")


# Create the combined visualization
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add battery SOC
fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df['battery_soc_percent'],
        name="Battery SOC (%)",
        line=dict(color='blue', width=2),
        hovertemplate="SOC: %{y:.1f}%<br>Time: %{x}",
    ),
    secondary_y=True,
)

# Add predicted production
fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df['predicted_production_kw'],
        name="Solar Production (kW)",
        line=dict(color='orange', width=2),
        hovertemplate="Production: %{y:.2f} kW<br>Time: %{x}",
    ),
    secondary_y=False,
)

# Add predicted consumption
fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df['predicted_consumption_kw'],
        name="Load Consumption (kW)",
        line=dict(color='red', width=2),
        hovertemplate="Consumption: %{y:.2f} kW<br>Time: %{x}",
    ),
    secondary_y=False,
)

# Add battery operating range indicators
fig.add_hline(y=80, line=dict(color="rgba(0,0,255,0.3)", width=1, dash="dash"), secondary_y=True)
fig.add_hline(y=20, line=dict(color="rgba(0,0,255,0.3)", width=1, dash="dash"), secondary_y=True)

# Update layout
fig.update_layout(
    title="Energy System Performance Overview",
    height=800,
    hovermode="x unified",
    showlegend=True,
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
        bgcolor="rgba(255, 255, 255, 0.8)"
    )
)

# Update axes titles
fig.update_yaxes(title_text="Power (kW)", secondary_y=False)
fig.update_yaxes(title_text="Battery SOC (%)", secondary_y=True)
fig.update_xaxes(title_text="Time")

# Show plot
fig.show()