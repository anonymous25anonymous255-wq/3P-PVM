import pandas as pd

df=pd.read_csv('./data/consumption/power_usage_pre_processed.csv')
df['StartDate']=pd.to_datetime(df['StartDate'])
df=df[('2017-01-01'<=df['StartDate']) & (df['StartDate']<='2019-12-31')]
df['StartDateplus']=df['StartDate']+pd.DateOffset(years=6)

#read 2020 data to get Feb 29
df_2020=pd.read_csv('./data/consumption/power_usage_pre_processed.csv')
df_2020['StartDate']=pd.to_datetime(df_2020['StartDate'])
leap_day_2020=df_2020[df_2020['StartDate'].dt.date==pd.to_datetime('2020-02-29').date()].copy()
leap_day_2020['StartDateplus']=leap_day_2020['StartDate']+pd.DateOffset(years=4)

#combine datasets
df=pd.concat([df,leap_day_2020],ignore_index=True)
#sort values by StartDatePlus to keep chronological order
df=df.sort_values('StartDateplus').reset_index(drop=True)
df['day_of_week']=df['StartDateplus'].dt.dayofweek


# Extract datetime components
df['year'] = df['StartDateplus'].dt.year
df['month'] = df['StartDateplus'].dt.month
df['day'] = df['StartDateplus'].dt.day
df['hour'] = df['StartDateplus'].dt.hour

# Reorder columns and keep only the ones you need
df = df[['StartDateplus','year','month','day','hour', 'day_of_week', 'Value (kWh)']]
df=df.rename(columns={'StartDateplus':'Datetime','Value (kWh)':'consumption_kwh'})
# Save to CSV
df.to_csv('./data/consumption/processed_power_usage_2023_2024.csv', index=False)
