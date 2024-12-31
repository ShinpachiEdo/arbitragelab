from My_Experience.data import charge_data, data_resampling, fetch_prices
import pandas as pd
import numpy as np
import arbitragelab.optimal_mean_reversion as omr
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
plt.ion()


symbol = 'BTC'
df_open, df_high, df_low, df_close, df_volume = charge_data (start_date = '2021-01-01', end_date = '2023-01-01') 
df_open_resampled,df_high_resampled,df_low_resampled,df_close_resampled,df_volume_resampled = data_resampling(df_open, df_high, df_low, df_close, df_volume, resampled= 'H')
#df_open_resampled,df_high_resampled,df_low_resampled,df_close_resampled,df_volume_resampled = resample_for_alpha(df_open, df_high, df_low, df_close, df_volume, offset = 23)
# Fetch prices
pair_prices = pd.DataFrame(index = df_close_resampled.index)
df_btc = fetch_prices('BTC', df_open_resampled, df_high_resampled, df_low_resampled, df_close_resampled, df_volume_resampled)
df_eth = fetch_prices('ETH', df_open_resampled, df_high_resampled, df_low_resampled, df_close_resampled, df_volume_resampled)
pair_prices['BTC'] = (df_btc['Close'])#.pct_change().cumsum()
pair_prices['ETH'] = (df_eth['Close'])#.pct_change().cumsum()
pair_prices.index.names = ['Date']
pair_prices.plot()
plt.show()

current_start = '2021-01-01'
current_end = '2021-05-01'

example = omr.OrnsteinUhlenbeck()

#You can input the np.array as data 
example.fit(pair_prices.loc['2021-01-01':'2021-05-01'], start='2021-01-01', end= '2021-05-01', data_frequency="H", discount_rate=0.00, transaction_cost=0.00, stop_loss=None)


start_date = datetime.strptime("2021-01-01", "%Y-%m-%d")
end_date = datetime.strptime("2023-01-01", "%Y-%m-%d")

# Ensure pair_prices index is timezone-naive
if pair_prices.index.tzinfo is not None:
    pair_prices.index = pair_prices.index.tz_convert(None)

current_start = start_date

while current_start < end_date:
    current_end = current_start + relativedelta(months=3)
    if current_end > end_date:
        current_end = end_date

    print(current_start, current_end)

    example = omr.OrnsteinUhlenbeck()
    
    # Fit the model for the current four-month period
    example.fit(pair_prices.loc[current_start.strftime("%Y-%m-%d"):current_end.strftime("%Y-%m-%d")], 
                start=current_start.strftime("%Y-%m-%d"), end=current_end.strftime("%Y-%m-%d"),
                data_frequency="H", 
                discount_rate=0.00, 
                transaction_cost=0.00, 
                stop_loss=None)

    #df = pd.DataFrame((pair_prices['BTC'].loc[current_start:] / pair_prices['BTC'].loc[current_start:].iloc[0] - example.B_value / pair_prices['ETH'].loc[current_start:].iloc[0] * pair_prices['ETH'].loc[current_start:]), columns=['initial_diff'])

    df = pd.DataFrame(example.portfolio_from_prices(pair_prices.loc[current_start:].transpose().values,example.B_value), 
                      columns=['initial_diff'])
    
    window_size = 30 * 24  # 30 days worth of hourly data
    df['moving_average'] = df['initial_diff'].rolling(window=window_size).mean()

    print(example.theta)
    print(example.B_value)
    df['theta'] = example.theta
    df['upper_band'] = df['theta'] + 0.1 * df['moving_average']
    df['lower_band'] = df['theta'] - 0.1 * df['moving_average']
    
    cutoff_date = current_end.strftime("%Y-%m-%d %H:%M:%S")
    fig, ax = plt.subplots()
    df.loc[:cutoff_date].plot(ax=ax, color='red')
    df.loc[cutoff_date:].plot(ax=ax, color='blue')
    plt.show()
    
    # Move to the next four-month period
    current_start = current_start + timedelta(days=90)
    
    

example.B_value
example.theta
example.mu
example.sigma_square
plt.plot(example.portfolio_from_prices(pair_prices.transpose().values,example.B_value))


current_start = datetime.strptime("2021-01-01", "%Y-%m-%d")
df = pd.DataFrame((pair_prices['BTC'].loc[current_start:] / pair_prices['BTC'].loc[current_start:].iloc[0] - example.B_value / pair_prices['ETH'].loc[current_start:].iloc[0] * pair_prices['ETH'].loc[current_start:]), columns=['initial_diff'])
df['theta'] = example.theta
df.plot()





a= example.check_fit()
a
# Calculate the optimal liquidation level
b = example.optimal_liquidation_level()

# Calculate the optimal entry level
d = example.optimal_entry_level()
self = example
# Calculate the optimal liquidation level accounting for stop-loss
#b_L = example.optimal_liquidation_level_stop_loss()

# Calculate the optimal entry interval accounting for stop-loss
#interval_L = example.optimal_entry_interval_stop_loss()

example.description()
b = example.optimal_liquidation_level()

# Calculate the optimal entry level
d = example.optimal_entry_level()
# Showcasing the results on the training data (pd.DataFrame)
fig = example.plot_levels(data=pair_prices, stop_loss=False)
fig.set_figheight(15)
fig.set_figwidth(10)
fig.show()


h = example.half_life()

print("half-life: ",h)



if False :
    (pair_prices['BTC']/pair_prices['BTC'].iloc[0]-self.B_value/pair_prices['ETH'].iloc[0]*pair_prices['ETH']).plot()


# Define the start and end dates for the data
start_date = datetime.strptime("2021-01-01", "%Y-%m-%d")
end_date = datetime.strptime("2023-01-01", "%Y-%m-%d")

B_values = []
theta_values = []
plt.figure()

# Ensure pair_prices index is timezone-naive
if pair_prices.index.tzinfo is not None:
    pair_prices.index = pair_prices.index.tz_convert(None)

# Loop through each four-month interval
current_start = start_date
while current_start < end_date:
    current_end = current_start + relativedelta(months=3)
    if current_end > end_date:
        current_end = end_date
    example = omr.OrnsteinUhlenbeck()

    # Fit the model for the current four-month period
    example.fit(pair_prices.loc[current_start:current_end], start=current_start.strftime("%Y-%m-%d"), end=current_end.strftime("%Y-%m-%d"),
                data_frequency="D", discount_rate=0.000, transaction_cost=0.00000, stop_loss=None)

    # Store the B_value
    B_values.append(example.B_value)
    theta_values.append(example.theta)
    # Move to the next four-month period
    current_start = current_start + timedelta(days=10)
    print(current_start)
    plt.plot((pair_prices['BTC'].loc[current_start:current_end]/pair_prices['BTC'].iloc[0]-example.B_value/pair_prices['ETH'].iloc[0]*pair_prices['ETH'].loc[current_start:current_end]))

# Generate labels for the x-axis
intervals = pd.date_range(start=start_date, end=end_date, freq='10d').strftime("%Y-%m-%d").tolist()

# Plot the B_values
plt.figure()
plt.plot(intervals[:-1], B_values, marker='o')
plt.xlabel('Interval Start Month')
plt.ylabel('B_value')
plt.title('B_value over Time in Three-Month Intervals')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Plot the B_values
plt.figure()
plt.plot(intervals[:-1], theta_values, marker='o')
plt.xlabel('Interval Start Month')
plt.ylabel('theta_value')
plt.title('Theta value over Time in Three-Month Intervals')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()