from My_Experience.data import charge_data, data_resampling, fetch_prices
import pandas as pd
import numpy as np
import arbitragelab.optimal_mean_reversion as omr
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

plt.ion()


symbol = 'BTC'
df_open, df_high, df_low, df_close, df_volume = charge_data (start_date = '2023-01-01', end_date = '2024-01-01') 
df_open_resampled,df_high_resampled,df_low_resampled,df_close_resampled,df_volume_resampled = data_resampling(df_open, df_high, df_low, df_close, df_volume, resampled= 'H')
#df_open_resampled,df_high_resampled,df_low_resampled,df_close_resampled,df_volume_resampled = resample_for_alpha(df_open, df_high, df_low, df_close, df_volume, offset = 23)
# Fetch prices
pair_prices = pd.DataFrame(index = df_close_resampled.index)
df_btc = fetch_prices('BTC', df_open_resampled, df_high_resampled, df_low_resampled, df_close_resampled, df_volume_resampled)
df_eth = fetch_prices('ETH', df_open_resampled, df_high_resampled, df_low_resampled, df_close_resampled, df_volume_resampled)
pair_prices['ETH'] = (df_eth['Close'])
pair_prices['BTC'] = (df_btc['Close'])
pair_prices.plot()
plt.show()

# Creating a class instance
example = omr.ExponentialOrnsteinUhlenbeck()# You can input the np.array as data

example.fit(pair_prices.loc['2021-01-01':'2021-03-01'], data_frequency="H", discount_rate=0.000,
            transaction_cost=0.00000, stop_loss=None)

example.B_value
example.theta
example.mu
example.sigma_square


df = pd.DataFrame(pair_prices['BTC']/pair_prices['BTC'].iloc[0]-example.B_value/pair_prices['ETH'].iloc[0]*pair_prices['ETH'])
df['theta'] = example.theta

df.plot()


# Solving the optimal stopping problem
b = example.xou_optimal_liquidation_level()

d, a = example.xou_optimal_entry_interval()
self = example
print("Optimal liquidation level:", round(b,5),
      "\nOptimal entry interval:", round(d, 5))

self = example
d_switch, b_switch = example.optimal_switching_levels()

print ("Optimal switching liquidation level:", round(b_switch,5),
       "\nOptimal switching entry interval:[", round(np.exp(example.a_tilde), 5),",",round(d_switch, 5),"]")

# Calculate the optimal liquidation level accounting for stop-loss
#b_L = example.optimal_liquidation_level_stop_loss()

# Calculate the optimal entry interval accounting for stop-loss
#interval_L = example.optimal_entry_interval_stop_loss()

example.description()

# Showcasing the results on the training data (pd.DataFrame)
fig = example.xou_plot_levels(pair_prices, switching=False)

# Adjusting the size of the plot
fig.set_figheight(7)
fig.set_figwidth(12)
fig.show()

h = example.half_life()

print("half-life: ",h)

##############################################################################
import arbitragelab.optimal_mean_reversion as omr
import numpy as np
import matplotlib.pyplot as plt

# Creating a class instance
example = omr.ExponentialOrnsteinUhlenbeck()

# We establish our training sample
delta_t = 1/252
np.random.seed(31)
xou_example =  example.ou_model_simulation(n=1000, theta_given=1, mu_given=0.6,
                                           sigma_given=0.2, delta_t_given=delta_t)
(xou_example).size
# Model fitting
example.fit(xou_example, data_frequency="D", discount_rate=0.05,
            transaction_cost=[0.02, 0.02])

# Solving the optimal stopping problem
b = example.xou_optimal_liquidation_level()

a,d = example.xou_optimal_entry_interval()

print("Optimal liquidation level:", round(b,5),
      "\nOptimal entry interval:[",round(a, 5),",",round(d, 5),"]")

# Solving the optimal switching problem
d_switch, b_switch = example.optimal_switching_levels()

print ("Optimal switching liquidation level:", round(b_switch,5),
       "\nOptimal switching entry interval:[", round(np.exp(example.a_tilde), 5),",",round(d_switch, 5),"]")

np.random.seed(31)
xou_plot_data =  example.xou_model_simulation(n=1000, theta_given=1, mu_given=0.6,
                                              sigma_given=0.2, delta_t_given=delta_t)

# Showcasing the results on the training data (pd.DataFrame)
fig = example.xou_plot_levels(xou_plot_data, switching=True)

# Adjusting the size of the plot
fig.set_figheight(7)
fig.set_figwidth(12)
plt.show()

# Or you can view the model statistics
example.xou_description(switching=True)