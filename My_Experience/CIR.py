import numpy as np
from arbitragelab.optimal_mean_reversion import CoxIngersollRoss
from My_Experience.data import charge_data, data_resampling, fetch_prices
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
plt.ion()

df_open, df_high, df_low, df_close, df_volume = charge_data (start_date = '2021-01-01', end_date = '2023-01-01') 
df_open_resampled,df_high_resampled,df_low_resampled,df_close_resampled,df_volume_resampled = data_resampling(df_open, df_high, df_low, df_close, df_volume, resampled= 'D')
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


example = CoxIngersollRoss()

example.fit(pair_prices, data_frequency="D", discount_rate=0.00, transaction_cost=0.00, stop_loss=None)

example.B_value

# Creating a class instance
example = CoxIngersollRoss()

# We establish our training sample
delta_t = 1/252
np.random.seed(30)
cir_example =  example.cir_model_simulation(n=1000, theta_given=0.2, mu_given=0.2,
                                            sigma_given=0.3, delta_t_given=delta_t)

# Plotting the generated CIR process
plt.figure(figsize=(12, 7))
plt.plot(cir_example)

# Model fitting
example.fit(cir_example, data_frequency="D", discount_rate=0.05,
            transaction_cost=[0.001, 0.001])

# Solving the optimal stopping problem
b = example.optimal_liquidation_level()

d = example.optimal_entry_level()

d_switch, b_switch = example.optimal_switching_levels()
np.random.seed(30)

cir_test = example.cir_model_simulation(n=1000)


fig = example.cir_plot_levels(cir_test, switching=True)

# Adjusting the size of the plot
fig.set_figheight(7)
fig.set_figwidth(12)


import numpy as np
from arbitragelab.optimal_mean_reversion import CoxIngersollRoss

example = CoxIngersollRoss()

# We establish our training sample
delta_t = 1/252
np.random.seed(30)
cir_example =  example.cir_model_simulation(n=1000, theta_given=0.2, mu_given=0.2,
                                            sigma_given=0.3, delta_t_given=delta_t)
# Model fitting
example.fit(cir_example, data_frequency="D", discount_rate=0.00,
            transaction_cost=[0.000, 0.000])

# You can separately solve optimal stopping
# and optimal switching problems

# Solving the optimal stopping problem
b = example.optimal_liquidation_level()

d = example.optimal_entry_level()

# Solving the optimal switching problem
d_switch, b_switch = example.optimal_switching_levels()

# You can display the results using the plot
fig = example.cir_plot_levels(cir_example, switching=True)

# Or you can view the model statistics
example.cir_description(switching=True)