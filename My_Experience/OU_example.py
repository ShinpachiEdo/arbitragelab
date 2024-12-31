import arbitragelab.optimal_mean_reversion as omr
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

# Import data from Yahoo finance
data1 =  yf.download("GLD GDX", start="2012-03-25", end="2013-12-09")
data2 =  yf.download("GLD GDX", start="2015-12-10", end="2016-02-20")
data3 =  yf.download("GLD GDX", start="2016-02-21", end="2020-08-20")

# You can use the pd.DataFrame of two asset prices
data_train_dataframe = data1["Adj Close"][["GLD", "GDX"]]

# And also we can create training dataset as an array of two asset prices
data_train = np.array(data1["Adj Close"][["GLD", "GDX"]])

# Create an out-of-sample dataset
data_test_and_retrain = data2["Adj Close"][["GLD", "GDX"]]

data_test_the_retrained = np.array(data3["Adj Close"][["GLD", "GDX"]])

data_train.shape

example = omr.OrnsteinUhlenbeck()

# The parameters can be allocated in an alternative way
example.fit(data_train_dataframe, data_frequency="D", discount_rate=0.05,
            transaction_cost=0.02, stop_loss=0.2)

example.check_fit()

# To calculate the optimal entry of liquidation levels separately
# you need to use following functions


# Calculate the optimal liquidation level
b = example.optimal_liquidation_level()

# Calculate the optimal entry level
d = example.optimal_entry_level()

# Calculate the optimal liquidation level accounting for stop-loss
b_L = example.optimal_liquidation_level_stop_loss()

# Calculate the optimal entry interval accounting for stop-loss
interval_L = example.optimal_entry_interval_stop_loss()

print("b*=",np.round(b, 4),"\nd*=",np.round(d, 4),"\nb_L*=",np.round(b_L, 4),"\n[a_L*,d_L*]=",np.round(interval_L, 4))

example.description()

fig = example.plot_levels(data=data_train_dataframe, stop_loss=False)
fig.set_figheight(15)
fig.set_figwidth(10)
fig.show()