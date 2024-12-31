import pandas as pd
import numpy as np
import datetime as dt  # For plotting x-axis as dates
import matplotlib.pyplot as plt
import statsmodels.api as sm
from data import charge_data, data_resampling, fetch_prices
# Tools to construct and trade spread
from arbitragelab.hedge_ratios import construct_spread
from arbitragelab.trading import BollingerBandsTradingRule
import arbitragelab.copula_approach.copula_calculation as ccalc



#data = pd.read_csv('data.csv', index_col=0, parse_dates=[0])
symbol = 'BTC'
df_open, df_high, df_low, df_close, df_volume = charge_data (start_date = '2020-01-01', end_date = '2022-12-31') 
df_open_resampled,df_high_resampled,df_low_resampled,df_close_resampled,df_volume_resampled = data_resampling(df_open, df_high, df_low, df_close, df_volume, resampled= 'h')
#df_open_resampled,df_high_resampled,df_low_resampled,df_close_resampled,df_volume_resampled = resample_for_alpha(df_open, df_high, df_low, df_close, df_volume, offset = 23)
# Fetch prices
pair_prices = pd.DataFrame(index = df_close_resampled.index)
df_btc = fetch_prices('BTC', df_open_resampled, df_high_resampled, df_low_resampled, df_close_resampled, df_volume_resampled)
df_eth = fetch_prices('ETH', df_open_resampled, df_high_resampled, df_low_resampled, df_close_resampled, df_volume_resampled)
pair_prices['BTC'] = df_btc['Close']
pair_prices['ETH'] = df_btc['Close']


# Importing the module and other libraries
from arbitragelab.copula_approach import fit_copula_to_empirical_data
from arbitragelab.copula_approach.archimedean import (Gumbel, Clayton, Frank, Joe, N13, N14)
from arbitragelab.copula_approach.elliptical import (StudentCopula, GaussianCopula)
from arbitragelab.trading.basic_copula import BasicCopulaTradingRule
import pandas as pd

# Instantiating the module with set open and exit probabilities
# and using the 'AND' exit logic:
cop_trading = BasicCopulaTradingRule(exit_rule='and', open_probabilities=(0.5, 0.95),
                                     exit_probabilities=(0.9, 0.5))



# Split data into train and test sets
prices_train = pair_prices.iloc[:int(len(pair_prices)*0.7)]
prices_test = pair_prices.iloc[int(len(pair_prices)*0.7):]

cdf_x = ccalc.construct_ecdf_lin(prices_train['BTC'])
cdf_y = ccalc.construct_ecdf_lin(prices_train['ETH'])

# Fitting copula to data and getting cdf for X and Y series
info_crit, fit_copula, ecdf_x, ecdf_y = fit_copula_to_empirical_data(x=prices_train['BTC'],
                                                                     y=prices_train['ETH'],
                                                                     copula=GaussianCopula)

# Printing fit scores (AIC, SIC, HQIC, log-likelihood)
print(info_crit)

# Setting initial probabilities
cop_trading.current_probabilities = (0.5, 0.5)
cop_trading.prev_probabilities = (0.5, 0.5)

# Adding copula to strategy
cop_trading.set_copula(fit_copula)

# Adding cdf for X and Y to strategy
cop_trading.set_cdf(cdf_x, cdf_y)

# Trading simulation
for time, values in prices_test.iterrows():
    x_price = values['BTC']
    y_price = values['ETH']

    # Adding price values
    cop_trading.update_probabilities(x_price, y_price)

    # Check if it's time to enter a trade
    trade, side = cop_trading.check_entry_signal()

    # Close previous trades if needed
    cop_trading.update_trades(update_timestamp=time)

    if trade:  # Open a new trade if needed
        cop_trading.add_trade(start_timestamp=time, side_prediction=side)

# Finally, check open trades at the end of the simulation
open_trades = cop_trading.open_trades

# And all trades that were opened and closed
closed_trades = cop_trading.closed_trades