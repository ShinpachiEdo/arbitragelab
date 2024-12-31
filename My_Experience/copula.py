import pandas as pd
import numpy as np
import datetime as dt  # For plotting x-axis as dates
import matplotlib.pyplot as plt
import statsmodels.api as sm

from arbitragelab.trading import BasicCopulaTradingRule
import arbitragelab.copula_approach.copula_calculation as ccalc
from arbitragelab.copula_approach.archimedean import (Gumbel, Clayton, Frank, Joe, N13, N14)
from arbitragelab.copula_approach.elliptical import (StudentCopula, GaussianCopula)
from My_Experience.data import charge_data, data_resampling, fetch_prices

import arbitragelab.optimal_mean_reversion as omr
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
plt.ion()


symbol = 'BTC'
df_open, df_high, df_low, df_close, df_volume = charge_data (start_date = '2020-01-01', end_date = '2024-01-01') 
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
len(pair_prices)

BCS = BasicCopulaTradingRule(exit_rule='and', open_probabilities=(0.5, 0.95),
                                exit_probabilities=(0.9, 0.5))

# Training and testing split
training_length = 1150 # From 01/02/2009 to 12/30/2011 (m/d/y)

prices_train = pair_prices.iloc[: training_length]
prices_test = pair_prices.iloc[training_length :]

# Empirical CDF for the training set.
# This step is only necessary for plotting.
cdf1 = ccalc.construct_ecdf_lin(prices_train['BTC'])
cdf2 = ccalc.construct_ecdf_lin(prices_train['ETH'])

# Fit different copulas, store the results in dictionaries
fit_result_gumbel, copula_gumbel, cdf_x_gumbel, cdf_y_gumbel =\
    ccalc.fit_copula_to_empirical_data(x=prices_train['BTC'], y=prices_train['ETH'], copula=Gumbel)

fit_result_frank, copula_frank, cdf_x_frank, cdf_y_frank =\
    ccalc.fit_copula_to_empirical_data(x=prices_train['BTC'], y=prices_train['ETH'], copula=Frank)

fit_result_clayton, copula_clayton, cdf_x_clayton, cdf_y_clayton =\
    ccalc.fit_copula_to_empirical_data(x=prices_train['BTC'], y=prices_train['ETH'], copula=Clayton)

fit_result_joe, copula_joe, cdf_x_joe, cdf_x_joe=\
    ccalc.fit_copula_to_empirical_data(x=prices_train['BTC'], y=prices_train['ETH'], copula=Joe)

fit_result_n14, copula_n14, cdf_x_n14, cdf_y_n14=\
    ccalc.fit_copula_to_empirical_data(x=prices_train['BTC'], y=prices_train['ETH'], copula=N14)

fit_result_gauss, copula_gauss, cdf_x_gauss, cdf_y_gauss =\
    ccalc.fit_copula_to_empirical_data(x=prices_train['BTC'], y=prices_train['ETH'], copula=GaussianCopula)

fit_result_t, copula_t, cdf_x_t, cdf_y_t=\
    ccalc.fit_copula_to_empirical_data(x=prices_train['BTC'], y=prices_train['ETH'], copula=StudentCopula)
    
# Print all the fit scores
print(fit_result_gumbel)
print(fit_result_frank)
print(fit_result_clayton)
print(fit_result_joe)
print(fit_result_n14)
print(fit_result_gauss)
print(fit_result_t)

# Print copula descriptions
print(copula_t.describe(), '\n')
print(copula_gauss.describe(), '\n')
print(copula_clayton.describe(), '\n')

# Plotting copulas
fig, ax = plt.subplots(figsize=(5,7), dpi=100)

ax.scatter(prices_train['BTC'].apply(cdf1), prices_train['ETH'].apply(cdf2), s=1)
ax.set_aspect('equal', adjustable='box')
ax.set_title(r'Empirical from BTC and ETH, Training Data')
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
plt.tight_layout()

copula_n14.plot_scatter(num_points=training_length)

copula_t.plot_scatter(num_points=training_length);


# Generate Trading Positions
# Using 'and' logic by default.

# ========== Use Student-t Copula ==========
# Instantiate the strategy
BCTR_t = BasicCopulaTradingRule(exit_rule='and', open_probabilities=(0.4, 0.6),
                                exit_probabilities=(0.5, 0.5))
# Adding copula, cdf for X and Y to strategy
BCTR_t.set_copula(copula_t)
BCTR_t.set_cdf(cdf_x_t, cdf_y_t)
# Setting initial probabilities
BCTR_t.current_probabilities = (0.5, 0.5)

# ========== Use Gaussian Copula ==========
# Instantiate the strategy
BCTR_gauss = BasicCopulaTradingRule(exit_rule='and', open_probabilities=(0.4, 0.6),
                                    exit_probabilities=(0.5, 0.5))
# Adding copula, cdf for X and Y to strategy
BCTR_gauss.set_copula(copula_gauss)
BCTR_gauss.set_cdf(cdf_x_gauss, cdf_y_gauss)
# Setting initial probabilities
BCTR_gauss.current_probabilities = (0.5, 0.5)

# ========== Use Clayton Copula ==========
# Instantiate the strategy
BCTR_clayton = BasicCopulaTradingRule(exit_rule='and', open_probabilities=(0.4, 0.6),
                                      exit_probabilities=(0.5, 0.5))
# Adding copula, cdf for X and Y to strategy
BCTR_clayton.set_copula(copula_clayton)
BCTR_clayton.set_cdf(cdf_x_clayton, cdf_y_clayton)
# Setting initial probabilities
BCTR_clayton.current_probabilities = (0.5, 0.5)

# ========== Use Student-t Copula with 'or' logic ==========
# Instantiate the strategy
BCTR_t_or = BasicCopulaTradingRule(exit_rule='or', open_probabilities=(0.4, 0.6),
                                   exit_probabilities=(0.5, 0.5))
# Adding copula, cdf for X and Y to strategy
BCTR_t_or.set_copula(copula_t)
BCTR_t_or.set_cdf(cdf_x_t, cdf_y_t)
# Setting initial probabilities
BCTR_t_or.current_probabilities = (0.5, 0.5)

# Simulate trading process on test data for StudentCopula
for time, values in prices_test.iterrows():
    x_price = values['BTC']
    y_price = values['ETH']

    # Adding price values
    BCTR_t.update_probabilities(x_price, y_price)
    
    # Check if it's time to enter a trade
    trade, side = BCTR_t.check_entry_signal()

    # Close previous trades if needed
    BCTR_t.update_trades(update_timestamp=time)

    if trade:  # Open a new trade if needed
        BCTR_t.add_trade(start_timestamp=time, side_prediction=side)

# Finally, check open trades at the end of the simulation
open_trades_t = BCTR_t.open_trades

# And all trades that were opened and closed
closed_trades_t = BCTR_t.closed_trades


# Simulate trading process on test data for StudentCopula
for time, values in prices_test.iterrows():
    x_price = values['BTC']
    y_price = values['ETH']

    # Adding price values
    BCTR_gauss.update_probabilities(x_price, y_price)
    BCTR_clayton.update_probabilities(x_price, y_price)
    BCTR_t_or.update_probabilities(x_price, y_price)
    
    # Check if it's time to enter a trade
    trade_gauss, side_gauss = BCTR_gauss.check_entry_signal()
    trade_clayton, side_clayton = BCTR_clayton.check_entry_signal()
    trade_t_or, side_t_or = BCTR_t_or.check_entry_signal()

    # Close previous trades if needed
    BCTR_gauss.update_trades(update_timestamp=time)
    BCTR_clayton.update_trades(update_timestamp=time)
    BCTR_t_or.update_trades(update_timestamp=time)

    if trade_gauss:  # Open a new trade if needed
        BCTR_gauss.add_trade(start_timestamp=time, side_prediction=side_gauss)
    if trade_clayton:  # Open a new trade if needed
        BCTR_clayton.add_trade(start_timestamp=time, side_prediction=side_clayton)
    if trade_t_or:  # Open a new trade if needed
        BCTR_t_or.add_trade(start_timestamp=time, side_prediction=side_t_or)
        
# Finally, check open trades at the end of the simulation
open_trades_gauss = BCTR_gauss.open_trades
open_trades_clayton = BCTR_clayton.open_trades
open_trades_t_or = BCTR_t_or.open_trades

# And all trades that were opened and closed
closed_trades_gauss = BCTR_gauss.closed_trades
closed_trades_clayton = BCTR_clayton.closed_trades
closed_trades_t_or = BCTR_t_or.closed_trades

# Following a logic of one open trade at a time - using the first trade
open_time = list(closed_trades_gauss.keys())[0]
close_time = list(closed_trades_gauss.values())[0]['t1']
position = list(closed_trades_gauss.values())[0]['side']

# Creating dataframe for viaualization
positions_gauss = pd.DataFrame(0, index=prices_test.index, columns=['Gauss Positions'])
positions_gauss[(open_time < positions_gauss.index) & (positions_gauss.index< close_time)] = position

# Plotting generated positions
fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 0.7]}, figsize=(10,6), dpi=150)
fig.suptitle('Copula Trading Strategy Results')

# Plotting repositioned log prices
axs[0].plot((prices_test['BTC'] / prices_test['BTC'][0]).map(np.log), label='BTC', color='cornflowerblue')
axs[0].plot((prices_test['ETH'] / prices_test['ETH'][0]).map(np.log), label='ETH', color='seagreen')
axs[0].title.set_text('Repositioned Log Prices')
axs[0].legend()
axs[0].grid()

# Plotting position from Gaussian copula strategy
axs[1].plot(positions_gauss , label='Positions', color='darkorange')
axs[1].title.set_text('Positions from Gaussian Copula, AND logic')
axs[1].set_yticks([-1,0,1])

fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # Avoid title overlap
plt.show()

# 1. Calculating weights
w1_star = -prices_test['ETH'][0] / (prices_test['BTC'][0] - prices_test['ETH'][0])
w2_star = w1_star - 1

w1 = w1_star / (w1_star + w2_star)
w2 = w2_star / (w1_star + w2_star)

print('Unnormalized weight: \n\
w1_star={}, \nw2_star={},\n\
Normalized weight:\n\
w1={} \nw2={}'.format(w1_star, w2_star, w1, w2))

# 2. Calculating Portfolio Series and daily P&L
portfolio_prices = w1 * prices_test['BTC'] - w2 * prices_test['ETH']
portfolio_pnl = np.diff(portfolio_prices, prepend=0)

# 3. Plotting portfolio prices
fig, ax = plt.subplots(figsize=(10,3), dpi=150)
ax.plot(portfolio_prices)
ax.title.set_text('Unit Portfolio Value for Pair ("BTC", "ETH") Calcualted From Price Series')
ax.grid()
fig.autofmt_xdate()
plt.show()

# 4. Calculating strategy daily P&L
pnl_gauss = portfolio_pnl * positions_gauss.values.T
equity_gauss = pnl_gauss.cumsum()

fig, ax = plt.subplots(figsize=(10,3), dpi=150)
ax.plot(equity_gauss, '--', color='g',label=r'Gauss AND')
ax.title.set_text('Strategy Performance on Unit Portfolio Calcualted From Daily P&L')
ax.grid()
fig.autofmt_xdate()
ax.legend()
plt.show()