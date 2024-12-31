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

def determine_frequency(df):
    time_diffs = df.index.to_series().diff().dropna().value_counts().idxmax()
    return time_diffs

def get_annualization_factor(time_diff):
    # Calculate the number of periods in a year based on the time difference
    if time_diff >= pd.Timedelta(days=1):
        periods_per_year = 365 / time_diff.days
    elif time_diff >= pd.Timedelta(hours=1):
        periods_per_year = 365 * 24 / time_diff.components.hours
    elif time_diff >= pd.Timedelta(minutes=1):
        periods_per_year = 365 * 24 * 60 / time_diff.components.minutes
    else:
        raise ValueError("Unsupported frequency: less than minutely.")
    
    return periods_per_year

def calculate_sharpe_ratio(df, col_name = 'ret_fee'):
    time_diff = determine_frequency(df=df)
    annualization_factor = get_annualization_factor(time_diff)
    
    # Calculate returns
    returns = df[col_name]
    
    # Calculate the mean and standard deviation of returns
    mean_return = returns.mean()
    std_return = returns.std()
    
    # Calculate the Sharpe Ratio
    sharpe_ratio = mean_return / std_return * np.sqrt(annualization_factor)  # Annualized Sharpe Ratio
    
    return sharpe_ratio

def calculate_max_drawdown(df, col_name = 'ret_fee'):
    # Calculate returns
    returns = df[col_name]
    
    # Calculate cumulative returns
    cumulative_return = (1 + returns).cumprod()
    
    # Calculate the running maximum
    running_max = cumulative_return.cummax()
    
    # Calculate drawdown
    drawdown = cumulative_return / running_max - 1
    
    # Calculate the maximum drawdown
    max_drawdown = drawdown.min()
    
    return abs(max_drawdown)

###########################################################################
start_date = '2022-12-01'
end_date = '2024-01-01'
calculation_method='returns'

def load_and_plot_price_differences(start_date, end_date, resample_frequency='h', calculation_method='difference', plot=True):
    """
    Load, resample, and calculate price differences or ratios, with an option to plot.

    Parameters:
    - start_date (str): The start date for the data.
    - end_date (str): The end date for the data.
    - resample_frequency (str): The frequency for resampling the data.
    - calculation_method (str): Method to calculate price differences, either 'difference' or 'ratio'.
    - plot (bool): Whether to plot the price differences/ratios. Default is True.

    Returns:
    - pair_prices (DataFrame): The DataFrame containing BTC and ETH price differences or ratios.
    - df_btc (DataFrame): The resampled BTC price data.
    - df_eth (DataFrame): The resampled ETH price data.
    """
    # Step 1: Load and resample data
    df_open, df_high, df_low, df_close, df_volume = charge_data(start_date=start_date, end_date=end_date)
    df_open_resampled, df_high_resampled, df_low_resampled, df_close_resampled, df_volume_resampled = data_resampling(
        df_open, df_high, df_low, df_close, df_volume, resampled=resample_frequency
    )

    # Step 2: Fetch prices for BTC and ETH, and calculate price differences or ratios
    pair_prices = pd.DataFrame(index=df_close_resampled.index)
    df_btc = fetch_prices('BTC', df_open_resampled, df_high_resampled, df_low_resampled, df_close_resampled, df_volume_resampled)
    df_eth = fetch_prices('ETH', df_open_resampled, df_high_resampled, df_low_resampled, df_close_resampled, df_volume_resampled)

    if calculation_method == 'difference':
        pair_prices['BTC'] = df_btc['Close'].diff().fillna(0)
        pair_prices['ETH'] = df_eth['Close'].diff().fillna(0)
    elif calculation_method == 'returns':
        pair_prices['BTC'] = df_btc['Close'].diff().fillna(0)
        pair_prices['ETH'] = df_eth['Close'].diff().fillna(0)
    elif calculation_method == 'ratio':
        pair_prices['BTC'] = df_btc['Close'] / df_btc['Close']
        pair_prices['ETH'] = df_eth['Close'] / df_eth['Close']
    else:
        raise ValueError("calculation_method must be either 'difference' or 'ratio'")

    pair_prices.index.names = ['Date']

    # Step 3: Optionally plot the price differences or ratios
    if plot:
        pair_prices.plot()
        plt.xlabel('Date')
        plt.ylabel('Price Difference' if calculation_method == 'difference' else 'Price Ratio')
        plt.title(f'BTC and ETH Price {"Differences" if calculation_method == "difference" else "Ratios"}')
        plt.show()

    return pair_prices, df_btc, df_eth

pair_prices, df_btc, df_eth = load_and_plot_price_differences(start_date=start_date, end_date=end_date, resample_frequency='h', calculation_method=calculation_method, plot=True)

def simulate_copula_trading_strategy(pair_prices, start_date, end_date, fit_weeks=4, test_weeks=4, plot = True):
    # Initialize the trading rule
    weeks = pd.date_range(start=start_date, end=end_date, freq='W', tz='UTC')

    # Initialize columns in pair_prices to store probabilities
    pair_prices['prob1'] = 0.0
    pair_prices['prob2'] = 0.0

    # Loop through the weeks, fit the copula, and simulate the trading strategy
    for i in range(0, len(weeks) - test_weeks, test_weeks):
        train_start = weeks[i] - pd.DateOffset(weeks=fit_weeks)
        train_end = weeks[i]
        test_start = train_end
        test_end = weeks[min(i + fit_weeks + test_weeks, len(weeks) - 1)]

        prices_train = pair_prices[train_start:train_end]
        prices_test = pair_prices[test_start:test_end]

        fit_result_t, copula_t, cdf_x_t, cdf_y_t = ccalc.fit_copula_to_empirical_data(
            x=prices_train['BTC'], y=prices_train['ETH'], copula=StudentCopula
        )

        # Print fit scores and copula description
        print(fit_result_t)
        print(copula_t.describe(), '\n')

        # Instantiate and configure the trading strategy
        BCTR_t = BasicCopulaTradingRule(exit_rule='or', open_probabilities=(0.3, 0.7), exit_probabilities=(0.4, 0.6))
        BCTR_t.set_copula(copula_t)
        BCTR_t.set_cdf(cdf_x_t, cdf_y_t)
        BCTR_t.current_probabilities = (0.5, 0.5)

        # Simulate trading process on test data
        for time, values in prices_test.iterrows():
            x_price = values['BTC']
            y_price = values['ETH']

            # Update probabilities with the current prices
            BCTR_t.update_probabilities(x_price, y_price)

            # Store the updated probabilities in pair_prices
            pair_prices.at[time, 'prob1'] = BCTR_t.current_probabilities[0]
            pair_prices.at[time, 'prob2'] = BCTR_t.current_probabilities[1]
            
    if plot:
        plt.figure(figsize=(12, 6))
        pair_prices[['prob1', 'prob2']].plot()
        plt.xlabel('Time')
        plt.ylabel('Probability')
        plt.title('Probabilities Over Time')
        plt.legend()
        plt.show()

    return pair_prices

pair_prices = simulate_copula_trading_strategy(
    pair_prices=pair_prices,
    start_date=start_date,
    end_date=end_date,
    fit_weeks=4,
    test_weeks=4
)

def calculate_positions_based_on_thresholds(pair_prices, threshold_upper=0.90, threshold_lower=0.10):
    """
    Calculate trading positions based on probability thresholds.

    Parameters:
    - pair_prices (DataFrame): A DataFrame containing the probabilities 'prob1' and 'prob2'.
    - threshold_upper (float): The upper threshold for entering a long position. Default is 0.90.
    - threshold_lower (float): The lower threshold for entering a short position. Default is 0.10.

    Returns:
    - pair_prices (DataFrame): The updated DataFrame with a new 'Position' column.
    """
    # Initialize the Position column
    pair_prices['Position'] = 0

    # Define positions based on thresholds
    pair_prices.loc[(pair_prices['prob1'] > threshold_upper) & (pair_prices['prob2'] < threshold_lower), 'Position'] = 1
    pair_prices.loc[(pair_prices['prob1'] < threshold_lower) & (pair_prices['prob2'] > threshold_upper), 'Position'] = -1

    return pair_prices

pair_prices = calculate_positions_based_on_thresholds(pair_prices, threshold_upper=0.90, threshold_lower=0.10)

def calculate_and_plot_returns_with_fees(pair_prices, df_btc, df_eth):
    # Step 7: Calculate returns and fees
    pair_prices['Shifted_Position'] = 0.5 * pair_prices['Position'].shift(1).fillna(0)
    pair_prices['ret_BTC'] = df_btc['Close'].pct_change().fillna(0)
    pair_prices['ret_ETH'] = df_eth['Close'].pct_change().fillna(0)

    diff_shift = pair_prices['Shifted_Position'].diff().fillna(0)

    pair_prices['Shifted_Position_BTC'] = -pair_prices['prob1'] + 0.5
    pair_prices['Shifted_Position_ETH'] = pair_prices['prob2'] - 0.5

    pair_prices['ret_strat_BTC'] = pair_prices['ret_BTC'] * pair_prices['Shifted_Position']
    pair_prices['ret_strat_ETH'] = -pair_prices['ret_ETH'] * pair_prices['Shifted_Position']
    pair_prices['ret_stratV2_BTC'] = pair_prices['ret_BTC'] * pair_prices['Shifted_Position_BTC']
    pair_prices['ret_stratV2_ETH'] = -pair_prices['ret_ETH'] * pair_prices['Shifted_Position_ETH']

    pair_prices['fee'] = 0.05 / 100 * np.abs(diff_shift).fillna(0)
    pair_prices['ret_strat_BTC_fee'] = pair_prices['ret_strat_BTC'] - pair_prices['fee']
    pair_prices['ret_strat_ETH_fee'] = pair_prices['ret_strat_ETH'] - pair_prices['fee']

    # Step 8: Plot the cumulative returns with and without fees
    plt.figure(figsize=(12, 6))
    (1 + pair_prices['ret_strat_BTC']).cumprod().plot(label='BTC', color='blue')
    (1 + pair_prices['ret_strat_ETH']).cumprod().plot(label='ETH', color='green')
    (1 + pair_prices['ret_strat_BTC_fee']).cumprod().plot(label='BTC with Fee', color='red')
    (1 + pair_prices['ret_strat_ETH_fee']).cumprod().plot(label='ETH with Fee', color='orange')
    pair_prices['ret_fee'] = pair_prices['ret_strat_ETH_fee'] + pair_prices['ret_strat_BTC_fee']
    pair_prices['ret'] = pair_prices['ret_strat_ETH'] + pair_prices['ret_strat_BTC']
    (1 + pair_prices['ret_fee']).cumprod().plot(label='Strategy with Fee', color='purple')
    (1 + pair_prices['ret']).cumprod().plot(label='Strategy without Fee', color='brown')

    plt.xlabel('Time')
    plt.ylabel('Cumulative Returns')
    plt.title('Cumulative Returns of Strategy with and without Fees')
    plt.legend()
    plt.show()

calculate_and_plot_returns_with_fees(pair_prices, df_btc, df_eth)

calculate_sharpe_ratio(pair_prices, col_name = 'ret')