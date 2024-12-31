
import pandas as pd 
import datetime
import os 
from tqdm import tqdm 

def charge_data (path = '/home/shinpachi/Lab/edo-data/edo-data/BTC/', start_date = '2021-01-01', end_date = '2023-01-01'):
    
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    current_date_start = start_date
    current_date_end = start_date + datetime.timedelta(days=1) - datetime.timedelta(seconds=1)

    feature1 = 'Open'
    feature2 = 'High'
    feature3 = 'Low'
    feature4 = 'Close'
    feature5 = 'Volume'
    sample = '1T'

    #coin_list = ['ADA', 'AVAX', 'BCH', 'BTC', 'DOT', 'DOGE', 'ETH', 'LTC', 'XRP', 'SOL']
    coin_list = ['BTC','ETH']#, 'DOGE', 'XRP', 'SOL']
    coin_list = [coin + '-USDT-SWAP' for coin in coin_list]
    df_open = pd.DataFrame({})
    df_high = pd.DataFrame({})
    df_low = pd.DataFrame({})
    df_close = pd.DataFrame({})
    df_volume = pd.DataFrame({})

    for _ in tqdm(range((end_date - start_date).days + 1)):
        file_name1 = os.path.join(path, feature1+'_'+sample+'_'+'Tardis_OHLCV_perpetual_Okex_all_coin_historical' +'_'+current_date_start.strftime("%Y-%m-%d")+'.parquet')
        file_name2 = os.path.join(path, feature2+'_'+sample+'_'+'Tardis_OHLCV_perpetual_Okex_all_coin_historical' +'_'+current_date_start.strftime("%Y-%m-%d")+'.parquet')
        file_name3 = os.path.join(path, feature3+'_'+sample+'_'+'Tardis_OHLCV_perpetual_Okex_all_coin_historical' +'_'+current_date_start.strftime("%Y-%m-%d")+'.parquet')
        file_name4 = os.path.join(path, feature4+'_'+sample+'_'+'Tardis_OHLCV_perpetual_Okex_all_coin_historical' +'_'+current_date_start.strftime("%Y-%m-%d")+'.parquet')
        file_name5 = os.path.join(path, feature5+'_'+sample+'_'+'Tardis_OHLCV_perpetual_Okex_all_coin_historical' +'_'+current_date_start.strftime("%Y-%m-%d")+'.parquet')
        
        df1 = pd.read_parquet(file_name1)
        df2 = pd.read_parquet(file_name2)
        df3 = pd.read_parquet(file_name3)
        df4 = pd.read_parquet(file_name4)
        df5 = pd.read_parquet(file_name5)

        df_open   = pd.concat([df_open,   df1[coin_list]], ignore_index=False)    
        df_high   = pd.concat([df_high,   df2[coin_list]], ignore_index=False)
        df_low    = pd.concat([df_low,    df3[coin_list]], ignore_index=False)    
        df_close  = pd.concat([df_close,  df4[coin_list]], ignore_index=False)
        df_volume = pd.concat([df_volume, df5[coin_list]], ignore_index=False)

        current_date_start += datetime.timedelta(days=1)
        current_date_end += datetime.timedelta(days=1)
    df_open.rename(columns={'BTC-USDT-SWAP': 'btc', 'ETH-USDT-SWAP': 'eth'}, inplace=True)
    df_high.rename(columns={'BTC-USDT-SWAP': 'btc', 'ETH-USDT-SWAP': 'eth'}, inplace=True)
    df_low.rename(columns={'BTC-USDT-SWAP': 'btc', 'ETH-USDT-SWAP': 'eth'}, inplace=True)
    df_close.rename(columns={'BTC-USDT-SWAP': 'btc', 'ETH-USDT-SWAP': 'eth'}, inplace=True)
    df_volume.rename(columns={'BTC-USDT-SWAP': 'btc', 'ETH-USDT-SWAP': 'eth'}, inplace=True)
    return df_open,df_high,df_low,df_close,df_volume

def resample_for_alpha(df_open, df_high, df_low, df_close, df_volume, offset = None):
    
    if offset is None:
        offset = df_open.index[-1].hour
    
    df_open_resampled = df_open.resample('1D', offset = f'{offset}h').first().iloc[1:]
    df_high_resampled = df_high.resample('1D', offset = f'{offset}h').max().iloc[1:]
    df_low_resampled = df_low.resample('1D', offset = f'{offset}h').min().iloc[1:]
    df_close_resampled = df_close.resample('1D', offset = f'{offset}h').last().iloc[1:]
    df_volume_resampled = df_volume.resample('1D', offset = f'{offset}h').sum().iloc[1:]
    
    return df_open_resampled, df_high_resampled, df_low_resampled, df_close_resampled, df_volume_resampled

def data_resampling(df_open, df_high, df_low, df_close, df_volume, resampled= '1h'):
    if resampled is not None :
        df_open_resampled = df_open.resample(resampled).first()
        df_high_resampled  = df_high.resample(resampled).max()
        df_low_resampled  = df_low.resample(resampled).min()
        df_close_resampled  = df_close.resample(resampled).last()
        df_volume_resampled  = df_volume.resample(resampled).sum()



    return df_open_resampled,df_high_resampled,df_low_resampled,df_close_resampled,df_volume_resampled

def fetch_prices(symbol, df_open, df_high, df_low, df_close, df_volume):
    symbol_mod = symbol.lower()
    try:
        # Assuming each DataFrame has the same index and can be joined on this index
        # Create a new DataFrame by joining the separate DataFrames
        prices_df = pd.DataFrame(index=df_open.index)
        prices_df['Open'] = df_open[symbol_mod]
        prices_df['High'] = df_high[symbol_mod]
        prices_df['Low'] = df_low[symbol_mod]
        prices_df['Close'] = df_close[symbol_mod]
        prices_df['Volume'] = df_volume[symbol_mod]
        #prices_df['Twap'] = twap[symbol_mod]
        
        # Assuming the index is already in datetime format and named appropriately
        # If not, you would convert and set it here
        # prices_df.index = pd.to_datetime(prices_df.index)  # Uncomment if the index needs to be converted

        # No need to set the index again if it's already set
        # prices_df.set_index('Date', inplace=True)  # Uncomment if you need to reset the index

        return prices_df
    except Exception as e:
        print(f"Failed to process data for {symbol_mod}, error: {str(e)}")
        return None

