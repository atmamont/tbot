import numpy as np
import pandas as pd
from pandas import DataFrame
from functools import reduce
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, CategoricalParameter
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class GaussianChannelStrategy(IStrategy):
    INTERFACE_VERSION = 3

    # Strategy parameters
    minimal_roi = {
        "0": 0.1
    }

    stoploss = -0.10
    trailing_stop = False
    timeframe = '1d'
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Hyperopt parameters
    rsi_period = IntParameter(10, 20, default=14, space="buy")
    rsi_threshold = IntParameter(45, 65, default=50, space="buy")
    uptrend_strength = IntParameter(1, 5, default=1, space="buy")
    confirmation_bars = IntParameter(1, 3, default=1, space="buy")
    poles = IntParameter(2, 9, default=4, space="buy")
    sampling_period = IntParameter(100, 200, default=144, space="buy")
    tr_multiplier = DecimalParameter(1.0, 2.0, default=1.414, decimals=3, space="buy")
    use_reduced_lag = CategoricalParameter([True, False], default=False, space="buy")
    use_fast_response = CategoricalParameter([True, False], default=False, space="buy")

    def gaussian_filter(self, source: pd.Series, n_poles: int, period: int) -> pd.Series:
        # Calculate beta and alpha components
        pi = np.pi
        beta = (1 - np.cos(4 * np.arcsin(1) / period)) / (pow(1.414, 2/n_poles) - 1)
        alpha = -beta + np.sqrt(pow(beta, 2) + 2 * beta)
        
        # Initialize the filtered series
        filtered = source.copy()
        
        # Apply the filter n_poles times
        for _ in range(n_poles):
            filtered = filtered.ewm(alpha=alpha).mean()
        
        return filtered

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=self.rsi_period.value)
        
        # Source data (hlc3)
        dataframe['src'] = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3
        
        # True Range
        dataframe['tr'] = ta.TRANGE(dataframe['high'], dataframe['low'], dataframe['close'])
        
        # Calculate lag for reduced lag mode
        lag = (self.sampling_period.value - 1) / (2 * self.poles.value)
        lag = int(lag)
        
        # Prepare source data with lag reduction if enabled
        if self.use_reduced_lag.value:
            dataframe['srcdata'] = dataframe['src'] + (dataframe['src'] - dataframe['src'].shift(lag))
            dataframe['trdata'] = dataframe['tr'] + (dataframe['tr'] - dataframe['tr'].shift(lag))
        else:
            dataframe['srcdata'] = dataframe['src']
            dataframe['trdata'] = dataframe['tr']
        
        # Apply Gaussian filter
        dataframe['filtn'] = self.gaussian_filter(
            dataframe['srcdata'], 
            self.poles.value, 
            self.sampling_period.value
        )
        
        dataframe['filt1'] = self.gaussian_filter(
            dataframe['srcdata'], 
            1, 
            self.sampling_period.value
        )
        
        dataframe['filtntr'] = self.gaussian_filter(
            dataframe['trdata'], 
            self.poles.value, 
            self.sampling_period.value
        )
        
        dataframe['filt1tr'] = self.gaussian_filter(
            dataframe['trdata'], 
            1, 
            self.sampling_period.value
        )
        
        # Apply fast response mode if enabled
        if self.use_fast_response.value:
            dataframe['filt'] = (dataframe['filtn'] + dataframe['filt1']) / 2
            dataframe['filttr'] = (dataframe['filtntr'] + dataframe['filt1tr']) / 2
        else:
            dataframe['filt'] = dataframe['filtn']
            dataframe['filttr'] = dataframe['filtntr']
        
        # Calculate bands
        dataframe['hband'] = dataframe['filt'] + dataframe['filttr'] * self.tr_multiplier.value
        dataframe['lband'] = dataframe['filt'] - dataframe['filttr'] * self.tr_multiplier.value
        
        # Trend direction
        dataframe['uptrend'] = dataframe['filt'] > dataframe['filt'].shift(1)
        
        # Uptrend strength check
        dataframe['strong_uptrend'] = True
        for i in range(self.uptrend_strength.value):
            dataframe['strong_uptrend'] &= dataframe['filt'].shift(i) > dataframe['filt'].shift(i+1)
        
        # Price above band check
        dataframe['price_above_band'] = True
        for i in range(self.confirmation_bars.value):
            dataframe['price_above_band'] &= dataframe['close'].shift(i) > dataframe['hband'].shift(i)
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        
        conditions.append(dataframe['price_above_band'])
        conditions.append(dataframe['rsi'] > self.rsi_threshold.value)
        conditions.append(dataframe['strong_uptrend'])
        
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'enter_long'
            ] = 1
        
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        
        conditions.append(qtpylib.crossed_below(dataframe['close'], dataframe['hband']))
        
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'exit_long'
            ] = 1
        
        return dataframe