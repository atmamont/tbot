# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement

# --- Do not remove these libs ---
from functools import reduce
from typing import Any, Callable, Dict, List

import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from skopt.space import Categorical, Dimension, Integer, Real  # noqa

from freqtrade.optimize.hyperopt_interface import IHyperOpt

# --------------------------------
# Add your lib to import here
import talib.abstract as ta  # noqa
import freqtrade.vendor.qtpylib.indicators as qtpylib

def bollinger_bands(stock_price, window_size, num_of_std):
    rolling_mean = stock_price.rolling(window=window_size).mean()
    rolling_std = stock_price.rolling(window=window_size).std()
    lower_band = rolling_mean - (rolling_std * num_of_std)
    return np.nan_to_num(rolling_mean), np.nan_to_num(lower_band)

class CombinedBinHAndClucHyperopt(IHyperOpt):

    @staticmethod
    def populate_indicators(dataframe: DataFrame, metadata: dict) -> DataFrame:
        mid, lower = bollinger_bands(dataframe['close'], window_size=40, num_of_std=2)
        dataframe['lower'] = lower
        dataframe['bbdelta'] = (mid - dataframe['lower']).abs()
        dataframe['closedelta'] = (dataframe['close'] - dataframe['close'].shift()).abs()
        dataframe['tail'] = (dataframe['close'] - dataframe['low']).abs()
        # strategy ClucMay72018
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['volume_mean_slow'] = dataframe['volume'].rolling(window=30).mean()

        return dataframe

    @staticmethod
    def indicator_space() -> List[Dimension]:
        return [
        ]

    @staticmethod
    def buy_strategy_generator(params: Dict[str, Any]) -> Callable:
        def populate_buy_trend(dataframe: DataFrame, metadata: dict) -> DataFrame:
            dataframe.loc[
                (  # strategy BinHV45
                        dataframe['lower'].shift().gt(0) &
                        dataframe['bbdelta'].gt(dataframe['close'] * 0.008) &
                        dataframe['closedelta'].gt(dataframe['close'] * 0.0175) &
                        dataframe['tail'].lt(dataframe['bbdelta'] * 0.25) &
                        dataframe['close'].lt(dataframe['lower'].shift()) &
                        dataframe['close'].le(dataframe['close'].shift())
                ) |
                (  # strategy ClucMay72018
                        (dataframe['close'] < dataframe['ema_slow']) &
                        (dataframe['close'] < 0.985 * dataframe['bb_lowerband']) &
                        (dataframe['volume'] < (dataframe['volume_mean_slow'].shift(1) * 20))
                ),
                'buy'
            ] = 1
            return dataframe

        return populate_buy_trend

    @staticmethod
    def sell_indicator_space() -> List[Dimension]:
        return [
        ]

    @staticmethod
    def sell_strategy_generator(params: Dict[str, Any]) -> Callable:
        def populate_sell_trend(dataframe: DataFrame, metadata: dict) -> DataFrame:
            dataframe.loc[
                (dataframe['close'] > dataframe['bb_middleband']),
                'sell'
            ] = 1
            return dataframe

        return populate_sell_trend
