
import numpy as np
import pandas as pd
from pandas import DataFrame
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, CategoricalParameter
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from datetime import datetime
import logging
logger = logging.getLogger(__name__)

class GaussianChannelStrategy(IStrategy):
    INTERFACE_VERSION = 3

    # Force override config values
    MINIMAL_ROI = {'0': float('inf')}
    STOPLOSS = -0.05
    USE_CUSTOM_STOPLOSS = False
    USE_EXIT_SIGNAL = True
    PROCESS_ONLY_NEW_CANDLES = True

    # Strategy settings
    timeframe = '1d'
    startup_candle_count = 20
    position_adjustment_enable = False
    
    # Plot config with TradingView colors
    plot_config = {
        'main_plot': {
            'src': {},
            'filt': {'color': '#0000ff', 'width': 2},
            'hband': {'color': '#00ff00'},
            'lband': {'color': '#ff0000'},
            'fill_area': {
                'fill_color': lambda df: df['isUptrendZone'].map({True: '#00ff0020', False: '#ff000020'})
            }
        },
        'subplots': {
            "RSI": {
                'rsi': {'color': 'blue'},
            }
        }
    }
    
    # Strategy Parameters (matched to TradingView)
    rsi_period = IntParameter(10, 30, default=14, space="buy")
    rsi_threshold = IntParameter(15, 65, default=15, space="buy")
    poles = IntParameter(1, 9, default=5, space="buy")
    sampling_period = IntParameter(2, 200, default=144, space="buy")
    tr_multiplier = DecimalParameter(1.0, 2.0, default=1.414, decimals=3, space="buy")
    use_reduced_lag = CategoricalParameter([True, False], default=False, space="buy")
    use_fast_response = CategoricalParameter([True, False], default=False, space="buy")

    def debug_msg(self, msg: str, *args, **kwargs) -> None:
        logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

    def gaussian_filter_9x(self, _a: float, _s: np.array, _i: int) -> np.array:
        _x = 1 - _a
        result = np.zeros_like(_s, dtype=float)
        
        # Initialize first values using source
        for t in range(min(9, len(_s))):
            result[t] = _s[t]
        
        # Calculate remaining values
        for t in range(9, len(_s)):
            _f = (pow(_a, _i) * _s[t] + 
                 _i * _x * result[t-1])
            
            if _i >= 2:
                _f -= (_i * (_i - 1) / 2) * pow(_x, 2) * result[t-2]
            if _i >= 3:
                _f += (_i * (_i - 1) * (_i - 2) / 6) * pow(_x, 3) * result[t-3]
            if _i >= 4:
                _f -= (_i * (_i - 1) * (_i - 2) * (_i - 3) / 24) * pow(_x, 4) * result[t-4]
            if _i >= 5:
                _f += (_i * (_i - 1) * (_i - 2) * (_i - 3) * (_i - 4) / 120) * pow(_x, 5) * result[t-5]
            
            result[t] = _f
        
        return result

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Source calculation
        dataframe['src'] = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3
        
        # RSI Calculation
        dataframe['rsi'] = ta.RSI(dataframe['src'], timeperiod=self.rsi_period.value)
        
        # Beta and Alpha Components
        beta = (1 - np.cos(4 * np.arcsin(1) / self.sampling_period.value)) / (pow(1.414, 2/self.poles.value) - 1)
        alpha = -beta + np.sqrt(pow(beta, 2) + 2 * beta)

        # Lag calculation
        lag = int((self.sampling_period.value - 1)/(2 * self.poles.value))

        # Calculate True Range
        dataframe['tr'] = ta.TRANGE(dataframe['high'], dataframe['low'], dataframe['close'])

        # Data preparation
        if self.use_reduced_lag.value:
            srcdata = dataframe['src'] + (dataframe['src'] - dataframe['src'].shift(lag))
            trdata = dataframe['tr'] + (dataframe['tr'] - dataframe['tr'].shift(lag))
        else:
            srcdata = dataframe['src']
            trdata = dataframe['tr']

        # Apply Gaussian filter
        filtn = self.gaussian_filter_9x(alpha, srcdata.to_numpy(), self.poles.value)
        filt1 = self.gaussian_filter_9x(alpha, srcdata.to_numpy(), 1)
        filtntr = self.gaussian_filter_9x(alpha, trdata.to_numpy(), self.poles.value)
        filt1tr = self.gaussian_filter_9x(alpha, trdata.to_numpy(), 1)

        # Apply lag reduction if enabled
        if self.use_fast_response.value:
            dataframe['filt'] = (filtn + filt1) / 2
            dataframe['filttr'] = (filtntr + filt1tr) / 2
        else:
            dataframe['filt'] = filtn
            dataframe['filttr'] = filtntr

        # Calculate bands
        dataframe['hband'] = dataframe['filt'] + dataframe['filttr'] * self.tr_multiplier.value
        dataframe['lband'] = dataframe['filt'] - dataframe['filttr'] * self.tr_multiplier.value

        # Trend direction (matches TradingView coloring)
        dataframe['isUptrendZone'] = dataframe['filt'] > dataframe['filt'].shift(1)
        
        # Debug key values
        if len(dataframe) > 0:
            self.debug_msg(f"Last values for {metadata['pair']}:")
            self.debug_msg(f"src: {dataframe['src'].iloc[-1]:.2f}")
            self.debug_msg(f"filt: {dataframe['filt'].iloc[-1]:.2f}")
            self.debug_msg(f"hband: {dataframe['hband'].iloc[-1]:.2f}")
            self.debug_msg(f"rsi: {dataframe['rsi'].iloc[-1]:.2f}")
            self.debug_msg(f"isUptrendZone: {dataframe['isUptrendZone'].iloc[-1]}")

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        entry_conditions = (
            qtpylib.crossed_above(dataframe['src'], dataframe['hband']) &
            dataframe['isUptrendZone'] &
            (dataframe['rsi'] > self.rsi_threshold.value)
        )
        
        dataframe.loc[entry_conditions, 'enter_long'] = 1
        
        # Debug entries with fixed datetime handling
        if entry_conditions.any():
            entry_dates = dataframe.index[entry_conditions]
            self.debug_msg(f"Entry signals for {metadata['pair']}: {[d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d) for d in entry_dates]}")
            for date in entry_dates:
                idx = dataframe.index.get_loc(date)
                self.debug_msg(f"Entry at {date}:")
                self.debug_msg(f"Price: {dataframe['src'].iloc[idx]:.2f}")
                self.debug_msg(f"Upper Band: {dataframe['hband'].iloc[idx]:.2f}")
                self.debug_msg(f"RSI: {dataframe['rsi'].iloc[idx]:.2f}")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        exit_conditions = (
            qtpylib.crossed_below(dataframe['src'], dataframe['hband']) |
            ~dataframe['isUptrendZone'] # & (dataframe['src'] < dataframe['filt']))
        )
        
        dataframe.loc[exit_conditions, 'exit_long'] = 1
        
        # Debug exits with fixed datetime handling
        if exit_conditions.any():
            exit_dates = dataframe.index[exit_conditions]
            self.debug_msg(f"Exit signals for {metadata['pair']}: {[d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d) for d in exit_dates]}")

        return dataframe