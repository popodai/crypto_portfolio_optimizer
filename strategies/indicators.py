"""
Technical indicators for strategy implementation.
Optimized for performance using vectorized operations.
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple
from functools import lru_cache


class TechnicalIndicators:
    """Collection of technical indicators with caching"""
    
    def __init__(self, cache_size: int = 128):
        self.cache_size = cache_size
        # Remove lru_cache as it doesn't work with pandas Series

    @staticmethod
    def _validate_series(series: pd.Series, min_length: int) -> bool:
        """Validate input series"""
        return len(series) >= min_length and not series.isnull().all()

    def ema(self, series: pd.Series, period: int) -> pd.Series:
        """
        Calculate Exponential Moving Average

        Args:
            series: Price series
            period: EMA period

        Returns:
            EMA series
        """
        if not self._validate_series(series, period):
            return pd.Series(index=series.index, dtype=float)

        return series.ewm(span=period, adjust=False).mean()
    
    def sma(self, series: pd.Series, period: int) -> pd.Series:
        """
        Calculate Simple Moving Average
        
        Args:
            series: Price series
            period: SMA period
            
        Returns:
            SMA series
        """
        if not self._validate_series(series, period):
            return pd.Series(index=series.index, dtype=float)
        
        return series.rolling(window=period).mean()
    
    def rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index
        
        Args:
            series: Price series
            period: RSI period (default: 14)
            
        Returns:
            RSI series
        """
        if not self._validate_series(series, period + 1):
            return pd.Series(index=series.index, dtype=float)
        
        # Calculate price changes
        delta = series.diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Calculate average gains and losses
        avg_gains = gains.rolling(window=period).mean()
        avg_losses = losses.rolling(window=period).mean()
        
        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def macd(
        self, 
        series: pd.Series, 
        fast: int = 12, 
        slow: int = 26, 
        signal: int = 9
    ) -> Dict[str, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Args:
            series: Price series
            fast: Fast EMA period (default: 12)
            slow: Slow EMA period (default: 26)
            signal: Signal line EMA period (default: 9)
            
        Returns:
            Dictionary with 'macd', 'signal', and 'histogram' series
        """
        if not self._validate_series(series, slow + signal):
            empty = pd.Series(index=series.index, dtype=float)
            return {'macd': empty, 'signal': empty, 'histogram': empty}
        
        # Calculate MACD line
        ema_fast = self.ema(series, fast)
        ema_slow = self.ema(series, slow)
        macd_line = ema_fast - ema_slow
        
        # Calculate signal line
        signal_line = self.ema(macd_line, signal)
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def bollinger_bands(
        self, 
        series: pd.Series, 
        period: int = 20, 
        num_std: float = 2
    ) -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands
        
        Args:
            series: Price series
            period: SMA period (default: 20)
            num_std: Number of standard deviations (default: 2)
            
        Returns:
            Dictionary with 'middle', 'upper', and 'lower' bands
        """
        if not self._validate_series(series, period):
            empty = pd.Series(index=series.index, dtype=float)
            return {'middle': empty, 'upper': empty, 'lower': empty}
        
        # Calculate middle band (SMA)
        middle_band = self.sma(series, period)
        
        # Calculate standard deviation
        rolling_std = series.rolling(window=period).std()
        
        # Calculate upper and lower bands
        upper_band = middle_band + (rolling_std * num_std)
        lower_band = middle_band - (rolling_std * num_std)
        
        return {
            'middle': middle_band,
            'upper': upper_band,
            'lower': lower_band
        }
    
    def atr(
        self, 
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series, 
        period: int = 14
    ) -> pd.Series:
        """
        Calculate Average True Range
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: ATR period (default: 14)
            
        Returns:
            ATR series
        """
        if not all(self._validate_series(s, period) for s in [high, low, close]):
            return pd.Series(index=close.index, dtype=float)
        
        # Calculate True Range
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    def adx(
        self, 
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series, 
        period: int = 14
    ) -> Dict[str, pd.Series]:
        """
        Calculate Average Directional Index
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: ADX period (default: 14)
            
        Returns:
            Dictionary with 'adx', 'plus_di', and 'minus_di' series
        """
        if not all(self._validate_series(s, period * 2) for s in [high, low, close]):
            empty = pd.Series(index=close.index, dtype=float)
            return {'adx': empty, 'plus_di': empty, 'minus_di': empty}
        
        # Calculate directional movement
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        # Where both are positive, keep only the larger
        plus_dm[plus_dm < minus_dm] = 0
        minus_dm[minus_dm < plus_dm] = 0
        
        # Calculate ATR
        atr = self.atr(high, low, close, period)
        
        # Calculate directional indicators
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        # Calculate DX
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
        
        # Calculate ADX
        adx = dx.rolling(window=period).mean()
        
        return {
            'adx': adx,
            'plus_di': plus_di,
            'minus_di': minus_di
        }
    
    def stochastic(
        self, 
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3
    ) -> Dict[str, pd.Series]:
        """
        Calculate Stochastic Oscillator
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            k_period: %K period (default: 14)
            d_period: %D period (default: 3)
            
        Returns:
            Dictionary with 'k' and 'd' series
        """
        if not all(self._validate_series(s, k_period) for s in [high, low, close]):
            empty = pd.Series(index=close.index, dtype=float)
            return {'k': empty, 'd': empty}
        
        # Calculate %K
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        
        # Calculate %D
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return {
            'k': k_percent,
            'd': d_percent
        }
    
    def pivot_points(
        self, 
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series
    ) -> Dict[str, pd.Series]:
        """
        Calculate Pivot Points
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            
        Returns:
            Dictionary with pivot levels
        """
        # Calculate pivot point
        pivot = (high + low + close) / 3
        
        # Calculate support and resistance levels
        r1 = 2 * pivot - low
        s1 = 2 * pivot - high
        
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)
        
        r3 = high + 2 * (pivot - low)
        s3 = low - 2 * (high - pivot)
        
        return {
            'pivot': pivot,
            'r1': r1, 'r2': r2, 'r3': r3,
            's1': s1, 's2': s2, 's3': s3
        }


class IndicatorCalculator:
    """
    Wrapper class for calculating indicators with proper data alignment
    """
    
    def __init__(self):
        self.indicators = TechnicalIndicators()
        self._cache = {}
    
    def calculate_all(
        self, 
        ohlc_data: pd.DataFrame,
        indicator_config: Dict[str, Dict[str, any]]
    ) -> pd.DataFrame:
        """
        Calculate all requested indicators
        
        Args:
            ohlc_data: DataFrame with OHLC data
            indicator_config: Configuration for indicators
            
        Returns:
            DataFrame with all calculated indicators
        """
        result = ohlc_data.copy()
        
        # Calculate each configured indicator
        for indicator_name, params in indicator_config.items():
            if not params.get('enabled', True):
                continue
                
            if indicator_name == 'ema':
                for period in params.get('periods', [12, 26]):
                    result[f'ema_{period}'] = self.indicators.ema(
                        ohlc_data['close'], period
                    )
                    
            elif indicator_name == 'sma':
                for period in params.get('periods', [20, 50]):
                    result[f'sma_{period}'] = self.indicators.sma(
                        ohlc_data['close'], period
                    )
                    
            elif indicator_name == 'rsi':
                result['rsi'] = self.indicators.rsi(
                    ohlc_data['close'], 
                    params.get('period', 14)
                )
                
            elif indicator_name == 'macd':
                macd_result = self.indicators.macd(
                    ohlc_data['close'],
                    params.get('fast', 12),
                    params.get('slow', 26),
                    params.get('signal', 9)
                )
                result['macd'] = macd_result['macd']
                result['macd_signal'] = macd_result['signal']
                result['macd_hist'] = macd_result['histogram']
                
            elif indicator_name == 'bollinger':
                bb_result = self.indicators.bollinger_bands(
                    ohlc_data['close'],
                    params.get('period', 20),
                    params.get('num_std', 2)
                )
                result['bb_upper'] = bb_result['upper']
                result['bb_middle'] = bb_result['middle']
                result['bb_lower'] = bb_result['lower']
                
            elif indicator_name == 'atr':
                result['atr'] = self.indicators.atr(
                    ohlc_data['high'],
                    ohlc_data['low'],
                    ohlc_data['close'],
                    params.get('period', 14)
                )
                
            elif indicator_name == 'adx':
                adx_result = self.indicators.adx(
                    ohlc_data['high'],
                    ohlc_data['low'],
                    ohlc_data['close'],
                    params.get('period', 14)
                )
                result['adx'] = adx_result['adx']
                result['plus_di'] = adx_result['plus_di']
                result['minus_di'] = adx_result['minus_di']
                
            elif indicator_name == 'stochastic':
                stoch_result = self.indicators.stochastic(
                    ohlc_data['high'],
                    ohlc_data['low'],
                    ohlc_data['close'],
                    params.get('k_period', 14),
                    params.get('d_period', 3)
                )
                result['stoch_k'] = stoch_result['k']
                result['stoch_d'] = stoch_result['d']
        
        return result