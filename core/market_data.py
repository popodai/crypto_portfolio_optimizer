"""
Market data fetching and management with caching support.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import yfinance as yf
from functools import lru_cache
import pickle
import os
from pathlib import Path

from core.types import TimeFrame
from config.base_config import DataConfig
from strategies.indicators import IndicatorCalculator


class MarketDataManager:
    """Manages market data fetching, caching, and preprocessing"""
    
    def __init__(self, config: DataConfig, cache_dir: str = "./cache"):
        self.config = config
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.indicator_calculator = IndicatorCalculator()
        
        # In-memory cache for current session
        self._data_cache: Dict[str, pd.DataFrame] = {}
        
    def get_historical_data(
        self,
        symbols: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        timeframe: Optional[TimeFrame] = None,
        use_cache: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch historical data for multiple symbols
        
        Args:
            symbols: List of symbols to fetch (default: from config)
            start_date: Start date (default: from config)
            end_date: End date (default: from config)
            timeframe: Timeframe (default: from config)
            use_cache: Whether to use cache
            
        Returns:
            Tuple of (price_data, ohlc_data)
        """
        # Use defaults from config if not provided
        symbols = symbols or self.config.assets
        start_date = start_date or self.config.start_date
        end_date = end_date or self.config.end_date
        timeframe = timeframe or self.config.timeframe
        
        # Check cache first
        cache_key = self._get_cache_key(symbols, start_date, end_date, timeframe)
        
        if use_cache and cache_key in self._data_cache:
            return self._data_cache[cache_key]
        
        # Try to load from disk cache
        if use_cache:
            cached_data = self._load_from_disk_cache(cache_key)
            if cached_data is not None:
                self._data_cache[cache_key] = cached_data
                return cached_data
        
        # Fetch fresh data
        price_data, ohlc_data = self._fetch_data(
            symbols, start_date, end_date, timeframe
        )
        
        # Cache the data
        if use_cache:
            self._data_cache[cache_key] = (price_data, ohlc_data)
            self._save_to_disk_cache(cache_key, (price_data, ohlc_data))
        
        return price_data, ohlc_data
    
    def _fetch_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: TimeFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch data from Yahoo Finance"""
        
        # Convert symbols to Yahoo Finance format
        yf_symbols = [f"{symbol}-USD" if symbol != self.config.quote_asset else symbol 
                      for symbol in symbols]
        
        # Map timeframe to yfinance interval
        interval_map = {
            TimeFrame.M1: '1m',
            TimeFrame.M5: '5m',
            TimeFrame.M15: '15m',
            TimeFrame.M30: '30m',
            TimeFrame.H1: '1h',
            TimeFrame.H4: '4h',
            TimeFrame.D1: '1d',
            TimeFrame.W1: '1wk'
        }
        
        interval = interval_map.get(timeframe, '1h')
        
        # Fetch data
        try:
            data = yf.download(
                yf_symbols,
                start=start_date,
                end=end_date,
                interval=interval,
                progress=False,
                auto_adjust=True
            )
            
            if data.empty:
                raise ValueError("No data returned from Yahoo Finance")
            
            # Handle single symbol case
            if len(symbols) == 1:
                data.columns = pd.MultiIndex.from_product(
                    [data.columns, [yf_symbols[0]]]
                )
            
            # Extract close prices
            close_prices = data['Close'].copy()
            close_prices.columns = [col.replace('-USD', '') for col in close_prices.columns]
            
            # Get OHLC data for the primary symbol (usually BTC)
            primary_symbol = f"{symbols[0]}-USD"
            if len(symbols) == 1:
                ohlc_data = data.loc[:, pd.IndexSlice[:, primary_symbol]].droplevel(1, axis=1)
            else:
                ohlc_data = data.loc[:, pd.IndexSlice[:, primary_symbol]].droplevel(1, axis=1)
            
            # Rename columns to lowercase
            ohlc_data.columns = [col.lower() for col in ohlc_data.columns]
            ohlc_data = ohlc_data[['open', 'high', 'low', 'close', 'volume']]
            
            # Remove any rows with NaN values
            close_prices = close_prices.dropna()
            ohlc_data = ohlc_data.dropna()
            
            # Ensure data types are float32 for memory efficiency
            close_prices = close_prices.astype(np.float32)
            ohlc_data = ohlc_data.astype(np.float32)
            
            return close_prices, ohlc_data
            
        except Exception as e:
            raise RuntimeError(f"Failed to fetch market data: {str(e)}")
    
    def add_indicators(
        self,
        ohlc_data: pd.DataFrame,
        indicator_config: Dict[str, Dict[str, any]]
    ) -> pd.DataFrame:
        """
        Add technical indicators to OHLC data
        
        Args:
            ohlc_data: DataFrame with OHLC data
            indicator_config: Configuration for indicators
            
        Returns:
            DataFrame with indicators added
        """
        return self.indicator_calculator.calculate_all(ohlc_data, indicator_config)
    
    def resample_data(
        self,
        data: pd.DataFrame,
        source_timeframe: TimeFrame,
        target_timeframe: TimeFrame
    ) -> pd.DataFrame:
        """
        Resample data to a different timeframe
        
        Args:
            data: Original data
            source_timeframe: Current timeframe
            target_timeframe: Desired timeframe
            
        Returns:
            Resampled data
        """
        # Map timeframes to pandas frequencies
        freq_map = {
            TimeFrame.M1: '1T',
            TimeFrame.M5: '5T',
            TimeFrame.M15: '15T',
            TimeFrame.M30: '30T',
            TimeFrame.H1: '1H',
            TimeFrame.H4: '4H',
            TimeFrame.D1: '1D',
            TimeFrame.W1: '1W'
        }
        
        target_freq = freq_map[target_timeframe]
        
        # Resample OHLC data
        resampled = data.resample(target_freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        return resampled
    
    def calculate_returns(
        self,
        price_data: pd.DataFrame,
        method: str = 'simple'
    ) -> pd.DataFrame:
        """
        Calculate returns from price data
        
        Args:
            price_data: DataFrame with price data
            method: 'simple' or 'log' returns
            
        Returns:
            DataFrame with returns
        """
        if method == 'simple':
            returns = price_data.pct_change()
        elif method == 'log':
            returns = np.log(price_data / price_data.shift(1))
        else:
            raise ValueError(f"Unknown return method: {method}")
        
        return returns.dropna()
    
    def _get_cache_key(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: TimeFrame
    ) -> str:
        """Generate cache key for data"""
        symbols_str = "_".join(sorted(symbols))
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")
        return f"{symbols_str}_{start_str}_{end_str}_{timeframe.value}"
    
    def _load_from_disk_cache(self, cache_key: str) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Load data from disk cache"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                # Cache file corrupted, ignore
                pass
        
        return None
    
    def _save_to_disk_cache(self, cache_key: str, data: Tuple[pd.DataFrame, pd.DataFrame]):
        """Save data to disk cache"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception:
            # Failed to save cache, ignore
            pass
    
    def clear_cache(self):
        """Clear all cached data"""
        self._data_cache.clear()
        
        # Clear disk cache
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                cache_file.unlink()
            except Exception:
                pass