"""
Base strategy class and implementations.
"""
from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Dict, Optional, Tuple, List
from datetime import datetime
import pandas as pd
import numpy as np

from core.types import MarketState, Signal
from config.base_config import StrategyConfig
from strategies.indicators import IndicatorCalculator


class BaseStrategy(ABC):
    """Abstract base class for trading strategies"""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.indicator_calculator = IndicatorCalculator()
        
    @abstractmethod
    def analyze(
        self, 
        market_data: pd.DataFrame,
        current_portfolio: Dict[str, Decimal]
    ) -> Signal:
        """
        Analyze market data and generate trading signal
        
        Args:
            market_data: DataFrame with OHLC and indicator data
            current_portfolio: Current portfolio allocations
            
        Returns:
            Trading signal with target allocations
        """
        pass
    
    @abstractmethod
    def get_market_state(self, indicators: pd.Series) -> MarketState:
        """
        Determine current market state from indicators
        
        Args:
            indicators: Series with current indicator values
            
        Returns:
            Current market state
        """
        pass
    
    def calculate_rebalance_threshold(
        self, 
        volatility: float,
        base_threshold: Decimal
    ) -> Decimal:
        """
        Calculate dynamic rebalance threshold based on volatility
        
        Args:
            volatility: Current market volatility (e.g., ATR%)
            base_threshold: Base rebalance threshold
            
        Returns:
            Adjusted rebalance threshold
        """
        if not self.config.dynamic_threshold:
            return base_threshold
        
        # Scale threshold based on volatility
        volatility_factor = Decimal(str(volatility)) * self.config.volatility_multiplier
        adjusted_threshold = base_threshold + volatility_factor
        
        # Cap at reasonable limits
        return min(adjusted_threshold, Decimal('0.20'))


class TrendFollowingStrategy(BaseStrategy):
    """
    Trend following strategy using EMA crossover with filters
    """
    
    def analyze(
        self, 
        market_data: pd.DataFrame,
        current_portfolio: Dict[str, Decimal]
    ) -> Signal:
        """Generate trading signal based on trend analysis"""
        
        # Get latest data
        current_data = market_data.iloc[-1]
        timestamp = market_data.index[-1]
        
        # Determine market state
        market_state = self.get_market_state(current_data)
        
        # Get target allocations for the market state
        # Fix: Handle both MarketState enum and string keys
        allocations_dict = self.config.allocations

        # Try different key formats
        if market_state in allocations_dict:
            target_allocations = allocations_dict[market_state].allocations
        elif market_state.value in allocations_dict:
            target_allocations = allocations_dict[market_state.value].allocations
        elif str(market_state) in allocations_dict:
            target_allocations = allocations_dict[str(market_state)].allocations
        else:
            # Default allocations if not found
            print(f"Warning: No allocations found for {market_state}, using defaults")
            target_allocations = {
                'BTC': Decimal('0.33'),
                'ETH': Decimal('0.33'),
                'USDT': Decimal('0.34')
            }

        # Calculate confidence score
        confidence = self._calculate_confidence(current_data, market_state)

        # Prepare metadata
        metadata = {
            'ema_short': current_data.get(f'ema_{self.config.indicators.ema_short}', None),
            'ema_long': current_data.get(f'ema_{self.config.indicators.ema_long}', None),
            'adx': current_data.get('adx', None),
            'atr': current_data.get('atr', None),
            'volatility': self._calculate_volatility(market_data)
        }

        return Signal(
            timestamp=timestamp,
            market_state=market_state,
            target_allocations=target_allocations,
            confidence=confidence,
            metadata=metadata
        )

    def get_market_state(self, indicators: pd.Series) -> MarketState:
        """Determine market state from indicators"""

        # Primary trend: EMA crossover
        ema_short = indicators.get(f'ema_{self.config.indicators.ema_short}')
        ema_long = indicators.get(f'ema_{self.config.indicators.ema_long}')

        if pd.isna(ema_short) or pd.isna(ema_long):
            return MarketState.SIDEWAYS

        # Basic trend determination
        if ema_short > ema_long:
            primary_trend = MarketState.BULL
        else:
            primary_trend = MarketState.BEAR

        # Apply filters if enabled
        if 'adx_filter' in self.config.enabled_indicators:
            adx = indicators.get('adx')
            if not pd.isna(adx):
                # Strong trend if ADX above threshold
                if adx < self.config.indicators.adx_threshold:
                    return MarketState.SIDEWAYS

        # Additional confirmation with RSI if enabled
        if 'rsi' in self.config.enabled_indicators:
            rsi = indicators.get('rsi')
            if not pd.isna(rsi):
                # Extreme RSI values might indicate reversal
                if rsi > self.config.indicators.rsi_overbought and primary_trend == MarketState.BULL:
                    return MarketState.SIDEWAYS
                elif rsi < self.config.indicators.rsi_oversold and primary_trend == MarketState.BEAR:
                    return MarketState.SIDEWAYS

        return primary_trend

    def _calculate_confidence(
        self,
        indicators: pd.Series,
        market_state: MarketState
    ) -> float:
        """Calculate confidence score for the signal"""

        confidence = 0.5  # Base confidence

        # EMA spread
        ema_short = indicators.get(f'ema_{self.config.indicators.ema_short}')
        ema_long = indicators.get(f'ema_{self.config.indicators.ema_long}')

        if not pd.isna(ema_short) and not pd.isna(ema_long):
            ema_spread = abs(ema_short - ema_long) / ema_long
            confidence += min(ema_spread * 10, 0.2)  # Max 0.2 boost

        # ADX strength
        adx = indicators.get('adx')
        if not pd.isna(adx):
            if adx > self.config.indicators.adx_threshold:
                confidence += 0.2
            elif adx < 20:
                confidence -= 0.1

        # RSI confirmation
        rsi = indicators.get('rsi')
        if not pd.isna(rsi):
            if market_state == MarketState.BULL and 40 < rsi < 70:
                confidence += 0.1
            elif market_state == MarketState.BEAR and 30 < rsi < 60:
                confidence += 0.1

        return max(0.0, min(1.0, confidence))

    def _calculate_volatility(self, market_data: pd.DataFrame) -> float:
        """Calculate current volatility as ATR percentage"""

        if 'atr' not in market_data.columns or 'close' not in market_data.columns:
            return 0.0

        current_atr = market_data['atr'].iloc[-1]
        current_close = market_data['close'].iloc[-1]

        if pd.isna(current_atr) or pd.isna(current_close) or current_close == 0:
            return 0.0

        return (current_atr / current_close) * 100


class MeanReversionStrategy(BaseStrategy):
    """
    Mean reversion strategy using Bollinger Bands and RSI
    """

    def analyze(
        self,
        market_data: pd.DataFrame,
        current_portfolio: Dict[str, Decimal]
    ) -> Signal:
        """Generate trading signal based on mean reversion"""

        current_data = market_data.iloc[-1]
        timestamp = market_data.index[-1]

        # Determine market state based on mean reversion signals
        market_state = self.get_market_state(current_data)

        # Get target allocations
        # Fix: Handle both MarketState enum and string keys
        allocations_dict = self.config.allocations

        # Try different key formats
        if market_state in allocations_dict:
            target_allocations = allocations_dict[market_state].allocations
        elif market_state.value in allocations_dict:
            target_allocations = allocations_dict[market_state.value].allocations
        elif str(market_state) in allocations_dict:
            target_allocations = allocations_dict[str(market_state)].allocations
        else:
            # Default allocations if not found
            print(f"Warning: No allocations found for {market_state}, using defaults")
            target_allocations = {
                'BTC': Decimal('0.33'),
                'ETH': Decimal('0.33'),
                'USDT': Decimal('0.34')
            }
        
        # Calculate confidence
        confidence = self._calculate_confidence(current_data)
        
        # Metadata
        metadata = {
            'close': current_data.get('close'),
            'bb_upper': current_data.get('bb_upper'),
            'bb_lower': current_data.get('bb_lower'),
            'rsi': current_data.get('rsi'),
            'position': self._get_bb_position(current_data)
        }
        
        return Signal(
            timestamp=timestamp,
            market_state=market_state,
            target_allocations=target_allocations,
            confidence=confidence,
            metadata=metadata
        )
    
    def get_market_state(self, indicators: pd.Series) -> MarketState:
        """Determine market state for mean reversion"""
        
        close = indicators.get('close')
        bb_upper = indicators.get('bb_upper')
        bb_lower = indicators.get('bb_lower')
        rsi = indicators.get('rsi')
        
        if any(pd.isna(x) for x in [close, bb_upper, bb_lower]):
            return MarketState.SIDEWAYS
        
        # Check Bollinger Band position
        bb_position = self._get_bb_position(indicators)
        
        # Combine with RSI for confirmation
        if not pd.isna(rsi):
            if bb_position < 0.2 and rsi < 30:
                # Oversold - expect bounce
                return MarketState.BULL
            elif bb_position > 0.8 and rsi > 70:
                # Overbought - expect pullback
                return MarketState.BEAR
        
        return MarketState.SIDEWAYS
    
    def _get_bb_position(self, indicators: pd.Series) -> float:
        """Get position within Bollinger Bands (0 = lower, 1 = upper)"""
        
        close = indicators.get('close')
        bb_upper = indicators.get('bb_upper')
        bb_lower = indicators.get('bb_lower')
        
        if any(pd.isna(x) for x in [close, bb_upper, bb_lower]):
            return 0.5
        
        if bb_upper == bb_lower:
            return 0.5
        
        position = (close - bb_lower) / (bb_upper - bb_lower)
        return max(0.0, min(1.0, position))
    
    def _calculate_confidence(self, indicators: pd.Series) -> float:
        """Calculate confidence for mean reversion signal"""
        
        confidence = 0.5
        
        # Extreme BB position increases confidence
        bb_position = self._get_bb_position(indicators)
        if bb_position < 0.1 or bb_position > 0.9:
            confidence += 0.3
        elif bb_position < 0.2 or bb_position > 0.8:
            confidence += 0.2
        
        # RSI confirmation
        rsi = indicators.get('rsi')
        if not pd.isna(rsi):
            if rsi < 20 or rsi > 80:
                confidence += 0.2
            elif rsi < 30 or rsi > 70:
                confidence += 0.1
        
        return min(1.0, confidence)