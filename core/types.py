"""
Core type definitions for the crypto portfolio backtesting system.
"""
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Union, TypedDict, NamedTuple
from datetime import datetime
import pandas as pd
import numpy as np


class MarketState(Enum):
    """Market regime states"""
    BULL = 'BULL'
    BEAR = 'BEAR'
    SIDEWAYS = 'SIDEWAYS'


class OrderSide(Enum):
    """Order side types"""
    BUY = 'BUY'
    SELL = 'SELL'


class TimeFrame(Enum):
    """Supported timeframes for data and indicators"""
    M1 = '1m'
    M5 = '5m'
    M15 = '15m'
    M30 = '30m'
    H1 = '1h'
    H4 = '4h'
    D1 = '1d'
    W1 = '1w'


class AssetBalance(TypedDict):
    """Asset balance information"""
    quantity: Decimal
    value_quote: Decimal
    allocation_pct: Decimal


class Portfolio(TypedDict):
    """Portfolio state at a point in time"""
    timestamp: datetime
    total_value: Decimal
    balances: Dict[str, AssetBalance]
    cash: Decimal


class Trade(NamedTuple):
    """Trade execution record"""
    timestamp: datetime
    symbol: str
    side: OrderSide
    quantity: Decimal
    price: Decimal
    fee: Decimal
    value: Decimal


class Signal(NamedTuple):
    """Trading signal from strategy"""
    timestamp: datetime
    market_state: MarketState
    target_allocations: Dict[str, Decimal]
    confidence: float
    metadata: Dict[str, any]


class BacktestResult(TypedDict):
    """Complete backtest result"""
    portfolio_history: pd.DataFrame
    trades: List[Trade]
    signals: List[Signal]
    metrics: Dict[str, float]
    final_portfolio: Portfolio


class OptimizationResult(TypedDict):
    """Optimization result for a parameter set"""
    parameters: Dict[str, Union[int, float, Decimal]]
    backtest_result: BacktestResult
    objective_value: float


class IndicatorConfig(TypedDict):
    """Configuration for a technical indicator"""
    name: str
    params: Dict[str, Union[int, float]]
    enabled: bool


class StrategyConfig(TypedDict):
    """Strategy configuration"""
    name: str
    indicators: List[IndicatorConfig]
    market_state_rules: Dict[str, any]
    rebalance_threshold: Decimal
    allocations: Dict[MarketState, Dict[str, Decimal]]