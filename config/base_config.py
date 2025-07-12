"""
Base configuration classes using Pydantic for validation.
"""
from decimal import Decimal
from typing import Dict, List, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field, validator
from enum import Enum

from core.types import MarketState, TimeFrame, IndicatorConfig


class AssetConfig(BaseModel):
    """Configuration for a single asset"""
    symbol: str
    min_order_size: Decimal = Decimal('0.001')
    precision: int = 8
    
    class Config:
        arbitrary_types_allowed = True


class DataConfig(BaseModel):
    """Data fetching configuration"""
    assets: List[str] = Field(default=['BTC', 'ETH'])
    quote_asset: str = Field(default='USDT')
    start_date: datetime
    end_date: datetime
    timeframe: TimeFrame = TimeFrame.H4
    data_source: str = Field(default='yfinance')
    
    @validator('end_date')
    def end_after_start(cls, v, values):
        if 'start_date' in values and v <= values['start_date']:
            raise ValueError('end_date must be after start_date')
        return v


class BacktestConfig(BaseModel):
    """Backtesting configuration"""
    initial_capital: Decimal = Field(default=Decimal('10000'))
    transaction_fee: Decimal = Field(default=Decimal('0.001'), ge=0, le=Decimal('0.01'))
    slippage: Decimal = Field(default=Decimal('0.0005'), ge=0, le=Decimal('0.01'))
    
    # DCA settings
    enable_dca: bool = Field(default=True)
    dca_amount: Decimal = Field(default=Decimal('1000'))
    dca_frequency_days: int = Field(default=30)
    
    # Risk management
    max_position_size: Decimal = Field(default=Decimal('0.5'), gt=0, le=1)
    stop_loss_pct: Optional[Decimal] = Field(default=None, ge=0, le=Decimal('0.5'))
    
    class Config:
        arbitrary_types_allowed = True


class IndicatorParams(BaseModel):
    """Parameters for technical indicators"""
    # EMA
    ema_short: int = Field(default=12, ge=1, le=50)
    ema_long: int = Field(default=26, ge=10, le=200)
    
    # ADX
    adx_period: int = Field(default=14, ge=5, le=50)
    adx_threshold: float = Field(default=25.0, ge=10, le=50)
    
    # ATR
    atr_period: int = Field(default=14, ge=5, le=50)
    
    # RSI
    rsi_period: int = Field(default=14, ge=5, le=50)
    rsi_overbought: float = Field(default=70.0, ge=60, le=90)
    rsi_oversold: float = Field(default=30.0, ge=10, le=40)
    
    # MACD
    macd_fast: int = Field(default=12, ge=5, le=50)
    macd_slow: int = Field(default=26, ge=10, le=100)
    macd_signal: int = Field(default=9, ge=3, le=30)


class AllocationSet(BaseModel):
    """Asset allocation for a specific market state"""
    allocations: Dict[str, Decimal] = Field(default_factory=dict)
    
    @validator('allocations')
    def allocations_sum_to_one(cls, v):
        total = sum(v.values())
        if abs(total - Decimal('1.0')) > Decimal('0.001'):
            raise ValueError(f'Allocations must sum to 1.0, got {total}')
        return v
    
    class Config:
        arbitrary_types_allowed = True


class StrategyConfig(BaseModel):
    """Complete strategy configuration"""
    name: str = Field(default='Dynamic Rebalancing')
    
    # Market state allocations - use string keys for JSON serialization
    allocations: Dict[str, AllocationSet] = Field(default_factory=dict)

    # Indicators
    indicators: IndicatorParams = Field(default_factory=IndicatorParams)
    enabled_indicators: List[str] = Field(
        default=['ema_cross', 'adx_filter', 'atr_volatility']
    )

    # Rebalancing
    base_rebalance_threshold: Decimal = Field(
        default=Decimal('0.05'), ge=Decimal('0.01'), le=Decimal('0.20')
    )
    dynamic_threshold: bool = Field(default=True)
    volatility_multiplier: Decimal = Field(
        default=Decimal('1.5'), ge=Decimal('0.5'), le=Decimal('5.0')
    )

    # Market state detection
    use_multi_timeframe: bool = Field(default=False)
    confirmation_timeframes: List[TimeFrame] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True
        use_enum_values = True
        json_encoders = {
            MarketState: lambda v: v.value,
            Decimal: lambda v: str(v)
        }


class OptimizationConfig(BaseModel):
    """Optimization configuration"""
    # Objective function
    objective: str = Field(default='sharpe_ratio')  # sharpe_ratio, total_return, calmar_ratio
    
    # Search space
    parameter_ranges: Dict[str, Dict[str, Union[int, float]]] = Field(
        default_factory=lambda: {
            'ema_short': {'min': 5, 'max': 20, 'step': 5},
            'ema_long': {'min': 20, 'max': 50, 'step': 5},
            'adx_threshold': {'min': 20, 'max': 30, 'step': 5},
        }
    )
    
    # Allocation optimization
    optimize_allocations: bool = Field(default=True)
    allocation_constraints: Dict[str, Dict[str, Decimal]] = Field(
        default_factory=lambda: {
            'BTC': {'min': Decimal('0.2'), 'max': Decimal('0.8')},
            'ETH': {'min': Decimal('0.0'), 'max': Decimal('0.5')},
            'USDT': {'min': Decimal('0.0'), 'max': Decimal('0.5')},
        }
    )
    
    # Optimization method
    method: str = Field(default='grid_search')  # grid_search, random_search, bayesian
    max_iterations: int = Field(default=1000, ge=10)
    n_jobs: int = Field(default=-1)  # -1 for all cores
    
    # Early stopping
    early_stopping: bool = Field(default=True)
    patience: int = Field(default=50)
    min_improvement: float = Field(default=0.001)
    
    class Config:
        arbitrary_types_allowed = True


class Config(BaseModel):
    """Main configuration container"""
    data: DataConfig
    backtest: BacktestConfig
    strategy: StrategyConfig
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    
    # Logging and output
    log_level: str = Field(default='INFO')
    output_dir: str = Field(default='./results')
    save_results: bool = Field(default=True)
    plot_results: bool = Field(default=True)