"""
Main backtesting engine that orchestrates the simulation.
"""
from decimal import Decimal
from typing import Dict, List, Optional, Type
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm

from core.types import (
    BacktestResult, Portfolio, Trade, Signal,
    MarketState, TimeFrame
)
from core.portfolio import PortfolioManager
from core.market_data import MarketDataManager
from strategies.base_strategy import BaseStrategy
from config.base_config import Config, StrategyConfig
from backtesting.metrics import MetricsCalculator


class BacktestEngine:
    """
    Main backtesting engine that simulates trading strategies
    """
    
    def __init__(
        self,
        config: Config,
        strategy_class: Type[BaseStrategy],
        verbose: bool = True
    ):
        self.config = config
        self.verbose = verbose
        
        # Initialize components
        self.data_manager = MarketDataManager(config.data)
        self.strategy = strategy_class(config.strategy)
        self.metrics_calculator = MetricsCalculator()
        
        # Data storage
        self.price_data: Optional[pd.DataFrame] = None
        self.ohlc_data: Optional[pd.DataFrame] = None
        self.indicator_data: Optional[pd.DataFrame] = None
        
    def run(self) -> BacktestResult:
        """
        Run the backtest simulation
        
        Returns:
            BacktestResult with complete simulation results
        """
        if self.verbose:
            print("=" * 60)
            print("Starting Backtest Simulation")
            print("=" * 60)
            print(f"Strategy: {self.config.strategy.name}")
            print(f"Period: {self.config.data.start_date.date()} to {self.config.data.end_date.date()}")
            print(f"Initial Capital: ${self.config.backtest.initial_capital:,.2f}")
            print("-" * 60)
        
        # 1. Load and prepare data
        self._load_data()
        
        # 2. Initialize portfolio
        portfolio_manager = PortfolioManager(
            self.config.backtest,
            self.config.data.assets,
            self.config.data.quote_asset
        )
        
        # 3. Run simulation
        self._run_simulation(portfolio_manager)
        
        # 4. Calculate metrics
        portfolio_history_df = self._create_portfolio_history_df(
            portfolio_manager.portfolio_history
        )
        
        metrics = portfolio_manager.get_performance_metrics()
        
        # 5. Create result
        result = BacktestResult(
            portfolio_history=portfolio_history_df,
            trades=portfolio_manager.trades,
            signals=portfolio_manager.signals,
            metrics=metrics,
            final_portfolio=portfolio_manager.portfolio_history[-1]
        )
        
        if self.verbose:
            self._print_results(result)
        
        return result
    
    def _load_data(self):
        """Load and prepare market data"""
        if self.verbose:
            print("\nLoading market data...")
        
        # Fetch historical data
        self.price_data, self.ohlc_data = self.data_manager.get_historical_data()
        
        # Calculate indicators
        indicator_config = self._create_indicator_config()
        self.indicator_data = self.data_manager.add_indicators(
            self.ohlc_data,
            indicator_config
        )
        
        if self.verbose:
            print(f"Loaded {len(self.price_data)} data points")
            print(f"Assets: {', '.join(self.config.data.assets)}")
    
    def _create_indicator_config(self) -> Dict[str, Dict[str, any]]:
        """Create indicator configuration from strategy config"""
        config = {}
        
        # EMA
        if 'ema_cross' in self.config.strategy.enabled_indicators:
            config['ema'] = {
                'enabled': True,
                'periods': [
                    self.config.strategy.indicators.ema_short,
                    self.config.strategy.indicators.ema_long
                ]
            }
        
        # ADX
        if 'adx_filter' in self.config.strategy.enabled_indicators:
            config['adx'] = {
                'enabled': True,
                'period': self.config.strategy.indicators.adx_period
            }
        
        # ATR
        if 'atr_volatility' in self.config.strategy.enabled_indicators:
            config['atr'] = {
                'enabled': True,
                'period': self.config.strategy.indicators.atr_period
            }
        
        # RSI
        if 'rsi' in self.config.strategy.enabled_indicators:
            config['rsi'] = {
                'enabled': True,
                'period': self.config.strategy.indicators.rsi_period
            }
        
        # MACD
        if 'macd' in self.config.strategy.enabled_indicators:
            config['macd'] = {
                'enabled': True,
                'fast': self.config.strategy.indicators.macd_fast,
                'slow': self.config.strategy.indicators.macd_slow,
                'signal': self.config.strategy.indicators.macd_signal
            }
        
        return config
    
    def _run_simulation(self, portfolio_manager: PortfolioManager):
        """Run the main simulation loop"""
        if self.verbose:
            print("\nRunning simulation...")
        
        # Get aligned data
        common_dates = self.price_data.index.intersection(self.indicator_data.index)
        
        # Progress bar
        iterator = tqdm(common_dates) if self.verbose else common_dates
        
        # Track state
        last_signal_date = None
        rebalance_count = 0
        
        for date in iterator:
            # Get current prices
            prices = {}
            for asset in self.config.data.assets:
                if asset in self.price_data.columns:
                    prices[asset] = Decimal(str(self.price_data.loc[date, asset]))
            
            # Check for DCA
            if portfolio_manager.process_dca(date):
                if self.verbose:
                    iterator.set_description(f"DCA executed on {date.date()}")
            
            # Update portfolio history
            portfolio_manager.update_history(prices, date)
            
            # Get current portfolio
            current_portfolio = portfolio_manager.get_current_portfolio(prices, date)
            
            # Skip if portfolio value too low
            if current_portfolio['total_value'] < self.config.backtest.initial_capital * Decimal('0.01'):
                continue
            
            # Get strategy signal
            market_data = self.indicator_data.loc[:date]
            current_allocations = {
                asset: balance['allocation_pct']
                for asset, balance in current_portfolio['balances'].items()
            }
            
            signal = self.strategy.analyze(market_data, current_allocations)
            portfolio_manager.add_signal(signal)
            
            # Calculate dynamic threshold
            volatility = signal.metadata.get('volatility', 0.0)
            threshold = self.strategy.calculate_rebalance_threshold(
                volatility,
                self.config.strategy.base_rebalance_threshold
            )
            
            # Execute rebalancing if needed
            trades = portfolio_manager.rebalance(
                signal.target_allocations,
                prices,
                date,
                threshold
            )
            
            if trades:
                rebalance_count += 1
                last_signal_date = date
                if self.verbose:
                    iterator.set_description(
                        f"Rebalanced ({signal.market_state.value}) - {len(trades)} trades"
                    )
        
        # Final portfolio update
        if common_dates[-1] != portfolio_manager.portfolio_history[-1]['timestamp']:
            final_prices = {}
            for asset in self.config.data.assets:
                if asset in self.price_data.columns:
                    final_prices[asset] = Decimal(str(self.price_data.iloc[-1][asset]))
            portfolio_manager.update_history(final_prices, self.price_data.index[-1])
        
        if self.verbose:
            print(f"\nSimulation complete. Total rebalances: {rebalance_count}")
    
    def _create_portfolio_history_df(self, portfolio_history: List[Portfolio]) -> pd.DataFrame:
        """Convert portfolio history to DataFrame"""
        data = []
        
        for portfolio in portfolio_history:
            row = {
                'timestamp': portfolio['timestamp'],
                'total_value': float(portfolio['total_value']),
                'cash': float(portfolio['cash'])
            }
            
            # Add individual asset values
            for asset, balance in portfolio['balances'].items():
                row[f'{asset}_quantity'] = float(balance['quantity'])
                row[f'{asset}_value'] = float(balance['value_quote'])
                row[f'{asset}_allocation'] = float(balance['allocation_pct'])
            
            data.append(row)
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def _print_results(self, result: BacktestResult):
        """Print backtest results summary"""
        print("\n" + "=" * 60)
        print("Backtest Results Summary")
        print("=" * 60)
        
        metrics = result['metrics']
        
        print(f"\nPerformance:")
        print(f"  Total Return:      {metrics['total_return']*100:>8.2f}%")
        print(f"  Annual Volatility: {metrics.get('volatility', 0)*100:>8.2f}%")
        print(f"  Sharpe Ratio:      {metrics.get('sharpe_ratio', 0):>8.2f}")
        print(f"  Max Drawdown:      {metrics.get('max_drawdown', 0)*100:>8.2f}%")
        
        print(f"\nTrading Activity:")
        print(f"  Total Trades:      {metrics['total_trades']:>8}")
        print(f"  Transaction Costs: ${metrics['transaction_costs']:>8,.2f}")
        print(f"  Win Rate:          {metrics.get('win_rate', 0)*100:>8.2f}%")
        
        print(f"\nFinal State:")
        print(f"  Initial Capital:   ${self.config.backtest.initial_capital:>10,.2f}")
        print(f"  Total Invested:    ${metrics['total_invested']:>10,.2f}")
        print(f"  Final Value:       ${metrics['final_value']:>10,.2f}")
        
        # Final allocations
        final_portfolio = result['final_portfolio']
        print(f"\nFinal Allocations:")
        for asset, balance in final_portfolio['balances'].items():
            if balance['allocation_pct'] > 0:
                print(f"  {asset:<6} {float(balance['allocation_pct'])*100:>6.2f}%")
        
        print("=" * 60)