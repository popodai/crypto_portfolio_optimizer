"""
Portfolio management and tracking functionality (simplified version).
"""
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np

from core.types import (
    Portfolio, AssetBalance, Trade, OrderSide,
    MarketState, Signal
)
from config.base_config import BacktestConfig


class PortfolioManager:
    """Manages portfolio state and executes trades"""

    def __init__(self, config: BacktestConfig, assets: List[str], quote_asset: str = 'USDT'):
        self.config = config
        self.assets = assets
        self.quote_asset = quote_asset

        # Initialize portfolio
        self.cash = config.initial_capital
        self.holdings: Dict[str, Decimal] = {asset: Decimal('0') for asset in assets}

        # Track history
        self.portfolio_history: List[Portfolio] = []
        self.trades: List[Trade] = []
        self.signals: List[Signal] = []

        # Performance tracking
        self.initial_capital = config.initial_capital
        self.total_invested = config.initial_capital
        self.transaction_costs = Decimal('0')

    def get_current_portfolio(self, prices: Dict[str, Decimal], timestamp: datetime) -> Portfolio:
        """Calculate current portfolio state"""
        balances = {}
        total_value = self.cash

        for asset in self.assets:
            if asset == self.quote_asset:
                continue

            quantity = self.holdings.get(asset, Decimal('0'))
            price = prices.get(asset, Decimal('0'))
            value = quantity * price
            total_value += value

            if quantity > 0:
                balances[asset] = AssetBalance(
                    quantity=quantity,
                    value_quote=value,
                    allocation_pct=Decimal('0')
                )

        # Add cash balance
        if self.cash > 0:
            balances[self.quote_asset] = AssetBalance(
                quantity=self.cash,
                value_quote=self.cash,
                allocation_pct=Decimal('0')
            )

        # Calculate allocation percentages
        if total_value > 0:
            for asset, balance in balances.items():
                balance['allocation_pct'] = balance['value_quote'] / total_value

        return Portfolio(
            timestamp=timestamp,
            total_value=total_value,
            balances=balances,
            cash=self.cash
        )

    def execute_trade(
        self,
        asset: str,
        side: OrderSide,
        quantity: Decimal,
        price: Decimal,
        timestamp: datetime
    ) -> Optional[Trade]:
        """Execute a single trade with fees"""
        # Calculate trade value and fee
        trade_value = quantity * price
        fee = trade_value * self.config.transaction_fee

        # Check if trade is possible
        if side == OrderSide.BUY:
            total_cost = trade_value + fee
            if total_cost > self.cash:
                return None

            # Execute buy
            self.cash -= total_cost
            self.holdings[asset] = self.holdings.get(asset, Decimal('0')) + quantity

        else:  # SELL
            # Check if we have enough to sell
            if quantity > self.holdings.get(asset, Decimal('0')):
                return None

            # Execute sell
            self.holdings[asset] -= quantity
            self.cash += trade_value - fee

        # Record trade
        trade = Trade(
            timestamp=timestamp,
            symbol=f"{asset}{self.quote_asset}",
            side=side,
            quantity=quantity,
            price=price,
            fee=fee,
            value=trade_value
        )

        self.trades.append(trade)
        self.transaction_costs += fee

        return trade

    def process_dca(self, timestamp: datetime) -> bool:
        """Process Dollar Cost Averaging if applicable"""
        if not self.config.enable_dca:
            return False

        # Simple DCA check
        if len(self.portfolio_history) > 0:
            days_since_start = (timestamp - self.portfolio_history[0]['timestamp']).days
            if days_since_start > 0 and days_since_start % self.config.dca_frequency_days == 0:
                self.cash += self.config.dca_amount
                self.total_invested += self.config.dca_amount
                return True

        return False

    def rebalance(
        self,
        target_allocations: Dict[str, Decimal],
        prices: Dict[str, Decimal],
        timestamp: datetime,
        threshold: Optional[Decimal] = None
    ) -> List[Trade]:
        """Execute portfolio rebalancing (simplified)"""
        if threshold is None:
            threshold = Decimal('0.05')

        # Get current portfolio state
        current_portfolio = self.get_current_portfolio(prices, timestamp)
        total_value = current_portfolio['total_value']

        if total_value <= 0:
            return []

        executed_trades = []

        # Simple rebalancing: sell overweight assets first
        for asset in self.assets:
            if asset == self.quote_asset:
                continue

            current_alloc = current_portfolio['balances'].get(
                asset, {'allocation_pct': Decimal('0')}
            )['allocation_pct']
            target_alloc = target_allocations.get(asset, Decimal('0'))

            diff = current_alloc - target_alloc

            if diff > threshold:  # Need to sell
                sell_value = total_value * diff
                price = prices.get(asset, Decimal('0'))
                if price > 0:
                    quantity = sell_value / price
                    trade = self.execute_trade(asset, OrderSide.SELL, quantity, price, timestamp)
                    if trade:
                        executed_trades.append(trade)

        # Then buy underweight assets
        for asset in self.assets:
            if asset == self.quote_asset:
                continue

            current_alloc = current_portfolio['balances'].get(
                asset, {'allocation_pct': Decimal('0')}
            )['allocation_pct']
            target_alloc = target_allocations.get(asset, Decimal('0'))

            diff = target_alloc - current_alloc

            if diff > threshold:  # Need to buy
                buy_value = total_value * diff
                price = prices.get(asset, Decimal('0'))
                if price > 0:
                    quantity = buy_value / price
                    trade = self.execute_trade(asset, OrderSide.BUY, quantity, price, timestamp)
                    if trade:
                        executed_trades.append(trade)

        return executed_trades

    def update_history(self, prices: Dict[str, Decimal], timestamp: datetime):
        """Update portfolio history"""
        portfolio = self.get_current_portfolio(prices, timestamp)
        self.portfolio_history.append(portfolio)

    def add_signal(self, signal: Signal):
        """Record a strategy signal"""
        self.signals.append(signal)

    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics"""
        if not self.portfolio_history:
            return {}

        # Convert portfolio history to returns
        values = [float(p['total_value']) for p in self.portfolio_history]
        returns = pd.Series(values).pct_change().dropna()

        # Calculate CORRECT total return based on total invested
        total_return = (values[-1] - float(self.total_invested)) / float(self.total_invested)

        metrics = {
            'total_return': total_return,
            'total_trades': len(self.trades),
            'transaction_costs': float(self.transaction_costs),
            'final_value': values[-1],
            'total_invested': float(self.total_invested),
            'initial_capital': float(self.initial_capital)
        }

        if len(returns) > 0:
            metrics['volatility'] = returns.std() * np.sqrt(252)
            if metrics['volatility'] > 0:
                metrics['sharpe_ratio'] = (returns.mean() * 252) / metrics['volatility']
            else:
                metrics['sharpe_ratio'] = 0

            # Calculate max drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            metrics['max_drawdown'] = drawdown.min()

            # Fix win rate calculation
            winning_trades = 0
            total_closed = 0

            # Group trades by asset
            from collections import defaultdict
            positions = defaultdict(list)

            for trade in self.trades:
                asset = trade.symbol.replace(self.quote_asset, '')
                positions[asset].append(trade)

            # Calculate P&L for each position
            for asset, trades in positions.items():
                buy_value = 0
                sell_value = 0

                for trade in trades:
                    if trade.side == OrderSide.BUY:
                        buy_value += float(trade.value + trade.fee)
                    else:  # SELL
                        sell_value += float(trade.value - trade.fee)
                        total_closed += 1

                        # Check if this sell was profitable
                        if sell_value > buy_value:
                            winning_trades += 1

            metrics['win_rate'] = winning_trades / total_closed if total_closed > 0 else 0
        
        return metrics