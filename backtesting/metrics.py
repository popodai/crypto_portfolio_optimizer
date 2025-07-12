"""
Performance metrics calculation for backtesting.
"""
import pandas as pd
import numpy as np
from typing import Dict, List
from decimal import Decimal


class MetricsCalculator:
    """Calculate various performance metrics for backtest results"""
    
    @staticmethod
    def calculate_returns(values: pd.Series) -> pd.Series:
        """Calculate simple returns from value series"""
        return values.pct_change().dropna()
    
    @staticmethod
    def calculate_log_returns(values: pd.Series) -> pd.Series:
        """Calculate log returns from value series"""
        return np.log(values / values.shift(1)).dropna()
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sharpe ratio (annualized)
        
        Args:
            returns: Daily returns series
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Annualized Sharpe ratio
        """
        if len(returns) == 0:
            return 0.0
        
        # Annualize returns and volatility
        annual_returns = returns.mean() * 252
        annual_volatility = returns.std() * np.sqrt(252)
        
        if annual_volatility == 0:
            return 0.0
        
        return (annual_returns - risk_free_rate) / annual_volatility
    
    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sortino ratio (uses downside deviation)
        
        Args:
            returns: Daily returns series
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Annualized Sortino ratio
        """
        if len(returns) == 0:
            return 0.0
        
        # Calculate downside returns
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf')  # No downside risk
        
        # Annualize
        annual_returns = returns.mean() * 252
        downside_deviation = downside_returns.std() * np.sqrt(252)
        
        if downside_deviation == 0:
            return 0.0
        
        return (annual_returns - risk_free_rate) / downside_deviation
    
    @staticmethod
    def calculate_max_drawdown(values: pd.Series) -> Dict[str, float]:
        """
        Calculate maximum drawdown and related metrics
        
        Args:
            values: Portfolio value series
            
        Returns:
            Dictionary with max_drawdown, max_dd_duration, etc.
        """
        if len(values) == 0:
            return {'max_drawdown': 0.0, 'max_dd_duration': 0}
        
        # Calculate running maximum
        running_max = values.expanding().max()
        
        # Calculate drawdown series
        drawdown = (values - running_max) / running_max
        
        # Find maximum drawdown
        max_drawdown = drawdown.min()
        
        # Calculate drawdown duration
        drawdown_start = None
        max_duration = 0
        current_duration = 0
        
        for i, dd in enumerate(drawdown):
            if dd < 0:
                if drawdown_start is None:
                    drawdown_start = i
                current_duration = i - drawdown_start
            else:
                if current_duration > max_duration:
                    max_duration = current_duration
                drawdown_start = None
                current_duration = 0
        
        return {
            'max_drawdown': max_drawdown,
            'max_dd_duration': max_duration
        }
    
    @staticmethod
    def calculate_calmar_ratio(total_return: float, max_drawdown: float) -> float:
        """
        Calculate Calmar ratio (return / max drawdown)
        
        Args:
            total_return: Total return (e.g., 0.5 for 50%)
            max_drawdown: Maximum drawdown (e.g., -0.2 for -20%)
            
        Returns:
            Calmar ratio
        """
        if max_drawdown == 0 or max_drawdown > 0:
            return 0.0
        
        return total_return / abs(max_drawdown)
    
    @staticmethod
    def calculate_win_rate(trades: List) -> float:
        """
        Calculate win rate from trades
        
        Args:
            trades: List of Trade objects
            
        Returns:
            Win rate (0 to 1)
        """
        if not trades:
            return 0.0
        
        # Group trades by symbol to match buys and sells
        profitable_trades = 0
        total_closed_trades = 0
        
        # Simple approach: count sells as closed trades
        for trade in trades:
            if trade.side.value == 'SELL':
                total_closed_trades += 1
                # Assume profitable if sold (simplified)
                if trade.value > trade.fee:
                    profitable_trades += 1
        
        if total_closed_trades == 0:
            return 0.0
        
        return profitable_trades / total_closed_trades
    
    @staticmethod
    def calculate_all_metrics(
        portfolio_values: pd.Series,
        trades: List,
        initial_capital: float,
        total_invested: float
    ) -> Dict[str, float]:
        """
        Calculate all performance metrics
        
        Args:
            portfolio_values: Series of portfolio values over time
            trades: List of Trade objects
            initial_capital: Initial capital
            total_invested: Total amount invested (including DCA)
            
        Returns:
            Dictionary of all metrics
        """
        calculator = MetricsCalculator()
        
        # Basic metrics
        final_value = portfolio_values.iloc[-1] if len(portfolio_values) > 0 else initial_capital
        total_return = (final_value - initial_capital) / initial_capital if initial_capital > 0 else 0
        
        # Calculate returns
        returns = calculator.calculate_returns(portfolio_values)
        
        # Risk metrics
        drawdown_metrics = calculator.calculate_max_drawdown(portfolio_values)
        
        # Compile all metrics
        metrics = {
            'total_return': total_return,
            'final_value': final_value,
            'total_invested': total_invested,
            'total_trades': len(trades),
            'volatility': returns.std() * np.sqrt(252) if len(returns) > 0 else 0,
            'sharpe_ratio': calculator.calculate_sharpe_ratio(returns),
            'sortino_ratio': calculator.calculate_sortino_ratio(returns),
            'max_drawdown': drawdown_metrics['max_drawdown'],
            'max_dd_duration': drawdown_metrics['max_dd_duration'],
            'calmar_ratio': calculator.calculate_calmar_ratio(
                total_return, drawdown_metrics['max_drawdown']
            ),
            'win_rate': calculator.calculate_win_rate(trades)
        }
        
        # Add trade analysis
        if trades:
            buy_trades = [t for t in trades if t.side.value == 'BUY']
            sell_trades = [t for t in trades if t.side.value == 'SELL']
            
            metrics['total_buys'] = len(buy_trades)
            metrics['total_sells'] = len(sell_trades)
            metrics['avg_trade_value'] = np.mean([t.value for t in trades])
            metrics['transaction_costs'] = sum(t.fee for t in trades)
        
        return metrics