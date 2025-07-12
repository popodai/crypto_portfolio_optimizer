"""
Visualization tools for backtest results.
"""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from core.types import BacktestResult, Trade, Signal, MarketState


class BacktestVisualizer:
    """Create visualizations for backtest results"""
    
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        plt.style.use(style)
        self.colors = {
            'portfolio': '#2E86AB',
            'benchmark': '#A23B72',
            'bull': '#00A878',
            'bear': '#FE5E41',
            'sideways': '#F18F01',
            'buy': '#00A878',
            'sell': '#FE5E41'
        }
    
    def plot_full_report(
        self,
        result: BacktestResult,
        benchmark_result: Optional[BacktestResult] = None,
        save_path: Optional[str] = None
    ):
        """Generate comprehensive backtest report"""
        fig = plt.figure(figsize=(16, 20))
        gs = GridSpec(5, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Portfolio Value Chart
        ax1 = fig.add_subplot(gs[0, :])
        self.plot_portfolio_value(result, benchmark_result, ax=ax1)
        
        # 2. Returns Distribution
        ax2 = fig.add_subplot(gs[1, 0])
        self.plot_returns_distribution(result, ax=ax2)
        
        # 3. Drawdown Chart
        ax3 = fig.add_subplot(gs[1, 1])
        self.plot_drawdown(result, ax=ax3)
        
        # 4. Asset Allocation Over Time
        ax4 = fig.add_subplot(gs[2, :])
        self.plot_allocation_history(result, ax=ax4)
        
        # 5. Market State Distribution
        ax5 = fig.add_subplot(gs[3, 0])
        self.plot_market_state_distribution(result, ax=ax5)
        
        # 6. Trade Analysis
        ax6 = fig.add_subplot(gs[3, 1])
        self.plot_trade_analysis(result, ax=ax6)
        
        # 7. Monthly Returns Heatmap
        ax7 = fig.add_subplot(gs[4, :])
        self.plot_monthly_returns_heatmap(result, ax=ax7)
        
        # Add title and metrics
        fig.suptitle('Backtest Report', fontsize=16, fontweight='bold')
        
        # Add text box with key metrics
        metrics_text = self._format_metrics_text(result['metrics'])
        fig.text(0.99, 0.01, metrics_text, transform=fig.transFigure,
                fontsize=10, verticalalignment='bottom',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def plot_portfolio_value(
        self,
        result: BacktestResult,
        benchmark_result: Optional[BacktestResult] = None,
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """Plot portfolio value over time"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot strategy portfolio
        portfolio_df = result['portfolio_history']
        ax.plot(portfolio_df.index, portfolio_df['total_value'],
               label='Strategy', color=self.colors['portfolio'], linewidth=2)
        
        # Plot benchmark if provided
        if benchmark_result:
            bench_df = benchmark_result['portfolio_history']
            ax.plot(bench_df.index, bench_df['total_value'],
                   label='Benchmark', color=self.colors['benchmark'],
                   linewidth=2, linestyle='--')
        
        # Add trade markers
        for trade in result['trades']:
            if trade.side.value == 'BUY':
                ax.scatter(trade.timestamp, 
                          portfolio_df.loc[trade.timestamp, 'total_value'],
                          color=self.colors['buy'], marker='^', s=50, alpha=0.7)
            else:
                ax.scatter(trade.timestamp,
                          portfolio_df.loc[trade.timestamp, 'total_value'],
                          color=self.colors['sell'], marker='v', s=50, alpha=0.7)
        
        ax.set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        return ax
    
    def plot_returns_distribution(
        self,
        result: BacktestResult,
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """Plot returns distribution"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        # Calculate returns
        portfolio_df = result['portfolio_history']
        returns = portfolio_df['total_value'].pct_change().dropna()
        
        # Plot histogram
        n, bins, patches = ax.hist(returns, bins=50, alpha=0.7,
                                  color=self.colors['portfolio'], edgecolor='black')
        
        # Add normal distribution overlay
        mu, sigma = returns.mean(), returns.std()
        x = np.linspace(returns.min(), returns.max(), 100)
        ax.plot(x, len(returns) * (bins[1] - bins[0]) * 
                (1/(sigma * np.sqrt(2*np.pi))) * 
                np.exp(-0.5*((x-mu)/sigma)**2),
                'r-', linewidth=2, label='Normal Distribution')
        
        # Add vertical lines for mean and median
        ax.axvline(returns.mean(), color='red', linestyle='--',
                  label=f'Mean: {returns.mean():.2%}')
        ax.axvline(returns.median(), color='green', linestyle='--',
                  label=f'Median: {returns.median():.2%}')
        
        ax.set_title('Returns Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Daily Returns')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_drawdown(
        self,
        result: BacktestResult,
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """Plot drawdown chart"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        portfolio_df = result['portfolio_history']
        
        # Calculate drawdown
        cumulative = portfolio_df['total_value']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        # Plot drawdown
        ax.fill_between(drawdown.index, 0, drawdown.values,
                       color=self.colors['bear'], alpha=0.3)
        ax.plot(drawdown.index, drawdown.values,
               color=self.colors['bear'], linewidth=1)
        
        # Mark maximum drawdown
        max_dd_idx = drawdown.idxmin()
        max_dd_value = drawdown.min()
        ax.scatter(max_dd_idx, max_dd_value, color='red', s=100, zorder=5)
        ax.annotate(f'Max DD: {max_dd_value:.1%}',
                   xy=(max_dd_idx, max_dd_value),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        ax.set_title('Drawdown Analysis', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(top=0.05)
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        return ax
    
    def plot_allocation_history(
        self,
        result: BacktestResult,
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """Plot asset allocation over time"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        
        portfolio_df = result['portfolio_history']
        
        # Extract allocation columns
        allocation_cols = [col for col in portfolio_df.columns if col.endswith('_allocation')]
        assets = [col.replace('_allocation', '') for col in allocation_cols]
        
        # Create stacked area plot
        allocation_data = portfolio_df[allocation_cols].fillna(0)
        allocation_data.columns = assets
        
        # Plot stacked area
        ax.stackplot(allocation_data.index,
                    *[allocation_data[asset] for asset in assets],
                    labels=assets,
                    alpha=0.7)
        
        # Add market state indicators
        if result['signals']:
            for signal in result['signals']:
                color = self.colors.get(signal.market_state.value.lower(), 'gray')
                ax.axvline(signal.timestamp, color=color, alpha=0.3, linewidth=1)
        
        ax.set_title('Asset Allocation Over Time', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Allocation (%)')
        ax.set_ylim(0, 1)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.grid(True, alpha=0.3)
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        return ax
    
    def plot_market_state_distribution(
        self,
        result: BacktestResult,
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """Plot distribution of market states"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        # Count market states from signals
        if not result['signals']:
            ax.text(0.5, 0.5, 'No signals available',
                   ha='center', va='center', transform=ax.transAxes)
            return ax
        
        state_counts = {}
        for signal in result['signals']:
            state = signal.market_state.value
            state_counts[state] = state_counts.get(state, 0) + 1
        
        # Create pie chart
        states = list(state_counts.keys())
        counts = list(state_counts.values())
        colors = [self.colors.get(state.lower(), 'gray') for state in states]
        
        wedges, texts, autotexts = ax.pie(counts, labels=states, colors=colors,
                                          autopct='%1.1f%%', startangle=90)
        
        ax.set_title('Market State Distribution', fontsize=14, fontweight='bold')
        
        return ax
    
    def plot_trade_analysis(
        self,
        result: BacktestResult,
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """Plot trade analysis"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        if not result['trades']:
            ax.text(0.5, 0.5, 'No trades executed',
                   ha='center', va='center', transform=ax.transAxes)
            return ax
        
        # Analyze trades by asset
        trade_summary = {}
        for trade in result['trades']:
            asset = trade.symbol.replace('USDT', '')
            if asset not in trade_summary:
                trade_summary[asset] = {'buy': 0, 'sell': 0, 'volume': 0}
            
            if trade.side.value == 'BUY':
                trade_summary[asset]['buy'] += 1
            else:
                trade_summary[asset]['sell'] += 1
            trade_summary[asset]['volume'] += float(trade.value)
        
        # Create grouped bar chart
        assets = list(trade_summary.keys())
        buy_counts = [trade_summary[asset]['buy'] for asset in assets]
        sell_counts = [trade_summary[asset]['sell'] for asset in assets]
        
        x = np.arange(len(assets))
        width = 0.35
        
        ax.bar(x - width/2, buy_counts, width, label='Buy',
              color=self.colors['buy'])
        ax.bar(x + width/2, sell_counts, width, label='Sell',
              color=self.colors['sell'])
        
        ax.set_xlabel('Asset')
        ax.set_ylabel('Number of Trades')
        ax.set_title('Trade Analysis by Asset', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(assets)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        return ax
    
    def plot_monthly_returns_heatmap(
        self,
        result: BacktestResult,
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """Plot monthly returns heatmap"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        
        portfolio_df = result['portfolio_history']
        
        # Calculate monthly returns
        monthly_returns = portfolio_df['total_value'].resample('M').last().pct_change()
        
        # Reshape for heatmap
        returns_pivot = pd.DataFrame({
            'Year': monthly_returns.index.year,
            'Month': monthly_returns.index.month,
            'Return': monthly_returns.values
        })
        
        returns_matrix = returns_pivot.pivot(index='Year', columns='Month', values='Return')
        
        # Create heatmap
        sns.heatmap(returns_matrix, annot=True, fmt='.1%',
                   cmap='RdYlGn', center=0, cbar_kws={'label': 'Monthly Return'},
                   ax=ax)
        
        ax.set_title('Monthly Returns Heatmap', fontsize=14, fontweight='bold')
        ax.set_xlabel('Month')
        ax.set_ylabel('Year')
        
        return ax
    
    def _format_metrics_text(self, metrics: Dict[str, float]) -> str:
        """Format metrics for display"""
        text_lines = [
            'Performance Metrics:',
            f"Total Return: {metrics.get('total_return', 0)*100:.2f}%",
            f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}",
            f"Max Drawdown: {metrics.get('max_drawdown', 0)*100:.2f}%",
            f"Win Rate: {metrics.get('win_rate', 0)*100:.1f}%",
            f"Total Trades: {metrics.get('total_trades', 0)}",
            f"Transaction Costs: ${metrics.get('transaction_costs', 0):,.2f}"
        ]
        
        return '\n'.join(text_lines)