"""
Main entry point for the crypto portfolio backtesting system.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path
import matplotlib.pyplot as plt

from config.base_config import (
    Config, DataConfig, BacktestConfig, StrategyConfig,
    OptimizationConfig, AllocationSet, MarketState
)
from backtesting.backtest_engine import BacktestEngine
from optimization.optimizer import StrategyOptimizer
from strategies.base_strategy import TrendFollowingStrategy, MeanReversionStrategy
from visualization.charts import BacktestVisualizer


# ============================================================================
# ตั้งค่าทั้งหมดที่นี่ (CONFIGURE EVERYTHING HERE)
# ============================================================================

# เลือกโหมดการทำงาน: 'backtest', 'optimize', หรือ 'compare'
MODE = 'optimize'

# เลือก Strategy: 'trend' หรือ 'mean_reversion'
STRATEGY = 'trend'

# ตั้งค่าข้อมูลและช่วงเวลา
DATA_CONFIG = {
    'assets': ['BTC', 'ETH'],
    'quote_asset': 'USDT',
    'start_date': datetime(2024, 1, 1),  # แค่ 1 ปี
    'end_date': datetime(2024, 12, 31),
    'timeframe': '4h'  # 4 hours
}
# ตั้งค่า Backtest
BACKTEST_CONFIG = {
    'initial_capital': Decimal('200'),  # เงินทุนเริ่มต้น
    'transaction_fee': Decimal('0.001'),  # ค่าธรรมเนียม 0.1%
    'enable_dca': True,                   # เปิด/ปิด DCA
    'dca_amount': Decimal('200'),        # จำนวนเงิน DCA
    'dca_frequency_days': 30              # ความถี่ DCA (ทุกกี่วัน)
}

# ตั้งค่า Strategy
STRATEGY_CONFIG = {
    'name': 'Dynamic Trend Following',

    # สัดส่วนการลงทุนในแต่ละสภาวะตลาด
    'allocations': {
        # ตลาดขาขึ้น (Bull Market)
        MarketState.BULL: {
            'BTC': Decimal('0.40'),   # 50% BT
            'ETH': Decimal('0.05'),   # 50% ETH
            'USDT': Decimal('0.55')   # 0% Cash
        },
        # ตลาดขาลง (Bear Market)
        MarketState.BEAR: {
            'BTC': Decimal('0.40'),   # 25% BTC
            'ETH': Decimal('0.05'),   # 25% ETH
            'USDT': Decimal('0.55')   # 50% Cash
        },
        # ตลาดไซด์เวย์ (Sideways Market)
        MarketState.SIDEWAYS: {
            'BTC': Decimal('0.40'),   # 35% BTC
            'ETH': Decimal('0.05'),   # 35% ETH
            'USDT': Decimal('0.55')   # 30% Cash
        }
    },

    # Indicators ที่ใช้
    'enabled_indicators': ['ema_cross', 'adx_filter', 'atr_volatility'],

    # พารามิเตอร์ของ Indicators
    'indicator_params': {
        'ema_short': 5,        # EMA ระยะสั้น
        'ema_long': 20,         # EMA ระยะยาว
        'adx_period': 10,       # ADX period
        'adx_threshold': 20,    # ADX threshold
        'atr_period': 15        # ATR period
    },

    # การ Rebalance
    'base_rebalance_threshold': Decimal('0.05'),  # เกณฑ์การ rebalance 5%
    'dynamic_threshold': True,                     # ใช้ dynamic threshold
    'volatility_multiplier': Decimal('1.5')        # ตัวคูณ volatility
}

# ตั้งค่า Optimization (ใช้เมื่อ MODE = 'optimize')
OPTIMIZATION_CONFIG = {
    'objective': 'sharpe_ratio',  # เป้าหมาย: 'sharpe_ratio', 'total_return', 'calmar_ratio'

    # ช่วงค่าพารามิเตอร์ที่จะค้นหา
    'parameter_ranges': {
        'ema_short': {'min': 5, 'max': 20, 'step': 5},
        'ema_long': {'min': 20, 'max': 50, 'step': 5},
        'adx_period': {'min': 10, 'max': 20, 'step': 5},
        'adx_threshold': {'min': 20, 'max': 30, 'step': 5},
        'atr_period': {'min': 10, 'max': 20, 'step': 5}
    },

    'optimize_allocations': True,  # optimize สัดส่วนการลงทุนด้วย
    'method': 'grid_search',       # วิธีการ: 'grid_search', 'random_search', 'bayesian'
    'n_jobs': -1                   # ใช้ CPU ทุก core (-1)
}

# โฟลเดอร์สำหรับบันทึกผลลัพธ์
OUTPUT_DIR = './results'

# ============================================================================
# ส่วนโค้ดด้านล่างไม่ต้องแก้ไข
# ============================================================================


def create_config_from_settings() -> Config:
    """Create configuration from settings above"""
    # Data configuration
    data_config = DataConfig(
        assets=DATA_CONFIG['assets'],
        quote_asset=DATA_CONFIG['quote_asset'],
        start_date=DATA_CONFIG['start_date'],
        end_date=DATA_CONFIG['end_date'],
        timeframe=DATA_CONFIG['timeframe']
    )

    # Backtest configuration
    backtest_config = BacktestConfig(
        initial_capital=BACKTEST_CONFIG['initial_capital'],
        transaction_fee=BACKTEST_CONFIG['transaction_fee'],
        enable_dca=BACKTEST_CONFIG['enable_dca'],
        dca_amount=BACKTEST_CONFIG['dca_amount'],
        dca_frequency_days=BACKTEST_CONFIG['dca_frequency_days']
    )

    # Strategy configuration
    allocations = {}
    for state, alloc_dict in STRATEGY_CONFIG['allocations'].items():
        # Convert MarketState enum to string value for consistency
        state_key = state.value if hasattr(state, 'value') else str(state)
        allocations[state_key] = AllocationSet(allocations=alloc_dict)

    from config.base_config import IndicatorParams
    indicators = IndicatorParams(**STRATEGY_CONFIG['indicator_params'])

    strategy_config = StrategyConfig(
        name=STRATEGY_CONFIG['name'],
        allocations=allocations,
        indicators=indicators,
        enabled_indicators=STRATEGY_CONFIG['enabled_indicators'],
        base_rebalance_threshold=STRATEGY_CONFIG['base_rebalance_threshold'],
        dynamic_threshold=STRATEGY_CONFIG['dynamic_threshold'],
        volatility_multiplier=STRATEGY_CONFIG['volatility_multiplier']
    )

    # Optimization configuration
    optimization_config = OptimizationConfig(
        objective=OPTIMIZATION_CONFIG['objective'],
        parameter_ranges=OPTIMIZATION_CONFIG['parameter_ranges'],
        optimize_allocations=OPTIMIZATION_CONFIG['optimize_allocations'],
        method=OPTIMIZATION_CONFIG['method'],
        n_jobs=OPTIMIZATION_CONFIG['n_jobs']
    )

    return Config(
        data=data_config,
        backtest=backtest_config,
        strategy=strategy_config,
        optimization=optimization_config,
        output_dir=OUTPUT_DIR
    )


def run_single_backtest(config: Config, strategy_name: str):
    """Run a single backtest"""
    print("\n" + "="*60)
    print("Running Single Backtest")
    print("="*60)

    # Select strategy
    if strategy_name == 'trend':
        strategy_class = TrendFollowingStrategy
    elif strategy_name == 'mean_reversion':
        strategy_class = MeanReversionStrategy
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    # Create and run backtest
    engine = BacktestEngine(config, strategy_class)
    result = engine.run()

    # Create visualizations
    visualizer = BacktestVisualizer()

    # Save results
    output_dir = Path(config.output_dir) / f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot and save
    visualizer.plot_full_report(result, save_path=str(output_dir / "report.png"))

    # Save metrics
    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(result['metrics'], f, indent=2, default=str)

    print(f"\nResults saved to: {output_dir}")

    return result


def run_optimization(config: Config, strategy_name: str):
    """Run strategy optimization"""
    print("\n" + "="*60)
    print("Running Strategy Optimization")
    print("="*60)

    # Select strategy
    if strategy_name == 'trend':
        strategy_class = TrendFollowingStrategy
    elif strategy_name == 'mean_reversion':
        strategy_class = MeanReversionStrategy
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    # Create optimizer
    optimizer = StrategyOptimizer(config, strategy_class)

    # Run optimization
    best_result = optimizer.optimize()

    # Save results
    output_dir = Path(config.output_dir) / f"optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save optimization results
    optimizer.save_results(str(output_dir / "optimization_results.json"))

    # Create visualizations for best result
    visualizer = BacktestVisualizer()
    visualizer.plot_full_report(
        best_result['backtest_result'],
        save_path=str(output_dir / "best_result_report.png")
    )

    print(f"\nOptimization results saved to: {output_dir}")

    return best_result


def run_comparison(config: Config):
    """Run comparison between strategies"""
    print("\n" + "="*60)
    print("Running Strategy Comparison")
    print("="*60)

    strategies = {
        'Trend Following': TrendFollowingStrategy,
        'Mean Reversion': MeanReversionStrategy
    }

    results = {}

    # Run backtest for each strategy
    for name, strategy_class in strategies.items():
        print(f"\nTesting {name} strategy...")
        engine = BacktestEngine(config, strategy_class, verbose=False)
        results[name] = engine.run()

    # Create comparison visualizations
    visualizer = BacktestVisualizer()

    # Compare portfolio values
    fig, ax = plt.subplots(figsize=(12, 6))

    for name, result in results.items():
        portfolio_df = result['portfolio_history']
        ax.plot(portfolio_df.index, portfolio_df['total_value'], label=name, linewidth=2)

    ax.set_title('Strategy Comparison: Portfolio Value', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Value ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Save comparison
    output_dir = Path(config.output_dir) / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_dir / "comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Save metrics comparison
    metrics_comparison = {}
    for name, result in results.items():
        metrics_comparison[name] = result['metrics']

    with open(output_dir / "metrics_comparison.json", 'w') as f:
        json.dump(metrics_comparison, f, indent=2, default=str)

    print(f"\nComparison results saved to: {output_dir}")

    return results


def main():
    """Main entry point"""
    # Create configuration from settings
    config = create_config_from_settings()

    # Print current settings
    print("\n" + "="*60)
    print("CRYPTO PORTFOLIO BACKTESTING SYSTEM")
    print("="*60)
    print(f"Mode: {MODE}")
    print(f"Strategy: {STRATEGY}")
    print(f"Assets: {', '.join(DATA_CONFIG['assets'])}")
    print(f"Period: {DATA_CONFIG['start_date'].date()} to {DATA_CONFIG['end_date'].date()}")
    print(f"Initial Capital: ${BACKTEST_CONFIG['initial_capital']:,}")
    print("="*60)

    # Execute based on mode
    if MODE == 'backtest':
        run_single_backtest(config, STRATEGY)
    elif MODE == 'optimize':
        run_optimization(config, STRATEGY)
    elif MODE == 'compare':
        run_comparison(config)
    else:
        print(f"Error: Unknown mode '{MODE}'")
        print("Please set MODE to 'backtest', 'optimize', or 'compare'")


if __name__ == '__main__':
    main()