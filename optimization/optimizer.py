"""
Strategy parameter and allocation optimizer.
"""
from decimal import Decimal
from typing import Dict, List, Optional, Type, Tuple, Any
from itertools import product
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass
import json

from core.types import OptimizationResult, MarketState
from config.base_config import Config, AllocationSet
from backtesting.backtest_engine import BacktestEngine
from strategies.base_strategy import BaseStrategy
from optimization.parameter_search import (
    GridSearchOptimizer,
    RandomSearchOptimizer,
    BayesianOptimizer
)


@dataclass
class OptimizationTask:
    """Single optimization task"""
    parameters: Dict[str, Any]
    config: Config
    strategy_class: Type[BaseStrategy]
    task_id: int


class StrategyOptimizer:
    """
    Optimizes strategy parameters and allocations
    """
    
    def __init__(
        self,
        base_config: Config,
        strategy_class: Type[BaseStrategy],
        n_jobs: Optional[int] = None
    ):
        self.base_config = base_config
        self.strategy_class = strategy_class
        self.n_jobs = n_jobs or cpu_count() - 1
        
        # Choose optimizer based on method
        self.param_optimizer = self._create_param_optimizer()
        
        # Results storage
        self.results: List[OptimizationResult] = []
        self.best_result: Optional[OptimizationResult] = None
        
    def optimize(self) -> OptimizationResult:
        """
        Run full optimization process
        
        Returns:
            Best optimization result
        """
        print("=" * 80)
        print("Starting Strategy Optimization")
        print("=" * 80)
        print(f"Method: {self.base_config.optimization.method}")
        print(f"Objective: {self.base_config.optimization.objective}")
        print(f"Using {self.n_jobs} parallel workers")
        print("-" * 80)
        
        # Phase 1: Optimize strategy parameters
        if self.base_config.optimization.parameter_ranges:
            print("\n[Phase 1] Optimizing Strategy Parameters...")
            best_params = self._optimize_parameters()
            print(f"Best parameters found: {best_params}")
        else:
            best_params = self._get_default_parameters()
            print("\n[Phase 1] Using default strategy parameters")
        
        # Phase 2: Optimize allocations (if enabled)
        if self.base_config.optimization.optimize_allocations:
            print("\n[Phase 2] Optimizing Asset Allocations...")
            best_allocations = self._optimize_allocations(best_params)
            print("Best allocations found")
        else:
            best_allocations = self._get_default_allocations()
            print("\n[Phase 2] Using default allocations")
        
        # Phase 3: Final validation run
        print("\n[Phase 3] Running Final Validation...")
        self.best_result = self._run_final_validation(best_params, best_allocations)
        
        self._print_optimization_summary()
        
        return self.best_result
    
    def _create_param_optimizer(self):
        """Create parameter optimizer based on method"""
        method = self.base_config.optimization.method
        
        if method == 'grid_search':
            return GridSearchOptimizer(self.base_config.optimization)
        elif method == 'random_search':
            return RandomSearchOptimizer(self.base_config.optimization)
        elif method == 'bayesian':
            return BayesianOptimizer(self.base_config.optimization)
        else:
            raise ValueError(f"Unknown optimization method: {method}")
    
    def _optimize_parameters(self) -> Dict[str, Any]:
        """Optimize strategy parameters"""
        # Generate parameter combinations
        param_space = self.param_optimizer.generate_parameter_space()
        
        print(f"Testing {len(param_space)} parameter combinations...")
        
        # Create tasks
        tasks = []
        for i, params in enumerate(param_space):
            config = self._create_config_with_params(params)
            task = OptimizationTask(params, config, self.strategy_class, i)
            tasks.append(task)
        
        # Run parallel optimization
        results = self._run_parallel_optimization(tasks, desc="Optimizing Parameters")
        
        # Find best result
        best_result = max(results, key=lambda r: r['objective_value'])
        
        return best_result['parameters']
    
    def _optimize_allocations(self, strategy_params: Dict[str, Any]) -> Dict[MarketState, AllocationSet]:
        """Optimize asset allocations"""
        # Generate allocation combinations
        allocation_space = self._generate_allocation_space()
        
        print(f"Testing {len(allocation_space)} allocation combinations...")
        
        # Create tasks with fixed strategy parameters
        tasks = []
        for i, allocations in enumerate(allocation_space):
            config = self._create_config_with_params(strategy_params)
            config.strategy.allocations = allocations
            task = OptimizationTask(strategy_params, config, self.strategy_class, i)
            tasks.append(task)
        
        # Run parallel optimization
        results = self._run_parallel_optimization(tasks, desc="Optimizing Allocations")
        
        # Find best result
        best_result = max(results, key=lambda r: r['objective_value'])
        
        # Return the complete allocations (including SIDEWAYS)
        best_task_config = None
        for task in tasks:
            task_result = self._run_single_backtest(task)
            if task_result and task_result['objective_value'] == best_result['objective_value']:
                best_task_config = task.config
                break

        if best_task_config:
            return best_task_config.strategy.allocations

        # Fallback to best result allocations
        return self._create_allocations_from_result(best_result)
    
    def _generate_allocation_space(self) -> List[Dict[MarketState, AllocationSet]]:
        """Generate allocation combinations to test"""
        constraints = self.base_config.optimization.allocation_constraints
        assets = self.base_config.data.assets
        
        # Generate ranges for each asset
        asset_ranges = {}
        for asset in assets:
            if asset in constraints:
                min_val = float(constraints[asset]['min'])
                max_val = float(constraints[asset]['max'])
                step = 0.05  # 5% steps
                asset_ranges[asset] = np.arange(min_val, max_val + step, step)
            else:
                asset_ranges[asset] = np.arange(0, 1.01, 0.05)
        
        # Generate valid combinations
        allocation_space = []
        
        # For each market state
        for bull_btc in asset_ranges.get('BTC', [0.5]):
            for bull_eth in asset_ranges.get('ETH', [0.5]):
                bull_usdt = 1.0 - bull_btc - bull_eth
                if bull_usdt < 0 or bull_usdt > 1:
                    continue
                
                for bear_btc in asset_ranges.get('BTC', [0.5]):
                    for bear_eth in asset_ranges.get('ETH', [0.5]):
                        bear_usdt = 1.0 - bear_btc - bear_eth
                        if bear_usdt < 0 or bear_usdt > 1:
                            continue
                        
                        # Create allocation set
                        allocations = {
                            MarketState.BULL: AllocationSet(allocations={
                                'BTC': Decimal(str(bull_btc)),
                                'ETH': Decimal(str(bull_eth)),
                                'USDT': Decimal(str(bull_usdt))
                            }),
                            MarketState.BEAR: AllocationSet(allocations={
                                'BTC': Decimal(str(bear_btc)),
                                'ETH': Decimal(str(bear_eth)),
                                'USDT': Decimal(str(bear_usdt))
                            }),
                            MarketState.SIDEWAYS: AllocationSet(allocations={
                                'BTC': Decimal(str((bull_btc + bear_btc) / 2)),
                                'ETH': Decimal(str((bull_eth + bear_eth) / 2)),
                                'USDT': Decimal(str((bull_usdt + bear_usdt) / 2))
                            })
                        }
                        
                        allocation_space.append(allocations)
        
        # Limit size if too large
        if len(allocation_space) > 1000:
            # Random sample
            indices = np.random.choice(len(allocation_space), 1000, replace=False)
            allocation_space = [allocation_space[i] for i in indices]
        
        return allocation_space
    
    def _run_parallel_optimization(
        self,
        tasks: List[OptimizationTask],
        desc: str = "Optimizing"
    ) -> List[OptimizationResult]:
        """Run optimization tasks in parallel"""
        results = []
        
        with Pool(processes=self.n_jobs) as pool:
            # Use imap_unordered for better progress tracking
            with tqdm(total=len(tasks), desc=desc) as pbar:
                for result in pool.imap_unordered(self._run_single_backtest, tasks):
                    if result is not None:
                        results.append(result)
                        self.results.append(result)
                    pbar.update()
        
        return results
    
    @staticmethod
    def _run_single_backtest(task: OptimizationTask) -> Optional[OptimizationResult]:
        """Run a single backtest (static method for multiprocessing)"""
        try:
            # Create backtest engine
            engine = BacktestEngine(task.config, task.strategy_class, verbose=False)
            
            # Run backtest
            result = engine.run()
            
            # Calculate objective value
            objective = task.config.optimization.objective
            metrics = result['metrics']
            
            if objective == 'sharpe_ratio':
                objective_value = metrics.get('sharpe_ratio', -999)
            elif objective == 'total_return':
                objective_value = metrics.get('total_return', -999)
            elif objective == 'calmar_ratio':
                # Return / Max Drawdown
                total_return = metrics.get('total_return', 0)
                max_dd = abs(metrics.get('max_drawdown', 1))
                objective_value = total_return / max_dd if max_dd > 0 else -999
            else:
                objective_value = -999
            
            return OptimizationResult(
                parameters=task.parameters,
                backtest_result=result,
                objective_value=objective_value
            )
            
        except Exception as e:
            # Log error and continue
            print(f"Error in task {task.task_id}: {str(e)}")
            return None
    
    def _run_final_validation(
        self,
        params: Dict[str, Any],
        allocations: Dict[MarketState, AllocationSet]
    ) -> OptimizationResult:
        """Run final validation with best parameters"""
        config = self._create_config_with_params(params)
        
        # Set allocations if provided
        if allocations:
            config.strategy.allocations = allocations
        
        # Run backtest with verbose output
        engine = BacktestEngine(config, self.strategy_class, verbose=True)
        result = engine.run()
        
        # Calculate objective
        objective_value = self._calculate_objective(result['metrics'])
        
        return OptimizationResult(
            parameters=params,
            backtest_result=result,
            objective_value=objective_value
        )
    
    def _create_config_with_params(self, params: Dict[str, Any]) -> Config:
        """Create config with updated parameters"""
        # Deep copy base config
        config_dict = self.base_config.dict()
        
        # Update strategy parameters
        for param, value in params.items():
            if hasattr(config_dict['strategy']['indicators'], param):
                config_dict['strategy']['indicators'][param] = value
        
        return Config(**config_dict)
    
    def _calculate_objective(self, metrics: Dict[str, float]) -> float:
        """Calculate objective value from metrics"""
        objective = self.base_config.optimization.objective
        
        if objective == 'sharpe_ratio':
            return metrics.get('sharpe_ratio', -999)
        elif objective == 'total_return':
            return metrics.get('total_return', -999)
        elif objective == 'calmar_ratio':
            total_return = metrics.get('total_return', 0)
            max_dd = abs(metrics.get('max_drawdown', 1))
            return total_return / max_dd if max_dd > 0 else -999
        
        return -999
    
    def _get_default_parameters(self) -> Dict[str, Any]:
        """Get default strategy parameters"""
        return {
            'ema_short': self.base_config.strategy.indicators.ema_short,
            'ema_long': self.base_config.strategy.indicators.ema_long,
            'adx_period': self.base_config.strategy.indicators.adx_period,
            'adx_threshold': self.base_config.strategy.indicators.adx_threshold,
            'atr_period': self.base_config.strategy.indicators.atr_period
        }
    
    def _get_default_allocations(self) -> Dict[MarketState, AllocationSet]:
        """Get default allocations"""
        return self.base_config.strategy.allocations
    
    def _print_optimization_summary(self):
        """Print optimization results summary"""
        print("\n" + "=" * 80)
        print("Optimization Summary")
        print("=" * 80)
        
        if self.best_result:
            print(f"\nBest Parameters:")
            for param, value in self.best_result['parameters'].items():
                print(f"  {param:<20} {value}")
            
            print(f"\nBest Performance:")
            metrics = self.best_result['backtest_result']['metrics']
            print(f"  Objective ({self.base_config.optimization.objective}): {self.best_result['objective_value']:.4f}")
            print(f"  Total Return:      {metrics['total_return']*100:.2f}%")
            print(f"  Sharpe Ratio:      {metrics.get('sharpe_ratio', 0):.2f}")
            print(f"  Max Drawdown:      {metrics.get('max_drawdown', 0)*100:.2f}%")
            
        print("\nTop 5 Results:")
        sorted_results = sorted(self.results, key=lambda r: r['objective_value'], reverse=True)[:5]
        for i, result in enumerate(sorted_results, 1):
            print(f"\n  {i}. Objective: {result['objective_value']:.4f}")
            print(f"     Return: {result['backtest_result']['metrics']['total_return']*100:.2f}%")
    
    def save_results(self, filename: str):
        """Save optimization results to file"""
        # Convert results to serializable format
        data = {
            'config': self.base_config.dict(),
            'best_result': {
                'parameters': self.best_result['parameters'],
                'metrics': self.best_result['backtest_result']['metrics'],
                'objective_value': self.best_result['objective_value']
            } if self.best_result else None,
            'all_results': [
                {
                    'parameters': r['parameters'],
                    'metrics': r['backtest_result']['metrics'],
                    'objective_value': r['objective_value']
                }
                for r in self.results
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)