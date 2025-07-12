"""
Parameter search algorithms for optimization.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple
from itertools import product
import numpy as np
from scipy.stats import uniform, randint

from config.base_config import OptimizationConfig


class BaseParameterSearch(ABC):
    """Base class for parameter search algorithms"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        
    @abstractmethod
    def generate_parameter_space(self) -> List[Dict[str, Any]]:
        """Generate parameter combinations to test"""
        pass


class GridSearchOptimizer(BaseParameterSearch):
    """Grid search optimization"""
    
    def generate_parameter_space(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations in grid"""
        param_ranges = self.config.parameter_ranges
        
        # Create lists of values for each parameter
        param_grids = {}
        for param, range_config in param_ranges.items():
            min_val = range_config['min']
            max_val = range_config['max']
            step = range_config.get('step', 1)
            
            if isinstance(min_val, int) and isinstance(max_val, int):
                values = list(range(min_val, max_val + 1, step))
            else:
                values = list(np.arange(min_val, max_val + step/2, step))
            
            param_grids[param] = values
        
        # Generate all combinations
        param_names = list(param_grids.keys())
        param_values = list(param_grids.values())
        
        combinations = []
        for values in product(*param_values):
            combo = dict(zip(param_names, values))
            combinations.append(combo)
        
        # Filter invalid combinations (e.g., EMA short >= EMA long)
        valid_combinations = []
        for combo in combinations:
            if self._is_valid_combination(combo):
                valid_combinations.append(combo)
        
        return valid_combinations
    
    def _is_valid_combination(self, params: Dict[str, Any]) -> bool:
        """Check if parameter combination is valid"""
        # EMA constraint
        if 'ema_short' in params and 'ema_long' in params:
            if params['ema_short'] >= params['ema_long']:
                return False
        
        # MACD constraint
        if 'macd_fast' in params and 'macd_slow' in params:
            if params['macd_fast'] >= params['macd_slow']:
                return False
        
        return True


class RandomSearchOptimizer(BaseParameterSearch):
    """Random search optimization"""
    
    def generate_parameter_space(self) -> List[Dict[str, Any]]:
        """Generate random parameter combinations"""
        param_ranges = self.config.parameter_ranges
        n_iter = min(self.config.max_iterations, 1000)
        
        combinations = []
        attempts = 0
        max_attempts = n_iter * 10
        
        while len(combinations) < n_iter and attempts < max_attempts:
            attempts += 1
            combo = {}
            
            for param, range_config in param_ranges.items():
                min_val = range_config['min']
                max_val = range_config['max']
                
                if isinstance(min_val, int) and isinstance(max_val, int):
                    value = np.random.randint(min_val, max_val + 1)
                else:
                    value = np.random.uniform(min_val, max_val)
                
                combo[param] = value
            
            if self._is_valid_combination(combo):
                combinations.append(combo)
        
        return combinations
    
    def _is_valid_combination(self, params: Dict[str, Any]) -> bool:
        """Check if parameter combination is valid"""
        # Same as GridSearchOptimizer
        if 'ema_short' in params and 'ema_long' in params:
            if params['ema_short'] >= params['ema_long']:
                return False
        
        if 'macd_fast' in params and 'macd_slow' in params:
            if params['macd_fast'] >= params['macd_slow']:
                return False
        
        return True


class BayesianOptimizer(BaseParameterSearch):
    """
    Simplified Bayesian optimization using Gaussian Process
    Note: For production use, consider libraries like scikit-optimize
    """
    
    def __init__(self, config: OptimizationConfig):
        super().__init__(config)
        self.n_initial = 20  # Initial random samples
        self.results_history = []
    
    def generate_parameter_space(self) -> List[Dict[str, Any]]:
        """
        Generate initial random samples for Bayesian optimization
        Real implementation would generate new points based on GP posterior
        """
        # For now, just generate random initial points
        # In practice, this would be called iteratively with feedback
        random_optimizer = RandomSearchOptimizer(self.config)
        return random_optimizer.generate_parameter_space()[:self.n_initial]
    
    def suggest_next_parameters(
        self, 
        tested_params: List[Dict[str, Any]], 
        results: List[float]
    ) -> Dict[str, Any]:
        """
        Suggest next parameters to test based on results
        Simplified implementation - uses expected improvement
        """
        if len(tested_params) < self.n_initial:
            # Still in initial random phase
            random_optimizer = RandomSearchOptimizer(self.config)
            return random_optimizer.generate_parameter_space()[len(tested_params)]
        
        # Find best result so far
        best_idx = np.argmax(results)
        best_params = tested_params[best_idx]
        best_result = results[best_idx]
        
        # Generate candidates around best point
        candidates = self._generate_candidates_around(best_params, n_candidates=100)
        
        # Simple acquisition function (expected improvement proxy)
        # In practice, would use GP to predict mean and variance
        scores = []
        for candidate in candidates:
            # Distance-based score (closer to untested regions is better)
            distances = [self._param_distance(candidate, tested) for tested in tested_params]
            min_distance = min(distances)
            score = min_distance  # Favor exploration
            scores.append(score)
        
        # Return candidate with highest score
        best_candidate_idx = np.argmax(scores)
        return candidates[best_candidate_idx]
    
    def _generate_candidates_around(
        self, 
        center_params: Dict[str, Any], 
        n_candidates: int
    ) -> List[Dict[str, Any]]:
        """Generate candidate parameters around a center point"""
        candidates = []
        param_ranges = self.config.parameter_ranges
        
        for _ in range(n_candidates):
            candidate = {}
            for param, value in center_params.items():
                if param in param_ranges:
                    range_config = param_ranges[param]
                    min_val = range_config['min']
                    max_val = range_config['max']
                    
                    # Add noise proportional to range
                    noise_scale = (max_val - min_val) * 0.2
                    
                    if isinstance(value, int):
                        noise = int(np.random.normal(0, noise_scale))
                        new_value = np.clip(value + noise, min_val, max_val)
                        candidate[param] = int(new_value)
                    else:
                        noise = np.random.normal(0, noise_scale)
                        candidate[param] = np.clip(value + noise, min_val, max_val)
            
            if self._is_valid_combination(candidate):
                candidates.append(candidate)
        
        return candidates
    
    def _param_distance(self, params1: Dict[str, Any], params2: Dict[str, Any]) -> float:
        """Calculate normalized distance between parameter sets"""
        distance = 0
        param_ranges = self.config.parameter_ranges
        
        for param in params1:
            if param in params2 and param in param_ranges:
                range_config = param_ranges[param]
                range_size = range_config['max'] - range_config['min']
                
                if range_size > 0:
                    diff = abs(params1[param] - params2[param])
                    normalized_diff = diff / range_size
                    distance += normalized_diff ** 2
        
        return np.sqrt(distance)
    
    def _is_valid_combination(self, params: Dict[str, Any]) -> bool:
        """Check if parameter combination is valid"""
        if 'ema_short' in params and 'ema_long' in params:
            if params['ema_short'] >= params['ema_long']:
                return False
        
        if 'macd_fast' in params and 'macd_slow' in params:
            if params['macd_fast'] >= params['macd_slow']:
                return False
        
        return True
        