import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from config.settings import settings
from utils.logger import TradingLogger

class BacktestEnvironment:
    """Enhanced backtesting environment with realistic market modeling"""
    
    def __init__(self, historical_data: pd.DataFrame):
        self.historical_data = historical_data
        self.current_step = 0
        self.balance = settings.INITIAL_BALANCE
        self.position = 0.0
        self.portfolio_value = settings.INITIAL_BALANCE
        self.logger = TradingLogger(settings.LOG_PATH)
        self.max_steps = len(historical_data) - 1
        
        # Price tracking
        self.current_price = historical_data['close'].iloc[0]
        self.previous_price = self.current_price
        
    def reset(self):
        """Reset environment to initial state"""
        self.current_step = 0
        self.balance = settings.INITIAL_BALANCE
        self.position = 0.0
        self.portfolio_value = settings.INITIAL_BALANCE
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get normalized market state"""
        row = self.historical_data.iloc[self.current_step]
        return np.array([
            row['open'] / 1e4,    # Normalized open price
            row['high'] / 1e4,    # Normalized high price
            row['low'] / 1e4,     # Normalized low price
            row['volume'] / 1e6,  # Normalized volume (millions)
            self.balance / settings.INITIAL_BALANCE,
            self.position / settings.MAX_POSITION,
            (self.current_price - self.previous_price) / self.previous_price  # Price change %
        ])
    
    def step(self, action: int) -> tuple:
        """Execute one step in the environment"""
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        # Store previous values
        self.previous_price = self.current_price
        self.current_price = self.historical_data['close'].iloc[self.current_step]
        
        # Execute trade
        self._process_action(action)
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Get next state
        next_state = self._get_state()
        
        return next_state, reward, done, {}

    def _process_action(self, action: int):
        """Execute trade with fee modeling"""
        if action == 0:  # Buy
            max_affordable = (self.balance * settings.MAX_RISK_PCT) / self.current_price
            amount = min(max_affordable, settings.MAX_POSITION - self.position)
            
            if amount > 0:
                cost = amount * self.current_price * (1 + settings.FEE_RATE)
                self.balance -= cost
                self.position += amount
                
        elif action == 1:  # Sell
            amount = min(self.position, settings.MAX_POSITION)
            
            if amount > 0:
                proceeds = amount * self.current_price * (1 - settings.FEE_RATE)
                self.balance += proceeds
                self.position -= amount

    def _calculate_reward(self) -> float:
        """Risk-adjusted reward function"""
        current_value = self.balance + (self.position * self.current_price)
        raw_return = current_value - self.portfolio_value
        
        # Drawdown penalty
        drawdown = max(0, (self.portfolio_value - current_value) / self.portfolio_value)
        
        # Position risk penalty
        position_penalty = abs(self.position) * 0.001
        
        # Volatility penalty
        price_change = abs(self.current_price - self.previous_price) / self.previous_price
        volatility_penalty = price_change * 0.5
        
        # Combine components
        reward = raw_return - (drawdown * 2) - position_penalty - volatility_penalty
        
        # Update portfolio value tracker
        self.portfolio_value = current_value
        
        return reward