from abc import ABC, abstractmethod
import numpy as np
from config.settings import settings

class TradingEnvironment(ABC):
    """Abstract base class for all trading environments"""
    
    def __init__(self):
        self.balance = settings.INITIAL_BALANCE
        self.position = 0.0
        self.current_step = 0
        self.trade_history = []
    
    @abstractmethod
    def get_state(self) -> np.ndarray:
        """Return current market state"""
        pass
    
    @abstractmethod
    def reset(self):
        """Reset environment to initial state"""
        pass
    
    @abstractmethod
    def step(self, action: int) -> tuple:
        """Execute one time step"""
        pass
    
    @abstractmethod
    def execute_order(self, order_type: str, amount: float, price: float):
        """Execute a trade order"""
        pass