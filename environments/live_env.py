import ccxt
import numpy as np
from .base_env import TradingEnvironment
# from config import settings
from config.keys import BINANCE_API_KEY, BINANCE_API_SECRET
from utils.risk_manager import RiskManager
from utils.logger import TradingLogger
from config.settings import settings  # Correct

class LiveTradingEnvironment(TradingEnvironment):
    """Live exchange trading environment"""
    
    def __init__(self):
        super().__init__()
        self.exchange = ccxt.binance({
            'apiKey': BINANCE_API_KEY,
            'secret': BINANCE_API_SECRET,
            'enableRateLimit': True
        })

        # First load market data
        self.load_markets()
        
        # Then initialize components
        self.rm = RiskManager()
        self.logger = TradingLogger(settings.LOG_PATH)
        
        # Price tracking
        self.current_price = 0.0
        self.previous_price = 0.0
        
        # Balance tracking
        self.previous_balance = 0.0
        self.previous_position = 0.0
        
        # State management
        self.current_step = 0
        
        # Initialize with current exchange state
        try:
            self.reset()
        except Exception as e:
            self.logger.log_error(f"Initialization failed: {str(e)}")
            raise

    def reset(self):
        """Implement abstract method from base class"""
        self.current_step = 0
        self.balance = self.exchange.fetch_balance()['USDT']['free']
        self.position = self.exchange.fetch_balance()[settings.SYMBOL.split('/')[0]]['free']
        return self.get_state()
        
    def load_markets(self):
        self.exchange.load_markets()
        self.market = self.exchange.market(settings.SYMBOL)
        
    def get_state(self) -> np.ndarray:
        """Get real-time market state"""
        ticker = self.exchange.fetch_ticker(settings.SYMBOL)
        ohlcv = self.exchange.fetch_ohlcv(settings.SYMBOL, settings.TIMEFRAME)[-1]
        return np.array([
            ohlcv[1], ohlcv[2], ohlcv[3], ohlcv[4],
            ohlcv[5], self.balance, self.position
        ])

    def _calculate_reward(self):
        """Calculate reward based on portfolio value change"""
        current_value = self.balance + (self.position * self.current_price)
        previous_value = self.previous_balance + (self.previous_position * self.previous_price)
        reward = current_value - previous_value
        self.previous_balance = self.balance
        self.previous_position = self.position
        self.previous_price = self.current_price
        return reward
    
    def step(self, action: int) -> tuple:
        self.current_step += 1
        done = False  # Live trading never stops automatically
        
        # Store previous values before executing action
        self.previous_balance = self.balance
        self.previous_position = self.position
        self.previous_price = self.current_price
        
        # Execute action
        self._process_action(action)
        
        # Get new state
        next_state = self.get_state()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        return next_state, reward, done, {}
    
    def _process_action(self, action: int):
        ticker = self.exchange.fetch_ticker(settings.SYMBOL)
        current_price = ticker['last']
        
        try:
            if action == 0:  # Buy
                amount = self.rm.calculate_position_size(
                    self.balance, current_price, settings.MAX_RISK_PCT
                )
                if self.rm.validate_order(amount, current_price, self.market):
                    order = self.exchange.create_limit_buy_order(
                        settings.SYMBOL, amount, current_price
                    )
                    self.logger.log_order(order)
                    
            elif action == 1:  # Sell
                if self.position > 0:
                    order = self.exchange.create_limit_sell_order(
                        settings.SYMBOL, self.position, current_price
                    )
                    self.logger.log_order(order)
                    
        except Exception as e:
            self.logger.log_error(f"Order failed: {str(e)}")

    def execute_order(self, action: int):
        try:
            ticker = self.exchange.fetch_ticker(settings.SYMBOL)
            self.current_price = ticker['last']
            
            if action == 0:  # Buy
                amount = self.rm.calculate_position_size(
                    self.get_balance('USDT'), 
                    self.current_price,
                    settings.MAX_RISK_PCT
                )
                if self.rm.validate_order(amount, self.current_price, self.market):
                    order = self.exchange.create_limit_buy_order(
                        settings.SYMBOL, 
                        amount, 
                        self.current_price
                    )
                    if order['status'] == 'filled':
                        self.logger.log_order(order)
                        self._log_current_balance()
                        
            elif action == 1:  # Sell
                asset = settings.SYMBOL.split('/')[0]
                amount = self.get_balance(asset)
                if amount > 0:
                    order = self.exchange.create_limit_sell_order(
                        settings.SYMBOL, 
                        amount, 
                        self.current_price
                    )
                    if order['status'] == 'filled':
                        self.logger.log_order(order)
                        self._log_current_balance()
                        
        except Exception as e:
            self.logger.log_error(f"Order failed: {str(e)}")

    def _log_current_balance(self):
        usdt = self.get_balance('USDT')
        asset = self.get_balance(settings.SYMBOL.split('/')[0])
        self.logger.log_balance(usdt, asset, self.current_price)

    def get_balance(self, currency: str) -> float:
        balances = self.exchange.fetch_balance()
        return balances[currency]['free']