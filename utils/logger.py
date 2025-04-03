import logging
from config.settings import settings

class TradingLogger:
    """Unified logging system"""
    
    def __init__(self, log_file=settings.LOG_PATH):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        
        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def log_order(self, order):
        self.logger.info(f"Order executed: {order}")
    
    def log_error(self, message):
        self.logger.error(f"Error: {message}")
    
    def log_performance(self, metrics):
        self.logger.info(f"Performance Update: {metrics}")

    def log_balance(self, usdt_balance: float, asset_balance: float, price: float):
        total_value = usdt_balance + (asset_balance * price)
        self.logger.info(
            f"Balance | USDT: {usdt_balance:.2f} | "
            f"{settings.SYMBOL.split('/')[0]}: {asset_balance:.4f} | "
            f"Total: {total_value:.2f} USDT"
        )

