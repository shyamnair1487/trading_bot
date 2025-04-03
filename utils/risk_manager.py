class RiskManager:
    """Risk management utilities"""
    
    def calculate_position_size(self, balance: float, price: float, risk_pct: float) -> float:
        """Calculate position size based on risk percentage"""
        if price <= 0:
            return 0.0
        return (balance * risk_pct) / price
    
    def validate_order(self, amount: float, price: float, market_info: dict) -> bool:
        """Check if order meets exchange requirements"""
        min_amount = market_info['limits']['amount']['min']
        min_cost = market_info['limits']['cost']['min']
        
        if amount < min_amount:
            return False
        if (amount * price) < min_cost:
            return False
        return True