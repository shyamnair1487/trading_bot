from dataclasses import dataclass

@dataclass
class Settings:
    # Training Parameters
    EPISODES: int = 1000             # Number of training episodes
    BATCH_SIZE: int = 64             # Experience replay batch size
    MEMORY_SIZE: int = 10000         # Replay buffer capacity
    GAMMA: float = 0.99              # Discount factor
    EPS_START: float = 1.0           # Initial exploration rate
    EPS_MIN = 0.01  # Minimum exploration rate
    EPS_END: float = 0.01            # Minimum exploration rate
    EPS_DECAY: float = 0.995         # Exploration decay rate
    TAU: float = 0.005               # Target network update rate
    
    # Trading Parameters
    SYMBOL: str = "SOL/USDT"
    TIMEFRAME: str = "1m"
    INITIAL_BALANCE: float = 10000.0
    MAX_RISK_PCT: float = 0.05       # Max % of capital per trade
    FEE_RATE: float = 0.001          # Transaction fee percentage
    MAX_POSITION: float = 100.0      # Maximum allowed position size
    
    # Risk Management
    RISK_FREE_RATE: float = 0.0001   # For Sharpe ratio calculation
    DRAWDOWN_PENALTY: float = 2.0    # Drawdown penalty multiplier
    
    # Network Architecture
    STATE_SIZE: int = 7              # Must match state representation
    ACTION_SIZE: int = 3             # Buy, Sell, Hold
    
    # Paths
    MODEL_PATH: str = "models/trained_model.weights.h5"  # Must use .h5 extension
    LOG_PATH: str = "logs/trading.log"

settings = Settings()
