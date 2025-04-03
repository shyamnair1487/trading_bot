import numpy as np
from environments.backtest_env import BacktestEnvironment
from models.q_network import DQNAgent
from utils.data_fetcher import fetch_historical_data  # Added missing import
from config.settings import settings

def generate_metrics():
    # Load test data
    data = fetch_historical_data(settings.SYMBOL, settings.TIMEFRAME)
    test_data = data.iloc[int(len(data)*0.8):]
    
    # Initialize components
    env = BacktestEnvironment(test_data)
    agent = DQNAgent(settings.STATE_SIZE, settings.ACTION_SIZE)
    agent.model.load_weights(settings.MODEL_PATH)
    
    # Initialize metrics
    metrics = {
        'portfolio_values': [],
        'positions': [],
        'episode_rewards': [],
        'drawdowns': []
    }
    
    # Run evaluation
    state = env.reset()
    done = False
    
    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        
        # Track metrics
        metrics['portfolio_values'].append(env.portfolio_value)
        metrics['positions'].append(env.position)
        metrics['episode_rewards'].append(reward)
        state = next_state
        
        # Calculate drawdown
        current_value = env.portfolio_value
        peak = max(metrics['portfolio_values']) if metrics['portfolio_values'] else current_value
        drawdown = (peak - current_value) / peak
        metrics['drawdowns'].append(drawdown)

    # Save metrics
    np.savez('test_metrics.npz', **metrics)
    print("Metrics saved to test_metrics.npz")

if __name__ == "__main__":
    generate_metrics()