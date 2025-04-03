import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os  # Added for directory creation
from environments.backtest_env import BacktestEnvironment
from models.q_network import DQNAgent
from utils.data_fetcher import fetch_historical_data
from config.settings import settings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # Enable all logs
import tensorflow as tf

# Configure GPU memory growth and mixed precision
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

def main():
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    log_dir = os.path.dirname(settings.LOG_PATH)
    os.makedirs(log_dir, exist_ok=True)
    
    # Load data from CSV
    full_data = pd.read_csv('historical_data.csv')
    full_data['timestamp'] = pd.to_datetime(full_data['timestamp'])
   
    # Print the first few rows to verify
    print("Data Loaded:\n", full_data.head())

    train_data = full_data.iloc[:int(len(full_data)*0.8)]
    test_data = full_data.iloc[int(len(full_data)*0.8):]
    
    # Initialize components
    env = BacktestEnvironment(train_data)
    agent = DQNAgent(
        state_size=settings.STATE_SIZE,
        action_size=settings.ACTION_SIZE
    )
    
    # Training metrics
    metrics = {
        'episode_rewards': [],
        'portfolio_values': [],
        'positions': [],
        'drawdowns': []
    }
    
    # Training loop
    with tqdm(total=settings.EPISODES, desc="Training Progress") as pbar:
        for episode in range(settings.EPISODES):
            state = env.reset()
            done = False
            
            while not done:
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                agent.train()
                state = next_state
                
                # Track metrics
                metrics['portfolio_values'].append(env.portfolio_value)
                metrics['positions'].append(env.position)
                
                # Calculate drawdown
                current_value = env.portfolio_value
                peak = max(metrics['portfolio_values']) if metrics['portfolio_values'] else current_value
                drawdown = (peak - current_value) / peak
                metrics['drawdowns'].append(drawdown)
                
            # Update progress
            metrics['episode_rewards'].append(env.portfolio_value - settings.INITIAL_BALANCE)
            pbar.update(1)
            pbar.set_postfix({
                'Portfolio': f"{env.portfolio_value:.2f}",
                'Epsilon': f"{agent.epsilon:.3f}"
            })
            
            # Update target network and save periodically
            if episode % 10 == 0:
                agent.update_target_network()
                agent.model.save_weights(settings.MODEL_PATH)
    
    # Final save
    agent.model.save_weights(settings.MODEL_PATH)
    np.savez('training_metrics.npz', **metrics)
    print(f"\nModel saved to {settings.MODEL_PATH}")
    print(f"Metrics saved to training_metrics.npz")

def _plot_performance(metrics, test_data):
    """Safe plotting using Agg backend"""
    import matplotlib
    matplotlib.use('Agg')  # Non-GUI backend
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(15, 10))
    
    # Equity curve
    plt.subplot(2, 2, 1)
    plt.plot(metrics['portfolio_values'])
    plt.title("Portfolio Value Over Time")
    plt.xlabel("Step")
    plt.ylabel("Value (USDT)")
    
    # Position sizing
    plt.subplot(2, 2, 2)
    plt.plot(metrics['positions'])
    plt.title("Position Sizing History")
    plt.xlabel("Step")
    plt.ylabel("Position Size")
    
    # Return distribution
    plt.subplot(2, 2, 3)
    returns = np.diff(metrics['portfolio_values'])
    plt.hist(returns, bins=50)
    plt.title("Return Distribution")
    plt.xlabel("Return")
    plt.ylabel("Frequency")
    
    # Drawdown analysis
    plt.subplot(2, 2, 4)
    peak = np.maximum.accumulate(metrics['portfolio_values'])
    drawdown = (peak - metrics['portfolio_values']) / peak
    plt.plot(drawdown)
    plt.title("Maximum Drawdown")
    plt.xlabel("Step")
    plt.ylabel("Drawdown %")
    
    plt.tight_layout()
    plt.savefig('backtest_results.png')
    print("Saved plot to backtest_results.png")
    plt.close()

if __name__ == "__main__":
    main()
