import numpy as np
from environments.backtest_env import BacktestEnvironment
from models.q_network import DQNAgent
from utils.data_fetcher import fetch_historical_data
from config.settings import settings

def main():
    # Load data
    data = fetch_historical_data(settings.SYMBOL, settings.TIMEFRAME)
    
    # Initialize components
    env = BacktestEnvironment(data)
    agent = DQNAgent(
        state_size=settings.STATE_SIZE,
        action_size=settings.ACTION_SIZE
    )
    
    # Training loop
    for episode in range(1000):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.memory.add(state, action, reward, next_state, done)
            agent.train()
            total_reward += reward
            state = next_state
            
        # Update target network
        if episode % 10 == 0:
            agent.update_target_network()
            
        print(f"Episode {episode} | Total Reward: {total_reward:.2f}")

if __name__ == "__main__":
    main()