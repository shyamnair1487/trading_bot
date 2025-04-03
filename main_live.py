from environments.live_env import LiveTradingEnvironment
from models.q_network import DQNAgent
from config.settings import settings


def main():
    env = LiveTradingEnvironment()
    agent = DQNAgent()
    
    try:
        agent.model.load_weights(settings.MODEL_PATH)
    except:
        print("No saved model found, starting fresh")
    
    while True:
        state = env.get_state()
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.memory.add(state, action, reward, next_state, done)
        agent.train()
        
        # Save model periodically
        if env.current_step % 100 == 0:
            agent.model.save_weights(settings.MODEL_PATH)

if __name__ == "__main__":
    main()