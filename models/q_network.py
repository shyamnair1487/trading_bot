import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, optimizers
from utils.memory import ReplayBuffer
from config.settings import settings

class DQNAgent:
    """Deep Q-Network Agent with Epsilon-Greedy Exploration"""
    
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(settings.MEMORY_SIZE)
        self.gamma = settings.GAMMA  # Discount factor
        self.epsilon = settings.EPS_START  # Exploration rate
        self.epsilon_min = settings.EPS_END
        self.epsilon_decay = settings.EPS_DECAY
        
        # Neural Networks
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_model.set_weights(self.model.get_weights())

    def _build_model(self):
        """Construct neural network architecture"""
        model = tf.keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(self.state_size,)),
            layers.BatchNormalization(),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.action_size)
        ])
        model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mse')
        return model

    def act(self, state):
        """Epsilon-greedy action selection"""
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)  # Random action
        q_values = self.model.predict(state[np.newaxis], verbose=0)
        return np.argmax(q_values[0])  # Best action

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.add(state, action, reward, next_state, done)

    def train(self):
        """Train network using experience replay"""
        if len(self.memory) < settings.BATCH_SIZE:
            return

        # Sample batch from memory
        states, actions, rewards, next_states, dones = self.memory.sample(settings.BATCH_SIZE)
        
        # Calculate target Q-values
        target_q = self.model.predict(states, verbose=0)
        next_q = self.target_model.predict(next_states, verbose=0)
        
        # Update targets
        for i in range(settings.BATCH_SIZE):
            if dones[i]:
                target_q[i][actions[i]] = rewards[i]
            else:
                target_q[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q[i])

        # Train model
        self.model.fit(states, target_q, batch_size=settings.BATCH_SIZE, verbose=0)
        
        # Decay exploration rate
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        """Sync target network with main network"""
        self.target_model.set_weights(self.model.get_weights())