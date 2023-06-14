#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import gym
import random

class TaxiAgent:
    def __init__(self, state_size, action_size, alpha, gamma, epsilon, decay_rate):
        self.state_size = state_size
        self.action_size = action_size
        self.qtable = np.zeros((state_size, action_size))
        self.alpha = alpha # Learning rate
        self.gamma = gamma # Discount rate
        self.epsilon = epsilon # Exploration rate
        self.decay_rate = decay_rate 

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            # Explore
            return random.randrange(self.action_size) 
        else:
            # Exploit
            return np.argmax(self.qtable[state, :])

    def train(self, state, action, new_state, reward):
        self.qtable[state, action] = self.qtable[state, action] + self.alpha * (
            reward + self.gamma * np.max(self.qtable[new_state, :]) - self.qtable[state, action])

    def update_epsilon(self, episode):
        self.epsilon = np.exp(-self.decay_rate * episode)

class TaxiEnvironment:
    def __init__(self):
        self.env = gym.make('Taxi-v3')

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

def main():
    # Create Taxi environment
    env = TaxiEnvironment()

    # Initialize Agent
    state_size = env.env.observation_space.n
    action_size = env.env.action_space.n
    agent = TaxiAgent(state_size, action_size, alpha=0.9, gamma=0.8, epsilon=1.0, decay_rate=0.005)

    # Hyperparameters
    num_episodes = 1000
    max_steps = 99  

    # Train model
    for episode in range(num_episodes):
        
        state = env.reset()
        done = False

        for s in range(max_steps):

            action = agent.act(state)
            new_state, reward, done, info = env.step(action)
            agent.train(state, action, new_state, reward)
            state = new_state

            # If done, finish episode
            if done:
                break

        # Decrease exploration rate
        agent.update_epsilon(episode)

    print(f"Training completed over {num_episodes} episodes")
    input("Press Enter to watch trained agent...")

    # watch trained agent
    state = env.reset()
    done = False
    rewards = 0

    for s in range(max_steps):

        action = np.argmax(agent.qtable[state, :])
        new_state, reward, done, info = env.step(action)
        rewards += reward
        env.render()
        print(f"score: {rewards}")
        state = new_state

        if done:
            break

    env.close()

if __name__ == "__main__":
    main()

