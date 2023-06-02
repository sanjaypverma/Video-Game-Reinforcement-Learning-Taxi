#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import gym
import random

class TaxiAgent:
    def __init__(self, state_size, action_size, learning_rate, discount_rate, epsilon, decay_rate):
        self.state_size = state_size
        self.action_size = action_size
        self.qtable = np.zeros((state_size, action_size))
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.epsilon = epsilon
        self.decay_rate = decay_rate

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_size - 1)  # explore
        else:
            return np.argmax(self.qtable[state, :])  # exploit

    def update_qtable(self, state, action, new_state, reward):
        self.qtable[state, action] = self.qtable[state, action] + self.learning_rate * (
                reward + self.discount_rate * np.max(self.qtable[new_state, :]) - self.qtable[state, action])

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
    # create Taxi environment
    env = TaxiEnvironment()

    # initialize agent
    state_size = env.env.observation_space.n
    action_size = env.env.action_space.n
    agent = TaxiAgent(state_size, action_size, learning_rate=0.9, discount_rate=0.8, epsilon=1.0, decay_rate=0.005)

    # training variables
    num_episodes = 1000
    max_steps = 99  # per episode

    # training
    for episode in range(num_episodes):
        # reset the environment
        state = env.reset()
        done = False

        for s in range(max_steps):
            # choose action
            action = agent.choose_action(state)

            # take action and observe reward
            new_state, reward, done, info = env.step(action)

            # update Q-table
            agent.update_qtable(state, action, new_state, reward)

            # update to the new state
            state = new_state

            # if done, finish episode
            if done:
                break

        # decrease epsilon
        agent.update_epsilon(episode)

    print(f"Training completed over {num_episodes} episodes")
    input("Press Enter to watch trained agent...")

    # watch trained agent
    state = env.reset()
    done = False
    rewards = 0

    for s in range(max_steps):
        print(f"TRAINED AGENT")
        print("Step {}".format(s + 1))

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

