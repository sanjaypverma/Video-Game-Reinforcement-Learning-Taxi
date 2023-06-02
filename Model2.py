### NO DIMENSION ERRORS BUT DOESN'T LEARN (CAN PLAY GAME) ###

import numpy as np
import gym
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = 1.0
        self.epsilon_decay = 0.005
        self.learning_rate = 0.75
        self.discount_rate = 0.8
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer='adam')
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state, verbose = 0)
        return np.argmax(q_values[0])

    def train(self, state, action, reward, next_state, done):
        target = reward + self.discount_rate * np.amax(self.model.predict(next_state, verbose = 0)[0])
        target_f = self.model.predict(state, verbose = 0)
        target_f[0][action] = target
        self.model.fit(state, target_f, epochs=1, verbose=0)
        if done:
            self.epsilon *= np.exp(-self.epsilon_decay)

class TaxiEnvironment:
    def __init__(self):
        self.env = gym.make('Taxi-v3')
        self.state_size = self.env.observation_space.n
        self.action_size = self.env.action_space.n

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def vectorize_state(self, state):
        return tf.keras.utils.to_categorical(state, num_classes=self.state_size).reshape(1, -1)

    
# create Taxi environment
env = TaxiEnvironment()

# create agent
agent = DQNAgent(env.state_size, env.action_size)

# training variables
num_episodes = 1
max_steps = 99

# training
for episode in range(num_episodes):
    #####
    total_reward = 0
    print("Episode: ", episode + 1)
    #####

    state = env.reset()
    state = env.vectorize_state(state)
    done = False

    for s in range(max_steps):
        action = agent.act(state)

        next_state, reward, done, info = env.step(action)
        next_state = env.vectorize_state(next_state)

        agent.train(state, action, reward, next_state, done)

        state = next_state

        ####
        total_reward += reward
        ####

        if done:
            break

    print(f"Total reward for Episode {episode + 1}: {total_reward}") ####

print(f"Training completed over {num_episodes} episodes")
input("Press Enter to watch trained agent...")

# watch trained agent
state = env.reset()
state = env.vectorize_state(state)
done = False
rewards = 0

for s in range(max_steps):
    print(f"TRAINED AGENT")
    print("Step {}".format(s+1))

    action = agent.act(state)
    next_state, reward, done, info = env.step(action)
    rewards += reward
    env.render()
    print(f"score: {rewards}")
    state = env.vectorize_state(next_state)

    if done:
        break

env.close()