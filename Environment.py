#Environment Class

import numpy as np
import gym
import random
import tensorflow as tf
from Agent import Agent

class Environment:
    def __init__(self, env_name):
        
        self.env = gym.make(env_name)
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
