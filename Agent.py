#Agent Class

import numpy as np
import gym
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import seaborn as sns
from pathlib import Path

from inputs import progress_model, complete_model

class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = 1.0
        self.epsilon_decay = 0.005
        self.learning_rate = 0.95
        self.learning_rate = 0.9
        self.discount_rate = 0.8
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(32, activation='relu', input_shape=(self.state_size,)))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer='adam')
        return model

    
    def saved(self,ccomplete_model):
            self.model.save(complete_model)
            print('Full model saved successfully')
        
        
    def load(self,complete_model): 
        
        check_path = Path(complete_model)
  
        if check_path.exists():
            print('Saved model found. Loading existing model')
            self.model=tf.keras.models.load_model(complete_model)
            print('Model loaded')
            
        else:
            print('Building Model')
            self.model=self._build_model()
            
        
        
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        q_values = self.model.predict(state, verbose=0)
        
        return np.argmax(q_values[0])

    def train(self, state, action, reward, next_state, done):
        target = reward
        
        if not done:
            target = (reward + self.discount_rate * np.amax(self.model.predict(next_state, verbose=0)[0]))
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
            
        if done:
            self.epsilon *= np.exp(-self.epsilon_decay)
