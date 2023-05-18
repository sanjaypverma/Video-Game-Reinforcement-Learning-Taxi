from os import path
import tensorflow as tf
from tensorflow.keras import layers
from keras.models import Sequential, load_model

import numpy as np
import random


class agent:
    
    def __init__(self, gamma = 0.9, epsilon = 0.9, epsilon_decay = 0.995, alpha = 0.1):
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        
        self.alpha = alpha
        
        self.model = Sequential([
            layers.Dense(128, input_shape = (500,500), activation = 'relu'),
            
            layers.Dense(64, activation = 'relu'),
            layer.Dropout(0.2),
            layers.Dense(32, activation = 'relu'),
            
            layers.Dense(6)
        ])
        
    def get_model_name(self):
        return 'taxi.h5'
    
    def save_model(self):
        self.model.save(self.get_model_name())
        
    def next_action(self, state):
        
        if random.random() <= self.epsilon:
            # If <= epsilon, perform a random action 
            # i.e. jump or don't jump (with equal probability)
            return random.randint(0, 5)
        else:
            # If > epsilon, perform a greedy action (use best known action)
            return self.Qfunction(state)
            
    def train(self, current_state, next_state, reward):
        
        Q_current_state = self.Qfunction(current_state)
        Q_next_state = self.Qfunction(next_state)
        
        Q_current_state = ((1-self.alpha) * Q_current_state) + (self.alpha * (reward + self.gamma * Q_next_state))
        
        # Pass through neural network
    
    def Qfunction(self, state):
        
        # Want to compute "something".predict(current_state)
        # return something (depending on what that something is may need to return some index... 
        # not entirely sure what is gonna get returned by the neural network)
        pass
    
    def update(self, current_state, next_state, reward):
        self.train(current_state, next_state, reward)
        self.epsilon *= self.epsilon_decay
