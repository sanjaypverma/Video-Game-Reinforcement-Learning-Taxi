import numpy as np
import gym
import random

        
class environment():
    
    def __init__(self):
        
        self.env=gym.make('Taxi-v3')
        self.state=env.rest()
        self.agent=agent()
        
    
    def start_training(self): 
        
        ##if things start to go wrong we may need to do resize of the state

        done = False 
        
        while not done: 
            
            env.render()
            
            action = env.action_space.sample() 
            
            observation, reward, done, info = env.step(action)        
            
            self.state = obervation
            
    
    def end_environment(self):
        
        close.env()
