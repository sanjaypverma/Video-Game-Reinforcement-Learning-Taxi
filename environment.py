import numpy as np
import gym
import random

        
env = gym.make('Taxi-v3')

#initial state
state = env.reset()


total_steps = 1000

for i in range(total_steps): 
	action = env.action_space.sample()
	env.step(action)
	

close.env()
