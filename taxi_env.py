import numpy as np
import gym
import random

from agent import agent

class environment():

	def __init__(self):

		self.env=gym.make('Taxi-v3')
		self.state=env.rest()
		self.agent=agent()

		self.state = tf.keras.utils.to_categorical(state,num_classes=env.observation_space.n)


	def start_training(self): 

		num_episodes = 15

		total_rewards=[]

		done = False 
		total_reward=0
		
		while not done: 

			env.render()
			q_values=model.predict(np.expand_dims(state,axis=0))
			action = np.argmax(q_values)



			next_state, reward, done, info = env.step(action)
			next_state = tf.keras.utils.to_categorical(next_state, num_classes=env.obervation_space.n)        
	
			self.state = next_state
			total_reward += reward
		total_rewards.append(total_reward)

	def average_rewards(self):

		avg_reward=sum(total_rewards)/num_episodes

		print(f"average reward over {num_episodes} evaluation episodes")


	def end_environment(self):

		close.env()
