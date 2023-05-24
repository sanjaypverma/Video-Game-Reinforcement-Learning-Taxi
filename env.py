import numpy as np
import gym
import random
import tensorflow as tf
from agent import agent
from inputs import path 

class environment():

	def __init__(self):

		self.env=gym.make('Taxi-v3')
		self.state=self.env.reset()
		self.state = tf.keras.utils.to_categorical(self.state,num_classes=self.env.observation_space.n)
		self.agent=agent()
		self.path=path

	def start_training(self): 

		num_episodes = 10000
		episode_rewards=[]


		for episodes in range(num_episodes):

			self.state=self.env.reset()		
			self.state = tf.keras.utils.to_categorical(self.state,num_classes=self.env.observation_space.n)

			done = False 
			total_reward=0
		
			while not done: 

#				env.render()
				
				action = self.agent.next_action(self.state)

				next_state, reward, done, _ = self.env.step(action)
				next_state = tf.keras.utils.to_categorical(next_state, num_classes=self.env.observation_space.n)
                
                self.agent.update_model(self.state, action, reward, next_state, done)
				self.state = next_state
				total_reward += reward
		
			total_rewards.append(total_reward)

			print(f'episode {episodes+1}/{total_episodes}-total reward {total_reward}')

		
	##save model

			if (episodes + 1) % 100 == 0:
				#agent=agent()
				self.agent.save_model()
				print("Model saved successfully.")

#		self.agent.save_model()
       		#print("Model saved successfully.")


	def continue_training(self):
		#agent=agent()
		self.agent.load_model()

	def average_rewards(self):

		avg_reward=sum(total_rewards)/num_episodes

		print(f"average reward over {num_episodes} evaluation episodes")


#	def end_environment(self):

#		close.env()



