import numpy as np
import gym
import random
import tensorflow as tf
from pathlib import Path
from agent import agent
from inputs import path, complete_model


class environment():

	def __init__(self):

		self.env=gym.make('Taxi-v3')
		self.state=self.env.reset()
		self.state = tf.keras.utils.to_categorical(self.state,num_classes=self.env.observation_space.n)
		self.agent=agent()
		self.path=path
		self.complete_model=complete_model
		self.episode_rewards=[]
		self.total_actions = []

	def start_training(self): 

		
		check_path = Path(self.path)

		if check_path.exists():
			print('checkpoint found. loading existing model and picking up where we left off')
			self.agent.load(self.path)
			self.continue_training(from_load=True)

		else:
			print('No existing checkpoint found. Starting training from begining.')
			self.continue_training(from_load=False)

	def continue_training(self,from_load=False):
		
		num_episodes=10000

		if from_load: 
			print('continuing from loaded model')

		else: 
			print('starting training from beginning')

		for episode in range(num_episodes):

			self.state=self.env.reset()		
			#self.env.render()
			self.state = tf.keras.utils.to_categorical(self.state,num_classes=self.env.observation_space.n)

			done = False 
			total_reward=0
			episode_actions = []
			

			while not done: 
				action = self.agent.next_action(self.state)
				episode_actions.append(action)
				next_state, reward, done, _ = self.env.step(action)
				next_state = tf.keras.utils.to_categorical(next_state, num_classes=self.env.observation_space.n)
				self.env.render()
				self.agent.update_model(self.state, action, reward, next_state, done)
				self.state = next_state
				total_reward += reward


			self.episode_rewards.append(total_reward)
			self.total_actions.append(episode_actions)   
   
			print(f'Episode {episode+1}/{num_episodes}-Total reward {total_reward}')
   
			avg_reward=np.mean(self.episode_rewards)
			print(f"average reward over {len(self.episode_rewards)} evaluation episodes is {avg_reward} ")

			if (episode + 1) % 25 == 0:
				self.agent.model_save(self.path)
				print("Model saved successfully.")
			
		self.agent.save_complete_model(self.complete_path)
		print('Full model saved successfully')
		

#	def continue_training(self):
#		check_path = Path(self.path)
#		
#		if check_path.exists():
#			print('checkpoint found. loading existing model and picking up where we left off')
#			self.agent.load()
#			self.start_training()
#			
#			# loaded_model=tf.keras.models.save_model(self.agent.model,path,overwrite=False)
#			
#
#		else: 
#			print('No existing checkpoint found. Starting training from begining.')
#	
#			continue_train = environment()
#			continue_train.start_training()
#   

   


	# MAKE SURE TO CALL THIS AFTER TRAINING IS DONE!
	def end_environment(self):

		self.env.close()


