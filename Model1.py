
import numpy as np
import gym
import random
import tensorflow as tf

class Agent:
	def __init__(self, state_size, action_size):
		self.state_size = state_size
		self.action_size = action_size
		


		# Create the Q-network
		self.model = tf.keras.Sequential([
				tf.keras.layers.Dense(32, activation='relu', input_shape=(state_size,)),
				tf.keras.layers.Flatten(),
				tf.keras.layers.Dense(16, activation='relu'),
				tf.keras.layers.Flatten(),
				tf.keras.layers.Dense(action_size, activation='linear')
		])
		self.model.compile(optimizer='adam', loss='mse')


	def act(self, state, epsilon):
		if np.random.rand() <= epsilon:
			# Explore
			return random.randrange(self.action_size)
		else:
			# Exploit
			state =  tf.keras.utils.to_categorical(state,num_classes=self.state_size)
#			state_shape = state.shape
#			print("State shape:", state_shape)
			q_values = self.model.predict(state, verbose = 0)

			return np.argmax(q_values[0])

	def train(self, state, action, reward, next_state, done, discount_rate, learning_rate):
		target = reward
		state =  tf.keras.utils.to_categorical(state,num_classes=self.state_size)
		next_state = tf.keras.utils.to_categorical(next_state,num_classes=self.state_size)

		state_shape = state.shape
#		next_state_shape = next_state.shape
#		print("State shape:", state_shape)
#		print("Next state shape:", next_state_shape)
		if not done:

			# Use the Q-network to estimate the target value
			target = (reward + discount_rate * np.amax(self.model.predict(next_state, verbose = 0)))
			target_f = self.model.predict(state, verbose = 0)
			target_f[0][action] = target

			# Train the Q-network with the updated target
			self.model.fit(state, target_f, epochs=1, verbose=0)

class Environment:
	def __init__(self, env_name):
		self.env = gym.make(env_name)
		self.state_size = self.env.observation_space.n
		self.action_size = self.env.action_space.n

	def reset(self):
		state = self.env.reset()
		state = np.array(state)
		return state

	def step(self, action):
		return self.env.step(action)


	def vectorize_state(self,state):
		return tf.keras.utils.to_categorical(state,num_classes=self.state_size).reshape(1,-1)

def main():
	#define path to save model 
#	complete_model = '/Users/ilianamarrujo/computing16B/project/PIC16BProject/save_complete_model'
#	saving_progress = '/Users/ilianamarrujo/computing16B/project/PIC16BProject/save_progress'
	
	# Create Taxi environment
	env_name = 'Taxi-v3'
	env = Environment(env_name)
#	state_size = env.state_size
#	initial_state = env.reset()
#	state_shape = initial_state.shape
#	print("Initial state shape:", state_shape)
#	print(state_size, 'state size')
	# Initialize agent
	agent = Agent(env.state_size, env.action_size)

	model = agent.model

	# Hyperparameters
	learning_rate = 0.9
	discount_rate = 0.8
	epsilon = 1.0
	decay_rate = 0.005

	# Training variables
	num_episodes = 15
	max_steps = 99  # per episode

	# Training
	for episode in range(num_episodes):

		total_reward = 0
		print("Episode: ", episode + 1)

		state = env.reset()

		state = tf.keras.utils.to_categorical(state,num_classes=env.env.observation_space.n)
		done = False

		for s in range(max_steps):
			
			action = agent.act(state, epsilon)
			next_state, reward, done, info = env.step(action)
			next_state = tf.keras.utils.to_categorical(next_state,num_classes=env.env.observation_space.n)

			agent.train(state, action, reward, next_state, done, discount_rate, learning_rate)
			state = next_state

			total_reward += reward

			if done:
				break

		if (episode + 1) % 2 == 0:
			tf.keras.models.save_model(model,filepath='/Users/ilianamarrujo/computing16B/project/PIC16BProject/save_progress')
			print("Model saved successfully.")

		print(f"Total reward for Episode {episode + 1}: {total_reward}") ####
		epsilon *= np.exp(-decay_rate * episode)


	print(f"Training completed over {num_episodes} episodes")

	tf.keras.models.save_model(model,filepath='/Users/ilianamarrujo/computing16B/project/PIC16BProject/save_complete_model')
	print('Full model saved successfully')
	
	input("Press Enter to watch trained agent...")

	# watch trained agent
	state = env.reset()
	state = env.vectorize_state(state)
	done = False
	rewards = 0

	for s in range(max_steps):
		print(f"TRAINED AGENT")
		print("Step {}".format(s+1))

		action = agent.act(state,epsilon)
		next_state, reward, done, info = env.step(action)
		rewards += reward
		env.env.render()
		print(f"score: {rewards}")
		state = env.vectorize_state(next_state)

		if done:
			break

#	tf.keras.models.save_model(model,filepath='/Users/ilianamarrujo/computing16B/project/PIC16BProject/save_complete_model')
#	print('Full model saved successfully')

	env.env.close()


if __name__ == "__main__":
	main()
