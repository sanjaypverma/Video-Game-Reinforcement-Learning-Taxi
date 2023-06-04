import numpy as np
import tensorflow as tf
from inputs import path, complete_model

class agent:
	
	def __init__(self,  gamma = 0.9, epsilon = 0.9, epsilon_decay = 0.995, alpha = 0.01, state_size=500, action_size=6):

		self.state_size = state_size
		self.action_size = action_size
		self.gamma = gamma
		self.epsilon = epsilon
		self.epsilon_decay = epsilon_decay
		self.alpha = alpha
		
		self.model = tf.keras.Sequential([
				tf.keras.layers.Dense(128, activation='relu', input_shape=(self.state_size,)),
				tf.keras.layers.Dense(64, activation = 'relu'),
				tf.keras.layers.Dropout(0.2),
				tf.keras.layers.Dense(32, activation = 'linear'), # Potentially change to ReLu
				tf.keras.layers.Dense(self.action_size, activation = 'linear')
				])

		self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.alpha,jit_compile=True)
		self.loss_fn = tf.keras.losses.MeanSquaredError()
		self.model.compile(optimizer = self.optimizer,
					   loss = self.loss_fn,
					   metrics = ['accuracy'])


	def model_save(self,path):
		tf.keras.models.save_model(self.model,path)
    
	def load(self,path): 
		self.model=tf.keras.models.load_model(path)

	def save_complete_model(self,complete_model): 
		tf.keras.models.save_model(self.model,complete_model,overwrite=True)
	     
	def next_action(self, state):
		
		if np.random.rand() <= self.epsilon:
			action = np.random.randint(self.action_size)
		else:
			q_values = self.model.predict(np.expand_dims(state, axis = 0), verbose=0)
			action = np.argmax(q_values)

		return action


	def update_model(self, state, action, reward, next_state, done):
        
		q_values = self.model.predict(np.expand_dims(state, axis = 0), verbose=0)
		next_q_values = self.model.predict(np.expand_dims(next_state, axis = 0),verbose=0)
		q_values[0][action] = reward + self.gamma * np.max(next_q_values)

		with tf.GradientTape() as tape:
			q_values_pred = self.model(np.expand_dims(state, axis = 0))

			loss_value= self.loss_fn(q_values_pred, q_values)

		grads = tape.gradient(loss_value, self.model.trainable_variables)
		self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
	
		if done:
			self.epsilon *= self.epsilon_decay
