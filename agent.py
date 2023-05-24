import numpy as np
import tensorflow as tf
from inputs import path

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
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.alpha)
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.model.compile(optimizer = self.optimizer,
                           loss = self.loss_fn,
                           metrics = ['accuracy'])


        
#	def stored_model_path(self):
#		path = '/Users/ilianamarrujo/computing16B/project/PIC16BProject/save_model'	
	 
	def save_model(self):
		self.model.save(path)

	def load_model(self): 
		self.model = tf.keras.models.load_model(path)

	     
	def next_action(self, state):
        
		if np.random.rand() <= epsilon:
			action = env.action_space.sample()
		else:
			q_values = model.predict(np.expand_dims(state, axis=0))
			action = np.argmax(q_values)
            
	def train(self, current_state, next_state, reward, done):
		target = reward
        
		if not done:
			target = (reward + self.gamma * Qfunction(next_state))
        
        
    
	def Qfunction(self, state):
		q_values = model.predict(np.expand_dims(state, axis=0))
		next_q_values = model.predict(np.expand_dims(next_state, axis=0))
		q_values[0][action] = reward + gamma * np.max(next_q_values)
    
	'''
	def update(self, current_state, next_state, reward):
		self.train(current_state, next_state, reward)
		self.epsilon *= self.epsilon_decay
	'''
