class agent:
    
    def __init__(self, gamma = 0.9, epsilon = 0.9, epsilon_decay = 0.995, alpha = 0.1, state_size, action_size):
        
        self.state_size = state_size
        self.action_size = action_size
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        
        self.alpha = alpha
        
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(env.observation_space.n,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(env.action_space.n, activation='linear')
        ])
        
    def get_model_name(self):
        return 'taxi.h5'
    
    def save_model(self):
        self.model.save(self.get_model_name())
        
    def next_action(self, state):
        
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        else:
            # If > epsilon, perform a greedy action (use best known action)
            return self.Qfunction(state)
            
    def train(self, current_state, next_state, reward, done):
        target = reward
        
        if not done:
            target = (reward + self.gamma * Qfunction(next_state))
        
        
    
    def Qfunction(self, state):
        
        
        pass
    
    '''
    def update(self, current_state, next_state, reward):
        self.train(current_state, next_state, reward)
        self.epsilon *= self.epsilon_decay
    '''
