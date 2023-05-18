class agent:
    
    def __init__(self, gamma = 0.9, epsilon = 0.9, epsilon_decay = 0.995, alpha = 0.1):
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        
        self.alpha = alpha
        
        self.model = Sequential([
            layers.Dense(128, input_shape = (2, ), activation = 'relu'),
            
            layers.Dense(64, activation = 'relu'),
            layer.Dropout(0.2),
            layers.Dense(32, activation = 'relu'),
            
            layers.Dense(6)
        ])
        
