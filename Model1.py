#/usr/bin/env python
# coding: utf-8

# In[1]:


#Imports 

import numpy as np
import gym
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import seaborn as sns

from pathlib import Path
import time




#Path files 
###CHANGE THESE SO THEY ARE TRUE TO UR COMPUTER####

complete_model = '/Users/ilianamarrujo/computing16B/project/PIC16BProject/save_complete_model'
progress_model = '/Users/ilianamarrujo/computing16B/project/PIC16BProject/save_progress'

graph_data = '/Users/ilianamarrujo/computing16B/project/PIC16BProject/output_data/graph_data.csv'
save_animation = '/Users/ilianamarrujo/computing16B/project/PIC16BProject/output_data/trained_agent_animation.mp4'




#Agent Class

class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = 1.0
        self.epsilon_decay = 0.0001
        self.learning_rate = 0.7
        self.discount_rate = 0.8
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(32, activation='relu', input_shape=(self.state_size,)))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer='adam')
        return model

    
    def saved(self,ccomplete_model):
#             tf.keras.models.save_model(model,filepath=complete_model)
            self.model.save(complete_model)
            print('Full model saved successfully')
        
        
    def load(self,complete_model): 
        
        check_path = Path(complete_model)
  
        if check_path.exists():
            print('Saved model found. Loading existing model')
            self.model=tf.keras.models.load_model(complete_model)
            print('Model loaded')
            
        else:
            print('Building Model')
            self.model=self._build_model()
            
        
        
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        q_values = self.model.predict(state, verbose=0)
        
        return np.argmax(q_values[0])

    def train(self, state, action, reward, next_state, done):
        target = reward
        
        if not done:
            target = (reward + self.discount_rate * np.amax(self.model.predict(next_state, verbose=0)[0]))
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
            
        if done:
            self.epsilon *= np.exp(-self.epsilon_decay)




#Environment Class

class Environment:
    def __init__(self, env_name):
        
        self.env = gym.make(env_name)
        self.state_size = self.env.observation_space.n
        self.action_size = self.env.action_space.n

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def vectorize_state(self, state):
        return tf.keras.utils.to_categorical(state, num_classes=self.state_size).reshape(1, -1)




#Plotting Class

class Plotting:
    def __init__(self):
        self.total_actions = []
        self.plotting_rewards = []  




#Main 

def main():
    # Create Taxi environment
    env_name = 'Taxi-v3'
    env = Environment(env_name)

    plotinst = Plotting()
    # Initialize agent
    agent = Agent(env.state_size, env.action_size)

    model = agent.model
    
    agent.load(complete_model)
            
    

    # Hyperparameters
    num_episodes = 150
    max_steps = 1000
    decay_rate = 0.0001

    # Training
    for episode in range(num_episodes):
        total_reward = 0
        episode_actions = []
        print("Episode: ", episode + 1)

        state = env.reset()
        state = env.vectorize_state(state)
        done = False

        for s in range(max_steps):
            action = agent.act(state)
            episode_actions.append(action)
            next_state, reward, done, info = env.step(action)
            next_state = env.vectorize_state(next_state)

            agent.train(state, action, reward, next_state, done)

            state = next_state

            total_reward += reward

            if done:
                break

    #		if (episode + 1) % 2 == 0:
    #			tf.keras.models.save_model(model,filepath=progress_model)
    #			print("Model saved successfully.")	

        print(f"Total reward for Episode {episode + 1}: {total_reward}")

        plotinst.plotting_rewards.append(total_reward)
        plotinst.total_actions.append(episode_actions)

        agent.epsilon *= np.exp(-decay_rate * episode)

    print(f"Training completed over {num_episodes} episodes")
    
    

    agent.saved(complete_model)
    
    

    df = pd.DataFrame()
    df["Episode Number"] = np.arange(1,len(plotinst.plotting_rewards)+1)
    df["Rewards"] = plotinst.plotting_rewards
    df["Average Rewards"] = np.cumsum(plotinst.plotting_rewards) / np.arange(1,len(plotinst.plotting_rewards)+1)
    df["Actions"] = plotinst.total_actions
    
    
    df.to_csv(graph_data, index=False)

    #rewards graph
    sns.lineplot(data = df, x = "Episode Number", y = "Rewards")
    plt.title("Rewards")
    plt.show()
    
    #average rewards graph
    sns.lineplot(data = df, x = "Episode Number", y = "Average Rewards")
    plt.title("Average Rewards")
    plt.show()
    
    #both rewards on the same plot
    sns.lineplot(data = df, x = "Episode Number", y = "Rewards",label="Rewards")
    sns.lineplot(data = df, x = "Episode Number", y = "Average Rewards",label="Average Rewards")
    plt.title("Rewards & Average Rewards")
    plt.legend()
    plt.show()
    
    #histogram of actions in the first episode
    sns.histplot(df["Actions"][0], discrete=True)
    plt.title("histogram of actions in the first episode")
    plt.show()
    
    #histogram of actions in the last episode
    sns.histplot(df["Actions"][len(df["Actions"])-2], discrete=True)
    plt.title("histogram of actions in the last episode")
    plt.show()
    
    input("Press Enter to watch the trained agent...")

    time.sleep(10)

    # Watch trained agent
#     state = env.reset()
#     state = env.vectorize_state(state)
#     done = False
#     rewards = 0

#     for s in range(max_steps):
#         print("TRAINED AGENT")
#         print("Step {}".format(s + 1))

#         action = agent.act(state)
#         next_state, reward, done, info = env.step(action)
#         rewards += reward
#         env.render()
#         print(f"Score: {rewards}")
#         state = env.vectorize_state(next_state)

#         if done:
#             break



#####################
    state = env.reset()
    state = env.vectorize_state(state)
    done = False
    positionPlay = []
    rewards = 0
    
    for s in range(max_steps):
        
        positionPlay.append(state)
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        rewards += reward
        env.render()
        print(f"Score: {rewards}")
        state = env.vectorize_state(next_state)

        if done:
            break


#     def visualization(ss):
#         #clear previous screenshot 
#         plt.clf()
#         env.render()

#         state = positionPlay[ss]
#         agent_position = np.unravel_index(state.argmax(), state.shape)
#         plt.scatter(agent_position[1], agent_position[0], color='red', s=100)

#         plt.title(f"Step: {ss + 1}")

#     fig = plt.figure()
#     game_animation = animation.FuncAnimation(fig, visualization, frame_num=len(positionPlay), interval=500, repeat=False)
#     game_animation.save(save_animation, writer='ffmpeg')
    
#     plt.show()



#     state = env.reset()
#     done = False
#     rewards = 0

#     # this loop is for the animation so you can visually see
#     # how the agent is performing.
#     for s in range(max_steps):

#         print(f"TRAINED AGENT")
#         print("Step {}".format(s+1))

#         # exploit a known action, we'll only used the
#         # exploitation since the agent is aleady trained
#         action = agent.act(state)
#         # take the action in the environment
#         new_state, reward, done, info = env.step(action)
#         # update reward
#         rewards += reward
#         # update the screenshot of the environment
#         env.render()

#         print(f"score: {rewards}")
#         state = new_state

#         if done == True:
#             break
            
#         if done:
#             break
    env.close()


if __name__ == "__main__":
    main()








# In[ ]:




