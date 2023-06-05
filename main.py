#Main 

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

from Agent import Agent
from Environment import Environment 
from Plotting import Plotting 
from inputs import progress_model, complete_model, graph_data, save_animation



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
    num_episodes = 3
    max_steps = 99
    decay_rate = 0.005

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


    def visualization(ss):
        #clear previous screenshot 
        plt.clf()
        env.render()

        state = agent_positions[ss]
        agent_position = np.unravel_index(state.argmax(), state.shape)
        plt.scatter(agent_position[1], agent_position[0], color='red', s=100)

        plt.title(f"Step: {ss + 1}")

    fig = plt.figure()
    game_animation = animation.FuncAnimation(fig, visualization, frame_num=len(agent_position), interval=500, repeat=False)
    game_animation.save(save_animation, writer='ffmpeg')
    
    plt.show()
    env.close()


if __name__ == "__main__":
    main()
