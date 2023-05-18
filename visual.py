import gym

env = gym.make('Taxi-v3')  # Create the Taxi environment

# Reset the environment to get the initial state
state = env.reset()

env.render()
