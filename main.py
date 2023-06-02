## run code ##


from agent import agent 
from env import environment
from tensorflow import keras
from inputs import path, complete_model


env=environment()
env.start_training()
#env.continue_training()

