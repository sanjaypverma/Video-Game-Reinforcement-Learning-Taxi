## run code & initilize params ##


from agent import agent 
from taxi_env import environment
from tensorflow import keras



env=environment()
env.start_training()

#agent.model.save("/Users/ilianamarrujo/computing16B/project/PIC16BProject/save_model")

