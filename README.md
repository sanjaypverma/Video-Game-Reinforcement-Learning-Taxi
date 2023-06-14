# PIC16BProject

The goal of this project was to teach a computer to play a video game through reinforcement learning. 

Initially, we wanted to solve a level of Geometry Dash but as a result of long training times we pivoted to a simpler and more well-documented game called "Taxi" that was available through the GYM API. Our new goal was to train a neural network to work through this game as efficiently as possible.

The three main technical components of this project were the agent, the environment, and the load/save implementation.

## **Agent:**

In practically all reinforcement models, the agent can be thought of as the entity that interacts with the environment to learn optimal behaviors. The agent’s goal is to maximize a predefined reward signal by learning from its experiences and adjusting its actions accordingly. In the Taxi game, the agent is the taxi-cab itself. The taxi-cab is allowed one of six possible actions within the environment: move north, south, east, west, pick up, or drop off the passenger. For each action taken by the agent, it receives a reward for its action from the environment (which we will specify later). As the model trains, it tries to maximize the reward it gets by looking for the best possible path to pick up and drop off the passenger. 

## **Envrionment:**

The environment class of our model was responsible for building the actual map through which the agent, or taxi, moved, as well as defining certain parameters such as the rewards for certain actions and the keys that correspond to certain inputs. The map shown above has 25 different positions where the taxi can be, 5 positions of the passenger (the four buildings in addition to being inside of the Taxi), and 4 dropoff locations, which combine for a discrete observation space of 500 states. However, each game has less than 500 reachable states that they will likely end up in as the dropoff location is always a different location than the initial location of the passenger. 

## **Load/Save**

Due to the computational power necessary to train our model, quite often we experienced GoogleColab breaking, our kernel dying, our computers going to sleep, or running out of space on our hard drives. For these reasons, it was extremely necessary to attempt to incorporate a save function where we could save the data as our model was training in the event one of the previously mentioned issues occurred.

## **Reflection**

Although our project did not go to plan, we were able to dive deeper into the world of AI and reinforcement learning, strengthening our programming skills and learning how to work new libraries such as TensorFlow. Neural networks are capable of learning virtually anything, but sometimes it is not the best way to approach a problem. As we have seen, we were able to train a computer and have the taxi move from space to space looking for the optimal path to pick up the person and drop them off, but our model was dumb and took an extremely long time, computational power, and storage to adequately train the game. As for ethical ramifications there are none. The point of this project was to have a computer play Taxi by finding the optimal path. This project was not intended to help the general public nor was it intended to help mankind. We strove to expand our programming knowledge and dive into the rising field of reinforcement learning. 
