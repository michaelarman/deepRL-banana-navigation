# Report

## Deep Q-Learning Network (DQN) Algorithm
The aim of the algorithm is to train a policy that tries to maximize the discounted, cumulative reward.
The main idea behind Q-learning is that if we had a function Q∗:State×Action→R, that could tell us what our return would be, and
if we were to take an action in a given state, then we could easily construct a policy that maximizes our rewards. 
This function is unknown to us but since neural networks are universal function approximators, we can simply create one and train it to resemble the Q* function.

In this project we use a DQN with a Replay Buffer which stores the transitions that the agent observes,
allowing us to reuse this data later. By sampling from it randomly, the transitions that build up a batch are decorrelated. 
It has been shown that this greatly stabilizes and improves the DQN training procedure.

The Network used in this project has the following configuration:
- Fully Connected Linear Layer - input: 37 (state size) -->  output: 64 with RELU activation
- Fully Connected Linear Layer - input: 64 --> output: 64 with RELU activation
- Fully Connected Linear Layer - input: 64 --> output: 4 (action size)

The training parameters that worked best for me are:
- BUFFER_SIZE = int(1e5)  # replay buffer size
- BATCH_SIZE = 64         # minibatch size
- GAMMA = 0.99            # discount factor
- TAU = 1e-3              # for soft update of target parameters
- LR = 5e-4               # learning rate 
- UPDATE_EVERY = 4        # how often to update the network
- NUM_EPISODES = 2000     # maximum number of training episodes
- MAX_TIME = 2000         # maximum number of timesteps per episode
- EPS_START = 1.0         # starting value of epsilon, for epsilon-greedy action selection
- EPS_END = 0.01          # minimum value of epsilon
- EPS_DECAY = 0.995       # multiplicative factor (per episode) for decreasing epsilon

## Results

The model was able to solve the environment within 500 episodes 


![image](https://user-images.githubusercontent.com/46076665/109403478-c0cac180-792b-11eb-9937-233872fccc7c.png)

## Future ideas for improving the agent's performance
There are many ways to experiment with the agent's performance such as
- Parameter optimization
- Using a different RL algorithm
- Trying a different configuration of the Q-Network (e.g. more hidden layers)
- Trying another choice for the Replay Buffer 
