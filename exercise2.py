import gym
import numpy as np


class Network:
    def __init__(self, env):
        self.env = env
        self.pvariance = 0.1
        self.nhiddens = 5
        self.ninputs = env.observation_space.shape[0]
        self.action_space = env.action_space
        if isinstance(env.action_space, gym.spaces.box.Box):
            self.noutputs = env.action_space.shape[0]
        else:
            self.noutputs = env.action_space.n

    def init_parameters(self):
        # initialize the training parameters randomly by using a gaussian
        # distribution with average 0.0 and variance 0.1
        # biases (thresholds) are initialized to 0.0
        self.W1 = np.random.randn(self.nhiddens, self.ninputs) * self.pvariance  # first connection layer
        self.W2 = np.random.randn(self.noutputs, self.nhiddens) * self.pvariance  # second connection layer
        self.b1 = np.zeros(shape=(self.nhiddens, 1))  # bias internal neurons
        self.b2 = np.zeros(shape=(self.noutputs, 1))  # bias motor neurons

    def update(self, observation):
        # convert the observation array into a matrix with 1 column and ninputs rows
        observation.resize(self.ninputs, 1)
        # compute the netinput of the first layer of neurons
        Z1 = np.dot(self.W1, observation) + self.b1
        # compute the activation of the first layer of neurons with the tanh function
        A1 = np.tanh(Z1)
        # compute the netinput of the second layer of neurons
        Z2 = np.dot(self.W2, A1) + self.b2
        # compute the activation of the second layer of neurons with the tanh function
        A2 = np.tanh(Z2)
        # if the action is discrete
        #  select the action that corresponds to the most activated unit
        if isinstance(self.env.action_space, gym.spaces.box.Box):
            action = A2
        else:
            action = np.argmax(A2)
        return action

    def evaluate(self, n_episode=10, n=100):
        fitness = 0
        for i_episode in range(n_episode):
            observation = self.env.reset()
            for t in range(n):
                self.env.render()
                action = self.update(observation)
                observation, reward, done, info = self.env.step(action)
                fitness += reward
                if done:
                    break
        return fitness / n / n_episode


env = gym.make("CartPole-v0")
for i in range(100):
    network = Network(env)
    network.init_parameters()
    fitness = network.evaluate()
    print("Network:", i, "\tfitness: ", fitness)
env.close()
