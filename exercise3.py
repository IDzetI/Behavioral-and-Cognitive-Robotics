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

    def compute_nparameters(self):
        return self.nhiddens * self.ninputs + self.noutputs * self.nhiddens + self.nhiddens + self.noutputs

    def set_genotype(self, genotype):
        border = self.nhiddens * self.ninputs
        self.W1 = genotype[:border].reshape(self.nhiddens, self.ninputs)
        self.W2 = genotype[border:border + self.noutputs * self.nhiddens].reshape(self.noutputs, self.nhiddens)
        border += self.noutputs * self.nhiddens
        self.b1 = genotype[border:border + self.nhiddens].reshape(self.nhiddens, 1)
        self.b2 = genotype[border + self.nhiddens:].reshape(self.noutputs, 1)

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

    def evaluate(self, n_episode=10, n=100, show=False):
        fitness = 0
        for i_episode in range(n_episode):
            observation = self.env.reset()
            for t in range(n):
                if show:
                    self.env.render()
                action = self.update(observation)
                observation, reward, done, info = self.env.step(action)
                fitness += reward
                if done:
                    break
        return fitness / n / n_episode


env = gym.make("CartPole-v0")
# env = gym.make("Acrobot-v1")
network = Network(env)

popsize = 10
variance = 0.1
perturb_variance = 0.02
ngeneration = 100
episodes = 3

nparameters = network.compute_nparameters()
population = np.random.randn(popsize, nparameters) * variance


for g in range(ngeneration):
    fitness = []
    # evaluating individual
    for i in range(popsize):
        network.set_genotype(genotype=population[i])
        fitness.append(network.evaluate(n_episode=episodes))

    # replacing the worst genotype with perturbed versions
    indexbest = np.argsort(fitness)
    for i in range(int(popsize / 2)):
        population[indexbest[i+int(popsize/2)]] = population[indexbest[i]] + np.random.rand(nparameters) * perturb_variance

    network.set_genotype(population[indexbest[0]])
    print('generation ', g, '\t fitness:', network.evaluate())
env.close()
