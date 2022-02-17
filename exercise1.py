import gym
import numpy as np

env = gym.make('CartPole-v0')
observation_total = []
for i_episode in range(10):
    observation = env.reset()
    observation_total.append(observation)
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()

y = np.zeros(observation_total[0].shape)
for observation in observation_total:
    y+= observation
    
print()
print("observation total shape")    
print("observation total:")    
print(y)
