import gym
from util import current_time
import numpy as np

# env = gym.make('Enduro-v0')
env = gym.make('SpaceInvaders-v0')

env.reset()
print np.int32(2147483647)
env.seed(int(np.int32(2147483647)))
t0, t1 = 0, 0
print current_time()
while t1 < 10000:
    # env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    t0 += 1
    t1 += 1
    if done:
        print("Episode finished after {} timesteps".format(t0))
        # print np.max(observation)
        # for _ in xrange(10):
        #     a, b, c, d = env.step(action)
        #     print a.shape, b, c
        # t0 = 0
        env.reset()
print current_time()
