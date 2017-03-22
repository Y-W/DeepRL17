import gym
from util import current_time
from game_replay_parallel import process_frame_for_storage
import numpy as np
import scipy.misc

# env = gym.make('Enduro-v0')
env = gym.make('SpaceInvaders-v0')

env.reset()
print np.int32(2147483647)
# env.seed(int(np.int32(2147483647)))
t0, t1 = 0, 0
print current_time()
while t1 < 10000:
    # env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    if t0 == 200:
        scipy.misc.imsave('tmp1.png', observation)
        scipy.misc.imsave('tmp2.png', process_frame_for_storage(observation))
    t0 += 1
    t1 += 1
    if done:
        print("Episode finished after {} timesteps".format(t0))
        # print np.max(observation)
        # for _ in xrange(10):
        #     a, b, c, d = env.step(action)
        #     print a.shape, b, c
        # t0 = 0
        # env.reset()
        break
print current_time()
