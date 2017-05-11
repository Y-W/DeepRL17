import os
import numpy as np

from mountain_car import MountainCar, state_bounds, action_n, visualize_states
from inc_learner import BinIncLearner
from framework import SARSA
from utils import Discretization

PARA_SIZE=32
REPLAY_SIZE=10000
DECAY=0.99
ALPHA=0.9

BIN_NUM=64

train_times=31250
vis_times=10

output_path='output/mountain_car_sarsa/'
if not os.path.exists(output_path):
    os.makedirs(output_path)

def random_action(_, n):
    return np.random.randint(n)


bin_shape = (BIN_NUM, BIN_NUM)
games = MountainCar(PARA_SIZE, REPLAY_SIZE)
inc_learner = BinIncLearner(Discretization(state_bounds, bin_shape), bin_shape, (action_n,), ALPHA)

framework = SARSA(games, inc_learner, random_action, DECAY)

for v in xrange(vis_times):
    framework.loop(train_times)
    visualize_states(output_path + 'states_%i.png' % v, inc_learner.eval_batch_no_default)

games.close()
