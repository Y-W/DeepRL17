from __future__ import division

import os
import numpy as np

from mountain_car import MountainCar, state_bounds, action_n, visualize_states
from batch_learner import BinBatchLearner
from framework import IDH_BASE, IDH_STEP
from utils import Discretization

PARA_SIZE=32
REPLAY_SIZE=10000
DECAY=0.99
ALPHA=0.9

BIN_NUM=64

train_times=31250
vis_times=10

output_path='output/mountain_car_idh/'
if not os.path.exists(output_path):
    os.makedirs(output_path)

def random_action(_, n):
    return np.random.randint(n)

bin_shape = (BIN_NUM, BIN_NUM)
games = MountainCar(PARA_SIZE, REPLAY_SIZE)

horizon=1
batch_learner = BinBatchLearner(Discretization(state_bounds, bin_shape), bin_shape, (action_n,))
framework = IDH_BASE(games, batch_learner, random_action)
framework.loop(train_times)
visualize_states(output_path + 'states_%i.png' % horizon, batch_learner.eval_batch_no_default)

horizon=2
eval_tmp = batch_learner.eval_batch
batch_learner = BinBatchLearner(Discretization(state_bounds, bin_shape), bin_shape, (action_n,))
framework = IDH_STEP(games, batch_learner, random_action, eval_tmp, horizon//2, DECAY)
framework.loop(train_times)
visualize_states(output_path + 'states_%i.png' % horizon, batch_learner.eval_batch_no_default)

horizon=4
eval_tmp = batch_learner.eval_batch
batch_learner = BinBatchLearner(Discretization(state_bounds, bin_shape), bin_shape, (action_n,))
framework = IDH_STEP(games, batch_learner, random_action, eval_tmp, horizon//2, DECAY)
framework.loop(train_times)
visualize_states(output_path + 'states_%i.png' % horizon, batch_learner.eval_batch_no_default)

horizon=8
eval_tmp = batch_learner.eval_batch
batch_learner = BinBatchLearner(Discretization(state_bounds, bin_shape), bin_shape, (action_n,))
framework = IDH_STEP(games, batch_learner, random_action, eval_tmp, horizon//2, DECAY)
framework.loop(train_times)
visualize_states(output_path + 'states_%i.png' % horizon, batch_learner.eval_batch_no_default)

horizon=16
eval_tmp = batch_learner.eval_batch
batch_learner = BinBatchLearner(Discretization(state_bounds, bin_shape), bin_shape, (action_n,))
framework = IDH_STEP(games, batch_learner, random_action, eval_tmp, horizon//2, DECAY)
framework.loop(train_times)
visualize_states(output_path + 'states_%i.png' % horizon, batch_learner.eval_batch_no_default)

horizon=32
eval_tmp = batch_learner.eval_batch
batch_learner = BinBatchLearner(Discretization(state_bounds, bin_shape), bin_shape, (action_n,))
framework = IDH_STEP(games, batch_learner, random_action, eval_tmp, horizon//2, DECAY)
framework.loop(train_times)
visualize_states(output_path + 'states_%i.png' % horizon, batch_learner.eval_batch_no_default)

horizon=64
eval_tmp = batch_learner.eval_batch
batch_learner = BinBatchLearner(Discretization(state_bounds, bin_shape), bin_shape, (action_n,))
framework = IDH_STEP(games, batch_learner, random_action, eval_tmp, horizon//2, DECAY)
framework.loop(train_times)
visualize_states(output_path + 'states_%i.png' % horizon, batch_learner.eval_batch_no_default)

horizon=128
eval_tmp = batch_learner.eval_batch
batch_learner = BinBatchLearner(Discretization(state_bounds, bin_shape), bin_shape, (action_n,))
framework = IDH_STEP(games, batch_learner, random_action, eval_tmp, horizon//2, DECAY)
framework.loop(train_times)
visualize_states(output_path + 'states_%i.png' % horizon, batch_learner.eval_batch_no_default)

horizon=256
eval_tmp = batch_learner.eval_batch
batch_learner = BinBatchLearner(Discretization(state_bounds, bin_shape), bin_shape, (action_n,))
framework = IDH_STEP(games, batch_learner, random_action, eval_tmp, horizon//2, DECAY)
framework.loop(train_times)
visualize_states(output_path + 'states_%i.png' % horizon, batch_learner.eval_batch_no_default)

horizon=512
eval_tmp = batch_learner.eval_batch
batch_learner = BinBatchLearner(Discretization(state_bounds, bin_shape), bin_shape, (action_n,))
framework = IDH_STEP(games, batch_learner, random_action, eval_tmp, horizon//2, DECAY)
framework.loop(train_times)
visualize_states(output_path + 'states_%i.png' % horizon, batch_learner.eval_batch_no_default)

horizon=1024
eval_tmp = batch_learner.eval_batch
batch_learner = BinBatchLearner(Discretization(state_bounds, bin_shape), bin_shape, (action_n,))
framework = IDH_STEP(games, batch_learner, random_action, eval_tmp, horizon//2, DECAY)
framework.loop(train_times)
visualize_states(output_path + 'states_%i.png' % horizon, batch_learner.eval_batch_no_default)


games.close()
