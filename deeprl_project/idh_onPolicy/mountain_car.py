from game_replay_parallel import GameEngine

import numpy as np

frame_size = (2,)
past_frame = 1
state_shape = (past_frame,) + frame_size
atari_game = 'MountainCar-v0'

def null_fn(x):
    return x

class MountainCar(GameEngine):
    def __init__(self, parallel_size, replay_size, max_samp_cnt=2):
        super(MountainCar, self).__init__(
            parallel_size, atari_game, None,
            past_frame, 
            null_fn, frame_size, np.float32, 
            null_fn, frame_size, np.float32,
            null_fn, max_samp_cnt,
            replay_size)
