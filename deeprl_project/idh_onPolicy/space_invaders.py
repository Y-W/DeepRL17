from game_replay_parallel import GameEngine

import numpy as np

frame_size = (105, 80)
past_frame = 4
atari_game = 'SpaceInvadersDeterministic-v3'

def process_frame_for_storage(f):
    f = np.mean(f, axis=2).round().astype(np.uint8)
    f = np.maximum(np.maximum(f[::2, ::2], f[::2, 1::2]), np.maximum(f[1::2, ::2], f[1::2, 1::2]))
    return f

def process_frame_for_output(f):
    f = f.astype(np.float32) * (1.0 / 255.0)
    return f

def clip_reward_fn(x):
    return np.clip(x, -1.0, 1.0)

def null_fn(x):
    return x

class SpaceInvaders(GameEngine):
    def __init__(self, parallel_size, replay_size, max_samp_cnt=2, clip_reward=False):
        if clip_reward:
            trans_reward_processor = clip_reward_fn
        else:
            trans_reward_processor = null_fn
        super(MountainCar, self).__init__(
            parallel_size, atari_game, None,
            past_frame, 
            process_frame_for_storage, frame_size, np.uint8, 
            process_frame_for_output, frame_size, np.float32,
            trans_reward_processor, max_samp_cnt,
            replay_size)
