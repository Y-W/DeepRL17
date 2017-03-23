import os
import multiprocessing as mp
import ctypes
import numpy as np
import gym

from util import list2NumpyBatches, numpyBatches2list

frame_size = (105, 80)
past_frame = 4
atari_game = 'SpaceInvaders-v0' # 'Enduro-v0'
action_n = 6

train_hist_len = 31250
def train_epsilon(k):
    return max(0.1, 1.0 - (0.9 / train_hist_len) * k)
def eval_epsilon(k):
    return 0.05


def process_frame_for_storage(f):
    f = np.mean(f, axis=2).round().astype(np.uint8)
    f = np.maximum(np.maximum(f[::2, ::2], f[::2, 1::2]), np.maximum(f[1::2, ::2], f[1::2, 1::2]))
    return f

def process_frame_for_output(f):
    f = f.astype(np.float32) * (1.0 / 255.0)
    return f


class AtariGame:
    def __init__(self, prng_seed, record_dir):
        self.env = gym.make(atari_game)
        self.env.seed(prng_seed)
        if record_dir is not None:
            self.env = gym.wrappers.Monitor(self.env, record_dir, force=True)
        self.action_n = self.env.action_space.n
    
    def reset(self):
        return self.env.reset()
    
    def action(self, action):
        ob, rew, done, _ = self.env.step(int(action))
        is_new_ep = False
        if done:
            ob = self.reset()
            is_new_ep = True
        return ob, rew, is_new_ep
    
    def close(self):
        self.env.close()


class AtariGame_ParallelWorker(mp.Process):
    def __init__(self, id, pipe, shared_arrs, is_eval, use_replay, record_dir):
        super(AtariGame_ParallelWorker, self).__init__()
        self.id = int(id)
        self.pipe = pipe
        self.shared_arrs = shared_arrs
        self.is_eval = is_eval
        self.use_replay = use_replay
        self.record_dir = record_dir

        self.prng = None
        self.game = None

        if not is_eval:
            if use_replay:
                self.history_length = train_hist_len + past_frame
            else:
                self.history_length = 1 + past_frame
            self.epsilon = train_epsilon
        else:
            self.history_length = past_frame
            self.epsilon = eval_epsilon

        self.store_frame = None
        self.store_action = None
        self.store_reward = None
        self.store_terminal = None
        self.store_p = None

        self.epsilon_cnt = None
    
    def run_init(self):
        self.prng = np.random.RandomState(hash(atari_game + str(self.id)) % 4294967296)
        self.game = AtariGame(int(self.prng.randint(4294967296)), self.record_dir)

        self.store_frame = np.zeros((self.history_length,) + frame_size, dtype=np.uint8)
        self.store_action = np.zeros((self.history_length,), dtype=np.uint8)
        self.store_reward = np.zeros((self.history_length,), dtype=np.float32)
        self.store_terminal = np.zeros((self.history_length,), dtype=np.bool_)
        self.store_p = 0

        self.epsilon_cnt = 0
        self.store_frame[self.store_p] = process_frame_for_storage(self.game.reset())
        self.store_terminal[self.store_p] = True

        self.action_input = np.frombuffer(self.shared_arrs[0], dtype=np.int32)
        self.ob_output = np.frombuffer(self.shared_arrs[1], dtype=np.float32).reshape((-1, past_frame) + frame_size)
        self.rew_output = np.frombuffer(self.shared_arrs[2], dtype=np.float32)
        self.terminal_output = np.frombuffer(self.shared_arrs[3], dtype=np.bool_)

        if not self.is_eval:
            self.trans_s0 = np.frombuffer(self.shared_arrs[4], dtype=np.float32).reshape((-1, past_frame) + frame_size)
            self.trans_action = np.frombuffer(self.shared_arrs[5], dtype=np.int32)
            self.trans_reward = np.frombuffer(self.shared_arrs[6], dtype=np.float32)
            self.trans_s1 = np.frombuffer(self.shared_arrs[7], dtype=np.float32).reshape((-1, past_frame) + frame_size)
            self.trans_terminal = np.frombuffer(self.shared_arrs[8], dtype=np.bool_)
        
        self.write_state(self.ob_output, self.store_p)
        self.rew_output[self.id] = self.store_reward[self.store_p]
        self.terminal_output[self.id] = self.store_terminal[self.store_p]
    
    def write_state(self, arr, pos):
        assert pos == self.store_p or (pos - self.store_p) % self.history_length >= past_frame
        terminated = False
        for i in xrange(past_frame):
            if terminated:
                arr[self.id, past_frame - 1 - i] = 0
            else:
                p = (pos - i) % self.history_length
                arr[self.id, past_frame - 1 - i] = process_frame_for_output(self.store_frame[p])
                terminated = self.store_terminal[p]

    def _take_action(self, action):
        ob, rew, is_new_ep = self.game.action(action)
        self.store_p = (self.store_p + 1) % self.history_length
        self.store_frame[self.store_p] = process_frame_for_storage(ob)
        self.store_action[self.store_p] = action
        self.store_reward[self.store_p] = rew
        self.store_terminal[self.store_p] = is_new_ep
    
    def take_action_epsilon(self):
        if self.action_input[self.id] >= 0:
            action = int(self.action_input[self.id])
            if self.prng.rand() < self.epsilon(self.epsilon_cnt):
                action = self.prng.randint(self.game.action_n)
            self.epsilon_cnt += 1
            self._take_action(action)
            self.write_state(self.ob_output, self.store_p)
            self.rew_output[self.id] = self.store_reward[self.store_p]
            self.terminal_output[self.id] = self.store_terminal[self.store_p]
    
    def get_trans(self):
        assert not self.is_eval
        if self.use_replay:
            p = (self.prng.randint(self.history_length - past_frame) + self.store_p + past_frame + 1) % self.history_length
        else:
            p = self.store_p
        self.write_state(self.trans_s0, (p-1) % self.history_length)
        self.trans_action[self.id] = self.store_action[p]
        if not self.is_eval:
            self.trans_reward[self.id] = np.clip(self.store_reward[p], -1, 1)
        else:
            self.trans_reward[self.id] = self.store_reward[p]
        if not self.store_terminal[p]:
            self.write_state(self.trans_s1, p)
        self.trans_terminal[self.id] = self.store_terminal[p]

    def run_close(self):
        self.game.close()
        del self.store_frame
        del self.store_action
        del self.store_reward
        del self.store_terminal
        
    def run(self):
        self.run_init()
        if not self.is_eval and self.use_replay:
            for _ in xrange(train_hist_len):
                self._take_action(self.prng.randint(self.game.action_n))
        while True:
            command = self.pipe.recv()
            if command == 0:
                self.run_close()
                break
            elif command == 1:
                self.take_action_epsilon()
                self.pipe.send(None)
            elif command == 2:
                self.get_trans()
                self.pipe.send(None)
            else:
                raise NotImplementedError()

class GameBatch_Parallel:
    def __init__(self, size, is_eval, use_replay, record_dir):
        self.size = size
        self.is_eval = is_eval
        self.use_replay = use_replay
        self.record_dir = record_dir

        self.state_shape = (past_frame,) + frame_size
        self.action_n = action_n

        self.shared_arrs = (mp.Array(ctypes.c_int32, size, lock=False),
                            mp.Array(ctypes.c_float, size * past_frame * frame_size[0] * frame_size[1], lock=False),
                            mp.Array(ctypes.c_float, size, lock=False),
                            mp.Array(ctypes.c_bool, size, lock=False))
        if not is_eval:
            self.shared_arrs += (mp.Array(ctypes.c_float, size * past_frame * frame_size[0] * frame_size[1], lock=False),
                                 mp.Array(ctypes.c_int32, size, lock=False),
                                 mp.Array(ctypes.c_float, size, lock=False),
                                 mp.Array(ctypes.c_float, size * past_frame * frame_size[0] * frame_size[1], lock=False),
                                 mp.Array(ctypes.c_bool, size, lock=False))
        
        self.action_input = np.frombuffer(self.shared_arrs[0], dtype=np.int32)
        self.ob_output = np.frombuffer(self.shared_arrs[1], dtype=np.float32).reshape((-1, past_frame) + frame_size)
        self.rew_output = np.frombuffer(self.shared_arrs[2], dtype=np.float32)
        self.terminal_output = np.frombuffer(self.shared_arrs[3], dtype=np.bool_)

        if not self.is_eval:
            self.trans_s0 = np.frombuffer(self.shared_arrs[4], dtype=np.float32).reshape((-1, past_frame) + frame_size)
            self.trans_action = np.frombuffer(self.shared_arrs[5], dtype=np.int32)
            self.trans_reward = np.frombuffer(self.shared_arrs[6], dtype=np.float32)
            self.trans_s1 = np.frombuffer(self.shared_arrs[7], dtype=np.float32).reshape((-1, past_frame) + frame_size)
            self.trans_terminal = np.frombuffer(self.shared_arrs[8], dtype=np.bool_)

        self.workers = []
        self.pipes = []
        for k in xrange(size):
            parent_conn, child_conn = mp.Pipe()
            self.pipes.append(parent_conn)
            worker = AtariGame_ParallelWorker(k, child_conn, self.shared_arrs, is_eval, use_replay, record_dir)
            self.workers.append(worker)
        
        for w in self.workers:
            w.start()
    
    def __del__(self):
        self.close()
    
    def close(self):
        for p in self.pipes:
            p.send(0)
        for w in self.workers:
            w.join()
        
    def take_action(self):
        for p in self.pipes:
            p.send(1)
        for p in self.pipes:
            p.recv()
    
    def get_trans(self):
        assert not self.is_eval
        for p in self.pipes:
            p.send(2)
        for p in self.pipes:
            p.recv()

class GameEngine_Train:
    def __init__(self, size, use_replay):
        self.size = size
        self.games = GameBatch_Parallel(size, False, use_replay, None)

    def __call__(self, actioner_fn):
        self.games.action_input[:] = actioner_fn(self.games.ob_output)
        self.games.take_action()
    
    def get_trans(self):
        self.games.get_trans()
        return self.games.trans_s0, self.games.trans_action, self.games.trans_reward, self.games.trans_s1, self.games.trans_terminal
    
    def close(self):
        self.games.close()

class GameEngine_Eval:
    def __init__(self, size):
        self.size = size
        self.games = GameBatch_Parallel(size, True, False, None)
    
    def run_episode(self, actioner_fn):
        total_scores = np.zeros(self.size, dtype=np.float32)
        episode_length = np.zeros(self.size, dtype=np.int32)
        running = np.ones(self.size, dtype=np.bool_)
        no_action = -np.ones(self.size, dtype=np.int32)
        while np.any(running):
            self.games.action_input[:] = np.where(running, actioner_fn(self.games.ob_output), no_action)
            self.games.take_action()
            total_scores += running * self.games.rew_output
            episode_length += running
            running = np.logical_and(running, np.logical_not(self.games.terminal_output))
        return total_scores.tolist(), episode_length.tolist()

    def __call__(self, actioner_fn, n):
        total_scores = []
        episode_length = []
        for _ in xrange(n):
            tr, el = self.run_episode(actioner_fn)
            total_scores.extend(tr)
            episode_length.extend(el)
        return total_scores, episode_length
    
    def close(self):
        self.games.close()

class GameEngine_Recorded:
    def __init__(self, record_dir, actioner_batch_size):
        if not os.path.exists(record_dir):
            os.makedirs(record_dir)
        self.actioner_batch_size = actioner_batch_size
        self.games = GameBatch_Parallel(1, True, False, record_dir)

    def __call__(self, actioner_fn):
        running = True
        aug_ob = np.zeros((actioner_batch_size,) + self.games.state_shape, dtype=np.float32)
        while running:
            aug_ob[0:1] = self.games.ob_output
            self.games.action_input[:] = actioner_fn(aug_ob)
            self.games.take_action()
            running = bool(self.games.terminal_output[0])
    
    def close(self):
        self.games.close()
