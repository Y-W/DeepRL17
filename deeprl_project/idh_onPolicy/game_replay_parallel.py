import os
import multiprocessing as mp
import ctypes
import numpy as np
import gym

from operator import mul


class AtariGame:
    def __init__(self, game_name, prng_seed, record_dir):
        self.env = gym.make(game_name)
        self.env.seed(prng_seed)
        if record_dir is not None:
            self.env = gym.wrappers.Monitor(self.env, record_dir, write_upon_reset=True, force=True)
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
    def __init__(self, game_name, record_dir,
                 frame_accum, 
                 frame_storage_processor, frame_storage_shape, frame_storage_type, 
                 frame_output_processor, frame_output_shape, frame_output_type,
                 trans_reward_processor, trans_max_count,
                 replay_size,
                 id, pipe, shared_arrs):
        super(AtariGame_ParallelWorker, self).__init__()
        self.game_name = game_name
        self.frame_accum = frame_accum
        self.frame_storage_processor = frame_storage_processor
        self.frame_storage_shape = frame_storage_shape
        self.frame_storage_type = frame_storage_type
        self.frame_output_processor = frame_output_processor
        self.frame_output_shape = frame_output_shape
        self.frame_output_type = frame_output_type
        self.trans_reward_processor = trans_reward_processor
        self.trans_max_count = trans_max_count
        self.replay_size = replay_size
        self.id = int(id)
        self.pipe = pipe
        self.shared_arrs = shared_arrs
        self.record_dir = record_dir

        self.prng = None
        self.game = None

        self.history_length = replay_size + frame_accum

        self.store_frame = None
        self.store_action = None
        self.store_reward = None
        self.store_step = None
        self.store_p = None
    
    def run_init(self):
        self.prng = np.random.RandomState(hash(self.game_name + str(self.id)) % 4294967296)
        self.game = AtariGame(self.game_name, int(self.prng.randint(4294967296)), self.record_dir)

        self.store_frame = np.zeros((self.history_length,) + self.frame_storage_shape, dtype=self.frame_storage_type)
        self.store_action = np.zeros((self.history_length,), dtype=np.uint8)
        self.store_reward = np.zeros((self.history_length,), dtype=np.float32)
        self.store_step = np.zeros((self.history_length,), dtype=np.uint16)
        self.store_p = 0

        self.store_frame[self.store_p] = self.frame_storage_processor(self.game.reset())
        self.store_step[self.store_p] = 0

        self.action_input = np.frombuffer(self.shared_arrs[0], dtype=np.int32)
        self.ob_output = np.frombuffer(self.shared_arrs[1], dtype=self.frame_output_type).reshape((-1, self.frame_accum) + self.frame_output_shape)
        self.rew_output = np.frombuffer(self.shared_arrs[2], dtype=np.float32)
        self.terminal_output = np.frombuffer(self.shared_arrs[3], dtype=np.bool_)

        self.trans_dist_input = np.frombuffer(self.shared_arrs[4], dtype=np.int32)
        self.trans_s = np.frombuffer(self.shared_arrs[5], dtype=self.frame_output_type).reshape((-1, self.trans_max_count, self.frame_accum) + self.frame_output_shape)
        self.trans_action = np.frombuffer(self.shared_arrs[6], dtype=np.int32).reshape((-1, self.trans_max_count))
        self.trans_reward = np.frombuffer(self.shared_arrs[7], dtype=np.float32).reshape((-1, self.trans_max_count))
        self.trans_terminal = np.frombuffer(self.shared_arrs[8], dtype=np.bool_).reshape((-1, self.trans_max_count))

        self.write_state(self.ob_output[self.id], self.store_p)
        self.rew_output[self.id] = self.store_reward[self.store_p]
        self.terminal_output[self.id] = (self.store_step[self.store_p] == 0)

    def write_state(self, arr, pos):
        assert pos == self.store_p or (pos - self.store_p) % self.history_length >= self.frame_accum
        terminated = False
        for i in xrange(self.frame_accum):
            if terminated:
                arr[self.frame_accum - 1 - i] = 0
            else:
                p = (pos - i) % self.history_length
                arr[self.frame_accum - 1 - i] = self.frame_output_processor(self.store_frame[p])
                terminated = (self.store_step[p] == 0)

    def _take_action(self, action):
        ob, rew, is_new_ep = self.game.action(action)
        self.store_p = (self.store_p + 1) % self.history_length
        self.store_frame[self.store_p] = self.frame_storage_processor(ob)
        self.store_action[self.store_p] = action
        self.store_reward[self.store_p] = rew
        if is_new_ep:
            self.store_step[self.store_p] = 0
        else:
            self.store_step[self.store_p] = self.store_step[(self.store_p - 1) % self.history_length] + 1
    
    def take_action(self):
        if self.action_input[self.id] >= 0:
            action = int(self.action_input[self.id])
            self._take_action(action)
            self.write_state(self.ob_output[self.id], self.store_p)
            self.rew_output[self.id] = self.store_reward[self.store_p]
            self.terminal_output[self.id] = (self.store_step[self.store_p] == 0)
    
    def write_trans(self, k, p):
        assert p != self.store_p
        self.write_state(self.trans_s[self.id, k], p)
        p = (p + 1) % self.history_length
        self.trans_action[self.id, k] = self.store_action[p]
        self.trans_reward[self.id, k] = self.trans_reward_processor(self.store_reward[p])

    def get_trans(self):
        low_bound = self.frame_accum
        high_bound = self.history_length - np.max(self.trans_dist_input)
        p = (self.prng.randint(high_bound - low_bound) + low_bound + self.store_p) % self.history_length

        for k in xrange(self.trans_max_count):
            ds = int(self.trans_dist_input[k])
            if ds >= 0:
                pp = (p + ds) % self.history_length
                self.trans_terminal[self.id, k] = (self.store_step[p] + ds != self.store_step[pp])
                if not self.trans_terminal[self.id, k]:
                    self.write_trans(k, pp)

    def run_close(self):
        self.game.close()
        del self.store_frame
        del self.store_action
        del self.store_reward
        del self.store_step
        
    def run(self):
        self.run_init()
        for _ in xrange(self.history_length):
            self._take_action(self.prng.randint(self.game.action_n))
        while True:
            command = self.pipe.recv()
            if command == 0:
                self.run_close()
                break
            elif command == 1:
                self.take_action()
                self.pipe.send(None)
            elif command == 2:
                self.get_trans()
                self.pipe.send(None)
            else:
                raise NotImplementedError()

class GameBatch_Parallel:
    def __init__(self, size, game_name, record_dir,
                 frame_accum, 
                 frame_storage_processor, frame_storage_shape, frame_storage_type, 
                 frame_output_processor, frame_output_shape, frame_output_type,
                 trans_reward_processor, trans_max_count,
                 replay_size):
        self.size = size

        self.state_shape = (frame_accum,) + frame_output_shape

        tmp_env = gym.make(game_name)
        self.action_n = tmp_env.action_space.n
        tmp_env.close()

        state_output_ctype = type(np.ctypeslib.as_ctypes(frame_output_type()))
        state_output_size = int(reduce(mul, self.state_shape, 1))

        self.shared_arrs = (mp.Array(ctypes.c_int32, size, lock=False),
                            mp.Array(state_output_ctype, size * state_output_size, lock=False),
                            mp.Array(ctypes.c_float, size, lock=False),
                            mp.Array(ctypes.c_bool, size, lock=False))

        self.shared_arrs += (mp.Array(ctypes.c_int32, trans_max_count, lock=False),
                             mp.Array(state_output_ctype, size * trans_max_count * state_output_size, lock=False),
                             mp.Array(ctypes.c_int32, size * trans_max_count, lock=False),
                             mp.Array(ctypes.c_float, size * trans_max_count, lock=False),
                             mp.Array(ctypes.c_bool, size * trans_max_count, lock=False))
        
        self.action_input = np.frombuffer(self.shared_arrs[0], dtype=np.int32)
        self.ob_output = np.frombuffer(self.shared_arrs[1], dtype=frame_output_type).reshape((-1, frame_accum) + frame_output_shape)
        self.rew_output = np.frombuffer(self.shared_arrs[2], dtype=np.float32)
        self.terminal_output = np.frombuffer(self.shared_arrs[3], dtype=np.bool_)

        self.trans_dist_input = np.frombuffer(self.shared_arrs[4], dtype=np.int32)
        self.trans_s = np.frombuffer(self.shared_arrs[5], dtype=frame_output_type).reshape((-1, trans_max_count, frame_accum) + frame_output_shape)
        self.trans_action = np.frombuffer(self.shared_arrs[6], dtype=np.int32).reshape((-1, trans_max_count))
        self.trans_reward = np.frombuffer(self.shared_arrs[7], dtype=np.float32).reshape((-1, trans_max_count))
        self.trans_terminal = np.frombuffer(self.shared_arrs[8], dtype=np.bool_).reshape((-1, trans_max_count))

        self.workers = []
        self.pipes = []
        for k in xrange(size):
            parent_conn, child_conn = mp.Pipe()
            self.pipes.append(parent_conn)

            worker = AtariGame_ParallelWorker(game_name, record_dir,
                        frame_accum, 
                        frame_storage_processor, frame_storage_shape, frame_storage_type, 
                        frame_output_processor, frame_output_shape, frame_output_type,
                        trans_reward_processor, trans_max_count,
                        replay_size,
                        k, child_conn, self.shared_arrs)
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

class GameEngine:
    def __init__(self, size, game_name, record_dir,
                 frame_accum, 
                 frame_storage_processor, frame_storage_shape, frame_storage_type, 
                 frame_output_processor, frame_output_shape, frame_output_type,
                 trans_reward_processor, trans_max_count,
                 replay_size):
        self.games = GameBatch_Parallel(size, game_name, record_dir,
                 frame_accum, 
                 frame_storage_processor, frame_storage_shape, frame_storage_type, 
                 frame_output_processor, frame_output_shape, frame_output_type,
                 trans_reward_processor, trans_max_count,
                 replay_size)
        self.state_shape = self.games.state_shape
        self.action_n = self.games.action_n
        self.trans_max_count = trans_max_count

    def __call__(self, actor_fn):
        self.games.action_input[:] = actor_fn(self.games.ob_output, self.action_n)
        self.games.take_action()
    
    def get_trans(self, dists):
        assert len(dists) <= self.trans_max_count
        assert min(dists) == 0
        for i in xrange(self.trans_max_count):
            if i < len(dists):
                self.games.trans_dist_input[i] = dists[i]
            else:
                self.games.trans_dist_input[i] = -1
        self.games.get_trans()
        return self.games.trans_s, self.games.trans_action, self.games.trans_reward, self.games.trans_terminal
    
    def samp_trans(self, dist):
        assert self.trans_max_count >= 2
        self.get_trans([0, dist])
    
    def close(self):
        self.games.close()
