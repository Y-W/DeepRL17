import os
import multiprocessing as mp
import numpy as np
import gym

from util import list2NumpyBatches, numpyBatches2list

frame_size = (105, 80)
past_frame = 4
atari_game = 'SpaceInvaders-v0' # 'Enduro-v0'

train_hist_len = 32768
def train_epsilon(k):
    return max(0.1, 1.0 - 0.9e-6 * k)
def eval_epsilon(k):
    return 0.05
train_hist_len = 31250 + past_frame - 1
eval_hist_len = past_frame


def process_frame_for_storage(f):
    f = np.mean(f, axis=2).round().astype(np.uint8)
    f = np.maximum(np.maximum(f[::2, ::2], f[::2, 1::2]), np.maximum(f[1::2, ::2], f[1::2, 1::2]))
    return f

def process_frame_for_output(f):
    f = f.astype(np.float32) * (1.0 / 255.0)
    return f


class AtariGame:
    def __init__(self, game_name, prng_seed, record_dir):
        self.env = gym.make(game_name)
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
    def __init__(self, id, pipe, is_eval, record_dir):
        super(AtariGame_ParallelWorker, self).__init__()
        self.id = id
        self.pipe = pipe
        self.is_eval = self.is_alive

        self.prng = None
        self.game = None

        self.record_dir = record_dir

        if not is_eval:
            self.history_length = train_hist_len
            self.epsilon = train_epsilon
        else:
            self.history_length = eval_hist_len
            self.epsilon = eval_epsilon

        self.store_frame = None
        self.store_action = None
        self.store_reward = None
        self.store_terminal = None
        self.store_p = None

        self.action_cnt = None
    
    def run_init(self):
        self.prng = np.random.RandomState(hash(atari_game + str(self.id)) % 4294967296)
        self.game = AtariGame(atari_game, int(self.prng.randint(4294967296)), self.record_dir)

        self.store_frame = np.zeros((self.history_length,) + frame_size, dtype=np.uint8)
        self.store_action = np.zeros((self.history_length,), dtype=np.uint8)
        self.store_reward = np.zeros((self.history_length,), dtype=np.float32)
        self.store_terminal = np.zeros((self.history_length,), dtype=np.bool_)
        self.store_p = 0

        self.action_cnt = 0
        self.store_frame[self.store_p] = process_frame_for_storage(self.game.reset())
        self.store_terminal[self.store_p] = True
    
    def get_state(self, pos):
        arr = np.zeros((past_frame,) + frame_size), dtype=np.float32)
        for i in xrange(past_frame):
            p = (pos - i) % self.history_length
            arr[past_frame - 1 - i] = process_frame_for_output(self.store_frame[p])
            if self.store_terminal[p]:
                break
        return arr

    def take_action(self, action):
        if action is None:
            return None
        if self.prng.rand() < self.epsilon(self.action_cnt):
            action = self.prng.randint(self.game.action_n)
        ob, rew, is_new_ep = self.game.action(action)
        self.action_cnt += 1
        self.store_p = (self.store_p + 1) % self.history_length
        self.store_frame[self.store_p] = process_frame_for_storage(ob)
        self.store_action[self.store_p] = action
        self.store_reward[self.store_p] = rew
        self.store_terminal[self.store_p] = is_new_ep
        return rew, is_new_ep
    
    def get_ob(self):
        return self.get_state(self.store_p)
    
    def samp_trans(self):
        p = (self.prng.randint(self.history_length - past_frame + 1) + self.store_p + past_frame) % self.history_length
        s0 = self.get_state((p-1) % self.history_length)
        act = self.store_action[p]
        rew = self.store_reward[p]
        if self.clip_reward:
            rew = np.clip(rew, -1, 1)
        if self.store_terminal[p]:
            s1 = None
        else:
            s1 = self.get_state(p)
        return (s0, act, rew, s1)
    
    def last_trans(self):
        p = self.store_p
        s0 = self.get_state((p-1) % self.history_length)
        act = self.store_action[p]
        rew = self.store_reward[p]
        if self.clip_reward:
            rew = np.clip(rew, -1, 1)
        if self.store_terminal[p]:
            s1 = None
        else:
            s1 = self.get_state(p)
        return (s0, act, rew, s1)
    
    def take_rand_action(self, n):
        for _ in xrange(n):
            action = self.prng.randint(self.game.action_n)
            self.take_action(action)

    def run_close(self):
        self.game.close()
        del self.store_frame
        del self.store_action
        del self.store_reward
        del self.store_terminal
        
    def run(self):
        self.run_init()
        while True:
            command, arg = self.pipe.recv()
            if command == 0:
                self.run_close()
                break
            elif command == 1:
                self.pipe.send(self.get_ob())
            elif command == 2:
                self.pipe.send(self.take_action(arg))
            elif command == 3:
                self.pipe.send(self.samp_trans())
            elif command == 4:
                self.take_rand_action(arg)
            elif command == 5:
                self.pipe.send(self.last_trans())
            else:
                raise NotImplementedError()

class GameBatch_Parallel:
    def __init__(self, size,
                 max_episode, record_dir,
                 history_length, clip_reward
                 ):
        self.size = size

        sample_game = gym.make(atari_game)
        self.state_shape = (past_frame, frame_size, frame_size)
        self.action_n = sample_game.action_space.n
        sample_game.close()

        self.workers = []
        self.pipes = []
        for k in xrange(size):
            parent_conn, child_conn = mp.Pipe()
            self.pipes.append(parent_conn)
            worker = AtariGame_ParallelWorker(k, child_conn, 
                                              atari_game, max_episode, record_dir,
                                              history_length, clip_reward, epsilon_for_all)
            worker.start()
            self.workers.append(worker)
    
    def __del__(self):
        self.close()
    
    def close(self):
        for p in self.pipes:
            p.send((0, None))
        for w in self.workers:
            w.join()
    
    def get_ob(self):
        for p in self.pipes:
            p.send((1, None))
        result = []
        for p in self.pipes:
            result.append(p.recv())
        return result
    
    def take_action(self, actions):
        for p, a in zip(self.pipes, actions):
            p.send((2, a))
        result = []
        for p in self.pipes:
            result.append(p.recv())
        return result
    
    def samp_trans(self):
        for p in self.pipes:
            p.send((3, None))
        result = []
        for p in self.pipes:
            result.append(p.recv())
        return result
    
    def take_rand_action(self, n):
        for p in self.pipes:
            p.send((4, n))
    
    def last_trans(self):
        for p in self.pipes:
            p.send((5, None))
        result = []
        for p in self.pipes:
            result.append(p.recv())
        return result


class GameEngine_Train:
    def __init__(self, size, history_length):
        self.size = size
        self.game_batch = GameBatch_Parallel(size, max_episode_train, None, history_length, clip_reward_on_train)
        self.game_batch.take_rand_action(history_length)

    def __call__(self, actioner_fn, actioner_batch_size):
        states = self.game_batch.get_ob()
        state_batches = list2NumpyBatches(states, actioner_batch_size)
        action_batches = [actioner_fn(states) for states in state_batches]
        actions = numpyBatches2list(action_batches, self.size)
        actions = [int(a) for a in actions]
        self.game_batch.take_action(actions)
    
    def sample(self, n):
        assert n == self.size
        return self.game_batch.samp_trans()
    
    def last_trans(self, n):
        assert n == self.size
        return self.game_batch.last_trans()

class GameEngine_Eval:
    def __init__(self, size):
        self.size = size
        self.game_batch = GameBatch_Parallel(size, None, None, None, False)
    
    def __call__(self, actioner_fn, actioner_batch_size):
        total_score = [0.0] * self.size
        running = [True] * self.size
        while any(running):
            obs = self.game_batch.get_ob()
            indx = [i for i, r in enumerate(running) if r]
            active_states = [obs[i] for i in indx]
            state_batches = list2NumpyBatches(active_states, actioner_batch_size)
            action_batches = [actioner_fn(states) for states in state_batches]
            actions = numpyBatches2list(action_batches, len(indx))

            actions_augmented = [None] * self.size
            for i, a in zip(indx, actions):
                actions_augmented[i] = int(a)
            info = self.game_batch.take_action(actions_augmented)

            for i, (r, t) in enumerate(info):
                if running[i]:
                    total_score[i] += r
                    if t:
                        running[i] = False                    
        self.game_batch.close()
        return total_score

class GameEngine_Recorded:
    def __init__(self, record_dir):
        if not os.path.exists(record_dir):
            os.makedirs(record_dir)
        self.game = GameBatch_Parallel(1, None, record_dir, None, None)
    def __call__(self, actioner_fn, actioner_batch_size):
        while True:
            ob = self.game.get_ob()[0]
            input_batch = np.zeros((actioner_batch_size,) + self.game.state_shape, dtype=np.float32)
            input_batch[0] = ob
            action_batch = actioner_fn(input_batch)
            action = int(action_batch[0])
            _, t = self.game.take_action([action])[0]
            if t:
                break
        self.game.close()
