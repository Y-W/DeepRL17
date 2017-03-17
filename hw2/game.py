import os
import multiprocessing
import numpy as np
import scipy.misc
import gym
import tensorflow as tf

from util import list2NumpyBatches, numpyBatches2list

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('game', 'SpaceInvaders-v0', 'Atari game to use')
tf.app.flags.DEFINE_integer('process', 4, 'Number of processes to use')

downsampled_frame_size = 84
past_frame_number = 4

def compress_frame(f):
    return np.mean(scipy.misc.imresize(f, (downsampled_frame_size, downsampled_frame_size)).astype(np.float32) * (1.0 / 255.0), axis=2)

def clip_reward(x):
    return float(min(max(x, -1.0), 1.0))


class AtariGame:
    def __init__(self, game_name, auto_restart, should_clip_reward, record_dir=None):
        self.auto_restart = auto_restart
        self.record_dir = record_dir
        self.should_clip_reward = should_clip_reward
        self.past_frames = np.zeros((past_frame_number, downsampled_frame_size, downsampled_frame_size), dtype=np.float32)
        self.env = gym.make(game_name)
        if record_dir is not None:
            self.env = gym.wrappers.Monitor(self.env, record_dir, force=True)
        self.terminated = False
        self.past_frames[-1] = compress_frame(self.env.reset())
        self.action_n = self.env.action_space.n
        self.state_shape = (past_frame_number, downsampled_frame_size, downsampled_frame_size)

    def reset(self):
        self.past_frames = np.zeros((past_frame_number, downsampled_frame_size, downsampled_frame_size), dtype=np.float32)
        self.past_frames[-1] = compress_frame(self.env.reset())
        self.terminated = False

    def get_ob(self):
        if self.terminated:
            return None
        else:
            return self.past_frames

    def take_action(self, action):
        if self.terminated and not self.auto_restart:
            return None
        if self.terminated and self.auto_restart:
            self.reset()
        ob, reward, done, _ = self.env.step(int(action))
        if not done:
            new_state = np.concatenate((self.past_frames[1:], compress_frame(ob)[np.newaxis, ...]), axis=0)
            if self.should_clip_reward:
                trans_tuple = (np.copy(self.past_frames), int(action), clip_reward(reward), np.copy(new_state))
            else:
                trans_tuple = (np.copy(self.past_frames), int(action), reward, np.copy(new_state))
            self.past_frames = new_state
            return trans_tuple
        else:
            if self.should_clip_reward:
                trans_tuple = (np.copy(self.past_frames), int(action), clip_reward(reward), None)
            else:
                trans_tuple = (np.copy(self.past_frames), int(action), reward, None)
            self.terminated = True
            if self.auto_restart:
                self.reset()
            return trans_tuple
    
    def close(self):
        self.env.close()


class GameBatch:
    def __init__(self, game_name, size, auto_restart, should_clip_reward):
        self.size = size
        self.auto_restart = auto_restart
        self.games = [AtariGame(game_name, auto_restart, should_clip_reward) for _ in xrange(self.size)]
        self.action_n = self.games[0].action_n
        self.state_shape = self.games[0].state_shape

    def get_ob(self):
        return [game.get_ob() for game in self.games]

    def take_action(self, actions):
        return [game.take_action(action) if action is not None else None for game, action in zip(self.games, actions)]
    
    def close(self):
        for game in self.games:
            game.close()


class GameBatch_ParallelWorker(multiprocessing.Process):
    def __init__(self, pipe, game_name, size, auto_restart, should_clip_reward):
        super(GameBatch_ParallelWorker, self).__init__()
        self.pipe = pipe
        self.game_batch = GameBatch(game_name, size, auto_restart, should_clip_reward)

    def run(self):
        while True:
            command = self.pipe.recv()
            if command == ():
                self.pipe.send(self.game_batch.get_ob())
            elif command == None:
                self.game_batch.close()
                break
            else:
                self.pipe.send(self.game_batch.take_action(command))

class GameBatch_Parallel:
    def __init__(self, size, auto_restart, should_clip_reward):
        self.size = size
        self.auto_restart = auto_restart
        self.should_clip_reward = should_clip_reward

        sample_game = AtariGame(FLAGS.game, auto_restart, should_clip_reward)
        self.state_shape = sample_game.state_shape
        self.action_n = sample_game.action_n
        sample_game.close()

        self.partition = []
        self.workers = []
        self.pipes = []
        for k in xrange(FLAGS.process):
            tmp_size = (size + FLAGS.process - 1 - k) // FLAGS.process
            self.partition.append(tmp_size)
            parent_conn, child_conn = multiprocessing.Pipe()
            self.pipes.append(parent_conn)
            worker = GameBatch_ParallelWorker(child_conn, FLAGS.game, tmp_size, auto_restart, should_clip_reward)
            worker.start()
            self.workers.append(worker)
    
    def get_ob(self):
        for p in self.pipes:
            p.send(())
        result = []
        for p in self.pipes:
            result.extend(p.recv())
        return result

    def take_action(self, actions):
        cnt = 0
        for i in xrange(len(self.workers)):
            action_sublist = tuple(actions[cnt:cnt+self.partition[i]])
            self.pipes[i].send(action_sublist)
            cnt += self.partition[i]
        assert cnt == self.size
        result = []
        for p in self.pipes:
            result.extend(p.recv())
        return result
    
    def close(self):
        for p in self.pipes:
            p.send(None)
        for w in self.workers:
            w.join()
    
    def __del__(self):
        self.close()


class GameEngine_Train:
    def __init__(self, size):
        self.size = size
        self.game_batch = GameBatch_Parallel(size, True, True)

    def __call__(self, actioner_fn, actioner_batch_size):
        states = self.game_batch.get_ob()
        state_batches = list2NumpyBatches(states, actioner_batch_size)
        action_batches = [actioner_fn(states) for states in state_batches]
        actions = numpyBatches2list(action_batches, self.size)
        actions = [int(a) for a in actions]
        return self.game_batch.take_action(actions)


class GameEngine_Eval:
    def __init__(self, size, decay=1.0):
        self.size = size
        self.decay = decay
        self.game_batch = GameBatch_Parallel(size, False, False)
    
    def __call__(self, actioner_fn, actioner_batch_size):
        total_score = [0.0] * self.size
        current_factor = 1.0
        obs = self.game_batch.get_ob()
        cnt_step = 0
        while any(ob is not None for ob in obs):
            indx, active_states = zip(*[(i, s) for i, s in enumerate(obs) if s is not None])
            state_batches = list2NumpyBatches(active_states, actioner_batch_size)
            action_batches = [actioner_fn(states) for states in state_batches]
            actions = numpyBatches2list(action_batches, len(indx))

            actions_augmented = [None] * self.size
            for i, a in zip(indx, actions):
                actions_augmented[i] = int(a)
            trans = self.game_batch.take_action(actions_augmented)

            for i in xrange(self.size):
                if trans[i] is not None:
                    total_score[i] += current_factor * trans[i][2]

            obs = self.game_batch.get_ob()
            current_factor *= self.decay
            cnt_step += 1
        self.game_batch.close()
        return total_score


class GameEngine_Recorded:
    def __init__(self, record_dir):
        if not os.path.exists(record_dir):
            os.makedirs(record_dir)
        self.game = AtariGame(FLAGS.game, False, False, record_dir=record_dir)
    def __call__(self, actioner_fn, actioner_batch_size):
        while True:
            ob = self.game.get_ob()
            input_batch = np.zeros((actioner_batch_size,) + self.game.state_shape, dtype=np.float32)
            input_batch[0] = ob
            action_batch = actioner_fn(input_batch)
            action = int(action_batch[0])
            _, _, _, s1 = self.game.take_action(action)
            if s1 is None:
                break
        self.game.close()