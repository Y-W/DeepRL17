import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from game_replay_parallel import action_n, samp_series, samp_n, GameEngine_Eval, GameEngine_Recorded, GameEngine_Train
from util import stats, stats_str, current_time

batch_size = 32
decay = 0.99
learning_rate = 0.001
# update_per_sim = 8
# sim_per_sync = 1250
# sim_per_light_eval = 1250
# sim_per_record_save = 12500
# sim_per_save = 12500
# total_sim = 312500

discount_factor = [decay ** k for k in samp_series]

class SARSA:
    def __init__(self,
                 output_dir,
                 learner_class,
                 update_per_sim,
                 sim_per_light_eval,
                 sim_per_record_save,
                 total_sim):
        self.output_dir = output_dir
        self.learner_class = learner_class
        self.update_per_sim = update_per_sim
        self.sim_per_light_eval = sim_per_light_eval
        self.sim_per_record_save = sim_per_record_save
        self.total_sim = total_sim

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.sim_cnt = 0

        self.model_dir = os.path.join(self.output_dir, 'model')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        
    def light_eval(self):
        granularity = 32
        x = np.linspace(-1.2, 0.6, granularity)
        y = np.linspace(-0.07, 0.07, granularity)
        xx, yy = np.meshgrid(x, y)
        states = np.stack((xx.reshape(-1), yy.reshape(-1)), axis=1)

        fig, axes = plt.subplots(figsize=(15, 7), nrows=1, ncols=action_n)
        values = np.zeros((states.shape[0], action_n), dtype=np.float32)
        p = 0
        while p < states.shape[0]:
            dp = min(states.shape[0] - p, batch_size)
            values[p:p+dp] = self.onlineQ.eval_batch(states[p:p+dp][:, np.newaxis, :])[:, :, 0]
            p = p + dp

        for a in xrange(action_n):
            axes[a].set_title("Action %i" % a)
            mp = axes[a].imshow(values[:, a].reshape(granularity, granularity),
                                extent=(-1.2, 0.6, 0.7, -0.7),
                                interpolation='bilinear', vmin=np.min(values[:, a])-0.1,
                                vmax=np.max(values[:, a])+0.1)
            axes[a].set_xlabel('position')
            axes[0].set_ylabel('velosity * 10')

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(mp, cax=cbar_ax)
        fig_path = os.path.join(self.output_dir, 'qvalues-%i.png' % self.sim_cnt)
        fig.savefig(fig_path, bbox_inches='tight')
        plt.close()


    def full_eval(self):
        return
    
    def update(self):
        s, act, rew, tm = self.games_train.get_trans()
        eval_batch = self.onlineQ.eval_batch
        
        eval_input = s[:, 1]
        eval_output = eval_batch(eval_input)

        target = np.zeros((batch_size, 1), dtype=np.float32)
        target[:, 0] = rew[:, 0] + decay * np.logical_not(tm[:, 1]).astype(np.float32) \
                       * eval_output[np.arange(batch_size), act[:, 1], 0]
        
        self.onlineQ.update_batch(s[:, 0], act[:, 0], target)

    def step_sim(self):
        self.games_train(lambda x: np.zeros(batch_size, dtype=np.int32))
        for _ in xrange(self.update_per_sim):
            self.update()
        self.sim_cnt += 1

    def record_save(self):
        self.onlineQ.save(os.path.join(self.model_dir, 'model-%i.ckpt' % self.sim_cnt))

    def info_log(self, s):
        with open(os.path.join(self.output_dir, 'log.txt'), 'a') as f:
            print >>f, s
            f.flush()

    def __call__(self):
        self.games_train = GameEngine_Train(batch_size)

        self.sess = tf.Session()
        self.game_info = self.games_train.games
        self.onlineQ = self.learner_class('online_Q', self.sess, self.game_info.state_shape, self.game_info.action_n, 1, batch_size, None, \
                                            learning_rate, os.path.join(self.output_dir, 'tf_log'))

        self.record_save()
        self.light_eval()
        
        while self.sim_cnt < self.total_sim:
            self.step_sim()
            if self.sim_cnt % self.sim_per_light_eval == 0:
                self.light_eval()
            if self.sim_cnt % self.sim_per_record_save == 0:
                self.record_save()
            if self.sim_cnt % (self.total_sim // 100) == 0:
                print current_time(), 'Sim', self.sim_cnt
        
        if self.sim_cnt % self.sim_per_record_save != 0:
            self.record_save()
        self.full_eval()

        self.games_train.close()
        self.sess.close()

class Onp_Cotrain:
    def __init__(self,
                 output_dir,
                 learner_class,
                 update_per_sim,
                 sim_per_light_eval,
                 sim_per_record_save,
                 total_sim):
        self.output_dir = output_dir
        self.learner_class = learner_class
        self.update_per_sim = update_per_sim
        self.sim_per_light_eval = sim_per_light_eval
        self.sim_per_record_save = sim_per_record_save
        self.total_sim = total_sim

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.sim_cnt = 0

        self.model_dir = os.path.join(self.output_dir, 'model')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        
    def light_eval(self):
        granularity = 32
        x = np.linspace(-1.2, 0.6, granularity)
        y = np.linspace(-0.07, 0.07, granularity)
        xx, yy = np.meshgrid(x, y)
        states = np.stack((xx.reshape(-1), yy.reshape(-1)), axis=1)

        fig, axes = plt.subplots(figsize=(15, 7), nrows=1, ncols=action_n)
        values = np.zeros((states.shape[0], action_n), dtype=np.float32)
        p = 0
        while p < states.shape[0]:
            dp = min(states.shape[0] - p, batch_size)
            values[p:p+dp] = np.sum(self.onlineQ.eval_batch(states[p:p+dp][:, np.newaxis, :]), axis=2)
            p = p + dp

        for a in xrange(action_n):
            axes[a].set_title("Action %i" % a)
            mp = axes[a].imshow(values[:, a].reshape(granularity, granularity),
                                extent=(-1.2, 0.6, 0.7, -0.7),
                                interpolation='bilinear', vmin=np.min(values[:, a])-0.1,
                                vmax=np.max(values[:, a])+0.1)
            axes[a].set_xlabel('position')
            axes[0].set_ylabel('velosity * 10')

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(mp, cax=cbar_ax)
        fig_path = os.path.join(self.output_dir, 'qvalues-%i.png' % self.sim_cnt)
        fig.savefig(fig_path, bbox_inches='tight')
        plt.close()


    def full_eval(self):
        return
    
    def update(self):
        s, act, rew, tm = self.games_train.get_trans()
        eval_batch = None
        eval_batch = self.onlineQ.eval_batch_large
        
        eval_input = s[:, 1:].reshape((-1,) + self.game_info.state_shape)
        eval_output = eval_batch(eval_input).reshape((batch_size, -1, self.game_info.action_n, samp_n))

        target = np.zeros(rew.shape, dtype=np.float32)
        target[:, 0] = rew[:, 0]
        for k in xrange(1, len(samp_series)):
            net_output = eval_output[:, k - 1]
            target[:, k] = discount_factor[k] * np.logical_not(tm[:, k]).astype(np.float32) \
                           * np.sum(net_output[np.arange(rew.shape[0]), act[:, k], :k], axis=1)
        
        self.onlineQ.update_batch(s[:, 0], act[:, 0], target)

    def step_sim(self):
        self.games_train(lambda x: np.zeros(batch_size, dtype=np.int32))
        for _ in xrange(self.update_per_sim):
            self.update()
        self.sim_cnt += 1

    def record_save(self):
        self.onlineQ.save(os.path.join(self.model_dir, 'model-%i.ckpt' % self.sim_cnt))

    def info_log(self, s):
        with open(os.path.join(self.output_dir, 'log.txt'), 'a') as f:
            print >>f, s
            f.flush()

    def __call__(self):
        self.games_train = GameEngine_Train(batch_size)

        self.sess = tf.Session()
        self.game_info = self.games_train.games
        self.onlineQ = self.learner_class('online_Q', self.sess, self.game_info.state_shape, self.game_info.action_n, samp_n, batch_size, batch_size * (samp_n - 1), \
                                            learning_rate, os.path.join(self.output_dir, 'tf_log'))

        self.record_save()
        self.light_eval()
        
        while self.sim_cnt < self.total_sim:
            self.step_sim()
            if self.sim_cnt % self.sim_per_light_eval == 0:
                self.light_eval()
            if self.sim_cnt % self.sim_per_record_save == 0:
                self.record_save()
            if self.sim_cnt % (self.total_sim // 100) == 0:
                print current_time(), 'Sim', self.sim_cnt
        
        if self.sim_cnt % self.sim_per_record_save != 0:
            self.record_save()
        self.full_eval()

        self.games_train.close()
        self.sess.close()


class Onp_seq_step:
    def __init__(self,
                 output_dir,
                 k,
                 prev_q,
                 sess,
                 learner_class,
                 update_per_sim,
                 sim_per_light_eval,
                 sim_per_record_save,
                 total_sim):
        self.output_dir = os.path.join(output_dir, 'q_%i' % (2 ** k))
        self.learner_class = learner_class
        self.update_per_sim = update_per_sim
        self.sim_per_light_eval = sim_per_light_eval
        self.sim_per_record_save = sim_per_record_save
        self.total_sim = total_sim
        self.k = k
        self.prev_q = prev_q
        self.sess = sess

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.sim_cnt = 0

        self.model_dir = os.path.join(self.output_dir, 'model')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        
    def light_eval(self):
        granularity = 32
        x = np.linspace(-1.2, 0.6, granularity)
        y = np.linspace(-0.07, 0.07, granularity)
        xx, yy = np.meshgrid(x, y)
        states = np.stack((xx.reshape(-1), yy.reshape(-1)), axis=1)

        fig, axes = plt.subplots(figsize=(15, 7), nrows=1, ncols=action_n)
        values = np.zeros((states.shape[0], action_n), dtype=np.float32)
        p = 0
        while p < states.shape[0]:
            dp = min(states.shape[0] - p, batch_size)
            values[p:p+dp] = self.onlineQ.eval_batch(states[p:p+dp][:, np.newaxis, :])[:, :, 0]
            p = p + dp

        for a in xrange(action_n):
            axes[a].set_title("Action %i" % a)
            mp = axes[a].imshow(values[:, a].reshape(granularity, granularity),
                                extent=(-1.2, 0.6, 0.7, -0.7),
                                interpolation='bilinear', vmin=np.min(values[:, a])-0.1,
                                vmax=np.max(values[:, a])+0.1)
            axes[a].set_xlabel('position')
            axes[0].set_ylabel('velosity * 10')

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(mp, cax=cbar_ax)
        fig_path = os.path.join(self.output_dir, 'qvalues-%i.png' % self.sim_cnt)
        fig.savefig(fig_path, bbox_inches='tight')
        plt.close()


    def full_eval(self):
        return
    
    def update(self):
        s, act, rew, tm = self.games_train.get_trans()
        eval_batch = None

        target = np.zeros((batch_size, 1), dtype=np.float32)
        if self.k == 0:
            target[:, 0] = rew[:, 0]
        else:
            eval_batch = self.prev_q.eval_batch

            target[:, 0] = eval_batch(s[:, 0])[np.arange(rew.shape[0]), act[:, 0], 0] + discount_factor[self.k] * np.logical_not(tm[:, self.k]).astype(np.float32) \
                           * eval_batch(s[:, self.k])[np.arange(rew.shape[0]), act[:, self.k], 0]
        
        self.onlineQ.update_batch(s[:, 0], act[:, 0], target)

    def step_sim(self):
        self.games_train(lambda x: np.zeros(batch_size, dtype=np.int32))
        for _ in xrange(self.update_per_sim):
            self.update()
        self.sim_cnt += 1

    def record_save(self):
        self.onlineQ.save(os.path.join(self.model_dir, 'model-%i.ckpt' % self.sim_cnt))

    def info_log(self, s):
        with open(os.path.join(self.output_dir, 'log.txt'), 'a') as f:
            print >>f, s
            f.flush()

    def __call__(self):
        self.games_train = GameEngine_Train(batch_size)

        self.game_info = self.games_train.games
        self.onlineQ = self.learner_class('online_Q_%i' % self.k, self.sess, self.game_info.state_shape, self.game_info.action_n, 1, batch_size, None, \
                                            learning_rate, os.path.join(self.output_dir, 'tf_log_%i' % self.k))

        self.record_save()
        self.light_eval()
        
        while self.sim_cnt < self.total_sim:
            self.step_sim()
            if self.sim_cnt % self.sim_per_light_eval == 0:
                self.light_eval()
            if self.sim_cnt % self.sim_per_record_save == 0:
                self.record_save()
            if self.sim_cnt % (self.total_sim // 100) == 0:
                print current_time(), 'Sim', self.sim_cnt
        
        if self.sim_cnt % self.sim_per_record_save != 0:
            self.record_save()
        self.full_eval()

        self.games_train.close()

        return self.onlineQ

class Onp_seq:
    def __init__(self,
                 output_dir,
                 learner_class,
                 update_per_sim,
                 sim_per_light_eval,
                 sim_per_record_save,
                 total_sim):
        self.output_dir = output_dir
        self.learner_class = learner_class
        self.update_per_sim = update_per_sim
        self.sim_per_light_eval = sim_per_light_eval
        self.sim_per_record_save = sim_per_record_save
        self.total_sim = total_sim


        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.sim_cnt = 0

        self.model_dir = os.path.join(self.output_dir, 'model')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
    
    def __call__(self):
        self.prev_q = None
        self.sess = tf.Session()
        for k in xrange(samp_n):
            self.prev_q = Onp_seq_step(
                self.output_dir,
                k,
                self.prev_q,
                self.sess,
                self.learner_class,
                self.update_per_sim,
                self.sim_per_light_eval,
                self.sim_per_record_save,
                self.total_sim
            )()
    