import os
import numpy as np
import tensorflow as tf

from game_replay_parallel import GameEngine_Eval, GameEngine_Recorded, GameEngine_Train
from util import stats, stats_str, current_time

batch_size = 32
decay = 0.99
learning_rate = 0.00025
full_eval_times = 12
# update_per_sim = 8
# sim_per_sync = 1250
# sim_per_light_eval = 1250
# sim_per_record_save = 12500
# sim_per_save = 12500
# total_sim = 312500

class Train:
    def __init__(self,
                 output_dir,
                 learner_class,
                 use_replay,
                 use_doubleQ,
                 update_per_sim,
                 sim_per_sync,
                 sim_per_light_eval,
                 sim_per_record_save,
                 total_sim):
        self.output_dir = output_dir
        self.learner_class = learner_class
        self.use_replay = use_replay
        self.use_doubleQ = use_doubleQ
        self.update_per_sim = update_per_sim
        self.sim_per_sync = sim_per_sync
        self.sim_per_light_eval = sim_per_light_eval
        self.sim_per_record_save = sim_per_record_save
        self.total_sim = total_sim

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.sim_cnt = 0
        self.use_targetFix = (sim_per_sync is not None)

        self.model_dir = os.path.join(self.output_dir, 'model')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        
    def light_eval(self):
        stats_seq = self.games_eval(self.onlineQ.eval_batch_action, 1)
        self.info_log(current_time() + ('Sim=%i Eval - ' % self.sim_cnt) \
                      + 'Reward : ' + stats_str(stats(stats_seq[0])) \
                      + ' - Episode Length : ' + stats_str(stats(stats_seq[1])))

    def full_eval(self):
        stats_seq = self.games_eval(self.onlineQ.eval_batch_action, full_eval_times)
        self.info_log(current_time() + ('Sim=%i Final - ' % self.sim_cnt) \
                      + 'Reward : ' + stats_str(stats(stats_seq[0])) \
                      + ' - Episode Length : ' + stats_str(stats(stats_seq[1])))
    
    def update(self):
        s0, act, rew, s1, tm = self.games_train.get_trans()
        target = None
        if not self.use_targetFix:
            target = rew + np.logical_not(tm) * np.max(self.onlineQ.eval_batch(s1), axis=1)
        elif not self.use_doubleQ:
            target = rew + np.logical_not(tm) * np.max(self.targetQ.eval_batch(s1), axis=1)
        else:
            best_act = self.onlineQ.eval_batch_action(s1)
            target = rew + np.logical_not(tm) * self.targetQ.eval_batch(s1)[np.arange(batch_size), best_act]
        self.onlineQ.update_batch(s0, act, target)

    def step_sim(self):
        self.games_train(self.onlineQ.eval_batch_action)
        for _ in xrange(self.update_per_sim):
            self.update()
        self.sim_cnt += 1

    def record_save(self):
        self.onlineQ.save(os.path.join(self.model_dir, 'model-%i.ckpt' % self.sim_cnt))
        self.games_record(self.onlineQ.eval_batch_action)

    def info_log(self, s):
        with open(os.path.join(self.output_dir, 'log.txt'), 'a') as f:
            print >>f, s
            f.flush()
    
    def sync(self):
        param_map = self.onlineQ.get_param()
        result_map = {}
        for k, v in param_map.iteritems():
            assert k.startswith('online_Q')
            new_k = k.replace('online_Q', 'target_Q', 1)
            result_map[new_k] = v
        self.targetQ.set_param(result_map)

    def __call__(self):
        self.games_train = GameEngine_Train(batch_size, self.use_replay)
        self.games_eval = GameEngine_Eval(batch_size)
        self.games_record = GameEngine_Recorded(os.path.join(self.output_dir, 'game_video'), batch_size)

        self.sess = tf.Session()
        game_info = self.games_train.games
        self.onlineQ = self.learner_class('online_Q', self.sess, game_info.state_shape, game_info.action_n, batch_size, \
                                          learning_rate, os.path.join(self.output_dir, 'tf_log'))
        if self.use_targetFix:
            self.targetQ = self.learner_class('target_Q', self.sess, game_info.state_shape, game_info.action_n, batch_size, \
                                          None, None)
            self.sync()
        
        self.record_save()
        self.light_eval()
        
        while self.sim_cnt < self.total_sim:
            self.step_sim()
            if self.use_targetFix and self.sim_cnt % self.sim_per_sync == 0:
                self.sync()
            if self.sim_cnt % self.sim_per_light_eval == 0:
                self.light_eval()
            if self.sim_cnt % self.sim_per_record_save == 0:
                self.record_save()
        
        if self.sim_cnt % self.sim_per_record_save != 0:
            self.record_save()
        self.full_eval()

        self.games_train.close()
        self.games_eval.close()
        self.games_record.close()
        self.sess.close()
