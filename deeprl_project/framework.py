import os
import numpy as np
import tensorflow as tf

from game_replay_parallel import samp_series, samp_n, GameEngine_Eval, GameEngine_Recorded, GameEngine_Train
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

discount_factor = [decay ** k for k in samp_series]

class Train:
    def __init__(self,
                 output_dir,
                 learner_class,
                 update_per_sim,
                 sim_per_sync,
                 sim_per_light_eval,
                 sim_per_record_save,
                 total_sim):
        self.output_dir = output_dir
        self.learner_class = learner_class
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
        s, act, rew, tm = self.games_train.get_trans()
        eval_batch = None
        if not self.use_targetFix:
            eval_batch = self.onlineQ.eval_batch_large
        else:
            eval_batch = self.targetQ.eval_batch_large
        
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
        self.games_train(self.onlineQ.eval_batch_action)
        for _ in xrange(self.update_per_sim):
            self.update()
        self.sim_cnt += 1

    def record_save(self):
        self.onlineQ.save(os.path.join(self.model_dir, 'model-%i.ckpt' % self.sim_cnt))
        # self.games_record(self.onlineQ.eval_batch_action)

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
        self.games_train = GameEngine_Train(batch_size)
        self.games_eval = GameEngine_Eval(batch_size)
        # self.games_record = GameEngine_Recorded(os.path.join(self.output_dir, 'game_video'), batch_size)

        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        self.game_info = self.games_train.games
        if not self.use_targetFix:
            self.onlineQ = self.learner_class('online_Q', self.sess, self.game_info.state_shape, self.game_info.action_n, samp_n, batch_size, batch_size * (samp_n - 1), \
                                            learning_rate, os.path.join(self.output_dir, 'tf_log'))
        else:
            self.onlineQ = self.learner_class('online_Q', self.sess, self.game_info.state_shape, self.game_info.action_n, samp_n, batch_size, None, \
                                            learning_rate, os.path.join(self.output_dir, 'tf_log'))
            self.targetQ = self.learner_class('target_Q', self.sess, self.game_info.state_shape, self.game_info.action_n, samp_n, batch_size, batch_size * (samp_n - 1), \
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
            if self.sim_cnt % (self.total_sim // 1000) == 0:
                print current_time(), 'Sim', self.sim_cnt
        
        if self.sim_cnt % self.sim_per_record_save != 0:
            self.record_save()
        self.full_eval()

        self.games_train.close()
        self.games_eval.close()
        # self.games_record.close()
        self.sess.close()
