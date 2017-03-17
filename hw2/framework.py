import os
import numpy as np
import tensorflow as tf

from game import GameEngine_Eval, GameEngine_Recorded, GameEngine_Train
from util import stats, stats_str, current_time

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_float('epsilon', 0.05, 'Epsilon')

class LightEval:
    def __init__(self, q, size=20):
        self.q = q
        self.size = size
    def __call__(self):
        return stats(GameEngine_Eval(self.size)(self.q.get_batch_actioner(FLAGS.epsilon), self.q.batch_size))


class FullEval:
    def __init__(self, q, record_dir, size=100):
        self.q = q
        self.size = size
        self.record_dir = record_dir
    def __call__(self):
        GameEngine_Recorded(self.record_dir)(self.q.get_batch_actioner(FLAGS.epsilon), self.q.batch_size)
        return stats(GameEngine_Eval(self.size)(self.q.get_batch_actioner(FLAGS.epsilon), self.q.batch_size))

#TODO log to file!
class Train:
    def __init__(self,
                 q,
                 pool,
                 game_engine,
                 decay,
                 init_game_batches,
                 game_batches_per_update,
                 sample_batches_per_update,
                 updates_per_light_eval,
                 updates_per_full_eval,
                 updates_per_save,
                 total_updates,
                 output_dir,
                ):
        self.q = q
        self.pool = pool
        self.game = game_engine
        self.init_game_batches = init_game_batches
        self.game_batches_per_update = game_batches_per_update
        self.sample_batches_per_update = sample_batches_per_update
        self.updates_per_light_eval = updates_per_light_eval
        self.updates_per_full_eval = updates_per_full_eval
        self.updates_per_save = updates_per_save
        self.total_updates = total_updates
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.decay = decay
        self.light_stats_dict = {}
        self.full_stats_dict = {}
        self.update_cnt = 0

    def light_eval(self):
        self.light_stats_dict[self.update_cnt] = LightEval(self.q)()
        print current_time() + ('Iter=%i - Light Eval - ' % self.update_cnt) + stats_str(self.light_stats_dict[self.update_cnt])

    def full_eval(self):
        self.full_stats_dict[self.update_cnt] = FullEval(self.q, os.path.join(self.output_dir, 'video-%i' % self.update_cnt))()
        print current_time() + ('Iter=%i - Full Eval - ' % self.update_cnt) + stats_str(self.full_stats_dict[self.update_cnt])
    
    def update(self):
        for _ in xrange(self.game_batches_per_update):
            self.pool.extend(self.game(self.q.get_batch_actioner(FLAGS.epsilon), self.q.batch_size))
        self.q.update_learner(self.pool.sample, self.sample_batches_per_update, self.decay)
        # print current_time() + ('Iter=%i - Updated' % self.update_cnt)
        self.update_cnt += 1
    
    def init_process(self):
        for _ in xrange(self.init_game_batches):
            self.pool.extend(self.game(self.q.get_batch_actioner(1.0), self.q.batch_size))

    def save_model(self):
        model_dir = os.path.join(self.output_dir, 'model-%i' % self.update_cnt)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.q.save(model_dir)

    def __call__(self):
        self.init_process()
        while self.update_cnt <= self.total_updates:
            if self.update_cnt % self.updates_per_light_eval == 0:
                self.light_eval()
            if self.update_cnt % self.updates_per_full_eval == 0 or self.update_cnt == self.total_updates:
                self.full_eval()
            if self.update_cnt % self.updates_per_save == 0 or self.update_cnt == self.total_updates:
                self.save_model()
            if self.update_cnt < self.total_updates:
                self.update()
