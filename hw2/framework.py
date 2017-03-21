import os
import numpy as np
import tensorflow as tf

from game_replay_parallel import GameEngine_Eval, GameEngine_Recorded, GameEngine_Train
from util import stats, stats_str, current_time


class LightEval:
    def __init__(self, q, size=20):
        self.q = q
        self.size = size
    def __call__(self):
        return stats(GameEngine_Eval(self.size)(self.q.get_batch_actioner(), self.q.batch_size))


class FullEval:
    def __init__(self, q, record_dir, size=100):
        self.q = q
        self.size = size
        self.record_dir = record_dir
    def __call__(self):
        GameEngine_Recorded(self.record_dir)(self.q.get_batch_actioner(), self.q.batch_size)
        return stats(GameEngine_Eval(self.size)(self.q.get_batch_actioner(), self.q.batch_size))


class Train:
    def __init__(self,
                 q,
                 game_engine,
                 decay,
                 game_batches_per_update,
                 sample_batches_per_update,
                 updates_per_light_eval,
                 updates_per_full_eval,
                 updates_per_save,
                 total_updates,
                 output_dir,
                 use_last_trans_only=False
                ):
        self.q = q
        self.game = game_engine
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
        self.use_last_trans_only = use_last_trans_only

    def light_eval(self):
        self.light_stats_dict[self.update_cnt] = LightEval(self.q)()
        self.info_log(current_time() + ('Iter=%i - Light Eval - ' % self.update_cnt) + stats_str(self.light_stats_dict[self.update_cnt]))

    def full_eval(self):
        self.full_stats_dict[self.update_cnt] = FullEval(self.q, os.path.join(self.output_dir, 'video-%i' % self.update_cnt))()
        self.info_log(current_time() + ('Iter=%i - Full Eval - ' % self.update_cnt) + stats_str(self.full_stats_dict[self.update_cnt]))
    
    def update(self):
        if self.game_batches_per_update > 10:
            print current_time() + ('Iter=%i - Full Eval - ' % self.update_cnt) + 'Start Running Game'
        for _ in xrange(self.game_batches_per_update):
            self.game(self.q.get_batch_actioner(), self.q.batch_size)
        if self.game_batches_per_update > 10:
            print current_time() + ('Iter=%i - Full Eval - ' % self.update_cnt) + 'Start Optimizing Q'
        if not self.use_last_trans_only:
            self.q.update_learner(self.game.sample, self.sample_batches_per_update, self.decay)
        else:
            self.q.update_learner(self.game.last_trans, self.sample_batches_per_update, self.decay)
        if self.game_batches_per_update > 10:
            print current_time() + ('Iter=%i - Full Eval - ' % self.update_cnt) + 'Finish Optimizing Q'
        self.update_cnt += 1

    def save_model(self):
        model_dir = os.path.join(self.output_dir, 'model-%i' % self.update_cnt)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.q.save(model_dir)

    def info_log(self, s):
        with open(os.path.join(self.output_dir, 'log.txt'), 'a') as f:
            print >>f, s
            f.flush()

    def __call__(self):
        self.save_model()
        self.full_eval()
        while self.update_cnt < self.total_updates:
            self.update()
            if self.update_cnt % self.updates_per_light_eval == 0:
                self.light_eval()
            if self.update_cnt % self.updates_per_full_eval == 0 or self.update_cnt == self.total_updates:
                self.full_eval()
            if self.update_cnt % self.updates_per_save == 0 or self.update_cnt == self.total_updates:
                self.save_model()
