import numpy as np

class SARSA:
    def __init__(self,
                 game_engine,
                 inc_learner,
                 policy_fn,
                 decay):
        self.game_engine = game_engine
        self.inc_learner = inc_learner
        self.policy_fn = policy_fn
        self.decay = decay
    
    def step(self):
        self.game_engine(self.policy_fn)
        trans_s, trans_action, trans_reward, trans_terminal = self.game_engine.samp_trans(1)
        s0 = trans_s[:, 0]
        s1 = trans_s[:, 1]
        r = trans_reward[:, 0]
        a0 = trans_action[:, 0]
        a1 = trans_action[:, 1]
        q1 = self.inc_learner.eval_batch(s1)[np.arange(s1.shape[0]), a1]
        target = r + self.decay * np.logical_not(trans_terminal[:, 1]) * q1
        self.inc_learner.update_batch(s0, a0, target)
    
    def loop(self, n):
        for _ in xrange(n):
            self.step()

class IDH:
    def __init__(self,
                 game_engine,
                 batch_leaner,
                 policy_fn,
                 eval_fn,
                 step_away,
                 decay):
        self.game_engine = game_engine
        self.batch_leaner = batch_leaner
        self.policy_fn = policy_fn
        self.eval_fn = eval_fn
        self.step_away = step_away
        self.decay = self.decay
    
    def add_batch(self):
        self.game_engine(self.policy_fn)
        trans_s, trans_action, trans_reward, trans_terminal = self.game_engine.samp_trans(self.step_away)
        s0 = trans_s[:, 0]
        s1 = trans_s[:, 1]
        a0 = trans_action[:, 0]
        a1 = trans_action[:, 1]
        q0 = self.eval_fn(s0)[np.arange(s0.shape[0]), a0]
        q1 = self.eval_fn(s1)[np.arange(s1.shape[0]), a1]
        target = q0 + (self.decay ** self.step_away) * np.logical_not(trans_terminal[:, 1]) * q1
        self.batch_leaner.add_batch(s0, a0, target)
    
    def loop(self, n):
        for _ in xrange(n):
            self.step()
