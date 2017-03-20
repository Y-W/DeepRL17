import os
import numpy as np


#TODO gradient clipping - huber loss
class Learner:
    def __init__(self, state_shape, action_n, batch_size, learning_rate):
        raise NotImplementedError()

    def eval_batch(self, input_batch):
        raise NotImplementedError()

    def update_batch(self, input_batch, action_batch, target_batch):
        raise NotImplementedError()
    
    def save(self, filePath):
        raise NotImplementedError()
    
    def load(self, filePath):
        raise NotImplementedError()


class Q:
    def __init__(self, learner, state_shape, action_n, batch_size):
        raise NotImplementedError()

    def get_batch_actioner(self):
        raise NotImplementedError()
    
    def update_learner(self, sampler, batch_num, decay):
        raise NotImplementedError()

    def save(self, dir_path):
        raise NotImplementedError()
    
    def load(self, dir_path):
        raise NotImplementedError()


class SimpleQ(Q):
    def __init__(self, learner):
        self.learner = learner
        self.state_shape = learner.state_shape
        self.action_n = learner.action_n
        self.batch_size = learner.batch_size
    
    def get_batch_actioner(self):
        class Actioner:
            def __init__(self, learner_eval):
                self.learner_eval = learner_eval
            def __call__(self, input_batch):
                return np.argmax(self.learner_eval(input_batch), axis=1)
        return Actioner(self.learner.eval_batch)
    
    def update_learner(self, sampler, batch_num, decay):
        inputs = []
        actions = []
        targets = []
        for _ in xrange(batch_num):
            trans = sampler(self.batch_size)
            s0_batch = np.zeros((self.batch_size,) + self.state_shape, dtype=np.float32)
            s1_batch = np.zeros((self.batch_size,) + self.state_shape, dtype=np.float32)
            action_batch = np.zeros(self.batch_size, dtype=np.int_)
            for i in xrange(self.batch_size):
                s0_batch[i] = trans[i][0]
                action_batch[i] = trans[i][1]
                if trans[i][3] is not None:
                    s1_batch[i] = trans[i][3]
            est = np.max(self.learner.eval_batch(s1_batch), axis=1)
            target_batch = np.zeros(self.batch_size, dtype=np.float32)
            for i in xrange(self.batch_size):
                if trans[i][3] is None:
                    target_batch[i] = trans[i][2]
                else:
                    target_batch[i] = trans[i][2] + decay * est[i]
            inputs.append(s0_batch)
            actions.append(action_batch)
            targets.append(target_batch)
        for i in xrange(batch_num):
            self.learner.update_batch(inputs[i], actions[i], targets[i])

    def save(self, dir_path):
        self.learner.save(os.path.join(dir_path, 'learner.ckpt'))
    
    def load(self, dir_path):
        self.learner.load(os.path.join(dir_path, 'learner.ckpt'))


class DoubleQ(Q):
    def __init__(self, learner1, learner2):
        self.learner1 = learner1
        self.learner2 = learner2
        self.state_shape = learner1.state_shape
        self.action_n = learner1.action_n
        self.batch_size = learner1.batch_size

    def get_batch_actioner(self):
        class Actioner:
            def __init__(self, learner_eval1, learner_eval2):
                self.learner_eval1 = learner_eval1
                self.learner_eval2 = learner_eval2
            def __call__(self, input_batch):
                return np.argmax(self.learner_eval1(input_batch) + self.learner_eval2(input_batch), axis=1)
        return Actioner(self.learner1.eval_batch, self.learner2.eval_batch)
    
    def update_learner(self, sampler, batch_num, decay):
        for _ in xrange(batch_num):
            trans = sampler(self.batch_size)
            s0_batch = np.zeros((self.batch_size,) + self.state_shape, dtype=np.float32)
            s1_batch = np.zeros((self.batch_size,) + self.state_shape, dtype=np.float32)
            action_batch = np.zeros(self.batch_size, dtype=np.int_)
            for i in xrange(self.batch_size):
                s0_batch[i] = trans[i][0]
                action_batch[i] = trans[i][1]
                if trans[i][3] is not None:
                    s1_batch[i] = trans[i][3]
            est = np.max(self.learner1.eval_batch(s1_batch), axis=1)
            target_batch = np.zeros(self.batch_size, dtype=np.float32)
            for i in xrange(self.batch_size):
                if trans[i][3] is None:
                    target_batch[i] = trans[i][2]
                else:
                    target_batch[i] = trans[i][2] + decay * est[i]
            self.learner2.update_batch(s0_batch, action_batch, target_batch)

            trans = sampler(self.batch_size)
            s0_batch = np.zeros((self.batch_size,) + self.state_shape, dtype=np.float32)
            s1_batch = np.zeros((self.batch_size,) + self.state_shape, dtype=np.float32)
            action_batch = np.zeros(self.batch_size, dtype=np.int_)
            for i in xrange(self.batch_size):
                s0_batch[i] = trans[i][0]
                action_batch[i] = trans[i][1]
                if trans[i][3] is not None:
                    s1_batch[i] = trans[i][3]
            est = np.max(self.learner2.eval_batch(s1_batch), axis=1)
            target_batch = np.zeros(self.batch_size, dtype=np.float32)
            for i in xrange(self.batch_size):
                if trans[i][3] is None:
                    target_batch[i] = trans[i][2]
                else:
                    target_batch[i] = trans[i][2] + decay * est[i]
            self.learner1.update_batch(s0_batch, action_batch, target_batch)

    def save(self, dir_path):
        self.learner1.save(os.path.join(dir_path, 'learner1.ckpt'))
        self.learner2.save(os.path.join(dir_path, 'learner2.ckpt'))
    
    def load(self, dir_path):
        self.learner1.load(os.path.join(dir_path, 'learner1.ckpt'))
        self.learner2.load(os.path.join(dir_path, 'learner2.ckpt'))


class LinearLeaner(Learner):
    def __init__(self, state_shape, action_n, batch_size, learning_rate):
        self.state_shape = state_shape
        self.feature_num = np.prod(state_shape)
        self.action_n = action_n
        self.batch_size = batch_size
        self.W = np.zeros((self.feature_num, self.action_n), dtype=np.float32)
        self.B = np.zeros((self.action_n,), dtype=np.float32)
        self.learning_rate = learning_rate

    def eval_batch(self, input_batch):
        input_batch = np.reshape(input_batch, (input_batch.shape[0], -1))
        return np.dot(input_batch, self.W) + self.B

    def update_batch(self, input_batch, action_batch, target_batch):
        input_batch = np.reshape(input_batch, (input_batch.shape[0], -1))
        action_one_hot = np.eye(self.action_n)[action_batch]
        out = np.dot(input_batch, self.W) + self.B
        grad_out = action_one_hot * np.clip(out - target_batch[:, np.newaxis], -1, 1) * (1.0 / self.batch_size)
        lr = self.learning_rate()
        self.B -= lr * np.sum(grad_out, axis=0)
        self.W -= lr * np.dot(input_batch.T, grad_out)

    def save(self, filePath):
        np.savez(filePath, W=self.W, B=self.B)
    
    def load(self, filePath):
        tmp = np.load(filePath)
        self.W = tmp['W']
        self.B = tmp['B']
