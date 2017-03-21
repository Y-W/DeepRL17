import os
import numpy as np
import tensorflow as tf


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
    def __init__(self, state_shape, action_n, batch_size, learning_rate, weight_decay):
        self.state_shape = state_shape
        self.feature_num = np.prod(state_shape)
        self.action_n = action_n
        self.batch_size = batch_size
        self.W = np.zeros((self.feature_num, self.action_n), dtype=np.float32)
        self.B = np.zeros((self.action_n,), dtype=np.float32)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

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
        self.B -= self.weight_decay * self.B
        self.W -= self.weight_decay * self.W

    def save(self, filePath):
        np.savez(filePath, W=self.W, B=self.B)
    
    def load(self, filePath):
        tmp = np.load(filePath)
        self.W = tmp['W']
        self.B = tmp['B']

class DeepLearner(Learner):
    def __init__(self, state_shape, action_n, batch_size, learning_rate):
        self.state_shape = state_shape
        self.action_n = action_n
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        #TODO set up the neural network just as in the paper
        with tf.Graph().as_default():
            with tf.name_scope('pre_proc'):
                self.input = tf.placeholder(tf.float32, 
                                            shape = (batch_size,) + state_shape,
                                            name = 'input')
            with tf.name_scope('conv_layers'):
                self.conv1 = tf.layers.conv2d(inputs = self.input,
                                              filters = 32,
                                              kernel_size= (8,8),
                                              strides = (4,4),
                                              data_format = 'channels_first',
                                              kernel_initializer = tf.contrib.layers.xavier_initializer(),
                                              bias_initializer = tf.constant_initializer(0.1),
                                              activation = tf.nn.relu,
                                              name = 'conv1')
                self.conv2 = tf.layers.conv2d(inputs = self.conv1,
                                              filters = 64,
                                              kernel_size= (4,4),
                                              strides = (2,2),
                                              data_format = 'channels_first',
                                              kernel_initializer = tf.contrib.layers.xavier_initializer(),
                                              bias_initializer = tf.constant_initializer(0.1),
                                              activation = tf.nn.relu,
                                              name = 'conv2')
                self.conv3 = tf.layers.conv2d(inputs = self.conv2,
                                              filters = 64,
                                              kernel_size= (3,3),
                                              strides = (1,1),
                                              data_format = 'channels_first',
                                              kernel_initializer = tf.contrib.layers.xavier_initializer(),
                                              bias_initializer = tf.constant_initializer(0.1),
                                              activation = tf.nn.relu,
                                              name = 'conv3')
            with tf.name_scope('FCLs'):
                self.flat = tf.reshape(self.conv3,(32,-1),name='flat') 
                self.fcn1 = tf.layers.dense(inputs = self.flat,
                                            units = 512,
                                            activation = tf.nn.relu,
                                            kernel_initializer = tf.contrib.layers.xavier_initializer(),
                                            name = 'fcn1')
                self.fcn2 = tf.layers.dense(inputs = self.fcn1,
                                            units = action_n,
                                            activation = None,
                                            kernel_initializer = tf.contrib.layers.xavier_initializer(),
                                            name = 'linear')
            with tf.name_scope('Loss'):
                self.actions = tf.placeholder(tf.int32,
                                              shape = batch_size,
                                              name = 'actions')
                self.act_one_hot = tf.one_hot(indices = self.actions,
                                              depth = action_n,
                                              on_value = 1.0,
                                              off_value = 0.0,
                                              name = 'act_one_hot')
                self.q_pred = tf.reduce_sum(self.fcn2 * self.act_one_hot,
                                            reduction_indices = 1,
                                            name = 'q_pred')
                self.targets = tf.placeholder(tf.float32,
                                              shape = batch_size,
                                              name = 'targets')
                self.err = tf.abs(self.q_pred - self.targets)
                self.hub_loss = tf.where(condition = self.err<1.0,
                                         x = 0.5 * tf.square(self.err),
                                         y = self.err - 0.5,
                                         name = 'clipped')
            with tf.name_scope('optim'):
                self.mean_loss = tf.reduce_mean(self.hub_loss,
                                                name = 'mean_loss')
                self.train = tf.train.AdamOptimizer(learning_rate=1e-4, 
                                                    epsilon = 1e-3).minimize(self.mean_loss)
            self.sess = tf.Session()
            self.sum_loss = tf.summary.scalar('loss',tf.reduce_mean(self.err))
            self.sum_writer = tf.summary.FileWriter('~/TensorBoard/',self.sess.graph)
            self.init = tf.global_variables_initializer()
            self.sess.run(self.init)
            self.saver = tf.train.Saver()

    def eval_batch(self, input_batch):
        #returns the forward pass of the input_batch
        return self.sess.run(self.fcn2,
                             feed_dict = {self.input:input_batch})

    def update_batch(self, input_batch, action_batch, target_batch):
        #modifies the weights according with y
        _, sum_loss = self.sess.run([self.train,self.sum_loss],
                      feed_dict = {self.input:input_batch,
                                   self.actions:action_batch,
                                   self.targets:target_batch})
        self.sum_writer.add_summary(sum_loss)
        self.sum_writer.flush()
    
    def save(self, filePath):
        self.saver.save(self.sess,filePath + 'model')
    
    def load(self, filePath):
        self.saver.restore(self.sess,filePath + 'model')
