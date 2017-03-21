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
    def __init__(self, name, sess, state_shape, action_n, batch_size, learning_rate, weight_decay, log_dir):
        self.state_shape = state_shape
        self.action_n = action_n
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        with tf.variable_scope(name):
            with tf.variable_scope('pre_proc'):
                self.input_tensor = tf.placeholder(tf.float32, 
                                            shape = (batch_size,) + state_shape,
                                            name = 'input')
                tf.summary.histogram('input', self.input_tensor)
            with tf.variable_scope('conv_layers'):
                self.conv1 = tf.layers.conv2d(inputs = self.input_tensor,
                                                filters = 16,
                                                kernel_size= (8,8),
                                                strides = (4,4),
                                                data_format = 'channels_first',
                                                kernel_initializer = tf.random_normal_initializer(
                                                        stddev=np.sqrt(2.0 / self.input_tensor.get_shape()[1].value)),
                                                bias_initializer = tf.constant_initializer(0.1),
                                                activation = tf.nn.relu,
                                                name = 'conv1')
                tf.summary.histogram('conv1', self.conv1)
                self.conv2 = tf.layers.conv2d(inputs = self.conv1,
                                                filters = 32,
                                                kernel_size= (4,4),
                                                strides = (2,2),
                                                data_format = 'channels_first',
                                                kernel_initializer = tf.random_normal_initializer(
                                                        stddev=np.sqrt(2.0 / self.conv1.get_shape()[1].value)),
                                                bias_initializer = tf.constant_initializer(0.1),
                                                activation = tf.nn.relu,
                                                name = 'conv2')
                tf.summary.histogram('conv2', self.conv2)
                # self.conv3 = tf.layers.conv2d(inputs = self.conv2,
                #                                 filters = 64,
                #                                 kernel_size= (3,3),
                #                                 strides = (1,1),
                #                                 data_format = 'channels_first',
                #                                 kernel_initializer = tf.contrib.layers.xavier_initializer(),
                #                                 bias_initializer = tf.constant_initializer(0.1),
                #                                 activation = tf.nn.relu,
                #                                 name = 'conv3')
                # tf.summary.histogram('conv3', self.conv3)
            with tf.variable_scope('FCLs'):
                self.flat = tf.reshape(self.conv2,(32,-1),name='flat') 
                self.fcn1 = tf.layers.dense(inputs = self.flat,
                                            units = 256,
                                            activation = tf.nn.relu,
                                            kernel_initializer = tf.random_normal_initializer(
                                                        stddev=np.sqrt(2.0 / self.flat.get_shape()[1].value)),
                                            name = 'fcn1')
                tf.summary.histogram('fcn1', self.fcn1)
                self.fcn2 = tf.layers.dense(inputs = self.fcn1,
                                            units = action_n,
                                            activation = None,
                                            kernel_initializer = tf.random_normal_initializer(
                                                        stddev=np.sqrt(1.0 / self.flat.get_shape()[1].value)),
                                            name = 'linear')
                tf.summary.histogram('output', self.fcn2)
            with tf.variable_scope('Loss'):
                self.actions_tensor = tf.placeholder(tf.int32,
                                                shape = batch_size,
                                                name = 'actions')
                self.act_one_hot = tf.one_hot(indices = self.actions_tensor,
                                                depth = action_n,
                                                on_value = 1.0,
                                                off_value = 0.0,
                                                name = 'act_one_hot')
                self.q_pred = tf.reduce_sum(self.fcn2 * self.act_one_hot,
                                            reduction_indices = 1,
                                            name = 'q_pred')
                self.targets_tensor = tf.placeholder(tf.float32,
                                                shape = batch_size,
                                                name = 'targets')
                self.err = tf.abs(self.q_pred - self.targets_tensor)
                tf.summary.histogram('abs_err', self.err)
                self.hub_loss = tf.where(condition = self.err<1.0,
                                            x = 0.5 * tf.square(self.err),
                                            y = self.err - 0.5,
                                            name = 'clipped')
                tf.summary.histogram('huber_loss', self.hub_loss)
            with tf.variable_scope('optim'):
                self.pred_loss = tf.reduce_mean(self.hub_loss,
                                                name = 'mean_loss')
                tf.summary.scalar('prediction_loss', self.pred_loss)
                self.weight_l2_loss = None
                for param in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name):
                    tf.summary.histogram('param/' + param.name, param)
                    if self.weight_l2_loss is None:
                        self.weight_l2_loss = tf.nn.l2_loss(param)
                    else:
                        self.weight_l2_loss = tf.add(self.weight_l2_loss, tf.nn.l2_loss(param))
                tf.summary.scalar('weight_loss', self.weight_l2_loss)
                self.total_loss = self.pred_loss + self.weight_decay * self.weight_l2_loss
                tf.summary.scalar('total_loss', self.total_loss)

                self.learning_rate_tensor = tf.placeholder(tf.float32, name='learning_rate')
                tf.summary.scalar('learning_rate', self.learning_rate_tensor)
                self.train_op = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate_tensor)\
                                        .minimize(self.total_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name))
            self.summary_op = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, scope=name))
            self.sess = sess # tf.Session()
            self.init_op = tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name))
            self.sess.run(self.init_op)
            self.saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name))

            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            self.train_log_writer = tf.summary.FileWriter(log_dir, self.sess.graph)
            self.train_log_writer.add_graph(self.sess.graph)

            self.udpate_step = 0

    def eval_batch(self, input_batch):
        #returns the forward pass of the input_batch
        return self.sess.run(self.fcn2,
                             feed_dict = {self.input_tensor:input_batch})

    def update_batch(self, input_batch, action_batch, target_batch):
        self.udpate_step += 1
        #modifies the weights according with y
        _, summary = self.sess.run([self.train_op, self.summary_op],
                      feed_dict = {self.input_tensor:input_batch,
                                   self.actions_tensor:action_batch,
                                   self.targets_tensor:target_batch,
                                   self.learning_rate_tensor: self.learning_rate()})
        self.train_log_writer.add_summary(summary, self.udpate_step)
        # self.sum_writer.flush()
    
    def save(self, filePath):
        self.saver.save(self.sess, filePath + 'model.ckpt')
    
    def load(self, filePath):
        self.saver.restore(self.sess, filePath + 'model.ckpt')
