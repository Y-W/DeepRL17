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
        estimates = []
        s1s = []
        terminals = []
        rewards = []
        for _ in xrange(batch_num):
            trans = sampler(self.batch_size)
            s0_batch = np.zeros((self.batch_size,) + self.state_shape, dtype=np.float32)
            s1_batch = np.zeros((self.batch_size,) + self.state_shape, dtype=np.float32)
            action_batch = np.zeros(self.batch_size, dtype=np.int_)
            terminal_batch = np.zeros(self.batch_size, dtype=np.int_)
            reward_batch = np.zeros(self.batch_size, dtype=np.float32)
            for i in xrange(self.batch_size):
                s0_batch[i] = trans[i][0]
                action_batch[i] = trans[i][1]
                reward_batch[i] = trans[i][2]
                if trans[i][3] is not None:
                    s1_batch[i] = trans[i][3]
                    terminal_batch[i] = 1
                else:
                    terminal_batch[i] = 0
            est = self.learner.eval_batch(s1_batch)
            inputs.append(s0_batch)
            actions.append(action_batch)
            estimates.append(est)
            s1s.append(s1_batch)
            terminals.append(terminal_batch)
            rewards.append(reward_batch)
        for i in xrange(batch_num):
            target_batch = rewards[i]
            s1_act = np.argmax(self.learner.eval_batch(s1s[i]), axis=1)
            target_batch += decay * terminals[i][:, np.newaxis] * estimates[i][np.arange(self.batch_size), s1_act]
            self.learner.update_batch(inputs[i], actions[i], target_batch)

    def save(self, dir_path):
        self.learner.save(os.path.join(dir_path, 'learner.ckpt'))
    
    def load(self, dir_path):
        self.learner.load(os.path.join(dir_path, 'learner.ckpt'))


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
                                                kernel_initializer = tf.contrib.layers.xavier_initializer(),
                                                bias_initializer = tf.constant_initializer(),
                                                activation = tf.nn.relu,
                                                name = 'conv1')
                tf.summary.histogram('conv1', self.conv1)
                self.conv2 = tf.layers.conv2d(inputs = self.conv1,
                                                filters = 32,
                                                kernel_size= (4,4),
                                                strides = (2,2),
                                                data_format = 'channels_first',
                                                kernel_initializer = tf.contrib.layers.xavier_initializer(),
                                                bias_initializer = tf.constant_initializer(),
                                                activation = tf.nn.relu,
                                                name = 'conv2')
                tf.summary.histogram('conv2', self.conv2)
            with tf.variable_scope('FCLs'):
                self.flat = tf.reshape(self.conv2,(batch_size,-1),name='flat') 
                self.fcn1 = tf.layers.dense(inputs = self.flat,
                                            units = 256,
                                            activation = tf.nn.relu,
                                            kernel_initializer = tf.contrib.layers.xavier_initializer(),
                                            name = 'fcn1')
                tf.summary.histogram('fcn1', self.fcn1)
                self.fcn2 = tf.layers.dense(inputs = self.fcn1,
                                            units = action_n,
                                            activation = None,
                                            kernel_initializer = tf.contrib.layers.xavier_initializer(),
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

class DeepDuelLearner(Learner):
    def __init__(self, name, sess, state_shape, action_n, batch_size, learning_rate, weight_decay, log_dir):
        self.state_shape = state_shape
        self.action_n = action_n
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        with tf.variable_scope(name):
            self.input_tensor = tf.placeholder(tf.float32, 
                                            shape = (batch_size,) + state_shape,
                                            name = 'input')
            tf.summary.histogram('input', self.input_tensor)
            with tf.variable_scope('value_net'):
                self.value_conv1_tensor = tf.layers.conv2d(inputs = self.input_tensor,
                                                filters = 16,
                                                kernel_size= (8,8),
                                                strides = (4,4),
                                                data_format = 'channels_first',
                                                kernel_initializer = tf.contrib.layers.xavier_initializer(),
                                                bias_initializer = tf.constant_initializer(),
                                                activation = tf.nn.relu,
                                                name = 'conv1')
                tf.summary.histogram('conv1', self.value_conv1_tensor)
                self.value_conv2_tensor = tf.layers.conv2d(inputs = self.value_conv1_tensor,
                                                filters = 32,
                                                kernel_size= (4,4),
                                                strides = (2,2),
                                                data_format = 'channels_first',
                                                kernel_initializer = tf.contrib.layers.xavier_initializer(),
                                                bias_initializer = tf.constant_initializer(),
                                                activation = tf.nn.relu,
                                                name = 'conv2')
                tf.summary.histogram('conv2', self.value_conv2_tensor)
                self.value_fcn1_tensor = tf.layers.dense(inputs = tf.reshape(self.value_conv2_tensor, (batch_size, -1)),
                                            units = 256,
                                            activation = tf.nn.relu,
                                            kernel_initializer = tf.contrib.layers.xavier_initializer(),
                                            bias_initializer = tf.constant_initializer(),
                                            name = 'fcn1')
                tf.summary.histogram('fcn1', self.value_fcn1_tensor)
                self.value_fcn2_tensor = tf.layers.dense(inputs = self.value_fcn1_tensor,
                                            units = 1,
                                            activation = None,
                                            kernel_initializer = tf.contrib.layers.xavier_initializer(),
                                            bias_initializer = tf.constant_initializer(),
                                            name = 'output')
                tf.summary.histogram('output', self.value_fcn2_tensor)

            with tf.variable_scope('action_net'):
                self.action_conv1_tensor = tf.layers.conv2d(inputs = self.input_tensor,
                                                filters = 16,
                                                kernel_size= (8,8),
                                                strides = (4,4),
                                                data_format = 'channels_first',
                                                kernel_initializer = tf.contrib.layers.xavier_initializer(),
                                                bias_initializer = tf.constant_initializer(),
                                                activation = tf.nn.relu,
                                                name = 'conv1')
                tf.summary.histogram('conv1', self.action_conv1_tensor)
                self.action_conv2_tensor = tf.layers.conv2d(inputs = self.action_conv1_tensor,
                                                filters = 32,
                                                kernel_size= (4,4),
                                                strides = (2,2),
                                                data_format = 'channels_first',
                                                kernel_initializer = tf.contrib.layers.xavier_initializer(),
                                                bias_initializer = tf.constant_initializer(),
                                                activation = tf.nn.relu,
                                                name = 'conv2')
                tf.summary.histogram('conv2', self.action_conv2_tensor)
                self.action_fcn1_tensor = tf.layers.dense(inputs = tf.reshape(self.action_conv2_tensor, (batch_size, -1)),
                                            units = 256,
                                            activation = tf.nn.relu,
                                            kernel_initializer = tf.contrib.layers.xavier_initializer(),
                                            bias_initializer = tf.constant_initializer(),
                                            name = 'fcn1')
                tf.summary.histogram('fcn1', self.action_fcn1_tensor)
                self.action_fcn2_tensor = tf.layers.dense(inputs = self.action_fcn1_tensor,
                                            units = action_n,
                                            activation = None,
                                            kernel_initializer = tf.contrib.layers.xavier_initializer(),
                                            bias_initializer = tf.constant_initializer(),
                                            name = 'fcn2')
                tf.summary.histogram('fcn2', self.action_fcn2_tensor)
                self.action_output = tf.subtract(self.action_fcn2_tensor, tf.reduce_mean(self.action_fcn2_tensor, axis=1, keep_dims=True), name='output')
                tf.summary.histogram('output', self.action_output)
            
            self.total_output = tf.add(self.value_fcn2_tensor, self.action_output, name='total_output')
            tf.summary.histogram('total_output', self.total_output)

            with tf.variable_scope('Loss'):
                self.actions_tensor = tf.placeholder(tf.int32,
                                                shape = batch_size,
                                                name = 'actions')
                self.act_one_hot = tf.one_hot(indices = self.actions_tensor,
                                                depth = action_n,
                                                on_value = 1.0,
                                                off_value = 0.0,
                                                name = 'act_one_hot')
                self.q_pred = tf.reduce_sum(self.total_output * self.act_one_hot,
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
        return self.sess.run(self.total_output,
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
