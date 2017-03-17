# Yijie Wang (wyijie93@gmail.com)

import os
import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('model_dir', None, 'Directory for model checkpoints')

tf.app.flags.DEFINE_integer('scales', 4, 'Number of scales')
tf.app.flags.DEFINE_string('conv_schema', '3,16', 'Convolution layers schema')

tf.app.flags.DEFINE_float('balance_tolerance', 0.1, 'Tolerance of splitting imbalance')
tf.app.flags.DEFINE_float('balance_loss', 10.0, 'Weight of balance-split loss')

tf.app.flags.DEFINE_integer('training_iterations', 100, 'Number of training iterations')
tf.app.flags.DEFINE_float('learning_rate_initial', 1e-3, 'Initial learning rate')
# tf.app.flags.DEFINE_float('learning_rate_final', 1e-3, 'Final learning rate')
tf.app.flags.DEFINE_float('momentum', 0.9, 'Momentum value')

class ConvBranch:
    def __init__(self, name, input_dim, label_dim, model_dir=None):
        if model_dir is None:
            model_dir = FLAGS.model_dir
        assert input_dim[0] == label_dim[0]
        self.name = name
        self.model_dir = model_dir
        self.tf_graph = tf.Graph()
        self.input_dim = input_dim
        self.label_dim = label_dim
        with self.tf_graph.as_default():
            self.data = tf.placeholder(tf.float32, shape=input_dim, name='data')
            self.label = tf.placeholder(tf.float32, shape=label_dim, name='label')
        self.makeConvBranch()
        self.makeEndPoints()

    
    def makeConvBranch(self, input=None, k_downsamp=None, convSeries=None, reuse_variable=False):
        if input is None:
            input = self.data
        if k_downsamp is None:
            k_downsamp = FLAGS.scales
        if convSeries is None:
            convSeries = [(int(conv.split(',')[0]), int(conv.split(',')[1])) for conv in FLAGS.conv_schema.split(';')]

        with self.tf_graph.as_default():
            with tf.variable_scope('conv_branch', values=[input], dtype=tf.float32, reuse=reuse_variable) as sc:
                scale_outputs = []
                for k in xrange(k_downsamp):
                    with tf.variable_scope('scale_%i' % k, [input]):
                        last_output = input
                        for i, (s, c) in enumerate(convSeries):
                            with tf.variable_scope('conv_%i' % i, [last_output]):
                                filters = tf.get_variable('filters',
                                                        shape=(
                                                            s, s, last_output.get_shape()[3].value, c),
                                                        initializer=tf.random_normal_initializer(
                                                            stddev=np.sqrt(2.0 / last_output.get_shape()[3].value)),
                                                        trainable=True)
                                last_output = tf.nn.conv2d(
                                    last_output, filters, strides=(1, 1, 1, 1), padding='VALID')
                                bias = tf.get_variable('bias_%i' % i, shape=(c,),
                                                    initializer=tf.zeros_initializer(), trainable=True)
                                last_output = tf.nn.relu(last_output + bias)
                        last_output = tf.reduce_mean(last_output, axis=(1, 2))
                        scale_outputs.append(last_output)
                    if k + 1 < k_downsamp:
                        _, h, w, _ = input.get_shape()
                        input = tf.image.resize_bilinear(
                            input, (h.value // 2, w.value // 2), name=('downsamp_%i' % k))
                all_outputs = tf.concat(scale_outputs, axis=1)
                with tf.variable_scope('dense', [all_outputs]):
                    weights = tf.get_variable('weights', shape=(all_outputs.get_shape()[1].value, 1),
                                            initializer=tf.random_normal_initializer(
                                                stddev=np.sqrt(1.0 / all_outputs.get_shape()[1].value)),
                                            trainable=True)
                    bias = tf.get_variable('bias', shape=(1, 1), initializer=tf.zeros_initializer(), trainable=True)
                    preact = tf.squeeze(tf.add(tf.matmul(all_outputs, weights), bias), name='preact')
                # self.branch = tf.sigmoid(preact, name='branching')
                self.branch = tf.hard_gate(tf.tanh(preact), name='branching')

    @staticmethod
    def gini_impurity(dist):
        return 1.0 - tf.reduce_sum(tf.multiply(dist, dist))

    def makeEndPoints(self, branching=None, label=None, balance_split_weight=None):
        if branching is None:
            branching = self.branch
        if label is None:
            label = self.label
        if balance_split_weight is None:
            balance_split_weight = FLAGS.balance_loss

        with self.tf_graph.as_default():
            imbalance = tf.abs(0.5 - tf.reduce_mean(branching))
            imbalance = tf.nn.relu(imbalance - FLAGS.balance_tolerance)
            self.split_loss = tf.square(imbalance)
            dist = tf.reduce_mean(tf.multiply(tf.expand_dims(branching, -1), label), axis=0)
            self.gini_loss = (1.0 - tf.reduce_mean(branching)) * ConvBranch.gini_impurity(tf.reduce_mean(label, axis=0) - dist) \
                    + tf.reduce_mean(branching) * ConvBranch.gini_impurity(dist)
            self.total_loss = self.split_loss * balance_split_weight + self.gini_loss

            self.branch_result = tf.greater(branching, 0.5)
    
    def setup_training(self):
        with self.tf_graph.as_default():
            self.global_step = tf.train.get_or_create_global_step()
            # self.learning_rate = tf.train.exponential_decay(FLAGS.learning_rate_initial, self.global_step, 
            #                          FLAGS.training_iterations, FLAGS.learning_rate_final / FLAGS.learning_rate_initial, 
            #                          staircase=False, name='learning_rate')
            self.learning_rate = FLAGS.learning_rate_initial
            self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=FLAGS.momentum)
            self.train_step = self.optimizer.minimize(self.total_loss, global_step=self.global_step, name='optimize_step')

    def train(self, data, label):
        self.setup_training()
        with tf.Session(graph=self.tf_graph) as sess:
            tf.logging.set_verbosity(tf.logging.INFO)
            sess.run(tf.global_variables_initializer())
            for _ in xrange(FLAGS.training_iterations):
                step, split_loss, gini_loss, _ = sess.run((self.global_step, self.split_loss, self.gini_loss, self.train_step),
                                                          feed_dict={self.data : data, self.label : label})
                tf.logging.info('Step=%i split_loss=%f gini_loss=%f' % (step, split_loss, gini_loss))
            saver = tf.train.Saver(tf.trainable_variables())
            saver.save(sess, os.path.join(self.model_dir, self.name+'.ckpt'))

    def eval_batches(self, inputs):
        with tf.Session(graph=self.tf_graph) as sess:
            saver = tf.train.Saver(tf.trainable_variables())
            saver.restore(sess, os.path.join(self.model_dir, self.name+'.ckpt'))
            results = []
            for input in inputs:
                result = sess.run(self.branch_result, feed_dict={self.data : input})
                results.append(result)
            return results
    
    def eval(self, inputs):
        input_batches = []
        p = 0
        while p < len(inputs):
            tmp = np.zeros(self.input_dim, dtype=np.float32)
            k = min(len(inputs) - p, self.input_dim[0])
            for i in xrange(k):
                tmp[i] = inputs[p+i]
            input_batches.append(tmp)
            p += k
        result_batches = self.eval_batches(input_batches)
        results = []
        for r in result_batches:
            k = min(len(inputs) - len(results), self.input_dim[0])
            results.extend(r.tolist()[:k])
        return results
