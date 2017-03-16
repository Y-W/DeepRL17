import random
import logging
import numpy as np
import tensorflow as tf

from game import GameEngine_Train
from util import ExponentialDecay, Constant
from q_learner import LinearLeaner, SimpleQ
from replay_pool import NoPool
from framework import Train


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('random_seed', 43, 'Random seed')
tf.app.flags.DEFINE_integer('batch', 32, 'Batch size')
tf.app.flags.DEFINE_integer('total_updates', 100000, 'Number of training updates')
tf.app.flags.DEFINE_float('learning_rate_initial', 1e-1, 'Initial learning rate')
tf.app.flags.DEFINE_float('learning_rate_final', 1e-2, 'Final learning rate')
tf.app.flags.DEFINE_float('decay', 0.99, 'Decay factor')
tf.app.flags.DEFINE_integer('batches_per_udpate', 1, 'Number of batches per update')
tf.app.flags.DEFINE_integer('init_game_batches', 0, 'Number of game batches to run before everything')
tf.app.flags.DEFINE_integer('updates_per_ckpt', 1000, 'Number of updates per checkpoint')
tf.app.flags.DEFINE_integer('full_eval', 10, 'Number of full evals')
tf.app.flags.DEFINE_string('output_dir', None, 'Output directory')

def main(argv=None):
    assert FLAGS.output_dir is not None

    random.seed(FLAGS.random_seed)
    np.random.seed(FLAGS.random_seed)

    logging.disable(logging.INFO)

    game_engine = GameEngine_Train(FLAGS.batch)

    learning_rate = ExponentialDecay(FLAGS.learning_rate_initial, FLAGS.learning_rate_final, FLAGS.total_updates)
    game_info = game_engine.game_batch
    learner = LinearLeaner(game_info.state_shape, game_info.action_n, FLAGS.batch, learning_rate)
    q = SimpleQ(learner)

    pool = NoPool(FLAGS.batches_per_udpate * FLAGS.batch)

    updates_per_full_eval = FLAGS.total_updates // FLAGS.full_eval
    train = Train(q, pool, game_engine, FLAGS.decay, FLAGS.init_game_batches,
                  FLAGS.batches_per_udpate, FLAGS.batches_per_udpate, FLAGS.updates_per_ckpt, 
                  updates_per_full_eval, FLAGS.updates_per_ckpt, FLAGS.total_updates, FLAGS.output_dir)
    
    train()

if __name__ == '__main__':
    tf.app.run()
