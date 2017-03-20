import logging
import numpy as np
import tensorflow as tf

from game_replay_parallel import GameEngine_Train
from util import ExponentialDecay, Constant
from q_learner import LinearLeaner, SimpleQ
from framework import Train


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch', 32, 'Batch size')
tf.app.flags.DEFINE_integer('total_updates', 100, 'Number of training updates')
tf.app.flags.DEFINE_float('learning_rate_initial', 1e-2, 'Initial learning rate')
tf.app.flags.DEFINE_float('learning_rate_final', 1e-2, 'Final learning rate')
tf.app.flags.DEFINE_float('weight_decay', 1e-4, 'Weight Decay')
tf.app.flags.DEFINE_float('decay', 0.99, 'Decay factor')
tf.app.flags.DEFINE_integer('batches_per_udpate', 1024, 'Number of batches per update')
tf.app.flags.DEFINE_integer('replay_limit_updates', 8, 'Limit for experience replay')
tf.app.flags.DEFINE_integer('light_eval', 100, 'Number of light evals')
tf.app.flags.DEFINE_integer('full_eval', 10, 'Number of full evals')
tf.app.flags.DEFINE_string('output_dir', 'outputs/q3', 'Output directory')

def main(argv=None):
    logging.disable(logging.INFO)

    game_engine = GameEngine_Train(FLAGS.batch, FLAGS.batches_per_udpate * FLAGS.replay_limit_updates)

    learning_rate = ExponentialDecay(FLAGS.learning_rate_initial, FLAGS.learning_rate_final, FLAGS.total_updates)
    game_info = game_engine.game_batch
    learner = LinearLeaner(game_info.state_shape, game_info.action_n, FLAGS.batch, learning_rate, FLAGS.weight_decay)
    q = SimpleQ(learner)

    updates_per_full_eval = FLAGS.total_updates // FLAGS.full_eval
    updates_per_light_eval = FLAGS.total_updates // FLAGS.light_eval
    train = Train(q, game_engine, FLAGS.decay, FLAGS.batches_per_udpate, FLAGS.batches_per_udpate,
                  updates_per_light_eval, updates_per_full_eval, updates_per_full_eval, 
                  FLAGS.total_updates, FLAGS.output_dir)
    
    train()

if __name__ == '__main__':
    tf.app.run()
