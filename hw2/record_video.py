# python record_video.py q5 31250

import sys
import os
import tensorflow as tf
from q_learner import LinearLeaner, DeepLearner, DeepDuelLearner
from game_replay_parallel import GameEngine_Recorded


def main():
    logging.disable(logging.INFO)

    ques = sys.argv[1]
    assert ques in ['q2', 'q3', 'q4', 'q5', 'q6', 'q7']
    n = int(sys.argv[2])
    print 'Question', ques, '#iter', n,

    learnerClass = LinearLeaner
    if ques in ['q5', 'q6']:
        learnerClass = DeepLearner
    if ques in ['q7']:
        learnerClass = DeepDuelLearner
    
    modelPath = 'outputs/%s/model/model-%i.ckpt' % (ques, n)
    print 'ModelPath', modelPath
    assert os.path.exists(modelPath + '.index')

    recordPath = 'outputs/%s/video/video-%i/' % (ques, n)
    print 'RecordPath', recordPath
    if not os.path.exists(recordPath):
        os.makedirs(recordPath)

    game = GameEngine_Recorded(record_dir, 1)

    sess = tf.Session()
    learner = learnerClass('online_Q', sess, game.games.state_shape, game.games.action_n, 1, None, None)
    game(learner.eval_batch_action)

    game.close()


if __name__ == '__main__':
    main()
