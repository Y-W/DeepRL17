import logging
import numpy as np
from q_learner import DeepDuelLearner
from framework import Train

output_dir = 'outputs/q7'
learner_class = DeepDuelLearner
use_replay = True
use_doubleQ = False
update_per_sim = 8
sim_per_sync = 1250
sim_per_light_eval = 3125
sim_per_record_save = 31250
total_sim = 312500

def main(argv=None):
    logging.disable(logging.INFO)

    Train(output_dir,
          learner_class,
          use_replay,
          use_doubleQ,
          update_per_sim,
          sim_per_sync,
          sim_per_light_eval,
          sim_per_record_save,
          total_sim)()

if __name__ == '__main__':
    main()
