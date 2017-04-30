import logging
import numpy as np
from q_learner import DeepLearner
from framework import SARSA

output_dir = 'outputs/mc_onp_sarsa'
learner_class = DeepLearner
update_per_sim = 1
sim_per_light_eval = 31250
sim_per_record_save = 31250
total_sim = 312500

def main(argv=None):
    logging.disable(logging.INFO)

    SARSA(output_dir,
          learner_class,
          update_per_sim,
          sim_per_light_eval,
          sim_per_record_save,
          total_sim)()

if __name__ == '__main__':
    main()
