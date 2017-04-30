import logging
import numpy as np
from q_learner import DeepLearner
from framework import Onp_seq

output_dir = 'outputs/mc_onp_seq'
learner_class = DeepLearner
update_per_sim = 1
sim_per_light_eval = 3125
sim_per_record_save = 31250
total_sim = 31250

def main(argv=None):
    logging.disable(logging.INFO)

    Onp_seq(output_dir,
          learner_class,
          update_per_sim,
          sim_per_light_eval,
          sim_per_record_save,
          total_sim)()

if __name__ == '__main__':
    main()
