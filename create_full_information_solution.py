import random
import numpy as np
from src import config, util
from run_pipeline import Pipeline


if __name__ == '__main__':

    # set seeds
    random.seed(123)
    numpy_random = np.random.RandomState(seed=123)
    args = config.parser.parse_args()
    args.mode = "create_full_information_solution"
    args.policy = "offline"
    args = util.set_additional_args(args)  # set additional args that automatically derive from set args

    pipeline = Pipeline(args=args, random_state=numpy_random)
    pipeline.run()


