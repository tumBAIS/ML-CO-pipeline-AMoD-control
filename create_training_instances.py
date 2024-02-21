import random
import numpy as np
from src import config, util
from run_pipeline import Pipeline

if __name__ == '__main__':

    # set random seed
    random.seed(123)
    numpy_random = np.random.RandomState(seed=123)
    args = config.parser.parse_args()
    args.mode = "create_training_instance"
    assert args.policy in ["policy_SB", "policy_CB"]
    args = util.set_additional_args(args)  # set additional args that automatically derive from set args
    print(args)

    pipeline = Pipeline(args=args, random_state=numpy_random)
    pipeline.run()
