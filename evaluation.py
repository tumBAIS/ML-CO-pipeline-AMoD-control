from src import config, util
import pandas as pd
import random
import numpy as np
from run_pipeline import Pipeline


def get_outcome_file(args):
    return (args.result_directory +
                                f"Result_{util.get_obj_str(args)}_{util.get_startDate_str(args)}" +
                                f"_{util.get_endDate_str(args)}" +
                               (f"_{util.get_ext_horizon_str(args)}" if args.mode in ["policy_SB", "policy_CB", "sampling"] else '') +
                                f"_{util.get_sys_time_period_str(args)}_{util.get_sparsityFactor_str(args)}" +
                               (f"_{util.get_red_factor_str(args)}" if args.mode == "sampling" else '') +
                                f"_{util.get_heurTimeRange(args)}_{util.get_heurDistanceRange(args)}_{util.get_fleetSize_str(args)}" +
                                f"_{util.get_predictor(args)}" +
                               (f"_{util.get_trIteration(args)}" if args.read_in_iterations is not None else ''))


def save_outcome(args, simulator):
    # save the evaluation in an output file at the end of the simulation.
    evaluationDataFrame = pd.DataFrame(simulator.evaluationList)
    evaluationDataFrame.to_csv(get_outcome_file(args))


if __name__ == '__main__':

    # set random seed
    random.seed(123)
    numpy_random = np.random.RandomState(seed=123)
    np.random.seed(seed=123)
    args = config.parser.parse_args()

    args.mode = "evaluation"
    args = util.set_additional_args(args)  # set additional args that automatically derive from set args

    pipeline = Pipeline(args=args, random_state=numpy_random)
    while pipeline.continue_running:
        pipeline.run()
    save_outcome(args, pipeline)