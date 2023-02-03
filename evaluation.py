from src import config, util
import json
import pandas as pd
import random
import numpy as np
from run_pipeline import Pipeline

def get_learned_parameters(args):
    if args.policy in ["policy_SB", "policy_CB"]:
        learned_parameter_file = open(util.get_learning_file(args, "outer"), "r")
        learned_parameter_file = json.load(learned_parameter_file)
        learned_parameter_dict = learned_parameter_file.get("parameter_evolution")[args.read_in_iterations if args.read_in_iterations is not None else -1]
        return learned_parameter_dict
    else:
        return None


def get_outcome_file(args):
    return (args.result_directory +
                                f"Result_{util.get_obj_str(args)}_{util.get_startDate_str(args)}" +
                                f"_{util.get_endDate_str(args)}" +
                               (f"_{util.get_ext_horizon_str(args)}" if args.mode in ["policy_SB", "policy_CB", "sampling"] else '') +
                                f"_{util.get_sys_time_period_str(args)}_{util.get_sparsityFactor_str(args)}" +
                               (f"_{util.get_red_factor_str(args)}" if args.mode == "sampling" else '') +
                                f"_{util.get_heurTimeRange(args)}_{util.get_heurDistanceRange(args)}_{util.get_fleetSize_str(args)}" +
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

    learned_parameter_dict = get_learned_parameters(args)

    pipeline = Pipeline(args=args, random_state=numpy_random, optim_param=learned_parameter_dict)
    while pipeline.continue_running:
        pipeline.run()
    save_outcome(args, pipeline)
