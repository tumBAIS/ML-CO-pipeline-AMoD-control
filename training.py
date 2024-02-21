import numpy as np
import random
from src import config
from prep.feature_data import Feature_data
from pipeline.predictors.NeuralNetwork import NNModel
from pipeline.predictors.Linear import LinearModel
from src import util
import tensorflow as tf
from learning_problem.bfgs_optimizer import OptimizerBFGS
from learning_problem.sgd_optimizer import OptimizerSGD
import multiprocessing as mp



if __name__ == '__main__':
    # set random seed
    tf.keras.utils.set_random_seed(2)
    random.seed(2)
    np.random.seed(2)
    numpy_random = np.random.RandomState(seed=2)
    Nfeval = 1

    # read in parameters from command line / default values from modules/config.py file
    args = config.parser.parse_args()
    args.mode = "training"
    args = util.set_additional_args(args)
    util.create_directories(args)
    print(args)
    print("Number of processors: ", mp.cpu_count())
    feature_data = Feature_data(args)
    feature_data.calculate_standardization_values(args=args)
    predictors = {"Linear": (LinearModel, OptimizerBFGS), "NN": (NNModel, OptimizerSGD)}
    Predictor, Optimizer = predictors[args.model]

    # generate Optimizer object
    optim = Optimizer(args, Predictor, feature_data)
    # learn the predictor over different training instances
    optim.optimize_problem()
