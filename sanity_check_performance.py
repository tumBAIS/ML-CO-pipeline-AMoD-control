import random
from src import config
from pipeline.predictors.NeuralNetwork import NNModel
from pipeline.predictors.Linear import LinearModel
from src import util
import tensorflow as tf
from learning_problem.bfgs_optimizer import OptimizerBFGS
from learning_problem.sgd_optimizer import OptimizerSGD
from learning_problem.util_learning import get_training_instances, get_data, calculate_objective_value, get_edges, save_evolution_learning, decode_solution, get_learnable_edges
import numpy as np
import multiprocessing as mp
from pipeline.graph_generator import GraphGenerator
from pipeline.combinatorial_optimization import CombinatorialOptimizer
from run_pipeline import get_solution
import json


def get_loss_value(args, training_instance_name, predictor):
    combinatorial_optimizer = CombinatorialOptimizer(args)
    graph_generator = GraphGenerator(args)
    training_data = get_data(args, instance_file=training_instance_name)
    training_data_standardized = predictor.feature_data.standardize(training_data)
    len_vehicleList = training_data["len_vehicleList"]
    sink_index = training_data["len_verticesList"] - 1
    edge_features, node_features = predictor.retrieve_features(training_data_standardized, training_data["node_attributes"])
    edges = get_edges(args, training_data)
    weights = predictor.predict_weights(edge_features, node_features, edges)
    graph_generator.sink_index = sink_index
    learnable_edges, _ = get_learnable_edges(edges)

    # get online solution
    edgesList_solution, _ = get_solution(args, edges, weights, len_vehicleList, graph_generator, combinatorial_optimizer)
    y_solution = decode_solution(edgesList_solution, learnable_edges)

    # get offline solution
    edgesList_full_information_solution = [edge for edge_group in training_data["full_information_solution_edges"].values() for edge in edge_group]
    y_full_information_solution = decode_solution(edgesList_full_information_solution, learnable_edges)

    edges_to_requests = [edge in edges["vehicles_requests"] or edge in edges["requests_requests"] for edge in learnable_edges]

    absolute_fit = int(sum(np.array(y_full_information_solution) == np.array(y_solution)))
    absolute_fit_to_requests = int(sum(np.array(y_full_information_solution)[edges_to_requests] == np.array(y_solution)[edges_to_requests]))

    return absolute_fit, absolute_fit_to_requests


def run_sanity_check(args, predictor):
    training_instances = get_training_instances(args)
    loss_over_training_instances = []
    for training_instance_name in training_instances:
        absolute_fit, absolute_fit_to_requests = get_loss_value(args, training_instance_name, predictor)
        loss_over_training_instances.append([absolute_fit, absolute_fit_to_requests])
    save_loss(args, loss_over_training_instances)


def save_loss(args, loss_over_training_instances):
    file_output = open(util.get_sanity_file(args) + "--performance", "w")
    json.dump(loss_over_training_instances, file_output)
    file_output.close()


if __name__ == '__main__':
    # set random seed
    tf.keras.utils.set_random_seed(2)
    random.seed(2)
    np.random.seed(2)
    numpy_random = np.random.RandomState(seed=2)
    Nfeval = 1

    # read in parameters from command line / default values from modules/config.py file
    args = config.parser.parse_args()
    args.mode = "sanity_check"
    args = util.set_additional_args(args)
    util.create_directories(args)
    print(args)
    print("Number of processors: ", mp.cpu_count())
    predictors = {"Linear": (LinearModel, OptimizerBFGS), "NN": (NNModel, OptimizerSGD)}
    Predictor, Optimizer = predictors[args.model]

    predictor = Predictor(args)
    predictor.load_model()
    # run sanity check
    run_sanity_check(args, predictor)
