import random
from src import config
from pipeline.predictors.NeuralNetwork import NNModel
from pipeline.predictors.Linear import LinearModel
from src import util
import tensorflow as tf
from learning_problem.bfgs_optimizer import OptimizerBFGS
from learning_problem.sgd_optimizer import OptimizerSGD
from learning_problem.util_learning import get_training_instances, get_data, calculate_objective_value, get_edges, decode_solution, get_learnable_edges
import numpy as np
from multiprocessing import Pool
import multiprocessing as mp
from functools import partial
from pipeline.graph_generator import GraphGenerator
from pipeline.combinatorial_optimization import CombinatorialOptimizer
from run_pipeline import get_solution
import json


def calculate_loss_for_instance(kwargs, perturbation_counter):
    np.random.seed(perturbation_counter)
    args, edges, weights, len_vehicleList, sink_index, learnable_edges = kwargs

    def perturb_weights(weights):
        weights_perturbed = {}
        for classe in weights.keys():
            if classe in ["source_vehicles", "vehicles_sink", "requests_sink", "artLocVertices_sink"]:
                weights_perturbed[classe] = weights[classe]
            else:
                weights_perturbed[classe] = list(weights[classe] + np.random.normal(loc=0, scale=args.perturbed_optimizer[1], size=len(weights[classe])))
        return weights_perturbed

    graph_generator = GraphGenerator(args)
    graph_generator.sink_index = sink_index
    combinatorial_optimizer = CombinatorialOptimizer(args)
    weights_perturbed = perturb_weights(weights)
    edgesList, weightsList_perturbed = graph_generator.get_edges_weights_lists(edges, weights_perturbed)
    edgesList_solution, _ = get_solution(args, edges, weights_perturbed, len_vehicleList, graph_generator, combinatorial_optimizer)
    objective_value_online = calculate_objective_value(edgesList=edgesList, weightsList=weightsList_perturbed, solution_edges=edgesList_solution)
    y_solution = decode_solution(edgesList_solution, learnable_edges)
    return objective_value_online, y_solution


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
    edgesList, weightsList = graph_generator.get_edges_weights_lists(edges, weights)

    # get online solution
    edgesList_solution, _ = get_solution(args, edges, weights, len_vehicleList, graph_generator, combinatorial_optimizer)
    objective_value_online = calculate_objective_value(edgesList=edgesList, weightsList=weightsList, solution_edges=edgesList_solution)

    # get offline solution
    edgesList_full_information_solution = [edge for edge_group in training_data["full_information_solution_edges"].values() for edge in edge_group]
    objective_value_offline = calculate_objective_value(edgesList=edgesList, weightsList=weightsList, solution_edges=edgesList_full_information_solution)

    with Pool(mp.cpu_count() - 2) as pool:
        saver = pool.map(partial(calculate_loss_for_instance, (args, edges, weights, len_vehicleList, sink_index, learnable_edges)), list(range(int(args.perturbed_optimizer[0]))))

    objective_value_saver = [value[0] for value in saver]
    objective_value_mean = np.mean(objective_value_saver)
    FYloss = float(objective_value_mean - objective_value_offline)
    NNloss = float(objective_value_online - objective_value_offline)
    return FYloss, NNloss


def run_sanity_check(args, predictor):
    training_instances = get_training_instances(args)
    loss_over_training_instances = []
    for training_instance_name in training_instances:
        FYloss, NNloss = get_loss_value(args, training_instance_name, predictor)
        loss_over_training_instances.append([FYloss, NNloss])
    save_loss(args, loss_over_training_instances)


def save_loss(args, loss_over_training_instances):
    file_output = open(util.get_sanity_file(args), "w")
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
