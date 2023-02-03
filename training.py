import numpy as np
import random
from src import config
import scipy
import time
import os
import pandas as pd
import json
from prep.feature_data import Feature_data
from multiprocessing import Pool
import multiprocessing as mp
from functools import partial
import pickle
from collections import OrderedDict
from pipeline.predictor import Predictor
from pipeline.graph_generator import GraphGenerator
from pipeline.combinatorial_optimization import CombinatorialOptimizer
from src import util


def get_gradient(edges, dict_features, predictor):
    if args.policy == "policy_SB":
        parameter_keys = predictor.refilter_parameter_dict(["learning_vector_origin", "learning_vector_destination", "learning_vector_future", "learning_vector_relation"]).keys()
    elif args.policy == "policy_CB":
        parameter_keys = predictor.refilter_parameter_dict(["learning_vector_origin", "learning_vector_destination", "learning_vector_future", "learning_vector_relation",
                                                            "learning_vector_location", "learning_vector_location_relation", "learning_vector_capacity"]).keys()
    else:
        raise Exception("Wrong policy defined")

    # vehicles to requests
    gradient_features_vehicles_requests = pd.DataFrame(edges, columns=["starting", "arriving"]).merge(dict_features["features_edges_vehicles_requests"], left_on=["starting", "arriving"], right_on=["starting", "arriving"])
    # requests to requests
    gradient_features_requests_requests = pd.DataFrame(edges, columns=["starting", "arriving"]).merge(dict_features["features_edges_requests_requests"], left_on=["starting", "arriving"], right_on=["starting", "arriving"]) if len(dict_features["features_edges_requests_requests"]) > 0 else pd.DataFrame()
    # requests to artRebVertices
    gradient_features_requests_artRebVertices = pd.DataFrame(edges, columns=["starting", "arriving"]).merge(dict_features["features_edges_requests_artRebVertices"], left_on=["starting", "arriving"], right_on=["starting", "arriving"]) if len(dict_features["features_edges_requests_artRebVertices"]) > 0 else pd.DataFrame()
    # vehicles to artRebVertices
    gradient_features_vehicles_artRebVertices = pd.DataFrame(edges, columns=["starting", "arriving"]).merge(dict_features["features_edges_vehicles_artRebVertices"], left_on=["starting", "arriving"], right_on=["starting", "arriving"]) if len(dict_features["features_edges_vehicles_artRebVertices"]) > 0 else pd.DataFrame()
    # artRebVertices to artCapVertices
    if args.policy == "policy_CB":
        gradient_features_artRebVertices_artCapVertices = pd.DataFrame(edges, columns=["starting", "arriving"]).merge(dict_features["features_edges_artRebVertices_artCapVertices"], left_on=["starting", "arriving"], right_on=["starting", "arriving"])
    else:
        gradient_features_artRebVertices_artCapVertices = pd.DataFrame()
    gradient_features = pd.concat([gradient_features_vehicles_requests, gradient_features_requests_requests, gradient_features_requests_artRebVertices, gradient_features_vehicles_artRebVertices, gradient_features_artRebVertices_artCapVertices])
    assert len(gradient_features) == len(edges)
    return gradient_features[parameter_keys].sum(axis=0)


def calculate_loss(edgeList, weightsList, offline_solution_edges, online_solution_edges):
    def get_solution_weights(solution_edges):
        weights = pd.concat([pd.DataFrame(edgeList, columns=["starting", "arriving"]), pd.DataFrame({"weight": weightsList})], axis=1).merge(pd.DataFrame(solution_edges, columns=["starting", "arriving"]), left_on=["starting", "arriving"], right_on=["starting", "arriving"])
        assert len(weights) == len(solution_edges)
        return list(weights["weight"])

    objective_value_online = -1 * sum(get_solution_weights(online_solution_edges))
    print("The objective value ONLINE is: {}".format(objective_value_online))
    objective_value_offline = -1 * sum(get_solution_weights(offline_solution_edges))
    print("The objective value OFFLINE is: {}".format(objective_value_offline))
    return objective_value_online, objective_value_offline


def run_training(training_instance, predictor, graph_generator, combinatorial_optimizer):

    edges_source_vehicles = training_instance["edges_source_vehicles"]
    edges_vehicles_sink = training_instance["edges_vehicles_sink"]
    edges_requests_sink = training_instance["edges_requests_sink"]
    edges_artLocVertices_sink = training_instance["edges_artLocVertices_sink"]
    full_information_solution_edges = [edge for edges in training_instance["full_information_solution_edges"].values() for edge in edges]
    graph_generator.sink_index = training_instance["len_verticesList"] - 1
    len_vehicleList = training_instance["len_vehicleList"]

    # we predict the weights on edges to artificial requests
    weights_source_vehicles = predictor.calculate_weights_source_vehicles(edges_source_vehicles)
    weights_vehicles_requests = predictor.calculate_weights_vehicles_requests(training_instance["features_edges_vehicles_requests"])
    weights_requests_requests = predictor.calculate_weights_requests_requests(training_instance["features_edges_requests_requests"])
    weights_vehicles_sink = predictor.calculate_weights_vehicles_sink(edges_vehicles_sink)
    weights_requests_sink = predictor.calculate_weights_requests_sink(edges_requests_sink)
    weights_vehicles_artRebVertices = predictor.calculate_weights_vehicles_artRebVertices(training_instance["features_edges_vehicles_artRebVertices"])
    weights_requests_artRebVertices = predictor.calculate_weights_requests_artRebVertices(training_instance["features_edges_requests_artRebVertices"])
    weights_artRebVertices_artCapVertices = predictor.calculate_weights_artRebVertices_artCapVertices(training_instance["features_edges_artRebVertices_artCapVertices"])
    weights_artLocVertices_sink = predictor.calculate_weights_artLocVertices_sink(edges_artLocVertices_sink)

    # the graph generator creates the graph
    edgeList, weightsList = graph_generator.generate_graph(edges_source_vehicles=edges_source_vehicles,
                                                                edges_vehicles_requests=[tuple(edge) for edge in training_instance["features_edges_vehicles_requests"][["starting", "arriving"]].values.tolist()],
                                                                edges_requests_requests=[tuple(edge) for edge in training_instance["features_edges_requests_requests"][["starting", "arriving"]].values.tolist()] if len(training_instance["features_edges_requests_requests"]) > 0 else [],
                                                                edges_vehicles_artRebVertices=[tuple(edge) for edge in training_instance["features_edges_vehicles_artRebVertices"][["starting", "arriving"]].values.tolist()] if len(training_instance["features_edges_vehicles_artRebVertices"]) > 0 else [],
                                                                edges_requests_artRebVertices=[tuple(edge) for edge in training_instance["features_edges_requests_artRebVertices"][["starting", "arriving"]].values.tolist()] if len(training_instance["features_edges_requests_artRebVertices"]) > 0 else [],
                                                                edges_artRebVertices_artCapVertices=[tuple(edge) for edge in training_instance["features_edges_artRebVertices_artCapVertices"][["starting", "arriving"]].values.tolist()] if args.policy == "policy_CB" else [],
                                                                edges_vehicles_sink=edges_vehicles_sink,
                                                                edges_requests_sink=edges_requests_sink,
                                                                edges_artLocVertices_sink=edges_artLocVertices_sink,
                                                                weights_source_vehicles=weights_source_vehicles,
                                                                weights_vehicles_requests=weights_vehicles_requests,
                                                                weights_requests_requests=weights_requests_requests,
                                                                weights_vehicles_artRebVertices=weights_vehicles_artRebVertices,
                                                                weights_requests_artRebVertices=weights_requests_artRebVertices,
                                                                weights_artRebVertices_artCapVertices=weights_artRebVertices_artCapVertices,
                                                                weights_vehicles_sink=weights_vehicles_sink,
                                                                weights_requests_sink=weights_requests_sink,
                                                                weights_artLocVertices_sink=weights_artLocVertices_sink)

    # run_k_disjoint shortest path+
    k, edges, time_needed_k_disjoint = combinatorial_optimizer.solve_optimization_problem(edges=edgeList, weights=weightsList, num_vertices=graph_generator.sink_index+1, num_k=len_vehicleList)

    # we remove unimportant edges, and when we added artificial edges we need to remove them
    edges, _ = graph_generator.deleting_sink_source_edges(edges)
    edgeList, weightsList = graph_generator.deleting_sink_source_edges(edgeList, weightsList)

    # calculate loss
    objective_value_online, objective_value_offline = calculate_loss(edgeList=edgeList, weightsList=weightsList, offline_solution_edges=full_information_solution_edges, online_solution_edges=edges)

    gradient_online = get_gradient(edges, training_instance, predictor)
    gradient_offline = get_gradient(full_information_solution_edges, training_instance, predictor)
    return objective_value_online, objective_value_offline, gradient_online, gradient_offline


class Optimizer():
    def __init__(self):
        self.loss_evolution_inner = []
        self.gradient_evolution_inner = []
        self.parameter_evolution_inner = []
        self.parameter_evolution_outer = []
        self.max_sec = args.stop_max_time * 60
        self.start_time = time.time()


    def get_parameter_set(self, param_dict):
        parameter_list = []
        for classes in param_dict:
            for key in list(param_dict[classes].keys()):
                parameter_list.append(param_dict[classes][key])
        return np.asarray(parameter_list)

    def set_parameter_set(self, param_list):
        if not isinstance(param_list, list):
            param_list = param_list.tolist()
        param_dict = {}
        counter = 0
        for classes in feature_data.dict.keys():
            param_dict[classes] = OrderedDict()
            for key in feature_data.dict[classes].keys():
                param_dict[classes][key] = param_list[counter]
                counter = counter + 1

        assert list(param_dict["learning_vector_origin"].keys()) == list(feature_data.dict["learning_vector_origin"].keys())
        assert list(param_dict["learning_vector_destination"].keys()) == list(feature_data.dict["learning_vector_destination"].keys())
        assert list(param_dict["learning_vector_future"].keys()) == list(feature_data.dict["learning_vector_future"].keys())
        assert list(param_dict["learning_vector_relation"].keys()) == list(feature_data.dict["learning_vector_relation"].keys())
        if args.policy == "policy_CB":
            assert list(param_dict["learning_vector_location"].keys()) == list(feature_data.dict["learning_vector_location"].keys())
            assert list(param_dict["learning_vector_location_relation"].keys()) == list(feature_data.dict["learning_vector_location_relation"].keys())
            assert list(param_dict["learning_vector_capacity"].keys()) == list(feature_data.dict["learning_vector_capacity"].keys())
        return param_dict

    def get_flat_parameter_set(self, param_set):
        param_dict = OrderedDict()
        for classe in param_set.keys():
            for feature in param_set[classe].keys():
                param_dict[feature] = param_set[classe][feature]
        return param_dict

    def get_flat_gradient_set(self, gradient_set, parameter_set):
        gradient_dict = OrderedDict()
        for feature in parameter_set.keys():
            gradient_dict[feature] = gradient_set[feature]
        return gradient_dict


    def set_Z(self):
        self.Z = np.random.normal(loc=0, scale=args.perturbed_optimizer[1], size=(int(args.perturbed_optimizer[0]), self.parameter_vector_size))

    def get_perturbed_w(self, x, iteration):
        x_pert = x + self.Z[iteration]
        x_pert_dict = self.set_parameter_set(x_pert)
        return x_pert_dict


    def get_data(self, instance_file):
        file_input = open(args.training_instance_directory + instance_file, "rb")
        instance_date = instance_file.split("instDate:")[-1].split("_")[0]
        print("Instance_Date: {} ".format(instance_date))
        training_instance = pickle.load(file_input)
        file_input.close()
        return training_instance

    def calculate_loss_for_instance(self, x, file):
        perturbed_counter = 0
        profit_online_list = []
        gradient_online_list = []
        training_instance = self.get_data(instance_file=file)
        predictor = Predictor(args, x, feature_data=feature_data)
        standardized_training_instance = predictor.feature_data.standardize(training_instance)
        graph_generator = GraphGenerator(args)
        combinatorial_optimizer = CombinatorialOptimizer(args)
        predictor.set_parameters(self.set_parameter_set(x))
        profit_online, profit_offline, gradient_online, gradient_offline = run_training(training_instance=standardized_training_instance,
                                                                                        predictor=predictor,
                                                                                        graph_generator=graph_generator,
                                                                                        combinatorial_optimizer=combinatorial_optimizer)
        profit_online_list.append(profit_online)
        gradient_online_list.append(gradient_online)
        while perturbed_counter < args.perturbed_optimizer[0]:
            w = self.get_perturbed_w(x, iteration=perturbed_counter)
            perturbed_counter += 1
            predictor.set_parameters(w)
            profit_online, _, gradient_online, _ = run_training(training_instance=standardized_training_instance, predictor=predictor, graph_generator=graph_generator, combinatorial_optimizer=combinatorial_optimizer)
            profit_online_list.append(profit_online)
            gradient_online_list.append(gradient_online)
        profit_online = np.mean(profit_online_list)
        gradient_online = pd.DataFrame(gradient_online_list)
        gradient_online = gradient_online.mean(axis=0)
        gradient = gradient_online.subtract(gradient_offline)
        loss = profit_online - profit_offline
        return [loss, gradient]


    def save_learning(self, evolution_dict, step):
        file_output = open(util.get_learning_file(args, step), "w")
        json.dump(evolution_dict, file_output)
        file_output.close()

    def objective(self, x):
        print("--------- Actual values: " + str(x) + " --------------------")

        relevant_instances = []
        for instance in os.listdir(args.training_instance_directory):
            if instance.find("TrainingInstance") == -1 or instance.find(args.objective) == -1:
                continue
            relevant_instances.append(instance)

        print("Number of processors: ", mp.cpu_count())
        with Pool(mp.cpu_count()-2) as pool:
            saver = pool.map(partial(self.calculate_loss_for_instance, x), relevant_instances)

        loss = sum([value[0] for value in saver])
        gradient = [value[1] for value in saver]
        gradient = pd.DataFrame(gradient).sum(axis=0)
        gradient = list(self.get_flat_gradient_set(gradient, self.get_flat_parameter_set(self.set_parameter_set(x))).values())

        print("=============================================================== LOSS: {}".format(loss))
        self.loss_evolution_inner.append(float(loss))
        self.parameter_evolution_inner.append(self.set_parameter_set(x))
        self.gradient_evolution_inner.append(self.set_parameter_set(gradient))

        # save the loss evolution
        evolution_dict = {"loss_evolution": self.loss_evolution_inner, "parameter_evolution": self.parameter_evolution_inner, "gradient_evolution": self.gradient_evolution_inner}
        self.save_learning(evolution_dict, "inner")
        return (float(loss), gradient)

    def callbackF(self, Xi):
        global Nfeval

        self.parameter_evolution_outer.append(self.set_parameter_set(Xi))

        # save the loss evolution
        evolution_dict = {"parameter_evolution": self.parameter_evolution_outer}

        self.save_learning(evolution_dict, "outer")

        elapsed = time.time() - self.start_time
        if elapsed > self.max_sec:
            raise Exception("Time limit reached after {} seconds".format(elapsed))
        else:
            print("ELAPSED TIME: {} sec".format(elapsed))


    def set_parameter_vector_size(self, parameter_dict):
        self.parameter_vector_size = len(self.get_parameter_set(parameter_dict))

    def optimize_problem(self, start_parameter_dict):
        self.set_parameter_vector_size(start_parameter_dict)
        self.set_Z()
        start_value = self.get_parameter_set(start_parameter_dict)
        x_point = scipy.optimize.minimize(self.objective, start_value, method="L-BFGS-B", jac=True, callback=self.callbackF, options={"maxiter": args.max_iter, "disp": True})
        return x_point



if __name__ == '__main__':
    # set random seed
    random.seed(2)
    np.random.seed(2)
    numpy_random = np.random.RandomState(seed=2)
    Nfeval = 1

    # read in parameters from command line / default values from modules/config.py file
    args = config.parser.parse_args()
    args.mode = "training"
    args = util.set_additional_args(args)
    util.create_directories(args)
    feature_data = Feature_data(args)
    start_parameter_dict = feature_data.get_starting_parameters()
    # calculate standardization values
    feature_data.calculate_standardization_values(args=args)


    # generate Optimizer object
    optim = Optimizer()
    # Learn the predictor over different training instances
    x_point = optim.optimize_problem(start_parameter_dict)

    # save output
    print("optimum at: {}".format(x_point.get("x")))
    print("minimum value = {}".format(x_point.get("fun")))
    print("result code = {}".format(x_point.get("success")))
    print("message = {}".format(x_point.get("message")))