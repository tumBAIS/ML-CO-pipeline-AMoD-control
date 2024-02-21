import time
import scipy
from learning_problem.util_learning import get_training_instances, get_data, calculate_objective_value, get_edges, save_evolution_learning
from multiprocessing import Pool
import multiprocessing as mp
from functools import partial
import pandas as pd
from pipeline.graph_generator import GraphGenerator
from pipeline.combinatorial_optimization import CombinatorialOptimizer
import numpy as np
from run_pipeline import get_solution


class OptimizerBFGS():
    def __init__(self, args, Predictor, feature_data):
        self.args = args
        self.predictor = Predictor(args, feature_data=feature_data)
        self.feature_data = feature_data
        self.loss_evolution_inner = []
        self.gradient_evolution_inner = []
        self.edges_to_request_count = []
        self.parameter_evolution_inner = []
        self.parameter_evolution_outer = []
        self.max_sec = args.stop_max_time * 60
        self.start_time = time.time()

    def set_Z(self, vector_size):
        self.Z = np.random.normal(loc=0, scale=self.args.perturbed_optimizer[1], size=(int(self.args.perturbed_optimizer[0]), vector_size))

    def calculate_loss_for_instance(self, file):
        profit_online_list = []
        gradient_online_list = []
        training_data = get_data(self.args, instance_file=file)
        training_data_standardized = self.predictor.feature_data.standardize(training_data)
        graph_generator = GraphGenerator(self.args)
        graph_generator.sink_index = training_data["len_verticesList"] - 1
        combinatorial_optimizer = CombinatorialOptimizer(self.args)
        edges = get_edges(self.args, training_data)
        edge_features, node_features = self.predictor.retrieve_features(training_data_standardized, training_data["node_attributes"])
        weights = self.predictor.predict_weights(edge_features, node_features, edges)
        edgesList, weightsList = graph_generator.get_edges_weights_lists(edges, weights)

        # get offline solution
        edgesList_full_information_solution = [edge for edge_group in training_data["full_information_solution_edges"].values() for edge in edge_group]
        objective_value_offline = calculate_objective_value(edgesList=edgesList, weightsList=weightsList, solution_edges=edgesList_full_information_solution)
        gradient_offline = self.predictor.get_gradient(edges, edgesList_full_information_solution, edge_features)

        # get online solution 0 (to be consistent with old learning)
        edgesList_solution, _ = get_solution(self.args, edges, weights, training_data["len_vehicleList"], graph_generator, combinatorial_optimizer)
        objective_value_online = calculate_objective_value(edgesList=edgesList, weightsList=weightsList, solution_edges=edgesList_solution)
        gradient_online = self.predictor.get_gradient(edges, edgesList_solution, edge_features)
        profit_online_list.append(objective_value_online)
        gradient_online_list.append(gradient_online)

        # get perturbed online solution
        num_edges_to_requests_in_solutions = []
        for perturbed_counter in list(range(int(self.args.perturbed_optimizer[0]))):
            weights_perturbed = self.predictor.predict_weights(edge_features, node_features, edges, perturbation=self.Z[perturbed_counter])
            edgesList, weightsList_perturbed = graph_generator.get_edges_weights_lists(edges, weights_perturbed)
            edgesList_solution, _ = get_solution(self.args, edges, weights_perturbed, training_data["len_vehicleList"], graph_generator, combinatorial_optimizer)
            objective_value_online = calculate_objective_value(edgesList=edgesList, weightsList=weightsList_perturbed, solution_edges=edgesList_solution)
            gradient_online = self.predictor.get_gradient(edges, edgesList_solution, edge_features)
            profit_online_list.append(objective_value_online)
            gradient_online_list.append(gradient_online)
            num_edges_to_requests_in_solutions.append(sum([edge in edgesList_solution for edge in edges["vehicles_requests"]]) + sum([edge in edgesList_solution for edge in edges["requests_requests"]]))
        objective_value_online_mean = np.mean(profit_online_list)
        print(f"In iteration: {np.mean(num_edges_to_requests_in_solutions)}")
        gradient_online_mean = pd.DataFrame(gradient_online_list).mean(axis=0)
        gradient = gradient_online_mean.subtract(gradient_offline)
        loss = objective_value_online_mean - objective_value_offline
        return [loss, gradient, np.mean(num_edges_to_requests_in_solutions)]

    def objective(self, x):
        relevant_instances = get_training_instances(self.args)
        self.predictor.load_model(x)

        with Pool(mp.cpu_count()-2) as pool:
            saver = pool.map(partial(self.calculate_loss_for_instance), relevant_instances)

        loss = sum([value[0] for value in saver])
        gradient = [value[1] for value in saver]
        mean_edges_to_requests_in_solutions = np.mean([value[2] for value in saver])
        gradient = pd.DataFrame(gradient).sum(axis=0)
        gradient = self.predictor.get_parameter_list_from_series(gradient)

        print("=============================================================== LOSS: {}".format(loss))
        self.loss_evolution_inner.append(float(loss))
        self.parameter_evolution_inner.append(self.predictor.get_parameter_dict_from_list(x))
        self.gradient_evolution_inner.append(self.predictor.get_parameter_dict_from_list(gradient))
        self.edges_to_request_count.append(mean_edges_to_requests_in_solutions)

        # save the loss evolution
        evolution_dict = {"loss_evolution": self.loss_evolution_inner, "parameter_evolution": self.parameter_evolution_inner, "gradient_evolution": self.gradient_evolution_inner,
                          "edges_to_request_evolution": self.edges_to_request_count}
        save_evolution_learning(self.args, evolution_dict, "inner")
        return (float(loss), gradient)

    def callbackF(self, Xi):
        global Nfeval

        self.parameter_evolution_outer.append(self.predictor.get_parameter_dict_from_list(Xi))

        # save the loss evolution
        evolution_dict = {"parameter_evolution": self.parameter_evolution_outer}

        save_evolution_learning(self.args, evolution_dict, "outer")

        elapsed = time.time() - self.start_time
        if elapsed > self.max_sec:
            raise Exception("Time limit reached after {} seconds".format(elapsed))
        else:
            print("ELAPSED TIME: {} sec".format(elapsed))

    def optimize_problem(self):
        self.predictor.load_model(self.feature_data.get_starting_parameters())

        self.set_Z(vector_size=len(self.predictor.get_parameters_as_list()))
        start_value = self.predictor.get_parameters_as_list()
        x_point = scipy.optimize.minimize(self.objective, start_value, method="L-BFGS-B", jac=True, callback=self.callbackF, options={"maxiter": self.args.max_iter, "disp": True})

        # save output
        print("optimum at: {}".format(x_point.get("x")))
        print("minimum value = {}".format(x_point.get("fun")))
        print("result code = {}".format(x_point.get("success")))
        print("message = {}".format(x_point.get("message")))
