from learning_problem.util_learning import get_training_instances, get_data, calculate_objective_value, get_edges, save_evolution_learning, decode_solution, get_learnable_edges
import numpy as np
from multiprocessing import Pool
import multiprocessing as mp
from functools import partial
from pipeline.graph_generator import GraphGenerator
from pipeline.combinatorial_optimization import CombinatorialOptimizer
from run_pipeline import get_solution


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


class OptimizerSGD():
    def __init__(self, args, Predictor, feature_data):
        self.args = args
        self.predictor = Predictor(args, feature_data=feature_data, learning_rate=args.learning_rate)
        self.overall_loss = []
        self.overall_accuracy = []

    def objective(self, training_instance_name):
        graph_generator = GraphGenerator(self.args)
        training_data = get_data(self.args, instance_file=training_instance_name)
        training_data_standardized = self.predictor.feature_data.standardize(training_data)
        edge_features, node_features = self.predictor.retrieve_features(training_data_standardized, training_data["node_attributes"])
        edges = get_edges(self.args, training_data)
        weights = self.predictor.predict_weights(edge_features, node_features, edges)
        sink_index = training_data["len_verticesList"] - 1
        learnable_edges, _ = get_learnable_edges(edges)
        edgesList, weightsList = graph_generator.get_edges_weights_lists(edges, weights)

        # get offline solution
        edgesList_full_information_solution = [edge for edge_group in training_data["full_information_solution_edges"].values() for edge in edge_group]
        y_full_information_solution = np.array(decode_solution(edgesList_full_information_solution, learnable_edges))
        objective_value_offline = calculate_objective_value(edgesList=edgesList, weightsList=weightsList, solution_edges=edgesList_full_information_solution)
        len_vehicleList = training_data["len_vehicleList"]

        # get perturbed online solution
        with Pool(mp.cpu_count()-2) as pool:
            saver = pool.map(partial(calculate_loss_for_instance, (self.args, edges, weights, len_vehicleList, sink_index, learnable_edges)), list(range(int(self.args.perturbed_optimizer[0]))))

        objective_value_saver = [value[0] for value in saver]
        y_solution_saver = [value[1] for value in saver]
        objective_value_mean = np.mean(objective_value_saver)
        y_solution_mean = np.mean(y_solution_saver, axis=0)
        loss = float(objective_value_mean - objective_value_offline)
        accuracy = np.mean(np.abs(y_full_information_solution - y_solution_mean))
        self.predictor.grad_optimize(y_full_information_solution, y_solution_mean, self.predictor.merge_features(edge_features), node_features, edges)
        return loss, accuracy

    def optimize_problem(self):
        training_instances = get_training_instances(self.args)
        for training_iter in range(self.args.max_iter):
            loss_value_instance = []
            accuracy_instance = []
            for training_instance_name in training_instances:
                loss, accuracy = self.objective(training_instance_name)
                loss_value_instance.append(loss)
                accuracy_instance.append(accuracy)
            self.overall_loss.append(np.mean(loss_value_instance))
            self.overall_accuracy.append(np.mean(accuracy_instance))
            self.predictor.save_model(self.args, training_iter)
            evolution_dict = {"loss_evolution": self.overall_loss, "accuracy_evolution": self.overall_accuracy}
            save_evolution_learning(self.args, evolution_dict)
            print("Epoch: {} ----> Loss: {} ---- Accuracy_n: {}".format(training_iter, np.mean(loss_value_instance), np.mean(accuracy_instance)))
