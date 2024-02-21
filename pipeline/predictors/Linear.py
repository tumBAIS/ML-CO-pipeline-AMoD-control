import pandas as pd
import numpy as np
from collections import OrderedDict
from pipeline.predictor import Predictor
import json
from src import util


class LinearModel(Predictor):
    def __init__(self, args, rebalancing_grid=None, feature_data=None, clock=None):
        super().__init__(args, rebalancing_grid=rebalancing_grid, feature_data=feature_data, clock=clock)
        self.parameters = None

    def get_parameters_as_list(self):
        return self.get_parameter_list_from_dict(self.parameters)

    def get_parameter_list_from_series(self, param_series):
        parameter_list = []
        for classes in self.feature_data.dict.keys():
            for key in list(self.feature_data.dict[classes].keys()):
                parameter_list.append(param_series[key])
        return parameter_list

    def get_parameter_list_from_dict(self, param_dict):
        parameter_list = []
        for classes in self.feature_data.dict.keys():
            for key in list(self.feature_data.dict[classes].keys()):
                parameter_list.append(param_dict[classes][key])
        return parameter_list

    def get_parameter_dict_from_list(self, param_list):
        if not isinstance(param_list, list):
            param_list = param_list.tolist()
        param_dict = {}
        counter = 0
        for classes in self.feature_data.dict.keys():
            param_dict[classes] = OrderedDict()
            for key in self.feature_data.dict[classes].keys():
                param_dict[classes][key] = param_list[counter]
                counter = counter + 1

        assert list(param_dict["learning_vector_origin"].keys()) == list(self.feature_data.dict["learning_vector_origin"].keys())
        assert list(param_dict["learning_vector_destination"].keys()) == list(self.feature_data.dict["learning_vector_destination"].keys())
        assert list(param_dict["learning_vector_future"].keys()) == list(self.feature_data.dict["learning_vector_future"].keys())
        assert list(param_dict["learning_vector_relation"].keys()) == list(self.feature_data.dict["learning_vector_relation"].keys())
        if self.args.policy == "policy_CB":
            assert list(param_dict["learning_vector_location"].keys()) == list(self.feature_data.dict["learning_vector_location"].keys())
            assert list(param_dict["learning_vector_location_relation"].keys()) == list(self.feature_data.dict["learning_vector_location_relation"].keys())
            assert list(param_dict["learning_vector_capacity"].keys()) == list(self.feature_data.dict["learning_vector_capacity"].keys())
        return param_dict

    def load_model(self, parameters=None):
        if parameters is None:
            parameters_file = open(util.get_learning_file(self.args, "outer"), "r")
            parameters_file = json.load(parameters_file)
            parameters = parameters_file.get("parameter_evolution")[self.args.read_in_iterations if self.args.read_in_iterations is not None else -1]
            if self.args.read_in_iterations == -1:
                self.args.read_in_iterations = len(parameters_file.get("parameter_evolution")) - 1
        if isinstance(parameters, list) or isinstance(parameters, np.ndarray):
            self.parameters = self.get_parameter_dict_from_list(parameters)
        else:
            self.parameters = parameters

    def forward_pass(self, data, perturbation=None):
        if perturbation is None:
            perturbation = np.zeros(len(self.get_parameter_list_from_dict(self.parameters)))
        if len(data) == 0:
            return []
        else:
            return (-1 * np.multiply(data, (np.array(self.get_parameter_list_from_dict(self.parameters)) + perturbation)).sum(axis=1)).tolist()

    def get_gradient(self, edges, edges_solution, features):
        gradient_features_vehicles_requests = features["vehicles_requests"][[edge in edges_solution for edge in edges["vehicles_requests"]]]
        gradient_features_requests_requests = features["requests_requests"][[edge in edges_solution for edge in edges["requests_requests"]]]
        gradient_features_requests_artRebVertices = features["requests_artRebVertices"][[edge in edges_solution for edge in edges["requests_artRebVertices"]]]
        gradient_features_vehicles_artRebVertices = features["vehicles_artRebVertices"][[edge in edges_solution for edge in edges["vehicles_artRebVertices"]]]
        gradient_features_artRebVertices_artCapVertices = features["artRebVertices_artCapVertices"][[edge in edges_solution for edge in edges["artRebVertices_artCapVertices"]]]

        gradient_features = pd.concat([gradient_features_vehicles_requests, gradient_features_requests_requests, gradient_features_requests_artRebVertices, gradient_features_vehicles_artRebVertices,
                                       gradient_features_artRebVertices_artCapVertices])
        return gradient_features.sum(axis=0)
