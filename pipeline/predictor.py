from src import util
from collections import OrderedDict
from prep.feature_data import Feature_data
import pandas as pd


class Predictor():
    def __init__(self, args, optim_param, rebalancing_grid=None, feature_data=None, clock=None):
        self.args = args
        self.parameter_dict = optim_param
        self.rebalancing_grid = rebalancing_grid
        if feature_data is None:
            self.feature_data = Feature_data(args=self.args, rebalancing_grid=rebalancing_grid, clock=clock)
        else:
            self.feature_data = feature_data


    def delete_feature_columns_vehicle(self, vehicleList):
        if self.args.policy in ["policy_SB", "policy_CB"]:
            vehicleList = vehicleList.drop(columns=self.parameter_dict["learning_vector_origin"].keys())
        return vehicleList.to_dict("records")

    def calculate_profit(self, features_edges):
        # calculate profit: - reward + (cost for driving to request + cost for fulfilling request)
        return list(- features_edges["objective_value"] + 100 * (self.args.costs_per_km * features_edges["distance_km"] + self.args.costs_per_km * features_edges["ride_distance"]))

    def set_parameters(self, w):
        self.parameter_dict = w

    def refilter_parameter_dict(self, list_feature_sets):
        relevant_parameters = OrderedDict()
        for classe in list_feature_sets:
            for feature in self.parameter_dict[classe].keys():
                relevant_parameters[feature] = self.parameter_dict[classe][feature]
        return relevant_parameters


    def calculate_weights_source_vehicles(self, edges_source_vehicles):
        return [0 for edge in edges_source_vehicles]


    def calculate_weights_vehicles_sink(self, edges_vehicles_sink):
        return [0 for edge in edges_vehicles_sink]


    def calculate_weights_requests_sink(self, edges_requests_sink):
        return [0 for edge in edges_requests_sink]


    def calculate_weights_artLocVertices_sink(self, artificialVertices_sink):
        return [0 for edge in artificialVertices_sink]


    def multiply(self, pandas, feature_sets):
        relevant_dict = self.refilter_parameter_dict(feature_sets)
        relevant_pandas = pandas[relevant_dict.keys()]
        return [-1 * weight for weight in list((relevant_pandas * relevant_dict).sum(axis=1))]


    def calculate_weights_vehicles_requests(self, features_edges_vehicles_requests):
        if self.args.mode in ["training", "evaluation"] and self.args.policy in ['policy_SB', 'policy_CB']:
            return self.multiply(features_edges_vehicles_requests, ["learning_vector_origin", "learning_vector_destination", "learning_vector_future", "learning_vector_relation"])
        else:
            return self.calculate_profit(features_edges_vehicles_requests)

    def calculate_weights_requests_requests(self, features_edges_requests_requests):
        if len(features_edges_requests_requests) == 0:
            return []
        if self.args.mode in ['policy_SB', 'policy_CB'] and self.args.mode in ["training", "evaluation"]:
            return self.multiply(features_edges_requests_requests, ["learning_vector_origin", "learning_vector_destination", "learning_vector_future", "learning_vector_relation"])
        else:
            return self.calculate_profit(features_edges_requests_requests)

    def calculate_weights_requests_artRebVertices(self, features_edges_requests_artificialVertices):
        if len(features_edges_requests_artificialVertices) == 0:
            return []
        if self.args.policy == "policy_SB" and self.args.mode in ["training", "evaluation"]:
            return self.multiply(features_edges_requests_artificialVertices, ["learning_vector_origin", "learning_vector_destination", "learning_vector_future", "learning_vector_relation"])
        elif self.args.policy == "policy_CB" and self.args.mode in ["training", "evaluation"]:
            return self.multiply(features_edges_requests_artificialVertices, ["learning_vector_location", "learning_vector_location_relation"])
        else:
            return self.calculate_profit(features_edges_requests_artificialVertices)


    def calculate_weights_vehicles_artRebVertices(self, features_edges_vehicles_artificialVertices):
        if len(features_edges_vehicles_artificialVertices) == 0:
            return []
        if self.args.policy == "policy_SB" and self.args.mode in ["training", "evaluation"]:
            return self.multiply(features_edges_vehicles_artificialVertices, ["learning_vector_origin", "learning_vector_destination", "learning_vector_future", "learning_vector_relation"])
        elif self.args.policy == "policy_CB" and self.args.mode in ["training", "evaluation"]:
            return self.multiply(features_edges_vehicles_artificialVertices, ["learning_vector_location", "learning_vector_location_relation"])
        else:
            return self.calculate_profit(features_edges_vehicles_artificialVertices)

    def calculate_weights_artRebVertices_artCapVertices(self, features_edges_artificialRebalancingVertices_artificialCapacityVertices):
        if len(features_edges_artificialRebalancingVertices_artificialCapacityVertices) == 0:
            return []
        if self.args.policy == 'policy_CB' and self.args.mode in ["training", "evaluation"]:
            return self.multiply(features_edges_artificialRebalancingVertices_artificialCapacityVertices, ["learning_vector_capacity"])
        else:
            return [0] * len(features_edges_artificialRebalancingVertices_artificialCapacityVertices)