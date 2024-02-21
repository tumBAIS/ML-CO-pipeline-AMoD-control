from collections import OrderedDict
from prep.feature_data import Feature_data
import pandas as pd
import numpy as np


class Predictor:
    def __init__(self, args, rebalancing_grid=None, feature_data=None, clock=None):
        self.args = args
        self.rebalancing_grid = rebalancing_grid
        if feature_data is None:
            self.feature_data = Feature_data(args=self.args, rebalancing_grid=rebalancing_grid, clock=clock)
        else:
            self.feature_data = feature_data

    def load_model(self, parameters=None):
        ...

    def forward_pass(self, features, perturbation=None):
        ...

    def retrieve_features(self, edge_attributes, node_attributes):
        # we predict the weights on edges to artificial requests
        features_edges = {"vehicles_requests": self.get_features_weights_vehicles_requests(edge_attributes["features_edges_vehicles_requests"]),
                "requests_requests": self.get_features_weights_requests_requests(edge_attributes["features_edges_requests_requests"]),
                "vehicles_artRebVertices": self.get_features_weights_vehicles_artRebVertices(edge_attributes["features_edges_vehicles_artRebVertices"]),
                "requests_artRebVertices": self.get_features_weights_requests_artRebVertices(edge_attributes["features_edges_requests_artRebVertices"]),
                "artRebVertices_artCapVertices": self.get_features_weights_artRebVertices_artCapVertices(edge_attributes["features_edges_artRebVertices_artCapVertices"])}
        columns = list(set(list(node_attributes["features_nodes_vehicles"].columns) + list(node_attributes["features_nodes_requests"].columns) +
                           list(node_attributes["features_nodes_artRebVertices"].columns) + list(node_attributes["features_nodes_artCapVertices"])))
        features_nodes = {"vehicles": self.fill_features(node_attributes["features_nodes_vehicles"], columns),
                          "requests": self.fill_features(node_attributes["features_nodes_requests"], columns),
                          "artRebVertices": self.fill_features(node_attributes["features_nodes_artRebVertices"], columns),
                          "artCapVertices": self.fill_features(node_attributes["features_nodes_artCapVertices"], columns)}
        return features_edges, features_nodes

    def fill_features(self, data, columns):
        feature_data = pd.DataFrame(0, index=np.arange(len(data)), columns=columns)
        feature_data[data.columns] = data[data.columns]
        return feature_data

    def predict_weights(self, edge_features, node_attributes, edges, edge_attributes=None, perturbation=None):
        weights_vehicles_requests = self.forward_pass(edge_features["vehicles_requests"].values, perturbation)
        weights_requests_requests = self.forward_pass(edge_features["requests_requests"].values, perturbation)
        weights_vehicles_artRebVertices = self.forward_pass(edge_features["vehicles_artRebVertices"].values, perturbation)
        weights_requests_artRebVertices = self.forward_pass(edge_features["requests_artRebVertices"].values, perturbation)
        weights_artRebVertices_artCapVertices = self.forward_pass(edge_features["artRebVertices_artCapVertices"].values, perturbation)
        return {"source_vehicles": [0 for edge in edges["source_vehicles"]],
                "vehicles_requests": weights_vehicles_requests,
                "requests_requests": weights_requests_requests,
                "vehicles_artRebVertices": weights_vehicles_artRebVertices,
                "requests_artRebVertices": weights_requests_artRebVertices,
                "artRebVertices_artCapVertices": weights_artRebVertices_artCapVertices,
                "vehicles_sink": [0 for edge in edges["vehicles_sink"]],
                "requests_sink": [0 for edge in edges["requests_sink"]],
                "artLocVertices_sink": [0 for edge in edges["artLocVertices_sink"]]}

    def delete_feature_columns_vehicle(self, vehicleList):
        if self.args.policy in ["policy_SB", "policy_CB"]:
            vehicleList = vehicleList.drop(columns=self.feature_data.dict["learning_vector_origin"].keys())
        return vehicleList.to_dict("records")

    def get_features(self, without_relation=False):
        features = []
        for classe in list(self.feature_data.dict.keys()):
            if classe in ["learning_vector_relation", "learning_vector_location_relation"] and without_relation:
                continue
            for feature in self.feature_data.dict[classe]:
                features.append(feature)
        return features

    def get_feature_data(self, data, feature_sets):
        relevant_features = self.refilter_feature_data(feature_sets)
        feature_data = pd.DataFrame(0, index=np.arange(len(data)), columns=self.get_features())
        feature_data[relevant_features] = data[relevant_features]
        return feature_data

    def refilter_feature_data(self, list_feature_sets=None):
        relevant_features = []
        for classe in list_feature_sets if list_feature_sets is not None else list(self.feature_data.dict.keys()):
            for feature in self.feature_data.dict[classe]:
                relevant_features.append(feature)
        return relevant_features

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

    def get_features_weights_vehicles_requests(self, features_edges_vehicles_requests):
        return self.get_feature_data(features_edges_vehicles_requests, ["learning_vector_origin", "learning_vector_destination", "learning_vector_future", "learning_vector_relation"])

    def get_features_weights_requests_requests(self, features_edges_requests_requests):
        if len(features_edges_requests_requests) == 0:
            return pd.DataFrame()
        else:
            return self.get_feature_data(features_edges_requests_requests, ["learning_vector_origin", "learning_vector_destination", "learning_vector_future", "learning_vector_relation"])

    def get_features_weights_requests_artRebVertices(self, features_edges_requests_artificialVertices):
        if len(features_edges_requests_artificialVertices) == 0:
            return pd.DataFrame()
        if self.args.policy == "policy_SB" and self.args.mode in ["training", "evaluation", "sanity_check"]:
            return self.get_feature_data(features_edges_requests_artificialVertices, ["learning_vector_origin", "learning_vector_destination", "learning_vector_future", "learning_vector_relation"])
        if self.args.policy == "policy_CB" and self.args.mode in ["training", "evaluation", "sanity_check"]:
            return self.get_feature_data(features_edges_requests_artificialVertices, ["learning_vector_location", "learning_vector_location_relation"])

    def get_features_weights_vehicles_artRebVertices(self, features_edges_vehicles_artificialVertices):
        if len(features_edges_vehicles_artificialVertices) == 0:
            return pd.DataFrame()
        if self.args.policy == "policy_SB" and self.args.mode in ["training", "evaluation", "sanity_check"]:
            return self.get_feature_data(features_edges_vehicles_artificialVertices, ["learning_vector_origin", "learning_vector_destination", "learning_vector_future", "learning_vector_relation"])
        elif self.args.policy == "policy_CB" and self.args.mode in ["training", "evaluation", "sanity_check"]:
            return self.get_feature_data(features_edges_vehicles_artificialVertices, ["learning_vector_location", "learning_vector_location_relation"])

    def get_features_weights_artRebVertices_artCapVertices(self, features_edges_artificialRebalancingVertices_artificialCapacityVertices):
        if len(features_edges_artificialRebalancingVertices_artificialCapacityVertices) == 0:
            return pd.DataFrame()
        if self.args.policy == 'policy_CB' and self.args.mode in ["training", "evaluation", "sanity_check"]:
            return self.get_feature_data(features_edges_artificialRebalancingVertices_artificialCapacityVertices, ["learning_vector_capacity"])
