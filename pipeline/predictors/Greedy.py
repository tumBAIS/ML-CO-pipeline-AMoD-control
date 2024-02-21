from pipeline.predictor import Predictor


class GreedyModel(Predictor):
    def __init__(self, args, rebalancing_grid=None, feature_data=None, clock=None, learning_rate=None):
        super().__init__(args, rebalancing_grid=rebalancing_grid, feature_data=feature_data, clock=clock)

    def retrieve_features(self, edge_attributes, node_attributes):
        features_edges = {"vehicles_requests": edge_attributes["features_edges_vehicles_requests"],
                          "requests_requests": edge_attributes["features_edges_requests_requests"],
                          "vehicles_artRebVertices": edge_attributes["features_edges_vehicles_artRebVertices"],
                          "requests_artRebVertices": edge_attributes["features_edges_requests_artRebVertices"],
                          "artRebVertices_artCapVertices": edge_attributes["features_edges_artRebVertices_artCapVertices"]}
        features_nodes = {"vehicles": node_attributes["features_nodes_vehicles"],
                          "requests": node_attributes["features_nodes_requests"],
                          "artRebVertices": node_attributes["features_nodes_artRebVertices"],
                          "artCapVertices": node_attributes["features_nodes_artCapVertices"]}
        return features_edges, features_nodes

    def predict_weights(self, edge_features, node_attributes, edges, edge_attributes=None, perturbation=None):
        weights_vehicles_requests = self.calculate_profit(edge_attributes["features_edges_vehicles_requests"])
        weights_requests_requests = self.calculate_profit(edge_attributes["features_edges_requests_requests"])
        weights_vehicles_artRebVertices = self.calculate_profit(edge_attributes["features_edges_vehicles_artRebVertices"])
        weights_requests_artRebVertices = self.calculate_profit(edge_attributes["features_edges_requests_artRebVertices"])
        weights_artRebVertices_artCapVertices = [0] * len(edge_attributes["features_edges_artRebVertices_artCapVertices"])
        return {"source_vehicles": [0 for edge in edges["source_vehicles"]],
                "vehicles_requests": weights_vehicles_requests,
                "requests_requests": weights_requests_requests,
                "vehicles_artRebVertices": weights_vehicles_artRebVertices,
                "requests_artRebVertices": weights_requests_artRebVertices,
                "artRebVertices_artCapVertices": weights_artRebVertices_artCapVertices,
                "vehicles_sink": [0 for edge in edges["vehicles_sink"]],
                "requests_sink": [0 for edge in edges["requests_sink"]],
                "artLocVertices_sink": [0 for edge in edges["artLocVertices_sink"]]}

    def calculate_profit(self, features_edges):
        if len(features_edges) == 0:
            return []
        # calculate profit: - reward + (cost for driving to request + cost for fulfilling request)
        return list(- features_edges["objective_value"] + 100 * (self.args.costs_per_km * features_edges["distance_km"] + self.args.costs_per_km * features_edges["ride_distance"]))
