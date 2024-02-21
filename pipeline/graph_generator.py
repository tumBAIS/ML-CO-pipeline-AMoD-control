import copy
from src.cplusplus_interface import Graph_Generator_Cplusplus
import pandas as pd
import numpy as np


class GraphGenerator:
    def __init__(self, args, look_up_grid=None):
        self.args = args
        # sink vertex
        self.sink = [{"type": "Sink", "location_lat": None, "location_lon": None}]
        # source vertex
        self.source = [{"type": "Source", "location_lat": None, "location_lon": None}]
        self.artificial_dummy_sink_index = None
        if look_up_grid is not None:
            self.cplusplus_interface = Graph_Generator_Cplusplus(args=self.args, look_up_grid=look_up_grid)

    def fill_cplusplus_interface(self, requestList, vehicleList, artRebVertexList):
        self.cplusplus_interface.set_values(requestList=requestList, vehicleList=vehicleList, artRebVertexList=artRebVertexList)

    def get_vertices_list(self, vehicleList, requestList, artRebVerticesList, artCapVerticesList):
        def set_vertex_idx(vertexList, start_idx, end_idx):
            vertexList = pd.DataFrame(vertexList)
            vertexList["vertex_idx"] = range(start_idx, end_idx)
            return vertexList.to_dict("records")

        idx_bins = {"vehicleVertices_idx": [1, 1+len(vehicleList)],
                "requestVertices_idx": [1 + len(vehicleList), 1 + len(vehicleList) + len(requestList)],
                "artRebVertices": [1 + len(vehicleList) + len(requestList), 1 + len(vehicleList) + len(requestList) + len(artRebVerticesList)],
                "artCapVertices": [1 + len(vehicleList) + len(requestList) + len(artRebVerticesList), 1 + len(vehicleList) + len(requestList) + len(artRebVerticesList) + len(artCapVerticesList)]}

        vehicleList = set_vertex_idx(vehicleList, start_idx=idx_bins["vehicleVertices_idx"][0], end_idx=idx_bins["vehicleVertices_idx"][1])
        requestList = set_vertex_idx(requestList, start_idx=idx_bins["requestVertices_idx"][0], end_idx=idx_bins["requestVertices_idx"][1])
        artRebVerticesList = set_vertex_idx(artRebVerticesList, start_idx=idx_bins["artRebVertices"][0], end_idx=idx_bins["artRebVertices"][1])
        artCapVerticesList = set_vertex_idx(artCapVerticesList, start_idx=idx_bins["artCapVertices"][0], end_idx=idx_bins["artCapVertices"][1])
        # combine all lists to an overall vertices list and set index of sink
        verticesList = self.source + vehicleList + requestList + artRebVerticesList + artCapVerticesList + self.sink
        self.sink_index = len(verticesList) - 1
        return verticesList, vehicleList, requestList, artRebVerticesList, artCapVerticesList, idx_bins

    def calculate_edges_source_vehicle(self):
        #print(" -- edges: Source -> Vehicles")
        edgeSourceVehicles = [tuple([0, vehicle]) for vehicle in range(1, self.args.num_vehicles + 1)]
        return edgeSourceVehicles

    def calculate_edges_vehicle_sink(self):
        #print(" -- edges: Source -> Vehicles")
        edgeVehiclesSink = [tuple([vehicle, self.sink_index]) for vehicle in range(1, self.args.num_vehicles + 1)]
        return edgeVehiclesSink

    def calculate_edges_vehicles_rides(self):
        #print(" -- edges: Vehicles -> Rides")
        edgeVehiclesRides, distancesVehiclesRides, durationsVehiclesRides = self.cplusplus_interface.calculateEdgesVehiclesRides()
        return edgeVehiclesRides, {"distance_km": distancesVehiclesRides, "duration_sec": durationsVehiclesRides}

    def calculate_edges_requests_requests(self):
        #print(" -- edges: Rides -> Rides")
        edgeRidesRides, distancesRidesRides, durationsRidesRides = self.cplusplus_interface.calculateEdgesRidesRides()
        return edgeRidesRides, {"distance_km": distancesRidesRides, "duration_sec": durationsRidesRides}

    def calculate_edges_rides_sink(self, requestList):
        #print(" -- edges: Rides -> Sink ")
        edgeRidesSinks = [tuple([request, self.sink_index]) for request in range(self.args.num_vehicles + 1, self.args.num_vehicles + 1 + len(requestList))]
        return edgeRidesSinks


    def calculate_edges_vehicles_artLoc(self):
        #print(" -- edges: Vehicles -> RebalancingLocations ")
        edgesVehiclesArtificialVertices, distancesVehiclesArtificialVertices, durationsVehiclesArtificialVertices = self.cplusplus_interface.caluclateEdgesVehiclesArtificialVertices()
        return edgesVehiclesArtificialVertices, {"distance_km": distancesVehiclesArtificialVertices, "duration_sec": durationsVehiclesArtificialVertices}


    def calculate_edges_rides_artLoc(self):
        #print(" -- edges: Rides -> RebalancingLocations ")
        edgesRidesArtificialVertices, distancesRidesArtificialVertices, durationsRidesArtificialVertices = self.cplusplus_interface.caluclateEdgesRidesArtificialVertices()
        return edgesRidesArtificialVertices, {"distance_km": distancesRidesArtificialVertices, "duration_sec": durationsRidesArtificialVertices}


    def calculate_edges_artLoc_sink(self, artLocVerticesList):
        # build edges between future optimal vehicle locations and sinks
        #print(" -- edges: Locations -> Sink ")
        edgeLocationsSinks = [tuple([location, self.sink_index]) for location in range(self.sink_index - len(artLocVerticesList), self.sink_index)]
        return edgeLocationsSinks

    def calculate_edges_artRebVertices_artCapVertices(self, artRebVerticesList, artCapVerticesList):
        #print(" -- edges RebalancingLocations -> Capacities ")
        if len(artCapVerticesList) == 0:
            return []
        #print(" -- edges Locations -> Capacities ", end="")
        edges = artCapVerticesList[["index_Rebalancing_Pickup", "vertex_idx"]].merge(artRebVerticesList[["index_Rebalancing_Pickup", "vertex_idx"]], left_on="index_Rebalancing_Pickup", right_on="index_Rebalancing_Pickup", how="left")
        return [tuple(edge) for edge in edges[["vertex_idx_y", "vertex_idx_x"]].values.tolist()]

    def reduce_weights_to_artificialVertices(self, edges, weights, clock, verticesList):
        # get all request that start in the future for edge RidesRides
        edgePandas = pd.DataFrame(edges, columns=["starting", "arriving"])
        edgePandas = edgePandas.merge(pd.DataFrame(verticesList), left_on="arriving", right_index=True, how="left")
        indicator_vector = np.where(edgePandas["tpep_pickup_datetime"] > clock.actual_system_time_period_end, self.args.reduce_weight_of_future_requests_factor, 1)
        return [a * b for a, b in zip(weights, list(indicator_vector))]

    def generate_graph(self, edges, weights):

        edges_graph = copy.deepcopy(edges)
        weights_graph = copy.deepcopy(weights)

        def enter_dummy_edges(edges, weights):
            # for each request vertex we add a dummy request vertex and add a dummy edge : request -> dummyRequest
            new_edges = []
            new_weights = []
            for weight, edge in zip(weights, edges):
                if edge[0] not in self.artificial_dummy_edge_mapping.keys():
                    self.artificial_dummy_edge_mapping[edge[0]] = self.artificial_dummy_sink_index
                    new_edges.append((edge[0], self.artificial_dummy_edge_mapping[edge[0]]))
                    new_weights.append(0)
                    self.artificial_dummy_sink_index += 1
                new_edges.append((self.artificial_dummy_edge_mapping[edge[0]], edge[1]))
                new_weights.append(weight)
            return new_edges, new_weights

        self.artificial_dummy_sink_index = self.sink_index
        if self.args.policy == "policy_CB":
            self.artificial_dummy_edge_mapping = {}
            edges_graph["requests_requests"], weights_graph["requests_requests"] = enter_dummy_edges(edges_graph["requests_requests"], weights_graph["requests_requests"])
            edges_graph["requests_artRebVertices"], weights_graph["requests_artRebVertices"] = enter_dummy_edges(edges_graph["requests_artRebVertices"], weights_graph["requests_artRebVertices"])
            edges_graph["requests_sink"], weights_graph["requests_sink"] = enter_dummy_edges(edges_graph["requests_sink"], weights_graph["requests_sink"])

            edges_graph["vehicles_sink"] = [(edge[0], self.artificial_dummy_sink_index) for edge in edges_graph["vehicles_sink"]]
            edges_graph["requests_sink"] = [(edge[0], self.artificial_dummy_sink_index) if edge[0] not in self.artificial_dummy_edge_mapping.keys() else (edge[0], edge[1]) for edge in edges_graph["requests_sink"]]
            edges_graph["artLocVertices_sink"] = [(edge[0], self.artificial_dummy_sink_index) for edge in edges_graph["artLocVertices_sink"]]

        return edges_graph, weights_graph

    def get_edges_weights_lists(self, edges, weights):
        # combine all edges and weights
        edgesList = edges["source_vehicles"] + edges["vehicles_requests"] + edges["requests_requests"] + edges["vehicles_artRebVertices"] + edges["requests_artRebVertices"] + \
                    edges["artRebVertices_artCapVertices"] + edges["vehicles_sink"] + edges["requests_sink"] + edges["artLocVertices_sink"]

        weightsList = weights["source_vehicles"] + weights["vehicles_requests"] + weights["requests_requests"] + weights["vehicles_artRebVertices"] + \
                  weights["requests_artRebVertices"] + weights["artRebVertices_artCapVertices"] + weights["vehicles_sink"] + weights["requests_sink"] + weights["artLocVertices_sink"]

        assert len(edgesList) == len(weightsList)
        return edgesList, weightsList

    def resolve_dummy_edges_mapping(self, edges, weights=None):
        # we reduce the list of edges to the original edge list without dummy edges
        # we can invert the artificial dummy edge mapping as it keeps for each request exactly one dummy request
        mapping_dummy_to_real = {v: k for k, v in self.artificial_dummy_edge_mapping.items()}
        real_edges = []
        real_weights = []
        for edge, weight in zip(edges, weights if weights is not None else [0] * len(edges)):
            # we keep edges which are not connected with requests
            if edge[1] == self.artificial_dummy_sink_index:
                edge = (edge[0], self.sink_index)
            if (edge[0] not in mapping_dummy_to_real.keys()) and (edge[1] not in mapping_dummy_to_real.keys()):
                real_edges.append(edge)
                real_weights.append(weight)
            # we reset dummy edges
            elif edge[0] in mapping_dummy_to_real.keys():
                real_edges.append((mapping_dummy_to_real[edge[0]], edge[1]))
                real_weights.append(weight)
        return real_edges, real_weights

    def get_full_information_solution_edges(self, edges, vehicleList, requestList, full_information_artRebVerticesList, artRebVerticesList):
        _, vehicleList, requestList, full_information_artRebVerticesList, _, idx_bins = self.get_vertices_list(vehicleList, requestList, full_information_artRebVerticesList, artCapVerticesList=[])
        edges = pd.DataFrame(edges, columns=["starting", "arriving"])
        edges_vehicles = edges.merge(pd.DataFrame(vehicleList)[["vertex_idx", "full_information_vertex_idx"]], left_on="starting", right_on="full_information_vertex_idx")
        edges_vehicles_requests = edges_vehicles.merge(pd.DataFrame(requestList)[["vertex_idx", "full_information_vertex_idx"]], left_on="arriving", right_on="full_information_vertex_idx")
        edges_vehicles_artRebVertices = edges_vehicles.merge(pd.DataFrame(full_information_artRebVerticesList)[["vertex_idx", "full_information_vertex_idx", "index_Rebalancing_Pickup"]], left_on="arriving", right_on="full_information_vertex_idx")
        edges_requests = edges.merge(pd.DataFrame(requestList)[["vertex_idx", "full_information_vertex_idx"]], left_on="starting", right_on="full_information_vertex_idx")
        edges_requests_requests = edges_requests.merge(pd.DataFrame(requestList)[["vertex_idx", "full_information_vertex_idx"]], left_on="arriving", right_on="full_information_vertex_idx")
        edges_requests_artRebVertices = edges_requests.merge(pd.DataFrame(full_information_artRebVerticesList)[["vertex_idx", "full_information_vertex_idx", "index_Rebalancing_Pickup"]], left_on="arriving", right_on="full_information_vertex_idx")
        if self.args.policy == "policy_CB":
            edges_vehicles_artRebVertices = edges_vehicles_artRebVertices[["vertex_idx_x", "index_Rebalancing_Pickup"]].rename(columns={"index_Rebalancing_Pickup": "vertex_idx_y"}).drop_duplicates()
            edges_requests_artRebVertices = edges_requests_artRebVertices[["vertex_idx_x", "index_Rebalancing_Pickup"]].rename(columns={"index_Rebalancing_Pickup": "vertex_idx_y"}).drop_duplicates()
            counter_artRebVertices_artCapVertices = pd.concat([edges_vehicles_artRebVertices[["vertex_idx_y"]], edges_requests_artRebVertices[["vertex_idx_y"]]]).groupby(["vertex_idx_y"], as_index=False).size()
            self.args.fix_capacity = max(counter_artRebVertices_artCapVertices["size"])
            edges_artRebVertices_artCapVertices = []
            for rebalancingVertex in counter_artRebVertices_artCapVertices.to_dict("records"):
                for counter in list(range(rebalancingVertex["size"])):
                    capacityNode_index = (len(artRebVerticesList) * counter) + rebalancingVertex["vertex_idx_y"]
                    edges_artRebVertices_artCapVertices.append((1 + len(vehicleList) + len(requestList) + rebalancingVertex["vertex_idx_y"],
                                                                1 + len(vehicleList) + len(requestList) + len(artRebVerticesList) + capacityNode_index))
            edges_vehicles_artRebVertices["vertex_idx_y"] = edges_vehicles_artRebVertices["vertex_idx_y"] + 1 + len(vehicleList) + len(requestList)
            edges_requests_artRebVertices["vertex_idx_y"] = edges_requests_artRebVertices["vertex_idx_y"] + 1 + len(vehicleList) + len(requestList)
        elif self.args.policy == "policy_SB":
            edges_artRebVertices_artCapVertices = []
        else:
            raise Exception("Wrong policy definded")

        return {"edgeVehiclesRequests_FI": [tuple(x) for x in edges_vehicles_requests[["vertex_idx_x", "vertex_idx_y"]].values.tolist()],
                "edgeRequestsRequests_FI": [tuple(x) for x in edges_requests_requests[["vertex_idx_x", "vertex_idx_y"]].values.tolist()],
                "edgeVehiclesArtRebVertices_FI": [tuple(x) for x in edges_vehicles_artRebVertices[["vertex_idx_x", "vertex_idx_y"]].values.tolist()],
                "edgeRequestsArtRebVertices_FI": [tuple(x) for x in edges_requests_artRebVertices[["vertex_idx_x", "vertex_idx_y"]].values.tolist()],
                "edgeArtRebVerticesArtCapVertices_FI": [tuple(x) for x in edges_artRebVertices_artCapVertices]}

    def get_maximum_capacity(self, full_information_solution_edges):
        maximum_capacity_counter = {}
        for edge_to_capacity_vertex in full_information_solution_edges["edgeLocationsCapacities_OfflineMap"]:
            if edge_to_capacity_vertex[1] not in maximum_capacity_counter.keys():
                maximum_capacity_counter[edge_to_capacity_vertex[1]] = 1
            else:
                maximum_capacity_counter[edge_to_capacity_vertex[1]] += 1
