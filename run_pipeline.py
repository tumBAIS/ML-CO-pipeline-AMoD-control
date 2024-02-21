from src import util
from src.environment import Environment
from pipeline.graph_generator import GraphGenerator
from pipeline.combinatorial_optimization import CombinatorialOptimizer
from pipeline.decoder import Decoder
import pandas as pd
import time
from pipeline.predictors.Linear import LinearModel
from pipeline.predictors.NeuralNetwork import NNModel
from pipeline.predictors.Greedy import GreedyModel


def get_solution(args, edges, weights, len_vehicleList, graph_generator, combinatorial_optimizer):

    # add dummy edges
    edges_graph, weights_graph = graph_generator.generate_graph(edges, weights)
    edgesList_graph, weightsList_graph = graph_generator.get_edges_weights_lists(edges_graph, weights_graph)

    # run_k_disjoint shortest path
    k, edges, time_needed_k_disjoint = combinatorial_optimizer.solve_optimization_problem(edges=edgesList_graph, weights=weightsList_graph, num_vertices=graph_generator.artificial_dummy_sink_index + 1, num_k=len_vehicleList)

    # delete dummy edges
    if args.policy == "policy_CB":
        edges, _ = graph_generator.resolve_dummy_edges_mapping(edges)

    return edges, time_needed_k_disjoint


def get_predictor(args, rebalancing_grid, clock):
    predictors = {"Linear": LinearModel, "NN": NNModel, "Greedy": GreedyModel}
    predictor = predictors[args.model](args, rebalancing_grid=rebalancing_grid, clock=clock)
    predictor.load_model()
    return predictor


class Pipeline:
    def __init__(self, args, random_state):
        args = util.check_args_compatibility(args)  # check if arguments are compatible with each other
        args = util.define_costs_per_km(args)
        util.create_directories(args)  # create missing directories
        self.args = args
        self.evaluationList = []  # tracks evaluation of performance
        self.continue_running = True  # as long as true the pipeline runs
        self.environment = Environment(args, random_state)  # environment representing Manhattan
        self.graph_generator = GraphGenerator(args, self.environment.look_up_grid)  # ... generates the dispatching graph
        self.predictor = get_predictor(args, rebalancing_grid=self.environment.rebalancing_grid, clock=self.environment.clock)  # ... predicts the weights on the edges
        self.combinatorial_optimizer = CombinatorialOptimizer(args)  # solves the combinatorial optimization problem, namely k-dSPP
        self.decoder = Decoder(args)  # ... decodes the combinatorial solution to the action space

    def fill_evaluation(self, requests_in_solution, driving_distance, time_needed_k_disjoint, verticesList):
        objective_value = sum(request['objective_value'] / 100 for request in requests_in_solution) - (driving_distance * self.args.costs_per_km)

        requestList = [request for request in verticesList if request["type"] == "Request"]

        evaluationDict = {
            "Actual date new": self.environment.clock.actual_system_time_period_end,
            "Amount of Customers": len(requestList),
            "Satisfied Customers": len(requests_in_solution),
            "Revenue": sum(request['total_amount'] for request in requests_in_solution),
            "Objective_value": objective_value,
            "Profit": sum(request['total_amount'] for request in requests_in_solution) - (driving_distance * self.args.costs_per_km),
            "Total_possible_Profit": sum(request['total_amount'] for request in requestList),
            "Total_distance_travelled": driving_distance,
            "Avg_distance_trav": driving_distance / self.args.num_vehicles,
            "Time_kDisjoint": round(time_needed_k_disjoint, 3),
        }
        print(f"New / satisfied: {len(requestList)} / {len(requests_in_solution)} in {round(self.time_end_pipeline - self.time_start_pipeline, 2)} seconds")
        self.evaluationList.append(evaluationDict)

    def run(self):
        self.time_start_pipeline = time.time()

        # the rolling horizon moves to next system time period
        self.environment.clock.tick(args=self.args)

        # we check if the end of the problem horizon
        if self.environment.clock.actual_system_time_period_end >= self.environment.clock.end_date:
            self.continue_running = False

        print("-------------------------------Period: {} - {}----------------------------------------".format(self.environment.clock.actual_system_time_period_start, self.environment.clock.actual_system_time_period_end))

        # we assign to each vehicle location an index of the look-up cell and the rebalancing cell
        vehicleList = self.environment.look_up_grid.calculateGridIndex(liste=self.environment.vehicleList, title_entry="index_Look_Up_Table_Dropoff", title_longitude="location_lon", title_latitude="location_lat")
        vehicleList = self.environment.rebalancing_grid.calculateGridIndex(liste=vehicleList, title_entry="index_Rebalancing_Dropoff", title_longitude="location_lon", title_latitude="location_lat")

        # new ride requests enter the system
        requestList = self.environment.simulate_ride_requests().to_dict("records")

        # we assign to each request pickup and dropoff location an index of the look-up cell and the rebalancing cell
        requestList = self.environment.look_up_grid.calculateGridIndex(liste=requestList, title_entry="index_Look_Up_Table_Pickup", title_longitude="location_lon", title_latitude="location_lat")
        requestList = self.environment.look_up_grid.calculateGridIndex(liste=requestList, title_entry="index_Look_Up_Table_Dropoff", title_longitude="location_lon_dropoff", title_latitude="location_lat_dropoff")
        requestList = self.environment.rebalancing_grid.calculateGridIndex(liste=requestList, title_entry="index_Rebalancing_Pickup", title_longitude="location_lon", title_latitude="location_lat")
        requestList = self.environment.rebalancing_grid.calculateGridIndex(liste=requestList, title_entry="index_Rebalancing_Dropoff", title_longitude="location_lon_dropoff", title_latitude="location_lat_dropoff")

        # update features related to the environment
        self.predictor.feature_data.update_data(clock=self.environment.clock, vehicleList=vehicleList, requestList=requestList)

        # when we consider artificial vertices we need to proceed here:
        if self.args.policy in ["policy_SB", "policy_CB", "sampling"]:
            # we need to get artificial requests in the prediction horizon
            artRebVerticesList = self.environment.get_artificial_requests()
            # ... and assign to each artificial request location an index of the look-up cell
            if self.args.policy in ["policy_CB"]:  # here an artificial vertex equals the center of a rebalancing cell
                artRebVerticesList = self.environment.look_up_grid.calculateGridIndex(liste=artRebVerticesList, title_entry="index_Look_Up_Table_Pickup", title_longitude="x_center", title_latitude="y_center")
                artRebVerticesList = self.environment.rebalancing_grid.calculateGridIndex(liste=artRebVerticesList, title_entry="index_Rebalancing_Pickup", title_longitude="x_center", title_latitude="y_center")
                artRebVerticesList = self.predictor.feature_data.get_average_historical_data(pd.DataFrame(artRebVerticesList))
                artRebVerticesList = artRebVerticesList.rename(columns={"x_center": "location_lon", "y_center": "location_lat"}).to_dict("records")
            else:  # here an artificial vertex equals and artificial future ride request
                artRebVerticesList = self.environment.look_up_grid.calculateGridIndex(liste=artRebVerticesList, title_entry="index_Look_Up_Table_Pickup", title_longitude="location_lon", title_latitude="location_lat")
                artRebVerticesList = self.environment.rebalancing_grid.calculateGridIndex(liste=artRebVerticesList, title_entry="index_Rebalancing_Pickup", title_longitude="location_lon", title_latitude="location_lat")
                artRebVerticesList = self.environment.look_up_grid.calculateGridIndex(liste=artRebVerticesList, title_entry="index_Look_Up_Table_Dropoff", title_longitude="location_lon_dropoff", title_latitude="location_lat_dropoff")
                artRebVerticesList = self.environment.rebalancing_grid.calculateGridIndex(liste=artRebVerticesList, title_entry="index_Rebalancing_Dropoff", title_longitude="location_lon_dropoff", title_latitude="location_lat_dropoff")
                artRebVerticesList = util.sort_columns(vertices=artRebVerticesList, columns=list(pd.DataFrame(requestList).columns))

            if self.args.mode == "create_training_instance":
                full_information_artRebVerticesList = self.environment.get_artificial_requests(for_training=True)
                full_information_artRebVerticesList = self.environment.rebalancing_grid.calculateGridIndex(liste=full_information_artRebVerticesList, title_entry="index_Rebalancing_Pickup", title_longitude="location_lon", title_latitude="location_lat")
                full_information_artRebVerticesList = self.environment.rebalancing_grid.calculateGridIndex(liste=full_information_artRebVerticesList, title_entry="index_Rebalancing_Dropoff", title_longitude="location_lon_dropoff", title_latitude="location_lat_dropoff")
                full_information_solution_edges = self.graph_generator.get_full_information_solution_edges(edges=self.environment.full_information_data["edges_solution"],
                                                                                                           vehicleList=vehicleList,
                                                                                                           requestList=requestList,
                                                                                                           full_information_artRebVerticesList=full_information_artRebVerticesList,
                                                                                                           artRebVerticesList=artRebVerticesList)
            if self.args.policy == "policy_CB":
                artCapVerticesList = self.environment.get_capacity_vertices(artRebVerticesList)
            else:
                artCapVerticesList = []

            if self.args.policy in ["sampling"]:
                # we treat the sampled requests as normal requests
                requestList = requestList + artRebVerticesList
                artRebVerticesList = []
        else:
            artRebVerticesList = []
            artCapVerticesList = []

        # we generate the vertices list for the graph
        verticesList, vehicleList, requestList, artRebVerticesList, artCapVerticesList, idx_bins = self.graph_generator.get_vertices_list(vehicleList, requestList, artRebVerticesList, artCapVerticesList)

        # we initialize the C++ interface for the graph generator to generate the graph quickly
        self.graph_generator.fill_cplusplus_interface(requestList=requestList, vehicleList=vehicleList, artRebVertexList=artRebVerticesList)
        # the graph generator generates edges
        edges = {}
        edges["source_vehicles"] = self.graph_generator.calculate_edges_source_vehicle()
        edges["vehicles_sink"] = self.graph_generator.calculate_edges_vehicle_sink()
        edges["requests_sink"] = self.graph_generator.calculate_edges_rides_sink(requestList)
        edges["vehicles_requests"], edges_vehicles_requests_attributes = self.graph_generator.calculate_edges_vehicles_rides()
        edges["requests_requests"], edges_requests_requests_attributes = self.graph_generator.calculate_edges_requests_requests()

        # if we prepare training or evaluate learned policies we have to calculate edges to future locations
        if self.args.policy in ["policy_SB", "policy_CB"]:
            edges["vehicles_artRebVertices"], edges_vehicles_artRebVertices_attributes = self.graph_generator.calculate_edges_vehicles_artLoc()
            edges["requests_artRebVertices"], edges_requests_artRebVertices_attributes = self.graph_generator.calculate_edges_rides_artLoc()
            edges["artLocVertices_sink"] = self.graph_generator.calculate_edges_artLoc_sink(artRebVerticesList if self.args.policy == "policy_SB" else artCapVerticesList)
            edges["artRebVertices_artCapVertices"] = self.graph_generator.calculate_edges_artRebVertices_artCapVertices(pd.DataFrame(artRebVerticesList), pd.DataFrame(artCapVerticesList))
        else:
            edges["vehicles_artRebVertices"] = []
            edges_vehicles_artRebVertices_attributes = []
            edges["requests_artRebVertices"] = []
            edges_requests_artRebVertices_attributes = []
            edges["artLocVertices_sink"] = []
            edges["artRebVertices_artCapVertices"] = []

        # if we prepare training or evaluate learned policies we have to calculate the features related to vertices
        if self.args.policy in ["policy_SB", "policy_CB"]:
            # If we need to predict the weights of the digraph, we must assign features to the requests ...
            requestList = self.predictor.feature_data.append_features_to_requests(requestsList=pd.DataFrame(requestList))
            # ... and features to the vehicles ...
            vehicleList = self.predictor.feature_data.append_features_to_vehicles(vehicleList=pd.DataFrame(vehicleList))
            # ... and features to the artificial rebalancing vertices
            artRebVerticesList = self.predictor.feature_data.append_features_to_artRebVertices(artRebVerticesList=pd.DataFrame(artRebVerticesList))
            # ... and features to the artificial capacity vertices
            if self.args.policy in ["policy_CB"]:
                artCapVerticesList = self.predictor.feature_data.append_features_to_artCapVertices(artCapVerticesList=pd.DataFrame(artCapVerticesList))

        # we add general attributes to the edges and nodes and assign features to edges and nodes if we prepare training or evaluate learned polices
        edge_attributes = {}
        edge_attributes["features_edges_vehicles_requests"], edges["vehicles_requests"] = self.predictor.feature_data.append_features_to_edges_vehicles_requests(edges["vehicles_requests"], pd.DataFrame(vehicleList), pd.DataFrame(requestList), pd.DataFrame(edges_vehicles_requests_attributes))
        edge_attributes["features_edges_requests_requests"], edges["requests_requests"] = self.predictor.feature_data.append_features_to_edges_requests_requests(edges["requests_requests"], pd.DataFrame(requestList), pd.DataFrame(edges_requests_requests_attributes))
        edge_attributes["features_edges_requests_artRebVertices"], edges["requests_artRebVertices"] = self.predictor.feature_data.append_features_to_edges_requests_artRebVertices(edges["requests_artRebVertices"], pd.DataFrame(requestList), pd.DataFrame(artRebVerticesList), pd.DataFrame(edges_requests_artRebVertices_attributes))
        edge_attributes["features_edges_vehicles_artRebVertices"], edges["vehicles_artRebVertices"] = self.predictor.feature_data.append_features_to_edges_vehicles_artRebVertices(edges["vehicles_artRebVertices"], pd.DataFrame(vehicleList), pd.DataFrame(artRebVerticesList), pd.DataFrame(edges_vehicles_artRebVertices_attributes))
        edge_attributes["features_edges_artRebVertices_artCapVertices"], edges["artRebVertices_artCapVertices"] = self.predictor.feature_data.append_features_to_edges_artRebVertices_artCapVertices(edges["artRebVertices_artCapVertices"], pd.DataFrame(artRebVerticesList), pd.DataFrame(artCapVerticesList))

        node_attributes = {}
        node_attributes["features_nodes_vehicles"] = self.predictor.feature_data.get_node_features(vehicleList, ["origin"], ["on_trip_till"])
        node_attributes["features_nodes_requests"] = self.predictor.feature_data.get_node_features(requestList, ["origin", "destination", "future"], ["tpep_pickup_datetime", "tpep_dropoff_datetime"])
        node_attributes["features_nodes_artRebVertices"] = self.predictor.feature_data.get_node_features(artRebVerticesList, ["location"] if self.args.policy == "policy_CB" else ["destination", "future"], [])
        node_attributes["features_nodes_artCapVertices"] = self.predictor.feature_data.get_node_features(artCapVerticesList, ["capacity"], [])

        # if we evaluate the SB or CB policy we standardize features
        if self.args.mode == "evaluation" and self.args.policy in ["policy_SB", "policy_CB"]:
            edge_attributes = self.predictor.feature_data.standardize(edge_attributes)

        # we predict the weights on edges to artificial requests
        edge_features, node_features = self.predictor.retrieve_features(edge_attributes, node_attributes)
        weights = self.predictor.predict_weights(edge_features, node_features, edges, edge_attributes)

        # if we evaluate the sampling benchmark, we multiply the weights on edges to artificial requests with a discount factor to prioritize real requests
        if self.args.policy == "sampling":
            weights["requests_requests"] = self.graph_generator.reduce_weights_to_artificialVertices(edges["requests_requests"], weights["requests_requests"], clock=self.environment.clock, verticesList=verticesList)
            weights["vehicles_requests"] = self.graph_generator.reduce_weights_to_artificialVertices(edges["vehicles_requests"], weights["vehicles_requests"], clock=self.environment.clock, verticesList=verticesList)

        if self.args.mode not in ["create_training_instance"]:
            edgesList_solution, time_needed_k_disjoint = get_solution(self.args, edges, weights, len(vehicleList), self.graph_generator, self.combinatorial_optimizer)
        else:
            if self.args.save_data_visualization:
                edgesList_solution = [edge for edgelist in full_information_solution_edges.values() for edge in edges]
            else:
                edgesList_solution = edges["source_vehicles"] + edges["vehicles_sink"]

        # the decoder transforms the k-dSPP in coherent vehicle trips
        vehicleTrips, verticesIndexVehiclemapping = self.decoder.get_graph(edgesList_solution, num_vehicles=len(vehicleList), len_verticesList=len(verticesList), idx_bins=idx_bins)

        # we save the data for later visualization purpose
        if self.args.save_data_visualization:
            util.save_data_visualization(args=self.args, solution_edges=edgesList_solution, vehicleRoutes=vehicleTrips, vehicleList=vehicleList, verticesList=verticesList,
                                         requestList=requestList, clock=self.environment.clock)

        if self.args.mode == 'evaluation':
            vehicles_in_solution, requests_in_solution, locations_in_solution = self.environment.extract_served_vertices(verticesIndexVehiclemapping, verticesList)
        # we save the full-information solution
        if self.args.mode == "create_full_information_solution":
            util.save_full_information_solution(self.args, vehicleList, requestList, vehicleTrips, verticesIndexVehiclemapping, edgesList_solution)
        # to generate training instances we need to save the epoch instance and the respective full-information solution
        if self.args.mode == "create_training_instance":
            util.save_training_data(args=self.args, edges=edges,
                                       full_information_solution_edges=full_information_solution_edges, vehicleList=vehicleList,
                                       edge_attributes=edge_attributes, node_attributes=node_attributes,
                                       look_up_grid=self.environment.look_up_grid, verticesList=verticesList)

        # the environment relocates the vehicles according to the vehicle trips in the solution
        if self.args.mode not in ["create_training_instance"]:
            vehicleList, driving_distance = self.environment.update_vehicle_list(vehicleTrips=vehicleTrips, verticesList=verticesList, vehicleList=vehicleList)
            self.environment.vehicleList = self.predictor.delete_feature_columns_vehicle(pd.DataFrame(vehicleList))

        self.time_end_pipeline = time.time()
        if self.args.mode == "evaluation":
            self.fill_evaluation(requests_in_solution, driving_distance, time_needed_k_disjoint, verticesList)
