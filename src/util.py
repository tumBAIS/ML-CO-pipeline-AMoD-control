import copy
from math import radians, cos, sin, asin, sqrt
import pandas as pd
import os
import pickle
import osmnx as ox
import geopandas as gpd


def haversine(location1, location2):
    lon1 = location1[0]  # representiv location_lon vehicle
    lat1 = location1[1]  # representiv location_lat vehicle
    lon2 = location2[0]  # representiv location_lon ride request
    lat2 = location2[1]  # representiv location_lat ride request
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r


def format_time(pandas_frame, pickup_time_name, dropoff_time_name):

    if len(pandas_frame) == 0:
        print("Pandas frame is empty")
        pandas_frame = pd.DataFrame(columns=["tpep_pickup_datetime", "tpep_dropoff_datetime", "trip_distance", "pickup_longitude",
                                             "pickup_latitude", "dropoff_longitude", "dropoff_latitude", "total_amount"])
    pandas_frame[pickup_time_name] = pd.to_datetime(pandas_frame[pickup_time_name], errors='coerce')
    pandas_frame[dropoff_time_name] = pd.to_datetime(pandas_frame[dropoff_time_name], errors='coerce')
    pandas_frame = pandas_frame.dropna()

    return pandas_frame


def prepare_list_for_CPlusPlus(args, vertex_list, date_columns, columns=None):
    if len(vertex_list) == 0:
        return pd.DataFrame()

    listCplusplus = pd.DataFrame(vertex_list)
    if columns is None:
        columns = list(listCplusplus.columns)
    listCplusplus = listCplusplus[columns]
    for date_column in date_columns:
        listCplusplus[date_column] = (listCplusplus[date_column] - args.start_date).dt.total_seconds()
    return listCplusplus


def sparse_taxi_data(args):
    print("We must generate the sparsed version of the taxi data to run this scenario ...")
    for month in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
        create_directory(args.directory_data + args.processed_ride_data_directory + "/" + "month-{:02}".format(month))
        for filename in os.listdir(args.directory_data + get_original_sparsityFactor_files(args.processed_ride_data_directory, args) + "/" + "month-{:02}".format(month)):
            print(filename)
            taxi_data = pd.read_csv(args.directory_data + get_original_sparsityFactor_files(args.processed_ride_data_directory, args) + "/" + "month-{:02}".format(month) + "/" + filename)
            taxi_data = taxi_data.sample(frac=args.sparsity_factor, random_state=123)
            taxi_data.to_csv(args.directory_data + args.processed_ride_data_directory + "/" + "month-{:02}".format(month) + "/" + filename)


def read_in_preprocessed_data(args, file, processed_ride_data_directory=None):
    print("Read in taxi data: {}".format(file))
    if processed_ride_data_directory is None:
        processed_ride_data_directory = args.processed_ride_data_directory

    # read in taxi data
    if not os.path.exists(args.directory_data + processed_ride_data_directory + file):
        sparse_taxi_data(args)
    taxi_data = pd.read_csv(args.directory_data + processed_ride_data_directory + file)
    # delete irrelevant columns
    taxi_data = taxi_data.loc[:, ~taxi_data.columns.str.contains('^Unnamed')]
    taxi_data = taxi_data.drop(columns=["in_NYC"], errors="ignore")

    # transform dates in pandas data frame into datetime objects
    taxi_data = format_time(taxi_data, "tpep_pickup_datetime", "tpep_dropoff_datetime")

    if args.objective == "amount_satisfied_customers":
        taxi_data["objective_value"] = 1
    elif args.objective == "profit":
        taxi_data["objective_value"] = taxi_data["total_amount"]

    # rename the longitude / latitude column with location_lon / location_lat
    taxi_data.rename(columns={"pickup_longitude": "location_lon"}, inplace=True)
    taxi_data.rename(columns={"pickup_latitude": "location_lat"}, inplace=True)

    taxi_data.rename(columns={"dropoff_longitude": "location_lon_dropoff"}, inplace=True)
    taxi_data.rename(columns={"dropoff_latitude": "location_lat_dropoff"}, inplace=True)

    # vanish comma figures so that we can treat weights as integers
    taxi_data["objective_value"] = 100 * taxi_data["objective_value"]

    # transform trip distance from miles to kilometers
    taxi_data["trip_distance"] = args.conversion_milesToKm * taxi_data["trip_distance"]
    taxi_data = taxi_data.rename(columns={"duration_sec": "ride_duration", "trip_distance": "ride_distance"})
    return taxi_data


def sort_columns(vertices, columns):
    if not isinstance(vertices, pd.DataFrame):
        vertices = pd.DataFrame(vertices)
    return vertices[columns].to_dict("records")


def create_directories(args):
    if args.mode == "create_full_information_solution":
        create_directory(args.full_information_solution_directory)
    elif args.mode == "create_training_instance":
        create_directory(args.training_instance_directory)
    elif args.mode == "training":
        create_directory(args.learned_parameters_directory)
    elif args.mode == "sanity_check":
        create_directory(args.sanity_directory)
    elif args.mode == "evaluation":
        create_directory(args.result_directory)


def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)
        print("Directory ", directory_path, " Created ")
    else:
        print("Directory ", directory_path, " already exists")


def define_costs_per_km(args):
    if args.objective == "amount_satisfied_customers":
        args.costs_per_km = args.amount_satisfied_customers_penalty_per_km
    elif args.objective == "profit":
        args.costs_per_km = args.costs_per_km
    else:
        raise Exception("Wrong objective defined.")
    return args


def check_args_compatibility(args):
    if args.policy == "offline":
        # System time period must be equal to problem horizon
        args.system_time_period = args.end_date - args.start_date
    return args


def get_abb_standardization():
    return "st"


def get_abb_model():
    return "m"


def get_standardization_str(args):
    return f"{get_abb_standardization()}-{args.standardization}"


def get_fleetSize_str(args):
    return f"{get_abbreviation_fleetSize()}-{args.num_vehicles}"


def get_abbreviation_fleetSize():
    return "fleetSi"


def get_abb_sparsityFactor():
    return "sF"


def get_sparsityFactor_str(args):
    return f"{get_abb_sparsityFactor()}-{args.sparsity_factor}"


def get_abb_obj_str():
    return "obj"


def get_obj_str(args):
    return f"{get_abb_obj_str()}-{args.objective}"


def get_trIteration(args):
    return "it-{}".format(args.read_in_iterations)


def get_abb_heurTimeRange():
    return "tR"


def get_heurTimeRange(args):
    return f"{get_abb_heurTimeRange()}-" + str(args.heuristic_time_range)


def get_abb_heurDistanceRange():
    return "dR"


def get_abb_learning_rate():
    return "lr"


def get_heurDistanceRange(args):
    return f"{get_abb_heurDistanceRange()}-" + str(args.heuristic_distance_range)


def get_sys_time_period_str(args):
    return "sysTimePeriod-{:02}".format(args.system_time_period.seconds / 60)


def get_startDate_str(args):
    return "startDat-" + args.start_date.strftime('%Y-%m-%d %H:%M:%S')


def get_endDate_str(args):
    return "endDat-" + args.end_date.strftime('%Y-%m-%d %H:%M:%S')


def get_instDate_str(args):
    return "instDate-{:02}-{:02}-{:02}".format(args.instance_date.hour, args.instance_date.minute, args.instance_date.second)


def get_abbreviation_ext_horizon():
    return "hoSi"


def get_abb_hidden_units():
    return "hu"


def get_abb_number_layers():
    return "nl"


def get_ext_horizon_str(args):
    if isinstance(args.extended_horizon, int):
        return get_abbreviation_ext_horizon() + "-{:02}".format(args.extended_horizon)
    else:
        return get_abbreviation_ext_horizon() + "-{:02}".format(args.extended_horizon.seconds / 60)


def get_perturbations(args):
    return "pert-{},{}".format(args.perturbed_optimizer[0], args.perturbed_optimizer[1])


def get_abbreviation_reduction_factor():
    return "rF"


def get_red_factor_str(args):
    return get_abbreviation_reduction_factor() + "-{}".format(args.reduce_weight_of_future_requests_factor)


def get_policy_str(args):
    return args.policy


def get_abb_max_cap():
    return "cap"


def get_max_capacities(args):
    return f"{get_abb_max_cap()}-{args.fix_capacity}"


def get_original_sparsityFactor_files(file, args):
    if "sF" in file:
        return file.split("_sF")[0]
    else:
        return file


def get_sanity_file(args):
    return args.sanity_directory + f"Sanity_{args.read_in_iterations}" + (f"-{get_policy_str(args)}_{get_standardization_str(args)}_"
                                                                          f"{get_sparsityFactor_str(args)}_{get_obj_str(args)}_"
                                                                          f"{get_ext_horizon_str(args)}_{get_fleetSize_str(args)}_"
                                                                          f"{get_perturbations(args)}_{get_predictor(args)}")


def get_learning_file(args, step, NNmodel=False):
    return args.learned_parameters_directory + (f"Learning_{step}" if step is not None else "Learning") + get_learning_file_name(args)


def get_learning_file_name(args):
    return f"-{get_obj_str(args)}_{get_ext_horizon_str(args)}_{get_fleetSize_str(args)}_{get_perturbations(args)}_{get_predictor(args)}"


def get_learning_rate(args):
    return f"{get_abb_learning_rate()}-{args.learning_rate}"


def get_hidden_units(args):
    return f"{get_abb_hidden_units()}-{args.hidden_units}"


def get_number_layers(args):
    return f"{get_abb_number_layers()}-{args.num_layers}"


def get_model(args):
    return f"{get_abb_model()}-{args.model}"


def get_abb_normalization():
    return "nrm"


def get_normalization(args):
    return f"{get_abb_normalization()}-{args.normalization}"


def get_predictor(args):
    if args.model in ["NN"]:
        return f"{get_model(args)}" + f"_{get_learning_rate(args)}" + f"_{get_hidden_units(args)}" + f"_{get_normalization(args)}_{get_number_layers(args)}"
    else:
        return f"{get_model(args)}"


def get_result_directory(args):
    return (args.result_directory + (f"_{get_policy_str(args)}_{get_fleetSize_str(args)}_{get_sparsityFactor_str(args)}_{get_obj_str(args)}" +
                            (f"_{get_ext_horizon_str(args)}" if args.policy in ["policy_SB", "policy_CB", "sampling"] else "") +
                            (f"_{get_red_factor_str(args)}" if args.policy == "sampling" else "") +
                            (f"_{get_standardization_str(args)}" if args.policy in ["policy_SB", "policy_CB"] else "") +
                            (f"_{get_max_capacities(args)}" if args.policy in ["policy_CB"] else "") if args.mode == "evaluation" else "") +
            (f"_{get_predictor(args)}" if args.model in ["NN", "Linear"] else "") +
            "/")


def set_additional_args(args):
    # set correct model
    if args.mode in ["evaluation", "training", "sanity_check"] and args.policy in ["policy_SB", "policy_CB"]:
        args.model = args.model
    else:
        args.model = "Greedy"
    # the vehicle distribution at the start of the simulation depends on: time, objective, number of vehicles, number of requests
    args.vehDistribution_file = args.vehDistribution_file + f"{args.start_date.hour}"
    args.vehDistribution_directory = args.directory_data + args.vehDistribution_directory + f"_{get_obj_str(args)}_{get_fleetSize_str(args)}_{get_sparsityFactor_str(args)}/"
    # the ride request data depends on the sparsity factor which determines the request density
    args.processed_ride_data_directory = args.processed_ride_data_directory + (("_" + get_sparsityFactor_str(args)) if args.sparsity_factor != 1 else "")
    # the historical ride request distribution data depends on the rebalancing cells, and the request density
    args.historicalData_directory = args.historicalData_directory + f"_{args.amount_of_cellsRebalancing[0]}_{args.amount_of_cellsRebalancing[1]}"
    args.historicalData_directory = args.historicalData_directory + (("_" + get_sparsityFactor_str(args)) if args.sparsity_factor != 1 else "")
    # training_instance_directory
    args.training_instance_directory = args.training_instance_directory + f"_{args.policy}_{args.num_vehicles}_{args.sparsity_factor}_{args.objective}_{args.extended_horizon}_{args.standardization}/"
    # learned_parameters_directory
    args.learned_parameters_directory = args.learned_parameters_directory + f"_{args.policy}_{args.num_vehicles}_{args.sparsity_factor}_{args.objective}_{args.extended_horizon}_{args.standardization}/"
    # result_directory
    args.result_directory = get_result_directory(args)
    return args


# returns false, if the request was not served
def get_served_vertices(item):
    if item == -1:
        return False
    else:
        return True


def get_full_information_solution_path(args):
    return args.full_information_solution_directory + f"FIsolution_{get_obj_str(args)}_{get_startDate_str(args)}_{get_endDate_str(args)}_{get_fleetSize_str(args)}_{get_sparsityFactor_str(args)}"


# read in full-information solution
def read_in_full_information_solution(args):
    instance_input_file = open(get_full_information_solution_path(args), "rb")
    instance_dict = pickle.load(instance_input_file)
    instance_input_file.close()
    return instance_dict


# save instance data in json format into document
def save_full_information_solution(args, vehicleList, requestList, vehicleRoutes, verticesIndexVehiclemapping, edges):
    instance_data = {"vehicleList": vehicleList,
                     "requestList": requestList,
                     "vehicleRoutes": vehicleRoutes,
                     "verticesIndexVehiclemapping": verticesIndexVehiclemapping,
                     "edges_solution": edges}

    instance_output_file = open(get_full_information_solution_path(args), "wb")

    pickle.dump(instance_data, instance_output_file)
    instance_output_file.close()


def get_file_data_visualization(args):
    return (args.directory_save_data_visualization +
            f"Source_{get_obj_str(args)}_{args.policy}_{get_startDate_str(args)}_{get_endDate_str(args)}_{get_fleetSize_str(args)}" +
            (f"_{get_max_capacities(args)}" if args.policy == "policy_CB" else ""))


def save_data_visualization(args, vehicleRoutes, vehicleList, requestList, verticesList, clock, solution_edges):
    if args.mode == "create_training_instance":
        args.policy = "offline"

    instance_data = {"vehicleRoutes": vehicleRoutes, "vehicleList": vehicleList, "requestList": requestList, "verticesList": verticesList, "solution_edges": solution_edges}
    directory_save_data_visualization = get_file_data_visualization(args)
    create_directory(directory_save_data_visualization)
    instance_output_file = open(directory_save_data_visualization + "/" + "Source" +
                                "_now:" + clock.actual_system_time_period_start.strftime('%Y-%m-%d %H:%M:%S'), "wb")
    pickle.dump(instance_data, instance_output_file)
    instance_output_file.close()


def get_network_graph(args):
    if os.path.isfile(args.directory_data + 'manhattan_graph.graphml'):
        graph = ox.load_graphml(args.directory_data + 'manhattan_graph.graphml')
    else:
        graph = ox.graph_from_place('Manhattan, New York, USA', network_type="drive")
        # save graph to disk
        ox.save_graphml(graph, args.directory_data + 'manhattan_graph.graphml')
    return graph


def get_manhattan_shapefile(args, dot=""):
    nyc_shape_withWater = gpd.read_file(dot + args.directory_data + args.manhattan_shape)
    shapefile = nyc_shape_withWater.loc[nyc_shape_withWater['NAME'] == "New York"]
    shapefile = shapefile.loc[shapefile['COUNTYFP'].isin(["085", "061", "047", "005", "081"])]
    return shapefile


def get_manhattan_exact_shapefile(args):
    nyc_shape = gpd.read_file(args.directory_data + args.manhattan_shape_exact)
    shapefile = nyc_shape.loc[nyc_shape['BoroName'] == "Manhattan"]
    return shapefile


def save_training_data(args, edges, full_information_solution_edges, vehicleList, edge_attributes, node_attributes, look_up_grid, verticesList):
    def save(training_data):
        instance_output_file = open(args.training_instance_directory + f"TrainingInstance:{get_obj_str(args)}_{get_startDate_str(args)}_{get_endDate_str(args)}_"
                                                                       f"{get_sys_time_period_str(args)}_{get_ext_horizon_str(args)}_{get_instDate_str(args)}_{get_fleetSize_str(args)}_NN", "wb")

        pickle.dump(training_data, instance_output_file)
        instance_output_file.close()

    missing_edges_vehicles_requests = [edge for edge in full_information_solution_edges["edgeVehiclesRequests_FI"] if edge not in edges["vehicles_requests"]]
    missing_edges_requests_requests = [edge for edge in full_information_solution_edges['edgeRequestsRequests_FI'] if edge not in edges["requests_requests"]]
    missing_edges_vehicles_artRebVertices = [edge for edge in full_information_solution_edges['edgeVehiclesArtRebVertices_FI'] if edge not in edges["vehicles_artRebVertices"]]
    missing_edges_requests_artRebVertices = [edge for edge in full_information_solution_edges['edgeRequestsArtRebVertices_FI'] if edge not in edges["requests_artRebVertices"]]
    missing_edges_artRebVertices_artCabVertices = [edge for edge in full_information_solution_edges['edgeArtRebVerticesArtCapVertices_FI'] if edge not in edges['artRebVertices_artCapVertices']]
    if len(missing_edges_vehicles_requests) > 0:
        raise Exception(f"Missing edges vehicles -> requests: {missing_edges_vehicles_requests}")
    if len(missing_edges_requests_requests) > 0:
        raise Exception(f"Missing edges requests -> requests: {missing_edges_requests_requests}")
    if len(missing_edges_vehicles_artRebVertices) > 0:
        if args.policy == "policy_SB":
            raise Exception(f"Missing edges vehicles -> artRebVertices: {missing_edges_vehicles_artRebVertices}")
    if len(missing_edges_requests_artRebVertices) > 0:
        if args.policy == "policy_SB":
            raise Exception(f"Missing edges requests -> artRebVertices: {missing_edges_requests_artRebVertices}")
    if len(missing_edges_artRebVertices_artCabVertices) > 0:
        raise Exception(f"Missing edges artRebVertices -> artCabVertices: {missing_edges_artRebVertices_artCabVertices}")

    # check if we do not incorporate these edges due to distance or duration cut!
    def check_distance_duration_cut(edges, starting_time, starting_lon, starting_lat):
        checker_list = []
        for edge in edges:
            distance_km = look_up_grid.table.distance[verticesList[edge[0]]['index_Look_Up_Table_Dropoff']][verticesList[edge[1]]['index_Look_Up_Table_Pickup']]
            duration_sec = look_up_grid.table.duration[verticesList[edge[0]]['index_Look_Up_Table_Dropoff']][verticesList[edge[1]]['index_Look_Up_Table_Pickup']]
            if (distance_km == 0) or (duration_sec == 0):
                distance_km = haversine([verticesList[edge[0]][starting_lon], verticesList[edge[0]][starting_lat]], [verticesList[edge[1]]["location_lon"], verticesList[edge[1]]["location_lat"]])
                duration_sec = distance_km * args.average_speed_secPERkm
            time_puffer = (verticesList[edge[1]]['tpep_pickup_datetime'] - verticesList[edge[0]][starting_time]).total_seconds()
            if (distance_km >= args.heuristic_distance_range) or (time_puffer >= args.heuristic_time_range) or (duration_sec >= time_puffer) or (time_puffer <= 0):
                checker_list.append(True)
            else:
                checker_list.append(False)
        return all(checker_list)

    def remove(edges, edges_to_remove):
        artRebVertices_idxs = []
        for edge in edges_to_remove:
            edges.remove(edge)
            artRebVertices_idxs.append(edge[1])
        return artRebVertices_idxs

    def remove_from_capacity(edges, vertices_to_capacity_remove):
        for vertex in vertices_to_capacity_remove:
            edge_to_delete = sorted([edge for edge in edges if edge[0] == vertex])[-1]
            edges.remove(edge_to_delete)

    if args.policy == "policy_CB":
        assert check_distance_duration_cut(missing_edges_vehicles_artRebVertices, "on_trip_till", "location_lon", "location_lat") and \
               check_distance_duration_cut(missing_edges_requests_artRebVertices, "tpep_dropoff_datetime", "location_lon_dropoff", "location_lat_dropoff")
        # we assume that not assignable edges lead to the sink
        artRebVertices_idxs_to_capacity_cells_to_remove_from_vehicle = remove(full_information_solution_edges['edgeVehiclesArtRebVertices_FI'], missing_edges_vehicles_artRebVertices)
        artRebVertices_idxs_to_capacity_cells_to_remove_from_request = remove(full_information_solution_edges['edgeRequestsArtRebVertices_FI'], missing_edges_requests_artRebVertices)
        remove_from_capacity(full_information_solution_edges['edgeArtRebVerticesArtCapVertices_FI'], artRebVertices_idxs_to_capacity_cells_to_remove_from_vehicle+artRebVertices_idxs_to_capacity_cells_to_remove_from_request)

    def check_if_full_information_solution_is_feasible_solution():
        print("We check if we can find the full-information solution")
        solution_edges = full_information_solution_edges["edgeVehiclesRequests_FI"] + full_information_solution_edges['edgeRequestsRequests_FI'] + full_information_solution_edges['edgeVehiclesArtRebVertices_FI'] + \
                         full_information_solution_edges['edgeRequestsArtRebVertices_FI'] + full_information_solution_edges['edgeArtRebVerticesArtCapVertices_FI']
        solution_edges_rebVertices_artCapVertices = copy.deepcopy(full_information_solution_edges['edgeArtRebVerticesArtCapVertices_FI'])
        training_edges = edges["source_vehicles"] + edges["vehicles_sink"] + edges["requests_sink"] + edges["artLocVertices_sink"] + \
                edges["vehicles_requests"] + edges["requests_requests"] + edges["vehicles_artRebVertices"] + \
                edges["requests_artRebVertices"] + edges["artRebVertices_artCapVertices"]
        solution_edges.sort(key=lambda tup: tup[0])
        vehicleTrips = {}
        verticesIndexVehicleMapping = {}

        for solution_edge in solution_edges:
            if solution_edge in edges["artRebVertices_artCapVertices"]:
                continue
            if solution_edge[0] not in verticesIndexVehicleMapping.keys():
                verticesIndexVehicleMapping[solution_edge[1]] = solution_edge[0]
                vehicleTrips[solution_edge[0] - 1] = [0, solution_edge[0], solution_edge[1]]
                if (args.policy == "policy_CB") and solution_edge in full_information_solution_edges['edgeVehiclesArtRebVertices_FI']:
                    edge_rebVertex_artCapVertex = [edge for edge in solution_edges_rebVertices_artCapVertices if edge[0] == solution_edge[1]][0]
                    vehicleTrips[solution_edge[0] - 1].append(edge_rebVertex_artCapVertex[1])
                    remove(solution_edges_rebVertices_artCapVertices, [edge_rebVertex_artCapVertex])
            else:
                vehicle = verticesIndexVehicleMapping[solution_edge[0]]
                verticesIndexVehicleMapping[solution_edge[1]] = vehicle
                vehicleTrips[vehicle-1].append(solution_edge[1])
                if (args.policy == "policy_CB") and solution_edge in full_information_solution_edges['edgeVehiclesArtRebVertices_FI'] + full_information_solution_edges['edgeRequestsArtRebVertices_FI']:
                    edge_rebVertex_artCapVertex = [edge for edge in solution_edges_rebVertices_artCapVertices if edge[0] == solution_edge[1]][0]
                    vehicleTrips[vehicle - 1].append(edge_rebVertex_artCapVertex[1])
                    remove(solution_edges_rebVertices_artCapVertices, [edge_rebVertex_artCapVertex])

        for vehicle_trip in vehicleTrips.values():
            vehicle_trip = vehicle_trip + [len(verticesList) - 1]
            for vertex_idx, vertex in enumerate((vehicle_trip)[:-1]):
                assert (vertex, vehicle_trip[vertex_idx+1]) in training_edges

    check_if_full_information_solution_is_feasible_solution()

    training_data = {"edges_source_vehicles": edges["source_vehicles"],
                  "edges_vehicles_sink": edges["vehicles_sink"],
                  "edges_requests_sink": edges["requests_sink"],
                  "edges_artLocVertices_sink": edges["artLocVertices_sink"],
                    "node_attributes": node_attributes,
                  "features_edges_vehicles_requests": edge_attributes["features_edges_vehicles_requests"],
                  "features_edges_requests_requests": edge_attributes["features_edges_requests_requests"],
                  "features_edges_vehicles_artRebVertices": edge_attributes["features_edges_vehicles_artRebVertices"],
                  "features_edges_requests_artRebVertices": edge_attributes["features_edges_requests_artRebVertices"],
                  "features_edges_artRebVertices_artCapVertices": edge_attributes["features_edges_artRebVertices_artCapVertices"],
                  "full_information_solution_edges": full_information_solution_edges, "len_verticesList": len(verticesList), "len_vehicleList": len(vehicleList)}
    save(training_data)