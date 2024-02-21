from src import util
import sys
sys.path.append('./cplusplus/cmake-build-release')
import pybind11module


class Graph_Generator_Cplusplus():
    def __init__(self, args, look_up_grid):
        self.args = args
        self.Cplusplus_interface = pybind11module.InterfaceGraphGenerator()
        self.Cplusplus_interface.look_up_table_distance = look_up_grid.table.distance
        self.Cplusplus_interface.look_up_table_duration = look_up_grid.table.duration
        self.Cplusplus_interface.average_speed_secPERkm = self.args.average_speed_secPERkm
        self.Cplusplus_interface.heuristic_distance_range = self.args.heuristic_distance_range
        self.Cplusplus_interface.heuristic_time_range = self.args.heuristic_time_range

    def set_values(self, requestList, vehicleList, artRebVertexList):
        columns_requests = ["tpep_pickup_datetime", "location_lon", "location_lat", "index_Look_Up_Table_Pickup", "tpep_dropoff_datetime", "location_lon_dropoff", "location_lat_dropoff", "index_Look_Up_Table_Dropoff"]
        requestCplusplus = util.prepare_list_for_CPlusPlus(args=self.args, vertex_list=requestList, columns=columns_requests, date_columns=["tpep_pickup_datetime", "tpep_dropoff_datetime"])
        columns_vehicles = ["on_trip_till", "location_lon", "location_lat", "index_Look_Up_Table_Dropoff"]
        vehicleCplusplus = util.prepare_list_for_CPlusPlus(args=self.args, vertex_list=vehicleList, columns=columns_vehicles, date_columns=["on_trip_till"])
        columns_artificialVertices = ["location_lon", "location_lat", "index_Look_Up_Table_Pickup", "tpep_pickup_datetime"]
        artRebVertexListCplusplus = util.prepare_list_for_CPlusPlus(args=self.args, vertex_list=artRebVertexList, columns=columns_artificialVertices, date_columns=["tpep_pickup_datetime"])

        self.Cplusplus_interface.request_vertices = requestCplusplus.values
        self.Cplusplus_interface.vehicle_vertices = vehicleCplusplus.values
        self.Cplusplus_interface.artificial_vertices = artRebVertexListCplusplus.values

        self.Cplusplus_interface.col_look_up_index_vehicle = list(vehicleCplusplus.columns).index("index_Look_Up_Table_Dropoff")
        self.Cplusplus_interface.col_look_up_index_request_pickup = list(requestCplusplus.columns).index("index_Look_Up_Table_Pickup")
        self.Cplusplus_interface.col_look_up_index_request_dropoff = list(requestCplusplus.columns).index("index_Look_Up_Table_Dropoff")
        self.Cplusplus_interface.col_look_up_index_artificialVertex_pickup = list(artRebVertexListCplusplus.columns).index("index_Look_Up_Table_Pickup") if len(artRebVertexListCplusplus) > 0 else -1
        self.Cplusplus_interface.num_vehicles = len(vehicleCplusplus)
        self.Cplusplus_interface.num_requests = len(requestCplusplus)
        self.Cplusplus_interface.num_artificialVertices = len(artRebVertexListCplusplus)
        self.Cplusplus_interface.col_time_vehicle = list(vehicleCplusplus.columns).index("on_trip_till")
        self.Cplusplus_interface.col_time_request_pickup = list(requestCplusplus.columns).index("tpep_pickup_datetime")
        self.Cplusplus_interface.col_time_request_dropoff = list(requestCplusplus.columns).index("tpep_dropoff_datetime")
        self.Cplusplus_interface.col_time_artificialVertex_pickup = list(artRebVertexListCplusplus.columns).index("tpep_pickup_datetime") if len(artRebVertexListCplusplus) > 0 else -1
        self.Cplusplus_interface.col_lon_request_pickup = list(requestCplusplus.columns).index("location_lon")
        self.Cplusplus_interface.col_lat_request_pickup = list(requestCplusplus.columns).index("location_lat")
        self.Cplusplus_interface.col_lon_request_dropoff = list(requestCplusplus.columns).index("location_lon_dropoff")
        self.Cplusplus_interface.col_lat_request_dropoff = list(requestCplusplus.columns).index("location_lat_dropoff")
        self.Cplusplus_interface.col_lon_vehicle_destination = list(vehicleCplusplus.columns).index("location_lon")
        self.Cplusplus_interface.col_lat_vehicle_destination = list(vehicleCplusplus.columns).index("location_lat")
        self.Cplusplus_interface.col_lon_artificialVertex_pickup = list(artRebVertexListCplusplus.columns).index("location_lon") if len(artRebVertexListCplusplus) > 0 else -1
        self.Cplusplus_interface.col_lat_artificialVertex_pickup = list(artRebVertexListCplusplus.columns).index("location_lat") if len(artRebVertexListCplusplus) > 0 else -1

    def calculateEdgesVehiclesRides(self):
        edges_vehicles_rides = self.Cplusplus_interface.calculateEdgesVehiclesRides()
        return edges_vehicles_rides

    def calculateEdgesRidesRides(self):
        edges_rides_rides = self.Cplusplus_interface.calculateEdgesRidesRides()
        return edges_rides_rides

    def caluclateEdgesRidesArtificialVertices(self):
        edges_rides_artificialVertices = self.Cplusplus_interface.caluclateEdgesRidesArtificialVertices()
        return edges_rides_artificialVertices

    def caluclateEdgesVehiclesArtificialVertices(self):
        edges_vehicles_artificialVertices = self.Cplusplus_interface.caluclateEdgesVehiclesArtificialVertices()
        return edges_vehicles_artificialVertices


class CombinatorialOptimizier():
    def __init__(self):
        pass

    def callrunKDisjoint(self, numVertices, edges, weights, numVehicles):
        CInstance = pybind11module.InstanceKDisjoint()
        CInstance.num_vertices = numVertices
        CInstance.num_vehicles = numVehicles
        CInstance.edges = edges
        CInstance.weights = weights
        CInstance.runKDisjoint()
        num_paths = CInstance.num_paths
        paths = CInstance.paths
        return [num_paths, paths]

    def callrunKEdgeDisjoint(self, numVertices, edges, weights, numVehicles):
        CInstance = pybind11module.InstanceKDisjoint()
        CInstance.num_vertices = numVertices
        CInstance.num_vehicles = numVehicles
        CInstance.edges = edges
        CInstance.weights = weights
        CInstance.runKEdgeDisjoint()
        num_paths = CInstance.num_paths
        paths = CInstance.paths_edge_disjoint
        return [num_paths, paths]

def callrunCalculateLoss(edges, all_edges_online, weights):
    CInstance = pybind11module.InstanceCalculateLoss()
    CInstance.edges = edges
    CInstance.all_edges_online = all_edges_online
    CInstance.weights = weights
    CInstance.runcalculateLoss()
    profit = CInstance.profit
    combinatorial_solution = CInstance.combinatorial_solution
    return profit, combinatorial_solution


def get_cost_smooth_features(base_features, granularity_counter, time_span_sec, mean_travel_distancePerTime, costs_per_km, type):
    CInstance = pybind11module.InstanceCalculateSmoothCostFeautres()
    CInstance.base_features = base_features.values
    CInstance.time_span_sec = time_span_sec
    CInstance.mean_travel_distancePerTime = mean_travel_distancePerTime
    CInstance.costs_per_km = costs_per_km
    CInstance.type = type
    CInstance.granularity_length = len(granularity_counter)
    CInstance.granularity_counter = granularity_counter
    CInstance.calculate_cost_smooth_feautres()
    features = CInstance.granularity_array
    return features
