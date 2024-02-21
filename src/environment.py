import json
import pandas as pd
from src.clock import Clock
from src.sampler import Sampler
from src import util
from prep.preprocess_grid import RebalancingGrid, LookUpGrid
from datetime import timedelta
import copy
from itertools import compress
import os
import scipy.stats as st


class Environment:
    def __init__(self, args, random_state):
        self.args = args
        self.look_up_grid = LookUpGrid(self.args)
        if args.mode == "create_training_instance":
            self.full_information_data = util.read_in_full_information_solution(args)  # To generate training data we need to read in the full-information solution
        self.random_state = random_state
        self.read_in_times()
        self.taxi_data = self.read_in_taxi_trip_data()
        self.vehicleList = self.read_in_vehicle_data()
        self.clock = Clock(self.args, start_date=self.start_date if args.mode != "create_training_instance" else args.instance_date, end_date=self.end_date, taxi_data=self.taxi_data)
        self.rebalancing_grid = RebalancingGrid(self.args)
        if self.args.policy in ['policy_SB', 'policy_CB', 'sampling']:
            self.sampler = Sampler(args, self.clock, self.rebalancing_grid)

    def get_capacity_vertices(self, artRebVerticesList):
        capacity_nodes = pd.DataFrame(artRebVerticesList)
        capacity_nodes["type"] = "Capacity"
        capacity_nodes["counter"] = 1
        capacity_list = [copy.deepcopy(capacity_nodes)]
        for counter in list(range(2, self.args.fix_capacity + 1)):
            capacity_nodes["counter"] = counter
            capacity_list.append(copy.deepcopy(capacity_nodes))
        return pd.concat(capacity_list).to_dict("records")

    def get_artificial_requests(self, for_training=False):
        if for_training or (self.args.mode == "create_training_instance" and self.args.policy == "policy_SB"):
            artificial_requests = self.simulate_ride_requests(start=self.clock.actual_system_time_period_end, end=self.clock.extended_horizon_end, vertex_type="Location")
        else:
            if self.args.policy == "policy_CB":
                artificial_requests = self.get_static_artificial_rebalancing_vertices()
            else:
                artificial_requests = self.sampler.sample_from_historical_data()
            artificial_requests["type"] = "Location"
        return artificial_requests.to_dict("records")

    def get_static_artificial_rebalancing_vertices(self):
        artRebVertices = pd.DataFrame(copy.deepcopy(self.rebalancing_grid.grid))
        artRebVertices = artRebVertices.drop(columns=["x_min", "x_max", "y_min", "y_max", "in_NYC"])
        return artRebVertices

    def simulate_ride_requests(self, start=None, end=None, vertex_type="Request"):
        if start is None:
            start = self.clock.actual_system_time_period_start
        if end is None:
            end = self.clock.actual_system_time_period_end
        requestList = copy.deepcopy(self.taxi_data.loc[(self.taxi_data["tpep_pickup_datetime"] >= start) & (self.taxi_data["tpep_pickup_datetime"] < end)])
        requestList = requestList.sort_values(by=["tpep_dropoff_datetime"])
        requestList["type"] = vertex_type
        if self.args.mode == "create_training_instance":
            requestList = requestList.rename(columns={"vertex_idx": "full_information_vertex_idx"})
        requestList = requestList.reset_index(drop=True)
        return requestList

    def extract_served_vertices(self, verticesIndexVehiclemapping, verticesList):
        # get satisfied requests
        boolean_vertices_in_solution = list(map(util.get_served_vertices, verticesIndexVehiclemapping))
        vertices_in_solution = list(compress(verticesList, boolean_vertices_in_solution))
        vehicles_in_solution = [request for request in vertices_in_solution if request["type"] == "Vehicle"]
        requests_in_solution = [request for request in vertices_in_solution if request["type"] == "Request"]
        locations_in_solution = [request for request in vertices_in_solution if request["type"] == "Location"]
        return vehicles_in_solution, requests_in_solution, locations_in_solution

    def get_distance_and_duration(self, location_lon_start, location_lat_start, location_lon_end, location_lat_end):
        look_up_index_start = self.look_up_grid.get_idx(location_lon_start, location_lat_start)
        look_up_index_end = self.look_up_grid.get_idx(location_lon_end, location_lat_end)
        distance = self.look_up_grid.table.distance[look_up_index_start][look_up_index_end]
        duration = self.look_up_grid.table.duration[look_up_index_start][look_up_index_end]
        if (distance) == 0 or (duration == 0):
            distance = util.haversine(location1=(location_lat_start, location_lon_start), location2=(location_lat_end, location_lon_end))
            duration = distance * self.args.average_speed_secPERkm
        return distance, duration

    # this is a function to get the actual position of a vehicle when driving to a rebalancing location.
    def get_rebalancing_location(self, origin_lon, origin_lat, origin_on_trip_till, destination_lon, destination_lat, now):
        # get the ETA of the vehicle to location
        distance_km, duration_sec = self.get_distance_and_duration(origin_lon, origin_lat, destination_lon, destination_lat)
        eta = origin_on_trip_till + timedelta(seconds=duration_sec)
        # Option 3: vehicle reached the location already
        if eta < now:
            last_vertex_dict = {"location_lat": destination_lat,
                                "location_lon": destination_lon,
                                "on_trip_till": now,
                                "type": "Location"}
            dist_travelled_rebalancing = distance_km
        # Option 2: the vehicle not yet reached the location
        else:
            driven_time = (now - origin_on_trip_till).total_seconds()
            dist_lat = destination_lat - origin_lat
            dist_lon = destination_lon - origin_lon
            location_lat = origin_lat + (driven_time / (duration_sec)) * dist_lat
            location_lon = origin_lon + (driven_time / (duration_sec)) * dist_lon
            last_vertex_dict = {"location_lat": location_lat,
                                "location_lon": location_lon,
                                "on_trip_till": now,
                                "type": "Location"}
            dist_travelled_rebalancing = (driven_time / (duration_sec)) * distance_km
        return last_vertex_dict, dist_travelled_rebalancing

    def update_vehicle_list(self, vehicleTrips, verticesList, vehicleList, load_instance=False):
        # realize vehicle trips and set new vehicle locations, and on_trip_till values
        # we calculate the driving distance
        # 1. driving to request
        distance_driving_to_request = 0
        # 2. transporting request
        distance_transporting_request = 0
        # 3. rebalancing
        distance_rebalancing = 0
        if load_instance is True:
            now = self.args.instance_date
        else:
            now = self.clock.actual_system_time_period_end
        for vehicle_idx in range(self.args.num_vehicles):
            vertex_idx = vehicle_idx + 1
            vehicle_vertex_dict = verticesList[vertex_idx]
            on_trip_till = vehicle_vertex_dict.get("on_trip_till")
            after_trip_lon = vehicle_vertex_dict["location_lon"]
            after_trip_lat = vehicle_vertex_dict["location_lat"]
            # The route of a vehicle is the sequence of vertices that a vehicle services.
            # [source -> vehicle -> request_1 -> .... - > request_n -> sink]
            vehicle_trip = vehicleTrips[vehicle_idx]
            # If the route only consists of one element, this is the vehicle itself.
            # Thus the vehicle was not in the graph because the vehicle is on tour
            if len(vehicle_trip) == 2:
                if on_trip_till < now:
                    vehicleList[vehicle_idx]["on_trip_till"] = now
            else:
                # there can be two types of vertices in the vehicle trip: requests, locations
                for index, vertex_in_trip in list(enumerate(vehicle_trip[1:])):
                    if on_trip_till > now:
                        break
                    vertex_in_trip_dict = verticesList[vertex_in_trip]
                    if vertex_in_trip_dict.get("type") == "Request":
                        if vertex_in_trip_dict.get("tpep_pickup_datetime") >= now:  # this is case when reading in full-information solution
                            continue
                        # the vehicle serves the request
                        distance_driving_to_request += self.get_distance_and_duration(after_trip_lon, after_trip_lat, vertex_in_trip_dict["location_lon"], vertex_in_trip_dict["location_lat"])[0]
                        distance_transporting_request += vertex_in_trip_dict["ride_distance"]
                        on_trip_till = vertex_in_trip_dict["tpep_dropoff_datetime"]
                        after_trip_lon = vertex_in_trip_dict["location_lon_dropoff"]
                        after_trip_lat = vertex_in_trip_dict["location_lat_dropoff"]
                        vertex_idx = vertex_in_trip_dict["vertex_idx"]
                    elif vertex_in_trip_dict.get("type") == "Location":
                        rebalancing_location, dist_travelled_rebalancing = self.get_rebalancing_location(origin_lon=after_trip_lon, origin_lat=after_trip_lat, origin_on_trip_till=on_trip_till,
                                                                                                         destination_lon=vertex_in_trip_dict["location_lon"], destination_lat=vertex_in_trip_dict["location_lat"], now=now)
                        distance_rebalancing += dist_travelled_rebalancing
                        on_trip_till = rebalancing_location["on_trip_till"]
                        after_trip_lon = rebalancing_location["location_lon"]
                        after_trip_lat = rebalancing_location["location_lat"]
                        vertex_idx = vertex_in_trip_dict["vertex_idx"]
                        break

                # if a vehicle does not service a request. It is on a trip till NOW,
                if on_trip_till < now and not load_instance:
                    on_trip_till = now
                vehicleList[vehicle_idx]["on_trip_till"] = on_trip_till
                vehicleList[vehicle_idx]["location_lon"] = after_trip_lon
                vehicleList[vehicle_idx]["location_lat"] = after_trip_lat
                vehicleList[vehicle_idx]["full_information_vertex_idx"] = vertex_idx
        return vehicleList, distance_driving_to_request + distance_transporting_request + distance_rebalancing

    def get_vehicle_occupied_sample_distribution(self):
        print("Here we have to reset the sample input file")
        sample_input_file = open(self.args.vehDistribution_directory + self.args.vehDistribution_file, "r")
        dist_dict = json.load(sample_input_file)
        distribution_name = dist_dict.get("distribution")
        distribution_param = dist_dict.get("parameters")
        arg = distribution_param[:-2]
        loc = distribution_param[-2]
        scale = distribution_param[-1]
        bernoulli_param = dist_dict.get("bernoulli_param")

        return distribution_name, loc, scale, arg, bernoulli_param

    def read_in_vehicle_data(self):
        print("We start reading in vehicle data ...")
        if self.args.mode == "create_training_instance":
            vehicleList = self.full_information_data.get("vehicleList")
            requestList = self.full_information_data.get("requestList")
            vehicleList = pd.DataFrame(vehicleList).rename(columns={"vertex_idx": "full_information_vertex_idx"}).to_dict("records")
            # we set the vehicles to its starting point according to the time point in the respective offline instance
            vehicleList, _ = self.update_vehicle_list(vehicleTrips=self.full_information_data.get("vehicleRoutes"),
                                                                       verticesList=[{"type": "Source"}] + vehicleList + requestList + [{"type": "Sink"}],
                                                                       vehicleList=vehicleList, load_instance=True)
            return vehicleList
        else:
            vehicleList = []
            if self.args.sample_starting_vehicle_activities:
                if not os.path.isfile(self.args.vehDistribution_directory + self.args.vehDistribution_file):
                    os.system(f"python prep/preprocess_vehicleDistribution.py "
                              f"--start_date='{self.args.start_date}' "
                              f"--end_date='{self.args.end_date}' "
                              f"--objective={self.args.objective} "
                              f"--num_vehicles={self.args.num_vehicles} "
                              f"--processed_ride_data_directory={self.args.processed_ride_data_directory} "
                              f"--historicalData_directory={self.args.historicalData_directory} "
                              f"--vehDistribution_directory={self.args.vehDistribution_directory} "
                              f"--vehDistribution_file={self.args.vehDistribution_file} "
                              f"--sparsity_factor={self.args.sparsity_factor} ")
                distribution_name, loc, scale, arg, bernoulli_param = self.get_vehicle_occupied_sample_distribution()
                distribution_time = getattr(st, distribution_name)
                distribution_bernoulli = st.bernoulli
            else:
                distribution_name = None
            # for each vehicle ...
            for i in range(self.args.num_vehicles):
                # ... we sample the origin location from the locations of the requests ...
                vehicle = self.taxi_data.sample(random_state=self.random_state)
                lat = vehicle.location_lat.values[0]
                lon = vehicle.location_lon.values[0]
                # ... and we sample on_trip_till from precalculated distribution ...
                if distribution_name is not None:
                    if distribution_bernoulli.rvs(bernoulli_param, size=1, random_state=self.random_state):
                        on_trip_till = int(distribution_time.rvs(size=1, loc=loc, scale=scale, random_state=self.random_state, *arg)[0])
                        on_trip_till = self.args.start_date + timedelta(seconds=on_trip_till)
                    else:
                        on_trip_till = self.args.start_date
                else:
                    on_trip_till = self.args.start_date
                # ... and add each vehicle as a dictionary to the vehicle list.
                vehicleList.append({
                    "location_lon": lon,
                    "location_lat": lat,
                    "on_trip_till": on_trip_till,
                    "type": "Vehicle",
                    "vertexLocation": i + 1
                })
            return vehicleList

    def read_in_times(self):
        if self.args.mode == "create_training_instance":
            self.start_date = self.args.instance_date
            self.end_date = self.start_date + self.args.system_time_period
        else:
            self.start_date = self.args.start_date
            self.end_date = self.args.end_date

    def read_in_taxi_trip_data(self):
        if self.args.mode == "create_training_instance":
            # get request list from full-information instances
            taxi_data = pd.DataFrame(self.full_information_data.get("requestList"))
        else:
            taxi_data = util.read_in_preprocessed_data(self.args, "/month-{:02}/trip_data_{}_{:02}_day-{:02}.csv".format(self.start_date.month, self.start_date.year, self.start_date.month, self.start_date.day - 1))
        return taxi_data
