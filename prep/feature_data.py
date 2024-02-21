import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn import preprocessing
from collections import OrderedDict
import os
import pickle
import json
from prep.preprocess_historical_ride_data import HistoricalData
from src import cplusplus_interface


class Feature_data():
    def __init__(self, args, rebalancing_grid=None, clock=None):
        self.args = args
        self.rebalancing_grid = rebalancing_grid
        self.dict = self.write_feature_set()
        self.clock = clock
        self.historical_data = HistoricalData(self.args, self.rebalancing_grid, self.clock, get_historical_data=False)
        if self.args.policy in ["policy_SB", "policy_CB"] and self.args.mode in ["evaluation", "sanity_check"]:
            self.standardization_values = self.load_standardization_values()

    def update_data(self, clock, vehicleList, requestList):
        self.clock = clock
        if self.args.policy in ["policy_SB", "policy_CB"]:
            self.time_span = (clock.extended_horizon - clock.system_time_period).total_seconds() / 60
            self.time_slot = int(((clock.actual_system_time_period_end - datetime(year=clock.actual_system_time_period_end.year,
                                                               month=clock.actual_system_time_period_end.month,
                                                               day=clock.actual_system_time_period_end.day)).total_seconds() / 60) / 15)
            self.request_and_vehicle_counts = self.get_request_and_vehicle_counts(pd.DataFrame(vehicleList), pd.DataFrame(requestList))
            self.get_distributions()
            self.sampling_features = self.get_sampling_features()
            self.rebalancing_features = self.get_rebalancing_features()
            with np.errstate(divide='ignore', invalid='ignore'):
                travelTime_array_timeSlot_day = np.divide(self.travelTime_array_timeSlot_day, self.frequency_array_timeSlot_day_original)
                distanceKM_array_timeSlot_day = np.divide(self.distanceKM_array_timeSlot_day, self.frequency_array_timeSlot_day_original)
                self.mean_travel_reward_value = 100 * (np.nanmean(np.divide(self.payment_array_timeSlot_day, self.frequency_array_timeSlot_day_original))) if self.args.objective == "profit" else 100
            self.mean_travel_time_value = np.nanmean(travelTime_array_timeSlot_day)
            self.mean_travel_rewardPerTime = self.args.time_span_sec * (self.mean_travel_reward_value / self.mean_travel_time_value)
            self.mean_travel_distance_value = np.nanmean(distanceKM_array_timeSlot_day)
            self.mean_travel_distancePerTime = self.args.time_span_sec * (self.mean_travel_distance_value / self.mean_travel_time_value)

    def get_distributions(self):
        self.frequency_array_timeSlot_day = self.historical_data.load_historical_data()
        self.frequency_array_timeSlot_day_original = self.historical_data.load_original_historical_data()
        self.travelTime_array_timeSlot_day = self.historical_data.load_travel_time()
        self.distanceKM_array_timeSlot_day = self.historical_data.load_distance()
        self.payment_array_timeSlot_day = self.historical_data.load_payments()

    def get_request_and_vehicle_counts(self, vehiclesList, requestsList):
        # create count of vehicles per cell
        count_vehicles = vehiclesList.groupby(["index_Rebalancing_Dropoff"], as_index=False).size().rename(columns={"size": "numberVehicles"})
        # create count of requests per cell
        count_requests = requestsList.groupby(["index_Rebalancing_Pickup"], as_index=False).size().rename(columns={"size": "numberRequests"})
        grid_count = pd.DataFrame(self.rebalancing_grid.grid.index, columns=["index_Rebalancing_Cell"])
        # get number vehicles per cell
        grid_count = grid_count.merge(count_vehicles, how="left", left_on="index_Rebalancing_Cell", right_on="index_Rebalancing_Dropoff").drop(columns=["index_Rebalancing_Dropoff"])
        grid_count["numberVehicles"] = grid_count["numberVehicles"].fillna(0)
        # get number requests per cell
        grid_count = grid_count.merge(count_requests, how="left", left_on="index_Rebalancing_Cell", right_on="index_Rebalancing_Pickup").drop(columns=["index_Rebalancing_Pickup"])
        grid_count["numberRequests"] = grid_count["numberRequests"].fillna(0)
        grid_count = grid_count.set_index("index_Rebalancing_Cell")
        # get ratio Requests/Vehicles
        grid_count["Requests/Vehicles"] = grid_count["numberRequests"] / grid_count["numberVehicles"]
        grid_count["Vehicles/Requests"] = grid_count["numberVehicles"] / grid_count["numberRequests"]
        # delete infinimum
        grid_count['Requests/Vehicles'].replace(np.inf, grid_count.loc[grid_count['Requests/Vehicles'] != np.inf, 'Requests/Vehicles'].max(), inplace=True)
        grid_count['Vehicles/Requests'].replace(np.inf, grid_count.loc[grid_count['Vehicles/Requests'] != np.inf, 'Vehicles/Requests'].max(), inplace=True)
        if self.args.standardization == 0:
            grid_count[["numberRequests", "numberVehicles", "Requests/Vehicles", "Vehicles/Requests"]] = \
                grid_count[["numberRequests", "numberVehicles", "Requests/Vehicles", "Vehicles/Requests"]].apply(lambda x: x / x.max())
        return grid_count

    def get_ratio_features(self, vertices, type, prefix):
        if "destination" == type:
            vertices = vertices.merge(self.request_and_vehicle_counts.add_prefix(prefix), how="left", left_on="index_Rebalancing_Pickup", right_index=True)
            return vertices
        if "origin" == type:
            vertices = vertices.merge(self.request_and_vehicle_counts.add_prefix(prefix), how="left", left_on="index_Rebalancing_Dropoff", right_index=True)
            return vertices
        if "origin" !=type and "destination" != type:
            raise Exception("Wrong type defined")

    def get_sampling_features(self):
        starting = np.sum(self.frequency_array_timeSlot_day / 52, axis=1)
        arriving = np.sum(self.frequency_array_timeSlot_day / 52, axis=0)
        sampling_features = pd.DataFrame({"starting": starting.tolist(), "arriving": arriving.tolist()})

        # normalize features for sampling features
        index_featureData = sampling_features.index
        column_names_featureData = sampling_features.columns
        featureData_Array = sampling_features.values  # returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        if self.args.standardization == 0:
            featureData_Array_scaled = min_max_scaler.fit_transform(featureData_Array)
        else:
            featureData_Array_scaled = featureData_Array
        sampling_features = pd.DataFrame(featureData_Array_scaled)
        sampling_features.columns = column_names_featureData
        sampling_features["index"] = index_featureData
        sampling_features = sampling_features.set_index("index", drop=True)
        return sampling_features

    def get_rebalancing_features(self):
        ## calculate the probable amount of vehicles that start from the probable arriving cell
        frequency_array_timeSlot_day_fraction = (self.time_span / 15) * (self.frequency_array_timeSlot_day / 52)
        # capacity_on_cell is the amount of vehicles starting in each cell
        capacity_on_starting_cell = np.sum(frequency_array_timeSlot_day_fraction, axis=1)
        # probability_of_arriving_cells is the probability for arriving in a cell when assuming to start in a origin cell
        with np.errstate(divide='ignore', invalid='ignore'):
            probability_of_arriving_cells = frequency_array_timeSlot_day_fraction / frequency_array_timeSlot_day_fraction.sum(axis=1)[:, None]
        probability_of_arriving_cells = np.nan_to_num(probability_of_arriving_cells)
        # amount of vehicles that start from the probability distributed arriving cell
        starting_from_cell = probability_of_arriving_cells * capacity_on_starting_cell
        # weighted average amount of vehicles that start from the probability distributed arriving cell
        starting_from_cell = np.sum(starting_from_cell, axis=1)

        scaler = preprocessing.MinMaxScaler()
        scaler = scaler.fit(np.array([0, 30]).reshape(-1, 1))
        if self.args.standardization == 0:
            starting_from_cell = scaler.transform(starting_from_cell.reshape(-1, 1))
        else:
            starting_from_cell = starting_from_cell.reshape(-1, 1)

        ## calculate the probable amount of vehicles that arrive at the probable arriving cell
        capacity_on_arriving_cell = np.sum(frequency_array_timeSlot_day_fraction, axis=0)
        arriving_at_cell = probability_of_arriving_cells * capacity_on_arriving_cell
        # weighted average amount of vehicles that start from the probability distributed arriving cell
        arriving_at_cell = np.sum(arriving_at_cell, axis=1)

        scaler = preprocessing.MinMaxScaler()
        scaler = scaler.fit(np.array([0, 30]).reshape(-1, 1))
        if self.args.standardization == 0:
            arriving_at_cell = scaler.transform(arriving_at_cell.reshape(-1, 1))
        else:
            arriving_at_cell = arriving_at_cell.reshape(-1, 1)
        rebalancing_grid = pd.DataFrame({"rebalancing_arriving": starting_from_cell.reshape(-1).tolist(),
                                         "rebalancing_starting": arriving_at_cell.reshape(-1).tolist()})
        return rebalancing_grid

    def get_time_features(self, grid_Pandas, on_sampled_vertices=False, prefix=""):

        # Feature 1: Duration of trip
        grid_Pandas[prefix + "duration_trip"] = grid_Pandas["ride_duration"]
        grid_Pandas[prefix + "ratio_duration_trip"] = grid_Pandas["objective_value"] / (grid_Pandas["ride_duration"] + 1)
        grid_Pandas[prefix + "cost_ratio_duration_trip"] = (grid_Pandas["ride_distance"] * 100 * self.args.costs_per_km) / (grid_Pandas["ride_duration"] + 1)

        # Feature 1.1: Distance of trip
        grid_Pandas[prefix + "distance_trip"] = grid_Pandas["ride_distance"]
        grid_Pandas[prefix + "ratio_distance_trip"] = (grid_Pandas["objective_value"] - (grid_Pandas["ride_distance"] * 100 * self.args.costs_per_km)) / (grid_Pandas["ride_distance"] + 1)

        # Feature 3. Time distance to dummy node
        if not on_sampled_vertices:
            grid_Pandas[prefix + "time_distance"] = grid_Pandas["objective_value"] / ((grid_Pandas["tpep_pickup_datetime"] - self.clock.actual_system_time_period_start).dt.total_seconds() + 1)
            grid_Pandas[prefix + "duration_dropoff"] = grid_Pandas["objective_value"] / (grid_Pandas["ride_duration"] + (grid_Pandas["tpep_pickup_datetime"] - self.clock.actual_system_time_period_start).dt.total_seconds() + 1)
            grid_Pandas[prefix + "costs_duration_dropoff"] = (grid_Pandas["ride_distance"] * 100 * self.args.costs_per_km) / (grid_Pandas["ride_duration"] + (grid_Pandas["tpep_pickup_datetime"] - self.clock.actual_system_time_period_start).dt.total_seconds() + 1)
        return grid_Pandas

    def get_features_smoothing(self, features, metric="revenue"):
        if metric == "revenue":
            factor = 1
            metric_column = "objective_value"
            mean_travel_metric_per_time = self.mean_travel_rewardPerTime
        elif metric == "cost":
            factor = -1 * 100 * self.args.costs_per_km
            metric_column = "ride_distance"
            mean_travel_metric_per_time = self.mean_travel_distancePerTime
        else:
            raise Exception("Wrong metric definded when calculating smooth features")

        granularity_counter = [self.clock.actual_system_time_period_start + timedelta(seconds=self.args.time_span_sec * i) for i in range(self.args.granularity_length)]
        # get features
        def get_smoothing_columns(row):
            column = np.array([0.0] * self.args.granularity_length)
            if metric == "cost":
                dropoff_time = row["tpep_pickup_datetime"]
                index = int((dropoff_time - self.clock.actual_system_time_period_start).total_seconds() / self.args.time_span_sec) + 1
                pickup_time = row["tpep_pickup_datetime"] - timedelta(seconds=row["duration_sec"])
                if pickup_time < self.clock.actual_system_time_period_start:
                    pickup_time = self.clock.actual_system_time_period_start
                index_begin = int((pickup_time - self.clock.actual_system_time_period_start).total_seconds() / self.args.time_span_sec) + 1
                percentage_within_index_begin = (granularity_counter[index_begin] - pickup_time).total_seconds() / self.args.time_span_sec
                if dropoff_time == pickup_time:
                    column[0] = row["distance_km"] * factor
                else:
                    total_amount_per_second = (row["distance_km"]) / (dropoff_time - pickup_time).total_seconds()
                    percentage_within_index = (dropoff_time - granularity_counter[index - 1]).total_seconds() / self.args.time_span_sec
                    if index > 1:
                        column[index_begin - 1] = total_amount_per_second * factor * percentage_within_index_begin * self.args.time_span_sec
                        column[index_begin:index - 1] = total_amount_per_second * factor * self.args.time_span_sec
                    column[index - 1] += total_amount_per_second * factor * percentage_within_index * self.args.time_span_sec
            # smooth feature during ride
            dropoff_time = row["tpep_pickup_datetime"] + timedelta(seconds=row["ride_duration"])
            index = int((dropoff_time - self.clock.actual_system_time_period_start).total_seconds() / self.args.time_span_sec) + 1
            pickup_time = row["tpep_pickup_datetime"]
            index_begin = int((pickup_time - self.clock.actual_system_time_period_start).total_seconds() / self.args.time_span_sec) + 1
            percentage_within_index_begin = (granularity_counter[index_begin] - pickup_time).total_seconds() / self.args.time_span_sec
            total_amount_per_second = (row[metric_column]) / (dropoff_time - pickup_time).total_seconds()
            if index >= self.args.granularity_length:
                index = self.args.granularity_length
                column[index_begin-1] = total_amount_per_second * factor * percentage_within_index_begin * self.args.time_span_sec
                column[index_begin:index] = total_amount_per_second * factor * self.args.time_span_sec
            else:
                percentage_within_index = (dropoff_time - granularity_counter[index-1]).total_seconds() / self.args.time_span_sec
                if index > 1:
                    column[index_begin - 1] = total_amount_per_second * factor * percentage_within_index_begin * self.args.time_span_sec
                    column[index_begin:index-1] = total_amount_per_second * factor * self.args.time_span_sec
                column[index - 1] = total_amount_per_second * factor * percentage_within_index * self.args.time_span_sec
                if row["type"] == "Location":
                    column[index - 1] = mean_travel_metric_per_time * factor * (1-percentage_within_index) + total_amount_per_second * factor * percentage_within_index * self.args.time_span_sec
                    column[index:self.args.granularity_length] = mean_travel_metric_per_time * factor
                    # smooth feature for driving to request
            return column.tolist()

        if metric == "cost":
            features["tpep_pickup_datetime_sec"] = (features["tpep_pickup_datetime"] - self.clock.actual_system_time_period_start).dt.total_seconds()
            smoothing_array = cplusplus_interface.get_cost_smooth_features(base_features=features[["ride_duration", "ride_distance", "distance_km", "duration_sec", "tpep_pickup_datetime_sec"]],
                                                                granularity_counter=[self.args.time_span_sec * i for i in range(self.args.granularity_length)],
                                                                time_span_sec=self.args.time_span_sec,
                                                                mean_travel_distancePerTime=self.mean_travel_distancePerTime,
                                                                costs_per_km=self.args.costs_per_km, type=features["type"][0])
        else:
            smoothing_array = features.apply(lambda row: get_smoothing_columns(row), axis=1)
        features[[f"smooth_{metric}_{i}" for i in list(range(1, self.args.granularity_length + 1))]] = pd.DataFrame.from_records(smoothing_array)

        return features

    def get_features_relation(self, edgesList):
        edgesList["distance_to_request"] = edgesList["distance_km"]
        edgesList["duration_to_request"] = edgesList["duration_sec"]
        edgesList["cost_ratio_driving_duration_trip"] = (edgesList["distance_km"] * 100 * self.args.costs_per_km) / edgesList["ride_duration"]
        edgesList["cost_ratio_driving_duration_dropoff"] = (edgesList["distance_km"] * 100 * self.args.costs_per_km) / ((edgesList["tpep_dropoff_datetime"] - self.clock.actual_system_time_period_start).dt.total_seconds() + 1)
        edgesList["cost_ratio_driving_distance_trip"] = (edgesList["objective_value"] - ((edgesList["distance_km"] + edgesList["ride_distance"]) * 100 * self.args.costs_per_km)) / \
                                           (edgesList["distance_km"] + edgesList["ride_distance"] + 1)
        if self.args.policy == "policy_SB":
            # add cost smoothing features
            edgesList = self.get_features_smoothing(edgesList, "cost")
        return edgesList

    def get_features_location_relation(self, edgesList):
        edgesList["distance_to_location"] = edgesList["distance_km"]
        edgesList["duration_to_location"] = edgesList["duration_sec"]
        return edgesList

    def get_relevant_feature_names(self, list_of_sets):
        return [feature_name for set in list_of_sets for feature_name in self.dict[f"learning_vector_{set}"].keys()]

    def get_feature_names(self):
        return [feature_name for set in self.dict.keys() for feature_name in self.dict[set].keys()]

    def load_standardization_values(self):
        standardization_values = open(self.args.training_instance_directory + "standardization_values_" + str(self.args.standardization), "r")
        standardization_multiplicator = json.load(standardization_values)
        standardization_values.close()
        return standardization_multiplicator

    def standardize(self, training_instance):
        def standardize_features(training_instance_features):
            relevant_features = [feature for feature in training_instance_features.columns if feature in self.standardization_values.keys()]
            training_instance_features[relevant_features] = training_instance_features[relevant_features] * pd.Series(self.standardization_values)[relevant_features]
            training_instance_features[relevant_features] = training_instance_features[relevant_features].fillna(0)
            training_instance_features[relevant_features] = training_instance_features[relevant_features].replace([np.inf, -np.inf], 0)

        standardize_features(training_instance["features_edges_vehicles_requests"])
        standardize_features(training_instance["features_edges_requests_requests"])
        standardize_features(training_instance["features_edges_vehicles_artRebVertices"])
        standardize_features(training_instance["features_edges_requests_artRebVertices"])
        if self.args.policy == "policy_CB":
            standardize_features(training_instance["features_edges_artRebVertices_artCapVertices"])

        return training_instance

    def calculate_standardization_values(self, args):
        print("Write standardization values ...")

        def append(features, features_edges):
            return pd.concat([features, features_edges[[column for column in self.get_feature_names() if column in features_edges.columns]]], axis=0)

        features = pd.DataFrame()
        list_of_instances = os.listdir(args.training_instance_directory)
        if self.args.standardization == 0:
            list_of_instances = [list_of_instances[0]]
        for instance in list_of_instances:
            if instance.find("TrainingInstance") == -1 or instance.find(args.objective) == -1:
                continue
            file_input = open(args.training_instance_directory + instance, "rb")
            train_dict = pickle.load(file_input)
            file_input.close()
            features = append(features, pd.DataFrame(train_dict["features_edges_vehicles_requests"]))
            features = append(features, pd.DataFrame(train_dict["features_edges_requests_requests"]))
            features = append(features, pd.DataFrame(train_dict["features_edges_vehicles_artRebVertices"]))
            features = append(features, pd.DataFrame(train_dict["features_edges_requests_artRebVertices"]))
            features = append(features, pd.DataFrame(train_dict["features_edges_artRebVertices_artCapVertices"]))
        dividend = features.mean()
        multiplicator = self.args.standardization / dividend
        # we replace division by 0 outcomes
        multiplicator = multiplicator.fillna(0)
        # we replace 0 outcomes
        multiplicator = {feature: (standardization_multiplicatior if standardization_multiplicatior != 0 else 1) for (feature, standardization_multiplicatior) in multiplicator.items()}
        standardization_values = open(args.training_instance_directory + "standardization_values_" + str(args.standardization), "w")
        json.dump(multiplicator, standardization_values)
        self.standardization_values = multiplicator

    def get_features_origin(self, verticesList):
        verticesList = self.get_ratio_features(vertices=verticesList, type="origin", prefix="origin_distribution_")
        verticesList = verticesList.merge(self.sampling_features.add_prefix("origin_prediction_"), how="left", left_on="index_Rebalancing_Dropoff", right_index=True)
        verticesList = verticesList.merge(self.rebalancing_features.add_prefix("origin_"), how="left", left_on="index_Rebalancing_Dropoff", right_index=True)
        return verticesList

    def get_features_destination(self, verticesList):
        verticesList = self.get_ratio_features(vertices=verticesList, type="destination", prefix="destination_distribution_")
        verticesList = verticesList.merge(self.sampling_features.add_prefix("destination_prediction_"), how="left", left_on="index_Rebalancing_Pickup", right_index=True)
        verticesList = verticesList.merge(self.rebalancing_features.add_prefix("destination_"), how="left", left_on="index_Rebalancing_Pickup", right_index=True)
        return verticesList

    def get_features_future(self, verticesList):
        verticesList = self.get_time_features(verticesList)
        if self.args.policy in ["policy_SB"]:
            verticesList = self.get_features_smoothing(verticesList)
        verticesList["objective_feature"] = verticesList["objective_value"]
        return verticesList

    def get_features_location(self, verticesList):
        verticesList = self.get_ratio_features(vertices=verticesList, type="destination", prefix="rebalancinglocation_")
        verticesList = verticesList.merge(self.sampling_features.add_prefix("rebalancinglocation_"), how="left", left_on="index_Rebalancing_Pickup", right_index=True)
        verticesList = verticesList.merge(self.rebalancing_features.add_prefix("rebalancinglocation_"), how="left", left_on="index_Rebalancing_Pickup", right_index=True)
        verticesList = self.get_time_features(verticesList, on_sampled_vertices=True, prefix="rebalancinglocation_")
        verticesList["rebalancinglocation_objective_feature"] = verticesList["objective_value"]
        return verticesList

    def get_features_capacity(self, verticesList):
        verticesList = self.get_ratio_features(vertices=verticesList, type="destination", prefix="capacity_")
        verticesList = verticesList.merge(self.sampling_features.add_prefix("capacity_"), how="left", left_on="index_Rebalancing_Pickup", right_index=True)
        verticesList = verticesList.merge(self.rebalancing_features.add_prefix("capacity_"), how="left", left_on="index_Rebalancing_Pickup", right_index=True)
        verticesList["capacity_counter"] = verticesList["counter"]
        verticesList["capacity_objective_feature"] = verticesList["objective_value"]
        verticesList = self.get_time_features(verticesList, on_sampled_vertices=True, prefix="capacity_")
        return verticesList

    def append_features(self, verticesList, feature_sets, old_columns):
        getter = {"origin": self.get_features_origin,
                  "destination": self.get_features_destination,
                  "future": self.get_features_future,
                  "location": self.get_features_location,
                  "capacity": self.get_features_capacity}

        for feature_set in feature_sets:
            verticesList = getter[feature_set](verticesList)

        verticesList.fillna(0, inplace=True)

        relevant_feature_names = self.get_relevant_feature_names(feature_sets)
        # check if assigned features coincides with features specified in feature space in util.py
        assert len(verticesList.columns) - len(old_columns) == len(relevant_feature_names)
        # bring features in correct order
        verticesList = verticesList[old_columns + relevant_feature_names]
        return verticesList.to_dict("records")

    def append_features_to_requests(self, requestsList):
        old_columns_request = list(requestsList.columns)
        requestsList = self.append_features(requestsList, ["origin", "destination", "future"], old_columns=old_columns_request)
        return requestsList

    def append_features_to_artRebVertices(self, artRebVerticesList):
        old_columns_artificialVertices = list(artRebVerticesList.columns)
        if self.args.policy == "policy_CB":
            artRebVerticesList = self.append_features(artRebVerticesList, ["location"], old_columns=old_columns_artificialVertices)
        elif self.args.policy == "policy_SB":
            artRebVerticesList = self.append_features(artRebVerticesList, ["destination", "future"], old_columns=old_columns_artificialVertices)
        return artRebVerticesList

    def append_features_to_artCapVertices(self, artCapVerticesList):
        old_columns_capacities = list(artCapVerticesList.columns)
        artCapVerticesList = self.append_features(artCapVerticesList, ["capacity"], old_columns=old_columns_capacities)
        return artCapVerticesList

    def append_features_to_vehicles(self, vehicleList):
        old_columns_vehicle = list(vehicleList.columns)
        vehicleList = self.append_features(vehicleList, ["origin"], old_columns=old_columns_vehicle)
        return vehicleList

    def get_node_features(self, vertices, feature_sets, additional_features):
        if len(vertices) == 0:
            return pd.DataFrame()
        if self.args.policy in ["policy_SB", "policy_CB"]:
            node_features = self.get(pd.DataFrame(vertices), feature_sets, additional_features)
            if len(additional_features) > 0:
                for additional_feature in additional_features:
                    node_features[additional_feature] = (node_features[additional_feature] - self.clock.actual_system_time_period_start).dt.total_seconds()
        else:
            node_features = pd.DataFrame(vertices)
        return node_features

    def get(self, vertices, feature_sets, additional_features):
        return vertices[self.get_relevant_feature_names(feature_sets) + additional_features]

    def append_features_to_edges_vehicles_requests(self, edges_vehicles_requests, vehicleList, requestList, edges_vehicles_requests_attributes):
        num_edges = len(edges_vehicles_requests)
        edges_vehicles_requests = pd.DataFrame(edges_vehicles_requests, columns=["starting", "arriving"])
        edges_vehicles_requests = pd.concat([edges_vehicles_requests, edges_vehicles_requests_attributes], axis=1)
        edges_vehicles_requests = edges_vehicles_requests.merge(requestList[["ride_duration", "type", "ride_distance", "objective_value", "tpep_pickup_datetime", "tpep_dropoff_datetime", "vertex_idx"]], left_on="arriving", right_on="vertex_idx")
        if self.args.policy in ["policy_SB", "policy_CB"]:
            edges_vehicles_requests = edges_vehicles_requests.merge(self.get(vehicleList, ["origin"], ["vertex_idx"]), left_on="starting", right_on="vertex_idx")
            edges_vehicles_requests = edges_vehicles_requests.merge(self.get(requestList, ["destination", "future"], ["vertex_idx"]), left_on="arriving", right_on="vertex_idx")
            edges_vehicles_requests = self.get_features_relation(edges_vehicles_requests)
        assert num_edges == len(edges_vehicles_requests)
        return edges_vehicles_requests, [tuple(edge) for edge in edges_vehicles_requests[["starting", "arriving"]].values.tolist()] if len(edges_vehicles_requests) > 0 else []

    def append_features_to_edges_requests_requests(self, edges_requests_requests, requestList, edges_requests_requests_attributes):
        if len(edges_requests_requests) == 0:
            return pd.DataFrame(), []
        num_edges = len(edges_requests_requests)
        edges_requests_requests = pd.DataFrame(edges_requests_requests, columns=["starting", "arriving"])
        edges_requests_requests = pd.concat([edges_requests_requests, edges_requests_requests_attributes], axis=1)
        edges_requests_requests = edges_requests_requests.merge(requestList[["ride_duration", "type", "ride_distance", "objective_value", "tpep_pickup_datetime", "tpep_dropoff_datetime", "vertex_idx"]], left_on="arriving", right_on="vertex_idx")
        if self.args.policy in ["policy_SB", "policy_CB"]:
            edges_requests_requests = edges_requests_requests.merge(self.get(requestList, ["origin"], ["vertex_idx"]), left_on="starting", right_on="vertex_idx")
            edges_requests_requests = edges_requests_requests.merge(self.get(requestList, ["destination", "future"], ["vertex_idx"]), left_on="arriving", right_on="vertex_idx")
            edges_requests_requests = self.get_features_relation(edges_requests_requests)
        assert num_edges == len(edges_requests_requests)
        return edges_requests_requests, [tuple(edge) for edge in edges_requests_requests[["starting", "arriving"]].values.tolist()]

    def append_features_to_edges_requests_artRebVertices(self, edges_requests_artRebVertices, requestList, artRebVerticesList, edges_requests_artRebVertices_attributes):
        if len(edges_requests_artRebVertices) == 0:
            return pd.DataFrame(), []
        num_edges = len(edges_requests_artRebVertices)
        edges_requests_artRebVertices = pd.DataFrame(edges_requests_artRebVertices, columns=["starting", "arriving"])
        edges_requests_artRebVertices = pd.concat([edges_requests_artRebVertices, edges_requests_artRebVertices_attributes], axis=1)
        if self.args.policy == "policy_SB":
            edges_requests_artRebVertices = edges_requests_artRebVertices.merge(artRebVerticesList[["ride_duration", "type", "ride_distance", "objective_value", "tpep_pickup_datetime", "tpep_dropoff_datetime", "vertex_idx"]], left_on="arriving", right_on="vertex_idx")
            edges_requests_artRebVertices = edges_requests_artRebVertices.merge(self.get(requestList, ["origin"], ["vertex_idx"]), left_on="starting", right_on="vertex_idx")
            edges_requests_artRebVertices = edges_requests_artRebVertices.merge(self.get(artRebVerticesList, ["destination", "future"], ["vertex_idx"]), left_on="arriving", right_on="vertex_idx")
            edges_requests_artRebVertices = self.get_features_relation(edges_requests_artRebVertices)
        elif self.args.policy == "policy_CB":
            edges_requests_artRebVertices = edges_requests_artRebVertices.merge(artRebVerticesList[["ride_duration", "ride_distance", "objective_value", "vertex_idx"]], left_on="arriving", right_on="vertex_idx")
            edges_requests_artRebVertices = edges_requests_artRebVertices.merge(self.get(artRebVerticesList, ["location"], ["vertex_idx"]), left_on="arriving", right_on="vertex_idx")
            edges_requests_artRebVertices = self.get_features_location_relation(edges_requests_artRebVertices)
        assert num_edges == len(edges_requests_artRebVertices)
        return edges_requests_artRebVertices, [tuple(edge) for edge in edges_requests_artRebVertices[["starting", "arriving"]].values.tolist()]

    def append_features_to_edges_vehicles_artRebVertices(self, edges_vehicles_artRebVertices, vehiclesList, artRebVerticesList, edges_vehicles_artRebVertices_attributes):
        if len(edges_vehicles_artRebVertices) == 0:
            return pd.DataFrame(), []
        num_edges = len(edges_vehicles_artRebVertices)
        edges_vehicles_artRebVertices = pd.DataFrame(edges_vehicles_artRebVertices, columns=["starting", "arriving"])
        edges_vehicles_artRebVertices = pd.concat([edges_vehicles_artRebVertices, edges_vehicles_artRebVertices_attributes], axis=1)
        if self.args.policy == "policy_SB":
            edges_vehicles_artRebVertices = edges_vehicles_artRebVertices.merge(artRebVerticesList[["ride_duration", "type", "ride_distance", "objective_value", "tpep_pickup_datetime", "tpep_dropoff_datetime", "vertex_idx"]], left_on="arriving", right_on="vertex_idx")
            edges_vehicles_artRebVertices = edges_vehicles_artRebVertices.merge(self.get(vehiclesList, ["origin"], ["vertex_idx"]), left_on="starting", right_on="vertex_idx")
            edges_vehicles_artRebVertices = edges_vehicles_artRebVertices.merge(self.get(artRebVerticesList, ["destination", "future"], ["vertex_idx"]), left_on="arriving", right_on="vertex_idx")
            edges_vehicles_artRebVertices = self.get_features_relation(edges_vehicles_artRebVertices)
        elif self.args.policy == "policy_CB":
            edges_vehicles_artRebVertices = edges_vehicles_artRebVertices.merge(artRebVerticesList[["ride_duration", "ride_distance", "objective_value", "vertex_idx"]], left_on="arriving", right_on="vertex_idx")
            edges_vehicles_artRebVertices = edges_vehicles_artRebVertices.merge(self.get(artRebVerticesList, ["location"], ["vertex_idx"]), left_on="arriving", right_on="vertex_idx")
            edges_vehicles_artRebVertices = self.get_features_location_relation(edges_vehicles_artRebVertices)
        assert num_edges == len(edges_vehicles_artRebVertices)
        return edges_vehicles_artRebVertices, [tuple(edge) for edge in edges_vehicles_artRebVertices[["starting", "arriving"]].values.tolist()]

    def append_features_to_edges_artRebVertices_artCapVertices(self, edges_artRebVertices_artCapVertices, artRebVerticesList, artCapVerticesList):
        if len(edges_artRebVertices_artCapVertices) == 0:
            return pd.DataFrame(), []
        num_edges = len(edges_artRebVertices_artCapVertices)
        edges_artRebVertices_artCapVertices = pd.DataFrame(edges_artRebVertices_artCapVertices, columns=["starting", "arriving"])
        if self.args.policy in ["policy_CB"]:
            edges_artRebVertices_artCapVertices = edges_artRebVertices_artCapVertices.merge(artCapVerticesList, left_on="arriving", right_on="vertex_idx")
        assert num_edges == len(edges_artRebVertices_artCapVertices)
        return edges_artRebVertices_artCapVertices, [tuple(edge) for edge in edges_artRebVertices_artCapVertices[["starting", "arriving"]].values.tolist()]

    def write_feature_set(self):
        feature_set = OrderedDict([
            ("learning_vector_origin", OrderedDict([  #8
                ### DISTRIBUTIONS
                ("origin_distribution_numberVehicles", 1),
                ("origin_distribution_numberRequests", 1),
                ("origin_distribution_Requests/Vehicles", 1),
                ("origin_distribution_Vehicles/Requests", 1),
                ### PREDICTION
                ("origin_prediction_starting", 1),
                ("origin_prediction_arriving", 1),
                ### REBALANCING
                ("origin_rebalancing_starting", 1),
                ("origin_rebalancing_arriving", 1),
            ])),
            ("learning_vector_destination", OrderedDict([  #8
                 ### DISTRIBUTIONS
                ("destination_distribution_numberVehicles", 1),
                ("destination_distribution_numberRequests", 1),
                ("destination_distribution_Requests/Vehicles", 1),
                ("destination_distribution_Vehicles/Requests", 1),
                 ### PREDICTION
                 # called mora features with starting and arriving
                ("destination_prediction_starting", 1),
                ("destination_prediction_arriving", 1),
                 ### REBALANCING
                ("destination_rebalancing_starting", 1),
                ("destination_rebalancing_arriving", 1),
            ])),
            ("learning_vector_future", OrderedDict([  #9
                ### TIME
                ("duration_trip", 1),
                ("ratio_duration_trip", 1),
                ("distance_trip", 1),
                ("ratio_distance_trip", 1),
                ("time_distance", 1),
                ("duration_dropoff", 1),
                ("cost_ratio_duration_trip", 1),
                ("costs_duration_dropoff", 1),
                 ### TOTAL AMOUNT
                ("objective_feature", 1),
            ])),
            ("learning_vector_relation", OrderedDict([  #5
                ("distance_to_request",1),
                ("duration_to_request", 1),
                ("cost_ratio_driving_duration_trip", 1),
                ("cost_ratio_driving_duration_dropoff", 1),
                ("cost_ratio_driving_distance_trip", 1),
            ])),
            ("learning_vector_location", OrderedDict([ #14
                ### DISTRIBUTIONS
                ("rebalancinglocation_numberVehicles", 1),
                ("rebalancinglocation_numberRequests", 1),
                ("rebalancinglocation_Requests/Vehicles", 1),
                ("rebalancinglocation_Vehicles/Requests", 1),
                ### PREDICTION
                # called mora features with starting and arriving
                ("rebalancinglocation_starting", 1),
                ("rebalancinglocation_arriving", 1),
                ### REBALANCING
                ("rebalancinglocation_rebalancing_starting", 1),
                ("rebalancinglocation_rebalancing_arriving", 1),
                ### TIME
                ("rebalancinglocation_duration_trip", 1),
                ("rebalancinglocation_ratio_duration_trip", 1),
                ("rebalancinglocation_distance_trip", 1),
                ("rebalancinglocation_ratio_distance_trip", 1),
                ("rebalancinglocation_cost_ratio_duration_trip", 1),
                ("rebalancinglocation_objective_feature", 1)
            ])),
            ("learning_vector_location_relation", OrderedDict([  #2
                ("distance_to_location", 1),
                ("duration_to_location", 1),
            ])),
            ("learning_vector_capacity", OrderedDict([  #15
                ### DISTRIBUTIONS
                ("capacity_numberVehicles", 1),
                ("capacity_numberRequests", 1),
                ("capacity_Requests/Vehicles", 1),
                ("capacity_Vehicles/Requests", 1),
                ### PREDICTION
                # called mora features with starting and arriving
                ("capacity_starting", 1),
                ("capacity_arriving", 1),
                ### REBALANCING
                ("capacity_rebalancing_starting", 1),
                ("capacity_rebalancing_arriving", 1),
                ### TIME
                ("capacity_duration_trip", 1),
                ("capacity_ratio_duration_trip", 1),
                ("capacity_distance_trip", 1),
                ("capacity_ratio_distance_trip", 1),
                ("capacity_cost_ratio_duration_trip", 1),
                ("capacity_objective_feature", 1),
                ### CAPACITY_COUNTER
                ("capacity_counter", 1),
            ])),
        ])
        if self.args.policy == "policy_SB":
            for idx in range(1, 26):
                ### SMOOTHING
                feature_set["learning_vector_future"][f"smooth_revenue_{idx}"] = 1
                ### SMOOTHING COSTS
                feature_set["learning_vector_relation"][f"smooth_cost_{idx}"] = 1
            del feature_set["learning_vector_location"]
            del feature_set["learning_vector_location_relation"]
            del feature_set["learning_vector_capacity"]
        return feature_set

    def get_starting_parameters(self):
        feature_set = self.write_feature_set()
        for classe in feature_set.keys():
            for feature in feature_set[classe].keys():
                feature_set[classe][feature] = 0
        return feature_set

    def get_average_historical_data(self, vertices):
        # 1. Get average trip duration
        with np.errstate(divide='ignore', invalid='ignore'):
            travel_times = np.nanmean(np.divide(self.travelTime_array_timeSlot_day, self.frequency_array_timeSlot_day_original), axis=1)
        travel_times = np.nan_to_num(travel_times)
        travel_times = travel_times.tolist()
        travel_times = [int(x) for x in travel_times]

        # 2. Get average trip distance
        with np.errstate(divide='ignore', invalid='ignore'):
            distance_km = np.nanmean(np.divide(self.distanceKM_array_timeSlot_day, self.frequency_array_timeSlot_day_original), axis=1)
        distance_km = np.nan_to_num(distance_km)
        distance_km = distance_km.tolist()

        # 3. Get payment per trip
        if self.args.objective == "amount_satisfied_customers":
            payments = np.array([100] * len(distance_km))
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                payments = np.nanmean(np.divide(self.payment_array_timeSlot_day, self.frequency_array_timeSlot_day_original), axis=1)
                payments = np.nan_to_num(payments)
            payments = 100 * payments

        artificial_grid = pd.DataFrame({"ride_duration": travel_times, "objective_value": payments.tolist(), "ride_distance": distance_km, "tpep_pickup_datetime": self.clock.extended_horizon_end})
        vertices = vertices.merge(artificial_grid, how="left", left_on="index_Rebalancing_Pickup", right_index=True)

        return vertices
