import json
import sys

import osmnx as ox
import os
import pickle
from datetime import datetime, timedelta
import time
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import collections
from src import config
from multiprocessing import Pool
import multiprocessing as mp
from src import util
from prep.preprocess_grid import LookUpGrid
from functools import partial

class DataLoader():
    def __init__(self, solution_directory, args):
        self.solution_directory = solution_directory
        self.start_time = args.start_date
        self.end_time = args.end_date
        self.time_interval = args.time_interval
        self.num_vehicles_in_simulation = args.num_vehicles_in_simulation
        self.look_up_grid = LookUpGrid(args)
        self.initialize_trip_diary()
        self.initialize_data()


    def initialize_data(self):
        self.data = {"vehicles": {}, "requests": {}}
        date = self.start_time
        for kind in ["vehicles", "requests"]:
            while date < self.end_time:
                self.data[kind][date] = []
                date = date + self.time_interval


    def get_trip_data(self, origin, destination, origin_lat, origin_lon, destination_lat, destination_lon):
        dist_km = self.look_up_grid.table.distance[origin.get("index_Look_Up_Table_Dropoff")][destination.get("index_Look_Up_Table_Pickup")]
        duration_sec = self.look_up_grid.table.duration[origin.get("index_Look_Up_Table_Dropoff")][destination.get("index_Look_Up_Table_Pickup")]
        if dist_km == 0 or duration_sec == 0:
            dist_km = util.haversine(location1=[origin_lat, origin_lon], location2=[destination_lat, destination_lon])
            duration_sec = dist_km * args.average_speed_secPERkm
            if duration_sec == 0:
                duration_sec = 1
        return dist_km, duration_sec


    def initialize_trip_diary(self):
        # vehicle is appearing (A)
        # vehicle is driving to request (D)
        # vehicle is transporting request (T)
        # vehicle is rebalancing (R)
        self.trip_diary = {vehicle_idx: [] for vehicle_idx in list(range(self.num_vehicles_in_simulation))}

    def get_shortest_path(self, start_lon, start_lat, end_lon, end_lat):
        origin_G = ox.nearest_nodes(self.look_up_grid.network_graph, start_lon, start_lat)
        dropoff_G = ox.nearest_nodes(self.look_up_grid.network_graph, end_lon, end_lat)
        try:
            distance_total = 0
            path = {"traversing_street": [], "summed_distance": []}
            route = nx.shortest_path(self.look_up_grid.network_graph, origin_G, dropoff_G)
            for node_idx in list(range(len(route)-1)):
                start_longitude = self.look_up_grid.network_graph.nodes()[route[node_idx]]["x"]
                start_latitude = self.look_up_grid.network_graph.nodes()[route[node_idx]]["y"]
                end_longitude = self.look_up_grid.network_graph.nodes()[route[node_idx+1]]["x"]
                end_latitude = self.look_up_grid.network_graph.nodes()[route[node_idx+1]]["y"]
                distance = util.haversine(location1=[start_latitude, start_longitude], location2=[end_latitude, end_longitude])
                distance_total += distance
                path["traversing_street"].append({"start_lon": start_longitude, "start_lat": start_latitude,
                                      "end_lon": end_longitude, "end_lat": end_latitude, "distance": distance})
                path["summed_distance"].append(distance_total)
            path["distance_total"] = distance_total
        except:
            print("Did not find route")
            path = 0
        return path


    def get_rebalancing_location(self, origin, destination, origin_lon, origin_lat, origin_on_trip_till, destination_lon, destination_lat, now):
        # get the ETA of the vehicle to location
        distance_km, duration_sec = self.get_trip_data(origin, destination, origin_lat, origin_lon, destination_lat, destination_lon)
        eta = origin_on_trip_till + timedelta(seconds=duration_sec)
        # Option 3: vehicle reached the location already
        if eta < now:
            return destination
        # Option 2: the vehicle not yet reached the location
        else:
            driven_time = (now - origin_on_trip_till).total_seconds()
            dist_lat = destination_lat - origin_lat
            dist_lon = destination_lon - origin_lon
            location_lat = origin_lat + (driven_time / (duration_sec)) * dist_lat
            location_lon = origin_lon + (driven_time / (duration_sec)) * dist_lon
            new_rebalancing_location = {"location_lat": location_lat,
                                        "location_lon": location_lon,
                                        "tpep_pickup_datetime": now,
                                        "index_Look_Up_Table_Pickup": self.look_up_grid.get_idx(location_lon, location_lat)}
            return new_rebalancing_location


    def fill_diary(self, trip, verticesList, vehicle_idx, name, actual_time):
        for location_idx in list(range(len(trip)-1)):
            start = verticesList[trip[location_idx]]
            end = verticesList[trip[location_idx+1]]
            if start["type"] == "Source":
                if end["type"] == "Vehicle":
                    # initialize starting position
                    if len(self.trip_diary[vehicle_idx]) == 0:
                        start_lon, start_lat, end_lon, end_lat = end["location_lon"], end["location_lat"], end["location_lon"], end["location_lat"]
                        self.trip_diary[vehicle_idx].append({"type": "A", "path": 0, "duration": 1,
                                                             "start_time": end["on_trip_till"].strftime('%Y-%m-%d %H:%M:%S'), "start_lat": start_lat, "start_lon": start_lon,
                                                             "end_time": end["on_trip_till"].strftime('%Y-%m-%d %H:%M:%S'), "end_lat": end_lat, "end_lon": end_lon})
                else:
                    raise Exception("Route defined wrongly!")
            elif start["type"] == "Vehicle":
                if end["type"] == "Request":
                    # vehicle is driving to request
                    start_lon, start_lat, end_lon, end_lat = start["location_lon"], start["location_lat"], end["location_lon"], end["location_lat"]
                    _, duration_sec = self.get_trip_data(origin=start, destination=end, origin_lat=start_lat, origin_lon=start_lon, destination_lat=end_lat, destination_lon=end_lon)
                    self.trip_diary[vehicle_idx].append({"type": "D", "path": self.get_shortest_path(start_lon=start_lon, start_lat=start_lat, end_lon=end_lon, end_lat=end_lat), "duration": duration_sec,
                                                         "start_time": (end["tpep_pickup_datetime"] - timedelta(seconds=duration_sec)).strftime('%Y-%m-%d %H:%M:%S'), "start_lat": start_lat, "start_lon": start_lon,
                                                         "end_time": end["tpep_pickup_datetime"].strftime('%Y-%m-%d %H:%M:%S'), "end_lat": end_lat, "end_lon": end_lon})
                    # vehicle is transporting request
                    start_lon, start_lat, end_lon, end_lat = end["location_lon"], end["location_lat"], end["location_lon_dropoff"], end["location_lat_dropoff"]
                    _, duration_sec = self.get_trip_data(origin=start, destination=end, origin_lat=start_lat, origin_lon=start_lon, destination_lat=end_lat, destination_lon=end_lon)
                    self.trip_diary[vehicle_idx].append({"type": "T", "path": self.get_shortest_path(start_lon=start_lon, start_lat=start_lat, end_lon=end_lon, end_lat=end_lat), "duration": duration_sec,
                                                         "start_time": end["tpep_pickup_datetime"].strftime('%Y-%m-%d %H:%M:%S'), "start_lat": start_lat, "start_lon": start_lon,
                                                         "end_time": end["tpep_dropoff_datetime"].strftime('%Y-%m-%d %H:%M:%S'), "end_lat": end_lat, "end_lon": end_lon})
                elif end["type"] == "Location":
                    # vehicle is rebalancing
                    start_lon, start_lat, end_lon, end_lat = start["location_lon"], start["location_lat"], end["location_lon"], end["location_lat"]
                    _, duration_sec = self.get_trip_data(origin=start, destination=end, origin_lat=start_lat, origin_lon=start_lon, destination_lat=end_lat, destination_lon=end_lon)
                    if name == "Offline": # in offline case it is not rebalancing but waiting - can be changed on request
                        self.trip_diary[vehicle_idx].append({"type": "I", "path": self.get_shortest_path(start_lon=start_lon, start_lat=start_lat, end_lon=start_lon, end_lat=start_lat), "duration": duration_sec,
                                                             "start_time": start["on_trip_till"].strftime('%Y-%m-%d %H:%M:%S'), "start_lat": start_lat, "start_lon": start_lon,
                                                             "end_time": start["on_trip_till"].strftime('%Y-%m-%d %H:%M:%S'), "end_lat": start_lat, "end_lon": start_lon})
                    else:
                        end = self.get_rebalancing_location(origin=start, destination=end, origin_lat=start_lat, origin_lon=start_lon, destination_lat=end_lat, destination_lon=end_lon,
                                                            origin_on_trip_till=start["on_trip_till"], now=actual_time)
                        start_lon, start_lat, end_lon, end_lat = start["location_lon"], start["location_lat"], end["location_lon"], end["location_lat"]
                        _, duration_sec = self.get_trip_data(origin=start, destination=end, origin_lat=start_lat, origin_lon=start_lon, destination_lat=end_lat, destination_lon=end_lon)
                        self.trip_diary[vehicle_idx].append({"type": "R", "path": self.get_shortest_path(start_lon=start_lon, start_lat=start_lat, end_lon=end_lon, end_lat=end_lat), "duration": duration_sec,
                                                             "start_time": start["on_trip_till"].strftime('%Y-%m-%d %H:%M:%S'), "start_lat": start_lat, "start_lon": start_lon,
                                                             "end_time": (start["on_trip_till"] + timedelta(seconds=duration_sec)).strftime('%Y-%m-%d %H:%M:%S'), "end_lat": end_lat, "end_lon": end_lon})
                    break
            elif start["type"] == "Request":
                if end["type"] == "Request":
                    # vehicle is driving to request
                    start_lon, start_lat, end_lon, end_lat = start["location_lon_dropoff"], start["location_lat_dropoff"], end["location_lon"], end["location_lat"]
                    _, duration_sec = self.get_trip_data(origin=start, destination=end, origin_lat=start_lat, origin_lon=start_lon, destination_lat=end_lat, destination_lon=end_lon)
                    self.trip_diary[vehicle_idx].append({"type": "D", "path": self.get_shortest_path(start_lon=start_lon, start_lat=start_lat, end_lon=end_lon, end_lat=end_lat), "duration": duration_sec,
                                                         "start_time": (end["tpep_pickup_datetime"] - timedelta(seconds=duration_sec)).strftime('%Y-%m-%d %H:%M:%S'), "start_lat": start_lat, "start_lon": start_lon,
                                                         "end_time": end["tpep_pickup_datetime"].strftime('%Y-%m-%d %H:%M:%S'), "end_lat": end_lat, "end_lon": end_lon})
                    # vehicle is transporting request
                    start_lon, start_lat, end_lon, end_lat = end["location_lon"], end["location_lat"], end["location_lon_dropoff"], end["location_lat_dropoff"]
                    _, duration_sec = self.get_trip_data(origin=start, destination=end, origin_lat=start_lat, origin_lon=start_lon, destination_lat=end_lat, destination_lon=end_lon)
                    self.trip_diary[vehicle_idx].append({"type": "T", "path": self.get_shortest_path(start_lon=start_lon, start_lat=start_lat, end_lon=end_lon, end_lat=end_lat), "duration": duration_sec,
                                                         "start_time": end["tpep_pickup_datetime"].strftime('%Y-%m-%d %H:%M:%S'), "start_lat": start_lat, "start_lon": start_lon,
                                                         "end_time": end["tpep_dropoff_datetime"].strftime('%Y-%m-%d %H:%M:%S'), "end_lat": end_lat, "end_lon": end_lon})
                elif end["type"] == "Location":
                    # vehicle is rebalancing
                    start_lon, start_lat, end_lon, end_lat = start["location_lon_dropoff"], start["location_lat_dropoff"], end["location_lon"], end["location_lat"]
                    _, duration_sec = self.get_trip_data(origin=start, destination=end, origin_lat=start_lat, origin_lon=start_lon, destination_lat=end_lat, destination_lon=end_lon)
                    if name == "Offline":  # in offline case it is not rebalancing but waiting - can be changed on request
                        self.trip_diary[vehicle_idx].append({"type": "I", "path": self.get_shortest_path(start_lon=start_lon, start_lat=start_lat, end_lon=start_lon, end_lat=start_lat), "duration": duration_sec,
                                                             "start_time": start["tpep_dropoff_datetime"].strftime('%Y-%m-%d %H:%M:%S'), "start_lat": start_lat, "start_lon": start_lon,
                                                             "end_time": end["tpep_dropoff_datetime"].strftime('%Y-%m-%d %H:%M:%S'), "end_lat": start_lat, "end_lon": start_lon})
                    else:
                        end = self.get_rebalancing_location(origin=start, destination=end, origin_lat=start_lat, origin_lon=start_lon, destination_lat=end_lat, destination_lon=end_lon,
                                                            origin_on_trip_till=start["tpep_dropoff_datetime"], now=actual_time)
                        start_lon, start_lat, end_lon, end_lat = start["location_lon_dropoff"], start["location_lat_dropoff"], end["location_lon"], end["location_lat"]
                        _, duration_sec = self.get_trip_data(origin=start, destination=end, origin_lat=start_lat, origin_lon=start_lon, destination_lat=end_lat, destination_lon=end_lon)
                        self.trip_diary[vehicle_idx].append({"type": "R", "path": self.get_shortest_path(start_lon=start_lon, start_lat=start_lat, end_lon=end_lon, end_lat=end_lat), "duration": duration_sec,
                                                             "start_time": start["tpep_dropoff_datetime"].strftime('%Y-%m-%d %H:%M:%S'), "start_lat": start["location_lat_dropoff"], "start_lon": start["location_lon_dropoff"],
                                                             "end_time": (start["tpep_dropoff_datetime"] + timedelta(seconds=duration_sec)).strftime('%Y-%m-%d %H:%M:%S'), "end_lat": end_lat, "end_lon": end_lon})
                elif end["type"] == "Source":
                    pass
            elif start["type"] == "Location":
                pass


    def load_simulation_data(self, dir):
        # vehicle is appearing (A)
        # vehicle is driving to request (D)
        # vehicle is transporting request (T)
        # vehicle is rebalancing (R)
        # vehicle is idling (I)
        os.makedirs(dir, exist_ok=True)
        for vehicle_diary_key, vehicle_diary in self.trip_diary.items():
            print("process vehicle diary: {}".format(vehicle_diary_key))
            vehicle_diary_iter = iter(vehicle_diary)
            activity = next(vehicle_diary_iter)
            state = {"location_lat": 0, "location_lon": 0, "type": "I", "color": colors["I"]}
            for time_point in self.data["vehicles"]:
                if time_point < activity["start_time"]:
                    state = {"location_lat": activity["start_lat"], "location_lon": activity["start_lon"], "type": "I", "color": colors["I"]}
                elif time_point < activity["end_time"]:
                    percentage_passed = (time_point - activity["start_time"]).total_seconds() / (activity["end_time"] - activity["start_time"]).total_seconds()
                    if percentage_passed > 1:
                        print("Percentage passed is higher than 1: {}".format(percentage_passed))
                    if activity["path"] == 0:
                        state = {"location_lat": 0, "location_lon": 0, "type": activity["type"], "color": colors[activity["type"]]}
                    else:
                        distance_travelled = activity["path"]["distance_total"] * percentage_passed
                        if activity["path"]["distance_total"] == 0:
                            state = {"location_lat": activity["start_lat"],
                                     "location_lon": activity["start_lon"], "type": activity["type"], "color": colors["I"]}
                        else:
                            traversing_street_idx = next(x[0] for x in enumerate(activity["path"]["summed_distance"]) if x[1] >= distance_travelled)
                            percentage_passed_snd = (distance_travelled - ([0] + activity["path"]["summed_distance"])[traversing_street_idx]) / activity["path"]["traversing_street"][traversing_street_idx]["distance"]
                            state = {"location_lat": (activity["path"]["traversing_street"][traversing_street_idx]["end_lat"] - activity["path"]["traversing_street"][traversing_street_idx]["start_lat"]) * percentage_passed_snd + activity["path"]["traversing_street"][traversing_street_idx]["start_lat"],
                                     "location_lon": (activity["path"]["traversing_street"][traversing_street_idx]["end_lon"] - activity["path"]["traversing_street"][traversing_street_idx]["start_lon"]) * percentage_passed_snd + activity["path"]["traversing_street"][traversing_street_idx]["start_lon"], "type": activity["type"], "color": colors[activity["type"]]}
                else:
                    try:
                        activity = next(vehicle_diary_iter)
                        state = {"location_lat": state["location_lat"], "location_lon": state["location_lon"], "type": "I", "color": colors["I"]}
                    except:
                        activity = vehicle_diary[-1]
                        state = {"location_lat": activity["end_lat"], "location_lon": activity["end_lon"], "type": "I", "color": colors["I"]}
                self.data["vehicles"][time_point].append(state)
        time_start_saving_files = time.time()
        for time_point in self.data["vehicles"]:
            json.dump(self.data["vehicles"][time_point], open(dir + str(time_point) + ".json", "w"))
        time_end_saving_files = time.time()
        print("File saving finished after {} sec".format(round(time_end_saving_files - time_start_saving_files, 2)))



    def load_trip_diary_data_from_disc(self, dir):
        time_start_load_saved_files = time.time()
        for vehicle_idx in list(range(self.num_vehicles_in_simulation)):
            print("Processing vehicle idx: {}".format(vehicle_idx))
            vehicle_trip_diary = json.load(open(dir + str(vehicle_idx) + ".json", "r"))
            for activity in vehicle_trip_diary:
                activity["start_time"] = datetime.strptime(activity["start_time"], '%Y-%m-%d %H:%M:%S')
                activity["end_time"] = datetime.strptime(activity["end_time"], '%Y-%m-%d %H:%M:%S')
            self.trip_diary[vehicle_idx] = vehicle_trip_diary
        time_end_load_saved_files = time.time()
        print("File loading finished after {} sec".format(round(time_end_load_saved_files - time_start_load_saved_files, 2)))

    def load_trip_diary_data(self, dir, name):
        print("Start filling diary ...")
        time_start_fill_diary = time.time()
        for solution_file in sorted(os.listdir(self.solution_directory)):
            if "Source" not in solution_file:
                continue
            print("Starting {}".format(solution_file))
            actual_time = datetime.strptime(solution_file.split("now:")[-1], '%Y-%m-%d %H:%M:%S')
            instance_input_file = open(self.solution_directory + "/" + solution_file, "rb")
            instance_dict = pickle.load(instance_input_file)
            instance_input_file.close()

            vehicleRoutes = instance_dict["vehicleRoutes"]
            verticesList = instance_dict["verticesList"]

            for vehicle_trip_idx, vehicle_trip in enumerate(vehicleRoutes[:args.num_vehicles_in_simulation]):
                self.fill_diary(vehicle_trip + [0], verticesList, vehicle_trip_idx, name, actual_time)
        time_end_fill_diary = time.time()
        print("Ended filling diary | Time needed: {}".format(round(time_end_fill_diary-time_start_fill_diary, 2)))
        time_start_saving_files = time.time()
        for vehicle_idx in self.trip_diary.keys():
            json.dump(self.trip_diary[vehicle_idx], open(dir + str(vehicle_idx) + ".json", "w"))
        time_end_saving_files = time.time()
        print("File saving finished after {} sec".format(round(time_end_saving_files - time_start_saving_files, 2)))




class simulation_Environment():

    def __init__(self):
        self.graph = util.get_network_graph(args)
        self.data = {}

    def input_data(self, solution_directories):
        for solution_directory in solution_directories:
            name = solution_directory["policy"]
            self.data[name] = self.load_simulation_data_from_disc(solution_directory["simulation_data_directory"])

    def load_simulation_data_from_disc(self, dir):
        time_start_load_saved_files = time.time()
        data = {}
        for time_point in os.listdir(dir):
            print("Processing: {}".format(time_point))
            data[datetime.strptime(time_point.split(".")[0], '%Y-%m-%d %H:%M:%S')] = json.load(open(dir + time_point, "r"))
        time_end_load_saved_files = time.time()
        print("File loading finished after {} sec".format(round(time_end_load_saved_files - time_start_load_saved_files, 2)))
        data = collections.OrderedDict(sorted(data.items()))
        return data

    def make_picture(self):
        simulations = list(self.data.keys())
        fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(15, 5)) #, figsize=(8, 15)
        for ax, simulation in zip(axes.flat, simulations):
            ax.set_title(simulation)
            ox.plot_graph(self.graph, node_size=0, bgcolor="white", edge_color="grey", show=False, close=False, ax=ax)
            ax.set_ylim([40.7, 40.82])
        plt.subplots_adjust(wspace=0, hspace=0)

        # To include legend
        ax.scatter(0, 0, s=10, c="orange", label=translate["T"])
        ax.scatter(0, 0, s=10, c="blue", label=translate["D"])
        ax.scatter(0, 0, s=10, c="green", label=translate["R"])
        ax.scatter(0, 0, s=10, c="red", label=translate["I"])
        fig.legend(loc='lower left', ncol=1, bbox_to_anchor=(0, 0.75), fontsize=10, frameon=False)
        plt.tight_layout()

        frames = []
        for t_idx, t in enumerate(list(self.data[simulations[0]].keys())[args.image_start:args.image_end]):
            print("Video processing time point: {}".format(str(t)))
            frames_t = []
            for ax, simulation in zip(axes.flat, simulations):
                data_pandas = pd.DataFrame(self.data[simulation][t])
                label = data_pandas["type"].apply(lambda x: translate[x])
                frame = ax.scatter(data_pandas["location_lon"], data_pandas["location_lat"], s=6, c=data_pandas["color"], label=list(label))
                frames_t.append(frame)
            frames.append(frames_t)

        anim = animation.ArtistAnimation(fig, frames, interval=100, blit=False, repeat=True) # fig, frames, interval=1000, blit=True, repeat_delay=1000

        anim.save('./visualization/movie_numVeh:{}_{}-{}.gif'.format(args.num_vehicles_in_simulation,
                                                       args.start_date + timedelta(seconds=args.image_start*args.time_interval.seconds),
                                                       args.start_date + timedelta(seconds=args.image_end*args.time_interval.seconds)), writer="ffmpeg")
        plt.close()

    def draw(self):
        pass


def calculate_diary_data(solution_directory):
    data_loader = DataLoader(solution_directory=solution_directory["solution_directory"], args=args)
    data_loader.load_trip_diary_data(solution_directory["trip_diary_directory"], name=solution_directory["policy"])


def calculate_simulation_data(solution_directory):
    data_loader = DataLoader(solution_directory=solution_directory["solution_directory"], args=args)

    data_loader.load_trip_diary_data_from_disc(solution_directory["trip_diary_directory"])
    data_loader.load_simulation_data(solution_directory["simulation_data_directory"])


def get_directories(args):
    solution_directory = util.get_file_data_visualization(args)
    util.create_directory(solution_directory)
    simulation_data_directory = util.get_file_data_visualization(args) + "/simulation_data/"
    util.create_directory(simulation_data_directory)
    trip_diary_directory = util.get_file_data_visualization(args) + "/trip_diary/"
    util.create_directory(trip_diary_directory)
    return {"solution_directory": solution_directory, "simulation_data_directory": simulation_data_directory, "trip_diary_directory": trip_diary_directory, "policy":args.policy}


if __name__ == '__main__':

    args = config.parser.parse_args()
    # run policy_SB
    os.chdir('../')
    os.system(f"python evaluation.py --policy=policy_SB --save_data_visualization=True --standardization=0 --read_in_iterations=15 --rebalancing_action_sampling=1 ")
    # run policy_CB
    os.system(f"python evaluation.py --policy=policy_CB --save_data_visualization=True --standardization=100 --read_in_iterations=45 --fix_capacity=4 ")
    # run sampling
    os.system(f"python evaluation.py --policy=sampling --save_data_visualization=True ")
    # run offline
    os.system(f"python create_full_information_solution.py --policy=offline --save_data_visualization=False ")
    minutes_ranges = range(0, 60, 1)
    for minutes_range in minutes_ranges:
        instance_date = args.start_date + timedelta(minutes=minutes_range)
        os.system(f"python create_training_instances.py --policy=policy_CB --save_data_visualization=True "
                  f"--instance_date='{instance_date}' ")

    args.calculate_diary = 1  # choices=[0, 1]
    args.calculate_simulation_data = 1  # choices=[0, 1]
    args.image_start = 0
    args.image_end = 359
    args.num_vehicles_in_simulation = 300
    args.time_interval = timedelta(seconds=5)

    translate = {"A": "idling", "D": "driving to request", "T": "transporting request", "R": "rebalancing", "I": "idle"}
    colors = {"A": "red", "D": "blue", "T": "orange", "R": "green", "I": "red"}
    solution_directories = []
    args.policy = "policy_CB"
    solution_directories.append(get_directories(args))
    args.policy = "policy_SB"
    solution_directories.append(get_directories(args))
    args.policy = "sampling"
    solution_directories.append(get_directories(args))
    args.policy = "offline"
    solution_directories.append(get_directories(args))

    if args.calculate_diary == 1:
        with Pool(mp.cpu_count()) as pool:
            pool.map(calculate_diary_data, solution_directories)
        #for solution_directory in solution_directories:
        #    calculate_diary_data(solution_directory)
    if args.calculate_simulation_data == 1:
        #with Pool(mp.cpu_count()) as pool:
        #    pool.map(calculate_simulation_data, solution_directories)
        for solution_directory in solution_directories:
            calculate_simulation_data(solution_directory)

    simulation = simulation_Environment()
    simulation.input_data(solution_directories)
    simulation.make_picture()