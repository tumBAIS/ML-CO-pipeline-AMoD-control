import osmnx as ox
import networkx as nx
import pandas as pd
import os
import numpy as np

class DistanceDurationCalculator():
    def __init__(self, args, look_up_grid):
        self.args = args
        if (not os.path.exists(self.args.directory_data + "LookUP_Duration_{}_{}.npy".format(self.args.amount_of_cellsLookUp[0], self.args.amount_of_cellsLookUp[1]))) or \
                (not os.path.exists(self.args.directory_data + "LookUP_Distance_{}_{}.npy".format(self.args.amount_of_cellsLookUp[0], self.args.amount_of_cellsLookUp[1]))):
            self.look_up_grid = look_up_grid
            self.manhattan_graph = self.get_manhattan_graph()
            self.speed_data = self.preprocess_speed_data()
            self.attach_speed_data_to_graph()
            self.calculate_durations_distances_LookUp()
        self.duration = np.load(self.args.directory_data + "LookUP_Duration_{}_{}.npy".format(self.args.amount_of_cellsLookUp[0], self.args.amount_of_cellsLookUp[1]))
        self.distance = np.load(self.args.directory_data + "LookUP_Distance_{}_{}.npy".format(self.args.amount_of_cellsLookUp[0], self.args.amount_of_cellsLookUp[1])) / 1000  # conversion of meters to km

    def calculate_durations_distances_LookUp(self):
        print("We start calculating duration and distance look up table ... ")
        self.durations = np.zeros((len(self.look_up_grid), len(self.look_up_grid)))
        self.distances = np.zeros((len(self.look_up_grid), len(self.look_up_grid)))
        cell_ox_ids = []
        if not os.path.isfile(self.args.directory_data + "LookUP_map_graph_idxs_{}_{}.npy".format(self.args.amount_of_cellsLookUp[0], self.args.amount_of_cellsLookUp[1])):
            for cell_idx, cell in enumerate(self.look_up_grid.to_dict("records")):
                if cell_idx % 100 == 0:
                    print("Calculate ox idx: {}/{}".format(cell_idx, len(self.look_up_grid)))
                cell_ox_ids.append(ox.distance.nearest_nodes(self.manhattan_graph, cell["x_center"], cell["y_center"]))
            np.save(self.args.directory_data + "LookUP_map_graph_idxs_{}_{}".format(self.args.amount_of_cellsLookUp[0], self.args.amount_of_cellsLookUp[1]), cell_ox_ids)
        cell_ox_ids = np.load(self.args.directory_data + "LookUP_map_graph_idxs_{}_{}.npy".format(self.args.amount_of_cellsLookUp[0], self.args.amount_of_cellsLookUp[1]))

        for starting_cell_idx, starting_cell_ox_id in enumerate(cell_ox_ids):
            print("Calculate durations and distances {}/{}".format(starting_cell_idx, len(cell_ox_ids)))
            duration_um = nx.shortest_path_length(self.manhattan_graph, source=starting_cell_ox_id, weight='um_travel_time')
            distance_um = nx.shortest_path_length(self.manhattan_graph, source=starting_cell_ox_id, weight='length')
            ending_cell_duration = []
            ending_cell_distance = []
            for ending_cell_ox_id in cell_ox_ids:
                try:
                    ending_cell_duration.append(duration_um[ending_cell_ox_id])
                    ending_cell_distance.append(distance_um[ending_cell_ox_id])
                except:
                    ending_cell_duration.append(1000000)
                    ending_cell_distance.append(100000)
            self.durations[starting_cell_idx][:] = ending_cell_duration
            self.distances[starting_cell_idx][:] = ending_cell_distance

        print("We save the LookUp files in numpy data format now ...")
        np.save(self.args.directory_data + "LookUP_Duration_{}_{}".format(self.args.amount_of_cellsLookUp[0], self.args.amount_of_cellsLookUp[1]), self.durations)
        np.save(self.args.directory_data + "LookUP_Distance_{}_{}".format(self.args.amount_of_cellsLookUp[0], self.args.amount_of_cellsLookUp[1]), self.distances)

    def attach_speed_data_to_graph(self):
        mean_speed_um = np.median([value * 1.60934 for value in self.speed_data.values()])
        for edge in self.manhattan_graph.edges:
            edge_obj = self.manhattan_graph[edge[0]][edge[1]][edge[2]]
            wayid = edge_obj['osmid']
            distance = edge_obj['length'] / 1000  # Convert from m to km
            try:
                speed = self.speed_data[wayid] * 1.60934  # Convert from mph to kph
                travel_time = distance / speed * 3600  # Convert from hours to seconds
            except:
                travel_time = distance / mean_speed_um * 3600
            self.manhattan_graph[edge[0]][edge[1]][edge[2]]['um_travel_time'] = travel_time

    def preprocess_speed_data(self):
        if os.path.isfile(self.args.directory_data + self.args.traffic_speed_data + "_preprocessed.csv"):
            um_speed_data = pd.read_csv(self.args.directory_data + self.args.traffic_speed_data + "_preprocessed.csv")
        else:
            um_speed_data = pd.read_csv(self.args.directory_data + self.args.traffic_speed_data + '.csv')
            um_speed_data = um_speed_data.rename(columns={"hour": "hour_of_day", "speed": "speed_mph_mean"}, errors="ignore")
            um_speed_data = um_speed_data[um_speed_data["hour_of_day"] == 8]
            um_speed_data = um_speed_data.groupby(['osm_way_id']).agg({'speed_mph_mean': 'mean'}).reset_index()
            um_speed_data.to_csv(self.args.directory_data + self.args.traffic_speed_data + "_preprocessed.csv", index=False)
        um_speed_data = dict([(t.osm_way_id, t.speed_mph_mean) for t in um_speed_data.itertuples()])
        return um_speed_data

    def get_manhattan_graph(self):
        if os.path.isfile(self.args.directory_data + self.args.manhattan_graph):
            graph = ox.load_graphml(self.args.directory_data + self.args.manhattan_graph)
        else:
            graph = ox.graph_from_place('Manhattan, New York, USA', network_type="drive")
            # save graph to disk
            ox.save_graphml(graph, self.args.directory_data + self.args.manhattan_graph)

        graph = nx.convert_node_labels_to_integers(graph, label_attribute='old_node_ID')
        graph = ox.add_edge_speeds(graph)
        graph = ox.speed.add_edge_travel_times(graph, precision=1)
        return graph

