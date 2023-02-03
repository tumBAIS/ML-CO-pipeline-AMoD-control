import numpy as np

from src import util
import json
from shapely.geometry import Point
import pandas as pd
import matplotlib.pyplot as plt
import os
from prep.preprocess_speed_duration_LookUp import DistanceDurationCalculator


class Grid():
    def __init__(self, args):
        self.args = args
        self.network_graph = util.get_network_graph(args)

    def calculateGridIndex(self, liste, title_entry, title_longitude, title_latitude, skip=False):
        for entry in liste:
            entry[title_entry] = self.get_idx(entry[title_longitude], entry[title_latitude], skip)
        return liste

    def get_original_idx(self, longitude, latitude):
        x = longitude
        y = latitude
        idx = int((y - self.y_min) / self.y_distance) * self.num_x + int((x - self.x_min) / self.x_distance)
        return idx

    def search_for_closest_cell(self, longitude, latitude):
        distance_saver = {}
        grid_cells = self.grid.to_dict("records")
        for cell_idx, cell in enumerate(grid_cells):
            distance_saver[cell_idx] = util.haversine((cell["x_center"], cell["y_center"]), (longitude, latitude))
        closest_cell_idx = min(distance_saver, key=distance_saver.get)
        return grid_cells[closest_cell_idx]["x_center"], grid_cells[closest_cell_idx]["y_center"]

    def get_idx(self, longitude, latitude, skip=False):
        original_idx = self.get_original_idx(longitude, latitude)
        if original_idx in self.mapping_old_idx:
            idx = self.mapping_old_idx.index(original_idx)
        else:
            if skip:
                return np.nan
            else:
                print(f"Set location lon: {longitude} | lat: {latitude} to closest graph node.")
                longitude, latitude = self.search_for_closest_cell(longitude=longitude, latitude=latitude)
                idx = self.get_idx(longitude, latitude)
        return idx

    def save_attributes_in_dict(self):
        return {"grid": self.grid, "x_min": self.x_min, "x_max": self.x_max, "y_min": self.y_min, "y_max": self.y_max, "num_x": self.num_x, "num_y": self.num_y,
                "x_distance": self.x_distance, "y_distance": self.y_distance, "x_cellLength_km": self.x_cellLength_km, "y_cellLength_km": self.y_cellLength_km,
                "name": self.name, "mapping_old_idx": self.mapping_old_idx}

    def load_attributes_from_dict(self, grid_data):
        self.grid = pd.DataFrame(grid_data["grid"])
        self.x_min = grid_data["x_min"]
        self.x_max = grid_data["x_max"]
        self.y_min = grid_data["y_min"]
        self.y_max = grid_data["y_max"]
        self.num_x = grid_data["num_x"]
        self.num_y = grid_data["num_y"]
        self.x_distance = grid_data["x_distance"]
        self.y_distance = grid_data["y_distance"]
        self.x_cellLength_km = grid_data["x_cellLength_km"]
        self.y_cellLength_km = grid_data["y_cellLength_km"]
        self.name = grid_data["name"]
        self.mapping_old_idx = grid_data["mapping_old_idx"]


    def load_grid(self, lon_min, lon_max, lat_min, lat_max, num_x, num_y, grid_name, with_sea_area=True):
        print("We load the {} grid ...".format(grid_name))
        if not os.path.exists(self.args.directory_data + "{}Grid_{}_{}".format(grid_name, num_x, num_y)):
            self.create_grid(lon_min, lon_max, lat_min, lat_max, num_y, num_x, grid_name, with_sea_area)
        self.load_attributes_from_dict(json.load(open(self.args.directory_data + "{}Grid_{}_{}".format(grid_name, num_x, num_y), "r")))


    def create_grid(self, lon_min, lon_max, lat_min, lat_max, num_y, num_x, grid_name, with_sea_area):
        self.grid = []
        # latitude / vertical / y
        # longitude / horizontal / x
        self.x_min, self.x_max, self.y_min, self.y_max, self.num_x, self.num_y = lon_min, lon_max, lat_min, lat_max, num_x, num_y
        self.x_distance = (self.x_max - self.x_min) / self.num_x
        self.y_distance = (self.y_max - self.y_min) / self.num_y
        self.y_cellLength_km = util.haversine([self.x_min, self.y_min], [self.x_min, self.y_min + self.y_distance])
        self.x_cellLength_km = util.haversine([self.x_min, self.y_min], [self.x_min + self.x_distance, self.y_min])
        self.name = grid_name

        #  y ^
        #    |
        #    |
        #    |
        #    |_____________________>
        #                      x

        y_max_cell = self.y_min
        for y_counter in range(self.num_y):
            y_min_cell = y_max_cell
            y_max_cell = y_max_cell + self.y_distance
            x_max_cell = self.x_min
            for x_counter in range(self.num_x):
                x_min_cell = x_max_cell
                x_max_cell = x_max_cell + self.x_distance
                y_center_cell = y_min_cell + ((y_max_cell - y_min_cell) / 2)
                x_center_cell = x_min_cell + ((x_max_cell - x_min_cell) / 2)
                self.grid.append({"x_min": x_min_cell, "x_max": x_max_cell, "x_center": x_center_cell,
                                  "y_min": y_min_cell, "y_max": y_max_cell, "y_center": y_center_cell, "in_NYC": False})

        print("The grid has a cell-x-length of: {} kilometers".format(round(self.x_cellLength_km, 2)))
        print("The grid has a cell-y-length of: {} kilometers".format(round(self.y_cellLength_km, 2)))

        shapefile = util.get_manhattan_shapefile(self.args)

        # Now we choose the cells which are relevant - are in Manhattan
        # To be sure that the set of relevant cells are a subset of cells we consider, we also consider cells from coastal area
        boundary = shapefile.iloc[0].geometry
        for cell_idx, cell in enumerate(self.grid):
            if cell_idx % 1000 == 0:
                print("Cell {} / {}".format(cell_idx, len(self.grid)))
            points = []
            points.append(Point(cell["x_center"], cell["y_center"]))
            if with_sea_area:
                points.append(Point(cell["x_center"] + self.x_distance, cell["y_center"]))
                points.append(Point(cell["x_center"] - self.x_distance, cell["y_center"]))
                points.append(Point(cell["x_center"], cell["y_center"] + self.y_distance))
                points.append(Point(cell["x_center"], cell["y_center"] - self.y_distance))
                points.append(Point(cell["x_min"], cell["y_max"]))
                points.append(Point(cell["x_min"], cell["y_min"]))
                points.append(Point(cell["x_max"], cell["y_max"]))
                points.append(Point(cell["x_max"], cell["y_min"]))
                points.append(Point(cell["x_min"] - self.x_distance, cell["y_max"] + self.y_distance))
                points.append(Point(cell["x_min"] - self.x_distance, cell["y_min"] - self.y_distance))
                points.append(Point(cell["x_max"] + self.x_distance, cell["y_max"] + self.y_distance))
                points.append(Point(cell["x_max"] + self.x_distance, cell["y_min"] - self.y_distance))
            for point in points:
                if point.within(boundary):
                    cell["in_NYC"] = True

        self.grid = pd.DataFrame(self.grid)
        self.grid["old_{}_index".format(grid_name)] = self.grid.index
        self.grid = self.grid[self.grid["in_NYC"]]
        self.mapping_old_idx = list(self.grid.index)
        self.grid = self.grid.to_dict('records')
        print("Remaining cells in grid: {}".format(len(self.grid)))

        grid_saver_file = open(self.args.directory_data + "{}Grid_{}_{}".format(grid_name, self.num_x, self.num_y), "w")
        json.dump(self.save_attributes_in_dict(), grid_saver_file)
        grid_saver_file.close()

        fig = plt.figure()
        shapefile.plot()
        for cell in self.grid:
            plt.scatter(cell["x_center"], cell["y_center"], s=0.2)
        plt.show()


class RebalancingGrid(Grid):
    def __init__(self, args):
        super().__init__(args)
        self.load_grid(lon_min=args.lon_lat_area[0], lon_max=args.lon_lat_area[1], lat_min=args.lon_lat_area[2], lat_max=args.lon_lat_area[3],
                       num_x=args.amount_of_cellsRebalancing[0], num_y=args.amount_of_cellsRebalancing[1], grid_name="Rebalancing", with_sea_area=False)


class LookUpGrid(Grid):
    # we use the look-up grid to store distances and durations between locations in Manhattan
    def __init__(self, args):
        super().__init__(args)
        self.load_grid(lon_min=args.lon_lat_area[0], lon_max=args.lon_lat_area[1], lat_min=args.lon_lat_area[2], lat_max=args.lon_lat_area[3],
                       num_x=args.amount_of_cellsLookUp[0], num_y=args.amount_of_cellsLookUp[1], grid_name="LookUp")
        self.table = DistanceDurationCalculator(self.args, self.grid)