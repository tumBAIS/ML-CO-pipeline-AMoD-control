from src import config, util
import pandas as pd
import os
from shapely.geometry import Point


def preprocess_nyc_data(args, months):
    shapefile = util.get_manhattan_shapefile(args, dot=".")
    boundary = shapefile.iloc[0].geometry
    util.create_directory("." + args.directory_data + args.processed_ride_data_directory)
    for month in months:
        util.create_directory("." + args.directory_data + args.processed_ride_data_directory + "/month-{:02}/".format(month))
        for filename in os.listdir("." + args.directory_data + args.ride_data_directory + "month-{:02}/".format(month)):
            taxi_data = pd.read_csv("." + args.directory_data + args.ride_data_directory + "month-{:02}/".format(month) + filename)

            # ignore unrealistic entries
            taxi_data = taxi_data.loc[args.conversion_milesToKm * 3600 * taxi_data["trip_distance"] / taxi_data["duration_sec"] < 200]  # average driving speed must be < 200 km/h
            taxi_data = taxi_data.loc[taxi_data["total_amount"] / taxi_data["duration_sec"] < 1]  # less than one € per one second of duration
            taxi_data = taxi_data.loc[taxi_data["total_amount"] / (args.conversion_milesToKm * taxi_data["trip_distance"]) < 100]  # less than 100€ for one km
            taxi_data = taxi_data.loc[taxi_data['total_amount'] > 0.0]  # prize must be positive
            taxi_data = taxi_data.loc[taxi_data['duration_sec'] < 10000]  # tour is shorther than 2.7h
            taxi_data = taxi_data.loc[taxi_data['trip_distance'] > 0]  # trip distance must be positive

            # check if entry is in Manhattan
            updated_taxi_data = []
            taxi_data = taxi_data.to_dict("records")
            for ride_request_idx, ride_request in enumerate(taxi_data):
                if ride_request_idx % 10000 == 0:
                    print("{} / {}".format(ride_request_idx, len(taxi_data)))
                if (Point(ride_request["pickup_longitude"], ride_request["pickup_latitude"]).within(boundary) and
                        Point(ride_request["dropoff_longitude"], ride_request["dropoff_latitude"]).within(boundary)):
                    updated_taxi_data.append(ride_request)

            taxi_data = pd.DataFrame(updated_taxi_data)
            taxi_data.to_csv("." + args.directory_data + args.processed_ride_data_directory + "/month-{:02}/".format(month) + filename, index=False)


if __name__ == '__main__':
    args = config.parser.parse_args()
    args.sparsity_factor = 1.0
    args = util.set_additional_args(args)
    preprocess_nyc_data(args=args, months=[4])
