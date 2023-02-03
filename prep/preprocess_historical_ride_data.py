import os.path
from datetime import datetime, timedelta
import numpy as np
from src import util


class HistoricalData:
    def __init__(self, args, rebalancing_grid, clock, get_historical_data=True):
        self.args = args
        self.rebalancing_grid = rebalancing_grid
        self.clock = clock
        if get_historical_data:
            self.prepare_historical_data(util.get_original_sparsityFactor_files(self.args.historicalData_directory, args), util.get_original_sparsityFactor_files(self.args.processed_ride_data_directory, args))
            self.prepare_historical_data(self.args.historicalData_directory, self.args.processed_ride_data_directory)

    def load_historical_data(self):
        time_slot = self.clock.calculate_time_slot()
        weekday = self.clock.get_weekday()
        return np.load(self.args.directory_data + self.args.historicalData_directory + "/Frequency/frequency_timeSlot:{}_day:{}.npy".format(time_slot, weekday))

    def load_original_historical_data(self):
        time_slot = self.clock.calculate_time_slot()
        weekday = self.clock.get_weekday()
        return np.load(self.args.directory_data + util.get_original_sparsityFactor_files(self.args.historicalData_directory, self.args) + "/Frequency/frequency_timeSlot:{}_day:{}.npy".format(time_slot, weekday))

    def load_travel_time(self):
        time_slot = self.clock.calculate_time_slot()
        weekday = self.clock.get_weekday()
        return np.load(self.args.directory_data + util.get_original_sparsityFactor_files(self.args.historicalData_directory, self.args) + "/TravelTime/travelTime_timeSlot:{}_day:{}.npy".format(time_slot, weekday))

    def load_distance(self):
        time_slot = self.clock.calculate_time_slot()
        weekday = self.clock.get_weekday()
        return np.load(self.args.directory_data + util.get_original_sparsityFactor_files(self.args.historicalData_directory, self.args) + "/Distance_km/distance_km_timeSlot:{}_day:{}.npy".format(time_slot, weekday))

    def load_payments(self):
        time_slot = self.clock.calculate_time_slot()
        weekday = self.clock.get_weekday()
        return np.load(self.args.directory_data + util.get_original_sparsityFactor_files(self.args.historicalData_directory, self.args) + "/Payment/payment_timeSlot:{}_day:{}.npy".format(time_slot, weekday))

    def prepare_historical_data(self, historicalData_directory, processed_ride_data_directory):
        time_slot = self.clock.calculate_time_slot()
        weekday = self.clock.get_weekday()
        if not os.path.exists(self.args.directory_data + historicalData_directory + "/Frequency/frequency_timeSlot:{}_day:{}.npy".format(time_slot, weekday)):
            self.create_historical_data(historicalData_directory, processed_ride_data_directory)

    def create_historical_data(self, historicalData_directory, processed_ride_data_directory, time_span=15):
        dirName = self.args.directory_data + historicalData_directory
        for directoryName in [dirName, dirName + "/Frequency"] + ([dirName + "/Distance_km", dirName + "/Payment", dirName + "/TravelTime"] if "sF" not in historicalData_directory else []):
            util.create_directory(directoryName)

        frequency_array = np.zeros((96, 7, len(self.rebalancing_grid.grid), len(self.rebalancing_grid.grid)))
        if "sF" not in historicalData_directory:
            travel_time_array = np.zeros((96, 7, len(self.rebalancing_grid.grid), len(self.rebalancing_grid.grid)))
            payment_array = np.zeros((96, 7, len(self.rebalancing_grid.grid), len(self.rebalancing_grid.grid)))
            distance_array = np.zeros((96, 7, len(self.rebalancing_grid.grid), len(self.rebalancing_grid.grid)))

        def get_time_slot_and_day(filename, month_check):
            filename = filename.split("data_")[-1]
            year_in = int(filename.split("_")[0])
            month_in = int(filename.split("_")[1])
            assert month_check == month_in
            day_in = int(filename.split("day-")[-1].split(".csv")[0]) + 1

            # calculate the day of the week
            date = datetime(year=year_in, month=month_in, day=day_in)
            day_week = date.weekday()
            day = datetime(year=year_in, month=month_in, day=day_in)
            return day, day_week

        # iterate over all ride requests and fill historical map
        for month in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
            for filename in os.listdir(self.args.directory_data + processed_ride_data_directory + "/month-{:02}/".format(month)):
                print("Check file {}".format(filename))
                taxi_data = util.read_in_preprocessed_data(args=self.args, file="/month-{:02}/".format(month) + filename, processed_ride_data_directory=processed_ride_data_directory)
                day, day_week = get_time_slot_and_day(filename, month)
                time_slots = int((24 * 60) / time_span)
                for time_slot in range(time_slots):
                    start_date = timedelta(minutes=time_span * time_slot)
                    end_date = timedelta(minutes=time_span * time_slot + time_span)
                    ride_requests = taxi_data.loc[(taxi_data["tpep_pickup_datetime"] >= day + start_date) &
                                                   (taxi_data["tpep_pickup_datetime"] < day + end_date)]
                    ride_requests = ride_requests.to_dict('records')
                    for request in ride_requests:
                        begin_loc_idx = self.rebalancing_grid.get_idx(longitude=request["location_lon"], latitude=request["location_lat"])
                        end_loc_idx = self.rebalancing_grid.get_idx(longitude=request["location_lon_dropoff"], latitude=request["location_lat_dropoff"])
                        travel_time_sec = (request["tpep_dropoff_datetime"] - request["tpep_pickup_datetime"]).total_seconds()
                        frequency_array[time_slot][day_week][begin_loc_idx][end_loc_idx] += 1
                        if "sF" not in historicalData_directory:
                            travel_time_array[time_slot][day_week][begin_loc_idx][end_loc_idx] += travel_time_sec
                            payment_array[time_slot][day_week][begin_loc_idx][end_loc_idx] += request["total_amount"]
                            distance_array[time_slot][day_week][begin_loc_idx][end_loc_idx] += request["ride_distance"]

        for time_span_key in range(96):
            for day in list(range(7)):
                np.save(dirName + "/Frequency/frequency_timeSlot:{}_day:{}.npy".format(time_span_key, day), frequency_array[time_span_key][day])
                if "sF" not in historicalData_directory:
                    np.save(dirName + "/TravelTime/travelTime_timeSlot:{}_day:{}.npy".format(time_span_key, day), travel_time_array[time_span_key][day])
                    np.save(dirName + "/Payment/payment_timeSlot:{}_day:{}.npy".format(time_span_key, day), payment_array[time_span_key][day])
                    np.save(dirName + "/Distance_km/distance_km_timeSlot:{}_day:{}.npy".format(time_span_key, day), distance_array[time_span_key][day])


