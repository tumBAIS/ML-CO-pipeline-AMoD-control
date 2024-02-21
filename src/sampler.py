import pandas as pd
import numpy as np
from prep.preprocess_historical_ride_data import HistoricalData


class Sampler:
    def __init__(self, args, clock, rebalancing_grid):
        self.args = args
        self.clock = clock
        self.rebalancing_grid = rebalancing_grid
        self.historical_data = HistoricalData(self.args, self.rebalancing_grid, self.clock)


    def form_new_requests_from_locations(self, new_requests_start, new_requests_end):
        new_requests_start = new_requests_start.drop(["x_min", "x_max", "y_min", "y_max", "in_NYC"], axis=1)
        new_requests_start.rename(columns={'y_center': 'location_lat', 'x_center': 'location_lon', 'old_Rebalancing_index': "old_Rebalancing_index_pickup"}, inplace=True)
        new_requests_start.reset_index(drop=True, inplace=True)

        new_requests_end = new_requests_end.drop(["x_min", "x_max", "y_min", "y_max", "in_NYC"], axis=1)
        new_requests_end = new_requests_end.rename(columns={'y_center': 'location_lat_dropoff', 'x_center': 'location_lon_dropoff', 'old_Rebalancing_index': "old_Rebalancing_index_dropoff"})
        new_requests_end.reset_index(drop=True, inplace=True)
        return new_requests_start.merge(new_requests_end, how="inner", left_index=True, right_index=True)


    def sample_from_historical_data(self):

        # 1. Determine capacity per cell
        # 2. Get the duration per trip
        # 2.1 Get the distance per trip
        # 3. Get the payment per trip
        # 4. Get pickup time
        # 5. Get dropoff time
        #
        # the frequency array is            x_end
        #                           x_start
        # we have to divide through 52 as there are 52 weeks per year and the amount of requests are summed over the complete year.

        # get frequency array
        frequency_array_timeSlot_day = self.historical_data.load_historical_data()
        frequency_array_timeSlot_day_original = self.historical_data.load_original_historical_data()
        frequency_array_timeSlot_day_original_shape = frequency_array_timeSlot_day_original.shape
        travelTime_array_timeSlot_day = self.historical_data.load_travel_time()
        distance_km_array_timeSlot_day = self.historical_data.load_distance()
        payment_array_timeSlot_day = self.historical_data.load_payments()

        if self.args.rebalancing_action_sampling:
            # 1. Determine capacity per cell
            time_span = (self.clock.extended_horizon_end - self.clock.actual_system_time_period_end).total_seconds() / 60
            frequency_array_timeSlot_day_fraction = (time_span / 15) * (frequency_array_timeSlot_day / 52)
            capacity_on_cell = np.sum(frequency_array_timeSlot_day_fraction, axis=1)
            capacity_on_cell = [round(x) for x in capacity_on_cell.tolist()]
            occurence = []
            pickup = []
            start_date_seconds = (self.clock.actual_system_time_period_end - self.args.start_date).total_seconds()
            end_date_seconds = (self.clock.extended_horizon_end - self.args.start_date).total_seconds()
            for index, capacity in enumerate(capacity_on_cell):
                if capacity > 0:
                    occurence.extend([index] * capacity)
                    pick = np.linspace(start=start_date_seconds, stop=end_date_seconds, num=capacity + 2)[1:-1].tolist()
                    pickup.extend(pick)
            new_requests = self.rebalancing_grid.grid.iloc[occurence]
            new_requests["tpep_pickup_datetime"] = [int(x) for x in pickup]
            new_requests["tpep_pickup_datetime"] = pd.to_timedelta(new_requests["tpep_pickup_datetime"], unit="seconds") + self.args.start_date

            # 2. Get the duration per trip
            travelTime_array_timeSlot_day = np.divide(travelTime_array_timeSlot_day, frequency_array_timeSlot_day_original)
            travel_times = np.nanmean(travelTime_array_timeSlot_day, axis=1)
            travel_times = np.nan_to_num(travel_times)
            travel_times = travel_times.tolist()
            travel_times = [int(x) for x in travel_times]

            # 2.2. Get the distance per trip
            distance_km_array_timeSlot_day = np.divide(distance_km_array_timeSlot_day, frequency_array_timeSlot_day_original)
            distance_km = np.nanmean(distance_km_array_timeSlot_day, axis=1)
            distance_km = np.nan_to_num(distance_km)
            distance_km = distance_km.tolist()

            # 3. Get payment per trip
            payment_array_timeSlot_day = np.divide(payment_array_timeSlot_day, frequency_array_timeSlot_day_original)
            total_amount = np.nanmean(payment_array_timeSlot_day, axis=1)
            total_amount = np.nan_to_num(total_amount)
            if self.args.objective == "amount_satisfied_customers":
                objective_value = np.array([100] * len(capacity_on_cell))
            elif self.args.objective == "profit":
                objective_value = 100 * total_amount
            else:
                raise Exception("Wrong optimization objective defined")

            artificial_grid = pd.DataFrame({"ride_duration": travel_times, "objective_value": objective_value.tolist(), "ride_distance": distance_km, "total_amount": total_amount})
            new_requests = new_requests.merge(artificial_grid, how="left", left_index=True, right_index=True)

            # 5. Get dropoff time
            new_requests["tpep_dropoff_datetime"] = new_requests["tpep_pickup_datetime"] + pd.to_timedelta(new_requests["ride_duration"], unit="seconds")
            new_requests = new_requests.rename(columns={"x_center": "location_lon", "y_center": "location_lat"})
            new_requests = new_requests.drop(columns=["x_min", "x_max", "y_min", "y_max", "in_NYC", "old_Rebalancing_index"])
            new_requests["location_lon_dropoff"] = new_requests["location_lon"]
            new_requests["location_lat_dropoff"] = new_requests["location_lat"]
        else:
            # 1. Determine capacity per cell
            frequency_array_timeSlot_day = frequency_array_timeSlot_day.flatten()
            frequency_array_timeSlot_day_original = frequency_array_timeSlot_day_original.flatten()
            sample_size = int((self.clock.prediction_horizon_min / 15) * np.sum(frequency_array_timeSlot_day / 52))
            sampled_requests = np.random.choice(np.arange(len(frequency_array_timeSlot_day)), size=sample_size, replace=True,
                                                p=frequency_array_timeSlot_day / np.sum(frequency_array_timeSlot_day)).tolist()

            new_requests_starting, new_requests_arriving = np.unravel_index(sampled_requests, frequency_array_timeSlot_day_original_shape)

            new_requests_start = self.rebalancing_grid.grid.iloc[new_requests_starting]
            new_requests_end = self.rebalancing_grid.grid.iloc[new_requests_arriving]

            new_requests = self.form_new_requests_from_locations(new_requests_start, new_requests_end)

            # 2. Get the duration per trip
            travelTime_array_timeSlot_day = travelTime_array_timeSlot_day.flatten()
            with np.errstate(divide='ignore', invalid='ignore'):
                travelTime_array_timeSlot_day = np.divide(travelTime_array_timeSlot_day, frequency_array_timeSlot_day_original)
            travelTime_array_timeSlot_day = np.nan_to_num(travelTime_array_timeSlot_day)
            travel_times = travelTime_array_timeSlot_day[sampled_requests].astype(int)
            travel_times = pd.DataFrame({"travel_times": travel_times.tolist()})
            new_requests["ride_duration"] = travel_times["travel_times"]

            # 2. Get the distance per trip
            distance_km_array_timeSlot_day = distance_km_array_timeSlot_day.flatten()
            with np.errstate(divide='ignore', invalid='ignore'):
                distance_km_array_timeSlot_day = np.divide(distance_km_array_timeSlot_day, frequency_array_timeSlot_day_original)
            distance_km_array_timeSlot_day = np.nan_to_num(distance_km_array_timeSlot_day)
            distance_km = distance_km_array_timeSlot_day[sampled_requests]
            distance_km = pd.DataFrame({"distance_km": distance_km.tolist()})
            new_requests["ride_distance"] = distance_km["distance_km"]

            # 3. Get the payment per trip
            payment_array_timeSlot_day = payment_array_timeSlot_day.flatten()
            with np.errstate(divide='ignore', invalid='ignore'):
                payment_array_timeSlot_day = np.divide(payment_array_timeSlot_day, frequency_array_timeSlot_day_original)
            payment_array_timeSlot_day = np.nan_to_num(payment_array_timeSlot_day)
            payments = payment_array_timeSlot_day[sampled_requests]
            payments = payments.tolist()
            new_requests["total_amount"] = payments
            if self.args.objective == "amount_satisfied_customers":
                new_requests["objective_value"] = 100
            elif self.args.objective == "profit":
                new_requests["objective_value"] = 100 * new_requests["total_amount"]
            else:
                raise Exception("Wrong optimization objective defined")

            # 4. Get pickup time
            random_times = np.random.randint(0, self.clock.prediction_horizon_min * 60, size=len(sampled_requests))
            random_times = pd.DataFrame({"tpep_pickup_datetime": random_times})
            random_times["tpep_pickup_datetime"] = pd.to_timedelta(random_times["tpep_pickup_datetime"], unit="seconds") + self.clock.actual_system_time_period_end
            new_requests = new_requests.merge(random_times, how="inner", left_index=True, right_index=True)

            # 5. Get dropoff time
            new_requests["tpep_dropoff_datetime"] = new_requests["tpep_pickup_datetime"] + pd.to_timedelta(new_requests["ride_duration"], unit="seconds")

            new_requests = new_requests.sort_values(by=["tpep_dropoff_datetime"])

        return new_requests
