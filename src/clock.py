from datetime import datetime


class Clock:

    def __init__(self, args, start_date, end_date, taxi_data):
        self.actual_system_time_period_start = start_date
        self.actual_system_time_period_end = start_date
        self.extended_horizon_end = start_date
        self.end_date = end_date
        self.system_time_period = args.system_time_period
        self.extended_horizon = args.extended_horizon
        self.prediction_horizon = self.extended_horizon - self.system_time_period
        self.prediction_horizon_min = self.prediction_horizon.total_seconds() / 60
        try:
            assert self.actual_system_time_period_start + self.extended_horizon <= taxi_data["tpep_dropoff_datetime"].max()
        except:
            print("CAUTION: The final horizon time exceeds the maximum pickup time of requests -------------------------------")

    # Go to next point in time
    def tick(self, args):
        if (self.actual_system_time_period_end + self.system_time_period) <= self.end_date:
            self.actual_system_time_period_start = self.actual_system_time_period_end
            self.actual_system_time_period_end = self.actual_system_time_period_end + self.system_time_period
            self.extended_horizon_end = self.actual_system_time_period_start + self.extended_horizon
            assert self.actual_system_time_period_end <= self.end_date
            if args.mode == "create_training_instance":
                assert self.actual_system_time_period_end < self.extended_horizon_end
        else:
            print("Time is up")
            self.actual_system_time_period_end = self.end_date

    def get_weekday(self):
        return self.actual_system_time_period_end.weekday()

    def calculate_time_slot(self):
        return int(((self.actual_system_time_period_end - datetime(year=self.actual_system_time_period_end.year, month=self.actual_system_time_period_end.month, day=self.actual_system_time_period_end.day)).total_seconds() / 60) / 15)
