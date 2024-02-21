from datetime import datetime, timedelta
import os
import sys
sys.path.append('./')
import numpy as np
import pandas as pd
import copy
from src.environment import Environment
import json
import scipy.stats as st
from src import config


def calculate_starting_distribution(args):
    def calculate_full_information_solutions(days, times):
        for day in days:
            for time in times:
                date = datetime(year=day.year, month=day.month, day=day.day, hour=time)
                start_date = date - timedelta(minutes=45)
                end_date = date + timedelta(minutes=45)
                os.system(f"python create_full_information_solution.py "
                          f"--start_date='{start_date}' "
                          f"--end_date='{end_date}' "
                          f"--num_vehicles='{args.num_vehicles}' "
                          f"--sparsity_factor='{args.sparsity_factor}' "
                          f"--objective='{args.objective}' "
                          f"--sample_starting_vehicle_activities=0 "
                          f"--full_information_solution_directory={args.vehDistribution_directory}")

    def make_pdf(dist, params, size=10000):
        # Separate parts of parameters
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]

        # Get sane start and end points of distribution
        start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
        end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

        # Build PDF and turn into pandas Series
        x = np.linspace(start, end, size)
        y = dist.pdf(x, loc=loc, scale=scale, *arg)
        pdf = pd.Series(y, x)

        return pdf

    def calculate_distribution(args, days, times):
        distribution = st.truncnorm
        for time in times:
            bernoulli_var = []
            exponent_var = []
            for day in days:
                args = copy.deepcopy(args)
                args.full_information_solution_directory = args.vehDistribution_directory
                args.instance_date = datetime(year=day.year, month=day.month, day=day.day, hour=time)
                args.start_date = args.instance_date - timedelta(minutes=45)
                args.end_date = args.instance_date + timedelta(minutes=45)
                args.mode = "create_training_instance"
                environment = Environment(args, np.random.RandomState(seed=123))
                # read in the data from the problem instance
                vehicleStatus = environment.vehicleList
                for vehicle in vehicleStatus:
                    if vehicle.get("on_trip_till") < args.instance_date:
                        bernoulli_var.append(0)
                    else:
                        bernoulli_var.append(1)
                        exponent_var.append((vehicle["on_trip_till"] - args.instance_date).total_seconds())

            num_on_tour = bernoulli_var.count(1)
            p_Bernoulli = num_on_tour / len(bernoulli_var)

            # calculate parameters for on_trip_till distribution
            data = exponent_var
            data = [x for x in data if 0 <= x <= 3000]
            data = np.asarray(data)
            best_fit_params = distribution.fit(data, f0=0, f1=3000)
            print("On average, there are {}/{} vehicles on trip, with beta parameters: {}".format(num_on_tour, len(bernoulli_var), best_fit_params))
            distribution_dict = {"distribution": distribution.name, "parameters": best_fit_params, "bernoulli_param": p_Bernoulli}

            # write output
            instance_output_file = open(args.vehDistribution_directory + args.vehDistribution_file, "w")
            json.dump(distribution_dict, instance_output_file)
            instance_output_file.close()

    days = [datetime(year=2015, month=1, day=7), datetime(year=2015, month=1, day=8), datetime(year=2015, month=1, day=9),
            datetime(year=2015, month=1, day=12), datetime(year=2015, month=1, day=13)]
    times = [args.start_date.hour]
    calculate_full_information_solutions(days, times)
    calculate_distribution(args, days, times)


if __name__ == '__main__':
    args = config.parser.parse_args()
    args.mode = "preprocess_vehicleDistribution"
    calculate_starting_distribution(args=args)
