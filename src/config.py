import argparse
from datetime import datetime, timedelta

parser = argparse.ArgumentParser(description="read input parameters for online k-disjoint shortest path algorithm")

parser.add_argument('--policy', type=str, help="specifies the policy", choices=['greedy', 'sampling', 'policy_SB', 'policy_CB', 'offline'], default="policy_CB")

parser.add_argument('--mode', type=str, help="specifies the running mode", default=None)

### hyperparameters algorithm

# determines the begin of the simulation
parser.add_argument('--start_date', type=lambda s: datetime.strptime(s, '%Y-%m-%d %H:%M:%S'), help="start date of simulation | Example: 2020-01-14 00:00:00 (YYYY-MM-DD hh:mm:ss)", default="2015-01-13 08:30:00")

# determines the end of the simulation
parser.add_argument('--end_date', type=lambda s: datetime.strptime(s, '%Y-%m-%d %H:%M:%S'), help="end date of simulation | Example: 2020-01-14 00:00:00 (YYYY-MM-DD hh:mm:ss)", default="2015-01-13 10:30:00")

# determine the step_size. The step_size defines size of each step size [now, now+time_step_size]
parser.add_argument('--system_time_period', type=lambda s: timedelta(minutes=int(s)), help="length of one system time period in minutes", default="1")

# determine the size of the extended horizon. The size defines the time horizon for which we receive sampled request data
parser.add_argument('--extended_horizon', type=lambda s: timedelta(minutes=int(s)), help="length of extended horizon", default="5")

# date to start training instance. Till this point in time all the trips from the instance are fixed
parser.add_argument('--instance_date', type=lambda s: datetime.strptime(s, '%Y-%m-%d %H:%M:%S'), help="start date of training instance | Example: 2020-04-01 00:00:00 (YYYY-MM-DD hh:mm:ss)", default="2015-01-13 09:36:00")

# determine the amount of vehicles in the simulation
parser.add_argument('--num_vehicles', type=int, help="number of vehicles", default=500)

# determine the amount of vehicles in the simulation
parser.add_argument('--sample_starting_vehicle_activities', type=int, help="0: No, 1: Yes", default=1)

# determines the relevant area for the simulation. We delete all vehicle trips out of this area
parser.add_argument('--lon_lat_area', type=float, nargs='*', help="longitude and latitude of relevant area: Example: [lon_min, lon_max, lat_min, lat_max]", default=[-74.24, -73.65, 40.48, 40.93])

# determine the optimization objective: Profit - we optimize the profit for the operator / amount_satisfied_customers: we optimize the amount of satisfied customers
parser.add_argument('--objective', type=str, choices=['profit', 'amount_satisfied_customers'], help="target for optimization", default='profit')

# determine the amount of vertical cells and horizontal cells for rebalancing
parser.add_argument('--amount_of_cellsRebalancing', type=int, nargs='*', help="amount of vertical and horizontal cells: [num_vert, num_horiz]", default=[150, 150])

# determine the amount of vertical cells and horizontal cells for look-up table
parser.add_argument('--amount_of_cellsLookUp', type=int, nargs='*', help="amount of vertical and horizontal cells: [num_x, num_y]", default=[250, 250])

# save parameters to sample starting vehicle activities
parser.add_argument('--vehDistribution_directory', type=str, help="directory to save the parameters for the vehicle starting distribution", default="vehDist/vehDist")

# save parameters to sample starting vehicle activities
parser.add_argument('--vehDistribution_file', type=str, default="vehDist_hour:")

# directory to store historical ride request distributions
parser.add_argument('--historicalData_directory', type=str, default="historical_data")

# heuristic distance range: maximum distance that vehicles can drive to requests in kilometers
parser.add_argument('--heuristic_distance_range', type=float, default=1.5)

# heuristic time range: maximum driving time that vehicles can need to drive to requests in seconds
parser.add_argument('--heuristic_time_range', type=float, default=720)

# driving costs for one kilometer - 0.45 US Dollar from 0.407 CHF
parser.add_argument('--costs_per_km', type=float, default=0.45)

# penalty for driven distance in the amount of satisfied customers case
parser.add_argument('--amount_satisfied_customers_penalty_per_km', type=float, default=0.00001)

# average driving speed in seconds per kilometer. 180 sec / km = 20 km / h
parser.add_argument('--average_speed_secPERkm', type=float, default=180.0)

# conversion factor to convert miles into km
parser.add_argument('--conversion_milesToKm', type=float, default=1.609344)

# number of granularity fields for the smoothing feature
parser.add_argument('--granularity_length', type=int, default=25)

# number of seconds per granularity field in the smoothing feature
parser.add_argument('--time_span_sec', type=int, default=120)

# number of capacity vertices per rabalancing cell in policy variant CB
parser.add_argument('--fix_capacity', type=int, help="fix maximum number of capacities per cell", default=1)

# source of traffic data
parser.add_argument('--traffic_speed_data', type=str, default="movement-speeds-hourly-new-york-2020-1")

# source of manhattan graph
parser.add_argument('--manhattan_graph', type=str, default="manhattan_graph.graphml")

# source of manhattan shape
parser.add_argument('--manhattan_shape', type=str, default="US_County_shape/tl_2020_us_county.shp")
# source of manhattan shape
parser.add_argument('--manhattan_shape_exact', type=str, default="US_County_shape/nybb_Correct.shp")



### hyperparameters learning

# directory to save all the full information solutions
parser.add_argument('--full_information_solution_directory', type=str, help="directory to save the full-information solutions", default="./full_information_solutions/")

# directory to save all the training instances
parser.add_argument('--training_instance_directory', type=str, help="directory to save the instances for training", default="./training_instances_std/training_instances_std")

# sparsity factor
parser.add_argument('--sparsity_factor', type=float, help="factor to sparify request density", default=0.3)

# standardization
parser.add_argument('--standardization', type=int, help="standardizes all features between 0 and x", default=100)

# list comprising the information for perturbed_optimizer:
# perturbed_optimizer[0]: indicating the n iterations. If n>0 -> perturbed_optimizer, if n=0 -> no perturbed_optimizer
# perturbed_optimizer[1]: the variance for the perturbation. If n=0 -> this value is negligible
parser.add_argument('--perturbed_optimizer', type=float, nargs='*', help="set n for perturbed optimizer; if n>0", default=[10.0, 1.0])

# set stopping criteria for learning - time
parser.add_argument('--stop_max_time', type=int, help="stopping criteria: after how many minutes the learning stops - in minutes", default=100)

# set stopping criteria for learning - accuracy
parser.add_argument('--max_iter', type=int, help="stopping criteria: stop after max_iter iterations", default=3)

# specify the directory to save the learned parameters and results from optimization
parser.add_argument('--learned_parameters_directory', type=str, help="directory to save all the learned parameters", default="./learned_parameters_std/learned_parameters")

parser.add_argument('--sanity_directory', type=str, help="directory to save the loss of sanity", default="./sanity_check/")

# set denominator to reduce weight prediction. If 0 or 1, we use the original weights.
parser.add_argument('--reduce_weight_of_future_requests_factor', type=float, help="reduces weights of future requests with this factor", default=0.2)

# define the model - GNN / Linear
parser.add_argument('--model', type=str, choices=["NN", "Linear", "GNN", "Greedy", "GNNBack", "GNNEdgespecific", "NNOuter", "NN_outer"], default="Linear")

# define the learning rate for deep learning
parser.add_argument('--learning_rate', type=float, default=0.01)

# hidden units in NN
parser.add_argument('--hidden_units', type=float, default=10)

# normalization in NN
parser.add_argument('--normalization', type=int, choices=[0, 1], default=0)

# number of layers in NN
parser.add_argument('--num_layers', type=int, default=2)



### hyperparameters read-in
# directory where you can find all the data to read-in
parser.add_argument('--directory_data', type=str, help="directory to store preprocessed files", default="./data/")

# directory where you can find all the data to build visualization
parser.add_argument('--directory_save_data_visualization', type=str, help="directory to save source data for visualization", default="./source_data_visualization/")

# when TRUE it enables to generate a gif of moving vehicles
parser.add_argument('--save_data_visualization', type=bool, help="True : save data for visualization | False : do not save data for visualization", default=False)

# directory with preprocessed ride request data
parser.add_argument('--processed_ride_data_directory', type=str, help="path to ride request data", default="taxi_data_Manhattan_2015_preprocessed")

# directory with original ride request data
parser.add_argument('--ride_data_directory', type=str, help="path to ride request data", default="taxi_data_Manhattan_2015/")

# directory to save the results
parser.add_argument('--result_directory', type=str, help="directory to save all the results", default="./results_deeplearning_std/results")

### for testing
# set iteration to read in parameters from
parser.add_argument('--read_in_iterations', type=int, help="iteration to read in data for testing", default=-1)

parser.add_argument('--rebalancing_action_sampling', type=int, choices=[0, 1], help="0: No, 1: Yes", default=0)
