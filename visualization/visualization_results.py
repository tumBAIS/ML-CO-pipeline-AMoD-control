import matplotlib.pyplot as plt
import os
import pandas as pd
from datetime import timedelta, datetime
import numpy as np
import json
from collections import OrderedDict
import seaborn as sns
import tikzplotlib
from src import util
import matplotlib.patches as patches


def get_abbreviation(x_tick):
    if x_tick == "num_vehicles":
        return util.get_abbreviation_fleetSize()
    elif x_tick == "extended_horizon":
        return util.get_abbreviation_ext_horizon()
    elif x_tick == "reduce_weight_of_future_requests_factor":
        return util.get_abbreviation_reduction_factor()
    elif x_tick == "heuristic_distance_range":
        return util.get_abb_heurDistanceRange()
    elif x_tick == "heuristic_time_range":
        return util.get_abb_heurTimeRange()
    elif x_tick == "sparsity_factor":
        return util.get_abb_sparsityFactor()
    elif x_tick == "objective":
        return util.get_abb_obj_str()
    elif x_tick == "standardization":
        return util.get_abb_standardization()
    elif x_tick == "fix_capacity":
        return util.get_abb_max_cap()
    else:
        raise Exception


def check_condition(filename, condition):
    if condition is None:
        return True
    skip = False
    for key in condition.keys():
        if key in ["result_directory", "policy"]:
            continue
        key_abb = get_abbreviation(key)
        instance_parameter = filename.split(key_abb + "-")[-1].split("_")[0].split("/")[0]
        if instance_parameter == "amount":
            instance_parameter = "amount_satisfied_customers"
        if key == "extended_horizon":
            instance_parameter = timedelta(minutes=float(instance_parameter))
        if isinstance(condition[key], list):
            if instance_parameter not in condition[key]:
                skip = True
        else:
            if instance_parameter != condition[key]:
                skip = True
    return skip


class Result_file():
    def __init__(self, filename, directory):
        self.filename = filename
        self.directory = directory
        self.get_times()


    def check_if_NoResult_file(self):
        if self.filename.find("Result_obj") == -1:
            return True
        else:
            return False

    def get_mode(self):
        if "sampling" in self.directory:
            self.mode = "sampling"
        if "greedy" in self.directory:
            self.mode = "greedy"
        if "policy_CB" in self.directory:
            self.mode = "policy_CB"
        if "policy_SB" in self.directory:
            self.mode = "policy_SB"
        if "offline" in self.directory:
            self.mode = "offline"

    def get_iteration(self):
        if self.mode in ["policy_CB", "policy_SB"]:
            self.iteration = int(self.filename.split("_it-")[-1].split("_")[0])
        else:
            self.iteration = -1

    def get_times(self):
        self.start_time = datetime.strptime(self.filename.split("_startDat-")[-1].split("_")[0], '%Y-%m-%d %H:%M:%S')
        self.end_time = datetime.strptime(self.filename.split("endDat-")[-1].split("_")[0], '%Y-%m-%d %H:%M:%S')
        self.system_time_period = float(self.filename.split("sysTimePeriod-")[-1].split("_")[0])


    def get_objective_value(self, metric):
        evaluation_data = pd.read_csv(self.directory + self.filename)
        if metric == "profit":
            self.objective_value = sum(evaluation_data["Profit"])
        elif metric == "amount_satisfied_customers":
            self.objective_value = sum(evaluation_data["Satisfied Customers"])
        elif metric == "service_ratio":
            self.objective_value = np.mean(evaluation_data["Satisfied Customers"] / evaluation_data["Amount of Customers"])
        elif metric == "km_per_request":
            self.objective_value = sum(evaluation_data["Total_distance_travelled"]) / sum(evaluation_data["Satisfied Customers"])
        elif metric == "avg_distance_trav":
            self.objective_value = sum(evaluation_data["Avg_distance_trav"])
        elif metric == "calculation_time":
            self.objective_value = np.mean(evaluation_data["Time_kDisjoint"])
        else:
            raise Exception("Wrong metric defined.")




    def get_x_tick(self, x_label):
        x_label_app = get_abbreviation(x_label)
        x_tick = (self.directory + self.filename).split(x_label_app + "-")[-1].split("_")[0].split("/")[0]
        try:
            self.x_tick = float(x_tick)
        except:
            self.x_tick = x_tick

def get_file_attributes(directory, filename, conditions, x_label, metric):
    file = Result_file(filename, directory)
    if (check_condition(directory + file.filename, conditions["condition_general"]) or file.check_if_NoResult_file()):
        return -1
    file.get_objective_value(metric=metric)
    file.get_mode()
    if file.mode == "policy_CB":
        skip = check_condition(directory + file.filename, conditions["condition_CB"])
    elif file.mode == "policy_SB":
        skip = check_condition(directory + file.filename, conditions["condition_SB"])
    elif file.mode == "sampling":
        skip = check_condition(directory + file.filename, conditions["condition_sampling"])
    elif file.mode == "offline":
        skip = check_condition(directory + file.filename, conditions["condition_offline"])
    elif file.mode == "greedy":
        skip = check_condition(directory + file.filename, conditions["condition_greedy"])
    else:
        raise Exception("There is a wrong mode in the dictionary")
    if skip:
        return -1
    file.get_iteration()
    file.get_x_tick(x_label=x_label)
    return file




def add_file_to_saver(saver, file):
    if file.iteration not in saver[file.mode][file.x_tick]:
        saver[file.mode][file.x_tick][file.iteration] = []
    saver[file.mode][file.x_tick][file.iteration].append(file.objective_value)
    return saver


def choose_best_learning_iteration(saver):
    best_iteration_saver = OrderedDict()
    for mode in saver.keys():
        best_iteration_saver[mode] = OrderedDict()
        for x_tick in saver[mode].keys():
            best_value = -1
            best_iteration = -1
            for learning_iteration in saver[mode][x_tick].keys():
                mean_value = sum(saver[mode][x_tick][learning_iteration]) / len(saver[mode][x_tick][learning_iteration])
                if mean_value > best_value:
                    best_value = mean_value
                    best_iteration = learning_iteration
            best_iteration_saver[mode][x_tick] = best_iteration
            print(f"Best learning iteration mode {mode} - x_tick {x_tick}: {best_iteration}")
    return best_iteration_saver

def mean_all_iterations(saver):
    for mode in saver.keys():
        for x_tick in saver[mode].keys():
            saver[mode][x_tick] = np.mean(list(saver[mode][x_tick].values()), axis=0)
    return saver

def fill_saver(directories, results_saver, conditions, x_label, metric):
    for directory in directories:
        for filename in os.listdir(directory):
            file = get_file_attributes(directory, filename, conditions, x_label, metric)
            if file == -1:
                continue
            else:
                results_saver = add_file_to_saver(results_saver, file)
    return results_saver


def create_result_saver(x_ticks, modes):
    results_saver = OrderedDict()
    for mode in modes:
        results_saver[mode] = OrderedDict()
        for x_tick in x_ticks:
            if isinstance(x_tick, timedelta):
                x_tick = float(x_tick.seconds / 60)
            else:
                x_tick = float(x_tick)
            results_saver[mode][x_tick] = OrderedDict()
    return results_saver


def sort_directories(saver):
    for mode in saver.keys():
        saver[mode] = OrderedDict(sorted(saver[mode].items(), key=lambda t: t[0]))
    return saver


def replace_with_mean(saver):
    for mode in saver.keys():
        for x_tick in saver[mode].keys():
            saver[mode][x_tick] = sum(saver[mode][x_tick]) / len(saver[mode][x_tick])
    return saver


def set_best_learning_iteration(saver, best_learning_iteration):
    for mode in saver.keys():
        for x_tick in saver[mode].keys():
            saver[mode][x_tick] = saver[mode][x_tick][best_learning_iteration[mode][x_tick]]
    return saver


def get_y_lim(metric, type):
    y_lim_saver = {"service_ratio": {"ratio": (-13, 15)}, "km_per_request": {"ratio": (0, 120)},  "avg_distance_trav": {"ratio": (0, 80)}}
    return y_lim_saver[metric][type]

def plot(saver, type, metric, x_label, running_type, boxplot=False):
    tikzplotlib_axis_parameters = {"samplingHyperparameterSelection": ["xmajorticks=true", "xlabel style={below=12mm}", "ymajorticks=true", "ylabel style={align=center}", "scaled y ticks=true", "xtick style={draw=none}",
                                                                       "ytick style={draw=none}"],
                                   "samplingReduceFutureSelect": ["xmajorticks=true", "xlabel style={align=center, below=7.7mm}", "ymajorticks=true", "ylabel style={align=center}", "scaled y ticks=true",
                                                                  "xtick style={draw=none}", "ytick style={draw=none}"],
                                   "hyperparameterOffline_distance": ["xmajorticks=true", "xlabel style={below=6mm}", "ymajorticks=true", "ylabel style={align=center}", "scaled y ticks=true", "xtick style={draw=none}",
                                                                       "ytick style={draw=none}"],
                                   "hyperparameterOffline_time": ["xmajorticks=true", "xlabel style={below=6mm}", "ymajorticks=true", "ylabel style={align=center}", "scaled y ticks=true", "xtick style={draw=none}",
                                                                       "ytick style={draw=none}"],
                                   "testmaxcapacity": ["xmajorticks=true", "xlabel style={align=center, below=5.8mm}", "ymajorticks=true", "ylabel style={align=center}", "scaled y ticks=true", "xtick style={draw=none}",
                                                "ytick style={draw=none}"],
                                   "hyperparameterSelection": ["xmajorticks=true", "xlabel style={below=11mm}", "ymajorticks=true", "ylabel style={align=center}", "scaled y ticks=true", "xtick style={draw=none}", "ytick style={draw=none}"],
                                   "absolute": ["xmajorticks=true", "ymajorticks=true", "ylabel style={align=center}", "scaled y ticks=true", "xtick style={draw=none}", "ytick style={draw=none}"],
                                   "ratio": ["xmajorticks=true", "ymajorticks=true", "ylabel style={align=center}", "scaled y ticks=false", "y tick label style={/pgf/number format/fixed}", "xtick style={draw=none}",
                                             "ytick style={draw=none}"]}



    patches_dict = {"vehicleTest": {"profit": {"absolute": (((1800, 31000), 400, 34000), (1700, 27000)),
                                               "ratio": (((1800, -1.5), 400, 10), (2300, 7))},
                               "amount_satisfied_customers": {"absolute": (((1800, 2.8), 400, 3.2), (1700, 2.5)),
                                                              "ratio": (((1800, -1), 400, 10), (1700, -2.5))}},
               "sparsityTest": {"profit": {"absolute": (((0.26, 31000), 0.08, 45000), (0.25, 25000)), "ratio": (((0.26, -2), 0.08, 9.5), (0.14, 8.3))}}}


    visualization = {
        "greedy": {"color": "#98C6EA", "marker": "o", "label": "greedy"},
        "sampling": {"color": "#005293", "marker": "*", "label": "sampling"},
        "offline": {"color": "#003359", "marker": "+", "label": "FI"},
        "policy_SB": {"color": "#E37222", "marker": "v", "label": "SB"},
        "policy_CB":{"color": "#A2AD00", "marker": "x", "label": "CB"}
    }
    x_label_name = {"num_vehicles": "Fleet size", "extended_horizon": "Extended horizon [hour:min:sec]",
                    "reduce_weight_of_future_requests_factor": "Factor reducing weights \\\\ to requests in prediction horizon",
                    "heuristic_distance_range": "Maximum distance to next request [km]", "heuristic_time_range": "Maximum time to next request [sec]",
                    "sparsity_factor": "Request density", "fix_capacity": "Num. of capacity vertices \\\\ in rebalancing cells"}

    def get_model(model):
        if model == "offline":
            return "FI"
        else:
            return model

    plt.rc('axes', labelsize=14)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=14)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=14)  # fontsize of the tick labels
    plt.rc('legend', fontsize=14)  # legend fontsize
    plt.rc('font', size=14)
    sns.set_style('whitegrid', {'axes.labelcolor': 'black'})  # darkgrid, white grid, dark, white and ticksplt.rc('axes', titlesize=18)     # fontsize of the axes title
    sns.color_palette('deep')
    fig, ax = plt.subplots(figsize=(6, 4.24))

    modes = list(saver.keys())
    if type == "absolute":
        denominator = {key: 1 for (key, value) in saver[modes[0]].items()}
        subtrahend = {key: 0 for (key, value) in saver[modes[0]].items()}
        label_suffix = ""
        factor = 1
        if metric == "profit":
            y_label = 'Profit [\$]'
        elif metric == "amount_satisfied_customers":
            y_label = 'Number of satisfied ride requests \\\\ in 1000'
        elif metric == "service_ratio":
            y_label = "Service rate [%]"
        elif metric == "km_per_request":
            y_label = "avg. km per request [km]"
        elif metric == "avg_distance_trav":
            y_label = "avg. km per vehicle [km]"
    elif type == "ratio":
        denominator = saver["greedy"]
        subtrahend = saver["greedy"]
        label_suffix = " / greedy"
        factor = 100
        modes.remove("greedy")
        modes.remove("offline")
        if metric == "profit":
            y_label = 'Increase of profit \\\\ compared to greedy bench. [%]'
        elif metric == "amount_satisfied_customers":
            y_label = 'Increase of num. sat. ride req. \\\\ compared to greedy bench. [%]'
        elif metric == "service_ratio":
            y_label = "Increase of num. of service rate \\\\ compared to greedy bench. [%]"
        elif metric == "km_per_request":
            y_label = "Increase of avg. km per request \\\\ compared to greedy bench. [%]"
        elif metric == "avg_distance_trav":
            y_label = "Increase of avg. km per vehicle \\\\ compared to greedy bench. [%]"
    else:
        raise Exception("Wrong type defined!")

    for mode in modes:
        if boxplot:
            if running_type in ["samplingHyperparameterSelection", "hyperparameterSelection"]:
                x_labels = [f"0{str(timedelta(seconds=minutes * 60))}" for minutes in list(saver[mode].keys())]
            else:
                x_labels = [minutes for minutes in list(saver[mode].keys())]
            ax.boxplot(saver[mode].values(), positions=list(range(len(saver[mode].values()))), patch_artist=True)
            plt.xticks(ticks=list(range(len(saver[mode].values()))), labels=x_labels, rotation=90)
            print([f"{mode} : {x_tick} :{np.sum(saver[mode][x_tick]) / len(saver[mode][x_tick])}" for x_tick in saver[mode].keys()])
        else:
            #print(mode + f":{[factor * x for x in [(saver[mode][x_tick] - subtrahend[x_tick]) / denominator[x_tick] for x_tick in saver[mode].keys()]]}")
            plt.plot(list(saver[mode].keys()), [factor * x for x in [(saver[mode][x_tick] - subtrahend[x_tick]) / denominator[x_tick] for x_tick in saver[mode].keys()]],
                     color=visualization[mode]["color"], marker=visualization[mode]["marker"], label=get_model(mode.split("_")[-1]) + label_suffix)
            #print(y_label + ": " + mode + ": " + str(np.mean([factor * x for x in [(saver[mode][x_tick] - subtrahend[x_tick]) / denominator[x_tick] for x_tick in saver[mode].keys()]])))
            if type == "ratio" and running_type in ["vehicleTest", "sparsityTest"]:
                print("Comparison : " + mode + " / greedy" + ": " + str(np.mean([factor * x for x in [(saver[mode][x_tick] - saver["greedy"][x_tick]) / saver["greedy"][x_tick] for x_tick in saver[mode].keys()]])))
                print("Comparison : " + mode + " / sampling" + ": " + str(np.mean([factor * x for x in [(saver[mode][x_tick] - saver["sampling"][x_tick]) / saver["sampling"][x_tick] for x_tick in saver[mode].keys()]])))
                print("Comparison MAX : " + mode + " / greedy" + ": " + str(np.max([factor * x for x in [(saver[mode][x_tick] - saver["greedy"][x_tick]) / saver["greedy"][x_tick] for x_tick in saver[mode].keys()]])))
                print("Comparison MAX : " + mode + " / sampling" + ": " + str(np.max([factor * x for x in [(saver[mode][x_tick] - saver["sampling"][x_tick]) / saver["sampling"][x_tick] for x_tick in saver[mode].keys()]])))
                print("Comparison MIN : " + mode + " / greedy" + ": " + str(np.min([factor * x for x in [(saver[mode][x_tick] - saver["greedy"][x_tick]) / saver["greedy"][x_tick] for x_tick in saver[mode].keys()]])))
                print("Comparison MIN : " + mode + " / sampling" + ": " + str(np.min([factor * x for x in [(saver[mode][x_tick] - saver["sampling"][x_tick]) / saver["sampling"][x_tick] for x_tick in saver[mode].keys()]])))
                x_tick_base = 2000 if running_type == "vehicleTest" else 0.3
                print("Comparison BASE : " + mode + " / greedy" + ": " + str(factor * (saver[mode][x_tick_base] - saver["greedy"][x_tick_base]) / saver["greedy"][x_tick_base]))
                print("Comparison BASE : " + mode + " / sampling" + ": " + str(factor * (saver[mode][x_tick_base] - saver["sampling"][x_tick_base]) / saver["sampling"][x_tick_base]))
                if running_type == "vehicleTest":
                    print("Comparison 500 : " + mode + " / greedy" + ": " + str(factor * (saver[mode][500] - saver["greedy"][500]) / saver["greedy"][500]))
                    print("Comparison 500 : " + mode + " / sampling" + ": " + str(factor * (saver[mode][500] - saver["sampling"][500]) / saver["sampling"][500]))
                if running_type == "sparsityTest":
                    print("Comparison 0.4 : " + mode + " / greedy" + ": " + str(factor * (saver[mode][0.4] - saver["greedy"][0.4]) / saver["greedy"][0.4]))
                    print("Comparison 0.4 : " + mode + " / sampling" + ": " + str(factor * (saver[mode][0.4] - saver["sampling"][0.4]) / saver["sampling"][0.4]))
                    print("Comparison HIGHER : " + mode + " / greedy" + ": " + str(np.mean([factor * x for x in [(saver[mode][x_tick] - saver["greedy"][x_tick]) / saver["greedy"][x_tick] for x_tick in [0.4, 0.5, 0.6, 0.7, 1.0]]])))
                    print("Comparison HIGHER : " + mode + " / sampling" + ": " + str(np.mean([factor * x for x in [(saver[mode][x_tick] - saver["sampling"][x_tick]) / saver["sampling"][x_tick] for x_tick in [0.4, 0.5, 0.6, 0.7, 1.0]]])))
    plt.xlabel(x_label_name[x_label])
    plt.ylabel(y_label)
    plt.legend()
    if running_type in patches_dict.keys():
        if metric in patches_dict[running_type].keys():
            if type in patches_dict[running_type][metric].keys():
                ((x, y), width, height), (x_text, y_text) = patches_dict[running_type][metric][type]
                rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='black', facecolor='none', linestyle="--", label='_nolegend_')
                ax.add_patch(rect)
                ax.text(x_text, y_text, "Base scenario", style='italic')
    fig.tight_layout()
    tikzplotlib.save("./visualization/evaluate_{}_{}_{}_{}.tex".format(running_type, x_label, metric, type if running_type != "hyperparameterSelection" else type + "_" + mode),
                     extra_axis_parameters=tikzplotlib_axis_parameters[running_type if running_type in tikzplotlib_axis_parameters.keys() else type])
    plt.show()




def divide_by_value(saver, metric):
    if metric == "amount_satisfied_customers":
        value = 1000
    else:
        value = 1
    for mode in saver.keys():
        for x_tick in saver[mode].keys():
            saver[mode][x_tick] = [x / value for x in saver[mode][x_tick]]
    return saver



class ArgsKeeper():
    def __init__(self):
        self.result_directory = None
        self.policy = None
        self.num_vehicles = None
        self.sparsity_factor = None
        self.objective = None
        self.extended_horizon = None
        self.fix_capacity = None
        self.standardization = None
        self.heuristic_distance_range = None
        self.heuristic_time_range = None
        self.mode = "evaluation"
        self.reduce_weight_of_future_requests_factor = None
        self.dict = {}

    def set_values(self, conditions):
        for condition in conditions:
            self.dict[condition] = conditions[condition]

    def set_value(self, key, value):
        if key == "result_directory":
            self.result_directory = value
        elif key == "policy":
            self.policy = value
        elif key == "num_vehicles":
            self.num_vehicles = value
        elif key == "sparsity_factor":
            self.sparsity_factor = value
        elif key == "objective":
            self.objective = value
        elif key == "extended_horizon":
            self.extended_horizon = value
        elif key == "fix_capacity":
            self.fix_capacity = value
        elif key == "reduce_weight_of_future_requests_factor":
            self.reduce_weight_of_future_requests_factor = value
        elif key == "heuristic_distance_range":
            self.heuristic_distance_range = value
        elif key == "heuristic_time_range":
            self.heuristic_time_range = value
        elif key == "standardization":
            self.standardization = value

    def get_args(self):
        for key in self.dict.keys():
            self.set_value(key, self.dict[key])


def get_x_label(args_keeper):
    for key in args_keeper.dict:
        if isinstance(args_keeper.dict[key], list):
            break
    return key

def get_results_directories(args_keeper, x_label):
    directories = []
    args_keeper.get_args()
    for specification in args_keeper.dict[x_label]:
        args_keeper.set_value(key=x_label, value=specification)
        directories.append(util.get_result_directory(args_keeper))
    return directories


def generate_relevant_directories(conditions, result_directory):
    modes = []
    conditions["condition_general"] = {**conditions["condition_general"], **{"result_directory": result_directory}}
    directories = []
    args_keeper = ArgsKeeper()
    args_keeper.set_values(conditions["condition_general"])
    x_label = get_x_label(args_keeper)
    # greedy
    if "condition_greedy" in conditions.keys():
        args_keeper.set_values(conditions["condition_greedy"])
        directories += get_results_directories(args_keeper, x_label)
        modes.append("greedy")
    # sampling
    if "condition_sampling" in conditions.keys():
        args_keeper.set_values(conditions["condition_sampling"])
        directories += get_results_directories(args_keeper, x_label)
        modes.append("sampling")
    # offline
    if "condition_offline" in conditions.keys():
        args_keeper.set_values(conditions["condition_offline"])
        directories += get_results_directories(args_keeper, x_label)
        modes.append("offline")
    # policy_SB
    if "condition_SB" in conditions.keys():
        args_keeper.set_values(conditions["condition_SB"])
        directories += get_results_directories(args_keeper, x_label)
        modes.append("policy_SB")
    # policy_CB
    if "condition_CB" in conditions.keys():
        args_keeper.set_values(conditions["condition_CB"])
        directories += get_results_directories(args_keeper, x_label)
        modes.append("policy_CB")
    return directories, x_label, modes


def visualize_results(conditions, running_type, metric):

    test_directories, x_label, modes = generate_relevant_directories(conditions, result_directory=f"./results/results_{running_type}_test")
    validation_directories, x_label, modes = generate_relevant_directories(conditions, result_directory=f"./results/results_{running_type}_validation")

    test_results_saver = create_result_saver(x_ticks=conditions["condition_general"][x_label], modes=modes)
    validation_results_saver = create_result_saver(x_ticks=conditions["condition_general"][x_label], modes=modes)
    validation_results_saver = fill_saver(validation_directories, validation_results_saver, conditions, x_label, metric="profit")
    best_learning_iteration = choose_best_learning_iteration(validation_results_saver)
    test_results_saver = fill_saver(test_directories, test_results_saver, conditions, x_label, metric)
    print("Delete test_results_saver and add validation_results_saver")
    test_results_saver = set_best_learning_iteration(test_results_saver, best_learning_iteration)
    test_results_saver = sort_directories(test_results_saver)
    test_results_saver = divide_by_value(test_results_saver, metric)
    results_saver = replace_with_mean(test_results_saver)

    plot(results_saver, type="absolute", metric=metric, x_label=x_label, running_type=running_type)
    plot(results_saver, type="ratio", metric=metric, x_label=x_label, running_type=running_type)

def evaluate_hyperparameters(conditions, running_type, metric):
    validation_directories, x_label, modes = generate_relevant_directories(conditions, result_directory=f"./results/results_{running_type}_validation")
    hyperparameters_results_saver = create_result_saver(x_ticks=conditions["condition_general"][x_label], modes=modes)
    hyperparameters_results_saver = fill_saver(validation_directories, hyperparameters_results_saver, conditions, x_label, metric)
    best_learning_iteration = choose_best_learning_iteration(hyperparameters_results_saver)
    hyperparameters_results_saver = set_best_learning_iteration(hyperparameters_results_saver, best_learning_iteration)
    hyperparameters_results_saver = sort_directories(hyperparameters_results_saver)
    hyperparameters_results_saver = divide_by_value(hyperparameters_results_saver, metric)

    plot(hyperparameters_results_saver, type="absolute", metric=metric, x_label=x_label, boxplot=True, running_type=running_type)


def extract_learned_parameters(conditions, running_type, metric, directory_parameters):

    if "condition_SB" in conditions.keys():
        learning_type = "SB"
    else:
        learning_type = "CB"

    translate_set_names = {"learning_vector_origin": "The following parameters describe the rebalancing cell where the vehicle starts.",
                           "learning_vector_destination": "The following parameters describe the rebalancing cell where the ride request starts.",
                           "learning_vector_future": "The following parameters describe the ride request.",
                           "learning_vector_relation": "The following parameters describe the tour from the vehicle location to the ride request location.",
                           "learning_vector_location": "The following parameters describe a rebalancing location",
                           "learning_vector_location_relation": "The following parameters describe the tour to a rebalancing cell.",
                           "learning_vector_capacity": "The following parameters describe the capacity vertices and its rebalancing cell."}

    translate_parameter_names = {
        "numberVehicles": "num. of vehicles",
        "numberRequests": "num. of requests",
        "Requests/Vehicles": "num. of requests / num. of vehicles",
        "Vehicles/Requests": "num. of vehicles / num. of requests",
        "prediction_starting": "est. num. of future starting requests",
        "prediction_arriving": "est. num. of future arriving requests",
        "location_arriving": "est. num. of future arriving requests",
        "location_starting": "est. num. of future starting requests",
        "capacity_arriving": "est. num. of future arriving requests",
        "capacity_starting": "est. num. of future starting requests",
        "rebalancing_arriving": "est. num. of future arriving vehicles",
        "rebalancing_starting": "est. num. of future starting vehicles",
        "cost_ratio_driving_duration_trip": "cost for tour / duration of tour",
        "cost_ratio_driving_distance_trip": "cost for tour / distance to request",
        "cost_ratio_duration_trip": "cost for tour / duration of tour",
        "costs_duration_dropoff": "costs for tour / time till drop-off",
        "cost_ratio_driving_duration_dropoff": "cost for tour / time till drop-off",
        "ratio_duration_trip": "reward of tour / duration of tour",
        "duration_trip": "duration of tour",
        "ratio_distance_trip": "reward of tour / distance of tour",
        "distance_trip": "distance of tour",
        "time_distance": "reward of tour / time till pick-up",
        "duration_dropoff": "reward of tour / time till drop-off",
        "objective_feature": "reward for tour",
        "distance_vehicle_request": "distance of tour",
        "duration_vehicle_request": "duration of tour",
        "distance_to_request": "distance to request",
        "duration_to_request": "duration to request",
        "distance_to_location": "distance to location",
        "duration_to_location": "duration to location",
        "capacity_counter": "idx of capacity vertex in reb. cell",
        "rebalancing": "est. num. of future arriving vehicles",
    }
    caption = {"CB": "All parameters with its respective $w$ values listed for the \gls*{abk:CB} policy when optimizing over profit; fleet size: 2000; request density: 0.3.",
               "SB": "All parameters with its respective $w$ values listed for the \gls*{abk:SB} policy when optimizing over profit; fleet size: 2000; request density: 0.3."}
    labels = {"CB": "table:parameters_CB", "SB": "table:parameters_SB"}

    def get_name(parameter_name):
        if "smooth_cost" in parameter_name:
            idx = int(parameter_name.split("_")[2])
            return "est. fut. cost at d.off loc. +[{},{})min".format(2*(idx-1), 2*idx)
        if "smooth_revenue" in parameter_name:
            idx = int(parameter_name.split("_")[2])
            return "est. fut. rew. at d.off loc. +[{},{})min".format(2*(idx-1), 2*idx)
        for key in translate_parameter_names.keys():
            if key in parameter_name:
                return translate_parameter_names[key]

    def round_number(number):
        rounded_number = round(number, 3)
        if rounded_number == 0:
            return 0.0
        else:
            return rounded_number

    validation_directories, x_label, modes = generate_relevant_directories(conditions, result_directory=f"./results_LRZ/results_{running_type}_validation")
    hyperparameters_results_saver = create_result_saver(x_ticks=conditions["condition_general"][x_label], modes=modes)
    hyperparameters_results_saver = fill_saver(validation_directories, hyperparameters_results_saver, conditions, x_label, metric)
    choose_iteration = list(choose_best_learning_iteration(hyperparameters_results_saver)[modes[0]].values())[0]

    # plot learning parameters
    learned_parameters = open(directory_parameters)
    learned_parameters = json.load(learned_parameters)

    print("\\begingroup \n\
\\setlength{\\tabcolsep}{6pt} % Default value: 6pt \n\
\\renewcommand{\\arraystretch}{0.6} % Default value: 1 \n\
\\begin{table}[h!] \n\
\\centering \n\
\\tiny \n\
\\resizebox{1\\textwidth}{!}{ \n\
\\begin{tabular}{c c c | c c c } \n\
\\toprule")
    for key in list(learned_parameters["parameter_evolution"][choose_iteration].keys()):
        for parameter in list(learned_parameters["parameter_evolution"][choose_iteration][key].keys()):
            if learned_parameters["parameter_evolution"][choose_iteration][key][parameter] == 0:
                del learned_parameters["parameter_evolution"][choose_iteration][key][parameter]
        keys_list = list(learned_parameters["parameter_evolution"][choose_iteration][key].keys())
        parameter_list = list(learned_parameters["parameter_evolution"][choose_iteration][key].values())
        length_list = len(parameter_list)
        length_column = int(length_list / 2)
        uneven = length_list % 2
        if uneven:
            length_column += 1
        print("\\multicolumn{6}{c}{" + translate_set_names[key] + "} \\\\ \n\
& parameter description & $w_k$ & & parameter description & $w_k$\\\\ \n\
\\midrule")
        for index_one in list(range(length_column)):
            index_two = index_one + length_column
            if index_two < length_list:
                try:
                    print(str(index_one) + " & " + get_name(keys_list[index_one]) + " & " + str(round_number(parameter_list[index_one])) + " & " + str(index_two) + " & " + get_name(keys_list[index_two]) + " & " + str(round_number(parameter_list[index_two])) +"\\\\")
                except:
                    print('\033[93m' + keys_list[index_one] + "|" + keys_list[index_two] + '\033[0m')
            else:
                try:
                    print(str(index_one) + " & " + get_name(keys_list[index_one]) + " & " + str(round_number(parameter_list[index_one])) + " & & & "  + "\\\\")
                except:
                    print('\033[93m' + keys_list[index_one] + '\033[0m')
        #if uneven:
        #    try:
        #        print(str(index_two + 1) + " & " + get_name(keys_list[index_two + 1]) + " & " + str(round(parameter_list[index_two + 1], 3)) + " & & & "  + "\\\\")
        #    except:
        #        print('\033[93m' + keys_list[index_two + 1] + '\033[0m')
        print("\\toprule")
    print("\end{tabular}} \n\
\\caption{" + caption[learning_type] + "} \n\
\\label{" + labels[learning_type] + "} \n\
\\end{table} \n\
\\endgroup \n\
\\clearpage")


def get_computational_time(conditions, running_type, metric):
    validation_directories, x_label, modes = generate_relevant_directories(conditions, result_directory=f"./results_LRZ/results_{running_type}_validation")
    hyperparameters_results_saver = create_result_saver(x_ticks=conditions["condition_general"][x_label], modes=modes)
    hyperparameters_results_saver = fill_saver(validation_directories, hyperparameters_results_saver, conditions, x_label, metric)
    best_learning_iteration = choose_best_learning_iteration(hyperparameters_results_saver)
    hyperparameters_results_saver = set_best_learning_iteration(hyperparameters_results_saver, best_learning_iteration)
    hyperparameters_results_saver = sort_directories(hyperparameters_results_saver)
    computational_time = divide_by_value(hyperparameters_results_saver, metric)


    for x_label in computational_time[list(computational_time.keys())[0]]:
        print("& " + str(x_label), end='')
    print("\\\\")
    print("\hline \hline")
    for mode in computational_time.keys():
        print(mode, end = '')
        for x_label in computational_time[mode]:
            print(" & " + str(np.round(np.mean(computational_time[mode][x_label]), decimals=3)), end='')
        print("\\\\")


def show_learning_evolution(parameter_directory, inner_evaluation, outer_evaluation):
    inner_evaluation = json.load(open(parameter_directory + inner_evaluation))
    outer_evaluation = json.load(open(parameter_directory + outer_evaluation))

    # get outer ticks
    def parameters_identical(parameters1, parameters2):
        identical = []
        for key in parameters1.keys():
            for feature in parameters1[key].keys():
                identical.append(parameters1[key][feature] == parameters2[key][feature])
        return all(identical)

    outer_epochs = []
    outer_evaluation["loss_evolution"] = []
    for outer_parameters in outer_evaluation["parameter_evolution"]:
        for epoch_idx, inner_parameters in enumerate(inner_evaluation["parameter_evolution"]):
            if parameters_identical(outer_parameters, inner_parameters):
                outer_epochs.append(epoch_idx)
                outer_evaluation["loss_evolution"].append(inner_evaluation["loss_evolution"][epoch_idx])


    plt.plot(inner_evaluation["loss_evolution"], color="blue")
    plt.scatter(outer_epochs, outer_evaluation["loss_evolution"], color="red")
    plt.title("Evolution of loss during training")
    plt.xlabel("Training epochs")
    plt.ylabel("Loss")
    plt.show()




if __name__ == '__main__':
    os.chdir('../')

    ### HYPERPARAMETER_SAMPLING_BENCHMARK
    """evaluate_hyperparameters(conditions={"condition_general": {"extended_horizon": [timedelta(minutes=5), timedelta(minutes=10), timedelta(minutes=15), timedelta(minutes=20), timedelta(minutes=25), timedelta(minutes=30)],
                                                               "objective": "profit", "num_vehicles": "2000", "heuristic_time_range": "720.0", "heuristic_distance_range": "1.5", "sparsity_factor": "0.3", "reduce_weight_of_future_requests_factor": "0.2"},
                                         "condition_sampling": {"policy": "sampling"}},
                                         running_type="samplingHyperparameterSelection", metric="profit")

    evaluate_hyperparameters(conditions={"condition_general": {
        "extended_horizon": timedelta(minutes=5), "reduce_weight_of_future_requests_factor": ["1.0", "0.9", "0.9", "0.8", "0.8", "0.7", "0.6", "0.5", "0.4", "0.3", "0.2", "0.1"],
        "objective": "profit", "num_vehicles": "2000", "heuristic_time_range": "720.0", "heuristic_distance_range": "1.5", "sparsity_factor": "0.3"},
                                         "condition_sampling": {"policy": "sampling"}},
                             running_type="samplingReduceFutureSelect", metric="profit")

    ### HYPERPARAMETER OFFLINE
    evaluate_hyperparameters(conditions={"condition_general": {"objective": "profit", "num_vehicles": "2000", "heuristic_time_range": "100000000.0", "heuristic_distance_range": ["0.3", "0.5", "0.7", "0.9", "1.1", "1.3", "1.5", "1.7"], "sparsity_factor": "0.3"},
        "condition_offline": {"policy": "offline"}},
        running_type="hyperparameterOffline_distance", metric="profit")

    evaluate_hyperparameters(conditions={
        "condition_general": {"objective": "profit", "num_vehicles": "2000", "heuristic_time_range": ["120.0", "180.0", "360.0", "540.0", "720.0", "900.0"], "heuristic_distance_range": "200.0",
                              "sparsity_factor": "0.3"},
        "condition_offline": {"policy": "offline"}},
                             running_type="hyperparameterOffline_time", metric="profit")

    evaluate_hyperparameters(conditions={
        "condition_general": {"objective": "profit", "num_vehicles": "2000", "heuristic_time_range": "720.0", "heuristic_distance_range": "1.5",
                              "sparsity_factor": "0.3", "extended_horizon": [timedelta(minutes=5), timedelta(minutes=10), timedelta(minutes=15), timedelta(minutes=20), timedelta(minutes=25), timedelta(minutes=30)]},
        "condition_SB": {"policy": "policy_SB", "standardization": "0"}},
        running_type="hyperparameterSelection", metric="profit")

    evaluate_hyperparameters(conditions={
        "condition_general": {"objective": "profit", "num_vehicles": "2000", "heuristic_time_range": "720.0", "heuristic_distance_range": "1.5",
                              "sparsity_factor": "0.3", "extended_horizon": [timedelta(minutes=5), timedelta(minutes=10), timedelta(minutes=15), timedelta(minutes=20), timedelta(minutes=25), timedelta(minutes=30)]},
        "condition_CB": {"policy": "policy_CB", "standardization": "100", "fix_capacity": "1"}},
        running_type="hyperparameterSelection", metric="profit")


    evaluate_hyperparameters(conditions={
        "condition_general": {"objective": "profit", "num_vehicles": "2000", "heuristic_time_range": "720.0", "heuristic_distance_range": "1.5", "fix_capacity": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
                              "sparsity_factor": "0.3"},
        "condition_CB": {"policy": "policy_CB", "extended_horizon": timedelta(minutes=5), "standardization": "100"}},
                             running_type="testmaxcapacity", metric="profit")"""


    """get_computational_time(conditions={"condition_general": {"objective": "profit", "num_vehicles": ["500", "1000", "2000", "3000", "4000", "5000"], "heuristic_time_range": "720.0", "heuristic_distance_range": "1.5", "sparsity_factor": "0.3"},
                                  "condition_greedy": {"policy": "greedy"},
                                  "condition_sampling": {"policy": "sampling", "extended_horizon": timedelta(minutes=5), "reduce_weight_of_future_requests_factor": "0.95"},
                                  "condition_offline": {"policy": "offline"},
                                  "condition_SB": {"policy": "policy_SB", "extended_horizon": timedelta(minutes=5), "standardization": "0"},
                                  "condition_CB": {"policy": "policy_CB", "extended_horizon": timedelta(minutes=5), "standardization": "100", "fix_capacity": "1"}}, running_type="vehicleTest", metric="calculation_time")"""


    # EVALUATE DIFFERENT VEHICLE FLEET SIZES
    visualize_results(conditions={"condition_general": {"objective": "profit", "num_vehicles": ["500", "1000", "2000", "3000", "4000", "5000"], "heuristic_time_range": "720.0", "heuristic_distance_range": "1.5", "sparsity_factor": "0.3"},
                                  "condition_greedy": {"policy": "greedy"},
                                  "condition_sampling": {"policy": "sampling", "extended_horizon": timedelta(minutes=5), "reduce_weight_of_future_requests_factor": "0.2"},
                                  "condition_offline": {"policy": "offline"},
                                  "condition_SB": {"policy": "policy_SB", "extended_horizon": timedelta(minutes=5), "standardization": "0"},
                                  "condition_CB": {"policy": "policy_CB", "extended_horizon": timedelta(minutes=5), "standardization": "100", "fix_capacity": "1"}}, running_type="vehicleTest", metric="profit")

    """visualize_results(conditions={"condition_general": {"objective": "profit", "num_vehicles": "2000", "heuristic_time_range": "720.0", "heuristic_distance_range": "1.5", "sparsity_factor": ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "1.0"]},
                                      "condition_greedy": {"policy": "greedy"},
                                      "condition_sampling": {"policy": "sampling", "extended_horizon": timedelta(minutes=5), "reduce_weight_of_future_requests_factor": "0.2"},
                                      "condition_offline": {"policy": "offline"},
                                      "condition_SB": {"policy": "policy_SB", "extended_horizon": timedelta(minutes=5), "standardization": "0"},
                                      "condition_CB": {"policy": "policy_CB", "extended_horizon": timedelta(minutes=5), "standardization": "100", "fix_capacity": "1"}}, running_type="sparsityTest", metric="profit")"""


    """visualize_results(conditions={
        "condition_general": {"objective": "amount_satisfied_customers", "num_vehicles": ["500", "1000", "2000", "3000", "4000", "5000"], "heuristic_time_range": "720.0", "heuristic_distance_range": "1.5", "sparsity_factor": "0.3"},
        "condition_greedy": {"policy": "greedy"},
        "condition_sampling": {"policy": "sampling", "extended_horizon": timedelta(minutes=5), "reduce_weight_of_future_requests_factor": "0.2"},
        "condition_offline": {"policy": "offline"},
        "condition_SB": {"policy": "policy_SB", "extended_horizon": timedelta(minutes=5), "standardization": "0"},
        "condition_CB": {"policy": "policy_CB", "extended_horizon": timedelta(minutes=5), "standardization": "100", "fix_capacity": "1"}}, running_type="vehicleTest", metric="amount_satisfied_customers")"""


    ### EVALUATE DIFFERENT METRICS
    """visualize_results(conditions={"condition_general": {"objective": "profit", "num_vehicles": ["500", "1000", "2000", "3000", "4000", "5000"], "heuristic_time_range": "720.0", "heuristic_distance_range": "1.5", "sparsity_factor": "0.3"},
                                  "condition_greedy": {"policy": "greedy"},
                                  "condition_sampling": {"policy": "sampling", "extended_horizon": timedelta(minutes=5), "reduce_weight_of_future_requests_factor": "0.2"},
                                  "condition_offline": {"policy": "offline"},
                                  "condition_SB": {"policy": "policy_SB", "extended_horizon": timedelta(minutes=5), "standardization": "0"},
                                  "condition_CB": {"policy": "policy_CB", "extended_horizon": timedelta(minutes=5), "standardization": "100", "fix_capacity": "1"}}, running_type="vehicleTest", metric="service_ratio")

    visualize_results(conditions={"condition_general": {"objective": "profit", "num_vehicles": ["500", "1000", "2000", "3000", "4000", "5000"], "heuristic_time_range": "720.0", "heuristic_distance_range": "1.5", "sparsity_factor": "0.3"},
                                  "condition_greedy": {"policy": "greedy"},
                                  "condition_sampling": {"policy": "sampling", "extended_horizon": timedelta(minutes=5), "reduce_weight_of_future_requests_factor": "0.2"},
                                  "condition_offline": {"policy": "offline"},
                                  "condition_SB": {"policy": "policy_SB", "extended_horizon": timedelta(minutes=5), "standardization": "0"},
                                  "condition_CB": {"policy": "policy_CB", "extended_horizon": timedelta(minutes=5), "standardization": "100", "fix_capacity": "1"}}, running_type="vehicleTest", metric="km_per_request")

    visualize_results(conditions={"condition_general": {"objective": "profit", "num_vehicles": ["500", "1000", "2000", "3000", "4000", "5000"], "heuristic_time_range": "720.0", "heuristic_distance_range": "1.5", "sparsity_factor": "0.3"},
                                  "condition_greedy": {"policy": "greedy"},
                                  "condition_sampling": {"policy": "sampling", "extended_horizon": timedelta(minutes=5), "reduce_weight_of_future_requests_factor": "0.2"},
                                  "condition_offline": {"policy": "offline"},
                                  "condition_SB": {"policy": "policy_SB", "extended_horizon": timedelta(minutes=5), "standardization": "0"},
                                  "condition_CB": {"policy": "policy_CB", "extended_horizon": timedelta(minutes=5), "standardization": "100", "fix_capacity": "1"}}, running_type="vehicleTest", metric="avg_distance_trav")"""



    ### SHOW LEARNED PARAMETERS
    # SLCA
    """extract_learned_parameters(conditions={"condition_general": {"objective": "profit", "num_vehicles": ["2000"], "heuristic_time_range": "720.0", "heuristic_distance_range": "1.5", "sparsity_factor": "0.3"},
                                  "condition_CB": {"policy": "policy_CB", "extended_horizon": timedelta(minutes=5), "standardization": "100", "fix_capacity": "1"}},
                               running_type="vehicleTest", metric="profit", directory_parameters="./learned_parameters/learned_parameters_policy_CB_2000_0.3_profit_0:05:00_100/Learning_outer-obj-profit_hoSi-5.0_fleetSi-2000_pert-50.0,1.0")"""

    # SLPA
    """extract_learned_parameters(conditions={"condition_general": {"objective": "profit", "num_vehicles": ["2000"], "heuristic_time_range": "720.0", "heuristic_distance_range": "1.5", "sparsity_factor": "0.3"},
                                  "condition_SB": {"policy": "policy_SB", "extended_horizon": timedelta(minutes=5), "standardization": "0"}},
                               running_type="vehicleTest", metric="profit", directory_parameters="./learned_parameters/learned_parameters_policy_SB_2000_0.3_profit_0:05:00_0/Learning_outer-obj-profit_hoSi-5.0_fleetSi-2000_pert-50.0,1.0")"""


    """show_learning_evolution(parameter_directory="./learned_parameters/learned_parameters_policy_SB_500_0.3_amount_satisfied_customers_0:05:00_100/",
        inner_evaluation="Learning_inner-obj-amount_satisfied_customers_hoSi-5.0_fleetSi-500_pert-50.0,1.0",
        outer_evaluation="Learning_outer-obj-amount_satisfied_customers_hoSi-5.0_fleetSi-500_pert-50.0,1.0")"""