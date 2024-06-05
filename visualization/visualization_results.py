import itertools
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
    elif x_tick == "learning_rate":
        return util.get_abb_learning_rate()
    elif x_tick == "normalization":
        return util.get_abb_normalization()
    elif x_tick == "num_layers":
        return util.get_abb_number_layers()
    elif x_tick == "hidden_units":
        return util.get_abb_hidden_units()
    elif x_tick == "model":
        return util.get_abb_model()
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
        instance_parameter = filename.split("_" + key_abb + "-")[-1].split("_")[0].split("/")[0]
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
        elif "greedy" in self.directory:
            self.mode = "greedy"
        elif "policy_CB" in self.directory and (("Linear" in self.directory) or ("NN" not in self.directory)):
            self.mode = "CB_Linear"
        elif "policy_SB" in self.directory and (("Linear" in self.directory) or ("NN" not in self.directory)):
            self.mode = "SB_Linear"
        elif "policy_CB" in self.directory and "NN" in self.directory:
            self.mode = "CB_NN"
        elif "policy_SB" in self.directory and "NN" in self.directory:
            self.mode = "SB_NN"
        elif "offline" in self.directory:
            self.mode = "offline"

    def get_iteration(self):
        if self.mode in ["CB_Linear", "SB_Linear", "CB_GNN", "SB_GNN", "CB_NN", "SB_NN", "CB_GNNBack", "SB_GNNBack", "CB_GNNEdgespecific", "SB_GNNEdgespecific"]:
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
        def convert_x_tick(x_tick):
            try:
                return float(x_tick)
            except:
                return x_tick

        x_label_app = [get_abbreviation(x_lab) for x_lab in x_label]
        x_ticks = [(self.directory + self.filename).split(x_lab_app + "-")[-1].split("_")[0].split("/")[0] for x_lab_app in x_label_app]
        self.x_tick = tuple([convert_x_tick(x_tick) for x_tick in x_ticks])


def get_file_attributes(directory, filename, conditions, x_label, metric, x_tick):
    file = Result_file(filename, directory)
    if (check_condition(directory + file.filename, conditions["condition_general"]) or file.check_if_NoResult_file()):
        return -1
    file.get_objective_value(metric=metric)
    file.get_mode()
    if file.mode == "CB_Linear":
        skip = check_condition(directory + file.filename, conditions["condition_CB_Linear"][x_tick])
    elif file.mode == "SB_Linear":
        skip = check_condition(directory + file.filename, conditions["condition_SB_Linear"][x_tick])
    elif file.mode == "CB_NN":
        skip = check_condition(directory + file.filename, conditions["condition_CB_NN"][x_tick])
    elif file.mode == "SB_NN":
        skip = check_condition(directory + file.filename, conditions["condition_SB_NN"][x_tick])
    elif file.mode == "sampling":
        skip = check_condition(directory + file.filename, conditions["condition_sampling"][x_tick])
    elif file.mode == "offline":
        skip = check_condition(directory + file.filename, conditions["condition_offline"][x_tick])
    elif file.mode == "greedy":
        skip = check_condition(directory + file.filename, conditions["condition_greedy"][x_tick])
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
            best_iteration = 1  # -1
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


def fill_saver(directories, results_saver, conditions, modes, x_ticks, x_label, metric):
    for mode in modes:
        for x_tick in x_ticks:
            if os.path.isdir(directories[mode][x_tick]):
                for filename in os.listdir(directories[mode][x_tick]):
                    file = get_file_attributes(directories[mode][x_tick], filename, conditions, x_label, metric, x_tick)
                    if file == -1:
                        continue
                    else:
                        results_saver = add_file_to_saver(results_saver, file)
    return results_saver


def convert_x_tick(x_tick):
    def convert_type(x_tick_item):
        if isinstance(x_tick_item, timedelta):
            return float(x_tick_item.seconds / 60)
        else:
            if "." in x_tick_item:
                return float(x_tick_item)
            else:
                try:
                    return int(x_tick_item)
                except:
                    return x_tick_item

    x_tick = [convert_type(x_tick_item) for x_tick_item in x_tick]
    return tuple(x_tick)


def create_result_saver(x_ticks, modes):
    results_saver = OrderedDict()
    for mode in modes:
        results_saver[mode] = OrderedDict()
        for x_tick in x_ticks:
            results_saver[mode][x_tick] = OrderedDict()
    return results_saver


def sort_directories(saver):
    for mode in saver.keys():
        saver[mode] = OrderedDict(sorted(saver[mode].items(), key=lambda t: t[0]))
    return saver


def replace_with_mean(saver):
    for mode in saver.keys():
        for x_tick in saver[mode].keys():
            if len(saver[mode][x_tick]) > 0:
                saver[mode][x_tick] = sum(saver[mode][x_tick]) / len(saver[mode][x_tick])
            else:
                saver[mode][x_tick] = 0
    return saver


def set_best_learning_iteration(saver, best_learning_iteration):
    for mode in saver.keys():
        for x_tick in saver[mode].keys():
            if len(saver[mode][x_tick]) > 0:
                saver[mode][x_tick] = saver[mode][x_tick][best_learning_iteration[mode][x_tick]]
    return saver


def get_y_lim(metric, type):
    y_lim_saver = {"service_ratio": {"ratio": (-13, 15)}, "km_per_request": {"ratio": (0, 120)}, "avg_distance_trav": {"ratio": (0, 80)}}
    return y_lim_saver[metric][type]


def plot(saver, type, metric, x_label, running_type, show_metric, boxplot=False, twitter=False):
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
                                   "hyperparameterSelection": ["xmajorticks=true", "xlabel style={below=11mm}", "ymajorticks=true", "ylabel style={align=center}", "scaled y ticks=true", "xtick style={draw=none}",
                                                               "ytick style={draw=none}"],
                                   "absolute": ["xmajorticks=true", "ymajorticks=true", "ylabel style={align=center}", "scaled y ticks=true", "xtick style={draw=none}", "ytick style={draw=none}"],
                                   "ratio": ["xmajorticks=true", "ymajorticks=true", "ylabel style={align=center}", "scaled y ticks=false", "y tick label style={/pgf/number format/fixed}", "xtick style={draw=none}",
                                             "ytick style={draw=none}"]}

    patches_dict = {"vehicleTest": {"profit": {"absolute": (((1800, 31000), 400, 34000), (1700, 27000)),
                                               "ratio": (((1800, -1.5), 400, 10), (2300, 6))},
                                    "amount_satisfied_customers": {"absolute": (((1800, 2.8), 400, 3.2), (1700, 2.5)),
                                                                   "ratio": (((1800, -1), 400, 10), (1700, -2.5))}},
                    "sparsityTest": {"profit": {"absolute": (((0.26, 31000), 0.08, 45000), (0.25, 25000)), "ratio": (((0.26, -2), 0.08, 9.5), (0.14, 8.3))}}}

    visualization = {
        "greedy": {"color": "#98C6EA", "marker": "o", "label": "greedy"},
        "sampling": {"color": "#005293", "marker": "*", "label": "sampling"},
        "offline": {"color": "#003359", "marker": "+", "label": "FI"},
        "SB_Linear": {"color": "#E37222", "marker": "v", "label": "SBLinear"},
        "CB_Linear": {"color": "#A2AD00", "marker": "x", "label": "CBLinear"},
        "SB_NN": {"color": "red", "marker": "v", "label": "SBNN"},
        "CB_NN": {"color": "black", "marker": "x", "label": "CBNN"},
        "SB_GNN": {"color": "red", "marker": "v", "label": "SBGNN"},
        "CB_GNN": {"color": "black", "marker": "x", "label": "CBGNN"},
        "SB_GNNBack": {"color": "red", "marker": "v", "label": "SBGNNBack"},
        "CB_GNNBack": {"color": "black", "marker": "x", "label": "CBGNNBack"},
        "SB_GNNEdgespecific": {"color": "red", "marker": "v", "label": "SBGNNEdgespecific"},
        "CB_GNNEdgespecific": {"color": "black", "marker": "x", "label": "CBGNNEdgespecific"}
        # "SB_Linear": {"color": "#E37222", "marker": "v", "label": "SB_Linear"},
        # "CB_Linear": {"color": "#A2AD00", "marker": "x", "label": "CB_Linear"}
    }
    x_label_name = {"num_vehicles": "Fleet size", "extended_horizon": "Extended horizon [hour:min:sec]",
                    "reduce_weight_of_future_requests_factor": "Factor reducing weights \\\\ to requests in prediction horizon",
                    "heuristic_distance_range": "Maximum distance to next request [km]", "heuristic_time_range": "Maximum time to next request [sec]",
                    "sparsity_factor": "Request density", "fix_capacity": "Num. of capacity vertices \\\\ in rebalancing cells", "learning_rate": "Learning rate",
                    "hidden_units": "Hidden Units", "normalization": "Normalization", "num_layers": "Number layers", "standardization": "Standardization"}

    def get_model(model):
        if model == "offline":
            return "FI"
        elif "_" in model:
            return "~".join(model.split("_"))
        else:
            return model

    plt.rc('axes', labelsize=14)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=14)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=14)  # fontsize of the tick labels
    plt.rc('legend', fontsize=14)  # legend fontsize
    plt.rc('font', size=14)
    sns.set_style('whitegrid', {'axes.labelcolor': 'black'})  # darkgrid, white grid, dark, white and ticksplt.rc('axes', titlesize=18)     # fontsize of the axes title
    sns.color_palette('deep')
    fig, ax = plt.subplots(figsize=(8, 6))

    modes = list(saver.keys())
    if type == "absolute":
        denominator = {key: 1 for (key, value) in saver[modes[0]].items()}
        subtrahend = {key: 0 for (key, value) in saver[modes[0]].items()}
        label_suffix = ""
        factor = 1
        if show_metric == "profit":
            y_label = 'Profit [\$]'
        elif show_metric == "amount_satisfied_customers":
            y_label = 'Number of satisfied ride requests \\\\ in 1000'
        elif show_metric == "service_ratio":
            y_label = "Service rate [%]"
        elif show_metric == "km_per_request":
            y_label = "avg. km per request [km]"
        elif show_metric == "avg_distance_trav":
            y_label = "avg. km per vehicle [km]"
    elif type == "ratio":
        denominator = saver["greedy"]
        subtrahend = saver["greedy"]
        label_suffix = " / greedy"
        factor = 100
        modes.remove("greedy")
        modes.remove("offline")
        if show_metric == "profit":
            if twitter:
                y_label = 'Increase of profit \n compared to greedy bench. [%]'
            else:
                y_label = 'Increase of profit \\\\ compared to greedy bench. [%]'
        elif show_metric == "amount_satisfied_customers":
            y_label = 'Increase of num. sat. ride req. \\\\ compared to greedy bench. [%]'
        elif show_metric == "service_ratio":
            y_label = "Increase of num. of service rate \\\\ compared to greedy bench. [%]"
        elif show_metric == "km_per_request":
            y_label = "Increase of avg. km per request \\\\ compared to greedy bench. [%]"
        elif show_metric == "avg_distance_trav":
            y_label = "Increase of avg. km per vehicle \\\\ compared to greedy bench. [%]"
    else:
        raise Exception("Wrong type defined!")

    for mode in modes:
        if boxplot:
            if running_type in ["samplingHyperparameterSelection", "hyperparameterSelection"]:
                x_labels = [f"0{str(timedelta(seconds=minutes[0] * 60))}" for minutes in list(saver[mode].keys())]
            else:
                x_labels = [minutes for minutes in list(saver[mode].keys())]
            ax.boxplot(saver[mode].values(), positions=list(range(len(saver[mode].values()))), patch_artist=True)
            plt.xticks(ticks=list(range(len(saver[mode].values()))), labels=x_labels, rotation=90)
            print([f"{mode} : {x_tick} :{np.sum(saver[mode][x_tick]) / len(saver[mode][x_tick])}" for x_tick in saver[mode].keys()])
        else:
            try:
                plt.plot(list(saver[mode].keys()), [factor * x for x in [(saver[mode][x_tick] - subtrahend[x_tick]) / denominator[x_tick] for x_tick in saver[mode].keys()]],
                         color=visualization[mode]["color"], marker=visualization[mode]["marker"], label=get_model(mode) + label_suffix)  # mode.split("_")[-1]
            except:
                print("Hier")
            if type == "ratio" and running_type in ["vehicleTest", "sparsityTest"]:
                print("Comparison : " + mode + " / greedy" + ": " + str(np.mean([factor * x for x in [(saver[mode][x_tick] - saver["greedy"][x_tick]) / saver["greedy"][x_tick] for x_tick in saver[mode].keys()]])))
                print("Comparison : " + mode + " / sampling" + ": " + str(np.mean([factor * x for x in [(saver[mode][x_tick] - saver["sampling"][x_tick]) / saver["sampling"][x_tick] for x_tick in saver[mode].keys()]])))
                print("Comparison MAX : " + mode + " / greedy" + ": " + str(np.max([factor * x for x in [(saver[mode][x_tick] - saver["greedy"][x_tick]) / saver["greedy"][x_tick] for x_tick in saver[mode].keys()]])))
                print(
                    "Comparison MAX : " + mode + " / sampling" + ": " + str(np.max([factor * x for x in [(saver[mode][x_tick] - saver["sampling"][x_tick]) / saver["sampling"][x_tick] for x_tick in saver[mode].keys()]])))
                print("Comparison MIN : " + mode + " / greedy" + ": " + str(np.min([factor * x for x in [(saver[mode][x_tick] - saver["greedy"][x_tick]) / saver["greedy"][x_tick] for x_tick in saver[mode].keys()]])))
                print(
                    "Comparison MIN : " + mode + " / sampling" + ": " + str(np.min([factor * x for x in [(saver[mode][x_tick] - saver["sampling"][x_tick]) / saver["sampling"][x_tick] for x_tick in saver[mode].keys()]])))
                print("WE NEED TO ENABLE THIS!!!!!!")
                # if running_type == "sparsityTest":
                #    print("Comparison 0.4 : " + mode + " / greedy" + ": " + str(factor * (saver[mode][0.4] - saver["greedy"][0.4]) / saver["greedy"][0.4]))
                #    print("Comparison 0.4 : " + mode + " / sampling" + ": " + str(factor * (saver[mode][0.4] - saver["sampling"][0.4]) / saver["sampling"][0.4]))
                #    print("Comparison HIGHER : " + mode + " / greedy" + ": " + str(np.mean([factor * x for x in [(saver[mode][x_tick] - saver["greedy"][x_tick]) / saver["greedy"][x_tick] for x_tick in [0.4, 0.5, 0.6, 0.7, 1.0]]])))
                #    print("Comparison HIGHER : " + mode + " / sampling" + ": " + str(np.mean([factor * x for x in [(saver[mode][x_tick] - saver["sampling"][x_tick]) / saver["sampling"][x_tick] for x_tick in [0.4, 0.5, 0.6, 0.7, 1.0]]])))
    plt.xlabel([x_label_name[x_lab] for x_lab in x_label])
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
    tikzplotlib.save("./visualization/evaluateNN_{}_{}_{}_{}.tex".format(running_type, x_label, metric, type if running_type != "hyperparameterSelection" else type + "_" + mode),
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
    def __init__(self, modes, x_labels, x_ticks):
        self.x_ticks = x_ticks
        self.x_labels = x_labels
        self.modes = modes
        self.result_directory = None
        self.dict = {x_tick: {mode: {} for mode in self.modes} for x_tick in self.x_ticks}
        self.reset_args()

    def reset_args(self):
        self.policy = None
        self.num_vehicles = None
        self.sparsity_factor = None
        self.objective = None
        self.extended_horizon = None
        self.fix_capacity = None
        self.standardization = None
        self.heuristic_distance_range = None
        self.heuristic_time_range = None
        self.model = None
        self.mode = "evaluation"
        self.reduce_weight_of_future_requests_factor = None
        self.learning_rate = None
        self.normalization = None
        self.num_layers = None
        self.hidden_units = None

    def get_values(self, keys):
        values = []
        for key in keys:
            values.append(self.dict[key])
        return values

    def set_values(self, modes, conditions, x_labels):
        for mode in modes:
            if sorted(list(conditions.keys())) != sorted(self.x_ticks):
                for x_tick in self.x_ticks:
                    # when we set the parameter values and the value is a list, this means that we consider a label, then we get the value from the x_ticks.
                    mapping_xtick_value = {parameter: value for parameter, value in zip(x_labels, x_tick)}
                    for condition in conditions:
                        value = conditions[condition]
                        if isinstance(value, list):
                            self.dict[x_tick][mode][condition] = mapping_xtick_value[condition]
                        else:
                            self.dict[x_tick][mode][condition] = value
            else:
                for x_tick, condition_lab in conditions.items():
                    for condition in condition_lab.keys():
                        self.dict[x_tick][mode][condition] = condition_lab[condition]

    def set_value(self, keys, values):
        for key, value in zip(keys, values):
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
            elif key == "learning_rate":
                self.learning_rate = value
            elif key == "normalization":
                self.normalization = value
            elif key == "num_layers":
                self.num_layers = value
            elif key == "hidden_units":
                self.hidden_units = value  # [value] * 2
            elif key == "model":
                self.model = value

    def get_args(self, x_tick, mode):
        for key in self.dict[x_tick][mode].keys():
            self.set_value([key], [self.dict[x_tick][mode][key]])


def get_x_label(conditions):
    x_label = []
    for key in conditions.keys():
        if isinstance(conditions[key], list):
            x_label.append(key)
    return x_label


def get_results_directories(args_keeper, mode, x_tick):
    args_keeper.get_args(x_tick, mode)
    result_directory = util.get_result_directory(args_keeper)
    args_keeper.reset_args()
    return result_directory


def get_metadata(conditions):
    modes = []
    for condition in conditions.keys():
        if "general" in condition:
            continue
        else:
            modes.append("_".join(condition.split("condition_")[1:]))

    x_labels = get_x_label(conditions["condition_general"])
    x_ticks = list(itertools.product(*[conditions["condition_general"][x_label] for x_label in x_labels]))
    x_ticks = [convert_x_tick(x_tick) for x_tick in x_ticks]
    args_keeper = ArgsKeeper(modes, x_labels, x_ticks)
    args_keeper.set_values(modes=modes, conditions=conditions["condition_general"], x_labels=x_labels)
    return args_keeper


def generate_relevant_directories(args_keeper, conditions, result_directory):
    args_keeper.set_value(["result_directory"], [result_directory])
    directories = create_result_saver(x_ticks=args_keeper.x_ticks, modes=args_keeper.modes)
    for mode in args_keeper.modes:
        args_keeper.set_values([mode], conditions[f"condition_{mode}"], args_keeper.x_labels)
        for x_tick in args_keeper.x_ticks:
            directories[mode][x_tick] = get_results_directories(args_keeper, mode, x_tick)
    return directories


def update_conditions(conditions, x_ticks):
    condition_new = {}
    for condition_key, condition_value in conditions.items():
        if "general" in condition_key:
            condition_new[condition_key] = condition_value
        else:
            if sorted([convert_x_tick(x_tick) for x_tick in condition_value.keys()]) == sorted(x_ticks):
                condition_new[condition_key] = {key: value for key, value in zip([convert_x_tick(x_tick) for x_tick in condition_value.keys()], condition_value.values())}
            else:
                condition_new[condition_key] = {x_tick: condition_value for x_tick in x_ticks}
    return condition_new


def visualize_results(conditions, running_type, metric, result_directory, twitter=False, running_type_validation=None, show_metric=None):
    if not show_metric:
        show_metric = metric

    if running_type_validation is None:
        running_type_validation = running_type
    args_keeper = get_metadata(conditions)
    conditions = update_conditions(conditions, x_ticks=args_keeper.x_ticks)

    # test_directories = generate_relevant_directories(args_keeper, conditions, result_directory=f"./results_original_saver/results_{running_type}_test")
    # validation_directories = generate_relevant_directories(args_keeper, conditions, result_directory=f"./results_original_saver/results_{running_type_validation}_validation")

    test_directories = generate_relevant_directories(args_keeper, conditions, result_directory=f"./{result_directory}/results_{running_type}_test")
    validation_directories = generate_relevant_directories(args_keeper, conditions, result_directory=f"./{result_directory}/results_{running_type_validation}_validation")

    test_results_saver = create_result_saver(x_ticks=args_keeper.x_ticks, modes=args_keeper.modes)
    validation_results_saver = create_result_saver(x_ticks=args_keeper.x_ticks, modes=args_keeper.modes)
    validation_results_saver = fill_saver(validation_directories, validation_results_saver, conditions, args_keeper.modes, args_keeper.x_ticks, args_keeper.x_labels, metric)
    best_learning_iteration = choose_best_learning_iteration(validation_results_saver)
    test_results_saver = fill_saver(test_directories, test_results_saver, conditions, args_keeper.modes, args_keeper.x_ticks, args_keeper.x_labels, show_metric)
    test_results_saver = set_best_learning_iteration(test_results_saver, best_learning_iteration)
    test_results_saver = sort_directories(test_results_saver)
    test_results_saver = divide_by_value(test_results_saver, metric)
    results_saver = replace_with_mean(test_results_saver)

    plot(results_saver, type="ratio", metric=metric, x_label=args_keeper.x_labels, running_type=running_type, show_metric=show_metric)


def evaluate_hyperparameters(conditions, running_type, metric, result_directory):
    args_keeper = get_metadata(conditions)
    conditions = update_conditions(conditions, x_ticks=args_keeper.x_ticks)
    validation_directories = generate_relevant_directories(args_keeper, conditions, result_directory=f"./{result_directory}/results_{running_type}_validation")
    hyperparameters_results_saver = create_result_saver(x_ticks=args_keeper.x_ticks, modes=args_keeper.modes)
    hyperparameters_results_saver = fill_saver(validation_directories, hyperparameters_results_saver, conditions, args_keeper.modes, args_keeper.x_ticks, args_keeper.x_labels, metric)
    best_learning_iteration = choose_best_learning_iteration(hyperparameters_results_saver)
    hyperparameters_results_saver = set_best_learning_iteration(hyperparameters_results_saver, best_learning_iteration)
    hyperparameters_results_saver = sort_directories(hyperparameters_results_saver)
    hyperparameters_results_saver = divide_by_value(hyperparameters_results_saver, metric)

    for key in hyperparameters_results_saver.keys():
        best_hyp = max(hyperparameters_results_saver[key], key=lambda x: np.mean(hyperparameters_results_saver[key][x]))
        print(f"The best hyperparameters in {key} are {best_hyp} with: {np.mean(hyperparameters_results_saver[key][best_hyp])}.")

    plot(hyperparameters_results_saver, type="absolute", metric=metric, x_label=args_keeper.x_labels, boxplot=True, running_type=running_type, show_metric=metric)
    results_saver = replace_with_mean(hyperparameters_results_saver)
    # plot(results_saver, type="ratio", metric=metric, x_label=x_label, running_type=running_type)


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
            return "est. fut. cost at d.off loc. +[{},{})min".format(2 * (idx - 1), 2 * idx)
        if "smooth_revenue" in parameter_name:
            idx = int(parameter_name.split("_")[2])
            return "est. fut. rew. at d.off loc. +[{},{})min".format(2 * (idx - 1), 2 * idx)
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
                    print(str(index_one) + " & " + get_name(keys_list[index_one]) + " & " + str(round_number(parameter_list[index_one])) + " & " + str(index_two) + " & " + get_name(keys_list[index_two]) + " & " + str(
                        round_number(parameter_list[index_two])) + "\\\\")
                except:
                    print('\033[93m' + keys_list[index_one] + "|" + keys_list[index_two] + '\033[0m')
            else:
                try:
                    print(str(index_one) + " & " + get_name(keys_list[index_one]) + " & " + str(round_number(parameter_list[index_one])) + " & & & " + "\\\\")
                except:
                    print('\033[93m' + keys_list[index_one] + '\033[0m')
        print("\\toprule")
    print("\end{tabular}} \n\
\\caption{" + caption[learning_type] + "} \n\
\\label{" + labels[learning_type] + "} \n\
\\end{table} \n\
\\endgroup \n\
\\clearpage")


def get_computational_time(conditions, running_type, metric, result_directory, show_metric=None):
    """args_keeper = get_metadata(conditions)
    conditions = update_conditions(conditions, x_ticks=args_keeper.x_ticks)
    validation_directories = generate_relevant_directories(args_keeper, conditions, result_directory=f"./results_LRZ/results_{running_type}_validation")
    hyperparameters_results_saver = create_result_saver(x_ticks=args_keeper.x_ticks, modes=args_keeper.modes)
    hyperparameters_results_saver = fill_saver(validation_directories, hyperparameters_results_saver, conditions, args_keeper.modes, args_keeper.x_ticks, args_keeper.x_labels, metric)
    best_learning_iteration = choose_best_learning_iteration(hyperparameters_results_saver)
    hyperparameters_results_saver = set_best_learning_iteration(hyperparameters_results_saver, best_learning_iteration)
    hyperparameters_results_saver = sort_directories(hyperparameters_results_saver)
    computational_time = divide_by_value(hyperparameters_results_saver, metric)"""

    if not show_metric:
        show_metric = metric

    args_keeper = get_metadata(conditions)
    conditions = update_conditions(conditions, x_ticks=args_keeper.x_ticks)
    validation_directories = generate_relevant_directories(args_keeper, conditions, result_directory=f"./{result_directory}/results_{running_type}_validation")
    hyperparameters_results_saver = create_result_saver(x_ticks=args_keeper.x_ticks, modes=args_keeper.modes)
    hyperparameters_results_saver = fill_saver(validation_directories, hyperparameters_results_saver, conditions, args_keeper.modes, args_keeper.x_ticks, args_keeper.x_labels, metric)
    best_learning_iteration = choose_best_learning_iteration(hyperparameters_results_saver)
    hyperparameters_results_saver = set_best_learning_iteration(hyperparameters_results_saver, best_learning_iteration)
    hyperparameters_results_saver = sort_directories(hyperparameters_results_saver)
    computational_time = divide_by_value(hyperparameters_results_saver, metric)

    for x_label in computational_time[list(computational_time.keys())[0]]:
        print("& " + str(x_label), end='')
    print("\\\\")
    print("\hline \hline")
    for mode in computational_time.keys():
        print(mode, end='')
        for x_label in computational_time[mode]:
            print(" & " + str(np.round(np.mean(computational_time[mode][x_label]), decimals=3)), end='')
        print("\\\\")


def show_learning_evolution(parameter_directory, inner_evaluation, outer_evaluation):
    inner_evaluation_values = json.load(open(parameter_directory + inner_evaluation))
    if outer_evaluation is not None:
        outer_evaluation = json.load(open(parameter_directory + outer_evaluation))
    else:
        outer_evaluation = {"parameter_evolution": []}

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
        for epoch_idx, inner_parameters in enumerate(inner_evaluation_values["parameter_evolution"]):
            if parameters_identical(outer_parameters, inner_parameters):
                outer_epochs.append(epoch_idx)
                outer_evaluation["loss_evolution"].append(inner_evaluation_values["loss_evolution"][epoch_idx])

    plt.plot(inner_evaluation_values["loss_evolution"], color="blue")
    plt.scatter(outer_epochs, outer_evaluation["loss_evolution"], color="red")
    plt.title(f"Evolution of loss during training \n {parameter_directory} \n {inner_evaluation}", fontsize=8)
    plt.xlabel("Training epochs")
    plt.ylabel("Loss")
    plt.show()

    if "edges_to_request_evolution" in inner_evaluation_values.keys():
        plt.plot(inner_evaluation_values["edges_to_request_evolution"])
        plt.title("Edges to requests evolution")
        plt.show()


def sanity_check(sanity_directory, policy):
    sanity_value_FY_dict = {"Linear": {500: 0, 1000: 0, 2000: 0, 3000: 0, 4000: 0, 5000: 0}, "NN": {500: 0, 1000: 0, 2000: 0, 3000: 0, 4000: 0, 5000: 0}}
    sanity_value_model_dict = {"Linear": {500: 0, 1000: 0, 2000: 0, 3000: 0, 4000: 0, 5000: 0}, "NN": {500: 0, 1000: 0, 2000: 0, 3000: 0, 4000: 0, 5000: 0}}
    for sanity_file_name in os.listdir(sanity_directory):
        if "Sanity" not in sanity_file_name:
            continue
        if policy not in sanity_file_name:
            continue
        fleet_size = int(sanity_file_name.split(f"_{util.get_abbreviation_fleetSize()}-")[1].split("_")[0])
        model = sanity_file_name.split(f"_{util.get_abb_model()}-")[1].split("_")[0]
        sanity_list_values = json.load(open(sanity_directory + sanity_file_name, "r"))
        sanity_list_values_array = np.array(sanity_list_values)
        sanity_value_FY_dict[model][fleet_size] = np.mean(sanity_list_values_array[:, 0])
        sanity_value_model_dict[model][fleet_size] = np.mean(sanity_list_values_array[:, 1])

    # Compare FY of Linear and NN
    x_Linear, y_Linear = zip(*sorted(sanity_value_FY_dict["Linear"].items()))
    plt.plot(x_Linear, y_Linear, color="blue", label="Loss~Linear~model")
    x_NN, y_NN = zip(*sorted(sanity_value_FY_dict["NN"].items()))
    plt.plot(x_NN, y_NN, color="red", label="Loss~NN~model")
    plt.legend()
    # plt.title(policy)
    plt.ylabel("Loss value")
    plt.xlabel("Fleet size")

    tikzplotlib.save(f"./visualization/comparisonLossLinearNN{policy}.tex")

    plt.show()

    # Compare FY_NN model_NN
    x_NN_model, y_NN_model = zip(*sorted(sanity_value_model_dict["NN"].items()))
    x_NN_FY, y_NN_FY = zip(*sorted(sanity_value_FY_dict["NN"].items()))
    plt.plot(x_NN_FY, np.nan_to_num((np.array(y_NN_FY) - np.array(y_NN_model)) / np.array(y_NN_model), posinf=0), color="red")
    plt.legend()
    plt.title(policy + "\n y_NN_FY: NN loss with perturbation \n y_NN_model: NN loss without perturbation")
    plt.ylabel("(y_NN_FY - y_NN_model) / y_NN_model")
    plt.xlabel("Fleet size")
    plt.tight_layout()
    plt.show()


def sanity_check_performance(sanity_performance_directory, policy):
    sanity_value_dict = {"Linear": {500: 0, 1000: 0, 2000: 0, 3000: 0, 4000: 0, 5000: 0}, "NN": {500: 0, 1000: 0, 2000: 0, 3000: 0, 4000: 0, 5000: 0}}
    sanity_value_request_dict = {"Linear": {500: 0, 1000: 0, 2000: 0, 3000: 0, 4000: 0, 5000: 0}, "NN": {500: 0, 1000: 0, 2000: 0, 3000: 0, 4000: 0, 5000: 0}}
    for sanity_file_name in os.listdir(sanity_performance_directory):
        if "Sanity" not in sanity_file_name:
            continue
        if "performance" not in sanity_file_name:
            continue
        if policy not in sanity_file_name:
            continue
        fleet_size = int(sanity_file_name.split(f"_{util.get_abbreviation_fleetSize()}-")[1].split("_")[0])
        model = sanity_file_name.split(f"_{util.get_abb_model()}-")[1].split("_")[0]
        if "performance" in model:
            model = model.split("--")[0]
        sanity_list_values = json.load(open(sanity_performance_directory + sanity_file_name, "r"))
        sanity_array_values = np.array(sanity_list_values)
        sanity_value_dict[model][fleet_size] = sanity_array_values[:, 0]
        sanity_value_request_dict[model][fleet_size] = sanity_array_values[:, 1]

    x_Linear, y_Linear = zip(*sorted(sanity_value_dict["Linear"].items()))
    x_NN, y_NN = zip(*sorted(sanity_value_dict["NN"].items()))
    plt.boxplot(((np.array(y_NN) - np.array(y_Linear)) / np.array(y_Linear)).T)
    plt.xticks([1, 2, 3, 4, 5, 6], [500, 1000, 2000, 3000, 4000, 5000])
    plt.legend()
    plt.ylabel("$f = (g_{NN} - g_{Linear}) / g_{Linear}$")
    plt.xlabel("Fleet size")
    plt.tight_layout()

    tikzplotlib.save(f"./visualization/comparisonEqualityLinearNN{policy}.tex")

    plt.show()


if __name__ == '__main__':
    os.chdir('../')

    ### HYPERPARAMETER_SAMPLING_BENCHMARK
    evaluate_hyperparameters(
        conditions={"condition_general": {"extended_horizon": [timedelta(minutes=5), timedelta(minutes=10), timedelta(minutes=15), timedelta(minutes=20), timedelta(minutes=25), timedelta(minutes=30)],
                                          "objective": "profit", "num_vehicles": "2000", "heuristic_time_range": "720.0", "heuristic_distance_range": "1.5", "sparsity_factor": "0.3",
                                          "reduce_weight_of_future_requests_factor": "0.2"},
                    "condition_sampling": {"policy": "sampling"}},
        running_type="samplingHyperparameterSelection", metric="profit", result_directory="results/results")

    evaluate_hyperparameters(conditions={"condition_general": {
        "extended_horizon": timedelta(minutes=5), "reduce_weight_of_future_requests_factor": ["1.0", "0.9", "0.8", "0.7", "0.6", "0.5", "0.4", "0.3", "0.2", "0.1"],
        "objective": "profit", "num_vehicles": "2000", "heuristic_time_range": "720.0", "heuristic_distance_range": "1.5", "sparsity_factor": "0.3"},
        "condition_sampling": {"policy": "sampling"}},
        running_type="samplingReduceFutureSelect", metric="profit", result_directory="results/results")

    ### HYPERPARAMETER OFFLINE
    evaluate_hyperparameters(conditions={
        "condition_general": {"objective": "profit", "num_vehicles": "2000", "heuristic_time_range": "100000000.0", "heuristic_distance_range": ["0.3", "0.5", "0.7", "0.9", "1.1", "1.3", "1.5", "1.7"],
                              "sparsity_factor": "0.3"},
        "condition_offline": {"policy": "offline"}},
                             running_type="hyperparameterOffline_distance", metric="profit", result_directory="results/results")

    evaluate_hyperparameters(conditions={
        "condition_general": {"objective": "profit", "num_vehicles": "2000", "heuristic_time_range": ["120.0", "180.0", "360.0", "540.0", "720.0", "900.0"], "heuristic_distance_range": "200.0",
                              "sparsity_factor": "0.3"},
        "condition_offline": {"policy": "offline"}},
        running_type="hyperparameterOffline_time", metric="profit", result_directory="results/results")

    ### HYPERPARAMETER CB AND SB POLICIES
    evaluate_hyperparameters(conditions={
        "condition_general": {"objective": "profit", "num_vehicles": "2000", "heuristic_time_range": "720.0", "heuristic_distance_range": "1.5",
                              "sparsity_factor": "0.3", "extended_horizon": [timedelta(minutes=5), timedelta(minutes=10), timedelta(minutes=15), timedelta(minutes=20), timedelta(minutes=25), timedelta(minutes=30)]},
        "condition_SB_Linear": {"policy": "policy_SB", "standardization": "0"}},
        running_type="hyperparameterSelection", metric="profit", result_directory="results/results")

    evaluate_hyperparameters(conditions={
        "condition_general": {"objective": "profit", "num_vehicles": "2000", "heuristic_time_range": "720.0", "heuristic_distance_range": "1.5",
                              "sparsity_factor": "0.3", "extended_horizon": [timedelta(minutes=5), timedelta(minutes=10), timedelta(minutes=15), timedelta(minutes=20), timedelta(minutes=25), timedelta(minutes=30)]},
        "condition_CB_Linear": {"policy": "policy_CB", "standardization": "100", "fix_capacity": "1"}},
        running_type="hyperparameterSelection", metric="profit", result_directory="results/results")

    evaluate_hyperparameters(conditions={
        "condition_general": {"objective": "profit", "num_vehicles": "2000", "heuristic_time_range": "720.0", "heuristic_distance_range": "1.5", "fix_capacity": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
                              "sparsity_factor": "0.3"},
        "condition_CB_Linear": {"policy": "policy_CB", "extended_horizon": timedelta(minutes=5), "standardization": "100"}},
        running_type="testmaxcapacity", metric="profit", result_directory="results/results")

    # EVALUATE DIFFERENT VEHICLE FLEET SIZES
    visualize_results(conditions={
        "condition_general": {"objective": "profit", "num_vehicles": ["500", "1000", "2000", "3000", "4000", "5000"], "heuristic_time_range": "720.0", "heuristic_distance_range": "1.5", "sparsity_factor": "0.3"},
        "condition_greedy": {"policy": "greedy"},
        "condition_sampling": {"policy": "sampling", "extended_horizon": timedelta(minutes=5), "reduce_weight_of_future_requests_factor": "0.2"},
        "condition_offline": {"policy": "offline"},
        "condition_SB_Linear": {"policy": "policy_SB", "extended_horizon": timedelta(minutes=5), "standardization": "0"},
        "condition_CB_Linear": {"policy": "policy_CB", "extended_horizon": timedelta(minutes=5), "standardization": "100", "fix_capacity": "1"}},
                      running_type="vehicleTest", metric="profit", result_directory="results/results")

    # EVALUATE DIFFERENT VEHICLE FLEET SIZES and DEEP LEARNING BENCHMARKS
    visualize_results(conditions={
        "condition_general": {"objective": "profit", "num_vehicles": ["500", "1000", "2000", "3000", "4000", "5000"], "heuristic_time_range": "720.0", "heuristic_distance_range": "1.5", "sparsity_factor": "0.3"},
        "condition_greedy": {"policy": "greedy"},
        "condition_sampling": {"policy": "sampling", "extended_horizon": timedelta(minutes=5), "reduce_weight_of_future_requests_factor": "0.2"},
        "condition_offline": {"policy": "offline"},
        "condition_SB_Linear": {"policy": "policy_SB", "extended_horizon": timedelta(minutes=5), "standardization": "0", "model": "Linear"},
        "condition_CB_Linear": {"policy": "policy_CB", "extended_horizon": timedelta(minutes=5), "standardization": "100", "fix_capacity": "1", "model": "Linear"},
        "condition_SB_NN": {
            ("500",): {"policy": "policy_SB", "extended_horizon": timedelta(minutes=5), "num_layers": "3", "standardization": "100", "model": "NN", "learning_rate": "0.001", "hidden_units": "10.0", "normalization": "0"},
            ("1000",): {"policy": "policy_SB", "extended_horizon": timedelta(minutes=5), "num_layers": "3", "standardization": "100", "model": "NN", "learning_rate": "0.001", "hidden_units": "1000.0",
                        "normalization": "0"},
            ("2000",): {"policy": "policy_SB", "extended_horizon": timedelta(minutes=5), "num_layers": "3", "standardization": "100", "model": "NN", "learning_rate": "0.001", "hidden_units": "1000.0",
                        "normalization": "0"},
            ("3000",): {"policy": "policy_SB", "extended_horizon": timedelta(minutes=5), "num_layers": "3", "standardization": "0", "model": "NN", "learning_rate": "0.001", "hidden_units": "1000.0",
                        "normalization": "0"},
            ("4000",): {"policy": "policy_SB", "extended_horizon": timedelta(minutes=5), "num_layers": "3", "standardization": "0", "model": "NN", "learning_rate": "0.1", "hidden_units": "1000.0", "normalization": "0"},
            ("5000",): {"policy": "policy_SB", "extended_horizon": timedelta(minutes=5), "num_layers": "0", "standardization": "0", "model": "NN", "learning_rate": "0.001", "hidden_units": "1000.0",
                        "normalization": "0"}},
        "condition_CB_NN": {
            ("500",): {"policy": "policy_CB", "extended_horizon": timedelta(minutes=5), "num_layers": "0", "standardization": "100", "fix_capacity": "1", "model": "NN", "learning_rate": "0.1", "hidden_units": "10.0",
                       "normalization": "0"},
            ("1000",): {"policy": "policy_CB", "extended_horizon": timedelta(minutes=5), "num_layers": "3", "standardization": "100", "fix_capacity": "1", "model": "NN", "learning_rate": "0.1", "hidden_units": "1000.0",
                        "normalization": "0"},
            ("2000",): {"policy": "policy_CB", "extended_horizon": timedelta(minutes=5), "num_layers": "0", "standardization": "0", "fix_capacity": "1", "model": "NN", "learning_rate": "0.1", "hidden_units": "10.0",
                        "normalization": "0"},
            ("3000",): {"policy": "policy_CB", "extended_horizon": timedelta(minutes=5), "num_layers": "0", "standardization": "100", "fix_capacity": "1", "model": "NN", "learning_rate": "0.1", "hidden_units": "10.0",
                        "normalization": "0"},
            ("4000",): {"policy": "policy_CB", "extended_horizon": timedelta(minutes=5), "num_layers": "3", "standardization": "0", "fix_capacity": "1", "model": "NN", "learning_rate": "0.1", "hidden_units": "1000.0",
                        "normalization": "0"},
            ("5000",): {"policy": "policy_CB", "extended_horizon": timedelta(minutes=5), "num_layers": "3", "standardization": "100", "fix_capacity": "1", "model": "NN", "learning_rate": "0.1", "hidden_units": "1000.0",
                        "normalization": "0"}}},
        running_type="vehicleTestHyperparameter", running_type_validation="vehicleTestHyperparameter", metric="profit", result_directory="results/results_deeplearning")

    # THIS BRINGS AMOUNT SATISFIED CUSTOMERS RESULTS
    visualize_results(conditions={
        "condition_general": {"objective": "amount_satisfied_customers", "num_vehicles": ["500", "1000", "2000", "3000", "4000", "5000"], "heuristic_time_range": "720.0", "heuristic_distance_range": "1.5",
                              "sparsity_factor": "0.3"},
        "condition_greedy": {"policy": "greedy"},
        "condition_sampling": {"policy": "sampling", "extended_horizon": timedelta(minutes=5), "reduce_weight_of_future_requests_factor": "0.2"},
        "condition_offline": {"policy": "offline"},
        "condition_SB_Linear": {"policy": "policy_SB", "extended_horizon": timedelta(minutes=5), "standardization": "0"},
        "condition_CB_Linear": {"policy": "policy_CB", "extended_horizon": timedelta(minutes=5), "standardization": "100", "fix_capacity": "1"}},
        running_type="vehicleTest", metric="amount_satisfied_customers",
        result_directory="results/results")

    # THIS BRINGS SPARSITY RESULTS
    visualize_results(conditions={
        "condition_general": {"objective": "profit", "num_vehicles": "2000", "heuristic_time_range": "720.0", "heuristic_distance_range": "1.5",
                              "sparsity_factor": ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "1.0"]},
        "condition_greedy": {"policy": "greedy"},
        "condition_sampling": {"policy": "sampling", "extended_horizon": timedelta(minutes=5), "reduce_weight_of_future_requests_factor": "0.2"},
        "condition_offline": {"policy": "offline"},
        "condition_SB_Linear": {"policy": "policy_SB", "extended_horizon": timedelta(minutes=5), "standardization": "0"},
        "condition_CB_Linear": {"policy": "policy_CB", "extended_horizon": timedelta(minutes=5), "standardization": "100", "fix_capacity": "1"}},
        running_type="sparsityTest", metric="profit",
        result_directory="results/results")

    ### EVALUATE DIFFERENT METRICS
    visualize_results(conditions={
        "condition_general": {"objective": "profit", "num_vehicles": ["500", "1000", "2000", "3000", "4000", "5000"], "heuristic_time_range": "720.0", "heuristic_distance_range": "1.5", "sparsity_factor": "0.3"},
        "condition_greedy": {"policy": "greedy"},
        "condition_sampling": {"policy": "sampling", "extended_horizon": timedelta(minutes=5), "reduce_weight_of_future_requests_factor": "0.2"},
        "condition_offline": {"policy": "offline"},
        "condition_SB_Linear": {"policy": "policy_SB", "extended_horizon": timedelta(minutes=5), "standardization": "0"},
        "condition_CB_Linear": {"policy": "policy_CB", "extended_horizon": timedelta(minutes=5), "standardization": "100", "fix_capacity": "1"}},
                      running_type="vehicleTest", metric="profit", result_directory="results/results", show_metric="service_ratio")

    visualize_results(conditions={
        "condition_general": {"objective": "profit", "num_vehicles": ["500", "1000", "2000", "3000", "4000", "5000"], "heuristic_time_range": "720.0", "heuristic_distance_range": "1.5", "sparsity_factor": "0.3"},
        "condition_greedy": {"policy": "greedy"},
        "condition_sampling": {"policy": "sampling", "extended_horizon": timedelta(minutes=5), "reduce_weight_of_future_requests_factor": "0.2"},
        "condition_offline": {"policy": "offline"},
        "condition_SB_Linear": {"policy": "policy_SB", "extended_horizon": timedelta(minutes=5), "standardization": "0"},
        "condition_CB_Linear": {"policy": "policy_CB", "extended_horizon": timedelta(minutes=5), "standardization": "100", "fix_capacity": "1"}},
                      running_type="vehicleTest", metric="profit", result_directory="results/results", show_metric="km_per_request")

    visualize_results(conditions={
        "condition_general": {"objective": "profit", "num_vehicles": ["500", "1000", "2000", "3000", "4000", "5000"], "heuristic_time_range": "720.0", "heuristic_distance_range": "1.5", "sparsity_factor": "0.3"},
        "condition_greedy": {"policy": "greedy"},
        "condition_sampling": {"policy": "sampling", "extended_horizon": timedelta(minutes=5), "reduce_weight_of_future_requests_factor": "0.2"},
        "condition_offline": {"policy": "offline"},
        "condition_SB_Linear": {"policy": "policy_SB", "extended_horizon": timedelta(minutes=5), "standardization": "0"},
        "condition_CB_Linear": {"policy": "policy_CB", "extended_horizon": timedelta(minutes=5), "standardization": "100", "fix_capacity": "1"}},
                      running_type="vehicleTest", metric="profit", result_directory="results/results", show_metric="avg_distance_trav")

    ### SHOW LEARNED PARAMETERS
    # SLCA
    """extract_learned_parameters(conditions={"condition_general": {"objective": "profit", "num_vehicles": ["2000"], "heuristic_time_range": "720.0", "heuristic_distance_range": "1.5", "sparsity_factor": "0.3"},
                                  "condition_CB": {"policy": "policy_CB", "extended_horizon": timedelta(minutes=5), "standardization": "100", "fix_capacity": "1"}},
                               running_type="vehicleTest", metric="profit", directory_parameters="./learned_parameters/learned_parameters_policy_CB_2000_0.3_profit_0:05:00_100/Learning_outer-obj-profit_hoSi-5.0_fleetSi-2000_pert-50.0,1.0")"""

    # SLPA
    """extract_learned_parameters(conditions={"condition_general": {"objective": "profit", "num_vehicles": ["2000"], "heuristic_time_range": "720.0", "heuristic_distance_range": "1.5", "sparsity_factor": "0.3"},
                                  "condition_SB": {"policy": "policy_SB", "extended_horizon": timedelta(minutes=5), "standardization": "0"}},
                               running_type="vehicleTest", metric="profit", directory_parameters="./learned_parameters/learned_parameters_policy_SB_2000_0.3_profit_0:05:00_0/Learning_outer-obj-profit_hoSi-5.0_fleetSi-2000_pert-50.0,1.0")"""

    """show_learning_evolution(parameter_directory="./learned_parameters/learned_parameters_policy_SB_1000_0.3_profit_0:05:00_0/",
                            inner_evaluation="Learning_inner-obj-profit_hoSi-5.0_fleetSi-1000_pert-50.0,1.0",
                            outer_evaluation="Learning_outer-obj-profit_hoSi-5.0_fleetSi-1000_pert-50.0,1.0")"""

    """show_learning_evolution(parameter_directory="./learned_parameters/learned_parameters_policy_CB_500_0.3_profit_0:05:00_0/",
        inner_evaluation="Learning_inner-obj-profit_hoSi-5.0_fleetSi-500_pert-10.0,1.0_m-Linear",
        outer_evaluation="Learning_outer-obj-profit_hoSi-5.0_fleetSi-500_pert-10.0,1.0_m-Linear")"""

    """show_learning_evolution(parameter_directory="./learned_parameters_20231213/learned_parameters_policy_CB_500_0.3_profit_0:05:00_100/",
                            inner_evaluation="Learning_inner-obj-profit_hoSi-5.0_fleetSi-500_pert-50.0,1.0_m-Linear",
                            outer_evaluation="Learning_outer-obj-profit_hoSi-5.0_fleetSi-500_pert-50.0,1.0_m-Linear")"""

    """show_learning_evolution(parameter_directory=f"./learned_parameters_20240212/learned_parameters_policy_SB_4000_0.3_profit_0:05:00_0/",
                            inner_evaluation=f"Learning-obj-profit_hoSi-5.0_fleetSi-4000_pert-50.0,1.0_m-NN_lr-0.1_hu-1000.0_nrm-0_nl-3", outer_evaluation=None)"""

    # SANITY CHECK
    sanity_check(sanity_directory="./results/sanity_check/", policy="policy_CB")
    sanity_check(sanity_directory="./results/sanity_check/", policy="policy_SB")

    sanity_check_performance(sanity_performance_directory="./results/sanity_check_performance/", policy="policy_CB")
    sanity_check_performance(sanity_performance_directory="./results/sanity_check_performance/", policy="policy_SB")