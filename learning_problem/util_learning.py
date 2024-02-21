import os
import pickle
from src import util
import json
import numpy as np


def get_training_instances(args):
    training_instances = []
    for instance in os.listdir(args.training_instance_directory):
        if instance.find("TrainingInstance") == -1 or instance.find(args.objective) == -1:
            continue
        training_instances.append(instance)
    return training_instances


def get_data(args, instance_file):
    file_input = open(args.training_instance_directory + instance_file, "rb")
    instance_date = instance_file.split("instDate:")[-1].split("_")[0]
    print("Instance_Date: {} ".format(instance_date))
    training_instance = pickle.load(file_input)
    file_input.close()
    return training_instance


def calculate_accuracy():
    return 0


def calculate_objective_value(edgesList, weightsList, solution_edges):
    weights = np.array(weightsList)[[edge in solution_edges for edge in edgesList]]
    assert len(weights) == len(solution_edges)
    return -1 * sum(weights)


def get_learnable_edges(edges):
    vehicles_requests_idxs = [0, len(edges["vehicles_requests"])]
    requests_requests_idxs = [vehicles_requests_idxs[1], vehicles_requests_idxs[1] + len(edges["requests_requests"])]
    vehicles_artRebVertices_idxs = [requests_requests_idxs[1], requests_requests_idxs[1] + len(edges["vehicles_artRebVertices"])]
    requests_artRebVertices_idxs = [vehicles_artRebVertices_idxs[1], vehicles_artRebVertices_idxs[1] + len(edges["requests_artRebVertices"])]
    artRebVertices_artCapVertices_idxs = [requests_artRebVertices_idxs[1], requests_artRebVertices_idxs[1] + len(edges["artRebVertices_artCapVertices"])]
    return (edges["vehicles_requests"] + edges["requests_requests"] + edges["vehicles_artRebVertices"] + edges["requests_artRebVertices"] + edges["artRebVertices_artCapVertices"],
            {"vehicles_requests": vehicles_requests_idxs,
             "requests_requests": requests_requests_idxs,
             "vehicles_artRebVertices": vehicles_artRebVertices_idxs,
             "requests_artRebVertices": requests_artRebVertices_idxs,
             "artRebVertices_artCapVertices": artRebVertices_artCapVertices_idxs})


def get_edges(args, training_instance):
    edges_source_vehicles = training_instance["edges_source_vehicles"]
    edges_vehicles_requests = [tuple(edge) for edge in training_instance["features_edges_vehicles_requests"][["starting", "arriving"]].values.tolist()]
    edges_requests_requests = [tuple(edge) for edge in training_instance["features_edges_requests_requests"][["starting", "arriving"]].values.tolist()] if len(training_instance["features_edges_requests_requests"]) > 0 else []
    edges_vehicles_artRebVertices = [tuple(edge) for edge in training_instance["features_edges_vehicles_artRebVertices"][["starting", "arriving"]].values.tolist()] if len(training_instance["features_edges_vehicles_artRebVertices"]) > 0 else []
    edges_requests_artRebVertices = [tuple(edge) for edge in training_instance["features_edges_requests_artRebVertices"][["starting", "arriving"]].values.tolist()] if len(training_instance["features_edges_requests_artRebVertices"]) > 0 else []
    edges_artRebVertices_artCapVertices = [tuple(edge) for edge in training_instance["features_edges_artRebVertices_artCapVertices"][["starting", "arriving"]].values.tolist()] if args.policy == "policy_CB" else []
    edges_vehicles_sink = training_instance["edges_vehicles_sink"]
    edges_requests_sink = training_instance["edges_requests_sink"]
    edges_artLocVertices_sink = training_instance["edges_artLocVertices_sink"]

    return {"source_vehicles": edges_source_vehicles,
            "vehicles_requests": edges_vehicles_requests,
            "requests_requests": edges_requests_requests,
            "vehicles_artRebVertices": edges_vehicles_artRebVertices,
            "requests_artRebVertices": edges_requests_artRebVertices,
            "artRebVertices_artCapVertices": edges_artRebVertices_artCapVertices,
            "vehicles_sink": edges_vehicles_sink,
            "requests_sink": edges_requests_sink,
            "artLocVertices_sink": edges_artLocVertices_sink}


def save_evolution_learning(args, evolution_dict, step=None):
    file_output = open(util.get_learning_file(args, step), "w")
    json.dump(evolution_dict, file_output)
    file_output.close()


def decode_solution(solution_edges, learnable_edges):
    return [edge in solution_edges for edge in learnable_edges]
