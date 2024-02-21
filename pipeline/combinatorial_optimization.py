import time
from src import cplusplus_interface


class CombinatorialOptimizer:
    def __init__(self, args):
        self.args = args
        self.cplusplus_optimizer = cplusplus_interface.CombinatorialOptimizier()

    def calculate_k_disjoint_shortest_path(self, numVertices, edges, weights, numVehicles):
        time_start_k_disjoint = time.time()
        if self.args.policy == "policy_CB":
            k, edges = self.cplusplus_optimizer.callrunKEdgeDisjoint(numVertices=numVertices, edges=edges, weights=weights, numVehicles=numVehicles)
            edges = [edge for trip in edges for edge in trip]
        else:
            k, edges = self.cplusplus_optimizer.callrunKDisjoint(numVertices=numVertices, edges=edges, weights=weights, numVehicles=numVehicles)
        time_end_k_disjoint = time.time()
        return k, edges, time_end_k_disjoint - time_start_k_disjoint

    def solve_optimization_problem(self, edges, weights, num_vertices, num_k):
        if self.args.mode not in ["create_training_instance"]:
            k, edges, time_needed_k_disjoint = self.calculate_k_disjoint_shortest_path(numVertices=num_vertices, edges=edges, weights=weights, numVehicles=num_k)
            return k, edges, time_needed_k_disjoint
        else:
            return 0, 0, 0
