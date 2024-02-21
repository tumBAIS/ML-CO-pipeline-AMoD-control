
class Decoder():
    def __init__(self, args):
        self.args = args

    def get_graph(self, edges, num_vehicles, len_verticesList, idx_bins):
        # vehicleTrips: for each vehicle a list with the succeeding vertices in a row (shortest path)
        # each route begins with the source
        vehicleTrips = [[0] for i in range(num_vehicles)]
        # for each vertex (vehicle or request) a mapping to the respective vehicle that completes the vertex
        verticesIndexVehicleMapping = [-1] * len_verticesList
        edges.sort(key=lambda tup: tup[0])
        sink_index = idx_bins["artCapVertices"][1]

        # get the vehicle that is responsible for fulfilling a vertex
        def getVehicle(vertex):
            return verticesIndexVehicleMapping[vertex]

        # set the vehicle that is responsible for fulfilling a vertex
        # add the vertex to the vehicle trip
        def setVehicle(vehicle, vertex):
            verticesIndexVehicleMapping[vertex] = vehicle
            vehicleTrips[vehicle - 1].append(vertex)

        # run through edges and assign them to connected trips / routes
        for edge in edges:
            if self.args.policy == "policy_CB":  # in CB policy several trips can share the same rebalancing vertex. To ease the process we finish a trip at the rebalancing vertex.
                if (edge[0] in range(idx_bins["artCapVertices"][0], idx_bins["artCapVertices"][1])) or (edge[1] in range(idx_bins["artCapVertices"][0], idx_bins["artCapVertices"][1])):
                    continue
            # begin of route
            if edge[0] == 0:
                setVehicle(edge[1], edge[1])
            # not begin of route
            else:
                v = getVehicle(edge[0])
                setVehicle(vehicle=v, vertex=edge[1])
                if self.args.policy == "policy_CB":
                    if edge[1] in range(idx_bins["artRebVertices"][0], idx_bins["artRebVertices"][1]):
                        setVehicle(vehicle=v, vertex=sink_index)
        # return the routes (shortest path) for each vehicle
        # source is 0, sink is len(vertices) - 1
        # Car1: [1] -> [0,    1,  301,  1520, 5202]
        #        ^      ^     ^    ^     ^      ^
        #        v_1  source v_1 r_301 r_1520  sink
        # Car2: [2] -> [0,    2,    532,  3543,   4567,  5202]
        #        ^      ^     ^      ^      ^      ^      ^
        #       v_2   source v_2, r_532, r_3543, r_4567, sink
        # ...
        # return a mapping between each vehicle and vertex. If a vehicle and a vertex are mapped, the vertex is also in
        # the trip (shortest path) of the vehicle
        return vehicleTrips, verticesIndexVehicleMapping
