#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <algorithm>
#include <vector>

#include <kdisjoint.hpp>
#include <kedgedisjoint.hpp>

// calculate the distance between point 1 and point 2 on earht
static float haversineFloat(float lat1, float lon1,
                            float lat2, float lon2)
{
    // distance between latitudes
    // and longitudes
    float dLat = (lat2 - lat1) *
                 M_PI / 180.0;
    float dLon = (lon2 - lon1) *
                 M_PI / 180.0;

    // convert to radians
    lat1 = (lat1) * M_PI / 180.0;
    lat2 = (lat2) * M_PI / 180.0;

    // apply formulae
    float a = pow(sin(dLat / 2), 2) +
              pow(sin(dLon / 2), 2) *
              cos(lat1) * cos(lat2);
    float rad = 6371;
    float c = 2 * asin(sqrt(a));
    return rad * c;
}


struct InterfaceGraphGenerator{
    using Edge = std::pair<unsigned int, unsigned int>;

    // STATIC
    std::vector<std::vector<double>> look_up_table_distance;
    std::vector<std::vector<double>> look_up_table_duration;
    float average_speed_secPERkm;
    float heuristic_distance_range = 0;
    float heuristic_time_range = 0;

    // DYNAMIC
    std::vector<std::vector<float>> request_vertices;
    std::vector<std::vector<float>> vehicle_vertices;
    std::vector<std::vector<float>> artificial_vertices;
    int col_look_up_index_vehicle;
    int col_look_up_index_request_pickup;
    int col_look_up_index_request_dropoff;
    int col_look_up_index_artificialVertex_pickup;
    int num_vehicles = 0;
    int num_requests = 0;
    int num_artificialVertices = 0;
    int col_time_vehicle;
    int col_time_request_pickup;
    int col_time_request_dropoff;
    int col_time_artificialVertex_pickup;
    int col_lon_request_pickup;
    int col_lat_request_pickup;
    int col_lon_request_dropoff;
    int col_lat_request_dropoff;
    int col_lon_vehicle_destination;
    int col_lat_vehicle_destination;
    int col_lon_artificialVertex_pickup;
    int col_lat_artificialVertex_pickup;


    std::tuple<std::vector<Edge>, std::vector<float>, std::vector<float>> calculateEdgesVehiclesRides(){
        return calculateEdges(vehicle_vertices, request_vertices,
                              col_look_up_index_vehicle, col_look_up_index_request_pickup,
                              1, 1 + num_vehicles, 1 + num_vehicles, 1 + num_vehicles + num_requests,
                              col_time_vehicle, col_time_request_pickup,
                              col_lon_vehicle_destination, col_lat_vehicle_destination, col_lon_request_pickup, col_lat_request_pickup);
    }

    std::tuple<std::vector<Edge>, std::vector<float>, std::vector<float>> calculateEdgesRidesRides(){
        return calculateEdges(request_vertices, request_vertices,
                              col_look_up_index_request_dropoff, col_look_up_index_request_pickup,
                              1 + num_vehicles, 1 + num_vehicles + num_requests, 1 + num_vehicles, 1 + num_vehicles + num_requests,
                              col_time_request_dropoff, col_time_request_pickup,
                              col_lon_request_dropoff, col_lat_request_dropoff, col_lon_request_pickup, col_lat_request_pickup);
    }

    std::tuple<std::vector<Edge>, std::vector<float>, std::vector<float>> caluclateEdgesRidesArtificialVertices(){
        return calculateEdges(request_vertices, artificial_vertices,
                              col_look_up_index_request_dropoff, col_look_up_index_artificialVertex_pickup,
                              1 + num_vehicles, 1 + num_vehicles + num_requests, 1 + num_vehicles + num_requests, 1 + num_vehicles + num_requests + num_artificialVertices,
                              col_time_request_dropoff, col_time_artificialVertex_pickup,
                              col_lon_request_dropoff, col_lat_request_dropoff, col_lon_artificialVertex_pickup, col_lat_artificialVertex_pickup);
    }

    std::tuple<std::vector<Edge>, std::vector<float>, std::vector<float>> caluclateEdgesVehiclesArtificialVertices(){
        return calculateEdges(vehicle_vertices, artificial_vertices,
                              col_look_up_index_vehicle, col_look_up_index_artificialVertex_pickup,
                              1, 1 + num_vehicles, 1 + num_vehicles + num_requests, 1 + num_vehicles + num_requests + num_artificialVertices,
                              col_time_vehicle, col_time_artificialVertex_pickup,
                              col_lon_vehicle_destination, col_lat_vehicle_destination, col_lon_artificialVertex_pickup, col_lat_artificialVertex_pickup);
    }


    std::tuple<std::vector<Edge>, std::vector<float>, std::vector<float>> calculateEdges(
            std::vector<std::vector<float>> startingVertices, std::vector<std::vector<float>> arrivingVertices,
            int look_up_index_starting, int look_up_index_arriving,
                        int starting_idx_begin, int starting_idx_end, int arriving_idx_begin, int arriving_idx_end,
                        int idx_starting_time, int idx_arriving_time,
                        int col_lon_start, int col_lat_start, int col_lon_end, int col_lat_end){
        // startingVertices : list of list with features of starting vertices
        // arrivingVertices : list of list with features of ending vertices
        // look_up_index_starting : index in look-up-table where edge starts
        // look_up_index_arriving : index in look-up-table where edge ends
        // starting_idx_begin : first index of starting vertices in vertices list
        // starting_idx_end : last index of starting vertices in vertices list
        // arriving_idx_begin : first index of arriving vertices in vertices list
        // arriving_idx_end : last index of arriving vertices in vertices list
        // idx_starting_time : column index of starting time in starting vertices
        // idx_arriving_time : column index of arriving time in arriving vertices
        std::vector<Edge> edges;
        std::vector<float> distances;
        std::vector<float> durations;
        for (int i=starting_idx_begin; i<starting_idx_end ;i++){
            for(int j=arriving_idx_begin; j<arriving_idx_end; j++){
                int starting_id = startingVertices[i - starting_idx_begin][look_up_index_starting];
                int destination_id = arrivingVertices[j - arriving_idx_begin][look_up_index_arriving];
                float distance_km = look_up_table_distance[starting_id][destination_id];
                float duration_sec = look_up_table_duration[starting_id][destination_id];
                    if (distance_km == 0 || duration_sec == 0) {
                        distance_km = haversineFloat(startingVertices[i - starting_idx_begin][col_lat_start],
                                                     startingVertices[i - starting_idx_begin][col_lon_start],
                                                     arrivingVertices[j - arriving_idx_begin][col_lat_end],
                                                     arrivingVertices[j - arriving_idx_begin][col_lon_end]);
                        duration_sec = distance_km * average_speed_secPERkm;
                    }
                float time_puffer = arrivingVertices[j - arriving_idx_begin][idx_arriving_time] - startingVertices[i - starting_idx_begin][idx_starting_time];
                if (distance_km < heuristic_distance_range && time_puffer < heuristic_time_range && duration_sec < time_puffer && time_puffer > 0) {
                // add the edge
                edges.emplace_back(std::make_pair(i, j));
                distances.emplace_back(distance_km);
                durations.emplace_back(duration_sec);
                }
            }
        }
        return std::make_tuple(edges, distances, durations);
    }
};



// run the k-disjoint shortest path algorithm from Suurballe
struct InstanceKDisjoint{
    unsigned int num_vertices = 0;
    int num_vehicles = 0;
    int k = 0;
    using Edge = std::pair<unsigned int, unsigned int>;
    std::vector<Edge> edges;
    std::vector<Edge> paths;
    std::vector<std::vector<Edge>> paths_edge_disjoint;
    std::vector<float> weights;

    void runKDisjoint(){
        auto return_value = kdisjoint_shortest_path<unsigned int, float>(num_vertices, edges, weights, num_vehicles);
        k = return_value.first;
        paths = return_value.second;
    }

    void runKEdgeDisjoint(){
        auto return_value = kedgedisjoint_shortest_path<unsigned int, float>(num_vertices, edges, weights, num_vehicles);
        k = return_value.first;
        paths_edge_disjoint = return_value.second;
    }

};



// calculate loss of solution
struct InstanceCalculateLoss{
    float profit = 0;
    using Edge = std::pair<unsigned int, unsigned int>;
    std::vector<Edge> edges;
    std::vector<Edge> all_edges_online;
    std::vector<float> weights;
    std::vector<int> combinatorial_solution;

    int getIndex(Edge v[], Edge K){
        auto it = find(v, v+all_edges_online.size(), K);
        int index = 0;
        if (it != v+all_edges_online.size()){
            index = std::distance(v, it);
        }
        else {
            index = -1;
        }
        return index;
    }

    void runcalculateLoss(){
        Edge all_edges_array[all_edges_online.size()];
        std::copy(all_edges_online.begin(), all_edges_online.end(), all_edges_array);
        for(const auto& edge: edges){
            int index = getIndex(all_edges_array, edge);
            if (index == -1){
                printf("The edge %u --> %u not has been found in all online edges \n", edge.first, edge.second);
            }
            assert (index != -1);
            //if (index == -1){
            //    printf ("(%i, %i)", std::get<0>(edge), std::get<1>(edge));
            //}
            profit = profit + weights[index];
            combinatorial_solution.push_back(index);
        }
    }
};

// calculate the cost smooth feautres
struct InstanceCalculateSmoothCostFeautres{
    std::vector<std::vector<float>> base_features;
    int granularity_length;
    int time_span_sec;
    float mean_travel_distancePerTime;
    float costs_per_km;
    std::string type;
    std::vector<float> granularity_counter;
    std::vector<std::vector<float>> granularity_array;


    void calculate_cost_smooth_feautres(){
        for (std::vector<float> base_feature_row: base_features) {
            std::vector<float> granularity_row = calculate_cost_smooth_feature(base_feature_row[0],
                                                                               base_feature_row[1],
                                                                               base_feature_row[2],
                                                                               base_feature_row[3],
                                                                               base_feature_row[4]);
            granularity_array.push_back(granularity_row);
        }
    }


    std::vector<float> calculate_cost_smooth_feature(int duration_sec_ride, float distance_km_ride, float distance_km, int duration_sec, int pickup_time){
        std::vector<float> granularity_row(granularity_length, 0.0);
        // here we have to distinguish between two cases:
        // 1. dropoff > actual_date_old -> in this case we calculate when the vehicle has to start driving and discount the costs from thereafter
        //if (dropoff_time_out > actual_date_old){
        //    dropoff_time = pickup_time - duration_sec;
        //}
        // 2. dropoff <= actual date_old -> in this case we consider the costs that 1.1 remain / 1.2 accumulate on the remaining time
        //if (dropoff_time_out <= actual_date_old){
        int dropoff_time = pickup_time - duration_sec;
        if (dropoff_time < 0) {
            dropoff_time = 0;
        }
        //}

        int index_dropoff = int(dropoff_time / time_span_sec) + 1;
        int index_pickup = int(pickup_time / time_span_sec) + 1;
        float percentage_within_index_begin = (granularity_counter[index_dropoff] - dropoff_time) / time_span_sec;
        float percentage_within_index_end = (pickup_time - granularity_counter[index_pickup-1]) / time_span_sec;
        if ((dropoff_time - pickup_time) == 0){
            granularity_row[0] = -1 * distance_km * 100 * costs_per_km;
        }
        else {
            float costs_per_seconds = (distance_km * 100 * costs_per_km) / (pickup_time - dropoff_time);
            if (index_pickup > 1){
                granularity_row[index_dropoff - 1] = -1 * costs_per_seconds * percentage_within_index_begin * time_span_sec;
                for(int i = index_dropoff; i<index_pickup-1; ++i)
                    granularity_row[i]= -1 * costs_per_seconds * time_span_sec;
            }
            granularity_row[index_pickup - 1] = -1 * costs_per_seconds * percentage_within_index_end * time_span_sec;
        }
        int dropoff_time_ride = pickup_time + duration_sec_ride;
        int index_dropoff_ride = int(dropoff_time_ride / time_span_sec) + 1;
        float percentage_within_index_begin_ride = (granularity_counter[index_pickup] - pickup_time) / time_span_sec;
        float total_amount_per_second = -1 * (distance_km_ride * 100 * costs_per_km) / (dropoff_time_ride - pickup_time);
        if (index_dropoff_ride >= granularity_length){
            index_dropoff_ride = granularity_length;
            granularity_row[index_pickup-1] = total_amount_per_second * percentage_within_index_begin_ride * time_span_sec;
            for (int i = index_pickup; i<index_dropoff_ride; ++i){
                granularity_row[i] = total_amount_per_second * time_span_sec;
            }
        }
        else {
            float percentage_within_index_end_ride = (dropoff_time_ride - granularity_counter[index_dropoff_ride-1]) / time_span_sec;
            if (index_dropoff_ride > 1) {
                granularity_row[index_pickup - 1] = total_amount_per_second * percentage_within_index_begin_ride * time_span_sec;
                for (int i = index_pickup; i<index_dropoff_ride - 1; ++i){
                    granularity_row[i] = total_amount_per_second * time_span_sec;
                }
                granularity_row[index_dropoff_ride - 1] = total_amount_per_second * percentage_within_index_end_ride * time_span_sec;
            }
            if (type == "Location"){
                granularity_row[index_dropoff_ride - 1] = - time_span_sec * mean_travel_distancePerTime  * costs_per_km * 100 * (1-percentage_within_index_end_ride) + total_amount_per_second * percentage_within_index_end_ride * time_span_sec;
                for (int i = index_dropoff_ride; i<granularity_length; ++i){
                    granularity_row[i] = - time_span_sec * mean_travel_distancePerTime * costs_per_km * 100;
                }
            }
        }
        return granularity_row;
    }
};


PYBIND11_MODULE(pybind11module, module) {


    // initialize problem instance - calculate map from offline space into online space
    pybind11::class_<InterfaceGraphGenerator>(module, "InterfaceGraphGenerator")
            .def(pybind11::init<>())
                    // VARIABLES
            .def_readwrite("look_up_table_distance", &InterfaceGraphGenerator::look_up_table_distance)
            .def_readwrite("look_up_table_duration", &InterfaceGraphGenerator::look_up_table_duration)
            .def_readwrite("average_speed_secPERkm", &InterfaceGraphGenerator::average_speed_secPERkm)
            .def_readwrite("heuristic_distance_range", &InterfaceGraphGenerator::heuristic_distance_range)
            .def_readwrite("heuristic_time_range", &InterfaceGraphGenerator::heuristic_time_range)
            .def_readwrite("request_vertices", &InterfaceGraphGenerator::request_vertices)
            .def_readwrite("vehicle_vertices", &InterfaceGraphGenerator::vehicle_vertices)
            .def_readwrite("artificial_vertices", &InterfaceGraphGenerator::artificial_vertices)
            .def_readwrite("col_look_up_index_vehicle", &InterfaceGraphGenerator::col_look_up_index_vehicle)
            .def_readwrite("col_look_up_index_request_pickup",
                           &InterfaceGraphGenerator::col_look_up_index_request_pickup)
            .def_readwrite("col_look_up_index_request_dropoff",
                           &InterfaceGraphGenerator::col_look_up_index_request_dropoff)
            .def_readwrite("col_look_up_index_artificialVertex_pickup",
                           &InterfaceGraphGenerator::col_look_up_index_artificialVertex_pickup)
            .def_readwrite("num_vehicles", &InterfaceGraphGenerator::num_vehicles)
            .def_readwrite("num_requests", &InterfaceGraphGenerator::num_requests)
            .def_readwrite("num_artificialVertices", &InterfaceGraphGenerator::num_artificialVertices)
            .def_readwrite("col_time_vehicle", &InterfaceGraphGenerator::col_time_vehicle)
            .def_readwrite("col_time_request_pickup", &InterfaceGraphGenerator::col_time_request_pickup)
            .def_readwrite("col_time_request_dropoff", &InterfaceGraphGenerator::col_time_request_dropoff)
            .def_readwrite("col_time_artificialVertex_pickup", &InterfaceGraphGenerator::col_time_artificialVertex_pickup)
            .def_readwrite("col_lon_request_pickup", &InterfaceGraphGenerator::col_lon_request_pickup)
            .def_readwrite("col_lat_request_pickup", &InterfaceGraphGenerator::col_lat_request_pickup)
            .def_readwrite("col_lon_request_dropoff", &InterfaceGraphGenerator::col_lon_request_dropoff)
            .def_readwrite("col_lat_request_dropoff", &InterfaceGraphGenerator::col_lat_request_dropoff)
            .def_readwrite("col_lon_vehicle_destination", &InterfaceGraphGenerator::col_lon_vehicle_destination)
            .def_readwrite("col_lat_vehicle_destination", &InterfaceGraphGenerator::col_lat_vehicle_destination)
            .def_readwrite("col_lon_artificialVertex_pickup", &InterfaceGraphGenerator::col_lon_artificialVertex_pickup)
            .def_readwrite("col_lat_artificialVertex_pickup", &InterfaceGraphGenerator::col_lat_artificialVertex_pickup)
            //.def_readwrite("actualTime", &InterfaceGraphGenerator::actualTime)
                    // FUNCTIONS
            .def("calculateEdgesVehiclesRides", &InterfaceGraphGenerator::calculateEdgesVehiclesRides)
            .def("calculateEdgesRidesRides", &InterfaceGraphGenerator::calculateEdgesRidesRides)
            .def("caluclateEdgesRidesArtificialVertices", &InterfaceGraphGenerator::caluclateEdgesRidesArtificialVertices)
            .def("caluclateEdgesVehiclesArtificialVertices", &InterfaceGraphGenerator::caluclateEdgesVehiclesArtificialVertices);


    // initialize problem instance
    pybind11::class_<InstanceKDisjoint>(module, "InstanceKDisjoint")
            .def(pybind11::init<>())
            .def_readwrite("num_vertices", &InstanceKDisjoint::num_vertices)
            .def_readwrite("num_vehicles", &InstanceKDisjoint::num_vehicles)
            .def_readwrite("num_paths", &InstanceKDisjoint::k)
            .def_readwrite("paths", &InstanceKDisjoint::paths)
            .def_readwrite("paths_edge_disjoint", &InstanceKDisjoint::paths_edge_disjoint)
            .def_readwrite("weights", &InstanceKDisjoint::weights)
            .def_readwrite("edges", &InstanceKDisjoint::edges)
            .def("runKEdgeDisjoint", &InstanceKDisjoint::runKEdgeDisjoint)
            .def("runKDisjoint", &InstanceKDisjoint::runKDisjoint);


    // calculate the loss value of a solution
    pybind11::class_<InstanceCalculateLoss>(module, "InstanceCalculateLoss")
            .def(pybind11::init<>())
            .def_readwrite("profit", &InstanceCalculateLoss::profit)
            .def_readwrite("edges", &InstanceCalculateLoss::edges)
            .def_readwrite("all_edges_online", &InstanceCalculateLoss::all_edges_online)
            .def_readwrite("weights", &InstanceCalculateLoss::weights)
            .def_readwrite("combinatorial_solution", &InstanceCalculateLoss::combinatorial_solution)
            .def("runcalculateLoss", &InstanceCalculateLoss::runcalculateLoss);


    // calculate the cost smooth features
    pybind11::class_<InstanceCalculateSmoothCostFeautres>(module, "InstanceCalculateSmoothCostFeautres")
            .def(pybind11::init<>())
            .def_readwrite("base_features", &InstanceCalculateSmoothCostFeautres::base_features)
            .def_readwrite("granularity_length", &InstanceCalculateSmoothCostFeautres::granularity_length)
            .def_readwrite("time_span_sec", &InstanceCalculateSmoothCostFeautres::time_span_sec)
            .def_readwrite("mean_travel_distancePerTime", &InstanceCalculateSmoothCostFeautres::mean_travel_distancePerTime)
            .def_readwrite("costs_per_km", &InstanceCalculateSmoothCostFeautres::costs_per_km)
            .def_readwrite("type", &InstanceCalculateSmoothCostFeautres::type)
            .def_readwrite("granularity_counter", &InstanceCalculateSmoothCostFeautres::granularity_counter)
            .def_readwrite("granularity_array", &InstanceCalculateSmoothCostFeautres::granularity_array)
            .def("calculate_cost_smooth_feautres", &InstanceCalculateSmoothCostFeautres::calculate_cost_smooth_feautres);

}