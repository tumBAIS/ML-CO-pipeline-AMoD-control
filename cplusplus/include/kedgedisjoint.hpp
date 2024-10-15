#pragma once

#include <boost/dynamic_bitset.hpp>
#include <boost/range/irange.hpp>
#include <kdisjoint.hpp>
#include <limits>
#include <queue>
#include <deque>
#include <iostream>
#include <lemon/suurballe.h>
#include <lemon/list_graph.h>




template<class Vertex, class Weight>
auto kedgedisjoint_shortest_path(Vertex num_vertices, std::vector<std::pair<Vertex, Vertex>> &_edges,
                                 std::vector<Weight> &_weights, int number_shortest_path) -> std::pair<int, std::vector<std::vector<std::pair<Vertex, Vertex>>>>{ //std::vector<std::pair<Vertex, Vertex>>> { //std::vector<lemon::ListPath<lemon::ListDigraph>>>{ //

CSRGraph graph(num_vertices, _edges, _weights);

std::vector<Vertex> p(graph.num_nodes());
boost::dynamic_bitset p_in_S(static_cast<std::size_t>(graph.num_nodes()));
std::vector<Vertex> t_p(graph.num_nodes());
std::vector<Vertex> t_s(graph.num_nodes());
std::vector<Weight> t_w(graph.num_nodes());
std::vector<Weight> d(graph.num_nodes());
for(auto j = 0; j < graph.num_nodes(); ++j) t_p[j] = j;

std::vector<Vertex> A(graph.num_nodes());
std::vector<Vertex> Z(graph.num_nodes());
std::vector<Weight> L(graph.num_nodes());

boost::dynamic_bitset Aj_in_Pm(static_cast<std::size_t>(graph.num_nodes()));

{
std::vector<Vertex> p_1(graph.num_nodes());
bellman_ford(graph, graph.source(), p_1, d);

{ // Reweight the edges to remove negatives
auto e_it = graph.edges_begin(), e_end = graph.edges_end();
for (; e_it != e_end; ++e_it) {
e_it->w = d[(*e_it).from] - d[(*e_it).to] + e_it->w;
}
}
}

// use lemon/suurballe.h
lemon::ListDigraph graph_lemon;
lemon::ListDigraph::NodeMap<Vertex> nodeToVertex(graph_lemon);
std::map<Vertex, int> VertexToId;
lemon::ListDigraph::ArcMap<float> weights(graph_lemon);
// add all vertices to graph_lemon
for (Vertex i : boost::irange(num_vertices)){
lemon::ListDigraph::Node node = graph_lemon.addNode();
VertexToId[i] = graph_lemon.id(node);
nodeToVertex[node] = i;
}

// add all edges to graph_lemon
// add all weights to graph_lemon
auto e_it = graph.edges_begin(), e_end = graph.edges_end();
for (; e_it != e_end; ++e_it) {
lemon::ListDigraph::Arc edge = graph_lemon.addArc(graph_lemon.nodeFromId(VertexToId[e_it->from]), graph_lemon.nodeFromId(VertexToId[e_it->to]));
weights[edge] = e_it->w;
}
// initialize Suurballe
lemon::Suurballe<lemon::ListDigraph, lemon::ListDigraph::ArcMap<float>> suurballe(graph_lemon, weights);

// calculate Suurballe
int succ_routes = suurballe.run(graph_lemon.nodeFromId(VertexToId[0]), graph_lemon.nodeFromId(VertexToId[num_vertices-1]), number_shortest_path);

// extract solution of Suurballe
std::vector<std::pair<Vertex, Vertex>> solution;
for (lemon::ListDigraph::ArcIt a(graph_lemon); a != lemon::INVALID; ++a){
if (suurballe.flow(a) == 1){
solution.emplace_back(std::make_pair(nodeToVertex[graph_lemon.source(a)], nodeToVertex[graph_lemon.target(a)]));
}
}

// extract paths from Suurballe
int num_paths = suurballe.pathNum();
std::vector<lemon::ListPath<lemon::ListDigraph>> paths;
std::vector<std::pair<Vertex, Vertex>> paths_solution;
std::vector<std::vector<std::pair<Vertex, Vertex>>> paths_solutions;
for (int i=0; i<num_paths; ++i){
paths.emplace_back(suurballe.path(i));
}
for (auto path : paths)
{
for (int i = 0; i < path.length(); i++)
{
paths_solution.emplace_back(std::make_pair(nodeToVertex[graph_lemon.source(path.nth(i))], nodeToVertex[graph_lemon.target(path.nth(i))]));
}
paths_solutions.emplace_back(paths_solution);
paths_solution.clear();
}


return std::make_pair(succ_routes, paths_solutions);
}
