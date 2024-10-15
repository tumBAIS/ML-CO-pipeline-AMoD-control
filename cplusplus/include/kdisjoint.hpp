#pragma once

#include <boost/dynamic_bitset.hpp>
#include <limits>
#include <queue>
#include <deque>
#include <iostream>


// This code is copied from the paper: A polynomial-time algorithm for user-based relocation in free-floating car sharing systems - https://doi.org/10.1016/j.trb.2020.11.001

/**
 * A compressed parse row graph implementation, allowing only to iterate over outgoing edges.
 * For a short intro to compressed parse row graphs:
 *  https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)
 *
 * @tparam Vertex
 * @tparam Weight
 */
template<class Vertex, class Weight>
class CSRGraph {
public:
    struct Edge {
        Vertex from;
        Vertex to;
        Weight w;

        Edge(Vertex _from, Vertex _to, Weight _w)
                : from(_from), to(_to), w(_w) {}
    };
private:
    Vertex num_vertices;
    Vertex source_node;
    Vertex sink_node;
    std::vector<Edge> edges;
    std::vector<std::size_t> fw_col_idx;
    std::vector<std::size_t> fw_row_ptr;
public:
    using edge_iterator = typename std::vector<Edge>::iterator;
    CSRGraph(Vertex _num_vertices, std::vector<std::pair<Vertex, Vertex>>& _edges, std::vector<Weight>& _weights);

    auto num_nodes() const -> Vertex { return num_vertices; }
    auto source() const -> Vertex { return source_node; }
    auto sink() const -> Vertex { return sink_node; }

    auto get_edge(Vertex origin, Vertex target) -> edge_iterator {
        auto end = fw_row_ptr[origin + 1];
        for(auto i = fw_row_ptr[origin]; i < end; ++i) {
            if(fw_col_idx[i] == target) {
                return edges.begin() + i;
            }
        }
        return edges.end();
    }

    auto out_edges(Vertex from) -> std::pair<edge_iterator, edge_iterator> {
        auto e_begin = edges.begin() + fw_row_ptr[from];
        auto e_end = edges.begin() + fw_row_ptr[from + 1];
        return std::make_pair(e_begin, e_end);
    }

    auto edges_begin() -> edge_iterator  {
        return edges.begin();
    }
    auto edges_end() -> edge_iterator {
        return edges.end();
    }
};

template<class Vertex, class Weight>
CSRGraph<Vertex, Weight>::CSRGraph(Vertex _num_vertices, std::vector<std::pair<Vertex, Vertex>>& _edges, std::vector<Weight>& _weights) {
    source_node = 0;
    sink_node = _num_vertices - 1;
    num_vertices = _num_vertices;

    edges.reserve(_edges.size());

    auto it_edge = _edges.begin();
    auto it_weight = _weights.begin();
    for (; it_edge != _edges.end(); ++it_edge, ++it_weight) {
        auto from = (*it_edge).first;
        auto to = (*it_edge).second;

        edges.emplace_back(from, to, *it_weight);
    }

    // now create the graph itself
    // first sort the edges first by their origin, then their target
    std::sort(edges.begin(), edges.end(), [](Edge &a, Edge &b) {
        return a.from < b.from || a.from == b.from && a.to < b.to;
    });

    fw_col_idx.resize(edges.size());
    fw_row_ptr.resize(num_vertices + 1);
    auto fw_row_cnt = 0;

    auto e_it = edges.begin();
    for(int i = 0; e_it != edges.end(); ++e_it, ++i) {
        fw_col_idx[i] = e_it->to;
        if (fw_row_cnt < e_it->from + 1) {
            fw_row_ptr[e_it->from] = i;
            fw_row_cnt = e_it->from + 1;
        }
    }
    fw_row_ptr[num_vertices] = edges.size();

    // fill uninitialized row_ptr when graph not strongly connected
    auto first_row = edges[0].from;
    auto next_row_ptr = edges.size();
    for (auto i = num_vertices - 1; i > first_row; --i) {
        if (fw_row_ptr[i] == 0) {
            fw_row_ptr[i] = next_row_ptr;
        } else {
            next_row_ptr = fw_row_ptr[i];
        }
    }
}

template<class Vertex, class Weight>
bool bellman_ford(CSRGraph<Vertex, Weight> &graph, Vertex source, std::vector<Vertex> &p, std::vector<Weight> &d) {
    for(auto i = 0; i < p.size(); ++i) p[i] = i;
    for(auto it = d.begin(); it != d.end(); ++it) *it = std::numeric_limits<Weight>::max();
    boost::dynamic_bitset on_queue(d.size());
    std::deque<Vertex> queue;
    queue.push_back(source);
    d[source] = 0;
    on_queue[source] = true;

    while(!queue.empty()) {
        auto u = queue.front();
        queue.pop_front();
        on_queue[u] = false;
        auto[e_it, e_end] = graph.out_edges(u);
        for (; e_it != e_end; ++e_it) {
            auto v = e_it->to;
            auto w = e_it->w;
            if (d[u] + w < d[v]) {
                d[v] = d[u] + w;
                p[v] = u;
                if (!on_queue[v]) {
                    queue.push_back(v);
                    on_queue[v] = true;
                }
            }
        }
    }
    return true;
}


template<class Vertex, class Weight>
auto kdisjoint_shortest_path(Vertex num_vertices, std::vector<std::pair<Vertex, Vertex>> &_edges,
                             std::vector<Weight> &_weights, int k) -> std::pair<int, std::vector<std::pair<Vertex, Vertex>>> {

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

        // store initial shortest path
        Vertex next = p_1[graph.sink()];
        Z[0] = next;

        while (true) {
            int prev = p_1[next];
            t_p[next] = prev;
            t_s[prev] = next;
            t_w[next] = graph.get_edge(prev, next)->w;
            // in P_1 there are no weighted nodes
            if (prev == graph.source()) {
                A[0] = next;
                Aj_in_Pm[next] = true;
                break;
            } else {
                next = prev;
            }
        }
    }
    for(auto j = 0; j < graph.num_nodes(); ++j) {
        L[j] = d[graph.sink()] < d[j] ? d[graph.sink()] : d[j];
    }

    using SearchNode = std::pair<Weight, Vertex>;
    using PQ = std::priority_queue<SearchNode, std::vector<SearchNode>, std::greater<SearchNode>>;
    PQ C;
    boost::dynamic_bitset in_C(static_cast<std::size_t>(graph.num_nodes()));

    for(auto it = d.begin(); it != d.end(); ++it) *it = std::numeric_limits<Weight>::max();
    {
        auto [e_it, e_end] = graph.out_edges(0);
        for (; e_it != e_end; ++e_it) {
            auto j = e_it->to;
            if(j != A[0]) {
                d[j] = e_it->w - L[j];
                p[j] = 0;
                C.emplace(d[j], j);
                in_C[j] = true;
            } else {
                Aj_in_Pm[j] = true;
            }
        }
        for(auto j = 1; j < graph.num_nodes(); ++j) {
            if(!Aj_in_Pm[j]) { in_C[j] = true; }
        }
    }

    auto M = 1;
    auto D = L[graph.sink()];

    // Interlacing Construction
    // find I = node in C with the minimal distance
    while(!C.empty()) {
        auto[weight, I] = C.top();
        C.pop();
        if (d[I] < weight || !in_C[I]) { continue; }
        L[I] += d[I];
        in_C[I] = false;

        if (I == graph.sink()) {
            // go to P_M+1
            while (I != 0) {
                auto J = p[I]; // J = node in P_M where S last leaves P_M prior to I
                auto Q = I; // Q = node following J on S
                t_p[Q] = J;
                t_w[Q] = graph.get_edge(J, Q)->w;
                while (t_p[J] == J && J != 0) {
                    t_s[J] = Q;
                    Q = J; J = p[Q]; t_p[Q] = J;
                    t_w[Q] = graph.get_edge(J, Q)->w;
                }

                if (J != 0) {
                    auto T = t_s[J]; // find T = node following J on P_M
                    t_s[J] = Q;

                    if(p_in_S[T]) {
                        // going further than T, reset t_p's along the way
                        I = p[T];
                        for(auto i = I; i != J;) {
                            auto prev_t = t_p[i];
                            t_p[i] = i;
                            i = prev_t;
                        }
                    } else {
                        I = T;
                    }
                } else {
                    t_s[J] = Q;
                    A[M] = Q; Z[M] = p[graph.sink()];
                    Aj_in_Pm[Q] = true;

                    D += L[graph.sink()];

                    C = PQ();
                    M++;

                    if (k > M) {
                        auto dZ = d[graph.sink()];
                        // New Initial Condition
                        // step 1: For each node j in C, add d_z to L_j
                        for(auto j = 1; j < graph.num_nodes(); ++j) {
                            if(in_C[j] && !Aj_in_Pm[j]) {
                                L[j] += dZ;
                            }
                        }

                        // step 2 + 3
                        in_C.reset();
                        for(auto it = d.begin(); it != d.end(); ++it) *it = std::numeric_limits<Weight>::max();
                        {
                            auto [e_it, e_end] = graph.out_edges(0);
                            for (; e_it != e_end; ++e_it) {
                                auto j = e_it->to;
                                if(!Aj_in_Pm[j]) {
                                    d[j] = e_it->w - L[j];
                                    p[j] = 0;
                                    p_in_S[j] = false;
                                    C.emplace(d[j], j);
                                    in_C[j] = true;
                                }
                            }
                            for(auto j = 1; j < graph.num_nodes(); ++j) {
                                if(!Aj_in_Pm[j]) { in_C[j] = true; }
                            }
                        }
                    } else {
                        // Terminate
                    }

                    break;
                }
            }

        } else if (t_p[I] != I) {
            // I in P_M
            // go to negative interlacing
            auto J = t_p[I], Q = I, T = p_in_S[I] ? p[I] : I;
            while (J != 0) {
                if (Q != I) {
                    // remove from C all nodes between and different from I and J in P
                    in_C[Q] = false;
                }
                if (!in_C[J] || L[J]  + t_w[Q] < L[Q]) {
                    break;
                }
                Q = J;
                J = t_p[Q];
            }

            // step 3
            for (auto s = t_p[I]; s != J; s = t_p[s]) {
                L[s] += d[I];
                p[s] = T;
                p_in_S[s] = true;
                auto[e_it, e_end] = graph.out_edges(s);
                for (; e_it != e_end; ++e_it) {
                    auto j = e_it->to;
                    if(in_C[j]) {
                        auto edge_cost = e_it->w;
                        auto delta = L[s] + edge_cost - L[j];
                        if (d[j] > delta) {
                            d[j] = delta;
                            p[j] = s;
                            p_in_S[j] = false;
                            C.emplace(delta, j);
                        }
                    }
                }
            }
            // step 4
            auto Lj = L[Q] - graph.get_edge(J, Q)->w;
            auto[e_it, e_end] = graph.out_edges(J);
            for (; e_it != e_end; ++e_it) {
                // no reverse check
                auto j = e_it->to;
                if(in_C[j]) {
                    auto edge_cost = e_it->w;
                    auto delta = Lj + edge_cost - L[j];
                    if (d[j] > delta) {
                        d[j] = delta;
                        p[j] = J;
                        p_in_S[j] = false;
                        C.emplace(delta, j);
                    }
                }
            }
            // step 5
            if (in_C[J]) {
                auto delta = Lj - L[J];
                if (d[J] > delta) {
                    d[J] = delta;
                    p[J] = T;
                    p_in_S[J] = true;
                    C.emplace(delta, J);
                }
            }
            // go to Interlacing Construction
        } else {
            // expand to neighbors
            auto[e_it, e_end] = graph.out_edges(I);
            for (; e_it != e_end; ++e_it) {
                // not part of a path, therefore no reversed edges possible
                auto j = e_it->to;
                if(in_C[j]) {
                    auto edge_cost = e_it->w;
                    auto delta = L[I] + edge_cost - L[j];
                    if (d[j] > delta) {
                        d[j] = delta;
                        p[j] = I;
                        p_in_S[j] = false;
                        C.emplace(delta, j);
                    }
                }
            }
        }
    }

    std::vector<std::pair<Vertex, Vertex>> sp_edges;
    for (auto i = 0; i < M; ++i) {
        sp_edges.emplace_back(Z[i], graph.sink());
        for (auto j = Z[i]; j != 0; j = t_p[j]) {
            sp_edges.emplace_back(t_p[j], j);
        }
    }

    return std::make_pair(M, std::move(sp_edges));
}