#include <vector>
#include <queue>
#include "StaticGrid.h"
#include <Ziran/CS/Util/Forward.h>
#include "BoundaryCondition.h"

namespace ZIRAN {
    template <class T, int dim>
    void dijkstra(StaticGrid<T, dim>& grid, int num_cells, Vector<int, dim> source, Matrix<int, Eigen::Dynamic, 1>& dist) 
    {
        typedef Vector<int, dim> IV;
        dist.resize(num_cells);
        const int INF = 100000000;
        dist.setConstant(INF);
        Matrix<int, Eigen::Dynamic, 1> S_flag(num_cells);
        S_flag.setZero();
        std::list<IV> U;
        U.push_back(source);
        dist(grid[source].cell_idx) = 0;
        while (!U.empty()) {
            typename std::list<IV>::iterator selected_it;
            T dist_to_S = INF;
            for(auto it = U.begin(); it != U.end(); ++it) {
                if (dist(grid[*it].cell_idx) < dist_to_S) {
                    dist_to_S = dist(grid[*it].cell_idx);
                    selected_it = it;
                }
            }
            S_flag(grid[*selected_it].cell_idx) = 1;
            // update dist of neighbors ;
            grid.iterateAxisAlignedOneLaynerNeighbor(*selected_it, [&](const IV& neighbor, auto& neighbor_state) {
                if (!neighbor_state.active_cell) return;
                if (S_flag(neighbor_state.cell_idx)) return;
                if (dist(neighbor_state.cell_idx) == INF)
                    U.push_back(neighbor);
                dist(neighbor_state.cell_idx) = std::min(dist(grid[*selected_it].cell_idx) + 1, dist(neighbor_state.cell_idx));
            });
            U.erase(selected_it);
        }
    }

    template <class T, int dim>
    int shortestDist(StaticGrid<T, dim>& grid, Vector<int, dim> source, Vector<int, dim> target) 
    {
        typedef Vector<int, dim> IV;
        const int INF = 100000000;
        std::unordered_map<IV, IV, VectorHash<dim>> parent;
        std::unordered_map<IV, int, VectorHash<dim>> dist;
        std::unordered_set<IV, VectorHash<dim>> U;
        std::unordered_set<IV, VectorHash<dim>> S;
        U.insert(source);
        dist[source] = 0;

        while(!U.empty()) {
            T dist_to_S = INF;
            IV next_to_S;
            for(auto it = U.begin(); it != U.end(); ++it) {
                if (dist[*it] < dist_to_S) {
                    dist_to_S = dist[*it];
                    next_to_S = *it;
                }
            }
            S.insert(next_to_S);
            if (next_to_S == target) return dist[next_to_S];
            grid.iterateAxisAlignedOneLaynerNeighbor(next_to_S, [&](const IV& neighbor, auto& neighbor_state) {
                if (!neighbor_state.active_cell) return;
                if (S.find(neighbor) != S.end()) return;
                if (U.find(neighbor) == U.end()) {
                    U.insert(neighbor);
                    parent[neighbor] = next_to_S;
                    dist[neighbor] = dist[next_to_S] + 1;
                } else {
                    if (dist[next_to_S] + 1 < dist[neighbor]) {
                        dist[neighbor] = dist[next_to_S] + 1;
                        parent[neighbor] = next_to_S;
                    }
                }
            });
            U.erase(next_to_S);
        }
    }

    template <class T, int dim>
    void shortestPath(StaticGrid<T, dim>& grid, Vector<int, dim> source, Vector<int, dim> target, StdVector<Vector<int, dim>>& path) 
    {
        typedef Vector<int, dim> IV;
        const int INF = 100000000;
        std::unordered_map<IV, IV, VectorHash<dim>> parent;
        std::unordered_map<IV, int, VectorHash<dim>> dist;
        std::unordered_set<IV, VectorHash<dim>> U;
        std::unordered_set<IV, VectorHash<dim>> S;
        U.insert(source);
        dist[source] = 0;

        while(!U.empty()) {
            T dist_to_S = INF;
            IV next_to_S;
            for(auto it = U.begin(); it != U.end(); ++it) {
                if (dist[*it] < dist_to_S) {
                    dist_to_S = dist[*it];
                    next_to_S = *it;
                }
            }
            S.insert(next_to_S);
            if (next_to_S == target) break;
            grid.iterateAxisAlignedOneLaynerNeighbor(next_to_S, [&](const IV& neighbor, auto& neighbor_state) {
                if (neighbor_state.cell_idx < 0) return;
                if (S.find(neighbor) != S.end()) return;
                if (U.find(neighbor) == U.end()) {
                    U.insert(neighbor);
                    parent[neighbor] = next_to_S;
                    dist[neighbor] = dist[next_to_S] + 1;
                } else {
                    if (dist[next_to_S] + 1 < dist[neighbor]) {
                        dist[neighbor] = dist[next_to_S] + 1;
                        parent[neighbor] = next_to_S;
                    }
                }
            });
            U.erase(next_to_S);
        }

        IV current = target;
        while(parent.find(current) != parent.end()) {
            path.push_back(current);
            current = parent[current];
        }
        path.push_back(current);
    }
}