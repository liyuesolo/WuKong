#pragma once

#include <Ziran/CS/Util/Forward.h>
#include "StaticGrid.h"

namespace ZIRAN {

template <class T, int dim>
class TopographyOptimization;

template <int dim>
struct VectorHash
{
    typedef Vector<int, dim> IV;
    size_t operator()(const IV& a) const{
        std::size_t h = 0;
        for (int d = 0; d < dim; ++d) {
            h ^= std::hash<int>{}(a(d)) + 0x9e3779b9 + (h << 6) + (h >> 2); 
        }
        return h;
    }
};

template <int dim>
struct VectorPairHash
{
    typedef Vector<int, dim> IV;
    size_t operator()(const std::pair<IV, IV>& a) const{
        std::size_t h = 0;
        for (int d = 0; d < dim; ++d) {
            h ^= std::hash<int>{}(a.first(d)) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<int>{}(a.second(d)) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return h;
    }
};

template <class T, int dim>
struct PairHash
{
    typedef StaticGridState<T, dim>* HashType;
    size_t operator()(const std::pair<HashType, HashType>& a) const{
        std::size_t h = 0;
        h ^= std::hash<HashType>{}(a.first);
        h ^= std::hash<HashType>{}(a.second);
        return h;
    }
};

template <class T, int dim, class Simulation>
class BoundaryCondition {
public:
    typedef Vector<T, dim> TV;
    typedef Vector<int, dim> IV;
    typedef Matrix<T, Eigen::Dynamic, 1> VectorXT;

    std::unordered_map<StaticGridState<T, dim>*, TV> neuman_data;
    std::unordered_map<IV, TV, VectorHash<dim>> dirichlet_data;
    std::unordered_set<IV, VectorHash<dim>> sync_data;
    std::unordered_map<StaticGridState<T, dim>*, IV> coord;
    std::unordered_map<StaticGridState<T, dim>*, int> group_id;
    std::unordered_map<std::pair<StaticGridState<T, dim>*, StaticGridState<T, dim>*>, std::pair<T, T>, PairHash<T, dim>> spring_data;  // two ends, k, l0
    std::unordered_map<StaticGridState<T, dim>*, T> contact_sphere; // radius at each contact center;
    Simulation &topo;
    std::unordered_map<IV, TV, VectorHash<dim>> cell_dirichlet_data;
    BoundaryCondition(Simulation &topo_in): topo(topo_in) {}
    std::unique_ptr<BoundaryCondition<T, dim, Simulation>> trivialize() const { // only keep trivial dirichlet boundary 
        auto new_bc = std::make_unique<BoundaryCondition<T, dim, Simulation>>(topo);
        new_bc->dirichlet_data = dirichlet_data;
        new_bc->sync_data = sync_data;
        new_bc->coord = coord;
        new_bc->group_id = group_id;
        new_bc->spring_data = spring_data;
        new_bc->contact_sphere = contact_sphere;
        for (auto it: new_bc->dirichlet_data) {
            for (int d = 0; d < dim; ++d) { if (it.second(d) != 0) it.second(d) = 1e30; }}
        return new_bc;
    }
    IV absToRel(const TV& abs_node) { return ((abs_node)/topo.dx).array().round().template cast<int>(); }
    void addDirichlet(TV abs_node, TV target_u)
    { 
        IV grid = absToRel(abs_node);
        if (dirichlet_data.find(grid) == dirichlet_data.end()) dirichlet_data[grid] = 1e30 * TV::Ones();
        for (int d = 0; d < dim; ++d)
        {
            if (abs(target_u(d)) < 1e10)
                dirichlet_data[grid](d) = target_u(d);
        }
    }
    
    void addDirichletCell(TV center, TV target_u)
    {
        topo.grid.iterateGridSerial([&](const IV& base_node, auto& cell) {
            TV cell_base = topo.dx * base_node.template cast<T>();
            TV cell_center = cell_base.array() + 0.5 * topo.dx;
            if ((center - cell_center).norm() < 0.01 * topo.dx)
                cell_dirichlet_data[base_node] = target_u;
        });
    }

    void addDirichletLambda(std::function<std::optional<TV>(const TV&)> node_helper) 
    {
        topo.iterateGridSerial([&](const IV& node, auto& grid_state, auto&) {
            if (grid_state.grid_idx < 0) return;
            TV grid_coord = node.template cast<T>() * topo.dx;
            std::optional<TV> u_opt = node_helper(grid_coord);
            if (u_opt) {
                if (dirichlet_data.find(node) == dirichlet_data.end()) dirichlet_data[node] = 1e30 * TV::Ones();
                TV target_u = u_opt.value();
                for (int d = 0; d < dim; ++d)
                {
                    if (abs(target_u(d)) < 1e10)
                        dirichlet_data[node](d) = target_u(d);
                }
            }
        });
    }
    void addDirichletSegment(TV start, TV end, TV target_u)
    {
        std::function<std::optional<TV>(const TV&)> node_helper = [&](const TV& x) -> std::optional<TV> {
            Matrix<T, dim, dim> R = Matrix<T, dim, dim>::Zero();
            TV n = (end - start).normalized();
            // find R such that R * orientaion = (1, 0, 0);
            R.row(0) = n;
            if constexpr (dim == 2) {
                R(1, 0) = - n(1);
                R(1, 1) = n(0);
            }
            else{
                // find a orthogonal vecor first
                TV n1(0,0,0);
                for (int d = 0; d < 3; ++d) {
                    if (n(d) != 0) {
                        n1(d) = n((d + 1) % 3);
                        n1((d + 1) % 3) = - n(d);
                        n1((d + 2) % 3) = 0;
                        n1 /= n1.norm();
                        break;
                    }
                }
                TV n2 = n.cross(n1);
                R.row(1) = n1;
                R.row(2) = n2;
            }
            TV rotated = R * (x - start);
            if (rotated[0] >= - 0.1 * topo.dx && rotated[0] <=  0.1 * topo.dx + (end - start).norm() && (rotated.template segment<dim - 1>(1).norm() <= topo.dx * 0.1))
                return target_u;
            else
                return std::nullopt;
        };
        addDirichletLambda(node_helper);
    }
    void addDirichletSphere(TV center, T radius, TV target_u)
    {
        std::function<std::optional<TV>(const TV&)> node_helper = [&](const TV& x) -> std::optional<TV> {
            if ((x - center).norm() <= radius + 0.01 * topo.dx)
                return target_u;
            else
                return std::nullopt;
        };
        addDirichletLambda(node_helper);
    }
    void addDirichletBox(TV min_corner, TV max_corner, TV target_u)
    {
        std::function<std::optional<TV>(const TV&)> node_helper = [&](const TV& x) -> std::optional<TV> {
            if ((x - min_corner).minCoeff() >= -0.01 * topo.dx && (max_corner - x).minCoeff() >= -0.01 * topo.dx)
                return target_u;
            else
                return std::nullopt;
        };
        addDirichletLambda(node_helper);
    }
    void addDirichletWall(TV origin, TV normal, TV target_u)
    {
        std::function<std::optional<TV>(const TV&)> node_helper = [&](const TV& x) -> std::optional<TV> {
            T dist = (x - origin).dot(normal.normalized());
            if (dist < 0.1 * topo.dx)
                return target_u;
            else
                return std::nullopt;
        };
        addDirichletLambda(node_helper);
    }
    void addNeuman(TV abs_node, TV f, int component_id)
    { 
        IV node = absToRel(abs_node);
        if (!topo.grid.allocated(node)) return;
        neuman_data[&topo.grid[node]] = f;
        coord[&topo.grid[node]] = node;
    }
    void addNeumannSolid(Eigen::Ref<const VectorXT> target_rho, TV fi)
    {
        topo.iterateCellSerial([&](const IV& cell, auto& cell_state, auto& grid) {
            if (!cell_state.active_cell) return;
            if (target_rho[cell_state.cell_idx] > 0.95)
            {
                grid.iterateCellKernel(cell, [&](const IV& node, auto& grid_state) {
                if (grid_state.grid_idx == -1) {
                    return;
                }
                neuman_data[&grid_state] = fi;
                coord[&grid_state] = node;
            });
            }
            
        });
    }
    std::optional<TV> dirichlet(const IV& node) 
    {
        if (dirichlet_data.find(node) != dirichlet_data.end()) return dirichlet_data[node];
        else return std::nullopt;
    }
    void addNeumannLambda(std::function<std::optional<TV>(const TV&)> force_helper) 
    {
        topo.iterateGridSerial([&](const IV& node, auto& grid_state, auto& grid) {
            
            if (grid_state.grid_idx < 0) return;
            TV grid_coord = node.template cast<T>() * topo.dx;
            std::optional<TV> f_opt = force_helper(grid_coord);
            if (f_opt) {
                neuman_data[&grid_state] = f_opt.value();
                coord[&grid_state] = node;
            }
        });
    }
    void addNeumanSegment(TV start, TV end, TV force, int component_id) 
    {
        std::function<std::optional<TV>(const TV&)> node_helper = [&](const TV& x) -> std::optional<TV> {
            Matrix<T, dim, dim> R = Matrix<T, dim, dim>::Zero();
            TV n = (end - start).normalized();
            // find R such that R * orientaion = (1, 0, 0);
            R.row(0) = n;
            if constexpr (dim == 2) {
                R(1, 0) = - n(1);
                R(1, 1) = n(0);
            }
            else{
                // find a orthogonal vecor first
                TV n1(0,0,0);
                for (int d = 0; d < 3; ++d) {
                    if (n(d) != 0) {
                        n1(d) = n((d + 1) % 3);
                        n1((d + 1) % 3) = - n(d);
                        n1((d + 2) % 3) = 0;
                        n1 /= n1.norm();
                        break;
                    }
                }
                TV n2 = n.cross(n1);
                R.row(1) = n1;
                R.row(2) = n2;
            }
            TV rotated = R * (x - start);
            if (rotated[0] >= - 0.1 * topo.dx && rotated[0] <=  0.1 * topo.dx + (end - start).norm() && (rotated.template segment<dim - 1>(1).norm() <= topo.dx * 0.1))
                return force;
            else
                return std::nullopt;
        };
        addNeumanLambda(node_helper, component_id);
    }
    void magnifyForce(T times) {
        for (auto& neuman: neuman_data) {
            neuman.second *= times;
        }
    }
    

    std::optional<TV> dirichletAbs(TV abs_node)
    {
        IV node = absToRel(abs_node);
        if (dirichlet_data.find(node) != dirichlet_data.end()) return dirichlet_data[node];
        else return std::nullopt;
    }
    // TV neuman(IV node)
    // {
    //     if (neuman_data.find(node) != neuman_data.end()) return neuman_data[node];
    //     else return TV::Zero();
    // }

    

    TV neumanAbs(TV abs_node, int component_id)
    {
        IV node = absToRel(abs_node);
        if (!topo.grid.allocated(node)) return TV::Zero();
        if (neuman_data.find(&topo.grid[node]) != neuman_data.end()) return neuman_data[&topo.grid[node]];
        else return TV::Zero();
    }
    int gridIndex(TV abs_node, int component_id)
    {
        IV node = absToRel(abs_node);
        const auto &g =  topo.grid[node];
        return g.grid_idx;
    }

    template <class OP>
    void iterateNeumanData(const OP& f) {
        for (auto neuman: neuman_data){
            auto& g = *neuman.first;
            if(g.grid_idx < 0) continue;
            f(g.grid_idx, coord[neuman.first], neuman.second);
        } 
    }

    template <class OP>
    void iterateContactSphere(const OP& f) {
        for (auto contact: contact_sphere) {
            if(contact.first->grid_idx < 0) continue;
            f(contact.first->grid_idx, coord[contact.first], contact.second, group_id[contact.first]);
        }
    }

    template <class OP>
    void iterateDirichletData(const OP& f) {
        for (auto dirichlet: dirichlet_data){
            
            if (!topo.grid.allocated(dirichlet.first)) continue;
            const auto& g = topo.grid[dirichlet.first];
            if(g.grid_idx < 0) continue;
            f(g.grid_idx, dirichlet.first, dirichlet.second);
            
        } 
    }

    template <class OP>
    void iterateSpringData(const OP& f) {
        for(auto spring: spring_data) {
            IV nodei = coord[spring.first.first];
            IV nodej = coord[spring.first.second];
            int dofi = spring.first.first->grid_idx;
            int dofj = spring.first.second->grid_idx;
            T k = spring.second.first;
            T l0 = spring.second.second;
            bool same_group = group_id[spring.first.first] == group_id[spring.first.second];
            f(dofi, dofj, nodei, nodej, k, l0, same_group);
        }
    }

};

}