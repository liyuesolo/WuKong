#ifndef ROD_H
#define ROD_H


#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <tbb/tbb.h>

#include "VecMatDef.h"

#include "RestState.h"

template<class T, int dim> class Rod;

template<class T, int dim>
struct RodCrossing
{
    int node_idx;
    std::vector<int> rods_involved;
    std::vector<Vector<T, 2>> sliding_ranges;

    RodCrossing(int id, std::vector<int> involved) : node_idx(id), rods_involved(involved) {}
};


template<class T, int dim>
class Rod
{

public:

    using Entry = Eigen::Triplet<T>;
    using TV = Vector<T, dim>;
    using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
    using VectorXi = Eigen::VectorXi;
    using Offset = Vector<int, dim + 1>;

    int rod_id;
    bool closed;
    // full states for all nodes in the system
    VectorXT& full_states;
    VectorXT& rest_states;
    
    // which node on this rod has been marked as crossing node
    std::vector<int> dof_node_location;


    //maps dim + 1 dof to global dof offset
    std::unordered_map<int, Offset> offset_map;

    std::unordered_map<int, int> reduced_map;

    // global indices of nodes on this rod
    std::vector<int> indices;

    // rest state data
    RestState<T, dim>* rest_state;

    template <class OP>
    void iterateSegments(const OP& f) 
    {
        for (int i = 0; i < indices.size() - 1; i++)
        {
            f(indices[i], indices[i+1]);
        }
    }
    

    template <class OP>
    void iterateSegmentsWithOffset(const OP& f) 
    {
        for (int i = 0; i < indices.size() - 1; i++)
        {
            f(indices[i], indices[i+1], offset_map[indices[i]], offset_map[indices[i+1]]);
        }
    }

    template <class OP>
    void iterate3NodesWithOffsets(const OP& f) 
    {
        for (int i = 1; i < indices.size() - 1; i++)
        {
            f(indices[i], indices[i+1], indices[i-1],
             offset_map[indices[i]], offset_map[indices[i+1]], offset_map[indices[i-1]]);
        }
        if (closed)
            f(indices.back(), indices[1], indices[indices.size() - 2],
             offset_map[indices.back()], offset_map[indices[1]], offset_map[indices[indices.size() - 2]]);
    }

    template <class OP>
    void iterate3Nodes(const OP& f) 
    {
        for (int i = 1; i < indices.size() - 1; i++)
        {
            f(indices[i], indices[i+1], indices[i-1]);
        }
        if (closed)
            f(indices.back(), indices[1], indices[indices.size() - 2]);
    }

    void x(int node_idx, TV& pos)
    {
        pos = TV::Zero();
        Offset idx = offset_map[node_idx];
        for (int d = 0; d < dim; d++)
        {
            pos[d] = full_states[idx[d]];
        }
        
    }

    void u(int node_idx, T& pos)
    {
        Offset idx = offset_map[node_idx];
        pos = full_states[idx[dim]];
    }

    void U(int node_idx, T& pos)
    {
        Offset idx = offset_map[node_idx];
        pos = rest_states[idx[dim]];
    }

    void X(int node_idx, TV& pos)
    {
        TV dXdu = TV::Zero(), d2Xdu2 = TV::Zero();
        Offset idx = offset_map[node_idx];
        T u = full_states(idx[dim]);
        rest_state->getMaterialPos(u, pos, dXdu, d2Xdu2, false, false);
    }

    void XdX(int node_idx, TV& pos, TV& dpos)
    {
        TV d2Xdu2 = TV::Zero();
        Offset idx = offset_map[node_idx];
        T u = full_states[idx[dim]];
        rest_state->getMaterialPos(u, pos, dpos, d2Xdu2, true, false);
    }

    void XdXddX(int node_idx, TV& pos, TV& dpos, TV& ddpos)
    {
        Offset idx = offset_map[node_idx];
        T u = full_states[idx[dim]];
        rest_state->getMaterialPos(u, pos, dpos, ddpos, true, true);
    }

    int entry(int node_idx)
    {
        return 0;
    }

    void getEntry(int node_idx, Offset& idx)
    {
        idx = offset_map[node_idx];
    }

    void frontOffset(Offset& idx)
    {
        idx = offset_map[indices.front()];
    }

    void backOffset(Offset& idx)
    {
        idx = offset_map[indices.back()];
    }

    void backOffsetReduced(Offset& idx)
    {
        idx = offset_map[indices.back()];
        for (int d = 0; d < dim + 1; d++)
            idx[d] = reduced_map[idx[d]];
    }

    void frontOffsetReduced(Offset& idx)
    {
        idx = offset_map[indices.front()];
        for (int d = 0; d < dim + 1; d++)
            idx[d] = reduced_map[idx[d]];
    }

    void frontDoF(Vector<T, dim + 1>& q)
    {
        q = Vector<T, dim + 1>::Zero();
        Offset idx = offset_map[indices.front()];
        for (int d = 0; d < dim + 1; d++)
            q[d] = full_states[idx[d]];
    }

    void backDoF(Vector<T, dim + 1>& q)
    {
        q = Vector<T, dim + 1>::Zero();
        Offset idx = offset_map[indices.back()];
        for (int d = 0; d < dim + 1; d++)
            q[d] = full_states[idx[d]];
    }

    void fixEndPointEulerian(std::unordered_map<int, T>& dirichlet_data)
    {
        dirichlet_data[reduced_map[offset_map[indices.front()][dim]]] = 0;
        dirichlet_data[reduced_map[offset_map[indices.back()][dim]]] = 0;
    }

    void fixEndPointLagrangian(std::unordered_map<int, T>& dirichlet_data)
    {
        for (int d = 0; d < dim; d++)
        {
            dirichlet_data[reduced_map[offset_map[indices.front()][d]]] = 0;
            dirichlet_data[reduced_map[offset_map[indices.back()][d]]] = 0;    
        }
    }

    void validCheck()
    {

    }

    int numSeg()
    {
        return indices.size() - 1;
    }

    void markDoF(std::vector<Entry>& w_entry, int& dof_cnt);

public:
    Rod (VectorXT& q, VectorXT& q0, int id) : full_states(q), rest_states(q0), rod_id(id), closed(false) {}
    Rod (VectorXT& q, VectorXT& q0, int id, bool c) : full_states(q), rest_states(q0), rod_id(id), closed(c) {}
    ~Rod() {}

// private:

};
#endif