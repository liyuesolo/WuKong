#ifndef ROD_H
#define ROD_H


#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <tbb/tbb.h>

#include "VecMatDef.h"
#include "Util.h"
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

    T B[2][2];

    Matrix<T, 2, 2> bending_coeffs;

    // unit m
    T a = 1e-4, b = 1e-4;

    T E = 3.5e9;
    T ks;
    T kt;

    int rod_id;
    bool closed;
    int theta_dof_start_offset = 0;
    // full states for all nodes in the system
    VectorXT& full_states;
    VectorXT& rest_states;

    std::vector<TV> reference_frame_us;
    // std::vector<TV> rest_tangents;
    
    VectorXT reference_angles;

    // this should be zero as we are doing staic solve
    // VectorXT reference_twist;
    
    // which node on this rod has been marked as crossing node, ranging from 0 to # nodes
    std::vector<int> dof_node_location;

    //maps dim + 1 dof to global dof offset
    std::unordered_map<int, Offset> offset_map;
    
    // reduced map maps global dof to actual dofs
    std::unordered_map<int, int> reduced_map;

    // global indices of nodes on this rod
    std::vector<int> indices;

    // rest state data
    RestState<T, dim>* rest_state;
    
public:
    // Rod.cpp
    void setupBishopFrame();
    T computeReferenceTwist(const TV& tangent, const TV& prev_tangent, int rod_idx);
    void curvatureBinormal(const TV& t1, const TV& t2, TV& kb);



public:


// ============================== iterators ===================================

    template <class OP>
    void iterateSegments(const OP& f) 
    {
        for (int i = 0; i < indices.size() - 1; i++)
        {
            f(indices[i], indices[i+1], i);
        }
    }
    

    template <class OP>
    void iterateSegmentsWithOffset(const OP& f) 
    {
        for (int i = 0; i < indices.size() - 1; i++)
        {
            f(indices[i], indices[i+1], offset_map[indices[i]], offset_map[indices[i+1]], i);
        }
    }

    template <class OP>
    void iterate3NodesWithOffsets(const OP& f) 
    {
        for (int i = 1; i < indices.size() - 1; i++)
        {
            f(indices[i], indices[i+1], indices[i-1],
             offset_map[indices[i]], offset_map[indices[i+1]], offset_map[indices[i-1]], i);
        }
        if (closed)
            f(indices.back(), indices[1], indices[indices.size() - 2],
             offset_map[indices.back()], offset_map[indices[1]], offset_map[indices[indices.size() - 2]], 0);
    }

    template <class OP>
    void iterate3Nodes(const OP& f) 
    {
        for (int i = 1; i < indices.size() - 1; i++)
        {
            f(indices[i], indices[i+1], indices[i-1], i);
        }
        if (closed)
            f(indices.back(), indices[1], indices[indices.size() - 2], 0);
    }

// ============================== Lagrangian Eulerian value helpers ===================================
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

// ============================== helpers ===================================

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

    int numSeg()
    {
        return indices.size() - 1;
    }

    void markDoF(std::vector<Entry>& w_entry, int& dof_cnt);

    void initCoeffs()
    {
        T coeff = 0.25 * E * M_PI * a * b;
        B[0][0] = coeff * a * a; B[0][1] = 0; 
        B[1][0] = 0; B[1][1] = coeff * b*b;

        bending_coeffs << coeff * a * a, 0, 0, coeff * b*b;
        
        ks = E * M_PI * a * b;
        T G = E/T(2)/(1.0 + 0.42);
        kt = 0.25 * G  * M_PI * a * b * (a*a + b*b);
    }

public:
    Rod (VectorXT& q, VectorXT& q0, int id) : full_states(q), rest_states(q0), rod_id(id), closed(false) { initCoeffs(); }
    Rod (VectorXT& q, VectorXT& q0, int id, bool c) : full_states(q), rest_states(q0), rod_id(id), closed(c) { initCoeffs(); }
    ~Rod() {}

// private:

};
#endif