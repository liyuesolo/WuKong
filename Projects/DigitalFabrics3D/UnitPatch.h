#ifndef UNIT_PATCH_H
#define UNIT_PATCH_H

#include "EoLRodSim.h"

template<class T, int dim>
class EoLRodSim;

template<class T, int dim>
class UnitPatch
{
public:
    using TV = Vector<T, dim>;

    using VectorXT = Matrix<T, Eigen::Dynamic, 1>;

    
    using StiffnessMatrix = Eigen::SparseMatrix<T>;
    using Entry = Eigen::Triplet<T>;

    using Offset = Vector<int, dim + 1>;
    using Range = Vector<T, 2>;

private:
    EoLRodSim<T, dim>& sim;

    VectorXT& deformed_states = sim.deformed_states;
    

public:
    UnitPatch(EoLRodSim<T, dim>& eol_sim) : sim(eol_sim) {}
    ~UnitPatch() {}

    void buildScene(int patch_type);
    void build3DtestScene(int sub_div);


private:
    
    void clearSimData();
    
    

    void addStraightYarnCrossNPoints(const TV& from, const TV& to,
        const std::vector<TV>& passing_points, 
        const std::vector<int>& passing_points_id, 
        int sub_div,
        std::vector<TV>& sub_points, std::vector<int>& node_idx,
        std::vector<int>& key_points_location,
        int start, bool pbc = false);


    void markCrossingDoF(
        std::vector<Eigen::Triplet<T>>& w_entry,
        int& dof_cnt);
    

};

#endif