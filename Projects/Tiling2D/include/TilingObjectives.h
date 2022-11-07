#ifndef TILING_OBJECTIVES_H
#define TILING_OBJECTIVES_H

#include <utility>
#include <iostream>
#include <fstream>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <tbb/tbb.h>

#include "VecMatDef.h"
#include "Tiling2D.h"
class Tiling2D;

class TilingObjectives
{
public:
    using TV = Vector<double, 2>;
    using TM = Matrix<double, 2, 2>;
    using IV = Vector<int, 2>;
    using IV3 = Vector<int, 3>;
    using Edge = Vector<int, 2>;

    typedef int StorageIndex;
    using StiffnessMatrix = Eigen::SparseMatrix<T, Eigen::ColMajor, StorageIndex>;
    // using StiffnessMatrix = Eigen::SparseMatrix<T>;
    using Entry = Eigen::Triplet<T>;
    using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
    using MatrixXT = Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorXi = Vector<int, Eigen::Dynamic>;

public:
    Tiling2D& tiling;
    FEMSolver& solver = tiling.solver;
    int n_dof_sim, n_dof_design;

    int IH = 19;
    
    T strain = 1.05;
    VectorXT strain_samples;

    T theta = 0.0;
    std::vector<TV> bounds;
    VectorXT targets;

public:
    virtual T value(const VectorXT& p_curr, bool simulate = true, bool use_prev_equil = false) {}
    virtual T gradient(const VectorXT& p_curr, VectorXT& dOdp, T& energy, bool simulate = true, bool use_prev_equil = false) {}

    TilingObjectives(Tiling2D& _tiling) : tiling(_tiling) {}
    ~TilingObjectives() {}
};


class UniaxialStressObjective : public TilingObjectives
{
public:

    T generateSingleTarget(const VectorXT& ti);
    void computeStressForDifferentStrain(const VectorXT& ti, VectorXT& stress);
    
    T value(const VectorXT& p_curr, bool simulate = true, bool use_prev_equil = false);
    T gradient(const VectorXT& p_curr, VectorXT& dOdp, 
        T& energy, bool simulate = true, bool use_prev_equil = false);

public:
    UniaxialStressObjective(Tiling2D& _tiling) : TilingObjectives(_tiling) {}
    ~UniaxialStressObjective() {}
};


#endif