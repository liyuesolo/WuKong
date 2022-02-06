#ifndef OBJECTIVES_H
#define OBJECTIVES_H


#include <utility>
#include <iostream>
#include <fstream>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <tbb/tbb.h>

#include "VecMatDef.h"
#include "Simulation.h"

class Simulation;

class Objectives
{
public:
    using TV = Vector<double, 3>;
    using TM = Matrix<double, 3, 3>;
    using IV = Vector<int, 3>;

    typedef long StorageIndex;
    using StiffnessMatrix = Eigen::SparseMatrix<T, Eigen::RowMajor, StorageIndex>;
    using Entry = Eigen::Triplet<T>;
    using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
    using MatrixXT = Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorXi = Vector<int, Eigen::Dynamic>;

public:
    Simulation& simulation;
    int n_dof_sim, n_dof_design;

    virtual T value(const VectorXT& p_curr) = 0;
    virtual T gradient(const VectorXT& p_curr, VectorXT& dOdp) = 0;
    virtual T gradient(const VectorXT& p_curr, VectorXT& dOdp, T& energy) = 0;
    virtual void updateDesignParameters(const VectorXT& design_parameters) = 0;
    virtual void getDesignParameters(VectorXT& design_parameters) = 0;
    virtual void getSimulationAndDesignDoF(int& sim_dof, int& design_dof) = 0;
    
    void diffTestGradientScale();
    void diffTestGradient();
public:
    Objectives(Simulation& _simulation) : simulation(_simulation) {}
    ~Objectives() {}
};

class ObjUTU : public Objectives
{
public:
    T value(const VectorXT& p_curr);
    T gradient(const VectorXT& p_curr, VectorXT& dOdp);
    T gradient(const VectorXT& p_curr, VectorXT& dOdp, T& energy);
    void updateDesignParameters(const VectorXT& design_parameters);
    void getDesignParameters(VectorXT& design_parameters);
    
    void getSimulationAndDesignDoF(int& _sim_dof, int& _design_dof);

public:
    ObjUTU(Simulation& _simulation) : Objectives(_simulation) {}
    ~ObjUTU() {}
};

#endif
