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

#include "Simulation.h"
class Simulation;

#include "VecMatDef.h"
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
    VectorXT equilibrium_prev;

    virtual T value(const VectorXT& p_curr, bool use_prev_equil = false) {}
    virtual T gradient(const VectorXT& p_curr, VectorXT& dOdp, bool use_prev_equil = false) {}
    virtual T gradient(const VectorXT& p_curr, VectorXT& dOdp, T& energy, bool use_prev_equil = false) {}
    virtual T evaluteGradientAndEnergy(const VectorXT& p_curr, VectorXT& dOdp, T& energy) {}
    virtual T hessianGN(const VectorXT& p_curr, StiffnessMatrix& H, bool use_prev_equil = false) {}
    virtual T hessian(const VectorXT& p_curr, StiffnessMatrix& H, bool use_prev_equil = false) {}

    virtual void updateDesignParameters(const VectorXT& design_parameters) {}
    virtual void getDesignParameters(VectorXT& design_parameters) {}
    virtual void getSimulationAndDesignDoF(int& sim_dof, int& design_dof) {}

    void saveState(const std::string& filename) { simulation.saveState(filename); }
    
    void diffTestGradientScale();
    void diffTestGradient();
public:
    Objectives(Simulation& _simulation) : simulation(_simulation) 
    {
        equilibrium_prev.resize(simulation.num_nodes * 3);
        equilibrium_prev.setZero();
    }
    ~Objectives() {}
};

class ObjUTU : public Objectives
{
public:
    T value(const VectorXT& p_curr, bool use_prev_equil = false);
    T gradient(const VectorXT& p_curr, VectorXT& dOdp, bool use_prev_equil = false);
    T gradient(const VectorXT& p_curr, VectorXT& dOdp, T& energy, bool use_prev_equil = false);
    T evaluteGradientAndEnergy(const VectorXT& p_curr, VectorXT& dOdp, T& energy);
    void updateDesignParameters(const VectorXT& design_parameters);
    void getDesignParameters(VectorXT& design_parameters);
    void getSimulationAndDesignDoF(int& _sim_dof, int& _design_dof);

    T hessianGN(const VectorXT& p_curr, StiffnessMatrix& H, bool use_prev_equil = false);
    T hessian(const VectorXT& p_curr, StiffnessMatrix& H, bool use_prev_equil = false) {}

public:
    ObjUTU(Simulation& _simulation) : Objectives(_simulation) 
    {
        
    }
    ~ObjUTU() {}
};

class ObjUMatching : public Objectives
{
private:
    VectorXT target;

public:
    void setTargetFromMesh(const std::string& filename);

    T value(const VectorXT& p_curr, bool use_prev_equil = false);
    T gradient(const VectorXT& p_curr, VectorXT& dOdp, bool use_prev_equil = false);
    T gradient(const VectorXT& p_curr, VectorXT& dOdp, T& energy, bool use_prev_equil = false);
    T evaluteGradientAndEnergy(const VectorXT& p_curr, VectorXT& dOdp, T& energy);
    void updateDesignParameters(const VectorXT& design_parameters);
    void getDesignParameters(VectorXT& design_parameters);
    void getSimulationAndDesignDoF(int& _sim_dof, int& _design_dof);

    T hessianGN(const VectorXT& p_curr, StiffnessMatrix& H, bool use_prev_equil = false);
    T hessian(const VectorXT& p_curr, StiffnessMatrix& H, bool use_prev_equil = false) {}

public:
    ObjUMatching(Simulation& _simulation) : Objectives(_simulation) 
    {
        
    }
    ~ObjUMatching() {}
};

#endif
