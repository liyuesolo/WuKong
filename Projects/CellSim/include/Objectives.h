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

#include "SpatialHash.h"

#include "VecMatDef.h"

enum Optimizer
{
    GradientDescent, GaussNewton, MMA, Newton, SGN
};

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

    Optimizer default_optimizer = GaussNewton;
public:
    template <class OP>
    void iterateTargets(const OP& f) {
        for (auto dirichlet: target_positions){
            f(dirichlet.first, dirichlet.second);
        } 
    }
public:
    Simulation& simulation;
    int n_dof_sim, n_dof_design;
    VectorXT equilibrium_prev;

    std::unordered_map<int, TV> target_positions;

    virtual T value(const VectorXT& p_curr, bool use_prev_equil = false) {}
    virtual T gradient(const VectorXT& p_curr, VectorXT& dOdp, bool use_prev_equil = false) {}
    virtual T gradient(const VectorXT& p_curr, VectorXT& dOdp, T& energy, bool use_prev_equil = false) {}
    virtual T evaluteGradientAndEnergy(const VectorXT& p_curr, VectorXT& dOdp, T& energy) {}
    virtual T hessianGN(const VectorXT& p_curr, StiffnessMatrix& H, bool simulate = true, bool use_prev_equil = false) {}
    virtual T hessian(const VectorXT& p_curr, StiffnessMatrix& H, bool use_prev_equil = false) {}
    virtual void dOdx(const VectorXT& p_curr, VectorXT& _dOdx) {}
    virtual void d2Odx2(const VectorXT& p_curr, std::vector<Entry>& d2Odx2_entries) {}

    virtual void updateDesignParameters(const VectorXT& design_parameters) {}
    virtual void getDesignParameters(VectorXT& design_parameters) {}
    virtual void getSimulationAndDesignDoF(int& sim_dof, int& design_dof) {}
    virtual void setSimulationAndDesignDoF(int _sim_dof, int _design_dof);
    virtual void updateTarget() {}

    virtual T maximumStepSize(const VectorXT& dp) { return 1.0; }
    virtual void setOptimizer(Optimizer opt) { default_optimizer = opt; }

    void saveState(const std::string& filename) { simulation.saveState(filename); }
    void saveDesignParameters(const std::string& filename, const VectorXT& design_parameters);
    
    void diffTestGradientScale();
    void diffTestGradient();
    void diffTestHessian();
    void diffTestHessianScale();
    void diffTestdOdx();
    void diffTestd2Odx2();

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
    void updateTarget() {}

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
    void updateTarget() {}
    T hessianGN(const VectorXT& p_curr, StiffnessMatrix& H, bool simulate = true, bool use_prev_equil = false);
    T hessian(const VectorXT& p_curr, StiffnessMatrix& H, bool use_prev_equil = false) {}

public:
    ObjUMatching(Simulation& _simulation) : Objectives(_simulation) 
    {
        
    }
    ~ObjUMatching() {}
};


class ObjNucleiTracking : public Objectives
{
public:
    using VtxList = std::vector<int>;
    SpatialHash hash;
    MatrixXT cell_trajectories;
    int frame = 0;
    bool use_log_barrier = false;
    T barrier_distance = 1e-5;
    T barrier_weight = 1e3;
    bool add_min_act = false;
    T w_min_act = 1.0;

public:
    T value(const VectorXT& p_curr, bool use_prev_equil = false);
    T gradient(const VectorXT& p_curr, VectorXT& dOdp, bool use_prev_equil = false);
    T gradient(const VectorXT& p_curr, VectorXT& dOdp, T& energy, bool use_prev_equil = false);
    T evaluteGradientAndEnergy(const VectorXT& p_curr, VectorXT& dOdp, T& energy);
    void updateDesignParameters(const VectorXT& design_parameters);
    void getDesignParameters(VectorXT& design_parameters);
    void getSimulationAndDesignDoF(int& _sim_dof, int& _design_dof);
    void dOdx(const VectorXT& p_curr, VectorXT& _dOdx);
    void d2Odx2(const VectorXT& p_curr, std::vector<Entry>& d2Odx2_entries);

    T hessianGN(const VectorXT& p_curr, StiffnessMatrix& H, bool simulate = true, bool use_prev_equil = false);
    T hessian(const VectorXT& p_curr, StiffnessMatrix& H, bool use_prev_equil = false) {}

    void initializeTarget();

    void loadTargetTrajectory(const std::string& filename);
    bool getTargetTrajectoryFrame(VectorXT& frame_data);
    void updateTarget();

    void loadTarget(const std::string& filename);
    void initializeTargetFromMap(const std::string& filename, int _frame);
    T maximumStepSize(const VectorXT& dp);
    template<int order>
    T barrier(T d, T eps)
    {
        if (d <= 0)
            std::cout << "d has to be a positive value" << std::endl;
        if constexpr (order == 0)
            return - std::pow(d / eps - 1, 2) * std::log(d / eps);
        else if constexpr (order == 1)
            return (d - eps) * (-2 * d * std::log(d / eps) + eps - d) / (eps * eps * d);
        else
            return 1 / (d * d) + (2 / (d * eps) - 2 * std::log(d / eps) - 3) / (eps * eps);
    }
public: 
    ObjNucleiTracking(Simulation& _simulation) : Objectives(_simulation) 
    {
        default_optimizer = MMA;
    }
    ~ObjNucleiTracking() {}
};



class ObjFindInit : public Objectives
{
public:
    using VtxList = std::vector<int>;
    MatrixXT cell_trajectories;
    int frame = 0;

    SpatialHash hash;
public:
    T value(const VectorXT& p_curr, bool use_prev_equil = false);
    T gradient(const VectorXT& p_curr, VectorXT& dOdp, bool use_prev_equil = false);
    T gradient(const VectorXT& p_curr, VectorXT& dOdp, T& energy, bool use_prev_equil = false);
    T evaluteGradientAndEnergy(const VectorXT& p_curr, VectorXT& dOdp, T& energy);
    void updateDesignParameters(const VectorXT& design_parameters);
    void getDesignParameters(VectorXT& design_parameters);
    void getSimulationAndDesignDoF(int& _sim_dof, int& _design_dof);

    T hessianGN(const VectorXT& p_curr, StiffnessMatrix& H, bool simulate = true, bool use_prev_equil = false) {}
    T hessian(const VectorXT& p_curr, StiffnessMatrix& H, bool use_prev_equil = false);
    
    void updateTarget();
    void loadTargetTrajectory(const std::string& filename);
    bool getTargetTrajectoryFrame(VectorXT& frame_data);

    T maximumStepSize(const VectorXT& dp);

public: 
    ObjFindInit(Simulation& _simulation) : Objectives(_simulation) 
    {
        default_optimizer = Newton;
    }
    ~ObjFindInit() {}
};

#endif
