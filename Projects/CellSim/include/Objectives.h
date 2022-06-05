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
#include "SoftPenalty.h"

enum Optimizer
{
    GradientDescent, GaussNewton, MMA, Newton, SGN, SQP, SSQP
};

enum PenaltyType
{
    LogBarrier, Qubic, Quadratic, BLEND
};

enum LinearSolverType
{
    PardisoLDLT, PardisoLU, EigenLU
};

class Objectives
{
public:
    using TV = Vector<double, 3>;
    using TM = Matrix<double, 3, 3>;
    using IV = Vector<int, 3>;
    using IV2 = Vector<int, 2>;
    using Edge = Vector<int, 2>;

    typedef int StorageIndex;
    using StiffnessMatrix = Eigen::SparseMatrix<T, Eigen::ColMajor, StorageIndex>;
    // using StiffnessMatrix = Eigen::SparseMatrix<T>;
    using Entry = Eigen::Triplet<T>;
    using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
    using MatrixXT = Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorXi = Vector<int, Eigen::Dynamic>;

    Optimizer default_optimizer = GaussNewton;
    LinearSolverType default_linear_solver = PardisoLDLT;
    
public:
    template <class OP>
    void iterateTargets(const OP& f) {
        for (auto dirichlet: target_positions){
            f(dirichlet.first, dirichlet.second);
        } 
    }
    template <class OP>
    void iterateWeightedTargets(const OP& f) {
        for (auto target_data: weight_targets){
            f(target_data.cell_idx, target_data.data_point_idx, target_data.target_pos, target_data.weights);
        } 
    }

public:

    Simulation& simulation;
    int n_dof_sim, n_dof_design;
    VectorXT equilibrium_prev;
    bool match_centroid = false;
    bool running_diff_test = false;
    bool use_penalty = false;
    T penalty_weight = 1e4;
    PenaltyType penalty_type = Qubic;
    bool perturb = true;
    bool add_reg = false;
    int power = 2;
    bool add_forward_potential = false;
    bool add_spatial_regularizor = false;
    
    T w_reg_spacial = 1e-3;
    T w_fp = 1e-3;
    T w_data = 1.0;
    std::string target_filename;
    T reg_w = 1e-6;
    T target_perturbation = 0.0;
    VectorXT rest_configuration;
    VectorXT prev_params;
    struct TargetData
    {
        int cell_idx;
        int data_point_idx;
        VectorXT weights;
        TV target_pos;

        TargetData(const VectorXT& _weights, int _idx, int _cell_idx) : data_point_idx(_idx), weights(_weights), cell_idx(_cell_idx) {}
        TargetData(const VectorXT& _weights, int _idx, int _cell_idx, const TV& _target_pos) : 
            data_point_idx(_idx), weights(_weights), cell_idx(_cell_idx), target_pos(_target_pos) {}
        TargetData() : data_point_idx(-1), cell_idx(-1) {}
    };

    

    Vector<T, 2> bound;
    Vector<bool, 2> mask;
    std::vector<TargetData> weight_targets;
    std::unordered_map<int, TV> target_positions;
    VectorXT target_obj_weights;

    virtual void setTargetObjWeights() {}

    virtual T value(const VectorXT& p_curr, bool simulate = true, bool use_prev_equil = false) {}
    virtual T gradient(const VectorXT& p_curr, VectorXT& dOdp, T& energy, bool simulate = true, bool use_prev_equil = false) {}
    virtual void hessianGN(const VectorXT& p_curr, MatrixXT& H, bool simulate = false, bool use_prev_equil = false) {}
    
    virtual void hessianSGN(const VectorXT& p_curr, StiffnessMatrix& H,
        bool simulate = false) {}
    virtual void hessian(const VectorXT& p_curr, StiffnessMatrix& H, bool use_prev_equil = false) {}


    virtual void assembleSGNHessian(StiffnessMatrix &A, const StiffnessMatrix &B,
	    const StiffnessMatrix &C, const StiffnessMatrix &dfdx,
	    const StiffnessMatrix &dfdp, StiffnessMatrix &KKT);
    virtual void assembleSGNHessianBCZero(StiffnessMatrix &A, const StiffnessMatrix &dfdx,
	    const StiffnessMatrix &dfdp, StiffnessMatrix &KKT);
    
    virtual void computeOx(const VectorXT& x, T& Ox) {}
    virtual void computedOdx(const VectorXT& x, VectorXT& dOdx) {}
    virtual void computed2Odx2(const VectorXT& x, std::vector<Entry>& d2Odx2_entries) {}

    virtual void computeOp(const VectorXT& p_curr, T& Op) {}
    virtual void computedOdp(const VectorXT& p_curr, VectorXT& dOdx) {}
    virtual void computed2Odp2(const VectorXT& p_curr, std::vector<Entry>& d2Odx2_entries) {}

    virtual void updateDesignParameters(const VectorXT& design_parameters) {}
    virtual void getDesignParameters(VectorXT& design_parameters) {}
    virtual void getSimulationAndDesignDoF(int& sim_dof, int& design_dof) {}
    virtual void setSimulationAndDesignDoF(int _sim_dof, int _design_dof);
    virtual void updateTarget() {}

    virtual T maximumStepSize(const VectorXT& dp) { return 1.0; }
    virtual void setOptimizer(Optimizer opt) { default_optimizer = opt; }
    virtual void setLinearSolver(LinearSolverType ls) { default_linear_solver = ls; }

    void saveState(const std::string& filename) { simulation.saveState(filename); }
    void saveDesignParameters(const std::string& filename, const VectorXT& design_parameters);
    
    void diffTestGradientScale();
    void diffTestGradient();
    void diffTestHessian();
    void diffTestHessianScale();
    void diffTestdOdx();
    void diffTestdOdxScale();
    void diffTestd2Odx2();
    void diffTestd2Odx2Scale();
    void diffTestPartialOPartialpScale();
    void diffTestPartialOPartialp();
    void diffTestPartial2OPartialp2Scale();
    void diffTestPartial2OPartialp2();

public:
    Objectives(Simulation& _simulation) : simulation(_simulation) 
    {
        // equilibrium_prev.resize(simulation.num_nodes * 3);
        // equilibrium_prev.setZero();
    }
    ~Objectives() {}
};

// class ObjUTU : public Objectives
// {
// public:
//     T value(const VectorXT& p_curr, bool use_prev_equil = false);
//     T gradient(const VectorXT& p_curr, VectorXT& dOdp, bool use_prev_equil = false);
//     T gradient(const VectorXT& p_curr, VectorXT& dOdp, T& energy, bool use_prev_equil = false);
//     T evaluteGradientAndEnergy(const VectorXT& p_curr, VectorXT& dOdp, T& energy);
//     void updateDesignParameters(const VectorXT& design_parameters);
//     void getDesignParameters(VectorXT& design_parameters);
//     void getSimulationAndDesignDoF(int& _sim_dof, int& _design_dof);
//     void updateTarget() {}

//     void hessianGN(const VectorXT& p_curr, StiffnessMatrix& H, bool simulate = false);
//     void hessian(const VectorXT& p_curr, StiffnessMatrix& H, bool use_prev_equil = false) {}

// public:
//     ObjUTU(Simulation& _simulation) : Objectives(_simulation) 
//     {
        
//     }
//     ~ObjUTU() {}
// };

// class ObjUMatching : public Objectives
// {
// private:
//     VectorXT target;

// public:
//     void setTargetFromMesh(const std::string& filename);

//     T value(const VectorXT& p_curr, bool use_prev_equil = false);
//     T gradient(const VectorXT& p_curr, VectorXT& dOdp, bool use_prev_equil = false);
//     T gradient(const VectorXT& p_curr, VectorXT& dOdp, T& energy, bool use_prev_equil = false);
//     T evaluteGradientAndEnergy(const VectorXT& p_curr, VectorXT& dOdp, T& energy);
//     void updateDesignParameters(const VectorXT& design_parameters);
//     void getDesignParameters(VectorXT& design_parameters);
//     void getSimulationAndDesignDoF(int& _sim_dof, int& _design_dof);
//     void updateTarget() {}
//     T hessianGN(const VectorXT& p_curr, StiffnessMatrix& H, bool simulate = true, bool use_prev_equil = false);
//     void hessian(const VectorXT& p_curr, StiffnessMatrix& H, bool use_prev_equil = false) {}

// public:
//     ObjUMatching(Simulation& _simulation) : Objectives(_simulation) 
//     {
        
//     }
//     ~ObjUMatching() {}
// };


class ObjNucleiTracking : public Objectives
{
public:
    using VtxList = std::vector<int>;
    SpatialHash hash;
    MatrixXT cell_trajectories;
    int frame = 0;
    
    T barrier_distance = 1e-3;
    T barrier_weight = 1e6;
    bool add_min_act = false;
    T w_min_act = 1.0;
    
    std::vector<std::vector<int>> vtx_edges;

    bool add_Hessian_PD_term = false;
    Vector<T, 2> bound;
    
    int wrapper_type = 0;
    
public:
    void buildVtxEdgeStructure();
    T simHessianPDEnergy(const VectorXT& p_curr);
    void simHessianPDGradient(const VectorXT& p_curr, T& energy, VectorXT& grad);
    void simHessianPDHessian(const VectorXT& p_curr, StiffnessMatrix& hess);
    
    T value(const VectorXT& p_curr, bool simulate = true, bool use_prev_equil = false);
    T gradient(const VectorXT& p_curr, VectorXT& dOdp, T& energy, bool simulate = true, bool use_prev_equil = false);
    void hessianGN(const VectorXT& p_curr, MatrixXT& H, bool simulate = false, bool use_prev_equil = false);

    void computeEnergyAllTerms(const VectorXT& p_curr, std::vector<T>& energies,
        bool simulate = true, bool use_prev_equil = false);

    void SGNforQP(const VectorXT& p_curr, StiffnessMatrix& objective_hessian, 
        StiffnessMatrix& constraint_jacobian);
    
    void hessianSGN(const VectorXT& p_curr, StiffnessMatrix& H,
        bool simulate = false);
    void hessian(const VectorXT& p_curr, StiffnessMatrix& H, bool use_prev_equil = false) {}

    void updateDesignParameters(const VectorXT& design_parameters);
    void getDesignParameters(VectorXT& design_parameters);
    void getSimulationAndDesignDoF(int& _sim_dof, int& _design_dof);

    void computeOx(const VectorXT& x, T& Ox);
    void computedOdx(const VectorXT& x, VectorXT& _dOdx);
    void computed2Odx2(const VectorXT& x, std::vector<Entry>& d2Odx2_entries);

    void computeOp(const VectorXT& p_curr, T& Op);
    void computedOdp(const VectorXT& p_curr, VectorXT& dOdp);
    void computed2Odp2(const VectorXT& p_curr, std::vector<Entry>& d2Odp2_entries);

    void computeKernelWeights();
    void computeCellTargetFromDatapoints();
    void computeCellTargetsFromDatapoints(const std::string& filename);

    void initializeTarget();

    void loadTargetTrajectory(const std::string& filename);

    bool getTargetTrajectoryFrame(VectorXT& frame_data);
    void updateTarget();
    void rotateTarget(T angle);
    void optimizeForStableTarget(T perturbation);
    void optimizeForStableTargetSpring(T perturbation);
    void optimizeForStableTargetDeformationGradient(T perturbation);
    void loadTarget(const std::string& filename, T perturbation = 0.0);
    void loadWeightedTarget(const std::string& filename);
    void loadWeightedCellTarget(const std::string& filename);
    void filterTrackingData3X2F();
    void filterTrackingData3X3F();
    void setTargetObjWeights();
    void loadOBJwithOnlyVertices(const std::string& filename, VectorXT& data);
    void loadTxtVertices(const std::string& filename, VectorXT& data);
    void checkData();

    void checkDistanceMetrics();
    void initializeTargetFromMap(const std::string& filename, int _frame);
    T maximumStepSize(const VectorXT& dp);

    void setFrame(int _frame) { frame = _frame; }
    
    template <int order>
    T soft_barrier(T p)
    {
        return 1.0;
    }

    template <int order>
    T wrapper(T p)
    {
        T b = bound[1] / 2.0;
        T a = -2.2;
        T c = 0.05;

        if constexpr (order == 0)
        {
            if (wrapper_type == 0)
                return p;
            else if (wrapper_type == 1)
                return p * p; 
            else if (wrapper_type == 2)
                return b + b * std::tanh(c * p + a);
        }
        else if constexpr (order == 1)
        {
            if (wrapper_type == 0)
                return 1.0;
            else if (wrapper_type == 1)
                return 2.0 * p; 
            else if (wrapper_type == 2)
                return c * b * (1.0 - std::pow(std::tanh(c * p + a), 2));
        }
        else if constexpr (order == 2)
        {
            if (wrapper_type == 0)
                return 1.0;
            else if (wrapper_type == 1)
                return 2.0;
            else if (wrapper_type == 2)
                return -2.0 * c * c * b * std::tanh(c * p + a) * (1.0 - std::pow(std::tanh(c * p + a), 2));
        }

    }

    void testWrapperDerivatives()
    {
        T p = 0.2;
        T eps = 1e-5;
        std::cout << (wrapper<0>(p + eps) - wrapper<0>(p)) / eps << " " << wrapper<1>(p) << std::endl;
        std::cout << (wrapper<1>(p + eps) - wrapper<1>(p)) / eps << " " << wrapper<2>(p) << std::endl;
    }
    
private:
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
    T value(const VectorXT& p_curr, bool simulate = true, bool use_prev_equil = false);
    T gradient(const VectorXT& p_curr, VectorXT& dOdp, T& energy, bool simulate = true, bool use_prev_equil = false);
    void hessianGN(const VectorXT& p_curr, MatrixXT& H, bool simulate = false, bool use_prev_equil = false) {}
    
    void hessianSGN(const VectorXT& p_curr, StiffnessMatrix& H, 
        bool simulate = false) {}
    void hessian(const VectorXT& p_curr, StiffnessMatrix& H, bool use_prev_equil = false);

    void updateDesignParameters(const VectorXT& design_parameters);
    void getDesignParameters(VectorXT& design_parameters);
    void getSimulationAndDesignDoF(int& _sim_dof, int& _design_dof);

    
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
