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


#include "VertexModel2D.h"

class VertexModel2D;

enum Optimizer
{
    GradientDescent, GaussNewton, MMA, Newton, SGN, SQP, SSQP
};

enum PenaltyType
{
    LogBarrier, Qubic, Quadratic, BLEND
};

struct TargetData
{
    using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
    using TV = Vector<double, 2>;

    int cell_idx;
    int data_point_idx;
    VectorXT weights;
    TV target_pos;

    TargetData(const VectorXT& _weights, int _idx, int _cell_idx) : data_point_idx(_idx), weights(_weights), cell_idx(_cell_idx) {}
    TargetData(const VectorXT& _weights, int _idx, int _cell_idx, const TV& _target_pos) : 
        data_point_idx(_idx), weights(_weights), cell_idx(_cell_idx), target_pos(_target_pos) {}
    TargetData() : data_point_idx(-1), cell_idx(-1) {}
};

class Objective
{
public:
    using TV = Vector<double, 2>;
    using TM = Matrix<double, 2, 2>;
    using IV = Vector<int, 2>;
    using Edge = Vector<int, 2>;
    using VtxList = std::vector<int>;

    typedef int StorageIndex;
    using StiffnessMatrix = Eigen::SparseMatrix<T, Eigen::ColMajor, StorageIndex>;
    // using StiffnessMatrix = Eigen::SparseMatrix<T>;
    using Entry = Eigen::Triplet<T>;
    using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
    using MatrixXT = Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorXi = Vector<int, Eigen::Dynamic>;

    Optimizer optimizer = SGN;

public:
    VertexModel2D& vertex_model;
    Vector<T, 2> bound = Vector<T, 2>::Zero();
    Vector<bool, 2> mask = Vector<bool, 2>::Zero();
    std::unordered_map<int, TV> target_positions;

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
    T w_fp = 1e-3;
    T w_data = 1.0;
    std::string target_filename;
    T target_perturbation = 0;
    T reg_w = 1e-6;

    // Regularizors
    bool add_spatial_regularizor = true;
    T w_reg_spacial = 0.01;

    bool add_l1_reg = false;
    T w_l1 = 0.01;
    bool contracting_term_only = false;
    VectorXT prev_params;

    std::vector<TargetData> weight_targets;
    
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

    void loadWeightedCellTarget(const std::string& filename, const std::string& data_file);
    void computeWeights(const std::string& filename, const std::string& data_file);

    void optimizeForStableTarget(T perturbation);
    void optimizeStableTargetsWithSprings(const std::string& rest_data_file, 
        const std::string& data_file, T perturbance = 0.0);
    void optimizeStableTargetsWithSprings(T perturbance = 0.0);

    void computeEnergySubTerms(std::vector<T>& energy_terms);
    
    void saveState(const std::string& filename);
    void loadTarget(const std::string& filename, T perturbation);
    void getSimulationAndDesignDoF(int& _sim_dof, int& _design_dof);

    void computeOx(const VectorXT& x, T& Ox);
    void computedOdx(const VectorXT& x, VectorXT& dOdx);
    void computed2Odx2(const VectorXT& x, std::vector<Entry>& d2Odx2_entries);

    void computeOp(const VectorXT& p_curr, T& Op);
    void computedOdp(const VectorXT& p_curr, VectorXT& dOdx);
    void computed2Odp2(const VectorXT& p_curr, std::vector<Entry>& d2Odx2_entries);

    void updateDesignParameters(const VectorXT& design_parameters);
    void getDesignParameters(VectorXT& design_parameters);
    
    T value(const VectorXT& p_curr, bool simulate = true, bool use_prev_equil = false);
    T gradient(const VectorXT& p_curr, VectorXT& dOdp, T& energy, bool simulate = true, bool use_prev_equil = false);
    void hessianSGN(const VectorXT& p_curr, StiffnessMatrix& H, bool simulate = false);
    void hessianGN(const VectorXT& p_curr, MatrixXT& H, bool simulate = false, bool use_prev_equil = false);

    void diffTestGradientScale();
    void diffTestGradient();
    
    void diffTestPartialOPartialPScale();
    void diffTestPartial2OPartialP2Scale();
    void diffTestPartial2OPartialP2();

    Objective(VertexModel2D& _vertex_model) : vertex_model(_vertex_model) {}
    ~Objective() {}
private:
    void load2DDataWithCellIdx(const std::string& filename, VectorXT& data);
};


#endif