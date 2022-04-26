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

class Objective
{
public:
    using TV = Vector<double, 2>;
    using TM = Matrix<double, 2, 2>;
    using IV = Vector<int, 2>;
    using Edge = Vector<int, 2>;

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

    VectorXT prev_params;

public:
    template <class OP>
    void iterateTargets(const OP& f) {
        for (auto dirichlet: target_positions){
            f(dirichlet.first, dirichlet.second);
        } 
    }

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

    Objective(VertexModel2D& _vertex_model) : vertex_model(_vertex_model) {}
    ~Objective() {}
};


#endif