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
#include <unordered_map>
#include "VecMatDef.h"

#include "FEMSolver.h"
class FEMSolver;

enum Optimizer
{
    GradientDescent, GaussNewton, MMA, Newton
};

class Objective
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

    Optimizer default_optimizer = GaussNewton;
    
public:

    FEMSolver& solver;
    int n_dof_sim, n_dof_design;
    VectorXT equilibrium_prev;
    VectorXT X0;
    bool add_reg_rest = true;
    T w_reg_rest = 1e-3;
    bool add_reg_laplacian = true;
    T w_reg_laplacian = 1e-3;

    bool use_ipc = false;
    T barrier_distance = 1e-3;
    T barrier_weight = 1.0;
    int num_ipc_vtx = 0;
    Eigen::MatrixXd ipc_vertices;
    Eigen::MatrixXi ipc_edges;
    Eigen::MatrixXi ipc_faces;
    T max_barrier_weight = 1e8;

    bool add_pbc = true;
    T pbc_w = 1e6;
    std::vector<IV> pbc_pairs;

    Eigen::MatrixXd surface_vertices;
    Eigen::MatrixXi surface_faces;
    Eigen::SparseMatrix<T> cot_mat;

    std::unordered_map<int, T> dirichlet_data;
    
    template <class OP>
    void iterateDirichletDoF(const OP& f) {
        for (auto dirichlet: dirichlet_data){
            f(dirichlet.first, dirichlet.second);
        } 
    }

    std::vector<Entry> entriesFromSparseMatrix(const StiffnessMatrix& A)
    {
        std::vector<Entry> triplets;
        triplets.reserve(A.nonZeros());
        for (int k=0; k < A.outerSize(); ++k)
            for (StiffnessMatrix::InnerIterator it(A,k); it; ++it)
            {
                triplets.push_back(Entry(it.row(), it.col(), it.value()));
            }
        return triplets;
    }
    
public:
    virtual T value(const VectorXT& p_curr, bool simulate = true, bool use_prev_equil = false) {}
    virtual T gradient(const VectorXT& p_curr, VectorXT& dOdp, T& energy, bool simulate = true, bool use_prev_equil = false) {}
    virtual void hessianGN(const VectorXT& p_curr, MatrixXT& H, bool simulate = false, bool use_prev_equil = false) {}

    virtual void computeOx(const VectorXT& x, T& Ox) {}
    virtual void computedOdx(const VectorXT& x, VectorXT& dOdx) {}
    virtual void computed2Odx2(const VectorXT& x, std::vector<Entry>& d2Odx2_entries) {}

    virtual void computeOp(const VectorXT& p_curr, T& Op) {}
    virtual void computedOdp(const VectorXT& p_curr, VectorXT& dOdx) {}
    virtual void computed2Odp2(const VectorXT& p_curr, std::vector<Entry>& d2Odx2_entries) {}

    virtual void updateDesignParameters(const VectorXT& design_parameters) {}
    virtual void getDesignParameters(VectorXT& design_parameters) {}

    virtual void loadTargetFromFile(const std::string& filename) {}
    virtual void loadTarget(const std::string& data_folder) {}
    
    virtual void initialize() {}
    virtual T maximumStepSize(const VectorXT& p_curr, const VectorXT& search_dir) {}

    virtual void getDirichletIndices(std::vector<int>& indices) {}
    virtual void getDirichletMask(VectorXT& mask) {}

    virtual void generateTarget(const std::string& target_folder) {}

    void setX0(const VectorXT& _X0) { X0 = _X0; }
    void projectDesignParameters(VectorXT& design_parameters);

    virtual void diffTestGradientScale();
    void diffTestGradient();
    void diffTestdOdx();
    void diffTestdOdxScale();

    void buildIPCRestData();
    void updateIPCVertices(const VectorXT& new_position);
    void updateCotMat(const VectorXT& new_position);
    void getSandwichPanelIndices(std::vector<int>& indices);
    
    Objective(FEMSolver& _solver) : solver(_solver) 
    {
        
    }
    ~Objective() {}
};

class ObjFTF : public Objective
{
public:
    VectorXT f_target;
    VectorXT f_current;
    bool sequence = false;
    int num_data_point = 2;

public:
    void computeOx(const VectorXT& x, T& Ox);
    void computedOdx(const VectorXT& x, VectorXT& dOdx);
    void computed2Odx2(const VectorXT& x, std::vector<Entry>& d2Odx2_entries);

    void computeOp(const VectorXT& p_curr, T& Op);
    void computedOdp(const VectorXT& p_curr, VectorXT& dOdp);
    void computed2Odp2(const VectorXT& p_curr, std::vector<Entry>& d2Odp2_entries);

    T value(const VectorXT& p_curr, bool simulate = true, bool use_prev_equil = false);
    T gradient(const VectorXT& p_curr, VectorXT& dOdp, T& energy, 
        bool simulate = true, bool use_prev_equil = false);
    void hessianGN(const VectorXT& p_curr, MatrixXT& H, 
        bool simulate = false, bool use_prev_equil = false);
    
    void updateDesignParameters(const VectorXT& design_parameters);
    void getDesignParameters(VectorXT& design_parameters);
    void loadTargetFromFile(const std::string& filename);
    void loadTarget(const std::string& data_folder);
    void initialize();
    void getDirichletIndices(std::vector<int>& indices);
    void getDirichletMask(VectorXT& mask);
    T maximumStepSize(const VectorXT& p_curr, const VectorXT& search_dir);
    void generateTarget(const std::string& target_folder);
    void generateSequenceData(VectorXT& x, bool simulate = true, bool use_prev_equil = false);

    void diffTestGradientScale();
    
public:
    ObjFTF(FEMSolver& _solver) : Objective(_solver) {}
    ~ObjFTF() {}
};

#endif