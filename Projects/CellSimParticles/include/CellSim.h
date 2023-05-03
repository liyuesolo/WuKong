#ifndef CELLSIM_H
#define CELLSIM_H

#include <utility>
#include <iostream>
#include <fstream>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <tbb/tbb.h>
#include <tgmath.h>

// https://gist.github.com/zishun/da277d30f4604108029d06db0e804773
template<class Matrix>
inline void write_binary(const std::string& filename, const Matrix& matrix){
    std::ofstream out(filename, std::ios::out | std::ios::binary | std::ios::trunc);
    if(out.is_open()) {
        typename Matrix::Index rows=matrix.rows(), cols=matrix.cols();
        out.write(reinterpret_cast<char*>(&rows), sizeof(typename Matrix::Index));
        out.write(reinterpret_cast<char*>(&cols), sizeof(typename Matrix::Index));
        out.write(reinterpret_cast<const char*>(matrix.data()), rows*cols*static_cast<typename Matrix::Index>(sizeof(typename Matrix::Scalar)) );
        out.close();
    }
    else {
        std::cout << "Can not write to file: " << filename << std::endl;
    }
}

template<class Matrix>
inline void read_binary(const std::string& filename, Matrix& matrix){
    std::ifstream in(filename, std::ios::in | std::ios::binary);
    if (in.is_open()) {
        typename Matrix::Index rows=0, cols=0;
        in.read(reinterpret_cast<char*>(&rows),sizeof(typename Matrix::Index));
        in.read(reinterpret_cast<char*>(&cols),sizeof(typename Matrix::Index));
        matrix.resize(rows, cols);
        in.read(reinterpret_cast<char*>(matrix.data()), rows*cols*static_cast<typename Matrix::Index>(sizeof(typename Matrix::Scalar)) );
        in.close();
    }
    else {
        std::cout << "Can not open binary matrix file: " << filename << std::endl;
    }
}

#include "VecMatDef.h"
#include "Timer.h"
#include "SDF.h"
#include "SpatialHash.h"

template <int dim>
struct VectorHash
{
    typedef Vector<int, dim> IV;
    size_t operator()(const IV& a) const{
        std::size_t h = 0;
        for (int d = 0; d < dim; ++d) {
            h ^= std::hash<int>{}(a(d)) + 0x9e3779b9 + (h << 6) + (h >> 2); 
        }
        return h;
    }
};

template <int dim>
struct VectorPairHash
{
    typedef Vector<int, dim> IV;
    size_t operator()(const std::pair<IV, IV>& a) const{
        std::size_t h = 0;
        for (int d = 0; d < dim; ++d) {
            h ^= std::hash<int>{}(a.first(d)) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<int>{}(a.second(d)) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return h;
    }
};  

enum Region
{
    Apical, Basal, Lateral, ALL
};

template<int dim>
class CellSim
{
public:
    using TV = Vector<T,  dim>;
    using TM = Matrix<T,  dim, dim>;
    using TV3 = Vector<double, 3>;
    using TV2 = Vector<T, 2>;
    using TM2 = Matrix<T, 2, 2>;
    using TM3 = Matrix<T, 3, 3>;
    using IV = Vector<int, dim>;
    using IV3 = Vector<int, 3>;
    using IV2 = Vector<int, 2>;

    using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
    using VectorXi = Vector<int, Eigen::Dynamic>;

    using MatrixXT = Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using MatrixXi = Matrix<int, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorXli = Matrix<long int, Eigen::Dynamic, 1>;

    using VtxList = std::vector<int>;
    using FaceList = std::vector<int>;

    using Edge = Vector<int, 2>;
    using StiffnessMatrix = Eigen::SparseMatrix<T>;
    using Entry = Eigen::Triplet<T>;

public:
    int num_nodes;
    int num_cells;
    VectorXi is_control_points;
    int global_frame = 0;
    int n_frames = 0;
    MatrixXT target_trajectories;
    VectorXT target_positions;
    T radius = 0.1;
    VectorXT undeformed, deformed, u;
    bool woodbury = false;
    
    std::unordered_map<int, T> dirichlet_data;

    // simulation 
    bool run_diff_test = false;
    T newton_tol = 1e-6;
    int max_newton_iter = 1000;
    bool lower_triangular = false;

    //position matching
    T w_matching = 1e3;
    VectorXT control_points;
    std::vector<Edge> control_edges;
    VectorXT control_edge_rest_length;
    std::unordered_map<int, int> ctrl_point_data_map;
    
    // repulsion
    T w_rep = 0.1;
    MatrixXi vv_flag;
    SpatialHash<dim> cell_hash;
    T collision_dhat = 0.02;
    
    // adhesion
    T w_adh = 0.1;
    std::vector<Edge> adhesion_edges;
    T adhesion_dhat = 0.02;

    // Membrane
    IMLS<dim> sdf;
    T bound_coeff = 1e3;
    bool use_surface_membrane = true;
    MatrixXT membrane_vtx;
    MatrixXi membrane_faces;
    
    // yolk
    int yolk_cell_starts = -1;
    TV centroid;
    T yolk_area_rest;
    T w_yolk = 1e4;
    T w_reg_edge = 0.01;
    std::vector<Edge> yolk_edges;
    VectorXT rest_edge_length;
    Eigen::MatrixXi yolk_faces;

    // IPC
    bool use_ipc = true;
    Eigen::MatrixXd ipc_vertices;
    Eigen::MatrixXi ipc_edges, ipc_faces;
    T max_barrier_weight = 1e10;
    T ipc_barrier_distance = 1e-3;
    T ipc_barrier_weight = 1e4;
    T ipc_min_dis = 1e-6;
    

    // printouts
    bool print_force_norm = true;
    bool verbose = true;
    

public:
    template <class OP>
    void iterateDirichletDoF(const OP& f) {
        for (auto dirichlet: dirichlet_data){
            f(dirichlet.first, dirichlet.second);
        } 
    }

    template <typename OP>
    void iterateCellSerial(const OP& f)
    {
        for (int i = 0; i < num_cells; i++)
        {
            f(i);
        }
    }

    template <typename OP>
    void iterateCellParallel(const OP& f)
    {
        tbb::parallel_for(0, num_cells, [&](int i)
        {
            f(i);
        });
    }

    template <typename OP>
    void iterateNodeParallel(const OP& f)
    {
        tbb::parallel_for(0, num_nodes, [&](int i)
        {
            f(i);
        });
    }

    std::vector<Entry> entriesFromSparseMatrix(const StiffnessMatrix& A, bool lower_tri_only = false)
    {
        std::vector<Entry> triplets;
        triplets.reserve(A.nonZeros());
        for (int k=0; k < A.outerSize(); ++k)
            for (StiffnessMatrix::InnerIterator it(A,k); it; ++it)
            {
                if (!lower_tri_only)
                    triplets.push_back(Entry(it.row(), it.col(), it.value()));
                else
                {
                    if (it.row() <= it.col())
                        triplets.push_back(Entry(it.row(), it.col(), it.value()));
                }
            }
        return triplets;
    }

private:
    template<int size>
    void addHessianBlock(
        std::vector<Entry>& triplets,
        const std::vector<int>& vtx_idx, 
        const Matrix<T, size, size>& hessian_block)
    {
        int dof_i = vtx_idx[0];
        int dof_j = vtx_idx[1];

        for (int k = 0; k < size; k++)
            for (int l = 0; l < size; l++)
            {
                if (std::abs(hessian_block(k, l)) > 1e-8)
                    triplets.push_back(Entry(dof_i * dim + k, dof_j * dim + l, hessian_block(k, l)));
            }
    }

    template<int size>
    void addHessianEntry(
        std::vector<Entry>& triplets,
        const std::vector<int>& vtx_idx, 
        const Matrix<T, size, size>& hessian)
    {
        if (vtx_idx.size() * dim != size)
            std::cout << "wrong hessian block size" << std::endl;

        for (int i = 0; i < vtx_idx.size(); i++)
        {
            int dof_i = vtx_idx[i];
            for (int j = 0; j < vtx_idx.size(); j++)
            {
                int dof_j = vtx_idx[j];
                for (int k = 0; k < dim; k++)
                    for (int l = 0; l < dim; l++)
                    {
                        if (std::abs(hessian(i * dim + k, j * dim + l)) > 1e-8)
                            triplets.push_back(Entry(dof_i * dim + k, dof_j * dim + l, hessian(i * dim + k, j * dim + l)));  
                    }
            }
        }
    }

    template<int size>
    void addForceEntry(VectorXT& residual, 
        const std::vector<int>& vtx_idx, 
        const Vector<T, size>& gradent)
    {
        if (vtx_idx.size() * dim != size)
            std::cout << "wrong gradient block size in addForceEntry" << std::endl;

        for (int i = 0; i < vtx_idx.size(); i++)
            residual.segment<dim>(vtx_idx[i] * dim) += gradent.template segment<dim>(i * dim);
    }

    template<int size>
    void getSubVector(const VectorXT& _vector, 
        const std::vector<int>& vtx_idx, 
        Vector<T, size>& sub_vec)
    {
        if (vtx_idx.size() * dim != size)
            std::cout << "wrong gradient block size in getSubVector" << std::endl;

        sub_vec = Vector<T, size>::Zero(vtx_idx.size() * dim);
        for (int i = 0; i < vtx_idx.size(); i++)
        {
            sub_vec.template segment<dim>(i * dim) = _vector.segment<dim>(vtx_idx[i] * dim);
        }
    }

    template<int size>
    bool isHessianBlockPD(const Matrix<T, size, size> & symMtr)
    {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, size, size>> eigenSolver(symMtr);
        // sorted from the smallest to the largest
        if (eigenSolver.eigenvalues()[0] >= 0.0) 
            return true;
        else
            return false;
        
    }

    template<int size>
    VectorXT computeHessianBlockEigenValues(const Matrix<T, size, size> & symMtr)
    {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, size, size>> eigenSolver(symMtr);
        return eigenSolver.eigenvalues();
    }

    template <int size>
    void projectBlockPD(Eigen::Matrix<T, size, size>& symMtr)
    {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, size, size>> eigenSolver(symMtr);
        if (eigenSolver.eigenvalues()[0] >= 0.0) {
            return;
        }
        Eigen::DiagonalMatrix<T, size> D(eigenSolver.eigenvalues());
        int rows = ((size == Eigen::Dynamic) ? symMtr.rows() : size);
        for (int i = 0; i < rows; i++) {
            if (D.diagonal()[i] < 0.0) {
                D.diagonal()[i] = 0.0;
            }
            else {
                break;
            }
        }
        symMtr = eigenSolver.eigenvectors() * D * eigenSolver.eigenvectors().transpose();
    }
    

public:
    CellSim<dim>(/* args */) {}
    ~CellSim<dim>() {}

// SIMULATION
    void reset();
    bool advanceOneStep(int step);
    bool staticSolve();
    void buildSystemMatrix(const VectorXT& _u, StiffnessMatrix& K);
    void buildSystemMatrixWoodbury(const VectorXT& _u, 
        StiffnessMatrix& K, MatrixXT& UV);
    T computeTotalEnergy(const VectorXT& _u, bool add_to_deform = true);
    T computeResidual(const VectorXT& _u,  VectorXT& residual);
    T lineSearchNewton(VectorXT& _u,  VectorXT& residual, int ls_max = 12);
    bool solveWoodbury(StiffnessMatrix& K, MatrixXT& UV,
         VectorXT& residual, VectorXT& du);
    bool linearSolve(StiffnessMatrix& K, 
        VectorXT& residual, VectorXT& du);
    void projectDirichletDoFMatrix(StiffnessMatrix& A, 
        const std::unordered_map<int, T>& data);
    T computeLineSearchInitStepsize(const VectorXT& _u, const VectorXT& du);
    void advanceOneFrame();
    void loadStates(const std::string& filename);

    void checkTotalGradient(bool perturb = true);
    void checkTotalHessian(bool perturb = true);
    void checkTotalGradientScale(bool perturb = true);
    void checkTotalHessianScale(bool perturb = true);

// IPC.cpp
    void updateBarrierInfo(bool first_step);
    void buildIPCRestData();
    void addIPCEnergy(T& energy);
    void addIPCForceEntries(VectorXT& residual);
    void addIPCHessianEntries(std::vector<Entry>& entries);
    void updateIPCVertices(const VectorXT& _u);
    T computeCollisionFreeStepsize(const VectorXT& _u, const VectorXT& du);
    void saveIPCMesh(bool save_edges = true);

// Processing.cpp
    void removeVtxInsideYolk();

// Yolk.cpp
    bool isYolkParticle(int i) 
    { 
        if (yolk_cell_starts == -1) 
            return false;
        else
            return i >= yolk_cell_starts; 
    }
    void constructYolkMesh2D();
    void constructYolkMesh3D();
    T computeYolkArea();
    void addYolkPreservationEnergy(T& energy);
    void addYolkPreservationForceEntries(VectorXT& residual);
    void addYolkPreservationHessianEntries(std::vector<Entry>& entries, MatrixXT& WoodBuryMatrix);
    void addYolkCollisionEnergy(T& energy);
    void addYolkCollisionForceEntries(VectorXT& residual);
    void addYolkCollisionHessianEntries(std::vector<Entry>& entries);

    void addYolkEdgeRegEnergy(T& energy);
    void addYolkEdgeRegForceEntries(VectorXT& residual);
    void addYolkEdgeRegHessianEntries(std::vector<Entry>& entries);

// PositionMatching.cpp
    void updatePerFrameData();
    void initializeControlPointsData();
    void addMatchingEnergy(T& energy);
    void addMatchingForceEntries(VectorXT& residual);
    void addMatchingHessianEntries(std::vector<Entry>& entries);

// Adhesion.cpp
    void computeInitialNeighbors();
    void addAdhesionEnergy(T& energy);
    void addAdhesionForceEntries(VectorXT& residual);
    void addAdhesionHessianEntries(std::vector<Entry>& entries, bool projectPD = false);

// Repulsion.cpp
    void addRepulsionEnergy(T& energy);
    void addRepulsionForceEntries(VectorXT& residual);
    void addRepulsionHessianEntries(std::vector<Entry>& entries, bool projectPD = false);

// Membrane.cpp
    void checkMembranePenetration();
    void constructMembraneLevelset();
    void addMembraneEnergy(T& energy);
    void addMembraneForceEntries(VectorXT& residual);
    void addMembraneHessianEntries(std::vector<Entry>& entries, bool projectPD = false);

// Scene.cpp
    void initializeCells();
    void initializeFrom3DData();
    
    void loadTargetTrajectories();
    void updateTargetPointsAsBC();
    void generateMeshForRendering(
        Eigen::MatrixXd& V, Eigen::MatrixXi& F, 
        Eigen::MatrixXd& C, bool show_yolk,
        T cell_radius, T yolk_radius);

    void computeBoundingBox(TV& min_corner, TV& max_corner)
    {
        min_corner.setConstant(1e6);
        max_corner.setConstant(-1e6);
        
        for (int i = 0; i < num_cells; i++)
        {
            for (int d = 0; d < dim; d++)
            {
                max_corner[d] = std::max(max_corner[d], undeformed[i * dim + d]);
                min_corner[d] = std::min(min_corner[d], undeformed[i * dim + d]);
            }
        }
    }
    void appendCylindersToEdges(const std::vector<std::pair<TV3, TV3>>& edge_pairs, 
        const TV3& color, T radius,
        Eigen::MatrixXd& _V, Eigen::MatrixXi& _F, Eigen::MatrixXd& _C);

    void appendSphereToPositionVector(const VectorXT& position, T radius, 
        const TV3& color,
        Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& C);

    void sampleBoundingSurface(MatrixXT& surface_samples)
    {
        VectorXT sdf_samples;
        sdf.sampleZeroLevelset(sdf_samples);
        surface_samples.resize(sdf_samples.rows()/ dim, 3);
        surface_samples.setZero();
        for (int i = 0; i < sdf_samples.rows() / dim; i++)
            surface_samples.row(i).segment<dim>(0) = sdf_samples.segment<dim>(i * dim);
    }

    void saveState(const std::string& filename, const VectorXT& positions)
    {
        std::ofstream out(filename);
        for (int i = 0; i < positions.rows()/dim; i++)
        {
            if constexpr (dim == 2)
                out << "v " << positions.segment<dim>(i * dim).transpose() << " 0.0" << std::endl;
            else
                out << "v " << positions.segment<dim>(i * dim).transpose() << std::endl;
        }
        out.close();
    }
};

#endif
