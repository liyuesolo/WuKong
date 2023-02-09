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
#include "VecMatDef.h"
#include "Timer.h"

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


class CellSim
{
public:
    using TV = Vector<T, 3>;
    using TV2 = Vector<T, 2>;
    using TM2 = Matrix<T, 2, 2>;
    using TM3 = Matrix<T, 3, 3>;
    using TM = Matrix<T, 3, 3>;
    using IV = Vector<int, 3>;
    using IV2 = Vector<int, 2>;
    using TetVtx = Matrix<T, 3, 4>;

    using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
    using VectorXi = Vector<int, Eigen::Dynamic>;

    using MatrixXT = Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using MatrixXi = Matrix<int, Eigen::Dynamic, Eigen::Dynamic>;

    using VtxList = std::vector<int>;
    using FaceList = std::vector<int>;

    using Edge = Vector<int, 2>;
    using StiffnessMatrix = Eigen::SparseMatrix<T>;
    using Entry = Eigen::Triplet<T>;

public:
    int num_nodes;
    int num_cells;

    VectorXT undeformed, deformed, u;
    TV mesh_centroid;

    std::unordered_map<int, T> dirichlet_data;

    // simulation 
    bool run_diff_test = false;
    T newton_tol = 1e-6;
    int max_newton_iter = 500;
    bool woodbury = true;
    bool project_block_hessian_PD = false;
    bool lower_triangular = false;

    // yolk
    int yolk_vtx_start = 0;
    std::vector<VtxList> yolk_cells;
    T yolk_vol_init= 0.0;
    bool add_yolk_volume = true;
    T By = 1e4;

    // cells
    std::vector<int> cell_vtx_start;
    std::vector<VtxList> faces;
    VectorXT cell_volume_init;
    T B = 1e4;

    // edges
    VectorXT rest_length;
    T w_edges = 0.1;
    std::vector<Edge> cell_edges; // cell edges

    // faces
    T w_faces = 0.1;

    // IPC
    bool use_ipc = true;
    T ipc_min_dis = 1.0;
    T max_barrier_weight = 1e8;
    T barrier_distance = 1e-4;
    T barrier_weight = 1e6;
    MatrixXT ipc_vertices;
    MatrixXi ipc_edges;
    MatrixXi ipc_faces;

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
    void iterateEdgeSerial(const OP& f)
    {
        for (int i = 0; i < cell_edges.size(); i++)
        {
            f(cell_edges[i], i);
        }
    }

    template <typename OP>
    void iterateFaceSerial(const OP& f)
    {
        for (int i = 0; i < faces.size(); i++)
        {
            f(faces[i], i);
        }
    }
    template <typename OP>
    void iterateYolkAndCellFaceSerial(const OP& f)
    {
        for (int i = 0; i < faces.size(); i++)
        {
            f(faces[i], i);
        }
        for (int i = 0; i < yolk_cells.size(); i++)
        {
            f(yolk_cells[i], i);
        }
    }

    template <typename OP>
    void iterateFaceParallel(const OP& f)
    {
        tbb::parallel_for(0, (int)faces.size(), [&](int i)
        {
            f(faces[i], i);
        });
    }

    template <typename OP>
    void iterateCellSerial(const OP& f)
    {
        for (int i = 0; i < cell_vtx_start.size(); i++)
        {
            VtxList vtx_list;
            if (i < cell_vtx_start.size() - 1)
            {
                for (int j = cell_vtx_start[i]; j < cell_vtx_start[i + 1]; j++)
                    vtx_list.push_back(j);
            }
            else
            {
                for (int j = cell_vtx_start[i]; j < yolk_vtx_start; j++)
                    vtx_list.push_back(j);
            }
            f(vtx_list, i);
        }
    }

    template <typename OP>
    void iterateCellParallel(const OP& f)
    {
        tbb::parallel_for(0, (int)cell_vtx_start.size(), [&](int i)
        {
            VtxList vtx_list;
            if (i < cell_vtx_start.size() - 1)
            {
                for (int j = cell_vtx_start[i]; j < cell_vtx_start[i + 1]; j++)
                    vtx_list.push_back(j);
            }
            else
            {
                for (int j = cell_vtx_start[i]; j < yolk_vtx_start; j++)
                    vtx_list.push_back(j);
            }
            f(vtx_list, i);
        });
    }

    template <typename OP>
    void iterateYolkFaceSerial(const OP& f)
    {
        for (int i = 0; i < yolk_cells.size(); i++)
        {
            f(yolk_cells[i], i);
        }
    }

    template <typename OP>
    void iterateYolkFaceParallel(const OP& f)
    {
        tbb::parallel_for(0, (int)yolk_cells.size(), [&](int i)
        {
            f(yolk_cells[i], i);
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
    template<int dim>
    void addHessianEntry(
        std::vector<Entry>& triplets,
        const std::vector<int>& vtx_idx, 
        const Matrix<T, dim, dim>& hessian)
    {
        if (vtx_idx.size() * 3 != dim)
            std::cout << "wrong hessian block size" << std::endl;
        int n_curr = triplets.size();
        int cnt = 0;
        
        int n_entry = (1 + vtx_idx.size() * 3) * vtx_idx.size() * 3 / 2;
        
        std::vector<T> values(vtx_idx.size() * vtx_idx.size() * 3 * 3);
        std::vector<Entry> local_triplets(vtx_idx.size() * vtx_idx.size() * 3 * 3);

        auto offset = [&](int i, int j, int k, int l) -> int
        {
            return i * vtx_idx.size() * 3 * 3 + j * 3 * 3 + k * 3 + l;
        };
        for (int i = 0; i < vtx_idx.size(); i++)
        {
            int dof_i = vtx_idx[i];
            for (int j = 0; j < vtx_idx.size(); j++)
            {
                int dof_j = vtx_idx[j];
                if (lower_triangular)
                    if (dof_j > dof_i) continue;
                for (int k = 0; k < 3; k++)
                    for (int l = 0; l < 3; l++)
                    {
                        if (lower_triangular && (dof_i == dof_j))
                            if (l > k) continue;
                        {
                            triplets.push_back(Entry(dof_i * 3 + k, dof_j * 3 + l, hessian(i * 3 + k, j * 3 + l)));                
                        }
                    }
            }
        }
    }

    inline T computeTetVolume(const TV& a, const TV& b, const TV& c, const TV& d)
    {
        return 1.0 / 6.0 * (b - a).cross(c - a).dot(d - a);
    };

    template<int dim>
    bool isHessianBlockPD(const Matrix<T, dim, dim> & symMtr)
    {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, dim, dim>> eigenSolver(symMtr);
        // sorted from the smallest to the largest
        if (eigenSolver.eigenvalues()[0] >= 0.0) 
            return true;
        else
            return false;
        
    }

    template<int dim>
    VectorXT computeHessianBlockEigenValues(const Matrix<T, dim, dim> & symMtr)
    {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, dim, dim>> eigenSolver(symMtr);
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

    

    template<int dim>
    void addHessianBlock(
        std::vector<Entry>& triplets,
        const std::vector<int>& vtx_idx, 
        const Matrix<T, dim, dim>& hessian_block)
    {

        int dof_i = vtx_idx[0];
        int dof_j = vtx_idx[1];
        if (lower_triangular)
            if (dof_j > dof_i) return;
        for (int k = 0; k < dim; k++)
            for (int l = 0; l < dim; l++)
            {
                if (lower_triangular && (dof_i == dof_j))
                        if (l > k) continue;
                if (std::abs(hessian_block(k, l)) > 1e-8)
                    triplets.push_back(Entry(dof_i * 3 + k, dof_j * 3 + l, hessian_block(k, l)));
            }
    }

    template<int dim>
    void addForceEntry(VectorXT& residual, 
        const std::vector<int>& vtx_idx, 
        const Vector<T, dim>& gradent)
    {
        if (vtx_idx.size() * 3 != dim)
            std::cout << "wrong gradient block size in addForceEntry" << std::endl;

        for (int i = 0; i < vtx_idx.size(); i++)
            residual.segment<3>(vtx_idx[i] * 3) += gradent.template segment<3>(i * 3);
    }

    template<int dim>
    void getSubVector(const VectorXT& _vector, 
        const std::vector<int>& vtx_idx, 
        Vector<T, dim>& sub_vec)
    {
        if (vtx_idx.size() * 3 != dim)
            std::cout << "wrong gradient block size in getSubVector" << std::endl;

        sub_vec = Vector<T, dim>::Zero(vtx_idx.size() * 3);
        for (int i = 0; i < vtx_idx.size(); i++)
        {
            sub_vec.template segment<3>(i * 3) = _vector.segment<3>(vtx_idx[i] * 3);
        }
    }

public:
    CellSim(/* args */) {}
    ~CellSim() {}

// SIMULATION
    bool advanceOneStep(int step);
    void buildSystemMatrix(const VectorXT& _u, StiffnessMatrix& K);
    void buildSystemMatrixWoodbury(const VectorXT& _u, StiffnessMatrix& K, MatrixXT& UV);
    T computeTotalEnergy(const VectorXT& _u, bool add_to_deform = true);
    T computeResidual(const VectorXT& _u,  VectorXT& residual);
    T lineSearchNewton(VectorXT& _u,  VectorXT& residual, int ls_max = 12);
    bool solveWoodburyCholmod(StiffnessMatrix& K, MatrixXT& UV,
         VectorXT& residual, VectorXT& du);
    void projectDirichletDoFMatrix(StiffnessMatrix& A, 
        const std::unordered_map<int, T>& data);
    T computeLineSearchInitStepsize(const VectorXT& _u, const VectorXT& du);

// Potentials

    // Yolk.cpp
    T computeYolkVolume();
    void addYolkVolumePreservationEnergy(T& energy);
    void addYolkVolumePreservationForceEntries(VectorXT& residual);
    void addYolkVolumePreservationHessianEntries(std::vector<Entry>& entries,
        MatrixXT& WoodBuryMatrix, bool projectPD = false);

    // CellVolume.cpp
    void computeVolumeAllCells(VectorXT& cell_volume_list);    
    void addCellVolumePreservationEnergy(T& energy);
    void addCellVolumePreservationForceEntries(VectorXT& residual);
    void addCellVolumePreservationHessianEntries(std::vector<Entry>& entries, 
        bool projectPD = false);

    // AreaTerms.cpp
    void addFaceAreaEnergy(Region region, T w, T& energy);
    void addFaceAreaForceEntries(Region region, T w, VectorXT& residual);
    void addFaceAreaHessianEntries(Region region, T w, 
        std::vector<Entry>& entries, bool projectPD = false);

    //EdgeTerms.cpp
    void computeRestLength();
    void addEdgeEnergy(Region region, T w, T& energy);
    void addEdgeForceEntries(Region region, T w, VectorXT& residual);
    void addEdgeHessianEntries(Region region, T w, 
        std::vector<Entry>& entries, bool projectPD = false);

    // IPC.cpp
    void saveIPCData(const std::string& folder, int iter, bool save_edges = false);
    void updateBarrierInfo(bool first_step);
    T computeCollisionFreeStepsize(const VectorXT& _u, const VectorXT& du);
    void computeIPCRestData();
    void updateIPCVertices(const VectorXT& _u);
    void addIPCEnergy(T& energy);
    void addIPCForceEntries(VectorXT& residual);
    void addIPCHessianEntries(std::vector<Entry>& entries,
        bool projectPD = false);

    // DerivativeTest.cpp
    void checkTotalGradient(bool perturb = true);
    void checkTotalHessian(bool perturb = true);
    void checkTotalGradientScale(bool perturb = true);
    void checkTotalHessianScale(bool perturb = true);

// Initialization
    void initializeCells();
    void constructCellMeshFromFile(const std::string& filename);

    void generateMeshForRendering(
        Eigen::MatrixXd& V, Eigen::MatrixXi& F, 
        Eigen::MatrixXd& C, bool yolk_only = false, bool cells_only = false);

    void computeBoundingBox(TV& min_corner, TV& max_corner)
    {
        min_corner.setConstant(1e6);
        max_corner.setConstant(-1e6);
        
        for (int i = 0; i < num_nodes; i++)
        {
            for (int d = 0; d < 3; d++)
            {
                max_corner[d] = std::max(max_corner[d], deformed[i * 3 + d]);
                min_corner[d] = std::min(min_corner[d], deformed[i * 3 + d]);
            }
        }
    }

    void positionsFromIndices(VectorXT& positions, const VtxList& indices, bool rest_state = false);
    void computeCentroid(const VectorXT& positions, TV& centroid);
};

#endif
