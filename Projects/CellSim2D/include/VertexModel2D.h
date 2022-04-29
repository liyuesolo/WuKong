#ifndef VERTEX_MODEL_2D_H
#define VERTEX_MODEL_2D_H

#include <utility>
#include <iostream>
#include <fstream>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <tbb/tbb.h>

#include "VecMatDef.h"

enum Region
{
    Apical, Basal, Lateral, ALL
};

class VertexModel2D
{
public:
    using TV = Vector<double, 2>;
    using TV3 = Vector<double, 3>;
    using TM = Matrix<double, 2, 2>;
    using IV = Vector<int, 2>;
    using IV3 = Vector<int, 3>;

    using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
    using MatrixXT = Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorXi = Vector<int, Eigen::Dynamic>;

    using VtxList = std::vector<int>;
    using FaceList = std::vector<int>;

    using Edge = Vector<int, 2>;
    // using StiffnessMatrix = Eigen::SparseMatrix<T>;
    typedef int StorageIndex;
    using StiffnessMatrix = Eigen::SparseMatrix<T, Eigen::ColMajor, StorageIndex>;
    using Entry = Eigen::Triplet<T>;

public:
    VectorXT undeformed, deformed, u;
    std::vector<Edge> edges;
    
    VectorXT apical_edge_contracting_weights;
    int basal_vtx_start = 0;
    int basal_edge_start = 0;
    int lateral_edge_start = 0;
    int num_nodes = 0;

    T w_ea = 1.0;
    T w_eb = 1.0;
    T w_el = 1.0;
    T radius = 1.0;
    T w_a = 1e4;
    T w_mb = 1e4;
    T w_yolk = 1e4;
    T w_c = 5.0;
    
    T yolk_area_rest = 0;
    int n_cells = 0;
    T barrier_weight = 1e-22;

    TV mesh_centroid = TV::Zero();

    T newton_tol = 1e-6;
    int max_newton_iter = 2000;
    bool save_mesh = false;

    bool verbose = false;
    bool run_diff_test = false;

    std::unordered_map<int, T> dirichlet_data;

    std::string data_folder = "/home/yueli/Documents/ETH/WuKong/output/cells2d/";

    bool use_ipc = true;
    Eigen::MatrixXd ipc_vertices;
    Eigen::MatrixXi ipc_edges, ipc_faces;
    T ipc_barrier_distance = 1e-3;
    T ipc_barrier_weight = 1e4;

private:
    bool validEdgeIdx(Region region, int idx)
    {
        if (region == Apical)
            return idx < basal_edge_start;
        else if (region == Basal)
            return idx >= basal_edge_start && idx < lateral_edge_start;
        else if (region == Lateral)
            return idx >= lateral_edge_start;
        else
            return false;
    }

public:
    template <class OP>
    void iterateDirichletDoF(const OP& f) {
        for (auto dirichlet: dirichlet_data){
            f(dirichlet.first, dirichlet.second);
        } 
    }

    template <typename OP>
    void iterateApicalEdgeSerial(const OP& f)
    {
        for (int i = 0; i < basal_edge_start; i++)
        {
            f(edges[i], i);
        }
    }

    template <typename OP>
    void iterateBasalEdgeSerial(const OP& f)
    {
        for (int i = basal_edge_start; i < lateral_edge_start; i++)
        {
            f(edges[i], i);
        }
    }

    template <typename OP>
    void iterateLateralEdgeSerial(const OP& f)
    {
        for (int i = lateral_edge_start; i < int(edges.size()); i++)
        {
            f(edges[i], i);
        }
    }

    template <typename OP>
    void iterateCellSerial(const OP& f)
    {
        for (int i = 0; i < basal_edge_start; i++)
        {
            VtxList edge_vtx_list;
            edge_vtx_list.push_back(edges[i][0]);
            edge_vtx_list.push_back(edges[i][1]);
            edge_vtx_list.push_back(edges[i][1] + basal_vtx_start);
            edge_vtx_list.push_back(edges[i][0] + basal_vtx_start);
            f(edge_vtx_list, i);
        }
    }

    template <typename OP>
    void iterateCellParallel(const OP& f)
    {
        tbb::parallel_for(0, basal_edge_start, [&](int i)
        {
            VtxList edge_vtx_list;
            edge_vtx_list.push_back(edges[i][0]);
            edge_vtx_list.push_back(edges[i][1]);
            edge_vtx_list.push_back(edges[i][1] + basal_vtx_start);
            edge_vtx_list.push_back(edges[i][0] + basal_vtx_start);
            f(edge_vtx_list, i);
        });
    }

    template<int dim>
    void addHessianEntry(
        std::vector<Entry>& triplets,
        const std::vector<int>& vtx_idx, 
        const Matrix<T, dim, dim>& hessian)
    {
        if (vtx_idx.size() * 2 != dim)
            std::cout << "wrong hessian block size" << std::endl;

        for (int i = 0; i < vtx_idx.size(); i++)
        {
            int dof_i = vtx_idx[i];
            for (int j = 0; j < vtx_idx.size(); j++)
            {
                int dof_j = vtx_idx[j];
                for (int k = 0; k < 2; k++)
                    for (int l = 0; l < 2; l++)
                    {
                        if (std::abs(hessian(i * 2 + k, j * 2 + l)) > 1e-8)
                            triplets.push_back(Entry(dof_i * 2 + k, dof_j * 2 + l, hessian(i * 2 + k, j * 2 + l)));                
                    }
            }
        }
    }
    

    template<int dim>
    void addHessianBlock(
        std::vector<Entry>& triplets,
        const std::vector<int>& vtx_idx, 
        const Matrix<T, dim, dim>& hessian_block)
    {
        int dof_i = vtx_idx[0];
        int dof_j = vtx_idx[1];

        for (int k = 0; k < dim; k++)
            for (int l = 0; l < dim; l++)
            {
                if (std::abs(hessian_block(k, l)) > 1e-8)
                    triplets.push_back(Entry(dof_i * 2 + k, dof_j * 2 + l, hessian_block(k, l)));
            }
    }

    template<int dim>
    void addForceEntry(VectorXT& residual, 
        const std::vector<int>& vtx_idx, 
        const Vector<T, dim>& gradent)
    {
        if (vtx_idx.size() * 2 != dim)
            std::cout << "wrong gradient block size in addForceEntry" << std::endl;

        for (int i = 0; i < vtx_idx.size(); i++)
            residual.segment<2>(vtx_idx[i] * 2) += gradent.template segment<2>(i * 2);
    }

    template<int dim>
    void getSubVector(const VectorXT& _vector, 
        const std::vector<int>& vtx_idx, 
        Vector<T, dim>& sub_vec)
    {
        if (vtx_idx.size() * 2 != dim)
            std::cout << "wrong gradient block size in getSubVector" << std::endl;

        sub_vec = Vector<T, dim>::Zero(vtx_idx.size() * 2);
        for (int i = 0; i < vtx_idx.size(); i++)
        {
            sub_vec.template segment<2>(i * 2) = _vector.segment<2>(vtx_idx[i] * 2);
        }
    }

    std::vector<Entry> entriesFromSparseMatrix(const StiffnessMatrix& A)
    {
        std::vector<Entry> triplets;

        for (int k=0; k < A.outerSize(); ++k)
            for (StiffnessMatrix::InnerIterator it(A,k); it; ++it)
                triplets.push_back(Entry(it.row(), it.col(), it.value()));
        return triplets;
    }

public:
    void saveCellCentroidsToFile(const std::string& filename);
    void saveStates(const std::string& filename);
    void computeAllCellCentroids(VectorXT& cell_centroids);
    void initializeScene();

    void appendSphereToPositionVector(const VectorXT& position, T radius, const TV3& color,
        Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& C);

    void generateMeshForRendering(Eigen::MatrixXd& V, Eigen::MatrixXi& F, 
        Eigen::MatrixXd& C, bool show_current, bool show_rest);
    
    void appendCylindersToEdges(const std::vector<std::pair<TV, TV>>& edge_pairs, 
        const TV3& color, T radius,
        Eigen::MatrixXd& _V, Eigen::MatrixXi& _F, Eigen::MatrixXd& _C);

    void buildSystemMatrix(const VectorXT& _u, StiffnessMatrix& K);
    T computeTotalEnergy(const VectorXT& _u);
    T computeResidual(const VectorXT& _u,  VectorXT& residual);
    T lineSearchNewton(VectorXT& _u,  VectorXT& residual, int ls_max = 15);
    void projectDirichletDoFMatrix(StiffnessMatrix& A, 
        const std::unordered_map<int, T>& data);
    bool advanceOneStep(int step);
    bool linearSolve(StiffnessMatrix& K, VectorXT& residual, VectorXT& du);
    T computeLineSearchInitStepsize(const VectorXT& _u, const VectorXT& du);

    void addEdgeEnergy(Region region, T w, T& energy);
    void addEdgeForceEntries(Region region, T w, VectorXT& residual);
    void addEdgeHessianEntries(Region region, T w, std::vector<Entry>& entries);

    T computeCellArea(int cell_idx, bool rest = false);
    void addAreaPreservationEnergy(T w, T& energy);
    void addAreaPreservationForceEntries(T w, VectorXT& residual);
    void addAreaPreservationHessianEntries(T w, std::vector<Entry>& entries);

    void addMembraneBoundTerm(T w, T& energy);
    void addMembraneBoundForceEntries(T w, VectorXT& residual);
    void addMembraneBoundHessianEntries(T w, std::vector<Entry>& entries);

    T computeAreaInversionFreeStepsize(const VectorXT& _u, const VectorXT& du);
    void addAreaBarrierEnergy(T w, T& energy);
    void addAreaBarrierForceEntries(T w, VectorXT& residual);
    void addAreaBarrierHessianEntries(T w, std::vector<Entry>& entries);

    T computeYolkArea();
    void addYolkPreservationEnergy(T w, T& energy);
    void addYolkPreservationForceEntries(T w, VectorXT& residual);
    void addYolkPreservationHessianEntries(T w, std::vector<Entry>& entries);

    void configContractingWeights();
    void addContractingEnergy(T& energy);
    void addContractingForceEntries(VectorXT& residual);
    void addContractingHessianEntries(std::vector<Entry>& entries);

    void buildIPCRestData();
    void addIPCEnergy(T& energy);
    void addIPCForceEntries(VectorXT& residual);
    void addIPCHessianEntries(std::vector<Entry>& entries);
    void updateIPCVertices(const VectorXT& _u);
    T computeCollisionFreeStepsize(const VectorXT& _u, const VectorXT& du);

    void checkTotalGradient(bool perturb = false);
    void checkTotalHessian(bool perturb = false);

    void checkTotalHessianScale(bool perturb = false);
    void checkTotalGradientScale(bool perturb = false);

    void positionsFromIndices(VectorXT& positions, const VtxList& indices, bool rest_state = false);
    void computeCellCentroid(int cell_idx, TV& centroid, bool rest = false);

    void reset();
    bool staticSolve();

    void getCellVtxIndices(VtxList& indices, int cell_idx);
    void loadEdgeWeights(const std::string& filename, VectorXT& weights);

    // For sensitvitity analysis
    void dOdpEdgeWeightsFromLambda(const VectorXT& lambda, VectorXT& dOdp);
    void computededp(VectorXT& dedp);
    void dfdpWeightsSparse(StiffnessMatrix& dfdp);
    void dxdpFromdxdpEdgeWeights(MatrixXT& dxdp);
    void dfdpWeightsDense(MatrixXT& dfdp);

    bool fetchNegativeEigenVectorIfAny(T& negative_eigen_value, VectorXT& negative_eigen_vector);
    void checkHessianPD(bool save_txt = false);

    void checkFunctionPerIteration(int step);
    void checkFinalState();
    VertexModel2D() {}
    ~VertexModel2D() {}
};

#endif