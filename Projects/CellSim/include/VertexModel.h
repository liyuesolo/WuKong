#ifndef VERTEXMODEL_H
#define VERTEXMODEL_H

#include <utility>
#include <iostream>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <tbb/tbb.h>

#include "VecMatDef.h"

class VertexModel
{
public:
    using TV = Vector<double, 3>;
    using TV2 = Vector<double, 2>;
    using TM2 = Matrix<double, 2, 2>;
    using TM3 = Matrix<double, 3, 3>;
    using IV = Vector<int, 3>;

    using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
    using VectorXi = Vector<int, Eigen::Dynamic>;

    using VtxList = std::vector<int>;
    using FaceList = std::vector<int>;

    using Edge = Vector<int, 2>;
    using StiffnessMatrix = Eigen::SparseMatrix<T>;
    using Entry = Eigen::Triplet<T>;
    
public:
    // iterators
    template <class OP>
    void iterateDirichletDoF(const OP& f) {
        for (auto dirichlet: dirichlet_data){
            f(dirichlet.first, dirichlet.second);
        } 
    }

    template <typename OP>
    void iterateFaceSerial(const OP& f)
    {
        int cnt = 0;
        for (VtxList& cell_face : faces)
        {
            f(cell_face, cnt);
            cnt++;
        }
    }

    template <typename OP>
    void iterateFaceParallel(const OP& f)
    {
        tbb::parallel_for(0, (int)faces.size(), [&](int i){
            f(faces[i], i);
        });
    }

    template <typename OP>
    void iterateEdgeSerial(const OP& f)
    {
        for (Edge& e : edges)
        {
            f(e);
        }   
    }

    template <typename OP>
    void iterateApicalEdgeSerial(const OP& f)
    {
        for (Edge& e : edges)
        {
            if (e[0] < basal_vtx_start && e[1] < basal_vtx_start)
                f(e);
        }   
    }

    template <typename OP>
    void iterateEdgeParallel(const OP& f)
    {
        tbb::parallel_for(0, (int)edges.size(), [&](int i){
            f(edges[i]);
        });
    }

public:
    // deformed and undeformed location of all vertices, u are the displacements
    VectorXT undeformed, deformed, u;

    std::vector<VtxList> faces; // all faces
    std::vector<FaceList> cell_faces; // face id list for each cell
    VectorXT cell_volume_init;
    std::vector<Edge> edges; // all edges
    
    
    int num_nodes;
    int basal_vtx_start;
    int basal_face_start;
    int lateral_face_start;

    T sigma = 1.0;
    T alpha = 2.13;
    T gamma = 0.98;
    T B = 100.0;
    T By = 100.0;

    bool run_diff_test = false;
    bool add_yolk_volume = true;
    bool use_cell_centroid = false;

    TV mesh_centroid;
    T yolk_vol_init;

    std::unordered_map<int, T> dirichlet_data;

    void splitCellsForRendering(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& C);

    void saveIndividualCellsWithOffset();

    void computeLinearModes();

    void computeCubeVolumeFromTet(const Vector<T, 24>& prism_vertices, T& volume);
    void computeCubeVolumeCentroid(const Vector<T, 24>& prism_vertices, T& volume);

    void computeCellCentroid(const VtxList& face_vtx_list, TV& centroid);
    void computeFaceCentroid(const VtxList& face_vtx_list, TV& centroid);

    T computeYolkVolume(bool verbose = false);
    void computeVolumeAllCells(VectorXT& cell_volume_list);
    void vertexModelFromMesh(const std::string& filename);
    void addTestPrism(int edge);

    void addTestPrismGrid(int n_row, int n_col);

    void generateMeshForRendering(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& C);

    void projectDirichletDoFMatrix(StiffnessMatrix& A, 
        const std::unordered_map<int, T>& data);

    void buildSystemMatrix(const VectorXT& _u, StiffnessMatrix& K);
    T computeTotalEnergy(const VectorXT& _u, bool verbose = false);
    T computeResidual(const VectorXT& _u,  VectorXT& residual, bool verbose = false);

    void faceHessianChainRuleTest();

    void checkTotalGradient();
    void checkTotalHessian();

    void checkTotalHessianScale();
    void checkTotalGradientScale();
    
    void positionsFromIndices(VectorXT& positions, const VtxList& indices);

    bool linearSolve(StiffnessMatrix& K, VectorXT& residual, VectorXT& du);
    
private:
    template<int dim>
    void addHessianEntry(
        std::vector<Entry>& triplets,
        const std::vector<int>& vtx_idx, 
        const Matrix<T, dim, dim>& hessian)
    {
        if (vtx_idx.size() * 3 != dim)
            std::cout << "wrong hessian block size" << std::endl;

        for (int i = 0; i < vtx_idx.size(); i++)
        {
            int dof_i = vtx_idx[i];
            for (int j = 0; j < vtx_idx.size(); j++)
            {
                int dof_j = vtx_idx[j];
                for (int k = 0; k < 3; k++)
                    for (int l = 0; l < 3; l++)
                        triplets.push_back(Entry(dof_i * 3 + k, dof_j * 3 + l, hessian(i * 3 + k, j * 3 + l)));                
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
                triplets.push_back(Entry(dof_i * 3 + k, dof_j * 3 + l, hessian_block(k, l)));
            }
    }

    template<int dim>
    void addForceEntry(VectorXT& residual, 
        const std::vector<int>& vtx_idx, 
        const Vector<T, dim>& gradent)
    {
        if (vtx_idx.size() * 3 != dim)
            std::cout << "wrong gradient block size" << std::endl;

        for (int i = 0; i < vtx_idx.size(); i++)
            residual.segment<3>(vtx_idx[i] * 3) += gradent.template segment<3>(i * 3);
    }

    template<int dim>
    void getSubVector(const VectorXT& _vector, 
        const std::vector<int>& vtx_idx, 
        Vector<T, dim>& sub_vec)
    {
        if (vtx_idx.size() * 3 != dim)
            std::cout << "wrong gradient block size" << std::endl;

        sub_vec = Vector<T, dim>::Zero(vtx_idx.size() * 3);
        for (int i = 0; i < vtx_idx.size(); i++)
        {
            sub_vec.template segment<3>(i * 3) = _vector.segment<3>(vtx_idx[i] * 3);
        }
    }

public:
    VertexModel() 
    {
        
    }
    ~VertexModel() {}
};

#endif