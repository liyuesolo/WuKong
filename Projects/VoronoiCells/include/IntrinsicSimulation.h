#ifndef INTRINSIC_SIMULATION_H
#define INTRINSIC_SIMULATION_H

#include <utility>
#include <iostream>
#include <fstream>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <tbb/tbb.h>
#include <unordered_map>
#include <complex>
#include <iomanip>

#include "geometrycentral/surface/edge_length_geometry.h"
#include "geometrycentral/surface/flip_geodesics.h"
#include "geometrycentral/surface/trace_geodesic.h"
#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/mesh_graph_algorithms.h"
#include "geometrycentral/surface/surface_mesh_factories.h"
#include "geometrycentral/surface/polygon_soup_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"
#include "geometrycentral/surface/exact_geodesics.h"

#include "VecMatDef.h"
#include "Util.h"
#include "Timer.h"

namespace gcs = geometrycentral::surface;
namespace gc = geometrycentral;

#define PARALLEL_GEODESIC

class IntrinsicSimulation
{
public:
    using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
    using MatrixXT = Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorXi = Vector<int, Eigen::Dynamic>;
    using MatrixXi = Matrix<int, Eigen::Dynamic, Eigen::Dynamic>;
    using VtxList = std::vector<int>;
    using TV = Vector<T, 3>;
    using TV2 = Vector<T, 2>;
    using TV3 = Vector<T, 3>;
    using IV = Vector<int, 3>;
    using TM = Matrix<T, 3, 3>;
    using StiffnessMatrix = Eigen::SparseMatrix<T>;
    using Entry = Eigen::Triplet<T>;
    using Face = Vector<int, 3>;
    using Edge = Vector<int, 2>;
    using FacePoint = std::pair<int, TV>;

    
    using gcEdge = geometrycentral::surface::Edge;
    using gcFace = geometrycentral::surface::Face;
    using gcVertex = geometrycentral::surface::Vertex;
    using Vector3 = geometrycentral::Vector3;
    using SurfacePoint = geometrycentral::surface::SurfacePoint;

    struct IxnData
    {
        TV start, end;
        T t;
        IxnData(const TV& _start, const TV& _end, T _t) 
            : start(_start), end(_end), t(_t) {};
    };
    

public:
    VectorXT extrinsic_vertices;
    VectorXi extrinsic_indices;

    VectorXT intrinsic_vertices_barycentric_coords;

    VectorXT deformed, undeformed;
    VectorXT delta_u;
    VectorXT u;
    std::unordered_map<int, T> dirichlet_data;
    bool run_diff_test = false;
    int max_newton_iter = 500;
    bool use_Newton = true;
    T newton_tol = 1e-6;

    std::vector<std::pair<SurfacePoint, gcFace>> mass_surface_points;
    std::vector<std::pair<SurfacePoint, gcFace>> mass_surface_points_undeformed;
    std::vector<Edge> spring_edges;
    std::vector<T> rest_length;
    bool verbose = false;
    bool use_intrinsic = false;
    T we = 1.0;
    T ref_dis = 1.0;
    bool retrace = true;

    std::unique_ptr<gcs::ManifoldSurfaceMesh> mesh;
    std::unique_ptr<gcs::VertexPositionGeometry> geometry;

    std::vector<std::pair<TV, TV>> all_intrinsic_edges;

    std::vector<T> current_length;
    std::vector<std::vector<SurfacePoint>> paths;
	std::vector<std::vector<IxnData>> ixn_data_list;
private:
    template <class OP>
    void iterateDirichletDoF(const OP& f) {
        for (auto dirichlet: dirichlet_data){
            f(dirichlet.first, dirichlet.second);
        } 
    }

    bool simDoFToPosition(VectorXT& sim_dof);

    Vector3 toVec3(const TV& vec) const
    {
        return Vector3{vec[0], vec[1], vec[2]};
    }

    TV toTV(const Vector3& vec) const
    {
        return TV(vec.x, vec.y, vec.z);
    }

    void printVec3(const Vector3& vec) const
    {
        std::cout << vec.x << " " << vec.y << " " << vec.z << std::endl;
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

    template<int size>
    VectorXT computeHessianBlockEigenValues(const Matrix<T, size, size> & symMtr)
    {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, size, size>> eigenSolver(symMtr);
        return eigenSolver.eigenvalues();
    }

    void edgeLengthHessian(const TV& v0, const TV& v1, Matrix<T, 6, 6>& hess)
    {
        TV dir = (v1-v0).normalized();
        hess.setZero();
        hess.block(0, 0, 3, 3) = (TM::Identity() - dir * dir.transpose())/(v1 - v0).norm();
        hess.block(3, 3, 3, 3) = hess.block(0, 0, 3, 3);
        hess.block(0, 3, 3, 3) = -hess.block(0, 0, 3, 3);
        hess.block(3, 0, 3, 3) = -hess.block(0, 0, 3, 3);
    }

public:
    void computeExactGeodesicEdgeFlip(const SurfacePoint& va, const SurfacePoint& vb, 
        T& dis, std::vector<SurfacePoint>& path, 
        std::vector<IxnData>& ixn_data, 
        bool trace_path = false);
    void computeExactGeodesic(const SurfacePoint& va, const SurfacePoint& vb, 
        T& dis, std::vector<SurfacePoint>& path, 
        std::vector<IxnData>& ixn_data, 
        bool trace_path = false);

    
    void initializeMassSpringSceneExactGeodesic();
    void generateMeshForRendering(MatrixXT& V, MatrixXi& F, MatrixXT& C);
    
    T computeTotalEnergy();
    T computeResidual(VectorXT& residual);
    void buildSystemMatrix(StiffnessMatrix& K);
    T lineSearchNewton(VectorXT& residual);
    void updateCurrentState();
    void traceGeodesics();
    void reset();
    // bool staticSolve();

    // bool staticSolveStep(int step);

    bool linearSolve(StiffnessMatrix& K, VectorXT& residual, VectorXT& du);

    void projectDirichletDoFMatrix(StiffnessMatrix& A, const std::unordered_map<int, T>& data);

    bool advanceOneStep(int step);

    void updateVisualization(bool all_edges = false);

    void addEdgeLengthEnergy(T w, T& energy);
    void addEdgeLengthForceEntries(T w, VectorXT& residual);
    void addEdgeLengthHessianEntries(T w, std::vector<Entry>& entries);

    void checkHessian();

    // DerivativeTest.cpp    
    void checkTotalGradientScale(bool perturb = false);
    void checkTotalGradient(bool perturb = false);
    void checkTotalHessian(bool perturb = false);
    void checkTotalHessianScale(bool perturb = false);

    void massPointPosition(int idx, TV& pos);
    void moveMassPoint(int idx, int bc);
    void getAllPointsPosition(VectorXT& positions);
public:
    IntrinsicSimulation() {}
    ~IntrinsicSimulation() {}
};

#endif