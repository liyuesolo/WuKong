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

public:
    VectorXT extrinsic_vertices;
    VectorXi extrinsic_indices;

    VectorXT intrinsic_vertices_barycentric_coords;
    VectorXT intrinsic_vertices_undeformed;
    VectorXT intrinsic_vertices_deformed;

    VectorXT deformed, undeformed;
    VectorXT u;
    std::unordered_map<int, T> dirichlet_data;
    bool run_diff_test = false;
    int max_newton_iter = 500;
    T newton_tol = 1e-6;

    std::vector<std::pair<gcVertex, gcFace>> mass_vertices;
    std::vector<Edge> spring_edges;
    std::vector<T> rest_length;
    

    T we = 1.0;

    std::unique_ptr<geometrycentral::surface::ManifoldSurfaceMesh> mesh;
    std::unique_ptr<geometrycentral::surface::VertexPositionGeometry> geometry;

    std::unique_ptr<geometrycentral::surface::FlipEdgeNetwork> edgeNetwork; 

    std::vector<std::pair<TV, TV>> all_intrinsic_edges;

private:
    template <class OP>
    void iterateDirichletDoF(const OP& f) {
        for (auto dirichlet: dirichlet_data){
            f(dirichlet.first, dirichlet.second);
        } 
    }

    bool simDoFToPosition(const VectorXT& sim_dof);

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

    

public:
    void initializeMassPointScene();
    void generateMeshForRendering(MatrixXT& V, MatrixXi& F, MatrixXT& C);
    
    T computeTotalEnergy(const VectorXT& _u);
    T computeResidual(const VectorXT& _u, VectorXT& residual);

    T lineSearchNewton(VectorXT& _u,  VectorXT& residual);

    // bool staticSolve();

    // bool staticSolveStep(int step);

    // bool linearSolve(StiffnessMatrix& K, VectorXT& residual, VectorXT& du);

    // void projectDirichletDoFMatrix(StiffnessMatrix& A, const std::unordered_map<int, T>& data);

    bool advanceOneStep(int step);

    void updateVisualization(bool all_edges = false);

    void addEdgeLengthEnergy(T w, T& energy);
    void addEdgeLengthForceEntries(T w, VectorXT& residual);
    void addEdgeLengthHessianEntries(T w, std::vector<Entry>& entries);

    // DerivativeTest.cpp
    void checkLengthDerivatives();
    void checkLengthDerivativesScale();
    void checkTotalGradientScale(bool perturb = false);
    void checkTotalGradient(bool perturb = false);
    // void checkTotalHessianScale(bool perturb = false);

public:
    IntrinsicSimulation() {}
    ~IntrinsicSimulation() {}
};

#endif