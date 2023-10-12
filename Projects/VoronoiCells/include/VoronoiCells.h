#ifndef VORONOI_CELLS_H
#define VORONOI_CELLS_H
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
#include <queue>
#include <unordered_map>
#include <unordered_set>

#include "VecMatDef.h"
#include "Util.h"

#include "geometrycentral/surface/edge_length_geometry.h"
#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/mesh_graph_algorithms.h"
#include "geometrycentral/surface/surface_mesh_factories.h"
#include "geometrycentral/surface/polygon_soup_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"
#include "geometrycentral/surface/poisson_disk_sampler.h"

namespace gcs = geometrycentral::surface;
namespace gc = geometrycentral;


enum DistanceMetric
{
    Geodesic, Euclidean
};

class VoronoiCells
{
public:
    using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
    using MatrixXT = Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorXi = Vector<int, Eigen::Dynamic>;
    using MatrixXi = Matrix<int, Eigen::Dynamic, Eigen::Dynamic>;
    using VtxList = std::vector<int>;
    using TV = Vector<T, 3>;
    using TV2 = Vector<T, 2>;
    using IV = Vector<int, 3>;
    using IV2 = Vector<int, 2>;
    using TM = Matrix<T, 3, 3>;
    using TM2 = Matrix<T, 2, 2>;
    using StiffnessMatrix = Eigen::SparseMatrix<T>;
    using Entry = Eigen::Triplet<T>;
    using Face = Vector<int, 3>;
    using Edge = Vector<int, 2>;

    using gcEdge = geometrycentral::surface::Edge;
    using gcFace = geometrycentral::surface::Face;
    using gcVertex = geometrycentral::surface::Vertex;
    using Vector3 = geometrycentral::Vector3;
    using SurfacePoint = geometrycentral::surface::SurfacePoint;

    struct IxnData
    {
        TV start, end;
        T t;
        int start_vtx_idx, end_vtx_idx;
        IxnData(const TV& _start, const TV& _end, T _t, int idx0, int idx1) 
            : start(_start), end(_end), t(_t), start_vtx_idx(idx0),
            end_vtx_idx(idx1) {};
        IxnData(const TV& _start, const TV& _end, T _t) 
            : start(_start), end(_end), t(_t), start_vtx_idx(-1),
            end_vtx_idx(-1) {};
    };

    struct FaceData
    {
        std::vector<int> site_indices;
        std::vector<TV> distances;
        FaceData (const std::vector<int>& _site_indices, 
            const std::vector<TV>& _distances) : 
            site_indices(_site_indices), distances(_distances) {}
        FaceData () {}

    };

    
    VectorXT extrinsic_vertices;
    VectorXi extrinsic_indices;

    bool use_debug_face_color = false;
    MatrixXT face_color;

    std::unique_ptr<gcs::ManifoldSurfaceMesh> mesh;
    std::unique_ptr<gcs::VertexPositionGeometry> geometry;

    MatrixXT surface_vtx;
    MatrixXi surface_indices;

    int n_sites;
    VectorXT voronoi_sites;
    VectorXT voronoi_cell_vertices;

    std::vector<SurfacePoint> samples;
    std::vector<FaceData> source_data;

    std::vector<VtxList> voronoi_cells;
    std::vector<std::pair<TV, TV>> voronoi_edges;

    DistanceMetric metric = Euclidean;    

private:
    // void triangulatePointCloud(const VectorXT& points, VectorXi& triangle_indices);
    
    TV toTV(const Vector3& vec) const
    {
        return TV(vec.x, vec.y, vec.z);
    }
    void loadGeometry();
    void updateFaceColor();
    
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
    T computeDistanceMatchingEnergy(const std::vector<int>& site_indices, 
        SurfacePoint& xi_current);
    T computeDistanceMatchingGradient(const std::vector<int>& site_indices, 
        SurfacePoint& xi_current, TV2& grad, T& energy);
    void computeDistanceMatchingHessian(const std::vector<int>& site_indices, 
        SurfacePoint& xi_current, TM2& hess);
    T computeDistanceMatchingEnergyGradientHessian(const std::vector<int>& site_indices, 
        SurfacePoint& xi_current, TM2& hess, TV2& grad, T& energy);
    void updateSurfacePoint(SurfacePoint& xi_current, const TV2& search_direction);

    void optimizeForExactVD(std::vector<std::pair<SurfacePoint, std::vector<int>>>& ixn_data);
    void intersectPrisms(std::vector<SurfacePoint>& samples,
            std::vector<FaceData>& source_data, 
            std::vector<std::pair<SurfacePoint, std::vector<int>>>& ixn_data);
    void intersectPrism(std::vector<SurfacePoint>& samples,
            std::vector<FaceData>& source_data, 
            std::vector<std::pair<TV, TV>>& edges, int face_idx);
    void propogateDistanceField(std::vector<SurfacePoint>& samples,
        std::vector<FaceData>& source_data);
    void constructVoronoiDiagram(bool exact = false, bool load_from_file = false);
    void saveVoronoiDiagram();
    void saveFacePrism(int face_idx);
    void generateMeshForRendering(MatrixXT& V, MatrixXi& F, MatrixXT& C);

    void computeGeodesicDistance(const SurfacePoint& a, const SurfacePoint& b,
        T& dis, std::vector<SurfacePoint>& path, 
        std::vector<IxnData>& ixn_data, bool trace_path = false);
    
    
    VoronoiCells() {}
    ~VoronoiCells() {}
};


#endif 