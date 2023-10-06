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

// #include <CGAL/Surface_mesh.h>
// #include <CGAL/convex_hull_2.h>
// #include <CGAL/convex_hull_3.h>
// #include <CGAL/boost/graph/convert_nef_polyhedron_to_polygon_mesh.h>
// #include <CGAL/Exact_integer.h>
// #include <CGAL/Extended_homogeneous.h>
// #include <CGAL/Exact_predicates_exact_constructions_kernel.h>
// #include <CGAL/Polyhedron_3.h>
// #include <CGAL/Nef_polyhedron_3.h>
// #include <CGAL/IO/Nef_polyhedron_iostream_3.h>
// #include <CGAL/Aff_transformation_3.h>
// typedef CGAL::Exact_predicates_exact_constructions_kernel inexact_Kernel;
// typedef CGAL::Polyhedron_3<inexact_Kernel>  Polyhedron;
// typedef CGAL::Polygon_2<inexact_Kernel> Polygon_2;
// typedef CGAL::Nef_polyhedron_3<inexact_Kernel>  Nef_polyhedron;
// typedef inexact_Kernel::Point_3 Point_3;
// typedef inexact_Kernel::Point_2 Point_2;
// typedef inexact_Kernel::Vector_3 Vector_3;
// typedef inexact_Kernel::Plane_3 Plane_3;
// typedef CGAL::Aff_transformation_3<inexact_Kernel> Aff_transformation_3;

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
    using TM = Matrix<T, 3, 3>;
    using StiffnessMatrix = Eigen::SparseMatrix<T>;
    using Entry = Eigen::Triplet<T>;
    using Face = Vector<int, 3>;
    using Edge = Vector<int, 2>;

    using gcEdge = geometrycentral::surface::Edge;
    using gcFace = geometrycentral::surface::Face;
    using gcVertex = geometrycentral::surface::Vertex;
    using Vector3 = geometrycentral::Vector3;
    using SurfacePoint = geometrycentral::surface::SurfacePoint;

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
public:
    void intersectPrisms(std::vector<SurfacePoint>& samples,
            std::vector<FaceData>& source_data, 
            std::vector<std::pair<TV, TV>>& edges);
    void intersectPrism(std::vector<SurfacePoint>& samples,
            std::vector<FaceData>& source_data, 
            std::vector<std::pair<TV, TV>>& edges, int face_idx);
    void propogateDistanceField(std::vector<SurfacePoint>& samples,
        std::vector<FaceData>& source_data);
    void constructVoronoiDiagram(bool exact = false, bool load_from_file = false);
    void saveVoronoiDiagram();
    void saveFacePrism(int face_idx);
    void generateMeshForRendering(MatrixXT& V, MatrixXi& F, MatrixXT& C);

    T computeGeodesicDistance(const SurfacePoint& a, const SurfacePoint& b);

    
    VoronoiCells() {}
    ~VoronoiCells() {}
};


#endif 