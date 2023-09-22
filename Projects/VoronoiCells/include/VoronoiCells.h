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

class VoronoiCells
{
public:
    using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
    using MatrixXT = Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorXi = Vector<int, Eigen::Dynamic>;
    using MatrixXi = Matrix<int, Eigen::Dynamic, Eigen::Dynamic>;
    using VtxList = std::vector<int>;
    using TV = Vector<T, 3>;
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
    std::vector<VtxList> voronoi_cells;
    std::vector<std::pair<TV, TV>> voronoi_edges;

private:
    // void triangulatePointCloud(const VectorXT& points, VectorXi& triangle_indices);
    void constructCentroidalVD(const VectorXi& triangulation, 
                                const VectorXT& site_locations,
                                VectorXT& nodal_positions,
                                std::vector<VtxList>& cell_connectivity,
                                std::vector<std::pair<TV, TV>>& path_for_viz,
                                bool generate_path = true);

    void constructIntrinsicVoronoiDiagram(const VectorXi& triangulation, 
                                const VectorXT& site_locations,
                                const std::vector<std::pair<int, TV>>& barycentric_coords,
                                VectorXT& nodal_positions,
                                std::vector<VtxList>& cell_connectivity,
                                std::vector<std::pair<TV, TV>>& path_for_viz,
                                bool generate_path = true);
    
    TV toTV(const Vector3& vec) const
    {
        return TV(vec.x, vec.y, vec.z);
    }
public:
    void constructVoronoiDiagram();
    void generateMeshForRendering(MatrixXT& V, MatrixXi& F, MatrixXT& C);

    
    VoronoiCells() {}
    ~VoronoiCells() {}
};


#endif 