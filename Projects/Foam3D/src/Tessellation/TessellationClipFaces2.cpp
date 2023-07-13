#include <geogram/basic/common.h>
#include <geogram/basic/logger.h>
#include <geogram/basic/command_line.h>
#include <geogram/basic/command_line_args.h>
#include <geogram/basic/stopwatch.h>
#include <geogram/basic/file_system.h>
#include <geogram/basic/process.h>
#include <geogram/mesh/mesh.h>
#include <geogram/mesh/mesh_geometry.h>
#include <geogram/mesh/mesh_topology.h>
#include <geogram/mesh/mesh_io.h>
#include <geogram/mesh/mesh_repair.h>
#include <geogram/mesh/mesh_fill_holes.h>
#include <geogram/mesh/mesh_preprocessing.h>
#include <geogram/mesh/mesh_degree3_vertices.h>
#include <geogram/mesh/mesh_tetrahedralize.h>
#include <geogram/mesh/mesh_halfedges.h>
#include <geogram/delaunay/delaunay.h>
#include <geogram/delaunay/periodic_delaunay_3d.h>
#include <geogram/voronoi/RVD.h>
#include <geogram/voronoi/RVD_callback.h>
#include <geogram/voronoi/RVD_mesh_builder.h>
#include <geogram/voronoi/convex_cell.h>
#include <geogram/numerics/predicates.h>

#include "../../include/Tessellation/Tessellation.h"
#include "Projects/Foam3D/include/Energy/PerTriangleFunction.h"
#include <set>
#include <chrono>

void Tessellation::clipFaces2() {
    GEO::initialize();
    GEO::CmdLine::import_arg_group("standard");
    GEO::CmdLine::import_arg_group("algo");

    GEO::Mesh boundaryMesh;
    for (int i = 0; i < boundary->v.size(); i++) {
        boundaryMesh.vertices.create_vertex(boundary->v[i].pos.data());
    }
    for (int i = 0; i < boundary->f.size(); i++) {
        boundaryMesh.facets.create_triangle(boundary->f[i].vertices(0), boundary->f[i].vertices(1),
                                            boundary->f[i].vertices(2));
    }
    boundaryMesh.facets.connect();

    VectorXT vertices, params;
    separateVerticesParams(c, vertices, params);
    GEO::PeriodicDelaunay3d dual(false);
    dual.set_vertices(vertices.rows() / 3, vertices.data());
    dual.set_weights(params.data());
    dual.set_stores_cicl(true);
    dual.compute();

    GEO::mesh_tetrahedralize(boundaryMesh);
    GEO::RestrictedVoronoiDiagram_var rvd = GEO::RestrictedVoronoiDiagram::create(&dual, &boundaryMesh);
    rvd->set_volumetric(true);

    GEO::Mesh clippedMesh;
    GEO::BuildRVDMesh rvdMeshBuilder(clippedMesh);
    rvdMeshBuilder.set_simplify_internal_tet_facets(true);
    rvdMeshBuilder.set_simplify_voronoi_facets(true);
    rvdMeshBuilder.set_generate_ids(true);
    rvd->for_each_polyhedron(rvdMeshBuilder);

    GEO::vector<std::string> names;
    clippedMesh.facets.attributes().list_attribute_names(names);
    for (auto s: names) {
        std::cout << "Facet " << s << std::endl;
    }
    clippedMesh.vertices.attributes().list_attribute_names(names);
    for (auto s: names) {
        std::cout << "Vertex " << s << std::endl;
    }
    clippedMesh.cells.attributes().list_attribute_names(names);
    for (auto s: names) {
        std::cout << "Cell " << s << std::endl;
    }

//    GEO::Attribute<int> cell_id(clippedMesh.facets.attributes(), "cell_id");
//    GEO::Attribute<int> facet_seed_id(clippedMesh.facets.attributes(), "facet_seed_id");
//    GEO::Attribute<int> seed_id(clippedMesh.facets.attributes(), "seed_id");
//    for (GEO::index_t v: clippedMesh.facets) {
//        std::cout << "Facet " << v << " " << cell_id[v] << " " << facet_seed_id[v] << " " << seed_id[v] << std::endl;
//    }
    GEO::Attribute<int> vertex_id(clippedMesh.vertices.attributes(), "vertex_id");
    GEO::Attribute<int> vertex_gen(clippedMesh.vertices.attributes(), "vertex_gen");
    GEO::Attribute<double> point(clippedMesh.vertices.attributes(), "point");
    for (GEO::index_t v: clippedMesh.vertices) {
        std::cout << "Vertex " << v << " " << vertex_id[v] << " " << point[v * 3 + 0] << " " << point[v * 3 + 1] << " "
                  << point[v * 3 + 2] << std::endl;
        std::cout << "   gen " << vertex_gen[v * 6 + 0] << " " << vertex_gen[v * 6 + 1] << " " << vertex_gen[v * 6 + 2]
                  << " " << vertex_gen[v * 6 + 3] << " " << vertex_gen[v * 6 + 4] << " " << vertex_gen[v * 6 + 5]
                  << std::endl;
    }

//    GEO::Mesh clippedMesh;
//    rvd->compute_RVD(clippedMesh, 0, true, false);
//    GEO::Attribute<GEO::index_t> region(clippedMesh.facets.attributes(), "region");
//    for (GEO::index_t v: clippedMesh.facets) {
//        std::cout << "Facet " << v << " belongs to cell " << region[v] << std::endl;
//    }
//    GEO::Attribute<GEO::index_t> point(clippedMesh.vertices.attributes(), "point");
//    for (GEO::index_t v: clippedMesh.vertices) {
//        std::cout << v << " " << point[v] << std::endl;
//    }

//            // Find a half edge for each vertex
//            GEO::vector < GEO::MeshHalfedges::Halfedge > v2h(clippedMesh.vertices.nb());
//    GEO::MeshHalfedges MH(clippedMesh);
//    for (GEO::index_t f = 0; f < clippedMesh.facets.nb(); ++f) {
//        for (GEO::index_t c = clippedMesh.facets.corners_begin(f);
//             c < clippedMesh.facets.corners_end(f); ++c
//                ) {
//            GEO::index_t v = clippedMesh.facet_corners.vertex(c);
//            v2h[v] = GEO::MeshHalfedges::Halfedge(f, c);
//        }
//    }
//
//    for (GEO::index_t v = 0; v < clippedMesh.vertices.nb(); ++v) {
//        GEO::MeshHalfedges::Halfedge H = v2h[v];
//        do {
//            MH.move_to_prev_around_vertex(H);
//        } while (H != v2h[v]);
//    }

    GEO::Mesh M_out;
}
