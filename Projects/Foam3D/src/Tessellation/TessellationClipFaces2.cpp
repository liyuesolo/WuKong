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
#include <tbb/tbb.h>

bool operator<(const Node &a, const Node &b) {
    if (a.type != b.type) return a.type < b.type;
    if (a.gen[0] != b.gen[0]) return a.gen[0] < b.gen[0];
    if (a.gen[1] != b.gen[1]) return a.gen[1] < b.gen[1];
    if (a.gen[2] != b.gen[2]) return a.gen[2] < b.gen[2];
    if (a.gen[3] != b.gen[3]) return a.gen[3] < b.gen[3];
    return false;
}

#define PRINT_INTERMEDIATE_TIMES false
#define PRINT_TOTAL_TIME true

static void
printTime(std::chrono::high_resolution_clock::time_point tstart, std::string description = "", bool final = false) {
    if (PRINT_INTERMEDIATE_TIMES || (final && PRINT_TOTAL_TIME)) {
        const auto tcurr = std::chrono::high_resolution_clock::now();
        std::cout << description << "Time: "
                  << std::chrono::duration_cast<std::chrono::microseconds>(tcurr - tstart).count() * 1.0e-6
                  << std::endl;
    }
}

void Tessellation::clipFaces2() {
    const auto tstart = std::chrono::high_resolution_clock::now();

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

    printTime(tstart, "Delaunay");

    GEO::mesh_tetrahedralize(boundaryMesh, false, false);
    GEO::RestrictedVoronoiDiagram_var rvd = GEO::RestrictedVoronoiDiagram::create(&dual, &boundaryMesh);
    rvd->set_volumetric(true);

//    for (int i = 0; i < boundaryMesh.cells.nb(); i++) {
//        std::cout << i << " find tet facet " << boundaryMesh.cells.find_tet_facet(i, 0, 2, 1) << std::endl;
//        std::cout << i << " find tet facet " << boundaryMesh.cells.find_tet_facet(i, 0, 4, 2) << std::endl;
//        std::cout << i << " find tet facet " << boundaryMesh.cells.find_tet_facet(i, 0, 1, 4) << std::endl;
//        if (boundaryMesh.cells.find_tet_facet(i, 0, 2, 1) < 16) {
//            std::cout << boundaryMesh.cells.facet(i, boundaryMesh.cells.find_tet_facet(i, 0, 2, 1)) << std::endl;
//        }
//        if (boundaryMesh.cells.find_tet_facet(i, 0, 4, 2) < 16) {
//            std::cout << boundaryMesh.cells.facet(i, boundaryMesh.cells.find_tet_facet(i, 0, 4, 2)) << std::endl;
//        }
//        if (boundaryMesh.cells.find_tet_facet(i, 0, 1, 4) < 16) {
//            std::cout << boundaryMesh.cells.facet(i, boundaryMesh.cells.find_tet_facet(i, 0, 1, 4)) << std::endl;
//        }
//    }

    printTime(tstart, "RVD");

    tbb::concurrent_vector<std::pair<int, int>> cellFacetToFacetPairs;
    tbb::parallel_for(size_t(0), boundary->f.size(), [&](size_t ff) {
        IV3 v = boundary->f[ff].vertices;
        for (int cc = 0; cc < boundaryMesh.cells.nb(); cc++) {
            int facet = boundaryMesh.cells.find_tet_facet(cc, v(0), v(2), v(1));
            if (facet != GEO::NO_FACET) {
                cellFacetToFacetPairs.emplace_back(boundaryMesh.cells.facet(cc, facet), ff);
                break;
            }
        }
    });
    std::map<int, int> cellFacetToFacet(cellFacetToFacetPairs.begin(), cellFacetToFacetPairs.end());


    printTime(tstart, "CellFacetMap");

    GEO::Mesh clippedMesh;
    GEO::BuildRVDMesh rvdMeshBuilder(clippedMesh);
    rvdMeshBuilder.set_simplify_internal_tet_facets(true);
    rvdMeshBuilder.set_simplify_voronoi_facets(true);
    rvdMeshBuilder.set_generate_ids(true);
    rvd->for_each_polyhedron(rvdMeshBuilder, true, true, false); // Parallel=true causes crash :(

    printTime(tstart, "RVDMeshBuilder");

//    std::cout << "WOW NB FACETS " << rvd->mesh()->facets.nb() << std::endl;
//    GEO::vector<std::string> names;
//    clippedMesh.facets.attributes().list_attribute_names(names);
//    for (auto s: names) {
//        std::cout << "Facet " << s << std::endl;
//    }
//    clippedMesh.vertices.attributes().list_attribute_names(names);
//    for (auto s: names) {
//        std::cout << "Vertex " << s << std::endl;
//    }
//    clippedMesh.cells.attributes().list_attribute_names(names);
//    for (auto s: names) {
//        std::cout << "Cell " << s << std::endl;
//    }
//
//    boundaryMesh.cell_facets.attributes().list_attribute_names(names);
//    for (auto s: names) {
//        std::cout << "Cell-facet " << s << std::endl;
//    }

//    GEO::Attribute<int> cell_id(clippedMesh.facets.attributes(), "cell_id");
//    GEO::Attribute<int> facet_seed_id(clippedMesh.facets.attributes(), "facet_seed_id");
//    GEO::Attribute<int> seed_id(clippedMesh.facets.attributes(), "seed_id");
//    for (GEO::index_t v: clippedMesh.facets) {
//        std::cout << "Facet " << v << " " << cell_id[v] << " " << facet_seed_id[v] << " " << seed_id[v] << std::endl;
//    }

//    GEO::Attribute<int> vertex_id(clippedMesh.vertices.attributes(), "vertex_id");
    GEO::Attribute<int> vertex_gen(clippedMesh.vertices.attributes(), "vertex_gen");
//    GEO::Attribute<double> point(clippedMesh.vertices.attributes(), "point");
//    for (GEO::index_t v: clippedMesh.vertices) {
//        std::cout << "Vertex " << v << " " << vertex_id[v] << " " << point[v * 3 + 0] << " " << point[v * 3 + 1] << " "
//                  << point[v * 3 + 2] << std::endl;
//        std::cout << "   gen " << vertex_gen[v * 6 + 0] << " " << vertex_gen[v * 6 + 1] << " " << vertex_gen[v * 6 + 2]
//                  << " " << vertex_gen[v * 6 + 3] << " " << vertex_gen[v * 6 + 4] << " " << vertex_gen[v * 6 + 5]
//                  << std::endl;
//    }

    int n_cells = c.rows() / 4;
    cells.resize(n_cells);

    for (GEO::index_t f: clippedMesh.facets) {
        int cellIdx = vertex_gen[clippedMesh.facets.vertex(f, 0) * 6 + 0];
        Face face;

        std::set<int> b_verts;

        for (GEO::index_t lv = 0; lv < clippedMesh.facets.nb_vertices(f); ++lv) {
            GEO::index_t v = clippedMesh.facets.vertex(f, lv);

            int gen[6];
            for (int i = 0; i < 6; i++) {
                gen[i] = vertex_gen[v * 6 + i];
            }

            Node node;
            if (gen[1] < 0 && gen[2] < 0 && gen[3] < 0) {
                node.type = NodeType::B_VERTEX;
                node.gen[0] = gen[4];
                node.gen[1] = -1;
                node.gen[2] = -1;
                node.gen[3] = -1;
                b_verts.insert(gen[4]);
            } else if (gen[1] < 0 && gen[2] < 0) {
                node.type = NodeType::B_EDGE;
                node.gen[0] = gen[4];
                node.gen[1] = gen[5];
                node.gen[2] = gen[0];
                node.gen[3] = gen[3] - 1;
                std::sort(std::begin(node.gen), std::begin(node.gen) + 2);
                std::sort(std::begin(node.gen) + 2, std::end(node.gen));
                b_verts.insert(gen[4]);
                b_verts.insert(gen[5]);
            } else if (gen[1] < 0) {
                node.type = NodeType::B_FACE;
                node.gen[0] = cellFacetToFacet.at(-gen[1] - 1);
                node.gen[1] = gen[0];
                node.gen[2] = gen[2] - 1;
                node.gen[3] = gen[3] - 1;
                std::sort(std::begin(node.gen) + 1, std::end(node.gen));
                b_verts.insert(boundary->f[node.gen[0]].vertices(0));
                b_verts.insert(boundary->f[node.gen[0]].vertices(1));
                b_verts.insert(boundary->f[node.gen[0]].vertices(2));
            } else {
                assert(gen[1] > 0 && gen[2] > 0 && gen[3] > 0);
                node.type = NodeType::STANDARD;
                node.gen[0] = gen[0];
                node.gen[1] = gen[1] - 1;
                node.gen[2] = gen[2] - 1;
                node.gen[3] = gen[3] - 1;
                std::sort(std::begin(node.gen), std::end(node.gen));
            }

            face.nodes.push_back(node);
        }

        if (b_verts.size() == 3) {
            std::vector<int> bv(b_verts.begin(), b_verts.end());
            for (int i = 0; i < boundary->f.size(); i++) {
                IV3 tri = boundary->f[i].vertices;
                std::sort(tri.data(), tri.data() + 3);
                if (std::memcmp(bv.data(), tri.data(), 3) == 0) {
                    face.bface = i;
                    break;
                }
            }
        }

        cells[cellIdx].faces.push_back(face);
    }

    printTime(tstart, "Cell data extraction");

    int ic = 0;
    for (Cell &cell: cells) {
        cell.cellIndex = ic;
        ic++;

        int i = 0;
        for (Face face: cell.faces) {
            for (Node n: face.nodes) {
                if (cell.nodeIndices.find(n) == cell.nodeIndices.end()) {
                    cell.nodeIndices[n] = i;
                    i++;
                }
            }
        }
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

    printTime(tstart, "Clipped Voronoi Diagram Construction ", true);
}
