#include <igl/readOBJ.h>
#include <igl/edges.h>
#include <igl/per_vertex_normals.h>
#include <igl/facet_adjacency_matrix.h>
#include "../include/VoronoiCells.h"

#include "geometrycentral/surface/edge_length_geometry.h"
#include "geometrycentral/surface/flip_geodesics.h"
#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/mesh_graph_algorithms.h"
#include "geometrycentral/surface/surface_mesh_factories.h"
#include "geometrycentral/surface/polygon_soup_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"
#include "geometrycentral/utilities/timing.h"
#include "geometrycentral/surface/exact_geodesics.h"

#include <CGAL/Simple_cartesian.h>
#include <CGAL/Advancing_front_surface_reconstruction.h>
#include <CGAL/tuple.h>
#include <boost/lexical_cast.hpp>

//CGAL 
typedef CGAL::Simple_cartesian<double> K;
typedef K::Point_3  Point_3;
typedef std::array<std::size_t,3> Facet;


void VoronoiCells::generateMeshForRendering(MatrixXT& V, MatrixXi& F, MatrixXT& C)
{
    V = surface_vtx; F = surface_indices;
    C.resize(F.rows(), 3); C.col(0).setConstant(0.0); C.col(1).setConstant(0.3); C.col(2).setConstant(1.0);
}

void VoronoiCells::triangulatePointCloud(const VectorXT& points, VectorXi& triangle_indices)
{
    if (n_sites != points.size() / 3)
    {
        std::cout << "# voronoi sites is incorrect" << std::endl;
        std::exit(0);
    }

    std::vector<Point_3> pointsCGAL;
    std::vector<Facet> facets;

    for (int i = 0; i < n_sites; i++)
        pointsCGAL.push_back(Point_3(points[i * 3 + 0],
        points[i * 3 + 1],
        points[i * 3 + 2]));
    
    
    CGAL::advancing_front_surface_reconstruction(pointsCGAL.begin(),
                                                pointsCGAL.end(),
                                                std::back_inserter(facets));
    
    triangle_indices.resize(facets.size() * 3);
    for (int i = 0; i < facets.size(); i++)
        triangle_indices.segment<3>(i * 3) = IV(facets[i][2], facets[i][1], facets[i][0]);
}


void VoronoiCells::constructCentroidalVD(const VectorXi& triangulation,
    const VectorXT& site_locations, VectorXT& nodal_positions,
    std::vector<VtxList>& cell_connectivity, 
    std::vector<std::pair<TV, TV>>& path_for_viz,
    bool generate_path)
{

    VectorXT mesh_vertices;
    iglMatrixFatten<T, 3>(surface_vtx, mesh_vertices);
    VectorXi mesh_indices;
    iglMatrixFatten<int, 3>(surface_indices, mesh_indices);

    using namespace geometrycentral;
    using namespace geometrycentral::surface;
    // == Geometry-central data
    std::unique_ptr<ManifoldSurfaceMesh> mesh;
    std::unique_ptr<VertexPositionGeometry> geometry;

    // An edge network while processing flips
    std::unique_ptr<FlipEdgeNetwork> edgeNetwork;

    int n_tri = mesh_indices.rows() / 3;
    std::vector<std::vector<size_t>> mesh_indices_gc(n_tri, std::vector<size_t>(3));
    for (int i = 0; i < n_tri; i++)
        for (int d = 0; d < 3; d++)
            mesh_indices_gc[i][d] = mesh_indices[i * 3 + d];
    int n_sites = mesh_vertices.rows() / 3;
    std::vector<Vector3> mesh_vertices_gc(n_sites);
    for (int i = 0; i < n_sites; i++)
        mesh_vertices_gc[i] = Vector3{mesh_vertices(i * 3 + 0), mesh_vertices(i * 3 + 1), mesh_vertices(i * 3 + 2)};
    
    auto lvals = makeManifoldSurfaceMeshAndGeometry(mesh_indices_gc, mesh_vertices_gc);
    std::tie(mesh, geometry) = std::tuple<std::unique_ptr<ManifoldSurfaceMesh>,
                    std::unique_ptr<VertexPositionGeometry>>(std::move(std::get<0>(lvals)),  // mesh
                                                             std::move(std::get<1>(lvals))); // geometry
    edgeNetwork = std::unique_ptr<FlipEdgeNetwork>(new FlipEdgeNetwork(*mesh, *geometry, {}));
    edgeNetwork->posGeom = geometry.get();
    
    MatrixXi igl_tri, igl_edges;
    vectorToIGLMatrix<int, 3>(triangulation, igl_tri);
    igl::edges(igl_tri, igl_edges);
    int n_vtx = mesh->nVertices();
    std::vector<bool> selected(n_vtx, false);
    for (int i = 0; i < igl_edges.rows(); i++)
    {
        int idx0 = igl_edges(i, 0), idx1 = igl_edges(i, 1);
        selected[idx0] = true; selected[idx1] = true;
        geometrycentral::surface::Vertex vA = edgeNetwork->tri->intrinsicMesh->vertex(idx0);
        geometrycentral::surface::Vertex vB = edgeNetwork->tri->intrinsicMesh->vertex(idx1);
        std::vector<geometrycentral::surface::Halfedge> path = shortestEdgePath(*edgeNetwork->tri, vA, vB);
        edgeNetwork->addPath(path);
        edgeNetwork->nFlips = 0;
        edgeNetwork->nShortenIters = 0;
        edgeNetwork->EPS_ANGLE = 1e-5;
        edgeNetwork->straightenAroundMarkedVertices = true;
        size_t iterLim = INVALID_IND;
        double lengthLim = 0.;
        edgeNetwork->addAllWedgesToAngleQueue();
        edgeNetwork->iterativeShorten(iterLim, lengthLim);
    }
    for (int i = 0; i < n_vtx; i++)
    {
        if (selected[i])
            continue;
        geometrycentral::surface::Vertex vtx = edgeNetwork->tri->intrinsicMesh->vertex(i); 
        geometrycentral::surface::Face f = edgeNetwork->tri->removeInsertedVertex(vtx);
    }
    edgeNetwork->makeDelaunay();
    
    int n_edges_before = edgeNetwork->tri->mesh.nEdges();
    std::vector<Vector3> nodes;
    std::vector<int> new_edge_indices;
    std::vector<Vertex> circumcenters;
    std::vector<std::vector<Vertex>> neignboring_faces(edgeNetwork->tri->mesh.nVertices());

    for (geometrycentral::surface::Face f : edgeNetwork->tri->mesh.faces()) 
    {
        // Vertex circumcenter = edgeNetwork->tri->insertCircumcenter(f);
        geometrycentral::surface::Vertex circumcenter = edgeNetwork->tri->insertBarycenter(f);
        // circumcenter_indices.push_back(circumcenter.getIndex());
        circumcenters.push_back(circumcenter);
        edgeNetwork->tri->refreshQuantities();
        // nodes.push_back(geometry->vertexPositions[circumcenter]);
        SurfacePoint p = edgeNetwork->tri->vertexLocations[circumcenter];
        Vector3 p3d = p.interpolate(geometry->inputVertexPositions);
        nodes.push_back(p3d);
        // std::cout << nodes.back() << std::endl;
        // std::getchar();
        for (geometrycentral::surface::Edge e : circumcenter.adjacentEdges())
            new_edge_indices.push_back(e.getIndex());
    }
    edgeNetwork->tri->refreshQuantities();



    for (geometrycentral::surface::Edge e : edgeNetwork->tri->intrinsicMesh->edges())
    {
        if (std::find(new_edge_indices.begin(), new_edge_indices.end(), e.getIndex()) == new_edge_indices.end())
        {
            edgeNetwork->tri->justFlip(e);
        }
    }

    voronoi_cell_vertices.resize(nodes.size() * 3);
    for (int i = 0; i < nodes.size(); i++)
    {
        voronoi_cell_vertices[i * 3 + 0] = nodes[i].x;
        voronoi_cell_vertices[i * 3 + 1] = nodes[i].y;
        voronoi_cell_vertices[i * 3 + 2] = nodes[i].z;
    }
    
    if (generate_path)
    {
        path_for_viz.resize(0);
        geometrycentral::surface::EdgeData<
            std::vector<geometrycentral::surface::SurfacePoint>
        > traces = edgeNetwork->tri->traceAllIntrinsicEdgesAlongInput();
    
        std::vector<std::vector<Vector3>> traces3D;
        
        for (geometrycentral::surface::Edge e : edgeNetwork->tri->mesh.edges()) 
        {
            if (std::find(new_edge_indices.begin(), new_edge_indices.end(), e.getIndex()) == new_edge_indices.end())
            {
                std::vector<TV> loop;
                for (geometrycentral::surface::SurfacePoint& p : traces[e]) 
                {
                    Vector3 vtx = p.interpolate(geometry->inputVertexPositions);
                    loop.push_back(TV(vtx.x, vtx.y, vtx.z));
                }
                for (int i = 0; i < loop.size(); i++)
                {
                    int j = (i + 1) % loop.size();
                    path_for_viz.push_back(std::make_pair(loop[i], loop[j]));
                }
            }
        }
        
        std::cout << path_for_viz.size() << std::endl;
    }

    // GeodesicAlgorithmExact mmp(*mesh, *geometry);
    
    // std::vector<SurfacePoint> sourcePoints;
    // SurfacePoint p = edgeNetwork->tri->vertexLocations[circumcenters[0]];
    // sourcePoints.push_back(p);
    // std::cout << "propagate start" << std::endl;
    // mmp.propagate(sourcePoints);
    // std::cout << "propagate" << std::endl;
    // // SurfacePoint queryPoint2(circumcenters[1]);
    // SurfacePoint queryPoint2 = edgeNetwork->tri->vertexLocations[circumcenters[1]];
    // double pathLength;
    
    // std::vector<SurfacePoint> path = mmp.traceBack(queryPoint2, pathLength);
    // std::cout << pathLength << std::endl;
    // if (generate_path)
    // {
    //     for (int i = 0; i < path.size() - 1; i++)
    //     {
    //         Vector3 vi = path[i].interpolate(geometry->inputVertexPositions);
    //         Vector3 vj = path[i+1].interpolate(geometry->inputVertexPositions);
    //         path_for_viz.push_back(std::make_pair(TV(vi.x, vi.y, vi.z), TV(vj.x, vj.y, vj.z)));
    //     }

    // }

}

void VoronoiCells::constructIntrinsicVoronoiDiagram(const VectorXi& triangulation, 
                                const VectorXT& site_locations,
                                const std::vector<std::pair<int, TV>>& barycentric_coords,
                                VectorXT& nodal_positions,
                                std::vector<VtxList>& cell_connectivity,
                                std::vector<std::pair<TV, TV>>& path_for_viz,
                                bool generate_path)
{
    VectorXT mesh_vertices;
    iglMatrixFatten<T, 3>(surface_vtx, mesh_vertices);
    VectorXi mesh_indices;
    iglMatrixFatten<int, 3>(surface_indices, mesh_indices);

    using namespace geometrycentral;
    using namespace geometrycentral::surface;
    using gcEdge = geometrycentral::surface::Edge;
    // == Geometry-central data
    std::unique_ptr<ManifoldSurfaceMesh> mesh;
    std::unique_ptr<VertexPositionGeometry> geometry;

    // An edge network while processing flips
    std::unique_ptr<FlipEdgeNetwork> edgeNetwork;

    int n_tri = mesh_indices.rows() / 3;
    std::vector<std::vector<size_t>> mesh_indices_gc(n_tri, std::vector<size_t>(3));
    for (int i = 0; i < n_tri; i++)
        for (int d = 0; d < 3; d++)
            mesh_indices_gc[i][d] = mesh_indices[i * 3 + d];
    int n_vtx_extrinsic = mesh_vertices.rows() / 3;
    std::vector<Vector3> mesh_vertices_gc(n_vtx_extrinsic);
    for (int i = 0; i < n_vtx_extrinsic; i++)
        mesh_vertices_gc[i] = Vector3{mesh_vertices(i * 3 + 0), mesh_vertices(i * 3 + 1), mesh_vertices(i * 3 + 2)};
    
    auto lvals = makeManifoldSurfaceMeshAndGeometry(mesh_indices_gc, mesh_vertices_gc);
    std::tie(mesh, geometry) = std::tuple<std::unique_ptr<ManifoldSurfaceMesh>,
                    std::unique_ptr<VertexPositionGeometry>>(std::move(std::get<0>(lvals)),  // mesh
                                                             std::move(std::get<1>(lvals))); // geometry
    edgeNetwork = std::unique_ptr<FlipEdgeNetwork>(new FlipEdgeNetwork(*mesh, *geometry, {}));
    edgeNetwork->posGeom = geometry.get();


    
    for (int i = 0; i < n_sites; i++)
    {
        Vector3 bary{barycentric_coords[i].second[0], 
                    barycentric_coords[i].second[1], 
                    barycentric_coords[i].second[2]};
        geometrycentral::surface::Face f = edgeNetwork->tri->inputMesh.face(barycentric_coords[i].first);
        SurfacePoint new_pt(f, bary);
        Vertex new_vtx = edgeNetwork->tri->insertVertex(new_pt);
        
    }
    
    int n_vtx = mesh->nVertices();
    std::vector<bool> selected(n_vtx, false);
    for (int i = 0; i < n_sites; i++)
    {
        selected[i+n_vtx_extrinsic] = true;
    }

    // MatrixXi igl_tri, igl_edges;
    // vectorToIGLMatrix<int, 3>(triangulation, igl_tri);
    // igl::edges(igl_tri, igl_edges);
    // for (int i = 0; i < igl_edges.rows(); i++)
    // {
    //     int idx0 = igl_edges(i, 0) + n_vtx_extrinsic, idx1 = igl_edges(i, 1) + n_vtx_extrinsic;
    //     selected[idx0] = true; selected[idx1] = true;
    //     geometrycentral::surface::Vertex vA = edgeNetwork->tri->intrinsicMesh->vertex(idx0);
    //     geometrycentral::surface::Vertex vB = edgeNetwork->tri->intrinsicMesh->vertex(idx1);
    //     std::vector<geometrycentral::surface::Halfedge> path = shortestEdgePath(*edgeNetwork->tri, vA, vB);
    //     edgeNetwork->addPath(path);
    //     edgeNetwork->nFlips = 0;
    //     edgeNetwork->nShortenIters = 0;
    //     edgeNetwork->EPS_ANGLE = 1e-5;
    //     edgeNetwork->straightenAroundMarkedVertices = true;
    //     size_t iterLim = INVALID_IND;
    //     double lengthLim = 0.;
    //     edgeNetwork->addAllWedgesToAngleQueue();
    //     edgeNetwork->iterativeShorten(iterLim, lengthLim);
    // }
    voronoi_cell_vertices.resize(3);
    for (int i = 0; i < n_vtx; i++)
    {
        if (selected[i])
            continue;
        geometrycentral::surface::Vertex vtx = edgeNetwork->tri->intrinsicMesh->vertex(i); 
        while (vtx.degree() != 3)
        {
            int cnt = 0;
            std::vector<gcEdge> gc_edges;
            for (geometrycentral::surface::Edge e : vtx.adjacentEdges())
            {
                if (cnt % 2 == 0)
                    gc_edges.push_back(e);
                cnt++;
            }
            for (gcEdge e : gc_edges)
                edgeNetwork->tri->justFlip(e);
        }
        // std::cout << cnt << " " << gc_edges.size() << std::endl;
        // geometrycentral::surface::Face f = edgeNetwork->tri->removeInsertedVertex(vtx);
        // edgeNetwork->tri->refreshQuantities();
        auto vtx_pos = geometry->vertexPositions[vtx];
        voronoi_cell_vertices[0] = vtx_pos.x;
        voronoi_cell_vertices[1] = vtx_pos.y;
        voronoi_cell_vertices[2] = vtx_pos.z;
        break;
    }
    // edgeNetwork->makeDelaunay();
    // std::cout << "make Delaunay Done" << std::endl;
    // int n_edges_before = edgeNetwork->tri->mesh.nEdges();
    // std::vector<Vector3> nodes;
    // std::vector<int> new_edge_indices;
    // std::vector<Vertex> circumcenters;
    // std::vector<std::vector<Vertex>> neignboring_faces(edgeNetwork->tri->mesh.nVertices());

    // for (geometrycentral::surface::Face f : edgeNetwork->tri->mesh.faces()) 
    // {
    //     // Vertex circumcenter = edgeNetwork->tri->insertCircumcenter(f);
    //     geometrycentral::surface::Vertex circumcenter = edgeNetwork->tri->insertBarycenter(f);
    //     // circumcenter_indices.push_back(circumcenter.getIndex());
    //     circumcenters.push_back(circumcenter);
    //     edgeNetwork->tri->refreshQuantities();
    //     // nodes.push_back(geometry->vertexPositions[circumcenter]);
    //     SurfacePoint p = edgeNetwork->tri->vertexLocations[circumcenter];
    //     Vector3 p3d = p.interpolate(geometry->inputVertexPositions);
    //     nodes.push_back(p3d);
    //     for (geometrycentral::surface::Edge e : circumcenter.adjacentEdges())
    //         new_edge_indices.push_back(e.getIndex());
    // }
    edgeNetwork->tri->refreshQuantities();

    if (generate_path)
    {
        path_for_viz.resize(0);
        geometrycentral::surface::EdgeData<
            std::vector<geometrycentral::surface::SurfacePoint>
        > traces = edgeNetwork->tri->traceAllIntrinsicEdgesAlongInput();
    
        std::vector<std::vector<Vector3>> traces3D;
        
        for (geometrycentral::surface::Edge e : edgeNetwork->tri->mesh.edges()) 
        {
            std::vector<TV> loop;
            for (geometrycentral::surface::SurfacePoint& p : traces[e]) 
            {
                Vector3 vtx = p.interpolate(geometry->inputVertexPositions);
                loop.push_back(TV(vtx.x, vtx.y, vtx.z));
            }
            for (int i = 0; i < loop.size()-1; i++)
            {
                int j = (i + 1) % loop.size();
                path_for_viz.push_back(std::make_pair(loop[i], loop[j]));
            }
        }
        
        std::cout << path_for_viz.size() << std::endl;
    }
}

void VoronoiCells::constructVoronoiDiagram()
{
    igl::readOBJ("/home/yueli/Documents/ETH/WuKong/Projects/VoronoiCells/data/sphere.obj", surface_vtx, surface_indices);
    // igl::readOBJ("/home/yueli/Documents/ETH/WuKong/Projects/VoronoiCells/data/drosophila_real_1.5k_remesh.obj", surface_vtx, surface_indices);
    // igl::readOBJ("/home/yueli/Documents/ETH/WuKong/Projects/VoronoiCells/data/drosophila_real_124_remesh.obj", surface_vtx, surface_indices);


    // sample one point per face
    int n_faces = surface_indices.rows();
    n_sites = n_faces;
    voronoi_sites.resize(n_sites * 3);
    std::vector<std::pair<int, TV>> barycentric_coords(n_faces);
    for (int i = 0; i < n_faces; i++)
    {
        T alpha = 0.2, beta = 0.5, gamma = 0.3;
        TV vi = surface_vtx.row(surface_indices(i, 0)) * alpha;
        TV vj = surface_vtx.row(surface_indices(i, 1)) * beta;
        TV vk = surface_vtx.row(surface_indices(i, 2)) * gamma;
        voronoi_sites.segment<3>(i * 3) = vi + vj + vk;
        barycentric_coords[i] = std::make_pair(i, TV(alpha, beta, gamma));
    }
    VectorXi triangulation;
    triangulatePointCloud(voronoi_sites, triangulation);

    VectorXT nodal_positions;
    std::vector<VtxList> cell_connectivity;
    std::vector<std::pair<TV, TV>> path_for_viz;

    START_TIMING(VD)
    constructIntrinsicVoronoiDiagram(triangulation, voronoi_sites, 
        barycentric_coords, nodal_positions, cell_connectivity, path_for_viz, true);
    FINISH_TIMING_PRINT(VD)    

    // std::vector<TV> sites_vector;
    // for (int i = 0; i < surface_vtx.rows(); i++)
    //     // if(i % 1 == 0)
    //     // if (i < 0.5 * surface_vtx.rows())
    //         sites_vector.push_back(surface_vtx.row(i));
    // n_sites = sites_vector.size();
    // voronoi_sites.resize(sites_vector.size() * 3);
    // for (int i = 0; i < sites_vector.size(); i++)
    //     voronoi_sites.segment<3>(i * 3) = sites_vector[i];
    
    // VectorXi triangulation;
    // triangulatePointCloud(voronoi_sites, triangulation);

    // VectorXT nodal_positions;
    // std::vector<VtxList> cell_connectivity;
    // std::vector<std::pair<TV, TV>> path_for_viz;
    
    // START_TIMING(VD)
    // constructCentroidalVD(triangulation, voronoi_sites, nodal_positions, cell_connectivity, path_for_viz, true);
    // FINISH_TIMING_PRINT(VD)

    voronoi_edges = path_for_viz;
    // int n_tri = triangulation.rows() / 3;
    // MatrixXi igl_faces;
    // vectorToIGLMatrix<int, 3>(triangulation, igl_faces);
    // MatrixXT igl_vertices;
    // vectorToIGLMatrix<T, 3>(voronoi_sites, igl_vertices);
    // std::ofstream out("triangulation_1.5k_edges.txt");
    // igl::edges(igl_faces, igl_edges);
    // int n_edges = igl_edges.rows();
    // out << n_edges << std::endl;
    // for (int i = 0; i < n_edges; i++)
    // {
    //     out << igl_edges(i, 0) << " " << igl_edges(i, 1) << std::endl;
    // }
    // out.close();

    // surface_vtx = igl_vertices;
    // surface_indices = igl_faces;
    // constructVoronoiDiagram(voronoi_sites.segment<3>(0), triangulation.segment<3>(0), 
    //     voronoi_cell_vertices, voronoi_cells);
}
