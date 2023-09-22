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

#include "../include/Util.h"

void VoronoiCells::generateMeshForRendering(MatrixXT& V, MatrixXi& F, MatrixXT& C)
{
    vectorToIGLMatrix<int, 3>(extrinsic_indices, F);
    int n_vtx_dof = extrinsic_vertices.rows();
    vectorToIGLMatrix<T, 3>(extrinsic_vertices, V);
    C.resize(F.rows(), 3);
    C.col(0).setZero(); C.col(1).setConstant(0.3); C.col(2).setOnes();
    if (use_debug_face_color)
        C = face_color;
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
    MatrixXT V; MatrixXi F;
    igl::readOBJ("/home/yueli/Documents/ETH/WuKong/Projects/VoronoiCells/data/grid.obj", V, F);
    MatrixXT N;
    igl::per_vertex_normals(V, F, N);

    // V.row(10) += 3.0 * N.row(10);

    iglMatrixFatten<T, 3>(V, extrinsic_vertices);
    iglMatrixFatten<int, 3>(F, extrinsic_indices);

    int n_tri = extrinsic_indices.rows() / 3;
    std::vector<std::vector<size_t>> mesh_indices_gc(n_tri, std::vector<size_t>(3));
    for (int i = 0; i < n_tri; i++)
        for (int d = 0; d < 3; d++)
            mesh_indices_gc[i][d] = extrinsic_indices[i * 3 + d];
    int n_vtx_extrinsic = extrinsic_vertices.rows() / 3;
    std::vector<gc::Vector3> mesh_vertices_gc(n_vtx_extrinsic);
    for (int i = 0; i < n_vtx_extrinsic; i++)
        mesh_vertices_gc[i] = gc::Vector3{extrinsic_vertices(i * 3 + 0), 
            extrinsic_vertices(i * 3 + 1), extrinsic_vertices(i * 3 + 2)};
    
    auto lvals = gcs::makeManifoldSurfaceMeshAndGeometry(mesh_indices_gc, mesh_vertices_gc);
    std::tie(mesh, geometry) = std::tuple<std::unique_ptr<gcs::ManifoldSurfaceMesh>,
                    std::unique_ptr<gcs::VertexPositionGeometry>>(std::move(std::get<0>(lvals)),  // mesh
                                                             std::move(std::get<1>(lvals))); // geometry

    gcs::PoissonDiskSampler poissonSampler(*mesh, *geometry);
    std::vector<SurfacePoint> samples = poissonSampler.sample(4.0);
    // std::vector<SurfacePoint> samples;
    // samples.push_back(SurfacePoint(mesh->face(80), gc::Vector3{0.2, 0.8, 0.}));
    // samples.push_back(SurfacePoint(mesh->face(110), gc::Vector3{1,0,0}));


    n_sites = samples.size();
    voronoi_sites.resize(n_sites * 3);
    int cnt = 0;
    for (SurfacePoint& pt : samples)
    {
        pt = pt.inSomeFace();
        voronoi_sites.segment<3>(cnt * 3) = toTV(pt.interpolate(geometry->vertexPositions));
        cnt++;
    }

    enum DistanceMetric
    {
        Geodesic, Euclidean
    };

    DistanceMetric metric = Euclidean;

    struct FaceData
    {
        std::vector<int> site_indices;
        TV distances;
        FaceData (const std::vector<int>& _site_indices, 
            const TV& _distances) : 
            site_indices(_site_indices), distances(_distances) {}
        FaceData () : distances(TV::Constant(1e10)) {}

    };
    
    std::vector<FaceData> source_data(n_tri, FaceData());

    auto comp = [&]( std::pair<int, int> a, std::pair<int, int> b ) 
    { 
        return source_data[a.first].distances.minCoeff() > source_data[b.first].distances.minCoeff(); 
    };
    std::priority_queue<std::pair<int, int>, 
        std::vector<std::pair<int, int>>, 
        decltype(comp)> queue(comp);
    // std::queue<std::pair<int, int>> queue;

    int sample_cnt = 0;
    for (SurfacePoint& pt : samples)
    {
        if (pt.type == gcs::SurfacePointType::Vertex)
        {
            std::cout<<"vertex"<<std::endl;
            auto he = pt.vertex.halfedge();
            do
            {
                queue.push(std::make_pair(he.face().getIndex(), sample_cnt));
                auto next_he = he.twin().next();
                he = next_he;
            } while (he != pt.vertex.halfedge());
            
        }
        else if (pt.type == gcs::SurfacePointType::Edge)
        {
            std::cout<<"edge"<<std::endl;
            auto he = pt.edge.halfedge();
            queue.push(std::make_pair(he.face().getIndex(), sample_cnt));
            queue.push(std::make_pair(he.twin().face().getIndex(), sample_cnt));    
        }
        else if (pt.type == gcs::SurfacePointType::Face)
        {
            int face_idx = pt.face.getIndex();
            queue.push(std::make_pair(face_idx, sample_cnt));    
        }
        else
        {
            std::cout << "point type error " << __FILE__ << std::endl;
            std::exit(0);
        }
        sample_cnt++;
    }

    while (queue.size())
    {
        std::pair<int, int> data_top = queue.top();
        queue.pop();

        int face_idx = data_top.first;
        int site_idx = data_top.second;

        SurfacePoint pt = samples[site_idx];
        TV site_location = toTV(pt.interpolate(geometry->vertexPositions));
        TV v0 = toTV(geometry->vertexPositions[mesh->face(face_idx).halfedge().vertex()]);
        TV v1 = toTV(geometry->vertexPositions[mesh->face(face_idx).halfedge().next().vertex()]);
        TV v2 = toTV(geometry->vertexPositions[mesh->face(face_idx).halfedge().next().next().vertex()]);
        TV current_distance;
        current_distance[0] = (site_location - v0).norm();
        current_distance[1] = (site_location - v1).norm();
        current_distance[2] = (site_location - v2).norm();
        // std::cout << "site " << site_idx << " pos " << site_location.transpose() << std::endl;
        // std::cout << "face " << face_idx << " dis " << current_distance.transpose() << " " << source_data[face_idx].distances.transpose() << std::endl;
        bool updated = false;
        for (int d = 0; d < 3; d++)
        {
            if (current_distance[d] < source_data[face_idx].distances[d])
            {
                source_data[face_idx].distances[d] = current_distance[d];
                updated = true;
            }
        }
        // std::cout << "update " << updated << std::endl;
        // std::getchar();
        if (updated)
        {
            source_data[face_idx].site_indices.push_back(site_idx);
            for (auto face : mesh->face(face_idx).adjacentFaces())
            {
                queue.push(std::make_pair(face.getIndex(), site_idx));
            }
        }
        
    }

    face_color.resize(n_tri, 3);
    for (int i = 0; i < n_tri; i++)
    {
        if (source_data[i].site_indices.size() == 0)
            face_color.row(i) = TV(1, 1 , 1);
        else if (source_data[i].site_indices.size() == 1)
            face_color.row(i) = TV(0, 1, 0);
        else if (source_data[i].site_indices.size() == 2)
            face_color.row(i) = TV(1, 0, 0);
        else if (source_data[i].site_indices.size() == 3)
            face_color.row(i) = TV(1, 1, 0);
        else if (source_data[i].site_indices.size() == 4)
            face_color.row(i) = TV(0, 1, 1);
    }
    use_debug_face_color = true;
        
}
