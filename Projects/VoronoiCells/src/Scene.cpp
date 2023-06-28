#include <igl/readOBJ.h>
#include <igl/edges.h>
#include "../include/IntrinsicSimulation.h"

void IntrinsicSimulation::generateMeshForRendering(MatrixXT& V, MatrixXi& F, MatrixXT& C)
{
    vectorToIGLMatrix<int, 3>(extrinsic_indices, F);
    vectorToIGLMatrix<T, 3>(extrinsic_vertices, V);
    C.resize(F.rows(), 3);
    C.col(0).setZero(); C.col(1).setConstant(0.3); C.col(2).setOnes();
}

void IntrinsicSimulation::initializeMassSpringSceneExactGeodesic()
{
    use_intrinsic = false;
    use_Newton = true;
    MatrixXT V; MatrixXi F;
    // igl::readOBJ("/home/yueli/Documents/ETH/WuKong/Projects/VoronoiCells/data/sphere642.obj", 
    igl::readOBJ("/home/yueli/Documents/ETH/WuKong/Projects/VoronoiCells/data/sphere.obj", 
        V, F);

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

    int n_faces = extrinsic_indices.rows()/ 3;    
    intrinsic_vertices_barycentric_coords.resize(n_faces * 2);
    
    VectorXT mass_point_Euclidean(n_faces * 3);
    for (int i = 0; i < n_faces; i++)
    {
        T alpha = 0.2, beta = 0.5;
        TV vi = extrinsic_vertices.segment<3>(extrinsic_indices[i * 3 + 0] * 3);
        TV vj = extrinsic_vertices.segment<3>(extrinsic_indices[i * 3 + 1] * 3);
        TV vk = extrinsic_vertices.segment<3>(extrinsic_indices[i * 3 + 2] * 3);
        intrinsic_vertices_barycentric_coords.segment<2>(i * 2) = TV2(alpha, beta);
        gcs::Face f = mesh->face(i);
        SurfacePoint new_pt(f, Vector3{alpha, beta, 1.0 - alpha - beta});
        mass_point_Euclidean.segment<3>(i * 3) =  vi * alpha + vj * beta + vk * (1.0 - alpha - beta);
        mass_surface_points.push_back(std::make_pair(new_pt, f));
    }
    undeformed = intrinsic_vertices_barycentric_coords;
    deformed = undeformed;
    u.resize(undeformed.rows()); u.setZero();

    std::cout << "add mass point" << std::endl;
    std::cout << undeformed.rows() << std::endl;
    for (int i = 0; i < 2; i++)
    {
        dirichlet_data[i] = 0.0;
    }

    VectorXi triangulation;
    triangulatePointCloud(mass_point_Euclidean, triangulation);

    MatrixXi igl_tri, igl_edges;
    vectorToIGLMatrix<int, 3>(triangulation, igl_tri);
    igl::edges(igl_tri, igl_edges);

    std::cout << "triangulate" << std::endl;

    all_intrinsic_edges.resize(0);

    // tbb::parallel_for(0, spring_edges.size(); [&](int i){
    // });

    // for (int i = 0; i < igl_edges.rows(); i++)
    // {
    //     SurfacePoint vA = mass_surface_points[igl_edges(i, 0)].first;
    //     SurfacePoint vB = mass_surface_points[igl_edges(i, 1)].first;
        
    //     spring_edges.push_back(Edge(igl_edges(i, 0), igl_edges(i, 1)));
        
    //     T geo_dis; std::vector<SurfacePoint> path;
    //     computeExactGeodesic(vA, vB, geo_dis, path, true);
    //     rest_length.push_back(0.0);
    //     for(int i = 0; i < path.size() - 1; i++)
    //     {
    //         all_intrinsic_edges.push_back(std::make_pair(
    //             toTV(path[i].interpolate(geometry->vertexPositions)),
    //             toTV(path[i+1].interpolate(geometry->vertexPositions))
    //         ));
    //     }
    // }
    int n_springs = igl_edges.rows();
    std::vector<std::vector<std::pair<TV, TV>>> sub_pairs(n_springs, std::vector<std::pair<TV, TV>>());
    rest_length.resize(n_springs);
    spring_edges.resize(n_springs);
#ifdef PARALLEL_GEODESIC
    tbb::parallel_for(0, n_springs, [&](int i)
#else
    for (int i = 0; i < n_springs; i++)
#endif
    {
        SurfacePoint vA = mass_surface_points[igl_edges(i, 0)].first;
        SurfacePoint vB = mass_surface_points[igl_edges(i, 1)].first;
        spring_edges[i] = Edge(igl_edges(i, 0), igl_edges(i, 1));

        T geo_dis; std::vector<SurfacePoint> path;
        std::vector<IxnData> ixn_data;
        computeExactGeodesic(vA, vB, geo_dis, path, ixn_data, true);
        rest_length[i] = 0.0;
        for(int j = 0; j < path.size() - 1; j++)
        {
            sub_pairs[i].push_back(std::make_pair(
                toTV(path[j].interpolate(geometry->vertexPositions)),
                toTV(path[j+1].interpolate(geometry->vertexPositions))
            ));
        }
    }
#ifdef PARALLEL_GEODESIC
    );
#endif
    for (int i = 0; i < igl_edges.rows(); i++)
    {
        all_intrinsic_edges.insert(all_intrinsic_edges.end(), sub_pairs[i].begin(), sub_pairs[i].end());
    }
    
}

void IntrinsicSimulation::initializeMassSpringScene()
{
    use_intrinsic = true;
    MatrixXT V; MatrixXi F;
    igl::readOBJ("/home/yueli/Documents/ETH/WuKong/Projects/VoronoiCells/data/sphere.obj", 
        V, F);

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
    edgeNetwork = std::unique_ptr<gcs::FlipEdgeNetwork>(new gcs::FlipEdgeNetwork(*mesh, *geometry, {}));
    edgeNetwork->posGeom = geometry.get();

    int n_faces = extrinsic_indices.rows()/ 3;    
    intrinsic_vertices_barycentric_coords.resize(n_faces * 2);
    std::vector<FacePoint> face_points;
    VectorXT mass_point_Euclidean(n_faces * 3);
    for (int i = 0; i < n_faces; i++)
    {
        T alpha = 0.2, beta = 0.5;
        TV vi = extrinsic_vertices.segment<3>(extrinsic_indices[i * 3 + 0] * 3);
        TV vj = extrinsic_vertices.segment<3>(extrinsic_indices[i * 3 + 1] * 3);
        TV vk = extrinsic_vertices.segment<3>(extrinsic_indices[i * 3 + 2] * 3);
        intrinsic_vertices_barycentric_coords.segment<2>(i * 2) = TV2(alpha, beta);
        face_points.push_back(std::make_pair(i, TV(alpha, beta, 1.0-alpha-beta)));
        mass_point_Euclidean.segment<3>(i * 3) =  vi * alpha + vj * beta + vk * (1.0 - alpha - beta);
    }
    undeformed = intrinsic_vertices_barycentric_coords;
    deformed = undeformed;
    u.resize(undeformed.rows()); u.setZero();

    for (const FacePoint& pt : face_points)
    {
        Vector3 bary{pt.second[0], 
                    pt.second[1], 
                    pt.second[2]};
        // this point is on the extrinsic mesh
        gcs::Face f = edgeNetwork->tri->inputMesh.face(pt.first);
        SurfacePoint new_pt(f, bary);
        // std::cout << "mass point position"<<std::endl;
        // printVec3(new_pt.interpolate(geometry->vertexPositions));
        // std::cout << "==================="<<std::endl;
        SurfacePoint new_pt_intrinsic = edgeNetwork->tri->equivalentPointOnIntrinsic(new_pt);
        gcVertex new_vtx = edgeNetwork->tri->insertVertex(new_pt_intrinsic);
        mass_vertices.push_back(std::make_pair(new_vtx, f));
    }
    
    std::cout << "add mass point" << std::endl;

    for (int i = 0; i < 2; i++)
    {
        dirichlet_data[i] = 0.0;
    }

    VectorXi triangulation;
    triangulatePointCloud(mass_point_Euclidean, triangulation);

    MatrixXi igl_tri, igl_edges;
    vectorToIGLMatrix<int, 3>(triangulation, igl_tri);
    igl::edges(igl_tri, igl_edges);

    std::cout << "triangulate" << std::endl;


    for (int i = 0; i < igl_edges.rows(); i++)
    {
        gcVertex vA = mass_vertices[igl_edges(i, 0)].first;
        gcVertex vB = mass_vertices[igl_edges(i, 1)].first;
        std::vector<gcs::Halfedge> path = shortestEdgePath(*edgeNetwork->tri, vA, vB);
        edgeNetwork->addPath(path);
        edgeNetwork->nFlips = 0;
        edgeNetwork->nShortenIters = 0;
        edgeNetwork->EPS_ANGLE = 1e-5;
        edgeNetwork->straightenAroundMarkedVertices = true;
        size_t iterLim = gc::INVALID_IND;
        double lengthLim = 0.;
        edgeNetwork->addAllWedgesToAngleQueue();
        edgeNetwork->iterativeShorten(iterLim, lengthLim);
        gcEdge ei = edgeNetwork->tri->intrinsicMesh->connectingEdge(vA, vB);
        if (ei != gcEdge())
        {
            spring_edges.push_back(Edge(igl_edges(i, 0), igl_edges(i, 1)));
            rest_length.push_back(edgeNetwork->tri->edgeLengths[ei] * 0.9);
        }
        else
        {
            std::cout << "nonexisting edge" << std::endl;
        }
        // edgeNetwork->isMarkedVertex.setDefault(false);
        // edgeNetwork->paths.clear();
        // edgeNetwork->tri->clearMarkedEdges();
    }


    all_intrinsic_edges.resize(0);
    gcs::EdgeData<std::vector<SurfacePoint>> tracedEdges(edgeNetwork->tri->mesh);

    for (Edge eij : spring_edges) {
        gcVertex vA = mass_vertices[eij[0]].first;
        gcVertex vB = mass_vertices[eij[1]].first;
        gcEdge e = edgeNetwork->tri->intrinsicMesh->connectingEdge(vA, vB);
        gcs::Halfedge he = e.halfedge();
        tracedEdges[e] = edgeNetwork->tri->traceIntrinsicHalfedgeAlongInput(he);
        std::vector<TV> loop;
        for (gcs::SurfacePoint& p : tracedEdges[e]) 
        {
            Vector3 vtx = p.interpolate(geometry->inputVertexPositions);
            loop.push_back(TV(vtx.x, vtx.y, vtx.z));
        }
        for (int i = 0; i < loop.size()-1; i++)
        {
            int j = (i + 1) % loop.size();
            all_intrinsic_edges.push_back(std::make_pair(loop[i], loop[j]));
        }
    } 
}

void IntrinsicSimulation::initialize3MassPointScene()
{
    use_intrinsic = true;

    MatrixXT V; MatrixXi F;
    igl::readOBJ("/home/yueli/Documents/ETH/WuKong/Projects/VoronoiCells/data/sphere.obj", 
        V, F);

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
    edgeNetwork = std::unique_ptr<gcs::FlipEdgeNetwork>(new gcs::FlipEdgeNetwork(*mesh, *geometry, {}));
    edgeNetwork->posGeom = geometry.get();

    // intrinsic_vertices_undeformed.resize(3 * 3);
    intrinsic_vertices_barycentric_coords.resize(3 * 2);
    std::vector<FacePoint> face_points;
    // face 55 36 39
    int cnt = 0;
    for (int face_idx : {55, 36, 49})
    {
        T alpha = 0.2, beta = 0.5, gamma = 1.0 - alpha - beta;
        TV vi = extrinsic_vertices.segment<3>(extrinsic_indices[face_idx * 3 + 0] * 3);
        TV vj = extrinsic_vertices.segment<3>(extrinsic_indices[face_idx * 3 + 1] * 3);
        TV vk = extrinsic_vertices.segment<3>(extrinsic_indices[face_idx * 3 + 2] * 3);
        // intrinsic_vertices_undeformed.segment<3>(cnt * 3) = vi * alpha + vj * beta + vk * gamma;
        intrinsic_vertices_barycentric_coords.segment<2>(cnt * 2) = TV2(alpha, beta);
        face_points.push_back(std::make_pair(face_idx, TV(alpha, beta, gamma)));
        cnt++;
    }
    undeformed = intrinsic_vertices_barycentric_coords;
    deformed = undeformed;
    u.resize(undeformed.rows()); u.setZero();

    for (const FacePoint& pt : face_points)
    {
        Vector3 bary{pt.second[0], 
                    pt.second[1], 
                    pt.second[2]};
        // this point is on the extrinsic mesh
        gcs::Face f = edgeNetwork->tri->inputMesh.face(pt.first);
        SurfacePoint new_pt(f, bary);
        // std::cout << "mass point position"<<std::endl;
        // printVec3(new_pt.interpolate(geometry->vertexPositions));
        // std::cout << "==================="<<std::endl;
        SurfacePoint new_pt_intrinsic = edgeNetwork->tri->equivalentPointOnIntrinsic(new_pt);
        gcVertex new_vtx = edgeNetwork->tri->insertVertex(new_pt_intrinsic);
        mass_vertices.push_back(std::make_pair(new_vtx, f));
    }

    // fix barycentric coordinates of the first point
    for (int i = 4; i < 6; i++)
    {
        dirichlet_data[i] = 0.0;
    }
    
    
    for (int i = 0; i < mass_vertices.size() - 1; i++)
    {
        gcVertex vA = mass_vertices[i].first;
        gcVertex vB = mass_vertices[i + 1].first;
        std::vector<gcs::Halfedge> path = shortestEdgePath(*edgeNetwork->tri, vA, vB);
        edgeNetwork->addPath(path);
        edgeNetwork->nFlips = 0;
        edgeNetwork->nShortenIters = 0;
        edgeNetwork->EPS_ANGLE = 1e-5;
        edgeNetwork->straightenAroundMarkedVertices = true;
        size_t iterLim = gc::INVALID_IND;
        double lengthLim = 0.;
        edgeNetwork->addAllWedgesToAngleQueue();
        edgeNetwork->iterativeShorten(iterLim, lengthLim);
        gcEdge ei = edgeNetwork->tri->intrinsicMesh->connectingEdge(vA, vB);
        spring_edges.push_back(Edge(i, i+1));
        rest_length.push_back(edgeNetwork->tri->edgeLengths[ei] * 0.5);
        edgeNetwork->isMarkedVertex.setDefault(false);
        edgeNetwork->paths.clear();
        edgeNetwork->tri->clearMarkedEdges();
    }

    all_intrinsic_edges.resize(0);
    gcs::EdgeData<std::vector<SurfacePoint>> tracedEdges(edgeNetwork->tri->mesh);

    for (Edge eij : spring_edges) {
        gcVertex vA = mass_vertices[eij[0]].first;
        gcVertex vB = mass_vertices[eij[1]].first;
        gcEdge e = edgeNetwork->tri->intrinsicMesh->connectingEdge(vA, vB);
        gcs::Halfedge he = e.halfedge();
        tracedEdges[e] = edgeNetwork->tri->traceIntrinsicHalfedgeAlongInput(he);
        std::vector<TV> loop;
        for (gcs::SurfacePoint& p : tracedEdges[e]) 
        {
            Vector3 vtx = p.interpolate(geometry->inputVertexPositions);
            loop.push_back(TV(vtx.x, vtx.y, vtx.z));
        }
        for (int i = 0; i < loop.size()-1; i++)
        {
            int j = (i + 1) % loop.size();
            all_intrinsic_edges.push_back(std::make_pair(loop[i], loop[j]));
        }
    }        
}
