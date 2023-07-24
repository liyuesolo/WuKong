#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/edges.h>
#include <igl/per_vertex_normals.h>
#include <igl/facet_adjacency_matrix.h>
#include "../include/IntrinsicSimulation.h"

void IntrinsicSimulation::generateMeshForRendering(MatrixXT& V, MatrixXi& F, MatrixXT& C)
{
    vectorToIGLMatrix<int, 3>(extrinsic_indices, F);
    int n_vtx_dof = extrinsic_vertices.rows();
    if (two_way_coupling)
        vectorToIGLMatrix<T, 3>(deformed.segment(deformed.rows() - n_vtx_dof, n_vtx_dof), V);
    else
        vectorToIGLMatrix<T, 3>(extrinsic_vertices, V);
    C.resize(F.rows(), 3);
    C.col(0).setZero(); C.col(1).setConstant(0.3); C.col(2).setOnes();
}

void IntrinsicSimulation::initializeTriangleDebugScene()
{
    use_Newton = false;

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

    int n_mass_point = 4;    

    intrinsic_vertices_barycentric_coords.resize(n_mass_point * 2);
    
    VectorXT mass_point_Euclidean(n_mass_point * 3);
    int valid_cnt = 0;
    
    for (int face_idx : {0, 1, 5, 6})
    {
        T alpha = 0.2, beta = 0.5;
        TV vi = extrinsic_vertices.segment<3>(extrinsic_indices[face_idx * 3 + 0] * 3);
        TV vj = extrinsic_vertices.segment<3>(extrinsic_indices[face_idx * 3 + 1] * 3);
        TV vk = extrinsic_vertices.segment<3>(extrinsic_indices[face_idx * 3 + 2] * 3);
        TV current = vi * alpha + vj * beta + vk * (1.0 - alpha - beta);
        
        intrinsic_vertices_barycentric_coords.segment<2>(valid_cnt * 2) = TV2(alpha, beta);
        gcs::Face f = mesh->face(face_idx);
        SurfacePoint new_pt(f, Vector3{alpha, beta, 1.0 - alpha - beta});
        new_pt = new_pt.inSomeFace();
        mass_surface_points.push_back(std::make_pair(new_pt, f));
        valid_cnt++;
    }
    // intrinsic_vertices_barycentric_coords.conservativeResize(valid_cnt * 2);
    // mass_point_Euclidean.conservativeResize(valid_cnt * 3);

    mass_surface_points_undeformed = mass_surface_points;
    undeformed = intrinsic_vertices_barycentric_coords;
    deformed = undeformed;
    u.resize(undeformed.rows()); u.setZero();
    delta_u.resize(undeformed.rows()); delta_u.setZero();

    std::cout << "add mass point" << std::endl;
    std::cout << "#dof : " << undeformed.rows() << " #points " << undeformed.rows() / 2 << std::endl;
    for (int i = 0; i < 2; i++)
    {
        dirichlet_data[i] = 0.0;
    }


    MatrixXi igl_tri, igl_edges;

    igl_edges.resize(6, 2);
    igl_edges.row(0) = Eigen::RowVector2i(0, 1);
    igl_edges.row(1) = Eigen::RowVector2i(0, 2);
    igl_edges.row(2) = Eigen::RowVector2i(1, 2);
    igl_edges.row(3) = Eigen::RowVector2i(1, 3);
    igl_edges.row(4) = Eigen::RowVector2i(1, 2);
    igl_edges.row(5) = Eigen::RowVector2i(2, 3);

    triangles.push_back(Triangle(0, 1, 2));
    triangles.push_back(Triangle(1, 3, 2));

    all_intrinsic_edges.resize(0);

    int n_springs = igl_edges.rows();
    std::vector<std::vector<std::pair<TV, TV>>> 
        sub_pairs(n_springs, std::vector<std::pair<TV, TV>>());
    rest_length.resize(n_springs);
    spring_edges.resize(n_springs);
    VectorXT ref_lengths(n_springs);
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
        rest_length[i] = 0.0 * geo_dis;
        ref_lengths[i] = geo_dis;
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

    ref_dis = ref_lengths.sum() / ref_lengths.rows();

    updateCurrentState();
    mass_surface_points_undeformed = mass_surface_points;
    verbose = true; 
    computeAllTriangleArea(rest_area);
    rest_area.setZero();
    wa = 1;
    we = 0;                                   
}

void IntrinsicSimulation::initializeMassSpringDebugScene()
{
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

    // int n_faces = extrinsic_indices.rows()/ 3 / 10;  
    // int n_faces = extrinsic_indices.rows()/ 3;    
    int n_faces = 2;    

    intrinsic_vertices_barycentric_coords.resize(n_faces * 2);
    
    VectorXT mass_point_Euclidean(n_faces * 3);
    int valid_cnt = 0;
    // 1, 5 is a discontinuity
    for (int face_idx : {1, 2})
    {
        T alpha = 0.2, beta = 0.5;
        TV vi = extrinsic_vertices.segment<3>(extrinsic_indices[face_idx * 3 + 0] * 3);
        TV vj = extrinsic_vertices.segment<3>(extrinsic_indices[face_idx * 3 + 1] * 3);
        TV vk = extrinsic_vertices.segment<3>(extrinsic_indices[face_idx * 3 + 2] * 3);
        TV current = vi * alpha + vj * beta + vk * (1.0 - alpha - beta);
        
        intrinsic_vertices_barycentric_coords.segment<2>(valid_cnt * 2) = TV2(alpha, beta);
        gcs::Face f = mesh->face(face_idx);
        SurfacePoint new_pt(f, Vector3{alpha, beta, 1.0 - alpha - beta});
        new_pt = new_pt.inSomeFace();
        mass_surface_points.push_back(std::make_pair(new_pt, f));
        valid_cnt++;
    }
    // intrinsic_vertices_barycentric_coords.conservativeResize(valid_cnt * 2);
    // mass_point_Euclidean.conservativeResize(valid_cnt * 3);

    mass_surface_points_undeformed = mass_surface_points;
    undeformed = intrinsic_vertices_barycentric_coords;
    deformed = undeformed;
    u.resize(undeformed.rows()); u.setZero();
    delta_u.resize(undeformed.rows()); delta_u.setZero();

    std::cout << "add mass point" << std::endl;
    std::cout << "#dof : " << undeformed.rows() << " #points " << undeformed.rows() / 2 << std::endl;
    for (int i = 0; i < 2; i++)
    {
        dirichlet_data[i] = 0.0;
    }


    MatrixXi igl_tri, igl_edges;

    igl_edges.resize(1, 2);
    igl_edges.row(0) = Eigen::RowVector2i(0, 1);

    all_intrinsic_edges.resize(0);

    int n_springs = igl_edges.rows();
    std::vector<std::vector<std::pair<TV, TV>>> 
        sub_pairs(n_springs, std::vector<std::pair<TV, TV>>());
    rest_length.resize(n_springs);
    spring_edges.resize(n_springs);
    VectorXT ref_lengths(n_springs);
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
        // computeGeodesicHeatMethod(vA, vB, geo_dis, path, ixn_data, true);
        // std::cout << "geodesic " << geo_dis << std::endl;
        rest_length[i] = 0.0 * geo_dis;
        ref_lengths[i] = geo_dis;
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

    ref_dis = ref_lengths.sum() / ref_lengths.rows();

    current_length = rest_length;
    for (int i = 0; i < n_springs; i++)
    {
        edge_map[spring_edges[i]] = i;
        edge_map[Edge(spring_edges[i][1], spring_edges[i][0])] = i;
    }

    two_way_coupling = true;
 
    if (two_way_coupling)
    {
        int n_dof = undeformed.rows();
        shell_dof_start = n_dof;
        undeformed.conservativeResize(n_dof + extrinsic_vertices.rows());
        undeformed.segment(n_dof, extrinsic_vertices.rows()) = extrinsic_vertices;
        deformed = undeformed;
        u.resize(undeformed.rows()); u.setZero();
        delta_u.resize(undeformed.rows()); delta_u.setZero();
        faces = extrinsic_indices;
        computeRestShape();
        buildHingeStructure();

        E_shell = 0;
        updateShellLameParameters();
    }

    computeAllTriangleArea(rest_area);
    undeformed_area = rest_area;
    rest_area.setZero();
    wa = 0.0;
    // we = 0;

    add_geo_elasticity = false;
    if (add_geo_elasticity)
    {
        E = 1e6;
        updateLameParameters();
        computeGeodesicTriangleRestShape();
        
    }

    add_volume = false && two_way_coupling;
    if (add_volume)
    {
        rest_volume = computeVolume();
        wv = 1e3;
        woodbury = false;
    }
    // std::cout << "rest volume " << rest_volume << std::endl;
    
    updateCurrentState();
    mass_surface_points_undeformed = mass_surface_points;
    verbose = false;
}

void IntrinsicSimulation::initializeMassSpringSceneExactGeodesic()
{
    use_Newton = true;
    two_way_coupling = true;
    
    MatrixXT V; MatrixXi F;
    igl::readOBJ("/home/yueli/Documents/ETH/WuKong/Projects/VoronoiCells/data/sphere642.obj", 
        V, F);
    // igl::readOBJ("/home/yueli/Documents/ETH/WuKong/Projects/VoronoiCells/data/grid.obj", 
    //     V, F);
    // igl::readOBJ("/home/yueli/Documents/ETH/WuKong/Projects/VoronoiCells/data/sphere2562.obj", 
    //     V, F);
    // igl::readOBJ("/home/yueli/Documents/ETH/WuKong/Projects/VoronoiCells/data/sphere.obj", 
    //     V, F);
    // igl::readOBJ("/home/yueli/Documents/ETH/WuKong/Projects/VoronoiCells/data/coarse.obj", 
    //     V, F);
    // igl::readOBJ("/home/yueli/Documents/ETH/WuKong/Projects/VoronoiCells/data/single_tet.obj", 
    //     V, F);
    // igl::readOBJ("/home/yueli/Documents/ETH/WuKong/Projects/VoronoiCells/data/torus.obj", 
    //     V, F);
    // igl::readOBJ("/home/yueli/Documents/ETH/WuKong/Projects/VoronoiCells/data/ellipsoid.obj", 
    //     V, F);
    // igl::readOBJ("/home/yueli/Documents/ETH/WuKong/Projects/VoronoiCells/data/bumpy.obj", 
    //     V, F);
    // igl::readOBJ("/home/yueli/Documents/ETH/WuKong/Projects/VoronoiCells/data/bumpy-cube-simplified.obj", 
    //     V, F);
    // igl::readOBJ("/home/yueli/Documents/ETH/WuKong/Projects/VoronoiCells/data/bunny.obj", 
    //     V, F);
    // igl::readOBJ("/home/yueli/Documents/ETH/WuKong/Projects/VoronoiCells/data/ellipsoid_outward_bump.obj", 
    //     V, F);
    
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

    int n_faces = extrinsic_indices.rows()/ 3;    

    intrinsic_vertices_barycentric_coords.resize(n_faces * 2);
    
    VectorXT mass_point_Euclidean(n_faces * 3);
    int valid_cnt = 0;
    
    // for (int i = 0; i < n_faces; i++)
    // {
    //     // T alpha = 0.2, beta = 0.5;
    //     T alpha = 0.9, beta = 2.0 * IRREGULAR_EPSILON;
    //     // T alpha = 0.1, beta = 10.0 * IRREGULAR_EPSILON;
    //     TV vi = extrinsic_vertices.segment<3>(extrinsic_indices[i * 3 + 0] * 3);
    //     TV vj = extrinsic_vertices.segment<3>(extrinsic_indices[i * 3 + 1] * 3);
    //     TV vk = extrinsic_vertices.segment<3>(extrinsic_indices[i * 3 + 2] * 3);
    //     TV current = vi * alpha + vj * beta + vk * (1.0 - alpha - beta);
    //     mass_point_Euclidean.segment<3>(i * 3) =  current;
    //     intrinsic_vertices_barycentric_coords.segment<2>(i * 2) = TV2(alpha, beta);
    //     gcs::Face f = mesh->face(i);
    //     SurfacePoint new_pt(f, Vector3{alpha, beta, 1.0 - alpha - beta});
    //     new_pt = new_pt.inSomeFace();
    //     mass_surface_points.push_back(std::make_pair(new_pt, f));
    // }
    

    gcs::PoissonDiskSampler poissonSampler(*mesh, *geometry);
    std::vector<SurfacePoint> samples = poissonSampler.sample(5.0);
    // std::vector<SurfacePoint> samples = poissonSampler.sample();
    intrinsic_vertices_barycentric_coords.resize(samples.size() * 2);
    mass_point_Euclidean.resize(samples.size() * 3);
    int cnt = 0;
    for (SurfacePoint& pt : samples)
    {
        pt = pt.inSomeFace();
        mass_point_Euclidean.segment<3>(cnt * 3) = toTV(pt.interpolate(geometry->vertexPositions));
        intrinsic_vertices_barycentric_coords.segment<2>(cnt * 2) = TV2(pt.faceCoords.x, pt.faceCoords.y);
        mass_surface_points.push_back(std::make_pair(pt, pt.face));
        cnt++;
    }



    mass_surface_points_undeformed = mass_surface_points;
    undeformed = intrinsic_vertices_barycentric_coords;
    deformed = undeformed;
    u.resize(undeformed.rows()); u.setZero();
    delta_u.resize(undeformed.rows()); delta_u.setZero();

    std::cout << "add mass point" << std::endl;
    std::cout << "#dof : " << undeformed.rows() << " #points " << undeformed.rows() / 2 << std::endl;
    for (int i = 0; i < 2; i++)
    {
        dirichlet_data[i] = 0.0;
    }

    // for (int i = undeformed.rows() / 2; i < undeformed.rows() / 2 + 2; i++)
    // {
    //     dirichlet_data[i] = 0.0;
    // }
   
    
    VectorXi triangulation;
    triangulatePointCloud(mass_point_Euclidean, triangulation);

    MatrixXi igl_tri, igl_edges;
    vectorToIGLMatrix<int, 3>(triangulation, igl_tri);
    igl::edges(igl_tri, igl_edges);

    MatrixXT igl_vertices;
    vectorToIGLMatrix<T, 3>(mass_point_Euclidean, igl_vertices);
    igl::writeOBJ("triangulation.obj", igl_vertices, igl_tri);

    triangles.resize(igl_tri.rows());
    for (int i = 0; i < igl_tri.rows(); i++)
        triangles[i] = igl_tri.row(i);

    std::cout << "#triangles " << igl_tri.rows() << std::endl;
    // formEdgesFromConnection(F, igl_edges);


    if (two_way_coupling)
    {
        int n_dof = undeformed.rows();
        shell_dof_start = n_dof;
        undeformed.conservativeResize(n_dof + extrinsic_vertices.rows());
        undeformed.segment(n_dof, extrinsic_vertices.rows()) = extrinsic_vertices;
        deformed = undeformed;
        u.resize(undeformed.rows()); u.setZero();
        delta_u.resize(undeformed.rows()); delta_u.setZero();
        faces = extrinsic_indices;
        computeRestShape();
        buildHingeStructure();

        E_shell = 1e3;
        updateShellLameParameters();

        for (int d = 0; d < 3; d++)
        {
            dirichlet_data[shell_dof_start + d] = 0.0;
        }
        
    }

    all_intrinsic_edges.resize(0);

    int n_springs = igl_edges.rows();
    std::vector<std::vector<std::pair<TV, TV>>> 
        sub_pairs(n_springs, std::vector<std::pair<TV, TV>>());
    rest_length.resize(n_springs);
    spring_edges.resize(n_springs);
    VectorXT ref_lengths(n_springs);
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
        // computeExactGeodesicEdgeFlip(vA, vB, geo_dis, path, ixn_data, true);
        rest_length[i] = 0.0 * geo_dis;
        ref_lengths[i] = geo_dis;
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

    undeformed_length = ref_lengths;
    for (int i = 0; i < igl_edges.rows(); i++)
    {
        all_intrinsic_edges.insert(all_intrinsic_edges.end(), sub_pairs[i].begin(), sub_pairs[i].end());
    }

    ref_dis = ref_lengths.sum() / ref_lengths.rows();

    current_length = rest_length;
    for (int i = 0; i < n_springs; i++)
    {
        edge_map[spring_edges[i]] = i;
        edge_map[Edge(spring_edges[i][1], spring_edges[i][0])] = i;
    }

    
    computeAllTriangleArea(rest_area);
    undeformed_area = rest_area;
    rest_area.setZero();
    wa = 0.0;
    // we = 0;

    add_geo_elasticity = false;
    if (add_geo_elasticity)
    {
        E = 1e6;
        updateLameParameters();
        computeGeodesicTriangleRestShape();
        
    }

    add_volume = false && two_way_coupling;
    if (add_volume)
    {
        rest_volume = computeVolume();
        wv = 1e3;
        woodbury = false;
    }
    // std::cout << "rest volume " << rest_volume << std::endl;
    
    updateCurrentState();
    mass_surface_points_undeformed = mass_surface_points;
    verbose = false;
}


void IntrinsicSimulation::formEdgesFromConnection( 
    const MatrixXi& F, MatrixXi& igl_edges)
{
    std::vector<std::pair<int,int>> edge_vectors;
    Eigen::SparseMatrix<int> adj;
    igl::facet_adjacency_matrix(F, adj);
    for (int k=0; k < adj.outerSize(); ++k)
        for (Eigen::SparseMatrix<int>::InnerIterator it(adj,k); it; ++it)
        {
            edge_vectors.push_back(std::make_pair(it.row(), it.col()));
        }
    igl_edges.resize(edge_vectors.size(), 2);
    for (int i = 0; i < edge_vectors.size(); i++)
    {
        igl_edges(i, 0) = edge_vectors[i].first;
        igl_edges(i, 1) = edge_vectors[i].second;
    }
}

void IntrinsicSimulation::movePointsPlotEnergy()
{
    VectorXT edge_length_squared_gradient(undeformed.rows());
    edge_length_squared_gradient.setZero();
    computeResidual(edge_length_squared_gradient); edge_length_squared_gradient *= -1.0;
    std::cout << edge_length_squared_gradient.transpose() << std::endl;

    // TV normal = toTV(geometry->faceNormals[mass_surface_points[1].second]);
    T a = edge_length_squared_gradient[2],
        b = edge_length_squared_gradient[3];
    edge_length_squared_gradient[2] = -b;
    edge_length_squared_gradient[3] = a;
    

    int n_steps = 100;
    T gmin = -1e-2, g_max = 1e-2;
    T delta = (g_max - gmin) / n_steps;
    std::ofstream out("energy.txt");
    std::vector<T> energies, step_sizes;
    for (T du = gmin; du < g_max + delta; du+= delta)
    {
        mass_surface_points = mass_surface_points_undeformed;
        delta_u = du * edge_length_squared_gradient;
        updateCurrentState();
        T energy = computeTotalEnergy();
        energies.push_back(energy);
        step_sizes.push_back(du);
        // out << energy << " ";
    }
    for (int i = 0; i < energies.size(); i++)
        out << energies[i] << " ";
    out << std::endl;
    for (int i = 0; i < step_sizes.size(); i++)
        out << step_sizes[i] << " ";
    out << std::endl;
    out.close();
    mass_surface_points = mass_surface_points_undeformed;
    delta_u.setZero();
    updateCurrentState();
}