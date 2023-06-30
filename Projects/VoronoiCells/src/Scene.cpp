#include <igl/readOBJ.h>
#include <igl/edges.h>
#include <igl/per_vertex_normals.h>
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
    // igl::readOBJ("/home/yueli/Documents/ETH/WuKong/Projects/VoronoiCells/data/sphere.obj", 
    //     V, F);
    // igl::readOBJ("/home/yueli/Documents/ETH/WuKong/Projects/VoronoiCells/data/torus.obj", 
    //     V, F);
    // igl::readOBJ("/home/yueli/Documents/ETH/WuKong/Projects/VoronoiCells/data/ellipsoid.obj", 
    //     V, F);
    // igl::readOBJ("/home/yueli/Documents/ETH/WuKong/Projects/VoronoiCells/data/bumpy-cube.obj", 
    //     V, F);
    igl::readOBJ("/home/yueli/Documents/ETH/WuKong/Projects/VoronoiCells/data/bumpy-cube-simplified.obj", 
        V, F);
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

    int n_faces = extrinsic_indices.rows()/ 3 / 10;    

    intrinsic_vertices_barycentric_coords.resize(n_faces * 2);
    
    VectorXT mass_point_Euclidean(n_faces * 3);
    for (int i = 0; i < n_faces; i++)
    {
        T alpha = 0.2, beta = 0.5;
        // T alpha = 0.99, beta = 0.0;
        // T alpha = 0.5, beta = 0.0;
        TV vi = extrinsic_vertices.segment<3>(extrinsic_indices[i * 3 + 0] * 3);
        TV vj = extrinsic_vertices.segment<3>(extrinsic_indices[i * 3 + 1] * 3);
        TV vk = extrinsic_vertices.segment<3>(extrinsic_indices[i * 3 + 2] * 3);
        intrinsic_vertices_barycentric_coords.segment<2>(i * 2) = TV2(alpha, beta);
        gcs::Face f = mesh->face(i);
        SurfacePoint new_pt(f, Vector3{alpha, beta, 1.0 - alpha - beta});
        mass_point_Euclidean.segment<3>(i * 3) =  vi * alpha + vj * beta + vk * (1.0 - alpha - beta);
        mass_surface_points.push_back(std::make_pair(new_pt, f));
    }
    mass_surface_points_undeformed = mass_surface_points;
    undeformed = intrinsic_vertices_barycentric_coords;
    deformed = undeformed;
    u.resize(undeformed.rows()); u.setZero();
    delta_u.resize(undeformed.rows()); u.setZero();

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
        rest_length[i] = 0.0;
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
    
}
