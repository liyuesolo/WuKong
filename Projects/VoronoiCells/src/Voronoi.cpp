#include <igl/readOBJ.h>
#include <igl/edges.h>
#include <igl/per_vertex_normals.h>
#include <igl/facet_adjacency_matrix.h>
#include <igl/adjacency_matrix.h>
#include <igl/copyleft/cgal/intersect_with_half_space.h>
#include <igl/copyleft/cgal/extract_feature.h>
#include <igl/copyleft/cgal/mesh_boolean.h>
#include <igl/copyleft/cgal/coplanar.h>
#include <igl/point_simplex_squared_distance.h>
#include <igl/point_mesh_squared_distance.h>
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

#define PARALLEL

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


void VoronoiCells::computeGeodesicDistance(const SurfacePoint& va, const SurfacePoint& vb,
    T& dis, std::vector<SurfacePoint>& path, 
        std::vector<IxnData>& ixn_data, bool trace_path)
{
    dis = 0.0;
    if (trace_path)
    {
        ixn_data.clear();
        path.clear();
    }
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


    // START_TIMING(contruct)
    auto lvals = gcs::makeManifoldSurfaceMeshAndGeometry(mesh_indices_gc, mesh_vertices_gc);
    // FINISH_TIMING_PRINT(contruct)
    std::unique_ptr<gcs::ManifoldSurfaceMesh> sub_mesh;
    std::unique_ptr<gcs::VertexPositionGeometry> sub_geometry;
    std::tie(sub_mesh, sub_geometry) = std::tuple<std::unique_ptr<gcs::ManifoldSurfaceMesh>,
                    std::unique_ptr<gcs::VertexPositionGeometry>>(std::move(std::get<0>(lvals)),  // mesh
                                                             std::move(std::get<1>(lvals))); // geometry
    

    // std::unique_ptr<gcs::ManifoldSurfaceMesh> sub_mesh = mesh->copy();
    // std::unique_ptr<gcs::VertexPositionGeometry> sub_geometry = geometry->copy();
    // std::cout << "constructing mmp"<<std::endl;
    gcs::GeodesicAlgorithmExact mmp(*sub_mesh, *sub_geometry);
    // std::cout << "constructing mmp done"<<std::endl;
    SurfacePoint va_sub(sub_mesh->face(va.face.getIndex()), va.faceCoords);
    SurfacePoint vb_sub(sub_mesh->face(vb.face.getIndex()), vb.faceCoords);
    // va_sub = va_sub.inSomeFace();
    // vb_sub = vb_sub.inSomeFace();
    // std::cout << "va idx " << va.face.getIndex() << " vb idx " << vb.face.getIndex() << std::endl;
    // std::cout << "start MMP" << std::endl;
    mmp.propagate(va_sub);
    if (trace_path)
    {
        // std::cout << "trace " << trace_path << std::endl;
        path = mmp.traceBack(vb_sub, dis);
        std::reverse(path.begin(), path.end());
        for (auto& pt : path)
        {
            T edge_t = -1.0; TV start = TV::Zero(), end = TV::Zero();
            bool is_edge_point = (pt.type == gcs::SurfacePointType::Edge);
            bool is_vtx_point = (pt.type == gcs::SurfacePointType::Vertex);
            Edge start_end; start_end.setConstant(-1);
            if (is_edge_point)
            {
                auto he = pt.edge.halfedge();
                SurfacePoint start_extrinsic = he.tailVertex();
                SurfacePoint end_extrinsic = he.tipVertex();
                start = toTV(start_extrinsic.interpolate(sub_geometry->vertexPositions));
                end = toTV(end_extrinsic.interpolate(sub_geometry->vertexPositions));
                start_end[0] = start_extrinsic.vertex.getIndex();
                start_end[1] = end_extrinsic.vertex.getIndex();
                TV ixn = toTV(pt.interpolate(sub_geometry->vertexPositions));
                TV test_interp = pt.tEdge * start + (1.0 - pt.tEdge) * end;
                if ((test_interp - ixn).norm() > 1e-6)
                {
                    std::swap(start, end);
                    std::swap(start_end[0], start_end[1]);
                }
                
                TV dir0 = (end-start).normalized();
                TV dir1 = (ixn-start).normalized();
                if ((dir0.cross(dir1)).norm() > 1e-6)
                {
                    std::cout << "start " << start.transpose() << " " << end.transpose() << std::endl;
                    std::cout << "error in cross product " << __FILE__ << std::endl;
                    std::exit(0);
                }
                test_interp = pt.tEdge * start + (1.0 - pt.tEdge) * end;
                if ((ixn - test_interp).norm() > 1e-6)
                {
                    std::cout << "error in interpolation " << __FILE__ << std::endl;
                    std::exit(0);
                }
                edge_t = pt.tEdge;
                // std::cout << pt.tEdge << " " << " cross " << (dir0.cross(dir1)).norm() << std::endl;
                // std::cout << ixn.transpose() << " " << (pt.tEdge * start + (1.0-pt.tEdge) * end).transpose() << std::endl;
                // std::getchar();
            }            
            else if (is_vtx_point)
            {
                // std::cout << "is vertex point" << std::endl;
                
                auto he = pt.vertex.halfedge();
                SurfacePoint start_extrinsic = he.tailVertex();
                SurfacePoint end_extrinsic = he.tipVertex();
                start = toTV(start_extrinsic.interpolate(sub_geometry->vertexPositions));
                end = toTV(end_extrinsic.interpolate(sub_geometry->vertexPositions));
                // std::cout << start.transpose() << " " << end.transpose() << std::endl;
                start_end[0] = start_extrinsic.vertex.getIndex();
                start_end[1] = end_extrinsic.vertex.getIndex();
                // std::cout << start_end[0] << " " << start_end[1] << std::endl;
                TV ixn = toTV(pt.interpolate(sub_geometry->vertexPositions));
                // std::cout << ixn.transpose() << std::endl;
                TV test_interp = start;
                if ((test_interp - ixn).norm() > 1e-6)
                {
                    std::swap(start, end);
                    std::swap(start_end[0], start_end[1]);
                }
                TV dir0 = (end-start).normalized();
                TV dir1 = (ixn-start).normalized();
                if ((dir0.cross(dir1)).norm() > 1e-6)
                {
                    std::cout << "error in cross product " << __FILE__ << std::endl;
                    std::exit(0);
                }
                edge_t = 1.0;
            }
            else
            {
                edge_t = 2.0;
            }
            
            ixn_data.push_back(IxnData(start, end, (1.0-edge_t), start_end[0], start_end[1]));
            pt.edge = mesh->edge(pt.edge.getIndex());
            pt.vertex = mesh->vertex(pt.vertex.getIndex());
            pt.face = mesh->face(pt.face.getIndex());
            
            pt = pt.inSomeFace();
        }
        TV v0 = toTV(path[0].interpolate(geometry->vertexPositions));
        TV v1 = toTV(path[path.size() - 1].interpolate(geometry->vertexPositions));
        TV ixn0 = toTV(path[1].interpolate(geometry->vertexPositions));
        TV ixn1 = toTV(path[path.size() - 2].interpolate(geometry->vertexPositions));
        
        if (path.size() > 2)
            if ((v0 - ixn0).norm() < 1e-6)
                path.erase(path.begin() + 1);
        if (path.size() > 2)
            if ((v1 - ixn1).norm() < 1e-6)
                path.erase(path.end() - 2);
        
    }
    else
        dis = mmp.getDistance(vb_sub);
    // std::cout << "done MMP" << std::endl;
}


void VoronoiCells::propogateDistanceField(std::vector<SurfacePoint>& samples,
    std::vector<FaceData>& source_data)
{
    int n_samples = samples.size();
    std::vector<gcs::VertexData<T>> dis_to_sources(n_samples);
    // START_TIMING(MMPallNodes)
    if (metric == Geodesic)
    {
        tbb::parallel_for(0, n_samples, [&](int i)
        {
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


            // START_TIMING(contruct)
            auto lvals = gcs::makeManifoldSurfaceMeshAndGeometry(mesh_indices_gc, mesh_vertices_gc);
            // FINISH_TIMING_PRINT(contruct)
            std::unique_ptr<gcs::ManifoldSurfaceMesh> sub_mesh;
            std::unique_ptr<gcs::VertexPositionGeometry> sub_geometry;
            std::tie(sub_mesh, sub_geometry) = std::tuple<std::unique_ptr<gcs::ManifoldSurfaceMesh>,
                            std::unique_ptr<gcs::VertexPositionGeometry>>(std::move(std::get<0>(lvals)),  // mesh
                                                                    std::move(std::get<1>(lvals))); // geometry
            
            gcs::GeodesicAlgorithmExact mmp(*sub_mesh, *sub_geometry);
            mmp.propagate(samples[i]);
            dis_to_sources[i] = mmp.getDistanceFunction();
        });
    }
    // FINISH_TIMING_PRINT(MMPallNodes)
    std::queue<std::pair<int, int>> queue;

    int sample_cnt = 0;
    for (SurfacePoint& pt : samples)
    {
        if (pt.type == gcs::SurfacePointType::Vertex)
        {
            // std::cout << "vertex" << std::endl;
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
            // std::cout << "edge" << std::endl;
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
        // std::pair<int, int> data_top = queue.top();
        std::pair<int, int> data_top = queue.front();
        queue.pop();

        int face_idx = data_top.first;
        int site_idx = data_top.second;

        SurfacePoint pt = samples[site_idx].inSomeFace();
        TV site_location = toTV(pt.interpolate(geometry->vertexPositions));
        TV v0 = toTV(geometry->vertexPositions[mesh->face(face_idx).halfedge().vertex()]);
        TV v1 = toTV(geometry->vertexPositions[mesh->face(face_idx).halfedge().next().vertex()]);
        TV v2 = toTV(geometry->vertexPositions[mesh->face(face_idx).halfedge().next().next().vertex()]);
        TV current_distance = TV::Zero();
        if (metric == Euclidean)
        {
            current_distance[0] = (site_location - v0).norm();
            current_distance[1] = (site_location - v1).norm();
            current_distance[2] = (site_location - v2).norm();
        }
        else if (metric == Geodesic)
        {
            
            current_distance[0] = dis_to_sources[site_idx][mesh->face(face_idx).halfedge().vertex()];
            current_distance[1] = dis_to_sources[site_idx][mesh->face(face_idx).halfedge().next().vertex()];
            current_distance[2] = dis_to_sources[site_idx][mesh->face(face_idx).halfedge().next().next().vertex()];
            // std::vector<IxnData> ixn_data;
            // std::vector<SurfacePoint> path; 
            // SurfacePoint vtx0 = SurfacePoint(mesh->face(face_idx).halfedge().vertex()).inSomeFace();
            // SurfacePoint vtx1 = SurfacePoint(mesh->face(face_idx).halfedge().next().vertex()).inSomeFace();
            // SurfacePoint vtx2 = SurfacePoint(mesh->face(face_idx).halfedge().next().next().vertex()).inSomeFace();
            // std::vector<SurfacePoint> vertices = {vtx0, vtx1, vtx2};
            // tbb::parallel_for(0, 3, [&](int i)
            // {
            //     computeGeodesicDistance(pt, vertices[i], 
            //         current_distance[i], path, ixn_data, false);
            // });
            
        }
        else
        {
            std::cout << "Unknown metric!!!! set to Euclidean" << std::endl;
            metric = Euclidean;
        }
        // std::cout << "site " << site_idx << " pos " << site_location.transpose() << std::endl;
        // std::cout << "face " << face_idx << " dis " << current_distance.transpose() << " " << source_data[face_idx].distances.transpose() << std::endl;
        bool updated = true;
        
        for (int i = 0; i < source_data[face_idx].site_indices.size(); i++)
        {
            TV existing_distance = source_data[face_idx].distances[i];
            bool larger_for_all_vertices = true;
            for (int d = 0; d < 3; d++)
            {
                if (current_distance[d] < existing_distance[d])
                {
                    larger_for_all_vertices = false;
                }
            }
            if (larger_for_all_vertices)
                updated = false;
        }
        
        
        if (updated)
        {
            source_data[face_idx].site_indices.push_back(site_idx);
            source_data[face_idx].distances.push_back(current_distance);
            for (auto face : mesh->face(face_idx).adjacentFaces())
            {
                queue.push(std::make_pair(face.getIndex(), site_idx));
            }
        }
        
    }
}

void VoronoiCells::intersectPrisms(std::vector<SurfacePoint>& samples,
            std::vector<FaceData>& source_data, 
            std::vector<std::pair<SurfacePoint, std::vector<int>>>& ixn_data)
{
    // return;
    T max_h = 1.0;
    T EPSILON = 1e-12;



    // check if pt is the same as v0, v1, v2
    auto check_if_existing_projection = [&](const TV2& v0, 
        const TV2& v1, const TV2& v2, const TV2& pt) -> bool
    {
        if ((pt - v0).norm() < EPSILON)
            return true;
        if ((pt - v1).norm() < EPSILON)
            return true;
        if ((pt - v2).norm() < EPSILON)
            return true;
        return false;
    };
    // std::vector<std::vector<std::pair<TV, TV>>> edges_thread(source_data.size(), 
    //     std::vector<std::pair<TV, TV>>());
    std::vector<std::vector<std::pair<SurfacePoint, std::vector<int>>>> ixn_data_thread(source_data.size(),
        std::vector<std::pair<SurfacePoint, std::vector<int>>>());

    std::vector<std::vector<IV2>> edge_graph_thread(source_data.size(), std::vector<IV2>());
#ifdef PARALLEL
    tbb::parallel_for(0, (int)source_data.size(), [&](int face_idx)
#else
    for (int face_idx = 0; face_idx < source_data.size(); face_idx++)
#endif
    {
        // START_TIMING(face)
        FaceData face_data = source_data[face_idx];
        
        // don't consider 1 vertex case
        if (face_data.site_indices.size() == 1)
#ifdef PARALLEL
            return;
#else
            continue;
#endif
        TV v0 = toTV(geometry->vertexPositions[mesh->face(face_idx).halfedge().vertex()]);
        TV v1 = toTV(geometry->vertexPositions[mesh->face(face_idx).halfedge().next().vertex()]);
        TV v2 = toTV(geometry->vertexPositions[mesh->face(face_idx).halfedge().next().next().vertex()]);
        // std::cout << "geometry done" << std::endl;


        // rigidly transform to center
        TV trans = (v0 + v1 + v2) / 3.0;
        TV v0_prime = v0 - trans;
        TV v1_prime = v1 - trans;
        TV v2_prime = v2 - trans;

        TV face_normal = -(v2_prime - v0_prime).normalized().cross(
            (v1_prime-v0_prime).normalized()).normalized();

        Matrix<T, 3, 3> R = Eigen::Quaternion<T>().setFromTwoVectors(face_normal, TV(0, 0, 1)).toRotationMatrix();

        v0_prime = R * v0_prime;
        v1_prime = R * v1_prime;
        v2_prime = R * v2_prime;

        TV2 v0_proj = v0_prime.segment<2>(0);
        TV2 v1_proj = v1_prime.segment<2>(0);
        TV2 v2_proj = v2_prime.segment<2>(0);
        MatrixXT prism(6, 3); prism.setZero();
        Eigen::MatrixXi prism_faces(8, 3); prism_faces.setZero();
        if (face_data.site_indices.size() != face_data.distances.size())
            std::exit(0);
        for (int i = 0; i < face_data.site_indices.size(); i++)
        {
            SurfacePoint pt = samples[face_data.site_indices[i]];
            TV site_location = toTV(pt.interpolate(geometry->vertexPositions));
            
            // initialize the prism
            if (i == 0)
            {
                prism_faces.row(0) = IV(1, 0, 2);
                prism_faces.row(1) = IV(3, 4, 5);
                prism_faces.row(2) = IV(3, 0, 4);
                prism_faces.row(3) = IV(4, 0, 1);
                prism_faces.row(4) = IV(5, 4, 1);
                prism_faces.row(5) = IV(5, 1, 2);
                prism_faces.row(6) = IV(5, 2, 3);
                prism_faces.row(7) = IV(3, 2, 0);
                prism.row(0) = v0_prime + TV(0, 0, -max_h); 
                prism.row(1) = v1_prime + TV(0, 0, -max_h); 
                prism.row(2) = v2_prime + TV(0, 0, -max_h);
                prism.row(3) = prism.row(0) + TV(0, 0, 2.0 * max_h).transpose();
                prism.row(4) = prism.row(1) + TV(0, 0, 2.0 * max_h).transpose();
                prism.row(5) = prism.row(2) + TV(0, 0, 2.0 * max_h).transpose();
            }    
            TV current_distance = face_data.distances[i];
            // lifting
            TV v0_prime_lifted = TV(v0_prime[0], v0_prime[1], std::pow(current_distance[0], 2));
            TV v1_prime_lifted = TV(v1_prime[0], v1_prime[1], std::pow(current_distance[1], 2));
            TV v2_prime_lifted = TV(v2_prime[0], v2_prime[1], std::pow(current_distance[2], 2));

            TV normal = -((v2_prime_lifted - v0_prime_lifted).normalized().cross((v1_prime_lifted-v0_prime_lifted).normalized())).normalized();
            
            if (normal.dot(TV(0, 0, 1)) < 0)
                normal *= -1.0;

            // igl::writeOBJ("plane.obj", plane_v, plane_f);

            TV point = (v0_prime_lifted + v1_prime_lifted + v2_prime_lifted) / 3.0;
            Eigen::Matrix<CGAL::Epeck::FT,8,3> BV;
            Eigen::Matrix<int,12,3> BF;
            // std::cout << "half box" << std::endl;
            igl::copyleft::cgal::half_space_box(point, normal, prism,BV,BF);
            // std::cout << "half box dpme" << std::endl;
            MatrixXT box(8, 3);
            for (int i = 0; i < 8; i++)
            {
                
                box.row(i) = TV(BV(i, 0).exact().convert_to<T>(), 
                    BV(i, 1).exact().convert_to<T>(), 
                    BV(i, 2).exact().convert_to<T>());
            }
            
            // igl::writeOBJ("box.obj", box, BF);
            // igl::copyleft::cgal::intersect_with_half_space(prism, prism_faces, 
            //             point, normal, prism, prism_faces, J);
            // START_TIMING(boolean)
            // std::cout << "boolean" << std::endl;
            MatrixXT new_vertices;
            MatrixXi new_faces;
            bool succeed = igl::copyleft::cgal::mesh_boolean(BV, BF, prism, prism_faces, 
                igl::MESH_BOOLEAN_TYPE_INTERSECT, new_vertices, new_faces);
            prism = new_vertices;
            prism_faces = new_faces;
            // std::cout << "succeed " << succeed << std::endl;

            // FINISH_TIMING_PRINT(boolean)
            // igl::writeOBJ("after.obj", prism, prism_faces);
            // std::cout << "after" << std::endl;
            // std::exit(0);
            if (!prism.rows() || !prism_faces.rows())
                break;
        }
        // std::getchar();
        // std::cout << "intersection done" << std::endl;
        if (!prism.rows() || !prism_faces.rows())
#ifdef PARALLEL
            return;
#else
            continue;
#endif
            
        Eigen::MatrixXi feature_edges;
        // START_TIMING(extract_feature)
        // std::cout << "extract_feature" << std::endl;
        igl::copyleft::cgal::extract_feature(prism, prism_faces, 1e-4, feature_edges);
        // FINISH_TIMING_PRINT(extract_feature)
        // std::cout << "extract_feature done" << std::endl;
        // remove middle vertices

        auto getNonZeroEntry = [&](const Eigen::VectorXi& vec)
        {
            std::vector<int> nz;
            for (int i = 0; i < vec.rows(); i++)
                if (vec[i] == 1)
                    nz.push_back(i);
            return nz;
        };

        MatrixXi adj(prism.rows(), prism.rows()); adj.setZero();
        // std::cout << feature_edges.rows() << " " << feature_edges.cols() << std::endl;
        for (int i = 0; i < feature_edges.rows(); i++)
        {
            // std::cout << feature_edges.row(i) << std::endl;
            adj(feature_edges(i, 0), feature_edges(i, 1)) = 1;
            adj(feature_edges(i, 1), feature_edges(i, 0)) = 1;
        }
        // igl::writeOBJ("cut"+std::to_string(face_idx)+".obj", prism, prism_faces);
        bool remove_vtx = true;
        std::vector<std::pair<int, int>> long_edges;
        // START_TIMING(remove)
        while (remove_vtx)
        {
            for (int i = 0; i < prism.rows(); i++)
            {
                TV vi = prism.row(i);
                std::vector<int> nz = getNonZeroEntry(adj.col(i));
                if (nz.size() == 2)
                {
                    TV vj = prism.row(nz[0]);
                    TV vk = prism.row(nz[1]);
                    TV e0 = (vj - vi).normalized();
                    TV e1 = (vk - vj).normalized();
                    if (std::abs(e1.dot(e0)) > 1.0 - EPSILON) // colinear
                    {
                        // std::cout << "colinear" << std::endl;
                        remove_vtx = true;
                        adj.col(i).setZero();
                        adj.row(i).setZero();
                        adj(nz[0], nz[1]) = 1;
                        adj(nz[1], nz[0]) = 1;
                        break;
                    }
                }
                remove_vtx = false;
            }
        }
        // FINISH_TIMING_PRINT(remove)
        // std::cout << "removing done " << std::endl;
        for (int i = 0; i < prism.rows(); i++)
        {
            std::vector<int> nz = getNonZeroEntry(adj.col(i));
            if (nz.size())
            {
                for (int j = 0; j < nz.size(); j++)
                {
                    // out << "l " << i + 1 << " " << nz[j] + 1 << std::endl;
                    long_edges.push_back(std::make_pair(i, nz[j]));
                    adj(i, nz[j]) = 0;
                    adj(nz[j], i) = 0;
                }
            }
        }
        // std::ofstream out("cut_edges"+std::to_string(face_idx)+".obj");
        
        T max_h_remaining = -1e10;

        for (int i = 0; i < prism.rows(); i++)
        {
            if (prism(i, 2) > max_h_remaining)
                max_h_remaining = prism(i, 2);
        }
        // std::cout << "extract long edges done" << std::endl;
        std::vector<bool> visited(prism.rows(), false);
        std::vector<SurfacePoint> ixn_in_triangle;
        for (auto pair : long_edges)
        {
            TV edge_vtx0 = prism.row(pair.first);
            TV edge_vtx1 = prism.row(pair.second);
            TV2 edge_vtx0_proj = edge_vtx0.segment<2>(0);
            TV2 edge_vtx1_proj = edge_vtx1.segment<2>(0);
            bool vtx0_exists = check_if_existing_projection(v0_proj, v1_proj, v2_proj, edge_vtx0_proj);
            bool vtx1_exists = check_if_existing_projection(v0_proj, v1_proj, v2_proj, edge_vtx1_proj);
            
            if (vtx0_exists || vtx1_exists)
                // not the highest line
                if (std::abs(edge_vtx0[2] - max_h_remaining) > EPSILON 
                    || std::abs(edge_vtx1[2] - max_h_remaining) > EPSILON)
                    continue;


            TV2 edge_vec = (edge_vtx1_proj - edge_vtx0_proj).normalized();
            
            // colinear with triangle edges and on the edge
            bool invalid = std::abs((v1_proj - v0_proj).normalized().dot(edge_vec)) > 1.0 - EPSILON 
                && std::abs((edge_vtx1_proj - v1_proj).normalized().dot(edge_vec)) > 1.0 - EPSILON;
            invalid |= std::abs((v1_proj - v2_proj).normalized().dot(edge_vec)) > 1.0 - EPSILON &&
                std::abs((edge_vtx1_proj - v1_proj).normalized().dot(edge_vec)) > 1.0 - EPSILON;
            invalid |= std::abs((v2_proj - v0_proj).normalized().dot(edge_vec)) > 1.0 - EPSILON &&
                std::abs((edge_vtx1_proj - v2_proj).normalized().dot(edge_vec)) > 1.0 - EPSILON;
            // std::cout << "check invalid done" << std::endl;

            
            if (invalid)
                continue;
            
            // compute in-triangle projection
            TV vtx0_proj = edge_vtx0; vtx0_proj[2] = 0.0;
            TV vtx1_proj = edge_vtx1; vtx1_proj[2] = 0.0;
            vtx0_proj = (R.transpose() * vtx0_proj + trans);
            vtx1_proj = (R.transpose() * vtx1_proj + trans);

            // if (visited[pair.first])
            //     ixn_in_triangle

            // edges_thread[face_idx].push_back(std::make_pair(vtx0_proj, vtx1_proj));     
            
            auto check_coplanar = [&](const TV& vi)
            {
                std::vector<int> connecting_sites;
                for (int i = 0; i < face_data.site_indices.size(); i++)
                {
                    MatrixXT testing_vertices(3, 3);
                    TV current_distance = face_data.distances[i];
                    // testing_vertices.row(0) = TV(v0_prime[0], v0_prime[1], std::pow(current_distance[0], 2));
                    // testing_vertices.row(1) = TV(v1_prime[0], v1_prime[1], std::pow(current_distance[1], 2));
                    // testing_vertices.row(2) = TV(v2_prime[0], v2_prime[1], std::pow(current_distance[2], 2));
                    
                    // int idx = 0; T dis_square; 
                    // // TV closest_point;
                    // std::cout << "point_simplex_squared_distance" << std::endl;
                    // MatrixXT query_point(1, 3); query_point.row(0) = vi;
                    // MatrixXi triangle_faces(1, 3); triangle_faces.row(0) = IV(0, 1, 2);
                    // TV closest_point;
                    // // igl::point_simplex_squared_distance<3>(query_point, testing_vertices, 
                    // //     triangle_faces, idx, dis_square, closest_point);
                    // Eigen::VectorXi I;
                    // Eigen::MatrixXd C;
                    // Eigen::VectorXd sqrD;
                    // igl::point_mesh_squared_distance(query_point, testing_vertices, triangle_faces, sqrD, I, C);
                    // std::cout << "point_simplex_squared_distance done" << std::endl;
                    // dis_square = sqrD[0];

                    TV v0_prime_lifted = TV(v0_prime[0], v0_prime[1], std::pow(current_distance[0], 2));
                    TV v1_prime_lifted = TV(v1_prime[0], v1_prime[1], std::pow(current_distance[1], 2));
                    TV v2_prime_lifted = TV(v2_prime[0], v2_prime[1], std::pow(current_distance[2], 2));

                    TV normal = -((v2_prime_lifted - v0_prime_lifted).normalized().cross((v1_prime_lifted-v0_prime_lifted).normalized())).normalized();
                    TV v = vi - v0_prime_lifted;
                    T dis_square = std::abs(v.dot(normal));
                    if (dis_square < EPSILON)
                    {
                        // std::cout << "coplanar" << std::endl;
                        connecting_sites.push_back(face_data.site_indices[i]);
                    }
                }
                return connecting_sites;
            };
            // std::cout << "check # connection" << std::endl;
            std::vector<int> connecting_sites_v0 = check_coplanar(edge_vtx0);
            std::vector<int> connecting_sites_v1 = check_coplanar(edge_vtx1);
            TV bary0 = computeBarycentricCoordinates(vtx0_proj, v0, v1, v2);
            TV bary1 = computeBarycentricCoordinates(vtx1_proj, v0, v1, v2);
            // std::cout << bary0.transpose() << " " << bary1.transpose() << std::endl;
            SurfacePoint p0 = SurfacePoint(mesh->face(face_idx), Vector3{bary0[0], bary0[1], bary0[2]});
            SurfacePoint p1 = SurfacePoint(mesh->face(face_idx), Vector3{bary1[0], bary1[1], bary1[2]});
            // std::cout << "done" << std::endl;
            ixn_data_thread[face_idx].push_back(std::make_pair(p0, connecting_sites_v0));
            ixn_data_thread[face_idx].push_back(std::make_pair(p1, connecting_sites_v1));

            // edge_graph_thread[face_idx].push_back(IV2());

            // edges.push_back(std::make_pair(vtx0_proj, vtx1_proj));
            // std::cout << "push back done" << std::endl;
            // for (int i = 0; i < face_data.site_indices.size(); i++)
            // {
            //     SurfacePoint pt = samples[face_data.site_indices[i]];
            //     TV site_location = toTV(pt.interpolate(geometry->vertexPositions));
            //     T left = (vtx0_proj - site_location).norm();
            //     T right = (vtx1_proj - site_location).norm();
            //     std::cout << (left - right) / right * 100.0 << std::endl;
            // }
        }

        
        // FINISH_TIMING_PRINT(face)
        
    }
#ifdef PARALLEL
            );
#endif
    
    
    // for (const std::vector<std::pair<TV, TV>>& data : edges_thread)
    // {
    //     if (data.size())
    //     {
    //         for (const std::pair<TV, TV>& edge : data)
    //             edges.push_back(edge);
    //     }
    // }
    for (const auto& data : ixn_data_thread)
    {
        if (data.size())
        {
            for (const auto& pair : data)
                ixn_data.push_back(pair);
        }
    }
    
}



void VoronoiCells::intersectPrism(std::vector<SurfacePoint>& samples,
            std::vector<FaceData>& source_data, 
            std::vector<std::pair<TV, TV>>& edges, int face_idx)
{
    T max_h = 0.2;
    T EPSILON = 1e-12;
    auto check_if_existing_projection = [&](const TV2& v0, 
        const TV2& v1, const TV2& v2, const TV2& pt) -> bool
    {
        if ((pt - v0).norm() < EPSILON)
            return true;
        if ((pt - v1).norm() < EPSILON)
            return true;
        if ((pt - v2).norm() < EPSILON)
            return true;
        return false;
    };
    
    
    FaceData face_data = source_data[face_idx];
    // don't consider 1 vertex case
    if (face_data.site_indices.size() == 1)
        return;
    TV v0 = toTV(geometry->vertexPositions[mesh->face(face_idx).halfedge().vertex()]);
    TV v1 = toTV(geometry->vertexPositions[mesh->face(face_idx).halfedge().next().vertex()]);
    TV v2 = toTV(geometry->vertexPositions[mesh->face(face_idx).halfedge().next().next().vertex()]);


    TV trans = (v0 + v1 + v2) / 3.0;
    TV v0_prime = v0 - trans;
    TV v1_prime = v1 - trans;
    TV v2_prime = v2 - trans;

    TV face_normal = -(v2_prime - v0_prime).normalized().cross((v1_prime-v0_prime).normalized()).normalized();

    Matrix<T, 3, 3> R = Eigen::Quaternion<T>().setFromTwoVectors(face_normal, TV(0, 0, 1)).toRotationMatrix();

    v0_prime = R * v0_prime;
    v1_prime = R * v1_prime;
    v2_prime = R * v2_prime;
    TV2 v0_proj = v0_prime.segment<2>(0);
    TV2 v1_proj = v1_prime.segment<2>(0);
    TV2 v2_proj = v2_prime.segment<2>(0);

    MatrixXT prism(6, 3);
    Eigen::MatrixXi prism_faces(8, 3);
    MatrixXT J;
    for (int i = 0; i < face_data.site_indices.size(); i++)
    {
        SurfacePoint pt = samples[face_data.site_indices[i]];
        TV site_location = toTV(pt.interpolate(geometry->vertexPositions));
        
        if (i == 0)
        {
            prism_faces.row(0) = IV(1, 0, 2);
            prism_faces.row(1) = IV(3, 4, 5);
            prism_faces.row(2) = IV(3, 0, 4);
            prism_faces.row(3) = IV(4, 0, 1);
            prism_faces.row(4) = IV(5, 4, 1);
            prism_faces.row(5) = IV(5, 1, 2);
            prism_faces.row(6) = IV(5, 2, 3);
            prism_faces.row(7) = IV(3, 2, 0);
            prism.row(0) = v0_prime + TV(0, 0, -max_h); 
            prism.row(1) = v1_prime + TV(0, 0, -max_h); 
            prism.row(2) = v2_prime + TV(0, 0, -max_h);
            prism.row(3) = prism.row(0) + TV(0, 0, 2.0 * max_h).transpose();
            prism.row(4) = prism.row(1) + TV(0, 0, 2.0 * max_h).transpose();
            prism.row(5) = prism.row(2) + TV(0, 0, 2.0 * max_h).transpose();
        }    

        TV v0_prime_lifted = TV(v0_prime[0], v0_prime[1], (site_location - v0).squaredNorm());
        TV v1_prime_lifted = TV(v1_prime[0], v1_prime[1], (site_location - v1).squaredNorm());
        TV v2_prime_lifted = TV(v2_prime[0], v2_prime[1], (site_location - v2).squaredNorm());

        TV normal = -(v2_prime_lifted - v0_prime_lifted).normalized().cross((v1_prime_lifted-v0_prime_lifted).normalized());
        std::cout << normal.transpose() << std::endl;
        TV point = (v0_prime_lifted + v1_prime_lifted + v2_prime_lifted) / 3.0;
        Eigen::Matrix<CGAL::Epeck::FT,8,3> BV;
        Eigen::Matrix<int,12,3> BF;
        // typedef CGAL::Plane_3<CGAL::Epeck> Plane;
        // typedef CGAL::Point_3<CGAL::Epeck> Point;
        // typedef CGAL::Vector_3<CGAL::Epeck> Vector;
        // Plane P(Point(point(0),point(1),point(2)),Vector(normal(0),normal(1),normal(2)));
        // igl::copyleft::cgal::half_space_box(P,prism,BV,BF);
        igl::copyleft::cgal::half_space_box(point, normal, prism,BV,BF);
        MatrixXT box(8, 3);
        for (int i = 0; i < 8; i++)
        {
            
            box.row(i) = TV(BV(i, 0).exact().convert_to<T>(), 
                BV(i, 1).exact().convert_to<T>(), 
                BV(i, 2).exact().convert_to<T>());
        }
        igl::writeOBJ("box.obj", box, BF);
        
        bool succeed = igl::copyleft::cgal::mesh_boolean(BV, BF, prism, prism_faces, 
            igl::MESH_BOOLEAN_TYPE_INTERSECT, prism, prism_faces);
        igl::writeOBJ("after.obj", prism, prism_faces);
        std::cout << "after" << std::endl;
        std::getchar();
    }
    Eigen::MatrixXi feature_edges;
    igl::copyleft::cgal::extract_feature(prism, prism_faces, 1e-4, feature_edges);
    // remove middle vertices

    auto getNonZeroEntry = [&](const Eigen::VectorXi& vec)
    {
        std::vector<int> nz;
        for (int i = 0; i < vec.rows(); i++)
            if (vec[i] == 1)
                nz.push_back(i);
        return nz;
    };

    MatrixXi adj(prism.rows(), prism.rows()); adj.setZero();
    // std::cout << feature_edges.rows() << " " << feature_edges.cols() << std::endl;
    for (int i = 0; i < feature_edges.rows(); i++)
    {
        // std::cout << feature_edges.row(i) << std::endl;
        adj(feature_edges(i, 0), feature_edges(i, 1)) = 1;
        adj(feature_edges(i, 1), feature_edges(i, 0)) = 1;
    }
    igl::writeOBJ("cut"+std::to_string(face_idx)+".obj", prism, prism_faces);
    if (!prism.rows() || !prism_faces.rows())
        return;
    bool remove_vtx = true;
    std::vector<std::pair<int, int>> long_edges;
    while (remove_vtx)
    {
        for (int i = 0; i < prism.rows(); i++)
        {
            TV vi = prism.row(i);
            std::vector<int> nz = getNonZeroEntry(adj.col(i));
            if (nz.size() == 2)
            {
                TV vj = prism.row(nz[0]);
                TV vk = prism.row(nz[1]);
                TV e0 = (vj - vi).normalized();
                TV e1 = (vk - vj).normalized();
                if (std::abs(e1.dot(e0)) > 1.0 - EPSILON) // colinear
                {
                    // std::cout << "colinear" << std::endl;
                    remove_vtx = true;
                    adj.col(i).setZero();
                    adj.row(i).setZero();
                    adj(nz[0], nz[1]) = 1;
                    adj(nz[1], nz[0]) = 1;
                    break;
                }
            }
            remove_vtx = false;
        }
    }
    
    for (int i = 0; i < prism.rows(); i++)
    {
        std::vector<int> nz = getNonZeroEntry(adj.col(i));
        if (nz.size())
        {
            for (int j = 0; j < nz.size(); j++)
            {
                // out << "l " << i + 1 << " " << nz[j] + 1 << std::endl;
                long_edges.push_back(std::make_pair(i, nz[j]));
                adj(i, nz[j]) = 0;
                adj(nz[j], i) = 0;
            }
        }
    }
    std::ofstream out("cut_edges"+std::to_string(face_idx)+".obj");
    T max_h_remaining = -1e10;

    for (int i = 0; i < prism.rows(); i++)
    {
        if (prism(i, 2) > max_h_remaining)
            max_h_remaining = prism(i, 2);
    }

    for (auto pair : long_edges)
    {
        TV edge_vtx0 = prism.row(pair.first);
        TV edge_vtx1 = prism.row(pair.second);
        TV2 edge_vtx0_proj = edge_vtx0.segment<2>(0);
        TV2 edge_vtx1_proj = edge_vtx1.segment<2>(0);
        bool vtx0_exists = check_if_existing_projection(v0_proj, v1_proj, v2_proj, edge_vtx0_proj);
        bool vtx1_exists = check_if_existing_projection(v0_proj, v1_proj, v2_proj, edge_vtx1_proj);

        if (vtx0_exists || vtx1_exists)
            if (std::abs(edge_vtx0[2] - max_h_remaining) > EPSILON 
                || std::abs(edge_vtx1[2] - max_h_remaining) > EPSILON)
                continue;

        TV2 edge_vec = (edge_vtx1_proj - edge_vtx0_proj).normalized();
        bool invalid = std::abs((v1_proj - v0_proj).normalized().dot(edge_vec)) > 1.0 - EPSILON 
            && std::abs((edge_vtx1_proj - v1_proj).normalized().dot(edge_vec)) > 1.0 - EPSILON;
        invalid |= std::abs((v1_proj - v2_proj).normalized().dot(edge_vec)) > 1.0 - EPSILON &&
            std::abs((edge_vtx1_proj - v1_proj).normalized().dot(edge_vec)) > 1.0 - EPSILON;
        invalid |= std::abs((v2_proj - v0_proj).normalized().dot(edge_vec)) > 1.0 - EPSILON &&
            std::abs((edge_vtx1_proj - v2_proj).normalized().dot(edge_vec)) > 1.0 - EPSILON;

        // TV edge_vec = (edge_vtx1 - edge_vtx0).normalized();
        // TV2 edge_vec2d = (edge_vtx1_proj - edge_vtx0_proj).normalized();
        // bool invalid = std::abs((v1_prime - v0_prime).normalized().dot(edge_vec)) > 1.0 - EPSILON 
        //     && std::abs((edge_vtx1_proj - v1_proj).normalized().dot(edge_vec2d)) > 1.0 - EPSILON;
        // invalid |= std::abs((v1_prime - v2_prime).normalized().dot(edge_vec)) > 1.0 - EPSILON &&
        //     std::abs((edge_vtx1_proj - v1_proj).normalized().dot(edge_vec2d)) > 1.0 - EPSILON;
        // invalid |= std::abs((v2_prime - v0_prime).normalized().dot(edge_vec)) > 1.0 - EPSILON &&
        //     std::abs((edge_vtx1_proj - v2_proj).normalized().dot(edge_vec2d)) > 1.0 - EPSILON;

        if (invalid)
        {
            // if (std::abs(edge_vtx0[2] - max_h_remaining) < EPSILON
            //     && std::abs(edge_vtx1[2] - max_h_remaining) < EPSILON)
            //     invalid = false;
        }
        
        if (invalid)
            continue;
        

        TV vtx0_proj = edge_vtx0; vtx0_proj[2] = 0.0;
        TV vtx1_proj = edge_vtx1; vtx1_proj[2] = 0.0;
        vtx0_proj = (R.transpose() * vtx0_proj + trans);
        vtx1_proj = (R.transpose() * vtx1_proj + trans);
        edges.push_back(std::make_pair(edge_vtx0, edge_vtx1));
    }

    for (int i = 0; i < edges.size(); i++)
    {
        out << "v " << edges[i].first.transpose() << std::endl;
        out << "v " << edges[i].second.transpose() << std::endl;
    }
    for (int i = 0; i < edges.size(); i++)
    {
        out << "l " << i*2 + 1 << " " << i*2 + 2 << std::endl;
    }
   
    out.close();

}

void VoronoiCells::saveVoronoiDiagram()
{
    std::ofstream out("samples.txt");
    out << samples.size() << std::endl;
    for (SurfacePoint& pt : samples)
    {
        pt = pt.inSomeFace();
        out << std::setprecision(16);
        out << pt.face.getIndex() << " " << pt.faceCoords.x << " " << 
            pt.faceCoords.y << " " << pt.faceCoords.z << std::endl;
    }
    out.close();
}

void VoronoiCells::loadGeometry()
{
    MatrixXT V; MatrixXi F;
    // igl::readOBJ("/home/yueli/Documents/ETH/WuKong/Projects/VoronoiCells/data/grid.obj", V, F);
    igl::readOBJ("/home/yueli/Documents/ETH/WuKong/Projects/VoronoiCells/data/sphere642.obj", V, F);
    MatrixXT N;
    igl::per_vertex_normals(V, F, N);


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

    
}

T VoronoiCells::computeDistanceMatchingEnergy(const std::vector<int>& site_indices, 
    SurfacePoint& xi_current)
{
    T energy = 0.0;
    TV current = toTV(xi_current.interpolate(geometry->vertexPositions));
    TV site0_location = toTV(samples[site_indices[0]].interpolate(geometry->vertexPositions));
    T dis_to_site0;
    std::vector<IxnData> ixn_data_site;
    std::vector<SurfacePoint> path; 
    if (metric == Euclidean)
        dis_to_site0 = (current - site0_location).norm();
    else
        computeGeodesicDistance(samples[site_indices[0]], xi_current, 
            dis_to_site0, path, ixn_data_site, false);
    
    for (int j = 1; j < site_indices.size(); j++)
    {
        int site_idx = site_indices[j];
        
        TV site_location = toTV(samples[site_indices[j]].interpolate(geometry->vertexPositions));
        T dis_to_site;
        if (metric == Euclidean)
            dis_to_site = (current - site_location).norm();
        else
            computeGeodesicDistance(samples[site_idx], xi_current, dis_to_site,
                path, ixn_data_site, false);
        energy += 0.5 * std::pow(dis_to_site - dis_to_site0, 2);
    }
    return energy;
}

T VoronoiCells::computeDistanceMatchingGradient(const std::vector<int>& site_indices, 
    SurfacePoint& xi_current, TV2& grad, T& energy)
{
    grad = TV2::Zero();
    energy = 0.0;
    TV current = toTV(xi_current.interpolate(geometry->vertexPositions));
    TV site0_location = toTV(samples[site_indices[0]].interpolate(geometry->vertexPositions));
    TV v0 = toTV(geometry->vertexPositions[xi_current.face.halfedge().vertex()]);
    TV v1 = toTV(geometry->vertexPositions[xi_current.face.halfedge().next().vertex()]);
    TV v2 = toTV(geometry->vertexPositions[xi_current.face.halfedge().next().next().vertex()]);

    // std::cout << "face# " << fi.getIndex() << " xi " << current.transpose() << std::endl;
    TV bary = toTV(xi_current.faceCoords);
    TV pt = v0 * bary[0] + v1 * bary[1] + v2 * bary[2];
    // std::cout << pt.transpose() << " " << current.transpose() << std::endl;
    // std::getchar();

    // std::cout << v0.transpose() << " " << v1.transpose() << " " << v2.transpose() << std::endl;
    T dis_to_site0;
    Matrix<T, 3, 2> dxdw; 
    dxdw.col(0) = v0 - v2;
    dxdw.col(1) = v1 - v2;
    
    TV dldx0;
    if (metric == Euclidean)
    {
        dis_to_site0 = (current - site0_location).norm();
        dldx0 = (current - site0_location).normalized();
    }
    else
    {
        std::vector<IxnData> ixn_data_site;
        std::vector<SurfacePoint> path;
        computeGeodesicDistance(samples[site_indices[0]], xi_current, dis_to_site0, path, 
            ixn_data_site, true);
        int length = path.size();
        
        TV vtx0 = toTV(path[0].interpolate(geometry->vertexPositions));
        TV vtx1 = toTV(path[length - 1].interpolate(geometry->vertexPositions));
        if (length == 2)
        {
            dldx0 = (vtx1 - vtx0).normalized();
        }
        else
        {
            TV ixn1 = toTV(path[length - 2].interpolate(geometry->vertexPositions));
            dldx0 = -(ixn1 - vtx1).normalized();
        }
    }
    
    for (int j = 1; j < site_indices.size(); j++)
    {
        int site_idx = site_indices[j];
        TV site_location = toTV(samples[site_indices[j]].interpolate(geometry->vertexPositions));
        T dis_to_site;
        TV dldxj;
        if (metric == Euclidean)
        {
            dis_to_site = (current - site_location).norm();
            dldxj = (current - site_location).normalized();
        }
        else
        {
            std::vector<IxnData> ixn_data_site;
            std::vector<SurfacePoint> path; 

            computeGeodesicDistance(samples[site_idx], xi_current, dis_to_site,
                path, ixn_data_site, true);

            int length = path.size();
        
            TV vtx0 = toTV(path[0].interpolate(geometry->vertexPositions));
            TV vtx1 = toTV(path[length - 1].interpolate(geometry->vertexPositions));
            if (length == 2)
            {
                dldxj = (vtx1 - vtx0).normalized();
            }
            else
            {
                TV ixn1 = toTV(path[length - 2].interpolate(geometry->vertexPositions));
                dldxj = -(ixn1 - vtx1).normalized();
            }
        }
        
        // std::cout << site0_location.transpose() << " " << site_location.transpose() << std::endl;
        // std::cout << "dis_to_site0 " << dis_to_site0 << " dis_to_site " << dis_to_site << std::endl;
        energy += 0.5 * std::pow(dis_to_site - dis_to_site0, 2); 

        T dOdl = (dis_to_site - dis_to_site0); 
        
        grad += dOdl * -dxdw.transpose() * dldx0;
        grad += dOdl * dxdw.transpose() * dldxj;
    }
    return grad.norm();
}

void VoronoiCells::computeDistanceMatchingHessian(const std::vector<int>& site_indices, 
        SurfacePoint& xi_current, TM2& hess)
{
    hess = TM2::Zero();
    
    TV current = toTV(xi_current.interpolate(geometry->vertexPositions));
    TV site0_location = toTV(samples[site_indices[0]].interpolate(geometry->vertexPositions));
    TV v0 = toTV(geometry->vertexPositions[xi_current.face.halfedge().vertex()]);
    TV v1 = toTV(geometry->vertexPositions[xi_current.face.halfedge().next().vertex()]);
    TV v2 = toTV(geometry->vertexPositions[xi_current.face.halfedge().next().next().vertex()]);

    T dis_to_site0;
    Matrix<T, 3, 2> dxdw; 
    dxdw.col(0) = v0 - v2;
    dxdw.col(1) = v1 - v2;
    
    TV dldx0;
    TM d2ldx02;
    if (metric == Euclidean)
    {
        dis_to_site0 = (current - site0_location).norm();
        dldx0 = (current - site0_location).normalized();
        d2ldx02 = (TM::Identity() - dldx0 * dldx0.transpose()) / dis_to_site0;
    }
    else
    {
        std::vector<IxnData> ixn_data_site;
        std::vector<SurfacePoint> path;
        computeGeodesicDistance(samples[site_indices[0]], xi_current, dis_to_site0, path, 
            ixn_data_site, true);
        int length = path.size();
        
        TV vtx0 = toTV(path[0].interpolate(geometry->vertexPositions));
        TV vtx1 = toTV(path[length - 1].interpolate(geometry->vertexPositions));
        if (length == 2)
        {
            dldx0 = (vtx1 - vtx0).normalized();
            d2ldx02 = (TM::Identity() - dldx0 * dldx0.transpose()) / dis_to_site0;
        }
        else
        {
            TV ixn1 = toTV(path[length - 2].interpolate(geometry->vertexPositions));
            dldx0 = -(ixn1 - vtx1).normalized();
        }
    }
    
    for (int j = 1; j < site_indices.size(); j++)
    {
        int site_idx = site_indices[j];
        TV site_location = toTV(samples[site_indices[j]].interpolate(geometry->vertexPositions));
        T dis_to_site;
        TV dldxj;
        TM d2ldxj2;
        if (metric == Euclidean)
        {
            dis_to_site = (current - site_location).norm();
            dldxj = (current - site_location).normalized();
            d2ldxj2 = (TM::Identity() - dldxj * dldxj.transpose()) / dis_to_site;
        }
        else
        {
            std::vector<IxnData> ixn_data_site;
            std::vector<SurfacePoint> path; 

            computeGeodesicDistance(samples[site_idx], xi_current, dis_to_site,
                path, ixn_data_site, true);

            int length = path.size();
        
            TV vtx0 = toTV(path[0].interpolate(geometry->vertexPositions));
            TV vtx1 = toTV(path[length - 1].interpolate(geometry->vertexPositions));
            if (length == 2)
            {
                dldxj = (vtx1 - vtx0).normalized();
            }
            else
            {
                TV ixn1 = toTV(path[length - 2].interpolate(geometry->vertexPositions));
                dldxj = -(ixn1 - vtx1).normalized();

                int ixn_dof = (length - 2) * 3;
                MatrixXT dfdc(ixn_dof, 3); dfdc.setZero();
                MatrixXT dfdx(ixn_dof, ixn_dof); dfdx.setZero();
                MatrixXT dxdt(ixn_dof, length-2); dxdt.setZero();
                MatrixXT d2gdcdx(ixn_dof, 3); d2gdcdx.setZero();

                for (int ixn_id = 0; ixn_id < length - 3; ixn_id++)
                {
                    // std::cout << "inside" << std::endl;
                    Matrix<T, 6, 6> hess;
                    TV ixn_i = toTV(path[1 + ixn_id].interpolate(geometry->vertexPositions));
                    TV ixn_j = toTV(path[1 + ixn_id+1].interpolate(geometry->vertexPositions));

                    edgeLengthHessian(ixn_i, ixn_j, hess);
                    dfdx.block(ixn_id*3, ixn_id * 3, 6, 6) += hess;
                }
                for (int ixn_id = 0; ixn_id < length - 2; ixn_id++)
                {
                    TV x_start = ixn_data_site[1+ixn_id].start;
                    TV x_end = ixn_data_site[1+ixn_id].end;
                    dxdt.block(ixn_id * 3, ixn_id, 3, 1) = (x_end - x_start);
                }
                TM dlndxn = (TM::Identity() - dldxj * dldxj.transpose()) / (ixn1 - vtx1).norm();
                dfdx.block(ixn_dof-3, ixn_dof-3, 3, 3) += dlndxn;
                dfdc.block(ixn_dof-3, 0, 3, 3) += -dlndxn;
                d2gdcdx.block(ixn_dof-3, 0, 3, 3) += -dlndxn;

                MatrixXT dxdtd2gdxdc = dxdt.transpose() * d2gdcdx;
                MatrixXT dxdtd2gdx2dxdt = dxdt.transpose() * dfdx * dxdt;
                MatrixXT dtdc = dxdtd2gdx2dxdt.colPivHouseholderQr().solve(-dxdtd2gdxdc);


                TM d2gdc2 = dlndxn;

                d2ldxj2 = d2gdc2 + d2gdcdx.transpose() * dxdt * dtdc;
            }
        }
        
        // std::cout << site0_location.transpose() << " " << site_location.transpose() << std::endl;
        // std::cout << "dis_to_site0 " << dis_to_site0 << " dis_to_site " << dis_to_site << std::endl;

        T dOdl = (dis_to_site - dis_to_site0); 
        TV2 dldw = dxdw.transpose() * (dldxj - dldx0);

        // grad += dOdl * dldw
        // hess += dOdl * d2ldw2 + dldw^T d2Odl2 * dldw 
        hess += dldw * dldw.transpose();
    }
    
}

T VoronoiCells::computeDistanceMatchingEnergyGradientHessian(const std::vector<int>& site_indices, 
        SurfacePoint& xi_current, TM2& hess, TV2& grad, T& energy)
{
    hess = TM2::Zero();
    grad = TV2::Zero();
    energy = 0.0;

    TV current = toTV(xi_current.interpolate(geometry->vertexPositions));
    TV site0_location = toTV(samples[site_indices[0]].interpolate(geometry->vertexPositions));
    TV v0 = toTV(geometry->vertexPositions[xi_current.face.halfedge().vertex()]);
    TV v1 = toTV(geometry->vertexPositions[xi_current.face.halfedge().next().vertex()]);
    TV v2 = toTV(geometry->vertexPositions[xi_current.face.halfedge().next().next().vertex()]);

    T dis_to_site0;
    Matrix<T, 3, 2> dxdw; 
    dxdw.col(0) = v0 - v2;
    dxdw.col(1) = v1 - v2;
    
    TV dldx0;
    TM d2ldx02;
    if (metric == Euclidean)
    {
        dis_to_site0 = (current - site0_location).norm();
        dldx0 = (current - site0_location).normalized();
        d2ldx02 = (TM::Identity() - dldx0 * dldx0.transpose()) / dis_to_site0;
    }
    else
    {
        std::vector<IxnData> ixn_data_site;
        std::vector<SurfacePoint> path;
        computeGeodesicDistance(samples[site_indices[0]], xi_current, dis_to_site0, path, 
            ixn_data_site, true);
        int length = path.size();
        
        TV vtx0 = toTV(path[0].interpolate(geometry->vertexPositions));
        TV vtx1 = toTV(path[length - 1].interpolate(geometry->vertexPositions));
        if (length == 2)
        {
            dldx0 = (vtx1 - vtx0).normalized();
            d2ldx02 = (TM::Identity() - dldx0 * dldx0.transpose()) / dis_to_site0;
        }
        else
        {
            TV ixn1 = toTV(path[length - 2].interpolate(geometry->vertexPositions));
            dldx0 = -(ixn1 - vtx1).normalized();
        }
    }
    
    for (int j = 1; j < site_indices.size(); j++)
    {
        int site_idx = site_indices[j];
        TV site_location = toTV(samples[site_indices[j]].interpolate(geometry->vertexPositions));
        T dis_to_site;
        TV dldxj;
        TM d2ldxj2;
        if (metric == Euclidean)
        {
            dis_to_site = (current - site_location).norm();
            dldxj = (current - site_location).normalized();
            d2ldxj2 = (TM::Identity() - dldxj * dldxj.transpose()) / dis_to_site;
        }
        else
        {
            std::vector<IxnData> ixn_data_site;
            std::vector<SurfacePoint> path; 

            computeGeodesicDistance(samples[site_idx], xi_current, dis_to_site,
                path, ixn_data_site, true);

            int length = path.size();
        
            TV vtx0 = toTV(path[0].interpolate(geometry->vertexPositions));
            TV vtx1 = toTV(path[length - 1].interpolate(geometry->vertexPositions));
            if (length == 2)
            {
                dldxj = (vtx1 - vtx0).normalized();
            }
            else
            {
                TV ixn1 = toTV(path[length - 2].interpolate(geometry->vertexPositions));
                dldxj = -(ixn1 - vtx1).normalized();

                int ixn_dof = (length - 2) * 3;
                MatrixXT dfdc(ixn_dof, 3); dfdc.setZero();
                MatrixXT dfdx(ixn_dof, ixn_dof); dfdx.setZero();
                MatrixXT dxdt(ixn_dof, length-2); dxdt.setZero();
                MatrixXT d2gdcdx(ixn_dof, 3); d2gdcdx.setZero();

                for (int ixn_id = 0; ixn_id < length - 3; ixn_id++)
                {
                    // std::cout << "inside" << std::endl;
                    Matrix<T, 6, 6> hess;
                    TV ixn_i = toTV(path[1 + ixn_id].interpolate(geometry->vertexPositions));
                    TV ixn_j = toTV(path[1 + ixn_id+1].interpolate(geometry->vertexPositions));

                    edgeLengthHessian(ixn_i, ixn_j, hess);
                    dfdx.block(ixn_id*3, ixn_id * 3, 6, 6) += hess;
                }
                for (int ixn_id = 0; ixn_id < length - 2; ixn_id++)
                {
                    TV x_start = ixn_data_site[1+ixn_id].start;
                    TV x_end = ixn_data_site[1+ixn_id].end;
                    dxdt.block(ixn_id * 3, ixn_id, 3, 1) = (x_end - x_start);
                }
                TM dlndxn = (TM::Identity() - dldxj * dldxj.transpose()) / (ixn1 - vtx1).norm();
                dfdx.block(ixn_dof-3, ixn_dof-3, 3, 3) += dlndxn;
                dfdc.block(ixn_dof-3, 0, 3, 3) += -dlndxn;
                d2gdcdx.block(ixn_dof-3, 0, 3, 3) += -dlndxn;

                MatrixXT dxdtd2gdxdc = dxdt.transpose() * d2gdcdx;
                MatrixXT dxdtd2gdx2dxdt = dxdt.transpose() * dfdx * dxdt;
                MatrixXT dtdc = dxdtd2gdx2dxdt.colPivHouseholderQr().solve(-dxdtd2gdxdc);


                TM d2gdc2 = dlndxn;

                d2ldxj2 = d2gdc2 + d2gdcdx.transpose() * dxdt * dtdc;
            }
        }
        
        // std::cout << site0_location.transpose() << " " << site_location.transpose() << std::endl;
        // std::cout << "dis_to_site0 " << dis_to_site0 << " dis_to_site " << dis_to_site << std::endl;

        T dOdl = (dis_to_site - dis_to_site0); 
        TV2 dldw = dxdw.transpose() * (dldxj - dldx0);

        energy += 0.5 * std::pow(dis_to_site - dis_to_site0, 2); 
        
        grad += dOdl * -dxdw.transpose() * dldx0;
        grad += dOdl * dxdw.transpose() * dldxj;
        hess += dldw * dldw.transpose();
    }
    return grad.norm();
}

void VoronoiCells::updateSurfacePoint(SurfacePoint& xi_current, const TV2& search_direction)
{
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


    // START_TIMING(contruct)
    auto lvals = gcs::makeManifoldSurfaceMeshAndGeometry(mesh_indices_gc, mesh_vertices_gc);
    // FINISH_TIMING_PRINT(contruct)
    std::unique_ptr<gcs::ManifoldSurfaceMesh> sub_mesh;
    std::unique_ptr<gcs::VertexPositionGeometry> sub_geometry;
    std::tie(sub_mesh, sub_geometry) = std::tuple<std::unique_ptr<gcs::ManifoldSurfaceMesh>,
                    std::unique_ptr<gcs::VertexPositionGeometry>>(std::move(std::get<0>(lvals)),  // mesh
                                                             std::move(std::get<1>(lvals))); // geometry

    gcFace face = sub_mesh->face(xi_current.face.getIndex());
    
    gcs::TraceOptions options; 
    options.includePath = true;
    Vector3 trace_vec{search_direction[0], search_direction[1], 
        0.0 - search_direction[0] - search_direction[1]};
    gcs::TraceGeodesicResult result = gcs::traceGeodesic(*sub_geometry, 
                                        face, 
                                        xi_current.faceCoords, 
                                        trace_vec, options);
    if (result.pathPoints.size() != 1)
    {
        SurfacePoint endpoint = result.endPoint.inSomeFace();
        xi_current = SurfacePoint(mesh->face(endpoint.face.getIndex()), endpoint.faceCoords);
    }
}

void VoronoiCells::optimizeForExactVD(std::vector<std::pair<SurfacePoint, std::vector<int>>>& ixn_data)
{
    int n_ixn = ixn_data.size();

#ifdef PARALLEL
    tbb::parallel_for(0, n_ixn, [&](int i)
#else
    for (int i = 0; i < n_ixn; i++)
#endif
    {
        
        SurfacePoint xi = ixn_data[i].first.inSomeFace();
        std::vector<int> site_indices = ixn_data[i].second;
    

        auto fdCheckGradient = [&]()
        {
            T eps = 1e-6;
            T E0, E1;
            TV2 grad;
            updateSurfacePoint(xi, TV2(-0.002, 0.004));
            SurfacePoint xi_tmp = xi;
            // std::cout << toTV(xi.interpolate(geometry->vertexPositions)).transpose() << std::endl;

            T g_norm = computeDistanceMatchingGradient(site_indices, xi, grad, E0);
            std::cout << "dgdw: " << grad.transpose() << std::endl;
            updateSurfacePoint(xi, TV2(eps, 0));
            E0 = computeDistanceMatchingEnergy(site_indices, xi);
            xi = xi_tmp;
            updateSurfacePoint(xi, TV2(-2.0 * eps, 0));
            E1 = computeDistanceMatchingEnergy(site_indices, xi);
            xi = xi_tmp;
            
            // std::cout << toTV(xi.interpolate(geometry->vertexPositions)).transpose() << std::endl;
            std::cout << "dgdw0 " << (E0 - E1) / eps / 2.0 << std::endl;
            updateSurfacePoint(xi, TV2(0.0, eps));
            E0 = computeDistanceMatchingEnergy(site_indices, xi);
            xi = xi_tmp;
            updateSurfacePoint(xi, TV2(0.0, -2.0 * eps));
            E1 = computeDistanceMatchingEnergy(site_indices, xi);
            xi = xi_tmp;
            std::cout << "dgdw1 " << (E0 - E1) / eps / 2.0 << std::endl;
            std::getchar();
        };

        auto fdCheckGradientScale = [&]()
        {
            T eps = 1e-4;
            T E0, E1;
            TV2 grad;
            updateSurfacePoint(xi, TV2(-0.002, 0.004));
            SurfacePoint xi_tmp = xi;
            // std::cout << toTV(xi.interpolate(geometry->vertexPositions)).transpose() << std::endl;

            T g_norm = computeDistanceMatchingGradient(site_indices, xi, grad, E0);
            TV2 dx;
            dx.setRandom();
            dx *= 1.0 / dx.norm();
            dx *= 0.01;
            
            T previous = 0.0;
            for (int i = 0; i < 10; i++)
            {
                xi = xi_tmp;
                updateSurfacePoint(xi, dx);
                T E1 = computeDistanceMatchingEnergy(site_indices, xi);
                T dE = E1 - E0;
                dE -= grad.dot(dx);
                // std::cout << "dE " << dE << std::endl;
                if (i > 0)
                {
                    std::cout << (previous/dE) << std::endl;
                }
                previous = dE;
                dx *= 0.5;
            }
            
            std::getchar();
        };

        // if (xi.type == gcs::SurfacePointType::Face)
        // {
        //     fdCheckGradientScale();
        // }

        T E0;
        TV2 grad;
        T tol = 1e-6;
        TM2 hess;
        int max_iter = 100;
        int iter = 0;
        while (true)
        {
            iter ++;
            if (iter > max_iter)
                break;
            // T g_norm = computeDistanceMatchingGradient(site_indices, xi, grad, E0);
            T g_norm = computeDistanceMatchingEnergyGradientHessian(site_indices, xi, hess, grad, E0);
            // std::getchar();
            T dis_error = std::sqrt(2.0*E0);
            if (g_norm < tol)
            {
                // std::cout << "|g|: " << g_norm << " obj: " << E0 << std::endl;
                break;
            }
            T alpha = 1.0;
            SurfacePoint xi_current = xi;
            // computeDistanceMatchingHessian(site_indices, xi, hess);
            TV2 dw = hess.colPivHouseholderQr().solve(-grad);
            
            for (int ls = 0; ls < 12; ls++)
            {
                xi = xi_current;
                updateSurfacePoint(xi, dw * alpha);
                T E1 = computeDistanceMatchingEnergy(site_indices, xi);
                // std::cout << "E0 " << E0 << " E1 " << E1 << std::endl;
                // std::getchar();
                if (E1 < E0)
                    break;
                alpha *= 0.5;
            }

        }
    }
#ifdef PARALLEL
    );
#endif
}

void VoronoiCells::constructVoronoiDiagram(bool exact, bool load_from_file)
{
    loadGeometry();
    gcs::PoissonDiskSampler poissonSampler(*mesh, *geometry);
    samples = poissonSampler.sample(1.0);
    int n_tri = mesh->nFaces();
    std::cout << "# sites " << samples.size() << std::endl;
    std::cout << "# faces " << n_tri << std::endl;
    if (load_from_file)
    {
        std::ifstream in("samples.txt");
        int n_samples; in >> n_samples;
        samples.resize(n_samples);
        for (int i = 0; i < n_samples; i++)
        {
            int face_idx;
            T wx, wy, wz;
            in >> face_idx >> wx >> wy >> wz;
            samples[i] = SurfacePoint(mesh->face(face_idx), Vector3{wx, wy, wz});
        }
        in.close();
    }

    n_sites = samples.size();
    voronoi_sites.resize(n_sites * 3);
    int cnt = 0;
    for (SurfacePoint& pt : samples)
    {
        pt = pt.inSomeFace();
        voronoi_sites.segment<3>(cnt * 3) = toTV(pt.interpolate(geometry->vertexPositions));
        cnt++;
    }
    
    source_data.resize(n_tri, FaceData());
    START_TIMING(PropogateDistance)
    propogateDistanceField(samples, source_data);
    FINISH_TIMING_PRINT(PropogateDistance)
    
    START_TIMING(PrismCutting)
    std::vector<std::pair<SurfacePoint, std::vector<int>>> ixn_data;

    intersectPrisms(samples, source_data, ixn_data);
    FINISH_TIMING_PRINT(PrismCutting)
    
    // remove duplication
    std::vector<SurfacePoint> unique_ixn_points;
    std::vector<IV2> unique_edges;

    

    if (exact)
    {
        START_TIMING(ExactVoronoi)
        optimizeForExactVD(ixn_data);
        FINISH_TIMING_PRINT(ExactVoronoi)
    }

    // voronoi_edges = edges;

    for (int i = 0; i < ixn_data.size(); i+=2)
    {

        TV v0 = toTV(ixn_data[i].first.interpolate(geometry->vertexPositions));
        TV v1 = toTV(ixn_data[i+1].first.interpolate(geometry->vertexPositions));
        // std::cout << v0.transpose() << " " << v1.transpose() << std::endl;
        voronoi_edges.push_back(std::make_pair(v0, v1));
    }
    
    
    
    use_debug_face_color = false;
    if (use_debug_face_color)
        updateFaceColor();
}

void VoronoiCells::updateFaceColor()
{
    int n_tri = mesh->nFaces();
    face_color.resize(n_tri, 3);
    face_color.setZero();
    
    tbb::parallel_for(0, n_tri, [&](int i)
    {
        if (source_data[i].site_indices.size() == 0)
        {
            std::cout << "error" << std::endl;
        }
        else if (source_data[i].site_indices.size() == 1)
            face_color.row(i) = TV(153.0/255.0, 204/255.0, 1.0);
        else if (source_data[i].site_indices.size() == 2)
            face_color.row(i) = TV(153.0/255.0, 153.0/255.0, 1.0);
        else if (source_data[i].site_indices.size() == 3)
            face_color.row(i) = TV(178.0/255.0, 102.0/255.0, 1.0);
        else if (source_data[i].site_indices.size() == 4)
            face_color.row(i) = TV(1.0, 0.0, 1.0);
        else if (source_data[i].site_indices.size() == 5)
            face_color.row(i) = TV(1.0, 0.0, 0.2);
        else if (source_data[i].site_indices.size() > 5)
            face_color.row(i) = TV(1.0, 0.0, 0.0);
    });
}


void VoronoiCells::saveFacePrism(int face_idx)
{
    std::vector<std::pair<TV, TV>> edges;
    intersectPrism(samples, source_data, edges, face_idx);
}