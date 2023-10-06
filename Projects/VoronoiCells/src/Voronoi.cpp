#include <igl/readOBJ.h>
#include <igl/edges.h>
#include <igl/per_vertex_normals.h>
#include <igl/facet_adjacency_matrix.h>
#include <igl/adjacency_matrix.h>
#include <igl/copyleft/cgal/intersect_with_half_space.h>
#include <igl/copyleft/cgal/extract_feature.h>
#include <igl/copyleft/cgal/mesh_boolean.h>
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

T VoronoiCells::computeGeodesicDistance(const SurfacePoint& a, const SurfacePoint& b)
{
    gcs::GeodesicAlgorithmExact mmp(*mesh, *geometry);
    mmp.propagate(a);
    return mmp.getDistance(b);
}


void VoronoiCells::propogateDistanceField(std::vector<SurfacePoint>& samples,
    std::vector<FaceData>& source_data)
{
    
    auto comp = [&]( std::pair<int, int> a, std::pair<int, int> b ) 
    { 
        T a_max = -1e10;
        for (int i = 0; i < source_data[a.first].distances.size(); i++)
        {
            a_max = std::max(a_max, source_data[a.first].distances[i].sum());
        }
        T b_max = -1e10;
        for (int i = 0; i < source_data[b.first].distances.size(); i++)
        {
            b_max = std::max(b_max, source_data[b.first].distances[i].sum());
        }
        return a_max > b_max;
    };
    // std::priority_queue<std::pair<int, int>, 
    //     std::vector<std::pair<int, int>>, 
    //     decltype(comp)> queue(comp);
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

        SurfacePoint pt = samples[site_idx];
        TV site_location = toTV(pt.interpolate(geometry->vertexPositions));
        TV v0 = toTV(geometry->vertexPositions[mesh->face(face_idx).halfedge().vertex()]);
        TV v1 = toTV(geometry->vertexPositions[mesh->face(face_idx).halfedge().next().vertex()]);
        TV v2 = toTV(geometry->vertexPositions[mesh->face(face_idx).halfedge().next().next().vertex()]);
        TV current_distance;
        if (metric == Euclidean)
        {
            current_distance[0] = (site_location - v0).norm();
            current_distance[1] = (site_location - v1).norm();
            current_distance[2] = (site_location - v2).norm();
        }
        else if (metric == Geodesic)
        {
            current_distance[0] = computeGeodesicDistance(pt, SurfacePoint(mesh->face(face_idx).halfedge().vertex()));
            current_distance[1] = computeGeodesicDistance(pt, SurfacePoint(mesh->face(face_idx).halfedge().next().vertex()));
            current_distance[2] = computeGeodesicDistance(pt, SurfacePoint(mesh->face(face_idx).halfedge().next().next().vertex()));
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
            TV& existing_distance = source_data[face_idx].distances[i];
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
            std::vector<std::pair<TV, TV>>& edges)
{
    // return;
    T max_h = 1.0;
    T EPSILON = 1e-12;

#define PARALLEL

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
    std::vector<std::vector<std::pair<TV, TV>>> edges_thread(source_data.size(), 
        std::vector<std::pair<TV, TV>>());

    // std::cout << "start" << std::endl;
    
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
            
            igl::copyleft::cgal::half_space_box(point, normal, prism,BV,BF);
            // MatrixXT box(8, 3);
            // for (int i = 0; i < 8; i++)
            // {
                
            //     box.row(i) = TV(BV(i, 0).exact().convert_to<T>(), 
            //         BV(i, 1).exact().convert_to<T>(), 
            //         BV(i, 2).exact().convert_to<T>());
            // }
            
            // igl::writeOBJ("box.obj", box, BF);
            // igl::copyleft::cgal::intersect_with_half_space(prism, prism_faces, 
            //             point, normal, prism, prism_faces, J);
            // START_TIMING(boolean)
            bool succeed = igl::copyleft::cgal::mesh_boolean(BV, BF, prism, prism_faces, 
                igl::MESH_BOOLEAN_TYPE_INTERSECT, prism, prism_faces);
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
            
            // colinear with triangle edges
            bool invalid = std::abs((v1_proj - v0_proj).normalized().dot(edge_vec)) > 1.0 - EPSILON 
                && std::abs((edge_vtx1_proj - v1_proj).normalized().dot(edge_vec)) > 1.0 - EPSILON;
            invalid |= std::abs((v1_proj - v2_proj).normalized().dot(edge_vec)) > 1.0 - EPSILON &&
                std::abs((edge_vtx1_proj - v1_proj).normalized().dot(edge_vec)) > 1.0 - EPSILON;
            invalid |= std::abs((v2_proj - v0_proj).normalized().dot(edge_vec)) > 1.0 - EPSILON &&
                std::abs((edge_vtx1_proj - v2_proj).normalized().dot(edge_vec)) > 1.0 - EPSILON;
            // std::cout << "check invalid done" << std::endl;

            if (invalid)
            {
                // if (std::abs(edge_vtx0[2] - max_h_remaining) > EPSILON
                //     && std::abs(edge_vtx1[2] - max_h_remaining) > EPSILON)
                //     invalid = false;
            }

            
            if (invalid)
                continue;
            
            

            TV vtx0_proj = edge_vtx0; vtx0_proj[2] = 0.0;
            TV vtx1_proj = edge_vtx1; vtx1_proj[2] = 0.0;
            vtx0_proj = (R.transpose() * vtx0_proj + trans);
            vtx1_proj = (R.transpose() * vtx1_proj + trans);

            edges_thread[face_idx].push_back(std::make_pair(vtx0_proj, vtx1_proj));
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
    
    
    for (const std::vector<std::pair<TV, TV>>& data : edges_thread)
    {
        if (data.size())
        {
            for (const std::pair<TV, TV>& edge : data)
                edges.push_back(edge);
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
    igl::readOBJ("/home/yueli/Documents/ETH/WuKong/Projects/VoronoiCells/data/sphere.obj", V, F);
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
    
    std::vector<std::pair<TV, TV>> edges;
    source_data.resize(n_tri, FaceData());
    propogateDistanceField(samples, source_data);
    
    START_TIMING(PrsimCutting)
    intersectPrisms(samples, source_data, edges);
    FINISH_TIMING_PRINT(PrsimCutting)
    
    if (exact)
    {

    }

    voronoi_edges = edges;

    
    
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