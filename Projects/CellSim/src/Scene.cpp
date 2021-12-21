#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/vertex_triangle_adjacency.h>
#include <igl/per_face_normals.h>

#include "../include/VertexModel.h"

void VertexModel::approximateMembraneThickness()
{
    T radii_max = -1e6, radii_min = 1e6;
    for (int i = 0; i < basal_vtx_start; i++)
    {
        radii_max = std::max(radii_max, (undeformed.segment<3>(i * 3) - mesh_centroid).norm());
        radii_min = std::min(radii_min, (undeformed.segment<3>(i * 3) - mesh_centroid).norm());
    }
    if (sphere_bound_penalty)
        Rc = radii_min;
    else
        Rc = 1.01 * radii_max;
    
    Rc = 1.005 * radii_max;
    total_volume = 4.0 / 3.0 * M_PI * std::pow(Rc, 3);
}

void VertexModel::initializeContractionData()
{
    // VtxList contracting_vertices;
    // if (scene_type == 0)
    // {
    //     // on low res sphere
    //     contracting_vertices = {37, 36, 39, 49, 48, 50, 23, 67, 33, 32, 34};
    //     for (int i = 0; i < contracting_vertices.size(); i++)
    //     {
    //         int j = (i + 1) % contracting_vertices.size();
    //         Edge e(contracting_vertices[i], contracting_vertices[j]);
    //         contracting_edges.push_back(e);
    //     }
    // }
    // else if (scene_type == 1)
    // {
    //     // on high res sphere
    //     contracting_vertices = {576, 577, 587, 618, 610, 608, 611, 615, 534,
    //         529, 528, 530, 543, 526, 515, 512, 513, 523, 554, 
    //         546, 544, 
    //         547, 551, 598, 
    //         593, 592, 594, 
    //         607, 590, 579};
        
    //     for (int i = 0; i < contracting_vertices.size(); i++)
    //     {
    //         int j = (i + 1) % contracting_vertices.size();
    //         Edge e(contracting_vertices[i], contracting_vertices[j]);
    //         contracting_edges.push_back(e);
    //     }    
    // }
    // else if (scene_type == 2)
    // {
    //     contracting_vertices = {238, 281, 324, 329, 331, 333, 334, 339, 
    //     341, 343, 347, 349, 351, 366, 370, 372, 374, 375, 379, 380, 381, 385, 386, 387};

    //     iterateEdgeSerial([&](Edge& e){
    //         auto find_v0 = std::find(contracting_vertices.begin(), contracting_vertices.end(), e[0]);
    //         auto find_v1 = std::find(contracting_vertices.begin(), contracting_vertices.end(), e[1]);

    //         if (find_v0 != contracting_vertices.end() && find_v1 != contracting_vertices.end())
    //         {
    //             contracting_edges.push_back(e);
    //         }
    //     });
    //     std::cout << contracting_edges.size() << std::endl;
    // }
    
    


    TV min_corner, max_corner;
    computeBoundingBox(min_corner, max_corner);
    TV mid_point = 0.5 * (min_corner + max_corner);
    TV delta = max_corner - min_corner;
    if (contract_apical_face)
    {
        iterateFaceSerial([&](VtxList& face_vtx_list, int face_idx)
        {
            if (face_idx < basal_face_start)
            {
                bool contract = true;
                TV centroid;
                computeFaceCentroid(face_vtx_list, centroid);
                if (centroid[0] < min_corner[0] + (max_corner[0] - min_corner[0]) * 0.92
                    || centroid[1] < mid_point[1] - 0.5 * delta[1] * 0.35 
                    || centroid[1] > mid_point[1] + 0.5 * delta[1] * 0.35)
                    contract = false;
                if (contract)
                    contracting_faces.push_back(face_idx);
            }
        });
    }
    else
    {
        auto validEdge = [&](const Edge& e)
        {
            TV x0 = deformed.segment<3>(e[0] * 3);
            TV x1 = deformed.segment<3>(e[1] * 3);

            if (x0[0] > min_corner[0] + (max_corner[0] - min_corner[0]) * 0.8 &&
                x0[1] > min_corner[0] + (max_corner[0] - min_corner[0]) * 0.8 &&
                e[0] < basal_vtx_start && e[1] < basal_vtx_start)
                return true;
            return false;
        };

        iterateFaceSerial([&](VtxList& face_vtx_list, int face_idx)
        {
            if (face_idx < basal_face_start)
            {
                bool add_all_edges_in_this_face = false;
                for (int i = 0; i < face_vtx_list.size(); i++)
                {
                    int j = (i + 1) % face_vtx_list.size();
                    if (validEdge(Edge(face_vtx_list[i], face_vtx_list[j])))
                    {
                        add_all_edges_in_this_face = true;
                        break;
                    }
                }
                if (add_all_edges_in_this_face)
                {
                    for (int i = 0; i < face_vtx_list.size(); i++)
                    {
                        int j = (i + 1) % face_vtx_list.size();
                        contracting_edges.push_back(Edge(face_vtx_list[i], face_vtx_list[j]));
                    }
                }
                
            }
        });
        // for (auto e : edges)
        // {
        //     TV x0 = deformed.segment<3>(e[0] * 3);
        //     TV x1 = deformed.segment<3>(e[1] * 3);

        //     if (x0[0] > min_corner[0] + (max_corner[0] - min_corner[0]) * 0.8 &&
        //         x0[1] > min_corner[0] + (max_corner[0] - min_corner[0]) * 0.8 &&
        //         e[0] < basal_vtx_start && e[1] < basal_vtx_start)
        //     {
        //         contracting_edges.push_back(e);
        //     }
        // }
    }
    

}

bool VertexModel::computeBoundingBox(TV& min_corner, TV& max_corner)
{
    min_corner.setConstant(1e6);
    max_corner.setConstant(-1e6);

    for (int i = 0; i < num_nodes; i++)
    {
        for (int d = 0; d < 3; d++)
        {
            max_corner[d] = std::max(max_corner[d], deformed[i * 3 + d]);
            min_corner[d] = std::min(min_corner[d], deformed[i * 3 + d]);
        }
    }
}

void VertexModel::addTestPrismGrid(int n_row, int n_col)
{

    T dx = std::min(1.0 / n_col, 1.0 / n_row);
    T dy = dx;
    T dz = dx;

    num_nodes = n_row * n_col * 2;
    basal_vtx_start = n_row * n_col;

    undeformed.resize(num_nodes * 3);
    for (int row = 0; row < n_row; row++)
    {
        for (int col = 0; col < n_col; col++)
        {
            int idx = row * n_col + col;
            undeformed.segment<3>(idx * 3) = TV(col * dx, 0.0, row * dy);
            undeformed.segment<3>((idx + basal_vtx_start) * 3) = TV(col * dx, -dz, row * dy);
        }
    }
    deformed = undeformed;
    u = VectorXT::Zero(deformed.size());
    f = VectorXT::Zero(deformed.rows());

    for (int row = 0; row < n_row - 1; row++)
    {
        for (int col = 0; col < n_col - 1; col++)
        {
            int idx0 = row * n_col + col;
            int idx1 = (row + 1) * n_col + col;
            int idx2 = (row + 1) * n_col + col + 1;
            int idx3 = row * n_col + col + 1;
            VtxList face = {idx0, idx1, idx2, idx3};
            std::reverse(face.begin(), face.end());
            faces.push_back(face);
            for (int i = 0; i < 4; i++)
            {
                int j = (i + 1) % 4;
                Edge e(face[i], face[j]);

                auto find_iter = std::find_if(edges.begin(), edges.end(), [&e](Edge& ei){
                    return (e[0] == ei[0] && e[1] == ei[1]) || (e[0] == ei[1] && e[1] == ei[0]);
                });
                if (find_iter == edges.end())
                {
                    edges.push_back(e);
                }
            }
            
        }
    }    

    basal_face_start = faces.size();
    
    for (int i = 0; i < basal_face_start; i++)
    {
        VtxList face = faces[i];
        for (int& idx_i : face)
            idx_i += basal_vtx_start;
        std::reverse(face.begin(), face.end());
        faces.push_back(face);
    }
    lateral_face_start = faces.size();

    std::vector<Edge> basal_and_lateral_edges;

    for (Edge edge : edges)
    {
        Edge basal_edge(edge[0] + basal_vtx_start, edge[1] + basal_vtx_start);
        Edge lateral0(edge[0], basal_edge[0]);
        Edge lateral1(edge[1], basal_edge[1]);
        basal_and_lateral_edges.push_back(basal_edge);
        basal_and_lateral_edges.push_back(lateral0);
        basal_and_lateral_edges.push_back(lateral1);

        VtxList lateral_face = {edge[0], edge[1], basal_edge[1], basal_edge[0]};
        std::reverse(lateral_face.begin(), lateral_face.end());
        faces.push_back(lateral_face);
    }
    edges.insert(edges.end(), basal_and_lateral_edges.begin(), basal_and_lateral_edges.end());
    
    B = 1e6;
    By = 1e4;
    alpha = 1.0; 
    gamma = 1.0;
    sigma = 0.01;

    use_cell_centroid = true;


    for (int d = 0; d < 3; d++)
    {
        dirichlet_data[d] = 0.0;
    }

    add_yolk_volume = false;

    mesh_centroid = TV::Zero();
    if (add_yolk_volume)
    {
        for (int i = 0; i < num_nodes; i++)
            mesh_centroid += undeformed.segment<3>(i * 3);
        mesh_centroid /= T(num_nodes);
    }
    
    yolk_vol_init = 0.0;
    if (add_yolk_volume)
    {
        yolk_vol_init = computeYolkVolume();
    }
    std::cout << "yolk volume init " << yolk_vol_init << std::endl;
}

void VertexModel::addTestPrism(int edge)
{
    woodbury = false;

    single_prism = true;
    num_nodes = edge * 2;
    if (edge == 4)
    {
        deformed.resize(8 * 3);
        deformed << -0.5, 0.5, 0.5, 
                0.5, 0.5, 0.5, 
                0.5, 0.5, -0.5,
                -0.5, 0.5, -0.5,
                -0.5, -0.5, 0.5, 
                0.5, -0.5, 0.5, 
                0.5, -0.5, -0.5,
                -0.5, -0.5, -0.5;
        
        faces.push_back({0, 1, 2, 3});
        
        faces.push_back({7, 6, 5, 4});
        
        faces.push_back({0, 4, 5, 1});
        faces.push_back({1, 5, 6, 2});
        faces.push_back({2, 6, 7, 3});
        faces.push_back({3, 7, 4, 0});

        for (VtxList& f : faces)
            std::reverse(f.begin(), f.end());
        
    }
    else if (edge == 5)
    {
        deformed.resize(10 * 3);
        deformed.segment<3>(15) = TV(0, 0, 1);
        deformed.segment<3>(18) = TV(std::sin(2.0/5.0 * M_PI), 0, std::cos(2.0/5.0 * M_PI));
        deformed.segment<3>(21) = TV(std::sin(4.0/5.0 * M_PI), 0, -std::cos(1.0/5.0 * M_PI));
        deformed.segment<3>(24) = TV(-std::sin(4.0/5.0 * M_PI), 0, -std::cos(1.0/5.0 * M_PI));
        deformed.segment<3>(27) = TV(-std::sin(2.0/5.0 * M_PI), 0, std::cos(2.0/5.0 * M_PI));

        deformed.segment<3>(0) = TV(0, 1, 1);
        deformed.segment<3>(3) = TV(std::sin(2.0/5.0 * M_PI), 1, std::cos(2.0/5.0 * M_PI));
        deformed.segment<3>(6) = TV(std::sin(4.0/5.0 * M_PI), 1, -std::cos(1.0/5.0 * M_PI));
        deformed.segment<3>(9) = TV(-std::sin(4.0/5.0 * M_PI), 1, -std::cos(1.0/5.0 * M_PI));
        deformed.segment<3>(12) = TV(-std::sin(2.0/5.0 * M_PI), 1, std::cos(2.0/5.0 * M_PI));
        

        faces.push_back({0, 1, 2, 3, 4});
        faces.push_back({9, 8, 7, 6, 5});
        
        faces.push_back({0, 5, 6, 1});
        faces.push_back({1, 6, 7, 2});
        faces.push_back({2, 7, 8, 3});
        faces.push_back({3, 8, 9, 4});
        faces.push_back({4, 9, 5, 0});

        for (VtxList& f : faces)
            std::reverse(f.begin(), f.end());
    }
    else if (edge == 6)
    {
        deformed.resize(12 * 3);
        deformed.segment<3>(18) = TV(1, -0.5, 0);
        deformed.segment<3>(21) = TV(0.5, -0.5, -0.5 * std::sqrt(3));
        deformed.segment<3>(24) = TV(-0.5, -0.5, -0.5 * std::sqrt(3));
        deformed.segment<3>(27) = TV(-1, -0.5, 0);
        deformed.segment<3>(30) = TV(-0.5, -0.5, 0.5 * std::sqrt(3));
        deformed.segment<3>(33) = TV(0.5, -0.5, 0.5 * std::sqrt(3));

        deformed.segment<3>(0) = TV(1, 0.5, 0);
        deformed.segment<3>(3) = TV(0.5, 0.5, -0.5 * std::sqrt(3));
        deformed.segment<3>(6) = TV(-0.5, 0.5, -0.5 * std::sqrt(3));
        deformed.segment<3>(9) = TV(-1, 0.5, 0);
        deformed.segment<3>(12) = TV(-0.5, 0.5, 0.5 * std::sqrt(3));
        deformed.segment<3>(15) = TV(0.5, 0.5, 0.5 * std::sqrt(3));

        faces.push_back({0, 1, 2, 3, 4, 5});
        faces.push_back({11, 10, 9, 8, 7, 6});
        
        faces.push_back({0, 6, 7, 1});
        faces.push_back({1, 7, 8, 2});
        faces.push_back({2, 8, 9, 3});
        faces.push_back({3, 9, 10, 4});
        faces.push_back({4, 10, 11, 5});
        faces.push_back({5, 11, 6, 0});

        for (VtxList& f : faces)
            std::reverse(f.begin(), f.end());
    }
    
    basal_vtx_start = edge;
    basal_face_start = 1;
    lateral_face_start = 2;

    u = VectorXT::Zero(deformed.size());
    f = VectorXT::Zero(deformed.rows());
    undeformed = deformed;

    for (int i = 0; i < edge; i++)
    {
        int j = (i + 1) % edge;
        edges.push_back(Edge(i, j));
        edges.push_back(Edge(i + edge, j + edge));
        edges.push_back(Edge(i, i + edge));
    }
    for (int d = 0; d < 3; d++)
    {
        dirichlet_data[d] = 0.0;
    }

    add_yolk_volume = false;
    add_contraction_term = false;
    use_sphere_radius_bound = false;
    perivitelline_pressure = false;
    use_yolk_pressure = false; 
    use_ipc_contact = false;
    add_perivitelline_liquid_volume = false;

    mesh_centroid = TV(0, -1, 0);
    B = 1e6;
    By = 1e4;

    alpha = 1.0; 
    gamma = 1.0;
    sigma = 1.0;

    use_cell_centroid = false;
    use_face_centroid = false;

    use_elastic_potential = true;

    if (use_elastic_potential)
    {
        use_cell_centroid = false;
        E = 100;
        nu = 0.48;
    }


    preserve_tet_vol = true;
    

    if (add_yolk_volume)
        yolk_vol_init = computeYolkVolume();

    if (preserve_tet_vol)
        computeTetVolInitial();

}

void VertexModel::saveIPCData(int iter)
{
    std::ofstream out("output/cells/surface/ipc_mesh_iter_" + std::to_string(iter) +".obj");
    for (int i = 0; i < ipc_vertices.rows(); i++)
    {
        out << "v " << ipc_vertices.row(i) << std::endl;
    }
    for (int i = 0; i < ipc_faces.rows(); i++)
    {
        IV obj_face = ipc_faces.row(i).transpose() + IV::Ones();
        out << "f " << obj_face.transpose() << std::endl;
    }
    out.close();
}

void VertexModel::saveCellMesh(int iter)
{
    std::ofstream out("output/cells/cell/cell_mesh_iter_" + std::to_string(iter) +".obj");
    Eigen::MatrixXd V, C;
    Eigen::MatrixXi F;
    generateMeshForRendering(V, F, C, false);
    for (int i = 0; i < V.rows(); i++)
    {
        out << "v " << V.row(i) << std::endl;
    }
    for (int i = 0; i < F.rows(); i++)
    {
        IV obj_face = F.row(i).transpose() + IV::Ones();
        out << "f " << obj_face.transpose() << std::endl;
    }
    out.close();
}



void VertexModel::vertexModelFromMesh(const std::string& filename)
{
    Eigen::MatrixXd V, N;
    Eigen::MatrixXi F;
    igl::readOBJ(filename, V, F);

    // face centroids corresponds to the vertices of the dual mesh 
    std::vector<TV> face_centroids(F.rows());
    deformed.resize(F.rows() * 3);

    tbb::parallel_for(0, (int)F.rows(), [&](int i)
    {
        TV centroid = 1.0/3.0*(V.row(F.row(i)[0]) + V.row(F.row(i)[1]) + V.row(F.row(i)[2]));
        face_centroids[i] = centroid;
        deformed.segment<3>(i * 3) = centroid;
    });

    std::vector<std::vector<int>> dummy;

    igl::vertex_triangle_adjacency(V.rows(), F, faces, dummy);
    igl::per_face_normals(V, F, N);

    // re-order so that the faces around one vertex is clockwise
    tbb::parallel_for(0, (int)faces.size(), [&](int vi)
    {
        std::vector<int>& one_ring_face = faces[vi];
        TV avg_normal = N.row(one_ring_face[0]);
        for (int i = 1; i < one_ring_face.size(); i++)
        {
            avg_normal += N.row(one_ring_face[i]);
        }
        avg_normal /= one_ring_face.size();

        TV vtx = V.row(vi);
        TV centroid0 = face_centroids[one_ring_face[0]];
        std::sort(one_ring_face.begin(), one_ring_face.end(), [&](int a, int b){
            TV E0 = (face_centroids[a] - vtx).normalized();
            TV E1 = (face_centroids[b] - vtx).normalized();
            TV ref = (centroid0 - vtx).normalized();
            T dot_sign0 = E0.dot(ref);
            T dot_sign1 = E1.dot(ref);
            TV cross_sin0 = E0.cross(ref);
            TV cross_sin1 = E1.cross(ref);
            // use normal and cross product to check if it's larger than 180 degree
            T angle_a = cross_sin0.dot(avg_normal) > 0 ? std::acos(dot_sign0) : 2.0 * M_PI - std::acos(dot_sign0);
            T angle_b = cross_sin1.dot(avg_normal) > 0 ? std::acos(dot_sign1) : 2.0 * M_PI - std::acos(dot_sign1);
            
            return angle_a < angle_b;
        });
    });

    // extrude 
    TV mesh_center = TV::Zero();
    for (int i = 0; i < V.rows(); i++)
        mesh_center += V.row(i);
    mesh_center /= T(V.rows());

    basal_vtx_start = deformed.size() / 3;
    deformed.conservativeResize(deformed.rows() * 2);

    T e0_norm = (V.row(F.row(0)[1]) - V.row(F.row(0)[0])).norm();
    T cell_height = 1.0 * e0_norm;

    tbb::parallel_for(0, (int)basal_vtx_start, [&](int i){
        TV apex = deformed.segment<3>(i * 3);
        deformed.segment<3>(basal_vtx_start * 3 + i * 3) = mesh_center + 
            (apex - mesh_center) * ((apex - mesh_center).norm() - cell_height);
    });

    // add apical edges
    for (auto one_ring_face : faces)
    {
        for (int i = 0; i < one_ring_face.size(); i++)
        {
            int next = (i + 1) % one_ring_face.size();
            Edge e(one_ring_face[i], one_ring_face[next]);
            auto find_iter = std::find_if(edges.begin(), edges.end(), [&e](Edge& ei){
                return (e[0] == ei[0] && e[1] == ei[1]) || (e[0] == ei[1] && e[1] == ei[0]);
            });
            if (find_iter == edges.end())
            {
                edges.push_back(e);
            }
        }
    }
    
    basal_face_start = faces.size();
    for (int i = 0; i < basal_face_start; i++)
    {
        VtxList new_face = faces[i];
        for (int& idx : new_face)
            idx += basal_vtx_start;
        std::reverse(new_face.begin(), new_face.end());
        faces.push_back(new_face);
    }
    
    lateral_face_start = faces.size();

    cell_face_indices.resize(basal_face_start, VtxList());

    std::vector<Edge> basal_and_lateral_edges;

    for (Edge edge : edges)
    {
        Edge basal_edge(edge[0] + basal_vtx_start, edge[1] + basal_vtx_start);
        Edge lateral0(edge[0], basal_edge[0]);
        Edge lateral1(edge[1], basal_edge[1]);
        basal_and_lateral_edges.push_back(basal_edge);
        basal_and_lateral_edges.push_back(lateral0);
        basal_and_lateral_edges.push_back(lateral1);

        VtxList lateral_face = {edge[0], edge[1], basal_edge[1], basal_edge[0]};
        lateral_edge_face_map[edge] = faces.size();
        Edge edge_rev(edge[1], edge[0]);
        lateral_edge_face_map[edge_rev] = faces.size();
        faces.push_back(lateral_face);
    }
    edges.insert(edges.end(), basal_and_lateral_edges.begin(), basal_and_lateral_edges.end());

    num_nodes = deformed.rows() / 3;
    u = VectorXT::Zero(deformed.rows());
    f = VectorXT::Zero(deformed.rows());
    undeformed = deformed;

    B = 1e6;
    By = 1e5;

    contract_apical_face = false;
    use_cell_centroid = true;
    
    use_elastic_potential = false;

    if (use_elastic_potential)
    {
        use_cell_centroid = false;
        E = 10;
        nu = 0.48;
    }

    if (scene_type == 1 || scene_type == 2)
    {
            if (contract_apical_face)
            {
                alpha = 0.4; // lateral
                gamma = 1.0; // basal
                sigma = 0.6; // apical
            }
            else
            {
                // this weights converge at rest state when height = 0.5
                // alpha = 50.0;
                // gamma = 20.0;
                // sigma = 0.5;

                // this weights lead to invagination when height = 1.0
                if (use_cell_centroid)
                {
                    alpha = 100.0; //without tet
                    // alpha = 200.0;
                    gamma = 10.0;
                    sigma = 0.5;
                }
                else
                {
                    // fixed tet sub div
                    // alpha = 300.0; //tet barrier
                    // gamma = 40.0;

                    alpha = 40.0; //tet barrier
                    gamma = 20.0;
                    sigma = 80.0;
                }
                

            }
    }
    else
    {
        alpha = 1.0; 
        gamma = 1.0;
        sigma = 0.1;
    }


    use_face_centroid = use_cell_centroid;


    for (int d = basal_vtx_start * 3; d < basal_vtx_start * 3 + 3; d++)
    {
        dirichlet_data[d] = 0.0;
    }
    

    add_yolk_volume = true;

    mesh_centroid = TV::Zero();
    if (add_yolk_volume)
    {
        for (int i = 0; i < num_nodes; i++)
            mesh_centroid += undeformed.segment<3>(i * 3);
        mesh_centroid /= T(num_nodes);
    }
    
    yolk_vol_init = 0.0;
    if (add_yolk_volume)
    {
        yolk_vol_init = computeYolkVolume();
    }
    // std::cout << "yolk volume init " << yolk_vol_init << std::endl;
    // std::cout << "basal vertex starts at " << basal_vtx_start << std::endl;

    add_contraction_term = true;
    
    // Gamma = 0.5;
    Gamma = 5.0;
    if (woodbury)
    {
        if (contract_apical_face)
            Gamma = 2000.0;
        else 
        {
            if (use_cell_centroid)
                Gamma = 1.0; //worked for the centroid formulation
            else
                Gamma = 1.0; // used for fixed tet subdiv
        }
    }

    if (add_contraction_term)
    {
        initializeContractionData();
    }

    use_sphere_radius_bound = true;
    
    sphere_bound_penalty = true;

    sphere_bound_barrier = false;


    if (use_sphere_radius_bound)
    {
        approximateMembraneThickness();
        std::cout << "Rc " << Rc << std::endl;
        
        if (sphere_bound_penalty)
            bound_coeff = 1e4;
        else if (sphere_bound_barrier)
        {
            bound_coeff = 0.1;
            membrane_dhat = 1e-2;
        }
        else
        {
            bound_coeff = 10e-15;
        }
    }

    std::cout << "# system DoF: " << deformed.rows() << std::endl;

    use_yolk_pressure = false;
    // pressure_constant = 1e-6; //low res worked
    pressure_constant = 0.1;
    

    // preserve_tet_vol = !use_cell_centroid;
    tet_vol_penalty = 1e10;

    if (preserve_tet_vol)
        computeTetVolInitial();

    use_ipc_contact = false;
    add_friction = false;
    
    if (use_ipc_contact)
    {
        computeIPCRestData();
        barrier_weight = 1e6;
        barrier_distance = 1e-3;
    }

    if (use_alm_on_cell_volume)
    {
        lambda_cell_vol = VectorXT::Zero(deformed.rows());
        kappa = 1e6;
        kappa_max = 1e3 * kappa;
    }

    use_fixed_centroid = false;
    if (use_fixed_centroid)
        updateFixedCentroids();

    add_perivitelline_liquid_volume = true;
    use_perivitelline_liquid_pressure = false;

    if (add_perivitelline_liquid_volume)
    {
        perivitelline_vol_init = total_volume - computeTotalVolumeFromApicalSurface();
        if (use_cell_centroid)
            Bp = 1e5; 
        else
            Bp = 1e6; 
        if (use_perivitelline_liquid_pressure)
        {
            perivitelline_pressure = 1;
        }
    }
    else
    {
        perivitelline_vol_init = 0.0;
    }
    project_block_hessian_PD = false;

    woodbury = add_perivitelline_liquid_volume || add_yolk_volume;

    weights_all_edges = 500.0;
    if (use_cell_centroid)
        weights_all_edges = 0.1;

    add_tet_vol_barrier = true;

    add_log_tet_barrier = true;
    tet_vol_barrier_dhat = 1e-6;
    if (use_cell_centroid && !add_log_tet_barrier)
        tet_vol_barrier_w = 10e-22;
    else
        tet_vol_barrier_w = 1e10;
    
    add_qubic_unilateral_term = true;
    qubic_active_percentage = 0.1;
    tet_vol_qubic_w = 1e3;

    
    add_yolk_tet_barrier = false;
    yolk_tet_vol_barrier_dhat = 1e-5;
    yolk_tet_vol_barrier_w = 1e6;

    check_all_vtx_membrane = true;
}

