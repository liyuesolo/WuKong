#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/edges.h>
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
    
    Rc = 1.001 * radii_max;
    total_volume = 4.0 / 3.0 * M_PI * std::pow(Rc, 3);
}

void VertexModel::initializeContractionData()
{
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
                // if (centroid[0] < min_corner[0] + (max_corner[0] - min_corner[0]) * 0.97
                //     || centroid[1] < mid_point[1] - 0.5 * delta[1] * 0.1
                //     || centroid[1] > mid_point[1] + 0.5 * delta[1] * 0.1)
                //     contract = false;
                bool bottom_y = centroid[1] < min_corner[1] + (max_corner[1] - min_corner[1]) * 0.2;
                bool middle_z = centroid[2] > mid_point[2] - 0.5 * delta[2] * 0.1 && 
                    centroid[2] < mid_point[2] + 0.5 * delta[2] * 0.1;
                bool middle_x = centroid[0] > mid_point[0] - 0.5 * delta[0] * 0.2 &&
                    centroid[0] < mid_point[0] + 0.5 * delta[0] * 0.4;
                if (!bottom_y || !middle_z || !middle_x)
                    contract = false;
                if (contract)
                    contracting_faces.push_back(face_idx);
            }
        });
    }
    else
    {
        auto isContractingEdgeCephalic = [&](const Edge& e)
        {
            TV x0 = deformed.segment<3>(e[0] * 3);
            TV x1 = deformed.segment<3>(e[1] * 3);
            T one_third = min_corner[0] + 1.0 / 3.0 * (max_corner[0] - min_corner[0]);
            T width = 0.005 * (max_corner[0] - min_corner[0]);
            bool region = std::abs(one_third - x0[0]) < width || std::abs(one_third - x1[0]) < width;
            bool apical_edge = e[0] < basal_vtx_start && e[1] < basal_vtx_start;
            if (region && apical_edge)
                return true;
            return false;
        };

        auto isContractingEdgeVentral = [&](const Edge& e)
        {
            
            TV x0 = deformed.segment<3>(e[0] * 3);
            TV x1 = deformed.segment<3>(e[1] * 3);

            T percent_y = 0.05;
            bool bottom_y = x0[1] < min_corner[1] + (max_corner[1] - min_corner[1]) * percent_y
                && x0[1] < min_corner[1] + (max_corner[1] - min_corner[1]) * percent_y;
            bool apical_edge = e[0] < basal_vtx_start && e[1] < basal_vtx_start;
            T percent_x = 0.2, percent_z = 0.1;
            bool middle_z = x0[2] > mid_point[2] - 0.5 * delta[2] * percent_z && 
                    x0[2] < mid_point[2] + 0.5 * delta[2] * percent_z &&
                    x1[2] > mid_point[2] - 0.5 * delta[2] * percent_z && 
                    x1[2] < mid_point[2] + 0.5 * delta[2] * percent_z;
            bool middle_x = x0[0] > mid_point[0] - 0.5 * delta[0] * percent_x &&
                x0[0] < mid_point[0] + 0.5 * delta[0] * percent_x &&
                x1[0] > mid_point[0] - 0.5 * delta[0] * percent_x &&
                x1[0] < mid_point[0] + 0.5 * delta[0] * percent_x;

            if (bottom_y && apical_edge && middle_z && middle_x)
                return true;
            return false;
        };

        auto isCFFace = [&](VtxList& face_vtx_list, int face_idx)
        {
            TV centroid;
            computeFaceCentroid(face_vtx_list, centroid);
            T one_third = min_corner[0] + 1.0 / 3.0 * (max_corner[0] - min_corner[0]);
            T width = 0.02 * (max_corner[0] - min_corner[0]);
            bool cephalic_region = std::abs(one_third - centroid[0]) < width;
            bool apical = face_idx < basal_face_start;
            T percent_y = 0.8;
            bool top_y = true;//centroid[1] > max_corner[1] - (max_corner[1] - min_corner[1]) * percent_y;

            if (cephalic_region && apical & top_y)
                return true;
            return false;
        };

        auto isVFFace = [&](VtxList& face_vtx_list, int face_idx)
        {
            TV centroid;
            computeFaceCentroid(face_vtx_list, centroid);
            bool apical = face_idx < basal_face_start;
            T percent_y = 0.1;
            bool bottom_y = centroid[1] < min_corner[1] + (max_corner[1] - min_corner[1]) * percent_y;
            T percent_x = 0.4, percent_z = 0.1;
            bool middle_z = centroid[2] > mid_point[2] - 0.5 * delta[2] * percent_z && 
                    centroid[2] < mid_point[2] + 0.5 * delta[2] * percent_z;
            bool middle_x = centroid[0] > mid_point[0] - 0.5 * delta[0] * percent_x &&
                centroid[0] < mid_point[0] + 0.5 * delta[0] * percent_x;
            if (bottom_y && apical && middle_z && middle_x)
                return true;
            return false;
        };

        auto isDTFFace = [&](VtxList& face_vtx_list, int face_idx)
        {
            TV centroid;
            computeFaceCentroid(face_vtx_list, centroid);
            bool apical = face_idx < basal_face_start;
            T percent_y = 0.1;
            bool top_y = centroid[1] > max_corner[1] - (max_corner[1] - min_corner[1]) * percent_y;
            T width = 0.1 * (max_corner[0] - min_corner[0]);
            T three_fifth = min_corner[0] + 3.0 / 5.0 * (max_corner[0] - min_corner[0]);
            bool dtf_region = std::abs(centroid[0] - three_fifth) < width;
            if (apical && top_y && dtf_region)
                return true;
            return false;
        };

        auto isPMFace = [&](VtxList& face_vtx_list, int face_idx)
        {   
            TV centroid;
            computeFaceCentroid(face_vtx_list, centroid);
            bool apical = face_idx < basal_face_start;
            T percent_y = 0.4;
            bool top_y = centroid[1] > max_corner[1] - (max_corner[1] - min_corner[1]) * percent_y;
            T width = 0.1 * (max_corner[0] - min_corner[0]);
            bool pm_region = centroid[0] > max_corner[0] - width;
            if (apical && pm_region && top_y)
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
                    Edge e = Edge(face_vtx_list[i], face_vtx_list[j]);
                    bool is_ventral_edge = isContractingEdgeVentral(e);
                    if (is_ventral_edge)
                    {
                        add_all_edges_in_this_face = true;
                        break;
                    }
                }
                add_all_edges_in_this_face = false;
                bool is_CF_face = isCFFace(face_vtx_list, face_idx);
                bool is_VF_face = isVFFace(face_vtx_list, face_idx);
                bool is_DTF_face = isDTFFace(face_vtx_list, face_idx);
                bool is_PM_face = isPMFace(face_vtx_list, face_idx);

                // if (is_CF_face || is_VF_face)// || is_DTF_face || is_PM_face)
                if (is_VF_face)
                {
                    for (int i = 0; i < face_vtx_list.size(); i++)
                    {
                        int j = (i + 1) % face_vtx_list.size();
                        // contracting_edges.push_back(Edge(face_vtx_list[i], face_vtx_list[j]));
                        Edge edge = Edge(face_vtx_list[i], face_vtx_list[j]);
                        auto find_iter = std::find_if(edges.begin(), edges.end(), 
                            [&edge](const Edge& ei) 
                            {
                                bool case1 = edge[0] == ei[0] && edge[1] == ei[1];
                                bool case2 = edge[0] == ei[1] && edge[1] == ei[0];
                                if (case1 || case2)
                                    return true;
                                return false;
                            });
                        int edge_id = std::distance(edges.begin(), find_iter);
                        if (find_iter != edges.end())
                        {
                            contracting_edges.push_back(edge);
                            edge_weights[edge_id] = Gamma;
                            // edge_weight_mask[edge_id] = 0.0;
                            // std::cout << edge_weights[edge_id] << " " << edge_id << std::endl;
                            // std::getchar();
                        }
                    }
                }
            }
        });
    }
    

}

void VertexModel::computeBoundingBox(TV& min_corner, TV& max_corner)
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
    undeformed *= 10.0;

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
    sigma = 1.0;
    gamma = 0.2;
    alpha = 4.0;

    use_cell_centroid = true;


    for (int d = basal_vtx_start * 3; d < basal_vtx_start * 3 + 9; d++)
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
    n_cells = basal_face_start;

    woodbury = false;
    add_contraction_term = false;
    use_sphere_radius_bound = false;
    perivitelline_pressure = false;
    use_yolk_pressure = false; 
    use_ipc_contact = false;
    add_perivitelline_liquid_volume = false;
    use_sdf_boundary = false;
    add_contraction_term = false;
    use_fixed_centroid = false;
    use_elastic_potential = false;
    preserve_tet_vol = false;
    add_area_term = true;
    add_tet_vol_barrier = true;
    add_yolk_tet_barrier = false;
    use_face_centroid = true;
    computeVolumeAllCells(cell_volume_init);



    TV min_corner, max_corner;
    computeBoundingBox(min_corner, max_corner);
    std::cout << "BBOX: " << min_corner.transpose() << " " << max_corner.transpose() << std::endl;
    std::cout << "# system DoF: " << deformed.rows() << std::endl;
    std::cout << "# cells: " << basal_face_start << std::endl;
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

void VertexModel::saveIPCData(const std::string& folder, int iter, bool save_edges)
{
    std::ofstream out(folder + "/ipc_faces_iter" + std::to_string(iter) + ".obj");
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
    if (save_edges)
    {
        out.open(folder + "/ipc_edges_iter" + std::to_string(iter) + ".obj");
        for (int i = 0; i < ipc_vertices.rows(); i++)
            out << "v " << ipc_vertices.row(i) << std::endl;
        for (int i = 0; i < ipc_edges.rows(); i++)
            out << "l " << ipc_edges.row(i) + Edge::Ones().transpose() << std::endl;
        out.close();
    }
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

void VertexModel::sdfFromHighResDualMesh(const std::string& filename)
{
    Eigen::MatrixXd V, N;
    Eigen::MatrixXi F;
    igl::readOBJ(filename, V, F);
    normalizeToUnit(V);
    
    T scale = 2.0;

    V *= scale;
    std::vector<TV> face_centroids(F.rows());
    VectorXT vtx_normals(F.rows() * 3);

    VectorXT vertices(F.rows() * 3);

    tbb::parallel_for(0, (int)F.rows(), [&](int i)
    {
        TV centroid = 1.0/3.0*(V.row(F.row(i)[0]) + V.row(F.row(i)[1]) + V.row(F.row(i)[2]));
        vertices.segment<3>(i * 3) = centroid;
        face_centroids[i] = centroid;
        TV ej = (V.row(F.row(i)[2]) - V.row(F.row(i)[1])).normalized();
        TV ei = (V.row(F.row(i)[0]) - V.row(F.row(i)[1])).normalized();
        vtx_normals.segment<3>(i * 3) = ej.cross(ei).normalized();
    });
    
    VectorXi indices;
    sdf.initializedMeshData(vertices, indices, vtx_normals, 1e-3);
}

void VertexModel::constructAnnulusScene()
{

}

void VertexModel::vertexModelFromMesh(const std::string& filename)
{
    for (auto& data : tet_index_quad_prism)
    {
        T cp = data[0];
        data[0] = data[1];
        data[1] = cp;
    }
    for (auto& data : tet_index_oct_prism)
    {
        T cp = data[0];
        data[0] = data[1];
        data[1] = cp;
    }
    for (auto& data : tet_index_sept_prism)
    {
        T cp = data[0];
        data[0] = data[1];
        data[1] = cp;
    }
    //     std::reverse(data.begin(), data.end());
    // for (auto& data : tet_index_sept_prism)
    //     std::reverse(data.begin(), data.end());
    // for (auto& data : tet_index_oct_prism)
    //     std::reverse(data.begin(), data.end());
    // for (auto& data : tet_index_penta_prism)
    //     std::reverse(data.begin(), data.end());
    // for (auto& data : tet_index_hexa_prism)
    //     std::reverse(data.begin(), data.end());

    Eigen::MatrixXd V, N;
    Eigen::MatrixXi F;
    igl::readOBJ(filename, V, F);

    // should be 500 µm length, 200 µm diameter
    // unit = 100.0; // in µm
    // unit = 0.01;
    unit = 5.0; // for real model
    tbb::parallel_for(0, (int)V.rows(), [&](int i)
    {
        TM scaling = TM::Identity();
        // scaling(0, 0) = 5; scaling(1, 1) = 2; scaling(2, 2) = 2; 
        scaling *= unit;
        V.row(i) *= scaling;
    });

    // igl::writeOBJ("/home/yueli/Documents/ETH/WuKong/Projects/CellSim/data/ellipsoid6k.obj", V, F);
    // std::exit(0);

    has_rest_shape = true;

    // face centroids corresponds to the vertices of the dual mesh 
    std::vector<TV> face_centroids(F.rows());
    deformed.resize(F.rows() * 3);

    VectorXT vtx_normals(F.rows() * 3);

    tbb::parallel_for(0, (int)F.rows(), [&](int i)
    {
        TV centroid = 1.0/3.0*(V.row(F.row(i)[0]) + V.row(F.row(i)[1]) + V.row(F.row(i)[2]));
        face_centroids[i] = centroid;
        deformed.segment<3>(i * 3) = centroid;
        TV ej = (V.row(F.row(i)[2]) - V.row(F.row(i)[1])).normalized();
        TV ei = (V.row(F.row(i)[0]) - V.row(F.row(i)[1])).normalized();
        vtx_normals.segment<3>(i * 3) = ej.cross(ei).normalized();
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
    num_nodes = deformed.rows() / 3;

    T avg_edge_norm = 0.0;
    Eigen::MatrixXi mesh_edges;
    igl::edges(F, mesh_edges);
    for (int i = 0; i < F.rows(); i++)
    {
        avg_edge_norm += (V.row(mesh_edges.row(i)[1]) - V.row(mesh_edges.row(i)[0])).norm();    
    }

    avg_edge_norm /= T(mesh_edges.rows());

    T e0_norm = (V.row(F.row(0)[1]) - V.row(F.row(0)[0])).norm();
    // T cell_height = 0.5 * e0_norm; //drosophila 1k
    // T cell_height = 0.8 * e0_norm; //drosophila 476
    T cell_height = 0.7 * e0_norm; // drosophila 120 and 241

    // cell_height = 4.0 * e0_norm;
    cell_height = avg_edge_norm;
    
    // cell_height = 0.1; //6k

    TV min_corner, max_corner;
    computeBoundingBox(min_corner, max_corner);
    std::cout << "BBOX: " << min_corner.transpose() << " " << max_corner.transpose() << std::endl;
    
    // auto disCurve = [&](int _x) -> T
    // {
    //     T c = 4.0;
    //     // T a = 4.0;
    //     T a = 14.0;
    //     T curved = -a * std::pow(_x, 2.0) + c;
    //     std::cout << _x << " " << _x * _x << " " << -a * std::pow(_x, 2.0) << " " << curved << std::endl;
    //     std::getchar();
    //     return curved;
    // };  

    // tbb::parallel_for(0, (int)basal_vtx_start, [&](int i)
    VectorXT percentage(basal_vtx_start);
    // T a = 14.0, c = 4.0;
    // T a = 10.0, c = 3.5;
    T a = 8.0, c = 3.0;
    
    if (resolution < 2)
    {
        a = 3.0; c = 1.0;
    }
        
    // else if (resolution == 2)
    //     a = 8.0, c = 3.0;
    
    for (int i = 0; i < basal_vtx_start; i++)
    {
        TV apex = deformed.segment<3>(i * 3);
        
        T x = (apex[0] - mesh_center[0]) / (max_corner[0] - min_corner[0]);
        T curved = -a * std::pow(x, 2.0) + c;
        // std::cout << "curved " << curved << std::endl;
        deformed.segment<3>(basal_vtx_start * 3 + i * 3) =
                deformed.segment<3>(i * 3) - 
                vtx_normals.segment<3>(i * 3) * curved * cell_height;
    }
    // );
    // std::cout << percentage.minCoeff() << " " << percentage.maxCoeff() << std::endl;
    // std::getchar();

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
    std::set<int> face_edge_numbers;

    basal_face_start = faces.size();
    for (int i = 0; i < basal_face_start; i++)
    {
        VtxList new_face = faces[i];
        face_edge_numbers.insert(new_face.size());
        for (int& idx : new_face)
            idx += basal_vtx_start;
        std::reverse(new_face.begin(), new_face.end());
        faces.push_back(new_face);
    }

    std::cout << "this mesh contains polygon of the following edge numbers: " << std::endl;
    for (int idx : face_edge_numbers)
        std::cout << idx << " ";
    std::cout << std::endl;

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

    n_edges = edges.size();
    
    u = VectorXT::Zero(deformed.rows());
    f = VectorXT::Zero(deformed.rows());
    undeformed = deformed;
    n_cells = basal_face_start;

    B = 1e4;
    By = 1e4;

    contract_apical_face = false;
    use_cell_centroid = true;
    
    use_elastic_potential = false;

    if (use_elastic_potential)
    {
        use_cell_centroid = false;
        E = 10;
        nu = 0.48;
    }

    alpha = 10.0; // WORKED
    gamma = 3.0;
    sigma = 2.0;// apical


    use_face_centroid = use_cell_centroid;
    // use_face_centroid = false;
    edge_weight_mask.resize(edges.size());
    edge_weight_mask.setOnes();

    for (int d = basal_vtx_start * 3; d < basal_vtx_start * 3 + 9; d++)
    {
        dirichlet_data[d] = 0.0;
    }
    
    computeVolumeAllCells(cell_volume_init);


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
    add_area_term = true;
    // Gamma = 0.5;
    Gamma = 1.0;
    // if (scene_type == 3)
    //     Gamma = 20.0;
    if (woodbury)
    {
        if (contract_apical_face)
            Gamma = 1.0;
        else 
        {
            Gamma = 20; 
            // Gamma = 5.0; 
            sigma = 1.0;
            gamma = 0.2;
            alpha = 4.0;
        }
    }
    // Gamma *= 0.01;
    assign_per_edge_weight = true;
    contracting_type = ApicalOnly;
    if (assign_per_edge_weight)
    {
        int apical_edge_cnt = 0;
        int ab_edge_cnt = 0;
        for (Edge& e : edges)
        {
            if (e[0] < basal_vtx_start && e[1] < basal_vtx_start)
            {
                apical_edge_cnt++;
                ab_edge_cnt++;
            }
            if (e[0] >= basal_vtx_start && e[1] >= basal_vtx_start)
            {
                ab_edge_cnt++;
            }
        }
        if (contracting_type == ALLEdges)
        {
            edge_weights.resize(n_edges);
            edge_weights.setConstant(0.1);
        }
        else if (contracting_type == ApicalBasal)
        {
            edge_weights.resize(ab_edge_cnt);
            edge_weights.segment(0, apical_edge_cnt).setConstant(0.1);
            edge_weights.segment(apical_edge_cnt, ab_edge_cnt - apical_edge_cnt).setConstant(gamma);
        }
        else
        {
            edge_weights.resize(apical_edge_cnt);
            edge_weights.segment(0, apical_edge_cnt).setConstant(0.1);
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
        // std::cout << "Rc " << Rc << std::endl;
        
        if (sphere_bound_penalty)
            bound_coeff = 1e1;
        else if (sphere_bound_barrier)
        {
            bound_coeff = 1e3;
            membrane_dhat = 1e-3;
        }
        else
        {
            bound_coeff = 10e-15;
            
        }
    }

    std::cout << "# system DoF: " << deformed.rows() << std::endl;
    std::cout << "# cells: " << basal_face_start << std::endl;

    use_yolk_pressure = false;
    // pressure_constant = 1e-6; //low res worked
    pressure_constant = 0.1;
    

    preserve_tet_vol = false;
    tet_vol_penalty = 1e4;

    if (preserve_tet_vol)
        computeTetVolInitial();

    use_ipc_contact = true;
    add_friction = false;
    
    if (use_ipc_contact)
    {
        add_basal_faces_ipc = true;
        computeIPCRestData();
        barrier_weight = 1e7;
        barrier_distance = 1e-3;
        if (add_friction)
        {
            friction_mu = 0.4;
        }
    }

    if (use_alm_on_cell_volume)
    {
        lambda_cell_vol = VectorXT::Zero(deformed.rows());
        kappa = 1e4;
        kappa_max = 1e4;
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
    // weights_all_edges = 1.0;
    weights_all_edges = 0.1 * Gamma;
    

    // edge_weights[0] -= 1e-6;

    add_tet_vol_barrier = true;

    add_log_tet_barrier = false;
    tet_vol_barrier_dhat = 1e-2;
    if (use_cell_centroid && !add_log_tet_barrier)
        // tet_vol_barrier_w = 1e-25;
        tet_vol_barrier_w = 1e-34;
    else
        tet_vol_barrier_w = 1e10;
    // std::cout << cell_volume_init[0] << std::endl;
    add_qubic_unilateral_term = false;
    qubic_active_percentage = 0.1;
    tet_vol_qubic_w = 1e3;

    
    add_yolk_tet_barrier = false;
    yolk_tet_vol_barrier_dhat = 1e-5;
    yolk_tet_vol_barrier_w = 1e6;

    check_all_vtx_membrane = true;

    if (dynamics)
    {
        eta = 1e0;
        computeNodalMass();
    }

    use_sdf_boundary = true;

    VectorXT edge_norm(edges.size());
    tbb::parallel_for(0, (int)edges.size(), [&](int i){
        TV vi = undeformed.segment<3>(edges[i][0] * 3);
        TV vj = undeformed.segment<3>(edges[i][1] * 3);
        edge_norm[i] = (vj - vi).norm();
    });
    
    
    if (use_sdf_boundary && use_sphere_radius_bound)
    {
        bound_coeff = 1e2;
        
        T normal_offset = 1e-3;// * unit;
        // T normal_offset = 0;
        // T normal_offset = 0.1;
        // T normal_offset = -1e-2;
        VectorXT vertices; VectorXi indices;
        getInitialApicalSurface(vertices, indices);
        vtx_normals.conservativeResize(vertices.rows());
        int offset = basal_vtx_start * 3;
        for (int i = 0; i < basal_face_start; i++)
        {
            TV xi = undeformed.segment<3>(faces[i][0] * 3);
            TV xj = undeformed.segment<3>(faces[i][1] * 3);
            TV xk = undeformed.segment<3>(faces[i][2] * 3);
            TV normal = (xk - xj).normalized().cross((xi - xj).normalized()).normalized();
            vtx_normals.segment<3>(offset + i * 3) = -normal;
        }
        
        TV v0 = deformed.segment<3>(edges[0][0] * 3);
        TV v1 = deformed.segment<3>(edges[0][1] * 3);
        // T ref_spacing = (v1 - v0).norm() * 0.5;
        T ref_spacing = edge_norm.sum() / edge_norm.rows();
        std::cout << ref_spacing << std::endl;
        
        // ref_spacing = 0.4;
        // std::cout << ref_spacing << std::endl;
        // std::getchar();
        sdf.setRefDis(ref_spacing);
        sdf.initializedMeshData(vertices, indices, vtx_normals, normal_offset);
        
        VectorXT vtx_normal_all = vtx_normals.segment(0, basal_vtx_start * 3);
        vtx_normal_all.conservativeResize(num_nodes * 3);
        total_volume = computeInitialApicalVolumeWithOffset(vtx_normal_all, normal_offset);
        deformed = undeformed;
        // std::cout << "perivitelline_vol_init " << perivitelline_vol_init << std::endl;
        perivitelline_vol_init = total_volume - computeTotalVolumeFromApicalSurface();
        std::cout << "perivitelline_vol_init " << perivitelline_vol_init << std::endl;
        std::cout << "Total volume: " << total_volume << std::endl;
        bool all_inside = true;
        int inside_cnt = 0;
        for (int i = 0; i < num_nodes; i++)
        {
            TV xi = deformed.segment<3>(i * 3);
            if (sdf.inside(xi))
                inside_cnt++;
                // continue;
            // std::cout << sdf.value(xi) << std::endl;
            // all_inside = false;
            // break;
        }
        std::cout << num_nodes - inside_cnt << "/" << num_nodes << " points are outside the sdf" << std::endl;
        // if (!all_inside)
        //     std::cout << "NOT ALL VERTICES ARE INSIDE THE SDF" << std::endl;
    }
    
    
    // removeAllTerms();
    
    // Gamma = 100.0;
    // add_contraction_term = true;
    // woodbury = false;
    // use_sphere_radius_bound = true;
    // alpha = 1.0; gamma = 1.0; sigma = 1.0;
    // B = 1e6;    
    // alpha = 10.0; // WORKED
    // gamma = 3.0;
    // // sigma = 2.0;
    // B = 1e6;
    // sigma = 0; alpha = 0.0; gamma = 0.0;
    // weights_all_edges = 0;
}

