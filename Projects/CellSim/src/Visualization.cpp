#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/vertex_triangle_adjacency.h>
#include <igl/per_face_normals.h>

#include<random>
#include<cmath>
#include<chrono>

#include "../include/VertexModel.h"

void VertexModel::getOutsideVtx(Eigen::MatrixXd& points, Eigen::MatrixXd& color, 
    int sdf_test_sample_idx_offset)
{
    std::vector<TV> outside_sdf_points;
    int outside_cnt = 0;
    int n_vtx = check_all_vtx_membrane ? num_nodes : basal_vtx_start;
    for (int i = 0; i < n_vtx; i++)
    {
        TV xi = deformed.segment<3>(i * 3);
        if (sdf.inside(xi))
            continue;
        outside_sdf_points.push_back(xi);
        outside_cnt++;
    }
    int np = sdf_test_sample_idx_offset;
    points.conservativeResize(np + outside_sdf_points.size(), 3);
    color.conservativeResize(points.rows(), 3);
    tbb::parallel_for(0, int(outside_sdf_points.size()), [&](int i){
        points.row(np + i) = outside_sdf_points[i];
        color.row(np + i) = TV(0, 1, 1);
    });
    std::cout << "[SDF]" << outside_cnt << "/" << n_vtx << " are outside the sdf" << std::endl;
}

void VertexModel::sampleBoundingSurface(Eigen::MatrixXd& V)
{
    if (use_sdf_boundary)
    {
        VectorXT boundary_points;
        sdf.sampleZeroLevelset(boundary_points);
        int n_pt = boundary_points.rows() / 3;
        V.resize(n_pt, 3);
        tbb::parallel_for(0, n_pt, [&](int i){
            V.row(i) = boundary_points.segment<3>(i * 3);
        });
    }
    else
    {
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::mt19937 generator (seed);
        std::uniform_real_distribution<T> uniform01(0.0, 1.0);
        int N = 10000;
        V.resize(N, 3);
        int type = 0;
        if (type == 0) //sphere
        {
            for (int i = 0; i < N; i++) 
            {
                T theta = 2 * M_PI * uniform01(generator);
                T phi = std::acos(1 - 2 * uniform01(generator));
                T x = Rc * std::sin(phi) * std::cos(theta);
                T y = Rc * std::sin(phi) * std::sin(theta);
                T z = Rc * std::cos(phi);
                V.row(i) = TV(x, y, z);
            }

        }
    }
    
}

void VertexModel::splitYolkForRendering(Eigen::MatrixXd& V, Eigen::MatrixXi& F, 
        Eigen::MatrixXd& C, bool a_bit)
{
    int face_cnt = 0, vtx_cnt = 0;
    T offset_percentage = 2.0;
    if (a_bit)
        offset_percentage = 1.5;
    std::vector<IV> tri_faces;
    std::vector<TV> vertices;
    std::vector<TV> colors;
    
    if (use_cell_centroid)
    {
        vertices.push_back(mesh_centroid);
        vtx_cnt++;

        iterateFaceSerial([&](VtxList& face_vtx_list, int face_idx)
        {
            if (face_idx < basal_face_start)
            {
                VectorXT positions;
                positionsFromIndices(positions, face_vtx_list);
                VectorXT positions_basal;
                VtxList basal_face_vtx_list = face_vtx_list;
                TV basal_centroid;
                computeFaceCentroid(basal_face_vtx_list, basal_centroid);
                TV basal_centroid_shifted = mesh_centroid + (basal_centroid - mesh_centroid) * offset_percentage;

                TV shift = basal_centroid_shifted - basal_centroid;
                VtxList new_face_vtx;
                vertices.push_back(basal_centroid_shifted);

                new_face_vtx.push_back(vtx_cnt);
                for (int i = 0; i < face_vtx_list.size(); i++)
                    new_face_vtx.push_back(vtx_cnt + i + 1);
                vtx_cnt++;

                for (int i = 0; i < face_vtx_list.size(); i++)
                {
                    int j = (i + 1) % face_vtx_list.size();
                    positions.segment<3>(i * 3) += shift;
                    vertices.push_back(positions.segment<3>(i * 3));
                    colors.push_back(Eigen::Vector3d(0.0, 1.0, 0.0));
                    colors.push_back(Eigen::Vector3d(0.0, 1.0, 0.0));
                    // colors.push_back(Eigen::Vector3d(1.0, 0.3, 0.0));
                    // colors.push_back(Eigen::Vector3d(1.0, 0.3, 0.0));

                    tri_faces.push_back(IV(new_face_vtx[1 + i], new_face_vtx[0], new_face_vtx[1 + j]));
                    tri_faces.push_back(IV(new_face_vtx[1 + i], new_face_vtx[1 + j], 0));
                    vtx_cnt++;
                }
            }
        });

        V.resize(vtx_cnt, 3);
        F.resize(tri_faces.size(), 3);
        C.resize(tri_faces.size(), 3);
        for (int i = 0; i < vtx_cnt; i++)
        {
            V.row(i) = vertices[i];
        }
        
        for (int i = 0; i < tri_faces.size(); i++)
        {
            F.row(i) = tri_faces[i];
            C.row(i) = colors[i];
        }
    }
}

void VertexModel::splitCellsForRendering(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& C, bool a_bit)
{
    if (single_prism)
    {
        if (use_cell_centroid)
        {

        }
        else
        {
            if (num_nodes == 4 * 2)
            {

            }
            else if (num_nodes == 5 * 2)
            {
                TV v0 = deformed.segment<3>(5 * 3);
                TV v1 = deformed.segment<3>(6 * 3);
                TV v2 = deformed.segment<3>(7 * 3);
                TV v3 = deformed.segment<3>(8 * 3);
                TV v4 = deformed.segment<3>(9 * 3);

                TV v5 = deformed.segment<3>(0 * 3);
                TV v6 = deformed.segment<3>(1 * 3);
                TV v7 = deformed.segment<3>(2 * 3);
                TV v8 = deformed.segment<3>(3 * 3);
                TV v9 = deformed.segment<3>(4 * 3);
                
                std::vector<TV> vertices;
                std::vector<IV> tri_faces;

                auto appendTet = [&](const std::vector<TV>& tets, int tet_idx)
                {
                    int nv = vertices.size();
                    TV tet_center = TV::Zero();
                    for (const TV& vtx : tets)
                    {
                        tet_center += vtx;
                    }
                    tet_center *= 0.25;

                    TV shift = 0.5 * tet_center;

                    for (const TV& vtx : tets)
                        vertices.push_back(vtx + shift);
                    // tri_faces.push_back(IV(nv + 2, nv + 1, nv + 0));
                    tri_faces.push_back(IV(nv + 1, nv + 2, nv + 0));
                    // tri_faces.push_back(IV(nv + 3, nv + 2, nv + 0));
                    tri_faces.push_back(IV(nv + 2, nv + 3, nv + 0));
                    // tri_faces.push_back(IV(nv + 1, nv + 3, nv + 0));
                    tri_faces.push_back(IV(nv + 3, nv + 1, nv + 0));
                    // tri_faces.push_back(IV(nv + 2, nv + 3, nv + 1));
                    tri_faces.push_back(IV(nv + 3, nv + 2, nv + 1));
                };

                appendTet({v8, v3, v9, v0}, 0);
                appendTet({v9, v3, v4, v0}, 1);
                appendTet({v0, v7, v2, v1}, 2);
                appendTet({v8, v9, v5, v0}, 3);
                appendTet({v0, v6, v7, v1}, 4);
                appendTet({v0, v5, v7, v6}, 5);
                appendTet({v8, v5, v7, v0}, 6);
                appendTet({v0, v8, v2, v7}, 7);
                appendTet({v3, v8, v2, v0}, 8);

                V.resize(vertices.size(), 3);
                F.resize(tri_faces.size(), 3);
                C.resize(tri_faces.size(), 3);

                for (int i = 0; i < vertices.size(); i++)
                {
                    V.row(i) = vertices[i];
                }
                
                for (int i = 0; i < tri_faces.size(); i++)
                {
                    F.row(i) = tri_faces[i];
                    C.row(i) = TV(0, 0.3, 1.0);
                }

            }
            else if (num_nodes == 6 * 2)
            {
                TV v0 = deformed.segment<3>(11 * 3);
                TV v1 = deformed.segment<3>(10 * 3);
                TV v2 = deformed.segment<3>(9 * 3);
                TV v3 = deformed.segment<3>(8 * 3);
                TV v4 = deformed.segment<3>(7 * 3);
                TV v5 = deformed.segment<3>(6 * 3);

                TV v6 = deformed.segment<3>(5 * 3);
                TV v7 = deformed.segment<3>(4 * 3);
                TV v8 = deformed.segment<3>(3 * 3);
                TV v9 = deformed.segment<3>(2 * 3);
                TV v10 = deformed.segment<3>(1 * 3);
                TV v11 = deformed.segment<3>(0 * 3);
                
                std::vector<TV> vertices;
                std::vector<IV> tri_faces;

                auto appendTet = [&](const std::vector<TV>& tets, int tet_idx)
                {
                    int nv = vertices.size();
                    TV tet_center = TV::Zero();
                    for (const TV& vtx : tets)
                    {
                        tet_center += vtx;
                    }
                    tet_center *= 0.25;

                    TV shift = 0.5 * tet_center;

                    for (const TV& vtx : tets)
                        vertices.push_back(vtx + shift);
                    // tri_faces.push_back(IV(nv + 2, nv + 1, nv + 0));
                    tri_faces.push_back(IV(nv + 1, nv + 2, nv + 0));
                    // tri_faces.push_back(IV(nv + 3, nv + 2, nv + 0));
                    tri_faces.push_back(IV(nv + 2, nv + 3, nv + 0));
                    // tri_faces.push_back(IV(nv + 1, nv + 3, nv + 0));
                    tri_faces.push_back(IV(nv + 3, nv + 1, nv + 0));
                    // tri_faces.push_back(IV(nv + 2, nv + 3, nv + 1));
                    tri_faces.push_back(IV(nv + 3, nv + 2, nv + 1));
                };

                appendTet({v9, v2, v10, v3}, 0);
                appendTet({v2, v10, v3, v4}, 1);
                appendTet({v1, v11, v0, v7}, 2);
                appendTet({v9, v2, v8, v10}, 3);
                appendTet({v10, v1, v7, v11}, 4);
                appendTet({v0, v11, v6, v7}, 5);
                appendTet({v1, v11, v4, v5}, 6);
                appendTet({v2, v10, v1, v8}, 7);
                appendTet({v2, v10, v4, v1}, 8);
                appendTet({v1, v11, v5, v0}, 9);
                appendTet({v10, v1, v8, v7}, 10);
                appendTet({v10, v1, v11, v4}, 11);

                V.resize(vertices.size(), 3);
                F.resize(tri_faces.size(), 3);
                C.resize(tri_faces.size(), 3);

                for (int i = 0; i < vertices.size(); i++)
                {
                    V.row(i) = vertices[i];
                }
                
                for (int i = 0; i < tri_faces.size(); i++)
                {
                    F.row(i) = tri_faces[i];
                    C.row(i) = TV(0, 0.3, 1.0);
                }

            }
        }
        
    }
    else
    {
        int face_cnt = 0, vtx_cnt = 0;
        T offset_percentage = 2.0;
        if (a_bit)
            offset_percentage = 1.2;
        std::vector<IV> tri_faces;
        std::vector<TV> vertices;
        std::vector<TV> colors;
        
        if (use_cell_centroid)
        {
            iterateFaceSerial([&](VtxList& face_vtx_list, int face_idx)
            {
                if (face_idx < basal_face_start)
                {
                    // std::cout << "f " << face_idx << std::endl;
                    VectorXT positions;
                    positionsFromIndices(positions, face_vtx_list);
                    VectorXT positions_basal;
                    VtxList basal_face_vtx_list = face_vtx_list;
                    
                    for (int& idx : basal_face_vtx_list)
                        idx += basal_vtx_start;

                    positionsFromIndices(positions_basal, basal_face_vtx_list);
                    TV apical_centroid, basal_centroid;
                    computeFaceCentroid(face_vtx_list, apical_centroid);

                    computeFaceCentroid(basal_face_vtx_list, basal_centroid);
                    
                    TV apical_centroid_shifted = mesh_centroid + (apical_centroid - mesh_centroid) * offset_percentage;
                    TV shift = apical_centroid_shifted - apical_centroid;
                    VtxList new_face_vtx;
                    // V.row(vtx_cnt) = apical_centroid_shifted;
                    vertices.push_back(apical_centroid_shifted);
                    // out << "v " << apical_centroid_shifted.transpose() << std::endl;
                    new_face_vtx.push_back(vtx_cnt);
                    for (int i = 0; i < face_vtx_list.size(); i++)
                        new_face_vtx.push_back(vtx_cnt + i + 1);
                    vtx_cnt++;
                    for (int i = 0; i < face_vtx_list.size(); i++)
                    {
                        int j = (i + 1) % face_vtx_list.size();
                        positions.segment<3>(i * 3) += shift;
                        // out << "v " << positions.segment<3>(i * 3).transpose() << std::endl;
                        // V.row(vtx_cnt) =  positions.segment<3>(i * 3);
                        vertices.push_back(positions.segment<3>(i * 3));
                        colors.push_back(Eigen::Vector3d(1.0, 0.3, 0.0));
                        tri_faces.push_back(IV(new_face_vtx[1 + i], new_face_vtx[0], new_face_vtx[1 + j]));
                        vtx_cnt++;
                    }

                    VtxList new_face_vtx_basal;
                    TV basal_centroid_shifted = basal_centroid + shift;
                    // out << "v " << basal_centroid_shifted.transpose() << std::endl;
                    // V.row(vtx_cnt) =  basal_centroid_shifted;
                    vertices.push_back(basal_centroid_shifted);
                    new_face_vtx_basal.push_back(vtx_cnt);
                    for (int i = 0; i < basal_face_vtx_list.size(); i++)
                        new_face_vtx_basal.push_back(vtx_cnt + i + 1);
                    vtx_cnt++;
                    for (int i = 0; i < basal_face_vtx_list.size(); i++)
                    {
                        int j = (i + 1) % basal_face_vtx_list.size();
                        positions_basal.segment<3>(i * 3) += shift;
                        // out << "v " << positions_basal.segment<3>(i * 3).transpose() << std::endl;
                        // V.row(vtx_cnt) =  positions_basal.segment<3>(i * 3);
                        vertices.push_back(positions_basal.segment<3>(i * 3));
                        colors.push_back(Eigen::Vector3d(0.0, 1.0, 0.0));
                        tri_faces.push_back(IV(new_face_vtx_basal[0], new_face_vtx_basal[1 + i], new_face_vtx_basal[1 + j]));
                        vtx_cnt++;
                    }

                    for (int i = 0; i < basal_face_vtx_list.size(); i++)
                    {
                        int j = (i + 1) % basal_face_vtx_list.size();
                        tri_faces.push_back(IV(new_face_vtx[1 + i], new_face_vtx[1 + j], new_face_vtx_basal[ 1 + j]));
                        tri_faces.push_back(IV(new_face_vtx[1 + i], new_face_vtx_basal[ 1 + j], new_face_vtx_basal[ 1 + i]));
                        colors.push_back(Eigen::Vector3d(0.0, 0.3, 1.0));
                        colors.push_back(Eigen::Vector3d(0.0, 0.3, 1.0));
                    }
                    
                }
            });
        }
        else
        {
            iterateFaceSerial([&](VtxList& face_vtx_list, int face_idx)
            {
                if (face_idx < basal_face_start)
                {
                    VectorXT positions;
                    positionsFromIndices(positions, face_vtx_list);
                    VectorXT positions_basal;
                    VtxList basal_face_vtx_list = face_vtx_list;
                    
                    for (int& idx : basal_face_vtx_list)
                        idx += basal_vtx_start;

                    positionsFromIndices(positions_basal, basal_face_vtx_list);
                    TV apical_centroid, basal_centroid;
                    computeFaceCentroid(face_vtx_list, apical_centroid);

                    computeFaceCentroid(basal_face_vtx_list, basal_centroid);
                    
                    TV apical_centroid_shifted = mesh_centroid + (apical_centroid - mesh_centroid) * offset_percentage;
                    TV shift = apical_centroid_shifted - apical_centroid;

                    
                    auto appendFace = [&](int n_edge, Region region, const VtxList& vtx_list)
                    {
                        TV c;
                        if (region == Apical)
                            c = Eigen::Vector3d(1.0, 0.3, 0.0);
                        else if (region == Lateral)
                            c = Eigen::Vector3d(0.0, 0.3, 1.0);
                        else
                            c = Eigen::Vector3d(0.0, 1.0, 0.0);
                        
                        if (n_edge == 4)
                        {
                            colors.push_back(c); colors.push_back(c);
                            tri_faces.push_back(IV(vtx_list[1], vtx_list[0], vtx_list[2]));
                            tri_faces.push_back(IV(vtx_list[2], vtx_list[0], vtx_list[3]));
                        }
                        else if (n_edge == 5)
                        {
                            colors.push_back(c); colors.push_back(c); colors.push_back(c);
                            tri_faces.push_back(IV(vtx_list[1], vtx_list[0], vtx_list[2]));
                            tri_faces.push_back(IV(vtx_list[2], vtx_list[0], vtx_list[3]));
                            tri_faces.push_back(IV(vtx_list[3], vtx_list[0], vtx_list[4]));
                        }
                        else if (n_edge == 6)
                        {
                            colors.push_back(c); colors.push_back(c); colors.push_back(c); colors.push_back(c);
                            tri_faces.push_back(IV(vtx_list[1], vtx_list[0], vtx_list[2]));
                            tri_faces.push_back(IV(vtx_list[2], vtx_list[0], vtx_list[3]));
                            tri_faces.push_back(IV(vtx_list[3], vtx_list[0], vtx_list[5]));
                            tri_faces.push_back(IV(vtx_list[3], vtx_list[5], vtx_list[4]));
                        }
                    };

                    VtxList new_face_vtx;
                    for (int i = 0; i < face_vtx_list.size(); i++)
                        new_face_vtx.push_back(vtx_cnt + i);

                    for (int i = 0; i < face_vtx_list.size(); i++)
                    {
                        int j = (i + 1) % face_vtx_list.size();
                        positions.segment<3>(i * 3) += shift;
                        vertices.push_back(positions.segment<3>(i * 3));
                        vtx_cnt++;
                    }
                    appendFace(face_vtx_list.size(), Apical, new_face_vtx);

                    VtxList new_face_vtx_basal;
                
                    for (int i = 0; i < basal_face_vtx_list.size(); i++)
                        new_face_vtx_basal.push_back(vtx_cnt + i );
                    
                    for (int i = 0; i < basal_face_vtx_list.size(); i++)
                    {
                        int j = (i + 1) % basal_face_vtx_list.size();
                        positions_basal.segment<3>(i * 3) += shift;
                        vertices.push_back(positions_basal.segment<3>(i * 3));
                        vtx_cnt++;
                    }
                    std::reverse(new_face_vtx_basal.begin(), new_face_vtx_basal.end());
                    appendFace(face_vtx_list.size(), Basal, new_face_vtx_basal);
                    std::reverse(new_face_vtx_basal.begin(), new_face_vtx_basal.end());

                    for (int i = 0; i < basal_face_vtx_list.size(); i++)
                    {
                        int j = (i + 1) % basal_face_vtx_list.size();
                        
                        VtxList new_face_vtx_lateral = {new_face_vtx[i], new_face_vtx[j], new_face_vtx_basal[j], new_face_vtx_basal[i]};
                        std::reverse(new_face_vtx_lateral.begin(), new_face_vtx_lateral.end());
                        appendFace(4, Lateral, new_face_vtx_lateral);
                    }
                    

                    
                }
            });
        }
        
        // std::cout << "set V done " << std::endl;
        V.resize(vtx_cnt, 3);
        F.resize(tri_faces.size(), 3);
        C.resize(tri_faces.size(), 3);
        for (int i = 0; i < vtx_cnt; i++)
        {
            V.row(i) = vertices[i];
        }
        
        for (int i = 0; i < tri_faces.size(); i++)
        {
            F.row(i) = tri_faces[i];
            C.row(i) = colors[i];
        }
    }
    
}

void VertexModel::saveBasalSurfaceMesh(const std::string& filename, bool invert_normal)
{
    std::ofstream out(filename);
    for (int i = basal_vtx_start; i < num_nodes; i++)
    {
        out << "v " << deformed.segment<3>(i * 3).transpose() << std::endl;
    }
    iterateBasalFaceSerial([&](VtxList& face_vtx_list, int i)
    {
        if (face_vtx_list.size() == 5)
        {
            IV idx(face_vtx_list[0], face_vtx_list[1], face_vtx_list[2]);
            out << "f " << (idx - IV::Ones() *(basal_vtx_start - 1)).transpose() << std::endl;
            idx = IV(face_vtx_list[0], face_vtx_list[2], face_vtx_list[3]);
            out << "f " << (idx - IV::Ones() *(basal_vtx_start - 1)).transpose() << std::endl;
            idx = IV(face_vtx_list[0], face_vtx_list[3], face_vtx_list[4]);
            out << "f " << (idx - IV::Ones() *(basal_vtx_start - 1)).transpose() << std::endl;
        }
        else if (face_vtx_list.size() == 6)
        {
            IV idx(face_vtx_list[0], face_vtx_list[1], face_vtx_list[2]);
            out << "f " << (idx - IV::Ones() *(basal_vtx_start - 1)).transpose() << std::endl;
            idx = IV(face_vtx_list[0], face_vtx_list[2], face_vtx_list[3]);
            out << "f " << (idx - IV::Ones() *(basal_vtx_start - 1)).transpose() << std::endl;
            idx = IV(face_vtx_list[0], face_vtx_list[3], face_vtx_list[5]);
            out << "f " << (idx - IV::Ones() *(basal_vtx_start - 1)).transpose() << std::endl;
            idx = IV(face_vtx_list[5], face_vtx_list[3], face_vtx_list[4]);
            out << "f " << (idx - IV::Ones() *(basal_vtx_start - 1)).transpose() << std::endl;
        }
    });
    out.close();
}

void VertexModel::getYolkForRendering(Eigen::MatrixXd& V, Eigen::MatrixXi& F, 
        Eigen::MatrixXd& C, bool rest_shape)
{
    std::vector<TV> face_centroid;
    iterateBasalFaceSerial([&](VtxList& face_vtx_list, int i){
        TV centroid = rest_shape ? undeformed.segment<3>(faces[i][0] * 3) : deformed.segment<3>(faces[i][0] * 3);
        for (int j = 1; j < faces[i].size(); j++)
        {
            if (rest_shape)
                centroid += undeformed.segment<3>(faces[i][j] * 3);
            else
                centroid += deformed.segment<3>(faces[i][j] * 3);
        }
        face_centroid.push_back(centroid / T(faces[i].size()));
    });

    int centroids_start = basal_vtx_start;

    V.resize(basal_vtx_start + face_centroid.size(), 3);
    for (int i = 0; i < basal_vtx_start; i++)
    {
        V.row(i) = rest_shape ? undeformed.segment<3>((i + basal_vtx_start) * 3) : deformed.segment<3>((i + basal_vtx_start) * 3);
    }

    for (int i = 0; i < face_centroid.size(); i++)
        V.row(basal_vtx_start + i) = face_centroid[i];
    
    int face_cnt = 0;
    iterateBasalFaceSerial([&](VtxList& face_vtx_list, int i){
        face_cnt += faces[i].size();
    });

    F.resize(face_cnt, 3);
    C.resize(face_cnt, 3);

    face_cnt = 0;
    iterateBasalFaceSerial([&](VtxList& face_vtx_list, int i)
    {
        for (int j = 0; j < faces[i].size(); j++)
        {
            int next = (j + 1) % faces[i].size();
            F.row(face_cnt) = Eigen::Vector3i(centroids_start + i - basal_face_start, 
                faces[i][j]- basal_vtx_start, faces[i][next]- basal_vtx_start);
            C.row(face_cnt++) = TV(0, 1.0, 0.0);
        }  
    });

    // std::cout << F << std::endl;
}

void VertexModel::saveIndividualCellsWithOffset()
{
    std::string filename = "cell_break_down.obj";
    std::ofstream out(filename);
    int face_cnt = 0, vtx_cnt = 0;
    T offset_percentage = 2.0;
    std::vector<IV> tri_faces;
    iterateFaceSerial([&](VtxList& face_vtx_list, int face_idx)
    {
        if (face_idx < basal_face_start)
        {
            VectorXT positions;
            positionsFromIndices(positions, face_vtx_list);
            VectorXT positions_basal;
            VtxList basal_face_vtx_list = face_vtx_list;
            
            for (int& idx : basal_face_vtx_list)
                idx += basal_vtx_start;

            positionsFromIndices(positions_basal, basal_face_vtx_list);
            TV apical_centroid, basal_centroid;
            computeFaceCentroid(face_vtx_list, apical_centroid);

            computeFaceCentroid(basal_face_vtx_list, basal_centroid);
            

            TV apical_centroid_shifted = mesh_centroid + (apical_centroid - mesh_centroid) * offset_percentage;
            TV shift = apical_centroid_shifted - apical_centroid;
            VtxList new_face_vtx;
            out << "v " << apical_centroid_shifted.transpose() << std::endl;
            new_face_vtx.push_back(vtx_cnt);
            for (int i = 0; i < face_vtx_list.size(); i++)
                new_face_vtx.push_back(vtx_cnt + i + 1);
            vtx_cnt++;
            for (int i = 0; i < face_vtx_list.size(); i++)
            {
                int j = (i + 1) % face_vtx_list.size();
                positions.segment<3>(i * 3) += shift;
                out << "v " << positions.segment<3>(i * 3).transpose() << std::endl;
                tri_faces.push_back(IV(new_face_vtx[1 + i], new_face_vtx[0], new_face_vtx[1 + j]));
                vtx_cnt++;
            }

            VtxList new_face_vtx_basal;
            TV basal_centroid_shifted = basal_centroid + shift;
            out << "v " << basal_centroid_shifted.transpose() << std::endl;
            new_face_vtx_basal.push_back(vtx_cnt);
            for (int i = 0; i < basal_face_vtx_list.size(); i++)
                new_face_vtx_basal.push_back(vtx_cnt + i + 1);
            vtx_cnt++;
            for (int i = 0; i < basal_face_vtx_list.size(); i++)
            {
                int j = (i + 1) % basal_face_vtx_list.size();
                positions_basal.segment<3>(i * 3) += shift;
                out << "v " << positions_basal.segment<3>(i * 3).transpose() << std::endl;
                tri_faces.push_back(IV(new_face_vtx_basal[0], new_face_vtx_basal[1 + i], new_face_vtx_basal[1 + j]));
                vtx_cnt++;
            }

            for (int i = 0; i < basal_face_vtx_list.size(); i++)
            {
                int j = (i + 1) % basal_face_vtx_list.size();
                tri_faces.push_back(IV(new_face_vtx[1 + i], new_face_vtx[1 + j], new_face_vtx_basal[ 1 + j]));
                tri_faces.push_back(IV(new_face_vtx[1 + i], new_face_vtx_basal[ 1 + j], new_face_vtx_basal[ 1 + i]));
            }

        }
    });
    for (IV f : tri_faces)
        out << "f " << f.transpose() + IV::Ones().transpose() << std::endl;
    out.close();
}

void VertexModel::generateMeshForRendering(Eigen::MatrixXd& V, 
    Eigen::MatrixXi& F, Eigen::MatrixXd& C, bool rest_state)
{
    bool triangulate_with_centroid = true;
    // compute polygon face centroid
    std::vector<TV> face_centroid(faces.size());
    tbb::parallel_for(0, (int)faces.size(), [&](int i){
        TV centroid = rest_state ? undeformed.segment<3>(faces[i][0] * 3) : deformed.segment<3>(faces[i][0] * 3);
        for (int j = 1; j < faces[i].size(); j++)
            centroid += rest_state ? undeformed.segment<3>(faces[i][j] * 3) : deformed.segment<3>(faces[i][j] * 3);
        face_centroid[i] = centroid / T(faces[i].size());
    });

    V.resize(deformed.rows() / 3 + face_centroid.size(), 3);

    int centroids_start = deformed.rows() / 3;
    
    for (int i = 0; i < deformed.rows()/ 3; i++)
        V.row(i) = rest_state ? undeformed.segment<3>(i * 3) : deformed.segment<3>(i * 3);
    
    for (int i = 0; i < face_centroid.size(); i++)
        V.row(deformed.rows() / 3 + i) = face_centroid[i];
    
    int face_start = 0;
    // std::cout << basal_face_start << " " << faces.size() << std::endl;
    int face_cnt = 0;
    for (int i = face_start; i < faces.size(); i++)
    {
        if (triangulate_with_centroid)
            face_cnt += faces[i].size();
        else
        {
            if (faces[i].size() == 4)
                face_cnt += 2;
            else if (faces[i].size() == 5)
                face_cnt += 3;
            else if (faces[i].size() == 6)
                face_cnt += 4;
            else if (faces[i].size() == 7)
                face_cnt += 5;
            else if (faces[i].size() == 8)
                face_cnt += 6;
            else if (faces[i].size() == 9)
                face_cnt += 7;
            else
            {
                std::cout << "Unknown polygon edges " << __FILE__ << std::endl;
            }
        }
    }
    F.resize(face_cnt, 3);

    face_cnt = 0;
    for (int i = face_start; i < faces.size(); i++)
    {
        if (triangulate_with_centroid)
        {
            for (int j = 0; j < faces[i].size(); j++)
            {
                int next = (j + 1) % faces[i].size();
                F.row(face_cnt++) = Eigen::Vector3i(centroids_start + i, faces[i][next], faces[i][j]);
            }       
        }
        else
        {
            if (faces[i].size() == 4)
            {
                F.row(face_cnt++) = Eigen::Vector3i(faces[i][1], faces[i][0], faces[i][2]);
                F.row(face_cnt++) = Eigen::Vector3i(faces[i][2], faces[i][0], faces[i][3]);
            }
            else if (faces[i].size() == 5)
            {
                F.row(face_cnt++) = Eigen::Vector3i(faces[i][1], faces[i][0], faces[i][4]);
                F.row(face_cnt++) = Eigen::Vector3i(faces[i][2], faces[i][1], faces[i][4]);
                F.row(face_cnt++) = Eigen::Vector3i(faces[i][3], faces[i][2], faces[i][4]);
            }
            else if (faces[i].size() == 6)
            {
                F.row(face_cnt++) = Eigen::Vector3i(faces[i][1], faces[i][0], faces[i][2]);
                F.row(face_cnt++) = Eigen::Vector3i(faces[i][2], faces[i][0], faces[i][3]);
                F.row(face_cnt++) = Eigen::Vector3i(faces[i][3], faces[i][0], faces[i][5]);
                F.row(face_cnt++) = Eigen::Vector3i(faces[i][3], faces[i][5], faces[i][4]);
            }
            else if (faces[i].size() == 7)
            {
                F.row(face_cnt++) = Eigen::Vector3i(faces[i][1], faces[i][0], faces[i][2]);
                F.row(face_cnt++) = Eigen::Vector3i(faces[i][2], faces[i][0], faces[i][6]);
                F.row(face_cnt++) = Eigen::Vector3i(faces[i][2], faces[i][6], faces[i][3]);
                F.row(face_cnt++) = Eigen::Vector3i(faces[i][3], faces[i][6], faces[i][5]);
                F.row(face_cnt++) = Eigen::Vector3i(faces[i][3], faces[i][5], faces[i][4]);
            }
            else if (faces[i].size() == 8)
            {
                F.row(face_cnt++) = Eigen::Vector3i(faces[i][2], faces[i][1], faces[i][0]);
                F.row(face_cnt++) = Eigen::Vector3i(faces[i][4], faces[i][3], faces[i][2]);
                F.row(face_cnt++) = Eigen::Vector3i(faces[i][2], faces[i][0], faces[i][7]);
                F.row(face_cnt++) = Eigen::Vector3i(faces[i][2], faces[i][7], faces[i][4]);
                F.row(face_cnt++) = Eigen::Vector3i(faces[i][4], faces[i][7], faces[i][5]);
                F.row(face_cnt++) = Eigen::Vector3i(faces[i][5], faces[i][7], faces[i][6]);
            }
            else if (faces[i].size() == 9)
            {
                F.row(face_cnt++) = Eigen::Vector3i(faces[i][1], faces[i][0], faces[i][8]);
                F.row(face_cnt++) = Eigen::Vector3i(faces[i][2], faces[i][1], faces[i][8]);
                F.row(face_cnt++) = Eigen::Vector3i(faces[i][2], faces[i][8], faces[i][7]);
                F.row(face_cnt++) = Eigen::Vector3i(faces[i][2], faces[i][7], faces[i][6]);
                F.row(face_cnt++) = Eigen::Vector3i(faces[i][2], faces[i][6], faces[i][3]);
                F.row(face_cnt++) = Eigen::Vector3i(faces[i][3], faces[i][6], faces[i][4]);
                F.row(face_cnt++) = Eigen::Vector3i(faces[i][4], faces[i][6], faces[i][5]);
            }
            else
            {
                std::cout << "Unknown polygon edges " << __FILE__ << std::endl;
            }
        }
    }

    // std::cout << face_cnt << " " << F.rows() << std::endl;
    
    C.resize(F.rows(), F.cols());
    face_cnt = 0;
    for (int i = face_start; i < faces.size(); i++)
    {
        if (triangulate_with_centroid)
        {
            for (int j = 0; j < faces[i].size(); j++)
            {
                if (i < basal_face_start)
                    C.row(face_cnt) = Eigen::Vector3d(1.0, 0.3, 0.0);
                else if (i < lateral_face_start)
                    C.row(face_cnt) = Eigen::Vector3d(0.0, 1.0, 0.0);
                else
                    C.row(face_cnt) = Eigen::Vector3d(0, 0.3, 1.0);
                if (contract_apical_face && add_contraction_term)
                {
                    if (std::find(contracting_faces.begin(), contracting_faces.end(), i) != contracting_faces.end())
                        C.row(face_cnt) = Eigen::Vector3d(1.0, 0.5, 0.5);
                }   
                face_cnt++;
            }
        }
        else
        {
            TV color;
            if (i < basal_face_start)
                color = Eigen::Vector3d(1.0, 0.3, 0.0);
            else if (i < lateral_face_start)
                color = Eigen::Vector3d(0.0, 1.0, 0.0);
            else
                color = Eigen::Vector3d(0, 0.3, 1.0);

            if (contract_apical_face && add_contraction_term)
            {
                if (std::find(contracting_faces.begin(), contracting_faces.end(), i) != contracting_faces.end())
                    color = Eigen::Vector3d(1.0, 0.0, 0.0);
            }   

            if (faces[i].size() == 4)
            {
                C.row(face_cnt++) = color;
                C.row(face_cnt++) = color;
            }
            else if (faces[i].size() == 5)
            {
                C.row(face_cnt++) = color;
                C.row(face_cnt++) = color;
                C.row(face_cnt++) = color;
            }
            else if (faces[i].size() == 6)
            {
                C.row(face_cnt++) = color;
                C.row(face_cnt++) = color;
                C.row(face_cnt++) = color;
                C.row(face_cnt++) = color;
            }
            else if (faces[i].size() == 7)
                for (int k = 0; k < 5; k++)
                    C.row(face_cnt++) = color;
            else if (faces[i].size() == 8)
                for (int k = 0; k < 6; k++)
                    C.row(face_cnt++) = color;
            else if (faces[i].size() == 9)
                for (int k = 0; k < 7; k++)
                    C.row(face_cnt++) = color;
            else
            {
                std::cout << "Unknown polygon edges " << __FILE__ << std::endl;
            }
            
        }
    }

    // tbb::parallel_for(0, (int)F.rows(), [&](int i)
    // {
    //     if (i < basal_face_start)
    //         C.row(i) = Eigen::Vector3d(1.0, 0.3, 0.0);
    //     else if (i < lateral_face_start)
    //         C.row(i) = Eigen::Vector3d(0.0, 1.0, 0.0);
    //     else
    //         C.row(i) = Eigen::Vector3d(0, 0.3, 1.0);
    // });

}

void VertexModel::appendCylinderOnContractingEdges(
    Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& C)
{
    if (!add_contraction_term)
        return;

    TV v0 = undeformed.segment<3>(edges[0][0] * 3);
    TV v1 = undeformed.segment<3>(edges[0][1] * 3);
    T visual_R = 0.1 * (v0 - v1).norm();

    int n_div = 10;
    T theta = 2.0 * EIGEN_PI / T(n_div);
    VectorXT points = VectorXT::Zero(n_div * 3);
    for(int i = 0; i < n_div; i++)
        points.segment<3>(i * 3) = TV(visual_R * std::cos(theta * T(i)), 
        0.0, visual_R*std::sin(theta*T(i)));
    
    int rod_offset_v = n_div * 2;
    int rod_offset_f = n_div * 2;

    int n_row_V = V.rows();
    int n_row_F = F.rows();

    int n_contracting_edges = contracting_edges.size();
    
    V.conservativeResize(n_row_V + n_contracting_edges * rod_offset_v, 3);
    F.conservativeResize(n_row_F + n_contracting_edges * rod_offset_f, 3);
    C.conservativeResize(n_row_F + n_contracting_edges * rod_offset_f, 3);

    tbb::parallel_for(0, n_contracting_edges, [&](int i){
        int rov = n_row_V + i * rod_offset_v;
        int rof = n_row_F + i * rod_offset_f;

        TV vtx_from = deformed.segment<3>(contracting_edges[i][0] * 3);
        TV vtx_to = deformed.segment<3>(contracting_edges[i][1] * 3);

        TV axis_world = vtx_to - vtx_from;
        TV axis_local(0, axis_world.norm(), 0);

        Matrix<T, 3, 3> R = Eigen::Quaternion<T>().setFromTwoVectors(axis_world, axis_local).toRotationMatrix();

        for(int i = 0; i < n_div; i++)
        {
            for(int d = 0; d < 3; d++)
            {
                V(rov + i, d) = points[i * 3 + d];
                V(rov + i+n_div, d) = points[i * 3 + d];
                if (d == 1)
                    V(rov + i+n_div, d) += axis_world.norm();
            }

            // central vertex of the top and bottom face
            V.row(rov + i) = (V.row(rov + i) * R).transpose() + vtx_from;
            V.row(rov + i + n_div) = (V.row(rov + i + n_div) * R).transpose() + vtx_from;

            F.row(rof + i*2 ) = IV(rov + i, rov + i+n_div, rov + (i+1)%(n_div));
            F.row(rof + i*2 + 1) = IV(rov + (i+1)%(n_div), rov + i+n_div, rov + (i+1)%(n_div) + n_div);

            C.row(rof + i*2 ) = TV(1.0, 0.0, 0.0);
            C.row(rof + i*2 + 1) = TV(1.0, 0.0, 0.0);
        }

    });
    
}

void VertexModel::appendCylinderOnApicalEdges(Eigen::MatrixXd& V, Eigen::MatrixXi& F, 
        Eigen::MatrixXd& C)
{
    // T visual_R = 0.01;
    // int n_div = 10;
    // T theta = 2.0 * EIGEN_PI / T(n_div);
    // VectorXT points = VectorXT::Zero(n_div * 3);
    // for(int i = 0; i < n_div; i++)
    //     points.segment<3>(i * 3) = TV(visual_R * std::cos(theta * T(i)), 
    //     0.0, visual_R*std::sin(theta*T(i)));
    
    // int rod_offset_v = n_div * 2;
    // int rod_offset_f = n_div * 2;

    // int n_row_V = V.rows();
    // int n_row_F = F.rows();

    // int n_edges_draw = 0;
    // iterateApicalEdgeSerial([&](Edge e))
    // {
    //     n_edges_draw++;
    // });
    
    // V.conservativeResize(n_row_V + n_contracting_edges * rod_offset_v, 3);
    // F.conservativeResize(n_row_F + n_contracting_edges * rod_offset_f, 3);
    // C.conservativeResize(n_row_F + n_contracting_edges * rod_offset_f, 3);

    // for (int i = 0; i < n_contracting_edges; i++)
    // {
    //     int rov = n_row_V + i * rod_offset_v;
    //     int rof = n_row_F + i * rod_offset_f;

    //     TV vtx_from = deformed.segment<3>(contracting_edges[i][0] * 3);
    //     TV vtx_to = deformed.segment<3>(contracting_edges[i][1] * 3);

    //     TV axis_world = vtx_to - vtx_from;
    //     TV axis_local(0, axis_world.norm(), 0);

    //     Matrix<T, 3, 3> R = Eigen::Quaternion<T>().setFromTwoVectors(axis_world, axis_local).toRotationMatrix();

    //     for(int i = 0; i < n_div; i++)
    //     {
    //         for(int d = 0; d < 3; d++)
    //         {
    //             V(rov + i, d) = points[i * 3 + d];
    //             V(rov + i+n_div, d) = points[i * 3 + d];
    //             if (d == 1)
    //                 V(rov + i+n_div, d) += axis_world.norm();
    //         }

    //         // central vertex of the top and bottom face
    //         V.row(rov + i) = (V.row(rov + i) * R).transpose() + vtx_from;
    //         V.row(rov + i + n_div) = (V.row(rov + i + n_div) * R).transpose() + vtx_from;

    //         F.row(rof + i*2 ) = IV(rov + i, rov + i+n_div, rov + (i+1)%(n_div));
    //         F.row(rof + i*2 + 1) = IV(rov + (i+1)%(n_div), rov + i+n_div, rov + (i+1)%(n_div) + n_div);

    //         C.row(rof + i*2 ) = TV(0.0, 0.0, 1.0);
    //         C.row(rof + i*2 + 1) = TV(0.0, 0.0, 1.0);
    //     }
    // }   
}


void VertexModel::saveAPrism(const std::string& filename, const VtxList& face_vtx_list)
{
    VectorXT positions;
    int n_points = face_vtx_list.size();
    VtxList cell_vtx_list(n_points * 2);
    for (int i = 0; i < n_points; i++)
    {
        cell_vtx_list[i + n_points] = face_vtx_list[i];
        cell_vtx_list[i] = face_vtx_list[i] - basal_vtx_start;
    }
    
    positionsFromIndices(positions, cell_vtx_list);


    std::ofstream out(filename);
    for (int i = 0; i < n_points * 2; i++)
    {
        out << "v " << positions.segment<3>(i * 3).transpose() << std::endl;        
    }

    if (n_points == 5)
    {
        out << "f 1 2 3" << std::endl;
        out << "f 1 3 4" << std::endl;
        out << "f 1 4 5" << std::endl;
        out << "f 6 7 8" << std::endl;
        out << "f 6 8 9" << std::endl;
        out << "f 6 9 10" << std::endl;

        for (int i = 0; i < n_points; i++)
        {
            out << "f " << i + n_points + 1 << " " << (i + 1) % n_points + 1 <<  " " << i + 1 << std::endl;
            out << "f " << i + n_points + 1 << " " << (i + 1) % n_points + n_points + 1 <<  " " << (i + 1) % n_points + 1 << std::endl;
        }
        
    }
    else if (n_points == 6)
    {
        out << "f 1 2 3" << std::endl;
        out << "f 1 3 4" << std::endl;
        out << "f 1 4 6" << std::endl;
        out << "f 6 4 5" << std::endl;
        out << "f 8 7 9" << std::endl;
        out << "f 9 7 10" << std::endl;
        out << "f 10 7 12" << std::endl;
        out << "f 10 12 11" << std::endl;

        for (int i = 0; i < n_points; i++)
        {
            out << "f " << i + n_points + 1 << " " << (i + 1) % n_points + 1 <<  " " << i + 1 << std::endl;
            out << "f " << i + n_points + 1 << " " << (i + 1) % n_points + n_points + 1 <<  " " << (i + 1) % n_points + 1 << std::endl;
        }
    }
    out.close();
}


void VertexModel::saveLowVolumeTets(const std::string& filename)
{
    std::vector<TM3> low_vol_tet_vtx;
    iterateFixedYolkTetsSerial([&](TM3& x_deformed, VtxList& indices)
    {
        T ei = 0.0;
        T d = computeTetVolume(mesh_centroid, x_deformed.col(0), x_deformed.col(1), x_deformed.col(2));
        if (d < 1e-6)
        {
            low_vol_tet_vtx.push_back(x_deformed);
        }
    });
    std::ofstream out(filename);
    for (auto vtx : low_vol_tet_vtx)
        for (int i = 0; i < vtx.cols(); i++)
        {
            out << "v " << vtx.col(i).transpose() << std::endl;
        }
    out << "v " << mesh_centroid.transpose() << std::endl;
    int centrod_idx = low_vol_tet_vtx.size() * 4;

    for (int i = 0; i < low_vol_tet_vtx.size(); i++)
    {
        out << "f " << i * 4 + 1 << " " << i * 4 + 2 << " " << i * 4 + 3 << std::endl;
        out << "f " << i * 4 + 1 << " " << i * 4 + 2 << " " << centrod_idx << std::endl;
        out << "f " << i * 4 + 2 << " " << i * 4 + 3 << " " << centrod_idx << std::endl;
        out << "f " << i * 4 + 3 << " " << i * 4 + 1 << " " << centrod_idx << std::endl;
    }
    out.close();
    
}

void VertexModel::loadMeshAndSaveCentroid(const std::string& folder, int start, int end)
{
    Eigen::MatrixXi F;
    Eigen::MatrixXd V;
    for (int i = start; i < end; i++)
    {
        igl::readOBJ(folder + "/cell_mesh_iter_" + std::to_string(i) + ".obj", V, F);
        for (int i = 0; i < num_nodes; i++)
        {
            deformed.segment<3>(i * 3) = V.row(i);
        }
        u = deformed - undeformed;
        VectorXT cell_centroids = VectorXT::Zero(basal_face_start * 3);
        iterateCellParallel([&](VtxList& face_vtx_list, int cell_idx){
            TV centroid;
            computeCellCentroid(face_vtx_list, centroid);
            cell_centroids.segment<3>(cell_idx * 3) = centroid;
        });
        std::ofstream out(folder + "/cell_centroids_iter_" + std::to_string(i) + ".obj");
        for (int i = 0; i < basal_face_start; i++)
            out << "v " << cell_centroids.segment<3>(i * 3).transpose() << std::endl;
        out.close();
    }
    
}