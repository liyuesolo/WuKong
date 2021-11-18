#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/vertex_triangle_adjacency.h>
#include <igl/per_face_normals.h>
#include <unordered_set>

#include <Spectra/SymEigsShiftSolver.h>
#include <Spectra/MatOp/SparseSymShiftSolve.h>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>

#include <Eigen/PardisoSupport>

#include "../include/VertexModel.h"
#include "../include/autodiff/VertexModelEnergy.h"

void VertexModel::computeLinearModes()
{
    int nmodes = 15;

    StiffnessMatrix K(deformed.rows(), deformed.rows());
    run_diff_test = true;
    buildSystemMatrix(u, K);
    
    bool use_Spectra = true;

    if (use_Spectra)
    {

        Spectra::SparseSymShiftSolve<T, Eigen::Upper> op(K);

        //0 cannot cannot be used as a shift
        T shift = -1e-4;
        Spectra::SymEigsShiftSolver<T, 
            Spectra::LARGEST_MAGN, 
            Spectra::SparseSymShiftSolve<T, Eigen::Upper> > 
            eigs(&op, nmodes, 2 * nmodes, shift);

        eigs.init();

        int nconv = eigs.compute();

        if (eigs.info() == Spectra::SUCCESSFUL)
        {
            Eigen::MatrixXd eigen_vectors = eigs.eigenvectors().real();
            Eigen::VectorXd eigen_values = eigs.eigenvalues().real();
            std::cout << eigen_values << std::endl;
            std::ofstream out("cell_eigen_vectors.txt");
            out << eigen_vectors.rows() << " " << eigen_vectors.cols() << std::endl;
            for (int i = 0; i < eigen_vectors.cols(); i++)
                out << eigen_values[eigen_vectors.cols() - 1 - i] << " ";
            out << std::endl;
            for (int i = 0; i < eigen_vectors.rows(); i++)
            {
                // for (int j = 0; j < eigen_vectors.cols(); j++)
                for (int j = eigen_vectors.cols() - 1; j >-1 ; j--)
                    out << eigen_vectors(i, j) << " ";
                out << std::endl;
            }       
            out << std::endl;
            out.close();
        }
        else
        {
            std::cout << "Eigen decomposition failed" << std::endl;
        }
    }
}

bool VertexModel::linearSolve(StiffnessMatrix& K, VectorXT& residual, VectorXT& du)
{
    StiffnessMatrix I(K.rows(), K.cols());
    I.setIdentity();

    StiffnessMatrix H = K;

    Eigen::PardisoLDLT<Eigen::SparseMatrix<T, Eigen::ColMajor, typename StiffnessMatrix::StorageIndex>> solver;
    
    T alpha = 10e-6;
    solver.analyzePattern(K);
    for (int i = 0; i < 50; i++)
    {
        // std::cout << i << std::endl;
        solver.factorize(K);
        if (solver.info() == Eigen::NumericalIssue)
        {
            // std::cout << "indefinite" << std::endl;
            K = H + alpha * I;        
            alpha *= 10;
            continue;
        }
        du = solver.solve(residual);

        T dot_dx_g = du.normalized().dot(residual.normalized());

        // VectorXT d_vector = solver.vectorD();
        int num_negative_eigen_values = 0;


        bool positive_definte = num_negative_eigen_values == 0;
        bool search_dir_correct_sign = dot_dx_g > 1e-6;
        bool solve_success = (K*du - residual).norm() < 1e-6 && solver.info() == Eigen::Success;

        if (positive_definte && search_dir_correct_sign && solve_success)
            return true;
        else
        {
            K = H + alpha * I;        
            alpha *= 10;
        }
    }
    return false;
}

void VertexModel::splitCellsForRendering(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& C)
{
    
    int face_cnt = 0, vtx_cnt = 0;
    T offset_percentage = 2.0;
    std::vector<IV> tri_faces;
    std::vector<TV> vertices;
    std::vector<TV> colors;
    // std::cout << basal_face_start << std::endl;
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
    
    basal_vtx_start = edge;
    basal_face_start = 1;
    lateral_face_start = 2;

    u = VectorXT::Zero(deformed.size());
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
    mesh_centroid = TV(0, -1, 0);
    B = 1e6;
    By = 1.0;
    alpha = 1.0; 
    gamma = 1.0;

    use_cell_centroid = true;

    sigma = 1;


    if (add_yolk_volume)
        yolk_vol_init = computeYolkVolume();
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
    T cell_height = 0.8 * e0_norm;

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
        faces.push_back(lateral_face);
    }
    edges.insert(edges.end(), basal_and_lateral_edges.begin(), basal_and_lateral_edges.end());

    num_nodes = deformed.rows() / 3;
    u = VectorXT::Zero(deformed.rows());
    undeformed = deformed;

    B = 1e6;
    By = 1e4;
    alpha = 1.0; 
    gamma = 1.0;
    sigma = 0.1;

    use_cell_centroid = false;


    for (int d = 0; d < 3; d++)
    {
        dirichlet_data[d] = 0.0;
    }

    // for (int i = 0; i < basal_vtx_start; i++)
    // {
    //     for (int d = 0; d < 3; d++)
    //     {
    //         dirichlet_data[i * 3 + d] = 0.0;
    //     }   
    // }
    

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
    std::cout << "yolk volume init " << yolk_vol_init << std::endl;
    std::cout << "basal vertex starts at " << basal_vtx_start << std::endl;
}



void VertexModel::generateMeshForRendering(Eigen::MatrixXd& V, 
    Eigen::MatrixXi& F, Eigen::MatrixXd& C)
{
    // compute polygon face centroid
    std::vector<TV> face_centroid(faces.size());
    tbb::parallel_for(0, (int)faces.size(), [&](int i){
        TV centroid = deformed.segment<3>(faces[i][0] * 3);
        for (int j = 1; j < faces[i].size(); j++)
            centroid += deformed.segment<3>(faces[i][j] * 3);
        face_centroid[i] = centroid / T(faces[i].size());
    });

    V.resize(deformed.rows() / 3 + face_centroid.size(), 3);

    int centroids_start = deformed.rows() / 3;
    
    for (int i = 0; i < deformed.rows()/ 3; i++)
        V.row(i) = deformed.segment<3>(i * 3);
    
    for (int i = 0; i < face_centroid.size(); i++)
        V.row(deformed.rows() / 3 + i) = face_centroid[i];
    
    int face_start = 0;
    // std::cout << basal_face_start << " " << faces.size() << std::endl;
    int face_cnt = 0;
    for (int i = face_start; i < faces.size(); i++)
        face_cnt += faces[i].size();
    F.resize(face_cnt, 3);

    face_cnt = 0;
    for (int i = face_start; i < faces.size(); i++)
    {
        for (int j = 0; j < faces[i].size(); j++)
        {
            int next = (j + 1) % faces[i].size();
            F.row(face_cnt++) = Eigen::Vector3i(centroids_start + i, faces[i][next], faces[i][j]);
        }       
    }

    // std::cout << face_cnt << " " << F.rows() << std::endl;
    
    C.resize(F.rows(), F.cols());
    face_cnt = 0;
    for (int i = face_start; i < faces.size(); i++)
    {
        for (int j = 0; j < faces[i].size(); j++)
        {
            if (i < basal_face_start)
                C.row(face_cnt) = Eigen::Vector3d(1.0, 0.3, 0.0);
            else if (i < lateral_face_start)
                C.row(face_cnt) = Eigen::Vector3d(0.0, 1.0, 0.0);
            else
                C.row(face_cnt) = Eigen::Vector3d(0, 0.3, 1.0);
            face_cnt++;
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

void VertexModel::computeCellCentroid(const VtxList& face_vtx_list, TV& centroid)
{
    centroid = TV::Zero();
    for (int vtx_idx : face_vtx_list)
    {
        centroid += deformed.segment<3>(vtx_idx * 3);
        centroid += deformed.segment<3>((vtx_idx + basal_vtx_start) * 3);
    }
    centroid /= T(face_vtx_list.size() * 2);
}

void VertexModel::computeFaceCentroid(const VtxList& face_vtx_list, TV& centroid)
{
    centroid = TV::Zero();
    for (int vtx_idx : face_vtx_list)
        centroid += deformed.segment<3>(vtx_idx * 3);

    centroid /= T(face_vtx_list.size());
}

void VertexModel::computeCubeVolumeFromTet(const Vector<T, 24>& prism_vertices, T& volume)
{
    auto computeTetVolume = [&](const TV& a, const TV& b, const TV& c, const TV& d)
    {
        return 1.0 / 6.0 * (b - a).cross(c - a).dot(d - a);
    };

    
    TV v0 = prism_vertices.segment<3>(4 * 3);
    TV v1 = prism_vertices.segment<3>(5 * 3);
    TV v2 = prism_vertices.segment<3>(7 * 3);
    TV v3 = prism_vertices.segment<3>(6 * 3);
    TV v4 = prism_vertices.segment<3>(0 * 3);
    TV v5 = prism_vertices.segment<3>(1 * 3);
    TV v6 = prism_vertices.segment<3>(3 * 3);
    TV v7 = prism_vertices.segment<3>(2 * 3);

	
    Vector<T, 6> tet_vol;
    tet_vol[0] = computeTetVolume(v2, v4, v6, v5);
    tet_vol[1] = computeTetVolume(v7, v2, v6, v5);
    tet_vol[2] = computeTetVolume(v3, v2, v7, v5);
    tet_vol[3] = computeTetVolume(v4, v2, v0, v5);
    tet_vol[4] = computeTetVolume(v1, v0, v2, v5);
    tet_vol[5] = computeTetVolume(v3, v1, v2, v5);

    auto saveTetObjs = [&](const std::vector<TV>& tets, int idx)
    {
        TV tet_center = TV::Zero();
        for (const TV& vtx : tets)
        {
            tet_center += vtx;
        }
        tet_center *= 0.25;

        TV shift = 0.5 * tet_center;
        
        std::ofstream out("tet" + std::to_string(idx) + ".obj");
        for (const TV& vtx : tets)
            out << "v " << (vtx + shift).transpose() << std::endl;
        out << "f 3 2 1" << std::endl;
        out << "f 4 3 1" << std::endl;
        out << "f 2 4 1" << std::endl;
        out << "f 3 4 2" << std::endl;
        out.close();
    };

    // saveTetObjs({v2, v4, v6, v5}, 0);
    // saveTetObjs({v7, v2, v6, v5}, 1);
    // saveTetObjs({v3, v2, v7, v5}, 2);
    // saveTetObjs({v4, v2, v0, v5}, 3);
    // saveTetObjs({v1, v0, v2, v5}, 4);
    // saveTetObjs({v3, v1, v2, v5}, 5);

    std::cout << "tet vol " << tet_vol.transpose() << std::endl;

    volume = tet_vol.sum();
}

void VertexModel::computeCubeVolumeCentroid(const Vector<T, 24>& prism_vertices, T& volume)
{
    auto computeTetVolume = [&](const TV& a, const TV& b, const TV& c, const TV& d)
    {
        T tet_vol = 1.0 / 6.0 * (b - a).cross(c - a).dot(d - a);
        std::cout << tet_vol << std::endl;
        return tet_vol;
    };

    TV apical_centroid = TV::Zero(), basal_centroid = TV::Zero(), cell_centroid = TV::Zero();
    for (int i = 0; i < 4; i++)
    {
        apical_centroid += prism_vertices.segment<3>(i * 3);
        basal_centroid += prism_vertices.segment<3>((i + 4) * 3);
    }
    cell_centroid = (apical_centroid + basal_centroid) / T(8);
    apical_centroid /= T(4);
    basal_centroid /= T(4);

    volume = 0.0;

    for (int i = 0; i < 4; i++)
    {
        
        int j = (i + 1) % 4;
        TV r0 = prism_vertices.segment<3>(i * 3);
        TV r1 = prism_vertices.segment<3>(j * 3);
        volume -= computeTetVolume(apical_centroid, r1, r0, cell_centroid);

        TV r2 = prism_vertices.segment<3>((i + 4) * 3);
        TV r3 = prism_vertices.segment<3>((j + 4) * 3);
        volume += computeTetVolume(basal_centroid, r3, r2, cell_centroid);

        TV lateral_centroid = T(0.25) * (r0 + r1 + r2 + r3);
        volume += computeTetVolume(lateral_centroid, r1, r0, cell_centroid);
        volume += computeTetVolume(lateral_centroid, r3, r1, cell_centroid);
        volume += computeTetVolume(lateral_centroid, r2, r3, cell_centroid);
        volume += computeTetVolume(lateral_centroid, r0, r2, cell_centroid);
    }
    
}

void VertexModel::computeVolumeAllCells(VectorXT& cell_volume_list)
{
    // each apical face corresponds to one cell
    cell_volume_list = VectorXT::Ones(basal_face_start);

    // use apical face to iterate other faces within this cell for now
    iterateFaceParallel([&](VtxList& face_vtx_list, int face_idx){
        if (face_idx < basal_face_start)
        {
            VectorXT positions;
            VtxList cell_vtx_list = face_vtx_list;
            for (int idx : face_vtx_list)
                cell_vtx_list.push_back(idx + basal_vtx_start);

            positionsFromIndices(positions, cell_vtx_list);

            if (face_vtx_list.size() == 4)
            {
                if (use_cell_centroid)
                    computeVolume4Points(positions, cell_volume_list[face_idx]);
                else
                    computeQuadBasePrismVolume(positions, cell_volume_list[face_idx]);
                // T tet_vol;
                
                // computeCubeVolumeFromTet(positions, tet_vol);
                // // // computeCubeVolumeCentroid(positions, tet_vol);
                // std::cout << tet_vol << std::endl;
                // std::getchar();
            }
            else if (face_vtx_list.size() == 5)
            {
                if (use_cell_centroid)
                    computeVolume5Points(positions, cell_volume_list[face_idx]);
                else
                    computePentaBasePrismVolume(positions, cell_volume_list[face_idx]);
            }
            else if (face_vtx_list.size() == 6)
                computeVolume6Points(positions, cell_volume_list[face_idx]);
        }
    });
}


T VertexModel::computeYolkVolume(bool verbose)
{
    T yolk_volume = 0.0;
    if (verbose)
        std::cout << "yolk tet volume: " << std::endl;
    iterateFaceSerial([&](VtxList& face_vtx_list, int face_idx)
    {
        VectorXT positions;
        positionsFromIndices(positions, face_vtx_list);
        
        
        if (face_idx < lateral_face_start && face_idx >= basal_face_start)
        {
            T cone_volume;
            if (face_vtx_list.size() == 4) 
                // computeConeVolume4Points(positions, mesh_centroid, cone_volume);
                computeQuadConeVolume(positions, mesh_centroid, cone_volume);
            else if (face_vtx_list.size() == 5) 
                computeConeVolume5Points(positions, mesh_centroid, cone_volume);
            else if (face_vtx_list.size() == 6) 
                computeConeVolume6Points(positions, mesh_centroid, cone_volume);
            else
                std::cout << "unknown polygon edge number" << __FILE__ << std::endl;
            yolk_volume += cone_volume;
            if (verbose)
                std::cout << cone_volume << " ";
        }
        
    });
    if (verbose)
        std::cout << std::endl;
    return yolk_volume;
}

T VertexModel::computeTotalEnergy(const VectorXT& _u, bool verbose)
{
    if (verbose)
        std::cout << std::endl;
    T energy = 0.0;
    
    VectorXT projected = _u;
    if (!run_diff_test)
    {
        iterateDirichletDoF([&](int offset, T target)
        {
            projected[offset] = target;
        });
    }
    deformed = undeformed + projected;

    T edge_length_term = 0.0, area_term = 0.0, volume_term = 0.0, yolk_volume_term = 0.0;
    // edge length term
    iterateApicalEdgeSerial([&](Edge& e){
        
        TV vi = deformed.segment<3>(e[0] * 3);
        TV vj = deformed.segment<3>(e[1] * 3);
        T edge_length = computeEdgeSquaredNorm(vi, vj);
        edge_length_term += sigma * edge_length;

    });
    if (verbose)
        std::cout << "\tE_edge " << edge_length_term << std::endl;
    energy += edge_length_term;

    VectorXT current_cell_volume;
    computeVolumeAllCells(current_cell_volume);
    
    // if (verbose)
        // std::cout << "current cell volume " << current_cell_volume.transpose() << std::endl;

    iterateFaceSerial([&](VtxList& face_vtx_list, int face_idx)
    {
        VectorXT positions;
        positionsFromIndices(positions, face_vtx_list);
        T area_energy = 0.0;
        // cell-wise volume preservation term
        if (face_idx < basal_face_start)
        {
            volume_term += 0.5 * B * std::pow(cell_volume_init[face_idx] - current_cell_volume[face_idx], 2);
        }
        else // basal and lateral faces area term
        {
            T coeff = face_idx >= lateral_face_start ? alpha : gamma;
            if (face_vtx_list.size() == 4)
                computeQuadFaceArea(coeff, positions, area_energy);
            else if (face_vtx_list.size() == 5)
                computePentFaceAreaEnergy(coeff, positions, area_energy);
            else if (face_vtx_list.size() == 6)
                computeHexFaceAreaEnergy(coeff, positions, area_energy);
            else
                std::cout << "unknown polygon edge case" << std::endl;
        }
        area_term += area_energy;
    });
    if (verbose)
    {
        std::cout << "\tE_area: " << area_term << std::endl;
        std::cout << "\tE_volume: " << volume_term << std::endl;
    }

    energy += volume_term;
    energy += area_term;

    if (add_yolk_volume)
    {
        T yolk_vol_curr = computeYolkVolume();
        yolk_volume_term +=  0.5 * By * std::pow(yolk_vol_curr - yolk_vol_init, 2);    
    }
    if (verbose)
        std::cout << "\tE_yolk_vol " << yolk_volume_term << std::endl;

    energy += yolk_volume_term;

    return energy;
}

T VertexModel::computeResidual(const VectorXT& _u,  VectorXT& residual, bool verbose)
{
    VectorXT residual_temp = residual;
    VectorXT projected = _u;
    if (!run_diff_test)
    {
        iterateDirichletDoF([&](int offset, T target)
        {
            projected[offset] = target;
        });
    }
    deformed = undeformed + projected;

    // edge length term
    iterateApicalEdgeSerial([&](Edge& e){
        TV vi = deformed.segment<3>(e[0] * 3);
        TV vj = deformed.segment<3>(e[1] * 3);
        Vector<T, 6> dedx;
        computeEdgeSquaredNormGradient(vi, vj, dedx);
        dedx *= -sigma;
        addForceEntry<6>(residual, {e[0], e[1]}, dedx);
    });

    if (verbose)
        std::cout << "edge length force norm: " << (residual - residual_temp).norm() << std::endl;
    residual_temp = residual;

    VectorXT current_cell_volume;
    computeVolumeAllCells(current_cell_volume);

    T yolk_vol_curr = 0.0;
    if (add_yolk_volume)
    {
        yolk_vol_curr = computeYolkVolume();
    }
    
    
    iterateFaceSerial([&](VtxList& face_vtx_list, int face_idx)
    {
        // cell-wise volume preservation term
        if (face_idx < basal_face_start)
        {
            VectorXT positions;
            VtxList cell_vtx_list = face_vtx_list;
            for (int idx : face_vtx_list)
                cell_vtx_list.push_back(idx + basal_vtx_start);

            positionsFromIndices(positions, cell_vtx_list);

            // cell-wise volume preservation term
            if (face_idx < basal_face_start)
            {
                T coeff = B * (current_cell_volume[face_idx] - cell_volume_init[face_idx]);

                if (face_vtx_list.size() == 4)
                {
                    Vector<T, 24> dedx;
                    if (use_cell_centroid)
                        computeVolume4PointsGradient(positions, dedx);
                    else
                        computeQuadBasePrismVolumeGradient(positions, dedx);
                    dedx *= -coeff;
                    addForceEntry<24>(residual, cell_vtx_list, dedx);
                }
                else if (face_vtx_list.size() == 5)
                {
                    Vector<T, 30> dedx;
                    if (use_cell_centroid)
                        computeVolume5PointsGradient(positions, dedx);
                    else
                        computePentaBasePrismVolumeGradient(positions, dedx);
                    dedx *= -coeff;
                    addForceEntry<30>(residual, cell_vtx_list, dedx);
                }
                else if (face_vtx_list.size() == 6)
                {
                    Vector<T, 36> dedx;
                    computeVolume6PointsGradient(positions, dedx);
                    dedx *= -coeff;
                    addForceEntry<36>(residual, cell_vtx_list, dedx);
                }
                else
                {
                    std::cout << "unknown polygon edge case" << std::endl;
                }
            }
        }
    });

    if (verbose)
        std::cout << "area force norm: " << (residual - residual_temp).norm() << std::endl;
    residual_temp = residual;

    iterateFaceSerial([&](VtxList& face_vtx_list, int face_idx)
    {
        // else // basal and lateral faces area term
        if (face_idx >= basal_face_start)
        {
            T coeff = face_idx >= lateral_face_start ? alpha : gamma;
            VectorXT positions;
            positionsFromIndices(positions, face_vtx_list);
            if (face_vtx_list.size() == 4)
            {
                Vector<T, 12> dedx;
                // computeArea4PointsGradient(coeff, positions, dedx);
                computeQuadFaceAreaGradient(coeff, positions, dedx);
                // dedx *= -coeff;
                dedx *=-1;
                addForceEntry<12>(residual, face_vtx_list, dedx);
            }
            else if (face_vtx_list.size() == 5)
            {
                Vector<T, 15> dedx;
                // computeArea5PointsGradient(coeff, positions, dedx);
                // dedx *= -coeff;
                computePentFaceAreaEnergyGradient(coeff, positions, dedx);
                dedx *= -1.0;
                addForceEntry<15>(residual, face_vtx_list, dedx);
            }
            else if (face_vtx_list.size() == 6)
            {
                Vector<T, 18> dedx;
                // computeArea6PointsGradient(coeff, positions, dedx);
                // dedx *= -coeff;
                computeHexFaceAreaEnergyGradient(coeff, positions, dedx);
                dedx *= -1.0;
                addForceEntry<18>(residual, face_vtx_list, dedx);
            }
            else
            {
                std::cout << "error " << __FILE__ << std::endl;
            }
        }
        // yolk volume preservation term
        if (add_yolk_volume)
        {
            if (face_idx < lateral_face_start && face_idx >= basal_face_start)
            {
                
                VectorXT positions;
                positionsFromIndices(positions, face_vtx_list);
                T coeff = By * (yolk_vol_curr - yolk_vol_init);
                if (face_vtx_list.size() == 4)
                {
                    Vector<T, 12> dedx;
                    // computeConeVolume4PointsGradient(positions, mesh_centroid, dedx);
                    computeQuadConeVolumeGradient(positions, mesh_centroid, dedx);
                    dedx *= -coeff;
                    addForceEntry<12>(residual, face_vtx_list, dedx);
                }
                else if (face_vtx_list.size() == 5)
                {
                    Vector<T, 15> dedx;
                    computeConeVolume5PointsGradient(positions, mesh_centroid, dedx);
                    dedx *= -coeff;
                    addForceEntry<15>(residual, face_vtx_list, dedx);
                }
                else if (face_vtx_list.size() == 6)
                {
                    Vector<T, 18> dedx;
                    computeConeVolume6PointsGradient(positions, mesh_centroid, dedx);
                    dedx *= -coeff;
                    addForceEntry<18>(residual, face_vtx_list, dedx);
                }
                else
                {
                    std::cout << "unknown polygon edge number" << std::endl;
                }
            }
        }
    });

    if (verbose)
        std::cout << "volume force norm: " << (residual - residual_temp).norm() << std::endl;
    residual_temp = residual;
    
    if (!run_diff_test)
        iterateDirichletDoF([&](int offset, T target)
        {
            residual[offset] = 0;
        });

    return residual.norm();
}

void VertexModel::positionsFromIndices(VectorXT& positions, const VtxList& indices)
{
    positions = VectorXT::Zero(indices.size() * 3);
    for (int i = 0; i < indices.size(); i++)
    {
        positions.segment<3>(i * 3) = deformed.segment<3>(indices[i] * 3);
    }
}

void VertexModel::buildSystemMatrix(const VectorXT& _u, StiffnessMatrix& K)
{
    VectorXT projected = _u;
    if (!run_diff_test)
    {
        iterateDirichletDoF([&](int offset, T target)
        {
            projected[offset] = target;
        });
    }
    deformed = undeformed + projected;

    std::vector<Entry> entries;
    
    // edge length term
    iterateApicalEdgeSerial([&](Edge& e){
        TV vi = deformed.segment<3>(e[0] * 3);
        TV vj = deformed.segment<3>(e[1] * 3);
        Matrix<T, 6, 6> hessian;
        computeEdgeSquaredNormHessian(vi, vj, hessian);
        hessian *= sigma;
        addHessianEntry<6>(entries, {e[0], e[1]}, hessian);
    });

    VectorXT current_cell_volume;
    computeVolumeAllCells(current_cell_volume);

    T yolk_vol_curr = 0.0;
    if (add_yolk_volume)
    {
        yolk_vol_curr = computeYolkVolume();
    }

    VectorXT dVdx_full = VectorXT::Zero(deformed.rows());

    iterateFaceSerial([&](VtxList& face_vtx_list, int face_idx)
    {
        if (add_yolk_volume)
        {
            if (face_idx < lateral_face_start && face_idx >= basal_face_start)
            {
                
                VectorXT positions;
                positionsFromIndices(positions, face_vtx_list);
                if (face_vtx_list.size() == 4)
                {
                    Vector<T, 12> dedx;
                    // computeConeVolume4PointsGradient(positions, mesh_centroid, dedx);
                    computeQuadConeVolumeGradient(positions, mesh_centroid, dedx);
                    addForceEntry<12>(dVdx_full, face_vtx_list, dedx);
                }
                else if (face_vtx_list.size() == 5)
                {
                    Vector<T, 15> dedx;
                    computeConeVolume5PointsGradient(positions, mesh_centroid, dedx);
                    addForceEntry<15>(dVdx_full, face_vtx_list, dedx);
                }
                else if (face_vtx_list.size() == 6)
                {
                    Vector<T, 18> dedx;
                    computeConeVolume6PointsGradient(positions, mesh_centroid, dedx);
                    addForceEntry<18>(dVdx_full, face_vtx_list, dedx);
                }
                else
                {
                    std::cout << "unknown polygon edge number" << std::endl;
                }
            }
        }
    });

    for (int dof_i = 0; dof_i < num_nodes; dof_i++)
    {
        for (int dof_j = 0; dof_j < num_nodes; dof_j++)
        {
            Vector<T, 6> dVdx;
            getSubVector<6>(dVdx_full, {dof_i, dof_j}, dVdx);
            TV dVdxi = dVdx.segment<3>(0);
            TV dVdxj = dVdx.segment<3>(3);
            Matrix<T, 3, 3> hessian_partial = By * dVdxi * dVdxj.transpose();
            addHessianBlock<3>(entries, {dof_i, dof_j}, hessian_partial);
        }
    }
    

    iterateFaceSerial([&](VtxList& face_vtx_list, int face_idx)
    {
        // cell-wise volume preservation term
        if (face_idx < basal_face_start)
        {
            VectorXT positions;
            VtxList cell_vtx_list = face_vtx_list;
            for (int idx : face_vtx_list)
                cell_vtx_list.push_back(idx + basal_vtx_start);

            positionsFromIndices(positions, cell_vtx_list);
            T V = current_cell_volume[face_idx];

            T coeff = B;
            if (face_vtx_list.size() == 4)
            {
                
                Matrix<T, 24, 24> d2Vdx2;
                if (use_cell_centroid)
                    computeVolume4PointsHessian(positions, d2Vdx2);
                else
                    computeQuadBasePrismVolumeHessian(positions, d2Vdx2);

                Vector<T, 24> dVdx;
                if (use_cell_centroid)
                    computeVolume4PointsGradient(positions, dVdx);
                else
                    computeQuadBasePrismVolumeGradient(positions, dVdx);
                    
                // break it down here to avoid super long autodiff code
                Matrix<T, 24, 24> hessian = B * (dVdx * dVdx.transpose() + 
                    (V - cell_volume_init[face_idx]) * d2Vdx2);

                addHessianEntry<24>(entries, cell_vtx_list, hessian);
            }
            else if (face_vtx_list.size() == 5)
            {
                Matrix<T, 30, 30> d2Vdx2;
                if (use_cell_centroid)
                    computeVolume5PointsHessian(positions, d2Vdx2);
                else 
                    computePentaBasePrismVolumeHessian(positions, d2Vdx2);

                Vector<T, 30> dVdx;
                if (use_cell_centroid)
                    computeVolume5PointsGradient(positions, dVdx);
                else
                    computePentaBasePrismVolumeGradient(positions, dVdx);
                
                // break it down here to avoid super long autodiff code
                Matrix<T, 30, 30> hessian = B * (dVdx * dVdx.transpose() + 
                    (V - cell_volume_init[face_idx]) * d2Vdx2);
                
                addHessianEntry<30>(entries, cell_vtx_list, hessian);
            }
            else if (face_vtx_list.size() == 6)
            {
                Matrix<T, 36, 36> d2Vdx2;
                computeVolume6PointsHessian(positions, d2Vdx2);

                Vector<T, 36> dVdx;
                computeVolume6PointsGradient(positions, dVdx);
                
                // break it down here to avoid super long autodiff code
                Matrix<T, 36, 36> hessian = B * (dVdx * dVdx.transpose() + 
                    (V - cell_volume_init[face_idx]) * d2Vdx2);
                
                addHessianEntry<36>(entries, cell_vtx_list, hessian);
            }
            else
            {
                std::cout << "unknown polygon edge case" << std::endl;
            }
            // std::cout << "Cell " << face_idx << std::endl;
        }
        else // basal and lateral faces area term
        {
            T coeff = face_idx >= lateral_face_start ? alpha : gamma;
            VectorXT positions;
            positionsFromIndices(positions, face_vtx_list);
            if (face_vtx_list.size() == 4)
            {
                Matrix<T, 12, 12> hessian;
                // computeArea4PointsHessian(coeff, positions, hessian);
                // hessian *= coeff;
                computeQuadFaceAreaHessian(coeff, positions, hessian);
                addHessianEntry<12>(entries, face_vtx_list, hessian);
            }
            else if (face_vtx_list.size() == 5)
            {
                Matrix<T, 15, 15> hessian;
                // computeArea5PointsHessian(coeff, positions, hessian);
                // hessian *= coeff;
                computePentFaceAreaEnergyHessian(coeff, positions, hessian);
                addHessianEntry<15>(entries, face_vtx_list, hessian);
            }
            else if (face_vtx_list.size() == 6)
            {
                Matrix<T, 18, 18> hessian;
                // computeArea6PointsHessian(coeff, positions, hessian);
                // hessian *= coeff;
                computeHexFaceAreaEnergyHessian(coeff, positions, hessian);
                addHessianEntry<18>(entries, face_vtx_list, hessian);
            }
            else
            {
                std::cout << "unknown " << std::endl;
            }
        }
        if (add_yolk_volume)
        {
            if (face_idx < lateral_face_start && face_idx >= basal_face_start)
            {
                
                VectorXT positions;
                positionsFromIndices(positions, face_vtx_list);
                if (face_vtx_list.size() == 4)
                {
                    
                    Matrix<T, 12, 12> d2Vdx2;
                    // computeConeVolume4PointsHessian(positions, mesh_centroid, d2Vdx2);
                    computeQuadConeVolumeHessian(positions, mesh_centroid, d2Vdx2);
                    Matrix<T, 12, 12> hessian = By * (yolk_vol_curr - yolk_vol_init) * d2Vdx2;
                    
                    addHessianEntry<12>(entries, face_vtx_list, hessian);
                }
                else if (face_vtx_list.size() == 5)
                {
                    Matrix<T, 15, 15> d2Vdx2;
                    computeConeVolume5PointsHessian(positions, mesh_centroid, d2Vdx2);

                    Matrix<T, 15, 15> hessian = By * (yolk_vol_curr - yolk_vol_init) * d2Vdx2;
                    
                    addHessianEntry<15>(entries, face_vtx_list, hessian);

                }
                else if (face_vtx_list.size() == 6)
                {
                    Matrix<T, 18, 18> d2Vdx2;
                    computeConeVolume6PointsHessian(positions, mesh_centroid, d2Vdx2);

                    Matrix<T, 18, 18> hessian = By * (yolk_vol_curr - yolk_vol_init) * d2Vdx2;
                    
                    addHessianEntry<18>(entries, face_vtx_list, hessian);
                }
                else
                {
                    std::cout << "unknown polygon edge case" << std::endl;
                }
            }
        }

        
    });

    

        
    K.resize(num_nodes * 3, num_nodes * 3);
    K.setFromTriplets(entries.begin(), entries.end());
    if (!run_diff_test)
        projectDirichletDoFMatrix(K, dirichlet_data);
    K.makeCompressed();
}

void VertexModel::projectDirichletDoFMatrix(StiffnessMatrix& A, 
    const std::unordered_map<int, T>& data)
{
    for (auto iter : data)
    {
        A.row(iter.first) *= 0.0;
        A.col(iter.first) *= 0.0;
        A.coeffRef(iter.first, iter.first) = 1.0;
    }

}

void VertexModel::faceHessianChainRuleTest()
{
    TV v0(0, 0, 0), v1(1, 0, 0), v2(1, 1, 0), v3(0, 1, 0);
    VectorXT positions(4 * 3);
    positions << 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0;
    TV centroid = 0.25 * (v0 + v1 + v2 + v3);

    auto updateCentroid = [&]()
    {
        centroid.setZero();

        for (int i = 0; i < 4; i++)
        {
            centroid += 0.25 * positions.segment<3>(i * 3);
        }
    };

    VtxList face_vtx_list = {0 ,1, 2, 3};
    auto computeEnergy = [&]()->T
    {
        updateCentroid();
        T energy  = 0.0;
        // for (int i = 0; i < face_vtx_list.size(); i++)
        // {
        //     int j = (i + 1) % face_vtx_list.size();
        //     TV vi = positions.segment<3>(face_vtx_list[i] * 3);
        //     TV vj = positions.segment<3>(face_vtx_list[j] * 3);

        //     T area = computeArea(vi, vj, centroid);
        //     energy += area;
        // }
        TV r0 = positions.segment<3>(0), r1 = positions.segment<3>(3), 
            r2 = positions.segment<3>(6), r3 = positions.segment<3>(9);
        energy = computeAreaFourPoints(r0, r1, r2, r3);
        return energy;
    };

    auto computeGradient = [&](VectorXT& residual)
    {
        updateCentroid();
        residual = VectorXT::Zero(positions.rows());
        TV dedc = TV::Zero();
        for (int i = 0; i < face_vtx_list.size(); i++)
        {
            int j = (i + 1) % face_vtx_list.size();
            TV vi = positions.segment<3>(face_vtx_list[i] * 3);
            TV vj = positions.segment<3>(face_vtx_list[j] * 3);
            Vector<T, 9> dedx;
            computeAreaGradient(vi, vj, centroid, dedx);
            
            // dedvij and dedcentroid/dcentroid dvij
            dedc += -dedx.segment<3>(6);
            addForceEntry<6>(residual, {face_vtx_list[i], face_vtx_list[j]}, -dedx.segment<6>(0));   
        }
        TM3 dcdx = TM3::Identity() / face_vtx_list.size();
        for (int vtx_id : face_vtx_list)
            addForceEntry<3>(residual, {vtx_id}, dedc.transpose() * dcdx );
    };

    auto computeGradientADFUll =[&](VectorXT& residual)
    {
        TV r0 = positions.segment<3>(0), r1 = positions.segment<3>(3), 
            r2 = positions.segment<3>(6), r3 = positions.segment<3>(9);
        Vector<T, 12> dedx;
        computeAreaFourPointsGradient(r0, r1, r2, r3, dedx);
        addForceEntry<12>(residual, face_vtx_list, -dedx);
    };

    auto hessianFullAutoDiff =[&](StiffnessMatrix& K)
    {
        TV r0 = positions.segment<3>(0), r1 = positions.segment<3>(3), 
            r2 = positions.segment<3>(6), r3 = positions.segment<3>(9);
        Matrix<T, 12, 12> d2edx2;
        computeAreaFourPointsHessian(r0, r1, r2, r3, d2edx2);
        std::vector<Entry> entries;
        addHessianEntry<12>(entries, face_vtx_list, d2edx2);
        K.resize(4 * 3, 4 * 3);
        K.setFromTriplets(entries.begin(), entries.end());
        K.makeCompressed();
    };
    
    auto computeHessianNPoints = [&](StiffnessMatrix& K)
    {
        Matrix<T, 12, 12> d2edx2;
        computeArea4PointsHessian(1.0, positions, d2edx2);
        std::vector<Entry> entries;
        addHessianEntry<12>(entries, face_vtx_list, d2edx2);
        K.resize(4 * 3, 4 * 3);
        K.setFromTriplets(entries.begin(), entries.end());
        K.makeCompressed();
    };

    auto computeHessian = [&](StiffnessMatrix& K)
    {
        updateCentroid();
        std::vector<Entry> entries;

        TM3 dcdx = TM3::Identity() / face_vtx_list.size();

        TM3 d2edc2_dcdvi_dcdvj = TM3::Zero();

        std::vector<std::vector<TM3>> deidcidv_list(face_vtx_list.size(), 
            std::vector<TM3>(face_vtx_list.size(), TM3::Zero()));
        
        for (int i = 0; i < face_vtx_list.size(); i++)
        {
            int j = (i + 1) % face_vtx_list.size();
            TV vi = positions.segment<3>(face_vtx_list[i] * 3);
            TV vj = positions.segment<3>(face_vtx_list[j] * 3);
            Matrix<T, 9, 9> sub_hessian;
            computeAreaHessian(vi, vj, centroid, sub_hessian);

            addHessianEntry<6>(entries, {face_vtx_list[i], face_vtx_list[j]}, 
                sub_hessian.block(0, 0, 6, 6));

            std::cout << sub_hessian << std::endl;
            std::cout << "--------------------------------" << std::endl;

            TM3 d2edcdx0 = sub_hessian.block(0, 6, 3, 3);
            TM3 d2edcdx1 = sub_hessian.block(3, 6, 3, 3);
            TM3 d2edc2 = sub_hessian.block(6, 6, 3, 3);

            std::cout << d2edcdx0 << std::endl;
            std::cout << "--------------------------------" << std::endl;
            std::cout << d2edcdx1 << std::endl;
            std::cout << "--------------------------------" << std::endl;
            std::cout << d2edc2 << std::endl;
            std::cout << "--------------------------------" << std::endl;
            std::getchar();
            
            d2edc2_dcdvi_dcdvj += d2edc2 * dcdx * dcdx;

            TM3 d2edcdx0_dcdx = d2edcdx0 * dcdx;
            TM3 d2edcdx1_dcdx = d2edcdx1 * dcdx;

            deidcidv_list[i][i] += d2edcdx0_dcdx;
            deidcidv_list[i][j] += d2edcdx1_dcdx;
        }
        // std::cout << d2edc2_dcdvi_dcdvj << std::endl;
        // std::getchar();
        for (int i = 0; i < face_vtx_list.size(); i++)
        {
            for (int j = 0; j < face_vtx_list.size(); j++)
            {
                // if (i == j)
                //      continue;
                TM3 chain_rule_term = d2edc2_dcdvi_dcdvj;
                // chain_rule_term.setZero();
                
                for (int k = 0; k < face_vtx_list.size(); k++)
                {
                    // d2e_k/dcdv_j
                    // std::cout <<deidcidv_list[k][j] << std::endl;
                    chain_rule_term += 1.0 * deidcidv_list[k][j];
                }
                // std::cout << d2edc2 * dcdx << std::endl;
                // std::cout << d2ecdx_list[i] << std::endl;
                // std::cout << chain_rule_term << std::endl;
                // std::cout << "--------------------------------" << std::endl;
                // std::getchar();
                addHessianBlock<3>(entries, {face_vtx_list[i], face_vtx_list[j]}, chain_rule_term);
            }
        }
        K.resize(4 * 3, 4 * 3);
        K.setFromTriplets(entries.begin(), entries.end());
        K.makeCompressed();
    };

    auto checkGradient = [&]()
    {
        int n_dof = 4 * 3;
        T epsilon = 1e-6;
        VectorXT gradient_FD(n_dof);
        gradient_FD.setZero();
        VectorXT gradient(n_dof);
        gradient.setZero();

        computeGradient(gradient);
        
        int cnt = 0;
        for(int dof_i = 0; dof_i < n_dof; dof_i++)
        {
            positions(dof_i) += epsilon;
            // std::cout << W * dq << std::endl;
            T E0 = computeEnergy();
            
            positions(dof_i) -= 2.0 * epsilon;
            T E1 = computeEnergy();
            positions(dof_i) += epsilon;
            // std::cout << "E1 " << E1 << " E0 " << E0 << std::endl;
            gradient_FD(dof_i) = (E1 - E0) / (2.0 *epsilon);
            if( gradient_FD(dof_i) == 0 && gradient(dof_i) == 0)
                continue;
            // if (std::abs( gradient_FD(d, n_node) - gradient(d, n_node)) < 1e-4)
            //     continue;
            std::cout << " dof " << dof_i << " " << gradient_FD(dof_i) << " " << gradient(dof_i) << std::endl;
            std::getchar();
            cnt++;   
        }
    };

    auto checkHessian = [&]()
    {
        T epsilon = 1e-6;
        int n_dof = 4 * 3;
        StiffnessMatrix A(n_dof, n_dof);
        computeHessian(A);

        StiffnessMatrix B = A;
        hessianFullAutoDiff(B);

        std::cout << A << std::endl;
        std::cout << "--------------------------------" << std::endl;
        std::cout << B << std::endl;
        std::getchar();

        for(int dof_i = 0; dof_i < n_dof; dof_i++)
        {
            positions(dof_i) += epsilon;
            VectorXT g0(n_dof), g1(n_dof);
            g0.setZero(); g1.setZero();
            computeGradient(g0);

            positions(dof_i) -= 2.0 * epsilon;
            computeGradient(g1);
            positions(dof_i) += epsilon;
            VectorXT row_FD = (g1 - g0) / (2.0 * epsilon);

            for(int i = 0; i < n_dof; i++)
            {
                if(A.coeff(dof_i, i) == 0 && row_FD(i) == 0)
                    continue;
                std::cout << "H(" << i << ", " << dof_i << ") " << " FD: " <<  row_FD(i) << " symbolic: " << A.coeff(i, dof_i) << std::endl;
                std::getchar();
            }
        }
    };

    // VectorXT g0, g1;
    // computeGradient(g0);
    // computeGradient(g1);
    // std::cout << g0.transpose() << std::endl;
    // std::cout << "--------------------------------" << std::endl;
    // std::cout << g1.transpose() << std::endl;
    // std::getchar();

    // checkGradient();
    checkHessian();
}

void VertexModel::checkTotalGradient()
{
    // sigma = 0; alpha = 0; gamma = 0; B = 0; 
    
    run_diff_test = true;
    VectorXT du(num_nodes * 3);
    du.setRandom();
    du *= 1.0 / du.norm();
    du *= 0.001;
    u += du;

    std::cout << "======================== CHECK GRADIENT ========================" << std::endl;
    int n_dof = num_nodes * 3;
    T epsilon = 1e-6;
    VectorXT gradient(n_dof);
    gradient.setZero();

    computeResidual(u, gradient);

    // std::cout << gradient.transpose() << std::endl;
    
    VectorXT gradient_FD(n_dof);
    gradient_FD.setZero();

    int cnt = 0;
    for(int dof_i = 0; dof_i < n_dof; dof_i++)
    {
        u(dof_i) += epsilon;
        // std::cout << W * dq << std::endl;
        T E0 = computeTotalEnergy(u);
        
        u(dof_i) -= 2.0 * epsilon;
        T E1 = computeTotalEnergy(u);
        u(dof_i) += epsilon;
        // std::cout << "E1 " << E1 << " E0 " << E0 << std::endl;
        gradient_FD(dof_i) = (E1 - E0) / (2.0 *epsilon);
        if( gradient_FD(dof_i) == 0 && gradient(dof_i) == 0)
            continue;
        // if (std::abs( gradient_FD(d, n_node) - gradient(d, n_node)) < 1e-4)
        //     continue;
        std::cout << " dof " << dof_i << " " << gradient_FD(dof_i) << " " << gradient(dof_i) << std::endl;
        std::getchar();
        cnt++;   
    }
    run_diff_test = false;
}

void VertexModel::checkTotalHessian()
{
    // sigma = 0; alpha = 0; gamma = 0; B = 0; 

    std::cout << "======================== CHECK HESSIAN ========================" << std::endl;
    run_diff_test = true;
    T epsilon = 1e-7;
    int n_dof = num_nodes * 3;
    
    VectorXT du(num_nodes * 3);
    du.setRandom();
    du *= 1.0 / du.norm();
    du *= 0.001;
    u += du;

    StiffnessMatrix A(n_dof, n_dof);
    buildSystemMatrix(u, A);
    // std::cout << "Full hessian" << std::endl;
    // std::cout << A << std::endl;
    for(int dof_i = 0; dof_i < n_dof; dof_i++)
    {
        u(dof_i) += epsilon;
        VectorXT g0(n_dof), g1(n_dof);
        g0.setZero(); g1.setZero();
        computeResidual(u, g0);

        u(dof_i) -= 2.0 * epsilon;
        computeResidual(u, g1);
        u(dof_i) += epsilon;
        VectorXT row_FD = (g1 - g0) / (2.0 * epsilon);

        for(int i = 0; i < n_dof; i++)
        {
            if(A.coeff(dof_i, i) == 0 && row_FD(i) == 0)
                continue;
            // if (std::abs( A.coeff(dof_i, i) - row_FD(i)) < 1e-4)
            //     continue;
            // std::cout << "node i: "  << std::floor(dof_i / T(dof)) << " dof " << dof_i%dof 
            //     << " node j: " << std::floor(i / T(dof)) << " dof " << i%dof 
            //     << " FD: " <<  row_FD(i) << " symbolic: " << A.coeff(i, dof_i) << std::endl;
            std::cout << "H(" << i << ", " << dof_i << ") " << " FD: " <<  row_FD(i) << " symbolic: " << A.coeff(i, dof_i) << std::endl;
            std::getchar();
        }
    }
    run_diff_test = false;
}

void VertexModel::checkTotalGradientScale()
{
    
    run_diff_test = true;
    std::cout << "======================== CHECK GRADIENT 2nd Scale ========================" << std::endl;
    T epsilon = 1e-6;
    VectorXT du(num_nodes * 3);
    du.setRandom();
    du *= 1.0 / du.norm();
    du *= 0.001;
    u += du;
    
    int n_dof = num_nodes * 3;

    VectorXT gradient(n_dof);
    gradient.setZero();
    computeResidual(u, gradient);
    
    gradient *= -1;
    T E0 = computeTotalEnergy(u);
    VectorXT dx(n_dof);
    dx.setRandom();
    dx *= 1.0 / dx.norm();
    dx *= 0.001;
    T previous = 0.0;
    for (int i = 0; i < 10; i++)
    {
        T E1 = computeTotalEnergy(u + dx);
        T dE = E1 - E0;
        
        dE -= gradient.dot(dx);
        // std::cout << "dE " << dE << std::endl;
        if (i > 0)
        {
            std::cout << (previous/dE) << std::endl;
        }
        previous = dE;
        dx *= 0.5;
    }
    run_diff_test = false;
}

void VertexModel::checkTotalHessianScale()
{
    run_diff_test = true;
    
    std::cout << "===================== check Hessian 2nd Scale =====================" << std::endl;

    VectorXT du(num_nodes * 3);
    du.setRandom();
    du *= 1.0 / du.norm();
    du *= 0.001;
    u += du;
    
    int n_dof = num_nodes * 3;

    StiffnessMatrix A;
    buildSystemMatrix(u, A);

    VectorXT f0(n_dof);
    f0.setZero();
    computeResidual(u, f0);
    f0 *= -1;
    
    VectorXT dx(n_dof);
    dx.setRandom();
    dx *= 1.0 / dx.norm();
    for(int i = 0; i < n_dof; i++) dx[i] += 0.5;
    dx *= 0.001;
    T previous = 0.0;
    for (int i = 0; i < 10; i++)
    {
        
        VectorXT f1(n_dof);
        f1.setZero();
        computeResidual(u + dx, f1);
        f1 *= -1;
        T df_norm = (f0 + (A * dx) - f1).norm();
        // std::cout << "df_norm " << df_norm << std::endl;
        if (i > 0)
        {
            std::cout << (previous/df_norm) << std::endl;
        }
        previous = df_norm;
        dx *= 0.5;
    }
    run_diff_test = false;
}