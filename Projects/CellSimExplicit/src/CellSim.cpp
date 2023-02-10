
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/edges.h>
#include <igl/vertex_triangle_adjacency.h>
#include <igl/per_face_normals.h>
#include <Eigen/CholmodSupport>
#include "../include/CellSim.h"

bool CellSim::advanceOneStep(int step)
{
    Timer step_timer(true);
    iterateDirichletDoF([&](int offset, T target)
    {
        // f[offset] = 0;
    });

    VectorXT residual(deformed.rows());
    residual.setZero();
    
    if (use_ipc)
    {
        updateBarrierInfo(step == 0);
        updateIPCVertices(u);
    }

    T residual_norm = computeResidual(u, residual);
    std::cout << "[Newton] computeResidual takes " << step_timer.elapsed_sec() << "s" << std::endl;
    step_timer.restart();
    // if (save_mesh)
    //     saveCellMesh(step);
    // std::cout << "[Newton] saveCellMesh takes " << step_timer.elapsed_sec() << "s" << std::endl;
    // if (verbose)
    std::cout << "[Newton] iter " << step << "/" << max_newton_iter << ": residual_norm " << residual.norm() << " tol: " << newton_tol << std::endl;

    if (residual_norm < newton_tol)
        return true;

    T dq_norm = lineSearchNewton(u, residual);
    
    step_timer.stop();
    if (verbose)
        std::cout << "[Newton] step takes " << step_timer.elapsed_sec() << "s" << std::endl;

    if(step == max_newton_iter || dq_norm > 1e10)
        return true;
    
    return false; 
}

T CellSim::computeLineSearchInitStepsize(const VectorXT& _u, const VectorXT& du)
{
    if (verbose)
        std::cout << "** step size **" << std::endl;
    T step_size = 1.0;
    if (use_ipc)
    {
        T ipc_step_size = computeCollisionFreeStepsize(_u, du);
        step_size = std::min(step_size, ipc_step_size);
        if (verbose)
            std::cout << "after ipc step size: " << step_size << std::endl;
    }

    // if (use_sphere_radius_bound && !sphere_bound_penalty && !use_sdf_boundary)
    // {
    //     T inside_membrane_step_size = computeInsideMembraneStepSize(_u, du);
    //     step_size = std::min(step_size, inside_membrane_step_size);
    //     if (verbose)
    //         std::cout << "after inside membrane step size: " << step_size << std::endl;
    // }

    // if (add_tet_vol_barrier)
    // {
    //     T inversion_free_step_size = computeInversionFreeStepSize(_u, du);
    //     // std::cout << "cell tet inversion free step size: " << inversion_free_step_size << std::endl;
    //     step_size = std::min(step_size, inversion_free_step_size);
    //     if (verbose)
    //         std::cout << "after tet inverison step size: " << step_size << std::endl;
    // }

    // if (add_yolk_tet_barrier)
    // {
    //     T inversion_free_step_size = computeYolkInversionFreeStepSize(_u, du);
    //     // std::cout << "yolk inversion free step size: " << inversion_free_step_size << std::endl;
    //     step_size = std::min(step_size, inversion_free_step_size);
    // }
    if (verbose)
        std::cout << "**       **" << std::endl;
    return step_size;
}

bool CellSim::solveWoodburyCholmod(StiffnessMatrix& K, MatrixXT& UV,
         VectorXT& residual, VectorXT& du)
{
    MatrixXT UVT= UV.transpose();
    
    Timer t(true);
    Eigen::CholmodSupernodalLLT<StiffnessMatrix, Eigen::Lower> solver;
    
    T alpha = 10e-6;
    solver.analyzePattern(K);
    // T time_analyze = t.elapsed_sec();
    // std::cout << "\t analyzePattern takes " << time_analyze << "s" << std::endl;
    int i = 0;
    int indefinite_count_reg_cnt = 0, invalid_search_dir_cnt = 0, invalid_residual_cnt = 0;
    for (; i < 50; i++)
    {
        // std::cout << i << std::endl;
        solver.factorize(K);
        // T time_factorize = t.elapsed_sec() - time_analyze;
        // std::cout << "\t factorize takes " << time_factorize << "s" << std::endl;
        // std::cout << "-----factorization takes " << t.elapsed_sec() << "s----" << std::endl;
        if (solver.info() == Eigen::NumericalIssue)
        {
            K.diagonal().array() += alpha;
            alpha *= 10;
            indefinite_count_reg_cnt++;
            continue;
        }

        // sherman morrison
        if (UV.cols() == 1)
        {
            VectorXT v = UV.col(0);
            MatrixXT rhs(K.rows(), 2); rhs.col(0) = residual; rhs.col(1) = v;
            // VectorXT A_inv_g = solver.solve(residual);
            // VectorXT A_inv_u = solver.solve(v);
            MatrixXT A_inv_gu = solver.solve(rhs);

            T dem = 1.0 + v.dot(A_inv_gu.col(1));

            du.noalias() = A_inv_gu.col(0) - (A_inv_gu.col(0).dot(v)) * A_inv_gu.col(1) / dem;
        }
        // UV is actually only U, since UV is the same in the case
        // C is assume to be Identity
        else // Woodbury https://en.wikipedia.org/wiki/Woodbury_matrix_identity
        {
            VectorXT A_inv_g = solver.solve(residual);

            MatrixXT A_inv_U(UV.rows(), UV.cols());
            // for (int col = 0; col < UV.cols(); col++)
                // A_inv_U.col(col) = solver.solve(UV.col(col));
            A_inv_U = solver.solve(UV);

            MatrixXT C(UV.cols(), UV.cols());
            C.setIdentity();
            C += UVT * A_inv_U;
            du = A_inv_g - A_inv_U * C.inverse() * UVT * A_inv_g;
        }
        

        T dot_dx_g = du.normalized().dot(residual.normalized());

        int num_negative_eigen_values = 0;
        int num_zero_eigen_value = 0;

        bool positive_definte = num_negative_eigen_values == 0;
        bool search_dir_correct_sign = dot_dx_g > 1e-3;
        if (!search_dir_correct_sign)
            invalid_search_dir_cnt++;
        
        bool solve_success = true;//(K * du + UV * UV.transpose()*du - residual).norm() < 1e-6 && solver.info() == Eigen::Success;
        
        if (!solve_success)
            invalid_residual_cnt++;
        // std::cout << "PD: " << positive_definte << " direction " 
        //     << search_dir_correct_sign << " solve " << solve_success << std::endl;

        if (positive_definte && search_dir_correct_sign && solve_success)
        {
            t.stop();
            if (verbose)
            {
                std::cout << "\t===== Linear Solve ===== " << std::endl;
                std::cout << "\tnnz: " << K.nonZeros() << std::endl;
                std::cout << "\t takes " << t.elapsed_sec() << "s" << std::endl;
                std::cout << "\t# regularization step " << i 
                    << " indefinite " << indefinite_count_reg_cnt 
                    << " invalid search dir " << invalid_search_dir_cnt
                    << " invalid solve " << invalid_residual_cnt << std::endl;
                std::cout << "\tdot(search, -gradient) " << dot_dx_g << std::endl;
                // std::cout << (K.selfadjointView<Eigen::Lower>() * du + UV * UV.transpose()*du - residual).norm() << std::endl;
                std::cout << "\t======================== " << std::endl;
            }
            return true;
        }
        else
        {
            // K = H + alpha * I;       
            // tbb::parallel_for(0, (int)K.rows(), [&](int row)    
            // {
            //     K.coeffRef(row, row) += alpha;
            // });  
            K.diagonal().array() += alpha;
            alpha *= 10;
        }
    }
    return false;
}

void CellSim::projectDirichletDoFMatrix(StiffnessMatrix& A, 
    const std::unordered_map<int, T>& data)
{
    for (auto iter : data)
    {
        A.row(iter.first) *= 0.0;
        A.col(iter.first) *= 0.0;
        A.coeffRef(iter.first, iter.first) = 1.0;
    }

}

void CellSim::buildSystemMatrixWoodbury(const VectorXT& _u, StiffnessMatrix& K, MatrixXT& UV)
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

    addEdgeHessianEntries(ALL, w_edges, entries, project_block_hessian_PD);
    addFaceAreaHessianEntries(ALL, w_faces, entries, project_block_hessian_PD);

    addCellVolumePreservationHessianEntries(entries, project_block_hessian_PD);

    if (add_yolk_volume)
        addYolkVolumePreservationHessianEntries(entries, UV, project_block_hessian_PD);

    if (use_ipc)
        addIPCHessianEntries(entries, project_block_hessian_PD);

    K.resize(num_nodes * 3, num_nodes * 3);
    K.reserve(0.5 * entries.size());
    K.setFromTriplets(entries.begin(), entries.end());
    
    if (!run_diff_test)
        projectDirichletDoFMatrix(K, dirichlet_data);
    
    K.makeCompressed();
}

void CellSim::buildSystemMatrix(const VectorXT& _u, StiffnessMatrix& K)
{
    

}

T CellSim::computeTotalEnergy(const VectorXT& _u, bool add_to_deform)
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
    if (add_to_deform)
    {
        deformed = undeformed + projected;
    }

    T edge_length_term = 0.0, area_term = 0.0, 
        volume_term = 0.0, yolk_volume_term = 0.0,
        contraction_term = 0.0, membrane_bound_term = 0.0;

    addEdgeEnergy(ALL, w_edges, edge_length_term);
    addFaceAreaEnergy(ALL, w_faces, area_term);
    energy += edge_length_term;
    energy += area_term;
    // ===================================== Cell Volume =====================================
    addCellVolumePreservationEnergy(volume_term);
    energy += volume_term;

    if (add_yolk_volume)
    {
        addYolkVolumePreservationEnergy(yolk_volume_term);
        if (verbose)
            std::cout << "\tE_yolk_vol " << yolk_volume_term << std::endl;
    }
    energy += yolk_volume_term;

    T contact_energy = 0.0;

    if (use_ipc)
    {
        addIPCEnergy(contact_energy);
        if (verbose)
            std::cout << "\tE_contact: " << contact_energy << std::endl;
        energy += contact_energy;
        // std::getchar();
    }

    
    return energy;
}
T CellSim::computeResidual(const VectorXT& _u,  VectorXT& residual)
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

    addEdgeForceEntries(ALL, w_edges, residual);
    if (print_force_norm)
        std::cout << "\tall edges contraction force norm: " << (residual - residual_temp).norm() << std::endl;
    residual_temp = residual;
    addFaceAreaForceEntries(ALL, w_faces, residual);
    if (print_force_norm)
        std::cout << "\tface area force norm: " << (residual - residual_temp).norm() << std::endl;
    residual_temp = residual;

    // ===================================== Cell Volume =====================================
    addCellVolumePreservationForceEntries(residual);
    if (print_force_norm)
        std::cout << "\tcell volume preservation force norm: " << (residual - residual_temp).norm() << std::endl;
    residual_temp = residual;
    // ===================================== Yolk =====================================

    if (add_yolk_volume)
    {
        addYolkVolumePreservationForceEntries(residual);
        if (print_force_norm)
            std::cout << "\tyolk volume preservation force norm: " << (residual - residual_temp).norm() << std::endl;
        residual_temp = residual;
    }

    if (use_ipc)
    {
        addIPCForceEntries(residual);
    
        if(print_force_norm)
            std::cout << "\tcontact force norm: " << (residual - residual_temp).norm() << std::endl;
        residual_temp = residual;
    }

    if (!run_diff_test)
        iterateDirichletDoF([&](int offset, T target)
        {
            residual[offset] = 0;
        });

    return residual.norm();
}

T CellSim::lineSearchNewton(VectorXT& _u,  VectorXT& residual, int ls_max)
{
    VectorXT du = residual;
    du.setZero();
    StiffnessMatrix K(residual.rows(), residual.rows());
    
    bool success = false;
    Timer ti(true);

    MatrixXT UV;
    if (woodbury)
    {
        MatrixXT UV;
        buildSystemMatrixWoodbury(_u, K, UV);
        success = solveWoodburyCholmod(K, UV, residual, du); 
    }
    else
    {
        buildSystemMatrix(_u, K);
        // success = linearSolve(K, residual, du);    
    }
    if (!success)
    {
        std::cout << "linear solve failed" << std::endl;
        return 1e16;
    }
    

    T norm = du.norm();

    T E0 = computeTotalEnergy(_u);
    T alpha = computeLineSearchInitStepsize(_u, du);
    int cnt = 1;
    while (true)
    {
        VectorXT u_ls = _u + alpha * du;
        T E1 = computeTotalEnergy(u_ls);
        if (E1 - E0 < 0 || cnt > ls_max)
        {
            _u = u_ls;
            if (cnt > ls_max)
                if (verbose)
                    std::cout << "---ls max---" << std::endl;
            if (verbose)
                std::cout << "# ls " << cnt << " |du| " << alpha * du.norm() << std::endl;
            break;
        }
        cnt++;
    }
    return norm;
}

void CellSim::positionsFromIndices(VectorXT& positions, const VtxList& indices, bool rest_state)
{
    positions = VectorXT::Zero(indices.size() * 3);
    for (int i = 0; i < indices.size(); i++)
    {
        positions.segment<3>(i * 3) = rest_state ? undeformed.segment<3>(indices[i] * 3) : deformed.segment<3>(indices[i] * 3);
    }
}

void CellSim::computeCentroid(const VectorXT& positions, TV& centroid)
{
    centroid = TV::Zero();
    int n_vtx = positions.rows() / 3;
    for (int i = 0; i < n_vtx; i++)
        centroid += positions.segment<3>(i * 3);
    centroid /= n_vtx;
}

void CellSim::generateMeshForRendering(
        Eigen::MatrixXd& V, Eigen::MatrixXi& F, 
        Eigen::MatrixXd& C, bool yolk_only, bool cells_only)
{
    std::vector<IV> tri_faces;
    std::vector<TV> vertices;
    std::vector<TV> colors;
    int face_cnt = 0, vtx_cnt = 0;
    if (!yolk_only)
        iterateCellSerial([&](VtxList& vtx_list, int cell_idx)
        {

            VectorXT positions;
            positionsFromIndices(positions, vtx_list);
            int n_vtx_half = vtx_list.size() / 2;
            
            VectorXT positions_basal = positions.segment(n_vtx_half * 3, n_vtx_half * 3);
            VectorXT positions_apical = positions.segment(0, n_vtx_half * 3);
            
            TV apical_centroid, basal_centroid;
            computeCentroid(positions_basal, basal_centroid);
            computeCentroid(positions_apical, apical_centroid);
            
            
            VtxList new_face_vtx;
            vertices.push_back(apical_centroid);
            new_face_vtx.push_back(vtx_cnt);
            for (int i = 0; i < n_vtx_half; i++)
                new_face_vtx.push_back(vtx_cnt + i + 1);
            vtx_cnt++;
            for (int i = 0; i < n_vtx_half; i++)
            {
                int j = (i + 1) % n_vtx_half;
                vertices.push_back(positions_apical.segment<3>(i * 3));
                colors.push_back(Eigen::Vector3d(1.0, 0.3, 0.0));
                tri_faces.push_back(IV(new_face_vtx[1 + i], new_face_vtx[0], new_face_vtx[1 + j]));
                vtx_cnt++;
            }

            VtxList new_face_vtx_basal;
            vertices.push_back(basal_centroid);
            new_face_vtx_basal.push_back(vtx_cnt);
            for (int i = 0; i < n_vtx_half; i++)
                new_face_vtx_basal.push_back(vtx_cnt + i + 1);
            vtx_cnt++;
            for (int i = 0; i < n_vtx_half; i++)
            {
                int j = (i + 1) % n_vtx_half;
                vertices.push_back(positions_basal.segment<3>(i * 3));
                colors.push_back(Eigen::Vector3d(0.0, 1.0, 0.0));
                tri_faces.push_back(IV(new_face_vtx_basal[0], new_face_vtx_basal[1 + i], new_face_vtx_basal[1 + j]));
                vtx_cnt++;
            }

            for (int i = 0; i < n_vtx_half; i++)
            {
                int j = (i + 1) % n_vtx_half;
                tri_faces.push_back(IV(new_face_vtx[1 + i], new_face_vtx[1 + j], new_face_vtx_basal[ 1 + j]));
                tri_faces.push_back(IV(new_face_vtx[1 + i], new_face_vtx_basal[ 1 + j], new_face_vtx_basal[ 1 + i]));
                colors.push_back(Eigen::Vector3d(0.0, 0.3, 1.0));
                colors.push_back(Eigen::Vector3d(0.0, 0.3, 1.0));
            }
        });
    if (!cells_only)
        for (auto yolk_cell : yolk_cells)
        {
            VectorXT positions;
            positionsFromIndices(positions, yolk_cell);
            // std::cout << yolk_cell.back() << " " << yolk_cell.front()<< " " << deformed.rows() << std::endl;
            // std::cout << positions.transpose() << std::endl;
            // std::getchar();
            TV centroid;
            computeCentroid(positions, centroid);
            VtxList new_face_vtx;
            vertices.push_back(centroid);
            new_face_vtx.push_back(vtx_cnt);
            for (int i = 0; i < yolk_cell.size(); i++)
                new_face_vtx.push_back(vtx_cnt + i + 1);
            vtx_cnt++;
            for (int i = 0; i < yolk_cell.size(); i++)
            {
                int j = (i + 1) % yolk_cell.size();
                vertices.push_back(positions.segment<3>(i * 3));
                colors.push_back(Eigen::Vector3d(0.0, 1.0, 0.0));
                tri_faces.push_back(IV(new_face_vtx[1 + i], new_face_vtx[0], new_face_vtx[1 + j]));
                vtx_cnt++;
            }
        }

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

void CellSim::initializeCells()
{
    std::string surface_mesh_file = "/home/yueli/Documents/ETH/WuKong/Projects/CellSim/data/drosophila_real_124_remesh.obj";
    constructCellMeshFromFile(surface_mesh_file);

}

void CellSim::constructCellMeshFromFile(const std::string& filename)
{
    Eigen::MatrixXd V, N;
    Eigen::MatrixXi F;
    igl::readOBJ(filename, V, F);
    
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

    int basal_vtx_start = deformed.size() / 3;
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

    T cell_height = avg_edge_norm;

    TV min_corner, max_corner;
    
    computeBoundingBox(min_corner, max_corner);
    std::cout << "BBOX: " << min_corner.transpose() << " " << max_corner.transpose() << std::endl;

    VectorXT percentage(basal_vtx_start);
    T a = 3.0, c = 1.0;
    
    for (int i = 0; i < basal_vtx_start; i++)
    {
        TV apex = deformed.segment<3>(i * 3);
        
        T x = (apex[0] - mesh_center[0]) / (max_corner[0] - min_corner[0]);
        T curved = -a * std::pow(x, 2.0) + c;
        
        deformed.segment<3>(basal_vtx_start * 3 + i * 3) =
                deformed.segment<3>(i * 3) - 
                vtx_normals.segment<3>(i * 3) * curved * cell_height;
    }
    undeformed = deformed;
    // std::cout << undeformed.segment(basal_vtx_start * 3, basal_vtx_start * 3).transpose() << std::endl;
    // std::cout << deformed.rows() << " " << basal_vtx_start * 3 << std::endl;
    // std::exit(0);
    int node_cnt = 0;
    
    std::vector<TV> new_vertices;
    T shrink = 0.96;
    for (auto one_ring_face : faces)
    {
        std::vector<TV> cell_vertices;
        
        cell_vtx_start.push_back(node_cnt);
        for (int i = 0; i < one_ring_face.size(); i++)
        {
            int next = (i + 1) % one_ring_face.size();
            cell_vertices.push_back(deformed.segment<3>(one_ring_face[i]* 3));
            node_cnt ++;
        }
        for (int i = 0; i < one_ring_face.size(); i++)
        {
            cell_vertices.push_back(deformed.segment<3>(basal_vtx_start * 3 + one_ring_face[i]* 3));
            node_cnt ++;
        }
        
        TV centroid = TV::Zero();
        for (const TV& vtx : cell_vertices)
        {
            centroid += vtx;
        }
        centroid /= T(cell_vertices.size());
        for (TV& vtx : cell_vertices)
        {
            vtx = centroid + shrink * (vtx - centroid);
        }
        new_vertices.insert(new_vertices.end(), cell_vertices.begin(), cell_vertices.end());
    }
    deformed.resize(new_vertices.size() * 3);
    for (int i = 0; i < new_vertices.size(); i++)
        deformed.segment<3>(i * 3) = new_vertices[i];
    num_nodes = new_vertices.size();
    yolk_vtx_start = new_vertices.size();

    for (auto one_ring_face : faces)
    {
        VtxList yolk_cell;
        for (int i = 0; i < one_ring_face.size(); i++)
        {
            yolk_cell.push_back(yolk_vtx_start + one_ring_face[i]);
        }
        yolk_cells.push_back(yolk_cell);
    }

    faces.resize(0);
    // iterate cells to get edges
    for (int i = 0; i < cell_vtx_start.size(); i++)
    {
        VtxList vtx_list;
        VtxList apical_list, basal_list, lateral_list;
        if (i < cell_vtx_start.size() - 1)
        {
            for (int j = cell_vtx_start[i]; j < cell_vtx_start[i + 1]; j++)
                vtx_list.push_back(j);
        }
        else
        {
            for (int j = cell_vtx_start[i]; j < yolk_vtx_start; j++)
                vtx_list.push_back(j);
        }
        int n_vtx_half = vtx_list.size() / 2;
        for (int i = 0; i < n_vtx_half; i++)
        {
            int j = (i + 1) % n_vtx_half;
            cell_edges.push_back(Edge(vtx_list[i], vtx_list[j]));
            apical_list.push_back(vtx_list[i]);
        }
        for (int i = 0; i < n_vtx_half; i++)
        {
            int j = (i + 1) % n_vtx_half;
            cell_edges.push_back(Edge(vtx_list[i + n_vtx_half], vtx_list[j + n_vtx_half]));
            basal_list.push_back(vtx_list[i + n_vtx_half]);
        }
        faces.push_back(apical_list);
        faces.push_back(basal_list);
        for (int i = 0; i < n_vtx_half; i++)
        {
            int j = (i + 1) % n_vtx_half;
            cell_edges.push_back(Edge(vtx_list[i], vtx_list[i + n_vtx_half]));
            faces.push_back({vtx_list[i], vtx_list[j],
                                vtx_list[j + n_vtx_half], vtx_list[i + n_vtx_half]});
        }
    }

    deformed.conservativeResize(num_nodes * 3 + basal_vtx_start * 3);
    deformed.segment(num_nodes * 3, basal_vtx_start * 3) = undeformed.segment(basal_vtx_start * 3, basal_vtx_start * 3);
    
    num_nodes = deformed.rows() / 3;
    std::cout << "num_nodes " << num_nodes << std::endl;
    undeformed = deformed;
    u = VectorXT::Zero(num_nodes * 3);
    num_cells = cell_vtx_start.size();

    
    yolk_vol_init = computeYolkVolume();

    computeVolumeAllCells(cell_volume_init);

    computeRestLength();
    use_ipc = true;
    if (use_ipc)
    {
        barrier_distance = 1e-5;
        computeIPCRestData();
        saveIPCData("./", 0, true);
    }

    print_force_norm = false;
    verbose = true;    
    max_newton_iter = 500;
    std::cout << "initialization done" << std::endl;
}