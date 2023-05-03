
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/edges.h>
#include <igl/vertex_triangle_adjacency.h>
#include <igl/per_face_normals.h>
#include <Eigen/CholmodSupport>
#include "../include/CellSim.h"
#include "../include/Timer.h"

template <int dim>
void CellSim<dim>::advanceOneFrame()
{
    updatePerFrameData();
    staticSolve();
    saveState("results/" + std::to_string(global_frame)+".obj", deformed);
}

template <int dim>
void CellSim<dim>::loadStates(const std::string& filename)
{
    std::ifstream in(filename);
    std::string vtx;
    T x, y, z;
    std::vector<T> coords;
    while (in >> vtx >> x >> y >> z)
    {
        coords.push_back(x);
        coords.push_back(y);
        coords.push_back(z);
    }
    deformed = Eigen::Map<VectorXT>(coords.data(), coords.size());
    u.setZero();
}

template <int dim>
bool CellSim<dim>::staticSolve()
{
    int cnt = 0;
    T residual_norm = 1e10, dq_norm = 1e10;
    T residual_norm_init = 0.0;
    while (true)
    {
        cell_hash.build(2.0 * collision_dhat, deformed);
        VectorXT residual(deformed.rows());
        residual.setZero();
        if (use_ipc)
        {
            updateBarrierInfo(cnt == 0);
            if (verbose)
                std::cout << "ipc barrier stiffness " << ipc_barrier_weight << std::endl;
            
            updateIPCVertices(u);
        }
        residual_norm = computeResidual(u, residual);
        if (cnt == 0)
            residual_norm_init = residual_norm;
        
        if (verbose)
            std::cout << "iter " << cnt << "/" << max_newton_iter << ": residual_norm " << residual.norm() << " tol: " << newton_tol << std::endl;


        if (residual_norm < newton_tol)
            break;

        dq_norm = lineSearchNewton(u, residual);

        if(cnt == max_newton_iter || dq_norm > 1e10 || dq_norm < 1e-12)
            break;
        cnt++;
    }
    iterateDirichletDoF([&](int offset, T target)
    {
        u[offset] = target;
    });

    deformed = undeformed + u;
    std::cout << "# of newton iter: " << cnt << " exited with |g|: " 
            << residual_norm << " |ddu|: " << dq_norm  
            << " |g_init|: " << residual_norm_init << std::endl;
    return true;
}

template <int dim>
bool CellSim<dim>::advanceOneStep(int step)
{
    Timer step_timer(true);
    iterateDirichletDoF([&](int offset, T target)
    {
        // f[offset] = 0;
    });

    cell_hash.build(2.0 * collision_dhat, deformed);

    VectorXT residual(deformed.rows());
    residual.setZero();
    if (use_ipc)
    {
        updateBarrierInfo(step == 0);
        std::cout << "ipc barrier stiffness " << ipc_barrier_weight << std::endl;
        
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
    {
        T yolk_volume = computeYolkArea();
        std::cout << "yolk volume init: " << yolk_area_rest << " " << yolk_volume << std::endl;
        return true;
    }

    T dq_norm = lineSearchNewton(u, residual);
    
    step_timer.stop();
    if (verbose)
        std::cout << "[Newton] step takes " << step_timer.elapsed_sec() << "s" << std::endl;

    if(step == max_newton_iter || dq_norm > 1e10)
    {
        
        return true;
    }
    
    return false; 
}

template <int dim>
T CellSim<dim>::computeLineSearchInitStepsize(const VectorXT& _u, const VectorXT& du)
{
    if (verbose)
        std::cout << "** step size **" << std::endl;
    T step_size = 1.0;
    if (use_ipc)
    {
        step_size = std::min(computeCollisionFreeStepsize(_u, du), step_size);
        if (verbose)
            std::cout << "step size after ipc " << step_size << std::endl;
    }
    if (verbose)
        std::cout << "**       **" << std::endl;
    return step_size;
}

template <int dim>
void CellSim<dim>::projectDirichletDoFMatrix(StiffnessMatrix& A, 
    const std::unordered_map<int, T>& data)
{
    for (auto iter : data)
    {
        A.row(iter.first) *= 0.0;
        A.col(iter.first) *= 0.0;
        A.coeffRef(iter.first, iter.first) = 1.0;
    }

}

template <int dim>
bool CellSim<dim>::solveWoodbury(StiffnessMatrix& K, MatrixXT& UV,
         VectorXT& residual, VectorXT& du)
{
    MatrixXT UVT= UV.transpose();
    
    Timer t(true);
    Eigen::CholmodSupernodalLLT<StiffnessMatrix, Eigen::Lower> solver;
    
    T alpha = 10e-6;
    StiffnessMatrix H(K.rows(), K.cols());
    H.setIdentity(); H.diagonal().array() = 1e-10;
    K += H;
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

template <int dim>
bool CellSim<dim>::linearSolve(StiffnessMatrix& K, 
    VectorXT& residual, VectorXT& du)
{
    // std::cout << "Linear Solver" << std::endl;
    Timer t(true);
    Eigen::CholmodSupernodalLLT<StiffnessMatrix, Eigen::Lower> solver;
    // Eigen::PardisoLLT<StiffnessMatrix, Eigen::Lower> solver;
    T alpha = 1e-6;
    StiffnessMatrix H(K.rows(), K.cols());
    H.setIdentity(); H.diagonal().array() = 1e-10;
    K += H;
    solver.analyzePattern(K);
    // T time_analyze = t.elapsed_sec();
    // std::cout << "\t analyzePattern takes " << time_analyze << "s" << std::endl;
    
    int indefinite_count_reg_cnt = 0, invalid_search_dir_cnt = 0, invalid_residual_cnt = 0;

    for (int i = 0; i < 50; i++)
    {
        solver.factorize(K);
        // std::cout << "factorize" << std::endl;
        if (solver.info() == Eigen::NumericalIssue)
        {
            K.diagonal().array() += alpha;
            alpha *= 10;
            indefinite_count_reg_cnt++;
            continue;
        }

        du = solver.solve(residual);
        
        T dot_dx_g = du.normalized().dot(residual.normalized());

        int num_negative_eigen_values = 0;
        int num_zero_eigen_value = 0;

        bool positive_definte = num_negative_eigen_values == 0;
        bool search_dir_correct_sign = dot_dx_g > 1e-6;
        if (!search_dir_correct_sign)
        {   
            invalid_search_dir_cnt++;
        }
        
        // bool solve_success = true;
        // bool solve_success = (K * du - residual).norm() / residual.norm() < 1e-6;
        bool solve_success = du.norm() < 1e3;
        
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
            K.diagonal().array() += alpha;
            alpha *= 10;
        }
    }
    return false;
}

template <int dim>
void CellSim<dim>::buildSystemMatrixWoodbury(const VectorXT& _u, 
        StiffnessMatrix& K, MatrixXT& UV)
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

    addYolkEdgeRegHessianEntries(entries);
    addMatchingHessianEntries(entries);
    addAdhesionHessianEntries(entries, false);
    addRepulsionHessianEntries(entries, false);
    addMembraneHessianEntries(entries, false);

    addYolkPreservationHessianEntries(entries, UV);
    if (use_ipc)
        addIPCHessianEntries(entries);
        
    K.resize(num_nodes * dim, num_nodes * dim);
    K.reserve(entries.size() / 2);
    K.setFromTriplets(entries.begin(), entries.end());
    if (!run_diff_test)
        projectDirichletDoFMatrix(K, dirichlet_data);
    // std::cout << K << std::endl;
    // std::getchar();
    K.makeCompressed();
}

template <int dim>
void CellSim<dim>::buildSystemMatrix(const VectorXT& _u, StiffnessMatrix& K)
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
    
    addYolkEdgeRegHessianEntries(entries);
    addMatchingHessianEntries(entries);
    addAdhesionHessianEntries(entries, false);
    addRepulsionHessianEntries(entries, false);
    addMembraneHessianEntries(entries, false);

    MatrixXT dummy_WoodBury_matrix;
    addYolkPreservationHessianEntries(entries, dummy_WoodBury_matrix);
    if (use_ipc)
        addIPCHessianEntries(entries);
        
    K.resize(num_nodes * dim, num_nodes * dim);
    K.reserve(entries.size() / 2);
    K.setFromTriplets(entries.begin(), entries.end());
    if (!run_diff_test)
        projectDirichletDoFMatrix(K, dirichlet_data);
    // std::cout << K << std::endl;
    // std::getchar();
    K.makeCompressed();
}

template <int dim>
T CellSim<dim>::computeTotalEnergy(const VectorXT& _u, bool add_to_deform)
{
    
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
    
    addYolkEdgeRegEnergy(energy);
    T e_adh = 0.0;
    addAdhesionEnergy(e_adh);
    // std::cout << "E e_adh " << e_adh << std::endl;
    energy += e_adh;
    addMatchingEnergy(energy);
    T e_rep = 0.0;
    addRepulsionEnergy(e_rep);
    // std::cout << "E rep " << e_rep << std::endl;
    energy += e_rep;
    addMembraneEnergy(energy);
    addYolkPreservationEnergy(energy);
    if (use_ipc)
        addIPCEnergy(energy);
    
    return energy;
}

template <int dim>
T CellSim<dim>::computeResidual(const VectorXT& _u,  VectorXT& residual)
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

    addYolkEdgeRegForceEntries(residual);
    addAdhesionForceEntries(residual);
    addMatchingForceEntries(residual);
    addRepulsionForceEntries(residual);
    addMembraneForceEntries(residual);
    addYolkPreservationForceEntries(residual);
    if (use_ipc)
        addIPCForceEntries(residual);

    if (!run_diff_test)
        iterateDirichletDoF([&](int offset, T target)
        {
            residual[offset] = 0;
        });

    return residual.norm();
}

template <int dim>
T CellSim<dim>::lineSearchNewton(VectorXT& _u,  VectorXT& residual, int ls_max)
{
    VectorXT du = residual;
    du.setZero();
    StiffnessMatrix K(residual.rows(), residual.rows());

    Timer ti(true);
    bool success = false;
    if (woodbury)
    {
        MatrixXT UV;
        buildSystemMatrixWoodbury(_u, K, UV);
        if (verbose)
            std::cout << "\t build system takes " << ti.elapsed_sec() << "s" << std::endl;
        success = solveWoodbury(K, UV, residual, du);
    }
    else
    {
        buildSystemMatrix(_u, K);
        if (verbose)
            std::cout << "\t build system takes " << ti.elapsed_sec() << "s" << std::endl;
        success = linearSolve(K, residual, du);    
    }

    if (!success)
    {
        std::cout << "linear solve failed" << std::endl;
        std::exit(0);
        return 1e16;
    }
    

    T norm = du.norm();

    T E0 = computeTotalEnergy(_u);
    T alpha = computeLineSearchInitStepsize(_u, du);
    int cnt = 1;
    std::vector<T> ls_energies;
    while (true)
    {
        VectorXT u_ls = _u + alpha * du;
        T E1 = computeTotalEnergy(u_ls);
        ls_energies.push_back(E1);
        if (E1 - E0 < 0 || cnt > ls_max)
        {
            _u = u_ls;
            if (cnt > ls_max)
            {
                if (verbose)
                    std::cout << "---ls max---" << std::endl;
                // checkTotalGradientScale();
                // checkTotalHessianScale();
                // print_force_norm = true;
                // std::cout << "|du|: " << du.norm() << std::endl;
                // std::cout << "alpha |du|: " << alpha * du.norm() << std::endl;
                // std::cout << "E0: " << E0 << " E1 " << E1 << std::endl;
                // for (T ei : ls_energies)
                //     std::cout << std::setprecision(16) << ei << std::endl;
                // saveState("e1.obj", undeformed + _u + du);
                // saveState("e0.obj", undeformed + _u);
                // std::getchar();
            }
            if (verbose)
                std::cout << "# ls " << cnt << " |du| " << alpha * du.norm() << std::endl;
            break;
        }
        alpha *= 0.5;
        cnt++;
    }
    return norm;
}


template <int dim>
void CellSim<dim>::generateMeshForRendering(
        Eigen::MatrixXd& V, Eigen::MatrixXi& F, 
        Eigen::MatrixXd& C, bool show_yolk,
        T cell_radius, T yolk_radius)
{
    V.resize(0, 0); F.resize(0, 0); C.resize(0, 0);

    appendSphereToPositionVector(deformed.segment(0, num_cells * dim), cell_radius, TV3(0, 0.3, 1.0), V, F, C);
    if (show_yolk && yolk_cell_starts != -1)
        appendSphereToPositionVector(deformed.segment(yolk_cell_starts * dim, (num_nodes - yolk_cell_starts) * dim), yolk_radius, TV3(0, 1.0, 0.0), V, F, C);
}



template <int dim>
void CellSim<dim>::appendSphereToPositionVector(const VectorXT& position, T radius, 
    const TV3& color,
    Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& C)
{
    int n_pt = position.rows() / dim;

    Eigen::MatrixXd v_sphere;
    Eigen::MatrixXi f_sphere;

    igl::readOBJ("/home/yueli/Documents/ETH/WuKong/Projects/DigitalFabrics/Data/sphere162.obj", v_sphere, f_sphere);
    
    Eigen::MatrixXd c_sphere(f_sphere.rows(), f_sphere.cols());
    
    v_sphere = v_sphere * radius;

    int n_vtx_prev = V.rows();
    int n_face_prev = F.rows();

    V.conservativeResize(V.rows() + v_sphere.rows() * n_pt, 3);
    F.conservativeResize(F.rows() + f_sphere.rows() * n_pt, 3);
    C.conservativeResize(C.rows() + f_sphere.rows() * n_pt, 3);

    tbb::parallel_for(0, n_pt, [&](int i)
    {
        Eigen::MatrixXd v_sphere_i = v_sphere;
        if (isYolkParticle(i))
            v_sphere_i *= 0.2;
        Eigen::MatrixXi f_sphere_i = f_sphere;
        Eigen::MatrixXd c_sphere_i = c_sphere;

        tbb::parallel_for(0, (int)v_sphere.rows(), [&](int row_idx){
            v_sphere_i.row(row_idx).segment<dim>(0) += position.segment<dim>(i * dim);
        });


        int offset_v = n_vtx_prev + i * v_sphere.rows();
        int offset_f = n_face_prev + i * f_sphere.rows();

        tbb::parallel_for(0, (int)f_sphere.rows(), [&](int row_idx){
            f_sphere_i.row(row_idx) += Eigen::Vector3i(offset_v, offset_v, offset_v);
        });

        tbb::parallel_for(0, (int)f_sphere.rows(), [&](int row_idx)
        {
            // if (is_control_points[i] != -1)
            //     c_sphere_i.row(row_idx) = TV3(1.0, 0, 0);
            // else if (isYolkParticle(i))
            //     c_sphere_i.row(row_idx) = TV3(0.0, 1.0, 0);
            // else
            //    c_sphere_i.row(row_idx) = color;
            c_sphere_i.row(row_idx) = color;
        });

        V.block(offset_v, 0, v_sphere.rows(), 3) = v_sphere_i;
        F.block(offset_f, 0, f_sphere.rows(), 3) = f_sphere_i;
        C.block(offset_f, 0, f_sphere.rows(), 3) = c_sphere_i;
    });
}


template <int dim>
void CellSim<dim>::reset()
{
    deformed = undeformed;
    u.setZero();
    buildIPCRestData();
}


template class CellSim<2>;
template class CellSim<3>;