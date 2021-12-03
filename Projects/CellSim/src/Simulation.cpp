#include "../include/Simulation.h"
#include <Eigen/PardisoSupport>
#include <iomanip>
#include <ipc/ipc.hpp>


void generatePolygonRendering(Eigen::MatrixXd& V, Eigen::MatrixXi& F, 
        Eigen::MatrixXd& C)
{

}

void Simulation::computeLinearModes()
{
    cells.computeLinearModes();
}

void Simulation::initializeCells()
{
    woodbury = true;
    cells.use_alm_on_cell_volume = false;

    std::string sphere_file;
    cells.scene_type = 1;

    if (cells.scene_type == 1 || cells.scene_type == 2)
        sphere_file = "/home/yueli/Documents/ETH/WuKong/Projects/CellSim/data/sphere.obj";
    else if(cells.scene_type == 0)
        sphere_file = "/home/yueli/Documents/ETH/WuKong/Projects/CellSim/data/sphere_lowres.obj";
    cells.vertexModelFromMesh(sphere_file);
    // cells.addTestPrism(6);
    // cells.addTestPrismGrid(10, 10);
    cells.computeVolumeAllCells(cells.cell_volume_init);

    // cells.checkTotalGradientScale(true);
    // cells.checkTotalHessianScale(true);
    // cells.checkTotalGradient(true);
    // cells.checkTotalHessian();
    // cells.faceHessianChainRuleTest();
    
    max_newton_iter = 500;
    // verbose = true;
    cells.print_force_norm = true;

}

void Simulation::reinitializeCells()
{
    
}

void Simulation::sampleBoundingSurface(Eigen::MatrixXd& V)
{
    cells.sampleBoundingSurface(V);
}

void Simulation::generateMeshForRendering(Eigen::MatrixXd& V, 
    Eigen::MatrixXi& F, Eigen::MatrixXd& C, 
    bool show_deformed, bool show_rest, 
    bool split, bool split_a_bit, bool yolk_only)
{
    // deformed = undeformed + 1.0 * u;
    V.resize(0, 0);
    F.resize(0, 0);
    C.resize(0, 0);
    
    Eigen::MatrixXd V_rest, C_rest;
    Eigen::MatrixXi F_rest, offset;
    
    if (show_deformed)
        cells.generateMeshForRendering(V, F, C);
    int nv = V.rows(), nf = F.rows();
    if (show_rest)
    {
        cells.generateMeshForRendering(V_rest, F_rest, C_rest, true);
        int nv_rest = V_rest.rows(), nf_rest = F_rest.rows();
        V.conservativeResize(V.rows() + V_rest.rows(), 3);
        F.conservativeResize(F.rows() + F_rest.rows(), 3);
        C.conservativeResize(C.rows() + C_rest.rows(), 3);
        C_rest.col(0).setConstant(1.0);
        C_rest.col(1).setConstant(1.0);
        C_rest.col(2).setConstant(0.0);
        offset = F_rest;
        offset.setConstant(nv);
        V.block(nv, 0, nv_rest, 3) = V_rest;
        F.block(nf, 0, nf_rest, 3) = F_rest + offset;
        C.block(nf, 0, nf_rest, 3) = C_rest;
    }
    if (split || split_a_bit)
    {
        cells.splitCellsForRendering(V, F, C, split_a_bit);
    }
    if (yolk_only)
    {
        if (show_deformed)
            cells.getYolkForRendering(V, F, C);
        int nv = V.rows(), nf = F.rows();
        if (show_rest)
        {
            cells.getYolkForRendering(V_rest, F_rest, C_rest, true);
            int nv_rest = V_rest.rows(), nf_rest = F_rest.rows();
            V.conservativeResize(V.rows() + V_rest.rows(), 3);
            F.conservativeResize(F.rows() + F_rest.rows(), 3);
            C.conservativeResize(C.rows() + C_rest.rows(), 3);
            C_rest.col(0).setConstant(1.0);
            C_rest.col(1).setConstant(1.0);
            C_rest.col(2).setConstant(0.0);
            offset = F_rest;
            offset.setConstant(nv);
            V.block(nv, 0, nv_rest, 3) = V_rest;
            F.block(nf, 0, nf_rest, 3) = F_rest + offset;
            C.block(nf, 0, nf_rest, 3) = C_rest;
        }
    }
}

void Simulation::advanceOneStep()
{

}

bool Simulation::staticSolve()
{
    // cells.saveHexTetsStep(0);
    // std::exit(0);
    VectorXT cell_volume_initial;
    cells.computeVolumeAllCells(cell_volume_initial);
    T yolk_volume_init = 0.0;
    if (cells.add_yolk_volume)
    {
        yolk_volume_init = cells.computeYolkVolume(false);
        // std::cout << "yolk volume initial: " << yolk_volume_init << std::endl;
    }

    T total_volume_apical_surface = cells.computeTotalVolumeFromApicalSurface();

    
    std::cout << cells.computeTotalEnergy(u, true) << std::endl;
    int cnt = 0;
    T residual_norm = 1e10, dq_norm = 1e10;

    cells.iterateDirichletDoF([&](int offset, T target)
    {
        f[offset] = 0;
    });

    while (true)
    {
        VectorXT residual(deformed.rows());
        residual.setZero();
        if (cells.use_fixed_cell_centroid)
            cells.updateFixedCellCentroid();
        
        residual_norm = computeResidual(u, residual);
        // cells.saveCellMesh(cnt);
        // cells.saveIPCData(cnt);
        // cells.saveHexTetsStep(cnt);
        // std::exit(0);
        // if (verbose)
            std::cout << "iter " << cnt << ": residual_norm " << residual.norm() << " tol: " << newton_tol << std::endl;
            // std::getchar();
        // if (cnt % 50 == 0)
        // if (cnt == 1)
        // {
        //     cells.checkTotalGradientScale();
        //     cells.checkTotalHessianScale();
        // }
        if (residual_norm < newton_tol)
            break;
        
        // t.start();
        dq_norm = lineSearchNewton(u, residual, 20, true);
        cells.updateALMData(u);
        // t.stop();
        // std::cout << "newton single step costs " << t.elapsed_sec() << "s" << std::endl;

        if(cnt == max_newton_iter || dq_norm > 1e10)
            break;
        cnt++;
    }

    cells.iterateDirichletDoF([&](int offset, T target)
    {
        u[offset] = target;
    });

    // T total_energy_final = cells.computeTotalEnergy(u, true);

    deformed = undeformed + u;
    cells.saveIPCData();

    VectorXT cell_volume_final;
    cells.computeVolumeAllCells(cell_volume_final);

    std::cout << "============================================================================" << std::endl;
    std::cout << std::endl;
    std::cout << "========================= Solver Info ================================="<< std::endl;
    std::cout << "# of system DoF " << deformed.rows() << std::endl;
    std::cout << "# of newton iter: " << cnt << " exited with |g|: " << residual_norm << "|dq|: " << dq_norm  << std::endl;
    // cells.computeLinearModes();
    std::cout << std::endl;
    std::cout << "========================= Cell Info =================================" << std::endl;
    std::cout << "\tcell volume sum initial " << cell_volume_initial.sum() << std::endl;
    std::cout << "\tcell volume sum final " << cell_volume_final.sum() << std::endl;
    if (cells.add_yolk_volume)
    {
        T yolk_volume = cells.computeYolkVolume(false);
        std::cout << "\tyolk volume initial: " << yolk_volume_init << std::endl;
        std::cout << "\tyolk volume final: " << yolk_volume << std::endl;
    }
    
    std::cout << "\ttotal volume initial from apical surface: " << total_volume_apical_surface << std::endl;
    std::cout << "\ttotal volume final from apical surface: " << cells.computeTotalVolumeFromApicalSurface() << std::endl;
    T total_energy_final = cells.computeTotalEnergy(u, true);
    std::cout << "\ttotal energy final: " << total_energy_final << std::endl;
    std::cout << "============================================================================" << std::endl;
    // std::cout << "total energy " << cells.computeTotalEnergy(u, true) << std::endl;
    // T vol;
    
    // cells.computeHexPrismVolumeFromTet(deformed, vol);
    // std::cout << "tet vol last print " << vol << std::endl;
    if (cnt == max_newton_iter || dq_norm > 1e10 || residual_norm > 1)
        return false;
    return true;
}

bool Simulation::WoodburySolve(StiffnessMatrix& K, const MatrixXT& UV,
         VectorXT& residual, VectorXT& du)
{
    StiffnessMatrix I(K.rows(), K.cols());
    I.setIdentity();

    StiffnessMatrix H = K;

    Eigen::PardisoLLT<Eigen::SparseMatrix<T, Eigen::ColMajor, typename StiffnessMatrix::StorageIndex>> solver;

    T alpha = 10e-6;
    solver.analyzePattern(K);
    int i = 0;
    for (; i < 50; i++)
    {
        // std::cout << i << std::endl;
        solver.factorize(K);
        if (solver.info() == Eigen::NumericalIssue)
        {
            K = H + alpha * I;        
            alpha *= 10;
            continue;
        }

        // sherman morrison
        if (UV.cols() == 1)
        {
            VectorXT v = UV.col(0);
            VectorXT A_inv_g = solver.solve(residual);
            VectorXT A_inv_u = solver.solve(v);

            T dem = 1.0 + v.dot(A_inv_u);

            du = A_inv_g - (A_inv_g.dot(v)) * A_inv_u / dem;
        }
        // UV is actually only U, since UV is the same in the case
        // C is assume to be Identity
        else // Woodbury https://en.wikipedia.org/wiki/Woodbury_matrix_identity
        {
            VectorXT A_inv_g = solver.solve(residual);

            MatrixXT A_inv_U(UV.rows(), UV.cols());
            for (int col = 0; col < UV.cols(); col++)
                A_inv_U.col(col) = solver.solve(UV.col(col));
            
            MatrixXT C(UV.cols(), UV.cols());
            C.setIdentity();
            C += UV.transpose() * A_inv_U;
            du = A_inv_g - A_inv_U * C.inverse() * UV.transpose() * A_inv_g;
        }
        

        T dot_dx_g = du.normalized().dot(residual.normalized());

        int num_negative_eigen_values = 0;
        int num_zero_eigen_value = 0;

        bool positive_definte = num_negative_eigen_values == 0;
        bool search_dir_correct_sign = dot_dx_g > 1e-6;
        bool solve_success = ((K + UV * UV.transpose())*du - residual).norm() < 1e-6 && solver.info() == Eigen::Success;
        // std::cout << "PD: " << positive_definte << " direction " 
        //     << search_dir_correct_sign << " solve " << solve_success << std::endl;

        if (positive_definte && search_dir_correct_sign && solve_success)
        {
            std::cout << "\t===== Linear Solve ===== " << std::endl;
            std::cout << "\t# regularization step " << i << std::endl;
            std::cout << "\tdot(search, -gradient) " << dot_dx_g << std::endl;
            std::cout << "\t======================== " << std::endl;
            return true;
        }
        else
        {
            K = H + alpha * I;        
            alpha *= 10;
        }
    }
    return false;
}

bool Simulation::linearSolve(StiffnessMatrix& K, VectorXT& residual, VectorXT& du)
{
#define USE_PARDISO

    StiffnessMatrix I(K.rows(), K.cols());
    I.setIdentity();

    StiffnessMatrix H = K;

#ifdef USE_PARDISO
    Eigen::PardisoLLT<Eigen::SparseMatrix<T, Eigen::ColMajor, typename StiffnessMatrix::StorageIndex>> solver;
#else
    Eigen::SimplicialLDLT<StiffnessMatrix> solver;
    // Eigen::CholmodSimplicialLLT<StiffnessMatrix> solver;
#endif

    T alpha = 10e-6;
    solver.analyzePattern(K);
    int i = 0;
    for (; i < 50; i++)
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

        int num_negative_eigen_values = 0;
        int num_zero_eigen_value = 0;
#ifndef USE_PARDISO
        VectorXT d_vector = solver.vectorD();
        // std::cout << d_vector << std::endl;
        // std::getchar();
        for (int i = 0; i < d_vector.size(); i++)
        {
            if (d_vector[i] < 0)
            {
                num_negative_eigen_values++;
                // break;
            }
            if (std::abs(d_vector[i]) < 1e-6)
                num_zero_eigen_value++;
        }
        if (num_zero_eigen_value > 0)
        {
            std::cout << "num_zero_eigen_value " << num_zero_eigen_value << std::endl;
            return false;
        }
#endif
        bool positive_definte = num_negative_eigen_values == 0;
        bool search_dir_correct_sign = dot_dx_g > 1e-6;
        bool solve_success = (K*du - residual).norm() < 1e-6 && solver.info() == Eigen::Success;
        // std::cout << "PD: " << positive_definte << " direction " 
        //     << search_dir_correct_sign << " solve " << solve_success << std::endl;

        if (positive_definte && search_dir_correct_sign && solve_success)
        {
            std::cout << "\t===== Linear Solve ===== " << std::endl;
            std::cout << "\t# regularization step " << i << std::endl;
            std::cout << "\tdot(search, -gradient) " << dot_dx_g << std::endl;
            std::cout << "\t======================== " << std::endl;
            return true;
        }
        else
        {
            K = H + alpha * I;        
            alpha *= 10;
        }
    }
    return false;
}

void Simulation::buildSystemMatrix(const VectorXT& _u, StiffnessMatrix& K)
{
    cells.buildSystemMatrix(_u, K);
}

T Simulation::computeTotalEnergy(const VectorXT& _u)
{
    T energy = cells.computeTotalEnergy(_u, verbose);
    return energy;
}

T Simulation::computeResidual(const VectorXT& _u,  VectorXT& residual)
{
    return cells.computeResidual(_u, residual, verbose);
}


void Simulation::sampleEnergyWithSearchAndGradientDirection(
    const VectorXT& _u,  
    const VectorXT& search_direction,
    const VectorXT& negative_gradient)
{
    T E0 = computeTotalEnergy(_u);
    
    std::cout << std::setprecision(12) << "E0 " << E0 << std::endl;
    T step_size = 5e-5;
    int step = 400;

    // T step_size = 1e0;
    // int step = 50;

    

    std::vector<T> energies;
    std::vector<T> energies_gd;
    std::vector<T> steps;
    int step_cnt = 1;
    for (T xi = -T(step/2) * step_size; xi < T(step/2) * step_size; xi+=step_size)
    {
        // cells.use_sphere_radius_bound = false;
        // cells.add_contraction_term = false;
        
        // cells.sigma = 0;
        // cells.gamma = 0;
        // cells.alpha = 0.0;
        // cells.B = 0;
        // cells.By = 0;
        T Ei = computeTotalEnergy(_u + xi * search_direction);
        
        // T Ei = cells.computeAreaEnergy(_u + xi * search_direction);
        // if (std::abs(xi) < 1e-6)
        //     std::getchar();
        energies.push_back(Ei);
        steps.push_back(xi);
    }
    
    for (T e : energies)
    {
        std::cout << std::setprecision(12) <<  e << " ";
    }
    std::cout << std::endl;
    for (T e : energies_gd)
    {
        std::cout << e << " ";
    }
    std::cout << std::endl;
    for (T idx : steps)
    {
        std::cout << idx << " ";
    }
    std::cout << std::endl;

}

void Simulation::buildSystemMatrixWoodbury(const VectorXT& _u, StiffnessMatrix& K, MatrixXT& UV)
{
    cells.buildSystemMatrixWoodbury(u, K, UV);
}

T Simulation::lineSearchNewton(VectorXT& _u,  VectorXT& residual, int ls_max, bool wolfe_condition)
{
    // for wolfe condition
    T c1 = 10e-4, c2 = 0.9;

    VectorXT du = residual;
    du.setZero();

    StiffnessMatrix K(residual.rows(), residual.rows());
    
    bool success = false;
    if (woodbury)
    {
        MatrixXT UV;
        buildSystemMatrixWoodbury(_u, K, UV);
        success = WoodburySolve(K, UV, residual, du);    
    }
    else
    {
        buildSystemMatrix(_u, K);
        success = linearSolve(K, residual, du);    
    }
    
    if (!success)
    {
        std::cout << "linear solve failed" << std::endl;
        return 1e16;
    }

    T norm = du.norm();
    
    T alpha = 1;

    if (cells.use_ipc_contact)
    {
        Eigen::MatrixXd current_position(cells.basal_vtx_start, 3), 
            next_step_position(cells.basal_vtx_start, 3);
            
        for (int i = 0; i < cells.basal_vtx_start; i++)
        {
            current_position.row(i) = undeformed.segment<3>(i * 3) + _u.segment<3>(i * 3);
            next_step_position.row(i) = undeformed.segment<3>(i * 3) + _u.segment<3>(i * 3) + du.segment<3>(i * 3);
        }
        
        alpha = ipc::compute_collision_free_stepsize(current_position, 
            next_step_position, cells.ipc_edges, cells.ipc_faces, ipc::BroadPhaseMethod::HASH_GRID, 1e-6, 1e7);
        // std::cout << "collision free step size " << alpha << std::endl;
    }

    T E0 = computeTotalEnergy(_u);
    // std::cout << "E0 " << E0 << std::endl;
    int cnt = 1;
    while (true)
    {
        VectorXT u_ls = _u + alpha * du;
        T E1 = computeTotalEnergy(u_ls);
        // std::cout << "ls# " << cnt << " E1 " << E1 << std::endl;
        // cells.computeTotalEnergy(u_ls, true);
        // if (wolfe_condition)
        if (false)
        {
            bool Armijo = E1 <= E0 + c1 * alpha * du.dot(-residual);
            // std::cout << c1 * alpha * du.dot(-residual) << std::endl;
            VectorXT gradient_forward = VectorXT::Zero(deformed.rows());
            computeResidual(u_ls, gradient_forward);
            bool curvature = -du.dot(-gradient_forward) <= -c2 * du.dot(-residual);
            // std::cout << "wolfe Armijo " << Armijo << " curvature " << curvature << std::endl;
            if ((Armijo && curvature) || cnt > ls_max)
            {
                _u = u_ls;
                if (cnt > ls_max)
                {
                    std::cout << "---ls max---" << std::endl;
                    // std::cout << "step size: " << alpha << std::endl;
                    // sampleEnergyWithSearchAndGradientDirection(_u, du, residual);
                    // cells.computeTotalEnergy(u_ls, true);
                    // cells.checkTotalGradientScale();
                    // cells.checkTotalHessianScale();
                    return 1e16;
                }
                std::cout << "# ls " << cnt << std::endl;
                break;
            }
        }
        else
        {
            if (E1 - E0 < 0 || cnt > ls_max)
            {
                _u = u_ls;
                if (cnt > ls_max)
                {
                    std::cout << "---ls max---" << std::endl;
                    // std::cout << "step size: " << alpha << std::endl;
                    // sampleEnergyWithSearchAndGradientDirection(_u, du, residual);
                    // cells.checkTotalGradient();
                    return 1e16;
                }
                std::cout << "# ls " << cnt << std::endl;
                break;
            }
        }
        alpha *= 0.5;
        cnt += 1;
    }
    return norm;
    if (cnt > ls_max)
    {
        // try gradien step
        std::cout << "taking gradient step " << std::endl;
        // std::cout << "|du|: " << du.norm() << " |g| " << residual.norm() << std::endl;
        // std::cout << "E0 " << E0 << std::endl;
        VectorXT negative_gradient_direction = residual.normalized();
        alpha = 1.0;
        cnt = 1;
        while (true)
        {
            VectorXT u_ls = _u + alpha * negative_gradient_direction;
            // _u = u_ls;
            // return 1e16;
            T E1 = computeTotalEnergy(u_ls);
            // std::cout << "ls gd # " << cnt << " E1 " << E1 << std::endl;
            if (E1 - E0 < 0 || cnt > 30)
            {
                _u = u_ls;
                if (cnt > 30)
                {
                    std::cout << "---gradient ls max---" << std::endl;
                    // cells.checkTotalGradient();
                    // std::cout << "|g|: " <<  residual.norm() << std::endl;
                    // cells.checkTotalGradientScale();
                    sampleEnergyWithSearchAndGradientDirection(_u, negative_gradient_direction, residual);
                    return 1e16;
                }
                // std::cout << "# ls " << cnt << std::endl;
                break;
            }
            alpha *= 0.5;
            cnt += 1;
        }
        
    }
    
    return norm;
}