#include "../include/Simulation.h"
#include <Eigen/PardisoSupport>
#include <iomanip>

void Simulation::computeLinearModes()
{
    cells.computeLinearModes();
}

void Simulation::initializeCells()
{
    std::string sphere_file = "/home/yueli/Documents/ETH/WuKong/Projects/CellSim/data/sphere.obj";
    // std::string sphere_file = "/home/yueli/Documents/ETH/WuKong/Projects/CellSim/data/sphere_lowres.obj";
    cells.vertexModelFromMesh(sphere_file);
    // cells.addTestPrism(5);
    // cells.addTestPrismGrid(10, 10);
    cells.computeVolumeAllCells(cells.cell_volume_init);

    // cells.checkTotalGradientScale(true);
    // cells.checkTotalHessianScale(true);
    // cells.checkTotalGradient();
    // cells.checkTotalHessian();
    // cells.faceHessianChainRuleTest();
    
    max_newton_iter = 300;
    // verbose = true;
}

void Simulation::generateMeshForRendering(Eigen::MatrixXd& V, 
    Eigen::MatrixXi& F, Eigen::MatrixXd& C, bool show_rest, bool split)
{
    deformed = undeformed + 1.0 * u;
    cells.generateMeshForRendering(V, F, C);
    if (show_rest)
    {
        int nv = V.rows(), nf = F.rows();
        
        V.conservativeResize(V.rows() * 2, 3);
        F.conservativeResize(F.rows() * 2, 3);
        C.conservativeResize(C.rows() * 2, 3);
        tbb::parallel_for(0, nv, [&](int i){
            V.row(nv + i) = cells.undeformed.segment<3>(i * 3);
        });
        tbb::parallel_for(0, nf, [&](int i){
            F.row(nf + i) = F.row(i) + Eigen::Vector3i(nv, nv, nv).transpose();
            C.row(nf + i) = TV(1, 1, 0);
        });
        
    }
    if (split)
    {
        cells.splitCellsForRendering(V, F, C);
    }
}

void Simulation::advanceOneStep()
{

}

bool Simulation::staticSolve()
{
    VectorXT cell_volume_initial;
    cells.computeVolumeAllCells(cell_volume_initial);
    if (cells.add_yolk_volume)
    {
        T yolk_volume_init = cells.computeYolkVolume(false);
        std::cout << "yolk volume initial: " << yolk_volume_init << std::endl;
    }

    std::cout << "total volume initial " << cell_volume_initial.sum() << std::endl;
    std::cout << "total energy " << cells.computeTotalEnergy(u, true) << std::endl;
    int cnt = 0;
    T residual_norm = 1e10, dq_norm = 1e10;

    // iterateDirichletDoF([&](int offset, T target)
    // {
    //     f[offset] = 0;
    // });

    while (true)
    {
        VectorXT residual(deformed.rows());
        residual.setZero();
        
        residual_norm = computeResidual(u, residual);
        
        // if (verbose)
            std::cout << "residual_norm " << residual.norm() << " tol: " << newton_tol << std::endl;
            // std::getchar();
        
        if (residual_norm < newton_tol)
            break;
        
        // t.start();
        dq_norm = lineSearchNewton(u, residual, 20, true);
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

    std::cout << "total energy " << cells.computeTotalEnergy(u, true) << std::endl;

    deformed = undeformed + u;

    VectorXT cell_volume_final;
    cells.computeVolumeAllCells(cell_volume_final);
    std::cout << "total volume " << cell_volume_final.sum() << std::endl;
    std::cout << "# of newton solve: " << cnt << " exited with |g|: " << residual_norm << "|dq|: " << dq_norm  << std::endl;
    if (cells.add_yolk_volume)
    {
        T yolk_volume = cells.computeYolkVolume(false);
        std::cout << "yolk volume final: " << yolk_volume << std::endl;
    }
    if (cnt == max_newton_iter || dq_norm > 1e10 || residual_norm > 1)
        return false;
    return true;
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
            // std::cout << "\t===== Linear Solve ===== " << std::endl;
            // std::cout << "\t# regularization step " << i << std::endl;
            // std::cout << "\tdot(search, -gradient) " << dot_dx_g << std::endl;
            // std::cout << "\t======================== " << std::endl;
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
    // T step_size = 5e-5;
    // int step = 400;

    T step_size = 1e-1;
    int step = 50;

    std::vector<T> energies;
    std::vector<T> energies_gd;
    std::vector<T> steps;
    int step_cnt = 1;
    for (T xi = -T(step/2) * step_size; xi < T(step/2) * step_size; xi+=step_size)
    {
        cells.use_sphere_radius_bound = false;
        cells.add_contraction_term = false;
        
        cells.sigma = 0;
        cells.gamma = 0;
        // cells.alpha = 0.0;
        cells.B = 0;
        cells.By = 0;
        T Ei = computeTotalEnergy(_u + xi * search_direction);
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

T Simulation::lineSearchNewton(VectorXT& _u,  VectorXT& residual, int ls_max, bool wolfe_condition)
{
    // for wolfe condition
    T c1 = 10e-4, c2 = 0.9;

    VectorXT du = residual;
    du.setZero();

    StiffnessMatrix K(residual.rows(), residual.rows());
    // t.start();
    buildSystemMatrix(_u, K);
    // std::cout << "\tbuild matrix " << t.elapsed_sec() << "s" << std::endl;
    // cells.checkTotalHessianScale();
    bool success = linearSolve(K, residual, du);    
    // std::cout << "\tlinear solve " << t.elapsed_sec() << "s" << std::endl;
    // t.stop();
    if (!success)
    {
        std::cout << "linear solve failed" << std::endl;
        return 1e16;
    }

    T norm = du.norm();
    
    T alpha = 1;
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
                    sampleEnergyWithSearchAndGradientDirection(_u, du, residual);
                    // cells.computeTotalEnergy(u_ls, true);
                    // cells.checkTotalGradientScale();
                    // cells.checkTotalHessianScale();
                    return 1e16;
                }
                // std::cout << "# ls " << cnt << std::endl;
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
                    sampleEnergyWithSearchAndGradientDirection(_u, du, residual);
                    // cells.checkTotalGradient();
                    return 1e16;
                }
                // std::cout << "# ls " << cnt << std::endl;
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
        // std::cout << "|du|: " << du.norm() << " |g| " << residual.norm() << std::endl;
        // std::cout << "E0 " << E0 << std::endl;
        VectorXT negative_gradient_direction = -residual.normalized();
        alpha = 1.0;
        cnt = 1;
        while (true)
        {
            VectorXT u_ls = _u + alpha * negative_gradient_direction;
            // _u = u_ls;
            // return 1e16;
            T E1 = computeTotalEnergy(u_ls);
            // std::cout << "ls gd # " << cnt << " E1 " << E1 << std::endl;
            if (E1 - E0 < 0 || cnt > ls_max)
            {
                _u = u_ls;
                if (cnt > ls_max)
                {
                    std::cout << "---gradient ls max---" << std::endl;
                    // cells.checkTotalGradient();
                    // std::cout << "|g|: " <<  residual.norm() << std::endl;
                    cells.checkTotalGradientScale();
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