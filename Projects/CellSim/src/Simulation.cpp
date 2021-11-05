#include "../include/Simulation.h"
#include <Eigen/PardisoSupport>

void Simulation::initializeCells()
{
    std::string sphere_file = "/home/yueli/Documents/ETH/WuKong/Projects/CellSim/data/sphere.obj";
    cells.vertexModelFromMesh(sphere_file);
    cells.computeCellInitialVolume();
}

void Simulation::generateMeshForRendering(Eigen::MatrixXd& V, 
    Eigen::MatrixXi& F, Eigen::MatrixXd& C)
{
    cells.generateMeshForRendering(V, F, C);
}

void Simulation::advanceOneStep()
{

}

bool Simulation::staticSolve()
{
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
        
        if (verbose)
            std::cout << "residual_norm " << residual.norm() << " tol: " << newton_tol << std::endl;
        
        if (residual_norm < newton_tol)
            break;
        
        dq_norm = lineSearchNewton(u, residual, 15);
        
        if(cnt == max_newton_iter || dq_norm > 1e10)
            break;
        cnt++;
    }

    // iterateDirichletDoF([&](int offset, T target)
    // {
    //     u[offset] = target;
    // });

    deformed = undeformed + u;

    std::cout << "# of newton solve: " << cnt << " exited with |g|: " << residual_norm << "|dq|: " << dq_norm  << std::endl;
    // std::cout << u.norm() << std::endl;
    if (cnt == max_newton_iter || dq_norm > 1e10 || residual_norm > 1)
        return false;
    return true;
}

bool Simulation::linearSolve(StiffnessMatrix& K, VectorXT& residual, VectorXT& du)
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

void Simulation::buildSystemMatrix(const VectorXT& _u, StiffnessMatrix& K)
{
    cells.buildSystemMatrix(_u, K);
}

T Simulation::computeTotalEnergy(const VectorXT& _u)
{
    T energy = cells.computeTotalEnergy(_u);
    return energy;
}

T Simulation::computeResidual(const VectorXT& _u,  VectorXT& residual)
{
    return cells.computeResidual(_u, residual);
}

T Simulation::lineSearchNewton(VectorXT& _u,  VectorXT& residual, int ls_max)
{
    VectorXT du = residual;
    du.setZero();

    StiffnessMatrix K(residual.rows(), residual.rows());
    buildSystemMatrix(_u, K);
    
    
    bool success = linearSolve(K, residual, du);
    
    // std::cout << "dx: " <<  du.norm() << std::endl;
    
    if (!success)
        return 1e16;
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
        if (E1 - E0 < 0 || cnt > ls_max)
        {
            _u = u_ls;
            break;
        }
        alpha *= 0.5;
        cnt += 1;
    }
    
    return norm;
}