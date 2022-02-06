#include "../include/SensitivityAnalysis.h"
#include <Eigen/PardisoSupport>
#include "../../../Solver/MMASolver.h"

void SensitivityAnalysis::initialize()
{
    n_dof_sim = simulation.num_nodes * 3;
    n_dof_design = simulation.cells.edge_weights.rows();
}

void SensitivityAnalysis::optimizePerEdgeWeigths()
{
    ObjUTU obj(simulation);
    simulation.save_mesh = false;
    simulation.cells.print_force_norm = false;
    simulation.verbose = false;

    obj.getSimulationAndDesignDoF(n_dof_sim, n_dof_design);
    // optimizeGradientDescent(obj);
    optimizeMMA(obj);
}

void SensitivityAnalysis::optimizeMMA(Objectives& objective)
{
    std::cout << "########### MMA ###########" << std::endl;
    objective.getDesignParameters(design_parameters);
    MMASolver mma(n_dof_design, 0);
    mma.SetAsymptotes(0.2, 0.65, 1.05);
    VectorXT min_p = VectorXT::Ones(n_dof_design) * -1e10;
    VectorXT max_p = VectorXT::Ones(n_dof_design) * 1e10;
    for (int iter = 0; iter < 10; iter++)
    {
        VectorXT dOdp(n_dof_design); dOdp.setZero();
        T O;
        T g_norm = objective.gradient(design_parameters, dOdp, O);
        std::cout << "[MMA] iter " << iter << " |g|: " << g_norm << " obj: " << O << std::endl;
        mma.UpdateEigen(design_parameters, dOdp, VectorXT(), VectorXT(), min_p, max_p);
        objective.updateDesignParameters(design_parameters);
    }
}

void SensitivityAnalysis::optimizeGradientDescent(Objectives& objective)
{
    std::cout << "########### GRADIENT DESCENT ###########" << std::endl;
    int num_iter_max = 100;
    T tol_g = 1e-6;
    objective.getDesignParameters(design_parameters);
    for (int iter = 0; iter < num_iter_max; iter++)
    {
        VectorXT dOdp;
        T g_norm = objective.gradient(design_parameters, dOdp);
        std::cout << "[GD] iter " << iter << " |g| " << g_norm << std::endl;
        // std::getchar();
        if (g_norm < tol_g)
            break;
        T E0 = objective.value(design_parameters);
        std::cout << "\tobj: " << E0 << std::endl;
        T alpha = 1.0;
        int ls_cnt = 0;
        while (true)
        {
            VectorXT p_ls = design_parameters - alpha * dOdp;
            T E1 = objective.value(p_ls);
            std::cout << "\tE1: " << E1 << std::endl;
            if (E1 < E0 || ls_cnt > 10)
            {
                design_parameters = p_ls;
                break;
            }
            else
            {
                ls_cnt++;
                alpha *= 0.5;
            }
        }
        std::cout <<  "\t #ls: " << ls_cnt << std::endl;
    }
    simulation.save_mesh = true;
    objective.value(design_parameters);
    
}

void SensitivityAnalysis::dxFromdpAdjoint(VectorXT& dx, const VectorXT& dp)
{
    // x(p + dp) = x(p) + dx/dp dp
    // df/dp = df/dx dx/dp + df/dp = 0 => dx/dp = -(df/dx)^-1 df/dp
    // x(p + dp) = x(p) + -(df/dx)^-1 df/dp dp
    // dx = -(df/dx)^-1 (df/dp dp)
    StiffnessMatrix d2edx2(n_dof_sim, n_dof_sim);
    if (simulation.woodbury)
    {
        MatrixXT UV;
        simulation.buildSystemMatrixWoodbury(simulation.u, d2edx2, UV);
    }
    else
    {
        simulation.buildSystemMatrix(simulation.u, d2edx2);
    }
    VectorXT dfdpdp = VectorXT::Zero(n_dof_sim);
    simulation.cells.multiplyDpWithDfdp(dfdpdp, dp);

    Eigen::PardisoLLT<Eigen::SparseMatrix<T, Eigen::ColMajor, int>> solver;
    solver.analyzePattern(d2edx2);
    solver.factorize(d2edx2);
    if (solver.info() == Eigen::NumericalIssue)
        std::cout << "Forward simulation hessian indefinite" << std::endl;

    dx = -solver.solve(dfdpdp);
}

void SensitivityAnalysis::diffTestdxdp()
{
    simulation.staticSolve();
    MatrixXT dxdp;
    simulation.cells.dxdpFromdxdpEdgeWeights(dxdp);
    T epsilon = 1e-6;
    for (int i = 0; i < n_dof_design; i++)
    {
        simulation.cells.edge_weights[i] += epsilon;
        simulation.reset();
        simulation.staticSolve();
        VectorXT x1 = simulation.deformed;
        simulation.cells.edge_weights[i] -= 2.0 * epsilon;
        simulation.reset();
        simulation.staticSolve();
        VectorXT x0 = simulation.deformed;
        for (int j = 0; j < n_dof_sim; j++)
        {
            T fd = (x1[j] - x0[j]) / (2.0 * epsilon);
            if (std::abs(fd) < 1e-6 && std::abs(dxdp(j, i)) < 1e-6)
                continue;
            std::cout << "symbolic: " << dxdp(j, i) << " fd " << fd << std::endl;
            std::getchar();
        }
        
        simulation.cells.edge_weights[i] += epsilon;
    }
    
}

void SensitivityAnalysis::buildSensitivityMatrix(MatrixXT& dxdp)
{
    // fetch d2edx2 from simulation
    StiffnessMatrix d2edx2(n_dof_sim, n_dof_sim);
    if (simulation.woodbury)
    {
        MatrixXT UV;
        simulation.buildSystemMatrixWoodbury(simulation.u, d2edx2, UV);
    }
    else
    {
        simulation.buildSystemMatrix(simulation.u, d2edx2);
    }
    
    //compute d2edxdp -> dfdp
    MatrixXT dfdp(n_dof_sim, n_dof_design);
    simulation.cells.dfdpWeights(dfdp);

    dxdp.resize(n_dof_sim, n_dof_design);
    dxdp.setZero();

    Eigen::PardisoLLT<Eigen::SparseMatrix<T, Eigen::ColMajor, int>> solver;
    solver.analyzePattern(d2edx2);
    solver.factorize(d2edx2);
    if (solver.info() == Eigen::NumericalIssue)
        std::cout << "Forward simulation hessian indefinite" << std::endl;
    for (int i = 0; i < n_dof_design; i++)
    {
        dxdp.col(i) = solver.solve(dfdp.col(i));
    }
    std::cout << "dxdp norm: " << dxdp.norm() << std::endl;
}

void SensitivityAnalysis::diffTestdfdp()
{
    MatrixXT dfdp(n_dof_sim, n_dof_design);
    simulation.cells.dfdpWeights(dfdp);

    MatrixXT dfdp_fd(n_dof_sim, n_dof_design);
    simulation.cells.dfdpWeightsFD(dfdp_fd);

    for (int i = 0; i < n_dof_design; i++)
    {
        for (int j = 0; j < n_dof_sim; j++)
        {
            if (std::abs(dfdp(j, i)) < 1e-6 && std::abs(dfdp_fd(j, i)) < 1e-6)
                continue;
            if (std::abs(dfdp_fd(j, i) - dfdp(j, i)) < 1e-3 * std::abs(dfdp_fd(j, i)))
                continue;
            std::cout << "design dof " << i << " sim dof: " << j << std::endl;
            std::cout << "symbolic: " << dfdp(j, i) << " fd: " << dfdp_fd(j, i) << std::endl;
            std::getchar();
        }
    }
    std::cout << "dfdp diff test passed" << std::endl;

    std::cout << (dfdp_fd - dfdp).norm() / T(n_dof_design * n_dof_sim) << std::endl;
}


void SensitivityAnalysis::computeEquilibriumState()
{
    simulation.staticSolve();
}

void SensitivityAnalysis::loadEquilibriumState()
{
    simulation.loadDeformedState("/home/yueli/Documents/ETH/WuKong/output/cells/cell/cell_mesh_iter_967.obj");
}

void SensitivityAnalysis::svdOnSensitivityMatrix()
{
    // computeEquilibriumState();
    loadEquilibriumState();
    MatrixXT dxdp;
    buildSensitivityMatrix(dxdp);

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(dxdp, Eigen::ComputeThinU | Eigen::ComputeThinV);
	MatrixXT U = svd.matrixU();
	VectorXT Sigma = svd.singularValues();
	MatrixXT V = svd.matrixV();

    std::cout << "sigma: " << std::endl;
    std::cout << Sigma << std::endl;    
    // std::cout << U.rows() << " " << U.cols() << std::endl;
    // std::cout << V.rows() << " " << V.cols() << std::endl;
    // std::cout << U.col(0).norm() << std::endl;

    std::cout << "V matrix: " << std::endl;
    std::cout << V << std::endl;
    
    std::ofstream out("cell_svd_vectors.txt");
    out << U.rows() << " " << U.cols() << std::endl;
    for (int i = 0; i < n_dof_design; i++)
        out << Sigma[i] << " ";
    out << std::endl;
    for (int i = 0; i < n_dof_sim; i++)
    {
        for (int j = 0; j < n_dof_design; j++)
            out << U(i, j) << " ";
        out << std::endl;
    }
    out << std::endl;
    out.close();

    out.open("dxdp.txt");
    for (int i = 0; i < n_dof_design; i++)
        dxdp.col(i).normalize();
    out << dxdp.rows() << " " << dxdp.cols() << std::endl;
    for (int i = 0; i < n_dof_design; i++)
        out << 1.0 << " ";
    out << std::endl;
    for (int i = 0; i < n_dof_sim; i++)
    {
        for (int j = 0; j < n_dof_design; j++)
            out << dxdp(i, j) << " ";
        out << std::endl;
    }
    out << std::endl;
    out.close();
}

