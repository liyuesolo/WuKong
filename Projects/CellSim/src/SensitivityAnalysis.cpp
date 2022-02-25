#include "../include/SensitivityAnalysis.h"
#include <Eigen/PardisoSupport>
#include <Spectra/SymEigsShiftSolver.h>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/DenseSymMatProd.h>
#include "../include/LinearSolver.h"


void SensitivityAnalysis::initialize()
{
    // simulation.initializeCells();
    simulation.save_mesh = false;
    simulation.cells.print_force_norm = false;
    simulation.verbose = false;
    
    objective.getSimulationAndDesignDoF(n_dof_sim, n_dof_design);
    design_parameter_bound[0] = 0;
    design_parameter_bound[1] = 10.0 * simulation.cells.unit;
    // simulation.cells.edge_weights.setRandom();
    // simulation.cells.edge_weights /= simulation.cells.edge_weights.norm();
    // simulation.cells.edge_weights.setConstant(0.05);
    VectorXT delta = simulation.cells.edge_weights * 0.1;
    // simulation.cells.edge_weights += delta;
    simulation.cells.edge_weights.setConstant(0.05);
    objective.getDesignParameters(design_parameters);
    std::cout << "# dof sim: " << n_dof_sim << " # dof design: " << n_dof_design << std::endl;
}

void SensitivityAnalysis::optimizePerEdgeWeigths()
{
    // ObjUTU obj(simulation);
    // ObjUMatching obj(simulation);
    // obj.setTargetFromMesh("output/cells/opt/target_simple.obj");

    
    simulation.cells.tet_vol_barrier_w = 1e-22;

    objective.getSimulationAndDesignDoF(n_dof_sim, n_dof_design);
    // optimizeGradientDescent();
    
    // optimizeMMA(); 
    optimizeGaussNewton();
}


void SensitivityAnalysis::generateNucleiDataSingleFrame(const std::string& filename)
{
    
    simulation.staticSolve();
    std::ofstream out(filename);
    objective.iterateTargets([&](int idx, TV& target){
        TV current;
        simulation.cells.computeCellCentroid(simulation.cells.faces[idx], current);
        out << idx << " " << current[0] << " " << current[1] << " " << current[2] << std::endl;
    });
    out.close();
}


bool SensitivityAnalysis::optimizeOneStep(int step, Optimizer optimizer)
{
    std::string method;
    if (optimizer == GaussNewton)
        method = "GN";
    else if (optimizer == GradientDescent)
        method = "GD";
    else if (optimizer == MMA)
        method = "MMA";
    else if (optimizer == Newton)
        method = "Newton";
    else if (optimizer == SGN)
        method = "SGN";
    else
        std::cout << "optimizer undefined" << std::endl;
    
    T tol_g = 1e-6;
    T g_norm = 0;
    T E0;
    VectorXT dOdp, dp;

    // mma 
    T mma_step_size = 1.0 * simulation.cells.unit;
    T lower_bound = 0, upper_bound = 10.0 * simulation.cells.unit;

    VectorXT min_p = VectorXT::Ones(n_dof_design) * lower_bound;
    VectorXT max_p = VectorXT::Ones(n_dof_design) * upper_bound;

    bool add_bound_contraint = true;

    if (optimizer == GradientDescent || optimizer == GaussNewton || optimizer == Newton || optimizer == SGN)
    {
        if (step == 0)
        {
            std::cout << "########### " << method << " ###########" << std::endl;
            objective.getDesignParameters(design_parameters);
            std::cout << "initial value max: " << design_parameters.maxCoeff() << " min: " << design_parameters.minCoeff() <<std::endl;
            g_norm = objective.gradient(design_parameters, dOdp, E0);   
        }
        else
        {
            g_norm = objective.evaluteGradientAndEnergy(design_parameters, dOdp, E0);
        }
        g_norm = dOdp.norm();

        std::cout << "[" << method << "] iter " << step << " |g| " << g_norm 
            << " max: " << dOdp.maxCoeff() << " min: " << dOdp.minCoeff()
            << " obj: " << E0 << std::endl;

        std::vector<int> projected_entries;
        auto projectGradient =[&](VectorXT& g, VectorXT& design_parameters_projected)
        {
            // design_parameters_projected = design_parameters;
            T epsilon = 1e-3;
            for (int i = 0; i < n_dof_design; i++)
            {
                if (design_parameters[i] < lower_bound + epsilon && g[i] > 0)
                {
                    // design_parameters_projected[i] = lower_bound;
                    // g[i] = 0;
                    projected_entries.push_back(i);
                }
                if (design_parameters[i] > upper_bound - epsilon && g[i] < 0)
                {
                    // design_parameters_projected[i] = upper_bound;
                    // g[i] = 0;
                    projected_entries.push_back(i);
                }

            }
            
        };
        VectorXT design_parameters_projected = design_parameters;

        if (add_bound_contraint)
            projectGradient(dOdp, design_parameters_projected);
        
        dp = dOdp;

        Timer gn_timer(true);
        if (optimizer == GaussNewton)
        {
            
            StiffnessMatrix H_GN;
            objective.hessianGN(design_parameters, H_GN, false, false);
            std::cout << "\tGN # projected entries " << projected_entries.size() << std::endl;
            for (int idx : projected_entries)
            {
                H_GN.row(idx) *= 0.0;
                H_GN.col(idx) *= 0.0;
                H_GN.coeffRef(idx, idx) = 1.0;
            }
            VectorXT rhs = -dOdp;
            // LinearSolver::linearSolve(H_GN, rhs, dp);
            simulation.verbose = true;
            simulation.linearSolve(H_GN, rhs, dp);
            simulation.verbose = false;
            dp *= -1.0;
            gn_timer.stop();
            std::cout << "\tGN takes " << gn_timer.elapsed_sec() << "s" << std::endl;
        }
        else if (optimizer == Newton)
        {
            StiffnessMatrix H;
            objective.hessian(design_parameters, H, false);
            simulation.linearSolve(H, dOdp, dp);
        }
        else if (optimizer == SGN)
        {
            StiffnessMatrix mat_SGN;
            std::vector<Entry> d2Odx2_entries;
            objective.d2Odx2(design_parameters, d2Odx2_entries);
            simulation.cells.edgeWeightsSGNMatrix(mat_SGN, d2Odx2_entries);
            VectorXT rhs_SGN(n_dof_design + n_dof_sim * 2);
            rhs_SGN.setZero();
            rhs_SGN.segment(n_dof_sim, n_dof_design) = -dOdp;
            VectorXT delta;
            LinearSolver::linearSolve(mat_SGN, rhs_SGN, delta, true, true, true);
            dp = delta.segment(n_dof_sim, n_dof_design);
            dp *= -1.0;
            gn_timer.stop();
            std::cout << "\tSGN takes " << gn_timer.elapsed_sec() << "s" << std::endl;
        }

        // filterGradient();
        

        
        VectorXT search_direction = -dp;
        T alpha = objective.maximumStepSize(search_direction);
        std::cout << "|dp|: " << alpha * search_direction.norm() << std::endl;

        for (int ls_cnt = 0; ls_cnt < 20; ls_cnt++)
        {
            VectorXT p_ls = design_parameters_projected + alpha * search_direction;
            if (add_bound_contraint)
                p_ls = p_ls.cwiseMax(lower_bound).cwiseMin(upper_bound);
            T E1 = objective.value(p_ls, true);
            std::cout << "[" << method << "]\tE1: " << E1 << std::endl;
            // std::getchar();
            if (E1 < E0 || ls_cnt > 20)
            {
                design_parameters = p_ls;
                break;
            }
            else
            {
                alpha *= 0.5;
            }
        }
    }
    else if (optimizer == MMA)
    {
        if (step == 0)
        {
            mma_solver.updateDoF(n_dof_design, 0);
            mma_solver.SetAsymptotes(0.2, 0.65, 1.05);
            objective.getDesignParameters(design_parameters);
            std::cout << "initial value max: " << design_parameters.maxCoeff() << " min: " << design_parameters.minCoeff() <<std::endl;
        }
        dOdp = VectorXT::Zero(n_dof_design);
        g_norm = objective.gradient(design_parameters, dOdp, E0);
        // min_p = (design_parameters.array() - mma_step_size).cwiseMax(lower_bound);
        // max_p = (design_parameters.array() + mma_step_size).cwiseMin(upper_bound);
        VectorXT tmp = design_parameters;
        mma_solver.UpdateEigen(design_parameters, dOdp, VectorXT(), VectorXT(), min_p, max_p);
        std::cout << "[" << method << "] iter " << step << " |g|: " << g_norm 
            << " |dp|: " << (design_parameters - tmp).norm() 
            << " obj: " << E0 << std::endl;
        objective.updateDesignParameters(design_parameters);
    }

    // std::cout << "[" << method << "] iter " << step << " |g| " << g_norm 
    //         << " max: " << dOdp.maxCoeff() << " min: " << dOdp.minCoeff()
    //         << " obj: " << E0 << std::endl;
    objective.saveState("output/cells/opt/" + method + "_iter_" + std::to_string(step) + ".obj");
    objective.updateDesignParameters(design_parameters);
    std::string filename = "output/cells/opt/" + method + "_iter_" + std::to_string(step) + ".txt";
    std::ofstream out(filename);
    out << design_parameters << std::endl;
    out.close();
    if (g_norm < tol_g)
        return true;
    return false;
}

void SensitivityAnalysis::optimizeMMA()
{
    std::cout << "########### MMA ###########" << std::endl;
    objective.getDesignParameters(design_parameters);
    T tol_g = 1e-6;
    MMASolver mma(n_dof_design, 0);
    mma.SetAsymptotes(0.2, 0.65, 1.05);
    T mma_step_size = 1;
    int max_mma_iter = 100;

    VectorXT min_p = VectorXT::Zero(n_dof_design);
    VectorXT max_p = VectorXT::Ones(n_dof_design);

    for (int iter = 0; iter < max_mma_iter; iter++)
    {
        VectorXT dOdp(n_dof_design); dOdp.setZero();
        T O;
        T g_norm = objective.gradient(design_parameters, dOdp, O);
        std::cout << "[MMA] iter " << iter << " |g|: " << g_norm << " obj: " << O << std::endl;
        objective.saveState("output/cells/opt/MMA_iter_" + std::to_string(iter) + ".obj");
        if (g_norm < tol_g)
            break;
        min_p = (design_parameters.array() - mma_step_size).cwiseMax(0.0);
        max_p = (design_parameters.array() + mma_step_size).cwiseMin(10.0);
        mma.UpdateEigen(design_parameters, dOdp, VectorXT(), VectorXT(), min_p, max_p);
        objective.updateDesignParameters(design_parameters);
        std::string filename = "output/cells/opt/MMA_iter_" + std::to_string(iter) + ".txt";
        std::ofstream out(filename);
        out << design_parameters << std::endl;
        out.close();
    }
}

void SensitivityAnalysis::optimizeGaussNewton()
{
    std::cout << "########### GAUSS NEWTON ###########" << std::endl;
    int num_iter_max = 100;
    T tol_g = 1e-6;
    int iter = 0;
    int max_GN_iter = 100;
    objective.getDesignParameters(design_parameters);
    

    while (true)
    {
        VectorXT dOdp; T E0;
        T g_norm;
        if (iter == 0)
            g_norm = objective.gradient(design_parameters, dOdp, E0);
        else 
            g_norm = objective.evaluteGradientAndEnergy(design_parameters, dOdp, E0);
    
        std::cout << "[GN] iter " << iter << " |g| " << g_norm 
            << " max: " << dOdp.maxCoeff() << " min: " << dOdp.minCoeff()
            << " obj: " << E0 << std::endl;
        objective.saveState("output/cells/opt/GN_iter_" + std::to_string(iter) + ".obj");

        if (g_norm < tol_g)
            break;
            
        StiffnessMatrix H_GH;
        objective.hessianGN(design_parameters, H_GH, false);
        VectorXT dp;
        simulation.linearSolve(H_GH, dOdp, dp);

        T alpha = 1.0;
        
        for (int ls_cnt = 0; ls_cnt < 15; ls_cnt++)
        {
            VectorXT p_ls = design_parameters - alpha * dp;
            // p_ls = p_ls.cwiseMax(0.0);
            p_ls = p_ls.cwiseMax(0.0);
            T E1 = objective.value(p_ls, true);
            std::cout << "[GN]\tE1: " << E1 << std::endl;
            // std::getchar();
            if (E1 < E0 || ls_cnt > 10)
            {
                design_parameters = p_ls;
                break;
            }
            else
            {
                alpha *= 0.5;
            }
        }
        std::string filename = "output/cells/opt/GN_iter_" + std::to_string(iter) + ".txt";
        std::ofstream out(filename);
        out << design_parameters << std::endl;
        out.close();

        if (iter > max_GN_iter)
            break;
        iter++;
    }
    
}

void SensitivityAnalysis::optimizeGradientDescent()
{
    std::cout << "########### GRADIENT DESCENT ###########" << std::endl;
    int num_iter_max = 100;
    T tol_g = 1e-6;
    objective.getDesignParameters(design_parameters);
    for (int iter = 0; iter < num_iter_max; iter++)
    {
        VectorXT dOdp; T E0;
        T g_norm;
        if (iter == 0)
            g_norm = objective.gradient(design_parameters, dOdp, E0);
        else 
            g_norm = objective.evaluteGradientAndEnergy(design_parameters, dOdp, E0);
        std::cout << "[GD] iter " << iter << " |g| " << g_norm 
            << " max: " << dOdp.maxCoeff() << " min: " << dOdp.minCoeff()
            << " obj: " << E0 << std::endl;
        objective.saveState("output/cells/opt/GD_iter_" + std::to_string(iter) + ".obj");
        if (g_norm < tol_g)
            break;
        T alpha = 1.0;
        int ls_cnt = 0;
        while (true)
        {
            VectorXT p_ls = design_parameters - alpha * dOdp;
            p_ls = p_ls.cwiseMax(0.0);
            T E1 = objective.value(p_ls, false);
            std::cout << "[GD]\tE1: " << E1 << std::endl;
            // std::getchar();
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
        std::cout <<  "[GD]\t #ls: " << ls_cnt << " |dp| " << alpha * g_norm << std::endl;
        // std::getchar();
        std::string filename = "output/cells/opt/GD_iter_" + std::to_string(iter) + ".txt";
        std::ofstream out(filename);
        out << design_parameters << std::endl;
        out.close();
    }
    simulation.save_mesh = true;
    objective.value(design_parameters);
    
}

void SensitivityAnalysis::dxFromdpAdjoint()
{
    std::cout << n_dof_design << std::endl;
    VectorXT dp(n_dof_design);
    std::ifstream in("/home/yueli/Documents/ETH/WuKong/output/cells/opt/MMA_iter_15.txt");
    for (int i = 0; i < n_dof_design; i++)
    {
        in >> dp[i];
    }
    in.close();

    // std::cout << dp << std::endl;
    
    dp -= simulation.cells.edge_weights;
    simulation.cells.edge_weights += dp;

    simulation.verbose = true;
    simulation.staticSolve();

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

    VectorXT dx = solver.solve(dfdpdp);
    dx.normalize();

    savedxdp(dx, dp, "dxdp.txt");
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
    simulation.loadDeformedState("/home/yueli/Documents/ETH/WuKong/output/cells/cell/result.obj");
}

void SensitivityAnalysis::eigenAnalysisOnSensitivityMatrix()
{
    loadEquilibriumState();
    MatrixXT dxdp;
    simulation.cells.dxdpFromdxdpEdgeWeights(dxdp);

    MatrixXT dxdpTdxdp = dxdp.transpose()*dxdp;
    int n_eigen = 20;
    Spectra::DenseSymMatProd<T> op(dxdpTdxdp);
    Spectra::SymEigsSolver<T, Spectra::LARGEST_ALGE, Spectra::DenseSymMatProd<T> > eigs(&op, n_eigen, dxdpTdxdp.rows());

    eigs.init();

    int nconv = eigs.compute();

    if (eigs.info() == Spectra::SUCCESSFUL)
    {
        Eigen::MatrixXd eigen_vectors = eigs.eigenvectors().real();
        Eigen::VectorXd eigen_values = eigs.eigenvalues().real();
        std::cout << eigen_values << std::endl;
        std::ofstream out("dxdp_eigen_vectors.txt");
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
}

void SensitivityAnalysis::svdOnSensitivityMatrix()
{
    // computeEquilibriumState();
    loadEquilibriumState();
    MatrixXT dxdp;
    // buildSensitivityMatrix(dxdp);
    simulation.cells.dxdpFromdxdpEdgeWeights(dxdp);

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(dxdp, Eigen::ComputeThinU | Eigen::ComputeThinV);
	MatrixXT U = svd.matrixU();
	VectorXT Sigma = svd.singularValues();
	MatrixXT V = svd.matrixV();

    std::cout << "sigma: " << std::endl;
    // std::cout << Sigma << std::endl;    
    std::cout << U.rows() << " " << U.cols() << std::endl;
    std::cout << V.rows() << " " << V.cols() << std::endl;
    // std::cout << U.col(0).norm() << std::endl;

    std::cout << "V matrix: " << std::endl;
    // std::cout << V << std::endl;
    
    // std::ofstream out("cell_svd_vectors.txt");
    std::ofstream out("cell_edge_weights_svd_vectors.txt");
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

    out.open("cell_edge_weights_svd_vectors_V.txt");
    out << n_dof_design << std::endl;
    for (int i = 0; i < n_dof_design; i++)
    {
        for (int j = 0; j < n_dof_design; j++)
            out << V(i, j) << " ";
        out << std::endl;
    }
    out << std::endl;
    out.close();

    // out.open("dxdp.txt");
    // for (int i = 0; i < n_dof_design; i++)
    //     dxdp.col(i).normalize();
    // out << dxdp.rows() << " " << dxdp.cols() << std::endl;
    // for (int i = 0; i < n_dof_design; i++)
    //     out << 1.0 << " ";
    // out << std::endl;
    // for (int i = 0; i < n_dof_sim; i++)
    // {
    //     for (int j = 0; j < n_dof_design; j++)
    //         out << dxdp(i, j) << " ";
    //     out << std::endl;
    // }
    // out << std::endl;
    // out.close();
}


void SensitivityAnalysis::savedxdp(const VectorXT& dx, 
        const VectorXT& dp, const std::string& filename)
{
    std::ofstream out(filename);
    out << dx.transpose() << std::endl;
    out << dp.transpose() << std::endl;
    out.close();
}
    