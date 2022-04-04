#include <igl/mosek/mosek_quadprog.h>
#include <igl/readOBJ.h>
#include "../include/SensitivityAnalysis.h"
#include <Eigen/PardisoSupport>
#include <Spectra/SymEigsShiftSolver.h>
#include <Spectra/MatOp/SparseSymShiftSolve.h>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include "../include/LinearSolver.h"
#include <LBFGSB.h>
#include "../include/IpoptSolver.h"
// #include <IpTNLP.hpp>

void SensitivityAnalysis::setSimulationEnergyWeights()
{
    auto& cell_model = simulation.cells;

    // cell_model.add_yolk_volume = false;
    // cell_model.use_sphere_radius_bound = true;
    // cell_model.add_perivitelline_liquid_volume = false;
    // cell_model.woodbury = false;
    cell_model.B *= 0.1;
    // cell_model.By *= 0.1;
    // cell_model.Bp *= 0.1;
    // cell_model.bound_coeff = 0.1;
}

void SensitivityAnalysis::initialize()
{
    // simulation.initializeCells();
    simulation.save_mesh = false;
    simulation.cells.print_force_norm = false;
    simulation.verbose = false;
    
    objective.getSimulationAndDesignDoF(n_dof_sim, n_dof_design);
    objective.bound[0] = 1e-3;
    objective.bound[1] = 5.5 * simulation.cells.unit;
    objective.equilibrium_prev = VectorXT::Zero(simulation.deformed.rows());
    // simulation.cells.edge_weights.setRandom();
    // simulation.cells.edge_weights /= simulation.cells.edge_weights.norm();
    // simulation.cells.edge_weights.setConstant(0.05);
    VectorXT delta = simulation.cells.edge_weights * 0.1;
    // simulation.cells.edge_weights += delta;
    // simulation.cells.edge_weights.setConstant(0.05);
    // std::ifstream in("/home/yueli/Documents/ETH/WuKong/output/cells/opt/load.txt");
    // for (int i = 0; i < simulation.cells.edge_weights.rows(); i++)
    //     in >> simulation.cells.edge_weights[i];
    // in.close();
    objective.getDesignParameters(design_parameters);
    objective.prev_params = design_parameters;
    // objective.rest_configuration = simulation.deformed;
    // Eigen::MatrixXd warm_start_V; Eigen::MatrixXi warm_start_F;
    // igl::readOBJ("/home/yueli/Documents/ETH/WuKong/Projects/CellSim/data/inv_state.obj", warm_start_V, warm_start_F);
    // VectorXT mesh_saved = Eigen::Map<VectorXT>(warm_start_V.data(), warm_start_V.size());
    // objective.rest_configuration = mesh_saved.segment(0, n_dof_sim);
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

void SensitivityAnalysis::mosekQPTest()
{
    bool equality = false;
    if (equality)
    {
        MatrixXT Q(3, 3);
        Q << 6, 2, 1, 2, 5, 2, 1, 2, 4;
        VectorXT c(3);
        c << -8, -3, -3;
            
        MatrixXT A(2, 3);
        A << 1, 0, 1, 0, 1, 1;
        VectorXT lc(2);
        lc << 3, 0;
        VectorXT uc(2);
        uc << 3, 0;

        VectorXT lx(3); 
        lx.setConstant(-1e3);
        
        VectorXT ux(3); 
        ux.setConstant(1e3);

        VectorXT x;    

        igl::mosek::MosekData mosek_data;
        std::vector<VectorXT> lagrange_multipliers;
        bool solve_success = igl::mosek::mosek_quadprog(Q.sparseView(), c, 0.0, A.sparseView(), lc, uc, lx, ux, mosek_data, x, lagrange_multipliers);
        
        VectorXT dLdp(3); 
        dLdp = c + Q * x;

        VectorXT dLdp2 = dLdp;
        std::cout << x.transpose() << std::endl;
        dLdp -= lagrange_multipliers[0].transpose() * A;
        dLdp += lagrange_multipliers[1].transpose() * A;
        dLdp -= lagrange_multipliers[2];
        dLdp += lagrange_multipliers[3];
        std::cout << dLdp.norm() << std::endl;
        
        // solution x(2 -1 1) lambda (3, -2)
    }
    else
    {
        MatrixXT Q(3, 3);
        Q << 1, -1, 1, -1, 2, -2, 1, -2, 4;
        VectorXT c(3);
        c << 2, -3, 1;
            
        MatrixXT A;
        VectorXT lc;
        VectorXT uc;

        VectorXT lx(3); 
        lx.setConstant(0.0);
        
        VectorXT ux(3); 
        ux.setConstant(1.0);

        VectorXT x;    

        igl::mosek::MosekData mosek_data;
        std::vector<VectorXT> lagrange_multipliers;
        bool solve_success = igl::mosek::mosek_quadprog(Q.sparseView(), c, 0.0, A.sparseView(), lc, uc, lx, ux, mosek_data, x, lagrange_multipliers);
        
        VectorXT dLdp(3); 
        dLdp = c + Q * x;

        VectorXT dLdp2 = dLdp;
        std::cout << x.transpose() << std::endl;
        // dLdp -= lagrange_multipliers[0].transpose() * A;
        // dLdp += lagrange_multipliers[1].transpose() * A;
        dLdp -= lagrange_multipliers[2];
        dLdp += lagrange_multipliers[3];
        std::cout << dLdp.norm() << std::endl;
    }
    std::exit(0);
}


bool SensitivityAnalysis::optimizeOneStep(int step, Optimizer optimizer)
{
    
    std::string method;
    if (optimizer == GaussNewton)
        method = "GN";
    else if (optimizer == PGN)
        method = "PGN";
    else if (optimizer == GradientDescent)
        method = "GD";
    else if (optimizer == MMA)
        method = "MMA";
    else if (optimizer == Newton)
        method = "Newton";
    else if (optimizer == SGN)
        method = "SGN";
    else if (optimizer == PSGN)
        method = "PSGN";
    else if (optimizer == SQP)
        method = "SQP";
    else if (optimizer == SSQP)
        method = "SSQP";
    else
        std::cout << "optimizer undefined" << std::endl;
    
    T tol_g = 1e-5;
    T g_norm = 0;
    T E0;
    VectorXT dOdp, dp;

    // mma 
    T mma_step_size = 1.0 * simulation.cells.unit;
    // T lower_bound = 0, upper_bound = 10.0 * simulation.cells.unit;
    T lower_bound = 0.001, upper_bound = 5.5 * simulation.cells.unit;

    VectorXT min_p = VectorXT::Ones(n_dof_design) * lower_bound;
    VectorXT max_p = VectorXT::Ones(n_dof_design) * upper_bound;
    
    if (optimizer == GradientDescent || optimizer == GaussNewton || 
        optimizer == Newton || optimizer == SGN || 
        optimizer == PSGN || optimizer == PGN || 
        optimizer == SQP || optimizer == SSQP)
    {
        if (step == 0 && !resume)
        {
            std::cout << "########### " << method << " ###########" << std::endl;
            objective.getDesignParameters(design_parameters);
            std::cout << "initial value max: " << design_parameters.maxCoeff() << " min: " << design_parameters.minCoeff() <<std::endl;
            g_norm = objective.gradient(design_parameters, dOdp, E0, /*simulate = */true); 
        }
        else
        {
            g_norm = objective.gradient(design_parameters, dOdp, E0, /*simulate = */false);
        }
        
        if (optimizer == PGN || optimizer == PSGN || optimizer == SQP)
        {
            T epsilon = 1e-5;
            VectorXT feasible_point_gradients = dOdp;
            for (int i = 0; i < n_dof_design; i++)
            {
                if (design_parameters[i] < lower_bound + epsilon && dOdp[i] >= 0)
                    feasible_point_gradients[i] = 0.0;
                if (design_parameters[i] > upper_bound - epsilon && dOdp[i] <= 0)
                    feasible_point_gradients[i] = 0.0;
            }
            g_norm = feasible_point_gradients.norm();
        }

        if (step == 0)
            initial_gradient_norm = g_norm;  
        
        std::cout << "[" << method << "] iter " << step << " |g| " << g_norm 
            << " |g_init| " << initial_gradient_norm << " obj: " << E0 << std::endl;

        dp = dOdp;

        Timer gn_timer(false);
        if (optimizer == GaussNewton)
        {
            gn_timer.start();
            MatrixXT H_GN;
            objective.hessianGN(design_parameters, H_GN, /*simulate = */false);
            
            // H_GN.diagonal().array() += 1e-3;
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(H_GN, Eigen::ComputeThinU | Eigen::ComputeThinV);
            MatrixXT V = svd.matrixV();
            // VectorXT Sigma = svd.singularValues();
            
            // std::cout << Sigma.tail<5>().transpose() << std::endl;

            VectorXT rhs = -dOdp;
            dp = H_GN.llt().solve(rhs);
            // MatrixXT V_block = V.block(0, n_dof_design - 6, n_dof_design, 5);
            // std::cout << (V_block.transpose() * (dp.normalized())).norm() << std::endl;

            // std::cout << (dOdp).norm() << " " << (dOdp).maxCoeff() << " " << (dOdp).minCoeff() << std::endl;
            // std::cout << (H_GN*dp).norm() << " " << (H_GN*dp).maxCoeff() << " " << (H_GN*dp).minCoeff() << std::endl;

            std::cout << "\t[EigenLLT] |Ax-b|/|b|: " << (H_GN * dp - rhs).norm() / rhs.norm() << std::endl;
            dp *= -1.0;
            gn_timer.stop();
            std::cout << "\tGN takes " << gn_timer.elapsed_sec() << "s" << std::endl;
        }
        else if (optimizer == SQP)
        {
            gn_timer.start();
            MatrixXT H_GN;
            objective.hessianGN(design_parameters, H_GN, /*simulate = */false);

            auto saveEigenVectors = [&]()
            {
                int n_eigen = 20;
                Spectra::DenseSymMatProd<T> op(H_GN);
                Spectra::SymEigsSolver<T, Spectra::LARGEST_ALGE, Spectra::DenseSymMatProd<T> > eigs(&op, n_eigen, H_GN.rows());

                eigs.init();

                int nconv = eigs.compute();

                if (eigs.info() == Spectra::SUCCESSFUL)
                {
                    Eigen::MatrixXd eigen_vectors = eigs.eigenvectors().real();
                    Eigen::VectorXd eigen_values = eigs.eigenvalues().real();
                    std::cout << "GaussNewton Hessian eigen values: " << eigen_values.tail<5>().transpose() << std::endl;
                }
                // std::exit(0);
            };
            
            
            // saveEigenVectors();

            Eigen::JacobiSVD<Eigen::MatrixXd> svd(H_GN, Eigen::ComputeThinU | Eigen::ComputeThinV);
            VectorXT Sigma = svd.singularValues();
            std::cout << "H_GN singular value last 5: " << Sigma.tail<5>().transpose() << std::endl;
            // H_GN.diagonal().array() += 1e-3;
            // MatrixXT dxdp;
            // simulation.cells.dxdpFromdxdpEdgeWeights(dxdp);
            // Eigen::JacobiSVD<Eigen::MatrixXd> svd2(dxdp, Eigen::ComputeThinU | Eigen::ComputeThinV);
            // Sigma = svd2.singularValues();
            // std::cout << "dxdp singular value first 5: " << Sigma.head<5>().transpose() << std::endl;
            // std::cout << "dxdp singular value last 5: " << Sigma.tail<5>().transpose() << std::endl;
            // std::cout << "min value in H_GN: " << H_GN.minCoeff() << " max value in H_GN : " << H_GN.maxCoeff() << std::endl;
            // std::cout << "min value in dxdp: " << dxdp.minCoeff() << " max value in dxdp : " << dxdp.maxCoeff() << std::endl;
            // std::exit(0);

            StiffnessMatrix Q = H_GN.sparseView();
            
            StiffnessMatrix A;
            VectorXT lc;
            VectorXT uc;
            VectorXT lx(n_dof_design); 
            lx.setConstant(lower_bound);
            lx -= design_parameters;
            
            VectorXT ux(n_dof_design); 
            ux.setConstant(upper_bound);
            ux -= design_parameters;
            

            igl::mosek::MosekData mosek_data;
            std::vector<VectorXT> lagrange_multipliers;
            bool solve_success = igl::mosek::mosek_quadprog(Q, dOdp, E0, A, lc, uc, lx, ux, mosek_data, dp, lagrange_multipliers);
            
            // T dot_search_grad = -dp.normalized().dot(dOdp.normalized());
            // std::cout << "\tmosek success " << solve_success << " dot(dp, -g): " << dot_search_grad << std::endl;
            
            VectorXT dLdp = dOdp + H_GN * dp;

            dLdp -= lagrange_multipliers[2];
            dLdp += lagrange_multipliers[3];
            std::cout << "\t[SQP] |dL/dp|: " << dLdp.norm() << std::endl;
            
            dp *= -1.0;
            gn_timer.stop();
            std::cout << "\t[SQP] takes " << gn_timer.elapsed_sec() << "s" << std::endl;
        }
        else if (optimizer == Newton)
        {
            StiffnessMatrix H;
            VectorXT rhs = -dOdp;
            objective.hessian(design_parameters, H, /*simulate = */false);
            simulation.linearSolve(H, rhs, dp);
            dp *= -1.0;
        }
        else if (optimizer == SGN)
        {
            StiffnessMatrix mat_SGN;
            objective.hessianSGN(design_parameters, mat_SGN, /*simulate = */false);
            
            VectorXT rhs_SGN(n_dof_design + n_dof_sim * 2);
            rhs_SGN.setZero();
            rhs_SGN.segment(n_dof_sim, n_dof_design) = -dOdp;
            VectorXT delta;
            
            gn_timer.start();
            PardisoLDLTSolver solver(mat_SGN, /*use_default=*/false);
            solver.setPositiveNegativeEigenValueNumber(n_dof_sim + n_dof_design, n_dof_sim);
            solver.setRegularizationIndices(n_dof_sim, n_dof_design);

            // EigenLUSolver solver(mat_SGN);
            // PardisoLUSolver solver(mat_SGN, false);
            
            solver.solve(rhs_SGN, delta);
            
            // VectorXT delta2;
            // EigenLUSolver solver_eigenLU(mat_SGN_copy);
            // solver_eigenLU.solve(rhs_SGN, delta2);

            dp = delta.segment(n_dof_sim, n_dof_design);
            
            dp *= -1.0;
            gn_timer.stop();
            std::cout << "\tSGN takes " << gn_timer.elapsed_sec() << "s" << std::endl;
        }
        // std::cout << "here" << std::endl;
        VectorXT search_direction = -dp;
        T alpha = 1.0;//objective.maximumStepSize(search_direction);
        
        std::cout << "[" << method << "]\t|dp|: " << alpha * search_direction.norm() << std::endl;

        int ls_max = 15;
        bool use_gradient = false;
        for (int ls_cnt = 0; ls_cnt < ls_max; ls_cnt++)
        {
            VectorXT p_ls = design_parameters + alpha * search_direction;
            if (optimizer == PGN || optimizer == PSGN || optimizer == SQP)
                p_ls = p_ls.cwiseMax(lower_bound).cwiseMin(upper_bound);
            
            T E1 = objective.value(p_ls, /*simulate=*/true, /*use_previous_equil=*/true);
            std::cout << "[" << method << "]\t ls " << ls_cnt << " E1: " << E1 << " E0: " << E0 << std::endl;
            // std::getchar();
            if (E1 < E0)
            {
                std::cout << "[" << method << "]\tfinal |dp|: " << (p_ls - design_parameters).norm() << " # ls " << ls_cnt << std::endl;
                design_parameters = p_ls;
                break;
            }
            else
            {
                alpha *= 0.5;
            }
            if (ls_cnt == ls_max - 1 && !use_gradient)
            {
                std::cout << "\t[" << method << "] taking gradient step" << std::endl;
                StiffnessMatrix Q(n_dof_design, n_dof_design);
                Q.setIdentity();
            
                StiffnessMatrix A;
                VectorXT lc, uc;
                VectorXT lx(n_dof_design); 
                lx.setConstant(lower_bound);
                lx -= design_parameters;
                
                VectorXT ux(n_dof_design); 
                ux.setConstant(upper_bound);
                ux -= design_parameters;
                
                igl::mosek::MosekData mosek_data;
                std::vector<VectorXT> lagrange_multipliers;
                bool solve_success = igl::mosek::mosek_quadprog(Q, dOdp, E0, A, lc, uc, lx, ux, mosek_data, search_direction, lagrange_multipliers);
                
                VectorXT dLdp = dOdp + search_direction;

                dLdp -= lagrange_multipliers[2];
                dLdp += lagrange_multipliers[3];
                std::cout << "\t[SQP] |dL/dp|: " << dLdp.norm() << std::endl;
                ls_cnt = 0;
                alpha = 1.0;
                use_gradient = true;
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
    if (save_results)
    {
        objective.saveState(data_folder + "/" + method + "_iter_" + std::to_string(step) + ".obj");
        objective.updateDesignParameters(design_parameters);
        std::string filename = data_folder + "/" + method + "_iter_" + std::to_string(step) + ".txt";
        saveDesignParameters(filename, design_parameters);
    }
    if (g_norm < tol_g || step > max_num_iter)
        return true;
    return false;
}

void SensitivityAnalysis::saveDesignParameters(const std::string& filename, const VectorXT& params)
{
    std::ofstream out(filename);
    out << std::setprecision(20) << params << std::endl;
    out.close();
}

void SensitivityAnalysis::optimizeLBFGSB()
{
    using namespace LBFGSpp;
    int cnt = 0;
    auto computeObjAndGradient = [&](const VectorXT& x, VectorXT& grad)
    {
        T energy = 0.0;
        objective.updateDesignParameters(x);
        objective.gradient(x, grad, energy, true);

        // StiffnessMatrix mat_SGN;
        // objective.hessianSGN(x, mat_SGN, /*simulate = */false);
        
        // VectorXT rhs_SGN(n_dof_design + n_dof_sim * 2);
        // rhs_SGN.setZero();
        // rhs_SGN.segment(n_dof_sim, n_dof_design) = -grad;
        // VectorXT delta;
        
        
        // PardisoLDLTSolver solver(mat_SGN, /*use_default=*/false);
        // solver.setPositiveNegativeEigenValueNumber(n_dof_sim + n_dof_design, n_dof_sim);
        // solver.setRegularizationIndices(n_dof_sim, n_dof_design);

        // solver.solve(rhs_SGN, delta);
        
        // grad = -delta.segment(n_dof_sim, n_dof_design);

        objective.saveState("output/cells/lbfgs/lbfgs_iter_" + std::to_string(cnt) + ".obj");
        std::string filename = "output/cells/lbfgs/lbfgs_iter_" + std::to_string(cnt) + ".txt";
        std::ofstream out(filename);
        out << x << std::endl;
        out.close();
        std::cout << "[L-BFGS-G] iter " << cnt << " |g|: " << grad.norm() << " obj: " << energy << std::endl;
        cnt++;
        return energy;
    };
    
    LBFGSBParam<T> param;
    LBFGSBSolver<T> solver(param);

    T lower_bound = 0.001, upper_bound = 5.5 * simulation.cells.unit;
    VectorXT lb = VectorXT::Constant(n_dof_design, lower_bound);
    VectorXT ub = VectorXT::Constant(n_dof_design, upper_bound);

    T fx;
    int niter = solver.minimize(computeObjAndGradient, design_parameters, fx, lb, ub);
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
            g_norm = objective.gradient(design_parameters, dOdp, E0, false);
    
        std::cout << "[GN] iter " << iter << " |g| " << g_norm 
            << " max: " << dOdp.maxCoeff() << " min: " << dOdp.minCoeff()
            << " obj: " << E0 << std::endl;
        objective.saveState("output/cells/opt/GN_iter_" + std::to_string(iter) + ".obj");

        if (g_norm < tol_g)
            break;
            
        MatrixXT H_GH;
        objective.hessianGN(design_parameters, H_GH, false);
        VectorXT dp = H_GH.llt().solve(dOdp);

        T alpha = 1.0;
        
        for (int ls_cnt = 0; ls_cnt < 15; ls_cnt++)
        {
            VectorXT p_ls = design_parameters - alpha * dp;
            // p_ls = p_ls.cwiseMax(0.0);
            p_ls = p_ls.cwiseMax(0.0);
            T E1 = objective.value(p_ls, true, true);
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
            g_norm = objective.gradient(design_parameters, dOdp, E0, false);
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
            T E1 = objective.value(p_ls, true, true);
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
    // loadEquilibriumState();
    simulation.loadDeformedState("current_mesh.obj");

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
    // loadEquilibriumState();
    // simulation.loadDeformedState("current_mesh.obj");
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
    out << V.rows() << " " << V.cols() << std::endl;
    for (int i = 0; i < n_dof_design; i++)
    {
        for (int j = 0; j < n_dof_design; j++)
            out << V(i, j) << " ";
        out << std::endl;
    }
    out << std::endl;
    out.close();

    // out.open("cell_edge_weights_svd_vectors_V.txt");
    // out << n_dof_design << std::endl;
    // for (int i = 0; i < n_dof_design; i++)
    // {
    //     for (int j = 0; j < n_dof_design; j++)
    //         out << V(i, j) << " ";
    //     out << std::endl;
    // }
    // out << std::endl;
    // out.close();

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

void SensitivityAnalysis::checkStatesAlongGradient()
{
    objective.bound << 0.001, 5.5 * simulation.cells.unit;
    simulation.newton_tol = 1e-6;
    objective.perturb = false;
    T E0;
    VectorXT dOdp, dp;
    VectorXT ew;
    simulation.loadDeformedState("/home/yueli/Documents/ETH/WuKong/output/cells/51/SQP_iter_10.obj");
    simulation.loadEdgeWeights("/home/yueli/Documents/ETH/WuKong/output/cells/51/SQP_iter_10.txt", ew);
    // simulation.loadDeformedState("/home/yueli/Documents/ETH/WuKong/output/cells/23/200.obj");
    // simulation.loadEdgeWeights("/home/yueli/Documents/ETH/WuKong/output/cells/23/200.txt", ew);
    simulation.cells.edge_weights = ew;
    
    // VectorXT target_deltax(objective.target_positions.size() * 3);
    // objective.iterateTargets([&](int cell_idx, TV& target_pos)
    // {
    //     target_deltax
    // });
    // MatrixXT dxdp;
    // simulation.cells.dxdpFromdxdpEdgeWeights(dxdp);
    // Eigen::JacobiSVD<Eigen::MatrixXd> svd_dxdp(dxdp, Eigen::ComputeThinU | Eigen::ComputeThinV);
	// MatrixXT U0 = svd_dxdp.matrixU();
	// VectorXT Sigma0 = svd_dxdp.singularValues();
	// MatrixXT V0 = svd_dxdp.matrixV();
    
    // simulation.deformed = simulation.undeformed + simulation.u + U0.col(0);
    // simulation.saveState("U0.obj");
    // simulation.deformed = simulation.undeformed + simulation.u + 0.5 * U0.col(0);
    // simulation.saveState("U0_0.5.obj");
    // simulation.deformed = simulation.undeformed + simulation.u + 0.1 * U0.col(0);
    // simulation.saveState("U0_0.1.obj");

    // std::cout << Sigma0.head<5>().transpose() << std::endl;
    // std::cout << Sigma0.tail<5>().transpose() << std::endl;
    // return;
    // objective.diffTestdOdxScale();
    // objective.diffTestd2Odx2();
    // objective.diffTestGradientScale();
    // objective.diffTestGradient();
    // return;


    objective.getDesignParameters(design_parameters);
    VectorXT u = simulation.u;
    T lower_bound = 0.001, upper_bound = 5.5 * simulation.cells.unit;
    if (save_results)
        simulation.saveState(data_folder + "/start.obj");
    MatrixXT H_GN;
    objective.hessianGN(design_parameters, H_GN, /*simulate = */false);
    T g_norm = objective.gradient(design_parameters, dOdp, E0, /*simulate = */false);
    simulation.checkHessianPD(false);
    
    std::vector<int> proj_entries;
    bool iecs = true;
    T eps = 1e-5;
    for (int i = 0; i < n_dof_design; i++)
    {
        if (design_parameters[i] < lower_bound || design_parameters[i] > upper_bound)
            iecs = false;
        
        if (design_parameters[i] < lower_bound + eps && dOdp[i] >= 0.0)
            proj_entries.push_back(i);
        if (design_parameters[i] > upper_bound - eps && dOdp[i] <= 0.0)
            proj_entries.push_back(i);
    }
    VectorXT projected_gradient = dOdp;
    for (int idx : proj_entries)
    {
        projected_gradient[idx] = 0;
    }
    g_norm = projected_gradient.norm();

    std::cout << "|g|: " << g_norm << " E0: " << E0 << std::endl;

    // std::cout << iecs << " " << proj_entries.size() << " " << projected_gradient.norm() << std::endl;
    // std::exit(0);

    // H_GN.diagonal().array() += 1e-6;

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(H_GN, Eigen::ComputeThinU | Eigen::ComputeThinV);
	
	VectorXT Sigma = svd.singularValues();
	std::cout << Sigma.tail<5>().transpose() << std::endl;
    H_GN.setIdentity();
    // H_GN.diagonal().array() += 1e-3;

    StiffnessMatrix Q = H_GN.sparseView();
    
    StiffnessMatrix A;
    VectorXT lc;
    VectorXT uc;
    VectorXT lx(n_dof_design); 
    lx.setConstant(lower_bound);
    lx -= design_parameters;
    
    VectorXT ux(n_dof_design); 
    ux.setConstant(upper_bound);
    ux -= design_parameters;

    // std::cout << ux.minCoeff() << " " << ux.maxCoeff() << " " << lx.minCoeff() << " " << lx.maxCoeff() << std::endl;
    

    igl::mosek::MosekData mosek_data;
    std::vector<VectorXT> lagrange_multipliers;
    bool solve_success = igl::mosek::mosek_quadprog(Q, dOdp, E0, A, lc, uc, lx, ux, mosek_data, dp, lagrange_multipliers);

    VectorXT qp_gradient = H_GN * dp + dOdp;
            
    std::vector<int> cons_entries;
    bool inequality_constraint_satisfied = true;
    for (int i = 0; i < n_dof_design; i++)
    {
        if (dp[i] < lx[i] || dp[i] > ux[i])
            inequality_constraint_satisfied = false;
        
        if (dp[i] < lx[i] + 1e-5 && qp_gradient[i] >= 0.0)
            cons_entries.push_back(i);
        if (dp[i] > ux[i] - 1e-5 && qp_gradient[i] <= 0.0)
            cons_entries.push_back(i);
    }
    std::cout << "\t[SQP] inequality_constraint_satisfied " << inequality_constraint_satisfied << std::endl;
    // std::cout << cons_entries.size() << std::endl;
    for (int idx : cons_entries)
    {
        qp_gradient[idx] = 0;
    }
    std::cout << "\t[SQP] dL/ddp hack: " <<  (qp_gradient).norm() << " " << std::endl;
    
    VectorXT dLdp = dOdp + H_GN * dp;

    dLdp -= lagrange_multipliers[2];
    dLdp += lagrange_multipliers[3];
    std::cout << "\t[SQP] |dL/ddp| LM: " << dLdp.norm() << std::endl;
    std::cout << dp.norm() << std::endl;
    
    int n_steps = 25;
    T alpha = 1.0;
    // for (int i = 0; i < n_steps; i++)
    // {
    //     std::cout << "step " << i << std::endl;
    //     VectorXT delta = alpha * dp;
    //     VectorXT p_ls = design_parameters + delta;
    //     T ei = objective.value(p_ls, true, true);
    //     std::cout << ei << " " << E0 << std::endl;
    //     if (save_results)
    //     {
    //         simulation.saveState(data_folder + "/ls_" + std::to_string(i) + ".obj");
    //         std::cout << ei << std::endl;
    //         std::string filename = data_folder + "/ls_" + std::to_string(i) + ".txt";
    //         std::ofstream out(filename);
    //         out << p_ls << std::endl;
    //         out.close();
    //     }
    //     alpha *= 0.5;
    // }
    // std::exit(0);

    // 

    // T step_size = 5e-5;
    // int step = 200; 
    VectorXT walking_direction = dp.normalized();
    T step_size = 5e-3;
    int step = 50; 

    // T step_size = 1e-5;
    // int step = 100000;
    std::vector<T> energies;
    std::vector<T> steps;
    // for (T xi = -T(step/2) * step_size; xi < T(step/2) * step_size; xi+=step_size)
    // for (T xi = 0.0002; xi < T(step/2) * step_size; xi+=step_size)
    for (T xi = 0; xi < T(step/2) * step_size; xi+=step_size)
    // for (T xi = -1.0 * 10e-3; xi < 1.0 * 10e-3; xi+=step_size)
    {
        T Ei = objective.value(design_parameters + xi * walking_direction, true, true);
        simulation.checkHessianPD(false);
        energies.push_back(Ei);
        steps.push_back(xi);
        T min_edge_length = 1e10;
        simulation.cells.iterateApicalEdgeSerial([&](Edge& e)
        {
            TV vi = simulation.deformed.segment<3>(e[0] * 3);
            TV vj = simulation.deformed.segment<3>(e[1] * 3);
            T edge_length = (vj - vi).norm();
            if (edge_length < min_edge_length)
                min_edge_length = edge_length;
        });
        if (save_results)
        {
            simulation.saveState(data_folder + "/" + std::to_string(energies.size() - 1) + ".obj");
            std::string filename = data_folder + "/"  + std::to_string(energies.size() - 1) + ".txt";
            VectorXT param = design_parameters + xi * walking_direction;
            saveDesignParameters(filename, design_parameters);
        }
        std::cout << "[debug]: " << xi  << " obj: " << Ei << " step " << energies.size() - 1 << " " << min_edge_length << std::endl;
        // std::getchar();
        // if (xi > 0.0004)
        //     break;
    }
    for (T e : energies)
    {
        std::cout << std::setprecision(12) <<  e << " ";
    }
    std::cout << std::endl;
    for (T idx : steps)
    {
        std::cout << idx << " ";
    }
    std::cout << std::endl;
    std::exit(0);
}

void SensitivityAnalysis::sampleEnergyWithSearchAndGradientDirection(const VectorXT& search_direction)
{
    T step_size = 1e-2;
    int step = 100; 

    std::vector<T> energies;
    std::vector<T> energies_gd;
    std::vector<T> steps;
    int step_cnt = 1;
    for (T xi = -T(step/2) * step_size; xi < T(step/2) * step_size; xi+=step_size)
    {
        T Ei = objective.value(design_parameters + xi * search_direction, true, true);
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

int SensitivityAnalysis::optimizeIPOPT()
{
    Ipopt::SmartPtr<Ipopt::IpoptApplication> app = IpoptApplicationFactory();
    
    app->RethrowNonIpoptException(true);
    

    app->Options()->SetNumericValue("tol", 1e-5);
    // app->Options()->SetStringValue("mu_strategy", "monotone");
    app->Options()->SetStringValue("mu_strategy", "adaptive");

    app->Options()->SetStringValue("output_file", data_folder + "ipopt.out");
    app->Options()->SetStringValue("hessian_approximation", "limited-memory");
    //        app->Options()->SetNumericValue("mu_max", 0.0001);
    //        app->Options()->SetNumericValue("constr_viol_tol", T(1e-7));
    //        app->Options()->SetNumericValue("acceptable_constr_viol_tol", T(1e-7));
    //        bound_relax_factor
    //        app->Options()->SetStringValue("derivative_test", "first-order");
    // The following overwrites the default name (ipopt.opt) of the
    // options file
    // app->Options()->SetStringValue("option_file_name", "hs071.opt");
    
    // Initialize the IpoptApplication and process the options
    
    Ipopt::ApplicationReturnStatus status;
    status = app->Initialize();
    if (status != Ipopt::Solve_Succeeded) 
    {
        std::cout << std::endl
                    << std::endl
                    << "*** Error during initialization!" << std::endl;
        return (int)status;
    }

    // Ask Ipopt to solve the problem
    std::cout << "Solving problem using IPOPT" << std::endl;
    
    Ipopt::SmartPtr<IpoptSolver> mynlp = new IpoptSolver(&objective, data_folder);
    
    status = app->OptimizeTNLP(mynlp);

    if (status == Ipopt::Solve_Succeeded) {
        std::cout << std::endl
                    << std::endl
                    << "*** The problem solved!" << std::endl;
    }
    else {
        std::cout << std::endl
                    << std::endl
                    << "*** The problem FAILED!" << std::endl;
    }
    return (int)status;
}