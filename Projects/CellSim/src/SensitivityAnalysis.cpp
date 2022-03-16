#include <igl/mosek/mosek_quadprog.h>
#include "../include/SensitivityAnalysis.h"
#include <Eigen/PardisoSupport>
#include <Spectra/SymEigsShiftSolver.h>
#include <Spectra/MatOp/SparseSymShiftSolve.h>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include "../include/LinearSolver.h"

void SensitivityAnalysis::setSimulationEnergyWeights()
{
    auto& cell_model = simulation.cells;

    cell_model.add_yolk_volume = false;
    cell_model.use_sphere_radius_bound = true;
    cell_model.add_perivitelline_liquid_volume = false;
    cell_model.woodbury = false;
    // cell_model.B = 1e4;
    // cell_model.bound_coeff = 0.1;
}

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
    // std::ifstream in("/home/yueli/Documents/ETH/WuKong/output/cells/opt/load.txt");
    // for (int i = 0; i < simulation.cells.edge_weights.rows(); i++)
    //     in >> simulation.cells.edge_weights[i];
    // in.close();
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
    
    T tol_g = 1e-6;
    T g_norm = 0;
    T E0;
    VectorXT dOdp, dp;

    // mma 
    T mma_step_size = 1.0 * simulation.cells.unit;
    T lower_bound = 0, upper_bound = 10.0 * simulation.cells.unit;

    VectorXT min_p = VectorXT::Ones(n_dof_design) * lower_bound;
    VectorXT max_p = VectorXT::Ones(n_dof_design) * upper_bound;

    if (optimizer == GradientDescent || optimizer == GaussNewton || 
        optimizer == Newton || optimizer == SGN || 
        optimizer == PSGN || optimizer == PGN || 
        optimizer == SQP || optimizer == SSQP)
    {
        if (step == 0)
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
        g_norm = dOdp.norm();


        // std::vector<int> projected_entries;
        std::unordered_set<int> binding_set;
        auto getProjectingEntries =[&](VectorXT& g)
        {
            T epsilon = 1e-5;
            for (int i = 0; i < n_dof_design; i++)
            {
                // if (design_parameters[i] < lower_bound + epsilon && g[i] > 0)
                if (design_parameters[i] < lower_bound + epsilon)
                {
                    binding_set.insert(i);
                }
                // if (design_parameters[i] > upper_bound - epsilon && g[i] < 0)
                if (design_parameters[i] > upper_bound - epsilon)
                {
                    binding_set.insert(i);
                }
            }
            std::cout << "\t[" << method << "] project " << binding_set.size() << "/" << n_dof_design << std::endl;
        };

        if (optimizer == PGN || optimizer == PSGN || optimizer == SQP)
        {
            getProjectingEntries(dOdp);
            VectorXT feasible_point_gradients = dOdp;
            for (int idx : binding_set)
                feasible_point_gradients[idx] = 0;
            // std::cout << "[" << method << "]\tprojected: " << feasible_point_gradients.norm() << std::endl;
            g_norm = feasible_point_gradients.norm();
            // VectorXT free_set_mask = VectorXT::Ones(n_dof_design);
            // for (int idx : binding_set)
            //     free_set_mask[idx] = 0.0;
            // search_direction = search_direction.array() * free_set_mask.array();
        }
        
        std::cout << "[" << method << "] iter " << step << " |g| " << g_norm 
            << " obj: " << E0 << std::endl;

        dp = dOdp;

        Timer gn_timer(false);
        if (optimizer == GaussNewton)
        {
            gn_timer.start();
            MatrixXT H_GN;
            objective.hessianGN(design_parameters, H_GN, /*simulate = */false);
            
            VectorXT rhs = -dOdp;
            dp = H_GN.llt().solve(rhs);
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
            H_GN.diagonal().array() += 1e-6;
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(H_GN, Eigen::ComputeThinU | Eigen::ComputeThinV);
            VectorXT Sigma = svd.singularValues();
            std::cout << Sigma.tail<5>().transpose() << std::endl;

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
            
            bool solve_success = igl::mosek::mosek_quadprog(Q, dOdp, E0, A, lc, uc, lx, ux, mosek_data, dp);
            T dot_search_grad = -dp.normalized().dot(dOdp.normalized());
            std::cout << "\tmosek success " << solve_success << " dot(dp, -g): " << dot_search_grad << std::endl;
            VectorXT error = H_GN * dp + dOdp;
            
            std::vector<int> cons_entries;
            bool inequality_constraint_satisfied = true;
            for (int i = 0; i < n_dof_design; i++)
            {
                if (dp[i] < lx[i] || dp[i] > ux[i])
                    inequality_constraint_satisfied = false;
                
                if (dp[i] < lx[i] + 1e-5)
                    cons_entries.push_back(i);
                if (dp[i] > ux[i] - 1e-5)
                    cons_entries.push_back(i);
            }
            std::cout << "\t[SQP] inequality_constraint_satisfied " << inequality_constraint_satisfied << std::endl;
            // std::cout << cons_entries.size() << std::endl;
            for (int idx : cons_entries)
            {
                error[idx] = 0;
            }
            std::cout << "\t[SQP] dL/ddp " <<  (error).norm() << " " << std::endl;
            // std::getchar();
            dp *= -1.0;
            gn_timer.stop();
            std::cout << "\t[SQP] takes " << gn_timer.elapsed_sec() << "s" << std::endl;
        }
        else if (optimizer == SSQP)
        {
            gn_timer.start();
            MatrixXT H_GN;
            objective.hessianGN(design_parameters, H_GN, /*simulate = */false);
            
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
            
            bool solve_success = igl::mosek::mosek_quadprog(Q, dOdp, E0, A, lc, uc, lx, ux, mosek_data, dp);
            T dot_search_grad = -dp.normalized().dot(dOdp.normalized());
            std::cout << "\tmosek success " << solve_success << " dot(dp, -g): " << dot_search_grad << std::endl;
            VectorXT error = H_GN * dp + dOdp;
            
            std::vector<int> cons_entries;
            bool inequality_constraint_satisfied = true;
            for (int i = 0; i < n_dof_design; i++)
            {
                if (dp[i] < lx[i] || dp[i] > ux[i])
                    inequality_constraint_satisfied = false;
                
                if (dp[i] < lx[i] + 1e-5)
                    cons_entries.push_back(i);
                if (dp[i] > ux[i] - 1e-5)
                    cons_entries.push_back(i);
            }
            std::cout << "\t[SQP] inequality_constraint_satisfied " << inequality_constraint_satisfied << std::endl;
            // std::cout << cons_entries.size() << std::endl;
            for (int idx : cons_entries)
            {
                error[idx] = 0;
            }
            std::cout << "\t[SQP] dL/ddp " <<  (error).norm() << " " << std::endl;
            // std::getchar();
            dp *= -1.0;
            gn_timer.stop();
            std::cout << "\t[SQP] takes " << gn_timer.elapsed_sec() << "s" << std::endl;
        }
        else if (optimizer == PGN)
        {
            gn_timer.start();
            MatrixXT H_GN;
            objective.hessianGN(design_parameters, H_GN, /*simulate = */false);
            for (int idx : binding_set)
            {
                H_GN.row(idx) *= 0.0;
                H_GN.col(idx) *= 0.0;
                H_GN.coeffRef(idx, idx) = 1.0;
            }
            dp = H_GN.llt().solve(dOdp);
            getProjectingEntries(dp);
            VectorXT rhs = -dOdp;
            for (int idx : binding_set)
            {
                H_GN.row(idx) *= 0.0;
                H_GN.col(idx) *= 0.0;
                H_GN.coeffRef(idx, idx) = 1.0;
            }
            dp = H_GN.llt().solve(rhs);
            std::cout << "\t[EigenLLT] |Ax-b|/|b|: " << (H_GN * dp - rhs).norm() / rhs.norm() << std::endl;
            dp *= -1.0;
            gn_timer.stop();
            std::cout << "\tGN takes " << gn_timer.elapsed_sec() << "s" << std::endl;
        }
        else if (optimizer == Newton)
        {
            StiffnessMatrix H;
            VectorXT rhs = -dOdp;
            objective.hessian(design_parameters, H, /*simulate = */false);
            simulation.linearSolve(H, rhs, dp);
            dp *= -1.0;
        }
        else if (optimizer == PSGN)
        {
            StiffnessMatrix mat_SGN;
            objective.hessianSGN(design_parameters, mat_SGN, /*simulate = */false);
            
            VectorXT rhs_SGN(n_dof_design + n_dof_sim * 2);
            rhs_SGN.setZero();
            rhs_SGN.segment(n_dof_sim, n_dof_design) = -dOdp;
            VectorXT delta;
            
            // This gives correct result
            // LinearSolver::solve<Eigen::SparseLU<StiffnessMatrix>>(mat_SGN, rhs_SGN, delta, n_dof_sim, n_dof_design);
            for (int idx : binding_set)
            {
                mat_SGN.row(idx + n_dof_sim) *= 0.0;
                mat_SGN.col(idx + n_dof_sim) *= 0.0;
                mat_SGN.coeffRef(idx + n_dof_sim, idx + n_dof_sim) = 1.0;
            }
            // StiffnessMatrix mat_SGN_copy = mat_SGN;
            // std::cout << projected_entries.size() << std::endl;

            // std::getchar();

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
            // getProjectingEntries(dp);
            // for (int idx : binding_set)
            // {
            //     mat_SGN.row(idx + n_dof_sim) *= 0.0;
            //     mat_SGN.col(idx + n_dof_sim) *= 0.0;
            //     mat_SGN.coeffRef(idx + n_dof_sim, idx + n_dof_sim) = 1.0;
            // }
            // solver.solve(rhs_SGN, delta);
            // dp = delta.segment(n_dof_sim, n_dof_design);
            dp *= -1.0;
            gn_timer.stop();
            std::cout << "\tSGN takes " << gn_timer.elapsed_sec() << "s" << std::endl;
            // gn_timer.restart();

            // MatrixXT H_GN;
            // objective.hessianGN(design_parameters, H_GN, false);
            // // gn_timer.stop();
            // // std::cout << "\tGN takes " << gn_timer.elapsed_sec() << "s" << std::endl;
            // for (int idx : projected_entries)
            // {
            //     H_GN.row(idx) *= 0.0;
            //     H_GN.col(idx) *= 0.0;
            //     H_GN.coeffRef(idx, idx) = 1.0;
            // }

            // VectorXT search_direction_G = -H_GN.llt().solve(dOdp);
            // gn_timer.stop();
            // std::cout << "\tGN takes " << gn_timer.elapsed_sec() << "s" << std::endl;
            // search_direction_GN *= -1.0;
            // std::cout << (search_direction_GN - dp).norm() << std::endl;
            // std::cout << search_direction_GN.normalized().dot(dOdp.normalized()) << std::endl;
            // std::cout << dp.normalized().dot(dOdp.normalized()) << std::endl;
            // for (int i = 0; i < n_dof_design; i++)
            // {
            //     std::cout << search_direction_GN[i] << " " << dp[i] << std::endl;
            //     std::getchar();
            // }
            // std::getchar();
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
        T alpha = objective.maximumStepSize(search_direction);
        
        std::cout << "[" << method << "]\t|dp|: " << alpha * search_direction.norm() << std::endl;

        // if (optimizer == PGN || optimizer == PSGN || optimizer == SQP)
        // {
        //     VectorXT free_set_mask = VectorXT::Ones(n_dof_design);
        //     for (int idx : binding_set)
        //         free_set_mask[idx] = 0.0;
        //     search_direction = search_direction.array() * free_set_mask.array();
        // }

        std::cout << "[" << method << "]\t starting alpha: " << alpha << std::endl;
        for (int ls_cnt = 0; ls_cnt < 20; ls_cnt++)
        {

            VectorXT p_ls = design_parameters + alpha * search_direction;
            if (optimizer == PGN || optimizer == PSGN || optimizer == SQP)
                p_ls = p_ls.cwiseMax(lower_bound).cwiseMin(upper_bound);
            T E1 = objective.value(p_ls, /*simulate=*/true, /*use_previous_equil=*/true);
            std::cout << "[" << method << "]\tE1: " << E1 << std::endl;
            // std::getchar();
            if (E1 < E0 || ls_cnt > 20)
            {
                if (ls_cnt > 20)
                {
                    sampleEnergyWithSearchAndGradientDirection(search_direction);
                    std::getchar();
                }
                std::cout << "[" << method << "]\tfinal |dp|: " << (p_ls - design_parameters).norm() << " # ls " << ls_cnt << std::endl;
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

void SensitivityAnalysis::optimizeNLOPT()
{
    


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
    T E0;
    VectorXT dOdp, dp;
    VectorXT ew;
    simulation.loadEdgeWeights("/home/yueli/Documents/ETH/WuKong/output/cells/opt/SQP_iter_25.txt", ew);
    simulation.cells.edge_weights = ew;
    simulation.loadDeformedState("/home/yueli/Documents/ETH/WuKong/output/cells/opt/SQP_iter_25.obj");
    
    objective.getDesignParameters(design_parameters);
    VectorXT u = simulation.u;
    T lower_bound = 0, upper_bound = 10.0 * simulation.cells.unit;
    simulation.saveState("output/cells/debug/start.obj");
    MatrixXT H_GN;
    objective.hessianGN(design_parameters, H_GN, /*simulate = */false);
    T g_norm = objective.gradient(design_parameters, dOdp, E0, /*simulate = */false);
    std::cout << "|g|: " << g_norm << " E0: " << E0 << std::endl;
    
    H_GN.diagonal().array() += 1e-6;

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
    
    bool solve_success = igl::mosek::mosek_quadprog(Q, dOdp, E0, A, lc, uc, lx, ux, mosek_data, dp);

    VectorXT error = H_GN * dp + dOdp;
            
    std::vector<int> cons_entries;
    bool inequality_constraint_satisfied = true;
    for (int i = 0; i < n_dof_design; i++)
    {
        if (dp[i] < lx[i] || dp[i] > ux[i])
            inequality_constraint_satisfied = false;
        
        if (dp[i] < lx[i] + 1e-5)
            cons_entries.push_back(i);
        if (dp[i] > ux[i] - 1e-5)
            cons_entries.push_back(i);
    }
    std::cout << "\t[SQP] inequality_constraint_satisfied " << inequality_constraint_satisfied << std::endl;
    // std::cout << cons_entries.size() << std::endl;
    for (int idx : cons_entries)
    {
        error[idx] = 0;
    }
    std::cout << "\t[SQP] dL/ddp " <<  (error).norm() << " " << std::endl;
    
    int n_steps = 25;
    T alpha = 0.125;
    for (int i = 0; i < n_steps; i++)
    {
        std::cout << "step " << i << std::endl;
        VectorXT delta = alpha * dp;
        VectorXT p_ls = design_parameters + delta;
        std::cout << "\t[SQP] |dp| " <<  delta.norm() << " max "  << delta.maxCoeff() 
            << " min: " << delta.minCoeff() << " " << std::endl;
        T ei = objective.value(p_ls, true, true);
        simulation.saveState("output/cells/debug/" + std::to_string(i) + ".obj");
        std::cout << ei << std::endl;
        alpha *= 0.5;
    }

    T step_size = 1e-6;
    int step = 100; 
    std::vector<T> energies;
    std::vector<T> steps;
    // for (T xi = -T(step/2) * step_size; xi < T(step/2) * step_size; xi+=step_size)
    for (T xi = 0; xi < T(step/2) * step_size; xi+=step_size)
    {
        T Ei = objective.value(design_parameters + xi * dp, true, true);
        std::cout << Ei << std::endl;
        energies.push_back(Ei);
        steps.push_back(xi);
        std::getchar();
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