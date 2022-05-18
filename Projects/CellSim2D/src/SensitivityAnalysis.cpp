#include <igl/mosek/mosek_quadprog.h>
#include <Eigen/PardisoSupport>
#include "../include/LinearSolver.h"
#include "../include/SensitivityAnalysis.h"
#include "../include/IpoptSolver.h"
void SensitivityAnalysis::initialize()
{
    vertex_model.verbose = false;
    objective.getSimulationAndDesignDoF(n_dof_sim, n_dof_design);
    objective.bound[0] = 1e-5;
    objective.bound[1] = vertex_model.w_c * 1.5;
    objective.mask[0] = true;
    objective.mask[1] = false;
    objective.equilibrium_prev = VectorXT::Zero(vertex_model.deformed.rows());
    
    std::cout << "n_dof_sim " <<  n_dof_sim << " n_dof_design: " << n_dof_design << std::endl;
}

void SensitivityAnalysis::checkStateAlongDirection()
{
    
    objective.perturb = false;
    T E0;
    VectorXT dOdp, dp;

    vertex_model.loadStates("/home/yueli/Documents/ETH/WuKong/output/cells/437/SQP_iter_18.obj");
    vertex_model.loadEdgeWeights("/home/yueli/Documents/ETH/WuKong/output/cells/437/SQP_iter_18.txt", vertex_model.apical_edge_contracting_weights);

    objective.getDesignParameters(design_parameters);
    VectorXT u = vertex_model.u;
    T lower_bound = objective.bound[0], upper_bound = objective.bound[1];
    if (save_results)
        vertex_model.saveStates(data_folder + "/start.obj");
    MatrixXT H_GN;
    objective.hessianGN(design_parameters, H_GN, /*simulate = */false);
    T g_norm = objective.gradient(design_parameters, dOdp, E0, /*simulate = */false);
    vertex_model.checkHessianPD(false);

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

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(H_GN, Eigen::ComputeThinU | Eigen::ComputeThinV);
	
	VectorXT Sigma = svd.singularValues();
	std::cout << "H_GN smallest 5 singular values: " << Sigma.tail<5>().transpose() << std::endl;
    std::cout << "H_GN largest 5 singular values: " << Sigma.head<5>().transpose() << std::endl;
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
    std::cout << "search direction dot: " << -dp.normalized().dot(dOdp.normalized()) << std::endl;
    int n_steps = 25;
    T alpha = 1.0;

    VectorXT walking_direction = dp.normalized();
    // VectorXT walking_direction = -dOdp.normalized();
    T step_size = 1e-6;
    int step = 100; 

    // T step_size = 1e-5;
    // int step = 100000;
    std::vector<T> energies;
    std::vector<std::vector<T>> energies_all_terms;
    std::vector<T> steps;
    for (T xi = -T(step/2) * step_size; xi < T(step/2) * step_size; xi+=step_size + 1e-6)
    // for (T xi = 0.0002; xi < T(step/2) * step_size; xi+=step_size)
    // for (T xi = 0; xi < T(step/2) * step_size; xi+=step_size)
    // for (T xi = -1.0 * 10e-3; xi < 1.0 * 10e-3; xi+=step_size)
    {
        T Ei = objective.value(design_parameters + xi * walking_direction, true, true);
        vertex_model.checkHessianPD(false);
        energies.push_back(Ei);
        std::vector<T> energy_all_terms;
        // objective.computeEnergyAllTerms(design_parameters + xi * walking_direction, energy_all_terms, true, true);
        energies_all_terms.push_back(energy_all_terms);
        steps.push_back(xi);
       
        if (save_results)
        {
            vertex_model.saveStates(data_folder + "/" + std::to_string(energies.size() - 1) + ".obj");
            std::string filename = data_folder + "/"  + std::to_string(energies.size() - 1) + ".txt";
            VectorXT param = design_parameters + xi * walking_direction;
            saveDesignParameters(filename, param);
        }
        std::cout << "[debug]: " << xi  << " obj: " << Ei << " step " << energies.size() - 1 << std::endl;
        
    }
    for (T e : energies)
    {
        std::cout << std::setprecision(12) <<  e << " ";
    }
    std::cout << std::endl;
    std::cout << "energy all terms" << std::endl;
    for (int j = 0; j < energies_all_terms[0].size(); j++)
    {
        std::cout << "term " << j << std::endl;
        for (int i = 0; i < energies_all_terms.size(); i++)
        {
            std::cout << std::setprecision(12) <<  energies_all_terms[i][j] << " ";
        }
        std::cout << std::endl;
    }
    
    for (T idx : steps)
    {
        std::cout << idx << " ";
    }
    std::cout << std::endl;
    std::exit(0);
}

bool SensitivityAnalysis::optimizeIPOPT()
{
    Ipopt::SmartPtr<Ipopt::IpoptApplication> app = IpoptApplicationFactory();
    
    app->RethrowNonIpoptException(true);
    

    app->Options()->SetNumericValue("tol", 1e-5);
    app->Options()->SetStringValue("mu_strategy", "monotone");
    // app->Options()->SetStringValue("mu_strategy", "adaptive");

    app->Options()->SetStringValue("output_file", data_folder + "/ipopt.out");
    // app->Options()->SetStringValue("hessian_approximation", "limited-memory");
    // app->Options()->SetIntegerValue("limited_memory_max_history", 50);
    app->Options()->SetIntegerValue("accept_after_max_steps", 20);
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
    
    // objective.bound[0] = 1e-5;
    // objective.bound[1] = 12.0 * simulation.cells.unit;

    Ipopt::SmartPtr<IpoptSolver> mynlp = new IpoptSolver(objective, data_folder);
    
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
    else if (optimizer == SQP)
        method = "SQP";
    else if (optimizer == SSQP)
        method = "SSQP";
    else
        std::cout << "optimizer undefined" << std::endl;

    T E0;
    VectorXT dOdp, dp;
    T g_norm = 1e6;
    T rel_tol = 1e-4;
    T abs_tol = 1e-6;
    T lower_bound = objective.bound[0], upper_bound = objective.bound[1];
    std::cout << "lower bound " << lower_bound << " upper bound " << upper_bound << std::endl;
    if (optimizer == GradientDescent || optimizer == GaussNewton || 
        optimizer == Newton || optimizer == SGN || 
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
        if (optimizer == SQP || optimizer == SSQP)
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
        {
            initial_gradient_norm = g_norm;
        }
        std::cout << "forward simulation hessian eigen values: ";
        vertex_model.checkHessianPD(false);
        std::cout << "[" << method << "] iter " << step << " |g| " << g_norm 
            << " |g_init| " << initial_gradient_norm 
            << " tol rel: " << rel_tol * initial_gradient_norm 
            << " tol abs: " << abs_tol 
            << " obj: " << E0 << std::endl;
        
        if ( g_norm < rel_tol * initial_gradient_norm || g_norm < abs_tol)
            return true;

        dp = dOdp;

        if (optimizer == GaussNewton)
        {

        }
        else if (optimizer == SQP)
        {
            
            MatrixXT H_GN;
            objective.hessianGN(design_parameters, H_GN, /*simulate = */false);

            T alpha_reg = 10e-6;
            while (true)
            {
                Eigen::LLT<Eigen::MatrixXd> lltOfA(H_GN);
                if(lltOfA.info() == Eigen::NumericalIssue)
                {
                    std::cout << "add reg to H_GN" << std::endl;
                    H_GN.diagonal().array() += alpha_reg;
                    alpha_reg *= 10.0;
                }
                else
                    break;
            }

            if (add_reg)
                H_GN.diagonal().array() += reg_w_H;
            
            T reg_alpha = 1e-6;
            StiffnessMatrix A;
            VectorXT lc;
            VectorXT uc;
            VectorXT lx(n_dof_design); 
            lx.setConstant(lower_bound);
            lx -= design_parameters;
            
            VectorXT ux(n_dof_design); 
            ux.setConstant(upper_bound);
            ux -= design_parameters;

            while (true)
            {
                Eigen::JacobiSVD<Eigen::MatrixXd> svd(H_GN, Eigen::ComputeThinU | Eigen::ComputeThinV);
                // MatrixXT U = svd.matrixU();
                VectorXT Sigma = svd.singularValues();
                // MatrixXT V = svd.matrixV();
                std::cout << "\t[SQP] GN Hessian singular values last: " << Sigma.tail<5>().transpose() << std::endl;
                std::cout << "\t[SQP] GN Hessian singular values first: " << Sigma.head<5>().transpose() << std::endl;            
                
                StiffnessMatrix Q = H_GN.sparseView();
                
                igl::mosek::MosekData mosek_data;
                std::vector<VectorXT> lagrange_multipliers;
                bool solve_success = igl::mosek::mosek_quadprog(Q, dOdp, E0, A, lc, uc, lx, ux, mosek_data, dp, lagrange_multipliers);
                
                T dot_search_grad = -dp.normalized().dot(dOdp.normalized());

                VectorXT dLdp = dOdp + H_GN * dp;
            
                dLdp -= lagrange_multipliers[2];
                dLdp += lagrange_multipliers[3];
                std::cout << "\t[SQP] |dL/dp|: " << dLdp.norm() << std::endl;
                std::cout << "dot(search_dir, -gradient) " << dot_search_grad << std::endl;
                VectorXT updated_p = design_parameters + dp;
                
                if (dot_search_grad < 1e-6)
                {
                    H_GN.diagonal().array() += reg_alpha;
                    reg_alpha *= 10.0;
                }
                else
                {
                    dp *= -1.0;
                    break;
                }
                
                // std::cout << "[" << method << "]\tdpT H dp: " << dp.transpose() * H_GN * dp << std::endl;    
            }
            
            
        }
        else if (optimizer == SGN)
        {
            StiffnessMatrix mat_SGN;
            objective.hessianSGN(design_parameters, mat_SGN, /*simulate = */false);
            
            if (add_reg)
                mat_SGN.diagonal().segment(n_dof_sim, n_dof_design).array() += reg_w_H;

            VectorXT rhs_SGN(n_dof_design + n_dof_sim * 2);
            rhs_SGN.setZero();
            rhs_SGN.segment(n_dof_sim, n_dof_design) = -dOdp;
            VectorXT delta;
            
            
            PardisoLDLTSolver solver(mat_SGN, /*use_default=*/false);
            solver.setPositiveNegativeEigenValueNumber(n_dof_sim + n_dof_design, n_dof_sim);
            solver.setRegularizationIndices(n_dof_sim, n_dof_design);

            solver.solve(rhs_SGN, delta);
            
            dp = -delta.segment(n_dof_sim, n_dof_design);
        }
        VectorXT search_direction = -dp;
        T alpha = 1.0;
        
        std::cout << "[" << method << "]\t|dp|: " << search_direction.norm() << std::endl;

        
        if (search_direction.norm() < 1e-10)
            return true;

        int ls_max = 15;
        int ls_cnt = 0;
        while (true)
        {
            ls_cnt++;
            VectorXT p_ls = design_parameters + alpha * search_direction;
            
            T E1 = objective.value(p_ls, /*simulate=*/true, /*use_previous_equil=*/true);
            if (save_ls_states)
            {
                vertex_model.saveStates(data_folder + "/" + method + "_iter_" + std::to_string(step) + "_ls_" + std::to_string(ls_cnt) + ".obj");
            }
            std::cout << "[" << method << "]\t ls " << ls_cnt << " E1: " << E1 << " E0: " << E0 << std::endl;

            if (E1 < E0 || ls_cnt > ls_max)
            {
                std::cout << "[" << method << "]\tfinal |dp|: " << (p_ls - design_parameters).norm() << " # ls " << ls_cnt << std::endl;
                design_parameters = p_ls;
                break;
            }
            alpha *= 0.5;
        }
        // for (int ls_cnt = 0; ls_cnt < ls_max; ls_cnt++)
        // {
        //     VectorXT p_ls = design_parameters + alpha * search_direction;
        //     // saveDesignParameters(data_folder + "/trouble.txt", p_ls);
            
        //     T E1 = objective.value(p_ls, /*simulate=*/true, /*use_previous_equil=*/true);
        //     if (save_ls_states)
        //     {
        //         vertex_model.saveStates(data_folder + "/" + method + "_iter_" + std::to_string(step) + "_ls_" + std::to_string(ls_cnt) + ".obj");
        //     }
        //     std::cout << "[" << method << "]\t ls " << ls_cnt << " E1: " << E1 << " E0: " << E0 << std::endl;
            
        //     if (E1 < E0)
        //     {
        //         std::cout << "[" << method << "]\tfinal |dp|: " << (p_ls - design_parameters).norm() << " # ls " << ls_cnt << std::endl;
        //         design_parameters = p_ls;
        //         break;
        //     }
        //     // if (ls_cnt == ls_max - 1)
        //     //     std::getchar();
        //     alpha *= 0.5;
        // }
    }
    if (save_results)
    {
        vertex_model.saveStates(data_folder + "/" + method + "_iter_" + std::to_string(step) + ".obj");
        std::string filename = data_folder + "/" + method + "_iter_" + std::to_string(step) + ".txt";
        saveDesignParameters(filename, design_parameters);
    }
    
    return false;

}

void SensitivityAnalysis::saveDesignParameters(const std::string& filename, const VectorXT& params)
{
    std::ofstream out(filename);
    out << std::setprecision(20) << params << std::endl;
    out.close();
}