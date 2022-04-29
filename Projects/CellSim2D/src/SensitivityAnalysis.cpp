#include <igl/mosek/mosek_quadprog.h>
#include <Eigen/PardisoSupport>
#include "../include/LinearSolver.h"
#include "../include/SensitivityAnalysis.h"

void SensitivityAnalysis::initialize()
{
    vertex_model.verbose = false;
    objective.getSimulationAndDesignDoF(n_dof_sim, n_dof_design);
    objective.bound[0] = 1e-5;
    objective.bound[1] = 10.0;
    objective.mask[0] = true;
    objective.mask[1] = false;
    objective.equilibrium_prev = VectorXT::Zero(vertex_model.deformed.rows());
    
    std::cout << "n_dof_sim " <<  n_dof_sim << " n_dof_design: " << n_dof_design << std::endl;
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
    T lower_bound = objective.bound[0], upper_bound = objective.bound[1];
    
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
            << " |g_init| " << initial_gradient_norm << " tol " << 1e-3 * initial_gradient_norm << " obj: " << E0 << std::endl;
        
        if ( g_norm < 1e-3 * initial_gradient_norm )
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
                // if (dot_search_grad < 0.01)
                if (dot_search_grad < 1e-3)
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

        
        if (search_direction.norm() < 1e-5)
            return true;

        int ls_max = 15;
        for (int ls_cnt = 0; ls_cnt < ls_max; ls_cnt++)
        {
            VectorXT p_ls = design_parameters + alpha * search_direction;
            // saveDesignParameters(data_folder + "/trouble.txt", p_ls);
            
            T E1 = objective.value(p_ls, /*simulate=*/true, /*use_previous_equil=*/true);
            
            std::cout << "[" << method << "]\t ls " << ls_cnt << " E1: " << E1 << " E0: " << E0 << std::endl;
            
            if (E1 < E0)
            {
                std::cout << "[" << method << "]\tfinal |dp|: " << (p_ls - design_parameters).norm() << " # ls " << ls_cnt << std::endl;
                design_parameters = p_ls;
                break;
            }
            alpha *= 0.5;
        }
    }
    if (save_results)
    {
        // objective.saveState(data_folder + "/" + method + "_iter_" + std::to_string(step) + ".obj");
        // objective.updateDesignParameters(design_parameters);
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