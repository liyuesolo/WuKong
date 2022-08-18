
#include <LBFGSB.h>
#include "../include/SensitivityAnalysis.h"

void SensitivityAnalysis::optimizeLBFGSB()
{
    objective.getDesignParameters(design_parameters);
    objective.solver.verbose = false;
    n_dof_design = objective.n_dof_design; n_dof_sim = objective.n_dof_sim;
    LBFGSB lbfgsb_solver;
    VectorXT lower_bounds = VectorXT::Constant(n_dof_design, -1.0);
    VectorXT upper_bounds = VectorXT::Constant(n_dof_design, 100.0);
    
    lbfgsb_solver.setBounds(lower_bounds, upper_bounds);
    VectorXT mask;
    objective.getDirichletMask(mask);
    int cnt = 0;
    VectorXT x_previous = design_parameters;
    auto computeObjAndGradient = [&](const VectorXT& x, VectorXT& grad)
    {
        
        VectorXT dp = x - x_previous;
        T alpha = objective.maximumStepSize(x_previous, dp);
        dp.array() *= mask.array();
        std::cout << "alpha " << alpha << std::endl;
        x_previous = x_previous + alpha * dp;
        
        objective.updateIPCVertices(x_previous);
        objective.updateCotMat(design_parameters);
        objective.updateDesignParameters(x_previous);
        
        std::string filename = "/home/yueli/Documents/ETH/SandwichStructure/opt/lbfgs_iter_" + std::to_string(cnt) + ".obj";
        solver.saveToOBJ(filename, true);

        T energy = 0.0;
        objective.gradient(x_previous, grad, energy, true, false);
        
        std::cout << "[L-BFGS-G] iter " << cnt << " |g|: " << grad.norm() << " obj: " << energy << std::endl;
        cnt++;
        return energy;
    };

    lbfgsb_solver.setObjective(computeObjAndGradient);
    lbfgsb_solver.setX(design_parameters);
    lbfgsb_solver.solve();
}

void SensitivityAnalysis::optimizeGradientDescent()
{
    objective.getDesignParameters(design_parameters);
    objective.solver.verbose = false;
    n_dof_design = objective.n_dof_design; n_dof_sim = objective.n_dof_sim;
    T tol_g = 1e-6;
    VectorXT mask;
    objective.getDirichletMask(mask);
    for (int iter = 0; iter < max_iter; iter++)
    {
        VectorXT dOdp(n_dof_design); dOdp.setZero();
        T E0;
        
        T g_norm = objective.gradient(design_parameters, dOdp, E0);   
        std::cout << "[GD] iter " << iter << " |g|: " << g_norm << " obj: " << E0 << std::endl;
        dOdp.array() *= mask.array();
        g_norm = dOdp.norm();
        int ls_max = 10;
        int ls_cnt = 0;
        T alpha = 1.0;
        alpha = objective.maximumStepSize(design_parameters, -dOdp);
        std::cout << "[GD] alpha " << alpha << std::endl;
        T E1 = 0.0;
        while (true)
        {
            ls_cnt++;
            VectorXT p_ls = design_parameters - alpha * dOdp;
            
            E1 = objective.value(p_ls, /*simulate=*/true, /*use_previous_equil=*/false);
            std::cout << "[GD]\t ls " << ls_cnt << " alpha " << alpha << " E1: " << E1 << " E0: " << E0 << std::endl;
            if (E1 < E0 || ls_cnt == ls_max)
            {
                if (ls_cnt == ls_max)
                {
                    design_parameters = p_ls;
                    break; 
                }
                else
                {
                    std::cout << "[GD]\tfinal |dp|: " << (p_ls - design_parameters).norm() << " # ls " << ls_cnt << std::endl;
                    design_parameters = p_ls;
                    break;    
                }
            }
            alpha *= 0.5;
        }
        objective.updateIPCVertices(design_parameters);
        objective.updateDesignParameters(design_parameters);
        std::string filename = "/home/yueli/Documents/ETH/SandwichStructure/opt/GD_iter_" + std::to_string(iter) + ".obj";
        solver.saveToOBJ(filename, true);
    }
    
}

void SensitivityAnalysis::optimizeGaussNewton()
{
    objective.getDesignParameters(design_parameters);
    objective.solver.verbose = false;
    n_dof_design = objective.n_dof_design; n_dof_sim = objective.n_dof_sim;
    T tol_g = 1e-6;
    
    for (int iter = 0; iter < max_iter; iter++)
    {
        VectorXT dOdp(n_dof_design); dOdp.setZero();
        T E0;
        T g_norm = objective.gradient(design_parameters, dOdp, E0);   
        std::cout << "[GD] iter " << iter << " |g|: " << g_norm << " obj: " << E0 << std::endl;
        g_norm = dOdp.norm();
        if (g_norm < tol_g)
            break;
        VectorXT rhs = -dOdp;
        
        MatrixXT H_GN;
        objective.hessianGN(design_parameters, H_GN, /*simulate = */false);
        T alpha_reg = 10e-6;
        int reg_cnt = 0;
        while (true)
        {
            Eigen::LLT<Eigen::MatrixXd> lltOfA(H_GN);
            if(lltOfA.info() == Eigen::NumericalIssue)
            {
                // std::cout << "add reg to H_GN" << std::endl;
                H_GN.diagonal().array() += alpha_reg;
                alpha_reg *= 10.0;
                reg_cnt ++;
            }
            else
                break;
        }
        std::cout << "[GN] #reg " << reg_cnt << std::endl;
        VectorXT dp = H_GN.llt().solve(rhs);

        int ls_max = 10;
        int ls_cnt = 0;
        T alpha = 1.0;
        alpha = objective.maximumStepSize(design_parameters, dp);
        std::cout << "[GN] alpha " << alpha << std::endl;
        T E1 = 0.0;
        while (true)
        {
            ls_cnt++;
            VectorXT p_ls = design_parameters + alpha * dp;
            
            E1 = objective.value(p_ls, /*simulate=*/true, /*use_previous_equil=*/false);
            std::cout << "[GN]\t ls " << ls_cnt << " alpha " << alpha << " E1: " << E1 << " E0: " << E0 << std::endl;
            if (E1 < E0 || ls_cnt == ls_max)
            {
                if (ls_cnt == ls_max)
                {
                    design_parameters = p_ls;
                    break; 
                }
                else
                {
                    std::cout << "[GN]\tfinal |dp|: " << (p_ls - design_parameters).norm() << " # ls " << ls_cnt << std::endl;
                    design_parameters = p_ls;
                    break;    
                }
            }
            alpha *= 0.5;
        }
        objective.updateIPCVertices(design_parameters);
        objective.updateDesignParameters(design_parameters);
        std::string filename = "/home/yueli/Documents/ETH/SandwichStructure/opt/GD_iter_" + std::to_string(iter) + ".obj";
        solver.saveToOBJ(filename, true);
    }
}

void SensitivityAnalysis::optimizeMMA()
{
    std::cout << "########### MMA ###########" << std::endl;
    objective.getDesignParameters(design_parameters);
    objective.solver.verbose = false;
    n_dof_design = objective.n_dof_design; n_dof_sim = objective.n_dof_sim;
    T tol_g = 1e-6;
    MMASolver mma(n_dof_design, 0);
    mma.SetAsymptotes(0.2, 0.65, 1.05);
    T mma_step_size = 0.02;
    int max_mma_iter = 1000;
    TV min_corner, max_corner;
    solver.computeBoundingBox(min_corner, max_corner);
    std::vector<int> dirichlet_indices;
    objective.getDirichletIndices(dirichlet_indices);
    VectorXT min_p = VectorXT::Ones(n_dof_design) * -10.0;
    VectorXT max_p = VectorXT::Ones(n_dof_design) * 100.0;
    VectorXT mask;
    objective.getDirichletMask(mask);
    T E0 = 0.0;
    for (int iter = 0; iter < max_mma_iter; iter++)
    {
        VectorXT dOdp(n_dof_design); dOdp.setZero();
        T O;
        std::cout << "compute gradient" << std::endl;
        T g_norm = objective.gradient(design_parameters, dOdp, O);
        if (iter == 0) E0 = O;
        std::cout << "[MMA] iter " << iter << " |g|: " << g_norm << " obj: " << O << " obj0: " << E0 << std::endl;
        
        if (g_norm < tol_g)
            break;
        min_p = (design_parameters.array() - mma_step_size).cwiseMax(0.0);
        max_p = (design_parameters.array() + mma_step_size).cwiseMin(10.0);
        objective.iterateDirichletDoF([&](int offset, T target)
        {
            min_p[offset] = target; max_p[offset] = target;
        });
        VectorXT current = design_parameters;
        mma.UpdateEigen(design_parameters, dOdp, VectorXT(), VectorXT(), min_p, max_p);
        VectorXT dp = design_parameters - current;
        T alpha = objective.maximumStepSize(current, dp);
        std::cout << "alpha " << alpha << std::endl;
        design_parameters = current + alpha * dp;
        objective.projectDesignParameters(design_parameters);
        objective.updateIPCVertices(design_parameters);
        objective.updateDesignParameters(design_parameters);
        objective.updateCotMat(design_parameters);
        std::string filename = "/home/yueli/Documents/ETH/SandwichStructure/opt/MMA_iter_" + std::to_string(iter) + ".obj";
        solver.saveToOBJ(filename, true);
    }
}