
#include <LBFGSB.h>
#include "../include/SensitivityAnalysis.h"



// void SensitivityAnalysis::optimizeGradientDescent()
// {
//     objective.getDesignParameters(design_parameters);
//     objective.solver.verbose = false;
//     n_dof_design = objective.n_dof_design; n_dof_sim = objective.n_dof_sim;
//     T tol_g = 1e-6;
//     VectorXT mask;
//     objective.getDirichletMask(mask);
//     for (int iter = 0; iter < max_iter; iter++)
//     {
//         VectorXT dOdp(n_dof_design); dOdp.setZero();
//         T E0;
        
//         T g_norm = objective.gradient(design_parameters, dOdp, E0);   
//         std::cout << "[GD] iter " << iter << " |g|: " << g_norm << " obj: " << E0 << std::endl;
//         dOdp.array() *= mask.array();
//         g_norm = dOdp.norm();
//         int ls_max = 10;
//         int ls_cnt = 0;
//         T alpha = 1.0;
//         alpha = objective.maximumStepSize(design_parameters, -dOdp);
//         std::cout << "[GD] alpha " << alpha << std::endl;
//         T E1 = 0.0;
//         while (true)
//         {
//             ls_cnt++;
//             VectorXT p_ls = design_parameters - alpha * dOdp;
            
//             E1 = objective.value(p_ls, /*simulate=*/true, /*use_previous_equil=*/false);
//             std::cout << "[GD]\t ls " << ls_cnt << " alpha " << alpha << " E1: " << E1 << " E0: " << E0 << std::endl;
//             if (E1 < E0 || ls_cnt == ls_max)
//             {
//                 if (ls_cnt == ls_max)
//                 {
//                     design_parameters = p_ls;
//                     break; 
//                 }
//                 else
//                 {
//                     std::cout << "[GD]\tfinal |dp|: " << (p_ls - design_parameters).norm() << " # ls " << ls_cnt << std::endl;
//                     design_parameters = p_ls;
//                     break;    
//                 }
//             }
//             alpha *= 0.5;
//         }
//         objective.updateIPCVertices(design_parameters);
//         objective.updateDesignParameters(design_parameters);
//         std::string filename = "/home/yueli/Documents/ETH/SandwichStructure/opt/GD_iter_" + std::to_string(iter) + ".obj";
//         solver.saveToOBJ(filename, true);
//     }
    
// }

// void SensitivityAnalysis::optimizeGaussNewton()
// {
//     objective.getDesignParameters(design_parameters);
//     objective.solver.verbose = false;
//     n_dof_design = objective.n_dof_design; n_dof_sim = objective.n_dof_sim;
//     T tol_g = 1e-6;
    
//     for (int iter = 0; iter < max_iter; iter++)
//     {
//         VectorXT dOdp(n_dof_design); dOdp.setZero();
//         T E0;
//         T g_norm = objective.gradient(design_parameters, dOdp, E0);   
//         std::cout << "[GN] iter " << iter << " |g|: " << g_norm << " obj: " << E0 << std::endl;
//         g_norm = dOdp.norm();
//         if (g_norm < tol_g)
//             break;
//         VectorXT rhs = -dOdp;
        
//         MatrixXT H_GN;
//         objective.hessianGN(design_parameters, H_GN, /*simulate = */false);
//         T alpha_reg = 10e-6;
//         int reg_cnt = 0;
//         while (true)
//         {
//             Eigen::LLT<Eigen::MatrixXd> lltOfA(H_GN);
//             if(lltOfA.info() == Eigen::NumericalIssue)
//             {
//                 // std::cout << "add reg to H_GN" << std::endl;
//                 H_GN.diagonal().array() += alpha_reg;
//                 alpha_reg *= 10.0;
//                 reg_cnt ++;
//             }
//             else
//                 break;
//         }
//         std::cout << "[GN] #reg " << reg_cnt << std::endl;
//         VectorXT dp = H_GN.llt().solve(rhs);

//         int ls_max = 10;
//         int ls_cnt = 0;
//         T alpha = 1.0;
//         alpha = objective.maximumStepSize(design_parameters, dp);
//         std::cout << "[GN] alpha " << alpha << std::endl;
//         T E1 = 0.0;
//         while (true)
//         {
//             ls_cnt++;
//             VectorXT p_ls = design_parameters + alpha * dp;
            
//             E1 = objective.value(p_ls, /*simulate=*/true, /*use_previous_equil=*/false);
//             std::cout << "[GN]\t ls " << ls_cnt << " alpha " << alpha << " E1: " << E1 << " E0: " << E0 << std::endl;
//             if (E1 < E0 || ls_cnt == ls_max)
//             {
//                 if (ls_cnt == ls_max)
//                 {
//                     design_parameters = p_ls;
//                     break; 
//                 }
//                 else
//                 {
//                     std::cout << "[GN]\tfinal |dp|: " << (p_ls - design_parameters).norm() << " # ls " << ls_cnt << std::endl;
//                     design_parameters = p_ls;
//                     break;    
//                 }
//             }
//             alpha *= 0.5;
//         }
//         objective.updateIPCVertices(design_parameters);
//         objective.updateDesignParameters(design_parameters);
//         std::string filename = "/home/yueli/Documents/ETH/SandwichStructure/opt/GN_iter_" + std::to_string(iter) + ".obj";
//         solver.saveToOBJ(filename, true);
//     }
// }

void SensitivityAnalysis::sampleGradientDirection()
{
    TV ti = TV(0.15, 0.55);
    objective.n_dof_design = ti.rows();
    design_parameters = ti;
    n_dof_design = objective.n_dof_design; 
    objective.targets.resize(3);
    objective.targets << -0.0062526045597, 0.0239273131735, 0.0717615511869;
    VectorXT dOdp(n_dof_design); dOdp.setZero();
    design_parameters = ti;
    T O = 0.0;
    // T g_norm = objective.gradient(design_parameters, dOdp, O);
    // dOdp << 0.130998, -0.0189072;
    dOdp << 1., 0.;
    // dOdp = TV(0.195, 0.795) - TV(0.105, 0.505);
    // std::cout << dOdp.transpose() << std::endl;
    // std::exit(0);
    VectorXT search_direction = dOdp;
    // std::cout << search_direction.transpose() << std::endl;
    
    T step_size = 1e-5;
    // T step_size = 1e-3;
    int step = 300; 

    std::vector<T> energies;
    std::vector<T> energies_gd;
    std::vector<T> steps;
    int step_cnt = 1;
    for (T xi = -T(step/2) * step_size; xi < T(step/2) * step_size; xi+=step_size)
    // T xi = T(step/2) * step_size - step_size;
    {
        // std::cout << design_parameters.transpose() << " xi " << xi << " dir " << search_direction.transpose() << std::endl;
        T Ei = objective.value(design_parameters + xi * search_direction);
        energies.push_back(Ei);
        steps.push_back(xi);
        
        step_cnt++;
    }
    
    for (T e : energies)
    {
        std::cout << std::setprecision(12) <<  e << ", ";
    }
    std::cout << std::endl;
    for (T e : energies_gd)
    {
        std::cout << e << ", ";
    }
    std::cout << std::endl;
    for (T idx : steps)
    {
        std::cout << idx << ", ";
    }
    std::cout << std::endl;
    std::cout << dOdp.transpose() << std::endl;
}

void SensitivityAnalysis::optimizeMMA()
{
    std::cout << "########### MMA ###########" << std::endl;
    // objective.targets.resize(1);
    // objective.targets[0] = 0.000394268;

    objective.targets.resize(3);
    objective.targets << -0.0062526,  0.0239273,  0.0717616;

    TV ti(0.12, 0.6);
    // TV ti(0.115, 0.75);
    objective.bounds.push_back(TV(0.1, 0.2));
    objective.bounds.push_back(TV(0.5, 0.8));
    objective.n_dof_design = ti.rows();
    design_parameters = ti;
    n_dof_design = objective.n_dof_design; 
    T tol_g = 1e-6;
    MMASolver mma(n_dof_design, 0);
    mma.SetAsymptotes(0.2, 0.65, 1.05);
    
    int max_mma_iter = 1000;
    
    VectorXT min_p(n_dof_design), max_p(n_dof_design);
    for (int i = 0; i < n_dof_design; i++)
    {
        min_p[i] = objective.bounds[i][0]; max_p[i] = objective.bounds[i][1];   
    }
    
    
    T E0 = 0.0;
    for (int iter = 0; iter < max_mma_iter; iter++)
    {
        VectorXT dOdp(n_dof_design); dOdp.setZero();
        T O = 0.0;
        // std::cout << "compute gradient" << std::endl;
        T g_norm = objective.gradient(design_parameters, dOdp, O);
        T epsilon = 1e-7;
        VectorXT feasible_point_gradients = dOdp;
        for (int i = 0; i < n_dof_design; i++)
        {
            if (design_parameters[i] < min_p[i] + epsilon && dOdp[i] >= 0)
                feasible_point_gradients[i] = 0.0;
            if (design_parameters[i] > max_p[i] - epsilon && dOdp[i] <= 0)
                feasible_point_gradients[i] = 0.0;
        }
        g_norm = feasible_point_gradients.norm();

        if (iter == 0) E0 = O;
        std::cout << "[MMA] iter " << iter << " |g|: " << g_norm << " obj: " << O << " obj0: " << E0 << std::endl;
        std::cout << "\t design parameters: " << design_parameters.transpose() << std::endl;
        if (g_norm < tol_g)
            break;
        
        VectorXT current = design_parameters;
        mma.UpdateEigen(design_parameters, dOdp, VectorXT(), VectorXT(), min_p, max_p);
        

    }
    std::cout << design_parameters.transpose() << std::endl;
}

void SensitivityAnalysis::optimizeLBFGSB()
{
    T epsilon = 1e-7;
    objective.targets.resize(3);
    objective.targets << -0.0062526, 0.0239273, 0.0717616;

    // TV ti(0.15, 0.65);
    TV ti(0.12, 0.60);
    objective.bounds.push_back(TV(0.1, 0.2));
    objective.bounds.push_back(TV(0.5, 0.8));
    objective.n_dof_design = ti.rows();
    design_parameters = ti;
    n_dof_design = objective.n_dof_design; 
    VectorXT min_p(n_dof_design), max_p(n_dof_design);
    for (int i = 0; i < n_dof_design; i++)
    {
        min_p[i] = objective.bounds[i][0]; max_p[i] = objective.bounds[i][1];   
    }


    LBFGSB lbfgsb_solver;
    
    lbfgsb_solver.setBounds(min_p, max_p);

    int cnt = 0;
//     VectorXT x_previous = design_parameters;
    auto computeObjAndGradient = [&](const VectorXT& x, VectorXT& grad)
    {
        T energy = 0.0;
        T g_norm = objective.gradient(x, grad, energy);
        VectorXT feasible_point_gradients = grad;
        for (int i = 0; i < n_dof_design; i++)
        {
            if (design_parameters[i] < min_p[i] + epsilon && grad[i] >= 0)
                feasible_point_gradients[i] = 0.0;
            if (design_parameters[i] > max_p[i] - epsilon && grad[i] <= 0)
                feasible_point_gradients[i] = 0.0;
        }
        g_norm = feasible_point_gradients.norm();

        std::cout << "[L-BFGS-B] iter " << cnt << " proj |g|: " << g_norm << " obj: " << energy << std::endl;
        cnt++;
        return energy;
    };

    lbfgsb_solver.setObjective(computeObjAndGradient);
    lbfgsb_solver.setX(design_parameters);
    lbfgsb_solver.solve();
    std::cout << design_parameters.transpose() << std::endl;
}