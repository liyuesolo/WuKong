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
#include <filesystem>
#include "../include/IpoptSolver.h"

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
    objective.bound[0] = 0.0;
    // objective.bound[1] = 20.0 * simulation.cells.unit;
    objective.bound[1] = 50;
    objective.mask[0] = true;
    objective.mask[1] = true;
    objective.equilibrium_prev = VectorXT::Zero(simulation.deformed.rows());

    // T mid = 0.5 * (objective.bound[0] + objective.bound[1]);
    // simulation.cells.edge_weights.setConstant(mid);
    // simulation.cells.edge_weights.setConstant(0.1);
    objective.getDesignParameters(design_parameters);
    objective.prev_params = design_parameters;
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
    for (int i = 0; i < simulation.cells.n_cells; i++)
    {
        TV current;
        simulation.cells.computeCellCentroid(simulation.cells.faces[i], current);
        out << i << " " << std::setprecision(20) << current[0] << " " << current[1] << " " << current[2] << std::endl;
    }
    
    // objective.iterateTargets([&](int idx, TV& target){
    //     TV current;
    //     simulation.cells.computeCellCentroid(simulation.cells.faces[idx], current);
    //     out << idx << " " << current[0] << " " << current[1] << " " << current[2] << std::endl;
    // });
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

void SensitivityAnalysis::optimizeSGN()
{
    T g_norm = 1e10;
    T tol_g = 1e-5;
    T E0;
    VectorXT dOdp, dp;
    T lower_bound = objective.bound[0], upper_bound = objective.bound[1];

    objective.getDesignParameters(design_parameters);
    
    for (int step = 0; step < max_num_iter; step++)
    {
        if (step == 0)
        {
            g_norm = objective.gradient(design_parameters, dOdp, E0, /*simulate = */true); 
        }
        else
        {
            g_norm = objective.gradient(design_parameters, dOdp, E0, /*simulate = */false);
        }
        
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

        if (step == 0)
        {
            initial_gradient_norm = g_norm;  
            tol_g = 1e-2 * initial_gradient_norm;
        }

        std::cout << "[" << "SGN" << "] iter " << step << " |g| " << g_norm 
            << " |g_init| " << initial_gradient_norm << " tol " << tol_g << " obj: " << E0 << std::endl;
        std::cout << "forward simulation hessian eigen values: ";
        // simulation.checkHessianPD(false);

        if (g_norm < tol_g )
            break;

        dp = dOdp;
        
        StiffnessMatrix mat_SGN;
        objective.hessianSGN(design_parameters, mat_SGN, /*simulate = */false);

        VectorXT rhs_SGN(n_dof_design + n_dof_sim * 2);
        rhs_SGN.setZero();
        rhs_SGN.segment(n_dof_sim, n_dof_design) = -dOdp;
        VectorXT delta;
        
        
        PardisoLDLTSolver solver(mat_SGN, /*use_default=*/false);
        solver.setPositiveNegativeEigenValueNumber(n_dof_sim + n_dof_design, n_dof_sim);
        solver.setRegularizationIndices(n_dof_sim, n_dof_design);

        
        solver.solve(rhs_SGN, delta);
        

        dp = -delta.segment(n_dof_sim, n_dof_design);
        

        if (dp.norm() < 1e-8)
            break;

        int ls_max = 15;
        bool use_gradient = false;
        T alpha = 1.0;
        for (int ls_cnt = 0; ls_cnt < ls_max; ls_cnt++)
        {
            VectorXT p_ls = design_parameters + alpha * dp;
            
            T E1 = objective.value(p_ls, /*simulate=*/true, /*use_previous_equil=*/true);
            std::cout << "[" << "SGN" << "]\t ls " << ls_cnt << " E1: " << E1 << " E0: " << E0 << std::endl;
            // std::getchar();
            if (E1 < E0)
            {
                std::cout << "[" << "SGN" << "]\tfinal |dp|: " << (p_ls - design_parameters).norm() << " # ls " << ls_cnt << std::endl;
                design_parameters = p_ls;
                break;
            }
            else
            {
                alpha *= 0.5;
            }
        }
    }
    objective.updateDesignParameters(design_parameters);
}

void SensitivityAnalysis::optimizeSQP()
{
    T g_norm = 1e10;
    T tol_g = 1e-5;
    T E0;
    VectorXT dOdp, dp;
    T lower_bound = objective.bound[0], upper_bound = objective.bound[1];

    objective.getDesignParameters(design_parameters);
    
    for (int step = 0; step < max_num_iter; step++)
    {
        if (step == 0)
        {
            g_norm = objective.gradient(design_parameters, dOdp, E0, /*simulate = */true); 
        }
        else
        {
            g_norm = objective.gradient(design_parameters, dOdp, E0, /*simulate = */false);
        }
        
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

        if (step == 0)
        {
            initial_gradient_norm = g_norm;  
            tol_g = 1e-2 * initial_gradient_norm;
        }

        std::cout << "[" << "SQP" << "] iter " << step << " |g| " << g_norm 
            << " |g_init| " << initial_gradient_norm << " tol " << tol_g << " obj: " << E0 << std::endl;
        std::cout << "forward simulation hessian eigen values: ";
        simulation.checkHessianPD(false);

        if (g_norm < tol_g )
            break;

        dp = dOdp;
        
        MatrixXT H_GN;
        objective.hessianGN(design_parameters, H_GN, /*simulate = */false);

        if (add_reg)
            H_GN.diagonal().array() += reg_w_H;

        StiffnessMatrix Q = H_GN.sparseView();    
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
        bool solve_success = igl::mosek::mosek_quadprog(Q, dOdp, E0, A, lc, uc, lx, ux, mosek_data, dp, lagrange_multipliers);

        if (dp.norm() < 1e-8)
            break;

        int ls_max = 15;
        bool use_gradient = false;
        T alpha = 1.0;
        for (int ls_cnt = 0; ls_cnt < ls_max; ls_cnt++)
        {
            VectorXT p_ls = design_parameters + alpha * dp;
            
            T E1 = objective.value(p_ls, /*simulate=*/true, /*use_previous_equil=*/true);
            std::cout << "[" << "SQP" << "]\t ls " << ls_cnt << " E1: " << E1 << " E0: " << E0 << std::endl;
            // std::getchar();
            if (E1 < E0)
            {
                std::cout << "[" << "SQP" << "]\tfinal |dp|: " << (p_ls - design_parameters).norm() << " # ls " << ls_cnt << std::endl;
                design_parameters = p_ls;
                break;
            }
            else
            {
                alpha *= 0.5;
            }
            if (ls_cnt == ls_max - 1 && !use_gradient)
            {
                std::cout << "\t[" << "SQP" << "] taking gradient step" << std::endl;
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
                bool solve_success = igl::mosek::mosek_quadprog(Q, dOdp, E0, A, lc, uc, lx, ux, mosek_data, dp, lagrange_multipliers);

                ls_cnt = 0;
                alpha = 1.0;
                use_gradient = true;
            }
        }
    }
    objective.updateDesignParameters(design_parameters);
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
    
    T tol_g = 1e-6;
    T rel_tol_g = 1e-4;
    T g_norm = 0;
    T E0;
    VectorXT dOdp, dp;
    bool abs_tol = false;

    // mma 
    T mma_step_size = 1.0 * simulation.cells.unit;
    
    T lower_bound = objective.bound[0], upper_bound = objective.bound[1];

    VectorXT min_p = VectorXT::Ones(n_dof_design) * lower_bound;
    VectorXT max_p = VectorXT::Ones(n_dof_design) * upper_bound;
    
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
            T epsilon = 1e-12;
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
        T tol_opt = abs_tol ? tol_g : rel_tol_g * initial_gradient_norm;
        std::cout << "[" << method << "] iter " << step << " |g| " << g_norm 
            << " |g_init| " << initial_gradient_norm << " tol " << tol_opt << " obj: " << E0 << std::endl;
        // std::cout << "forward simulation hessian eigen values: "; simulation.checkHessianPD(false);
        // simulation.checkInfoForSA();
        if (g_norm < tol_opt || step > max_num_iter)
            return true;

        dp = dOdp;

        Timer gn_timer(false);
        if (optimizer == GaussNewton)
        {
            gn_timer.start();
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

            Eigen::JacobiSVD<Eigen::MatrixXd> svd(H_GN, Eigen::ComputeThinU | Eigen::ComputeThinV);
            VectorXT Sigma = svd.singularValues();
            
            std::cout << "\tGN Hessian singular values last: " << Sigma.tail<5>().transpose() << std::endl;
            std::cout << "\tGN Hessian singular values first: " << Sigma.head<5>().transpose() << std::endl;            

            VectorXT rhs = -dOdp;
            dp = H_GN.llt().solve(rhs);
            // MatrixXT V_block = V.block(0, n_dof_design - 6, n_dof_design, 5);
            // std::cout << (V_block.transpose() * (dp.normalized())).norm() << std::endl;

            // std::cout << (dOdp).norm() << " " << (dOdp).maxCoeff() << " " << (dOdp).minCoeff() << std::endl;
            // std::cout << (H_GN*dp).norm() << " " << (H_GN*dp).maxCoeff() << " " << (H_GN*dp).minCoeff() << std::endl;

            // std::cout << "\t[EigenLLT] |Ax-b|/|b|: " << (H_GN * dp - rhs).norm() / rhs.norm() << std::endl;
            dp *= -1.0;

            int s_cnt = 0, l_cnt = 0;
            for (int i = 0; i < n_dof_design; i++)
            {
                if (design_parameters[i] + dp[i] < lower_bound)
                    s_cnt++;
                if (design_parameters[i] + dp[i] > upper_bound)
                    l_cnt++;
            }
            std::cout << s_cnt << "/" << n_dof_design << "are below lower bound" << std::endl;
            std::cout << l_cnt << "/" << n_dof_design << "are above upper bound" << std::endl;
            std::cout << "min " << (design_parameters + dp).minCoeff() << " max " << (design_parameters + dp).maxCoeff() << std::endl;
            
        }
        else if (optimizer == SQP)
        {
            gn_timer.start();
            MatrixXT H_GN;
            objective.hessianGN(design_parameters, H_GN, /*simulate = */false);

            // Eigen::JacobiSVD<Eigen::MatrixXd> svd(H_GN, Eigen::ComputeThinU | Eigen::ComputeThinV);
            // MatrixXT U = svd.matrixU();
            // VectorXT Sigma = svd.singularValues();
            // MatrixXT V = svd.matrixV();
            // std::cout << "\t[SQP] GN Hessian singular values last: " << Sigma.tail<5>().transpose() << std::endl;
            // std::cout << "\t[SQP] GN Hessian singular values first: " << Sigma.head<5>().transpose() << std::endl;

            if (add_reg)
            {
                std::cout << "add reg " << reg_w_H << std::endl;
                H_GN.diagonal().array() += reg_w_H;
            }
            
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
            dp *= -1.0;
            
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
            gn_timer.start();
            objective.hessianSGN(design_parameters, mat_SGN, /*simulate = */false);
            std::cout << "\tSGN matrix assembly takes " << gn_timer.elapsed_sec() << "s" << std::endl;
            if (add_reg)
            {
                // mat_SGN.diagonal().segment(0, n_dof_sim).array() += reg_w_H;
                mat_SGN.diagonal().segment(n_dof_sim, n_dof_design).array() += reg_w_H;
            }

            VectorXT rhs_SGN(n_dof_design + n_dof_sim * 2);
            rhs_SGN.setZero();
            rhs_SGN.segment(n_dof_sim, n_dof_design) = -dOdp;
            VectorXT delta;
            T reg_alpha = 1e-6;
            while (true)
            {
                PardisoLDLTSolver solver(mat_SGN, /*use_default=*/false);
                solver.setPositiveNegativeEigenValueNumber(n_dof_sim + n_dof_design, n_dof_sim);
                solver.setRegularizationIndices(n_dof_sim, n_dof_design);

                // EigenLUSolver solver(mat_SGN);
                // PardisoLUSolver solver(mat_SGN, false);
                
                solver.solve(rhs_SGN, delta);
                dp = delta.segment(n_dof_sim, n_dof_design);
                T dot_search_grad = -dp.normalized().dot(dOdp.normalized());
                if (dot_search_grad < 0.05)
                {
                    mat_SGN.diagonal().segment(n_dof_sim, n_dof_design).array() += reg_alpha;
                    reg_alpha *= 10.0;
                }
                else
                {
                    std::cout << "dot search grad " << dot_search_grad << std::endl;
                    dp *= -1.0;
                    break;
                }
            }
        }
        else if (optimizer == SSQP)
        {
            StiffnessMatrix Q, A;
            objective.SGNforQP(design_parameters, Q, A);
            // std::ofstream out("Q.txt"); out << Q; out.close();
            Eigen::PardisoLLT<StiffnessMatrix> solver;
            std::cout << "analyzePattern here" << std::endl;
            solver.analyzePattern(Q);
            T Q_reg_alpha = 1e-6;
            while (true)
            {
                solver.factorize(Q);
                std::cout << "factorize here" << std::endl;
                if (solver.info() == Eigen::NumericalIssue)
                {
                    Q.diagonal().array() += Q_reg_alpha;
                    std::cout << "add reg here" << std::endl;
                    Q_reg_alpha *= 10.0;
                    continue;
                }
                break;
            }
            
            VectorXT c(n_dof_design + n_dof_sim);
            c.setZero();
            c.segment(n_dof_sim, n_dof_design) = -dOdp;
            
            VectorXT lc(n_dof_sim), uc(n_dof_sim);
            lc.setZero(); uc.setZero();

            VectorXT lx(n_dof_design + n_dof_sim);
            lx.setConstant(-1e8);
            lx.segment(n_dof_sim, n_dof_design).setConstant(lower_bound);
            lx.segment(n_dof_sim, n_dof_design) -= design_parameters;
            
            VectorXT ux(n_dof_design + n_dof_sim); 
            ux.setConstant(1e8);
            ux.segment(n_dof_sim, n_dof_design).setConstant(upper_bound);
            ux.segment(n_dof_sim, n_dof_design) -= design_parameters;
            
            VectorXT delta_x_p(n_dof_design + n_dof_sim); delta_x_p.setZero();
            igl::mosek::MosekData mosek_data;
            std::vector<VectorXT> lagrange_multipliers;
            bool solve_success = igl::mosek::mosek_quadprog(Q, c, 0, A, 
                lc, uc, lx, ux, mosek_data, delta_x_p, lagrange_multipliers);
            std::cout << "mosek success: " << solve_success << std::endl;
            dp = -delta_x_p.segment(n_dof_sim, n_dof_design);
            std::cout << "|dxdp| " << delta_x_p.norm() << std::endl;
            VectorXT dLdp = Q * dp + c;

            dLdp -= lagrange_multipliers[0].transpose() * A;
            dLdp += lagrange_multipliers[1].transpose() * A;
            dLdp -= lagrange_multipliers[2];
            dLdp += lagrange_multipliers[3];
            std::cout << "\t[SQP] |dL|: " << dLdp.norm() << std::endl;

            auto saveQPData = [&]()
            {
                std::ofstream out("Q.txt"); out << Q; out.close();
                out.open("c.txt"); out << c; out.close();
                out.open("A.txt"); out << A; out.close();
                out.open("b.txt"); out << lc; out.close();
                out.open("lx.txt"); out << lx; out.close();
                out.open("ux.txt"); out << ux; out.close();
            };

            // saveQPData();
            // std::exit(0);
        }
        gn_timer.stop();
        std::cout << "\t" << method <<" takes " << gn_timer.elapsed_sec() << "s" << std::endl;
        // std::cout << "here" << std::endl;
        VectorXT search_direction = -dp;
        T alpha = objective.maximumStepSize(search_direction);
        
        std::cout << "[" << method << "]\t|dp|: " << search_direction.norm() << std::endl;
        
        
        if (search_direction.norm() < 1e-5)
            return true;

        int ls_max = 15;
        bool use_gradient = false;
        T c1 = 10e-4, c2 = 0.9;
        bool wolfe_condition = false;
        int ls_cnt = 0;

        while (true)
        {
            ls_cnt++;
            VectorXT p_ls = design_parameters + alpha * search_direction;
            
            T E1 = objective.value(p_ls, /*simulate=*/true, /*use_previous_equil=*/true);
            
            std::cout << "[" << method << "]\t ls " << ls_cnt << " alpha " << alpha << " E1: " << E1 << " E0: " << E0 << std::endl;

            if (E1 < E0 || ls_cnt == ls_max)
            {
                if (ls_cnt == ls_max && !use_gradient)
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
                else
                {
                    std::cout << "[" << method << "]\tfinal |dp|: " << (p_ls - design_parameters).norm() << " # ls " << ls_cnt << std::endl;
                    design_parameters = p_ls;
                    break;    
                }
                
            }
            alpha *= 0.5;
        }
        
        
        // for (int ls_cnt = 0; ls_cnt < ls_max; ls_cnt++)
        // {
        //     VectorXT p_ls = design_parameters + alpha * search_direction;
            
        //     T E1 = objective.value(p_ls, /*simulate=*/true, /*use_previous_equil=*/true);
            
        //     std::cout << "[" << method << "]\t ls " << ls_cnt << " E1: " << E1 << " E0: " << E0 << std::endl;
        //     // {
        //     //     T total_energy_sim = simulation.cells.computeTotalEnergy(simulation.u);
        //     //     std::cout << "[" << method << "]\t\t total energy sim: " << total_energy_sim << std::endl;
        //     // }
        //     // if (wolfe_condition)
        //     // {
        //     //     bool Armijo = E1 <= E0 + c1 * alpha * search_direction.dot(dOdp);
        //     //     VectorXT g1 = VectorXT::Zero(n_dof_design);
        //     //     T E_tmp;
        //     //     objective.gradient(p_ls, g1, E_tmp, true, true);
        //     //     bool curvature = -search_direction.dot(g1) <= -c2 * search_direction.dot(dOdp);
        //     //     if (Armijo && curvature)
        //     //     {
        //     //         std::cout << "[" << method << "]\tfinal |dp|: " << (p_ls - design_parameters).norm() << " # ls " << ls_cnt << std::endl;
        //     //         design_parameters = p_ls;
        //     //         break;
        //     //     }
        //     // }
        //     // else
        //     {
        //         if (E1 < E0)
        //         {
        //             std::cout << "[" << method << "]\tfinal |dp|: " << (p_ls - design_parameters).norm() << " # ls " << ls_cnt << std::endl;
        //             design_parameters = p_ls;
        //             break;
        //         }
        //     }
        //     alpha *= 0.5;
        //     continue;
        //     if (ls_cnt == ls_max - 1 && !use_gradient)
        //     {
        //         std::cout << "\t[" << method << "] taking gradient step" << std::endl;
        //         StiffnessMatrix Q(n_dof_design, n_dof_design);
        //         Q.setIdentity();
            
        //         StiffnessMatrix A;
        //         VectorXT lc, uc;
        //         VectorXT lx(n_dof_design); 
        //         lx.setConstant(lower_bound);
        //         lx -= design_parameters;
                
        //         VectorXT ux(n_dof_design); 
        //         ux.setConstant(upper_bound);
        //         ux -= design_parameters;
                
        //         igl::mosek::MosekData mosek_data;
        //         std::vector<VectorXT> lagrange_multipliers;
        //         bool solve_success = igl::mosek::mosek_quadprog(Q, dOdp, E0, A, lc, uc, lx, ux, mosek_data, search_direction, lagrange_multipliers);
                
        //         VectorXT dLdp = dOdp + search_direction;

        //         dLdp -= lagrange_multipliers[2];
        //         dLdp += lagrange_multipliers[3];
        //         std::cout << "\t[SQP] |dL/dp|: " << dLdp.norm() << std::endl;
        //         ls_cnt = 0;
        //         alpha = 1.0;
        //         use_gradient = true;
        //     }
        // }
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
        std::cout << "forward simulation hessian eigen values: ";
        simulation.checkHessianPD(false);
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
        VectorXT final_parameters = design_parameters;
        
        saveDesignParameters(filename, final_parameters);
    }
    
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
    LBFGSB lbfgsb_solver;
    VectorXT lower_bounds = VectorXT::Zero(n_dof_design);
    VectorXT upper_bounds = VectorXT::Constant(n_dof_design, objective.bound[1]);
    std::cout << objective.bound[1] << std::endl;

    lbfgsb_solver.setBounds(lower_bounds, upper_bounds);

    int cnt = 0;
    auto computeObjAndGradient = [&](const VectorXT& x, VectorXT& grad)
    {
        T energy = 0.0;
        objective.updateDesignParameters(x);
        objective.gradient(x, grad, energy, true, true);
        objective.equilibrium_prev = objective.simulation.u;
        objective.saveState(data_folder + "/" + "lbfgs_iter_" + std::to_string(cnt) + ".obj");
        std::string filename = data_folder + "/" + "lbfgs_iter_" + std::to_string(cnt) + ".txt";
        saveDesignParameters(filename, x);
        // std::cout << "[L-BFGS-G] iter " << cnt << " |g|: " << grad.norm() << " obj: " << energy << std::endl;
        cnt++;
        return energy;
    };

    lbfgsb_solver.setObjective(computeObjAndGradient);
    lbfgsb_solver.setX(design_parameters);
    lbfgsb_solver.solve();
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

void SensitivityAnalysis::checkStatesAlongGradientSGN()
{
    simulation.newton_tol = 1e-6;
    objective.perturb = false;
    save_results = true;
    T E0;
    VectorXT dOdp, dp;
    simulation.loadDeformedState("/home/yueli/Documents/ETH/WuKong/output/cells/562/SGN_iter_449.obj");
    simulation.loadEdgeWeights("/home/yueli/Documents/ETH/WuKong/output/cells/562/SGN_iter_449.txt", design_parameters);
    
    // simulation.checkHessianPD(false);
    T g_norm = objective.gradient(design_parameters, dOdp, E0, /*simulate = */false);


    MatrixXT dxdp;
    simulation.cells.dxdpFromdxdpEdgeWeights(dxdp);
    Eigen::JacobiSVD<Eigen::MatrixXd> svd_dxdp(dxdp, Eigen::ComputeThinU | Eigen::ComputeThinV);
	
	VectorXT Sigma0 = svd_dxdp.singularValues();

    std::cout << "dxdp largest 5 singular values: " << Sigma0.head<5>().transpose() << std::endl;
    std::cout << "dxdp smallest 5 singular values: " << Sigma0.tail<5>().transpose() << std::endl;

    objective.getDesignParameters(design_parameters);
    VectorXT u = simulation.u;
    T lower_bound = objective.bound[0], upper_bound = objective.bound[1];
    if (save_results)
        simulation.saveState(data_folder + "/start.obj");

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

    dp = delta.segment(n_dof_sim, n_dof_design);
    
    VectorXT walking_direction = dp.normalized();
    // VectorXT walking_direction = -dOdp.normalized();
    T step_size = 1e-2;
    int step = 25; 

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
        simulation.checkHessianPD(false);
        energies.push_back(Ei);
        std::vector<T> energy_all_terms;
        objective.computeEnergyAllTerms(design_parameters + xi * walking_direction, energy_all_terms, true, true);
        energies_all_terms.push_back(energy_all_terms);
        steps.push_back(xi);
        if (save_results)
        {
            simulation.saveState(data_folder + "/" + std::to_string(energies.size() - 1) + ".obj");
            std::string filename = data_folder + "/"  + std::to_string(energies.size() - 1) + ".txt";
            VectorXT param = design_parameters + xi * walking_direction;
            saveDesignParameters(filename, param);
        }
        std::cout << "[debug]: " << xi  << " obj: " << Ei << " step " << energies.size() - 1 << std::endl;
        // std::getchar();
        // if (xi > 0.0004)
        //     break;
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

void SensitivityAnalysis::checkStatesAlongGradient()
{
    
    simulation.newton_tol = 1e-6;
    objective.perturb = false;
    save_results = true;
    T E0;
    VectorXT dOdp, dp;
    // VectorXT ew;
    // simulation.loadDeformedState("/home/yueli/Documents/ETH/WuKong/output/cells/109/SQP_iter_10.obj");
    // simulation.loadEdgeWeights("/home/yueli/Documents/ETH/WuKong/output/cells/109/SQP_iter_10.txt", ew);
    simulation.loadDeformedState("/home/yueli/Documents/ETH/WuKong/output/cells/805/SQP_iter_129.obj");
    simulation.loadEdgeWeights("/home/yueli/Documents/ETH/WuKong/output/cells/805/SQP_iter_129.txt", simulation.cells.edge_weights);
    // simulation.cells.edge_weights = ew;
    
    VectorXT current_state = simulation.deformed;

    MatrixXT dxdp;
    simulation.cells.dxdpFromdxdpEdgeWeights(dxdp);
    Eigen::JacobiSVD<Eigen::MatrixXd> svd_dxdp(dxdp, Eigen::ComputeThinU | Eigen::ComputeThinV);
	MatrixXT U0 = svd_dxdp.matrixU();
	VectorXT Sigma0 = svd_dxdp.singularValues();
	MatrixXT V0 = svd_dxdp.matrixV();
    
    std::cout << "dxdp largest 5 singular values: " << Sigma0.head<5>().transpose() << std::endl;
    std::cout << "dxdp smallest 5 singular values: " << Sigma0.tail<5>().transpose() << std::endl;

    objective.getDesignParameters(design_parameters);
    VectorXT u = simulation.u;
    T lower_bound = objective.bound[0], upper_bound = objective.bound[1];
    if (save_results)
        simulation.saveState(data_folder + "/start.obj");
    MatrixXT H_GN;
    objective.hessianGN(design_parameters, H_GN, /*simulate = */false);
    T g_norm = objective.gradient(design_parameters, dOdp, E0, /*simulate = */false);
    
    // StiffnessMatrix Hessian(n_dof_sim, n_dof_sim);
    // simulation.cells.buildSystemMatrix(simulation.u, Hessian);
    // MatrixXT Hessian_dense = Hessian;
    // Eigen::JacobiSVD<Eigen::MatrixXd> svd_H(Hessian_dense, Eigen::ComputeThinU | Eigen::ComputeThinV);
    // VectorXT singular_value_hessian = svd_H.singularValues();
    // std::cout << "hessian largest 5 singular values: " << singular_value_hessian.head<5>().transpose() << std::endl;
    // std::cout << "hessian smallest 5 singular values: " << singular_value_hessian.tail<5>().transpose() << std::endl;

    std::cout << "forward simulation hessian ev " << std::endl; simulation.checkHessianPD(false);
    
    std::vector<int> proj_entries;
    bool iecs = true;
    T eps = 1e-5;
    for (int i = 0; i < n_dof_design; i++)
    {
        if (design_parameters[i] < lower_bound || design_parameters[i] > upper_bound)
            iecs = false;
        
        if (design_parameters[i] < lower_bound + eps && dOdp[i] > 1e-8)
            proj_entries.push_back(i);
        if (design_parameters[i] > upper_bound - eps && dOdp[i] < 1e-8)
            proj_entries.push_back(i);
    }
    VectorXT projected_gradient = dOdp;
    for (int idx : proj_entries)
    {
        projected_gradient[idx] = 0;
    }
    g_norm = projected_gradient.norm();
    std::cout << proj_entries.size() << "/" << n_dof_design << " are at their bounds " << std::endl;

    std::cout << "|g|: " << g_norm << " E0: " << E0 << std::endl;
    // std::getchar();
    
    // std::cout << iecs << " " << proj_entries.size() << " " << projected_gradient.norm() << std::endl;


    Eigen::JacobiSVD<Eigen::MatrixXd> svd(H_GN, Eigen::ComputeThinU | Eigen::ComputeThinV);
	
	VectorXT Sigma = svd.singularValues();
	std::cout << "H_GN smallest 5 singular values: " << Sigma.tail<5>().transpose() << std::endl;
    std::cout << "H_GN largest 5 singular values: " << Sigma.head<5>().transpose() << std::endl;
    // std::exit(0);

    MatrixXT V = svd.matrixV();
    
    if (add_reg)
        H_GN.diagonal().array() += reg_w_H;

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

    // T step_size = 5e-5;
    // int step = 200; 
    VectorXT walking_direction = dp.normalized();
    // VectorXT walking_direction = -dOdp.normalized();
    T step_size = 1e-2;
    // int step = 200; 
    int step = 50; 

    // T step_size = 1e-5;
    // int step = 100000;
    std::vector<T> energies;
    std::vector<std::vector<T>> energies_all_terms;
    std::vector<T> steps;
    std::vector<T> distance_to_current;
    std::vector<T> e_dx;

    for (T xi = -T(step/2) * step_size; xi < T(step/2) * step_size; xi+=step_size)
    {
        T Ei = objective.value(design_parameters + xi * walking_direction, true, true);
        
        T xTx = (simulation.deformed - current_state).sum();

        for (int i = 0; i < n_dof_sim; i++)
            distance_to_current.push_back(simulation.deformed[i]);
        // for (int i = 0; i < n_dof_sim; i++)
        //     distance_to_current.push_back(simulation.deformed[i]);
        
        // distance_to_current.push_back(xTx);

        VectorXT dx = dxdp * xi * walking_direction;
        VectorXT delta_p = xi * walking_direction;
        simulation.deformed = current_state + dx;

        simulation.checkHessianPD(false);
        energies.push_back(Ei);
        std::vector<T> energy_all_terms;
        objective.computeEnergyAllTerms(design_parameters + xi * walking_direction, energy_all_terms, false, true);
        energies_all_terms.push_back(energy_all_terms);
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
            // simulation.deformed = current_state + 100.0 * (simulation.deformed - current_state);
            simulation.saveState(data_folder + "/" + std::to_string(energies.size() - 1) + ".obj", false);
            std::string filename = data_folder + "/"  + std::to_string(energies.size() - 1) + ".txt";
            VectorXT param = design_parameters + xi * walking_direction;
            saveDesignParameters(filename, param);
            saveDesignParameters(data_folder + "/" + std::to_string(energies.size() - 1) + "_dp.txt", delta_p);
        }
        std::cout << "[debug]: " << xi  << " obj: " << Ei << " step " << energies.size() - 1 << " min length " << min_edge_length << std::endl;
        // std::getchar();
        // if (xi > 0.0004)
        //     break;
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
    // std::cout << "sum dx" << std::endl;
    // int n_step = steps.size();
    // for (int i = 0; i < n_dof_sim; i++)
    // {
    //     for (int step = 0; step < n_step; step++)
    //     {
    //         std::cout << distance_to_current[step * n_dof_sim + i] << " ";
    //     }    
    //     std::cout << std::endl;
    // }
    
    std::cout << "|dp|" << std::endl;
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
}

int SensitivityAnalysis::optimizeIPOPT()
{
    objective.use_penalty = false;
    Ipopt::SmartPtr<Ipopt::IpoptApplication> app = IpoptApplicationFactory();
    
    app->RethrowNonIpoptException(true);
    

    app->Options()->SetNumericValue("tol", 1e-4);
    app->Options()->SetNumericValue("acceptable_tol", 1e-4);
    app->Options()->SetStringValue("mu_strategy", "monotone");
    app->Options()->SetNumericValue("acceptable_obj_change_tol",  1e-4);
    app->Options()->SetIntegerValue("max_iter",  300);
    // app->Options()->SetStringValue("mu_strategy", "adaptive");
    app->Options()->SetStringValue("output_file", data_folder + "/ipopt_frame_" + std::to_string(objective.frame) + ".out");

    bool use_GN_hessian = true;
    if (!use_GN_hessian)
    {
        app->Options()->SetStringValue("hessian_approximation", "limited-memory");
        app->Options()->SetIntegerValue("limited_memory_max_history", 25);
    }
    app->Options()->SetIntegerValue("accept_after_max_steps", 25);
    // app->Options()->SetIntegerValue("print_level", 6);
    
    //        app->Options()->SetNumericValue("mu_max", 0.0001);
    
    //        bound_relax_factor
    // app->Options()->SetStringValue("derivative_test", "first-order");
    
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
    bool has_bound = true;
    Ipopt::SmartPtr<IpoptSolver> mynlp = new IpoptSolver(objective, data_folder, 
                                                        has_bound, use_GN_hessian, true);
    
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

void SensitivityAnalysis::saveConfig()
{
    std::ofstream out(data_folder + "/config.txt");
    out << "objective.use_penalty\t" << objective.use_penalty << std::endl;
    if (objective.use_penalty)
    {
        out << "objective.penalty_weight\t" << objective.penalty_weight << std::endl;
    }
    out << "objective.add_forward_potential\t" << objective.add_forward_potential << std::endl;
    out << "objective.w_fp\t" << objective.w_fp << std::endl;
    out << "objective.add_reg\t" << objective.add_reg << std::endl;
    out << "objective.reg_w\t" << objective.reg_w << std::endl;
    out << "objective.w_data\t" << objective.w_data << std::endl;
    out << "objective.add_spatial_regularizor\t" << objective.add_spatial_regularizor << std::endl;
    out << "objective.w_reg_spacial\t" << objective.w_reg_spacial << std::endl;
    out << "objective.match_centroid\t" << objective.match_centroid << std::endl;
    out << "objective.target_filename\t" << objective.target_filename << std::endl;
    out << "objective.target_perturbation\t" << objective.target_perturbation << std::endl;
    out << "objective.power\t" << objective.power << std::endl;
    out << "sa.add_reg\t" << add_reg << std::endl;
    out << "sa.reg_w_H\t" << reg_w_H << std::endl;
    out << "simulation.newton_tol\t" << simulation.newton_tol << std::endl;
    out.close();
}

void SensitivityAnalysis::runTracking(int start_frame, int end_frame, 
    bool load_weights, const std::string& filename)
{
    simulation.max_newton_iter = 300;
    objective.loadTargetTrajectory("/home/yueli/Documents/ETH/WuKong/Projects/CellSim/data/trajectories.dat");
    
    objective.match_centroid = false;
    objective.add_forward_potential = false;
    objective.w_fp = 1e-2;
    objective.power = 2;
    
    objective.add_reg = true;
    objective.reg_w = 1e-5;
    add_reg = !objective.add_reg;
    reg_w_H = 1e-6;
    objective.use_penalty = false;
    objective.penalty_weight = 0.0;

    objective.add_spatial_regularizor = true;
    if (objective.add_spatial_regularizor)
    {
        objective.w_reg_spacial = 1e-3;
        objective.buildVtxEdgeStructure();
    }

    simulation.save_mesh = false;
    simulation.cells.print_force_norm = false;
    simulation.verbose = false;
    
    objective.getSimulationAndDesignDoF(n_dof_sim, n_dof_design);
    objective.bound[0] = 0;
    objective.bound[1] = 40;
    objective.equilibrium_prev = VectorXT::Zero(simulation.deformed.rows());

    if (load_weights)
    {
        simulation.loadEdgeWeights(filename, design_parameters);
        objective.prev_params = design_parameters;
    }
    else
    {
        simulation.cells.edge_weights.setConstant(0.01);
        objective.prev_params = simulation.cells.edge_weights;
    }
    std::string base_folder = data_folder;
    for (int frame_id = start_frame; frame_id < end_frame + 1; frame_id++)
    {
        std::cout << "################################## FRAME " << frame_id << " ##################################" << std::endl;
        objective.setFrame(frame_id);
        std::string weights_filename = "/home/yueli/Documents/ETH/WuKong/Projects/CellSim/data/";
        std::filesystem::create_directories(base_folder + "/" + std::to_string(frame_id));
        data_folder = base_folder + "/" + std::to_string(frame_id);
        if (simulation.cells.resolution == 0)
            weights_filename += "weights_124.txt";
        else if (simulation.cells.resolution == 1)
            weights_filename += "weights_463.txt";
        else if (simulation.cells.resolution == 2)
            weights_filename += "weights_1500.txt";
        objective.loadWeightedCellTarget(weights_filename);
        
        // optimizeSQP();
        // optimizeSGN();
        optimizeIPOPT();
        if (save_results)
        {
            objective.saveState(data_folder + "/frame_" + std::to_string(frame_id) + ".obj");
            std::string filename = data_folder + "/frame_" + std::to_string(frame_id) + ".txt";
            saveDesignParameters(filename, design_parameters);
        }
        objective.prev_params = design_parameters;
        std::cout << "####################################################################" << std::endl;
        std::cout << std::endl;
    }
}
