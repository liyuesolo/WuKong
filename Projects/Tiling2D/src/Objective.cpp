#include <igl/edges.h>
#include <igl/boundary_loop.h>
#include <ipc/ipc.hpp>
#include <igl/cotmatrix.h>
#include <ipc/barrier/adaptive_stiffness.hpp>
#include <Eigen/CholmodSupport>
#include "../include/Objective.h"

void Objective::projectDesignParameters(VectorXT& design_parameters)
{
    iterateDirichletDoF([&](int offset, T target)
    {
        design_parameters[offset] = target;
    });
}

void Objective::getSandwichPanelIndices(std::vector<int>& indices)
{
    
    indices = std::vector<int>();
    TV min_corner, max_corner;
    solver.computeBoundingBox(min_corner, max_corner);

    T width = (max_corner[1] - min_corner[1]) / 1.1 * 0.055;
    for (int i = 0; i < solver.num_nodes; i++)
    {
        TV xi = solver.undeformed.segment<2>(i * 2);
        bool y_bottom = xi[1] < min_corner[1] + width;
        bool y_top = xi[1] > max_corner[1] - width;
        if (y_bottom || y_top)
            indices.push_back(i);
    }
    
    // std::ofstream out("test.obj");
    // for (int idx : indices)
    // {
    //     out << "v " << solver.undeformed.segment<2>(idx * 2).transpose() << " 0" << std::endl;
    // }
    // out.close();
}

void Objective::diffTestGradient()
{
    solver.verbose = false;
    T epsilon = 1e-5;
    VectorXT dOdp(n_dof_design);
    dOdp.setZero();
    VectorXT p;
    getDesignParameters(p);
    T _dummy = 0.0;
    // simulation.newton_tol = 1e-9;
    solver.staticSolve();
    VectorXT init = solver.u;
    equilibrium_prev = init;
    // target_obj_weights.setZero();
    // w_reg_spacial = 0.0;
    gradient(p, dOdp, _dummy, true, true);
    VectorXT dirichlet_mask;
    getDirichletMask(dirichlet_mask);
    for(int _dof_i = 0; _dof_i < n_dof_design; _dof_i++)
    {
        if (dirichlet_mask[_dof_i] == 0.0)
            continue;
        std::cout << "dof i " << _dof_i << std::endl;
        int dof_i = _dof_i;
        p[dof_i] += epsilon;
        // std::cout << "p_i+1: " << p[dof_i] << " ";
        equilibrium_prev = init;
        T E1 = value(p, true, true);
        // saveState("debug_p_i_plus_1.obj");
        p[dof_i] -= 2.0 * epsilon;
        // std::cout << "p_i-1: " << p[dof_i] << std::endl;
        equilibrium_prev = init;
        T E0 = value(p, true, true);
        // saveState("debug_p_i_minus_1.obj");
        p[dof_i] += epsilon;
        T fd = (E1 - E0) / (2.0 *epsilon);
        if(dOdp[dof_i] == 0 && fd == 0)
            continue;
        // if (std::abs(dOdp[dof_i] - fd) < 1e-3 * std::abs(dOdp[dof_i]))
        //     continue;
        std::cout << "dof " << dof_i << " symbolic " << dOdp[dof_i] << " fd " << fd << std::endl;
        std::getchar();
    }
}



void Objective::diffTestGradientScale()
{
    solver.verbose = false;
    std::cout << "###################### CHECK GRADIENT SCALE ######################" << std::endl;   
    VectorXT dOdp(n_dof_design);
    VectorXT p;
    getDesignParameters(p);
    T E0 = 0.0;
    VectorXT init;
    
    solver.staticSolve();
    init = solver.u;
    equilibrium_prev = init;
    
    gradient(p, dOdp, E0, true, true);
    // std::cout << dOdp.minCoeff() << " " << dOdp.maxCoeff() << std::endl;
    VectorXT dp(n_dof_design);
    dp.setRandom();
    dp *= 1.0 / dp.norm();
    VectorXT dirichlet_mask;
    getDirichletMask(dirichlet_mask);
    dp.array() *= dirichlet_mask.array();
    // dp *= 0.001;
    T previous = 0.0;
    
    for (int i = 0; i < 10; i++)
    {
        // T E1 = value(p + dp, true, true);
        // VectorXT p1 = (p + dp).cwiseMax(bound[0]).cwiseMin(bound[1]);
        // dp = p1 - p;
        equilibrium_prev = init;
        T E1 = value(p + dp, true, true);
        T dE = E1 - E0;
        
        dE -= dOdp.dot(dp);
        // std::cout << "dE " << dE << std::endl;
        if (i > 0)
        {
            // std::cout << "scale" << std::endl;
            std::cout << (previous/dE) << std::endl;
            // std::getchar();
        }
        previous = dE;
        dp *= 0.5;
    }
}

void Objective::diffTestdOdxScale()
{
    std::cout << "###################### CHECK dOdx SCALE ######################" << std::endl;   
    VectorXT dOdx(n_dof_sim);
    VectorXT dx(n_dof_sim);
    dx.setRandom();
    dx *= 1.0 / dx.norm();
    dx *= 0.001;
    T previous = 0.0;
    VectorXT x = solver.deformed;
    computedOdx(x, dOdx);
    T E0 = 0.0;
    computeOx(x, E0);
    for (int i = 0; i < 10; i++)
    {
        T E1 = 0.0;
        computeOx(x + dx, E1);
        T dE = E1 - E0;
        
        dE -= dOdx.dot(dx);
        
        if (i > 0)
        {
            // std::cout << "scale" << std::endl;
            std::cout << (previous/dE) << std::endl;
            // std::getchar();
        }
        previous = dE;
        dx *= 0.5;
    }
}

void Objective::diffTestdOdx()
{
    T back_up = solver.penalty_weight;
    solver.penalty_weight = 1.0;
    std::cout << "###################### CHECK dOdx ENTRY ######################" << std::endl;   
    VectorXT dOdx(n_dof_sim);
    
    VectorXT x = solver.deformed;
    computedOdx(x, dOdx);
    
    VectorXT gradient_FD(n_dof_sim);
    gradient_FD.setZero();
    int cnt = 0;
    T epsilon = 1e-5;
    for(int dof_i = 0; dof_i < n_dof_sim; dof_i++)
    {
        x(dof_i) += epsilon;
        // std::cout << W * dq << std::endl;
        T E0 = 0.0; computeOx(x, E0);
        
        x(dof_i) -= 2.0 * epsilon;
        T E1 = 0.0; computeOx(x, E1);
        x(dof_i) += epsilon;
        // std::cout << "E1 " << E1 << " E0 " << E0 << std::endl;
        gradient_FD(dof_i) = (E0 - E1) / (2.0 *epsilon);
        if( gradient_FD(dof_i) == 0 && dOdx(dof_i) == 0)
            continue;
        if (std::abs( gradient_FD(dof_i) - dOdx(dof_i)) < 1e-3 * std::abs(dOdx(dof_i)))
            continue;
        std::cout << " dof " << dof_i << " " << gradient_FD(dof_i) << " " << dOdx(dof_i) << std::endl;
        std::getchar();
        cnt++;   
    }
    solver.penalty_weight = back_up;
    std::cout << "FD test passed" << std::endl;
}

void Objective::updateIPCVertices(const VectorXT& new_position)
{
    for (int i = 0; i < solver.num_nodes; i++)
        ipc_vertices.row(i) = new_position.segment<2>(i * 2);
}

void Objective::updateCotMat(const VectorXT& new_position)
{
    if (!add_reg_laplacian)
        return;
    for (int i = 0; i < solver.num_nodes; i++)
        surface_vertices.row(i) = new_position.segment<2>(i * 2);
    igl::cotmatrix(surface_vertices, surface_faces, cot_mat);
    cot_mat *= -1.0;
}

void Objective::buildIPCRestData()
{
    ipc_vertices.resize(solver.num_nodes, 2);
    for (int i = 0; i < solver.num_nodes; i++)
        ipc_vertices.row(i) = solver.undeformed.segment<2>(i * 2);
    num_ipc_vtx = ipc_vertices.rows();
    
    if (solver.use_quadratic_triangle)
        ipc_faces.resize(solver.num_ele * 4, 3);
    else
        ipc_faces.resize(solver.num_ele, 3);

    for (int i = 0; i < solver.num_ele; i++)
    {
        if (solver.use_quadratic_triangle)
        {
            ipc_faces.row(i* 4 + 0) = IV3(solver.indices[i * 6 + 0], solver.indices[i * 6 + 3], solver.indices[i * 6 + 5]);
            ipc_faces.row(i* 4 + 1) = IV3(solver.indices[i * 6 + 3], solver.indices[i * 6 + 1], solver.indices[i * 6 + 4]);
            ipc_faces.row(i* 4 + 2) = IV3(solver.indices[i * 6 + 5], solver.indices[i * 6 + 3], solver.indices[i * 6 + 4]);
            ipc_faces.row(i* 4 + 3) = IV3(solver.indices[i * 6 + 5], solver.indices[i * 6 + 4], solver.indices[i * 6 + 2]);
        }
        else
            ipc_faces.row(i) = solver.indices.segment<3>(i * 3);
    }
    igl::edges(ipc_faces, ipc_edges);
    ipc_faces.resize(0, 0);
    
    std::cout << "ipc has ixn in rest state: " << ipc::has_intersections(ipc_vertices, ipc_edges, ipc_faces) << std::endl;
    
    TV min_corner, max_corner;
    solver.computeBoundingBox(min_corner, max_corner);
    T bb_diag = (max_corner - min_corner).norm();
    VectorXT dedx(n_dof_design); dedx.setZero(); 
    
    T dummy = 0.0;
    gradient(X0, dedx, dummy, true, false);
    Eigen::MatrixXd ipc_vertices_deformed = ipc_vertices;
    for (int i = 0; i < solver.num_nodes; i++) 
        ipc_vertices_deformed.row(i) = X0.segment<2>(i * 2);

    ipc::Constraints ipc_constraints;
    ipc::construct_constraint_set(ipc_vertices, ipc_vertices_deformed, 
        ipc_edges, ipc_faces, barrier_distance, ipc_constraints);
    VectorXT dbdx = ipc::compute_barrier_potential_gradient(ipc_vertices_deformed, 
        ipc_edges, ipc_faces, ipc_constraints, barrier_distance);
    barrier_weight = ipc::initial_barrier_stiffness(bb_diag, barrier_distance, 1.0, dedx, dbdx, max_barrier_weight);
    // barrier_weight = 1e3;
    std::cout << "barrier weight " << barrier_weight << std::endl;
    // std::getchar();
}

void ObjFTF::getDesignParameters(VectorXT& design_parameters)
{
    design_parameters = solver.undeformed;
}

void ObjFTF::getDirichletMask(VectorXT& mask)
{
    T epsilon = 1e-4;
    mask = VectorXT::Ones(solver.num_nodes * 2);
    TV min_corner, max_corner;
    solver.computeBoundingBox(min_corner, max_corner);
    T width = (max_corner[1] - min_corner[1]) / 1.1 * 0.055;
    tbb::parallel_for(0, solver.num_nodes, [&](int i)
    {
        TV xi = solver.undeformed.segment<2>(i * 2);
        // bool x_left = xi[0] < min_corner[0] + epsilon;
        // bool x_right = xi[0] > max_corner[0] - epsilon;
        // bool y_bottom = xi[1] < min_corner[1] + epsilon;
        // bool y_top = xi[1] > max_corner[1] - epsilon;
        // if (x_left || x_right || y_bottom || y_top)
        bool y_bottom = xi[1] < min_corner[1] + width;
        bool y_top = xi[1] > max_corner[1] - width;
        if (y_bottom || y_top)
        {
            mask.segment<2>(i * 2).setZero();
        }
    });
}

void ObjFTF::diffTestGradientScale()
{
    solver.verbose = false;
    std::cout << "###################### CHECK GRADIENT SCALE ######################" << std::endl;   
    VectorXT dOdp(n_dof_design);
    VectorXT p;
    getDesignParameters(p);
    T E0 = 0.0;
    VectorXT init;
    if (sequence)
    {
        generateSequenceData(init, true, false);
    }
    else
    {
        solver.staticSolve();
        init = solver.u;
        equilibrium_prev = init;
    }
    gradient(p, dOdp, E0, true, true);
    // std::cout << dOdp.minCoeff() << " " << dOdp.maxCoeff() << std::endl;
    VectorXT dp(n_dof_design);
    dp.setRandom();
    dp *= 1.0 / dp.norm();
    VectorXT dirichlet_mask;
    getDirichletMask(dirichlet_mask);
    dp.array() *= dirichlet_mask.array();
    // dp *= 0.001;
    T previous = 0.0;
    
    for (int i = 0; i < 10; i++)
    {
        // T E1 = value(p + dp, true, true);
        // VectorXT p1 = (p + dp).cwiseMax(bound[0]).cwiseMin(bound[1]);
        // dp = p1 - p;
        equilibrium_prev = init;
        T E1 = value(p + dp, true, true);
        T dE = E1 - E0;
        
        dE -= dOdp.dot(dp);
        // std::cout << "dE " << dE << std::endl;
        if (i > 0)
        {
            // std::cout << "scale" << std::endl;
            std::cout << (previous/dE) << std::endl;
            // std::getchar();
        }
        previous = dE;
        dp *= 0.5;
    }
}

void ObjFTF::getDirichletIndices(std::vector<int>& indices)
{
    T epsilon = 1e-4;
    indices = std::vector<int>();
    TV min_corner, max_corner;
    solver.computeBoundingBox(min_corner, max_corner);

    for (int i = 0; i < solver.num_nodes; i++)
    {
        TV xi = solver.undeformed.segment<2>(i * 2);
        bool x_left = xi[0] < min_corner[0] + epsilon;
        bool x_right = xi[0] > max_corner[0] - epsilon;
        bool y_bottom = xi[1] < min_corner[1] + epsilon;
        bool y_top = xi[1] > max_corner[1] - epsilon;
        if (x_left || x_right || y_bottom || y_top)
            indices.push_back(i);
    }
}

T ObjFTF::maximumStepSize(const VectorXT& p_curr, const VectorXT& search_dir)
{
    Eigen::MatrixXd current_position = ipc_vertices, 
        next_step_position = ipc_vertices;
        
    for (int i = 0; i < solver.num_nodes; i++)
    {
        current_position.row(i) = p_curr.segment<2>(i * 2);
        next_step_position.row(i) = p_curr.segment<2>(i * 2) + search_dir.segment<2>(i * 2);
    }
    return ipc::compute_collision_free_stepsize(current_position, 
            next_step_position, ipc_edges, ipc_faces, ipc::BroadPhaseMethod::HASH_GRID, 1e-6, 1e7);
}

void ObjFTF::updateDesignParameters(const VectorXT& design_parameters)
{
    solver.undeformed = design_parameters;
    
}

T ObjFTF::value(const VectorXT& p_curr, bool simulate, bool use_prev_equil)
{
    updateDesignParameters(p_curr);
    T Ox = 0.0, Op = 0.0;
    if (sequence)
    {
        VectorXT x_sequence;
        generateSequenceData(x_sequence, simulate, use_prev_equil);
        computeOx(x_sequence, Ox);
    }
    else
    {
        if (simulate)
        {
            solver.reset();
            if (use_prev_equil)
                solver.u = equilibrium_prev;
            solver.staticSolve();
        }
        computeOx(solver.deformed, Ox);
    }
    computeOp(p_curr, Op);
    return Ox + Op;
}

T ObjFTF::gradient(const VectorXT& p_curr, VectorXT& dOdp, T& energy, 
    bool simulate, bool use_prev_equil)
{
    dOdp = VectorXT::Zero(n_dof_design);
    updateDesignParameters(p_curr);
    
    VectorXT dOdx; T Ox = 0.0;
    if (sequence)
    {
        VectorXT x_sequence;
        generateSequenceData(x_sequence, simulate, use_prev_equil);
        computeOx(x_sequence, Ox);
        computedOdx(x_sequence, dOdx);
        for (int i = 0; i < num_data_point; i++)
        {
            StiffnessMatrix d2edx2(n_dof_sim, n_dof_sim);
            bool flag = solver.project_block_PD;
            solver.project_block_PD = false;
            VectorXT ui = x_sequence.segment(i * n_dof_sim, n_dof_sim) - solver.undeformed;
            solver.buildSystemMatrix(ui, d2edx2);
            solver.project_block_PD = flag;
            
            solver.iterateDirichletDoF([&](int offset, T target)
            {
                dOdx[offset] = 0;
            });

            Eigen::CholmodSupernodalLLT<StiffnessMatrix, Eigen::Lower> cholmod_solver;
            cholmod_solver.compute(d2edx2);

            VectorXT lambda;
            lambda = -1.0 * cholmod_solver.solve(dOdx.segment(i * n_dof_sim, n_dof_sim));
            
            StiffnessMatrix dfdX(n_dof_sim, n_dof_sim);
            solver.builddfdX(ui, dfdX);
            dOdp += lambda.transpose() * -dfdX;
            if (!use_prev_equil)
                equilibrium_prev.segment(i * n_dof_sim, n_dof_sim) = solver.u;
        }
    }
    else
    {
        if (simulate)
        {
            solver.reset();
            if (use_prev_equil)
                solver.u = equilibrium_prev;
            solver.staticSolve();
        }
        computeOx(solver.deformed, Ox);
        computedOdx(solver.deformed, dOdx);
        StiffnessMatrix d2edx2(n_dof_sim, n_dof_sim);
        bool flag = solver.project_block_PD;
        solver.project_block_PD = false;
        solver.buildSystemMatrix(solver.u, d2edx2);
        solver.project_block_PD = flag;
        
        solver.iterateDirichletDoF([&](int offset, T target)
        {
            dOdx[offset] = 0;
        });

        Eigen::CholmodSupernodalLLT<StiffnessMatrix, Eigen::Lower> cholmod_solver;
        cholmod_solver.compute(d2edx2);

        VectorXT lambda;
        lambda = -1.0 * cholmod_solver.solve(dOdx);
        
        StiffnessMatrix dfdX(n_dof_sim, n_dof_sim);
        solver.builddfdX(solver.u, dfdX);
        dOdp = lambda.transpose() * -dfdX;
        if (!use_prev_equil)
            equilibrium_prev = solver.u;
    }

    VectorXT partialO_partialp;
    computedOdp(p_curr, partialO_partialp);

    T Op = 0.0; computeOp(p_curr, Op);
    dOdp += partialO_partialp;
    energy = Ox + Op;

    iterateDirichletDoF([&](int offset, T target)
    {
        dOdp[offset] = 0;
    });

    return dOdp.norm();
}

void ObjFTF::hessianGN(const VectorXT& p_curr, MatrixXT& H, bool simulate, bool use_prev_equil)
{   
    updateDesignParameters(p_curr);
    if (sequence)
    {
        VectorXT x_sequence;
        generateSequenceData(x_sequence, simulate, use_prev_equil);
        
        MatrixXT dxdp(num_data_point * n_dof_sim, n_dof_design);
        for (int i = 0; i < num_data_point; i++)
        {
            StiffnessMatrix d2edx2(n_dof_sim, n_dof_sim);
            bool flag = solver.project_block_PD;
            solver.project_block_PD = false;
            VectorXT ui = x_sequence.segment(i * n_dof_sim, n_dof_sim) - solver.undeformed;
            solver.buildSystemMatrix(ui, d2edx2);
            solver.project_block_PD = flag;
            Eigen::CholmodSupernodalLLT<StiffnessMatrix, Eigen::Lower> cholmod_solver;
            cholmod_solver.analyzePattern(d2edx2);
            cholmod_solver.factorize(d2edx2);
            if (cholmod_solver.info() == Eigen::NumericalIssue)
            {
                std::cout << "forward hessian indefinite when computing dxdp" << std::endl;
            }
            StiffnessMatrix dfdX(n_dof_sim, n_dof_sim);
            solver.builddfdX(ui, dfdX);
            MatrixXT dfdp = dfdX;
            dxdp.block(i * n_dof_sim, 0, n_dof_sim, n_dof_design) = cholmod_solver.solve(dfdp);
        }
        std::vector<Entry> d2Odx2_entries;
        computed2Odx2(x_sequence, d2Odx2_entries);
        StiffnessMatrix d2Odx2_matrix(n_dof_sim * num_data_point, n_dof_sim * num_data_point);
        d2Odx2_matrix.setFromTriplets(d2Odx2_entries.begin(), d2Odx2_entries.end());
        // solver.projectDirichletDoFMatrix(d2Odx2_matrix, solver.dirichlet_data);
        H = dxdp.transpose() * d2Odx2_matrix * dxdp;
    }
    else
    {
        if (simulate)
        {
            solver.reset();
            if (use_prev_equil)
                solver.u = equilibrium_prev;
            solver.staticSolve();
        }
        MatrixXT dxdp(n_dof_sim, n_dof_design);
        StiffnessMatrix d2edx2(n_dof_sim, n_dof_sim);
        solver.buildSystemMatrix(solver.u, d2edx2);
        Eigen::CholmodSupernodalLLT<StiffnessMatrix, Eigen::Lower> cholmod_solver;
        cholmod_solver.analyzePattern(d2edx2);
        cholmod_solver.factorize(d2edx2);
        if (cholmod_solver.info() == Eigen::NumericalIssue)
        {
            std::cout << "forward hessian indefinite when computing dxdp" << std::endl;
        }
        StiffnessMatrix dfdX;
        solver.builddfdX(solver.u, dfdX);
        MatrixXT dfdp = dfdX;
        dxdp.noalias() = cholmod_solver.solve(dfdp);
        
        std::vector<Entry> d2Odx2_entries;
        computed2Odx2(solver.deformed, d2Odx2_entries);
        StiffnessMatrix d2Odx2_matrix(n_dof_sim, n_dof_sim);
        d2Odx2_matrix.setFromTriplets(d2Odx2_entries.begin(), d2Odx2_entries.end());
        solver.projectDirichletDoFMatrix(d2Odx2_matrix, solver.dirichlet_data);
        H = dxdp.transpose() * d2Odx2_matrix * dxdp;
    }
    
    std::vector<Entry> d2Odp2_entries;
    computed2Odp2(p_curr, d2Odp2_entries);

    for (auto entry : d2Odp2_entries)
        H(entry.row(), entry.col()) += entry.value();
    iterateDirichletDoF([&](int offset, T target)
    {
        H.row(offset).setZero();
        H.col(offset).setZero();
        H(offset, offset) = 1.0;
    });

}

void ObjFTF::generateSequenceData(VectorXT& x, 
    bool simulate, bool use_prev_equil)
{
    solver.reset();
    x = VectorXT::Zero(f_target.rows());
    f_current = VectorXT::Zero(f_target.rows());
    T dp = 0.05;
    solver.penalty_weight = 1e4;
    
    VectorXT u_prev = solver.u;
    if (use_prev_equil)
        u_prev = equilibrium_prev.segment(0, n_dof_sim);

    TV min_corner, max_corner;
    solver.computeBoundingBox(min_corner, max_corner);
    TV min1(min_corner[0] - 1e-6, max_corner[1] - 1e-6);
    TV max1(max_corner[0] + 1e-6, max_corner[1] + 1e-6);

    T dy = max_corner[1] - min_corner[1];
    int cnt = 0;
    for (T dis = 0.0; dis < 0.4 + dp; dis += dp)
    {
        solver.penalty_pairs.clear();
        solver.addPenaltyPairsBox(min1, max1, TV(0, -dis * dy));
        solver.addPenaltyPairsBoxXY(TV(min_corner[0] - 1e-6, max_corner[1] - 1e-6), 
            TV(min_corner[0] + 1e-6, max_corner[1] + 1e-6), 
            TV(0, -dis * dy));
        if (simulate)
        {
            solver.u = equilibrium_prev.segment(cnt * n_dof_sim, n_dof_sim);
            solver.staticSolve();
            // u_prev = solver.u;
        }
        else
        {
            solver.u = equilibrium_prev.segment(cnt * n_dof_sim, n_dof_sim);
        }
        VectorXT interal_force(solver.num_nodes * 2);
        interal_force.setZero();
        solver.addBCPenaltyForceEntries(solver.penalty_weight, interal_force);
        x.segment(cnt * n_dof_sim, n_dof_sim) = solver.undeformed + solver.u;
        f_current.segment(cnt * n_dof_sim, n_dof_sim) = interal_force;
        cnt++;
        if (cnt >= num_data_point)
            break;
    }
}

void ObjFTF::computeOx(const VectorXT& x, T& Ox)
{
    VectorXT f;
    f = VectorXT::Zero(f_target.rows());
    if (sequence)
    {
        f = f_current;
    }
    else
    {
        solver.deformed = x;
        solver.addBCPenaltyForceEntries(solver.penalty_weight, f);
    }
    // std::cout << "=========================== f ===========================" << std::endl;
    // std::cout << f.transpose() << std::endl;
    // std::cout << "=========================== f target ===========================" << std::endl;
    // std::cout << f_target.transpose() << std::endl;
    // std::cout << "-----------------------------------------" << std::endl;
    // std::exit(0);
    Ox += 0.5 * (f - f_target).dot(f - f_target);
}

void ObjFTF::computedOdx(const VectorXT& x, VectorXT& dOdx)
{
    VectorXT f;
    f = VectorXT::Zero(f_target.rows());
    if (sequence)
    {
        f = f_current;
    }
    else
    {
        solver.deformed = x;
        solver.addBCPenaltyForceEntries(solver.penalty_weight, f);
    }
    dOdx = -solver.penalty_weight * (f - f_target);
}

void ObjFTF::computed2Odx2(const VectorXT& x, std::vector<Entry>& d2Odx2_entries)
{
    if (sequence)
    {
        for (int i = 0; i < num_data_point; i++)
        {
            solver.iterateBCPenaltyPairs([&](int offset, T target)
            {
                d2Odx2_entries.push_back(Entry(i * n_dof_sim + offset, i * n_dof_sim + offset, 
                    std::pow(solver.penalty_weight, 2)));
            });    
        }
    }
    else
    {
        solver.iterateBCPenaltyPairs([&](int offset, T target)
        {
            d2Odx2_entries.push_back(Entry(offset, offset, std::pow(solver.penalty_weight, 2)));
        });
    }
}

void ObjFTF::loadTargetFromFile(const std::string& filename)
{
    std::ifstream in(filename);
    std::vector<T> data_vec;
    T fi;
    while (in >> fi)
    {
        data_vec.push_back(fi);
    }
    in.close();
    f_target = Eigen::Map<VectorXT>(data_vec.data(), data_vec.size());
    for (int i = 0; i < f_target.rows(); i++)
    {
        if (std::abs(f_target[i]) > 1e-6 && i % 2 == 1)
        {
            f_target[i] *= 0.9;
        }
    }
    
}

void ObjFTF::initialize()
{
    n_dof_design = solver.num_nodes * 2;
    n_dof_sim = solver.num_nodes * 2;
    
    if (add_reg_laplacian)
    {
        int n_faces = solver.surface_indices.rows() / 3;
        surface_vertices.resize(solver.num_nodes, 2);
        surface_faces.resize(n_faces, 3);
        tbb::parallel_for(0, solver.num_nodes, [&](int i)
        {
            surface_vertices.row(i) = solver.undeformed.segment<2>(i * 2);
        });

        tbb::parallel_for(0, n_faces, [&](int i)
        {
            surface_faces.row(i) = solver.surface_indices.segment<3>(i * 3);
        });

        igl::cotmatrix(surface_vertices, surface_faces, cot_mat);
        cot_mat *= -1.0;
    }
    setX0(solver.undeformed);
    if (sequence)
        equilibrium_prev = VectorXT::Zero(solver.undeformed.rows() * num_data_point);
    else
        equilibrium_prev = VectorXT::Zero(solver.undeformed.rows());
    if (use_ipc)
        buildIPCRestData();

    std::vector<int> dirichlet_indices;
    // getDirichletIndices(dirichlet_indices);
    getSandwichPanelIndices(dirichlet_indices);
    if (add_pbc)
        pbc_pairs = solver.pbc_pairs[0];
    for (int idx : dirichlet_indices)
    {
        dirichlet_data[idx * 2 + 0] = solver.undeformed[idx * 2 + 0];
        dirichlet_data[idx * 2 + 1] = solver.undeformed[idx * 2 + 1];
    }
    std::cout << "objective initialized" << std::endl;
}



void ObjFTF::computeOp(const VectorXT& p_curr, T& Op)
{
    if (add_reg_rest)
    {
        T reg_rest = w_reg_rest * 0.5 * (p_curr - X0).dot(p_curr - X0);
        Op += reg_rest;
    }
    if (use_ipc)
    {
        T contact_energy = 0.0;
    
        Eigen::MatrixXd ipc_vertices_deformed = ipc_vertices;
        for (int i = 0; i < solver.num_nodes; i++) 
        {
            ipc_vertices_deformed.row(i) = p_curr.segment<2>(i * 2);
        }

        ipc::Constraints ipc_constraints;
        ipc::construct_constraint_set(ipc_vertices, ipc_vertices_deformed, 
            ipc_edges, ipc_faces, barrier_distance, ipc_constraints);

        contact_energy = barrier_weight * ipc::compute_barrier_potential(ipc_vertices_deformed, 
        ipc_edges, ipc_faces, ipc_constraints, barrier_distance);

        Op += contact_energy;
    }
    if (add_reg_laplacian)
    {
        VectorXT vx(solver.num_nodes), vy(solver.num_nodes);
        for (int i = 0; i < solver.num_nodes; i++)
        {
            vx[i] = p_curr[i * 2 + 0];
            vy[i] = p_curr[i * 2 + 1];
        }
        T e_lap = w_reg_laplacian * 0.5 * vx.transpose() * cot_mat * vx;
        e_lap += w_reg_laplacian * 0.5 * vy.transpose() * cot_mat * vy;
        
        // std::cout << "e lap " << e_lap << std::endl;
        Op += e_lap;
    }
    if (add_pbc)
    {
        int cnt = 0;
        for (auto pbc_pair : pbc_pairs)
        {
            // strain term
            int idx0 = pbc_pair[0], idx1 = pbc_pair[1];
            
            TV xi = p_curr.segment<2>(idx0 * 2);
            TV xj = p_curr.segment<2>(idx1 * 2);
            
            cnt++;
            if (cnt == 1)
                continue;
            
            TV xi_ref = p_curr.segment<2>(pbc_pairs[0][0] * 2);
            TV xj_ref = p_curr.segment<2>(pbc_pairs[0][1] * 2);

            TV pair_dis_vec = xj - xi - (xj_ref - xi_ref);
            Op += 0.5 * pbc_w * pair_dis_vec.dot(pair_dis_vec);
        }
    }
}
void ObjFTF::computedOdp(const VectorXT& p_curr, VectorXT& dOdp)
{
    dOdp = VectorXT::Zero(p_curr.rows());
    if (add_reg_rest)
        dOdp += w_reg_rest * (p_curr - X0);
    if (use_ipc)
    {
        Eigen::MatrixXd ipc_vertices_deformed = ipc_vertices;
        for (int i = 0; i < solver.num_nodes; i++) 
        {
            ipc_vertices_deformed.row(i) = p_curr.segment<2>(i * 2);
        }

        ipc::Constraints ipc_constraints;
        ipc::construct_constraint_set(ipc_vertices, ipc_vertices_deformed, 
            ipc_edges, ipc_faces, barrier_distance, ipc_constraints);
        VectorXT contact_gradient = barrier_weight * ipc::compute_barrier_potential_gradient(ipc_vertices_deformed, 
            ipc_edges, ipc_faces, ipc_constraints, barrier_distance);
        dOdp += contact_gradient;
    }
    if (add_reg_laplacian)
    {
        VectorXT vx(solver.num_nodes), vy(solver.num_nodes);
        for (int i = 0; i < solver.num_nodes; i++)
        {
            vx[i] = p_curr[i * 2 + 0];
            vy[i] = p_curr[i * 2 + 1];
        }
        VectorXT dedvx = w_reg_laplacian * cot_mat * vx;
        VectorXT dedvy = w_reg_laplacian * cot_mat * vy;
        for (int i = 0; i < solver.num_nodes; i++)
        {
            dOdp[i * 2 + 0] += dedvx[i];
            dOdp[i * 2 + 1] += dedvy[i];
        }
    }
    if (add_pbc)
    {
        int cnt = 0;
        for (auto pbc_pair : pbc_pairs)
        {
            // strain term
            int idx0 = pbc_pair[0], idx1 = pbc_pair[1];
            
            TV xi = p_curr.segment<2>(idx0 * 2);
            TV xj = p_curr.segment<2>(idx1 * 2);
            
            cnt++;
            if (cnt == 1)
                continue;
            
            TV xi_ref = p_curr.segment<2>(pbc_pairs[0][0] * 2);
            TV xj_ref = p_curr.segment<2>(pbc_pairs[0][1] * 2);

            TV pair_dis_vec = xj - xi - (xj_ref - xi_ref);
            dOdp.segment<2>(idx0 * 2) += -pbc_w * pair_dis_vec;
            dOdp.segment<2>(idx1 * 2) += pbc_w * pair_dis_vec;

            dOdp.segment<2>(pbc_pairs[0][0] * 2) += pbc_w * pair_dis_vec;
            dOdp.segment<2>(pbc_pairs[0][1] * 2) += -pbc_w * pair_dis_vec;
        }
    }
}
void ObjFTF::computed2Odp2(const VectorXT& p_curr, std::vector<Entry>& d2Odp2_entries)
{
    if (add_reg_rest)
    {
        for (int i = 0; i < n_dof_design; i++)
        {
            d2Odp2_entries.push_back(Entry(i, i, w_reg_rest));
        }
        
    }

    if (use_ipc)
    {
        Eigen::MatrixXd ipc_vertices_deformed = ipc_vertices;
        for (int i = 0; i < solver.num_nodes; i++) 
        {
            ipc_vertices_deformed.row(i) = p_curr.segment<2>(i * 2);
        }

        ipc::Constraints ipc_constraints;
        ipc::construct_constraint_set(ipc_vertices, ipc_vertices_deformed, 
            ipc_edges, ipc_faces, barrier_distance, ipc_constraints);
        StiffnessMatrix contact_hessian = barrier_weight *  ipc::compute_barrier_potential_hessian(ipc_vertices_deformed, 
            ipc_edges, ipc_faces, ipc_constraints, barrier_distance, true);
        
        std::vector<Entry> contact_entries = entriesFromSparseMatrix(contact_hessian);
    
        d2Odp2_entries.insert(d2Odp2_entries.end(), contact_entries.begin(), contact_entries.end());
    }
    if (add_reg_laplacian)
    {
        std::vector<Entry> contact_entries = entriesFromSparseMatrix(cot_mat);
        for (auto entry : contact_entries)
        {
            d2Odp2_entries.push_back(Entry(entry.row() * 2 + 0, entry.col() * 2 + 0, entry.value() * w_reg_laplacian));
            d2Odp2_entries.push_back(Entry(entry.row() * 2 + 1, entry.col() * 2 + 1, entry.value() * w_reg_laplacian));
        }   
    }
    if (add_pbc)
    {
        int cnt = 0;
        for (auto pbc_pair : pbc_pairs)
        {
            // strain term
            int idx0 = pbc_pair[0], idx1 = pbc_pair[1];
            
            TV xi = p_curr.segment<2>(idx0 * 2);
            TV xj = p_curr.segment<2>(idx1 * 2);
            
            cnt++;
            if (cnt == 1)
                continue;
            
            TV xi_ref = p_curr.segment<2>(pbc_pairs[0][0] * 2);
            TV xj_ref = p_curr.segment<2>(pbc_pairs[0][1] * 2);

            TV pair_dis_vec = xj - xi - (xj_ref - xi_ref);
            std::vector<int> nodes = {idx0, idx1, pbc_pairs[0][0], pbc_pairs[0][1]};
            std::vector<T> sign_J = {-1, 1, 1, -1};
            std::vector<T> sign_F = {1, -1, -1, 1};

            for(int k = 0; k < 4; k++)
                for(int l = 0; l < 4; l++)
                    for(int i = 0; i < 2; i++)
                        d2Odp2_entries.push_back(Entry(nodes[k]*2 + i, nodes[l] * 2 + i, -pbc_w *sign_F[k]*sign_J[l]));
        }
    }
}

void ObjFTF::loadTarget(const std::string& data_folder)
{
    if (sequence)
    {
        std::vector<VectorXT> forces;
        for (int i = 0; i < num_data_point; i++)
        {
            std::ifstream in(data_folder + std::to_string(i) + ".txt");
            std::vector<T> data_vec;
            T fi;
            while (in >> fi)
            {
                data_vec.push_back(fi);
            }
            in.close();
            forces.push_back(Eigen::Map<VectorXT>(data_vec.data(), data_vec.size()));
        }
        int n_entry_per_force = forces[0].rows();
        f_target.resize(forces.size() * n_entry_per_force);
        for (int i = 0; i < num_data_point; i++)
            f_target.segment(i * n_entry_per_force, n_entry_per_force) = forces[i];
    }
    else
    {
        
    }
}

void ObjFTF::generateTarget(const std::string& target_folder)
{
    if (sequence)
    {
        T dp = 0.05;
        solver.penalty_weight = 1e4;
        std::vector<T> displacements;
        std::vector<T> force_norms;
        VectorXT u_prev = solver.u;

        TV min_corner, max_corner;
        solver.computeBoundingBox(min_corner, max_corner);
        TV min1(min_corner[0] - 1e-6, max_corner[1] - 1e-6);
        TV max1(max_corner[0] + 1e-6, max_corner[1] + 1e-6);

        T dy = max_corner[1] - min_corner[1];
        int cnt = 0;
        for (T dis = 0.0; dis < 0.4 + dp; dis += dp)
        {
            std::cout << "\t---------pencent " << dp << std::endl;
            std::cout << dis << std::endl;
            T displacement_sum = 0.0;
            solver.penalty_pairs.clear();
            solver.addPenaltyPairsBox(min1, max1, TV(0, -dis * dy));
            solver.addPenaltyPairsBoxXY(TV(min_corner[0] - 1e-6, max_corner[1] - 1e-6), 
                TV(min_corner[0] + 1e-6, max_corner[1] + 1e-6), 
                TV(0, -dis * dy));
            // solver.y_bar = max_corner[1] - dis * dy;
            solver.u = u_prev;
            solver.staticSolve();
            u_prev = solver.u;
            VectorXT interal_force(solver.num_nodes * 2);
            interal_force.setZero();
            solver.addBCPenaltyForceEntries(solver.penalty_weight, interal_force);
            displacements.push_back(dis * dy);
            force_norms.push_back(interal_force.norm());
            solver.savePenaltyForces(target_folder + std::to_string(cnt) + ".txt");
            solver.saveToOBJ(target_folder + std::to_string(cnt) + ".obj");
            cnt++;
        }
        std::ofstream out(target_folder + "log.txt");
        out << "displacement in cm" << std::endl;
        for (T v : displacements)
            out << v << " ";
        out << std::endl;
        out << "force in N" << std::endl;
        for (T v : force_norms)
            out << v << " ";
        out << std::endl;
        out.close();
    }
}