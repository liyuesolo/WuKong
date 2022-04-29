#include <Eigen/PardisoSupport>
#include "../include/Objective.h"

void Objective::getDesignParameters(VectorXT& design_parameters)
{
    design_parameters = vertex_model.apical_edge_contracting_weights;
}


void Objective::loadTarget(const std::string& filename, T perturbation)
{
    target_filename = filename;
    target_perturbation = perturbation;
    std::ifstream in(filename);
    int idx; T x, y;
    
    while(in >> idx >> x >> y)
    {
        TV perturb = perturbation * TV::Random();
        target_positions[idx] = TV(x, y) + perturb;
    }
    in.close();
}         

void Objective::updateDesignParameters(const VectorXT& design_parameters)
{
    vertex_model.apical_edge_contracting_weights = design_parameters;
}

T Objective::value(const VectorXT& p_curr, bool simulate, bool use_prev_equil)
{
    updateDesignParameters(p_curr);
    if (simulate)
    {
        vertex_model.reset();
        if (use_prev_equil)
            vertex_model.u = equilibrium_prev;
        while (true)
        {
            vertex_model.staticSolve();
            if (!perturb)
                break;
            VectorXT negative_eigen_vector;
            T negative_eigen_value;
            bool has_neg_ev = vertex_model.fetchNegativeEigenVectorIfAny(negative_eigen_value,
                negative_eigen_vector);
            if (has_neg_ev)
            {
                std::cout << "unstable state for the forward problem" << std::endl;
                std::cout << "nodge it along the negative eigen vector" << std::endl;
                VectorXT nodge_direction = negative_eigen_vector;
                T step_size = 1e-3;
                vertex_model.u += step_size * nodge_direction;
                // std::getchar();
            }
            else
                break;
        }
    }
    
    T energy = 0.0;
    computeOx(vertex_model.deformed, energy);
    
    T Op; computeOp(p_curr, Op);
    energy += Op;

    return energy;
}

void Objective::getSimulationAndDesignDoF(int& _sim_dof, int& _design_dof)
{
    _sim_dof = vertex_model.num_nodes * 2;
    _design_dof = vertex_model.apical_edge_contracting_weights.rows();
    n_dof_sim = _sim_dof;
    n_dof_design = _design_dof;
}

T Objective::gradient(const VectorXT& p_curr, VectorXT& dOdp, T& energy, bool simulate, bool use_prev_equil)
{
    updateDesignParameters(p_curr);
    if (simulate)
    {
        vertex_model.reset();
        if (use_prev_equil)
            vertex_model.u = equilibrium_prev;
        while (true)
        {
            vertex_model.staticSolve();
            if (!perturb)
                break;
            VectorXT negative_eigen_vector;
            T negative_eigen_value;
            bool has_neg_ev = vertex_model.fetchNegativeEigenVectorIfAny(negative_eigen_value,
                negative_eigen_vector);
            if (has_neg_ev)
            {
                std::cout << "unstable state for the forward problem" << std::endl;
                std::cout << "nodge it along the negative eigen vector" << std::endl;
                VectorXT nodge_direction = negative_eigen_vector;
                T step_size = 1e-3;
                vertex_model.u += step_size * nodge_direction;   
            }
            else
                break;
        }
    }
    
    energy = 0.0;
    VectorXT dOdx;

    computeOx(vertex_model.deformed, energy);
    computedOdx(vertex_model.deformed, dOdx);
    
    StiffnessMatrix d2edx2(n_dof_sim, n_dof_sim);
    vertex_model.buildSystemMatrix(vertex_model.u, d2edx2);

    vertex_model.iterateDirichletDoF([&](int offset, T target)
    {
        dOdx[offset] = 0;
    });
    
    VectorXT lambda;
    Eigen::PardisoLLT<StiffnessMatrix> solver;
    solver.analyzePattern(d2edx2);
    T alpha = 1.0;
    for (int i = 0; i < 50; i++)
    {
        solver.factorize(d2edx2);
        if (solver.info() == Eigen::NumericalIssue)
        { 
            std::cout << "[ERROR] simulation Hessian indefinite" << std::endl;
            d2edx2.diagonal().array() += alpha;
            alpha *= 10;
            continue;
        }
        break;    
    }
    lambda = solver.solve(dOdx);

    vertex_model.dOdpEdgeWeightsFromLambda(lambda, dOdp);
    // std::cout << "|dOdp matching|: " << dOdp.norm();
    // VectorXT dOdp_tmp = dOdp;
    // \partial O \partial p for the edge contracting force
    {
        if (add_forward_potential)
        {
            VectorXT dOdp_force(n_dof_design); dOdp_force.setZero();
            vertex_model.computededp(dOdp_force);
            dOdp += w_fp * dOdp_force;
        }
    }

    // std::exit(0);
    if (!use_prev_equil)
        equilibrium_prev = vertex_model.u;

    VectorXT partialO_partialp;
    computedOdp(p_curr, partialO_partialp);

    T Op; computeOp(p_curr, Op);
    dOdp += partialO_partialp;
    energy += Op;
    return dOdp.norm();
}

void Objective::hessianGN(const VectorXT& p_curr, MatrixXT& H, bool simulate)
{
    updateDesignParameters(p_curr);
    
    MatrixXT dxdp;
    
    vertex_model.dxdpFromdxdpEdgeWeights(dxdp);
    
    std::vector<Entry> d2Odx2_entries;
    computed2Odx2(vertex_model.deformed, d2Odx2_entries);
    StiffnessMatrix d2Odx2_matrix(n_dof_sim, n_dof_sim);
    d2Odx2_matrix.setFromTriplets(d2Odx2_entries.begin(), d2Odx2_entries.end());
    vertex_model.projectDirichletDoFMatrix(d2Odx2_matrix, vertex_model.dirichlet_data);
    
    // MatrixXT d2Odx2_dense = d2Odx2_matrix;
    // std::cout << d2Odx2_dense.minCoeff() << " " << d2Odx2_dense.maxCoeff() << std::endl;
    
    MatrixXT dxdpTHdxdp = dxdp.transpose() * d2Odx2_matrix * dxdp;

    H = dxdpTHdxdp;

    // 2 dx/dp^T d2O/dxdp
    if (add_forward_potential)
    {
        MatrixXT dfdp;
        vertex_model.dfdpWeightsDense(dfdp);
        dfdp *= -w_fp;
        H += dxdp.transpose() * dfdp + dfdp.transpose() * dxdp;
    }

    std::vector<Entry> d2Odp2_entries;
    computed2Odp2(p_curr, d2Odp2_entries);

    for (auto entry : d2Odp2_entries)
        H(entry.row(), entry.col()) += entry.value();
}

void Objective::hessianSGN(const VectorXT& p_curr, StiffnessMatrix& H, bool simulate)
{
    updateDesignParameters(p_curr);
    if (simulate)
    {
        vertex_model.reset();
        vertex_model.staticSolve();
    }

    std::vector<Entry> d2Odx2_entries;
    computed2Odx2(vertex_model.deformed, d2Odx2_entries);

    StiffnessMatrix dfdx(n_dof_sim, n_dof_sim);
    vertex_model.buildSystemMatrix(vertex_model.u, dfdx);
    dfdx *= -1.0;

    StiffnessMatrix dfdp;
    vertex_model.dfdpWeightsSparse(dfdp);

    StiffnessMatrix d2Odx2_mat(n_dof_sim, n_dof_sim);
    d2Odx2_mat.setFromTriplets(d2Odx2_entries.begin(), d2Odx2_entries.end());

    int nx = n_dof_sim, np = n_dof_design, nxnp = n_dof_sim + n_dof_design;

    H.resize(n_dof_sim * 2 + n_dof_design, n_dof_sim * 2 + n_dof_design);
    std::vector<Entry> entries;
    
    entries.insert(entries.end(), d2Odx2_entries.begin(), d2Odx2_entries.end());

    for (int i = 0; i < dfdx.outerSize(); i++)
    {
        for (StiffnessMatrix::InnerIterator it(dfdx, i); it; ++it)
        {
            entries.push_back(Entry(it.row() + nxnp, it.col(), it.value()));
            entries.push_back(Entry(it.row(), it.col() + nxnp, it.value()));
        }
    }
    
    for (int i = 0; i < dfdp.outerSize(); i++)
    {
        for (StiffnessMatrix::InnerIterator it(dfdp, i); it; ++it)
        {
            entries.push_back(Entry(it.col() + nx, it.row() + nxnp, it.value()));
            entries.push_back(Entry(it.row() + nxnp, it.col() + nx, it.value()));

            if (add_forward_potential)
            {
                entries.push_back(Entry(it.row(), it.col() + nx, -w_fp * it.value()));
                entries.push_back(Entry(it.col() + nx, it.row(), -w_fp * it.value()));
            }
        }
    }

    std::vector<Entry> d2Odp2_entries;
    computed2Odp2(p_curr, d2Odp2_entries);

    for (auto entry : d2Odp2_entries)
        entries.push_back(Entry(entry.row() + n_dof_sim, 
                                entry.col() + n_dof_sim, 
                                entry.value()));

    for (int i = 0; i < n_dof_sim; i++)
        entries.push_back(Entry(i, i, 1e-10));
    for (int i = 0; i < n_dof_design; i++)
        entries.push_back(Entry(i + n_dof_sim, i + n_dof_sim, 1e-10));
    for (int i = 0; i < n_dof_sim; i++)
        entries.push_back(Entry(i + n_dof_sim + n_dof_design, i + n_dof_sim + n_dof_design, -1e-10));
    
    
    H.setFromTriplets(entries.begin(), entries.end());
    H.makeCompressed();

}

void Objective::computeOp(const VectorXT& p_curr, T& Op)
{
    Op = 0.0;

    if (add_reg)
    {
        T reg_term = 0.5 * reg_w * (p_curr - prev_params).dot(p_curr - prev_params);
        Op += reg_term;
    }

    if (use_penalty)
    {
        T penalty_term = 0.0;
        for (int i = 0; i < n_dof_design; i++)
        {
            if (penalty_type == Qubic)
            {
                if (p_curr[i] < bound[0] && mask[0])
                    penalty_term += penalty_weight * std::pow(-(p_curr[i] - bound[0]), 3);
                if (p_curr[i] > bound[1] && mask[1])
                    penalty_term += penalty_weight * std::pow(p_curr[i] - bound[1], 3);
            }
        }
        Op += penalty_term;
    }
}

void Objective::computedOdp(const VectorXT& p_curr, VectorXT& dOdp)
{
    dOdp = VectorXT::Zero(n_dof_design);
    if (add_reg)
    {
        dOdp += reg_w * (p_curr - prev_params);
    }

    if (use_penalty)
    {
        for (int i = 0; i < n_dof_design; i++)
        {
            if (penalty_type == Qubic)
            {
                if (p_curr[i] < bound[0] && mask[0])
                {
                    dOdp[i] += -penalty_weight * 3.0 * std::pow(-(p_curr[i] - bound[0]), 2);
                }
                if (p_curr[i] > bound[1] && mask[1])
                {
                    dOdp[i] += penalty_weight * 3.0 * std::pow((p_curr[i] - bound[1]), 2);
                }
            }
        }
    }
}

void Objective::computed2Odp2(const VectorXT& p_curr, std::vector<Entry>& d2Odp2_entries)
{
    if (add_reg)
    {
        for (int i = 0; i < n_dof_design; i++)
        {
            d2Odp2_entries.push_back(Entry(i, i, reg_w));
        }
    }
    
    if (use_penalty)
    {
        for (int i = 0; i < n_dof_design; i++)
        {
            if (penalty_type == Qubic)
            {
                if (p_curr[i] < bound[0] && mask[0])
                {
                    T vi = -(p_curr[i] - bound[0]);
                    d2Odp2_entries.push_back(Entry(i, i, penalty_weight * 6.0 * vi));
                }
                if (p_curr[i] > bound[1] && mask[1])
                {
                    T vi = (p_curr[i] - bound[1]);
                    d2Odp2_entries.push_back(Entry(i, i, penalty_weight * 6.0 * vi));
                }
            }
        }
    }
}

void Objective::computeOx(const VectorXT& x, T& Ox)
{
    Ox = 0.0;
    vertex_model.deformed = x;
    
    T min_dis = 1e10, max_dis = -1e10, avg_dis = 0;
    int cnt = 0;
    if (match_centroid)
    {
        iterateTargets([&](int cell_idx, TV& target_pos)
        {
            TV centroid;
            vertex_model.computeCellCentroid(cell_idx, centroid);
            Ox += 0.5 * (centroid - target_pos).dot(centroid - target_pos);
        });
    }
    
    if (add_forward_potential)
    {
        VectorXT dx = vertex_model.deformed - vertex_model.undeformed;
        T simulation_potential = vertex_model.computeTotalEnergy(dx);
        simulation_potential *= w_fp;
        Ox += simulation_potential;
        // std::cout << "constracting energy: " << simulation_potential << std::endl;
    }
}

void Objective::computedOdx(const VectorXT& x, VectorXT& dOdx)
{
    vertex_model.deformed = x;
    dOdx.resize(n_dof_sim);
    dOdx.setZero();
    
    if (match_centroid)
    {
        iterateTargets([&](int cell_idx, TV& target_pos)
        {
            TV centroid;
            vertex_model.computeCellCentroid(cell_idx, centroid);
            std::vector<int> indices;
            vertex_model.getCellVtxIndices(indices, cell_idx);
            T dcdx = 1.0 / T(indices.size());
            for (int idx : indices)
            {
                TV dOdc = (centroid - target_pos);
                dOdx.segment<2>(idx * 2) += dOdc * dcdx;
            }
        });
    }
    if (add_forward_potential)
    {
        VectorXT cell_forces(n_dof_sim); cell_forces.setZero();
        VectorXT dx = vertex_model.deformed - vertex_model.undeformed;
        vertex_model.computeResidual(dx, cell_forces);
        dOdx -= w_fp * cell_forces;
    }
}

void Objective::computed2Odx2(const VectorXT& x, std::vector<Entry>& d2Odx2_entries)
{
    vertex_model.deformed = x;
    
    if (match_centroid)
    {
        iterateTargets([&](int cell_idx, TV& target_pos)
        {
            TV centroid;
            vertex_model.computeCellCentroid(cell_idx, centroid);
            std::vector<int> indices;
            vertex_model.getCellVtxIndices(indices, cell_idx);
            TM dcdx = TM::Identity() / T(indices.size());
            TM d2Odc2 = TM::Identity();
            TM tensor_term = TM::Zero(); // c is linear in x
            TM local_hessian = dcdx.transpose() * d2Odc2 * dcdx + tensor_term;

            for (int idx_i : indices)
                for (int idx_j : indices)
                    for (int d = 0; d < 2; d++)
                        for (int dd = 0; dd < 2; dd++)
                            d2Odx2_entries.push_back(Entry(idx_i * 2 + d, idx_j * 2 + dd, local_hessian(d, dd)));
        });
    }
    
    if (add_forward_potential)
    {
        std::vector<Entry> sim_H_entries;
        VectorXT dx = vertex_model.deformed - vertex_model.undeformed;
        StiffnessMatrix sim_H(n_dof_sim, n_dof_sim);
        vertex_model.buildSystemMatrix(dx, sim_H);
        sim_H *= w_fp;
        // std::vector<Entry> sim_potential_H_entries = vertex_model.entriesFromSparseMatrix(sim_H);
        // d2Odx2_entries.insert(d2Odx2_entries.end(), sim_potential_H_entries.begin(), sim_potential_H_entries.end());
    }
}

void Objective::diffTestGradientScale()
{
    std::cout << "###################### CHECK GRADIENT SCALE ######################" << std::endl;   
    VectorXT dOdp(n_dof_design);
    VectorXT p;
    getDesignParameters(p);
    T E0;
    // gradient(p, dOdp, E0, false);
    gradient(p, dOdp, E0);
    // std::cout << dOdp.minCoeff() << " " << dOdp.maxCoeff() << std::endl;
    VectorXT dp(n_dof_design);
    dp.setRandom();
    dp *= 1.0 / dp.norm();
    // dp *= 0.001;
    T previous = 0.0;
    
    for (int i = 0; i < 10; i++)
    {
        T E1 = value(p + dp);
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

void Objective::diffTestGradient()
{
    T epsilon = 1e-5;
    VectorXT dOdp(n_dof_design);
    dOdp.setZero();
    VectorXT p;
    getDesignParameters(p);
    T _dummy;
    gradient(p, dOdp, _dummy, true, false);

    for(int _dof_i = 0; _dof_i < n_dof_design; _dof_i++)
    {
        int dof_i = _dof_i;
        p[dof_i] += epsilon;
        T E1 = value(p);
        p[dof_i] -= 2.0 * epsilon;
        T E0 = value(p);
        p[dof_i] += epsilon;
        T fd = (E1 - E0) / (2.0 *epsilon);
        std::cout << "dof " << dof_i << " symbolic " << dOdp[dof_i] << " fd " << fd << std::endl;
        std::getchar();
    }
}