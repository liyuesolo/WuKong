#include <Eigen/PardisoSupport>
#include <iomanip>
#include "../include/Objective.h"
#include "../include/autodiff/Deformation.h"

void Objective::getDesignParameters(VectorXT& design_parameters)
{
    design_parameters = vertex_model.apical_edge_contracting_weights;
}

void Objective::optimizeForStableTarget(T perturbation)
{
    TV c0, c1;
    vertex_model.computeCellCentroid(0, c0);
    vertex_model.computeCellCentroid(1, c1);
    
    T length = (c0 - c1).norm();
    
    VectorXT targets_inititial(target_positions.size() * 2);
    VectorXT targets_opt(target_positions.size() * 2);
    VectorXT targets_rest(target_positions.size() * 2);

    for (auto data : target_positions)
    {
        targets_rest.segment<2>(data.first * 2) = data.second;
        targets_inititial.segment<2>(data.first * 2) = data.second + perturbation * TV::Random() * length;
        targets_opt.segment<2>(data.first * 2) = targets_inititial.segment<2>(data.first * 2);
    }
    
    int n_cells = vertex_model.n_cells;
    T det_F_sum_init = 0.0, def_F_sum_final = 0.0;
    int dof = targets_inititial.rows();
    vertex_model.iterateCellSerial([&](VtxList& indices, int cell_idx)
    {
        int cell_i = cell_idx;
        int cell_j = (cell_idx + 1) % n_cells;
        int cell_k = (cell_idx - 1 + n_cells) % n_cells;

        TV ti, tj, tk;
        ti = targets_rest.segment<2>(cell_i * 2);
        tj = targets_rest.segment<2>(cell_j * 2);
        tk = targets_rest.segment<2>(cell_k * 2);

        TV ti_prime, tj_prime, tk_prime;
        ti_prime = targets_opt.segment<2>(cell_i * 2);
        tj_prime = targets_opt.segment<2>(cell_j * 2);
        tk_prime = targets_opt.segment<2>(cell_k * 2);

        TM rest; rest.col(0) = tj - ti; rest.col(1) = tk - ti;
        TM deformed; deformed.col(0) = tj_prime - ti_prime; deformed.col(1) = tk_prime - ti_prime;
        TM F = deformed * rest.inverse();
        det_F_sum_init += F.determinant();
    });

    auto fetchEntryFromVector = [&](const std::vector<int>& indices, 
        const VectorXT& vector, VectorXT& entries)
    {
        entries.resize(indices.size() * 2);
        for (int i = 0; i < indices.size(); i++)
            entries.segment<2>(i * 2) = vector.segment<2>(indices[i] * 2);
    };

    T w_p = 0.1;

    auto energyValue = [&](const VectorXT& x)
    {
        T energy = 0.0;
        energy += 0.5 * (x - targets_inititial).dot(x - targets_inititial);

        vertex_model.iterateCellSerial([&](VtxList& indices, int cell_idx)
        {
            int cell_i = cell_idx;
            int cell_j = (cell_idx + 1) % n_cells;
            int cell_k = (cell_idx - 1 + n_cells) % n_cells;

            VectorXT undeformed, deformed;
            fetchEntryFromVector({cell_k, cell_i, cell_j}, targets_rest, undeformed);
            fetchEntryFromVector({cell_k, cell_i, cell_j}, x, deformed);
            T ei;
            computeDeformationPenaltyDet(w_p, undeformed, deformed, ei);
            energy += ei;
        });

        return energy;
    };

    auto energyGradient = [&](const VectorXT& x, VectorXT& dedx)
    {
        dedx = x - targets_inititial;

        vertex_model.iterateCellSerial([&](VtxList& indices, int cell_idx)
        {
            int cell_i = cell_idx;
            int cell_j = (cell_idx + 1) % n_cells;
            int cell_k = (cell_idx - 1 + n_cells) % n_cells;

            VectorXT undeformed, deformed;
            std::vector<int> idx_local = {cell_k, cell_i, cell_j};
            fetchEntryFromVector(idx_local, targets_rest, undeformed);
            fetchEntryFromVector(idx_local, x, deformed);
            Vector<T, 6> dedx_local;
            computeDeformationPenaltyDetGradient(w_p, undeformed, deformed, dedx_local);
            for (int i = 0; i < idx_local.size(); i++)
            {
                dedx.segment<2>(idx_local[i] * 2) += dedx_local.segment<2>(i * 2);
            }
        });

        return dedx.norm();
    };

    auto energyHessian = [&](const VectorXT& x, std::vector<Entry>& entries)
    {
        vertex_model.iterateCellSerial([&](VtxList& indices, int cell_idx)
        {
            entries.push_back(Entry(cell_idx, cell_idx, 1.0));

            int cell_i = cell_idx;
            int cell_j = (cell_idx + 1) % n_cells;
            int cell_k = (cell_idx - 1 + n_cells) % n_cells;

            VectorXT undeformed, deformed;
            std::vector<int> idx_local = {cell_k, cell_i, cell_j};
            fetchEntryFromVector(idx_local, targets_rest, undeformed);
            fetchEntryFromVector(idx_local, x, deformed);
            Matrix<T, 6, 6> hessian_local;
            computeDeformationPenaltyDetHessian(w_p, undeformed, deformed, hessian_local);
            for (int i = 0; i < idx_local.size(); i++)
            {
                for (int j = 0; j < idx_local.size(); j++)
                {
                    for (int k = 0; k < 2; k++)
                    {
                        for (int l = 0; l < 2; l++)
                        {
                            entries.push_back(Entry(idx_local[i] * 2 + k, idx_local[j] * 2 + l, hessian_local(i * 2 + k, j * 2 + l)));
                        }   
                    }
                }
            }
        });
    };

    auto diffTestGradient = [&]()
    {
        T epsilon = 1e-7;
        VectorXT dx(dof);
        dx.setRandom();
        dx *= 1.0 / dx.norm();
        dx *= 0.001;
        if (perturb)
            targets_opt += dx;
        VectorXT g(dof);
        g.setZero();
        energyGradient(targets_opt, g);
        
        T E0 = energyValue(targets_opt);
        
        T previous = 0.0;
        for (int i = 0; i < 10; i++)
        {
            T E1 = energyValue(targets_opt + dx);
            T dE = E1 - E0;
            
            dE -= g.dot(dx);
            // std::cout << "dE " << dE << std::endl;
            if (i > 0)
            {
                std::cout << (previous/dE) << std::endl;
            }
            previous = dE;
            dx *= 0.5;
        }
    };

    auto diffTestHessian = [&]()
    {
        StiffnessMatrix H(dof, dof);
        std::vector<Entry> entries;
        energyHessian(targets_opt, entries);
        H.setFromTriplets(entries.begin(), entries.end());

        VectorXT f0(dof);
        energyGradient(targets_opt, f0);
        VectorXT dx(dof);
        dx.setRandom();
        dx *= 1.0 / dx.norm();
        dx *= 0.001;
        T previous = 0.0;
        for (int i = 0; i < 10; i++)
        {
            VectorXT f1(dof);
            energyGradient(targets_opt + dx, f1);
            T df_norm = (f0 + (H * dx) - f1).norm();
            // std::cout << "df_norm " << df_norm << std::endl;
            if (i > 0)
            {
                std::cout << (previous/df_norm) << std::endl;
            }
            previous = df_norm;
            dx *= 0.5;
        }
    };
    
    // diffTestGradient();
    // diffTestHessian();

    T tol = 1e-5;
    T g_norm = 1e10;
    int ls_max = 10;
    int opt_iter = 0;

    int max_iter = 10000;
    vertex_model.verbose = false;
    T g_norm0 = 0;
    while (true)
    {
        T O; 
        VectorXT dOdx;
        g_norm = energyGradient(targets_opt, dOdx);
        O = energyValue(targets_opt);
        std::cout << "iter " << opt_iter << " |g|: " << g_norm << " E: " << O << std::endl;
        if (opt_iter == 0)
            g_norm0 = g_norm;
        if (g_norm < tol * g_norm0 || opt_iter > max_iter)
            break;
        StiffnessMatrix H(dof, dof);
        std::vector<Entry> entries;
        energyHessian(targets_opt, entries);
        H.setFromTriplets(entries.begin(), entries.end());
        VectorXT g = -dOdx, dx = VectorXT::Zero(dof);
        vertex_model.linearSolve(H, g, dx);
        T alpha = 1.0;
        int i = 0;
        for (; i < ls_max; i++)
        {
            VectorXT x_ls = targets_opt + alpha * dx;
            T O_ls = energyValue(x_ls);
            if (O_ls < O)
            {
                targets_opt = x_ls;
                break;
            }
            alpha *= 0.5;
        }
        if (i == ls_max)
        {

        }
        std::cout << "#ls " << i << "/" << ls_max << std::endl;
        opt_iter++;
    }

    vertex_model.iterateCellSerial([&](VtxList& indices, int cell_idx)
    {
        int cell_i = cell_idx;
        int cell_j = (cell_idx + 1) % n_cells;
        int cell_k = (cell_idx - 1 + n_cells) % n_cells;

        TV ti, tj, tk;
        ti = targets_rest.segment<2>(cell_i * 2);
        tj = targets_rest.segment<2>(cell_j * 2);
        tk = targets_rest.segment<2>(cell_k * 2);

        TV ti_prime, tj_prime, tk_prime;
        ti_prime = targets_opt.segment<2>(cell_i * 2);
        tj_prime = targets_opt.segment<2>(cell_j * 2);
        tk_prime = targets_opt.segment<2>(cell_k * 2);

        TM rest; rest.col(0) = tj - ti; rest.col(1) = tk - ti;
        TM deformed; deformed.col(0) = tj_prime - ti_prime; deformed.col(1) = tk_prime - ti_prime;
        TM F = deformed * rest.inverse();
        def_F_sum_final += F.determinant();
    });
    std::cout << det_F_sum_init << " " << def_F_sum_final << std::endl;

    std::ofstream out("init.obj");
    for (int i = 0; i < n_cells; i++)
    {
        out << "v " << targets_inititial.segment<2>(i * 2).transpose() << " 0" << std::endl;
    }
    out.close();
    out.open("opt.obj");
    for (int i = 0; i < n_cells; i++)
    {
        out << "v " << targets_opt.segment<2>(i * 2).transpose() << " 0" << std::endl;
    }
    out.close();
    out.open("rest.obj");
    for (int i = 0; i < n_cells; i++)
    {
        out << "v " << targets_rest.segment<2>(i * 2).transpose() << " 0" << std::endl;
    }
    out.close();

    for (auto& data : target_positions)
    {
        data.second = targets_opt.segment<2>(data.first * 2);
    }
}

void Objective::loadTarget(const std::string& filename, T perturbation)
{
    target_filename = filename;
    target_perturbation = perturbation;
    std::ifstream in(filename);
    int idx; T x, y;
    TV c0, c1;
    vertex_model.computeCellCentroid(0, c0);
    vertex_model.computeCellCentroid(1, c1);
    
    T length = (c0 - c1).norm();

    while(in >> idx >> x >> y)
    {
        TV perturb = perturbation * TV::Random() * length;
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
            // T energy = vertex_model.computeTotalEnergy(vertex_model.u);
            // std::cout << std::setprecision(12) << "energy: " << energy << std::endl;
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
                T step_size = 1e-2;
                vertex_model.u += step_size * nodge_direction;
                // std::getchar();
            }
            else
                break;
        }
        vertex_model.u0 = vertex_model.u;
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
        {
            vertex_model.u = equilibrium_prev;
            vertex_model.u0 = equilibrium_prev;
        }
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
                T step_size = 1e-2;
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
            VectorXT negative_eigen_vector;
            T negative_eigen_value;
            bool has_neg_ev = vertex_model.fetchNegativeEigenVectorIfAny(negative_eigen_value,
                negative_eigen_vector);
            std::cout << "has neg ev: " << has_neg_ev << " " << negative_eigen_value << std::endl;
            std::exit(0);
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
    {
        equilibrium_prev = vertex_model.u;
    }

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
    if (add_l1_reg)
    {
        for (int i = 0; i < n_dof_design; i++)
        {
            if (p_curr[i] > 0)
                dOdp[i] += w_l1;
            else if (p_curr[i] < 0)
                dOdp[i] -= w_l1;
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

void Objective::computeEnergySubTerms(std::vector<T>& energy_terms)
{
    T Ox = 0.0;
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
    energy_terms.push_back(Ox);

    if (add_forward_potential)
    {
        T sim_energy = 0.0;
        VectorXT dx = vertex_model.deformed - vertex_model.undeformed;
        T simulation_potential = vertex_model.computeTotalEnergy(dx);
        simulation_potential *= w_fp;
        sim_energy += simulation_potential;
        energy_terms.push_back(sim_energy);
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
        if (contracting_term_only)
        {
            T contracting_energy = 0.0;
            vertex_model.addContractingEnergy(contracting_energy);
            Ox += w_fp * contracting_energy;
        }
        else
        {
            T simulation_potential = vertex_model.computeTotalEnergy(dx);
            simulation_potential *= w_fp;
            Ox += simulation_potential;
        }
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
        if (contracting_term_only)
        {
            vertex_model.addContractingForceEntries(cell_forces);
        }
        else
        {
            vertex_model.computeResidual(dx, cell_forces);
        }
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
        if (contracting_term_only)
        {
            vertex_model.addContractingHessianEntries(sim_H_entries);
            for (auto entry : sim_H_entries)
            {
                d2Odx2_entries.push_back(Entry(entry.row(), entry.col(), w_fp * entry.value()));
            }
            
        }
        else
        {
            StiffnessMatrix sim_H(n_dof_sim, n_dof_sim);
            vertex_model.buildSystemMatrix(dx, sim_H);
            sim_H *= w_fp;
            sim_H_entries = vertex_model.entriesFromSparseMatrix(sim_H);
            d2Odx2_entries.insert(d2Odx2_entries.end(), sim_H_entries.begin(), sim_H_entries.end());
        }
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

void Objective::saveState(const std::string& filename)
{
    vertex_model.saveStates(filename);
}