#include "EoLRodSim.h"
#include <Eigen/SparseCore>

template<class T, int dim>
T EoLRodSim<T, dim>::computeTotalEnergy(Eigen::Ref<const VectorXT> dq, 
    Eigen::Ref<const DOFStack> lambdas, T kappa, bool verbose)
{
    VectorXT dq_projected = dq;
    if(!add_penalty && !run_diff_test)
        iterateDirichletData([&](const auto& node_id, const auto& target, const auto& mask)
        {
            for(int d = 0; d < dof; d++)
                if (mask(d))
                    dq_projected(node_id * dof + d) = target(d);
        });

    DOFStack dq_full(dof, n_nodes);

    Eigen::Map<VectorXT>(dq_full.data(), dq_full.size()) = W * dq_projected;
    DOFStack q_temp = q0 + dq_full;

    T total_energy = 0;
    T E_stretching = 0, E_bending = 0, E_shearing = 0, 
        E_eul_reg = 0, E_pbc = 0, E_penalty = 0, E_contact = 0;
    if (add_stretching)
        E_stretching += addStretchingEnergy(q_temp);
    if (add_bending)
        E_bending += addBendingEnergy(q_temp);
    if (add_shearing)
    {
        E_shearing += addShearingEnergy(q_temp, true);
        E_shearing += addShearingEnergy(q_temp, false);
    }
    if (add_eularian_reg)
        E_eul_reg += addEulerianRegEnergy(q_temp);
    if (add_pbc)
    {
        if(use_alm)
            E_pbc += addPBCEnergyALM(q_temp, lambdas, kappa);
        else
            E_pbc += addPBCEnergy(q_temp);
    }
    if (add_penalty)
        iterateDirichletData([&](const auto& node_id, const auto& target, const auto& mask)
        {
            for(int d = 0; d < dof; d++)
                if (mask(d))
                    E_penalty += 0.5 * kc * std::pow(target(d) - dq_full(node_id * dof + d), 2);
        });
    if (add_contact_penalty)
        E_contact += addParallelContactEnergy(q_temp);
    total_energy = E_stretching + E_bending + E_shearing + E_eul_reg + E_pbc + E_penalty + E_contact;
    if (verbose)
        std::cout << "E_stretching " << E_stretching << " E_bending " << E_bending << 
        " E_shearing " << E_shearing << " E_eul_reg " << E_eul_reg << 
        " E_pbc " << E_pbc << " E_penalty " << E_penalty << " E_contact " << E_contact << std::endl;
    return total_energy;
}



template<class T, int dim>
T EoLRodSim<T, dim>::computeResidual(Eigen::Ref<VectorXT> residual, 
    Eigen::Ref<const VectorXT> dq, Eigen::Ref<const DOFStack> lambdas, T kappa)
{
    VectorXT dq_projected = dq;
    if(!add_penalty && !run_diff_test)
        iterateDirichletData([&](const auto& node_id, const auto& target, const auto& mask)
        {
            for(int d = 0; d < dof; d++)
                if (mask(d))
                    dq_projected(node_id * dof + d) = target(d);
        });

    DOFStack dq_full(dof, n_nodes);
    Eigen::Map<VectorXT>(dq_full.data(), dq_full.size()) = W * dq_projected;
    DOFStack q_temp = q0 + dq_full;
    

    DOFStack gradient_full(dof, n_nodes);
    gradient_full.setZero();

    if (add_stretching)
        addStretchingForce(q_temp, gradient_full);
    if (add_bending)
        addBendingForce(q_temp, gradient_full);
    if (add_shearing)
    {
        addShearingForce(q_temp, gradient_full, true);
        addShearingForce(q_temp, gradient_full, false);
    }
    if (add_pbc)
    {
        if(use_alm)
            addPBCForceALM(q_temp, gradient_full, lambdas, kappa);
        else
            addPBCForce(q_temp, gradient_full);
    }
    if (add_eularian_reg)
        addEulerianRegForce(q_temp, gradient_full);
    if (add_contact_penalty)
        addParallelContactForce(q_temp, gradient_full);
    if (add_penalty)
    {
        iterateDirichletData([&](const auto& node_id, const auto& target, const auto& mask)
        {
            for(int d = 0; d < dof; d++)
                if (mask(d))
                    gradient_full(d, node_id) -= kc * (dq_full(node_id * dof + d) - target(d));
        });
        residual = W.transpose() * Eigen::Map<const VectorXT>(gradient_full.data(), gradient_full.size());
        return residual.norm();
    }
    else
    {
        residual = W.transpose() * Eigen::Map<const VectorXT>(gradient_full.data(), gradient_full.size());
        
        if (!run_diff_test)
            iterateDirichletData([&](const auto& node_id, const auto& target, const auto& mask)
                {
                    for(int d = 0; d < dof; d++)
                        if (mask(d))
                            residual(node_id * dof + d) = 0.0;
                });
        return residual.norm();
    }
    
}
template<class T, int dim>
void EoLRodSim<T, dim>::addMassMatrix(std::vector<Eigen::Triplet<T>>& entry_K)
{
    for(int i = 0; i < n_nodes * dof; i++)
        entry_K.push_back(Eigen::Triplet<T>(i, i, km));    
}

template<class T, int dim>
void EoLRodSim<T, dim>::addStiffnessMatrix(std::vector<Entry>& entry_K, 
    Eigen::Ref<const VectorXT> dq, T kappa)
{
    VectorXT dq_projected = dq;
    if(!add_penalty && !run_diff_test)
        iterateDirichletData([&](const auto& node_id, const auto& target, const auto& mask)
        {
            for(int d = 0; d < dof; d++)
                if (mask(d))
                    dq_projected(node_id * dof + d) = target(d);
        });

    DOFStack dq_full(dof, n_nodes);
    Eigen::Map<VectorXT>(dq_full.data(), dq_full.size()) = W * dq_projected;

    DOFStack q_temp = q0 + dq_full;
    
    if (add_stretching)
        addStretchingK(q_temp, entry_K);
    if (add_bending)
        addBendingK(q_temp, entry_K);
    if (add_shearing)
    {
        addShearingK(q_temp, entry_K, true);
        addShearingK(q_temp, entry_K, false);
    }
    if (add_eularian_reg)
        addEulerianRegK(entry_K);
    if (add_pbc)
    {
        if (use_alm)
            addPBCKALM(q_temp, entry_K, kappa);
        else
            addPBCK(q_temp, entry_K);
    }
    if (add_contact_penalty)
        addParallelContactK(q_temp, entry_K);
}

template<class T, int dim>
void EoLRodSim<T, dim>::addConstraintMatrix(std::vector<Eigen::Triplet<T>>& entry_K)
{
    // penalty term
    iterateDirichletData([&](const auto& node_id, const auto& target, const auto& mask)
    {
        for(int d = 0; d < dof; d++)
            if (mask(d))
                entry_K.push_back(Eigen::Triplet<T>(node_id * dof + d, node_id * dof + d, kc));
    });
}

template<class T, int dim>
void EoLRodSim<T, dim>::buildSystemMatrix(
    Eigen::Ref<const VectorXT> dq, StiffnessMatrix& K, T kappa)
{
    
    std::vector<Entry> entry_K;
    if (add_regularizor)
        addMassMatrix(entry_K);
    addStiffnessMatrix(entry_K, dq, kappa);

    if (add_penalty)
        addConstraintMatrix(entry_K);
    
    StiffnessMatrix A(n_nodes * dof, n_nodes * dof);
    A.setFromTriplets(entry_K.begin(), entry_K.end());
    
    K = W.transpose() * A * W;

    if(!add_penalty && !run_diff_test)
        projectDirichletEntrySystemMatrix(K);


    K.makeCompressed();
}

template<class T, int dim>
bool EoLRodSim<T, dim>::projectDirichletEntrySystemMatrix(StiffnessMatrix& A)
{
    // project Dirichlet data, set Dirichlet nodal index to be 1
    iterateDirichletData([&](const auto& node_id, const auto& target, const auto& mask)
    {
        for(int d = 0; d < dof; d++)
            if (mask(d))
            {
                A.row(node_id * dof + d) *= 0.0;
                A.col(node_id * dof + d) *= 0.0;
                A.coeffRef(node_id * dof + d, node_id * dof + d) = 1.0;
            }
    });
    // A.makeCompressed();
}

template<class T, int dim>
bool EoLRodSim<T, dim>::linearSolve(StiffnessMatrix& K, 
    Eigen::Ref<const VectorXT> residual, Eigen::Ref<VectorXT> ddq)
{
    
    StiffnessMatrix I(K.rows(), K.cols());
    I.setIdentity();

    StiffnessMatrix H = K;
    Eigen::SimplicialLLT<StiffnessMatrix> solver;
    
    T mu = 10e-6;
    while(true)
    {
        solver.compute(K);
        if (solver.info() == Eigen::NumericalIssue)
        {
            // std::cout<< "indefinite" << std::endl;
            K = H + mu * I;        
            mu *= 10;
        }
        else
            break;
    }
    
    ddq = solver.solve(residual);
    
    if(solver.info()!=Eigen::Success) 
    {
        std::cout << "solving Ax=b failed ||Ax-b||: " << (K*ddq - residual).norm() << std::endl;
        return false;
    }
    return true;
}

template<class T, int dim>
T EoLRodSim<T, dim>::newtonLineSearch(Eigen::Ref<VectorXT> dq, 
    Eigen::Ref<const VectorXT> residual, 
    Eigen::Ref<const DOFStack> lambdas, T kappa,
    int line_search_max)
{
    bool verbose = false;
    int nz_stretching = 16 * n_rods;
    int nz_penalty = dof * dirichlet_data.size();

    VectorXT ddq(n_dof);
    ddq.setZero();

    StiffnessMatrix K;
    buildSystemMatrix(dq, K, kappa);
    bool success = linearSolve(K, residual, ddq);
    
    // T norm = ddq.cwiseAbs().maxCoeff();
    T norm = ddq.norm();
    // if (norm < 1e-6) return norm;
    T alpha = 1;
    T E0 = computeTotalEnergy(dq, lambdas, kappa);
    // std::cout << "E0: " << E0 << std::endl;
    int cnt = 0;
    bool set_to_gradient = true;
    while(true)
    {
        VectorXT dq_ls = dq + alpha * ddq;
        T E1 = computeTotalEnergy(dq_ls, lambdas, kappa, verbose);
        // std::cout << "E1: " << E1 << std::endl;
        if (E1 - E0 < 0) {
            dq = dq_ls;
            break;
        }
        alpha *= T(0.5);
        cnt += 1;
        if (cnt > 15)
        {
            // checkGradient(dq_ls);
            // std::cout << "!!!!!!!!!!!!!!!!!! line count: !!!!!!!!!!!!!!!!!!" << cnt << std::endl;
            // // std::cout << residual.transpose() << std::endl;
            // if(set_to_gradient)
            // {
            //     ddq = residual;
            //     alpha = 1.0;
            //     cnt = 0;
            //     set_to_gradient = false;
            // }
            // else
            {
                dq = dq_ls;
                return 1e16;
            }
            
            // // // return 0.0;
            // // verbose = true;
            // // T E0 = computeTotalEnergy(dq, true);
            // std::cout << "E0: " << E0 << std::endl;
            // // k_pbc = 0;
            // T E2 = computeTotalEnergy(dq + residual, true);
            // std::cout << "E2 " << E2 << std::endl;
            // // std::cout << "CHECKING GRADIENT AND HESSIAN " << std::endl;
            // // // checkGradient(dq_ls);
            // // // checkHessian(dq_ls);
            // // checkGradient(dq_ls);
            // // print_force_mag = true;
            // std::cout << "residual.norm() " << residual.norm() << std::endl;
            // // std::cout << residual.transpose() << std::endl;
            // // // std::cout << "E1: " << E1 << std::endl;
            // // // ddq = residual;
            // // // alpha = 100.0;
            // std::getchar();
        }
        if (cnt == line_search_max) 
            return 1e16;
            
    }
    // std::cout << "#ls: " << cnt << std::endl;
    return norm;    
}

template<class T, int dim>
void EoLRodSim<T, dim>::implicitUpdate(Eigen::Ref<VectorXT> dq)
{
    auto ALMUpdate = [&](Eigen::Ref<DOFStack> lambdas, T& kappa){
        int cons_cnt = 0;
        DOFStack dq_full(dof, n_nodes);
        Eigen::Map<VectorXT>(dq_full.data(), dq_full.size()) = W * dq;
        
        if(!add_penalty && !run_diff_test)
            iterateDirichletData([&](const auto& node_id, const auto& target, const auto& mask)
            {
                for(int d = 0; d < dof; d++)
                    if (std::abs(target(d)) <= 1e10 && mask(d))
                        dq_full(d, node_id) = target(d);
            });

        DOFStack q_temp = q0 + dq_full;
        iteratePBCReferencePairs([&](int yarn_type, int node_i, int node_j){
        
            int ref_i = pbc_ref_unique[yarn_type](0);
            int ref_j = pbc_ref_unique[yarn_type](1);

            TVDOF qi = q_temp.col(node_i);
            TVDOF qj = q_temp.col(node_j);
            TVDOF qi_ref = q_temp.col(ref_i);
            TVDOF qj_ref = q_temp.col(ref_j);

            if (ref_i == node_i && ref_j == node_j)
                return;

            TVDOF pair_dis_vec = qj - qi - (qj_ref - qi_ref);
            lambdas.col(cons_cnt) -= kappa * pair_dis_vec;
            cons_cnt++;
        });
        if (kappa < 1e8)
            kappa *= 2.0;
    };

    int cnt = 0;
    T residual_norm = 1e10, dq_norm = 1e10;
    
    int max_newton_iter = 1000;

    DOFStack lambdas(dof, n_pb_cons);
    lambdas.setZero();
    T kappa = 1e3;

    while (true)
    {
    
        VectorXT residual(n_dof);
        residual.setZero();
        
        computeResidual(residual, dq, lambdas, kappa);

        residual_norm = residual.norm();
        if (verbose)
            std::cout << "residual_norm " << residual_norm << std::endl;
        // std::getchar();
        if (residual_norm < newton_tol)
            break;
        
        dq_norm = newtonLineSearch(dq, residual, lambdas, kappa, 50);
        if (use_alm)
            ALMUpdate(lambdas, kappa);
        // std::cout << lambdas.transpose() << std::endl;
        // std::getchar();


        // std::cout << "dq_norm " << dq_norm << std::endl;
        // if (dq_norm < 1e-9 || dq_norm > 1e10)
        // if (dq_norm > 1e10)  
            // break;
        if(cnt == max_newton_iter)
            break;
        cnt++;
    }

    // if (verbose)
        std::cout << "# of newton solve: " << cnt << " exited with |g|: " << residual_norm << "|dq|: " << dq_norm  << std::endl;
}

template<class T, int dim>
void EoLRodSim<T, dim>::advanceOneStep()
{
    // newton_tol = 1e-6;
    VectorXT dq(n_dof);
    dq.setZero();
    implicitUpdate(dq);
    
    VectorXT dq_projected = dq;
    if(!add_penalty && !run_diff_test)
        iterateDirichletData([&](const auto& node_id, const auto& target, const auto& mask)
        {
            for(int d = 0; d < dof; d++)
                if (mask(d))
                    dq_projected(node_id * dof + d) = target(d);
        });
    DOFStack dq_full(dof, n_nodes);
    Eigen::Map<VectorXT>(dq_full.data(), dq_full.size()) = W * dq_projected;
    q += dq_full;

    
    // std::cout << "total Eulerian displacement " << dq_full.transpose().block(0, dim, n_nodes, 2).cwiseAbs().sum() << std::endl;
    // computeDeformationGradientUnitCell();
    // fitDeformationGradientUnitCell();
    // auto checkHessian = [&](){
    //     StiffnessMatrix A(n_nodes * dof, n_nodes * dof);
    //     std::vector<Eigen::Triplet<T>> entry_K;
    //     addStiffnessMatrix(entry_K, dq);
    //     A.setFromTriplets(entry_K.begin(), entry_K.end());
    //     projectDirichletEntrySystemMatrix(A);
    //     Eigen::SimplicialLLT<StiffnessMatrix> solver;
    //     solver.compute(A);
    //     Eigen::MatrixXd A_dense = A;
    //     Eigen::EigenSolver<Eigen::MatrixXd> eigen_solver;
    //     eigen_solver.compute(A_dense, /* computeEigenvectors = */ false);
    //     auto eigen_values = eigen_solver.eigenvalues();
    //     T min_ev = 1e10;
    //     for (int i = 0; i < A.cols(); i++)
    //         if (eigen_values[i].real() < min_ev)
    //             min_ev = eigen_values[i].real();
    //     std::cout << min_ev << std::endl;
    //     // std::cout << "The eigenvalues of the Hessian are: " << eigen_solver.eigenvalues().transpose() << std::endl;
    //     if (solver.info() == Eigen::NumericalIssue)
    //         std::cout << "Indefinite Hessian at Equilibrium" << std::endl;
    //     else
    //         std::cout << "Definite Hessian at Equilibrium" << std::endl;
    // };

    // checkHessian();
}

template<class T, int dim>
void EoLRodSim<T, dim>::fixEulerian()
{
    for(int i = 0; i < n_nodes; i++)
        if (connections.col(i).prod() < 0)
            dirichlet_data[i]= std::make_pair(TVDOF::Zero(), fix_eulerian);
    dirichlet_data[12] = std::make_pair(TVDOF::Zero(), fix_lagrangian);
}


template<class T, int dim>
void EoLRodSim<T, dim>::freeEulerian()
{
    dirichlet_data.clear();

    for (int i = 0; i < n_nodes; i++)
    {
        if (connections.col(i).prod() < 0)
            dirichlet_data[i]= std::make_pair(TVDOF::Zero(), fix_eulerian);    
    }

    dirichlet_data[12] = std::make_pair(TVDOF::Zero(), fix_lagrangian);
}

template class EoLRodSim<double, 3>;
template class EoLRodSim<double, 2>;