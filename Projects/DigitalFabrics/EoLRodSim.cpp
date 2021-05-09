#include "EoLRodSim.h"
#include <Eigen/SparseCore>

template<class T, int dim>
T EoLRodSim<T, dim>::computeTotalEnergy(Eigen::Ref<const VectorXT> dq, bool verbose)
{
    // advect q to compute internal energy
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

    T total_energy = 0;
    T E_stretching = 0, E_bending = 0, E_shearing = 0, E_eul_reg = 0, E_pbc = 0, E_penalty = 0;
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
        E_pbc += addPBCEnergy(q_temp);
    if (add_penalty)
        iterateDirichletData([&](const auto& node_id, const auto& target, const auto& mask)
        {
            for(int d = 0; d < dof; d++)
                if (std::abs(target(d)) <= 1e10 && mask(d))
                    E_penalty += 0.5 * kc * std::pow(target(d) - dq(d, node_id), 2);
        });
    total_energy = E_stretching + E_bending + E_shearing + E_eul_reg + E_pbc + E_penalty;
    if (verbose)
        std::cout << "E_stretching " << E_stretching << " E_bending " << E_bending << 
        " E_shearing " << E_shearing << " E_eul_reg " << E_eul_reg << 
        " E_pbc " << E_pbc << " E_penalty " << E_penalty << std::endl;
    return total_energy;
}



template<class T, int dim>
T EoLRodSim<T, dim>::computeResidual(Eigen::Ref<VectorXT> residual, Eigen::Ref<const VectorXT> dq)
{
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
        addPBCForce(q_temp, gradient_full);
    if (add_eularian_reg)
        addEulerianRegForce(q_temp, gradient_full);
    if (add_penalty)
        iterateDirichletData([&](const auto& node_id, const auto& target, const auto& mask)
        {
            for(int d = 0; d < dof; d++)
                if (std::abs(target(d)) <= 1e10 && mask(d))
                    gradient_full(d, node_id) -= kc * (dq(d, node_id) - target(d));
        });
    else
    {
        if (!run_diff_test)
            iterateDirichletData([&](const auto& node_id, const auto& target, const auto& mask)
                {
                    for(int d = 0; d < dof; d++)
                        if (std::abs(target(d)) <= 1e10 && mask(d))
                            gradient_full(d, node_id) = 0.0;
                });
    }
        
    // std::cout << gradient_full.norm() << std::endl;
    residual = W.transpose() * Eigen::Map<const VectorXT>(gradient_full.data(), gradient_full.size());
    // std::cout << residual.norm() << std::endl;
    // std::getchar();
    return residual.norm();
}
template<class T, int dim>
void EoLRodSim<T, dim>::addMassMatrix(std::vector<Eigen::Triplet<T>>& entry_K)
{
    for(int i = 0; i < n_nodes * dof; i++)
        entry_K.push_back(Eigen::Triplet<T>(i, i, km));    
}

template<class T, int dim>
void EoLRodSim<T, dim>::addStiffnessMatrix(std::vector<Entry>& entry_K, Eigen::Ref<const VectorXT> dq)
{
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
        addPBCK(q_temp, entry_K);
}

template<class T, int dim>
void EoLRodSim<T, dim>::addConstraintMatrix(std::vector<Eigen::Triplet<T>>& entry_K)
{
    // penalty term
    iterateDirichletData([&](const auto& node_id, const auto& target, const auto& mask)
    {
        for(int d = 0; d < dof; d++)
            if (std::abs(target(d)) <= 1e10 && mask(d))
                entry_K.push_back(Eigen::Triplet<T>(node_id * dof + d, node_id * dof + d, kc));
    });
}

template<class T, int dim>
void EoLRodSim<T, dim>::buildSystemMatrix(
    Eigen::Ref<const VectorXT> dq, StiffnessMatrix& K)
{
    
    std::vector<Entry> entry_K;
    if (add_regularizor)
        addMassMatrix(entry_K);
    addStiffnessMatrix(entry_K, dq);
    if (add_penalty)
        addConstraintMatrix(entry_K);
    
    StiffnessMatrix A(n_nodes * dof, n_nodes * dof);
    A.setFromTriplets(entry_K.begin(), entry_K.end());
    
    if(!add_penalty && !run_diff_test)
        projectDirichletEntrySystemMatrix(A);
    
    K = W.transpose() * A * W;
    K.makeCompressed();
}

template<class T, int dim>
bool EoLRodSim<T, dim>::projectDirichletEntrySystemMatrix(StiffnessMatrix& A)
{
    // project Dirichlet data, set the row and col of Dirichlet nodal dof to be zero first
    for (int k=0; k<A.outerSize(); ++k)
        for (typename StiffnessMatrix::InnerIterator it(A,k); it; ++it)
        {
            int node_i = std::floor(it.row() / dof);
            int dof_i = it.row() % dof;
            int node_j = std::floor(it.col() / dof);
            int dof_j = it.col() % dof;            
            if(dirichlet_data.find(node_i) != dirichlet_data.end())
            {
                TVDOF mask = dirichlet_data[node_i].second;
                if(mask(dof_i))
                    it.valueRef() = 0.0;
            }
            if(dirichlet_data.find(node_j) != dirichlet_data.end())
            {
                TVDOF mask = dirichlet_data[node_j].second;
                if(mask(dof_j))
                    it.valueRef() = 0.0;
            }
        }
    // project Dirichlet data, set Dirichlet nodal index to be 1
    iterateDirichletData([&](const auto& node_id, const auto& target, const auto& mask)
    {
        for(int d = 0; d < dof; d++)
            if (std::abs(target(d)) <= 1e10 && mask(d))
                A.coeffRef(node_id * dof + d, node_id * dof + d) = 1.0;
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
T EoLRodSim<T, dim>::newtonLineSearch(Eigen::Ref<VectorXT> dq, Eigen::Ref<const VectorXT> residual, int line_search_max)
{
    bool verbose = false;
    int nz_stretching = 16 * n_rods;
    int nz_penalty = dof * dirichlet_data.size();

    VectorXT ddq(n_dof);
    ddq.setZero();

    StiffnessMatrix K;
    buildSystemMatrix(dq, K);
    bool success = linearSolve(K, residual, ddq);
    
    // T norm = ddq.cwiseAbs().maxCoeff();
    T norm = ddq.norm();
    // if (norm < 1e-6) return norm;
    T alpha = 1;
    T E0 = computeTotalEnergy(dq);
    // std::cout << "E0: " << E0 << std::endl;
    int cnt = 0;
    bool set_to_gradient = true;
    while(true)
    {
        VectorXT dq_ls = dq + alpha * ddq;
        T E1 = computeTotalEnergy(dq_ls, verbose);
        // std::cout << "E1: " << E1 << std::endl;
        if (E1 - E0 < 0) {
            dq = dq_ls;
            break;
        }
        alpha *= T(0.5);
        cnt += 1;
        if (cnt > 15)
        {
            dq = dq_ls;
            return 1e16;
            // checkGradient(dq_ls);
            // std::cout << "!!!!!!!!!!!!!!!!!! line count: !!!!!!!!!!!!!!!!!!" << cnt << std::endl;
            // // std::cout << residual.transpose() << std::endl;
            if(set_to_gradient)
            {
                ddq = residual;
                alpha = 1.0;
                cnt = 0;
                set_to_gradient = false;
            }
            else
            {
                dq = dq_ls;
                // std::cout << residual.norm() << std::endl;
                // T E0 = computeTotalEnergy(dq, true);
                // T E2 = computeTotalEnergy(dq + residual, true);
                // E2 = computeTotalEnergy(dq + 0.01 * residual, true);
                // E2 = computeTotalEnergy(dq + alpha * residual, true);
                // std::getchar();
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
    int cnt = 0;
    T residual_norm = 1e10, dq_norm = 1e10;
    // this is not a hack, just used to debug intermediate results
    int hard_set_exit_number = INT_MAX;
    while (true)
    {
    
        VectorXT residual(n_dof);
        residual.setZero();
        // set Dirichlet boundary condition
        
        
        computeResidual(residual, dq);

        residual_norm = residual.norm();
        // std::cout << "residual_norm " << residual_norm << std::endl;
        // std::getchar();
        if (residual_norm < newton_tol)
            break;
        
        dq_norm = newtonLineSearch(dq, residual, 100);
        // std::cout << "dq_norm " << dq_norm << std::endl;
        // if (dq_norm < 1e-9 || dq_norm > 1e10)
        // if (dq_norm > 1e10)
            // break;
        if(cnt == hard_set_exit_number)
            break;
        cnt++;
    }

    if (verbose)
        std::cout << "# of newton solve: " << cnt << " exited with |g|: " << residual_norm << "|dq|: " << dq_norm  << std::endl;
}

template<class T, int dim>
void EoLRodSim<T, dim>::advanceOneStep()
{
    // newton_tol = 1e-6;
    VectorXT dq(n_dof);
    dq.setZero();
    implicitUpdate(dq);
    
    DOFStack dq_full(dof, n_nodes);
    Eigen::Map<VectorXT>(dq_full.data(), dq_full.size()) = W * dq;
    q += dq_full;

    // computeDeformationGradientUnitCell();
    // fitDeformationGradientUnitCell();
    auto checkHessian = [&](){
        StiffnessMatrix A(n_nodes * dof, n_nodes * dof);
        std::vector<Eigen::Triplet<T>> entry_K;
        addStiffnessMatrix(entry_K, dq);
        A.setFromTriplets(entry_K.begin(), entry_K.end());
        projectDirichletEntrySystemMatrix(A);
        Eigen::SimplicialLLT<StiffnessMatrix> solver;
        solver.compute(A);
        Eigen::MatrixXd A_dense = A;
        Eigen::EigenSolver<Eigen::MatrixXd> eigen_solver;
        eigen_solver.compute(A_dense, /* computeEigenvectors = */ false);
        auto eigen_values = eigen_solver.eigenvalues();
        T min_ev = 1e10;
        for (int i = 0; i < A.cols(); i++)
            if (eigen_values[i].real() < min_ev)
                min_ev = eigen_values[i].real();
        std::cout << min_ev << std::endl;
        // std::cout << "The eigenvalues of the Hessian are: " << eigen_solver.eigenvalues().transpose() << std::endl;
        if (solver.info() == Eigen::NumericalIssue)
            std::cout << "Indefinite Hessian at Equilibrium" << std::endl;
        else
            std::cout << "Definite Hessian at Equilibrium" << std::endl;
    };

    // checkHessian();
}

template class EoLRodSim<double, 3>;
template class EoLRodSim<double, 2>;