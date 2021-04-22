#include "EoLRodSim.h"
template<class T, int dim>
T EoLRodSim<T, dim>::computeTotalEnergy(Eigen::Ref<const DOFStack> dq)
{
    // advect q to compute internal energy
    DOFStack q_temp = q + dq;

    T total_energy = 0;
    if (add_stretching)
        total_energy += addStretchingEnergy(q_temp);
    if (add_bending)
        total_energy += addBendingEnergy(q_temp);
    if (add_shearing)
    {
        total_energy += addShearingEnergy(q_temp, true);
        total_energy += addShearingEnergy(q_temp, false);
    }
    if (add_eularian_reg)
        total_energy += addEulerianRegEnergy(q_temp);
    if (add_pbc)
        total_energy += addPBCEnergy(q_temp);
    if (add_penalty)
        iterateDirichletData([&](const auto& node_id, const auto& target, const auto& mask)
        {
            for(int d = 0; d < dof; d++)
                if (std::abs(target(d)) <= 1e10 && mask(d))
                    total_energy += 0.5 * kc * std::pow(target(d) - dq(d, node_id), 2);
        });

    return total_energy;
}

template<class T, int dim>
T EoLRodSim<T, dim>::computeResidual(Eigen::Ref<DOFStack> residual, Eigen::Ref<const DOFStack> dq)
{
    const DOFStack q_temp = q + dq;
    if (add_stretching)
        addStretchingForce(q_temp, residual);
    if (add_bending)
        addBendingForce(q_temp, residual);
    if (add_shearing)
    {
        addShearingForce(q_temp, residual, true);
        addShearingForce(q_temp, residual, false);
    }
    if (add_pbc)
        addPBCForce(q_temp, residual);
    if (add_eularian_reg)
        addEulerianRegForce(q_temp, residual);
    if (add_penalty)
        iterateDirichletData([&](const auto& node_id, const auto& target, const auto& mask)
        {
            for(int d = 0; d < dof; d++)
                if (std::abs(target(d)) <= 1e10 && mask(d))
                    residual(d, node_id) -= kc * (dq(d, node_id) - target(d));
        });
    return residual.norm();
}
template<class T, int dim>
void EoLRodSim<T, dim>::addMassMatrix(std::vector<Eigen::Triplet<T>>& entry_K)
{
    for(int i = 0; i < n_nodes * dof; i++)
        entry_K.push_back(Eigen::Triplet<T>(i, i, km));    
}

template<class T, int dim>
void EoLRodSim<T, dim>::addStiffnessMatrix(std::vector<Eigen::Triplet<T>>& entry_K, Eigen::Ref<const DOFStack> dq)
{
    const DOFStack q_temp = q + dq;
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
void EoLRodSim<T, dim>::addConstraintMatrix(std::vector<Eigen::Triplet<T>>& entry_K, Eigen::Ref<const DOFStack> dq)
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
void EoLRodSim<T, dim>::buildSystemMatrix(std::vector<Eigen::Triplet<T>>& entry_K, Eigen::Ref<const DOFStack> dq)
{
    if (add_regularizor)
        addMassMatrix(entry_K);
    addStiffnessMatrix(entry_K, dq);
    if (add_penalty)
        addConstraintMatrix(entry_K, dq);
}

template<class T, int dim>
bool EoLRodSim<T, dim>::linearSolve(const std::vector<Eigen::Triplet<T>>& entry_K, 
    Eigen::Ref<const DOFStack> residual, Eigen::Ref<DOFStack> ddq)
{
    using StiffnessMatrix = Eigen::SparseMatrix<T>;
    ddq.setZero();
    StiffnessMatrix A(n_nodes * dof, n_nodes * dof);
    
    StiffnessMatrix I(n_nodes * dof, n_nodes * dof);
    I.setIdentity();

    A.setFromTriplets(entry_K.begin(), entry_K.end()); 

    StiffnessMatrix H = A;
    Eigen::SimplicialLLT<Eigen::SparseMatrix<T>> solver;

    T mu = 10e-6;
    while(true)
    {
        solver.compute(A);
        if (solver.info() == Eigen::NumericalIssue)
        {
            // std::cout<< "indefinite" << std::endl;
            A = H + mu * I;
            mu *= 10;
        }
        else
            break;
    }
    const auto& rhs = Eigen::Map<const VectorXT>(residual.data(), residual.size());
    Eigen::Map<VectorXT>(ddq.data(), ddq.size()) = solver.solve(rhs);
    return true;
}

template<class T, int dim>
T EoLRodSim<T, dim>::newtonLineSearch(Eigen::Ref<DOFStack> dq, Eigen::Ref<const DOFStack> residual, int line_search_max)
{
    int nz_stretching = 16 * n_rods;
    int nz_penalty = dof * dirichlet_data.size();

    DOFStack ddq(dof, n_nodes);
    ddq.setZero();

    std::vector<Eigen::Triplet<T>> entry_K;
    buildSystemMatrix(entry_K, dq);
    linearSolve(entry_K, residual, ddq);
    T norm = ddq.cwiseAbs().maxCoeff();
    // T norm = ddq.norm();
    if (norm < 1e-5) return norm;
    T alpha = 1;
    T E0 = computeTotalEnergy(dq);
    // std::cout << "E0: " << E0 << std::endl;
    int cnt = 0;
    while(true)
    {
        DOFStack dq_ls = dq + alpha * ddq;
        T E1 = computeTotalEnergy(dq_ls);
        // std::cout << "E1: " << E1 << std::endl;
        if (E1 - E0 < 0) {
            dq = dq_ls;
            break;
        }
        alpha *= T(0.5);
        cnt += 1;
        if (cnt > 500)
        {
            std::cout << "line search count: " << cnt << std::endl;
        }
        if (cnt == line_search_max) 
            return 1e30;
            
    }
    return norm;    
}

template<class T, int dim>
void EoLRodSim<T, dim>::implicitUpdate(Eigen::Ref<DOFStack> dq)
{
    int cnt = 0;
    T norm = 1e10;
    while (true)
    {
        // set Dirichlet boundary condition
        // iterateDirichletData([&](const auto& node_id, const auto& target, const auto& mask)
        // {
        //     for(int d = 0; d < dof; d++)
        //         if (std::abs(target(d)) <= 1e10 && mask(d))
        //             dq(d, node_id) = target(d);
        // });
        
        DOFStack residual(dof, n_nodes);
        residual.setZero();
        T residual_norm = computeResidual(residual, dq);
        norm = newtonLineSearch(dq, residual);
        // std::cout << "|g|: " << norm << std::endl;
        if (norm < newton_tol)
            break;
        
        cnt++;
    }
    std::cout << "# of newton solve: " << cnt << " exited with |g|: " << norm << std::endl;
}

template<class T, int dim>
void EoLRodSim<T, dim>::advanceOneStep()
{
    DOFStack dq(dof, n_nodes);
    dq.setZero();
    implicitUpdate(dq);
    q += dq;
}

template class EoLRodSim<double, 3>;
template class EoLRodSim<double, 2>;