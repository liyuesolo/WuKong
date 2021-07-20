#include "EoLRodSim.h"
#include <Eigen/SparseCore>
#include <fstream>
template<class T, int dim>
T EoLRodSim<T, dim>::computeTotalEnergy(Eigen::Ref<const VectorXT> dq, 
        bool verbose)
{
    VectorXT dq_projected = dq;
    
    if(!add_penalty && !run_diff_test)
        iterateDirichletDoF([&](int offset, T target)
        {
            dq_projected[offset] = target;
        });

   
    deformed_states = rest_states + W * dq_projected;
    for (auto& rod : Rods)
    {
        rod->reference_angles = deformed_states.template segment(rod->theta_dof_start_offset, 
            rod->indices.size() - 1);
    }
    for (auto& crossing : rod_crossings)
        crossing->omega = deformed_states.template segment<3>(crossing->dof_offset);

    T total_energy = 0;
    T E_stretching = 0, E_bending = 0, E_shearing = 0, E_twisting = 0, E_bending_twisting = 0,
        E_eul_reg = 0, E_pbc = 0, E_penalty = 0, E_contact = 0;
    
    if (add_stretching)
        E_stretching += addStretchingEnergy();
    if constexpr (dim == 3)
    {
        if (add_bending && add_twisting)
            E_bending_twisting = add3DBendingAndTwistingEnergy();
        if (add_rigid_joint)
            E_bending_twisting += addJointBendingAndTwistingEnergy();
    }
    if (add_pbc)
        E_pbc += addPBCEnergy();
    if (add_eularian_reg)
        E_eul_reg += addEulerianRegEnergy();
    if (add_contact_penalty)
        E_contact += addParallelContactEnergy();
    total_energy = E_stretching + E_bending + E_shearing + E_twisting + E_bending_twisting + E_eul_reg + E_pbc + E_penalty + E_contact;
    
    if (verbose)
        std::cout << "E_stretching " << E_stretching << " E_bending " 
        << E_bending << " E_twisting " << E_twisting << 
        " E_bending_twisting " << E_bending_twisting <<  
        " E_shearing " << E_shearing << " E_eul_reg " << E_eul_reg << 
        " E_pbc " << E_pbc << " E_penalty " << E_penalty << " E_contact " << E_contact << std::endl;
    return total_energy;
}


template<class T, int dim>
T EoLRodSim<T, dim>::computeResidual(Eigen::Ref<VectorXT> residual, Eigen::Ref<const VectorXT> dq)
{
    VectorXT dq_projected = dq;
    if(!add_penalty && !run_diff_test)
        iterateDirichletDoF([&](int offset, T target)
        {
            dq_projected[offset] = target;
        });

    deformed_states = rest_states + W * dq_projected;
    for (auto& rod : Rods)
    {
        rod->reference_angles = deformed_states.template segment(rod->theta_dof_start_offset, 
            rod->indices.size() - 1);
        rod->rotateReferenceFrameToLastNewtonStepAndComputeReferenceTwsit();
    }

    for (auto& crossing : rod_crossings)
    {
        crossing->omega = deformed_states.template segment<3>(crossing->dof_offset);
        // crossing->updateRotation(deformed_states.template segment<3>(crossing->dof_offset));
    }
        // crossing->updateRotation(deformed_states.template segment<3>(crossing->dof_offset));

    VectorXT full_residual(deformed_states.rows());
    full_residual.setZero();

    if (add_stretching)
        addStretchingForce(full_residual);
    if constexpr (dim == 3)
    {
        if (add_bending && add_twisting)
            add3DBendingAndTwistingForce(full_residual);
        if (add_rigid_joint)
            addJointBendingAndTwistingForce(full_residual);
    }
    if (add_pbc)
        addPBCForce(full_residual);
    if (add_eularian_reg)
        addEulerianRegForce(full_residual);
    if (add_contact_penalty)
        addParallelContactForce(full_residual);
    
    // std::cout << " full residual " << std::endl;
    // std::cout << full_residual << std::endl;
    // std::getchar();
    
    residual = W.transpose() * full_residual;
    
    if (!run_diff_test)
        iterateDirichletDoF([&](int offset, T target)
        {
            residual[offset] = 0;
        });
    
    return residual.norm();
}

template<class T, int dim>
void EoLRodSim<T, dim>::addStiffnessMatrix(std::vector<Eigen::Triplet<T>>& entry_K,
         Eigen::Ref<const VectorXT> dq)
{
    VectorXT dq_projected = dq;
    if(!add_penalty && !run_diff_test)
        iterateDirichletDoF([&](int offset, T target)
        {
            dq_projected[offset] = target;
        });

    deformed_states = rest_states + W * dq_projected;
    for (auto& rod : Rods)
    {
        rod->reference_angles = deformed_states.template segment(rod->theta_dof_start_offset, rod->indices.size() - 1);
    }
    for (auto& crossing : rod_crossings)
        crossing->omega = deformed_states.template segment<3>(crossing->dof_offset);

    if (add_stretching)
        addStretchingK(entry_K);
    if constexpr (dim == 3)
    {
        if (add_bending && add_twisting)
            add3DBendingAndTwistingK(entry_K);
        if (add_rigid_joint)
            addJointBendingAndTwistingK(entry_K);
    }
    if (add_pbc)
        addPBCK(entry_K);
    if (add_eularian_reg)
        addEulerianRegK(entry_K);
    if (add_contact_penalty)
        addParallelContactK(entry_K);
}

template<class T, int dim>
void EoLRodSim<T, dim>::buildSystemDoFMatrix(
    Eigen::Ref<const VectorXT> dq, StiffnessMatrix& K)
{
    std::vector<Entry> entry_K;
    addStiffnessMatrix(entry_K, dq);
    StiffnessMatrix A(deformed_states.rows(), deformed_states.rows());

    A.setFromTriplets(entry_K.begin(), entry_K.end());
    // std::cout << A.rows() << " " << A.cols() << std::endl;
    // std::cout << A << std::endl;
    K = W.transpose() * A * W;
    
    if(!add_penalty && !run_diff_test)
        projectDirichletDoFSystemMatrix(K);

    // std::cout << K << std::endl;
    
    K.makeCompressed();
}

template<class T, int dim>
bool EoLRodSim<T, dim>::projectDirichletDoFSystemMatrix(StiffnessMatrix& A)
{
    iterateDirichletDoF([&](int offset, T target)
    {
        A.row(offset) *= 0.0;
        A.col(offset) *= 0.0;
        A.coeffRef(offset, offset) = 1.0;
    });
}

template<class T, int dim>
bool EoLRodSim<T, dim>::linearSolve(StiffnessMatrix& K, 
    Eigen::Ref<const VectorXT> residual, Eigen::Ref<VectorXT> ddq)
{
    
    StiffnessMatrix I(K.rows(), K.cols());
    I.setIdentity();

    StiffnessMatrix H = K;
    Eigen::SimplicialLLT<StiffnessMatrix> solver;
    // std::cout << H << std::endl;
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
T EoLRodSim<T, dim>::lineSearchNewton(Eigen::Ref<VectorXT> dq, 
        Eigen::Ref<const VectorXT> residual, 
        int line_search_max)
{
    bool verbose = false;
    
    VectorXT ddq(W.cols());
    ddq.setZero();

    StiffnessMatrix K;
    buildSystemDoFMatrix(dq, K);
    bool success = linearSolve(K, residual, ddq);

    if (residual.dot(ddq) < 1e-6)
    {
        // ddq = residual;
        // std::cout << "dx dot -g < 0 " << std::endl;
        
    }
    
    // T norm = ddq.cwiseAbs().maxCoeff();
    T norm = ddq.norm();
    // std::cout << norm << std::endl;
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
            // testGradient(dq);
            // testHessian(dq);
            break;
        }
        alpha *= T(0.5);
        cnt += 1;
        if (cnt > 15)
        {
            // std::cout << "sss" << std::endl;
            // testGradient(dq);
            // testHessian(dq);
            dq = dq_ls;
            break;
            // return 1e16;
        }
        if (cnt == line_search_max) 
            break;
            // return 1e16;
            
    }
    
    for (auto& crossing : rod_crossings)
    {
        Vector<T, 3> new_omega = dq.template segment<3>(crossing->reduced_dof_offset);
        crossing->updateRotation(new_omega);
        dq.template segment<3>(crossing->reduced_dof_offset).setZero();
    }
    
    // std::cout << "#ls: " << cnt << std::endl;
    return norm;   
}

template<class T, int dim>
void EoLRodSim<T, dim>::staticSolve(Eigen::Ref<VectorXT> dq)
{
    int cnt = 0;
    T residual_norm = 1e10, dq_norm = 1e10;
    
    int max_newton_iter = 1000;

    while (true)
    {
        VectorXT residual(W.cols());
        residual.setZero();
        
        computeResidual(residual, dq);

        residual_norm = residual.norm();
        if (verbose)
            std::cout << "residual_norm " << residual_norm << std::endl;
        
        // std::cout << residual << std::endl;
        // std::getchar();

        if (residual_norm < newton_tol)
            break;
        
        dq_norm = lineSearchNewton(dq, residual, 50);

        if(cnt == max_newton_iter)
            break;
        cnt++;
    }
    
    // if (verbose)
        std::cout << "# of newton solve: " << cnt << " exited with |g|: " << residual_norm << "|dq|: " << dq_norm  << std::endl;
}

template<class T, int dim>
void EoLRodSim<T, dim>::checkHessianPD(Eigen::Ref<const VectorXT> dq)
{
    StiffnessMatrix K;
    buildSystemDoFMatrix(dq, K);
    
    Eigen::SimplicialLLT<StiffnessMatrix> solver;
    solver.compute(K);

    Eigen::MatrixXd A_dense = K;
    Eigen::EigenSolver<Eigen::MatrixXd> eigen_solver;

    eigen_solver.compute(A_dense, /* computeEigenvectors = */ false);
    auto eigen_values = eigen_solver.eigenvalues();
    T min_ev = 1e10;
    for (int i = 0; i < K.cols(); i++)
        if (eigen_values[i].real() < min_ev)
            min_ev = eigen_values[i].real();
    std::cout << min_ev << std::endl;
    if (solver.info() == Eigen::NumericalIssue)
        std::cout << "Indefinite Hessian at Equilibrium" << std::endl;
    else
        std::cout << "Definite Hessian at Equilibrium" << std::endl;
}

template<class T, int dim>
void EoLRodSim<T, dim>::advanceOneStep()
{
    // newton_tol = 1e-6;
    n_dof = W.cols();
    VectorXT dq(n_dof);
    dq.setZero();
    
    staticSolve(dq);
    
    VectorXT dq_projected = dq;

    iterateDirichletDoF([&](int offset, T target){
            dq_projected[offset] = target;
        });
    deformed_states = rest_states + W * dq_projected;
    T e_total = 0.0;
    if (add_stretching)
        e_total += addStretchingEnergy();
    if constexpr (dim == 3)
        e_total += add3DBendingAndTwistingEnergy();
    std::cout << "E total: " << e_total << std::endl;

    checkHessianPD(dq);

    for (auto& rod : Rods)
    {
        rod->reference_angles = deformed_states.template segment(rod->theta_dof_start_offset, 
            rod->indices.size() - 1);
        VectorXT reference_twist = rod->reference_twist + rod->reference_angles;
        // std::cout << rod->reference_angles.transpose() << std::endl;
    }
    
}

template<class T, int dim>
void EoLRodSim<T, dim>::convertxXforMaple(
        std::vector<TV>& x, 
        const std::vector<TV>& X,
        Eigen::Ref<const DOFStack> q_temp,
        std::vector<int>& nodes)
{
    int cnt = 0;
    for (int node : nodes)
    {
        x[cnt] = q_temp.col(node).template segment<dim>(0);
        x[cnt + nodes.size()] = X[cnt];
        cnt++;
    }

}

template<class T, int dim>
void EoLRodSim<T, dim>::toMapleNodesVector(std::vector<Vector<T, dim + 1>>& x, Eigen::Ref<const DOFStack> q_temp,
    std::vector<int>& nodes, int yarn_type)
{
    int cnt = 0;
    for (int node : nodes)
    {
        Vector<T, dim + 1> xu;
        xu.template segment<dim>(0) = q_temp.col(node).template segment<dim>(0);
        xu[dim] = q_temp(dim + yarn_type, node);
        x[cnt++] = xu;
    }
}


template<class T, int dim>
void EoLRodSim<T, dim>::getMaterialPositions(
    Eigen::Ref<const DOFStack> q_temp,
    const std::vector<int>& nodes, std::vector<TV>& X, int uv_offset,
    std::vector<TV>& dXdu, std::vector<TV>& d2Xdu2, bool g, bool h)
{
    X.resize(nodes.size(), TV::Zero());
    dXdu.resize(nodes.size(), TV::Zero());
    d2Xdu2.resize(nodes.size(), TV::Zero());

    int cnt = 0;
    for(int node : nodes)
    {
        curvature_functions[uv_offset]->getMaterialPos(q_temp(dim + uv_offset, node), 
            X[cnt], dXdu[cnt], d2Xdu2[cnt], g, h);
        cnt++;
    }
}


template class EoLRodSim<double, 3>;
template class EoLRodSim<double, 2>;