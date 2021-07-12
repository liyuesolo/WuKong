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
   
    T total_energy = 0;
    T E_stretching = 0, E_bending = 0, E_shearing = 0, E_twisting = 0, E_bending_twisting = 0,
        E_eul_reg = 0, E_pbc = 0, E_penalty = 0, E_contact = 0;
    
    if (add_stretching)
        E_stretching += addStretchingEnergy();
    if constexpr (dim == 3)
    {
        if (add_bending && add_twisting)
            E_bending_twisting = add3DBendingAndTwistingEnergy();
    }
    else if constexpr (dim == 2)
    {
        if (add_bending)
            E_bending += addBendingEnergy();
        if (add_twisting)
            E_twisting += addTwistingEnergy();
    }
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
    }
    VectorXT full_residual(deformed_states.rows());
    full_residual.setZero();

    if (add_stretching)
        addStretchingForce(full_residual);
    if constexpr (dim == 3)
    {
        if (add_bending && add_twisting)
            add3DBendingAndTwistingForce(full_residual);
    }
    else if constexpr (dim == 2)
    {
        if (add_bending)
            addBendingForce(full_residual);
        if (add_twisting)
            addTwistingForce(full_residual);
    }
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
    if (add_stretching)
        addStretchingK(entry_K);
    if constexpr (dim == 3)
    {
        if (add_bending && add_twisting)
            add3DBendingAndTwistingK(entry_K);
    }
    else if constexpr (dim == 2)
    {
        if (add_bending)
            addBendingK(entry_K);
        if (add_twisting)
            addTwistingK(entry_K);
    }
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
            // testGradient(dq);
            // testHessian(dq);
            dq = dq_ls;
            return 1e16;
        }
        if (cnt == line_search_max) 
            return 1e16;
            
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
    else if constexpr (dim == 2)
        e_total += addBendingEnergy();
    std::cout << "E total: " << e_total << std::endl;

    for (auto& rod : Rods)
    {
        rod->reference_angles = deformed_states.template segment(rod->theta_dof_start_offset, 
            rod->indices.size() - 1);
        std::cout << rod->reference_angles.transpose() << std::endl;
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