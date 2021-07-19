#include "EoLRodSim.h"


template<class T, int dim>
void EoLRodSim<T, dim>::derivativeTest()
{
    run_diff_test = true;
    add_regularizor = false;
    add_stretching = false;
    add_penalty = false;
    add_bending = false;
    add_shearing = false;
    add_twisting = false;
    add_rigid_joint = true;
    add_pbc = false;
    add_contact_penalty = false;
    add_eularian_reg = false;
    deformed_states /= unit;
    VectorXT dq(W.cols());
    // VectorXT dq(W.rows());
    dq.setRandom();
    dq *= 1.0 / dq.norm();
    for (int i = 0; i < dq.rows(); i++)
        dq(i) = dq(i) * 0.5 + 0.5;
    dq *= 0.01;
    // dq(2) += 1.0;
    // dq(3) += 1.0;
    // testGradient(dq);
    testHessian(dq);
}


template<class T, int dim>
void EoLRodSim<T, dim>::checkMaterialPositionDerivatives()
{
    T epsilon = 1e-6;
    if (new_frame_work)
    {
        for (auto& rod : Rods)
        {
            rod->iterateSegmentsWithOffset([&](int node_i, int node_j, Offset offset_i, Offset offset_j, int rod_idx)
            {
                // std::cout << "node i " << node_i << " node j " << node_j << std::endl;
                TV X1, X0, dX0;
                rod->XdX(node_i, X0, dX0);
                deformed_states[offset_i[dim]] += epsilon;
                rod->X(node_i, X1);
                deformed_states[offset_i[dim]] -= epsilon;
                for (int d = 0; d < dim; d++)
                {
                    std::cout << "dXdu: " << (X1[d] - X0[d]) / epsilon << " " << dX0[d] << std::endl;
                }
                // std::getchar();
            });
        }
    }
    else
    {
        int yarn_type = 0;
        for (int i = 0; i < n_nodes; i++)
        {
            std::vector<int> nodes = { i };
            std::vector<TV> X0, X1; std::vector<TV> dXdu, dXdu1; std::vector<TV> d2Xdu2, dummy2;
            
            getMaterialPositions(q, nodes, X0, yarn_type, dXdu, d2Xdu2, true, true);

            TV x0 = X0[0];
            q(dim, i) += epsilon;
            getMaterialPositions(q, nodes, X1, yarn_type, dXdu1, dummy2, true, false);
            TV x1 = X1[0];

            for (int d = 0; d < dim; d++)
            {
                std::cout << "dXdu: " << (x1[d] - x0[d]) / epsilon << " " << dXdu[0][d] << std::endl;
                std::cout << "d2Xdu2 " << (dXdu1[0][d] - dXdu[0][d]) / epsilon << " " << d2Xdu2[0][d] << std::endl;
                std::getchar();
            }
            q(dim, i) -= epsilon;
        }
    }
}

template<class T, int dim>
void EoLRodSim<T, dim>::testGradient(Eigen::Ref<VectorXT> dq)
{
    run_diff_test = true;
    std::cout << "======================== CHECK GRADIENT ========================" << std::endl;
    T epsilon = 1e-6;
    n_dof = W.cols();
    // n_dof = W.rows();
    
    VectorXT gradient(n_dof);
    gradient.setZero();

    computeResidual(gradient, dq);
    // std::cout << gradient << std::endl;
    VectorXT gradient_FD(n_dof);
    gradient_FD.setZero();
    
    // dq.setZero();

    int cnt = 0;
    for(int dof_i = 0; dof_i < n_dof; dof_i++)
    {
        dq(dof_i) += epsilon;
        // std::cout << W * dq << std::endl;
        T E0 = computeTotalEnergy(dq);
        
        dq(dof_i) -= 1.0 * epsilon;
        T E1 = computeTotalEnergy(dq);
        // std::cout << "E1 " << E1 << " E0 " << E0 << std::endl;
        gradient_FD(dof_i) = (E1 - E0) / (1*epsilon);
        if( gradient_FD(dof_i) == 0 && gradient(dof_i) == 0)
            continue;
        // if (std::abs( gradient_FD(d, n_node) - gradient(d, n_node)) < 1e-4)
        //     continue;
        std::cout << " dof " << dof_i << " " << gradient_FD(dof_i) << " " << gradient(dof_i) << std::endl;
        std::getchar();
        cnt++;   
    }
}

template<class T, int dim>
void EoLRodSim<T, dim>::testHessian(Eigen::Ref<VectorXT> dq)
{
    run_diff_test = true;
    std::cout << "======================== CHECK HESSIAN ========================" << std::endl;
    T epsilon = 1e-6;
    StiffnessMatrix A;
    buildSystemDoFMatrix(dq, A);
    // std::cout << A.coeff(1, 0) << std::endl;
    n_dof = W.cols();
    for(int dof_i = 0; dof_i < n_dof; dof_i++)
    {
        dq(dof_i) += epsilon;
        VectorXT g0(n_dof), g1(n_dof);
        g0.setZero(); g1.setZero();
        computeResidual(g0, dq);

        dq(dof_i) -= 1.0 * epsilon;
        computeResidual(g1, dq);
            
        VectorXT row_FD = (g1 - g0) / (epsilon);

        for(int i = 0; i < n_dof; i++)
        {
            if(A.coeff(dof_i, i) == 0 && row_FD(i) == 0)
                continue;
            // if (std::abs( A.coeff(dof_i, i) - row_FD(i)) < 1e-4)
            //     continue;
            // std::cout << "node i: "  << std::floor(dof_i / T(dof)) << " dof " << dof_i%dof 
            //     << " node j: " << std::floor(i / T(dof)) << " dof " << i%dof 
            //     << " FD: " <<  row_FD(i) << " symbolic: " << A.coeff(i, dof_i) << std::endl;
            std::cout << "H(" << i << ", " << dof_i << ") " << " FD: " <<  row_FD(i) << " symbolic: " << A.coeff(i, dof_i) << std::endl;
            std::getchar();
        }
    }
}


template class EoLRodSim<double, 3>;
template class EoLRodSim<double, 2>;