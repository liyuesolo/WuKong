#include "EoLRodSim.h"

template<class T, int dim>
void EoLRodSim<T, dim>::setUniaxialStrain(T theta, T s, TV& strain_dir)
{
    pbc_strain_data.clear();
    strain_dir = TV::Zero();
    if constexpr (dim == 2)
        strain_dir = TV(std::cos(theta), std::sin(theta));
    iteratePBCReferencePairs([&](int dir_id, int node_i, int node_j){
        TV Xj = q0.col(node_j).template segment<dim>(0);
        TV Xi = q0.col(node_i).template segment<dim>(0);

        if constexpr (dim == 2)
        {
            T Dij = (Xj - Xi).dot(strain_dir);
            T dij = Dij * s;
            pbc_strain_data.push_back(std::make_pair(IV2(node_i, node_j), std::make_pair(strain_dir, dij)));
        }
    });
}

template<class T, int dim>
void EoLRodSim<T, dim>::computeMacroStress(TM& sigma)
{
    TV xj = q.col(pbc_ref_unique[0](1)).template segment<dim>(0);
    TV xi = q.col(pbc_ref_unique[0](0)).template segment<dim>(0);
    TV xl = q.col(pbc_ref_unique[1](1)).template segment<dim>(0);
    TV xk = q.col(pbc_ref_unique[1](0)).template segment<dim>(0);

    TV Xj = q0.col(pbc_ref_unique[0](1)).template segment<dim>(0);
    TV Xi = q0.col(pbc_ref_unique[0](0)).template segment<dim>(0);
    TV Xl = q0.col(pbc_ref_unique[1](1)).template segment<dim>(0);
    TV Xk = q0.col(pbc_ref_unique[1](0)).template segment<dim>(0);

    TM X = TM::Zero(), x = TM::Zero();
    X.col(0) = Xi - Xj;
    X.col(1) = Xk - Xl;

    x.col(0) = xi - xj;
    x.col(1) = xk - xl;

    TM F_macro = x * X.inverse();

    TM strain_marco = 0.5 * (F_macro.transpose() * F_macro) - TM::Identity();
    // std::cout << strain_marco << std::endl;
    

    TM R90 = TM::Zero();
    if constexpr (dim == 2)
    {
        R90.row(0) = TV(0, -1);
        R90.row(1) = TV(1, 0);
    }

    TV n0 = (R90 * (xj - xi)).normalized(), n1 = (R90 * (xl - xk)).normalized();

    DOFStack f(dof, n_nodes);
    f.setZero();
    computeResidual(f, q-q0);
    std::vector<TV> f_bc(2, TV::Zero());

    iteratePBCReferencePairs([&](int dir_id, int node_i, int node_j){
        f_bc[dir_id] -= f.col(node_j).template segment<dim>(0);
    });

    TM F_bc = TM::Zero(), n_bc = TM::Zero();
    F_bc.col(0) = f_bc[0]; F_bc.col(1) = f_bc[1];
    n_bc.col(0) = n1; n_bc.col(1) = n0;

    TM stress_marco = F_bc * n_bc.inverse();
    // std::cout << n_bc << std::endl;
    // std::cout << stress_marco << std::endl;
    sigma = stress_marco;
}

template class EoLRodSim<double, 3>;
template class EoLRodSim<double, 2>;