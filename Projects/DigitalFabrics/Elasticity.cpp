#include "EoLRodSim.h"

template<class T, int dim>
void EoLRodSim<T, dim>::setUniaxialStrain(T theta, T s, TV& strain_dir)
{
    
    pbc_strain_data.clear();
    pbc_strain_data.resize(0);
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
            // std::cout << Dij << " " << dij << std::endl;
            pbc_strain_data.push_back(std::make_pair(IV2(node_i, node_j), std::make_pair(strain_dir, dij)));
        }
    });
}

template<class T, int dim>
void EoLRodSim<T, dim>::computeMacroStress(TM& sigma, TV strain_dir)
{
    bool COUT_ALL = false;

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
    
    // if (COUT_ALL)
        // std::cout << "Fd dot d: " << strain_dir.dot((F_macro * strain_dir).normalized()) << std::endl;

    TM strain_marco = 0.5 * (F_macro.transpose() + F_macro) - TM::Identity();

    if (COUT_ALL)
        std::cout << "macro green strain " << std::endl << strain_marco << std::endl;
    

    TM R90 = TM::Zero();
    if constexpr (dim == 2)
    {
        R90.row(0) = TV(0, -1);
        R90.row(1) = TV(1, 0);
    }

    TV n0 = (R90 * (xj - xi)).normalized(), n1 = (R90 * (xl - xk)).normalized();

    // std::cout << n0.dot(xj - xi) << " " << n1.dot(xl - xk) << std::endl;

    DOFStack f(dof, n_nodes), zero_delta(dof, n_nodes);
    f.setZero(); zero_delta.setZero();
    add_pbc = false;
    computeResidual(f, zero_delta);
    add_pbc = true;

    // DOFStack temp(dof, n_nodes);
    // temp.setZero();
    // addStretchingForce(q, temp);
    // std::cout << "------------------------------- stretching -------------------------------" << std::endl;
    // std::cout << temp.transpose() << std::endl;
    // std::cout << "------------------------------- stretching -------------------------------" << std::endl;
    // std::cout << "stretching norm: " << temp.norm() << std::endl;
    // temp.setZero();
    // addShearingForce(q, temp, true);
    // addShearingForce(q, temp, false);
    // std::cout << "------------------------------- shearing -------------------------------" << std::endl;
    // std::cout << temp.transpose() << std::endl;
    // std::cout << "------------------------------- shearing -------------------------------" << std::endl;
    // std::cout << "shearing norm: " << temp.norm() << std::endl;
    
    



    if(COUT_ALL)
    {
        std::cout << "------------------------------- f -------------------------------" << std::endl;
        std::cout << f.transpose() << std::endl;
        std::cout << "------------------------------- f -------------------------------" << std::endl;
    }
    
    std::vector<TV> f_bc(2, TV::Zero());

    iteratePBCReferencePairs([&](int dir_id, int node_i, int node_j){
        T length = dir_id == 1 ? (xj - xi).norm() : (xl - xk).norm();
        int bc_node = dir_id == 0 ? node_j : node_i;
        f_bc[dir_id].template segment<dim>(0) += f.col(bc_node).template segment<dim>(0) / length;
        // std::cout << "node j "<< node_j << " " << f.col(node_j).template segment<dim>(0).transpose() << std::endl;
    });

    TM F_bc = TM::Zero(), n_bc = TM::Zero();
    F_bc.col(0) = f_bc[0]; F_bc.col(1) = f_bc[1];
    n_bc.col(0) = n1; n_bc.col(1) = n0;

    auto stress_marco = F_bc * n_bc.inverse();

    if (COUT_ALL)
    {    
        std::cout << "stress: " << std::endl;
        std::cout << stress_marco << std::endl;
    }
    // sigma = stress_marco;
    if (COUT_ALL)
        std::cout << "strain_dir" << strain_dir.transpose() << " " << (stress_marco * strain_dir).transpose() << std::endl;

    sigma = stress_marco.template block<dim, dim>(0, 0);
}

template<class T, int dim>
void EoLRodSim<T, dim>::computeDeformationGradientUnitCell()
{
    IV2 ref0 = pbc_ref_unique[0];
    IV2 ref1 = pbc_ref_unique[1];
    
    TM x, X;
    X.col(0) = q0.col(ref0[1]).template segment<dim>(0) - q0.col(ref0[0]).template segment<dim>(0);
    X.col(1) = q0.col(ref1[1]).template segment<dim>(0) - q0.col(ref1[0]).template segment<dim>(0);
    x.col(0) = q.col(ref0[1]).template segment<dim>(0) - q.col(ref0[0]).template segment<dim>(0);
    x.col(1) = q.col(ref1[1]).template segment<dim>(0) - q.col(ref1[0]).template segment<dim>(0);

    TM F = x * X.inverse();
    
    std::cout << "F from ref pairs: " << F << std::endl;
    // std::cout << "F-FT: " << F - F.transpose() << std::endl;
}


// min_F 1/2||F(Xi-Xj) - (xi-xj)||^2
// least square deformation fitting
template<class T, int dim>
void EoLRodSim<T, dim>::fitDeformationGradientUnitCell()
{
    auto computeEnergy = [&](TM _F){
        VectorXT energy(n_rods);
        energy.setZero();
        tbb::parallel_for(0, n_rods, [&](int i){
            TV xi = q.col(rods(0, i)).template segment<dim>(0);
            TV xj = q.col(rods(1, i)).template segment<dim>(0);
            TV Xi = q0.col(rods(0, i)).template segment<dim>(0);
            TV Xj = q0.col(rods(1, i)).template segment<dim>(0);
            energy[i] += 0.5 * (_F * (Xi - Xj) - (xi - xj)).squaredNorm();
        });
        return energy.sum();
    };

    auto computeGradient = [&](TM _F){
        TM dedF = TM::Zero();
        for (int i = 0; i < n_rods; i++)
        {
            TV xi = q.col(rods(0, i)).template segment<dim>(0);
            TV xj = q.col(rods(1, i)).template segment<dim>(0);
            TV Xi = q0.col(rods(0, i)).template segment<dim>(0);
            TV Xj = q0.col(rods(1, i)).template segment<dim>(0);
            dedF += -(_F * (Xi - Xj) - (xi - xj)) * (Xi - Xj).transpose();
        }
        return dedF;
    };


    auto polarDecomposition = [&](TM F, TM& R, TM& RU)
    {
        Eigen::JacobiSVD<TM> svd;
        svd.compute(F, Eigen::ComputeFullU | Eigen::ComputeFullV );
        TM U = svd.matrixU();
        TM V = svd.matrixV();
        TV S = svd.singularValues();
        R = U*V.transpose();
        const auto& SVT = S.asDiagonal() * V.adjoint();
        // from libigl
        if(R.determinant() < 0)
        {
            auto W = V.eval();
            W.col(V.cols()-1) *= -1.;
            R = U*W.transpose();
            RU = W*SVT;
        }
        else
            RU = V*SVT;      
    };


    TM F = TM::Identity();

    while (true)
    {
        TM dedF = computeGradient(F);
        if (dedF.norm() < 1e-5)
            break;
        T E0 = computeEnergy(F);
        T alpha = 1.0;
        while (true)
        {
            TM F_ls = F + alpha * dedF;
            T E1 = computeEnergy(F_ls);
            if (E1 - E0 < 0)
            {
                F = F_ls;
                break;
            }
            alpha *= 0.5;
        }
    }

    TM R, RU;
    polarDecomposition(F, R, RU);

    std::cout << "F fited " <<  F << std::endl;
    // std::cout << "F-F^T: " << F - F.transpose() << std::endl;
    // std::cout << "R: " << R << std::endl;
}

template class EoLRodSim<double, 3>;
template class EoLRodSim<double, 2>;