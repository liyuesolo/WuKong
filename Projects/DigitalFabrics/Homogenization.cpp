#include "Homogenization.h"

template<class T, int dim>
void Homogenization<T, dim>::initalizeSim()
{
    
    sim.buildPlanePeriodicBCScene3x3Subnodes(8);
    // sim.buildPlanePeriodicBCScene3x3();
    TV strain_dir;
    // sim.setUniaxialStrain(0.298451, 1.1, strain_dir);
    // sim.setUniaxialStrain(M_PI/4 + 0.1, 1.1, strain_dir);
        
    // sim.setUniaxialStrain(6.04757, 1.1, strain_dir1);
    // sim.setBiaxialStrain(6.04757, 1.1, strain_dir1, 6.04757, 0.9915, strain_dir2);
    sim.setBiaxialStrain(M_PI/4 + 0.1, 1.1, M_PI/4 + 0.1, 1.0);
    // sim.setUniaxialStrain(6.06327, 1.1, strain_dir);
    // sim.setUniaxialStrain(0.0, 1.1, strain_dir);

    // sim.setUniaxialStrain(M_PI/4, 1.1, strain_dir);
    sim.advanceOneStep();
    TM stress, strain;
    computeMacroStressStrain(stress, strain);
    std::cout << stress << std::endl << strain << std::endl;

    // strain_dir.normalize(); 
    // TM sigma;
    // sim.computeMacroStress(sigma, strain_dir);
    // T youngs_modulus = strain_dir.dot(sigma * strain_dir);
    // std::cout << youngs_modulus << std::endl;
}

template<class T, int dim>
void Homogenization<T, dim>::computeMacroStressStrain(TM& stress_marco, TM& strain_marco)
{
    TV xj = sim.q.col(sim.pbc_ref_unique[0](1)).template segment<dim>(0);
    TV xi = sim.q.col(sim.pbc_ref_unique[0](0)).template segment<dim>(0);
    TV xl = sim.q.col(sim.pbc_ref_unique[1](1)).template segment<dim>(0);
    TV xk = sim.q.col(sim.pbc_ref_unique[1](0)).template segment<dim>(0);

    TV Xj = sim.q0.col(sim.pbc_ref_unique[0](1)).template segment<dim>(0);
    TV Xi = sim.q0.col(sim.pbc_ref_unique[0](0)).template segment<dim>(0);
    TV Xl = sim.q0.col(sim.pbc_ref_unique[1](1)).template segment<dim>(0);
    TV Xk = sim.q0.col(sim.pbc_ref_unique[1](0)).template segment<dim>(0);

    TM X = TM::Zero(), x = TM::Zero();
    X.col(0) = Xi - Xj;
    X.col(1) = Xk - Xl;

    x.col(0) = xi - xj;
    x.col(1) = xk - xl;

    TM F_macro = x * X.inverse();
    
    strain_marco = 0.5 * (F_macro.transpose() + F_macro) - TM::Identity();

    TM R90 = TM::Zero();
    if constexpr (dim == 2)
    {
        R90.row(0) = TV(0, -1);
        R90.row(1) = TV(1, 0);
    }

    TV n0 = (R90 * (xj - xi)).normalized(), n1 = (R90 * (xl - xk)).normalized();
    

    DOFStack f(sim.dof, sim.n_nodes);
    f.setZero(); 
    sim.addPBCForce(sim.q, f);
    
    std::vector<TV> f_bc(2, TV::Zero());

    sim.iteratePBCReferencePairs([&](int dir_id, int node_i, int node_j){
        T length = dir_id == 1 ? (xj - xi).norm() : (xl - xk).norm();
        length *= sim.unit;
        int bc_node = dir_id == 0 ? node_j : node_i;
        f_bc[dir_id].template segment<dim>(0) += f.col(bc_node).template segment<dim>(0) / length;
        
    });

    TM F_bc = TM::Zero(), n_bc = TM::Zero();
    F_bc.col(0) = f_bc[0]; F_bc.col(1) = f_bc[1];
    n_bc.col(0) = n1; n_bc.col(1) = n0;

    stress_marco = F_bc * n_bc.inverse();
}
template<class T, int dim>
void Homogenization<T, dim>::computeMacroStress(TM& sigma, TV strain_dir)
{
    bool COUT_ALL = false;

    TV xj = sim.q.col(sim.pbc_ref_unique[0](1)).template segment<dim>(0);
    TV xi = sim.q.col(sim.pbc_ref_unique[0](0)).template segment<dim>(0);
    TV xl = sim.q.col(sim.pbc_ref_unique[1](1)).template segment<dim>(0);
    TV xk = sim.q.col(sim.pbc_ref_unique[1](0)).template segment<dim>(0);

    TV Xj = sim.q0.col(sim.pbc_ref_unique[0](1)).template segment<dim>(0);
    TV Xi = sim.q0.col(sim.pbc_ref_unique[0](0)).template segment<dim>(0);
    TV Xl = sim.q0.col(sim.pbc_ref_unique[1](1)).template segment<dim>(0);
    TV Xk = sim.q0.col(sim.pbc_ref_unique[1](0)).template segment<dim>(0);

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

    DOFStack f(sim.dof, sim.n_nodes);
    f.setZero(); 
    sim.addPBCForce(sim.q, f);


    if(COUT_ALL)
    {
        std::cout << "------------------------------- f -------------------------------" << std::endl;
        std::cout << f.transpose() << std::endl;
        std::cout << "------------------------------- f -------------------------------" << std::endl;
    }
    
    std::vector<TV> f_bc(2, TV::Zero());

    sim.iteratePBCReferencePairs([&](int dir_id, int node_i, int node_j){
        T length = dir_id == 1 ? (xj - xi).norm() : (xl - xk).norm();
        length *= sim.unit;
        int bc_node = dir_id == 0 ? node_j : node_i;
        f_bc[dir_id].template segment<dim>(0) += f.col(bc_node).template segment<dim>(0) / length;
        // std::cout << "node j "<< node_j << " " << f.col(node_j).template segment<dim>(0).transpose() << std::endl;
    });
    
    // if(f_bc[0].dot(n0) < 0)
    // {
    //     f_bc[0] = -f_bc[0];
    // }

    TM F_bc = TM::Zero(), n_bc = TM::Zero();
    F_bc.col(0) = f_bc[0]; F_bc.col(1) = f_bc[1];
    n_bc.col(0) = n1; n_bc.col(1) = n0;

    auto stress_marco = F_bc * n_bc.inverse();

    // if (COUT_ALL)
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
void Homogenization<T, dim>::marcoYoungsModulusFitting()
{
    
    sim.buildPlanePeriodicBCScene3x3Subnodes(8);
    T s = 1.1;
    int n_angles = 400;
    T cycle = 2. * M_PI;
    // T cycle = M_PI / 4.0;
    std::vector<T> thetas, youngs_moduli;
    for (T theta = 0; theta <= cycle; theta += cycle/(T)n_angles)
    {
        thetas.push_back(theta);
        T theta_6 = (int)(theta * 1e6)/T(1e6);
        std::cout << theta_6 << std::endl;
        T youngs_modulus = YoungsModulusFromUniaxialStrain(theta_6, s);
        std::cout << "theta: " << theta << " youngs_modulus " << youngs_modulus << std::endl;
        youngs_moduli.push_back(youngs_modulus);
    }
    for(T theta : thetas)
        std::cout << theta << " ";
    std::cout << std::endl;
    for(T youngs_modulus : youngs_moduli)
        std::cout << youngs_modulus << " ";
    std::cout << std::endl;
}

template<class T, int dim>
T Homogenization<T, dim>::YoungsModulusFromUniaxialStrain(T theta, T s)
{
    TV strain_dir, strain_dir_orth;
    sim.setUniaxialStrain(theta, s, strain_dir);
    // sim.setBiaxialStrain(theta, s, strain_dir, theta, 0.992, strain_dir_orth);
    sim.advanceOneStep();
    strain_dir.normalize(); 
    TM sigma;
    computeMacroStress(sigma, strain_dir);
    T youngs_modulus = strain_dir.dot(sigma * strain_dir);
    sim.resetScene();
    return youngs_modulus;
}


// template<class T, int dim>
// void Homogenization<T, dim>::fitComplianceTensor()
// {
//     sim.buildPlanePeriodicBCScene3x3Subnodes(8);
//     sim.setVerbose(false);
//     /*
//     compliance tensor should be dim x dim by dim x dim
//     */

//     // A : C: A
//     //sum_ijkl Aij Cijkl Akl
//     ComplianceTensor S;
//     S.setIdentity();
//     int n_angles = 40;
//     T s1 = 1.05, s2 = 1.0;

//     std::vector<TVEntry> strain_entries, stress_entries;

//     auto gatherData = [&, s1, s2, n_angles]()
//     {
//         for (T theta = 0; theta <= 2.0 * M_PI; theta += 2.0 * M_PI /(T)n_angles)
//         {
//             sim.setBiaxialStrain(theta, s1, theta, s2);
//             sim.advanceOneStep();
//             TM stress, strain;
//             computeMacroStressStrain(stress, strain);
//             // std::cout << "strain " << std::endl;
//             // std::cout << strain << std::endl;
//             // std::cout << "stress " << std::endl;
//             // std::cout << stress << std::endl;
//             if constexpr (dim == 2)
//             {
//                 TVEntry se0, se1, se2;
//                 se0[0] = stress(0, 0); se0[1] = stress(1, 1); se0[2] = stress(0, 1);
//                 se1[0] = strain(0, 0); se1[1] = strain(1, 1); se1[2] = 2.0 * strain(0, 1);

//                 strain_entries.push_back(se1);
//                 stress_entries.push_back(se0);
//             }
//             sim.resetScene();
//         }
//     };

//     auto computeEnergy = [&, n_angles](ComplianceTensor& S)
//     {
//         T energy = 0.0;
//         for (int i = 0; i < n_angles; i++)
//         {
//             energy += 0.5 * (S * stress_entries[i] - strain_entries[i]).squaredNorm() / strain_entries[i].squaredNorm();
//         }
//         energy /= T(stress_entries.size());
//         return energy;
//     };

//     auto computeGradient = [&, n_angles](ComplianceTensor& S)
//     {
//         ComplianceTensor gradient;
//         gradient.setZero();
//         for (int i = 0; i < n_angles; i++)
//         {
//             gradient -= (S * stress_entries[i] - strain_entries[i]) * stress_entries[i].transpose() / strain_entries[i].squaredNorm();
//         }
//         gradient /= T(stress_entries.size());
//         return gradient;
//     };

//     auto leastSquareFit = [&]()
//     {
//         while (true)
//         {
//             ComplianceTensor gradient = computeGradient(S);
//             if (gradient.norm() < 1e-4)
//                 break;
//             // std::cout << gradient.norm() << std::endl;
//             T E0 = computeEnergy(S);
//             // std::cout << "E0: " << E0 << std::endl;
//             T alpha = 1.0;
//             // std::getchar();
//             while(true)
//             {
//                 ComplianceTensor S_ls = S + alpha * gradient;
//                 T E1 = computeEnergy(S_ls);
//                 // std::cout << "E1 " << E1 << std::endl;
//                 // std::getchar();
//                 if (E1 - E0 < 0)
//                 {
//                     S = S_ls;
//                     break;
//                 }
//                 alpha *= 0.5;
//             }
//         }
//         // std::cout << S << std::endl;
//         return S.inverse();
//     };

//     auto checkGradientFD = [&]()
//     {
//         T epsilon = 1e-4;
//         ComplianceTensor gradient = computeGradient(S);
//         for (int i = 0; i < S.rows(); i++)
//             for (int j = 0; j < S.cols(); j++)
//             {
//                 S(i, j) += epsilon;
//                 T E0 = computeEnergy(S);
//                 S(i, j) -= 2.0 * epsilon;
//                 T E1 = computeEnergy(S);
//                 S(i, j) += epsilon;

//                 T gradient_FD = (E1 - E0) / (2.0 * epsilon);

//                 if (gradient_FD == 0.0 && gradient(i, j) == 0.0)
//                     continue;
//                 std::cout << gradient_FD << " " << gradient(i, j) << std::endl;
//                 std::getchar();
//             }
//     };
    
//     gatherData();
//     // checkGradientFD();
//     ComplianceTensor C = leastSquareFit();
//     std::cout << "C: " << std::endl << C << std::endl;
// }

template<class T, int dim>
void Homogenization<T, dim>::fitComplianceTensor()
{
    sim.buildPlanePeriodicBCScene3x3Subnodes(8);
    sim.setVerbose(false);
    /*
    compliance tensor should be dim x dim by dim x dim
    */

    // A : C: A
    //sum_ijkl Aij Cijkl Akl
    ComplianceTensor S;
    // S.setOnes();
    S.setIdentity();
    int n_angles = 40;
    T s1 = 1.05, s2 = 1.0;

    std::vector<TVEntry> strain_entries, stress_entries;

    auto gatherData = [&, s1, s2, n_angles]()
    {
        for (T theta = 0; theta <= 2.0 * M_PI; theta += 2.0 * M_PI /(T)n_angles)
        {
            sim.setBiaxialStrain(theta, s1, theta, s2);
            sim.advanceOneStep();
            TM stress, strain;
            computeMacroStressStrain(stress, strain);
            // std::cout << "strain " << std::endl;
            // std::cout << strain << std::endl;
            // std::cout << "stress " << std::endl;
            // std::cout << stress << std::endl;
            if constexpr (dim == 2)
            {
                TVEntry se0, se1, se2;
                se0[0] = stress(0, 0); se0[1] = stress(1, 1); se0[2] = stress(1, 0);
                se1[0] = strain(0, 0); se1[1] = strain(1, 1); se1[2] = 2.0 * strain(1, 0);

                strain_entries.push_back(se1);
                stress_entries.push_back(se0);
            }
            sim.resetScene();
        }
    };

    auto computeEnergy = [&, n_angles](ComplianceTensor& S)
    {
        T energy = 0.0;
        for (int i = 0; i < n_angles; i++)
        {
            energy += 0.5 * (S * stress_entries[i] - strain_entries[i]).squaredNorm() / strain_entries[i].squaredNorm();
        }
        return energy;
    };

    auto computeGradient = [&, n_angles](ComplianceTensor& S)
    {
        ComplianceTensor gradient;
        gradient.setZero();
        for (int i = 0; i < n_angles; i++)
        {
            gradient -= (S * stress_entries[i] - strain_entries[i]) * stress_entries[i].transpose() / strain_entries[i].squaredNorm();
        }
        return gradient;
    };

    auto leastSquareFit = [&]()
    {
        while (true)
        {
            ComplianceTensor gradient = computeGradient(S);
            if (gradient.norm() < 1e-4)
                break;
            std::cout << gradient.norm() << std::endl;
            T E0 = computeEnergy(S);
            // std::cout << "E0: " << E0 << std::endl;
            T alpha = 1.0;
            // std::getchar();
            while(true)
            {
                ComplianceTensor S_ls = S + alpha * gradient;
                T E1 = computeEnergy(S_ls);
                // std::cout << "E1 " << E1 << std::endl;
                // std::getchar();
                if (E1 - E0 < 0)
                {
                    S = S_ls;
                    break;
                }
                alpha *= 0.5;
            }
        }
        // std::cout << S << std::endl;
        return S.inverse();
    };

    auto checkGradientFD = [&]()
    {
        T epsilon = 1e-4;
        ComplianceTensor gradient = computeGradient(S);
        for (int i = 0; i < S.rows(); i++)
            for (int j = 0; j < S.cols(); j++)
            {
                S(i, j) += epsilon;
                T E0 = computeEnergy(S);
                S(i, j) -= 2.0 * epsilon;
                T E1 = computeEnergy(S);
                S(i, j) += epsilon;

                T gradient_FD = (E1 - E0) / (2.0 * epsilon);

                if (gradient_FD == 0.0 && gradient(i, j) == 0.0)
                    continue;
                std::cout << gradient_FD << " " << gradient(i, j) << std::endl;
                std::getchar();
            }
    };
    
    gatherData();
    std::cout << "======== Simulation Data Generation Done ========" << std::endl;
    // checkGradientFD();
    ComplianceTensor C = leastSquareFit();
    std::cout << "C: " << std::endl << C << std::endl;

    // std::vector<T> thetas, youngs_moduli;
    // for (T theta = 0; theta <= 2.0 * M_PI; theta += 2.0 * M_PI /(T)n_angles)
    // {
    //     thetas.push_back(theta);
    //     T theta_6 = (int)(theta * 1e6)/T(1e6);
    //     TV d(std::cos(theta), std::sin(theta));
    //     TM ddT = d*d.transpose();
    //     const auto& ddT_vec = Eigen::Map<const TVEntry>(ddT.data(), ddT.size());
    //     T youngs_modulus = 1.0 / ddT_vec.dot(C * ddT_vec);
    //     youngs_moduli.push_back(youngs_modulus);
    // }
    // for(T youngs_modulus : youngs_moduli)
    //     std::cout << youngs_modulus << " ";
    // std::cout << std::endl;
}

// template class Homogenization<double, 3>;
template class Homogenization<double, 2>;