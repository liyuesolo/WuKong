#include "Homogenization.h"

template<class T, int dim>
void Homogenization<T, dim>::initalizeSim()
{
    
    sim.buildPlanePeriodicBCScene3x3Subnodes(8);
    // sim.buildPlanePeriodicBCScene3x3();
    TV strain_dir;
    // sim.setUniaxialStrain(0.298451, 1.1, strain_dir);
    // sim.setUniaxialStrain(M_PI/4 + 0.1, 1.1, strain_dir);
        
    sim.setUniaxialStrain(6.04757, 1.05, strain_dir);
    // sim.setBiaxialStrain(6.04757, 1.1, strain_dir1, 6.04757, 0.9915, strain_dir2);
    // sim.setBiaxialStrain(M_PI/4 + 0.1, 1.1, M_PI/4 + 0.1, 1.0);
    // sim.setUniaxialStrain(6.06327, 1.1, strain_dir);
    // sim.setUniaxialStrain(0.0, 1.1, strain_dir);

    // sim.setUniaxialStrain(M_PI/4, 1.1, strain_dir);
    // sim.advanceOneStep();
    // TM stress, strain;
    // computeMacroStressStrain(stress, strain);
    // std::cout << stress << std::endl << strain << std::endl;

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
    f *= -1;
    std::vector<TV> f_bc(2, TV::Zero());

    sim.iteratePBCReferencePairs([&](int dir_id, int node_i, int node_j){
        T length = dir_id == 1 ? (xj - xi).norm() : (xl - xk).norm();
        length /= sim.unit;
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
    f *= -1;

    if(COUT_ALL)
    {
        std::cout << "------------------------------- f -------------------------------" << std::endl;
        std::cout << f.transpose() << std::endl;
        std::cout << "------------------------------- f -------------------------------" << std::endl;
    }
    
    std::vector<TV> f_bc(2, TV::Zero());

    sim.iteratePBCReferencePairs([&](int dir_id, int node_i, int node_j){
        T length = dir_id == 1 ? (xj - xi).norm() : (xl - xk).norm();
        length /= sim.unit;
        int bc_node = dir_id == 0 ? node_j : node_i;
        f_bc[dir_id].template segment<dim>(0) += f.col(bc_node).template segment<dim>(0) / length; 
    });

    TM F_bc = TM::Zero(), n_bc = TM::Zero();
    F_bc.col(0) = f_bc[0]; F_bc.col(1) = f_bc[1];
    n_bc.col(0) = n1; n_bc.col(1) = n0;

    sigma = F_bc * n_bc.inverse();
    sigma /= strain_marco.norm();
}

template<class T, int dim>
void Homogenization<T, dim>::marcoYoungsModulusFitting()
{
    // sim.unit = 1.0;
    sim.buildPlanePeriodicBCScene3x3Subnodes(8);
    // sim.k_pbc = 1e5;
    T s = 1.001;
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
    // sim.setBiaxialStrain(theta, s, theta, 0.992);
    sim.advanceOneStep();
    
    TM sigma, epsilon;
    computeMacroStress(sigma, strain_dir);
    // computeMacroStressStrain(sigma, epsilon);
    T youngs_modulus = strain_dir.dot(sigma * strain_dir);

    sim.resetScene();
    // return youngs_modulus / (s-1.0);
    return youngs_modulus;
}


template<class T, int dim>
void Homogenization<T, dim>::fitComplianceTensor()
{
    // sim.unit = 1.0;
    sim.buildPlanePeriodicBCScene3x3Subnodes(8);
    // sim.k_pbc = 1e5;
    // sim.k_pbc=1e5;
    sim.setVerbose(false);
    /*
    compliance tensor should be dim x dim by dim x dim
    */

    // A : C: A
    //sum_ijkl Aij Cijkl Akl

    CDoF2D S_entry;
    S_entry.setZero();
    ComplianceTensor S;
    S.setOnes();
    // S.setZero();
    int n_angles = 400;
    T s1 = 1.001, s2 = 1.0;

    std::vector<TVEntry> strain_entries, stress_entries;

    auto gatherData = [&, s1, s2, n_angles]()
    {
        for (T theta = 0; theta <= 2.0 * M_PI; theta += 2.0 * M_PI /(T)n_angles)
        {
            sim.setBiaxialStrain(theta, s1, theta, s2);
            TV strain_dir;
            // sim.setUniaxialStrain(theta, s1, strain_dir);
            sim.advanceOneStep();
            TM stress, strain;
            computeMacroStressStrain(stress, strain);

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

    auto computeEnergy = [&, n_angles](CDoF2D& S_entry)
    {
        T energy = 0.0;
        for (int i = 0; i < n_angles; i++)
        {
            // energy += 0.5 * (S * stress_entries[i] - strain_entries[i]).squaredNorm() / strain_entries[i].squaredNorm();
            auto x = stress_entries[i];
            auto b = strain_entries[i];
            T V[1];
            #include "Maple/LSV.mcg"
            energy += V[0];
        }
        return energy;
    };

    auto computeGradient = [&, n_angles](CDoF2D& S_entry)
    {
        CDoF2D gradient;
        gradient.setZero();
        for (int i = 0; i < n_angles; i++)
        {
            CDoF2D F;
            auto x = stress_entries[i];
            auto b = strain_entries[i];
            #include "Maple/LSF.mcg"
            gradient += F;
        }
        return gradient;
    };


    auto computeHessian = [&, n_angles]()
    {
        CHessian2D H;
        H.setZero();
        for (int a = 0; a < n_angles; a++)
        {
            auto x = stress_entries[a];
            auto b = strain_entries[a];
            T J[6][6];
            memset(J, 0, sizeof(J));
            #include "Maple/LSJ.mcg"
            for(int i = 0; i< 6; i++)
                for(int j = 0; j< 6; j++)
                    H(i, j) += -J[i][j];
            // H += (stress_entries[i] * stress_entries[i].transpose()).transpose() / strain_entries[i].squaredNorm();
        }
        return H;
    };
    

    auto leastSquareFit = [&]()
    {
        CDoF2D gradient = computeGradient(S_entry);
        
        CHessian2D H = computeHessian();
        CDoF2D dx = H.colPivHouseholderQr().solve(gradient);
        
        S_entry += dx;

        S(0, 0) = S_entry[0]; S(0, 1) = S_entry[1]; S(0, 2) = S_entry[2];
        S(1, 0) = S_entry[1]; S(1, 1) = S_entry[3]; S(1, 2) = S_entry[4];
        S(2, 0) = S_entry[2]; S(2, 1) = S_entry[4]; S(2, 2) = S_entry[5];

        std::cout << S_entry << std::endl;
    };
    
    gatherData();
    std::cout << "======== Simulation Data Generation Done ========" << std::endl;

    leastSquareFit();
    std::cout << "S" << std::endl;
    std::cout << S << std::endl;
    std::vector<T> thetas, youngs_moduli;
    for (T theta = 0; theta <= 2.0 * M_PI; theta += 2.0 * M_PI /(T)n_angles)
    {
        thetas.push_back(theta);
        TVEntry ddT_vec;
        ddT_vec[0] = std::cos(theta) * std::cos(theta);
        ddT_vec[1] = std::sin(theta) * std::sin(theta);
        ddT_vec[2] = std::cos(theta) * std::sin(theta);
        T youngs_modulus = 1.0 / ddT_vec.dot(S * ddT_vec);
        youngs_moduli.push_back(youngs_modulus);
    }
    for(T youngs_modulus : youngs_moduli)
        std::cout << youngs_modulus << " ";
    std::cout << std::endl;
}

template class Homogenization<double, 3>;
template class Homogenization<double, 2>;