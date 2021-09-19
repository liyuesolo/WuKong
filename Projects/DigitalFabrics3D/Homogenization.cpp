#include "Homogenization.h"
#include <fstream>
#include <iomanip>

template<class T, int dim>
void Homogenization<T, dim>::testOneSample()
{
    initialize();
    // sim.disable_sliding = false;
    TV strain_dir, ortho_dir;
    
    if (sim.add_pbc)
        sim.setUniaxialStrain(45/180.0 * M_PI, s1, strain_dir, ortho_dir);
    // sim.setUniaxialStrain(0.1885, s1, strain_dir, ortho_dir);
    
    // sim.advanceOneStep();
    // TM2 stress_macro, strain_macro;
    // computeMacroStressStrain(stress_macro, strain_macro);
    // std::cout << stress_macro.norm() << std::endl;
    // TV2 E_nu;
    // materialParametersFromUniaxialStrain(1.885, s1, E_nu);
    // std::cout << "theta: " << 3.3929 << " youngs_modulus " << E_nu(0) << " Poisson Ratio: " << E_nu(1) << std::endl;
    // sim.setUniaxialStrain(0.0/180.0 * M_PI, s1, strain_dir, ortho_dir);
}

template<class T, int dim>
void Homogenization<T, dim>::initialize()
{
    
    sim.print_force_mag = false;
    sim.disable_sliding = true;
    sim.verbose = false;
    // sim.buildPlanePeriodicBCScene3x3Subnodes(8);
    sim.buildSceneFromUnitPatch(2);
    
    // sim.buildPlanePeriodicBCScene3x3();
    
    
    sim.add_penalty = false;

    sim.add_shearing = false;

    
    sim.newton_tol = 1e-6;
        
    sim.k_strain = 1e8;
    
    
    sim.k_yc = 1e8;
    sim.k_pbc = 1e8;
    sim.kr = 1e3;
    
    s1 = 1.1;
    s2 = 1.0;
}

template<class T, int dim>
void Homogenization<T, dim>::computeMacroStressStrain(TM2& stress_marco, TM2& strain_marco)
{

    TV xi = deformed_states.template segment<dim>(sim.pbc_pairs_reference[0].first.first[0]);
    TV xj = deformed_states.template segment<dim>(sim.pbc_pairs_reference[0].first.second[0]);
    TV xk = deformed_states.template segment<dim>(sim.pbc_pairs_reference[1].first.first[0]);
    TV xl = deformed_states.template segment<dim>(sim.pbc_pairs_reference[1].first.second[0]);

    TV Xi = rest_states.template segment<dim>(sim.pbc_pairs_reference[0].first.first[0]);
    TV Xj = rest_states.template segment<dim>(sim.pbc_pairs_reference[0].first.second[0]);
    TV Xk = rest_states.template segment<dim>(sim.pbc_pairs_reference[1].first.first[0]);
    TV Xl = rest_states.template segment<dim>(sim.pbc_pairs_reference[1].first.second[0]);

    TM2 X = TM2::Zero(), x = TM2::Zero();
    X.col(0) = (Xi - Xj).template segment<2>(0);
    X.col(1) = (Xk - Xl).template segment<2>(0);

    x.col(0) = (xi - xj).template segment<2>(0);
    x.col(1) = (xk - xl).template segment<2>(0);


    TM2 F_macro = x * X.inverse();
    
    strain_marco = 0.5 * (F_macro.transpose() + F_macro) - TM2::Identity();

    TM2 R90 = TM2::Zero();
    

    R90.row(0) = TV2(0, -1);
    R90.row(1) = TV2(1, 0);

    TV2 n0 = (R90 * (xj - xi).template segment<2>(0)).normalized(), 
        n1 = (R90 * (xl - xk).template segment<2>(0)).normalized();
    
    VectorXT f(deformed_states.rows());
    f.setZero(); 
    sim.addPBCForce(f);
    f *= -1;
    std::vector<TV> f_bc(2, TV::Zero());

    
    sim.iteratePBCPairsWithDirection([&](int direction, Offset offset_i, Offset offset_j)
    {
        
        TV Xj_bc = rest_states.template segment<dim>(offset_j[0]);
        TV Xi_bc = rest_states.template segment<dim>(offset_i[0]);

        // T length = (Xj_bc - Xi_bc).norm();
        T length = (Xj_bc - Xi_bc).norm();
        Offset bc_node = direction == 0 ? offset_j : offset_i;
        if (std::abs(length) >= 1e-6)
            f_bc[direction].template segment<dim>(0) += f.template segment<dim>(bc_node[0]);
        
    });

    TM2 F_bc = TM2::Zero(), n_bc = TM2::Zero();
    F_bc.col(0) = f_bc[0].template segment<2>(0); 
    F_bc.col(1) = f_bc[1].template segment<2>(0);

    // std::cout << F_bc << std::endl;

    n_bc.col(0) = n1; n_bc.col(1) = n0;

    stress_marco = F_bc * n_bc.inverse();
    
    
}

template<class T, int dim>
void Homogenization<T, dim>::computeYoungsModulusPoissonRatioBatch()
{
    initialize();
    
    int n_angles = 400;
    T cycle = 2. * M_PI;
    std::vector<T> thetas;
    std::vector<T> youngs_moduli;
    std::vector<T> poisson_ratio;
    for (T theta = 0; theta <= cycle; theta += cycle/(T)n_angles)
    {
        T theta6 = std::round( theta * 1e4 ) / 1e4;
        
        thetas.push_back(theta6);
        TV2 E_nu;
        materialParametersFromUniaxialStrain(theta6, s1, E_nu);
        // std::cout << "theta: " << theta / M_PI * 180.0 << " youngs_modulus " << E_nu(0) << " Poisson Ratio: " << E_nu(1) << std::endl;
        std::cout << "theta: " << theta6 << " youngs_modulus " << E_nu(0) << " Poisson Ratio: " << E_nu(1) << std::endl;
        T E = E_nu(0); T nu = E_nu(1);
        youngs_moduli.push_back(E);
        // std::cout << E << " " << youngs_moduli[0] << std::endl;
        poisson_ratio.push_back(nu);
    }
    for(T theta : thetas)
        std::cout << theta << " ";
    std::cout << std::endl;
    for(T E : youngs_moduli)
        std::cout << E << " ";
    std::cout << std::endl;
    for(T nu : poisson_ratio)
        std::cout << nu << " ";
    std::cout << std::endl;
    
}

template<class T, int dim>
void Homogenization<T, dim>::materialParametersFromUniaxialStrain(T theta, T s, TV2& E_nu)
{
    TV strain_dir, ortho_dir;
    sim.setUniaxialStrain(theta, s, strain_dir, ortho_dir);
    
    // sim.setBiaxialStrain(theta, s, theta, 1.0, , strain_dir, ortho_dir);
    sim.advanceOneStep();
    
    E_nu = TV2::Zero();
    TM2 stress, strain;
    computeMacroStressStrain(stress, strain);
    T stretch_in_d = strain_dir.template segment<2>(0).dot(strain * strain_dir.template segment<2>(0));
    
    E_nu(0) = strain_dir.template segment<2>(0).dot(stress * strain_dir.template segment<2>(0)) / stretch_in_d;
    E_nu(1) = -ortho_dir.template segment<2>(0).dot(strain * ortho_dir.template segment<2>(0)) / stretch_in_d;
    sim.resetScene();
    
}


template<class T, int dim>
void Homogenization<T, dim>::fitComplianceTensor()
{
    
    initialize();

    CDoF2D S_entry;
    S_entry.setZero();
    ComplianceTensor S;
    
    S.setZero();
    int n_angles = 400;
    
    std::vector<TVEntry> strain_entries, stress_entries;

    auto gatherData = [&, n_angles]()
    {
        for (T theta = 0; theta <= 2.0 * M_PI; theta += 2.0 * M_PI /(T)n_angles)
        {
            TV strain_dir, ortho_dir;
            
            sim.setUniaxialStrain(theta, s1, strain_dir, ortho_dir);
            sim.advanceOneStep();
            TM2 stress, strain;
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

    

    auto computeEnergy = [&]()
    {
        T energy = 0.0;
        for (int i = 0; i < stress_entries.size(); i++)
        {
            // energy += 0.5 * (S * stress_entries[i] - strain_entries[i]).squaredNorm() / strain_entries[i].squaredNorm();
            auto x = stress_entries[i];
            auto b = strain_entries[i];
            T V[1];
            #include "/home/yueli/Documents/ETH/WuKong/Projects/DigitalFabrics/Maple/LSV.mcg"
            energy += V[0];
        }
        return energy;
    };

    auto computeGradient = [&]()
    {
        CDoF2D gradient;
        gradient.setZero();
        for (int i = 0; i < stress_entries.size(); i++)
        {
            CDoF2D F;
            auto x = stress_entries[i];
            auto b = strain_entries[i];
            #include "/home/yueli/Documents/ETH/WuKong/Projects/DigitalFabrics/Maple/LSF.mcg"
            gradient += F;
        }
        return gradient;
    };


    auto computeHessian = [&]()
    {
        CHessian2D H;
        H.setZero();
        for (int a = 0; a < stress_entries.size(); a++)
        {
            auto x = stress_entries[a];
            auto b = strain_entries[a];
            T J[6][6];
            memset(J, 0, sizeof(J));
            #include "/home/yueli/Documents/ETH/WuKong/Projects/DigitalFabrics/Maple/LSJ.mcg"
            for(int i = 0; i< 6; i++)
                for(int j = 0; j< 6; j++)
                    H(i, j) += -J[i][j];
            // H += (stress_entries[i] * stress_entries[i].transpose()).transpose() / strain_entries[i].squaredNorm();
        }
        return H;
    };
    

    auto leastSquareFit = [&]()
    {
        CDoF2D gradient = computeGradient();
        
        CHessian2D H = computeHessian();
        CDoF2D dx = H.colPivHouseholderQr().solve(gradient);
        
        S_entry += dx;

        
        S(0, 0) = S_entry[0]; S(0, 1) = S_entry[1]; S(0, 2) = S_entry[2];
        S(1, 0) = S_entry[1]; S(1, 1) = S_entry[3]; S(1, 2) = S_entry[4];
        S(2, 0) = S_entry[2]; S(2, 1) = S_entry[4]; S(2, 2) = S_entry[5];

        // std::cout << "Sσ - ε " << (S * stress_entries[0] - strain_entries[0]).norm() << std::endl;

        // std::cout << S_entry << std::endl;
    };
    
    gatherData();
    // loadDataFromFile();
    std::cout << "# sample: " << stress_entries.size() << std::endl;
    std::cout << "======== Simulation Data Generation Done ========" << std::endl;
    T e = computeEnergy();
    std::cout << "e: " << e << std::endl;
    leastSquareFit();

    e = computeEnergy();
    auto gd = computeGradient();
    std::cout << "e: " << e << " |g|: " << gd.norm() << std::endl;
    std::cout << "S" << std::endl;
    std::cout << S << std::endl;

    std::vector<T> thetas, youngs_moduli, poisson_ratio;
    for (T theta = 0; theta <= 2.0 * M_PI; theta += 2.0 * M_PI /(T)400)
    {
        thetas.push_back(theta);
        TVEntry ddT_vec, nnT_vec;
        ddT_vec[0] = std::cos(theta) * std::cos(theta);
        ddT_vec[1] = std::sin(theta) * std::sin(theta);
        ddT_vec[2] = std::cos(theta) * std::sin(theta);
        T youngs_modulus = 1.0 / ddT_vec.dot(S * ddT_vec);
        nnT_vec[0] = std::sin(theta) * std::sin(theta);
        nnT_vec[1] = std::cos(theta) * std::cos(theta);
        nnT_vec[2] = -std::cos(theta) * std::sin(theta);
        T nu = -youngs_modulus * ddT_vec.dot(S*nnT_vec);
        youngs_moduli.push_back(youngs_modulus);
        poisson_ratio.push_back(nu);
    }
    for(T theta : thetas)
        std::cout << theta << " ";
    std::cout << std::endl;
    for(T youngs_modulus : youngs_moduli)
        std::cout << youngs_modulus << " ";
    std::cout << std::endl;
    for(T nu : poisson_ratio)
        std::cout << nu << " ";
    std::cout << std::endl;
}


template<class T, int dim>
void Homogenization<T, dim>::fitComplianceFullTensor()
{
    // initialize();
    // /*
    // compliance tensor should be dim x dim by dim x dim
    // */

    // // A : C: A
    // //sum_ijkl Aij Cijkl Akl

    // Vector<T, 16> S_entry;
    // S_entry.setZero();
    // ComplianceTensorFull S;
    // S.setOnes();
    // // S.setZero();
    // int n_angles = 400;
    
    // std::vector<TVEntryFull> strain_entries, stress_entries;

    // auto gatherData = [&, n_angles]()
    // {
    //     for (T theta = 0; theta <= 2.0 * M_PI; theta += 2.0 * M_PI /(T)n_angles)
    //     // for (T theta : {1.25664, 1.88496})
    //     {
    //         TV strain_dir, ortho_dir;
    //         sim.setUniaxialStrain(theta, s1, strain_dir, ortho_dir);
    //         // sim.setBiaxialStrain(theta, s1, theta, s2, strain_dir, ortho_dir);
    //         sim.advanceOneStep();
    //         TM stress, strain;
    //         computeMacroStressStrain(stress, strain);
    //         // std::cout << "E sim : " << strain_dir.dot(stress * strain_dir) / strain.norm() << std::endl;
    //         // std::getchar();
    //         if constexpr (dim == 2)
    //         {
    //             TVEntryFull se0, se1, se2;
    //             se0[0] = stress(0, 0); se0[3] = stress(1, 1); se0[1] = stress(0, 1); se0[2] = stress(1, 0);
    //             se1[0] = strain(0, 0); se1[3] = strain(1, 1); se1[1] = strain(0, 1); se1[2] = strain(1, 0);

    //             strain_entries.push_back(se1);
    //             stress_entries.push_back(se0);
    //         }
    //         sim.resetScene();
    //     }
    // };

    // auto loadDataFromFile = [&]()
    // {
    //     std::string base_path = "/home/yueli/Downloads/hexagon_uniaxial_distanceBased/";
    //     for (int degree = 0; degree < 180; degree++)
    //     {
    //         int cnt = 0;
    //         std::ifstream in(base_path + "uniaxial_"+std::to_string(degree)+".000000degrees.txt");
    //         T eps00, eps01, eps10, eps11, sigma00, sigma01, sigma10, sigma11, dummy;
    //         while(in >> eps00 >> eps01 >> eps10 >> eps11 >> sigma00 >> sigma01 >> sigma10 >> sigma11 >> dummy >> dummy >> dummy >> dummy >> dummy)
    //         {
                
    //             TVEntryFull se0, se1, se2;
    //             se0[0] = sigma00;  se0[1] = sigma01; se0[2] = sigma10; se0[3] = sigma11;
    //             se1[0] = eps00;  se1[1] = eps01; se1[2] = eps10; se1[3] = eps11;
    //             if(se1.norm() < 1e-6)
    //                 continue;
    //             stress_entries.push_back(se0);
    //             strain_entries.push_back(se1);
                
    //         }
    //         in.close();
    //     }
    // };

    // auto computeEnergy = [&]()
    // {
    //     T energy = 0.0;
    //     for (int i = 0; i < stress_entries.size(); i++)
    //     {
            
    //         // energy += 0.5 * (S * stress_entries[i] - strain_entries[i]).squaredNorm() / strain_entries[i].squaredNorm();
    //         auto x = stress_entries[i];
    //         auto b = strain_entries[i];
    //         T V[1];
    //         #include "Maple/LSVFull.mcg"
    //         energy += V[0];
    //     }
    //     return energy;
    // };

    // auto computeGradient = [&]()
    // {
    //     Vector<T, 16> gradient;
    //     gradient.setZero();
    //     for (int i = 0; i < stress_entries.size(); i++)
    //     {
    //         Vector<T, 16> F;
    //         auto x = stress_entries[i];
    //         auto b = strain_entries[i];
    //         #include "Maple/LSFFull.mcg"
    //         gradient += F;
    //     }
    //     return gradient;
    // };


    // auto computeHessian = [&]()
    // {
    //     Matrix<T, 16, 16> H;
    //     H.setZero();
    //     for (int a = 0; a < stress_entries.size(); a++)
    //     {
    //         auto x = stress_entries[a];
    //         auto b = strain_entries[a];
    //         T J[16][16];
    //         memset(J, 0, sizeof(J));
    //         #include "Maple/LSJFull.mcg"
    //         for(int i = 0; i< 16; i++)
    //             for(int j = 0; j< 16; j++)
    //                 H(i, j) += -J[i][j];
    //         // H += (stress_entries[i] * stress_entries[i].transpose()).transpose() / strain_entries[i].squaredNorm();
    //     }
    //     return H;
    // };
    

    // auto leastSquareFit = [&]()
    // {
    //     Vector<T, 16> gradient = computeGradient();
        
    //     Matrix<T, 16, 16> H = computeHessian();
    //     Vector<T, 16> dx = H.colPivHouseholderQr().solve(gradient);
    //     // Vector<T, 16> dx = H.inverse() * gradient;
        
    //     S_entry += dx;

    //     for(int i = 0; i< 4; i++)
    //         for(int j = 0; j< 4; j++)
    //             S(i, j) = S_entry(i * 4 + j);
    // };
    
    // gatherData();
    // // loadDataFromFile();

    // std::cout << "======== Simulation Data Generation Done ========" << std::endl;
    // T e = computeEnergy();
    // std::cout << "e: " << e << std::endl;
    // leastSquareFit();

    // e = computeEnergy();
    // auto gd = computeGradient();
    // std::cout << "e: " << e << " |g|: " << gd.norm() << std::endl;
    // std::cout << "S" << std::endl;
    // std::cout << S << std::endl;
    // std::cout << std::endl;
    // // S << 0.00484941, -2.40251e-07, 2.40148e-07, -0.00457267, 2.00858e-11, 0.00471005, 0.00471203, 2.12701e-11, 2.00858e-11, 0.00471005, 0.00471203, 2.12701e-11, -0.00457267, 2.48572e-07, -2.48449e-07, 0.00484941;

    // std::vector<T> thetas, youngs_moduli;
    // for (T theta = 0; theta <= 2.0 * M_PI; theta += 2.0 * M_PI /(T)400)
    // // for (T theta : {1.25664, 1.88496})
    // {
    //     thetas.push_back(theta);
    //     TVEntryFull ddT_vec;
    //     ddT_vec[0] = std::cos(theta) * std::cos(theta);
    //     ddT_vec[1] = std::cos(theta) * std::sin(theta);
    //     ddT_vec[2] = std::cos(theta) * std::sin(theta);
    //     ddT_vec[3] = std::sin(theta) * std::sin(theta);
    //     T youngs_modulus = 1.0 / ddT_vec.dot(S * ddT_vec);
    //     youngs_moduli.push_back(youngs_modulus);
    // }
    // for(T theta : thetas)
    //     std::cout << theta << " ";
    // std::cout << std::endl;
    // for(T youngs_modulus : youngs_moduli)
    //     std::cout << youngs_modulus << " ";
    // std::cout << std::endl;
}

template class Homogenization<double, 3>;
template class Homogenization<double, 2>;   