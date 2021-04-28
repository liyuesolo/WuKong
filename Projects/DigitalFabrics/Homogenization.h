#ifndef HOMOGENIZATION_H
#define HOMOGENIZATION_H

#include "EoLRodSim.h"

template<class T, int dim>
class EoLRodSim;

template<class T, int dim>
class Homogenization
{
    using TVDOF = Vector<T, dim+2>;
    using DOFStack = Matrix<T, dim + 2, Eigen::Dynamic>;
    using IV2 = Vector<int, 2>;
    using TV = Vector<T, dim>;
    using TM = Matrix<T, dim, dim>;

public:
    EoLRodSim<T, dim>& sim;
public:
    Homogenization(EoLRodSim<T, dim>& eol_sim) : sim(eol_sim) {}
    ~Homogenization() {}

    void initalizeSim()
    {
        sim.buildPlanePeriodicBCScene3x3Subnodes(4);
        // sim.buildPlanePeriodicBCScene3x3();
        TV strain_dir;
        // sim.setUniaxialStrain(0.76969, 1.1, strain_dir);
        // sim.setUniaxialStrain(0., 1.1, strain_dir);
        sim.setUniaxialStrain(M_PI/4, 1.8, strain_dir);
    }

    void marcoYoungsModulusFitting()
    {
        sim.disable_sliding = true;
        sim.buildPlanePeriodicBCScene3x3();
        T s = 1.1;
        int n_angles = 400;
        T cycle = 2. * M_PI;
        // T cycle = M_PI / 4.0;
        std::vector<T> thetas, youngs_moduli;
        for (T theta = 0; theta <= cycle; theta += cycle/(T)n_angles)
        {
            thetas.push_back(theta);
            // std::cout << theta << std::endl;
            T youngs_modulus = YoungsModulusFromUniaxialStrain(theta, s);
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

    T YoungsModulusFromUniaxialStrain(T theta, T s)
    {
        TV strain_dir;
        sim.setUniaxialStrain(theta, s, strain_dir);
        sim.advanceOneStep();
        strain_dir.normalize(); 
        TM sigma;
        sim.computeMacroStress(sigma, strain_dir);
        T youngs_modulus = strain_dir.dot(sigma * strain_dir);
        sim.resetScene();
        return youngs_modulus;
    }
};

#endif