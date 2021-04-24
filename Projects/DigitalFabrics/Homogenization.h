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
        sim.buildPlanePeriodicBCScene3x3();
        TV strain_dir;
        sim.setUniaxialStrain(0.628319, 1.1, strain_dir);
    }

    void marcoYoungsModulusFitting()
    {
        T s = 1.1;
        int n_angles = 20;
        T cycle = 2.0 * M_PI;
        std::vector<T> thetas, youngs_moduli;
        for (T theta = 0; theta <= cycle; theta += cycle/(T)n_angles)
        {
            thetas.push_back(theta);
            T youngs_modulus = YoungsModulusFromUniaxialStrain(theta, s);
            // std::cout << youngs_modulus << std::endl;
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
        sim.computeMacroStress(sigma);
        T youngs_modulus = strain_dir.dot(sigma * strain_dir);
        sim.resetScene();
        return youngs_modulus;
    }
};

#endif