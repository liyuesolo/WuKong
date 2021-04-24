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

public:
    EoLRodSim<T, dim>& sim;
public:
    Homogenization(EoLRodSim<T, dim>& eol_sim) : sim(eol_sim) {}
    ~Homogenization() {}

    void initalizeSim()
    {
        sim.buildPlanePeriodicBCScene3x3();
    }

    void marcoYoungsModulusFitting()
    {
        T s = 1.1;
        int n_angles = 20;
        // for (T theta = 0; theta < M_PI/2.0; theta += (M_PI/2.0)/(T)n_angles)
        // {
        //     T E_theta = YoungsModulusFromUniaxialStrain(theta, s);
        // }
        YoungsModulusFromUniaxialStrain(M_PI*0.2, s);
    }

    T YoungsModulusFromUniaxialStrain(T theta, T s)
    {
        sim.iteratePBCReferencePairs([&](int node_i, int node_j){
            TV Xj = sim.q0.col(node_j).template segment<dim>(0);
            TV Xi = sim.q0.col(node_i).template segment<dim>(0);

            TV strain_dir = TV::Zero();
            if constexpr (dim == 2)
            {
                strain_dir = TV(std::cos(theta), std::sin(theta));
                T Dij = (Xj - Xi).dot(strain_dir);
                T dij = Dij * s;
                sim.pbc_strain_data.push_back(std::make_pair(IV2(node_i, node_j), std::make_pair(strain_dir, dij)));
            }
        });
        return 1.0;
    }
};

#endif