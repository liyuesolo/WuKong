#ifndef EOL_ROD_SIM_H
#define EOL_ROD_SIM_H

#include <utility>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <tbb/tbb.h>

#include "VecMatDef.h"

enum YarnType {
        WARP=0, WEFT=1, ABANDON=2
    };

template<class T>
class EoLRodSim
{
public:
    using TV2 = Vector<T, 2>;
    using TV3 = Vector<T, 3>;
    using TV5 = Vector<T, 5>;
    
    using TM3 = Matrix<T, 3, 3>;

    using TV3Stack = Matrix<T, 3, Eigen::Dynamic>;
    using IV3Stack = Matrix<int, 3, Eigen::Dynamic>;
    using TV2Stack = Matrix<T, 3, Eigen::Dynamic>;
    using TV5Stack = Matrix<T, 5, Eigen::Dynamic>;

    
    using IV2 = Vector<int, 2>;
    using IV3 = Vector<int, 3>;

    int dof = 5;
    
    TV5Stack q, dq;
    IV3Stack rods;
    int n_nodes;
    int n_rods;
    IV2 n_rod_uv;

    T dt = 1e-3;
    T newton_tol = 1e-4;
    T E = 100;
    T R = 0.001;

    TV3 gravity = TV3::Zero();

public:
    EoLRodSim() 
    {
        gravity[1] = -9.8;
    }

    ~EoLRodSim() {}
    
    //  
    T computeTotalEnergy()
    {
        return 0;
    }

    // T computeResidual(TV5Stack& residual)
    // {

    //     return residual.norm();
    // }

    // void addMassMatrix(std::vector<Eigen::Triplet<T>>& entryK)
    // {
    //     // tbb::parallel_for(0, n_rods, [&](int rod_idx){
    //     //     TV2 delta_u = rod_net.rods[rod_idx];
    //     // });
        
    // }

    void addStiffnessMatrix()
    {

    }

    void addConstraintMatrix()
    {

    }

    // void buildSystemMatrix(std::vector<Eigen::Triplet<T>>& entryK)
    // {

    // }

    void implicitUpdate()
    {
        // T E0 = computeTotalEnergy();
        // TV5Stack residual(dof, n_nodes);
        // residual.setZero();
        // while(true)
        // {
        //     T norm = computeResidual(residual);
        //     if (norm < newton_tol)
        //         break;
            
        //     T E1 = computeTotalEnergy();

        // }
    }

    void advanceOneStep()
    {
        implicitUpdate();
        q += dt * dq;
    }

public:
    // IO.cpp
    void buildRodNetwork(int width, int height);
    void buildMeshFromRodNetwork(Eigen::MatrixXd& V, Eigen::MatrixXi& F);
};

#endif