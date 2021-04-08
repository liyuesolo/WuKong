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

    using VectorXT = Matrix<T, Eigen::Dynamic, 1>;

    
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
    T E = 1e7;
    T R = 0.01;

    T rho = 1;

    TV3 gravity = TV3::Zero();

public:
    EoLRodSim() 
    {
        gravity[1] = -9.8;
    }

    ~EoLRodSim() {}
    
    //https://math.stackexchange.com/questions/99299/best-fitting-plane-given-a-set-of-points
    void computeCrossingNormal()
    {

    }

    //  
    T computeTotalEnergy(const TV5Stack& qi)
    {
        VectorXT rod_energy(n_rods);
        rod_energy.setZero();

        tbb::parallel_for(0, n_rods, [&](int rod_idx){
            int node0 = rods.col(rod_idx)[0];
            int node1 = rods.col(rod_idx)[1];
            TV3 x0 = qi.col(node0).segment(0, 3);
            TV3 x1 = qi.col(node1).segment(0, 3);
            TV2 u0 = qi.col(node0).segment(3, 5);
            TV2 u1 = qi.col(node1).segment(3, 5);
            
            T delta_l = (u0 - u1)[0] ? rods.col(rod_idx)[2] == WARP : (u0 - u1)[1];
            T coeff = rho * delta_l;

            // add graviational energy
            rod_energy[rod_idx] += coeff * 0.5 * (x0 + x1).dot(gravity);

            // add kinetic energy

            
        });

        return rod_energy.sum();
    }

    T computeResidual(TV5Stack& residual, const TV5Stack& qi)
    {
        residual.resize(dof, n_nodes);
        residual.setZero();

        for (int rod_idx = 0; rod_idx < n_rods; rod_idx++)
        {
            int node0 = rods.col(rod_idx)[0];
            int node1 = rods.col(rod_idx)[1];
            TV3 x0 = qi.col(node0).segment(0, 3);
            TV3 x1 = qi.col(node1).segment(0, 3);
            TV2 u0 = qi.col(node0).segment(3, 5);
            TV2 u1 = qi.col(node1).segment(3, 5);


            // compute gravity                
            T delta_l = (u0 - u1)[0] ? rods.col(rod_idx)[2] == WARP : (u0 - u1)[1];

            //fg = -dVg/dx
            residual.col(node0).segment(0, 3) -= 0.5 * delta_l * rho * gravity;
            residual.col(node1).segment(0, 3) -= 0.5 * delta_l * rho * gravity;
            
            //fg = -dVg/du
            if(rods.col(rod_idx)[2] == WARP)
            {
                residual.col(node0)[4] -= 0.5 * delta_l * rho * gravity.dot(x1-x0);
                residual.col(node1)[4] -= -0.5 * delta_l * rho * gravity.dot(x1-x0);
            }
            else if(rods.col(rod_idx)[2] == WEFT)
            {
                residual.col(node0)[5] -= 0.5 * delta_l * rho * gravity.dot(x1-x0);
                residual.col(node1)[5] -= -0.5 * delta_l * rho * gravity.dot(x1-x0);
            }

            // compute Mq


        }

        return residual.norm();
    }

    

    void addMassMatrix(std::vector<Eigen::Triplet<T>>& entry_K)
    {
        TM3 I = TM3::Identity();

        // tbb::parallel_for(0, n_rods, [&](int rod_idx){
        // for (int rod_idx = 0; i < n_rods; i++){
        //     int node0 = rods.col(rod_idx)[0];
        //     int node1 = rods.col(rod_idx)[1];
        //     TV2 u0 = q.col(node0).segment(3, 5);
        //     TV2 u1 = q.col(node1).segment(3, 5);
        //     TV3 x0 = q.col(node0).segment(0, 3);
        //     TV3 x1 = q.col(node1).segment(0, 3);

        //     T delta_u = (u0 - u1).norm();
        //     T coeff = T(1)/6 * rho * delta_u;    
        //     TV3 w = (x1 - x0) / delta_u;
        //     TM3 wTw = w.transpose() * w;
            
        //     for(int i = 0; i < 3; i++)
        //     {
        //         // push 2I
        //         if (i == j)
        //         {
        //             entryK.push_back(node0 * dof * 2 + i, node0 * dof * 2 + j, T(2) * coeff);
        //             entryK.push_back(node0 * dof * 2 + i, node0 * dof * 2 + j, T(2) * coeff);
        //         }
                
        //     }

        // }
        // });
        
    }

    void addStiffnessMatrix()
    {

    }

    void addConstraintMatrix()
    {

    }

    void buildSystemMatrix(std::vector<Eigen::Triplet<T>>& entry_K)
    {
        addMassMatrix(entry_K);
        addStiffnessMatrix();
        addConstraintMatrix();
    }

    bool linearSolve(const std::vector<Eigen::Triplet<T>>& entry_K, 
        const TV5Stack& residual, TV5Stack& delta_dof)
    {

        return true;
    }

    void implicitUpdate()
    {
        T E0 = computeTotalEnergy(q);
        TV5Stack residual(dof, n_nodes);
        residual.setZero();

        // delta_dof is the delta at each newton step
        TV5Stack delta_dof(dof, n_nodes);
        
        // qi is the temporary q_n+1 during newton loop
        TV5Stack qi = q;

        while(true)
        {
            delta_dof.setZero();

            T norm = computeResidual(residual, qi);
            if (norm < newton_tol)
                break;
            std::vector<Eigen::Triplet<T>> entry_K;
            // assemble system matrix: mass, stiffness, penalty
            buildSystemMatrix(entry_K);
            // solve delta dof
            bool succeed = linearSolve(entry_K, residual, delta_dof);
            T alpha = 1.0;
            TV5Stack qi_ls = qi + alpha * delta_dof;
            T E1 = computeTotalEnergy(qi_ls);
            int ls_cnt = 0;
            while(E0 <= E1)
            {
                alpha *= 0.5;
                qi_ls = qi + alpha * delta_dof;
                E1 = computeTotalEnergy(qi_ls);
            }
        }
    }

    void advanceOneStep()
    {
        implicitUpdate();
    }

public:
    // IO.cpp
    void buildRodNetwork(int width, int height);
    void buildMeshFromRodNetwork(Eigen::MatrixXd& V, Eigen::MatrixXi& F);
};

#endif