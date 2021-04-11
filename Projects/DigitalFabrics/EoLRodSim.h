#ifndef EOL_ROD_SIM_H
#define EOL_ROD_SIM_H

#include <utility>
#include <iostream>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <tbb/tbb.h>

#include "VecMatDef.h"


#define WARP 0
#define WEFT 1

template<class T, int dim = 3>
class EoLRodSim
{
public:
    using TV2 = Vector<T, 2>;
    using TV3 = Vector<T, 3>;
    using TV = Vector<T, dim>;
    using TV5 = Vector<T, 5>;
    
    using TM3 = Matrix<T, 3, 3>;
    using TM5 = Matrix<T, 5, 5>;

    using TM = Matrix<T, dim, dim>;

    using TV3Stack = Matrix<T, 3, Eigen::Dynamic>;
    using TVStack = Matrix<T, dim, Eigen::Dynamic>;
    using IV3Stack = Matrix<int, 3, Eigen::Dynamic>;
    using TV2Stack = Matrix<T, 3, Eigen::Dynamic>;
    using DOFStack = Matrix<T, dim + 2, Eigen::Dynamic>;

    using VectorXT = Matrix<T, Eigen::Dynamic, 1>;

    using IV2 = Vector<int, 2>;
    using IV3 = Vector<int, 3>;

    int dof = dim + 2;
    
    DOFStack q, dq;
    IV3Stack rods;
    TV3Stack normal;
    int n_nodes;
    int n_rods;
    IV2 n_rod_uv;
    

    T dt = 1;
    T newton_tol = 1e-4;
    T E = 1e7;
    T R = 0.01;

    T rho = 1;
    T ks = 1.0;
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
    // T computeTotalEnergy(const TV5Stack& qi)
    // {
    //     VectorXT rod_energy(n_rods);
    //     rod_energy.setZero();

    //     tbb::parallel_for(0, n_rods, [&](int rod_idx){
    //         int node0 = rods.col(rod_idx)[0];
    //         int node1 = rods.col(rod_idx)[1];
    //         TV3 x0 = qi.col(node0).segment(0, 3);
    //         TV3 x1 = qi.col(node1).segment(0, 3);
    //         TV2 u0 = qi.col(node0).segment(3, 5);
    //         TV2 u1 = qi.col(node1).segment(3, 5);
            
    //         // T delta_l = (u0 - u1)[0] ? rods.col(rod_idx)[2] == WARP : (u0 - u1)[1];
    //         T coeff = rho * delta_l;

    //         // add graviational energy
    //         rod_energy[rod_idx] += coeff * 0.5 * (x0 + x1).dot(gravity);

    //         // add kinetic energy
    //         // 1/2 * qTMq


    //     });

    //     return rod_energy.sum();
    // }

    T computeTotalEnergy(const DOFStack& qi)
    {
        VectorXT rod_energy(n_rods);
        rod_energy.setZero();

        tbb::parallel_for(0, n_rods, [&](int rod_idx){
            int node0 = rods.col(rod_idx)[0];
            int node1 = rods.col(rod_idx)[1];
            TV x0 = qi.col(node0).template segment<dim>(0);
            TV x1 = qi.col(node1).template segment<dim>(0);
            TV2 u0 = qi.col(node0).template segment<2>(dim);
            TV2 u1 = qi.col(node1).template segment<2>(dim);
            TV2 delta_u = u1 - u0;

            int yarn_type = rods.col(rod_idx)[2];

            int uv_offset = 0 ? yarn_type == WARP : 1;

            // add elastic potential here 1/2 ks delta_u * (||w|| - 1)^2
            TV w = (x1 - x0) / std::abs(delta_u[uv_offset]);
            rod_energy[rod_idx] += 0.5 * ks * std::abs(delta_u[uv_offset]) * std::pow(w.norm() - 1.0, 2);
           
            // add constraint term here 1/2 kc (q - q')^T (q - q')


        });

        return rod_energy.sum();
    }


    // M ((qn+1 - qn)/dt - qdotn)/dt = dT/dq - dV/dq
    // Residual = M(qn+1 - qn - dq*dt) - dt*2 (dTdq-dVdq)
    
    // T computeResidual(TV5Stack& residual, const TV5Stack& qi)
    // {
    //     residual.resize(dof, n_nodes);
    //     residual.setZero();

    //     for (int rod_idx = 0; rod_idx < n_rods; rod_idx++)
    //     {
    //         int node0 = rods.col(rod_idx)[0];
    //         int node1 = rods.col(rod_idx)[1];
    //         TV3 x0 = qi.col(node0).segment(0, 3);
    //         TV3 x1 = qi.col(node1).segment(0, 3);
    //         TV2 u0 = qi.col(node0).segment(3, 5);
    //         TV2 u1 = qi.col(node1).segment(3, 5);


    //         // compute gravity                
    //         T delta_l = (u0 - u1)[0] ? rods.col(rod_idx)[2] == WARP : (u0 - u1)[1];

    //         //fg = -dVg/dx
    //         residual.col(node0).segment(0, 3) -= 0.5 * delta_l * rho * gravity * dt * dt;
    //         residual.col(node1).segment(0, 3) -= 0.5 * delta_l * rho * gravity * dt * dt;
            
    //         //fg = -dVg/du
    //         if(rods.col(rod_idx)[2] == WARP)
    //         {
    //             residual.col(node0)[4] -= 0.5 * delta_l * rho * gravity.dot(x1-x0) * dt * dt;
    //             residual.col(node1)[4] -= -0.5 * delta_l * rho * gravity.dot(x1-x0) * dt * dt;
    //         }
    //         else if(rods.col(rod_idx)[2] == WEFT)
    //         {
    //             residual.col(node0)[5] -= 0.5 * delta_l * rho * gravity.dot(x1-x0) * dt * dt;
    //             residual.col(node1)[5] -= -0.5 * delta_l * rho * gravity.dot(x1-x0) * dt * dt;
    //         }

    //         // // compute M(qn+1 - qn) - dq*dt
    //         // T coeff = T(1) / 6 * rho * delta_l;

    //         // TV3 delta_q0 = qi.col(node0).segment<3>(0) - q.col(node0).segment<3>(0);
    //         // TV3 delta_q1 = qi.col(node1).segment<3>(0) - q.col(node1).segment<3>(0);
    //         // TV2 delta_u0 = qi.col(node0).segment<2>(3) - q.col(node0).segment<2>(3);
    //         // TV2 delta_u1 = qi.col(node1).segment<2>(3) - q.col(node1).segment<2>(3);
    //         // // 2I3 * x0
    //         // residual.col(node0).segment(0, 3) += 
    //         //     coeff * 2.0 * delta_q0 - dq.col(node0).segment<3>(0) * dt;
    //         // // I3 x0
    //         // residual.col(node0).segment(0, 3) += 
    //         //     coeff * 2.0 * delta_q0 - dq.col(node0).segment<3>(0) * dt;
    //         // // -2w u0
    //         // if(rods.col(rod_idx)[2] == WARP)
    //         // {
    //         //     residual.col(node0)[4] -= 
    //         //         coeff * 2.0 * delta_u0[0] - dq.col(node0)[4] * dt;
    //         // }

    //         // // I3 * x1
    //         // residual.col(node1).segment(0, 3) += 
    //         //     coeff * 1.0 * delta_q1 - dq.col(node1).segment<3>(0) * dt;
    //         // // 2I3 * x1
    //         // residual.col(node1).segment(0, 3) += 
    //         //     coeff * 2.0 * delta_q1 - dq.col(node1).segment<3>(0) * dt;


    //     }

    //     return residual.norm();
    // }

    T computeResidual(DOFStack& residual, const DOFStack& qi)
    {
        residual.resize(dof, n_nodes);
        residual.setZero();

        for (int rod_idx = 0; rod_idx < n_rods; rod_idx++)
        {
            int node0 = rods.col(rod_idx)[0];
            int node1 = rods.col(rod_idx)[1];
            TV x0 = qi.col(node0).template segment<dim>(0);
            TV x1 = qi.col(node1).template segment<dim>(0);
            TV2 u0 = qi.col(node0).template segment<2>(dim);
            TV2 u1 = qi.col(node1).template segment<2>(dim);
            TV2 delta_u = u1 - u0;

            T l = (x1 - x0).norm();
            TV d = (x1 - x0).normalized();

            int yarn_type = rods.col(rod_idx)[2];

            int uv_offset = 0 ? yarn_type == WARP : 1;

            TV w = (x1 - x0) / std::abs(delta_u[uv_offset]);
            residual.col(node0).template segment<dim>(0) = ks * (w.norm() - 1.0) * d;
            residual.col(node1).template segment<dim>(0) = -ks * (w.norm() - 1.0) * d;
            residual.col(node0)[node0 * dof + dim + uv_offset] = -0.5 * ks * std::pow(w.norm() - 1.0, 2);
            residual.col(node1)[node1 * dof + dim + uv_offset] = 0.5 * ks * std::pow(w.norm() - 1.0, 2);
        }

        return residual.norm();
    }

    

    void addMassMatrix(std::vector<Eigen::Triplet<T>>& entry_K)
    {
        return;
        // TM3 I = TM3::Identity();

        // // tbb::parallel_for(0, n_rods, [&](int rod_idx){
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

    void addStiffnessMatrix(std::vector<Eigen::Triplet<T>>& entry_K, const DOFStack& qi)
    {
        

        // 
        for (int rod_idx = 0; rod_idx < n_rods; rod_idx++)
        {
            int node0 = rods.col(rod_idx)[0];
            int node1 = rods.col(rod_idx)[1];
            TV x0 = qi.col(node0).template segment<dim>(0);
            TV x1 = qi.col(node1).template segment<dim>(0);
            TV2 u0 = qi.col(node0).template segment<2>(dim);
            TV2 u1 = qi.col(node1).template segment<2>(dim);
            TV2 delta_u = u1 - u0;

            T l = (x1 - x0).norm();
            TV d = (x1 - x0).normalized();

            TM P = TM::Identity() - d * d.transpose();

            int yarn_type = rods.col(rod_idx)[2];
            int uv_offset = 0 ? yarn_type == WARP : 1;

            TV w = (x1 - x0) / std::abs(delta_u[uv_offset]);


            // add streching K here
            {
                TM dfxdx = ks/l * P - ks / std::abs(delta_u[uv_offset]) * TM::Identity();
                TV dfxdu = ks * w.norm() / std::abs(delta_u[uv_offset]) * d;
                T dfudu = -ks * std::pow(w.norm(), 2) / std::abs(delta_u[uv_offset]);
                TV dfudx = ks / std::abs(delta_u[uv_offset]) * w;

                for(int i = 0; i < dim; i++)
                {
                    //dfx/dx
                    for(int j = 0; j < dim; j++)
                    {
                        //dfx0/dx0
                        entry_K.push_back(Eigen::Triplet<T>(node0 * dof + i, node0 * dof + j, dfxdx(i, j)));
                        //dfx1/dx1
                        entry_K.push_back(Eigen::Triplet<T>(node1 * dof + i, node1 * dof + j, dfxdx(i, j)));
                        //dfx0/dx1
                        entry_K.push_back(Eigen::Triplet<T>(node0 * dof + i, node1 * dof + j, -dfxdx(i, j)));
                        //dfx1/dx0
                        entry_K.push_back(Eigen::Triplet<T>(node1 * dof + i, node0 * dof + j, -dfxdx(i, j)));
                    }
                    // dfx1/du1
                    entry_K.push_back(Eigen::Triplet<T>(node1 * dof + i, node1 * dof + dim + uv_offset, dfxdu(i)));
                    // dfx1/du0
                    entry_K.push_back(Eigen::Triplet<T>(node1 * dof + i, node0 * dof + dim + uv_offset, -dfxdu(i)));
                    // dfx0/du1
                    entry_K.push_back(Eigen::Triplet<T>(node0 * dof + i, node1 * dof + dim + uv_offset, -dfxdu(i)));
                    // dfx0/du0
                    entry_K.push_back(Eigen::Triplet<T>(node0 * dof + i, node0 * dof + dim + uv_offset, dfxdu(i)));

                    // dfu0/dx0
                    entry_K.push_back(Eigen::Triplet<T>(node0 * dof + dim + uv_offset, node0 * dof + i, dfudx(i)));
                    // dfu1/dx1
                    entry_K.push_back(Eigen::Triplet<T>(node1 * dof + dim + uv_offset, node1 * dof + i, dfudx(i)));
                    // dfu1/dx0
                    entry_K.push_back(Eigen::Triplet<T>(node1 * dof + dim + uv_offset, node0 * dof + i, -dfudx(i)));
                    // dfu0/dx1
                    entry_K.push_back(Eigen::Triplet<T>(node0 * dof + dim + uv_offset, node1 * dof + i, -dfudx(i)));
                }

                //dfu0/du0
                entry_K.push_back(Eigen::Triplet<T>(node0 * dof + dim + uv_offset, node0 * dof + dim + uv_offset, dfudu));
                //dfu1/du1
                entry_K.push_back(Eigen::Triplet<T>(node1 * dof + dim + uv_offset, node1 * dof + dim + uv_offset, dfudu));
                //dfu1/du0
                entry_K.push_back(Eigen::Triplet<T>(node1 * dof + dim + uv_offset, node0 * dof + dim + uv_offset, -dfudu));
                //dfu0/du1
                entry_K.push_back(Eigen::Triplet<T>(node0 * dof + dim + uv_offset, node1 * dof + dim + uv_offset, -dfudu));
            }

            
        }
    }

    void addConstraintMatrix(std::vector<Eigen::Triplet<T>>& entry_K, const DOFStack& qi)
    {

    }

    void buildSystemMatrix(std::vector<Eigen::Triplet<T>>& entry_K, const DOFStack& qi)
    {
        // addMassMatrix(entry_K);
        addStiffnessMatrix(entry_K, qi);
        addConstraintMatrix(entry_K, qi);
    }

    bool linearSolve(const std::vector<Eigen::Triplet<T>>& entry_K, 
        const DOFStack& residual, DOFStack& delta_dof)
    {

        return true;
    }

    void implicitUpdate()
    {
        T E0 = computeTotalEnergy(q);
        DOFStack residual(dof, n_nodes);
        residual.setZero();

        // delta_dof is the delta at each newton step
        DOFStack delta_dof(dof, n_nodes);
        
        // qi is the temporary q_n+1 during newton loop
        DOFStack qi = q;

        while(true)
        {
            delta_dof.setZero();

            T norm = computeResidual(residual, qi);
            if (norm < newton_tol)
                break;
            std::vector<Eigen::Triplet<T>> entry_K;
            // assemble system matrix: mass, stiffness, penalty
            buildSystemMatrix(entry_K, qi);
            // solve delta dof
            bool succeed = linearSolve(entry_K, residual, delta_dof);
            T alpha = 1.0;
            DOFStack qi_ls = qi + alpha * delta_dof;
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

    void updatedqExplicit()
    {
        //M(dqn+1 - dqn)/dt = dT/ddq -dV/ddq
        //dqn+1 = dqn + inv(M) * (dT/ddq - dV/ddq) * dt
    }

    void symplecticUpdate()
    {
        updatedqExplicit();
        q += dt * dq;
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