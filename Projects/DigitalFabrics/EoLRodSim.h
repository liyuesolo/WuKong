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


template <int dim>
struct VectorHash
{
    typedef Vector<int, dim> IV;
    size_t operator()(const IV& a) const{
        std::size_t h = 0;
        for (int d = 0; d < dim; ++d) {
            h ^= std::hash<int>{}(a(d)) + 0x9e3779b9 + (h << 6) + (h >> 2); 
        }
        return h;
    }
};

template <int dim>
struct VectorPairHash
{
    typedef Vector<int, dim> IV;
    size_t operator()(const std::pair<IV, IV>& a) const{
        std::size_t h = 0;
        for (int d = 0; d < dim; ++d) {
            h ^= std::hash<int>{}(a.first(d)) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<int>{}(a.second(d)) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return h;
    }
};


template<class T, int dim = 3>
class EoLRodSim
{
public:
    using Simulation = EoLRodSim<T, dim>;
    

    using TV2 = Vector<T, 2>;
    using TV3 = Vector<T, 3>;
    using TV = Vector<T, dim>;
    using TV5 = Vector<T, 5>;
    using TVDOF = Vector<T, dim+2>;
    
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
    T kc = 1e5;
    TV3 gravity = TV3::Zero();

    std::unordered_map<int, std::pair<TVDOF, TVDOF>> dirichlet_data;

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
        T total_energy = 0;
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
           
        });

        total_energy += rod_energy.sum();
         // add constraint term here 1/2 kc (q - q')^T (q - q')

        for(auto node_target : dirichlet_data)
        {
            int node_id = node_target.first;
            TVDOF target = node_target.second.first;
            TVDOF mask = node_target.second.second;
            TVDOF delta_dis = (target - qi.col(node_id)).array() * mask.array();
            total_energy += 0.5 * kc * (delta_dis).dot(delta_dis);
        }

        return total_energy;
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

        for(auto node_target : dirichlet_data)
        {
            int node_id = node_target.first;
            TVDOF target = node_target.second.first;
            TVDOF mask = node_target.second.second;
            TVDOF delta_dis = (qi.col(node_id) - target).array() * mask.array();

            residual.col(node_id) += kc * delta_dis;
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

        // penalty term
        for(auto node_target : dirichlet_data)
        {
            int node_id = node_target.first;
            TVDOF mask = node_target.second.second;
            for(int dof_id = 0; dof_id < dof; dof_id++)
            {
                if(std::abs(mask[dof_id] - 1) < 1e-6)
                    entry_K.push_back(Eigen::Triplet<T>(node_id * dof + dof_id, node_id * dof + dof_id, kc));
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
        Eigen::SparseMatrix<T> A(n_nodes * dof, n_nodes * dof);
        int nz_stretching = 16 * n_rods;
        int nz_penalty = dof * dirichlet_data.size();
        A.reserve(nz_stretching + nz_penalty);
        A.setFromTriplets(entry_K.begin(), entry_K.end());
        Eigen::SparseLU<Eigen::SparseMatrix<T>> solver;
        solver.compute(A);
        
        const auto& rhs = Eigen::Map<const VectorXT>(residual.data(), residual.size());
        Eigen::Map<VectorXT>(delta_dof.data(), delta_dof.size()) = solver.solve(rhs);
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
            std::cout << "solve one step" << std::endl;
            std::exit(0);
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

    void advanceOneStep()
    {
        implicitUpdate();
    }

public:
    // IO.cpp
    void buildRodNetwork(int width, int height);
    void buildMeshFromRodNetwork(Eigen::MatrixXd& V, Eigen::MatrixXi& F);

    // BoundaryCondtion.cpp
    void addBCStretchingTest();
};

#endif