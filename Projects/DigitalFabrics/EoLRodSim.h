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


template<class T, int dim>
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
    using TMDOF = Matrix<T, dim + 2, dim + 2>;

    using TV3Stack = Matrix<T, 3, Eigen::Dynamic>;
    using TVStack = Matrix<T, dim, Eigen::Dynamic>;
    using IV3Stack = Matrix<int, 3, Eigen::Dynamic>;
    using IV4Stack = Matrix<int, 4, Eigen::Dynamic>;
    using TV2Stack = Matrix<T, 3, Eigen::Dynamic>;
    using DOFStack = Matrix<T, dim + 2, Eigen::Dynamic>;

    using VectorXT = Matrix<T, Eigen::Dynamic, 1>;

    using IV2 = Vector<int, 2>;
    using IV3 = Vector<int, 3>;
    using IV4 = Vector<int, 4>;
    


    int dof = dim + 2;
    
    DOFStack q;
    IV3Stack rods;
    IV4Stack connections;
    TV3Stack normal;
    int n_nodes;
    int n_rods;
    IV2 n_rod_uv;
    
    const static int grid_range = 3;
    int final_dim;

    T dt = 1;
    T newton_tol = 1e-5;
    T E = 1e7;
    T R = 0.01;

    T rho = 1;
    T ks = 1.0;  // stretching term
    T kc = 1e1;  //constraint term
    T kn = 1e-3; //
    T kb = 1.0;
    T km = 1e-2; //mass term
    TV3 gravity = TV3::Zero();

    std::unordered_map<int, std::pair<TVDOF, TVDOF>> dirichlet_data;

public:

    EoLRodSim()
    {
        gravity[1] = -9.8;
    }
    ~EoLRodSim() {}
    
    //https://math.stackexchange.com/questions/99299/best-fitting-plane-given-a-set-of-points
    void computeCrossingNormal() {}


    template <class OP>
    void iterateDirichletData(const OP& f) {
        for (auto dirichlet: dirichlet_data){
            f(dirichlet.first, dirichlet.second.first, dirichlet.second.second);
        } 
    }

    template <class OP>
    void iterateYarnCrossings(const OP& f) {
        for (int i = 0; i < n_nodes; i++)
            f(i, connections(0, i), connections(1, i), connections(2, i), connections(3, i));
    }

    T computeTotalEnergy(Eigen::Ref<const DOFStack> dq)
    {
        // advect q to compute internal energy
        DOFStack q_temp = q + dq;

        T total_energy = 0;
        VectorXT rod_energy(n_rods);
        rod_energy.setZero();

        tbb::parallel_for(0, n_rods, [&](int rod_idx){
        // for (int rod_idx = 0; rod_idx < n_rods; rod_idx++) {
            int node0 = rods.col(rod_idx)[0];
            int node1 = rods.col(rod_idx)[1];
            TV x0 = q_temp.col(node0).template segment<dim>(0);
            TV x1 = q_temp.col(node1).template segment<dim>(0);
            TV2 u0 = q_temp.col(node0).template segment<2>(dim);
            TV2 u1 = q_temp.col(node1).template segment<2>(dim);
            TV2 delta_u = u1 - u0;

            int yarn_type = rods.col(rod_idx)[2];

            int uv_offset = yarn_type == WARP ? 0 : 1;
        
            // add elastic potential here 1/2 ks delta_u * (||w|| - 1)^2
            TV w = (x1 - x0) / std::abs(delta_u[uv_offset]);
            rod_energy[rod_idx] += 0.5 * ks * std::abs(delta_u[uv_offset]) * std::pow(w.norm() - 1.0, 2);
            
        // }
        });
        total_energy += rod_energy.sum();

        iterateYarnCrossings([&](int middle, int bottom, int top, int left, int right){
            if (left != -1 && right != -1)
            {
                TV x2 = q_temp.col(left).template segment<dim>(0);
                TV x1 = q_temp.col(right).template segment<dim>(0);
                TV x0 = q_temp.col(middle).template segment<dim>(0);
                T u2 = q_temp(dim, left);
                T u0 = q_temp(dim, right);
                T u1 = q_temp(dim, middle);
                T l1 = (x1 - x0).norm(), l2 = (x2 - x0).norm();
                TV d1 = (x1 - x0).normalized(), d2 = (x2 - x0).normalized();
                T theta = std::acos(-d1.dot(d2));
                total_energy += kb * theta * theta / (u1 - u2);
                // std::cout << theta << " " << middle << " " << (u1 - u2) << std::endl;
            }
            if (top != -1 && bottom != -1)
            {
                TV x2 = q_temp.col(bottom).template segment<dim>(0);
                TV x1 = q_temp.col(top).template segment<dim>(0);
                TV x0 = q_temp.col(middle).template segment<dim>(0);
                T v2 = q_temp(dim + 1, bottom);
                T v0 = q_temp(dim + 1, top);
                T v1 = q_temp(dim + 1, middle);
                T l1 = (x1 - x0).norm(), l2 = (x2 - x0).norm();
                TV d1 = (x1 - x0).normalized(), d2 = (x2 - x0).normalized();
                T theta = std::acos(-d1.dot(d2));
                total_energy += kb * theta * theta / (v1 - v2);
                // std::cout << theta << " " << middle << " " << (v1 - v2) << std::endl;
            }
        });
        
        // add constraint term here 1/2 kc (q - q')^T (q - q')
        iterateDirichletData([&](const auto& node_id, const auto& target, const auto& mask)
        {
            for(int d = 0; d < dof; d++)
                if (std::abs(target(d)) <= 1e10 && mask(d))
                    total_energy += 0.5 * kc * std::pow(target(d) - dq(d, node_id), 2);
        });
        return total_energy;
    }



    T computeResidual(Eigen::Ref<DOFStack> residual, Eigen::Ref<const DOFStack> dq)
    {
        const DOFStack q_temp = q + dq;
        for (int rod_idx = 0; rod_idx < n_rods; rod_idx++)
        {
            int node0 = rods.col(rod_idx)[0];
            int node1 = rods.col(rod_idx)[1];
            TV x0 = q_temp.col(node0).template segment<dim>(0);
            TV x1 = q_temp.col(node1).template segment<dim>(0);
            TV2 u0 = q_temp.col(node0).template segment<2>(dim);
            TV2 u1 = q_temp.col(node1).template segment<2>(dim);
            TV2 delta_u = u1 - u0;
            
            T l = (x1 - x0).norm();
            TV d = (x1 - x0).normalized();

            int yarn_type = rods.col(rod_idx)[2];

            int uv_offset = yarn_type == WARP ? 0 : 1;

            TV w = (x1 - x0) / std::abs(delta_u[uv_offset]);
            //fx
            residual.col(node0).template segment<dim>(0) += ks * (w.norm() - 1.0) * d;
            residual.col(node1).template segment<dim>(0) += -ks * (w.norm() - 1.0) * d;
            //fu
            residual.col(node0)[dim + uv_offset] += -0.5 * ks * (std::pow(w.norm(), 2) - 1.0);
            residual.col(node1)[dim + uv_offset] += 0.5 * ks * (std::pow(w.norm(), 2) - 1.0);
        }
        
        iterateYarnCrossings([&](int middle, int bottom, int top, int left, int right){
            if (left != -1 && right != -1)
            {
                TV x2 = q_temp.col(left).template segment<dim>(0);
                TV x1 = q_temp.col(right).template segment<dim>(0);
                TV x0 = q_temp.col(middle).template segment<dim>(0);
                T u2 = q_temp(dim, left);
                T u0 = q_temp(dim, middle);
                T u1 = q_temp(dim, right);
                T l1 = (x1 - x0).norm(), l2 = (x2 - x0).norm();
                TV d1 = (x1 - x0).normalized(), d2 = (x2 - x0).normalized();
                TM P1 = TM::Identity() - d1 * d1.transpose();
                TM P2 = TM::Identity() - d2 * d2.transpose();
                T theta = std::acos(-d1.dot(d2));

                std::cout << theta << " " << middle << " u1 - u2 " << (u1 - u2) << std::endl;
                TV Fx1 = -(2.0 * kb * theta) / (l1 * (u1 - u2) * std::sin(theta)) * P1 * d2;
                TV Fx2 = -(2.0 * kb * theta) / (l2 * (u1 - u2) * std::sin(theta)) * P2 * d1;
                residual.col(right).template segment<dim>(0) += Fx1;
                residual.col(left).template segment<dim>(0) += Fx2;
                residual.col(middle).template segment<dim>(0) += -(Fx1 + Fx2);
                residual(dim, right) += kb * theta * theta / std::pow(u1-u2, 2);
                residual(dim, left) += -kb * theta * theta / std::pow(u1-u2, 2);
                
            }
            if (top != -1 && bottom != -1)
            {
                TV x2 = q_temp.col(bottom).template segment<dim>(0);
                TV x1 = q_temp.col(top).template segment<dim>(0);
                TV x0 = q_temp.col(middle).template segment<dim>(0);
                T v2 = q_temp(dim + 1, bottom);
                T v0 = q_temp(dim + 1, middle);
                T v1 = q_temp(dim + 1, top);
                T l1 = (x1 - x0).norm(), l2 = (x2 - x0).norm();
                TV d1 = (x1 - x0).normalized(), d2 = (x2 - x0).normalized();
                TM P1 = TM::Identity() - d1 * d1.transpose();
                TM P2 = TM::Identity() - d2 * d2.transpose();
                T theta = std::acos(-d1.dot(d2));   
                std::cout << theta << " " << middle << " v1 - v2 " << (v1 - v2) << std::endl;
            
                TV Fx1 = -(2.0 * kb * theta) / (l1 * (v1 - v2) * std::sin(theta)) * P1 * d2;
                TV Fx2 = -(2.0 * kb * theta) / (l2 * (v1 - v2) * std::sin(theta)) * P2 * d1;
                residual.col(top).template segment<dim>(0) += Fx1;
                residual.col(bottom).template segment<dim>(0) += Fx2;
                residual.col(middle).template segment<dim>(0) += -(Fx1 + Fx2);
                residual(dim + 1, top) += kb * theta * theta / std::pow(v1-v2, 2);
                residual(dim + 1, bottom) += -kb * theta * theta / std::pow(v1-v2, 2);
            }
        });

        iterateDirichletData([&](const auto& node_id, const auto& target, const auto& mask)
        {
            for(int d = 0; d < dof; d++)
                if (std::abs(target(d)) <= 1e10 && mask(d))
                    residual(d, node_id) -= kc * (dq(d, node_id) - target(d));
        });
        return residual.norm();
    }

    void initializeSystemMatrix(Eigen::Ref<const TVStack> dq, 
        std::vector<int> &entryCol, 
        std::vector<TMDOF> &entryVal)
    {
        final_dim = std::pow(grid_range, dof);
        int n_rows = n_nodes;
        entryCol.resize(n_rows * final_dim);
        entryVal.resize(n_rows * final_dim);
        tbb::parallel_for(0, n_rows, [&](int g) {
            for (int i = 0; i < final_dim; ++i) {
                entryCol[g * final_dim + i] = -1;
                entryVal[g * final_dim + i] = TMDOF::Zero();
            }
        });
    }

    void addMassMatrix(std::vector<Eigen::Triplet<T>>& entry_K)
    {
        for(int i = 0; i < n_nodes * dof; i++)
            entry_K.push_back(Eigen::Triplet<T>(i, i, km));    
    }

    void addStiffnessMatrix(std::vector<Eigen::Triplet<T>>& entry_K, Eigen::Ref<const DOFStack> dq)
    {
        DOFStack q_temp = q + dq;
        for (int rod_idx = 0; rod_idx < n_rods; rod_idx++)
        {
            int node0 = rods.col(rod_idx)[0];
            int node1 = rods.col(rod_idx)[1];
            TV x0 = q_temp.col(node0).template segment<dim>(0);
            TV x1 = q_temp.col(node1).template segment<dim>(0);
            TV2 u0 = q_temp.col(node0).template segment<2>(dim);
            TV2 u1 = q_temp.col(node1).template segment<2>(dim);
            TV2 delta_u = u1 - u0;

            T l = (x1 - x0).norm();
            TV d = (x1 - x0).normalized();

            TM P = TM::Identity() - d * d.transpose();
            int yarn_type = rods.col(rod_idx)[2];

            int uv_offset = yarn_type == WARP ? 0 : 1;

            TV w = (x1 - x0) / std::abs(delta_u[uv_offset]);
            
            // add streching K here
            {
                TM dfxdx = -1.0 * (ks/l * P - ks / std::abs(delta_u[uv_offset]) * TM::Identity());
                TV dfxdu = -1.0 * (ks * w.norm() / std::abs(delta_u[uv_offset]) * d);
                T dfudu = -1.0 * (-ks * w.squaredNorm() / std::abs(delta_u[uv_offset]));
                TV dfudx = -1.0 * (ks / std::abs(delta_u[uv_offset]) * w);

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

    void addConstraintMatrix(std::vector<Eigen::Triplet<T>>& entry_K, Eigen::Ref<const DOFStack> dq)
    {
        // penalty term
        iterateDirichletData([&](const auto& node_id, const auto& target, const auto& mask)
        {
            for(int d = 0; d < dof; d++)
                if (std::abs(target(d)) <= 1e10 && mask(d))
                    entry_K.push_back(Eigen::Triplet<T>(node_id * dof + d, node_id * dof + d, kc));
        });
    }

    void buildSystemMatrix(std::vector<Eigen::Triplet<T>>& entry_K, Eigen::Ref<const DOFStack> dq)
    {
        addMassMatrix(entry_K);
        addStiffnessMatrix(entry_K, dq);
        addConstraintMatrix(entry_K, dq);
    }

    bool linearSolve(const std::vector<Eigen::Triplet<T>>& entry_K, 
        Eigen::Ref<const DOFStack> residual, Eigen::Ref<DOFStack> ddq)
    {
        ddq.setZero();
        Eigen::SparseMatrix<T> A(n_nodes * dof, n_nodes * dof);
        A.setFromTriplets(entry_K.begin(), entry_K.end()); 
        Eigen::SparseLU<Eigen::SparseMatrix<T>> solver;
        solver.compute(A);
        const auto& rhs = Eigen::Map<const VectorXT>(residual.data(), residual.size());
        Eigen::Map<VectorXT>(ddq.data(), ddq.size()) = solver.solve(rhs);
        return true;
    }


    T newtonLineSearch(Eigen::Ref<DOFStack> dq, Eigen::Ref<const DOFStack> residual, int line_search_max = 100)
    {
        int nz_stretching = 16 * n_rods;
        int nz_penalty = dof * dirichlet_data.size();

        DOFStack ddq(dof, n_nodes);
        ddq.setZero();

        std::vector<Eigen::Triplet<T>> entry_K;
        buildSystemMatrix(entry_K, dq);
        linearSolve(entry_K, residual, ddq);
        T norm = ddq.norm();
        if (norm < 1e-5) return norm;
        T alpha = 1;
        T E0 = computeTotalEnergy(dq);
        std::cout << "E0: " << E0 << std::endl;
        int cnt = 0;
        while(true)
        {
            DOFStack dq_ls = dq + alpha * ddq;
            T E1 = computeTotalEnergy(dq_ls);
            std::cout << "E1: " << E1 << std::endl;
            if (E1 - E0 < 0) {
                dq = dq_ls;
                break;
            }
            alpha *= T(0.5);
            cnt += 1;
            if (cnt == line_search_max) 
                return 1e30;
        }
        return norm;    
    }

    void implicitUpdate(Eigen::Ref<DOFStack> dq)
    {
        int cnt = 0;
        T norm = 1e10;
        while (true)
        {
            // set Dirichlet boundary condition
            // iterateDirichletData([&](const auto& node_id, const auto& target, const auto& mask)
            // {
            //     for(int d = 0; d < dof; d++)
            //         if (std::abs(target(d)) <= 1e10 && mask(d))
            //             dq(d, node_id) = target(d);
            // });
            
            DOFStack residual(dof, n_nodes);
            residual.setZero();
            q(0, 3) += 0.1;
            q(2, 3) += 0.1;
            q(3, 5) += 0.1;
            q(1, 4) += 0.1;
            q(2, 1) += 0.1;
            T residual_norm = computeResidual(residual, dq);
            checkGradient(dq);
            // checkHessian(dq);
            // std::cout << "|g|: " << residual_norm << std::endl;
            norm = newtonLineSearch(dq, residual);
            if (norm < newton_tol)
                break;
            cnt++;
        }
        std::cout << "# of newton solve: " << cnt << " exited with |g|: " << norm << std::endl;
    }

    void advanceOneStep()
    {
        DOFStack dq(dof, n_nodes);
        dq.setZero();
        implicitUpdate(dq);
        q += dq;
    }

public:
    // Scene.cpp
    void build5NodeTestScene();
    void buildRodNetwork(int width, int height);
    void buildMeshFromRodNetwork(Eigen::MatrixXd& V, Eigen::MatrixXi& F);

    // BoundaryCondtion.cpp
    void addBCStretchingTest();
    void addBCShearingTest();

    // DerivativeTest.cpp
    void checkGradient(Eigen::Ref<DOFStack> dq);
    void checkHessian(Eigen::Ref<DOFStack> dq);
};

#endif