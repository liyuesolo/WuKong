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
    
    DOFStack q, q0;
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
    T kc = 1e2;  //constraint term
    T kn = 1e-3; //
    T kb = 1.0;
    T km = 1e-3; //mass term
    T kx = 1.0; // shearing term

    T L = 1;
    

    TV gravity = TV::Zero();

    bool add_stretching = true;
    bool add_bending = true;
    bool add_shearing = true;
    bool add_penalty = true;
    bool add_regularizor = true;

    std::unordered_map<int, std::pair<TVDOF, TVDOF>> dirichlet_data;

public:

    EoLRodSim()
    {
        gravity[1] = -9.8;
    }
    ~EoLRodSim() {}
    
    void cout3Nodes(int n0, int n1, int n2)
    {
        std::cout << n0 << " " << n1 << " " << n2 << std::endl;
    } 

    template <class OP>
    void iterateDirichletData(const OP& f) {
        for (auto dirichlet: dirichlet_data){
            f(dirichlet.first, dirichlet.second.first, dirichlet.second.second);
        } 
    }

    template <class OP>
    void iterateYarnCrossingsSerial(const OP& f) {
        for (int i = 0; i < n_nodes; i++)
            f(i, connections(0, i), connections(1, i), connections(2, i), connections(3, i));
    }

    template <class OP>
    void iterateYarnCrossingsParallel(const OP& f) {
        tbb::parallel_for(0, n_nodes, [&](int i)
        {
            f(i, connections(0, i), connections(1, i), connections(2, i), connections(3, i));
        }); 
    }

    T computeTotalEnergy(Eigen::Ref<const DOFStack> dq)
    {
        // advect q to compute internal energy
        DOFStack q_temp = q + dq;

        T total_energy = 0;
        if (add_stretching)
            total_energy += addStretchingEnergy(q_temp);
        if (add_bending)
            total_energy += addBendingEnergy(q_temp);
        if (add_shearing)
        {
            total_energy += addShearingEnergy(q_temp, true);
            total_energy += addShearingEnergy(q_temp, false);
        }
        if (add_penalty)
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
        if (add_stretching)
            addStretchingForce(q_temp, residual);
        if (add_bending)
            addBendingForce(q_temp, residual);
        if (add_shearing)
        {
            addShearingForce(q_temp, residual, true);
            addShearingForce(q_temp, residual, false);
        }
        if (add_penalty)
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
        const DOFStack q_temp = q + dq;
        if (add_stretching)
            addStretchingK(q_temp, entry_K);
        if (add_bending)
            addBendingK(q_temp, entry_K);
        if (add_shearing)
        {
            addShearingK(q_temp, entry_K, true);
            addShearingK(q_temp, entry_K, false);
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
        if (add_regularizor)
            addMassMatrix(entry_K);
        addStiffnessMatrix(entry_K, dq);
        if (add_penalty)
            addConstraintMatrix(entry_K, dq);
    }

    bool linearSolve(const std::vector<Eigen::Triplet<T>>& entry_K, 
        Eigen::Ref<const DOFStack> residual, Eigen::Ref<DOFStack> ddq)
    {
        ddq.setZero();
        Eigen::SparseMatrix<T> A(n_nodes * dof, n_nodes * dof);
        A.setFromTriplets(entry_K.begin(), entry_K.end()); 
        // A.setIdentity();
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
        // std::cout << "E0: " << E0 << std::endl;
        int cnt = 0;
        while(true)
        {
            DOFStack dq_ls = dq + alpha * ddq;
            T E1 = computeTotalEnergy(dq_ls);
            // std::cout << "E1: " << E1 << std::endl;
            if (E1 - E0 < 0) {
                dq = dq_ls;
                break;
            }
            alpha *= T(0.5);
            cnt += 1;
            if (cnt > 100)
            {
                std::cout << "line search count: " << cnt << std::endl;
            }
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
            T residual_norm = computeResidual(residual, dq);
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
    void resetScene() { q = q0; }
public:
    // Scene.cpp
    
    void build5NodeTestScene();
    void buildLongRodForBendingTest();
    void buildShearingTest();
    void buildRodNetwork(int width, int height);
    void buildMeshFromRodNetwork(Eigen::MatrixXd& V, Eigen::MatrixXi& F);

    // BoundaryCondtion.cpp
    void addBCStretchingTest();
    void addBCShearingTest();
    
    // DerivativeTest.cpp
    void runDerivativeTest();
    void checkGradient(Eigen::Ref<DOFStack> dq);
    void checkHessian(Eigen::Ref<DOFStack> dq);

    // Bending.cpp
    void entryHelperBending(std::vector<TV>& x, T u1, T u2, 
        std::vector<Eigen::Triplet<T>>& entry_K, int n0, int n1, int n2, int uv_offset);
    void addBendingK(Eigen::Ref<const DOFStack> q_temp, std::vector<Eigen::Triplet<T>>& entry_K);  
    void addBendingForce(Eigen::Ref<const DOFStack> q_temp, Eigen::Ref<DOFStack> residual);
    T addBendingEnergy(Eigen::Ref<const DOFStack> q_temp);

    // Stretching.cpp
    void addStretchingK(Eigen::Ref<const DOFStack> q_temp, std::vector<Eigen::Triplet<T>>& entry_K);  
    void addStretchingForce(Eigen::Ref<const DOFStack> q_temp, Eigen::Ref<DOFStack> residual);
    T addStretchingEnergy(Eigen::Ref<const DOFStack> q_temp);

    // Shearing.cpp
    void addShearingK(Eigen::Ref<const DOFStack> q_temp, std::vector<Eigen::Triplet<T>>& entry_K, bool top_right);  
    void addShearingForce(Eigen::Ref<const DOFStack> q_temp, Eigen::Ref<DOFStack> residual, bool top_right);
    T addShearingEnergy(Eigen::Ref<const DOFStack> q_temp, bool top_right);
};

#endif