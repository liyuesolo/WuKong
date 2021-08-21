#include "EoLRodSim.h"

#include "LBFGS.h"

using namespace LBFGSpp;

#include <Spectra/SymEigsShiftSolver.h>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/DenseSymMatProd.h>

// template<class T, int dim>
// class LBFGSWrapper
// {
// public:
//     using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
//     using TV = Vector<T, dim>;
//     using Offset = Vector<int, dim + 1>;
//     using StiffnessMatrix = Eigen::SparseMatrix<T>;
    
// private:
//     int n;
//     EoLRodSim<T, dim>& sim;
    
//     int n0 = 29, n1 = 30;
//     TV x0, x1;
//     TV x0_target = TV(0.0377535,   0.0789261,  0);
//     TV x1_target = TV(0.0479829,    0.0789278, 0);

//     VectorXT computedOdx()
//     {
//         sim.Rods[5]->x(n0, x0);
//         sim.Rods[6]->x(n1, x1);
//         VectorXT dOdx = VectorXT::Zero(sim.deformed_states.rows());
//         Offset offset0, offset1;
//         sim.Rods[5]->getEntry(n0, offset0);
//         sim.Rods[6]->getEntry(n1, offset1);
//         dOdx.template segment<dim>(offset0[0]) = x0 - x0_target;
//         dOdx.template segment<dim>(offset1[0]) = x1 - x1_target;
//         return sim.W.transpose() * dOdx;
//     }

//     void updateDesignParameter(const VectorXT& new_p)
//     {
//         int cnt = 0;
//         for (auto& crossing : sim.rod_crossings)
//         {
//             if (crossing->is_fixed)
//             {
//                 cnt += crossing->rods_involved.size() * 2;
//                 continue;
//             }
//             for (int i = 0; i < crossing->rods_involved.size(); i++)
//             {
//                 // if (i == 0)
//                 //     crossing->sliding_ranges[i].setZero();
//                 // else
//                 // {
//                 //     crossing->sliding_ranges[i] = new_p.template segment<2>(cnt);
//                 // }
//                 crossing->sliding_ranges[i] = new_p.template segment<2>(cnt);
//                 crossing->sliding_ranges[i][0] = std::max(0.0, crossing->sliding_ranges[i][0]);
//                 crossing->sliding_ranges[i][1] = std::max(0.0, crossing->sliding_ranges[i][1]);
//                 // std::cout << crossing->sliding_ranges[i].transpose() << std::endl;
//                 cnt += 2;
//             }
//         }
//     }
    
// public:
//     LBFGSWrapper(int n_, EoLRodSim<T, dim>& _sim) : n(n_), sim(_sim) {}
//     double operator()(const Eigen::VectorXd& x, Eigen::VectorXd& grad)
//     {
//         updateDesignParameter(x);

//         sim.resetScene();
//         VectorXT dq(sim.W.cols()); dq.setZero();
//         sim.forward(dq);
//         sim.Rods[5]->x(n0, x0);
//         sim.Rods[6]->x(n1, x1);

//         int nx = sim.deformed_states.rows();

//         StiffnessMatrix H;
//         sim.buildSystemDoFMatrix(dq, H);
//         VectorXT dOdx = computedOdx();
//         // std::cout << "|dOdx| " << dOdx.norm() << std::endl;
//         Eigen::SimplicialLLT<StiffnessMatrix> solver;
//         solver.compute(H);

//         VectorXT lambda = solver.solve(dOdx);
//         // std::cout << "|lambda| " << lambda.norm() << std::endl;
//         VectorXT dOdp(n); dOdp.setZero();

//         VectorXT dfdp_full(n * nx);
        
//         sim.parallelContactdfdp(dfdp_full);
//         dfdp_full *= -1.0;
//         //de/dx = -f
        
//         // std::cout << "|dfdp_full| " << dfdp_full.norm() << std::endl;
        
//         for (int i = 0; i < n; i++)
//         {
//             VectorXT dfdp = sim.W.transpose() * dfdp_full.segment(i * nx, nx);
            
//             grad[i] = -lambda.dot(dfdp);
//         }
//         T obj = 0.5 * ((x0 - x0_target).dot((x0 - x0_target)) + (x1 - x1_target).dot((x1 - x1_target))); 
//         std::cout << "E " << obj << std::endl;
//         return obj;
//     }
// };


// template<class T, int dim>
// void EoLRodSim<T, dim>::resetScene()
// { 
//     deformed_states = rest_states; 
//     for (auto& rod : Rods)
//     {
//         rod->reference_twist.setZero();
//         rod->reference_angles.setZero();
//     }
//     for (auto& crossing : rod_crossings)
//     {
//         crossing->omega.setZero();
//         crossing->rotation_accumulated.setIdentity();
//     }

//     for (auto& rod : Rods)
//     {
//         rod->setupBishopFrame();
//     }
// }

// template<class T, int dim>
// void EoLRodSim<T, dim>::inverse()
// {
//     if constexpr (dim == 3)
//     {
        
//         int n_design = 1;
        
//         std::cout << "#design parameters: " << n_design << std::endl;
//         VectorXT p(n_design);
//         p.setZero();
        
//         // int loop_cnt = 0;
//         // for (auto& crossing : rod_crossings)
//         // {
//         //     for (int i = 0; i < crossing->rods_involved.size(); i++)
//         //     {
//         //         p[loop_cnt++] = crossing->sliding_ranges[i][0];
//         //         p[loop_cnt++] = crossing->sliding_ranges[i][1];
                
//         //     }
            
//         // }
//         // std::cout << p.transpose() << std::endl;
        
//         int n0 = 29, n1 = 30;
//         TV x0, x1;
//         TV x0_target = TV(0.0377535,   0.0789261,  0);
//         TV x1_target = TV(0.0479829,    0.0789278, 0);

                

            
//         auto objective = [&]()
//         {
//             resetScene();
//             VectorXT dq(W.cols()); dq.setZero();
//             forward(dq);
//             Rods[5]->x(n0, x0);
//             Rods[6]->x(n1, x1);
//             return 0.5 * ((x0 - x0_target).dot((x0 - x0_target)) + (x1 - x1_target).dot((x1 - x1_target)));
//         };

//         auto computedOdx = [&]()
//         {
            
//             Rods[5]->x(n0, x0);
//             Rods[6]->x(n1, x1);
//             VectorXT dOdx = VectorXT::Zero(deformed_states.rows());
//             Offset offset0, offset1;
//             Rods[5]->getEntry(n0, offset0);
//             Rods[6]->getEntry(n1, offset1);
//             dOdx.template segment<dim>(offset0[0]) = x0 - x0_target;
//             dOdx.template segment<dim>(offset1[0]) = x1 - x1_target;
//             return W.transpose() * dOdx;
//         };

        

//         auto gradient = [&]()
//         {
//             resetScene();

//             VectorXT dq(W.cols()); dq.setZero();
//             forward(dq);
            
//             int nx = deformed_states.rows();

//             StiffnessMatrix H;
//             buildSystemDoFMatrix(dq, H);
//             VectorXT dOdx = computedOdx();
//             // std::cout << "|dOdx| " << dOdx.norm() << std::endl;
//             Eigen::SimplicialLLT<StiffnessMatrix> solver;
//             solver.compute(H);

//             VectorXT lambda = solver.solve(dOdx);
//             // std::cout << "|lambda| " << lambda.norm() << std::endl;
//             VectorXT dOdp(n_design); dOdp.setZero();

//             VectorXT dfdp_full(n_design * nx);
            
//             parallelContactdfdp(dfdp_full);
//             dfdp_full *= -1.0;
//             //de/dx = -f
            
//             // std::cout << "|dfdp_full| " << dfdp_full.norm() << std::endl;
            
//             for (int i = 0; i < n_design; i++)
//             {
//                 VectorXT dfdp = W.transpose() * dfdp_full.segment(i * nx, nx);
                
//                 dOdp[i] += -lambda.dot(dfdp);
//             }
//             // std::getchar();
//             return dOdp;

//         };

//         auto gradientSA = [&]()
//         {
//             resetScene();

//             VectorXT dq(W.cols()); dq.setZero();
//             forward(dq);
            
//             int nx = deformed_states.rows();

//             StiffnessMatrix H;
//             buildSystemDoFMatrix(dq, H);
//             VectorXT dOdx = computedOdx();
//             // std::cout << "|dOdx| " << dOdx.norm() << std::endl;
//             Eigen::SimplicialLLT<StiffnessMatrix> solver;
//             solver.compute(H);

            
//             VectorXT dOdp(n_design); dOdp.setZero();

//             VectorXT dfdp_full(n_design * nx);
            
//             parallelContactdfdp(dfdp_full);
//             dfdp_full *= -1.0;
            
//             Eigen::MatrixXd dXdu(W.cols(), n_design);
//             dXdu.setZero();
            
//             for (int i = 0; i < n_design; i++)
//             {
//                 VectorXT dfdp = W.transpose() * dfdp_full.segment(i * nx, nx);
                
//                 dXdu.col(i) += -solver.solve(dfdp);
//             }
//             // std::cout << dXdu << std::endl;
//             // Eigen::MatrixXd dXdu2 = dXdu.transpose() * dXdu;
//             Eigen::MatrixXd dXdu2 = dXdu * dXdu.transpose();

//             // std::cout << dXdu2 << std::endl;

//             Spectra::DenseSymMatProd<T> op(dXdu2);

//             int n_eigen = 20;
            
//             Spectra::SymEigsSolver< T, Spectra::LARGEST_ALGE, Spectra::DenseSymMatProd<T> > eigs(&op, n_eigen, dXdu2.rows());

//             eigs.init();
//             int nconv = eigs.compute();
//             Eigen::VectorXd eigen_values = eigs.eigenvalues().real();
//             Eigen::MatrixXd eigen_vectors = eigs.eigenvectors().real();

//             std::cout << "eigen values" << std::endl;
//             std::cout << eigen_values << std::endl;
//             // std::getchar();

//             // deformed_states = rest_states + eigen_vectors.col(3);
//             return dOdp;

//         };

//         // auto updateDesignParameter = [&](const VectorXT& new_p)
//         // {
//         //     int cnt = 0;
//         //     for (auto& crossing : rod_crossings)
//         //     {
//         //         if (crossing->is_fixed)
//         //         {
//         //             cnt += crossing->rods_involved.size() * 2;
//         //             continue;
//         //         }
//         //         for (int i = 0; i < crossing->rods_involved.size(); i++)
//         //         {
//         //             if (i == 0)
//         //                 crossing->sliding_ranges[i].setZero();
//         //             else
//         //             {
//         //                 crossing->sliding_ranges[i] = new_p.template segment<2>(cnt);
//         //             }
//         //             // crossing->sliding_ranges[i] = new_p.template segment<2>(cnt);
//         //             crossing->sliding_ranges[i][0] = std::max(0.0, crossing->sliding_ranges[i][0]);
//         //             crossing->sliding_ranges[i][1] = std::max(0.0, crossing->sliding_ranges[i][1]);
//         //             // std::cout << crossing->sliding_ranges[i].transpose() << std::endl;
//         //             cnt += 2;
//         //         }
//         //     }
//         // };

//         updateDesignParameter(p);

//         auto diffTest = [&]()
//         {
//             T epsilon = 1e-8;
//             VectorXT g = gradient();
            
//             T E0 = objective();
//             for (int i = 0; i < n_design; i++)
//             {
//                 if (!flag[i])
//                     continue;
//                 // std::cout << i << std::endl;
//                 p[i] += epsilon;
//                 updateDesignParameter(p);
//                 T E1 = objective();
                
//                 std::cout << "FD " << (E1 - E0) / epsilon << " symbolic " << g[i] << std::endl;
//                 // // std::getchar();
//                 p[i] -= epsilon;
//                 // std::cout << E1 << std::endl;
//             }
//         };

//         // diffTest();

//         // LBFGSParam<T> param;
//         // param.epsilon = 1e-4;
//         // param.max_iterations = 100;

//         // LBFGSSolver<T, LineSearchBracketing> solver(param);
//         // LBFGSWrapper<T, dim> fun(n_design, *this);
        
        
//         // T fx;
//         // VectorXT x = VectorXT::Constant(n_design, 0.0);
//         // int niter = solver.minimize(fun, x, fx);

//         // std::cout << niter << " iterations" << std::endl;
//         // std::cout << "x = \n" << x.transpose() << std::endl;
//         // std::cout << "f(x) = " << fx << std::endl;
//         // VectorXT g = gradientSA();
//         for (int iter = 0; iter < 10; iter++)
//         {
//             VectorXT g = gradient();
//             break;
//             std::cout << "|g| " << g.norm() << std::endl;
            
//             if (g.norm() < 1e-6)
//             {
//                 break;
//             }
//             T E0 = objective();
//             std::cout << "E " << E0 << std::endl;
//             T alpha = 1.0;
//             while (true)
//             {
//                 VectorXT p_ls = p - alpha * g;
//                 updateDesignParameter(p_ls);
//                 // std::getchar();
//                 T E1 = objective();
//                 // std::getchar();
//                 if (E1 < E0)
//                 {
//                     p = p_ls;
//                     break;
//                 }
//                 else
//                 {
//                     alpha *= 0.5;
//                 }
//             }
            

//         }
//         std::cout << p.transpose() << std::endl;
//     }
// }

template<class T, int dim>
class LBFGSWrapper
{
public:
    using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
    using TV = Vector<T, dim>;
    using Offset = Vector<int, dim + 1>;
    using StiffnessMatrix = Eigen::SparseMatrix<T>;
    using Entry = Eigen::Triplet<T>;
private:
    int n;
    EoLRodSim<T, dim>& sim;
    std::vector<int> handle_joints = { 5, 10, 15, 21, 22, 23};
    bool sliding = false;

    VectorXT computedOdx()
    {
        VectorXT dOdx = VectorXT::Zero(sim.W.cols());
        for (auto crossing : sim.rod_crossings)
        {
            Offset offset;
            sim.Rods[crossing->rods_involved.front()]->getEntryReduced(crossing->node_idx, offset);
            TV crossing_position;
            sim.getCrossingPosition(crossing->node_idx, crossing_position);

            T x = crossing_position[0], 
                y = crossing_position[1], 
                z = crossing_position[2];

            dOdx[offset[0]] += (x * x - y * y - z) * 2.0 * x;
            dOdx[offset[1]] += (x * x - y * y - z) * -2.0 * y;
            dOdx[offset[2]] += (x * x - y * y - z) * -1.0;
        }
        return dOdx;
    }

    void updateDesignParameter(const VectorXT& new_p)
    {
        int i = 0;
        for (int crossing_idx : handle_joints)
        {
            auto crossing = sim.rod_crossings[crossing_idx];
            Offset offset;
            sim.Rods[crossing->rods_involved.front()]->getEntryReduced(crossing->node_idx, offset);

            int offset_omega = crossing->reduced_dof_offset;

            if (sliding)
            {
                for (int d = 0; d < dim + 1; d++)
                {
                    sim.dirichlet_dof[offset[d]] = new_p[i++];
                }
            }
            else
            {
                for (int d = 0; d < dim; d++)
                {
                    sim.dirichlet_dof[offset[d]] = new_p[i++];
                }
                for (int d = 0; d < dim; d++)
                {
                    sim.dirichlet_dof[offset_omega + d] = new_p[i++];
                }
            }
        }
    }
    
public:
    LBFGSWrapper(int n_, EoLRodSim<T, dim>& _sim, bool _sliding) : n(n_), sim(_sim), sliding(_sliding) {}
    double operator()(const Eigen::VectorXd& x, Eigen::VectorXd& grad)
    {
        updateDesignParameter(x);

        sim.resetScene();
            
        VectorXT dq(sim.W.cols()); dq.setZero();
        sim.forward(dq);
        
        StiffnessMatrix H;

        std::vector<Entry> entry_K;
        sim.addStiffnessMatrix(entry_K, dq);

        StiffnessMatrix A(sim.deformed_states.rows(), sim.deformed_states.rows());
        A.setFromTriplets(entry_K.begin(), entry_K.end());
        H = sim.W.transpose() * A * sim.W;
        StiffnessMatrix dfdx = H;
        
        sim.projectDirichletDoFSystemMatrix(H);
        
        VectorXT dOdx = computedOdx();
        std::cout << "|dOdx| " << dOdx.norm() << std::endl;

        Eigen::SimplicialLLT<StiffnessMatrix> solver;
        solver.compute(H);

        if (solver.info() == Eigen::NumericalIssue)
        {
            std::cout << "not PD" << std::endl;
        }

        VectorXT lambda = solver.solve(-dOdx);
        std::cout << "|lambda| " << lambda.norm() << std::endl;
        VectorXT dOdp(n); dOdp.setZero();

        
        int i = 0;
        for (int crossing_idx : handle_joints)
        {
            auto crossing = sim.rod_crossings[crossing_idx];
            Offset offset;
            sim.Rods[crossing->rods_involved.front()]->getEntryReduced(crossing->node_idx, offset);

            int offset_omega = crossing->reduced_dof_offset;

            if (sliding)
            {
                for (int d = 0; d < dim + 1; d++)
                {
                    VectorXT dfdp = dfdx.col(offset[d]);
                    for (auto data : sim.dirichlet_dof)
                        dfdp[data.first] = 0;
                    dOdp[i++] += lambda.dot(dfdp);
                }
            }
            else
            {
                for (int d = 0; d < dim; d++)
                {
                    VectorXT dfdp = dfdx.col(offset[d]);
                    for (auto data : sim.dirichlet_dof)
                        dfdp[data.first] = 0;
                    
                    dOdp[i++] += lambda.dot(dfdp);
                }
                for (int d = 0; d < dim; d++)
                {
                    VectorXT dfdp = dfdx.col(offset_omega + d);
                    for (auto data : sim.dirichlet_dof)
                        dfdp[data.first] = 0;

                    dOdp[i++] += lambda.dot(dfdp);
                }
            }

        }
        
        grad = dOdp;

        T energy = 0.0;
        for (auto& crossing : sim.rod_crossings)
        {
            TV crossing_position;
            sim.getCrossingPosition(crossing->node_idx, crossing_position);

            T x = crossing_position[0], y = crossing_position[1], z = crossing_position[2];
            energy += 0.5 * std::pow( x * x - y * y - z, 2);
        }
        std::cout << "E " << energy << std::endl;
        return energy;
    }
};

template<class T, int dim>
void EoLRodSim<T, dim>::inverse()
{
    if constexpr (dim == 3)
    {
        bool sliding = true;
        bool u_only = true;
        auto system_dirichlet_dof = dirichlet_dof;

        int n_design = 0;
        TV target = TV(0.0133826, 0.0601781, 0.0324977);
        // TV target = TV(0.0, 1, 0.2) * unit;
        Mask mask(false, false, true);
        mask.setConstant(true);
        std::vector<int> handle_joints = { 5, 10, 15, 21, 22, 23};
        // std::vector<int> handle_joints = { 5, 10, 15, 21, 22, 23, 6, 7, 8, 11, 12, 13};
        // std::vector<int> handle_joints = {10, 22};
        std::unordered_map<int, int> sliding_rods;
        sliding_rods[10] = 0; 
        sliding_rods[5] = 0; 
        sliding_rods[15] = 0;
        sliding_rods[21] = 1; 
        sliding_rods[22] = 1;
        sliding_rods[23] = 1;
        // sliding_rods[6] = 1; 
        // sliding_rods[7] = 1;
        // sliding_rods[8] = 1;
        // sliding_rods[11] = 1; 
        // sliding_rods[12] = 1;
        // sliding_rods[13] = 1;

        if (sliding)
        {
            for (int idx : handle_joints)
                rod_crossings[idx]->is_fixed = false;
        }
        fixCrossing();
        for (int idx : handle_joints)
        {
            if (sliding)
            {
                if (u_only)
                    n_design += 1;
                else
                    n_design += 4;
            }
            else
                n_design += 6;
        }
        
        std::cout << "#design parameters: " << n_design << std::endl;
        VectorXT p(n_design);
        p.setZero();
        
        T w = 1e-2;
        
        auto objective = [&]()
        {
            resetScene();

            VectorXT dq(W.cols()); dq.setZero();
            forward(dq);
            
            T energy = 0.0;
            // for (auto& crossing : rod_crossings)
            // {
            //     TV crossing_position;
            //     getCrossingPosition(crossing->node_idx, crossing_position);

            //     T x = crossing_position[0], y = crossing_position[1], z = crossing_position[2];
            //     energy += 0.5 * std::pow( x * x - y * y - z, 2);
            // }
            for (auto& crossing : {rod_crossings[4]})
            {
                TV crossing_position;
                getCrossingPosition(crossing->node_idx, crossing_position);

                for (int d = 0; d < dim; d++)
                {
                    if (mask[d])
                        energy += 0.5 * std::pow(crossing_position[d] - target[d], 2);
                }
                
                // energy += 0.5 * (crossing_position - target).dot(crossing_position - target);
            }
            std::cout << "obj matching " << energy << std::endl;
            // energy += addStretchingEnergy() * w;
            return energy;
        };

        auto computedOdx = [&](VectorXT& dOdx)
        {
            // dOdx = VectorXT::Zero(W.cols());
            // for (auto crossing : rod_crossings)
            // {
            //     Offset offset;
            //     Rods[crossing->rods_involved.front()]->getEntryReduced(crossing->node_idx, offset);
            //     TV crossing_position;
            //     getCrossingPosition(crossing->node_idx, crossing_position);

            //     T x = crossing_position[0], 
            //       y = crossing_position[1], 
            //       z = crossing_position[2];

            //     dOdx[offset[0]] += (x * x - y * y - z) * 2.0 * x;
            //     dOdx[offset[1]] += (x * x - y * y - z) * -2.0 * y;
            //     dOdx[offset[2]] += (x * x - y * y - z) * -1.0;
            // }

            dOdx = VectorXT::Zero(W.cols());
            for (auto crossing : {rod_crossings[4]})
            {
                Offset offset;
                Rods[crossing->rods_involved.front()]->getEntryReduced(crossing->node_idx, offset);
                TV crossing_position;
                getCrossingPosition(crossing->node_idx, crossing_position);

                // dOdx.template segment<dim>(offset[0]) += (crossing_position - target);
                for (int d = 0; d < dim; d++)
                {
                    if (mask[d])
                        dOdx[offset[d]] += (crossing_position[d] - target[d]);
                }
            }
            VectorXT f(W.rows()); f.setZero();
            addStretchingForce(f);
            // dOdx += -W.transpose() * f * w;
            
        };


        auto gradient = [&]()
        {
            resetScene();
            
            VectorXT dq(W.cols()); dq.setZero();
            forward(dq);

            // TV c4;            
            // getCrossingPosition(4, c4);
            // std::cout << c4.transpose() << std::endl;

            StiffnessMatrix H;

            std::vector<Entry> entry_K;
            addStiffnessMatrix(entry_K, dq);

            StiffnessMatrix A(deformed_states.rows(), deformed_states.rows());
            A.setFromTriplets(entry_K.begin(), entry_K.end());
            H = W.transpose() * A * W;
            StiffnessMatrix dfdx = H;
            // projectDirichletDoFMatrix(dfdx, system_dirichlet_dof);
            projectDirichletDoFSystemMatrix(H);
            
            VectorXT dOdx; computedOdx(dOdx);
            std::cout << "|dOdx| " << dOdx.norm() << std::endl;

            Eigen::SimplicialLLT<StiffnessMatrix> solver;
            solver.compute(H);

            if (solver.info() == Eigen::NumericalIssue)
            {
                std::cout << "not PD" << std::endl;
            }

            VectorXT lambda = solver.solve(-dOdx);
            std::cout << "|lambda| " << lambda.norm() << std::endl;
            VectorXT dOdp(n_design); dOdp.setZero();

            
            int i = 0;
            for (int crossing_idx : handle_joints)
            {
                auto crossing = rod_crossings[crossing_idx];
                Offset offset;
                Rods[crossing->rods_involved.front()]->getEntryReduced(crossing->node_idx, offset);

                int offset_omega = crossing->reduced_dof_offset;

                if (sliding)
                {
                    if (u_only)
                    {
                        int sliding_rod = sliding_rods[crossing_idx];
                        VectorXT dfdp = dfdx.col(offset[dim]);
                        for (auto data : dirichlet_dof)
                            dfdp[data.first] = 0;
                        dOdp[i++] += lambda.dot(dfdp);
                    }
                    else
                    {
                        for (int d = 0; d < dim + 1; d++)
                        {
                            VectorXT dfdp = dfdx.col(offset[d]);
                            for (auto data : dirichlet_dof)
                                dfdp[data.first] = 0;
                            dOdp[i++] += lambda.dot(dfdp);
                        }
                    }
                }
                else
                {
                    for (int d = 0; d < dim; d++)
                    {
                        VectorXT dfdp = dfdx.col(offset[d]);
                        for (auto data : dirichlet_dof)
                            dfdp[data.first] = 0;
                        
                        dOdp[i++] += lambda.dot(dfdp);
                    }
                    for (int d = 0; d < dim; d++)
                    {
                        VectorXT dfdp = dfdx.col(offset_omega + d);
                        for (auto data : dirichlet_dof)
                            dfdp[data.first] = 0;

                        dOdp[i++] += lambda.dot(dfdp);
                    }
                }

            }
            
            return dOdp;

        };

        auto updateDesignParameter = [&](const VectorXT& new_p)
        {
            int i = 0;
            for (int crossing_idx : handle_joints)
            {
                auto crossing = rod_crossings[crossing_idx];
                Offset offset;
                Rods[crossing->rods_involved.front()]->getEntryReduced(crossing->node_idx, offset);

                int offset_omega = crossing->reduced_dof_offset;

                if (sliding)
                {
                    if (u_only)
                    {
                        dirichlet_dof[offset[dim]] = new_p[i++];
                    }
                    else
                    {
                        for (int d = 0; d < dim + 1; d++)
                        {
                            dirichlet_dof[offset[d]] = new_p[i++];
                        }
                    }
                }
                else
                {
                    for (int d = 0; d < dim; d++)
                    {
                        dirichlet_dof[offset[d]] = new_p[i++];
                    }
                    // if (crossing_idx == 10)
                    //     dirichlet_dof[offset[1]] = -0.1 * unit;
                    // if (crossing_idx == 22)
                    // {
                    //     dirichlet_dof[offset[0]] = 0.1 * unit;
                    //     dirichlet_dof[offset[2]] = 0.01 * unit;
                    // }
                    for (int d = 0; d < dim; d++)
                    {
                        dirichlet_dof[offset_omega + d] = new_p[i++];
                    }
                }
            }

        };

        auto gradientSA = [&]()
        {
            resetScene();
            
            VectorXT dq(W.cols()); dq.setZero();
            forward(dq);
            

            StiffnessMatrix H;

            std::vector<Entry> entry_K;
            addStiffnessMatrix(entry_K, dq);

            StiffnessMatrix A(deformed_states.rows(), deformed_states.rows());
            A.setFromTriplets(entry_K.begin(), entry_K.end());
            H = W.transpose() * A * W;
            StiffnessMatrix dfdx = H;
            // projectDirichletDoFMatrix(dfdx, system_dirichlet_dof);
            projectDirichletDoFSystemMatrix(H);
            
            VectorXT dOdx; computedOdx(dOdx);
            std::cout << "|dOdx| " << dOdx.norm() << std::endl;

            Eigen::SimplicialLLT<StiffnessMatrix> solver;
            solver.compute(H);

            if (solver.info() == Eigen::NumericalIssue)
            {
                std::cout << "not PD" << std::endl;
            }
            
            VectorXT dOdp(n_design); dOdp.setZero();

            
            Eigen::MatrixXd dXdu(W.cols(), n_design);
            dXdu.setZero();

            int i = 0;
            for (int crossing_idx : handle_joints)
            {
                auto crossing = rod_crossings[crossing_idx];
                Offset offset;
                Rods[crossing->rods_involved.front()]->getEntryReduced(crossing->node_idx, offset);

                int offset_omega = crossing->reduced_dof_offset;

                if (sliding)
                {
                    for (int d = 0; d < dim + 1; d++)
                    {
                        VectorXT dfdp = dfdx.col(offset[d]);
                        for (auto data : dirichlet_dof)
                            dfdp[data.first] = 0;
                        dXdu.col(i) += -solver.solve(dfdp);
                    }
                }
                else
                {
                    for (int d = 0; d < dim; d++)
                    {
                        VectorXT dfdp = dfdx.col(offset[d]);
                        for (auto data : dirichlet_dof)
                            dfdp[data.first] = 0;
                        
                        dXdu.col(i) += -solver.solve(dfdp);
                    }
                    for (int d = 0; d < dim; d++)
                    {
                        VectorXT dfdp = dfdx.col(offset_omega + d);
                        for (auto data : dirichlet_dof)
                            dfdp[data.first] = 0;

                        dXdu.col(i) += -solver.solve(dfdp);
                    }
                }

            }

            // std::cout << dXdu << std::endl;
            // Eigen::MatrixXd dXdu2 = dXdu.transpose() * dXdu;

            Eigen::MatrixXd dXdu2 = dXdu * dXdu.transpose();

            // std::cout << dXdu2 << std::endl;

            Spectra::DenseSymMatProd<T> op(dXdu2);

            int n_eigen = 50;
            
            Spectra::SymEigsSolver< T, Spectra::LARGEST_ALGE, Spectra::DenseSymMatProd<T> > eigs(&op, n_eigen, dXdu2.rows());

            eigs.init();
            int nconv = eigs.compute();
            Eigen::VectorXd eigen_values = eigs.eigenvalues().real();
            Eigen::MatrixXd eigen_vectors = eigs.eigenvectors().real();
            std::cout << "eigen values" << std::endl;
            std::cout << eigen_values << std::endl;
            // std::getchar();
            // deformed_states = rest_states + W * eigen_vectors.col(0);
            VectorXT du = dXdu.transpose() * eigen_vectors.col(0);
            updateDesignParameter(du);
            VectorXT ddq(W.cols());
            ddq.setZero();
            forward(ddq);
            return dOdp;

        };

       
        
        

        updateDesignParameter(p);

        auto diffTest = [&]()
        {
            T epsilon = 1e-6;
            VectorXT g = gradient();
            std::cout << "|g|: " << g.norm() << std::endl;
            
            T E0 = objective();
            for (int i = 0; i < n_design; i++)
            {
                p[i] += epsilon;
                updateDesignParameter(p);
                T E1 = objective();
                
                T FD = (E1 - E0) / epsilon;
                if (std::abs(FD) > 1e-6 && std::abs(g[i]) > 1e-6)
                {
                    std::cout << "dof " << i << " FD " << FD << " symbolic " << g[i] << std::endl;
                    std::getchar();
                }
                p[i] -= epsilon;
                // std::cout << E1 << std::endl;
            }
        };

        // diffTest();
        // gradientSA();
        // LBFGSParam<T> param;
        // param.epsilon = 1e-4;
        // param.max_iterations = 100;

        // LBFGSSolver<T, LineSearchBracketing> solver(param);
        // LBFGSWrapper<T, dim> fun(n_design, *this, false);
        
        // T fx;
        // VectorXT x = VectorXT::Constant(n_design, 0.0);
        // int niter = solver.minimize(fun, x, fx);

        // std::cout << niter << " iterations" << std::endl;
        // std::cout << "x = \n" << x.transpose() << std::endl;
        // std::cout << "f(x) = " << fx << std::endl;
        
        bool stuck = false;
        for (int iter = 0; iter < 200; iter++)
        {
            VectorXT g = gradient();
            std::cout << "|g| " << g.norm() << std::endl;
            // return;
            if (g.norm() < 1e-4 || stuck)
            {
                break;
            }
            T E0 = objective();
            // if (E0 < 1e-6)
            // {
            //     break;
            // }
            std::cout << "E " << E0 << std::endl;
            T alpha = 1.0;
            int ls_cnt = 0;
            while (true)
            {
                VectorXT p_ls = p - alpha * g;
                updateDesignParameter(p_ls);
                // std::getchar();
                T E1 = objective();
                // std::getchar();
                if (E1 < E0 || ls_cnt > 15)
                {
                    if (ls_cnt > 15)
                        stuck = true;
                    p = p_ls;
                    break;
                }
                else
                {
                    ls_cnt++;
                    alpha *= 0.5;
                }
            }
        
        }
        std::cout << p.transpose() << std::endl;
    }
}

template class EoLRodSim<double, 3>;
template class EoLRodSim<double, 2>;