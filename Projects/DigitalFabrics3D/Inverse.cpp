#include "EoLRodSim.h"

#include "LBFGS.h"

using namespace LBFGSpp;

#include <Spectra/SymEigsShiftSolver.h>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/DenseSymMatProd.h>


template<class T, int dim>
void EoLRodSim<T, dim>::inverse()
{
    if constexpr (dim == 3)
    {
        auto system_dirichlet_dof = dirichlet_dof;

        int n_row = 11, n_col = 11;
        int half_row = (n_row - 1) / 2;
        int half_col = (n_col - 1) / 2;

        int n_design = 0;
        TV target = targets[1];
     
        Mask mask(false, false, true);
        mask.setConstant(true);
        std::vector<std::pair<int, int>> handle_nodes;

        int tip_node_id = n_row - 1;

        for (int col = 1; col < n_col - 1; col++)
        {
            handle_nodes.push_back(std::make_pair(Rods[col]->indices.front(), col)); 
        }

        for (int row = 1; row < n_row - 1; row++)
        {
            handle_nodes.push_back(std::make_pair(Rods[n_col + row]->indices.back(), n_col + row)); 
        }

        // handle_nodes.push_back(std::make_pair(Rods[1]->indices.front(), 1));
        // handle_nodes.push_back(std::make_pair(Rods[2]->indices.front(), 2));
        // handle_nodes.push_back(std::make_pair(Rods[3]->indices.front(), 3));
        
        // handle_nodes.push_back(std::make_pair(Rods[6]->indices.back(), 6));
        // handle_nodes.push_back(std::make_pair(Rods[7]->indices.back(), 7));
        // handle_nodes.push_back(std::make_pair(Rods[8]->indices.back(), 8));
        
        VectorXT dq_initialization(W.cols());
        dq_initialization.setZero();

        VectorXT backup_twist, backup_theta, backup_rotations, backup_prev_tangents, backup_u;
        
        auto backUpState = [&]()
        {
            backup_twist.resize(0);
            backup_rotations.resize(0);
            backup_prev_tangents.resize(0);
            backup_u.resize(0);
            backup_theta.resize(0);

            for (auto& rod : Rods)
            {
                int current_row = backup_twist.rows();
                backup_twist.conservativeResize(current_row + rod->numSeg());
                backup_twist.segment(current_row, rod->numSeg()) = rod->reference_twist;
                backup_theta.conservativeResize(current_row + rod->numSeg());
                backup_theta.segment(current_row, rod->numSeg()) = rod->reference_angles;

                current_row = backup_prev_tangents.rows();
                backup_prev_tangents.conservativeResize(current_row + rod->numSeg() * dim);
                backup_u.conservativeResize(current_row + rod->numSeg() * dim);
                
                for (int i = 0; i < rod->numSeg(); i++)
                {
                    backup_prev_tangents.template segment<dim>(current_row + i * dim) = rod->prev_tangents[i];
                    backup_u.template segment<dim>(current_row + i * dim) = rod->reference_frame_us[i];
                }
                
            }

            for (auto& crossing : rod_crossings)
            {
                int current_row = backup_rotations.rows();
                backup_rotations.conservativeResize(current_row + 9);
                backup_rotations.segment(current_row, 9) = Eigen::Map<VectorXT>(crossing->rotation_accumulated.data(), 
                    crossing->rotation_accumulated.size());
            }
        };

        auto recoverState = [&]()
        {
            int current_row_twist = 0;
            int current_row_TV = 0;
            for (auto& rod : Rods)
            {
                
                rod->reference_twist = backup_twist.segment(current_row_twist, rod->numSeg());
                rod->reference_angles = backup_theta.segment(current_row_twist, rod->numSeg());
                current_row_twist += rod->numSeg();
                
                for (int i = 0; i < rod->numSeg(); i++)
                {
                    rod->prev_tangents[i] = backup_prev_tangents.template segment<dim>(current_row_TV + i * dim);
                    rod->reference_frame_us[i] = backup_u.template segment<dim>(current_row_TV + i * dim);
                }
                current_row_TV += rod->numSeg() * dim;
            }
            int current_row_rotation = 0;
            for (auto& crossing : rod_crossings)
            {
                Eigen::Map<VectorXT>(crossing->rotation_accumulated.data(), 
                    crossing->rotation_accumulated.size()) = backup_rotations.segment(current_row_rotation, 9);
                current_row_rotation += 9;
            }
        };

        
        

        for (auto data : handle_nodes)
        {
            n_design += 3;
        }
        
        std::cout << "#design parameters: " << n_design << std::endl;
        VectorXT p(n_design);
        p.setZero();
        backUpState();
        T w = 1e-6;
        
        auto objective = [&]()
        {
            resetScene();
            recoverState();
            VectorXT dq(W.cols()); dq.setZero();
            bool converged = forward(dq);
            if (!converged)
                return 1e10;
            T energy = 0.0;
            
            for (auto& crossing : {rod_crossings[tip_node_id]})
            {
                TV crossing_position;
                getCrossingPosition(crossing->node_idx, crossing_position);

                for (int d = 0; d < dim; d++)
                {
                    if (mask[d])
                        energy += 0.5 * std::pow(crossing_position[d] - target[d], 2);
                }
            }
            T obj_matching = energy;
            std::cout << "\t\tobj matching " << energy << std::endl;
            // energy += addStretchingEnergy() * w;
            energy += computeTotalEnergy(dq) * w;
            std::cout << "\t\tstretching " << energy - obj_matching<< std::endl;
            return energy;
        };

        auto computedOdx = [&](VectorXT& dOdx, const VectorXT& dq)
        {
        
            dOdx = VectorXT::Zero(W.cols());
            for (auto crossing : {rod_crossings[tip_node_id]})
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
            // VectorXT f(W.rows()); f.setZero();
            // addStretchingForce(f);
            // dOdx += -W.transpose() * f * w;

            // VectorXT f(W.rows()); f.setZero();
            // computeResidual(f);
            // dOdx += -W.transpose() * f * w;
            VectorXT f(W.cols());
            f.setZero();
            computeResidual(f, dq);
            
            dOdx += -f * w;
            
        };

        auto computed2Odx2 = [&](StiffnessMatrix& d2Odx2, const VectorXT& dq)
        {
            StiffnessMatrix hessian(W.rows(), W.rows());
            std::vector<Entry> hessian_entries;
            for (auto crossing : {rod_crossings[tip_node_id]})
            {
                Offset offset;
                Rods[crossing->rods_involved.front()]->getEntry(crossing->node_idx, offset);
                TV crossing_position;
                getCrossingPosition(crossing->node_idx, crossing_position);

                for (int d = 0; d < dim; d++)
                {
                    if (mask[d])
                        hessian_entries.push_back(Entry(offset[d], offset[d], 1.0));
                }
            }
            // addStretchingK(hessian_entries);
            addStiffnessMatrix(hessian_entries, dq);
            hessian.setFromTriplets(hessian_entries.begin(), hessian_entries.end());

            d2Odx2 = w * W.transpose() * hessian * W;
            // d2Odx2 = hessian;
        };


        auto gradient = [&]()
        {
            resetScene();
            recoverState();
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
            
            VectorXT dOdx; computedOdx(dOdx, dq);
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
            for (std::pair<int, int> idx_rod_idx : handle_nodes)
            {
                Offset offset;
                Rods[idx_rod_idx.second]->getEntryReduced(idx_rod_idx.first, offset);
                for (int d = 0; d < dim; d++)
                {
                    VectorXT dfdp = dfdx.col(offset[d]);
                    for (auto data : dirichlet_dof)
                        dfdp[data.first] = 0;
                    
                    dOdp[i++] += lambda.dot(dfdp);
                }
            }
            return dOdp;

        };

        auto updateDesignParameter = [&](const VectorXT& new_p)
        {
            int i = 0;
            for (std::pair<int, int> idx_rod_idx : handle_nodes)
            {
                Offset offset;
                Rods[idx_rod_idx.second]->getEntryReduced(idx_rod_idx.first, offset);
                for (int d = 0; d < dim; d++)
                {
                    dirichlet_dof[offset[d]] = new_p[i++];
                }
            }
            
        };

        auto GaussNewtonStep = [&](StiffnessMatrix& H_GN)
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
            projectDirichletDoFSystemMatrix(H);
            H.makeCompressed();
            VectorXT dOdx; computedOdx(dOdx, dq);
            // std::cout << "|dOdx| " << dOdx.norm() << std::endl;

            Eigen::SimplicialLLT<StiffnessMatrix> solver;
            solver.compute(H);

            if (solver.info() == Eigen::NumericalIssue)
            {
                std::cout << "not PD" << std::endl;
            }

            StiffnessMatrix dXdu(W.cols(), n_design);
            // dXdu.setZero();
            std::vector<Entry> dXdu_entry;

            int i = 0;
            for (std::pair<int, int> idx_rod_idx : handle_nodes)
            {
                Offset offset;
                Rods[idx_rod_idx.second]->getEntryReduced(idx_rod_idx.first, offset);
                for (int d = 0; d < dim; d++)
                {
                    VectorXT dfdp = dfdx.col(offset[d]);
                    for (auto data : dirichlet_dof)
                        dfdp[data.first] = 0;
                    
                    VectorXT dXdu_col_i = -solver.solve(dfdp);
                    for (int j = 0; j < dXdu_col_i.rows(); j++)
                    {
                        if (std::abs(dXdu_col_i[j]) > 1e-10)
                            dXdu_entry.push_back(Entry(j, i, dXdu_col_i[j]));
                    }
                    i++;
                }
            }

            
            dXdu.setFromTriplets(dXdu_entry.begin(), dXdu_entry.end());
            
            StiffnessMatrix d2Odx2(W.cols(), W.cols());
            computed2Odx2(d2Odx2, dq);

            H_GN = dXdu.transpose() * d2Odx2 * dXdu;

        };

        auto searchDirection = [&](VectorXT& dp)
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
            projectDirichletDoFSystemMatrix(H);
            H.makeCompressed();
            VectorXT dOdx; computedOdx(dOdx, dq);

            Eigen::SimplicialLLT<StiffnessMatrix> solver;
            solver.compute(H);

            VectorXT lambda = solver.solve(-dOdx);
            
            VectorXT dOdp(n_design); dOdp.setZero();

            StiffnessMatrix dXdu(W.cols(), n_design);
            
            std::vector<Entry> dXdu_entry;

            int i = 0;
            for (std::pair<int, int> idx_rod_idx : handle_nodes)
            {
                Offset offset;
                Rods[idx_rod_idx.second]->getEntryReduced(idx_rod_idx.first, offset);
                for (int d = 0; d < dim; d++)
                {
                    VectorXT dfdp = dfdx.col(offset[d]);
                    for (auto data : dirichlet_dof)
                        dfdp[data.first] = 0;
                    
                    VectorXT dXdu_col_i = -solver.solve(dfdp);
                    for (int j = 0; j < dXdu_col_i.rows(); j++)
                    {
                        if (std::abs(dXdu_col_i[j]) > 1e-10)
                            dXdu_entry.push_back(Entry(j, i, dXdu_col_i[j]));
                    }
                    dOdp[i++] += lambda.dot(dfdp);
                }
            }
            std::cout << "|g| " << dOdp.norm() << std::endl;
            if (dOdp.norm() < 1e-6)
            {
                
                return false;
            }

            dXdu.setFromTriplets(dXdu_entry.begin(), dXdu_entry.end());
            
            StiffnessMatrix d2Odx2(W.cols(), W.cols());
            computed2Odx2(d2Odx2, dq);

            StiffnessMatrix H_GN = dXdu.transpose() * d2Odx2 * dXdu;
                
            solver.compute(H_GN);
            T mu = 10e-6;
        
            while (solver.info() == Eigen::NumericalIssue)
            {
                for (int j = 0; j < H_GN.rows(); j++)
                {
                    H_GN.coeffRef(j, j) += mu;
                }
                mu *= 10.0;
                solver.compute(H_GN);
            }

            dp = solver.solve(-dOdp);
            return true;
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

        auto diffTest2ndOrder = [&]()
        {
            
            VectorXT g = gradient();

            VectorXT dx(n_design);
            dx.setRandom();
            dx *= 1.0 / dx.norm();
            dx *= 1e-4;
            T E0 = objective();
            T previous = 0.0;
            for (int i = 0; i < 10; i++)
            {
                updateDesignParameter(p + dx);
                T E1 = objective();
                T dE = E1 - E0;
                
                dE -= g.dot(dx);
                // std::cout << "dE " << dE << std::endl;
                if (i > 0)
                {
                    std::cout << (previous/dE) << std::endl;
                }
                previous = dE;
                dx *= 0.5;
            }

        };

        auto gradientDescent = [&]()
        {
            bool stuck = false;
            for (int iter = 0; iter < 10; iter++)
            {
                VectorXT g = gradient();
                std::cout << "|g| " << g.norm() << std::endl;
                
                if (g.norm() < 1e-4 || stuck)
                {
                    break;
                }
                T E0 = objective();
                
                std::cout << "\tE " << E0 << std::endl;
                T alpha = 1.0;
                int ls_cnt = 0;
                while (true)
                {
                    VectorXT p_ls = p - alpha * g;
                    updateDesignParameter(p_ls);
                    // std::getchar();
                    T E1 = objective();
                    // std::getchar();
                    if (E1 < E0 || ls_cnt > 20)
                    {
                        dq_initialization = W.transpose() * (deformed_states - rest_states);
                        perturb = dq_initialization;
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
            std::cout << "stuck " << stuck << std::endl;
            std::cout << p.transpose() << std::endl;
        };

        auto gaussNewton = [&]()
        {
            for (int iter = 0; iter < 40; iter++)
            {
                VectorXT g = gradient();
                std::cout << "|g| " << g.norm() << std::endl;
             
                if (g.norm() < 1e-6)
                {
                    break;
                }
                StiffnessMatrix H_GN;
                GaussNewtonStep(H_GN);

                Eigen::SimplicialLLT<StiffnessMatrix> solver;
                solver.compute(H_GN);
                T mu = 10e-6;
                StiffnessMatrix I(H_GN.rows(), H_GN.cols());
                I.setIdentity();

                StiffnessMatrix H = H_GN;
                while (solver.info() == Eigen::NumericalIssue)
                {
                    H = H_GN + mu * I;
                    mu *= 10.0;
                    solver.compute(H);
                }

                VectorXT dp = solver.solve(-g);
                // if (dp.norm() > T(1) * n_design)
                //     dp = dp / dp.norm() * n_design;
                std::cout << "|dp| " << dp.norm() << std::endl;
                std::cout << "\tdot " << dp.normalized().dot(-g.normalized()) << std::endl;
                T E0 = objective();
                // std::cout << "E " << E0 << std::endl;
                T alpha = 1.0;
                int ls_cnt = 0;
                while (true)
                {
                    VectorXT p_ls = p + alpha * dp;
                    updateDesignParameter(p_ls);
                    
                    T E1 = objective();
                    
                    if (E1 < E0 || ls_cnt > 15)
                    {
                        if (ls_cnt > 15)
                        {
                            p = p_ls;
                            break;
                        }
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
        };

        auto gaussNewtonFast = [&]()
        {
            for (int iter = 0; iter < 10; iter++)
            {
                VectorXT dp;
                if (searchDirection(dp))
                {
                    std::cout << "|dp| " << dp.norm() << std::endl;
                    T E0 = objective();
                    // std::cout << "E " << E0 << std::endl;
                    T alpha = 1.0;
                    int ls_cnt = 0;
                    while (true)
                    {
                        VectorXT p_ls = p + alpha * dp;
                        updateDesignParameter(p_ls);
                        
                        T E1 = objective();
                        
                        if (E1 < E0 || ls_cnt > 15)
                        {
                            dq_initialization = W.transpose() * (deformed_states - rest_states);
                            perturb = dq_initialization;
                            backUpState();
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
                else
                {
                    break;
                }
            }
            std::cout << "max iter reached" << std::endl;
        };

        // gradientDescent();
        // gaussNewton();
        gaussNewtonFast();
        // diffTest();
        // std::cout << "|u| " << p.norm() << std::endl;
    }
}

template class EoLRodSim<double, 3>;
template class EoLRodSim<double, 2>;