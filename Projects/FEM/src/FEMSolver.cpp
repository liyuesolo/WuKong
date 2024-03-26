#include "../include/FEMSolver.h"
#include "../include/autodiff/FEMEnergy.h"
#include "../include/Timer.h"

#include "../Solver/CHOLMODSolver.hpp"
#include <Eigen/CholmodSupport>

#include <Spectra/SymEigsShiftSolver.h>
#include <Spectra/MatOp/SparseSymShiftSolve.h>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>

#include <fstream>
#include <iomanip>

template <int dim>
T FEMSolver<dim>::computeInteralEnergy(const VectorXT& _u)
{
    VectorXT projected = _u;
    if (!run_diff_test)
    {
        iterateDirichletDoF([&](int offset, T target)
        {
            projected[offset] = target;
        });
    }
    deformed = undeformed + projected;

    VectorXT energies_neoHookean = VectorXT::Zero(num_ele);

    iterateElementParallel([&](const EleNodes& x_deformed, 
        const EleNodes& x_undeformed, const EleIdx& indices, int tet_idx)
    {
        T ei;
        if constexpr (dim == 2)
        {
            computeLinear2DNeoHookeanEnergy(E, nu, x_deformed, x_undeformed, ei);
        }
        else if constexpr (dim == 3)
            computeLinearTet3DNeoHookeanEnergy(E, nu, x_deformed, x_undeformed, ei);
        energies_neoHookean[tet_idx] += ei;
    });

    return energies_neoHookean.sum();
}

template <int dim>
T FEMSolver<dim>::computeTotalEnergy(const VectorXT& _u)
{
    T total_energy = 0.0;

    VectorXT projected = _u;
    if (!run_diff_test)
    {
        iterateDirichletDoF([&](int offset, T target)
        {
            projected[offset] = target;
        });
    }
    deformed = undeformed + projected;

    if(use_elasticity){
        T e_NH = 0.0;
        addElastsicPotential(e_NH);
        total_energy += e_NH;

        std::cout<<"Neo-Hookean Energy: "<<e_NH<<" ";
    }
    

    if (use_penalty)
    {
        T penalty_energy = 0.0;
        addBCPenaltyEnergy(penalty_energy);
        total_energy += penalty_energy;
    }

    total_energy -= _u.dot(f);

    if(USE_IPC_3D)
    {
        T contact_energy = 0.0;
        addIPC3DEnergy(contact_energy);
        std::cout<<"contact energy: "<<contact_energy<<" ";
        total_energy += contact_energy;
    }
    else if(use_ipc_2d)
    {
        T contact_energy = 0.0;
        if(USE_MORTAR_METHOD)
            addMortarEnergy(contact_energy);
        else if(USE_TRUE_IPC_2D)
            addIPC2DtrueEnergy(contact_energy);
        else
        {
            if(!USE_NEW_FORMULATION)
            {
                if(USE_RIMLS) addFastRIMLSSCEnergy(contact_energy);
                else addFastIMLSSCEnergy(contact_energy);
            }
                
            else
            {
                T temp;
                addFastIMLSEnergy(temp);
                addFastIMLS12Energy(contact_energy);
            }
                
        }
            
        std::cout<<"contact energy: "<<contact_energy<<" ";
        total_energy += contact_energy;
    }

    // if(use_pos_penalty)
    // {
    //     T pospen_energy = 0.0;
    //     addPosPenaltyEnergy(pospen_energy);
    //     total_energy += pospen_energy;

    //     std::cout<<"Pos Pen Energy: "<<pospen_energy<<std::endl;
    // }

    // if(use_PBC_penalty)
    // {
    //     T PBC_energy = 0.0;
    //     addPBCPenaltyEnergy(PBC_energy);
    //     total_energy += PBC_energy;
    // }

    // if(use_virtual_spring)
    // {
    //     T virtual_spring_energy = 0.0;
    //     addVirtualSpringEnergy(virtual_spring_energy);
    //     total_energy += virtual_spring_energy;
    // }

    // if(use_rotational_penalty)
    // {
    //     T rotatinal_energy = 0.0;
    //     addRotationalPenaltyEnergy(rotatinal_energy);
    //     total_energy += rotatinal_energy;
    // }

    if(USE_NEW_FORMULATION)
    {
        //addL2DistanceEnergy(total_energy);
        T L2_energy = 0.0;
        addL2CorrectEnergy(L2_energy);
        total_energy += L2_energy;
        std::cout<<"L2 Energy: "<<L2_energy<<std::endl;

        T IMLSSameSide_energy = 0.0;
        //addIMLSPenEnergy(IMLSSameSide_energy);
        addFastIMLSSameSideEnergy(IMLSSameSide_energy);
        total_energy += IMLSSameSide_energy;
        std::cout<<"IMLS Pen Energy: "<<IMLSSameSide_energy<<std::endl;
    }

    // if(SLIDING_TEST)
    // {
    //     T spring_energy = 0.0;
    //     addSlidingSpringEnergy(spring_energy);
    //     total_energy += spring_energy;
    // }

    // if (false)
    // {
    //     T spring_energy = 0.0;
    //     addVirtualSpringEnergy2(spring_energy);
    //     total_energy += spring_energy;
    //     //std::cout<<"Spring Energy: "<<spring_energy<<std::endl;
    // }

    if(USE_DYNAMICS)
    {
        VectorXT a = get_a(deformed);
        T dynamics_energy = (deformed.transpose()*M*deformed-2*deformed.transpose()*M*(x_prev+h*v_prev))(0)/(2*pow(h,2));
        total_energy += dynamics_energy;
        std::cout<<"Dynamics Energy: "<<dynamics_energy<<" ";

        if(USE_FRICTION&&USE_IMLS)
        {
            T friction_energy = 0;
            addFrictionEnergy(friction_energy);
            total_energy += friction_energy;
            std::cout<<"Friction Energy: "<<friction_energy<<" ";
        }
    }
    std::cout<<std::endl;

    if(USE_SHELL)
    {
        T shell_energy = 0;
        addShellEnergy(shell_energy);
        total_energy += shell_energy;
        addUnilateralQubicPenaltyEnergy(bar_param, total_energy);
        std::cout<<"shell energy: "<<total_energy<<std::endl;
    }
    
    //std::cout<<"total energy: "<<total_energy<<std::endl;
    return total_energy;
}

template <int dim>
T FEMSolver<dim>::computeResidual(const VectorXT& _u, VectorXT& residual, double re)
{
    
    VectorXT projected = _u;

    if (!run_diff_test)
    {
        iterateDirichletDoF([&](int offset, T target)
        {
            projected[offset] = target;
        });
    }

    // if (use_ipc_2d)
    //     updateIPC2DVertices(_u);

    deformed = undeformed + projected;

    //std::cout<<deformed<<std::endl;

    residual = f;
    
    std::cout << "external force " << (residual).norm() << std::endl;
    VectorXT residual_backup = residual;
        

    if(use_elasticity)
        addElasticForceEntries(residual);

    std::cout << "elastic force " << (residual - residual_backup).norm() << std::endl;
    std::cout << "force1 " << (residual).norm() << std::endl;


    if (use_penalty)
        addBCPenaltyForceEntries(residual);

    if(USE_IPC_3D)
    {
        residual_backup = residual;
        addIPC3DForceEntries(residual);
        std::cout << "contact + penalty force " << (residual-residual_backup).norm() << std::endl;
    }
        
    else if(use_ipc_2d)
    {
        residual_backup = residual;
        VectorXT residual_ipc(deformed.rows());
        if(USE_MORTAR_METHOD)
            addMortarForceEntries(residual);
        else if(USE_TRUE_IPC_2D)
            addIPC2DtrueForceEntries(residual);
        else
        {
            if(!USE_NEW_FORMULATION)    
            {
                residual_backup = residual;
                if(USE_RIMLS) addFastRIMLSSCForceEntries(residual);
                else addFastIMLSSCForceEntries(residual);
                //if(USE_FRICTION)
                //{
                    //prev_contact_force = residual-residual_backup;
                //}
            }
            else
            {
                T temp;
                addFastIMLSEnergy(temp);
                addFastIMLS12ForceEntries(residual);
            }
                
        }
            
        //addIPC2DForceEntries(residual_ipc);
        std::cout << "contact + penalty force " << (residual-residual_backup).norm() << std::endl;
    }
     std::cout << "force2 " << (residual).norm() << std::endl;

    // residual_backup = residual;
    // if(use_pos_penalty)
    // {
    //     addPosPenaltyForceEntries(residual);
    //     std::cout << "Pospen force " << (residual - residual_backup).norm() << std::endl;
    // }
    // //std::cout << "Pospen force " << (residual - residual_backup).norm() << std::endl;

    // if(use_PBC_penalty)
    // {
    //     addPBCPenaltyForceEntries(residual);
    // }

    // if(use_virtual_spring)
    // {
    //     addVirtualSpringForceEntries(residual);
    // }

    // if(use_rotational_penalty)
    // {
    //     addRotationalPenaltyForceEntries(residual);
    // }

    residual_backup = residual;
    if(USE_NEW_FORMULATION)
    {
        //addL2DistanceForceEntries(residual);
        addL2CorrectForceEntries(residual);
        std::cout << "L2 force " << (residual - residual_backup).norm() << std::endl;
        
        // for(int i=0; i<num_nodes; ++i)
        // {
        //     std::cout<<"Original Dof"<<std::endl;
        //     std::cout<<deformed.segment<dim>(dim*i).transpose()<<std::endl;
        //     std::cout<<"Additional Dof"<<std::endl;
        //     std::cout<<deformed.segment<dim>(dim*(i+num_nodes)).transpose()<<std::endl;
        // }
        residual_backup = residual;
        //addIMLSPenForceEntries(residual);
        addFastIMLSSameSideForceEntries(residual);
        std::cout << "IMLSPen force " << (residual - residual_backup).norm() << std::endl;
    }
     std::cout << "force3 " << (residual).norm() << std::endl;

    // if(SLIDING_TEST)
    // {
    //     residual_backup = residual;
    //     addSlidingSpringForceEntries(residual);
    //     //std::cout << "spring force " << (residual - residual_backup).norm() << std::endl;
    // }

    // if (false)
    // {
    //     residual_backup = residual;
    //     addVirtualSpringForceEntries2(residual);
    //     //std::cout << "spring force " << (residual - residual_backup).norm() << std::endl;
    // }
    if(USE_DYNAMICS)
    {
        VectorXT dynamics_grad = M*(2*deformed-2*(x_prev+v_prev*h))/(2*pow(h,2));
        residual.segment(0, num_nodes * dim) -= dynamics_grad.segment(0, num_nodes * dim);

        if(USE_FRICTION&&USE_IMLS)
        {
            residual_backup = residual;
            addFrictionForceEntries(residual);
            std::cout << "Friction force: " << (residual - residual_backup).norm()<<" Normal force: "<<prev_contact_force.norm()<<" mu: "<<friction_mu<< std::endl;
        }
    }

    if(USE_SHELL)
    {
        addShellForceEntries(residual);
        addUnilateralQubicPenaltyForceEntries(bar_param,residual);
    }

    // std::getchar();
    if (!run_diff_test)
        iterateDirichletDoF([&](int offset, T target)
        {
            residual[offset] = 0;
        });

        
    return residual.norm();
}

template <int dim>
void FEMSolver<dim>::reset()
{
    deformed = undeformed;
    u.setZero();
    
    int num_nodes_all = num_nodes;
    //if(CALCULATE_IMLS_PROJECTION) num_nodes_all += projectedPts.rows();
    if(USE_NEW_FORMULATION)  num_nodes_all += additional_dof;
    ipc_vertices.resize(num_nodes_all, dim);
    for (int i = 0; i < num_nodes_all; i++)
        ipc_vertices.row(i) = undeformed.segment<dim>(i * dim);
        
    num_ipc_vtx = ipc_vertices.rows();
}

template <int dim>
void FEMSolver<dim>::buildSystemMatrix(const VectorXT& _u, StiffnessMatrix& K)
{
    VectorXT projected = _u;
    if (!run_diff_test)
    {
        iterateDirichletDoF([&](int offset, T target)
        {
            projected[offset] = target;
        });
    }
    deformed = undeformed + projected;
    
    std::vector<Entry> entries;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    if(use_elasticity)
        addElasticHessianEntries(entries);

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    //std::cout << "Time difference (elastic hessian) = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;

    if (use_penalty)
        addBCPenaltyHessianEntries(entries);
    

    begin = std::chrono::steady_clock::now();
    if(USE_IPC_3D)
        addIPC3DHessianEntries(entries, project_block_PD);
    else if (use_ipc_2d)
    {
        if(USE_MORTAR_METHOD)
            addMortarHessianEntries(entries, project_block_PD);
        else if(USE_TRUE_IPC_2D)
            addIPC2DtrueHessianEntries(entries, project_block_PD);
        else
        {
            if(!USE_NEW_FORMULATION)
            {
                if(USE_RIMLS) addFastRIMLSSCHessianEntries(entries, project_block_PD);
                else addFastIMLSSCHessianEntries(entries, project_block_PD);
                // UV.resize(dim*num_nodes,1);
                // VectorXa temp(dim*num_nodes); temp.setZero();
                // addFastIMLSSCForceEntries(temp);
                // UV.col(0) = temp;
            }else
            {
                T temp;
                addFastIMLSEnergy(temp);
                addFastIMLS12HessianEntries(entries, project_block_PD);
            }
        }
            
    }
    end = std::chrono::steady_clock::now();
    //std::cout << "Time difference (Mortar hessian) = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;

    begin = std::chrono::steady_clock::now();
    // if(use_pos_penalty)
    // {
    //     addPosPenaltyHessianEntries(entries, project_block_PD);
    // }

    // if(use_PBC_penalty)
    // {
    //     addPBCPenaltyHessianEntries(entries, project_block_PD);
    // }

    // if(use_virtual_spring)
    // {
    //     addVirtualSpringHessianEntries(entries, project_block_PD);
    // }

    // if(use_rotational_penalty)
    // {
    //     addRotationalPenaltyHessianEntries(entries, project_block_PD);
    // }

    if(USE_NEW_FORMULATION)
    {
        //addL2DistanceHessianEntries(entries, project_block_PD);
        addL2CorrectHessianEntries(entries, project_block_PD);
        //addIMLSPenHessianEntries(entries, project_block_PD);
        addFastIMLSSameSideHessianEntries(entries, project_block_PD);
    }

    // if(SLIDING_TEST)
    // {
    //     addSlidingSpringHessianEntries(entries, project_block_PD);
    // }

    // if(false)
    // {
    //     addVirtualSpringHessianEntries2(entries, project_block_PD);
    // }

    if(USE_DYNAMICS)
    {
        constructMassMatrix(entries,project_block_PD,true);
        if(USE_FRICTION&&USE_IMLS)
        {
            addFrictionHessianEntries(entries,project_block_PD);
        }
    }

    if(USE_SHELL)
    {
        addShellHessianEntries(entries, project_block_PD);
        addUnilateralQubicPenaltyHessianEntries(bar_param,entries);
    }

    K.setFromTriplets(entries.begin(), entries.end());
    if (!run_diff_test)
        projectDirichletDoFMatrix(K, dirichlet_data);
    
    K.makeCompressed();
    end = std::chrono::steady_clock::now();
    //std::cout<<K<<std::endl;
    //std::cout << "Time difference (rest hessian) = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
}

template <int dim>
void FEMSolver<dim>::projectDirichletDoFMatrix(StiffnessMatrix& A, const std::unordered_map<int, T>& data)
{
    for (auto iter : data)
    {
        A.row(iter.first) *= 0.0;
        A.col(iter.first) *= 0.0;
        A.coeffRef(iter.first, iter.first) = 1.0;
    }

}

template <int dim>
bool FEMSolver<dim>::linearSolve(StiffnessMatrix& K, 
    VectorXT& residual, VectorXT& du)
{

    Timer t(true);
    // StiffnessMatrix I(K.rows(), K.cols());
    // I.setIdentity();
    // K = I;
    T alpha = 10e-6;
    // StiffnessMatrix H = K;
    std::unordered_map<int, int> diagonal_entry_location;
    
    int indefinite_count_reg_cnt = 0, invalid_search_dir_cnt = 0, invalid_residual_cnt = 0;
    
    //Noether::CHOLMODSolver<typename StiffnessMatrix::StorageIndex> cholmod_solver;
    Eigen::CholmodSupernodalLLT<StiffnessMatrix> cholmod_solver;
    cholmod_solver.analyzePattern(K);

    for (int i = 0; i < 50; i++)
    {
        cholmod_solver.factorize(K);
        if (cholmod_solver.info() == Eigen::NumericalIssue)
        {
            std::cout << "indefinite" << std::endl;
            indefinite_count_reg_cnt++;
            // tbb::parallel_for(0, (int)K.rows(), [&](int row)
            // {
            //     K.coeffRef(row, row) += alpha;
            // });  
            //std::cout<<K.rows()<<std::endl;
            for(int row = 0; row<K.rows(); ++row)
            {
                //std::cout<<K.coeffRef(row, row)<<" "<<row<<" out of "<<K.rows()<<std::endl;
                K.coeffRef(row, row) += alpha;
                
            }
            std::cout<<alpha<<std::endl;  
            //K.diagonal().array() += alpha;
            alpha *= 10;
            // cholmod_solver.A->x = K.valuePtr();
            continue;
        }
        
        if(!use_sherman_morrison)
            du = cholmod_solver.solve(residual);
        else if (UV.cols() == 1)
        {
            VectorXT v = UV.col(0);
            VectorXT A_inv_g = cholmod_solver.solve(residual);
            VectorXT A_inv_u = cholmod_solver.solve(v);            
            T dem = 1.0 + v.dot(A_inv_u);            
            du = A_inv_g - (A_inv_g.dot(v)) * A_inv_u / dem;
        }
        
        VectorXT linSys_error = VectorXT::Zero(du.rows());
        VectorXT ones = VectorXT::Ones(du.rows());
        // cholmod_solver.multiply(du.data(), linSys_error.data());
        // linSys_error -= residual;
        bool solve_success = true;
        if (!solve_success)
            invalid_residual_cnt++;
        if (solve_success)
        {
            t.stop();
            std::cout << "\t===== Linear Solve ===== " << std::endl;
            
            std::cout << "\t takes " << t.elapsed_sec() << "s" << std::endl;
            std::cout << "\t# regularization step " << i 
                << " indefinite " << indefinite_count_reg_cnt 
                << " invalid solve " << invalid_residual_cnt << std::endl;
            std::cout << "\t======================== " << std::endl;
            return true;
        }
        else
        {
            std::cout << "|Ax-b|: " << linSys_error.norm() << std::endl;
            tbb::parallel_for(0, (int)K.rows(), [&](int row)
            {
                K.coeffRef(row, row) += alpha;
            });
            // cholmod_solver.A->x = K.valuePtr();
            // cholmod_solver.A->x[0] += alpha;
            // std::cout << mat_value[0] << std::endl;
            alpha *= 10;
        }

    }
    du = residual.normalized();
    std::cout << "linear solve failed" << std::endl;
    return true;
}

template <int dim>
T FEMSolver<dim>::lineSearchNewton(VectorXT& _u, VectorXT& residual)
{
    VectorXT du = residual;
    du.setZero();

    StiffnessMatrix K(residual.rows(), residual.rows());
    //Timer ti(true);
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    buildSystemMatrix(_u, K);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time difference (Build System Matrix) = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;

    //std::cout << "build system takes " <<  ti.elapsed_sec() << std::endl;
    bool success = linearSolve(K, residual, du);

    //eigenAnalysis(K);
    
    if (!success)
        return 1e16;
    T norm = du.norm();


    if(du.norm()<1e-10 && residual.norm()<1) return norm;
    while(du.norm() > 1)
    {
        du *= 0.5;
    }
    
    //if(du.norm() > radius/2.) du = du*(radius/2.)/du.norm();

    std::cout<<"du norm: " << du.norm() << std::endl;
    T alpha = computeInversionFreeStepsize(_u, du);
    std::cout << "** step size **" << std::endl;
    std::cout << "after tet inv step size: " << alpha << std::endl;

    if(USE_IPC_3D)
    {
        T ipc_step_size = computeCollisionFreeStepsize3D(_u, du);
        alpha = std::min(alpha, ipc_step_size);
        std::cout << "after ipc 3d step size: " << alpha << std::endl;
    }
    else if(BARRIER_ENERGY)
    {
        //T ipc_step_size = computeCollisionFreeStepsize2D(_u, du);
        T ipc_step_size = computeCollisionFreeStepsizeUnsigned(_u, du);
        //std::cout<<"IPC3D result: "<<ipc_step_size2<<std::endl;
        alpha = std::min(alpha, ipc_step_size);
        std::cout << "after barrier step size: " << alpha << std::endl;
    }
    else if (use_ipc_2d)
    {
        if(USE_TRUE_IPC_2D)
        {
            T ipc_step_size = computeCollisionFreeStepsize2D(_u, du);
            //T ipc_step_size = computeCollisionFreeStepsizeUnsigned(_u, du);
            alpha = std::min(alpha, ipc_step_size);
            std::cout << "after true ipc step size: " << alpha << std::endl;
        }else
        {
            T ipc_step_size = computeCollisionFreeStepsize2D(_u, du);
            //T ipc_step_size = computeCollisionFreeStepsizeUnsigned(_u, du);
            alpha = std::min(alpha, ipc_step_size);
            std::cout << "after ipc step size: " << alpha << std::endl;
        }
        
    }
    // else if(SLIDING_TEST)
    // {
    //     T ipc_step_size = computeCollisionFreeStepsize2Dtrue(_u, du);
    //     alpha = std::min(alpha, ipc_step_size);
    //     std::cout << "after true ipc step size: " << alpha << std::endl;
    // }
    std::cout << "**       **" << std::endl;

    begin = std::chrono::steady_clock::now();
    T E0 = computeTotalEnergy(_u);
    end = std::chrono::steady_clock::now();
    //std::cout << "Time difference (Compute Energy) = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;

    int cnt = 0;
    while (true)
    {
        VectorXT u_ls = _u + alpha * du;
        T E1 = computeTotalEnergy(u_ls);
        //std::cout<<"E1-E0: "<<E1 - E0<<std::endl;
        if (E1 - E0 < 0 || cnt > 8)
        {
            // if (cnt > 15)
            //     std::cout << "cnt > 15" << std::endl;

            // computeTotalEnergy(u_ls);
            // checkTotalGradientScale(1);
            // checkTotalHessianScale(1);
            _u = u_ls;

            if(cnt>8)
            {
                // checkTotalGradientScale(1);
                // checkTotalHessianScale(1);
                // std::ofstream file( "line_search2.csv", std::ios::app );
                // for(int k=-9; k<10; ++k)
                // {
                //     VectorXT residual(deformed.rows());
                //     residual.setZero();
                    
                //     double residual_norm = computeTotalEnergy(_u+0.5*k*_u);
                //     file<<residual_norm;
                //     if(k < 9) file<<",";
                //     else file<<std::endl;
                // }
                // for(int k=-9; k<10; ++k)
                // {
                //     file<<0.5*k;
                //     if(k < 9) file<<",";
                //     else file<<std::endl;
                // }
            }
            
            computeTotalEnergy(u_ls);
            break;
        }
        alpha *= 0.5;
        cnt += 1;
    }
        // T c1 = 10e-4, c2 = 0.9;
        // bool Armijo = E1 <= E0 + c1 * alpha * du.dot(-residual);
        // // std::cout << c1 * alpha * du.dot(-residual) << std::endl;
        // VectorXT gradient_forward = VectorXT::Zero(deformed.rows());
        // computeResidual(u_ls, gradient_forward);
        // bool curvature = -du.dot(-gradient_forward) <= -c2 * du.dot(-residual);
        // // std::cout << “wolfe Armijo ” << Armijo << ” curvature ” << curvature << std::endl;
        // if ((Armijo && curvature) || cnt > 15)
        // {
        //     _u = u_ls;
        //     if (cnt > 15)
        //     {
        //         if (verbose)
        //             std::cout <<"---ls max---"<< std::endl;
        //         // std::cout << “step size: ” << alpha << std::endl;
        //         // sampleEnergyWithSearchAndGradientDirection(_u, du, residual);
        //         // cells.computeTotalEnergy(u_ls, true);
        //         // cells.checkTotalGradientScale();
        //         // cells.checkTotalHessianScale();
        //         // return 1e16;
        //     }
        //     std::cout << "# ls " << cnt << std::endl;
        //     break;
        // }
        // alpha *= 0.5;
        // cnt += 1;
    //}

    std::cout << "#ls " << cnt << " alpha = " << alpha << std::endl;
    return norm;
}

template <int dim>
T FEMSolver<dim>::computeInversionFreeStepsize(const VectorXT& _u, const VectorXT& du)
{
    return 1.0;
    // Matrix<T, 4, 3> dNdb;
    //     dNdb << -1.0, -1.0, -1.0, 
    //         1.0, 0.0, 0.0,
    //         0.0, 1.0, 0.0,
    //         0.0, 0.0, 1.0;
           
    // VectorXT step_sizes = VectorXT::Zero(num_ele);

    // iterateElementParallel([&](const EleNodes& x_deformed, 
    //     const EleNodes& x_undeformed, const EleIdx& indices, int tet_idx)
    // {
    //     TM dXdb = x_undeformed.transpose() * dNdb;
    //     TM dxdb = x_deformed.transpose() * dNdb;
    //     TM A = dxdb * dXdb.inverse();
    //     T a, b, c, d;
    //     a = A.determinant();
    //     b = A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0) + A(0, 0) * A(2, 2) - A(0, 2) * A(2, 0) + A(1, 1) * A(2, 2) - A(1, 2) * A(2, 1);
    //     c = A.diagonal().sum();
    //     d = 0.8;

    //     T t = getSmallestPositiveRealCubicRoot(a, b, c, d);
    //     if (t < 0 || t > 1) t = 1;
    //         step_sizes(tet_idx) = t;
    // });
    // return step_sizes.minCoeff();
}

template <int dim>
bool FEMSolver<dim>::staticSolveStep(int step)
{
    if (step == 0)
    {
        reset();
        iterateDirichletDoF([&](int offset, T target)
        {
            f[offset] = 0;
        });
        u.setZero();
        // u *= 1.0 / u.norm();
        // u *= 0.001;
    }

    VectorXT residual(deformed.rows());

    Eigen::MatrixXd ipc_vertices_deformed = ipc_vertices;
    int num_nodes_all = num_nodes;
    //if(CALCULATE_IMLS_PROJECTION) num_nodes_all += projectedPts.rows();
    if(USE_NEW_FORMULATION) num_nodes_all += additional_dof;

    for (int i = 0; i < num_nodes_all; i++) 
    {
        //ipc_vertices_deformed.row(i);
        ipc_vertices_deformed.row(i) = deformed.segment<dim>(i * dim);
    }

    //findProjection(ipc_vertices_deformed, true);
    residual.setZero();
    T residual_norm = computeResidual(u, residual, false);

    if (USE_TRUE_IPC_2D || USE_IPC_3D)
    {
        updateBarrierInfo(step == 0);
        updateIPC2DVertices(u);
        std::cout << "ipc barrier stiffness " << barrier_weight << std::endl;
    }

    // saveIPCMesh("/home/yueli/Documents/ETH/WuKong/output/ThickShell/IPC_mesh_iter_" + std::to_string(step) + ".obj");
    saveToOBJ("dynamics3/iter_" + std::to_string(step) + ".obj");
    std::cout << "iter " << step << "/" << max_newton_iter << ": residual_norm " << residual_norm << " tol: " << newton_tol << std::endl;

    if (residual_norm < newton_tol)
        return true;
    

    T dq_norm = lineSearchNewton(u, residual);
    //std::cout<<u.transpose()<<std::endl;
    // saveToOBJ("/home/yueli/Documents/ETH/WuKong/output/ThickShell/iter_" + std::to_string(step) + ".obj");
    //saveToOBJ("/home/yueli/Documents/ETH/WuKong/output/ThickShell/structure_iter_" + std::to_string(step) + ".obj");
    iterateDirichletDoF([&](int offset, T target)
    {
        u[offset] = target;
    });
    deformed = undeformed + u;

    // displayBoundaryInfo();

    if(step == max_newton_iter || dq_norm > 1e10 || dq_norm < 1e-10)
        return true;

    // if(use_virtual_spring)
    //     addNeumannBC();
    return false;

}

template <int dim>
bool FEMSolver<dim>::staticSolve()
{
    int cnt = 0;
    T residual_norm = 1e10, dq_norm = 1e10;

    iterateDirichletDoF([&](int offset, T target)
    {
        f[offset] = 0;
    });

    while (true)
    {
        
        VectorXT residual(deformed.rows());
        residual.setZero();


        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        residual_norm = computeResidual(u, residual, false);
        if (USE_TRUE_IPC_2D||USE_IPC_3D)
        {
            updateBarrierInfo(cnt == 0);
            updateIPC2DVertices(u);
            std::cout << "ipc barrier stiffness " << barrier_weight << std::endl;
        }
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        //std::cout << "Time difference(compute Residual) = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
        saveToOBJ("dynamics3/iter_" + std::to_string(cnt) + ".obj");
        
        if (verbose)
            std::cout << "iter " << cnt << "/" << max_newton_iter 
            << ": residual_norm " << residual.norm() << " tol: " << newton_tol << std::endl;
        
        if (residual_norm < newton_tol)
            break;
        
        dq_norm = lineSearchNewton(u, residual);

        if(cnt == max_newton_iter || dq_norm > 1e10 || dq_norm < 1e-10)
            break;

        // if(use_virtual_spring)
        //     addNeumannBC();
        cnt++;
    }

    iterateDirichletDoF([&](int offset, T target)
    {
        u[offset] = target;
    });
    deformed = undeformed + u;

    std::cout << "# of newton solve: " << cnt << " exited with |g|: " 
        << residual_norm << "|ddu|: " << dq_norm  << std::endl;
    num_cnt = cnt;
    // std::cout << u.norm() << std::endl;
    if (cnt == max_newton_iter || dq_norm > 1e10 || residual_norm > 1)
        return false;
    return true;
    
}

template class FEMSolver<2>;
template class FEMSolver<3>;