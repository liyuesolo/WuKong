#include <unordered_set>
#include <fstream>
#include <Spectra/SymEigsShiftSolver.h>
#include <Spectra/MatOp/SparseSymShiftSolve.h>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>

#include <Eigen/PardisoSupport>

#include <ipc/ipc.hpp>

#include "../include/VertexModel.h"
#include "../include/autodiff/VertexModelEnergy.h"
#include "../include/autodiff/YolkEnergy.h"
#include "../include/autodiff/TetVolBarrier.h"
#include "../include/autodiff/EdgeEnergy.h"


void VertexModel::computeLinearModes()
{
    int nmodes = 15;

    StiffnessMatrix K(deformed.rows(), deformed.rows());
    run_diff_test = true;

    // buildSystemMatrix(u, K);

    std::vector<Entry> entries;
    deformed = undeformed + u;
    MatrixXT dummy;
    addFaceAreaHessianEntries(Basal, gamma, entries);
    addFaceAreaHessianEntries(Lateral, alpha, entries);
    // addYolkVolumePreservationHessianEntries(entries, dummy);
    // addCellVolumePreservationHessianEntries(entries);
    
    K.setFromTriplets(entries.begin(), entries.end());
    std::cout << "build K" << std::endl;
    bool use_Spectra = true;

    if (use_Spectra)
    {

        Spectra::SparseSymShiftSolve<T, Eigen::Upper> op(K);

        //0 cannot cannot be used as a shift
        T shift = -1e-4;
        Spectra::SymEigsShiftSolver<T, 
            Spectra::LARGEST_MAGN, 
            Spectra::SparseSymShiftSolve<T, Eigen::Upper> > 
            eigs(&op, nmodes, 2 * nmodes, shift);

        eigs.init();

        int nconv = eigs.compute();

        if (eigs.info() == Spectra::SUCCESSFUL)
        {
            Eigen::MatrixXd eigen_vectors = eigs.eigenvectors().real();
            Eigen::VectorXd eigen_values = eigs.eigenvalues().real();
            std::cout << eigen_values << std::endl;
            std::ofstream out("cell_eigen_vectors.txt");
            out << eigen_vectors.rows() << " " << eigen_vectors.cols() << std::endl;
            for (int i = 0; i < eigen_vectors.cols(); i++)
                out << eigen_values[eigen_vectors.cols() - 1 - i] << " ";
            out << std::endl;
            for (int i = 0; i < eigen_vectors.rows(); i++)
            {
                // for (int j = 0; j < eigen_vectors.cols(); j++)
                for (int j = eigen_vectors.cols() - 1; j >-1 ; j--)
                    out << eigen_vectors(i, j) << " ";
                out << std::endl;
            }       
            out << std::endl;
            out.close();
        }
        else
        {
            std::cout << "Eigen decomposition failed" << std::endl;
        }
    }
    else
    {
        Eigen::MatrixXd A_dense = K;
        Eigen::EigenSolver<Eigen::MatrixXd> eigen_solver;
        eigen_solver.compute(A_dense, /* computeEigenvectors = */ true);
        auto eigen_values = eigen_solver.eigenvalues();
        auto eigen_vectors = eigen_solver.eigenvectors();
        
        std::vector<T> ev_all(A_dense.cols());
        for (int i = 0; i < A_dense.cols(); i++)
        {
            ev_all[i] = eigen_values[i].real();
        }
        
        std::vector<int> indices;
        for (int i = 0; i < A_dense.cols(); i++)
        {
            indices.push_back(i);    
        }
        std::sort(indices.begin(), indices.end(), [&ev_all](int a, int b){ return ev_all[a] < ev_all[b]; } );
        // std::sort(ev_all.begin(), ev_all.end());

        for (int i = 0; i < nmodes; i++)
            std::cout << ev_all[indices[i]] << std::endl;
        

        std::ofstream out("cell_eigen_vectors.txt");
        out << nmodes << " " << A_dense.cols() << std::endl;
        for (int i = 0; i < nmodes; i++)
            out << ev_all[indices[i]] << " ";
        out << std::endl;
        for (int i = 0; i < nmodes; i++)
        {
            out << eigen_vectors.col(indices[i]).real() << std::endl;
        }

        out.close();
    }
}

bool VertexModel::linearSolve(StiffnessMatrix& K, VectorXT& residual, VectorXT& du)
{
    StiffnessMatrix I(K.rows(), K.cols());
    I.setIdentity();

    StiffnessMatrix H = K;

    Eigen::PardisoLDLT<Eigen::SparseMatrix<T, Eigen::ColMajor, typename StiffnessMatrix::StorageIndex>> solver;
    
    T alpha = 10e-6;
    solver.analyzePattern(K);
    for (int i = 0; i < 50; i++)
    {
        // std::cout << i << std::endl;
        solver.factorize(K);
        if (solver.info() == Eigen::NumericalIssue)
        {
            // std::cout << "indefinite" << std::endl;
            K = H + alpha * I;        
            alpha *= 10;
            continue;
        }
        du = solver.solve(residual);

        T dot_dx_g = du.normalized().dot(residual.normalized());

        // VectorXT d_vector = solver.vectorD();
        int num_negative_eigen_values = 0;


        bool positive_definte = num_negative_eigen_values == 0;
        bool search_dir_correct_sign = dot_dx_g > 1e-6;
        bool solve_success = (K*du - residual).norm() < 1e-6 && solver.info() == Eigen::Success;

        if (positive_definte && search_dir_correct_sign && solve_success)
            return true;
        else
        {
            K = H + alpha * I;        
            alpha *= 10;
        }
    }
    return false;
}



void VertexModel::updateALMData(const VectorXT& _u)
{
    if (!use_alm_on_cell_volume)
        return;
    
    VectorXT projected = _u;
    iterateDirichletDoF([&](int offset, T target)
    {
        projected[offset] = target;
    });
    deformed = undeformed + projected;

    if (use_alm_on_cell_volume)
    {
        VectorXT current_cell_volume;
        computeVolumeAllCells(current_cell_volume);

        iterateFaceSerial([&](VtxList& face_vtx_list, int face_idx)
        {
            VectorXT positions;
            positionsFromIndices(positions, face_vtx_list);
            T area_energy = 0.0;
            
            // cell-wise volume preservation term
            if (face_idx < basal_face_start)
            {
                T ci = current_cell_volume[face_idx] - cell_volume_init[face_idx];
                lambda_cell_vol[face_idx] -= kappa * ci;
            }
        });
    }
    // if (kappa < kappa_max)
    //     kappa *= 2.0;
}

void VertexModel::computeCellInfo()
{
    std::cout << "=================== Cell Info ===================" << std::endl;
    if (preserve_tet_vol)
    {
        VectorXT tet_vol_curr;
        computeTetVolCurent(tet_vol_curr);
        std::cout << "\ttet vol sum: " << tet_vol_curr.sum() << std::endl;
    }
    else
    {
        VectorXT current_cell_volume;
        computeVolumeAllCells(current_cell_volume);
        std::cout << "\ttet vol sum: " << current_cell_volume.sum() << std::endl;
    }

    T yolk_vol_curr = computeYolkVolume();
    std::cout << "\tyolk vol sum: " << yolk_vol_curr << std::endl;

    T perivitelline_vol_curr = total_volume - computeTotalVolumeFromApicalSurface();
    std::cout << "\tperivitelline vol sum: " << perivitelline_vol_curr << std::endl;
}

T VertexModel::computeTotalEnergy(const VectorXT& _u, bool verbose)
{
    if (verbose)
        std::cout << std::endl;
    T energy = 0.0;
    
    VectorXT projected = _u;
    if (!run_diff_test)
    {
        iterateDirichletDoF([&](int offset, T target)
        {
            projected[offset] = target;
        });
    }
    deformed = undeformed + projected;

    T edge_length_term = 0.0, area_term = 0.0, 
        volume_term = 0.0, yolk_volume_term = 0.0,
        contraction_term = 0.0, sphere_bound_term = 0.0;
    
    // ===================================== Edge constriction =====================================
    if (add_contraction_term)
    {
        if (contract_apical_face)
        {
            addFaceAreaEnergy(Apical, Gamma, contraction_term);
        }
        else
        {
            iterateContractingEdgeSerial([&](Edge& e){    
                TV vi = deformed.segment<3>(e[0] * 3);
                TV vj = deformed.segment<3>(e[1] * 3);
                T edge_length = computeEdgeSquaredNorm(vi, vj);
                contraction_term += Gamma * edge_length;
            });
        }
    }
    
    energy += contraction_term;

    if (use_elastic_potential)
    {
        T elastic_energy;
        addElasticityEnergy(elastic_energy);
        energy += elastic_energy;
        if (verbose)
            std::cout << "\tE_neo " << elastic_energy << std::endl;    
    }
    else
    {
        // ===================================== Edge length =====================================
        if (contract_apical_face)
        {
            addFaceAreaEnergy(Apical, sigma, contraction_term);
        }
        else
            addEdgeEnergy(Apical, sigma, edge_length_term);

        addEdgeEnergy(ALL, weights_all_edges, edge_length_term);

        if (verbose)
        {
            std::cout << "\tE_edge " << edge_length_term << std::endl;
            if (add_contraction_term)
                std::cout << "\tE_contract " << contraction_term << std::endl;
        }
        energy += edge_length_term;

        // ===================================== Cell Volume =====================================
        if (preserve_tet_vol)
            addTetVolumePreservationEnergy(volume_term);
        else
            addCellVolumePreservationEnergy(volume_term);

        // ===================================== Face Area =====================================
        addFaceAreaEnergy(Basal, gamma, area_term);
        addFaceAreaEnergy(Lateral, alpha, area_term);

        if (verbose)
        {
            std::cout << "\tE_area: " << area_term << std::endl;
            std::cout << "\tE_volume: " << volume_term << std::endl;
        }

        energy += volume_term;
        energy += area_term;
    }
    T vol_barrier_energy = 0.0;
    if (add_tet_vol_barrier)
    {
        if (use_cell_centroid)
            addSingleTetVolBarrierEnergy(vol_barrier_energy);
        else
            addFixedTetLogBarrierEnergy(vol_barrier_energy);
        energy += vol_barrier_energy;
        if (verbose)
            std::cout << "\tE_tet_barrier " << vol_barrier_energy << std::endl;
    }
    if (add_yolk_volume)
    {
        addYolkVolumePreservationEnergy(yolk_volume_term);
        if (verbose)
            std::cout << "\tE_yolk_vol " << yolk_volume_term << std::endl;
    }

    energy += yolk_volume_term;

    if (use_sphere_radius_bound)
    {
        addMembraneBoundEnergy(sphere_bound_term);
        if (verbose)
            std::cout << "\tE_inside_sphere " << sphere_bound_term << std::endl;
    }
    energy += sphere_bound_term;
    
    T contact_energy = 0.0;

    if (use_ipc_contact)
    {
        addIPCEnergy(contact_energy);
        if (verbose)
            std::cout << "\tE_contact: " << contact_energy << std::endl;
        energy += contact_energy;
        // std::getchar();
    }

    if (add_perivitelline_liquid_volume)
    {
        T volume_penalty_previtelline = 0.0;
        addPerivitellineVolumePreservationEnergy(volume_penalty_previtelline);
        energy += volume_penalty_previtelline;
        if (verbose)
            std::cout << "\tE_previtelline_vol: " << volume_penalty_previtelline << std::endl;
    }

    return energy;
}


T VertexModel::computeResidual(const VectorXT& _u,  VectorXT& residual, bool verbose)
{
    // if (use_ipc_contact)
    //     updateIPCVertices(_u);
    VectorXT residual_temp = residual;
    VectorXT projected = _u;
    if (!run_diff_test)
    {
        iterateDirichletDoF([&](int offset, T target)
        {
            projected[offset] = target;
        });
    }
    deformed = undeformed + projected;

    // ===================================== Edge constriction =====================================
    if (add_contraction_term)
    {
        if (contract_apical_face)
            addFaceAreaForceEntries(Apical, Gamma, residual);
        else
            iterateContractingEdgeSerial([&](Edge& e){
                TV vi = deformed.segment<3>(e[0] * 3);
                TV vj = deformed.segment<3>(e[1] * 3);
                Vector<T, 6> dedx;
                computeEdgeSquaredNormGradient(vi, vj, dedx);
                dedx *= -Gamma;
                addForceEntry<6>(residual, {e[0], e[1]}, dedx);
            }); 
    }

    if (print_force_norm)
        std::cout << "\tcontracting force norm: " << (residual - residual_temp).norm() << std::endl;
    residual_temp = residual;


    if (use_elastic_potential)
    {
        addElasticityForceEntries(residual);
        if (print_force_norm)
            std::cout << "\telastic force norm: " << (residual - residual_temp).norm() << std::endl;
        residual_temp = residual;
    }
    else
    {
        if (contract_apical_face)
        {
            addFaceAreaForceEntries(Apical, sigma, residual);
            if (print_force_norm)
                std::cout << "\tapical area force norm: " << (residual - residual_temp).norm() << std::endl;
        }
        else
        {
            addEdgeForceEntries(Apical, sigma, residual);
            if (print_force_norm)
                std::cout << "\tapical edge force norm: " << (residual - residual_temp).norm() << std::endl;
        }
        
        addEdgeForceEntries(ALL, weights_all_edges, residual);
        if (print_force_norm)
            std::cout << "\tall edges contraction force norm: " << (residual - residual_temp).norm() << std::endl;
        residual_temp = residual;

        if (preserve_tet_vol)
            addTetVolumePreservationForceEntries(residual);
        else
            addCellVolumePreservationForceEntries(residual);

        if (print_force_norm)
            std::cout << "\tcell volume preservation force norm: " << (residual - residual_temp).norm() << std::endl;
        residual_temp = residual;
        
        addFaceAreaForceEntries(Basal, gamma, residual);
        addFaceAreaForceEntries(Lateral, alpha, residual);

        if (print_force_norm)
            std::cout << "\tbasal and lateral area force norm: " << (residual - residual_temp).norm() << std::endl;
        residual_temp = residual;
    }


    if (add_tet_vol_barrier)
    {
        if (use_cell_centroid)
            addSingleTetVolBarrierForceEntries(residual);
        else
            addFixedTetLogBarrierForceEneries(residual);
        if (print_force_norm)
            std::cout << "\ttet barrier force norm: " << (residual - residual_temp).norm() << std::endl;
        residual_temp = residual;
    }

    if (add_yolk_volume)
        addYolkVolumePreservationForceEntries(residual);
    
    {
        VectorXT yolk_force = (residual - residual_temp);
        int cnt = 0, negative_cnt = 0;
        for (int i = basal_vtx_start; i < num_nodes; i++)
        {
            TV xi = deformed.segment<3>(i * 3);
            T dot_pro = yolk_force.segment<3>(i * 3).dot(xi - mesh_centroid);
            if (dot_pro < -1e-8)
            {
                negative_cnt++;
            }
            // else 
            // {
            //     if (yolk_force.norm() > 1e-8)
            //         std::cout << "!!!!dot " << dot_pro << std::endl;
            // }
            
            cnt ++;
        }
        std::cout << "yolk force " << negative_cnt << "/" << cnt << " is negative" << std::endl;
    }

    if (print_force_norm)
        std::cout << "\tyolk volume preservation force norm: " << (residual - residual_temp).norm() << std::endl;
    residual_temp = residual;

    // ===================================== Membrane =====================================
    if (use_sphere_radius_bound)
    {
        addMembraneBoundForceEntries(residual);
        if(print_force_norm)
            std::cout << "\tsphere bound norm: " << (residual - residual_temp).norm() << std::endl;
        residual_temp = residual;
    }

    // ===================================== IPC =====================================
    if (use_ipc_contact)
    {
        addIPCForceEntries(residual);
    
        if(print_force_norm)
            std::cout << "\tcontact force norm: " << (residual - residual_temp).norm() << std::endl;
        residual_temp = residual;
    }

    
    // ===================================== Previtelline Volume =====================================
    if (add_perivitelline_liquid_volume)
    {
        addPerivitellineVolumePreservationForceEntries(residual);
        if (print_force_norm)
            std::cout << "\tprevitelline vol force norm: " << (residual - residual_temp).norm() << std::endl;
        residual_temp = residual;
    }

    if (!run_diff_test)
        iterateDirichletDoF([&](int offset, T target)
        {
            residual[offset] = 0;
        });

    return residual.norm();
}



void VertexModel::buildSystemMatrixWoodbury(const VectorXT& _u, StiffnessMatrix& K, MatrixXT& UV)
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
    
    if (add_contraction_term)
    {
        if (contract_apical_face)
            addFaceAreaHessianEntries(Apical, Gamma, entries, project_block_hessian_PD);
        else
            iterateContractingEdgeSerial([&](Edge& e){
                TV vi = deformed.segment<3>(e[0] * 3);
                TV vj = deformed.segment<3>(e[1] * 3);
                Matrix<T, 6, 6> hessian;
                computeEdgeSquaredNormHessian(vi, vj, hessian);
                hessian *= Gamma;
                addHessianEntry<6>(entries, {e[0], e[1]}, hessian);
            });
    }
    
    if (use_elastic_potential)
    {
        addElasticityHessianEntries(entries, project_block_hessian_PD);
    }
    else
    {
        if (contract_apical_face)
        {
            addFaceAreaHessianEntries(Apical, sigma, entries, project_block_hessian_PD);
        }
        else
        {
            addEdgeHessianEntries(Apical, sigma, entries, project_block_hessian_PD);
        }
        
        addEdgeHessianEntries(ALL, weights_all_edges, entries, project_block_hessian_PD);

        if (preserve_tet_vol)
            addTetVolumePreservationHessianEntries(entries, project_block_hessian_PD);
        else
            addCellVolumePreservationHessianEntries(entries, project_block_hessian_PD);
        // ===================================== Face Area =====================================

        addFaceAreaHessianEntries(Basal, gamma, entries, project_block_hessian_PD);
        addFaceAreaHessianEntries(Lateral, alpha, entries, project_block_hessian_PD);

    }

    if (add_tet_vol_barrier)
    {
        if (use_cell_centroid)
            addSingleTetVolBarrierHessianEntries(entries, project_block_hessian_PD);
        else
            addFixedTetLogBarrierHessianEneries(entries, project_block_hessian_PD);
    }

    if (add_yolk_volume)
        addYolkVolumePreservationHessianEntries(entries, UV, project_block_hessian_PD);
    if (add_perivitelline_liquid_volume)
        addPerivitellineVolumePreservationHessianEntries(entries, UV, project_block_hessian_PD);
    

    // ===================================== Membrane =====================================
    if (use_sphere_radius_bound)
    {
        addMembraneBoundHessianEntries(entries, project_block_hessian_PD);
    }

    // ===================================== IPC =====================================
    if (use_ipc_contact)
    {
        addIPCHessianEntries(entries, project_block_hessian_PD);
    }

    K.resize(num_nodes * 3, num_nodes * 3);
    K.setFromTriplets(entries.begin(), entries.end());
        
        
    if (!run_diff_test)
        projectDirichletDoFMatrix(K, dirichlet_data);
    // std::cout << K << std::endl;
    // std::ofstream out("hessian.txt");
    // out << K;
    // out.close();
    // std::getchar();
    K.makeCompressed();
}

void VertexModel::buildSystemMatrix(const VectorXT& _u, StiffnessMatrix& K)
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
    
    if (add_contraction_term)
    {
        if (contract_apical_face)
            addFaceAreaHessianEntries(Apical, Gamma, entries, project_block_hessian_PD);
        else
        {
            iterateContractingEdgeSerial([&](Edge& e){
                TV vi = deformed.segment<3>(e[0] * 3);
                TV vj = deformed.segment<3>(e[1] * 3);
                Matrix<T, 6, 6> hessian;
                computeEdgeSquaredNormHessian(vi, vj, hessian);
                hessian *= Gamma;
                addHessianEntry<6>(entries, {e[0], e[1]}, hessian);
            });
        }
    }

    if (use_elastic_potential)
    {
        addElasticityHessianEntries(entries, project_block_hessian_PD);
    }
    else
    {
        if (contract_apical_face)
        {
            addFaceAreaHessianEntries(Apical, sigma, entries, project_block_hessian_PD);
        }
        else
            addEdgeHessianEntries(Apical, sigma, entries, project_block_hessian_PD);
        
        addEdgeHessianEntries(ALL, weights_all_edges, entries, project_block_hessian_PD);
        if (preserve_tet_vol)
            addTetVolumePreservationHessianEntries(entries, project_block_hessian_PD);
        else
            addCellVolumePreservationHessianEntries(entries, project_block_hessian_PD);

        addFaceAreaHessianEntries(Basal, gamma, entries, project_block_hessian_PD);
        addFaceAreaHessianEntries(Lateral, gamma, entries, project_block_hessian_PD);
        
    }
    
    if (add_tet_vol_barrier)
    {
        if (use_cell_centroid)
            addSingleTetVolBarrierHessianEntries(entries, project_block_hessian_PD);
        else
            addFixedTetLogBarrierHessianEneries(entries, project_block_hessian_PD);
    }

    MatrixXT dummy_WoodBury_matrix;

    if (add_yolk_volume)
        addYolkVolumePreservationHessianEntries(entries, dummy_WoodBury_matrix, project_block_hessian_PD);

    if (add_perivitelline_liquid_volume)
        addPerivitellineVolumePreservationHessianEntries(entries, dummy_WoodBury_matrix, project_block_hessian_PD);
    

    if (use_sphere_radius_bound)
    {
        addMembraneBoundHessianEntries(entries, project_block_hessian_PD);
    }

    if (use_ipc_contact)
    {
        addIPCHessianEntries(entries, project_block_hessian_PD);
    }

        
    K.resize(num_nodes * 3, num_nodes * 3);
    K.setFromTriplets(entries.begin(), entries.end());
    if (!run_diff_test)
        projectDirichletDoFMatrix(K, dirichlet_data);
    // std::cout << K << std::endl;
    // std::ofstream out("hessian.txt");
    // out << K;
    // out.close();
    // std::getchar();
    K.makeCompressed();
}

void VertexModel::projectDirichletDoFMatrix(StiffnessMatrix& A, 
    const std::unordered_map<int, T>& data)
{
    for (auto iter : data)
    {
        A.row(iter.first) *= 0.0;
        A.col(iter.first) *= 0.0;
        A.coeffRef(iter.first, iter.first) = 1.0;
    }

}
