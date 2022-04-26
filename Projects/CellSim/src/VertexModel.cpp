#include <unordered_set>
#include <fstream>
#include <Eigen/PardisoSupport>
#include <Spectra/SymEigsShiftSolver.h>
#include <Spectra/MatOp/SparseSymShiftSolve.h>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>


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

    buildSystemMatrix(u, K);

    // bool fix_dirichlet = true;
    // iterateDirichletDoF([&](int offset, T target)
    // {
    //     std::cout << offset << " " << target << std::endl;
    //     K.coeffRef(offset, offset) = 1e10;
    // });

    // std::vector<Entry> entries;
    // deformed = undeformed + u;
    // MatrixXT dummy;

    // iterateContractingEdgeSerial([&](Edge& e){
    //     TV vi = deformed.segment<3>(e[0] * 3);
    //     TV vj = deformed.segment<3>(e[1] * 3);
    //     Matrix<T, 6, 6> hessian;
    //     computeEdgeSquaredNormHessian(vi, vj, hessian);
    //     hessian *= Gamma;
    //     addHessianEntry<6>(entries, {e[0], e[1]}, hessian);
    // });
    
    // addEdgeHessianEntries(Apical, sigma, entries, project_block_hessian_PD);
    
    // addEdgeHessianEntries(ALL, weights_all_edges, entries, project_block_hessian_PD);
    // addFaceAreaHessianEntries(Basal, gamma, entries, project_block_hessian_PD);
    // addFaceAreaHessianEntries(Lateral, alpha, entries, project_block_hessian_PD);
    
    // addCellVolumePreservationHessianEntries(entries);

    // if (add_yolk_volume)
    //     addYolkVolumePreservationHessianEntries(entries, dummy);
    // if (add_tet_vol_barrier)
    //     addSingleTetVolBarrierHessianEntries(entries, project_block_hessian_PD);
    // if (add_perivitelline_liquid_volume)
    //     addPerivitellineVolumePreservationHessianEntries(entries, dummy);
    // if (use_sphere_radius_bound)
    //     addMembraneBoundHessianEntries(entries, project_block_hessian_PD);
    // if (use_ipc_contact)
    //     addIPCHessianEntries(entries, project_block_hessian_PD);
    
    // K.setFromTriplets(entries.begin(), entries.end());
    std::cout << "build K" << std::endl;

    bool use_Spectra = true;

    if (use_Spectra)
    {

        Spectra::SparseSymShiftSolve<T, Eigen::Lower> op(K);

        //0 cannot cannot be used as a shift
        T shift = -1e-4;
        Spectra::SymEigsShiftSolver<T, 
            Spectra::LARGEST_MAGN, 
            Spectra::SparseSymShiftSolve<T, Eigen::Lower> > 
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
        int compressed_cell_cnt = 0;
        for (int i = 0; i < current_cell_volume.rows(); i++)
        {
            if (current_cell_volume[i] < cell_volume_init[i])
                compressed_cell_cnt++;
        }
        std::cout << "\tcell vol sum: " << current_cell_volume.sum() << 
        " initial: " << cell_volume_init.sum() << 
        " " << compressed_cell_cnt << "/" << basal_face_start << " cells are compressed" << std::endl;
    }

    T yolk_vol_curr = computeYolkVolume();
    std::cout << "\tyolk vol sum: " << yolk_vol_curr << 
        " initial " << yolk_vol_init << std::endl;

    T perivitelline_vol_curr = total_volume - computeTotalVolumeFromApicalSurface();
    std::cout << "\tperivitelline vol sum: " << perivitelline_vol_curr <<
        " initial " << perivitelline_vol_init << std::endl;
    VectorXT residual(num_nodes * 3);
    residual.setZero();
    bool _print = print_force_norm;
    print_force_norm = false;
    computeResidual(u, residual);
    T total_energy = computeTotalEnergy(u);
    print_force_norm = _print;
    std::cout << "\t |g_norm|: " << residual.norm() << " total energy: " << total_energy << std::endl;

    bool all_inside = true;
    int inside_cnt = 0;
    for (int i = 0; i < num_nodes; i++)
    {
        TV xi = deformed.segment<3>(i * 3);
        if (sdf.inside(xi))
            inside_cnt++;
            // continue;
        // std::cout << sdf.value(xi) << std::endl;
        // all_inside = false;
        // break;
    }
    // if (!all_inside)
        // std::cout << "NOT ALL VERTICES ARE INSIDE THE SDF" << std::endl;
    std::cout << num_nodes - inside_cnt << "/" << num_nodes << " are not inside the SDF" << std::endl;

}

T VertexModel::computeTotalEnergy(const VectorXT& _u, bool verbose, bool add_to_deform)
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
    if (add_to_deform)
    {
        deformed = undeformed + projected;
    }

    T edge_length_term = 0.0, area_term = 0.0, 
        volume_term = 0.0, yolk_volume_term = 0.0,
        contraction_term = 0.0, membrane_bound_term = 0.0;
    
    // ===================================== Edge constriction =====================================
    if (add_contraction_term)
    {
        if (contract_apical_face)
        {
            addFaceContractionEnergy(Gamma, contraction_term);
        }
        else
        {
            if (assign_per_edge_weight)
                addPerEdgeEnergy(contraction_term);
            else
                addEdgeContractionEnergy(Gamma, contraction_term);
        }
    }

    if (dynamics)
    {
        T kinetic_term = 0.0;
        addInertialEnergy(kinetic_term);
        energy += kinetic_term;
        if (verbose)
            std::cout << "\tE_inertial " << kinetic_term << std::endl;    
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
        {
            if (use_fixed_centroid)
                addCellVolumePreservationEnergyFixedCentroid(volume_term);
            else
                addCellVolumePreservationEnergy(volume_term);
        }

        // ===================================== Face Area =====================================

        addFaceAreaEnergy(Apical, sigma, area_term);
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

    if (add_yolk_tet_barrier)
    {
        T yolk_tet_barrier_energy = 0.0;
        addYolkTetLogBarrierEnergy(yolk_tet_barrier_energy);
        energy += yolk_tet_barrier_energy;
        if (verbose)
            std::cout << "\tE_yolk_tet_barrier " << yolk_tet_barrier_energy << std::endl;
    }

    if (use_sphere_radius_bound)
    {
        if(use_sdf_boundary)
            addMembraneSDFBoundEnergy(membrane_bound_term);
        else
            addMembraneBoundEnergy(membrane_bound_term);
        if (verbose)
            std::cout << "\tE_inside_sphere " << membrane_bound_term << std::endl;
    }
    energy += membrane_bound_term;
    
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
            addFaceContractionForceEntries(Gamma, residual);
        else
        {
            if (assign_per_edge_weight)
                addPerEdgeForceEntries(residual);
            else
                addEdgeContractionForceEntries(Gamma, residual);
        }
    }

    if (print_force_norm)
        std::cout << "\tcontracting force norm: " << (residual - residual_temp).norm() << std::endl;
    residual_temp = residual;

    if (dynamics)
    {
        addInertialForceEntries(residual);
        if (print_force_norm)
            std::cout << "\tkinetic force norm: " << (residual - residual_temp).norm() << std::endl;
        residual_temp = residual;
    }


    if (use_elastic_potential)
    {
        addElasticityForceEntries(residual);
        if (print_force_norm)
            std::cout << "\telastic force norm: " << (residual - residual_temp).norm() << std::endl;
        residual_temp = residual;
    }
    else
    {
        addEdgeForceEntries(ALL, weights_all_edges, residual);
        if (print_force_norm)
            std::cout << "\tall edges contraction force norm: " << (residual - residual_temp).norm() << std::endl;
        residual_temp = residual;

        if (preserve_tet_vol)
            addTetVolumePreservationForceEntries(residual);
        else
        {
            if (use_fixed_centroid)
                addCellVolumePreservationForceEntriesFixedCentroid(residual);
            else
                addCellVolumePreservationForceEntries(residual);
        }

        if (print_force_norm)
            std::cout << "\tcell volume preservation force norm: " << (residual - residual_temp).norm() << std::endl;
        residual_temp = residual;
        
        addFaceAreaForceEntries(Apical, sigma, residual);
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
    if(false)
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

    if (add_yolk_tet_barrier)
    {
        addYolkTetLogBarrierForceEneries(residual);
        if (print_force_norm)
            std::cout << "\tyolk tet barrier force norm: " << (residual - residual_temp).norm() << std::endl;
        residual_temp = residual;
    }

    // ===================================== Membrane =====================================
    if (use_sphere_radius_bound)
    {
        if (use_sdf_boundary)
            addMembraneSDFBoundForceEntries(residual);
        else
            addMembraneBoundForceEntries(residual);
        if(print_force_norm)
            std::cout << "\tmembrane bound norm: " << (residual - residual_temp).norm() << std::endl;
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
            addFaceContractionHessianEntries(Gamma, entries, project_block_hessian_PD);
        else
        {
            if (assign_per_edge_weight)
                addPerEdgeHessianEntries(entries, project_block_hessian_PD);
            else
                addEdgeContractionHessianEntries(Gamma, entries, project_block_hessian_PD);
        }
    }

    if (dynamics)
        addInertialHessianEntries(entries);
    
    if (use_elastic_potential)
    {
        addElasticityHessianEntries(entries, project_block_hessian_PD);
    }
    else
    {
        addEdgeHessianEntries(ALL, weights_all_edges, entries, project_block_hessian_PD);

        if (preserve_tet_vol)
            addTetVolumePreservationHessianEntries(entries, project_block_hessian_PD);
        else
        {
            if (use_fixed_centroid)
                addCellVolumePreservationHessianEntriesFixedCentroid(entries, project_block_hessian_PD);
            else
                addCellVolumePreservationHessianEntries(entries, project_block_hessian_PD);
        }
        // ===================================== Face Area =====================================

        addFaceAreaHessianEntries(Apical, sigma, entries, project_block_hessian_PD);
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
    if (add_yolk_tet_barrier)
        addYolkTetLogBarrierHessianEneries(entries, project_block_hessian_PD);
        
    if (add_perivitelline_liquid_volume)
        addPerivitellineVolumePreservationHessianEntries(entries, UV, project_block_hessian_PD);
    

    // ===================================== Membrane =====================================
    if (use_sphere_radius_bound)
    {
        if(use_sdf_boundary)
            addMembraneSDFBoundHessianEntries(entries, project_block_hessian_PD);
        else
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

    // Matrix<T, 2, 2> I;
    // I.setIdentity();    
    // MatrixXT dense_part = UV * I * UV.transpose();
    // dense_part += K;
    // K = dense_part.sparseView();
    // std::ofstream out("hessian.txt");
    // out << K;
    // out.close();
    // std::exit(0);
    // std::getchar();
    K.makeCompressed();
}

void VertexModel::buildSystemMatrix(const VectorXT& _u, StiffnessMatrix& K)
{
    bool using_woodbury_formulation = woodbury;
    woodbury = false;
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
            addFaceContractionHessianEntries(Gamma, entries, project_block_hessian_PD);
        else
        {
            if (assign_per_edge_weight)
                addPerEdgeHessianEntries(entries, project_block_hessian_PD);
            else
                addEdgeContractionHessianEntries(Gamma, entries, project_block_hessian_PD);
        }
    }

    if (dynamics)
        addInertialHessianEntries(entries);

    if (use_elastic_potential)
    {
        addElasticityHessianEntries(entries, project_block_hessian_PD);
    }
    else
    {
        addEdgeHessianEntries(ALL, weights_all_edges, entries, project_block_hessian_PD);
        if (preserve_tet_vol)
            addTetVolumePreservationHessianEntries(entries, project_block_hessian_PD);
        else
        {
            if (use_fixed_centroid)
                addCellVolumePreservationHessianEntriesFixedCentroid(entries, project_block_hessian_PD);
            else
                addCellVolumePreservationHessianEntries(entries, project_block_hessian_PD);
        }

        addFaceAreaHessianEntries(Apical, sigma, entries, project_block_hessian_PD);
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

    MatrixXT dummy_WoodBury_matrix;

    if (add_yolk_volume)
        addYolkVolumePreservationHessianEntries(entries, dummy_WoodBury_matrix, project_block_hessian_PD);
    if (add_yolk_tet_barrier)
        addYolkTetLogBarrierHessianEneries(entries, project_block_hessian_PD);
    if (add_perivitelline_liquid_volume)
        addPerivitellineVolumePreservationHessianEntries(entries, dummy_WoodBury_matrix, project_block_hessian_PD);
    

    if (use_sphere_radius_bound)
    {
        if(use_sdf_boundary)
            addMembraneSDFBoundHessianEntries(entries, project_block_hessian_PD);
        else
            addMembraneBoundHessianEntries(entries, project_block_hessian_PD);
    }

    if (use_ipc_contact)
        addIPCHessianEntries(entries, project_block_hessian_PD);

        
    K.resize(num_nodes * 3, num_nodes * 3);
    K.setFromTriplets(entries.begin(), entries.end());
    if (!run_diff_test)
        projectDirichletDoFMatrix(K, dirichlet_data);
    // std::cout << K << std::endl;
    // std::ofstream out("hessian.txt");
    // out << K;
    // out.close();
    // std::exit(0);
    // std::getchar();
    K.makeCompressed();
    woodbury = using_woodbury_formulation;
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

T VertexModel::computeLineSearchInitStepsize(const VectorXT& _u, const VectorXT& du, bool verbose)
{
    if (verbose)
        std::cout << "** step size **" << std::endl;
    T step_size = 1.0;
    if (use_ipc_contact)
    {
        T ipc_step_size = computeCollisionFreeStepsize(_u, du);
        step_size = std::min(step_size, ipc_step_size);
        if (verbose)
            std::cout << "after ipc step size: " << step_size << std::endl;
    }

    if (use_sphere_radius_bound && !sphere_bound_penalty && !use_sdf_boundary)
    {
        T inside_membrane_step_size = computeInsideMembraneStepSize(_u, du);
        step_size = std::min(step_size, inside_membrane_step_size);
        if (verbose)
            std::cout << "after inside membrane step size: " << step_size << std::endl;
    }

    if (add_tet_vol_barrier)
    {
        T inversion_free_step_size = computeInversionFreeStepSize(_u, du);
        // std::cout << "cell tet inversion free step size: " << inversion_free_step_size << std::endl;
        step_size = std::min(step_size, inversion_free_step_size);
        if (verbose)
            std::cout << "after tet inverison step size: " << step_size << std::endl;
    }

    if (add_yolk_tet_barrier)
    {
        T inversion_free_step_size = computeYolkInversionFreeStepSize(_u, du);
        // std::cout << "yolk inversion free step size: " << inversion_free_step_size << std::endl;
        step_size = std::min(step_size, inversion_free_step_size);
    }
    if (verbose)
        std::cout << "**       **" << std::endl;
    return step_size;
}