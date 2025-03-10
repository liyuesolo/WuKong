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

    VectorXT edge_norm(edges.size());
    tbb::parallel_for(0, (int)edges.size(), [&](int i){
        TV vi = deformed.segment<3>(edges[i][0] * 3);
        TV vj = deformed.segment<3>(edges[i][1] * 3);
        edge_norm[i] = (vj - vi).norm();
    });
    std::cout << "min edge length " << edge_norm.minCoeff() << " max " << edge_norm.maxCoeff() 
        << " avg " << edge_norm.sum() / T(edge_norm.rows()) << std::endl;
    bool all_inside = true;
    int inside_cnt = 0;
    T penetration = 0.0;
    for (int i = 0; i < num_nodes; i++)
    {
        TV xi = deformed.segment<3>(i * 3);
        if (sdf.inside(xi))
        {
            inside_cnt++;
        }
        else
        {
            penetration += sdf.value(xi);
        }

            // continue;
        // std::cout << sdf.value(xi) << std::endl;
        // all_inside = false;
        // break;
    }
    // if (!all_inside)
        // std::cout << "NOT ALL VERTICES ARE INSIDE THE SDF" << std::endl;
    std::cout << num_nodes - inside_cnt << "/" << num_nodes << " are not inside the SDF" << std::endl;
    TV min_corner, max_corner;
    computeBoundingBox(min_corner, max_corner);
    std::cout << "total penetration " << penetration << " embryo ap length: " << max_corner[0] - min_corner[1] << std::endl;
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
        addEdgeEnergy(Apical, sigma, edge_length_term);
        addEdgeEnergy(Basal, gamma, edge_length_term);
        addEdgeEnergy(Lateral, alpha, edge_length_term);

        // addEdgeEnergy(ALL, weights_all_edges, edge_length_term);

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
        energy += volume_term;
        if (verbose)
            std::cout << "\tE_volume: " << volume_term << std::endl;
            
        // ===================================== Face Area =====================================
        if (add_area_term)
        {
            // if (has_rest_shape)
            // {
            //     addFaceAreaEnergyWithRestShape(Apical, sigma, area_term);
            //     addFaceAreaEnergyWithRestShape(Basal, gamma, area_term);
            //     addFaceAreaEnergyWithRestShape(Lateral, alpha, area_term);
            // }
            // else
            {
                addFaceAreaEnergy(Apical, sigma, area_term);
                addFaceAreaEnergy(Basal, gamma, area_term);
                addFaceAreaEnergy(Lateral, alpha, area_term);
            }    
            energy += area_term;
            if (verbose)
                std::cout << "\tE_area: " << area_term << std::endl;
        }
        
        
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
        if (print_force_norm)
            std::cout << "\tcontracting force norm: " << (residual - residual_temp).norm() << std::endl;
        residual_temp = residual;
    }


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
        addEdgeForceEntries(Apical, sigma, residual);
        addEdgeForceEntries(Basal, gamma, residual);
        addEdgeForceEntries(Lateral, alpha, residual);

        // addEdgeForceEntries(ALL, weights_all_edges, residual);
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
        
        if (add_area_term)
        {
            // if (has_rest_shape)
            // {
            //     addFaceAreaForceEntriesWithRestShape(Apical, sigma, residual);
            //     addFaceAreaForceEntriesWithRestShape(Basal, gamma, residual);
            //     addFaceAreaForceEntriesWithRestShape(Lateral, alpha, residual);
            // }
            // else
            {
                addFaceAreaForceEntries(Apical, sigma, residual);
                addFaceAreaForceEntries(Basal, gamma, residual);
                addFaceAreaForceEntries(Lateral, alpha, residual);
            }
        }
        

        if (print_force_norm)
            std::cout << "\tface area force norm: " << (residual - residual_temp).norm() << std::endl;
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
    {
        addYolkVolumePreservationForceEntries(residual);
        if (print_force_norm)
            std::cout << "\tyolk volume preservation force norm: " << (residual - residual_temp).norm() << std::endl;
        residual_temp = residual;
    }
    
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


void VertexModel::buildSystemMatrixWoodburyFaster(const VectorXT& _u, 
        StiffnessMatrix& K, MatrixXT& UV)
{
    bool profile = true;
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

    // iterateEdgeSerial([&](Edge& e){
    //     TV vi = deformed.segment<3>(e[0] * 3);
    //     TV vj = deformed.segment<3>(e[1] * 3);
    //     Matrix<T, 6, 6> hessian;
    //     computeEdgeSquaredNormHessian(vi, vj, hessian);
    //     Matrix<T, 6, 6> spring_hessian;
    //     TV Xi = undeformed.segment<3>(e[0] * 3);
    //     TV Xj = undeformed.segment<3>(e[1] * 3);
    //     T l0 = (Xj - Xi).norm();
    //     Vector<T, 6> x;
    //     x << vi, vj;
    //     int apical_edge_cnt = 0;
    //     if (e[0] < basal_vtx_start && e[1] < basal_vtx_start)
    //     {
    //         hessian *= edge_weights[apical_edge_cnt++];
    //         addHessianEntry<6>(entries, {e[0], e[1]}, hessian);
    //         computeEdgeEnergyRestLength3DHessian(sigma, l0, x, spring_hessian);
    //     }
    //     if (e[0] >= basal_vtx_start && e[1] >= basal_vtx_start)
    //     {
    //         computeEdgeEnergyRestLength3DHessian(gamma, l0, x, spring_hessian);
    //     }
    //     bool case1 = e[0] < basal_vtx_start && e[1] >= basal_vtx_start;
    //     bool case2 = e[1] < basal_vtx_start && e[0] >= basal_vtx_start;
    //     if (case1 || case2)
    //     {
    //         computeEdgeEnergyRestLength3DHessian(alpha, l0, x, spring_hessian);
    //     }
    //     addHessianEntry<6>(entries, {e[0], e[1]}, spring_hessian);
    // });
    Timer tp_timer(true);
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
    if (profile)
    {
        std::cout << tp_timer.elapsed_sec() << "s" << std::endl;
        tp_timer.restart();
    }
    if (dynamics)
        addInertialHessianEntries(entries);
    
    if (use_elastic_potential)
    {
        addElasticityHessianEntries(entries, project_block_hessian_PD);
    }
    else
    {
        addEdgeHessianEntries(Apical, sigma, entries, project_block_hessian_PD);
        addEdgeHessianEntries(Basal, gamma, entries, project_block_hessian_PD);
        addEdgeHessianEntries(Lateral, alpha, entries, project_block_hessian_PD);
        // addEdgeHessianEntries(ALL, weights_all_edges, entries, project_block_hessian_PD);
        if (profile)
        {
            std::cout << tp_timer.elapsed_sec() << "s" << std::endl;
            tp_timer.restart();
        }
        if (preserve_tet_vol)
            addTetVolumePreservationHessianEntries(entries, project_block_hessian_PD);
        else
        {
            if (use_fixed_centroid)
                addCellVolumePreservationHessianEntriesFixedCentroid(entries, project_block_hessian_PD);
            else
                addCellVolumePreservationHessianEntries(entries, project_block_hessian_PD);
        }
        if (profile)
        {
            std::cout << tp_timer.elapsed_sec() << "s" << std::endl;
            tp_timer.restart();
        }
        // ===================================== Face Area =====================================
        if (add_area_term)
        {
            // if (has_rest_shape)
            // {
            //     addFaceAreaHessianEntriesWithRestShape(Apical, sigma, entries, project_block_hessian_PD);
            //     addFaceAreaHessianEntriesWithRestShape(Basal, gamma, entries, project_block_hessian_PD);
            //     addFaceAreaHessianEntriesWithRestShape(Lateral, alpha, entries, project_block_hessian_PD);
            // }
            // else
            {
                addFaceAreaHessianEntries(Apical, sigma, entries, project_block_hessian_PD);
                addFaceAreaHessianEntries(Basal, gamma, entries, project_block_hessian_PD);
                addFaceAreaHessianEntries(Lateral, alpha, entries, project_block_hessian_PD);
                if (profile)
                {
                    std::cout << tp_timer.elapsed_sec() << "s" << std::endl;
                    tp_timer.restart();
                }
            }
        }
        

    }

    if (add_tet_vol_barrier)
    {
        if (use_cell_centroid)
            addSingleTetVolBarrierHessianEntries(entries, project_block_hessian_PD);
        else
            addFixedTetLogBarrierHessianEneries(entries, project_block_hessian_PD);
    }
    if (profile)
    {
        std::cout << tp_timer.elapsed_sec() << "s" << std::endl;
        tp_timer.restart();
    }

    if (add_yolk_volume)
        addYolkVolumePreservationHessianEntries(entries, UV, project_block_hessian_PD);
    if (add_yolk_tet_barrier)
        addYolkTetLogBarrierHessianEneries(entries, project_block_hessian_PD);
    if (profile)
    {
        std::cout << tp_timer.elapsed_sec() << "s" << std::endl;
        tp_timer.restart();
    }
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
    if (profile)
    {
        std::cout << tp_timer.elapsed_sec() << "s" << std::endl;
        tp_timer.restart();
    }
    // ===================================== IPC =====================================
    if (use_ipc_contact)
    {
        addIPCHessianEntries(entries, project_block_hessian_PD);
    }
    if (profile)
    {
        std::cout << tp_timer.elapsed_sec() << "s" << std::endl;
        tp_timer.restart();
    }
    if (use_pre_build_structure)
    {
        std::cout << "use prebuild structure" << std::endl;
        // K.resize(num_nodes * 3, num_nodes * 3);
        // K.reserve(Hessian_copy.nonZeros());
        // val.resize(Hessian_copy.nonZeros());
        // memset(&val[0], 0.0, K.nonZeros() * sizeof(T));
        // for (auto entry : entries)
        // {
        //     std::ptrdiff_t ptr_diff = ij_to_value_array[IV2(entry.row(), entry.col())];
        //     *(&val[0] + ptr_diff) += entry.value();
        // }
        // tp_timer.restart();
        K = Hessian_copy;
        // memset(K.valuePtr(), 0.0, K.nonZeros() * sizeof(T));
        // StiffnessMatrix test(num_nodes * 3, num_nodes * 3);
        // test.setFromTriplets(entries.begin(), entries.end());
        // // K.setZero();
        // // VectorXT value(Hessian_copy.nonZeros());
        // for (auto entry : entries)
        // {
        //     // if (ij_to_value_array.find(IV2(entry.row(), entry.col())) != ij_to_value_array.end())
        //     {
        //         std::ptrdiff_t ptr_diff = ij_to_value_array[IV2(entry.row(), entry.col())];
        //         // std::cout << ptr_diff << " " << val.size() << std::endl;
        //         // std::getchar();
        //         // *(&val[0] + ptr_diff) += entry.value();
        //         *(K.valuePtr() + ptr_diff) += entry.value();
        //     }
        //     // else
        //     // {
        //     //     std::cout << IV2(entry.row(), entry.col()).transpose() << std::endl;
        //     //     std::getchar();
        //     // }
        // }
        
        memcpy(K.valuePtr(), val.data(),
            val.size() * sizeof(val[0]));
        std::vector<T>().swap(val);
        // memcpy(K.innerIndexPtr(), inner_indices.data(),
        //     inner_indices.size() * sizeof(inner_indices[0]));
        // memcpy(K.outerIndexPtr(), outer_indices.data(),
        //     outer_indices.size() * sizeof(outer_indices[0]));
        
        K.finalize();
        // std::cout << (K - test).norm() << std::endl;
        // std::getchar();
    }
    else
    {
        K.resize(num_nodes * 3, num_nodes * 3);
        K.reserve(0.5 * entries.size());
        K.setFromTriplets(entries.begin(), entries.end());
    }
    
    if (!run_diff_test)
        projectDirichletDoFMatrix(K, dirichlet_data);
    if (profile)
    {
        std::cout << "assemble " << tp_timer.elapsed_sec() << "s" << std::endl;
        tp_timer.restart();
    }
    if (!use_pre_build_structure)
    {
        K.makeCompressed();
        // val.resize(K.innerSize());
        // memcpy(val.data(), K.valuePtr(),
        //     val.size() * sizeof(T));
        for (int i = 0; i < K.outerSize(); i++)
        {
            for (StiffnessMatrix::InnerIterator it(K, i); it; ++it)
            {
                T* ref = &it.valueRef();
                std::ptrdiff_t prt_diff = ref - K.valuePtr();
                // std::cout << prt_diff << std::endl;
                // std::getchar();
                // ij_to_value_array[IV2(it.row(), it.col())] = prt_diff;
                uint64_t lower_bit = it.row();
                uint64_t upper_bit = it.col();
                uint64_t whole = upper_bit << 32 | lower_bit;
                ijv[whole] = prt_diff;
                // *(K.valuePtr() + diff) = value;
            }
        }
        // std::cout << K.nonZeros() << std::endl;
        // std::getchar();
        
        // inner_indices.resize(sizeof(K.innerIndexPtr()));
        // outer_indices.resize(sizeof(K.outerIndexPtr()));
        // std::cout << inner_indices.size() << " " << K.nonZeros() << std::endl;
        // std::getchar();
        // memcpy(inner_indices.data(), K.innerIndexPtr(),
        //     inner_indices.size() * sizeof(StorageIndex));
        // memcpy(outer_indices.data(), K.outerIndexPtr(), 
        //     outer_indices.size() * sizeof(StorageIndex));
        Hessian_copy = K;
        use_pre_build_structure = true;
    }
    val.resize(Hessian_copy.nonZeros());
    memset(&val[0], 0.0, Hessian_copy.nonZeros() * sizeof(T));
    // std::getchar();
}


void VertexModel::buildSystemMatrixWoodbury(const VectorXT& _u, StiffnessMatrix& K, MatrixXT& UV)
{
    bool profile = false;
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
    entries.reserve(Hessian_copy.nonZeros());
    
    Timer tp_timer(true);
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
    if (profile)
    {
        std::cout << tp_timer.elapsed_sec() << "s" << std::endl;
        tp_timer.restart();
    }
    if (dynamics)
        addInertialHessianEntries(entries);
    
    if (use_elastic_potential)
    {
        addElasticityHessianEntries(entries, project_block_hessian_PD);
    }
    else
    {
        addEdgeHessianEntries(Apical, sigma, entries, project_block_hessian_PD);
        addEdgeHessianEntries(Basal, gamma, entries, project_block_hessian_PD);
        addEdgeHessianEntries(Lateral, alpha, entries, project_block_hessian_PD);
        // addEdgeHessianEntries(ALL, weights_all_edges, entries, project_block_hessian_PD);
        if (profile)
        {
            std::cout << tp_timer.elapsed_sec() << "s" << std::endl;
            tp_timer.restart();
        }
        if (preserve_tet_vol)
            addTetVolumePreservationHessianEntries(entries, project_block_hessian_PD);
        else
        {
            if (use_fixed_centroid)
                addCellVolumePreservationHessianEntriesFixedCentroid(entries, project_block_hessian_PD);
            else
                addCellVolumePreservationHessianEntries(entries, project_block_hessian_PD);
        }
        if (profile)
        {
            std::cout << tp_timer.elapsed_sec() << "s" << std::endl;
            tp_timer.restart();
        }
        // ===================================== Face Area =====================================
        if (add_area_term)
        {
            // if (has_rest_shape)
            // {
            //     addFaceAreaHessianEntriesWithRestShape(Apical, sigma, entries, project_block_hessian_PD);
            //     addFaceAreaHessianEntriesWithRestShape(Basal, gamma, entries, project_block_hessian_PD);
            //     addFaceAreaHessianEntriesWithRestShape(Lateral, alpha, entries, project_block_hessian_PD);
            // }
            // else
            {
                addFaceAreaHessianEntries(Apical, sigma, entries, project_block_hessian_PD);
                addFaceAreaHessianEntries(Basal, gamma, entries, project_block_hessian_PD);
                addFaceAreaHessianEntries(Lateral, alpha, entries, project_block_hessian_PD);
                if (profile)
                {
                    std::cout << tp_timer.elapsed_sec() << "s" << std::endl;
                    tp_timer.restart();
                }
            }
        }
        

    }

    if (add_tet_vol_barrier)
    {
        if (use_cell_centroid)
            addSingleTetVolBarrierHessianEntries(entries, project_block_hessian_PD);
        else
            addFixedTetLogBarrierHessianEneries(entries, project_block_hessian_PD);
    }
    if (profile)
    {
        std::cout << tp_timer.elapsed_sec() << "s" << std::endl;
        tp_timer.restart();
    }

    if (add_yolk_volume)
        addYolkVolumePreservationHessianEntries(entries, UV, project_block_hessian_PD);
    if (add_yolk_tet_barrier)
        addYolkTetLogBarrierHessianEneries(entries, project_block_hessian_PD);
    if (profile)
    {
        std::cout << tp_timer.elapsed_sec() << "s" << std::endl;
        tp_timer.restart();
    }
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
    if (profile)
    {
        std::cout << tp_timer.elapsed_sec() << "s" << std::endl;
        tp_timer.restart();
    }
    // ===================================== IPC =====================================
    if (use_ipc_contact)
    {
        addIPCHessianEntries(entries, project_block_hessian_PD);
    }
    if (profile)
    {
        std::cout << tp_timer.elapsed_sec() << "s" << std::endl;
        tp_timer.restart();
    }
    K.resize(num_nodes * 3, num_nodes * 3);
    K.reserve(0.5 * entries.size());
    K.setFromTriplets(entries.begin(), entries.end());
    Hessian_copy = K;
    if (!run_diff_test)
        projectDirichletDoFMatrix(K, dirichlet_data);
    if (profile)
    {
        std::cout << "assemble " << tp_timer.elapsed_sec() << "s" << std::endl;
        tp_timer.restart();
    }
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
        addEdgeHessianEntries(Apical, sigma, entries, project_block_hessian_PD);
        addEdgeHessianEntries(Basal, gamma, entries, project_block_hessian_PD);
        addEdgeHessianEntries(Lateral, alpha, entries, project_block_hessian_PD);
        // addEdgeHessianEntries(ALL, weights_all_edges, entries, project_block_hessian_PD);
        if (preserve_tet_vol)
            addTetVolumePreservationHessianEntries(entries, project_block_hessian_PD);
        else
        {
            if (use_fixed_centroid)
                addCellVolumePreservationHessianEntriesFixedCentroid(entries, project_block_hessian_PD);
            else
                addCellVolumePreservationHessianEntries(entries, project_block_hessian_PD);
        }
        if (add_area_term)
        {
            // if (has_rest_shape)
            // {
            //     addFaceAreaHessianEntriesWithRestShape(Apical, sigma, entries, project_block_hessian_PD);
            //     addFaceAreaHessianEntriesWithRestShape(Basal, gamma, entries, project_block_hessian_PD);
            //     addFaceAreaHessianEntriesWithRestShape(Lateral, alpha, entries, project_block_hessian_PD);
            // }
            // else
            {
                addFaceAreaHessianEntries(Apical, sigma, entries, project_block_hessian_PD);
                addFaceAreaHessianEntries(Basal, gamma, entries, project_block_hessian_PD);
                addFaceAreaHessianEntries(Lateral, alpha, entries, project_block_hessian_PD);
            }
        }
        
        
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
    K.reserve(entries.size() / 2);
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