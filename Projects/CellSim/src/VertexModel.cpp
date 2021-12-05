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


void VertexModel::computeLinearModes()
{
    int nmodes = 15;

    StiffnessMatrix K(deformed.rows(), deformed.rows());
    run_diff_test = true;
    buildSystemMatrix(u, K);
    
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
        eigen_solver.compute(A_dense, /* computeEigenvectors = */ false);
        auto eigen_values = eigen_solver.eigenvalues();
        T min_ev = 1e10;
        T second_min_ev = 1e10;
        for (int i = 0; i < A_dense.cols(); i++)
            if (eigen_values[i].real() < min_ev)
            {
                min_ev = eigen_values[i].real();
            }
        
        // std::cout << min_ev << std::endl;
        
        std::vector<T> ev_all(A_dense.cols());
        for (int i = 0; i < A_dense.cols(); i++)
        {
            ev_all[i] = eigen_values[i].real();
        }
        std::sort(ev_all.begin(), ev_all.end());
        // for (int i = 0; i < 10; i++)
        for (int i = 0; i < nmodes; i++)
            std::cout << ev_all[i] << " ";
        std::cout << std::endl;
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
    
    // ===================================== Edge length =====================================
    iterateApicalEdgeSerial([&](Edge& e){    
        TV vi = deformed.segment<3>(e[0] * 3);
        TV vj = deformed.segment<3>(e[1] * 3);
        T edge_length = computeEdgeSquaredNorm(vi, vj);
        edge_length_term += sigma * edge_length;

    });

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

    if (verbose)
    {
        std::cout << "\tE_edge " << edge_length_term << std::endl;
        if (add_contraction_term)
            std::cout << "\tE_contract " << contraction_term << std::endl;
    }
    energy += edge_length_term;
    energy += contraction_term;

    iterateFaceSerial([&](VtxList& face_vtx_list, int face_idx)
    {
        
        if (face_idx < basal_face_start)
        {
            VectorXT positions;
            VtxList cell_vtx_list = face_vtx_list;
            for (int idx : face_vtx_list)
                cell_vtx_list.push_back(idx + basal_vtx_start);
            T volume_barrier_energy = 0.0;
            positionsFromIndices(positions, cell_vtx_list);

            if (add_single_tet_vol_barrier)
            {
                if(face_vtx_list.size() == 6)
                {
                    if (use_cell_centroid)
                    {
                        computeVolumeBarrier6Points(tet_barrier_stiffness, positions, volume_barrier_energy);
                    }
                    else
                        computeHexBasePrismVolumeBarrier(tet_barrier_stiffness, positions, volume_barrier_energy);
                    energy += volume_barrier_energy;
                    // std::cout << volume_barrier_energy << std::endl;
                }
            }
        }
    });

    // ===================================== Cell Volume =====================================
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

    if (add_yolk_volume)
    {
        addYolkVolumePreservationEnergy(yolk_volume_term);
        if (verbose)
            std::cout << "\tE_yolk_vol " << yolk_volume_term << std::endl;
    }

    energy += yolk_volume_term;

    if (use_sphere_radius_bound)
    {
        for (int i = 0; i < basal_vtx_start; i++)
        {
            T e = 0.0;;
            T Rk = (deformed.segment<3>(i * 3) - mesh_centroid).norm();
            if (sphere_bound_penalty)
            {
                if (Rk >= Rc)
                {
                    computeRadiusPenalty(bound_coeff, Rc, deformed.segment<3>(i * 3), mesh_centroid, e);
                    sphere_bound_term += e;
                }
            }
            else
            {
                sphereBoundEnergy(bound_coeff, Rc, deformed.segment<3>(i * 3), mesh_centroid, e);
                // std::cout << e << " Rk " << Rk << " Rc " << Rc << std::endl;
                // std::getchar();
                sphere_bound_term += e;
            }
        }
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
    if (use_ipc_contact)
        updateIPCVertices(_u);
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

    // ===================================== Edge length =====================================
    iterateApicalEdgeSerial([&](Edge& e){
        TV vi = deformed.segment<3>(e[0] * 3);
        TV vj = deformed.segment<3>(e[1] * 3);
        Vector<T, 6> dedx;
        computeEdgeSquaredNormGradient(vi, vj, dedx);
        dedx *= -sigma;
        addForceEntry<6>(residual, {e[0], e[1]}, dedx);
    });

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
        std::cout << "\tedge length force norm: " << (residual - residual_temp).norm() << std::endl;
    residual_temp = residual;

    // ===================================== Cell Volume =====================================

    iterateFaceSerial([&](VtxList& face_vtx_list, int face_idx)
    {
        // cell-wise volume preservation term
        if (face_idx < basal_face_start)
        {
            VectorXT positions;
            VtxList cell_vtx_list = face_vtx_list;
            for (int idx : face_vtx_list)
                cell_vtx_list.push_back(idx + basal_vtx_start);

            positionsFromIndices(positions, cell_vtx_list);

            // cell-wise volume preservation term
            if (face_idx < basal_face_start)
            {
                         
                if (face_vtx_list.size() == 6)
                {
                    Vector<T, 36> dedx;
                    if (add_single_tet_vol_barrier)
                    {
                        if (use_cell_centroid)
                            computeVolumeBarrier6PointsGradient(tet_barrier_stiffness, positions, dedx);
                        else
                            computeHexBasePrismVolumeBarrierGradient(tet_barrier_stiffness, positions, dedx);
                        dedx *= -1;
                        addForceEntry<36>(residual, cell_vtx_list, dedx);
                    }
                }
            }
        }
    });

    addCellVolumePreservationForceEntries(residual);

    if (print_force_norm)
        std::cout << "\tcell volume preservation force norm: " << (residual - residual_temp).norm() << std::endl;
    residual_temp = residual;
    // ===================================== Face Area =====================================
    
    addFaceAreaForceEntries(Basal, gamma, residual);
    addFaceAreaForceEntries(Lateral, alpha, residual);

    if (print_force_norm)
        std::cout << "\tarea force norm: " << (residual - residual_temp).norm() << std::endl;
    residual_temp = residual;

    // ===================================== Yolk =====================================

    if (add_yolk_volume)
        addYolkVolumePreservationForceEntries(residual);
    
    if (print_force_norm)
        std::cout << "\tyolk volume preservation force norm: " << (residual - residual_temp).norm() << std::endl;
    residual_temp = residual;

    // ===================================== Membrane =====================================
    if (use_sphere_radius_bound)
    {
        for (int i = 0; i < basal_vtx_start; i++)
        {
            Vector<T, 3> dedx;
            if (sphere_bound_penalty)
            {
                T Rk = (deformed.segment<3>(i * 3) - mesh_centroid).norm();
                if (Rk >= Rc)
                {
                    computeRadiusPenaltyGradient(bound_coeff, Rc, deformed.segment<3>(i * 3), mesh_centroid, dedx);
                    addForceEntry<3>(residual, {i}, -dedx);
                }
            }
            else
            {
                sphereBoundEnergyGradient(bound_coeff, Rc, deformed.segment<3>(i*3), mesh_centroid, dedx);
                // std::cout << dedx.transpose() << std::endl;
                addForceEntry<3>(residual, {i}, -dedx);
            }
        }
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
    
    // edge length term
    iterateApicalEdgeSerial([&](Edge& e){
        TV vi = deformed.segment<3>(e[0] * 3);
        TV vj = deformed.segment<3>(e[1] * 3);
        Matrix<T, 6, 6> hessian;
        computeEdgeSquaredNormHessian(vi, vj, hessian);
        hessian *= sigma;
        addHessianEntry<6>(entries, {e[0], e[1]}, hessian);
    });

    if (add_contraction_term)
    {
        if (contract_apical_face)
            addFaceAreaHessianEntries(Apical, Gamma, entries, false);
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

    if (add_yolk_volume)
        addYolkVolumePreservationHessianEntries(entries, UV, false);
    if (add_perivitelline_liquid_volume)
        addPerivitellineVolumePreservationHessianEntries(entries, UV, false);
    
    
    // ===================================== Cell Volume =====================================
    iterateFaceSerial([&](VtxList& face_vtx_list, int face_idx)
    {
        // cell-wise volume preservation term
        if (face_idx < basal_face_start)
        {
            VectorXT positions;
            VtxList cell_vtx_list = face_vtx_list;
            for (int idx : face_vtx_list)
                cell_vtx_list.push_back(idx + basal_vtx_start);

            positionsFromIndices(positions, cell_vtx_list);
            
            if (face_vtx_list.size() == 6)
            {
                Matrix<T, 36, 36> hessian;

                if(add_single_tet_vol_barrier)
                {
                    if (use_cell_centroid)
                    {
                        computeVolumeBarrier6PointsHessian(tet_barrier_stiffness, positions, hessian);
                    }
                    addHessianEntry<36>(entries, cell_vtx_list, hessian);
                }
            }
        }
        
    });

    addCellVolumePreservationHessianEntries(entries, false);
    // ===================================== Face Area =====================================
    addFaceAreaHessianEntries(Basal, gamma, entries, false);
    addFaceAreaHessianEntries(Lateral, alpha, entries, false);
    // ===================================== Membrane =====================================
    if (use_sphere_radius_bound)
    {
        for (int i = 0; i < basal_vtx_start; i++)
        {
            Matrix<T, 3, 3> hessian;
            if (sphere_bound_penalty)
            {
                T Rk = (deformed.segment<3>(i * 3) - mesh_centroid).norm();
                if (Rk >= Rc)
                {
                    computeRadiusPenaltyHessian(bound_coeff, Rc, deformed.segment<3>(i * 3), mesh_centroid, hessian);
                    addHessianEntry<3>(entries, {i}, hessian);    
                }
            }
            else
            {
                sphereBoundEnergyHessian(bound_coeff, Rc, deformed.segment<3>(i*3), mesh_centroid, hessian);
                addHessianEntry<3>(entries, {i}, hessian);
            }
        }
    }

    // ===================================== IPC =====================================
    if (use_ipc_contact)
    {
        addIPCHessianEntries(entries);
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
    
    // edge length term
    iterateApicalEdgeSerial([&](Edge& e){
        TV vi = deformed.segment<3>(e[0] * 3);
        TV vj = deformed.segment<3>(e[1] * 3);
        Matrix<T, 6, 6> hessian;
        computeEdgeSquaredNormHessian(vi, vj, hessian);
        hessian *= sigma;
        addHessianEntry<6>(entries, {e[0], e[1]}, hessian);
    });

    if (add_contraction_term)
    {
        if (contract_apical_face)
            addFaceAreaHessianEntries(Apical, Gamma, entries, false);
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
    MatrixXT dummy_WoodBury_matrix;

    if (add_yolk_volume)
        addYolkVolumePreservationHessianEntries(entries, dummy_WoodBury_matrix, false);

    if (add_perivitelline_liquid_volume)
        addPerivitellineVolumePreservationHessianEntries(entries, dummy_WoodBury_matrix, false);
    

    iterateFaceSerial([&](VtxList& face_vtx_list, int face_idx)
    {
        // cell-wise volume preservation term
        if (face_idx < basal_face_start)
        {
            VectorXT positions;
            VtxList cell_vtx_list = face_vtx_list;
            for (int idx : face_vtx_list)
                cell_vtx_list.push_back(idx + basal_vtx_start);

            positionsFromIndices(positions, cell_vtx_list);
            
            if (face_vtx_list.size() == 6)
            {
                Matrix<T, 36, 36> hessian;
                if (add_single_tet_vol_barrier)
                {
                    if (use_cell_centroid)
                        computeVolumeBarrier6PointsHessian(tet_barrier_stiffness, positions, hessian);
                    else
                        computeHexBasePrismVolumeBarrierHessian(tet_barrier_stiffness, positions, hessian);
                    addHessianEntry<36>(entries, cell_vtx_list, hessian);
                }
            }
            
        }
        
    });

    addCellVolumePreservationHessianEntries(entries, false);

    addFaceAreaHessianEntries(Basal, gamma, entries, false);
    addFaceAreaHessianEntries(Lateral, gamma, entries, false);
    

    if (use_sphere_radius_bound)
    {
        for (int i = 0; i < basal_vtx_start; i++)
        {
            Matrix<T, 3, 3> hessian;
            if (sphere_bound_penalty)
            {
                T Rk = (deformed.segment<3>(i * 3) - mesh_centroid).norm();
                if (Rk >= Rc)
                {
                    computeRadiusPenaltyHessian(bound_coeff, Rc, deformed.segment<3>(i * 3), mesh_centroid, hessian);
                    addHessianEntry<3>(entries, {i}, hessian);    
                }
            }
            else
            {
                sphereBoundEnergyHessian(bound_coeff, Rc, deformed.segment<3>(i*3), mesh_centroid, hessian);
                addHessianEntry<3>(entries, {i}, hessian);
            }
        }
    }

    if (use_ipc_contact)
    {
        addIPCHessianEntries(entries);

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
