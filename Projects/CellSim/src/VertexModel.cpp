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


void VertexModel::updateFixedCellCentroid()
{
    fixed_cell_centroids = VectorXT::Zero(basal_face_start * 3);
    iterateFaceSerial([&](VtxList& face_vtx_list, int face_idx)
    {
        VectorXT positions;
        positionsFromIndices(positions, face_vtx_list);
        T area_energy = 0.0;
        
        // cell-wise volume preservation term
        if (face_idx < basal_face_start)
        {
            TV centroid;
            computeCellCentroid(face_vtx_list, centroid);
            fixed_cell_centroids.segment<3>(face_idx * 3) = centroid;
        }
    });
}

void VertexModel::computeCellCentroid(const VtxList& face_vtx_list, TV& centroid)
{
    centroid = TV::Zero();
    for (int vtx_idx : face_vtx_list)
    {
        centroid += deformed.segment<3>(vtx_idx * 3);
        centroid += deformed.segment<3>((vtx_idx + basal_vtx_start) * 3);
    }
    centroid /= T(face_vtx_list.size() * 2);
}

void VertexModel::computeFaceCentroid(const VtxList& face_vtx_list, TV& centroid)
{
    centroid = TV::Zero();
    for (int vtx_idx : face_vtx_list)
        centroid += deformed.segment<3>(vtx_idx * 3);

    centroid /= T(face_vtx_list.size());
}


T VertexModel::computeTotalVolumeFromApicalSurface()
{
    T volume = 0.0;
    iterateFaceSerial([&](VtxList& face_vtx_list, int face_idx)
    {
        VectorXT positions;
        positionsFromIndices(positions, face_vtx_list);
        
        if (face_idx < basal_face_start)
        {
            T cone_volume;
            if (face_vtx_list.size() == 4) 
                computeConeVolume4Points(positions, mesh_centroid, cone_volume);
            else if (face_vtx_list.size() == 5) 
                computeConeVolume5Points(positions, mesh_centroid, cone_volume);
            else if (face_vtx_list.size() == 6) 
                computeConeVolume6Points(positions, mesh_centroid, cone_volume);
            else
                std::cout << "unknown polygon edge number" << __FILE__ << std::endl;
            volume += cone_volume;
        }
        
    });
    return -volume;
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

void VertexModel::computeVolumeAllCells(VectorXT& cell_volume_list)
{
    // each apical face corresponds to one cell
    cell_volume_list = VectorXT::Ones(basal_face_start);

    // use apical face to iterate other faces within this cell for now
    iterateFaceSerial([&](VtxList& face_vtx_list, int face_idx){
        if (face_idx < basal_face_start)
        {
            VectorXT positions;
            VtxList cell_vtx_list = face_vtx_list;
            for (int idx : face_vtx_list)
                cell_vtx_list.push_back(idx + basal_vtx_start);
            
            // for (int idx : cell_vtx_list)
            //     std::cout << idx << " ";
            // std::cout << std::endl;

            positionsFromIndices(positions, cell_vtx_list);
            
            if (face_vtx_list.size() == 4)
            {
                if (use_cell_centroid)
                    computeVolume4Points(positions, cell_volume_list[face_idx]);
                else
                    computeQuadBasePrismVolume(positions, cell_volume_list[face_idx]);
                // T tet_vol;
                
                // computeCubeVolumeFromTet(positions, tet_vol);
                // // // computeCubeVolumeCentroid(positions, tet_vol);
                // std::cout << tet_vol << std::endl;
                // std::getchar();
            }
            else if (face_vtx_list.size() == 5)
            {
                if (use_cell_centroid)
                    computeVolume5Points(positions, cell_volume_list[face_idx]);
                else
                    computePentaBasePrismVolume(positions, cell_volume_list[face_idx]);
                
                // T tet_vol;
                // computePentaPrismVolumeFromTet(deformed, tet_vol);
                // std::cout << tet_vol << std::endl;
                // computePentaPrismVolumeFromTet(positions, tet_vol);
                // std::cout << tet_vol << std::endl;
                // std::cout << cell_volume_list[face_idx] << std::endl;
                // std::getchar();
            }
            else if (face_vtx_list.size() == 6)
            {
                if (use_cell_centroid)
                {
                    if (use_fixed_cell_centroid)
                        computeVolume6PointsFixedCentroid(positions, fixed_cell_centroids.segment<3>(face_idx * 3), cell_volume_list[face_idx]);
                    else
                        computeVolume6Points(positions, cell_volume_list[face_idx]);
                }
                else
                    computeHexBasePrismVolume(positions, cell_volume_list[face_idx]);
                // std::cout << cell_volume_list[face_idx] << std::endl;
                // T tet_vol, tet_ad;
                // computeVolume6Points(positions, tet_ad);
                // computeHexPrismVolumeFromTet(positions, tet_vol);
                // std::cout << "tet manual " << tet_vol << " tet ad " << tet_ad << std::endl;
                // std::getchar();
            }
        }
    });
}


T VertexModel::computeYolkVolume(bool verbose)
{
    T yolk_volume = 0.0;
    if (verbose)
        std::cout << "yolk tet volume: " << std::endl;
    iterateFaceSerial([&](VtxList& face_vtx_list, int face_idx)
    {
        VectorXT positions;
        positionsFromIndices(positions, face_vtx_list);
        
        if (face_idx < lateral_face_start && face_idx >= basal_face_start)
        {
            T cone_volume;
            if (face_vtx_list.size() == 4) 
                computeConeVolume4Points(positions, mesh_centroid, cone_volume);
                // computeQuadConeVolume(positions, mesh_centroid, cone_volume);
            else if (face_vtx_list.size() == 5) 
                computeConeVolume5Points(positions, mesh_centroid, cone_volume);
            else if (face_vtx_list.size() == 6) 
                computeConeVolume6Points(positions, mesh_centroid, cone_volume);
            else
                std::cout << "unknown polygon edge number" << __FILE__ << std::endl;
            yolk_volume += cone_volume;
            if (verbose)
            {
                std::cout << cone_volume << " ";
                if (cone_volume < 0)
                    std::cout << "negative volume " << cone_volume << std::endl;
            }
        }
        
    });
    if (verbose)
        std::cout << std::endl;
    return yolk_volume; 
}



T VertexModel::computeAreaEnergy(const VectorXT& _u)
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

    T energy = 0.0;
    use_face_centroid = true;
    // std::cout << lateral_face_start << std::endl;
    // std::cout << faces.size() << std::endl;
    iterateFaceSerial([&](VtxList& face_vtx_list, int face_idx)
    {
        VectorXT positions;
        positionsFromIndices(positions, face_vtx_list);
        T area_energy = 0.0;
        // cell-wise volume preservation term
        
        // if (face_idx >= basal_face_start + 43)
        // if (face_idx >= lateral_face_start)
        if (face_idx == 196)
        {
            // std::ofstream out("bug_face.obj");
            // for (int i = 0; i < face_vtx_list.size(); i++)
            // {
            //     out << "v " << positions.segment<3>(i * 3).transpose() << std::endl;
            // }
            // TV face_centroid = TV::Zero();
            // computeFaceCentroid(face_vtx_list, face_centroid);
            // out << "v " << face_centroid.transpose() << std::endl;
            
            // out << "f 5 1 2 " << std::endl;
            // out << "f 5 2 3 " << std::endl;
            // out << "f 5 3 4 " << std::endl;
            // out << "f 5 4 1 " << std::endl;
            // out.close();
            // std::getchar();
            T coeff = face_idx >= lateral_face_start ? alpha : gamma;
            if (face_vtx_list.size() == 4)
            {
                // computeArea4PointsSquaredSum(coeff, positions, area_energy);
                if (use_face_centroid)
                {
                    computeArea4PointsSquared(coeff, positions, area_energy);
                    area_energy = std::sqrt(area_energy);
                }
                else
                    computeQuadFaceAreaSquaredSum(coeff, positions, area_energy);
                // computeArea4PointsSquared(coeff, positions, area_energy);
                // computeQuadFaceArea(coeff, positions, area_energy);
            }
            else if (face_vtx_list.size() == 5)
            {
                if (use_face_centroid)
                    computeArea5PointsSquared(coeff, positions, area_energy);
                else
                    computePentFaceAreaSquaredSum(coeff, positions, area_energy);
            }
            else if (face_vtx_list.size() == 6)
            {
                if (use_face_centroid)
                    computeArea6PointsSquared(coeff, positions, area_energy);
                else
                    computeHexFaceAreaSquaredSum(coeff, positions, area_energy);
            }
            else
                std::cout << "unknown polygon edge case" << std::endl;
        }
        energy += area_energy;
    });
    return energy;
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
        edge_contraction_term = 0.0, sphere_bound_term = 0.0;
    
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
        iterateContractingEdgeSerial([&](Edge& e){    
            TV vi = deformed.segment<3>(e[0] * 3);
            TV vj = deformed.segment<3>(e[1] * 3);
            T edge_length = computeEdgeSquaredNorm(vi, vj);
            edge_contraction_term += Gamma * edge_length;
        });
    }

    if (verbose)
    {
        std::cout << "\tE_edge " << edge_length_term << std::endl;
        if (add_contraction_term)
            std::cout << "\tE_edge_contract " << edge_contraction_term << std::endl;
    }
    energy += edge_length_term;
    energy += edge_contraction_term;

    
    VectorXT current_cell_volume;
    computeVolumeAllCells(current_cell_volume);
    

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
    iterateFaceSerial([&](VtxList& face_vtx_list, int face_idx)
    {
        VectorXT positions;
        positionsFromIndices(positions, face_vtx_list);
        T area_energy = 0.0;
        
        // cell-wise volume preservation term
        if (face_idx < basal_face_start)
        {
            T ci = current_cell_volume[face_idx] - cell_volume_init[face_idx];
            if (use_alm_on_cell_volume)
                volume_term += -lambda_cell_vol[face_idx] * ci + 0.5 * kappa * std::pow(ci, 2);
            else
                volume_term += 0.5 * B * std::pow(ci, 2);
            
        }
        else // basal and lateral faces area term
        {
            T coeff = face_idx >= lateral_face_start ? alpha : gamma;
            if (face_vtx_list.size() == 4)
            {
                // computeArea4PointsSquaredSum(coeff, positions, area_energy);
                if (use_face_centroid)
                    computeArea4PointsSquared(coeff, positions, area_energy);
                else
                    computeQuadFaceAreaSquaredSum(coeff, positions, area_energy);
                // computeArea4PointsSquared(coeff, positions, area_energy);
                // computeQuadFaceArea(coeff, positions, area_energy);
            }
            else if (face_vtx_list.size() == 5)
            {
                if (use_face_centroid)
                    computeArea5PointsSquared(coeff, positions, area_energy);
                else
                    computePentFaceAreaSquaredSum(coeff, positions, area_energy);
            }
            else if (face_vtx_list.size() == 6)
            {
                if (use_face_centroid)
                    computeArea6PointsSquared(coeff, positions, area_energy);
                else
                    computeHexFaceAreaSquaredSum(coeff, positions, area_energy);
            }
            else
                std::cout << "unknown polygon edge case" << std::endl;
        }
        area_term += area_energy;
    });

    if (verbose)
    {
        std::cout << "\tE_area: " << area_term << std::endl;
        std::cout << "\tE_volume: " << volume_term << std::endl;
    }

    energy += volume_term;
    energy += area_term;

    if (add_yolk_volume)
    {
        T yolk_vol_curr = computeYolkVolume();
        if (use_yolk_pressure)
        {
            yolk_volume_term += -pressure_constant * yolk_vol_curr;
        }
        else
        {
            yolk_volume_term +=  0.5 * By * std::pow(yolk_vol_curr - yolk_vol_init, 2);    
        }
    }
    if (verbose)
        std::cout << "\tE_yolk_vol " << yolk_volume_term << std::endl;

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
        Eigen::MatrixXd ipc_vertices_deformed(basal_vtx_start, 3);
        for (int i = 0; i < basal_vtx_start; i++)
        {
            ipc_vertices_deformed.row(i) = deformed.segment<3>(i * 3);
        }
        
        ipc::Constraints ipc_constraints;
        ipc::construct_constraint_set(ipc_vertices, ipc_vertices_deformed, 
            ipc_edges, ipc_faces, barrier_distance, ipc_constraints);

        try
        {
            contact_energy = barrier_weight * ipc::compute_barrier_potential(ipc_vertices_deformed, 
            ipc_edges, ipc_faces, ipc_constraints, barrier_distance);
        }
        catch(const std::runtime_error& e)
        {
            std::cout << "error catch " << std::endl;
            std::cerr << e.what() << '\n';
        }
        

        energy += contact_energy;
        if (verbose)
            std::cout << "\tE_contact: " << contact_energy << std::endl;
        // std::getchar();
    }

    if (add_pervitelline_liquid_volume)
    {
        T volume_penalty_previtelline = 0.0;
        T previtelline_vol_curr = total_volume - computeTotalVolumeFromApicalSurface();
        volume_penalty_previtelline += 0.5 * Bp * std::pow(previtelline_vol_curr - previtelline_vol_init, 2);
        energy += volume_penalty_previtelline;
        if (verbose)
            std::cout << "\tE_previtelline_vol: " << contact_energy << std::endl;
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

    VectorXT current_cell_volume;
    computeVolumeAllCells(current_cell_volume);

    T yolk_vol_curr = 0.0;
    if (add_yolk_volume)
    {
        yolk_vol_curr = computeYolkVolume();
    }
    
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
                T ci = (current_cell_volume[face_idx] - cell_volume_init[face_idx]);
                
                if (face_vtx_list.size() == 4)
                {
                    Vector<T, 24> dedx;
                    if (use_cell_centroid)
                        computeVolume4PointsGradient(positions, dedx);
                    else
                        computeQuadBasePrismVolumeGradient(positions, dedx);
                    if (use_alm_on_cell_volume)
                    {
                        Vector<T, 24> negative_gradient =  
                            lambda_cell_vol[face_idx] * dedx - kappa * ci * dedx;
                        addForceEntry<24>(residual, cell_vtx_list, negative_gradient);
                    }
                    else
                    {
                        dedx *= -B * ci;
                        addForceEntry<24>(residual, cell_vtx_list, dedx);
                    }
                    
                }
                else if (face_vtx_list.size() == 5)
                {
                    Vector<T, 30> dedx;
                    if (use_cell_centroid)
                        computeVolume5PointsGradient(positions, dedx);
                    else
                        computePentaBasePrismVolumeGradient(positions, dedx);
                    if (use_alm_on_cell_volume)
                    {
                        Vector<T, 30> negative_gradient =  
                            lambda_cell_vol[face_idx] * dedx - kappa * ci * dedx;
                        addForceEntry<30>(residual, cell_vtx_list, negative_gradient);
                    }
                    else
                    {
                        dedx *= -B * ci;
                        addForceEntry<30>(residual, cell_vtx_list, dedx);
                    }
                }
                else if (face_vtx_list.size() == 6)
                {
                    Vector<T, 36> dedx;
                    if (use_cell_centroid)
                    {
                        if (use_fixed_cell_centroid)
                            computeVolume6PointsFixedCentroidGradient(positions, fixed_cell_centroids.segment<3>(face_idx * 3), dedx);
                        else
                            computeVolume6PointsGradient(positions, dedx);
                    }
                    else
                        computeHexBasePrismVolumeGradient(positions, dedx);
                    if (use_alm_on_cell_volume)
                    {
                        Vector<T, 36> negative_gradient =  
                            lambda_cell_vol[face_idx] * dedx - kappa * ci * dedx;
                        addForceEntry<36>(residual, cell_vtx_list, negative_gradient);
                    }
                    else
                    {
                        dedx *= -B * ci;
                        addForceEntry<36>(residual, cell_vtx_list, dedx);
                    }

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
                else
                {
                    std::cout << "unknown polygon edge case" << std::endl;
                }
            }
        }
    });

    if (print_force_norm)
        std::cout << "\tcell volume preservation force norm: " << (residual - residual_temp).norm() << std::endl;
    residual_temp = residual;
    // ===================================== Face Area =====================================
    iterateFaceSerial([&](VtxList& face_vtx_list, int face_idx)
    {
        // else // basal and lateral faces area term
        if (face_idx >= basal_face_start)
        {
            T coeff = face_idx >= lateral_face_start ? alpha : gamma;
            VectorXT positions;
            positionsFromIndices(positions, face_vtx_list);
            if (face_vtx_list.size() == 4)
            {
                Vector<T, 12> dedx;
                if (use_face_centroid)
                    computeArea4PointsSquaredGradient(coeff, positions, dedx);
                else
                    computeQuadFaceAreaSquaredSumGradient(coeff, positions, dedx);
                // dedx *= -coeff;
                dedx *=-1;
                addForceEntry<12>(residual, face_vtx_list, dedx);
            }
            else if (face_vtx_list.size() == 5)
            {
                Vector<T, 15> dedx;
                // computeArea5PointsGradient(coeff, positions, dedx);
                // dedx *= -coeff;
                if (use_face_centroid)
                    computeArea5PointsSquaredGradient(coeff, positions, dedx);
                else
                    computePentFaceAreaSquaredSumGradient(coeff, positions, dedx);
                dedx *= -1.0;
                addForceEntry<15>(residual, face_vtx_list, dedx);
            }
            else if (face_vtx_list.size() == 6)
            {
                Vector<T, 18> dedx;
                // computeArea6PointsGradient(coeff, positions, dedx);
                // dedx *= -coeff;
                // computeHexFaceAreaEnergyGradient(coeff, positions, dedx);
                if (use_face_centroid)
                    computeArea6PointsSquaredGradient(coeff, positions, dedx);
                else
                    computeHexFaceAreaSquaredSumGradient(coeff, positions, dedx);
                dedx *= -1.0;
                addForceEntry<18>(residual, face_vtx_list, dedx);
            }
            else
            {
                std::cout << "error " << __FILE__ << std::endl;
            }
        }
    });

    if (print_force_norm)
        std::cout << "\tarea force norm: " << (residual - residual_temp).norm() << std::endl;
    residual_temp = residual;

    // ===================================== Yolk =====================================
    iterateFaceSerial([&](VtxList& face_vtx_list, int face_idx)
    {
        if (add_yolk_volume)
        {
            if (face_idx < lateral_face_start && face_idx >= basal_face_start)
            {
                
                VectorXT positions;
                positionsFromIndices(positions, face_vtx_list);
                T coeff;
                if (use_yolk_pressure)
                    coeff = -pressure_constant;
                else
                    coeff = By * (yolk_vol_curr - yolk_vol_init);
                if (face_vtx_list.size() == 4)
                {
                    Vector<T, 12> dedx;
                    computeConeVolume4PointsGradient(positions, mesh_centroid, dedx);
                    // computeQuadConeVolumeGradient(positions, mesh_centroid, dedx);
                    dedx *= -coeff;
                    addForceEntry<12>(residual, face_vtx_list, dedx);
                }
                else if (face_vtx_list.size() == 5)
                {
                    Vector<T, 15> dedx;
                    computeConeVolume5PointsGradient(positions, mesh_centroid, dedx);
                    dedx *= -coeff;
                    addForceEntry<15>(residual, face_vtx_list, dedx);
                }
                else if (face_vtx_list.size() == 6)
                {
                    Vector<T, 18> dedx;
                    computeConeVolume6PointsGradient(positions, mesh_centroid, dedx);
                    dedx *= -coeff;
                    addForceEntry<18>(residual, face_vtx_list, dedx);
                }
                else
                {
                    std::cout << "unknown polygon edge number" << std::endl;
                }
            }
        }
    });
    
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
        Eigen::MatrixXd ipc_vertices_deformed(basal_vtx_start, 3);
        for (int i = 0; i < basal_vtx_start; i++)
        {
            ipc_vertices_deformed.row(i) = deformed.segment<3>(i * 3);
        }

        ipc::Constraints ipc_constraints;
        ipc::construct_constraint_set(ipc_vertices, ipc_vertices_deformed, 
            ipc_edges, ipc_faces, barrier_distance, ipc_constraints);

        VectorXT contact_gradient;

        try
        {
            contact_gradient = barrier_weight * ipc::compute_barrier_potential_gradient(ipc_vertices_deformed, 
            ipc_edges, ipc_faces, ipc_constraints, barrier_distance);
        }
        catch(const std::runtime_error& e)
        {
            std::cerr << e.what() << '\n';
        }
    
        residual.segment(0, basal_vtx_start * 3) += -contact_gradient;

    
        if(print_force_norm)
            std::cout << "\tcontact force norm: " << (residual - residual_temp).norm() << std::endl;
        residual_temp = residual;
    }

    
    // ===================================== Previtelline Volume =====================================
    if (add_pervitelline_liquid_volume)
    {
        T previtelline_vol_curr = total_volume - computeTotalVolumeFromApicalSurface();

        iterateFaceSerial([&](VtxList& face_vtx_list, int face_idx)
        {
            VectorXT positions;
            positionsFromIndices(positions, face_vtx_list);
            
            if (face_idx < basal_face_start)
            {
                T coeff = -Bp * (previtelline_vol_curr - previtelline_vol_init);
                // negative is correct 
                if (face_vtx_list.size() == 4)
                {
                    Vector<T, 12> dedx;
                    computeConeVolume4PointsGradient(positions, mesh_centroid, dedx);
                    dedx *= coeff;
                    addForceEntry<12>(residual, face_vtx_list, dedx);
                }
                else if (face_vtx_list.size() == 5)
                {
                    Vector<T, 15> dedx;
                    computeConeVolume5PointsGradient(positions, mesh_centroid, dedx);
                    dedx *= coeff;
                    addForceEntry<15>(residual, face_vtx_list, dedx);
                }
                else if (face_vtx_list.size() == 6)
                {
                    Vector<T, 18> dedx;
                    computeConeVolume6PointsGradient(positions, mesh_centroid, dedx);
                    dedx *= coeff;
                    addForceEntry<18>(residual, face_vtx_list, dedx);
                }
                else
                {
                    std::cout << "unknown polygon edge number" << std::endl;
                }
            }
            
        });
    }

    if (!run_diff_test)
        iterateDirichletDoF([&](int offset, T target)
        {
            residual[offset] = 0;
        });

    return residual.norm();
}

void VertexModel::positionsFromIndices(VectorXT& positions, const VtxList& indices)
{
    positions = VectorXT::Zero(indices.size() * 3);
    for (int i = 0; i < indices.size(); i++)
    {
        positions.segment<3>(i * 3) = deformed.segment<3>(indices[i] * 3);
    }
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
        iterateContractingEdgeSerial([&](Edge& e){
            TV vi = deformed.segment<3>(e[0] * 3);
            TV vj = deformed.segment<3>(e[1] * 3);
            Matrix<T, 6, 6> hessian;
            computeEdgeSquaredNormHessian(vi, vj, hessian);
            hessian *= Gamma;
            addHessianEntry<6>(entries, {e[0], e[1]}, hessian);
        });
    }

    VectorXT current_cell_volume;
    computeVolumeAllCells(current_cell_volume);

    T yolk_vol_curr = 0.0;
    if (add_yolk_volume)
    {
        yolk_vol_curr = computeYolkVolume();
    }

    VectorXT v0 = VectorXT::Zero(deformed.rows());
    if (!use_yolk_pressure)
    {
        VectorXT dVdx_full = VectorXT::Zero(deformed.rows());

        iterateFaceSerial([&](VtxList& face_vtx_list, int face_idx)
        {
            if (add_yolk_volume)
            {
                if (face_idx < lateral_face_start && face_idx >= basal_face_start)
                {
                    
                    VectorXT positions;
                    positionsFromIndices(positions, face_vtx_list);
                    if (face_vtx_list.size() == 4)
                    {
                        Vector<T, 12> dedx;
                        computeConeVolume4PointsGradient(positions, mesh_centroid, dedx);
                        // computeQuadConeVolumeGradient(positions, mesh_centroid, dedx);
                        addForceEntry<12>(dVdx_full, face_vtx_list, dedx);
                    }
                    else if (face_vtx_list.size() == 5)
                    {
                        Vector<T, 15> dedx;
                        computeConeVolume5PointsGradient(positions, mesh_centroid, dedx);
                        addForceEntry<15>(dVdx_full, face_vtx_list, dedx);
                    }
                    else if (face_vtx_list.size() == 6)
                    {
                        Vector<T, 18> dedx;
                        computeConeVolume6PointsGradient(positions, mesh_centroid, dedx);
                        addForceEntry<18>(dVdx_full, face_vtx_list, dedx);
                    }
                    else
                    {
                        std::cout << "unknown polygon edge number" << std::endl;
                    }
                }
            }
        });

        v0 += dVdx_full * std::sqrt(By);
        if (!run_diff_test)
        {
            iterateDirichletDoF([&](int offset, T target)
            {
                v0[offset] = 0.0;
            });
        }
    }

    VectorXT v1 = VectorXT::Zero(deformed.rows());
    if (add_pervitelline_liquid_volume)
    {
        VectorXT dVdx_full = VectorXT::Zero(deformed.rows());

        iterateFaceSerial([&](VtxList& face_vtx_list, int face_idx)
        {
            // negative sign is correct here
            if (face_idx < basal_face_start)
            {
                VectorXT positions;
                positionsFromIndices(positions, face_vtx_list);
                if (face_vtx_list.size() == 4)
                {
                    Vector<T, 12> dedx;
                    computeConeVolume4PointsGradient(positions, mesh_centroid, dedx);
                    addForceEntry<12>(dVdx_full, face_vtx_list, -dedx);
                }
                else if (face_vtx_list.size() == 5)
                {
                    Vector<T, 15> dedx;
                    computeConeVolume5PointsGradient(positions, mesh_centroid, dedx);
                    addForceEntry<15>(dVdx_full, face_vtx_list, -dedx);
                }
                else if (face_vtx_list.size() == 6)
                {
                    Vector<T, 18> dedx;
                    computeConeVolume6PointsGradient(positions, mesh_centroid, dedx);
                    addForceEntry<18>(dVdx_full, face_vtx_list, -dedx);
                }
                else
                {
                    std::cout << "unknown polygon edge number" << std::endl;
                }
            }
        });

        v1 += dVdx_full * std::sqrt(Bp);
        if (!run_diff_test)
        {
            iterateDirichletDoF([&](int offset, T target)
            {
                v1[offset] = 0.0;
            });
        }
    }

    UV.resize(num_nodes * 3, 2);
    UV.col(0) = v0; UV.col(1) = v1;
    
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
            T V = current_cell_volume[face_idx];

            if (face_vtx_list.size() == 4)
            {
                
                Matrix<T, 24, 24> d2Vdx2;
                if (use_cell_centroid)
                    computeVolume4PointsHessian(positions, d2Vdx2);
                else
                    computeQuadBasePrismVolumeHessian(positions, d2Vdx2);

                Vector<T, 24> dVdx;
                if (use_cell_centroid)
                    computeVolume4PointsGradient(positions, dVdx);
                else
                    computeQuadBasePrismVolumeGradient(positions, dVdx);
                    
                // break it down here to avoid super long autodiff code
                Matrix<T, 24, 24> hessian;
                if (use_alm_on_cell_volume)
                {
                    hessian = -lambda_cell_vol[face_idx] * d2Vdx2 + 
                            kappa * (dVdx * dVdx.transpose() + 
                            (V - cell_volume_init[face_idx]) * d2Vdx2);
                }
                else
                {    
                    hessian = B * (dVdx * dVdx.transpose() + 
                        (V - cell_volume_init[face_idx]) * d2Vdx2);
                }
            
                addHessianEntry<24>(entries, cell_vtx_list, hessian);
            }
            else if (face_vtx_list.size() == 5)
            {
                Matrix<T, 30, 30> d2Vdx2;
                if (use_cell_centroid)
                    computeVolume5PointsHessian(positions, d2Vdx2);
                else 
                    computePentaBasePrismVolumeHessian(positions, d2Vdx2);

                Vector<T, 30> dVdx;
                if (use_cell_centroid)
                    computeVolume5PointsGradient(positions, dVdx);
                else
                    computePentaBasePrismVolumeGradient(positions, dVdx);
                
                // break it down here to avoid super long autodiff code
                
                Matrix<T, 30, 30> hessian;
                if (use_alm_on_cell_volume)
                {
                    hessian = -lambda_cell_vol[face_idx] * d2Vdx2 + 
                            kappa * (dVdx * dVdx.transpose() + 
                            (V - cell_volume_init[face_idx]) * d2Vdx2);
                }
                else
                {    
                    hessian = B * (dVdx * dVdx.transpose() + 
                        (V - cell_volume_init[face_idx]) * d2Vdx2);
                }
                
                addHessianEntry<30>(entries, cell_vtx_list, hessian);
            }
            else if (face_vtx_list.size() == 6)
            {
                Matrix<T, 36, 36> d2Vdx2;
                if (use_cell_centroid)
                {
                    if (use_fixed_cell_centroid)
                        computeVolume6PointsFixedCentroidHessian(positions, fixed_cell_centroids.segment<3>(face_idx * 3), d2Vdx2);
                    else
                        computeVolume6PointsHessian(positions, d2Vdx2);
                }
                else
                    computeHexBasePrismVolumeHessian(positions, d2Vdx2);

                Vector<T, 36> dVdx;
                if (use_cell_centroid)
                {
                    if (use_fixed_cell_centroid)
                        computeVolume6PointsFixedCentroidGradient(positions, fixed_cell_centroids.segment<3>(face_idx * 3), dVdx);
                    else
                        computeVolume6PointsGradient(positions, dVdx);
                }
                else
                    computeHexBasePrismVolumeGradient(positions, dVdx);
                
                // break it down here to avoid super long autodiff code
                Matrix<T, 36, 36> hessian;
                
                if (use_alm_on_cell_volume)
                {
                    hessian = -lambda_cell_vol[face_idx] * d2Vdx2 + 
                            kappa * (dVdx * dVdx.transpose() + 
                            (V - cell_volume_init[face_idx]) * d2Vdx2);
                }
                else
                {    
                    hessian = B * (dVdx * dVdx.transpose() + 
                        (V - cell_volume_init[face_idx]) * d2Vdx2);
                }

                addHessianEntry<36>(entries, cell_vtx_list, hessian);

                if(add_single_tet_vol_barrier)
                {
                    if (use_cell_centroid)
                    {
                        computeVolumeBarrier6PointsHessian(tet_barrier_stiffness, positions, hessian);
                    }
                    addHessianEntry<36>(entries, cell_vtx_list, hessian);
                }
            }
            else
            {
                std::cout << "unknown polygon edge case" << std::endl;
            }
            // std::cout << "Cell " << face_idx << std::endl;
        }
        // ===================================== Face Area =====================================
        else // basal and lateral faces area term
        {
            T coeff = face_idx >= lateral_face_start ? alpha : gamma;
            VectorXT positions;
            positionsFromIndices(positions, face_vtx_list);
            if (face_vtx_list.size() == 4)
            {
                Matrix<T, 12, 12> hessian;
                if (use_face_centroid)
                    computeArea4PointsSquaredSumHessian(coeff, positions, hessian);
                else
                    computeQuadFaceAreaSquaredSumHessian(coeff, positions, hessian);
                addHessianEntry<12>(entries, face_vtx_list, hessian);
            }
            else if (face_vtx_list.size() == 5)
            {
                Matrix<T, 15, 15> hessian;
                // computeArea5PointsHessian(coeff, positions, hessian);
                // hessian *= coeff;
                if (use_face_centroid)
                    computeArea5PointsSquaredHessian(coeff, positions, hessian);
                else
                    computePentFaceAreaSquaredSumHessian(coeff, positions, hessian);
                addHessianEntry<15>(entries, face_vtx_list, hessian);
            }
            else if (face_vtx_list.size() == 6)
            {
                Matrix<T, 18, 18> hessian;
                // computeArea6PointsHessian(coeff, positions, hessian);
                // hessian *= coeff;
                if (use_face_centroid)
                    computeArea6PointsSquaredHessian(coeff, positions, hessian);
                else
                    computeHexFaceAreaSquaredSumHessian(coeff, positions, hessian);
                addHessianEntry<18>(entries, face_vtx_list, hessian);
            }
            else
            {
                std::cout << "unknown " << std::endl;
            }
        }
        // ===================================== Yolk Volume =====================================
        if (add_yolk_volume)
        {
            if (face_idx < lateral_face_start && face_idx >= basal_face_start)
            {
                
                VectorXT positions;
                positionsFromIndices(positions, face_vtx_list);
                
                if (face_vtx_list.size() == 4)
                {
                    
                    Matrix<T, 12, 12> d2Vdx2;
                    computeConeVolume4PointsHessian(positions, mesh_centroid, d2Vdx2);
                    // computeQuadConeVolumeHessian(positions, mesh_centroid, d2Vdx2);
                    Matrix<T, 12, 12> hessian;
                    if (use_yolk_pressure)
                        hessian = -pressure_constant * d2Vdx2;
                    else
                        hessian = By * (yolk_vol_curr - yolk_vol_init) * d2Vdx2;
                    
                    addHessianEntry<12>(entries, face_vtx_list, hessian);
                }
                else if (face_vtx_list.size() == 5)
                {
                    Matrix<T, 15, 15> d2Vdx2;
                    computeConeVolume5PointsHessian(positions, mesh_centroid, d2Vdx2);
                    Matrix<T, 15, 15> hessian;
                    if (use_yolk_pressure)
                        hessian = -pressure_constant * d2Vdx2;
                    else
                        hessian = By * (yolk_vol_curr - yolk_vol_init) * d2Vdx2;
                    
                    addHessianEntry<15>(entries, face_vtx_list, hessian);

                }
                else if (face_vtx_list.size() == 6)
                {
                    Matrix<T, 18, 18> d2Vdx2;
                    computeConeVolume6PointsHessian(positions, mesh_centroid, d2Vdx2);
                    Matrix<T, 18, 18> hessian;
                    if (use_yolk_pressure)
                        hessian = -pressure_constant * d2Vdx2;
                    else
                        hessian = By * (yolk_vol_curr - yolk_vol_init) * d2Vdx2;
                    
                    addHessianEntry<18>(entries, face_vtx_list, hessian);
                }
                else
                {
                    std::cout << "unknown polygon edge case" << std::endl;
                }
            }
        }

        if (add_pervitelline_liquid_volume)
        {
            T previtelline_vol_curr = total_volume - computeTotalVolumeFromApicalSurface();

            if (face_idx < basal_face_start)
            {
                VectorXT positions;
                positionsFromIndices(positions, face_vtx_list);
                T ci = previtelline_vol_curr - previtelline_vol_init;

                if (face_vtx_list.size() == 4)
                {
                    
                    Matrix<T, 12, 12> d2Vdx2;
                    computeConeVolume4PointsHessian(positions, mesh_centroid, d2Vdx2);
                    Matrix<T, 12, 12> hessian = Bp * ci * d2Vdx2;
                    
                    addHessianEntry<12>(entries, face_vtx_list, hessian);
                }
                else if (face_vtx_list.size() == 5)
                {
                    Matrix<T, 15, 15> d2Vdx2;
                    computeConeVolume5PointsHessian(positions, mesh_centroid, d2Vdx2);
                    Matrix<T, 15, 15> hessian = Bp * ci * d2Vdx2;
                    
                    addHessianEntry<15>(entries, face_vtx_list, hessian);

                }
                else if (face_vtx_list.size() == 6)
                {
                    Matrix<T, 18, 18> d2Vdx2;
                    computeConeVolume6PointsHessian(positions, mesh_centroid, d2Vdx2);
                    Matrix<T, 18, 18> hessian = Bp * ci * d2Vdx2;
                    
                    addHessianEntry<18>(entries, face_vtx_list, hessian);
                }
                else
                {
                    std::cout << "unknown polygon edge case" << std::endl;
                }
            }
        }

        
    });
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
        Eigen::MatrixXd ipc_vertices_deformed(basal_vtx_start, 3);
        for (int i = 0; i < basal_vtx_start; i++)
        {
            ipc_vertices_deformed.row(i) = deformed.segment<3>(i * 3);
        }
        
        ipc::Constraints ipc_constraints;
        ipc::construct_constraint_set(ipc_vertices, ipc_vertices_deformed, 
            ipc_edges, ipc_faces, barrier_distance, ipc_constraints);

        StiffnessMatrix contact_hessian = barrier_weight *  ipc::compute_barrier_potential_hessian(ipc_vertices_deformed, 
            ipc_edges, ipc_faces, ipc_constraints, barrier_distance, false);

        std::vector<Entry> contact_entries = entriesFromSparseMatrix(contact_hessian);
        entries.insert(entries.end(), contact_entries.begin(), contact_entries.end());

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
        iterateContractingEdgeSerial([&](Edge& e){
            TV vi = deformed.segment<3>(e[0] * 3);
            TV vj = deformed.segment<3>(e[1] * 3);
            Matrix<T, 6, 6> hessian;
            computeEdgeSquaredNormHessian(vi, vj, hessian);
            hessian *= Gamma;
            addHessianEntry<6>(entries, {e[0], e[1]}, hessian);
        });
    }

    VectorXT current_cell_volume;
    computeVolumeAllCells(current_cell_volume);

    T yolk_vol_curr = 0.0;
    if (add_yolk_volume)
    {
        yolk_vol_curr = computeYolkVolume();
    }

    if (!use_yolk_pressure)
    {
        VectorXT dVdx_full = VectorXT::Zero(deformed.rows());

        iterateFaceSerial([&](VtxList& face_vtx_list, int face_idx)
        {
            if (add_yolk_volume)
            {
                if (face_idx < lateral_face_start && face_idx >= basal_face_start)
                {
                    
                    VectorXT positions;
                    positionsFromIndices(positions, face_vtx_list);
                    if (face_vtx_list.size() == 4)
                    {
                        Vector<T, 12> dedx;
                        computeConeVolume4PointsGradient(positions, mesh_centroid, dedx);
                        addForceEntry<12>(dVdx_full, face_vtx_list, dedx);
                    }
                    else if (face_vtx_list.size() == 5)
                    {
                        Vector<T, 15> dedx;
                        computeConeVolume5PointsGradient(positions, mesh_centroid, dedx);
                        addForceEntry<15>(dVdx_full, face_vtx_list, dedx);
                    }
                    else if (face_vtx_list.size() == 6)
                    {
                        Vector<T, 18> dedx;
                        computeConeVolume6PointsGradient(positions, mesh_centroid, dedx);
                        addForceEntry<18>(dVdx_full, face_vtx_list, dedx);
                    }
                    else
                    {
                        std::cout << "unknown polygon edge number" << std::endl;
                    }
                }
            }
        });

        for (int dof_i = 0; dof_i < num_nodes; dof_i++)
        {
            for (int dof_j = 0; dof_j < num_nodes; dof_j++)
            {
                Vector<T, 6> dVdx;
                getSubVector<6>(dVdx_full, {dof_i, dof_j}, dVdx);
                TV dVdxi = dVdx.segment<3>(0);
                TV dVdxj = dVdx.segment<3>(3);
                Matrix<T, 3, 3> hessian_partial = By * dVdxi * dVdxj.transpose();
                addHessianBlock<3>(entries, {dof_i, dof_j}, hessian_partial);
            }
        }
    }

    if (add_pervitelline_liquid_volume)
    {
        VectorXT dVdx_full = VectorXT::Zero(deformed.rows());

        iterateFaceSerial([&](VtxList& face_vtx_list, int face_idx)
        {
            if (face_idx < basal_face_start)
            {
                
                VectorXT positions;
                positionsFromIndices(positions, face_vtx_list);
                if (face_vtx_list.size() == 4)
                {
                    Vector<T, 12> dedx;
                    computeConeVolume4PointsGradient(positions, mesh_centroid, dedx);
                    addForceEntry<12>(dVdx_full, face_vtx_list, -dedx);
                }
                else if (face_vtx_list.size() == 5)
                {
                    Vector<T, 15> dedx;
                    computeConeVolume5PointsGradient(positions, mesh_centroid, dedx);
                    addForceEntry<15>(dVdx_full, face_vtx_list, -dedx);
                }
                else if (face_vtx_list.size() == 6)
                {
                    Vector<T, 18> dedx;
                    computeConeVolume6PointsGradient(positions, mesh_centroid, dedx);
                    addForceEntry<18>(dVdx_full, face_vtx_list, -dedx);
                }
                else
                {
                    std::cout << "unknown polygon edge number" << std::endl;
                }
            }
        });

        for (int dof_i = 0; dof_i < num_nodes; dof_i++)
        {
            for (int dof_j = 0; dof_j < num_nodes; dof_j++)
            {
                Vector<T, 6> dVdx;
                getSubVector<6>(dVdx_full, {dof_i, dof_j}, dVdx);
                TV dVdxi = dVdx.segment<3>(0);
                TV dVdxj = dVdx.segment<3>(3);
                Matrix<T, 3, 3> hessian_partial = Bp * dVdxi * dVdxj.transpose();
                addHessianBlock<3>(entries, {dof_i, dof_j}, hessian_partial);
            }
        }
    }
    

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
            T V = current_cell_volume[face_idx];

            T coeff = use_alm_on_cell_volume ? kappa : B;
            if (face_vtx_list.size() == 4)
            {
                
                Matrix<T, 24, 24> d2Vdx2;
                if (use_cell_centroid)
                    computeVolume4PointsHessian(positions, d2Vdx2);
                else
                    computeQuadBasePrismVolumeHessian(positions, d2Vdx2);

                Vector<T, 24> dVdx;
                if (use_cell_centroid)
                    computeVolume4PointsGradient(positions, dVdx);
                else
                    computeQuadBasePrismVolumeGradient(positions, dVdx);
                    
                // break it down here to avoid super long autodiff code
                Matrix<T, 24, 24> hessian;

                if (use_alm_on_cell_volume)
                {
                    hessian = -lambda_cell_vol[face_idx] * d2Vdx2 + 
                            kappa * (dVdx * dVdx.transpose() + 
                            (V - cell_volume_init[face_idx]) * d2Vdx2);
                }
                else
                {    
                    hessian = B * (dVdx * dVdx.transpose() + 
                        (V - cell_volume_init[face_idx]) * d2Vdx2);
                }

                addHessianEntry<24>(entries, cell_vtx_list, hessian);
            }
            else if (face_vtx_list.size() == 5)
            {
                Matrix<T, 30, 30> d2Vdx2;
                if (use_cell_centroid)
                    computeVolume5PointsHessian(positions, d2Vdx2);
                else 
                    computePentaBasePrismVolumeHessian(positions, d2Vdx2);

                Vector<T, 30> dVdx;
                if (use_cell_centroid)
                    computeVolume5PointsGradient(positions, dVdx);
                else
                    computePentaBasePrismVolumeGradient(positions, dVdx);
                
                // break it down here to avoid super long autodiff code
                Matrix<T, 30, 30> hessian;

                if (use_alm_on_cell_volume)
                {
                    hessian = -lambda_cell_vol[face_idx] * d2Vdx2 + 
                            kappa * (dVdx * dVdx.transpose() + 
                            (V - cell_volume_init[face_idx]) * d2Vdx2);
                }
                else
                {    
                    hessian = B * (dVdx * dVdx.transpose() + 
                        (V - cell_volume_init[face_idx]) * d2Vdx2);
                }

                addHessianEntry<30>(entries, cell_vtx_list, hessian);
            }
            else if (face_vtx_list.size() == 6)
            {
                Matrix<T, 36, 36> d2Vdx2;
                if (use_cell_centroid)
                    computeVolume6PointsHessian(positions, d2Vdx2);
                else
                    computeHexBasePrismVolumeHessian(positions, d2Vdx2);

                Vector<T, 36> dVdx;
                if (use_cell_centroid)
                    computeVolume6PointsGradient(positions, dVdx);
                else
                    computeHexBasePrismVolumeGradient(positions, dVdx);
                
                // break it down here to avoid super long autodiff code
                Matrix<T, 36, 36> hessian;

                if (use_alm_on_cell_volume)
                {
                    hessian = -lambda_cell_vol[face_idx] * d2Vdx2 + 
                            kappa * (dVdx * dVdx.transpose() + 
                            (V - cell_volume_init[face_idx]) * d2Vdx2);
                }
                else
                {    
                    hessian = B * (dVdx * dVdx.transpose() + 
                        (V - cell_volume_init[face_idx]) * d2Vdx2);
                }

                addHessianEntry<36>(entries, cell_vtx_list, hessian);

                if (add_single_tet_vol_barrier)
                {
                    if (use_cell_centroid)
                        computeVolumeBarrier6PointsHessian(tet_barrier_stiffness, positions, hessian);
                    else
                        computeHexBasePrismVolumeBarrierHessian(tet_barrier_stiffness, positions, hessian);
                    addHessianEntry<36>(entries, cell_vtx_list, hessian);
                }
            }
            else
            {
                std::cout << "unknown polygon edge case" << std::endl;
            }
            // std::cout << "Cell " << face_idx << std::endl;
        }
        else // basal and lateral faces area term
        {
            T coeff = face_idx >= lateral_face_start ? alpha : gamma;
            VectorXT positions;
            positionsFromIndices(positions, face_vtx_list);
            if (face_vtx_list.size() == 4)
            {
                Matrix<T, 12, 12> hessian;
                // computeArea4PointsHessian(coeff, positions, hessian);
                // hessian *= coeff;
                // computeQuadFaceAreaHessian(coeff, positions, hessian);
                // computeArea4PointsSquaredHessian(coeff, positions, hessian);
                if (use_face_centroid)
                    computeArea4PointsSquaredSumHessian(coeff, positions, hessian);
                else
                    computeQuadFaceAreaSquaredSumHessian(coeff, positions, hessian);
                addHessianEntry<12>(entries, face_vtx_list, hessian);
            }
            else if (face_vtx_list.size() == 5)
            {
                Matrix<T, 15, 15> hessian;
                // computeArea5PointsHessian(coeff, positions, hessian);
                // hessian *= coeff;
                if (use_face_centroid)
                    computeArea5PointsSquaredHessian(coeff, positions, hessian);
                else
                    computePentFaceAreaSquaredSumHessian(coeff, positions, hessian);
                addHessianEntry<15>(entries, face_vtx_list, hessian);
            }
            else if (face_vtx_list.size() == 6)
            {
                Matrix<T, 18, 18> hessian;
                // computeArea6PointsHessian(coeff, positions, hessian);
                // hessian *= coeff;
                if (use_face_centroid)
                    computeArea6PointsSquaredHessian(coeff, positions, hessian);
                else
                    computeHexFaceAreaSquaredSumHessian(coeff, positions, hessian);
                addHessianEntry<18>(entries, face_vtx_list, hessian);
            }
            else
            {
                std::cout << "unknown " << std::endl;
            }
        }
        if (add_yolk_volume)
        {
            if (face_idx < lateral_face_start && face_idx >= basal_face_start)
            {
                
                VectorXT positions;
                positionsFromIndices(positions, face_vtx_list);
                
                if (face_vtx_list.size() == 4)
                {
                    
                    Matrix<T, 12, 12> d2Vdx2;
                    computeConeVolume4PointsHessian(positions, mesh_centroid, d2Vdx2);
                    // computeQuadConeVolumeHessian(positions, mesh_centroid, d2Vdx2);
                    Matrix<T, 12, 12> hessian;
                    if (use_yolk_pressure)
                        hessian = -pressure_constant * d2Vdx2;
                    else
                        hessian = By * (yolk_vol_curr - yolk_vol_init) * d2Vdx2;
                    
                    addHessianEntry<12>(entries, face_vtx_list, hessian);
                }
                else if (face_vtx_list.size() == 5)
                {
                    Matrix<T, 15, 15> d2Vdx2;
                    computeConeVolume5PointsHessian(positions, mesh_centroid, d2Vdx2);
                    Matrix<T, 15, 15> hessian;
                    if (use_yolk_pressure)
                        hessian = -pressure_constant * d2Vdx2;
                    else
                        hessian = By * (yolk_vol_curr - yolk_vol_init) * d2Vdx2;
                    
                    addHessianEntry<15>(entries, face_vtx_list, hessian);

                }
                else if (face_vtx_list.size() == 6)
                {
                    Matrix<T, 18, 18> d2Vdx2;
                    computeConeVolume6PointsHessian(positions, mesh_centroid, d2Vdx2);
                    Matrix<T, 18, 18> hessian;
                    if (use_yolk_pressure)
                        hessian = -pressure_constant * d2Vdx2;
                    else
                        hessian = By * (yolk_vol_curr - yolk_vol_init) * d2Vdx2;
                    
                    addHessianEntry<18>(entries, face_vtx_list, hessian);
                }
                else
                {
                    std::cout << "unknown polygon edge case" << std::endl;
                }
            }
        }
        
    });

    if (add_pervitelline_liquid_volume)
    {
        T previtelline_vol_curr = total_volume - computeTotalVolumeFromApicalSurface();

        iterateFaceSerial([&](VtxList& face_vtx_list, int face_idx)
        {
            // cell-wise volume preservation term
            if (face_idx < basal_face_start)
            {
                VectorXT positions;
                positionsFromIndices(positions, face_vtx_list);
                T ci = previtelline_vol_curr - previtelline_vol_init;
                if (face_vtx_list.size() == 4)
                {
                    
                    Matrix<T, 12, 12> d2Vdx2;
                    computeConeVolume4PointsHessian(positions, mesh_centroid, d2Vdx2);
                    Matrix<T, 12, 12> hessian = Bp * ci * d2Vdx2;
                    
                    addHessianEntry<12>(entries, face_vtx_list, hessian);
                }
                else if (face_vtx_list.size() == 5)
                {
                    Matrix<T, 15, 15> d2Vdx2;
                    computeConeVolume5PointsHessian(positions, mesh_centroid, d2Vdx2);
                    Matrix<T, 15, 15> hessian = Bp * ci * d2Vdx2;
                    
                    addHessianEntry<15>(entries, face_vtx_list, hessian);

                }
                else if (face_vtx_list.size() == 6)
                {
                    Matrix<T, 18, 18> d2Vdx2;
                    computeConeVolume6PointsHessian(positions, mesh_centroid, d2Vdx2);
                    Matrix<T, 18, 18> hessian = Bp * ci * d2Vdx2;
                    
                    addHessianEntry<18>(entries, face_vtx_list, hessian);
                }
                else
                {
                    std::cout << "unknown polygon edge case" << std::endl;
                }
            }
        });
    }

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
        Eigen::MatrixXd ipc_vertices_deformed(basal_vtx_start, 3);
        for (int i = 0; i < basal_vtx_start; i++)
        {
            ipc_vertices_deformed.row(i) = deformed.segment<3>(i * 3);
        }
        
        ipc::Constraints ipc_constraints;
        ipc::construct_constraint_set(ipc_vertices, ipc_vertices_deformed, 
            ipc_edges, ipc_faces, barrier_distance, ipc_constraints);

        StiffnessMatrix contact_hessian = barrier_weight *  ipc::compute_barrier_potential_hessian(ipc_vertices_deformed, 
            ipc_edges, ipc_faces, ipc_constraints, barrier_distance, false);

        std::vector<Entry> contact_entries = entriesFromSparseMatrix(contact_hessian);
        entries.insert(entries.end(), contact_entries.begin(), contact_entries.end());

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
