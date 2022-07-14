#include "../include/VertexModel.h"
#include "../include/autodiff/TetVolBarrier.h"
#include "../include/autodiff/TetVolBarrierScaled.h"

void VertexModel::addFixedTetLogBarrierEnergy(T& energy)
{
    T barrier_energy = 0.0;
    iterateFixedTetsSerial([&](TetVtx& x_deformed, TetVtx& x_undeformed, VtxList& indices)
    {
        T e = 0.0;
        T d_cur = computeTetVolume(x_deformed.col(0), x_deformed.col(1), x_deformed.col(2), x_deformed.col(3));
        T d_bar = computeTetVolume(x_undeformed.col(0), x_undeformed.col(1), x_undeformed.col(2), x_undeformed.col(3));
        T d = d_cur / d_bar;
        if (d < tet_vol_barrier_dhat)
        {
            computeTetInversionBarrierScaled(tet_vol_barrier_w, tet_vol_barrier_dhat, x_deformed, x_undeformed, e);
            barrier_energy += e;
        }
    });
    energy += barrier_energy;
}

void VertexModel::addFixedTetLogBarrierForceEneries(VectorXT& residual)
{
    iterateFixedTetsSerial([&](TetVtx& x_deformed, TetVtx& x_undeformed, VtxList& indices)
    {
        T d_cur = computeTetVolume(x_deformed.col(0), x_deformed.col(1), x_deformed.col(2), x_deformed.col(3));
        T d_bar = computeTetVolume(x_undeformed.col(0), x_undeformed.col(1), x_undeformed.col(2), x_undeformed.col(3));
        T d = d_cur / d_bar;
        if (d < tet_vol_barrier_dhat)
        {   
            Vector<T, 12> dedx;
            computeTetInversionBarrierScaledGradient(tet_vol_barrier_w, tet_vol_barrier_dhat, x_deformed, x_undeformed, dedx);
            addForceEntry<12>(residual, indices, -dedx);
        }
    });
}

void VertexModel::addFixedTetLogBarrierHessianEneries(std::vector<Entry>& entries, bool projectPD)
{
    iterateFixedTetsSerial([&](TetVtx& x_deformed, TetVtx& x_undeformed, VtxList& indices)
    {
        T d_cur = computeTetVolume(x_deformed.col(0), x_deformed.col(1), x_deformed.col(2), x_deformed.col(3));
        T d_bar = computeTetVolume(x_undeformed.col(0), x_undeformed.col(1), x_undeformed.col(2), x_undeformed.col(3));
        T d = d_cur / d_bar;
        if (d < tet_vol_barrier_dhat)
        {
            Matrix<T, 12, 12> hessian;
            computeTetInversionBarrierScaledHessian(tet_vol_barrier_w, tet_vol_barrier_dhat, x_deformed, x_undeformed, hessian);
            if (projectPD)
                projectBlockPD<12>(hessian);
            addHessianEntry<12>(entries, indices, hessian);
        }
    });
}

void VertexModel::computeCentroidTetVolume(const VectorXT& positions, 
        const VtxList& face_vtx_list, VectorXT& tets_volume,
        std::vector<TetVtx>& tets)
{
    tets_volume = VectorXT::Zero(face_vtx_list.size() * 6);

    TV cell_centroid = TV::Zero();
    TV apical_centroid = TV::Zero();
    TV basal_centroid = TV::Zero();
    
    for (int i = 0; i < face_vtx_list.size(); i++)
    {
        apical_centroid += positions.segment<3>(i * 3);
        basal_centroid += positions.segment<3>((i + face_vtx_list.size()) * 3);
    }		

    cell_centroid = (apical_centroid + basal_centroid) / T(face_vtx_list.size() * 2);

    apical_centroid /= T(face_vtx_list.size());
    basal_centroid /= T(face_vtx_list.size());

    int cnt = 0;
    for (int i = 0; i < face_vtx_list.size(); i++)
    {
        int j = (i + 1) % face_vtx_list.size();
        TV r0 = positions.segment<3>(i * 3);
        TV r1 = positions.segment<3>(j * 3);
        T volume = -computeTetVolume(apical_centroid, r1, r0, cell_centroid);
        tets_volume[cnt++] = volume;

        TV r2 = positions.segment<3>((i + face_vtx_list.size()) * 3);
        TV r3 = positions.segment<3>((j + face_vtx_list.size()) * 3);
        volume = computeTetVolume(basal_centroid, r3, r2, cell_centroid);
        tets_volume[cnt++] = volume;

        TV lateral_centroid = T(0.25) * (r0 + r1 + r2 + r3);
        volume = computeTetVolume(lateral_centroid, r1, r0, cell_centroid);
        tets_volume[cnt++] = volume;

        volume = computeTetVolume(lateral_centroid, r3, r1, cell_centroid);
        tets_volume[cnt++] = volume;

        volume = computeTetVolume(lateral_centroid, r2, r3, cell_centroid);
        tets_volume[cnt++] = volume;

        volume = computeTetVolume(lateral_centroid, r0, r2, cell_centroid);
        tets_volume[cnt++] = volume;
    }
}

void VertexModel::computeTetBarrierWeightMask(const VectorXT& positions, 
    const VtxList& face_vtx_list, VectorXT& mask_log_term, 
    VectorXT& mask_qubic_term, T cell_volume)
{
    T qubic_term_active_vol = qubic_active_percentage * cell_volume / T(face_vtx_list.size() * 6);
    T log_term_active_vol = log_active_percentage * cell_volume / T(face_vtx_list.size() * 6);

    TV cell_centroid = TV::Zero();
    TV apical_centroid = TV::Zero();
    TV basal_centroid = TV::Zero();
    
    for (int i = 0; i < face_vtx_list.size(); i++)
    {
        apical_centroid += positions.segment<3>(i * 3);
        basal_centroid += positions.segment<3>((i + face_vtx_list.size()) * 3);
    }		

    cell_centroid = (apical_centroid + basal_centroid) / T(face_vtx_list.size() * 2);

    apical_centroid /= T(face_vtx_list.size());
    basal_centroid /= T(face_vtx_list.size());
    mask_qubic_term = VectorXT::Zero(face_vtx_list.size() * 6);
    mask_log_term = VectorXT::Zero(face_vtx_list.size() * 6);
    int cnt = 0;
    for (int i = 0; i < face_vtx_list.size(); i++)
    {
        int j = (i + 1) % face_vtx_list.size();
        TV r0 = positions.segment<3>(i * 3);
        TV r1 = positions.segment<3>(j * 3);
        T volume = -computeTetVolume(apical_centroid, r1, r0, cell_centroid);
        if (volume < qubic_term_active_vol) mask_qubic_term(cnt) = 1.0;
        // std::cout << "current volume " << volume << " active value: " << qubic_term_active_vol << std::endl;
        if (volume < log_term_active_vol) mask_log_term(cnt) = 1.0;
        cnt++;

        TV r2 = positions.segment<3>((i + face_vtx_list.size()) * 3);
        TV r3 = positions.segment<3>((j + face_vtx_list.size()) * 3);
        volume = computeTetVolume(basal_centroid, r3, r2, cell_centroid);
        // std::cout << "current volume " << volume << " active value: " << qubic_term_active_vol << std::endl;
        if (volume < qubic_term_active_vol) mask_qubic_term(cnt) = 1.0;
        if (volume < log_term_active_vol) mask_log_term(cnt) = 1.0;
        cnt++;
        TV lateral_centroid = T(0.25) * (r0 + r1 + r2 + r3);
        volume = computeTetVolume(lateral_centroid, r1, r0, cell_centroid);
        if (volume < qubic_term_active_vol) mask_qubic_term(cnt) = 1.0;
        if (volume < log_term_active_vol) mask_log_term(cnt) = 1.0;
        cnt++;
        volume = computeTetVolume(lateral_centroid, r3, r1, cell_centroid);
        if (volume < qubic_term_active_vol) mask_qubic_term(cnt) = 1.0;
        if (volume < log_term_active_vol) mask_log_term(cnt) = 1.0;
        cnt++;
        volume = computeTetVolume(lateral_centroid, r2, r3, cell_centroid);
        if (volume < qubic_term_active_vol) mask_qubic_term(cnt) = 1.0;
        if (volume < log_term_active_vol) mask_log_term(cnt) = 1.0;
        cnt++;
        volume = computeTetVolume(lateral_centroid, r0, r2, cell_centroid);
        if (volume < qubic_term_active_vol) mask_qubic_term(cnt) = 1.0;
        if (volume < log_term_active_vol) mask_log_term(cnt) = 1.0;
        cnt++;
    }
}

T VertexModel::computeInversionFreeStepSize(const VectorXT& _u, const VectorXT& du)
{
    T step_size = 1.0;
    if (!add_tet_vol_barrier)
        return 1.0;
    int cnt = 0;
    while (true)
    {
        if (cnt > 60)
        {
            std::cout << "unable to find inversion free state " << std::endl;
            saveCellMesh(-1);
            // std::exit(0);
            return step_size;
        }
        deformed = undeformed + _u + step_size * du;
        bool constraint_violated = false;
        if (use_cell_centroid)
        {
            // return step_size;
            iterateFaceSerial([&](VtxList& face_vtx_list, int face_idx)
            {
                if (constraint_violated)
                    return;
                // cell-wise volume preservation term
                if (face_idx < basal_face_start)
                {
                    VectorXT positions;
                    VtxList cell_vtx_list = face_vtx_list;
                    for (int idx : face_vtx_list)
                        cell_vtx_list.push_back(idx + basal_vtx_start);

                    positionsFromIndices(positions, cell_vtx_list);
                    
                    T scaling_factor = cell_volume_init[face_idx] / T(face_vtx_list.size() * 6);
                    if (!scaled_barrier)
                        scaling_factor = 1.0;
                    TV cell_centroid = TV::Zero();
                    TV apical_centroid = TV::Zero();
                    TV basal_centroid = TV::Zero();
                    
                    for (int i = 0; i < face_vtx_list.size(); i++)
                    {
                        apical_centroid += positions.segment<3>(i * 3);
                        basal_centroid += positions.segment<3>((i + face_vtx_list.size()) * 3);
                    }		

                    cell_centroid = (apical_centroid + basal_centroid) / T(face_vtx_list.size() * 2);

                    apical_centroid /= T(face_vtx_list.size());
                    basal_centroid /= T(face_vtx_list.size());

                    for (int i = 0; i < face_vtx_list.size(); i++)
                    {
                        int j = (i + 1) % face_vtx_list.size();
                        TV r0 = positions.segment<3>(i * 3);
                        TV r1 = positions.segment<3>(j * 3);
                        if (-computeTetVolume(apical_centroid, r1, r0, cell_centroid) / scaling_factor < TET_VOL_MIN)
                        {
                            constraint_violated = true;
                            break;
                        }

                        TV r2 = positions.segment<3>((i + face_vtx_list.size()) * 3);
                        TV r3 = positions.segment<3>((j + face_vtx_list.size()) * 3);
                        if (computeTetVolume(basal_centroid, r3, r2, cell_centroid) / scaling_factor < TET_VOL_MIN)
                        {
                            constraint_violated = true;
                            break;
                        }

                        TV lateral_centroid = T(0.25) * (r0 + r1 + r2 + r3);
                        if (computeTetVolume(lateral_centroid, r1, r0, cell_centroid) / scaling_factor < TET_VOL_MIN)
                        {
                            constraint_violated = true;
                            break;
                        }
                        if (computeTetVolume(lateral_centroid, r3, r1, cell_centroid) / scaling_factor < TET_VOL_MIN)
                        {
                            constraint_violated = true;
                            break;
                        }
                        if (computeTetVolume(lateral_centroid, r2, r3, cell_centroid) / scaling_factor < TET_VOL_MIN)
                        {
                            constraint_violated = true;
                            break;
                        }
                        if (computeTetVolume(lateral_centroid, r0, r2, cell_centroid) / scaling_factor < TET_VOL_MIN)
                        {
                            constraint_violated = true;
                            break;
                        }
                    }
                    if (constraint_violated)
                        return;
                }
            });
        }
        else
        {
            iterateFixedTetsSerial([&](TetVtx& x_deformed, TetVtx& x_undeformed, VtxList& indices)
            {
                if (constraint_violated)
                    return;
                
                T d_cur = computeTetVolume(x_deformed.col(0), x_deformed.col(1), x_deformed.col(2), x_deformed.col(3));
                T d_bar = computeTetVolume(x_undeformed.col(0), x_undeformed.col(1), x_undeformed.col(2), x_undeformed.col(3));
                T d = d_cur / d_bar;
                if (d < TET_VOL_MIN) constraint_violated = true;
            });
        }
        
        if (constraint_violated)
        {
            step_size *= 0.8;
            cnt++;
        }
        else
            return step_size;
    }
}

void VertexModel::addSingleTetVolBarrierEnergy(T& energy)
{
    VectorXT energies = VectorXT::Zero(basal_face_start);
    iterateFaceParallel([&](VtxList& face_vtx_list, int face_idx)
    {
        if (face_idx >= basal_face_start)
            return;
        if (!add_log_tet_barrier)
        {
            VectorXT positions;
            VtxList cell_vtx_list = face_vtx_list;
            for (int idx : face_vtx_list)
                cell_vtx_list.push_back(idx + basal_vtx_start);
            T volume_barrier_energy = 0.0;
            positionsFromIndices(positions, cell_vtx_list);
            T scaling_factor = cell_volume_init[face_idx] / T(face_vtx_list.size() * 6);
            if (scaled_barrier)
            {
                if(face_vtx_list.size() == 4)
                    computeVolumeBarrier4PointsScaled(tet_vol_barrier_w, scaling_factor, positions, energies[face_idx]);
                else if(face_vtx_list.size() == 5)
                    computeVolumeBarrier5PointsScaled(tet_vol_barrier_w, scaling_factor, positions, energies[face_idx]);
                else if(face_vtx_list.size() == 6)
                    computeVolumeBarrier6PointsScaled(tet_vol_barrier_w, scaling_factor, positions, energies[face_idx]);
                else if(face_vtx_list.size() == 7)
                    computeVolumeBarrier7PointsScaled(tet_vol_barrier_w, scaling_factor, positions, energies[face_idx]);
                else if(face_vtx_list.size() == 8)
                    computeVolumeBarrier8PointsScaled(tet_vol_barrier_w, scaling_factor, positions, energies[face_idx]);
                else if(face_vtx_list.size() == 9)
                    computeVolumeBarrier9PointsScaled(tet_vol_barrier_w, scaling_factor, positions, energies[face_idx]);
            }
            else
            {
                if(face_vtx_list.size() == 4)
                    computeVolumeBarrier4Points(tet_vol_barrier_w, positions, energies[face_idx]);
                else if(face_vtx_list.size() == 5)
                    computeVolumeBarrier5Points(tet_vol_barrier_w, positions, energies[face_idx]);
                else if(face_vtx_list.size() == 6)
                    computeVolumeBarrier6Points(tet_vol_barrier_w, positions, energies[face_idx]);
                else if(face_vtx_list.size() == 7)
                    computeVolumeBarrier7Points(tet_vol_barrier_w, positions, energies[face_idx]);
                else if(face_vtx_list.size() == 8)
                    computeVolumeBarrier8Points(tet_vol_barrier_w, positions, energies[face_idx]);
                else if(face_vtx_list.size() == 9)
                    computeVolumeBarrier9Points(tet_vol_barrier_w, positions, energies[face_idx]);
            }
        }
    });
    energy += energies.sum();
    // iterateFaceSerial([&](VtxList& face_vtx_list, int face_idx)
    // {
    //     if (face_idx < basal_face_start)
    //     {
    //         VectorXT positions;
    //         VtxList cell_vtx_list = face_vtx_list;
    //         for (int idx : face_vtx_list)
    //             cell_vtx_list.push_back(idx + basal_vtx_start);
    //         T volume_barrier_energy = 0.0;
    //         positionsFromIndices(positions, cell_vtx_list);
    //         VectorXT qubic_mask, log_mask;
    //         computeTetBarrierWeightMask(positions, face_vtx_list, log_mask, qubic_mask, cell_volume_init[face_idx]);

    //         T log_term_active_vol = log_active_percentage * cell_volume_init[face_idx] / T(face_vtx_list.size() * 6);

    //         if(face_vtx_list.size() == 4)
    //         {_log_tet_barrier)
    //                 return;
    //             if (add
    //             else
    //                 computeVolumeBarrier4Points(tet_vol_barrier_w, positions, volume_barrier_energy);
    //             energy += volume_barrier_energy;
    //         }
    //         else if(face_vtx_list.size() == 5)
    //         {
    //             if (add_log_tet_barrier)
    //                 computeVolLogBarrier5Points(tet_vol_barrier_w, log_term_active_vol, positions, log_mask, volume_barrier_energy);
    //             else
    //                 computeVolumeBarrier5Points(tet_vol_barrier_w, positions, volume_barrier_energy);
    //             energy += volume_barrier_energy;
    //             if (add_qubic_unilateral_term)
    //             {
    //                 T qubic_term = 0.0;
    //                 T target = qubic_active_percentage * cell_volume_init[face_idx];
    //                 computeVolQubicUnilateralPenalty5Points(tet_vol_qubic_w, target, positions, qubic_mask, qubic_term);
    //                 energy += qubic_term;
    //             }
    //         }
    //         else if(face_vtx_list.size() == 6)
    //         {
    //             if (add_log_tet_barrier)
    //                 computeVolLogBarrier6Points(tet_vol_barrier_w, log_term_active_vol, positions, log_mask, volume_barrier_energy);
    //             else
    //                 computeVolumeBarrier6Points(tet_vol_barrier_w, positions, volume_barrier_energy);
    //             energy += volume_barrier_energy;
    //             if (add_qubic_unilateral_term)
    //             {
    //                 T qubic_term = 0.0;
    //                 T target = qubic_active_percentage * cell_volume_init[face_idx];
    //                 computeVolQubicUnilateralPenalty6Points(tet_vol_qubic_w, target, positions, qubic_mask, qubic_term);
    //                 energy += qubic_term;
    //             }
    //         }
    //         else if(face_vtx_list.size() == 7)
    //         {
    //             if (add_log_tet_barrier)
    //                 return;
    //             else
    //                 computeVolumeBarrier7Points(tet_vol_barrier_w, positions, volume_barrier_energy);
    //             energy += volume_barrier_energy;
    //         }
    //         else if(face_vtx_list.size() == 8)
    //         {
    //             if (add_log_tet_barrier)
    //                 return;
    //             else
    //                 computeVolumeBarrier8Points(tet_vol_barrier_w, positions, volume_barrier_energy);
    //             energy += volume_barrier_energy;
    //         }
    //         else if(face_vtx_list.size() == 9)
    //         {
    //             if (add_log_tet_barrier)
    //                 return;
    //             else
    //                 computeVolumeBarrier9Points(tet_vol_barrier_w, positions, volume_barrier_energy);
    //             energy += volume_barrier_energy;
    //         }
    //     }
    // });
}

void VertexModel::addSingleTetVolBarrierForceEntries(VectorXT& residual)
{
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
            
            
            VectorXT qubic_mask, log_mask;
            computeTetBarrierWeightMask(positions, face_vtx_list, log_mask, qubic_mask, cell_volume_init[face_idx]);

            T log_term_active_vol = log_active_percentage * cell_volume_init[face_idx] / T(face_vtx_list.size() * 6);

            // if (qubic_mask.sum() > 1e-8)
            //     std::cout << "cubic term kicks in" << std::endl;
            T scaling_factor = cell_volume_init[face_idx] / T(face_vtx_list.size() * 6);
            if (face_vtx_list.size() == 4)
            {
                Vector<T, 24> dedx;
                if (scaled_barrier)
                    computeVolumeBarrier4PointsScaledGradient(tet_vol_barrier_w, scaling_factor, positions, dedx);
                else
                    computeVolumeBarrier4PointsGradient(tet_vol_barrier_w, positions, dedx);
                addForceEntry<24>(residual, cell_vtx_list, -dedx);
                
            }
            else if (face_vtx_list.size() == 5)
            {
                Vector<T, 30> dedx;
                if (scaled_barrier)
                    computeVolumeBarrier5PointsScaledGradient(tet_vol_barrier_w, scaling_factor, positions, dedx);
                else
                    computeVolumeBarrier5PointsGradient(tet_vol_barrier_w, positions, dedx);
                addForceEntry<30>(residual, cell_vtx_list, -dedx);
                if (add_qubic_unilateral_term)
                {
                    T target = qubic_active_percentage * cell_volume_init[face_idx];
                    computeVolQubicUnilateralPenalty5PointsGradient(tet_vol_qubic_w, target, positions, qubic_mask, dedx);
                    if (qubic_mask.sum() > 1e-8)
                    {
                        std::cout << "qubic term" << std::endl;
                        std::cout <<dedx.norm() << " weights " << qubic_mask.transpose() << std::endl;
                    }
                    addForceEntry<30>(residual, cell_vtx_list, -dedx);
                }
                // if (dedx.norm() > 1)
                // {
                //     std::cout << "force " << dedx.norm() << std::endl;
                //     VectorXT cell_vol_curr;
                //     computeVolumeAllCells(cell_vol_curr);
                //     VectorXT tet_volumes;
                //     std::vector<TetVtx> tets;
                //     computeCentroidTetVolume(positions, face_vtx_list, tet_volumes, tets);
                //     std::cout << "tets volume " << tet_volumes.transpose() << std::endl;
                //     std::cout << "cell volume " << cell_vol_curr[face_idx] << std::endl;
                //     std::cout << "cell volume init " << cell_volume_init[face_idx] << std::endl;
                //     std::cout << tet_vol_barrier_w / std::pow(cell_vol_curr[face_idx], 4) << std::endl;
                //     T ei_ = 0.0;
                //     for (int i = 0; i < tet_volumes.rows(); i++)
                //     {
                //         ei_ += tet_vol_barrier_w / std::pow(tet_volumes[i], 4);
                //     }
                //     T ei = 0.0;
                //     computeVolumeBarrier5Points(tet_vol_barrier_w, positions, ei);
                //     std::cout << "barrier energy: " << ei << " " << ei_ << std::endl;
                //     saveSingleCellEdges("troubled_cell.obj", cell_vtx_list, positions);
                //     std::getchar();
                // }
            }
            else if (face_vtx_list.size() == 6)
            {
                Vector<T, 36> dedx;
                if (scaled_barrier)
                    computeVolumeBarrier6PointsScaledGradient(tet_vol_barrier_w, scaling_factor, positions, dedx);
                else
                    computeVolumeBarrier6PointsGradient(tet_vol_barrier_w, positions, dedx);
                addForceEntry<36>(residual, cell_vtx_list, -dedx);
                if (add_qubic_unilateral_term)
                {
                    T target = qubic_active_percentage * cell_volume_init[face_idx];
                    computeVolQubicUnilateralPenalty6PointsGradient(tet_vol_qubic_w, target, positions, qubic_mask, dedx);
                    addForceEntry<36>(residual, cell_vtx_list, -dedx);
                }
                // if (dedx.norm() > 1)
                // {
                //     std::cout << "force " << dedx.norm() << std::endl;
                //     saveSingleCellEdges("troubled_cell.obj", cell_vtx_list, positions);
                //     std::getchar();
                // }
            }
            else if (face_vtx_list.size() == 7)
            {
                Vector<T, 42> dedx;
                if (scaled_barrier)
                    computeVolumeBarrier7PointsScaledGradient(tet_vol_barrier_w, scaling_factor, positions, dedx);
                else
                    computeVolumeBarrier7PointsGradient(tet_vol_barrier_w, positions, dedx);
                addForceEntry<42>(residual, cell_vtx_list, -dedx);
                // if (dedx.norm() > 1)
                // {
                //     std::cout << "force " << dedx.norm() << std::endl;
                //     saveSingleCellEdges("troubled_cell.obj", cell_vtx_list, positions);
                //     std::getchar();
                // }   
            }
            else if (face_vtx_list.size() == 8)
            {
                Vector<T, 48> dedx;
                if (scaled_barrier)
                    computeVolumeBarrier8PointsScaledGradient(tet_vol_barrier_w, scaling_factor, positions, dedx);
                else
                    computeVolumeBarrier8PointsGradient(tet_vol_barrier_w, positions, dedx);
                addForceEntry<48>(residual, cell_vtx_list, -dedx);
                // if (dedx.norm() > 1)
                // {
                //     std::cout << "force " << dedx.norm() << std::endl;
                //     saveSingleCellEdges("troubled_cell.obj", cell_vtx_list, positions);
                //     std::getchar();
                // }
            }
            else if (face_vtx_list.size() == 9)
            {
                Vector<T, 54> dedx;
                if (scaled_barrier)
                    computeVolumeBarrier9PointsScaledGradient(tet_vol_barrier_w, scaling_factor, positions, dedx);
                else
                    computeVolumeBarrier9PointsGradient(tet_vol_barrier_w, positions, dedx);
                addForceEntry<54>(residual, cell_vtx_list, -dedx);
                // if (dedx.norm() > 1)
                // {
                //     std::cout << "force " << dedx.norm() << std::endl;
                //     saveSingleCellEdges("troubled_cell.obj", cell_vtx_list, positions);
                //     std::getchar();
                // }
            }
        }
    });
}

void VertexModel::addSingleTetVolBarrierHessianEntries(std::vector<Entry>& entries, 
    bool projectPD)
{
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
            VectorXT qubic_mask, log_mask;
            // computeTetBarrierWeightMask(positions, face_vtx_list, log_mask, qubic_mask, cell_volume_init[face_idx]);
            T scaling_factor = cell_volume_init[face_idx] / T(face_vtx_list.size() * 6);

            T log_term_active_vol = log_active_percentage * cell_volume_init[face_idx] / T(face_vtx_list.size() * 6);

            if (face_vtx_list.size() == 4)
            {
                Matrix<T, 24, 24> hessian;
                if (scaled_barrier)
                    computeVolumeBarrier4PointsScaledHessian(tet_vol_barrier_w, scaling_factor, positions, hessian);
                else
                    computeVolumeBarrier4PointsHessian(tet_vol_barrier_w, positions, hessian);
                if (projectPD)
                    projectBlockPD<24>(hessian);
                addHessianEntry<24>(entries, cell_vtx_list, hessian);
            }
            else if (face_vtx_list.size() == 5)
            {
                Matrix<T, 30, 30> hessian;
                if (scaled_barrier)
                    computeVolumeBarrier5PointsScaledHessian(tet_vol_barrier_w, scaling_factor, positions, hessian);
                else
                    computeVolumeBarrier5PointsHessian(tet_vol_barrier_w, positions, hessian);
                if (projectPD)
                    projectBlockPD<30>(hessian);
                addHessianEntry<30>(entries, cell_vtx_list, hessian);
                
                if (add_qubic_unilateral_term)
                {
                    T target = qubic_active_percentage * cell_volume_init[face_idx];
                    computeVolQubicUnilateralPenalty5PointsHessian(tet_vol_qubic_w, target, positions, qubic_mask, hessian);
                    if (projectPD)
                        projectBlockPD<30>(hessian);
                    addHessianEntry<30>(entries, cell_vtx_list, hessian);
                }
            }
            else if (face_vtx_list.size() == 6)
            {
                Matrix<T, 36, 36> hessian;
                if (scaled_barrier)
                    computeVolumeBarrier6PointsScaledHessian(tet_vol_barrier_w, scaling_factor, positions, hessian);
                else
                    computeVolumeBarrier6PointsHessian(tet_vol_barrier_w, positions, hessian);
                if (projectPD)
                    projectBlockPD<36>(hessian);
                addHessianEntry<36>(entries, cell_vtx_list, hessian);
                
                if (add_qubic_unilateral_term)
                {
                    T target = qubic_active_percentage * cell_volume_init[face_idx];
                    computeVolQubicUnilateralPenalty6PointsHessian(tet_vol_qubic_w, target, positions, qubic_mask, hessian);
                    if (projectPD)
                        projectBlockPD<36>(hessian);
                    addHessianEntry<36>(entries, cell_vtx_list, hessian);
                }
            }
            else if (face_vtx_list.size() == 7)
            {
                Matrix<T, 42, 42> hessian;
                if (scaled_barrier) 
                    computeVolumeBarrier7PointsScaledHessian(tet_vol_barrier_w, scaling_factor, positions, hessian);
                else
                    computeVolumeBarrier7PointsHessian(tet_vol_barrier_w, positions, hessian);
                if (projectPD)
                    projectBlockPD<42>(hessian);
                addHessianEntry<42>(entries, cell_vtx_list, hessian);
            }
            else if (face_vtx_list.size() == 8)
            {
                Matrix<T, 48, 48> hessian;
                if (scaled_barrier)
                    computeVolumeBarrier8PointsScaledHessian(tet_vol_barrier_w, scaling_factor, positions, hessian);
                else
                    computeVolumeBarrier8PointsHessian(tet_vol_barrier_w, positions, hessian);
                if (projectPD)
                    projectBlockPD<48>(hessian);
                addHessianEntry<48>(entries, cell_vtx_list, hessian);
            }
            else if (face_vtx_list.size() == 9)
            {
                Matrix<T, 54, 54> hessian;
                if (scaled_barrier)
                    computeVolumeBarrier9PointsScaledHessian(tet_vol_barrier_w, scaling_factor, positions, hessian);
                else
                    computeVolumeBarrier9PointsHessian(tet_vol_barrier_w, positions, hessian);
                if (projectPD)
                    projectBlockPD<54>(hessian);
                addHessianEntry<54>(entries, cell_vtx_list, hessian);
            }
        }
        
    });
}