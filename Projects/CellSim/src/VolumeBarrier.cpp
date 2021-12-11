#include "../include/VertexModel.h"
#include "../include/autodiff/TetVolBarrier.h"

void VertexModel::addFixedTetLogBarrierEnergy(T& energy)
{
    T barrier_energy = 0.0;
    iterateFixedTetsSerial([&](TetVtx& x_deformed, TetVtx& x_undeformed, VtxList& indices)
    {
        T e = 0.0;
        T d = computeTetVolume(x_deformed.col(0), x_deformed.col(1), x_deformed.col(2), x_deformed.col(3));
        if (d < tet_vol_barrier_dhat)
        {
            computeTetInversionBarrier(tet_vol_barrier_w, tet_vol_barrier_dhat, x_deformed, e);
            barrier_energy += e;
        }
    });
    energy += barrier_energy;
}

void VertexModel::addFixedTetLogBarrierForceEneries(VectorXT& residual)
{
    iterateFixedTetsSerial([&](TetVtx& x_deformed, TetVtx& x_undeformed, VtxList& indices)
    {
        T d = computeTetVolume(x_deformed.col(0), x_deformed.col(1), x_deformed.col(2), x_deformed.col(3));
        if (d < tet_vol_barrier_dhat)
        {
            Vector<T, 12> dedx;
            computeTetInversionBarrierGradient(tet_vol_barrier_w, tet_vol_barrier_dhat, x_deformed, dedx);
            addForceEntry<12>(residual, indices, -dedx);
        }
    });
}

void VertexModel::addFixedTetLogBarrierHessianEneries(std::vector<Entry>& entries, bool projectPD)
{
    iterateFixedTetsSerial([&](TetVtx& x_deformed, TetVtx& x_undeformed, VtxList& indices)
    {
        T d = computeTetVolume(x_deformed.col(0), x_deformed.col(1), x_deformed.col(2), x_deformed.col(3));
        if (d < tet_vol_barrier_dhat)
        {
            Matrix<T, 12, 12> hessian;
            computeTetInversionBarrierHessian(tet_vol_barrier_w, tet_vol_barrier_dhat, x_deformed, hessian);
            if (projectPD)
                projectBlockPD<12>(hessian);
            addHessianEntry<12>(entries, indices, hessian);
        }
    });
}

T VertexModel::computeInversionFreeStepSize(const VectorXT& _u, const VectorXT& du)
{
    T step_size = 1.0;
    while (true)
    {
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
                        if (-computeTetVolume(apical_centroid, r1, r0, cell_centroid) < 1e-8)
                        {
                            constraint_violated = true;
                            break;
                        }

                        TV r2 = positions.segment<3>((i + face_vtx_list.size()) * 3);
                        TV r3 = positions.segment<3>((j + face_vtx_list.size()) * 3);
                        if (computeTetVolume(basal_centroid, r3, r2, cell_centroid) < 1e-8)
                        {
                            constraint_violated = true;
                            break;
                        }

                        TV lateral_centroid = T(0.25) * (r0 + r1 + r2 + r3);
                        if (computeTetVolume(lateral_centroid, r1, r0, cell_centroid) < 1e-8)
                        {
                            constraint_violated = true;
                            break;
                        }
                        if (computeTetVolume(lateral_centroid, r3, r1, cell_centroid) < 1e-8)
                        {
                            constraint_violated = true;
                            break;
                        }
                        if (computeTetVolume(lateral_centroid, r2, r3, cell_centroid) < 1e-8)
                        {
                            constraint_violated = true;
                            break;
                        }
                        if (computeTetVolume(lateral_centroid, r0, r2, cell_centroid) < 1e-8)
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
                
                T d = computeTetVolume(x_deformed.col(0), x_deformed.col(1), x_deformed.col(2), x_deformed.col(3));
                if (d < 1e-7)
                {
                    constraint_violated = true;
                }
            });
        }
        
        if (constraint_violated)
            step_size *= 0.8;
        else
            return step_size;
    }
}

void VertexModel::addSingleTetVolBarrierEnergy(T& energy)
{
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

            if(face_vtx_list.size() == 4)
            {
                computeVolumeBarrier4Points(tet_vol_barrier_w, positions, volume_barrier_energy);
                energy += volume_barrier_energy;
            }
            else if(face_vtx_list.size() == 5)
            {
                computeVolumeBarrier5Points(tet_vol_barrier_w, positions, volume_barrier_energy);
                energy += volume_barrier_energy;
            }
            else if(face_vtx_list.size() == 6)
            {
                computeVolumeBarrier6Points(tet_vol_barrier_w, positions, volume_barrier_energy);
                energy += volume_barrier_energy;
            }
        }
    });
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

            // cell-wise volume preservation term
            if (face_idx < basal_face_start)
            {
                if (face_vtx_list.size() == 4)
                {
                    Vector<T, 24> dedx;
                    computeVolumeBarrier4PointsGradient(tet_vol_barrier_w, positions, dedx);
                    addForceEntry<24>(residual, cell_vtx_list, -dedx);
                }
                else if (face_vtx_list.size() == 5)
                {
                    Vector<T, 30> dedx;
                    computeVolumeBarrier5PointsGradient(tet_vol_barrier_w, positions, dedx);
                    addForceEntry<30>(residual, cell_vtx_list, -dedx);
                }
                else if (face_vtx_list.size() == 6)
                {
                    Vector<T, 36> dedx;
                    computeVolumeBarrier6PointsGradient(tet_vol_barrier_w, positions, dedx);
                    addForceEntry<36>(residual, cell_vtx_list, -dedx);
                }
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
            
            if (face_vtx_list.size() == 4)
            {
                Matrix<T, 24, 24> hessian;
                computeVolumeBarrier4PointsHessian(tet_vol_barrier_w, positions, hessian);
                addHessianEntry<24>(entries, cell_vtx_list, hessian);
            }
            else if (face_vtx_list.size() == 5)
            {
                Matrix<T, 30, 30> hessian;
                computeVolumeBarrier5PointsHessian(tet_vol_barrier_w, positions, hessian);
                addHessianEntry<30>(entries, cell_vtx_list, hessian);
            }
            else if (face_vtx_list.size() == 6)
            {
                Matrix<T, 36, 36> hessian;
                computeVolumeBarrier6PointsHessian(tet_vol_barrier_w, positions, hessian);
                addHessianEntry<36>(entries, cell_vtx_list, hessian);
            }
        }
        
    });
}