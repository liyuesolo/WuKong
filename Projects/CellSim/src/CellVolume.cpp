#include "../include/VertexModel.h"
#include "../include/autodiff/CellVolume.h"


void VertexModel::computePentaPrismTetVol(const Vector<T, 30>& prism_vertices, Vector<T, 9>& tet_vol)
{
    auto computeTetVolume = [&](const TV& a, const TV& b, const TV& c, const TV& d)
    {
        return 1.0 / 6.0 * (b - a).cross(c - a).dot(d - a);
    };

    TV v0 = prism_vertices.segment<3>(9 * 3);
    TV v1 = prism_vertices.segment<3>(8 * 3);
    TV v2 = prism_vertices.segment<3>(7 * 3);
    TV v3 = prism_vertices.segment<3>(6 * 3);
    TV v4 = prism_vertices.segment<3>(5 * 3);

    TV v5 = prism_vertices.segment<3>(4 * 3);
    TV v6 = prism_vertices.segment<3>(3 * 3);
    TV v7 = prism_vertices.segment<3>(2 * 3);
    TV v8 = prism_vertices.segment<3>(1 * 3);
    TV v9 = prism_vertices.segment<3>(0 * 3);

    tet_vol[0] = computeTetVolume(v3, v8, v9, v0);
    tet_vol[1] = computeTetVolume(v3, v9, v4, v0);
    tet_vol[2] = computeTetVolume(v7, v0, v2, v1);
    tet_vol[3] = computeTetVolume(v9, v8, v5, v0);
    tet_vol[4] = computeTetVolume(v6, v0, v7, v1);
    tet_vol[5] = computeTetVolume(v5, v0, v7, v6);
    tet_vol[6] = computeTetVolume(v5, v8, v7, v0);
    tet_vol[7] = computeTetVolume(v8, v0, v2, v7);
    tet_vol[8] = computeTetVolume(v8, v3, v2, v0);

}

void VertexModel::computeHexPrismTetVol(const Vector<T, 36>& prism_vertices, Vector<T, 12>& tet_vol)
{
    std::vector<TV> vtx(12);
    for (int i = 0; i < 6; i++)
    {
        vtx[i] = prism_vertices.segment<3>((11 - i) * 3);
        vtx[i + 6] = prism_vertices.segment<3>((5 - i) * 3);
    }

    tet_vol[0] = computeTetVolume(vtx[9], vtx[2], vtx[10], vtx[3]);
    tet_vol[1] = computeTetVolume(vtx[2], vtx[10], vtx[3], vtx[4]);
    tet_vol[2] = computeTetVolume(vtx[1], vtx[11], vtx[0], vtx[7]);
    tet_vol[3] = computeTetVolume(vtx[9], vtx[2], vtx[8], vtx[10]);
    tet_vol[4] = computeTetVolume(vtx[10], vtx[1], vtx[7], vtx[11]);
    tet_vol[5] = computeTetVolume(vtx[0], vtx[11], vtx[6], vtx[7]);
    tet_vol[6] = computeTetVolume(vtx[1], vtx[11], vtx[4], vtx[5]);
    tet_vol[7] = computeTetVolume(vtx[2], vtx[10], vtx[1], vtx[8]);
    tet_vol[8] = computeTetVolume(vtx[2], vtx[10], vtx[4], vtx[1]);
    tet_vol[9] = computeTetVolume(vtx[1], vtx[11], vtx[5], vtx[0]);
    tet_vol[10] = computeTetVolume(vtx[10], vtx[1], vtx[8], vtx[7]);
    tet_vol[11] = computeTetVolume(vtx[10], vtx[1], vtx[11], vtx[4]);
}

void VertexModel::computeTetVolCurent(VectorXT& tet_vol_current)
{
    tet_vol_current = VectorXT::Zero(tet_vol_init.rows());
    int cnt = 0;
    iterateFaceSerial([&](VtxList& face_vtx_list, int face_idx){
        if (face_idx < basal_face_start)
        {
            VectorXT positions;
            VtxList cell_vtx_list = face_vtx_list;
            for (int idx : face_vtx_list)
                cell_vtx_list.push_back(idx + basal_vtx_start);
            positionsFromIndices(positions, cell_vtx_list);

            if (face_vtx_list.size() == 4)
            {

            }
            else if (face_vtx_list.size() == 5)
            {
                Vector<T, 9> tet_vol;
                computePentaPrismTetVol(positions, tet_vol);
                tet_vol_current.segment<9>(cnt) = tet_vol;
                cnt += 9;
            }
            else if (face_vtx_list.size() == 6)
            {
                Vector<T, 12> tet_vol;
                computeHexPrismTetVol(positions, tet_vol);
                tet_vol_current.segment<12>(cnt) = tet_vol;
                cnt += 12;
            }
        }
    });
}

void VertexModel::computeTetVolInitial()
{
    int total_tet_num = 0.0;
    iterateFaceSerial([&](VtxList& face_vtx_list, int face_idx){
        if (face_idx < basal_face_start)
        {
            if (face_vtx_list.size() == 4)
                total_tet_num += 6;
            else if (face_vtx_list.size() == 5)
                total_tet_num += 9;
            else if (face_vtx_list.size() == 6)
                total_tet_num += 12;
        }
    });
    tet_vol_init = VectorXT::Zero(total_tet_num);
    int cnt = 0;
    iterateFaceSerial([&](VtxList& face_vtx_list, int face_idx){
        if (face_idx < basal_face_start)
        {
            VectorXT positions;
            VtxList cell_vtx_list = face_vtx_list;
            for (int idx : face_vtx_list)
                cell_vtx_list.push_back(idx + basal_vtx_start);
            positionsFromIndices(positions, cell_vtx_list);

            if (face_vtx_list.size() == 4)
            {

            }
            else if (face_vtx_list.size() == 5)
            {
                Vector<T, 9> tet_vol;
                computePentaPrismTetVol(positions, tet_vol);
                tet_vol_init.segment<9>(cnt) = tet_vol;
                cnt += 9;
            }
            else if (face_vtx_list.size() == 6)
            {
                Vector<T, 12> tet_vol;
                computeHexPrismTetVol(positions, tet_vol);
                tet_vol_init.segment<12>(cnt) = tet_vol;
                cnt += 12;
            }
        }
    });
}


void VertexModel::addTetVolumePreservationEnergy(T& energy)
{
    int cnt = 0;
    iterateFaceSerial([&](VtxList& face_vtx_list, int face_idx)
    {
        if (face_idx < basal_face_start)
        {
            T tet_vol_preservation_energy = 0.0;
            VectorXT positions;
            VtxList cell_vtx_list = face_vtx_list;
            for (int idx : face_vtx_list)
                cell_vtx_list.push_back(idx + basal_vtx_start);

            positionsFromIndices(positions, cell_vtx_list);

            if (face_vtx_list.size() == 4)
            {

            }
            else if (face_vtx_list.size() == 5)
            {
                computePentaBasePrismVolumePenalty(tet_vol_penalty, positions, tet_vol_init.segment<9>(cnt), tet_vol_preservation_energy);
                cnt += 9;
            }
            else if (face_vtx_list.size() == 6)
            {
                computeHexBasePrismVolumePenalty(tet_vol_penalty, positions, tet_vol_init.segment<12>(cnt), tet_vol_preservation_energy);
                cnt += 12;
            }

            energy += tet_vol_preservation_energy;
        }
    });
}
void VertexModel::addTetVolumePreservationForceEntries(VectorXT& residual)
{
    int cnt = 0;
    iterateFaceSerial([&](VtxList& face_vtx_list, int face_idx)
    {
        if (face_idx < basal_face_start)
        {
            
            VectorXT positions;
            VtxList cell_vtx_list = face_vtx_list;
            for (int idx : face_vtx_list)
                cell_vtx_list.push_back(idx + basal_vtx_start);

            positionsFromIndices(positions, cell_vtx_list);

            if (face_vtx_list.size() == 4)
            {

            }
            else if (face_vtx_list.size() == 5)
            {
                Vector<T, 30> dedx;
                computePentaBasePrismVolumePenaltyGradient(tet_vol_penalty, positions, 
                    tet_vol_init.segment<9>(cnt), dedx);
                addForceEntry<30>(residual, cell_vtx_list, -dedx);
                cnt += 9;
                
            }
            else if (face_vtx_list.size() == 6)
            {
                Vector<T, 36> dedx;
                computeHexBasePrismVolumePenaltyGradient(tet_vol_penalty, positions, 
                    tet_vol_init.segment<12>(cnt), dedx);
                addForceEntry<36>(residual, cell_vtx_list, -dedx);
                cnt += 12;
            }
        }
    });
}
void VertexModel::addTetVolumePreservationHessianEntries(std::vector<Entry>& entries, 
    bool projectPD)
{
    int cnt = 0;
    iterateFaceSerial([&](VtxList& face_vtx_list, int face_idx)
    {
        if (face_idx < basal_face_start)
        {
            T tet_vol_preservation_energy = 0.0;
            VectorXT positions;
            VtxList cell_vtx_list = face_vtx_list;
            for (int idx : face_vtx_list)
                cell_vtx_list.push_back(idx + basal_vtx_start);

            positionsFromIndices(positions, cell_vtx_list);

            if (face_vtx_list.size() == 4)
            {

            }
            else if (face_vtx_list.size() == 5)
            {
                Matrix<T, 30, 30> hessian;
                computePentaBasePrismVolumePenaltyHessian(tet_vol_penalty, positions, 
                    tet_vol_init.segment<9>(cnt), hessian);
                if (projectPD)
                    projectBlockPD<30>(hessian);
                addHessianEntry<30>(entries, cell_vtx_list, hessian);
                cnt += 9;
                
            }
            else if (face_vtx_list.size() == 6)
            {
                Matrix<T, 36, 36> hessian;
                computeHexBasePrismVolumePenaltyHessian(tet_vol_penalty, positions, 
                    tet_vol_init.segment<12>(cnt), hessian);
                if (projectPD)
                    projectBlockPD<36>(hessian);
                addHessianEntry<36>(entries, cell_vtx_list, hessian);
                cnt += 12;
            }
        }
    });
}

void VertexModel::computeVolumeAllCells(VectorXT& cell_volume_list)
{
    // each apical face corresponds to one cell
    cell_volume_list = VectorXT::Zero(basal_face_start);

    // use apical face to iterate other faces within this cell for now
    int cnt = 0;
    iterateFaceSerial([&](VtxList& face_vtx_list, int face_idx){
        if (face_idx < basal_face_start)
        {
            VectorXT positions;
            VtxList cell_vtx_list = face_vtx_list;
            for (int idx : face_vtx_list)
                cell_vtx_list.push_back(idx + basal_vtx_start);
            

            positionsFromIndices(positions, cell_vtx_list);
            
            if (face_vtx_list.size() == 4)
            {
                if (preserve_tet_vol)
                {

                }
                else
                {
                    if (use_cell_centroid)
                        computeVolume4Points(positions, cell_volume_list[face_idx]);
                    else
                        computeQuadBasePrismVolume(positions, cell_volume_list[face_idx]);
                }
            }
            else if (face_vtx_list.size() == 5)
            {
                if (preserve_tet_vol)
                {
                    Vector<T, 9> tet_vol;
                    computePentaPrismTetVol(positions, tet_vol);
                    cell_volume_list[face_idx] += tet_vol.sum();
                }
                else
                {
                    if (use_cell_centroid)
                    {
                        if (use_fixed_cell_centroid)
                            computeVolume5PointsFixedCentroid(positions, fixed_cell_centroids.segment<3>(face_idx * 3), cell_volume_list[face_idx]);
                        else
                            computeVolume5Points(positions, cell_volume_list[face_idx]);
                    }
                    else
                        computePentaBasePrismVolume(positions, cell_volume_list[face_idx]);
                }
                
            }
            else if (face_vtx_list.size() == 6)
            {
                if (preserve_tet_vol)
                {
                    Vector<T, 12> tet_vol;
                    computeHexPrismTetVol(positions, tet_vol);
                    cell_volume_list[face_idx] += tet_vol.sum();
                }
                else
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
                }
                
            }
        }
    });
}


void VertexModel::addCellVolumePreservationEnergy(T& energy)
{
    VectorXT current_cell_volume;
    computeVolumeAllCells(current_cell_volume);

    T volume_term = 0.0;
    iterateFaceSerial([&](VtxList& face_vtx_list, int face_idx)
    {
        VectorXT positions;
        positionsFromIndices(positions, face_vtx_list);
        
        // cell-wise volume preservation term
        if (face_idx < basal_face_start)
        {
            T ci = current_cell_volume[face_idx] - cell_volume_init[face_idx];
            if (use_alm_on_cell_volume)
                volume_term += -lambda_cell_vol[face_idx] * ci + 0.5 * kappa * std::pow(ci, 2);
            else
                volume_term += 0.5 * B * std::pow(ci, 2);
            
        }
    });
    energy += volume_term;
}

void VertexModel::addCellVolumePreservationForceEntries(VectorXT& residual)
{
    VectorXT current_cell_volume;
    computeVolumeAllCells(current_cell_volume);

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
                    {
                        if (use_fixed_cell_centroid)
                            computeVolume5PointsFixedCentroidGradient(positions, fixed_cell_centroids.segment<3>(face_idx * 3), dedx);
                        else
                            computeVolume5PointsGradient(positions, dedx);
                    }
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
                }
                else
                {
                    std::cout << "unknown polygon edge case" << std::endl;
                }
            }
        }
    });
}

void VertexModel::addCellVolumePreservationHessianEntries(std::vector<Entry>& entries, bool projectPD)
{
    VectorXT current_cell_volume;
    computeVolumeAllCells(current_cell_volume);

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
                hessian.setZero();
                if (use_alm_on_cell_volume)
                {
                    hessian = -lambda_cell_vol[face_idx] * d2Vdx2 + 
                            kappa * (dVdx * dVdx.transpose() + 
                            (V - cell_volume_init[face_idx]) * d2Vdx2);
                }
                else
                {    
                    
                    hessian += B * dVdx * dVdx.transpose();
                    hessian += B * (V - cell_volume_init[face_idx]) * d2Vdx2;
                }
                if(projectPD)
                    projectBlockPD<24>(hessian);
                addHessianEntry<24>(entries, cell_vtx_list, hessian);
            }
            else if (face_vtx_list.size() == 5)
            {
                Matrix<T, 30, 30> d2Vdx2;
                if (use_cell_centroid)
                {
                    if (use_fixed_cell_centroid)
                        computeVolume5PointsFixedCentroidHessian(positions, fixed_cell_centroids.segment<3>(face_idx * 3), d2Vdx2);
                    else
                        computeVolume5PointsHessian(positions, d2Vdx2);
                }
                else 
                    computePentaBasePrismVolumeHessian(positions, d2Vdx2);

                Vector<T, 30> dVdx;
                if (use_cell_centroid)
                {
                    if (use_fixed_cell_centroid)
                        computeVolume5PointsFixedCentroidGradient(positions, fixed_cell_centroids.segment<3>(face_idx * 3), dVdx);
                    else
                        computeVolume5PointsGradient(positions, dVdx);
                }
                else
                    computePentaBasePrismVolumeGradient(positions, dVdx);
                
                // break it down here to avoid super long autodiff code
                
                Matrix<T, 30, 30> hessian;
                hessian.setZero();
                if (use_alm_on_cell_volume)
                {
                    hessian = -lambda_cell_vol[face_idx] * d2Vdx2 + 
                            kappa * (dVdx * dVdx.transpose() + 
                            (V - cell_volume_init[face_idx]) * d2Vdx2);
                }
                else
                {    
                    hessian += B * dVdx * dVdx.transpose();
                    hessian += B * (V - cell_volume_init[face_idx]) * d2Vdx2;
                }
                if(projectPD)
                    projectBlockPD<30>(hessian);
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
                hessian.setZero();
                if (use_alm_on_cell_volume)
                {
                    hessian = -lambda_cell_vol[face_idx] * d2Vdx2 + 
                            kappa * (dVdx * dVdx.transpose() + 
                            (V - cell_volume_init[face_idx]) * d2Vdx2);
                }
                else
                {    
                    hessian += B * dVdx * dVdx.transpose();
                    hessian += B * (V - cell_volume_init[face_idx]) * d2Vdx2;
                }
                if(projectPD)
                    projectBlockPD<36>(hessian);
                addHessianEntry<36>(entries, cell_vtx_list, hessian);
            }
            else
            {
                std::cout << "unknown polygon edge case" << std::endl;
            }
            // std::cout << "Cell " << face_idx << std::endl;
        }
    });
}