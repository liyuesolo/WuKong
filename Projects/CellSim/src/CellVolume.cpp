#include "../include/VertexModel.h"
#include "../include/autodiff/CellVolume.h"

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
                if(projectPD)
                    projectBlockPD<24>(hessian);
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