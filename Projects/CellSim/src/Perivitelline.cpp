
#include "../include/VertexModel.h"
#include "../include/autodiff/YolkEnergy.h"

bool use_centroid_subdivide = true;

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
            {
                if (use_centroid_subdivide)
                    computeConeVolume4Points(positions, mesh_centroid, cone_volume);
                else
                    computeQuadConeVolume(positions, mesh_centroid, cone_volume);
            }
            else if (face_vtx_list.size() == 5) 
            {
                if (use_centroid_subdivide)
                    computeConeVolume5Points(positions, mesh_centroid, cone_volume);
                else
                    computePentaConeVolume(positions, mesh_centroid, cone_volume);
            }
            else if (face_vtx_list.size() == 6) 
            {
                if (use_centroid_subdivide)
                    computeConeVolume6Points(positions, mesh_centroid, cone_volume);
                else
                    computeHexConeVolume(positions, mesh_centroid, cone_volume);
            }
            else
                std::cout << "unknown polygon edge number" << __FILE__ << std::endl;
            volume += cone_volume;
        }
        
    });
    return -volume;
}

void VertexModel::addPerivitellineVolumePreservationEnergy(T& energy)
{
    T volume_penalty_previtelline = 0.0;
    T perivitelline_vol_curr = total_volume - computeTotalVolumeFromApicalSurface();
    if (use_perivitelline_liquid_pressure)
        volume_penalty_previtelline = -perivitelline_pressure * perivitelline_vol_curr;
    else
        volume_penalty_previtelline += 0.5 * Bp * std::pow(perivitelline_vol_curr - perivitelline_vol_init, 2);
    energy += volume_penalty_previtelline;
}

void VertexModel::addPerivitellineVolumePreservationForceEntries(VectorXT& residual)
{
    T perivitelline_vol_curr = total_volume - computeTotalVolumeFromApicalSurface();

    iterateFaceSerial([&](VtxList& face_vtx_list, int face_idx)
    {
        VectorXT positions;
        positionsFromIndices(positions, face_vtx_list);
        
        if (face_idx < basal_face_start)
        {
            T coeff = use_perivitelline_liquid_pressure ? pressure_constant
                : -Bp * (perivitelline_vol_curr - perivitelline_vol_init);
            // negative is correct 
            if (face_vtx_list.size() == 4)
            {
                Vector<T, 12> dedx;
                if (use_centroid_subdivide)
                    computeConeVolume4PointsGradient(positions, mesh_centroid, dedx);
                else
                    computeQuadConeVolumeGradient(positions, mesh_centroid, dedx);
                dedx *= coeff;
                addForceEntry<12>(residual, face_vtx_list, dedx);
            }
            else if (face_vtx_list.size() == 5)
            {
                Vector<T, 15> dedx;
                if (use_centroid_subdivide)
                    computeConeVolume5PointsGradient(positions, mesh_centroid, dedx);
                else
                    computePentaConeVolumeGradient(positions, mesh_centroid, dedx);
                dedx *= coeff;
                addForceEntry<15>(residual, face_vtx_list, dedx);
            }
            else if (face_vtx_list.size() == 6)
            {
                Vector<T, 18> dedx;
                if (use_centroid_subdivide)
                    computeConeVolume6PointsGradient(positions, mesh_centroid, dedx);
                else
                    computeHexConeVolumeGradient(positions, mesh_centroid, dedx);
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

void VertexModel::addPerivitellineVolumePreservationHessianEntries(std::vector<Entry>& entries,
    MatrixXT& WoodBuryMatrix, bool projectPD)
{
    
    if (add_perivitelline_liquid_volume && !use_perivitelline_liquid_pressure)
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
                    if (use_centroid_subdivide)
                        computeConeVolume4PointsGradient(positions, mesh_centroid, dedx);
                    else
                        computeQuadConeVolumeGradient(positions, mesh_centroid, dedx);
                    addForceEntry<12>(dVdx_full, face_vtx_list, -dedx);
                }
                else if (face_vtx_list.size() == 5)
                {
                    Vector<T, 15> dedx;
                    if (use_centroid_subdivide)
                        computeConeVolume5PointsGradient(positions, mesh_centroid, dedx);
                    else
                        computePentaConeVolumeGradient(positions, mesh_centroid, dedx);
                    addForceEntry<15>(dVdx_full, face_vtx_list, -dedx);
                }
                else if (face_vtx_list.size() == 6)
                {
                    Vector<T, 18> dedx;
                    if (use_centroid_subdivide)
                        computeConeVolume6PointsGradient(positions, mesh_centroid, dedx);
                    else
                        computeHexConeVolumeGradient(positions, mesh_centroid, dedx);
                    addForceEntry<18>(dVdx_full, face_vtx_list, -dedx);
                }
                else
                {
                    std::cout << "unknown polygon edge number" << std::endl;
                }
            }
        });

        if (woodbury)
        {
            dVdx_full *= std::sqrt(Bp);
            if (!run_diff_test)
            {
                iterateDirichletDoF([&](int offset, T target)
                {
                    dVdx_full[offset] = 0.0;
                });
            }
            int n_row = num_nodes * 3, n_col = WoodBuryMatrix.cols();
            WoodBuryMatrix.conservativeResize(n_row, n_col + 1);
            WoodBuryMatrix.col(n_col) = dVdx_full;
        }
        else
        {
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
        
    }


    iterateFaceSerial([&](VtxList& face_vtx_list, int face_idx)
    {
        VectorXT positions;
        positionsFromIndices(positions, face_vtx_list);
        T perivitelline_vol_curr = total_volume - computeTotalVolumeFromApicalSurface();

            if (face_idx < basal_face_start)
            {
                VectorXT positions;
                positionsFromIndices(positions, face_vtx_list);
                T ci = perivitelline_vol_curr - perivitelline_vol_init;
                T coeff = use_perivitelline_liquid_pressure ? -pressure_constant : Bp * ci;
                if (face_vtx_list.size() == 4)
                {
                    
                    Matrix<T, 12, 12> d2Vdx2;
                    if (use_centroid_subdivide)
                        computeConeVolume4PointsHessian(positions, mesh_centroid, d2Vdx2);
                    else
                        computeQuadConeVolumeHessian(positions, mesh_centroid, d2Vdx2);
                    Matrix<T, 12, 12> hessian = coeff * d2Vdx2;
                    if(projectPD)
                        projectBlockPD<12>(hessian);
                    addHessianEntry<12>(entries, face_vtx_list, hessian);
                }
                else if (face_vtx_list.size() == 5)
                {
                    Matrix<T, 15, 15> d2Vdx2;
                    if (use_centroid_subdivide)
                        computeConeVolume5PointsHessian(positions, mesh_centroid, d2Vdx2);
                    else
                        computePentaConeVolumeHessian(positions, mesh_centroid, d2Vdx2);
                    Matrix<T, 15, 15> hessian = coeff * d2Vdx2;
                    if(projectPD)
                        projectBlockPD<15>(hessian);
                    addHessianEntry<15>(entries, face_vtx_list, hessian);

                }
                else if (face_vtx_list.size() == 6)
                {
                    Matrix<T, 18, 18> d2Vdx2;
                    if (use_centroid_subdivide)
                        computeConeVolume6PointsHessian(positions, mesh_centroid, d2Vdx2);
                    else
                        computeHexConeVolumeHessian(positions, mesh_centroid, d2Vdx2);
                    Matrix<T, 18, 18> hessian = coeff * d2Vdx2;
                    if(projectPD)
                        projectBlockPD<18>(hessian);
                    addHessianEntry<18>(entries, face_vtx_list, hessian);
                }
                else
                {
                    std::cout << "unknown polygon edge case" << std::endl;
                }
            }
        }); 
}