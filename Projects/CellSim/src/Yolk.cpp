
#include "../include/VertexModel.h"
#include "../include/autodiff/YolkEnergy.h"

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

void VertexModel::addYolkVolumePreservationEnergy(T& energy)
{
    T yolk_vol_curr = computeYolkVolume();
    if (use_yolk_pressure)
    {
        energy += -pressure_constant * yolk_vol_curr;
    }
    else
    {
        energy +=  0.5 * By * std::pow(yolk_vol_curr - yolk_vol_init, 2);    
    }
}

void VertexModel::addYolkVolumePreservationForceEntries(VectorXT& residual)
{
    T yolk_vol_curr = computeYolkVolume();


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
}

void VertexModel::addYolkVolumePreservationHessianEntries(std::vector<Entry>& entries,
    MatrixXT& WoodBuryMatrix, bool projectPD)
{
    T yolk_vol_curr = computeYolkVolume();

    
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

        if (woodbury)
        {
            dVdx_full *= std::sqrt(By);
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
                    Matrix<T, 3, 3> hessian_partial = By * dVdxi * dVdxj.transpose();
                    if(projectPD)
                        projectBlockPD<3>(hessian_partial);
                    addHessianBlock<3>(entries, {dof_i, dof_j}, hessian_partial);
                }
            }
        }
        
    }

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
                    
                    Matrix<T, 12, 12> d2Vdx2;
                    computeConeVolume4PointsHessian(positions, mesh_centroid, d2Vdx2);
                    // computeQuadConeVolumeHessian(positions, mesh_centroid, d2Vdx2);
                    Matrix<T, 12, 12> hessian;
                    if (use_yolk_pressure)
                        hessian = -pressure_constant * d2Vdx2;
                    else
                        hessian = By * (yolk_vol_curr - yolk_vol_init) * d2Vdx2;
                    if(projectPD)
                        projectBlockPD<12>(hessian);
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
                    if(projectPD)
                        projectBlockPD<15>(hessian);
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
                    if(projectPD)
                        projectBlockPD<18>(hessian);
                    addHessianEntry<18>(entries, face_vtx_list, hessian);
                }
                else
                {
                    std::cout << "unknown polygon edge case" << std::endl;
                }
            }
        }

        
    });
}