
#include "../include/VertexModel.h"
#include "../include/autodiff/YolkEnergy.h"
#include "../include/autodiff/TetVolBarrier.h"
#include <fstream>

// bool use_cell_centroid = false;

T VertexModel::computeYolkInversionFreeStepSize(const VectorXT& _u, const VectorXT& du)
{
    T step_size = 1.0;
    while (true)
    {
        deformed = undeformed + _u + step_size * du;
        
        bool constraint_violated = false;
        if (use_cell_centroid)
        {
            iterateFaceSerial([&](VtxList& face_vtx_list, int face_idx)
            {
                if (face_idx < lateral_face_start && face_idx >= basal_face_start)
                {    
                    VectorXT positions;
                    positionsFromIndices(positions, face_vtx_list);
                    TV basal_centroid = TV::Zero();
                    for (int i = 0; i < face_vtx_list.size(); i++)
                        basal_centroid += positions.segment<3>(i * 3);
                    for (int i = 0; i < face_vtx_list.size(); i++)
                    {
                        int j = (i + 1) % face_vtx_list.size();
                        TV r0 = positions.segment<3>(i * 3);
                        TV r1 = positions.segment<3>(j * 3);
                        if (computeTetVolume(basal_centroid, r1, r0, mesh_centroid) < 1e-8)
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
            iterateFixedYolkTetsSerial([&](TM3& x_deformed, VtxList& indices)
            {
                if (constraint_violated)
                    return;
                T d = computeTetVolume(mesh_centroid, x_deformed.col(0), x_deformed.col(1), x_deformed.col(2));
                if (d < 1e-6)
                {
                    // std::cout << d << std::endl;
                    constraint_violated = true;
                    return;
                }
            });
        if (constraint_violated)
            step_size *= 0.8;
        else
            return step_size;
        // if (step_size < 1e-6)
        // {
        //     saveLowVolumeTets("low_vol_tet.obj");
        //     saveBasalSurfaceMesh("low_vol_tet_basal_surface.obj");

        //     ÷std::exit(0);
        // }
    }
}

void VertexModel::addYolkTetLogBarrierEnergy(T& energy)
{
    T yolk_tet_barrier_energy = 0.0;
    iterateFixedYolkTetsSerial([&](TM3& x_deformed, VtxList& indices)
    {
        T ei = 0.0;
        T d = computeTetVolume(mesh_centroid, x_deformed.col(0), x_deformed.col(1), x_deformed.col(2));
        
        if (d < yolk_tet_vol_barrier_dhat)
        {
            computeTetInversionBarrierFixedCentroid(yolk_tet_vol_barrier_w, yolk_tet_vol_barrier_dhat,
                x_deformed, mesh_centroid, ei);
            yolk_tet_barrier_energy += ei;
        }
    });
    energy += yolk_tet_barrier_energy;
}
void VertexModel::addYolkTetLogBarrierForceEneries(VectorXT& residual)
{
    iterateFixedYolkTetsSerial([&](TM3& x_deformed, VtxList& indices)
    {
        T d = computeTetVolume(mesh_centroid, x_deformed.col(0), x_deformed.col(1), x_deformed.col(2));
        if (d < yolk_tet_vol_barrier_dhat)
        {
            std::cout << "small tet "<< std::endl;
            Vector<T, 9> dedx;
            computeTetInversionBarrierFixedCentroidGradient(yolk_tet_vol_barrier_w, yolk_tet_vol_barrier_dhat,
                x_deformed, mesh_centroid, dedx);
            addForceEntry<9>(residual, indices, -dedx);
        }
    });
}
void VertexModel::addYolkTetLogBarrierHessianEneries(std::vector<Entry>& entries, bool projectPD)
{
    iterateFixedYolkTetsSerial([&](TM3& x_deformed, VtxList& indices)
    {
        T d = computeTetVolume(mesh_centroid, x_deformed.col(0), x_deformed.col(1), x_deformed.col(2));
        if (d < yolk_tet_vol_barrier_dhat)
        {
            Matrix<T, 9, 9> hessian;
            computeTetInversionBarrierFixedCentroidHessian(yolk_tet_vol_barrier_w, yolk_tet_vol_barrier_dhat,
                x_deformed, mesh_centroid, hessian);
            if (projectPD)
                projectBlockPD<9>(hessian);
            addHessianEntry<9>(entries, indices, hessian);
        }
    });
}

T VertexModel::computeYolkVolume(bool verbose)
{
    VectorXT volumes = VectorXT::Zero(lateral_face_start - basal_face_start);
    iterateFaceParallel([&](VtxList& face_vtx_list, int face_idx)
    {
        if (face_idx >= lateral_face_start || face_idx < basal_face_start)
            return;
        VectorXT positions;
        positionsFromIndices(positions, face_vtx_list);
        if (use_cell_centroid)
        {
            if (face_vtx_list.size() == 4) 
                computeConeVolume4Points(positions, mesh_centroid, volumes[face_idx - basal_face_start]);
            else if (face_vtx_list.size() == 5) 
                computeConeVolume5Points(positions, mesh_centroid, volumes[face_idx - basal_face_start]);
            else if (face_vtx_list.size() == 6) 
                computeConeVolume6Points(positions, mesh_centroid, volumes[face_idx - basal_face_start]);
            else if (face_vtx_list.size() == 7) 
                computeConeVolume7Points(positions, mesh_centroid, volumes[face_idx - basal_face_start]);
            else if (face_vtx_list.size() == 8) 
                computeConeVolume8Points(positions, mesh_centroid, volumes[face_idx - basal_face_start]);
            else if (face_vtx_list.size() == 9) 
                computeConeVolume9Points(positions, mesh_centroid, volumes[face_idx - basal_face_start]);
        }
    });
    return volumes.sum();

    // T yolk_volume = 0.0;
    
    // iterateFaceSerial([&](VtxList& face_vtx_list, int face_idx)
    // {
    //     VectorXT positions;
    //     positionsFromIndices(positions, face_vtx_list);
        
    //     if (face_idx < lateral_face_start && face_idx >= basal_face_start)
    //     {
    //         T cone_volume;
    //         if (face_vtx_list.size() == 4) 
    //         {
    //             if (use_cell_centroid)
    //                 computeConeVolume4Points(positions, mesh_centroid, cone_volume);
    //             else
    //                 computeQuadConeVolume(positions, mesh_centroid, cone_volume);
    //         }
    //         else if (face_vtx_list.size() == 5) 
    //         {
    //             if (use_cell_centroid)
    //                 computeConeVolume5Points(positions, mesh_centroid, cone_volume);
    //             else
    //                 computePentaConeVolume(positions, mesh_centroid, cone_volume);
    //         }
    //         else if (face_vtx_list.size() == 6) 
    //         {
    //             if (use_cell_centroid)
    //                 computeConeVolume6Points(positions, mesh_centroid, cone_volume);
    //             else
    //                 computeHexConeVolume(positions, mesh_centroid, cone_volume);
    //         }
    //         else if (face_vtx_list.size() == 7) 
    //         {
    //             if (use_cell_centroid)
    //                 computeConeVolume7Points(positions, mesh_centroid, cone_volume);
    //         }
    //         else if (face_vtx_list.size() == 8) 
    //         {
    //             if (use_cell_centroid)
    //                 computeConeVolume8Points(positions, mesh_centroid, cone_volume);
    //         }    
    //         else
    //             std::cout << "unknown polygon edge number" << __FILE__ << std::endl;
    //         yolk_volume += cone_volume;
    //         if (verbose)
    //         {
    //             std::cout << cone_volume << " ";
    //             if (cone_volume < 0)
    //                 std::cout << "negative volume " << cone_volume << std::endl;
    //         }
    //     }
        
    // });
    // // if (verbose)
    //     // std::cout << "yolk_volume " << yolk_volume << std::endl;
    // // std::getchar();
    // return yolk_volume; 
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
                    if (use_cell_centroid)
                        computeConeVolume4PointsGradient(positions, mesh_centroid, dedx);
                    else
                        computeQuadConeVolumeGradient(positions, mesh_centroid, dedx);
                    dedx *= -coeff;
                    addForceEntry<12>(residual, face_vtx_list, dedx);
                }
                else if (face_vtx_list.size() == 5)
                {
                    Vector<T, 15> dedx;
                    if (use_cell_centroid)
                        computeConeVolume5PointsGradient(positions, mesh_centroid, dedx);
                    else
                        computePentaConeVolumeGradient(positions, mesh_centroid, dedx);
                    dedx *= -coeff;
                    // bool trouble_tet = false;
                    // for (int i = 0; i < 5; i++)
                    // {
                    //     TV xi = positions.segment<3>(i * 3);
                    //     if (dedx.norm() < 1e-8)
                    //         continue;
                    //     if (dedx.segment<3>(i * 3).dot(xi - mesh_centroid) > 0)
                    //     {
                    //         trouble_tet = true;
                    //         break;
                    //     }
                    // }
                    // if (trouble_tet)
                    // {
                    //     saveBasalSurfaceMesh("trouble_surface.obj");
                    //     std::exit(0);
                    //     VectorXT rest_positions;
                    //     positionsFromIndices(rest_positions, face_vtx_list, true);
                    //     std::ofstream out("trouble_tet_rest.obj");
                    //     for (int i = 0; i < 5; i++)
                    //     {
                    //         TV xi = rest_positions.segment<3>(i * 3);
                    //         out << "v " <<  xi.transpose() << std::endl;
                    //     }
                    //     out << "v " <<  mesh_centroid.transpose() << std::endl;
                    //     for (int i = 0; i < 5; i++)
                    //     {
                            
                    //         out << "f 1 2 3" << std::endl; 
                    //         out << "f 1 3 4" << std::endl; 
                    //         out << "f 1 4 5" << std::endl; 
                    //         out << "f 2 1 6" << std::endl; 
                    //         out << "f 3 2 6" << std::endl; 
                    //         out << "f 4 3 6" << std::endl; 
                    //         out << "f 5 4 6" << std::endl; 
                    //         out << "f 1 5 6" << std::endl; 
                    //     }
                    //     out.close();

                    //     out.open("trouble_tet.obj");
                    //     for (int i = 0; i < 5; i++)
                    //     {
                    //         TV xi = positions.segment<3>(i * 3);
                    //         out << "v " <<  xi.transpose() << std::endl;
                    //     }
                    //     out << "v " <<  mesh_centroid.transpose() << std::endl;
                    //     for (int i = 0; i < 5; i++)
                    //     {
                            
                    //         out << "f 1 2 3" << std::endl; 
                    //         out << "f 1 3 4" << std::endl; 
                    //         out << "f 1 4 5" << std::endl; 
                    //         out << "f 2 1 6" << std::endl; 
                    //         out << "f 3 2 6" << std::endl; 
                    //         out << "f 4 3 6" << std::endl; 
                    //         out << "f 5 4 6" << std::endl; 
                    //         out << "f 1 5 6" << std::endl; 
                    //     }
                    //     out.close();
                    //     std::getchar();
                    // }
                    addForceEntry<15>(residual, face_vtx_list, dedx);
                }
                else if (face_vtx_list.size() == 6)
                {
                    Vector<T, 18> dedx;
                    if (use_cell_centroid)
                        computeConeVolume6PointsGradient(positions, mesh_centroid, dedx);
                    else
                        computeHexConeVolumeGradient(positions, mesh_centroid, dedx);
                    dedx *= -coeff;
                    // bool trouble_tet = false;
                    // for (int i = 0; i < 6; i++)
                    // {
                    //     TV xi = positions.segment<3>(i * 3);
                    //     if (dedx.norm() < 1e-8)
                    //         continue;
                    //     if (dedx.segment<3>(i * 3).dot(xi - mesh_centroid) > 0)
                    //     {
                    //         trouble_tet = true;
                    //         break;
                    //     }
                    // }
                    // if (trouble_tet)
                    // {
                    //     saveAPrism("trouble_prism.obj", face_vtx_list);
                    //     saveBasalSurfaceMesh("trouble_surface.obj");
                    //     // std::exit(0);
                    //     VectorXT rest_positions;
                    //     positionsFromIndices(rest_positions, face_vtx_list, true);
                    //     std::ofstream out("trouble_tet_rest.obj");
                    //     for (int i = 0; i < 6; i++)
                    //     {
                    //         TV xi = rest_positions.segment<3>(i * 3);
                    //         out << "v " <<  xi.transpose() << std::endl;
                    //     }
                    //     out << "v " <<  mesh_centroid.transpose() << std::endl;
                    //     for (int i = 0; i < 6; i++)
                    //     {
                            
                    //         out << "f 1 2 3" << std::endl; 
                    //         out << "f 1 3 4" << std::endl; 
                    //         out << "f 1 4 6" << std::endl; 
                    //         out << "f 6 4 5" << std::endl; 
                    //         out << "f 2 1 7" << std::endl; 
                    //         out << "f 3 2 7" << std::endl; 
                    //         out << "f 4 3 7" << std::endl; 
                    //         out << "f 5 4 7" << std::endl; 
                    //         out << "f 6 5 7" << std::endl; 
                    //         out << "f 1 6 7" << std::endl; 
                    //     }
                    //     out.close();

                    //     out.open("trouble_tet.obj");
                    //     for (int i = 0; i < 6; i++)
                    //     {
                    //         TV xi = positions.segment<3>(i * 3);
                    //         out << "v " <<  xi.transpose() << std::endl;
                    //     }
                    //     out << "v " <<  mesh_centroid.transpose() << std::endl;
                    //     for (int i = 0; i < 6; i++)
                    //     {
                            
                    //         out << "f 1 2 3" << std::endl; 
                    //         out << "f 1 3 4" << std::endl; 
                    //         out << "f 1 4 6" << std::endl; 
                    //         out << "f 6 4 5" << std::endl; 
                    //         out << "f 2 1 7" << std::endl; 
                    //         out << "f 3 2 7" << std::endl; 
                    //         out << "f 4 3 7" << std::endl; 
                    //         out << "f 5 4 7" << std::endl; 
                    //         out << "f 6 5 7" << std::endl; 
                    //         out << "f 1 6 7" << std::endl; 
                    //     }
                    //     out.close();
                    //     std::getchar();
                    // }
                    addForceEntry<18>(residual, face_vtx_list, dedx);
                }
                else if (face_vtx_list.size() == 7)
                {
                    Vector<T, 21> dedx;
                    if (use_cell_centroid)
                        computeConeVolume7PointsGradient(positions, mesh_centroid, dedx);
                    dedx *= -coeff;
                    addForceEntry<21>(residual, face_vtx_list, dedx);
                }
                else if (face_vtx_list.size() == 8)
                {
                    Vector<T, 24> dedx;
                    if (use_cell_centroid)
                        computeConeVolume8PointsGradient(positions, mesh_centroid, dedx);
                    dedx *= -coeff;
                    addForceEntry<24>(residual, face_vtx_list, dedx);
                }
                else if (face_vtx_list.size() == 9)
                {
                    Vector<T, 27> dedx;
                    if (use_cell_centroid)
                        computeConeVolume9PointsGradient(positions, mesh_centroid, dedx);
                    dedx *= -coeff;
                    addForceEntry<27>(residual, face_vtx_list, dedx);
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
                        if (use_cell_centroid)
                            computeConeVolume4PointsGradient(positions, mesh_centroid, dedx);
                        else
                            computeQuadConeVolumeGradient(positions, mesh_centroid, dedx);
                        addForceEntry<12>(dVdx_full, face_vtx_list, dedx);
                    }
                    else if (face_vtx_list.size() == 5)
                    {
                        Vector<T, 15> dedx;
                        if (use_cell_centroid)
                            computeConeVolume5PointsGradient(positions, mesh_centroid, dedx);
                        else
                            computePentaConeVolumeGradient(positions, mesh_centroid, dedx);
                        addForceEntry<15>(dVdx_full, face_vtx_list, dedx);
                    }
                    else if (face_vtx_list.size() == 6)
                    {
                        Vector<T, 18> dedx;
                        if (use_cell_centroid)
                            computeConeVolume6PointsGradient(positions, mesh_centroid, dedx);
                        else
                            computeHexConeVolumeGradient(positions, mesh_centroid, dedx);
                        addForceEntry<18>(dVdx_full, face_vtx_list, dedx);
                    }
                    else if (face_vtx_list.size() == 7)
                    {
                        Vector<T, 21> dedx;
                        if (use_cell_centroid)
                            computeConeVolume7PointsGradient(positions, mesh_centroid, dedx);
                        addForceEntry<21>(dVdx_full, face_vtx_list, dedx);
                    }
                    else if (face_vtx_list.size() == 8)
                    {
                        Vector<T, 24> dedx;
                        if (use_cell_centroid)
                            computeConeVolume8PointsGradient(positions, mesh_centroid, dedx);
                        addForceEntry<24>(dVdx_full, face_vtx_list, dedx);
                    }
                    else if (face_vtx_list.size() == 9)
                    {
                        Vector<T, 27> dedx;
                        if (use_cell_centroid)
                            computeConeVolume9PointsGradient(positions, mesh_centroid, dedx);
                        addForceEntry<27>(dVdx_full, face_vtx_list, dedx);
                    }
                    else
                    {
                        std::cout << "unknown polygon edge number" << std::endl;
                    }
                    if (face_vtx_list.size() == 4)
                    {
                        
                        Matrix<T, 12, 12> d2Vdx2;
                        if (use_cell_centroid)
                            computeConeVolume4PointsHessian(positions, mesh_centroid, d2Vdx2);
                        else
                            computeQuadConeVolumeHessian(positions, mesh_centroid, d2Vdx2);
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
                        if (use_cell_centroid)
                            computeConeVolume5PointsHessian(positions, mesh_centroid, d2Vdx2);
                        else
                            computePentaConeVolumeHessian(positions, mesh_centroid, d2Vdx2);
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
                        if (use_cell_centroid)
                            computeConeVolume6PointsHessian(positions, mesh_centroid, d2Vdx2);
                        else
                            computeHexConeVolumeHessian(positions, mesh_centroid, d2Vdx2);
                        Matrix<T, 18, 18> hessian;
                        if (use_yolk_pressure)
                            hessian = -pressure_constant * d2Vdx2;
                        else
                            hessian = By * (yolk_vol_curr - yolk_vol_init) * d2Vdx2;
                        if(projectPD)
                            projectBlockPD<18>(hessian);
                        addHessianEntry<18>(entries, face_vtx_list, hessian);
                    }
                    else if (face_vtx_list.size() == 7)
                    {
                        Matrix<T, 21, 21> d2Vdx2;
                        if (use_cell_centroid)
                            computeConeVolume7PointsHessian(positions, mesh_centroid, d2Vdx2);
                        Matrix<T, 21, 21> hessian;
                        hessian = By * (yolk_vol_curr - yolk_vol_init) * d2Vdx2;
                        if(projectPD)
                            projectBlockPD<21>(hessian);
                        addHessianEntry<21>(entries, face_vtx_list, hessian);
                    }
                    else if (face_vtx_list.size() == 8)
                    {
                        Matrix<T, 24, 24> d2Vdx2;
                        if (use_cell_centroid)
                            computeConeVolume8PointsHessian(positions, mesh_centroid, d2Vdx2);
                        Matrix<T, 24, 24> hessian;
                        hessian = By * (yolk_vol_curr - yolk_vol_init) * d2Vdx2;
                        if(projectPD)
                            projectBlockPD<24>(hessian);
                        addHessianEntry<24>(entries, face_vtx_list, hessian);
                    }
                    else if (face_vtx_list.size() == 9)
                    {
                        Matrix<T, 27, 27> d2Vdx2;
                        if (use_cell_centroid)
                            computeConeVolume9PointsHessian(positions, mesh_centroid, d2Vdx2);
                        Matrix<T, 27, 27> hessian;
                        hessian = By * (yolk_vol_curr - yolk_vol_init) * d2Vdx2;
                        if(projectPD)
                            projectBlockPD<27>(hessian);
                        addHessianEntry<27>(entries, face_vtx_list, hessian);
                    }
                    else
                    {
                        std::cout << "unknown polygon edge case" << std::endl;
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
                    if (hessian_partial.nonZeros() > 0)
                        addHessianBlock<3>(entries, {dof_i, dof_j}, hessian_partial);
                }
            }
        }
        
    }

    // iterateFaceSerial([&](VtxList& face_vtx_list, int face_idx)
    // {
    //     if (add_yolk_volume)
    //     {
    //         if (face_idx < lateral_face_start && face_idx >= basal_face_start)
    //         {
                
    //             VectorXT positions;
    //             positionsFromIndices(positions, face_vtx_list);
                
    //             if (face_vtx_list.size() == 4)
    //             {
                    
    //                 Matrix<T, 12, 12> d2Vdx2;
    //                 if (use_cell_centroid)
    //                     computeConeVolume4PointsHessian(positions, mesh_centroid, d2Vdx2);
    //                 else
    //                     computeQuadConeVolumeHessian(positions, mesh_centroid, d2Vdx2);
    //                 Matrix<T, 12, 12> hessian;
    //                 if (use_yolk_pressure)
    //                     hessian = -pressure_constant * d2Vdx2;
    //                 else
    //                     hessian = By * (yolk_vol_curr - yolk_vol_init) * d2Vdx2;
    //                 if(projectPD)
    //                     projectBlockPD<12>(hessian);
    //                 addHessianEntry<12>(entries, face_vtx_list, hessian);
    //             }
    //             else if (face_vtx_list.size() == 5)
    //             {
    //                 Matrix<T, 15, 15> d2Vdx2;
    //                 if (use_cell_centroid)
    //                     computeConeVolume5PointsHessian(positions, mesh_centroid, d2Vdx2);
    //                 else
    //                     computePentaConeVolumeHessian(positions, mesh_centroid, d2Vdx2);
    //                 Matrix<T, 15, 15> hessian;
    //                 if (use_yolk_pressure)
    //                     hessian = -pressure_constant * d2Vdx2;
    //                 else
    //                     hessian = By * (yolk_vol_curr - yolk_vol_init) * d2Vdx2;
    //                 if(projectPD)
    //                     projectBlockPD<15>(hessian);
    //                 addHessianEntry<15>(entries, face_vtx_list, hessian);

    //             }
    //             else if (face_vtx_list.size() == 6)
    //             {
    //                 Matrix<T, 18, 18> d2Vdx2;
    //                 if (use_cell_centroid)
    //                     computeConeVolume6PointsHessian(positions, mesh_centroid, d2Vdx2);
    //                 else
    //                     computeHexConeVolumeHessian(positions, mesh_centroid, d2Vdx2);
    //                 Matrix<T, 18, 18> hessian;
    //                 if (use_yolk_pressure)
    //                     hessian = -pressure_constant * d2Vdx2;
    //                 else
    //                     hessian = By * (yolk_vol_curr - yolk_vol_init) * d2Vdx2;
    //                 if(projectPD)
    //                     projectBlockPD<18>(hessian);
    //                 addHessianEntry<18>(entries, face_vtx_list, hessian);
    //             }
    //             else if (face_vtx_list.size() == 7)
    //             {
    //                 Matrix<T, 21, 21> d2Vdx2;
    //                 if (use_cell_centroid)
    //                     computeConeVolume7PointsHessian(positions, mesh_centroid, d2Vdx2);
    //                 Matrix<T, 21, 21> hessian;
    //                 hessian = By * (yolk_vol_curr - yolk_vol_init) * d2Vdx2;
    //                 if(projectPD)
    //                     projectBlockPD<21>(hessian);
    //                 addHessianEntry<21>(entries, face_vtx_list, hessian);
    //             }
    //             else if (face_vtx_list.size() == 8)
    //             {
    //                 Matrix<T, 24, 24> d2Vdx2;
    //                 if (use_cell_centroid)
    //                     computeConeVolume8PointsHessian(positions, mesh_centroid, d2Vdx2);
    //                 Matrix<T, 24, 24> hessian;
    //                 hessian = By * (yolk_vol_curr - yolk_vol_init) * d2Vdx2;
    //                 if(projectPD)
    //                     projectBlockPD<24>(hessian);
    //                 addHessianEntry<24>(entries, face_vtx_list, hessian);
    //             }
    //             else
    //             {
    //                 std::cout << "unknown polygon edge case" << std::endl;
    //             }
    //         }
        // }

        
    // });
}