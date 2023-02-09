
#include "../include/CellSim.h"
#include "../include/autodiff/YolkEnergy.h"
#include <fstream>

// bool use_cell_centroid = false;

// T CellSim::computeYolkInversionFreeStepSize(const VectorXT& _u, const VectorXT& du)
// {
//     T step_size = 1.0;
//     while (true)
//     {
//         deformed = undeformed + _u + step_size * du;
        
//         bool constraint_violated = false;
//         if (use_cell_centroid)
//         {
//             iterateFaceSerial([&](VtxList& face_vtx_list, int face_idx)
//             {
//                 if (face_idx < lateral_face_start && face_idx >= basal_face_start)
//                 {    
//                     VectorXT positions;
//                     positionsFromIndices(positions, face_vtx_list);
//                     TV basal_centroid = TV::Zero();
//                     for (int i = 0; i < face_vtx_list.size(); i++)
//                         basal_centroid += positions.segment<3>(i * 3);
//                     for (int i = 0; i < face_vtx_list.size(); i++)
//                     {
//                         int j = (i + 1) % face_vtx_list.size();
//                         TV r0 = positions.segment<3>(i * 3);
//                         TV r1 = positions.segment<3>(j * 3);
//                         if (computeTetVolume(basal_centroid, r1, r0, mesh_centroid) < 1e-8)
//                         {
//                             constraint_violated = true;
//                             break;
//                         }
//                     }
//                     if (constraint_violated)
//                         return;
//                 }
//             });
//         }
//         else
//             iterateFixedYolkTetsSerial([&](TM3& x_deformed, VtxList& indices)
//             {
//                 if (constraint_violated)
//                     return;
//                 T d = computeTetVolume(mesh_centroid, x_deformed.col(0), x_deformed.col(1), x_deformed.col(2));
//                 if (d < 1e-6)
//                 {
//                     // std::cout << d << std::endl;
//                     constraint_violated = true;
//                     return;
//                 }
//             });
//         if (constraint_violated)
//             step_size *= 0.8;
//         else
//             return step_size;
//         // if (step_size < 1e-6)
//         // {
//         //     saveLowVolumeTets("low_vol_tet.obj");
//         //     saveBasalSurfaceMesh("low_vol_tet_basal_surface.obj");

//         //     ÷std::exit(0);
//         // }
//     }
// }


T CellSim::computeYolkVolume()
{
    VectorXT volumes = VectorXT::Zero(yolk_cells.size());
    iterateYolkFaceParallel([&](VtxList& face_vtx_list, int face_idx)
    {
        VectorXT positions;
        positionsFromIndices(positions, face_vtx_list);
        if (face_vtx_list.size() == 4) 
            computeConeVolume4Points(positions, mesh_centroid, volumes[face_idx]);
        else if (face_vtx_list.size() == 5) 
            computeConeVolume5Points(positions, mesh_centroid, volumes[face_idx]);
        else if (face_vtx_list.size() == 6) 
            computeConeVolume6Points(positions, mesh_centroid, volumes[face_idx]);
        else if (face_vtx_list.size() == 7) 
            computeConeVolume7Points(positions, mesh_centroid, volumes[face_idx]);
        else if (face_vtx_list.size() == 8) 
            computeConeVolume8Points(positions, mesh_centroid, volumes[face_idx]);
        else if (face_vtx_list.size() == 9) 
            computeConeVolume9Points(positions, mesh_centroid, volumes[face_idx]);
    });
    return volumes.sum();
}

void CellSim::addYolkVolumePreservationEnergy(T& energy)
{
    T yolk_vol_curr = computeYolkVolume();
    energy +=  0.5 * By * std::pow(yolk_vol_curr - yolk_vol_init, 2);
}

void CellSim::addYolkVolumePreservationForceEntries(VectorXT& residual)
{
    T yolk_vol_curr = computeYolkVolume();

    iterateYolkFaceSerial([&](VtxList& face_vtx_list, int face_idx)
    {
        VectorXT positions;
        positionsFromIndices(positions, face_vtx_list);
        T coeff = By * (yolk_vol_curr - yolk_vol_init);
        if (face_vtx_list.size() == 4)
        {
            Vector<T, 12> dedx;
            computeConeVolume4PointsGradient(positions, mesh_centroid, dedx);
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
        else if (face_vtx_list.size() == 7)
        {
            Vector<T, 21> dedx;
            computeConeVolume7PointsGradient(positions, mesh_centroid, dedx);
            dedx *= -coeff;
            addForceEntry<21>(residual, face_vtx_list, dedx);
        }
        else if (face_vtx_list.size() == 8)
        {
            Vector<T, 24> dedx;
            computeConeVolume8PointsGradient(positions, mesh_centroid, dedx);
            dedx *= -coeff;
            addForceEntry<24>(residual, face_vtx_list, dedx);
        }
        else if (face_vtx_list.size() == 9)
        {
            Vector<T, 27> dedx;
            computeConeVolume9PointsGradient(positions, mesh_centroid, dedx);
            dedx *= -coeff;
            addForceEntry<27>(residual, face_vtx_list, dedx);
        }
        else
        {
            std::cout << "unknown polygon edge number" << std::endl;
        }
    });
}

void CellSim::addYolkVolumePreservationHessianEntries(std::vector<Entry>& entries,
    MatrixXT& WoodBuryMatrix, bool projectPD)
{
    T yolk_vol_curr = computeYolkVolume();

    VectorXT dVdx_full = VectorXT::Zero(deformed.rows());

    iterateYolkFaceSerial([&](VtxList& face_vtx_list, int face_idx)
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
        else if (face_vtx_list.size() == 7)
        {
            Vector<T, 21> dedx;
            computeConeVolume7PointsGradient(positions, mesh_centroid, dedx);
            addForceEntry<21>(dVdx_full, face_vtx_list, dedx);
        }
        else if (face_vtx_list.size() == 8)
        {
            Vector<T, 24> dedx;
            computeConeVolume8PointsGradient(positions, mesh_centroid, dedx);
            addForceEntry<24>(dVdx_full, face_vtx_list, dedx);
        }
        else if (face_vtx_list.size() == 9)
        {
            Vector<T, 27> dedx;
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
            computeConeVolume4PointsHessian(positions, mesh_centroid, d2Vdx2);
            Matrix<T, 12, 12> hessian;
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
            hessian = By * (yolk_vol_curr - yolk_vol_init) * d2Vdx2;
            if(projectPD)
                projectBlockPD<18>(hessian);
            addHessianEntry<18>(entries, face_vtx_list, hessian);
        }
        else if (face_vtx_list.size() == 7)
        {
            Matrix<T, 21, 21> d2Vdx2;
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
        for (int dof_i = yolk_vtx_start; dof_i < num_nodes; dof_i++)
        {
            for (int dof_j = yolk_vtx_start; dof_j < num_nodes; dof_j++)
            {
                Vector<T, 6> dVdx;
                getSubVector<6>(dVdx_full, {dof_i, dof_j}, dVdx);
                TV dVdxi = dVdx.segment<3>(0);
                TV dVdxj = dVdx.segment<3>(3);
                Matrix<T, 3, 3> hessian_partial = By * dVdxi * dVdxj.transpose();
                // if (hessian_partial.nonZeros() > 0)
                addHessianBlock<3>(entries, {dof_i, dof_j}, hessian_partial);
            }
        }
    }
    
}