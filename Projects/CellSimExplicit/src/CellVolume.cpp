#include "../include/CellSim.h"
#include "../include/autodiff/CellEnergy.h"
void CellSim::computeVolumeAllCells(VectorXT& cell_volume_list)
{
    cell_volume_list = VectorXT::Zero(num_cells);
    iterateCellParallel([&](VtxList& cell_vtx_list, int cell_idx)
    {
        VectorXT positions;
        positionsFromIndices(positions, cell_vtx_list);
        int n_vtx_base = cell_vtx_list.size() / 2;
        if (n_vtx_base == 4)
            computeVolume4Points(positions, cell_volume_list[cell_idx]);
        else if (n_vtx_base == 5)
            computeVolume5Points(positions, cell_volume_list[cell_idx]);
        else if (n_vtx_base == 6)
            computeVolume6Points(positions, cell_volume_list[cell_idx]);
        else if (n_vtx_base == 7)
            computeVolume7Points(positions, cell_volume_list[cell_idx]);
        else if (n_vtx_base == 8)
            computeVolume8Points(positions, cell_volume_list[cell_idx]);
        else if (n_vtx_base == 9)
            computeVolume9Points(positions, cell_volume_list[cell_idx]);
    });
}
void CellSim::addCellVolumePreservationEnergy(T& energy)
{
    VectorXT current_cell_volume;
    computeVolumeAllCells(current_cell_volume);

    T volume_term = 0.0;
    iterateCellSerial([&](VtxList& cell_vtx_list, int cell_idx)
    {
        VectorXT positions;
        positionsFromIndices(positions, cell_vtx_list);
        T ci = current_cell_volume[cell_idx] - cell_volume_init[cell_idx];
        volume_term += 0.5 * B * std::pow(ci, 2);
    });
    energy += volume_term;
}
void CellSim::addCellVolumePreservationForceEntries(VectorXT& residual)
{
    VectorXT current_cell_volume;
    computeVolumeAllCells(current_cell_volume);

    iterateCellSerial([&](VtxList& cell_vtx_list, int cell_idx)
    {
        int n_vtx_base = cell_vtx_list.size() / 2;
        VectorXT positions;
        positionsFromIndices(positions, cell_vtx_list);
        T ci = current_cell_volume[cell_idx] - cell_volume_init[cell_idx];
        if (n_vtx_base == 4)
        {
            Vector<T, 24> dedx;
            computeVolume4PointsGradient(positions, dedx);
            dedx *= -B * ci;
            addForceEntry<24>(residual, cell_vtx_list, dedx);
            
        }
        else if (n_vtx_base == 5)
        {
            Vector<T, 30> dedx;
            computeVolume5PointsGradient(positions, dedx);
            dedx *= -B * ci;
            addForceEntry<30>(residual, cell_vtx_list, dedx);
        }
        else if (n_vtx_base == 6)
        {
            Vector<T, 36> dedx;
            computeVolume6PointsGradient(positions, dedx);
            dedx *= -B * ci;
            addForceEntry<36>(residual, cell_vtx_list, dedx);
        }
        else if (n_vtx_base == 7)
        {
            Vector<T, 42> dedx;
            computeVolume7PointsGradient(positions, dedx);
            dedx *= -B * ci;
            addForceEntry<42>(residual, cell_vtx_list, dedx);
        }
        else if (n_vtx_base == 8)
        {
            Vector<T, 48> dedx;
            computeVolume8PointsGradient(positions, dedx);
            dedx *= -B * ci;
            addForceEntry<48>(residual, cell_vtx_list, dedx);
        }
        else if (n_vtx_base == 9)
        {
            Vector<T, 54> dedx;
            computeVolume9PointsGradient(positions, dedx);
            dedx *= -B * ci;
            addForceEntry<54>(residual, cell_vtx_list, dedx);
        }
        else
        {
            // std::cout << "unknown polygon edge case" << std::endl;
        }
    });
}

void CellSim::addCellVolumePreservationHessianEntries(std::vector<Entry>& entries, 
    bool projectPD)
{
    VectorXT current_cell_volume;
    computeVolumeAllCells(current_cell_volume);
    iterateCellSerial([&](VtxList& cell_vtx_list, int cell_idx)
    {
        int n_vtx_base = cell_vtx_list.size() / 2;
        VectorXT positions;
        positionsFromIndices(positions, cell_vtx_list);
        T V = current_cell_volume[cell_idx];
        if (n_vtx_base == 4)
        {
            
            Matrix<T, 24, 24> d2Vdx2;
            computeVolume4PointsHessian(positions, d2Vdx2);
            Vector<T, 24> dVdx;
            computeVolume4PointsGradient(positions, dVdx);
            Matrix<T, 24, 24> hessian;
            hessian.setZero();
            hessian += B * dVdx * dVdx.transpose();
            hessian += B * (V - cell_volume_init[cell_idx]) * d2Vdx2;
            if(projectPD)
                projectBlockPD<24>(hessian);
            
            addHessianEntry<24>(entries, cell_vtx_list, hessian);            
        }
        else if (n_vtx_base == 5)
        {
            Matrix<T, 30, 30> d2Vdx2;
            computeVolume5PointsHessian(positions, d2Vdx2);
            Vector<T, 30> dVdx;
            computeVolume5PointsGradient(positions, dVdx);
                
            Matrix<T, 30, 30> hessian;
            hessian.setZero();
            hessian += B * dVdx * dVdx.transpose();
            hessian += B * (V - cell_volume_init[cell_idx]) * d2Vdx2;
            if(projectPD)
                projectBlockPD<30>(hessian);
            
            addHessianEntry<30>(entries, cell_vtx_list, hessian);
        }
        else if (n_vtx_base == 6)
        {
            Matrix<T, 36, 36> d2Vdx2;
            computeVolume6PointsHessian(positions, d2Vdx2);
            Vector<T, 36> dVdx;
            computeVolume6PointsGradient(positions, dVdx);

            Matrix<T, 36, 36> hessian;
            hessian.setZero();
            hessian += B * dVdx * dVdx.transpose();
            hessian += B * (V - cell_volume_init[cell_idx]) * d2Vdx2;
            if(projectPD)
                projectBlockPD<36>(hessian);
            
            addHessianEntry<36>(entries, cell_vtx_list, hessian);
        }
        else if (n_vtx_base == 7)
        {
            Matrix<T, 42, 42> d2Vdx2;
            computeVolume7PointsHessian(positions, d2Vdx2);
            Vector<T, 42> dVdx;
            computeVolume7PointsGradient(positions, dVdx);
            
            Matrix<T, 42, 42> hessian;
            hessian.setZero();

            hessian += B * dVdx * dVdx.transpose();
            hessian += B * (V - cell_volume_init[cell_idx]) * d2Vdx2;
            if(projectPD)
                projectBlockPD<42>(hessian);
            addHessianEntry<42>(entries, cell_vtx_list, hessian);
            
        }
        else if (n_vtx_base == 8)
        {
            Matrix<T, 48, 48> d2Vdx2;
            computeVolume8PointsHessian(positions, d2Vdx2);
            
            Vector<T, 48> dVdx;
            computeVolume8PointsGradient(positions, dVdx);
            
            Matrix<T, 48, 48> hessian;
            hessian.setZero();
            
            hessian += B * dVdx * dVdx.transpose();
            hessian += B * (V - cell_volume_init[cell_idx]) * d2Vdx2;
            
            if(projectPD)
                projectBlockPD<48>(hessian);
            
            addHessianEntry<48>(entries, cell_vtx_list, hessian);
        }
        else if (n_vtx_base == 9)
        {
            Matrix<T, 54, 54> d2Vdx2;
            computeVolume9PointsHessian(positions, d2Vdx2);
            Vector<T, 54> dVdx;
            computeVolume9PointsGradient(positions, dVdx);
            
            // break it down here to avoid super long autodiff code
            Matrix<T, 54, 54> hessian;
            hessian.setZero();
            hessian += B * dVdx * dVdx.transpose();
            hessian += B * (V - cell_volume_init[cell_idx]) * d2Vdx2;            
            if(projectPD)
                projectBlockPD<54>(hessian);

            addHessianEntry<54>(entries, cell_vtx_list, hessian);
        }
        else
        {
            // std::cout << "unknown polygon edge case" << std::endl;
        }
    });
}