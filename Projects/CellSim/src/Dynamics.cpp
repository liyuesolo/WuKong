#include "../include/VertexModel.h"

void VertexModel::computeNodalMass()
{
    vtx_mass.resize(num_nodes);
    vtx_mass.setZero();
    if (!add_mass)
    {
        vtx_mass.setOnes();
        return;
    }

    VectorXT current_cell_volume;
    computeVolumeAllCells(current_cell_volume);
    
    iterateCellSerial([&](VtxList& face_vtx_list, int cell_idx)
    {
        VtxList cell_vtx_list = face_vtx_list;
        for (int idx : face_vtx_list)
            cell_vtx_list.push_back(idx + basal_vtx_start);
        for (int idx : cell_vtx_list)
        {
            vtx_mass[idx] += density * current_cell_volume[cell_idx] / T(cell_vtx_list.size());
        }
    });
}

void VertexModel::addInertialEnergy(T& energy)
{
    VectorXT energies(num_nodes);
    energies.setZero();
    tbb::parallel_for(0, num_nodes, [&](int i) {   
        for (int d = 0; d < 3; d++)
        {
            // energies[i] = 0.5 * vtx_mass[i] * std::pow(deformed[i * 3 + d] - undeformed[i * 3 + d] - dt * vtx_vel[i * 3 + d], 2);
            energies[i] += 1.0 / (2.0 * eta * dt) * vtx_mass[i] * std::pow(deformed[i * 3 + d] - undeformed[i * 3 + d], 2);
        }
    });
    energy += energies.sum();
}

void VertexModel::addInertialForceEntries(VectorXT& residual)
{
    for (int i = 0; i < num_nodes; i++)
    {
        for (int d = 0; d < 3; d++)
        {
            // residual[i * 3 + d] += vtx_mass[i] * (deformed[i * 3 + d] - undeformed[i * 3 + d] - dt * vtx_vel[i * 3 + d]);
            residual[i * 3 + d] -= 1.0 / (eta * dt) *  vtx_mass[i] * (deformed[i * 3 + d] - undeformed[i * 3 + d]);
        }
    }
    
}

void VertexModel::addInertialHessianEntries(std::vector<Entry>& entires)
{
    for (int i = 0; i < num_nodes; i++)
    {
        for (int d = 0; d < 3; d++)
        {
            entires.push_back(Entry(i * 3 + d, i * 3 + d, 1.0 / (eta * dt) * vtx_mass[i]));
        }
    }
}

