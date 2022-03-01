#include "../../include/Objectives.h"
#include <Eigen/PardisoSupport>
#include "../../include/DataIO.h"
#include "../../icp/simpleicp.h"

T ObjFindInit::value(const VectorXT& p_curr, bool simulate, bool use_prev_equil)
{
    updateDesignParameters(p_curr);
    T energy = 0.0;
    iterateTargets([&](int cell_idx, TV& target_pos)
    {
        VtxList face_vtx_list = simulation.cells.faces[cell_idx];
        TV centroid;
        simulation.cells.computeCellCentroid(face_vtx_list, centroid);
        energy += 0.5 * (centroid - target_pos).dot(centroid - target_pos);
    });
    
    simulation.cells.addSingleTetVolBarrierEnergy(energy);

    return energy;
}

T ObjFindInit::gradient(const VectorXT& p_curr, VectorXT& dOdp, T& energy, bool simulate)
{
    updateDesignParameters(p_curr);
    
    energy = 0.0;
    VectorXT dOdx(n_dof_sim);
    dOdx.setZero();

    iterateTargets([&](int cell_idx, TV& target_pos)
    {
        
        VtxList face_vtx_list = simulation.cells.faces[cell_idx];
        VtxList cell_vtx_list = face_vtx_list;
        for (int idx : face_vtx_list)
            cell_vtx_list.push_back(idx + simulation.cells.basal_vtx_start);
        TV centroid;
        simulation.cells.computeCellCentroid(face_vtx_list, centroid);
        energy += 0.5 * (centroid - target_pos).dot(centroid - target_pos);
        T coeff = cell_vtx_list.size();
        for (int idx : cell_vtx_list)
        {
            dOdx.segment<3>(idx * 3) += (centroid - target_pos) / coeff;
        }
    });

    simulation.cells.addSingleTetVolBarrierEnergy(energy);
    VectorXT vol_barrier_force(n_dof_sim); vol_barrier_force.setZero();
    simulation.cells.addSingleTetVolBarrierForceEntries(vol_barrier_force);
    dOdx += -vol_barrier_force;

    dOdp = dOdx;
}

void ObjFindInit::hessian(const VectorXT& p_curr, StiffnessMatrix& H, bool simulate)
{
    std::vector<Entry> entries;

    iterateTargets([&](int cell_idx, TV& target_pos)
    {
        VtxList face_vtx_list = simulation.cells.faces[cell_idx];
        VtxList cell_vtx_list = face_vtx_list;
        for (int idx : face_vtx_list)
            cell_vtx_list.push_back(idx + simulation.cells.basal_vtx_start);
        TV centroid;
        simulation.cells.computeCellCentroid(face_vtx_list, centroid);
        T coeff = cell_vtx_list.size();
        for (int idx_i : cell_vtx_list)
            for (int idx_j : cell_vtx_list)
                for (int d = 0; d < 3; d++)
                    entries.push_back(Entry(idx_i * 3 + d, idx_j * 3 + d, 1.0 / coeff / coeff));
    });

    simulation.cells.addSingleTetVolBarrierHessianEntries(entries);

    H.resize(n_dof_design, n_dof_design);
    H.setFromTriplets(entries.begin(), entries.end());
    H.makeCompressed();
}


void ObjFindInit::updateDesignParameters(const VectorXT& design_parameters)
{
    simulation.deformed = design_parameters;
}

void ObjFindInit::getDesignParameters(VectorXT& design_parameters)
{
    design_parameters = simulation.deformed;
}

void ObjFindInit::getSimulationAndDesignDoF(int& _sim_dof, int& _design_dof)
{
    _sim_dof = simulation.num_nodes * 3;
    _design_dof = simulation.num_nodes * 3;
    setSimulationAndDesignDoF(_sim_dof, _design_dof);
}



bool ObjFindInit::getTargetTrajectoryFrame(VectorXT& frame_data)
{
    if (cell_trajectories.rows() == 0)
    {
        std::cout << "load cell trajectory first" << std::endl;
        return false;
    }
    if (frame > cell_trajectories.cols())
    {
        std::cout << "frame exceed " << cell_trajectories.cols() << std::endl;
        return false;
    }
    frame_data = cell_trajectories.col(frame);
    int n_pt = frame_data.rows() / 3;
    Matrix<T, 3, 3> R;
    R << 0.960277, -0.201389, 0.229468, 0.2908, 0.871897, -0.519003, -0.112462, 0.558021, 0.887263;
    Matrix<T, 3, 3> R2 = Eigen::AngleAxis<T>(0.20 * M_PI + 0.5 * M_PI, TV(-1.0, 0.0, 0.0)).toRotationMatrix();

    for (int i = 0; i < n_pt; i++)
    {
        TV pos = frame_data.segment<3>(i * 3);
        TV updated = (pos - TV(605.877,328.32,319.752)) / 1096.61;
        updated = R2 * R * updated;
        frame_data.segment<3>(i * 3) = updated * 0.9 * simulation.cells.unit; 
    }
    
    return true;
}

void ObjFindInit::loadTargetTrajectory(const std::string& filename)
{
    DataIO data_io;
    data_io.loadTrajectories(filename, cell_trajectories);
}

void ObjFindInit::updateTarget()
{
    target_positions.clear();
    VectorXT cell_centroids;
    simulation.cells.getAllCellCentroids(cell_centroids);
    VectorXT data_points;
    bool success = getTargetTrajectoryFrame(data_points);

    TV min_corner, max_corner;
    simulation.cells.computeBoundingBox(min_corner, max_corner);
    T spacing = 0.05 * (max_corner - min_corner).norm();

    T max_dis = 0.05 * (max_corner - min_corner).norm();
    bool inverse = true;
    std::vector<std::pair<int, int>> pairs;
    
    if (inverse)
    {
        std::vector<bool> visited(data_points.rows() / 3, false);
        hash.build(spacing, data_points);

        for (int i = 0; i < cell_centroids.rows() / 3; i++)
        {
            std::vector<int> neighbors;
            TV current = cell_centroids.segment<3>(i * 3);
            hash.getOneRingNeighbors(current, neighbors);
            T min_dis = 1e6;
            int min_dis_pt = -1;
            for (int idx : neighbors)
            {
                TV neighbor = data_points.segment<3>(idx * 3);
                
                T dis = (current - neighbor).norm();
                
                if (dis < min_dis)
                {
                    min_dis = dis;
                    min_dis_pt = idx;
                }
            }
            if (min_dis_pt != -1 && min_dis < max_dis && !visited[min_dis_pt])
            {
                target_positions[i] = data_points.segment<3>(min_dis_pt * 3);
                pairs.push_back(std::make_pair(i, min_dis_pt));
                visited[min_dis_pt] = true;
            }
        }   
    }
    else
    {
        std::vector<bool> visited(cell_centroids.rows() / 3, false);
        hash.build(spacing, cell_centroids);
        for (int i = 0; i < data_points.rows() / 3; i++)
        {
            std::vector<int> neighbors;
            TV current = data_points.segment<3>(i * 3);
            hash.getOneRingNeighbors(current, neighbors);
            T min_dis = 1e6;
            int min_dis_pt = -1;
            for (int idx : neighbors)
            {
                TV neighbor = cell_centroids.segment<3>(idx * 3);
                
                T dis = (current - neighbor).norm();
                
                if (dis < min_dis)
                {
                    min_dis = dis;
                    min_dis_pt = idx;
                }
            }
            if (min_dis_pt != -1 && min_dis < max_dis && !visited[i])
            {
                target_positions[min_dis_pt] = current;
                pairs.push_back(std::make_pair(min_dis_pt, i));
                visited[i] = true;
            }
        }   
    }
    // std::ofstream out("idx_map.txt");
    // for (auto pair : pairs)
    //     out << pair.first << " " << pair.second << std::endl;
    // out.close();
}

T ObjFindInit::maximumStepSize(const VectorXT& dp)
{
    VectorXT dummy = VectorXT::Zero(dp.rows());
    return simulation.cells.computeInversionFreeStepSize(dummy, dp);

}