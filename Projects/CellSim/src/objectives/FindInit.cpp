#include "../../include/Objectives.h"
#include <Eigen/PardisoSupport>
#include "../../include/DataIO.h"
#include "../../icp/simpleicp.h"

T ObjFindInit::value(const VectorXT& p_curr, bool use_prev_equil)
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
    
    return energy;
}

T ObjFindInit::gradient(const VectorXT& p_curr, VectorXT& dOdp, bool use_prev_equil)
{
    updateDesignParameters(p_curr);

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

        T coeff = cell_vtx_list.size();
        for (int idx : cell_vtx_list)
        {
            dOdx.segment<3>(idx * 3) += (centroid - target_pos) / coeff;
        }
    });

    dOdp = dOdx;
}

T ObjFindInit::gradient(const VectorXT& p_curr, VectorXT& dOdp, T& energy, bool use_prev_equil)
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

    dOdp = dOdx;
}

T ObjFindInit::hessian(const VectorXT& p_curr, StiffnessMatrix& H, bool use_prev_equil)
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

    H.resize(n_dof_design, n_dof_design);
    H.setFromTriplets(entries.begin(), entries.end());
    H.makeCompressed();
}

T ObjFindInit::evaluteGradientAndEnergy(const VectorXT& p_curr, VectorXT& dOdp, T& energy)
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

    dOdp = dOdx;
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
        frame_data.segment<3>(i * 3) = updated * 0.8 * simulation.cells.unit; 
    }
    
    return true;
}

void ObjFindInit::loadTargetTrajectory(const std::string& frame)
{
    DataIO data_io;
    data_io.loadTrajectories(frame, cell_trajectories);
}

void ObjFindInit::updateTarget()
{
    target_positions.clear();
    VectorXT cell_centroids;
    simulation.cells.getAllCellCentroids(cell_centroids);
    VectorXT data_points;
    bool success = getTargetTrajectoryFrame(data_points);

    MatrixXT query = Eigen::Map<MatrixXT>(cell_centroids.data(), cell_centroids.rows() / 3, 3);
    MatrixXT fixed = Eigen::Map<MatrixXT>(data_points.data(), data_points.rows() / 3, 3);

    VectorXi closest_indices = KnnSearch(fixed, query);
    // std::cout << query.rows() << " " << closest_indices.rows() << std::endl;
    // std::cout << closest_indices.transpose() << std::endl;
    // std::exit(0);
    for (int i = 0; i < closest_indices.rows(); i++)
    {
        target_positions[i] = data_points.segment<3>(closest_indices[i] * 3);
    }
}