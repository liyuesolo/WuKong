#include "../../include/Objectives.h"
#include <Eigen/PardisoSupport>
#include "../../include/DataIO.h"
#include "../../include/LinearSolver.h"

void ObjNucleiTracking::initializeTarget()
{
    // std::vector<int> VF_cell_idx;
    // simulation.cells.getVFCellIds(VF_cell_idx);
    // // std::cout << "VF cells " << VF_cell_idx.size() << std::endl;
    // for (int idx : VF_cell_idx)
    // {
    //     TV cell_centroid;
    //     VtxList face_vtx_list = simulation.cells.faces[idx];
    //     simulation.cells.computeCellCentroid(face_vtx_list, cell_centroid);
    //     target_positions[idx] = cell_centroid;
    // }
    for (int i = 0; i < simulation.cells.basal_face_start; i++)
    {
        target_positions[i] = TV::Zero();
    }
    
}

void ObjNucleiTracking::loadTarget(const std::string& filename)
{
    std::vector<int> VF_cell_idx;
    simulation.cells.getVFCellIds(VF_cell_idx);
    std::ifstream in(filename);
    int idx; T x, y, z;
    while(in >> idx >> x >> y >> z)
    {
        // if (std::find(VF_cell_idx.begin(), VF_cell_idx.end(), idx) != VF_cell_idx.end())
            target_positions[idx] = TV(x, y, z);
    }
    in.close();
}

void ObjNucleiTracking::d2Odx2(const VectorXT& p_curr, std::vector<Entry>& d2Odx2_entries)
{
    updateDesignParameters(p_curr);
    iterateTargets([&](int cell_idx, TV& target_pos)
    {
        VtxList face_vtx_list = simulation.cells.faces[cell_idx];
        VtxList cell_vtx_list = face_vtx_list;
        for (int idx : face_vtx_list)
            cell_vtx_list.push_back(idx + simulation.cells.basal_vtx_start);
        
        T coeff = cell_vtx_list.size();
        for (int idx_i : cell_vtx_list)
            for (int idx_j : cell_vtx_list)
                for (int d = 0; d < 3; d++)
                    d2Odx2_entries.push_back(Entry(idx_i * 3 + d, idx_j * 3 + d, 1.0 / coeff / coeff));
    });
}

void ObjNucleiTracking::dOdx(const VectorXT& p_curr, VectorXT& _dOdx)
{
    updateDesignParameters(p_curr);
    
    _dOdx.resize(n_dof_sim);
    _dOdx.setZero();

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
            _dOdx.segment<3>(idx * 3) += (centroid - target_pos) / coeff;
        }
    });
}


T ObjNucleiTracking::value(const VectorXT& p_curr, bool simulate, bool use_prev_equil)
{
    // simulation.loadDeformedState("current_mesh.obj");
    updateDesignParameters(p_curr);
    if (simulate)
    {
        simulation.reset();
        if (use_prev_equil)
            simulation.u = equilibrium_prev;
        simulation.staticSolve();
    }

    T energy = 0.0;
    iterateTargets([&](int cell_idx, TV& target_pos)
    {
        VtxList face_vtx_list = simulation.cells.faces[cell_idx];
        TV centroid;
        simulation.cells.computeCellCentroid(face_vtx_list, centroid);
        energy += 0.5 * (centroid - target_pos).dot(centroid - target_pos);
    });
    if (use_log_barrier)
        for (int i = 0; i < n_dof_design; i++)
            if (p_curr[i] < barrier_distance)
                energy += barrier_weight * barrier<0>(p_curr[i], barrier_distance);
    
    if (add_min_act)
    {
        for (int i = 0; i < n_dof_design; i++)
        {
            energy += w_min_act * std::abs(p_curr[i]);
        }
    }
        
    return energy;
}

T ObjNucleiTracking::gradient(const VectorXT& p_curr, VectorXT& dOdp, T& energy, bool simulate)
{
    updateDesignParameters(p_curr);
    if (simulate)
    {
        simulation.reset();
        simulation.staticSolve();
    }
    
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
    
    
    StiffnessMatrix d2edx2(n_dof_sim, n_dof_sim);
    VectorXT lambda;
    // here d2e/dx2 is -df/dx, negative sign is cancelled with -df/dp
    if (simulation.woodbury)
    {
        MatrixXT UV;
        simulation.buildSystemMatrixWoodbury(simulation.u, d2edx2, UV);
        PardisoLLTSolver solver(d2edx2, UV);
        solver.solve(dOdx, lambda);
        // LinearSolver::WoodburySolve(d2edx2, UV, dOdx, lambda, true, false, false);
    }
    else
    {   
        simulation.buildSystemMatrix(simulation.u, d2edx2);
        simulation.linearSolveNaive(d2edx2, dOdx, lambda);
    }
    
    
    simulation.cells.dOdpEdgeWeightsFromLambda(lambda, dOdp);

    if (use_log_barrier)
    {
        T e_barrier = 0.0;
        for (int i = 0; i < n_dof_design; i++)
        {
            if (p_curr[i] < barrier_distance)
            {
                energy += barrier_weight * barrier<0>(p_curr[i], barrier_distance);
                dOdp[i] += barrier_weight * barrier<1>(p_curr[i], barrier_distance);
            }
        }
        // std::cout << "barrier energy: " << e_barrier << " " << p_curr.minCoeff() << std::endl;
    }
    
    if (add_min_act)
    {
        for (int i = 0; i < n_dof_design; i++)
        {
            if (p_curr[i] < 0)
                dOdp[i] -= w_min_act;
            else
                dOdp[i] += w_min_act;
            energy += w_min_act * std::abs(p_curr[i]);
        }
    }

    equilibrium_prev = simulation.u;
    return dOdp.norm();
}

void ObjNucleiTracking::updateDesignParameters(const VectorXT& design_parameters)
{
    simulation.cells.edge_weights = design_parameters;
}

void ObjNucleiTracking::getDesignParameters(VectorXT& design_parameters)
{
    design_parameters = simulation.cells.edge_weights;
}

void ObjNucleiTracking::getSimulationAndDesignDoF(int& _sim_dof, int& _design_dof)
{
    _sim_dof = simulation.num_nodes * 3;
    _design_dof = simulation.cells.edge_weights.rows();
    n_dof_sim = _sim_dof;
    n_dof_design = _design_dof;
}

void ObjNucleiTracking::hessianSGN(const VectorXT& p_curr, 
    StiffnessMatrix& H, bool simulate)
{
    updateDesignParameters(p_curr);
    if (simulate)
    {
        simulation.reset();
        simulation.staticSolve();
    }

    std::vector<Entry> d2Odx2_entries;
    d2Odx2(p_curr, d2Odx2_entries);

    StiffnessMatrix dfdx(n_dof_sim, n_dof_sim);
    simulation.buildSystemMatrix(simulation.u, dfdx);
    dfdx *= -1.0;

    StiffnessMatrix dfdp;
    simulation.cells.dfdpWeightsSparse(dfdp);

    MatrixXT dfdp_dense = MatrixXT(dfdp);
    // Eigen::JacobiSVD<Eigen::MatrixXd> svd(dfdp_dense, Eigen::ComputeThinU | Eigen::ComputeThinV);
    // VectorXT Sigma = svd.singularValues();
    // std::cout << Sigma[0] << " " << Sigma[Sigma.rows() - 1] << std::endl;

    StiffnessMatrix d2Odx2_mat(n_dof_sim, n_dof_sim);
    d2Odx2_mat.setFromTriplets(d2Odx2_entries.begin(), d2Odx2_entries.end());

    int nx = n_dof_sim, np = n_dof_design, nxnp = n_dof_sim + n_dof_design;

    H.resize(n_dof_sim * 2 + n_dof_design, n_dof_sim * 2 + n_dof_design);
    std::vector<Entry> entries;
    
    entries.insert(entries.end(), d2Odx2_entries.begin(), d2Odx2_entries.end());

    for (int i = 0; i < dfdx.outerSize(); i++)
    {
        for (StiffnessMatrix::InnerIterator it(dfdx, i); it; ++it)
        {
            entries.push_back(Entry(it.row() + nxnp, it.col(), it.value()));
            entries.push_back(Entry(it.row(), it.col() + nxnp, it.value()));
        }
    }
    
    for (int i = 0; i < dfdp.outerSize(); i++)
    {
        for (StiffnessMatrix::InnerIterator it(dfdp, i); it; ++it)
        {
            entries.push_back(Entry(it.col() + nx, it.row() + nxnp, it.value()));
            entries.push_back(Entry(it.row() + nxnp, it.col() + nx, it.value()));
        }
    }

    // for (int i = 0; i < n_dof_design; i++)
    //     entries.push_back(Entry(i + n_dof_sim, i + n_dof_sim, 100));
    
    // for (int i = 0; i < nxnp; i++)
    //     entries.push_back(Entry(i, i, 1e-4));
    for (int i = nx; i < nxnp; i++)
        entries.push_back(Entry(i, i, 1e-8));
    // for (int i = nxnp; i < nxnp + n_dof_sim; i++)
    //     entries.push_back(Entry(i, i, -1e-6));
    
    H.setFromTriplets(entries.begin(), entries.end());
    H.makeCompressed();
}

void ObjNucleiTracking::hessianGN(const VectorXT& p_curr, StiffnessMatrix& H, bool simulate)
{
    updateDesignParameters(p_curr);
    if (simulate)
    {
        simulation.reset();
        simulation.staticSolve();
    }
    
    MatrixXT dxdp;
    simulation.cells.dxdpFromdxdpEdgeWeights(dxdp);
    std::vector<Entry> d2Odx2_entries;
    d2Odx2(p_curr, d2Odx2_entries);
    StiffnessMatrix d2Odx2_matrix(n_dof_sim, n_dof_sim);
    d2Odx2_matrix.setFromTriplets(d2Odx2_entries.begin(), d2Odx2_entries.end());

    StiffnessMatrix dxdp_sparse = dxdp.sparseView();
    // Timer tt(true);
    H = dxdp_sparse.transpose() * d2Odx2_matrix * dxdp_sparse;
    // tt.stop();
    // std::cout << "multiplication: " << tt.elapsed_sec() << std::endl;

    if (use_log_barrier)
        tbb::parallel_for(0, (int)H.rows(), [&](int row)
        {
            if (p_curr[row] < barrier_distance)
                H.coeffRef(row, row) += barrier_weight * barrier<2>(p_curr[row], barrier_distance);
        });
    H.makeCompressed();
}

T ObjNucleiTracking::maximumStepSize(const VectorXT& dp)
{
    VectorXT p_curr;
    getDesignParameters(p_curr);
    T step_size = 1.0;
    if (!use_log_barrier)
        return step_size;
    while (true)
    {
        VectorXT forward = p_curr + step_size * dp;
        if (forward.minCoeff() < 0)
            step_size *= 0.8;
        else
            return step_size;
    }
    
}

bool ObjNucleiTracking::getTargetTrajectoryFrame(VectorXT& frame_data)
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

void ObjNucleiTracking::loadTargetTrajectory(const std::string& filename)
{
    DataIO data_io;
    data_io.loadTrajectories(filename, cell_trajectories);
}

void ObjNucleiTracking::initializeTargetFromMap(const std::string& filename, int _frame)
{
    VectorXT data_points;
    frame = _frame;
    bool success = getTargetTrajectoryFrame(data_points);
    std::ifstream in(filename);
    int idx0, idx1;
    std::vector<int> vf_cell_indices;
    simulation.cells.getVFCellIds(vf_cell_indices);
    while (in >> idx0 >> idx1)
    {
        if (std::find(vf_cell_indices.begin(), vf_cell_indices.end(), idx0) 
            != vf_cell_indices.end())
            target_positions[idx0] = data_points.segment<3>(idx1 * 3);
    }
    in.close();
}

void ObjNucleiTracking::updateTarget()
{
    target_positions.clear();
    VectorXT cell_centroids;
    simulation.cells.getAllCellCentroids(cell_centroids);
    VectorXT data_points;
    bool success = getTargetTrajectoryFrame(data_points);

    TV min_corner, max_corner;
    simulation.cells.computeBoundingBox(min_corner, max_corner);
    T spacing = 0.05 * (max_corner - min_corner).norm();

    T max_dis = 0.02 * (max_corner - min_corner).norm();
    bool inverse = true;
    std::vector<std::pair<int, int>> pairs;
    if (inverse)
    {
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
            if (min_dis_pt != -1 && min_dis < max_dis)
            {
                target_positions[i] = data_points.segment<3>(min_dis_pt * 3);
                pairs.push_back(std::make_pair(i, min_dis_pt));
            }
        }   
    }
    else
    {
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
            if (min_dis_pt != -1 && min_dis < max_dis)
            {
                target_positions[min_dis_pt] = current;
                pairs.push_back(std::make_pair(min_dis_pt, i));
            }
        }   
    }
    // std::ofstream out("idx_map.txt");
    // for (auto pair : pairs)
    //     out << pair.first << " " << pair.second << std::endl;
    // out.close();
}

