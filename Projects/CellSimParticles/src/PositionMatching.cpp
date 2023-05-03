#include "../include/CellSim.h"
#include "../autodiff/EdgeEnergy.h"

template <int dim>
void CellSim<dim>::updatePerFrameData()
{
    if constexpr (dim == 3)
    {
        undeformed = deformed;
        VectorXT frame_data = target_trajectories.col(global_frame);
        int n_ctrl_pts = control_points.rows() / dim;
        for (int i = 0; i < n_ctrl_pts; i++)
        {
            control_points.segment<dim>(i * dim) = frame_data.segment<dim>(ctrl_point_data_map[i] * dim);
        }
        u.setZero();
        if (use_ipc)
            buildIPCRestData();
    }
}

template <int dim>
void CellSim<dim>::initializeControlPointsData()
{
    if constexpr (dim == 3)
    {
        std::vector<TV> control_points_vector;
        VectorXT frame0_data = target_trajectories.col(0);
        SpatialHash<dim> cell_center_hash;
        cell_center_hash.build(2.0 * radius, undeformed.segment(0, num_cells * dim));
        int n_data_pt = frame0_data.rows() / 3;
        for (int i = 0; i < n_data_pt; i++)
        {
            TV xi = frame0_data.segment<dim>(i * dim);
            std::vector<int> neighbors;
            cell_center_hash.getOneRingNeighbors(xi, neighbors);
            // std::cout << neighbors.size() << std::endl;
            int cnt = 0;
            for (int j : neighbors)
            {
                TV xj = undeformed.segment<dim>(j * dim);
                if ((xi - xj).norm() < 1.0 * radius)
                {
                    cnt++;
                    break;
                }
            }
            if (cnt > 0)
            {
                control_points_vector.push_back(xi);
                ctrl_point_data_map[control_points_vector.size() - 1] = i;
                for (int j : neighbors)
                {
                    TV xj = undeformed.segment<dim>(j * dim);
                    if ((xi - xj).norm() < 1.0 * radius)
                    {
                        control_edges.push_back(Edge(j, control_points_vector.size() - 1));
                    }
                }
            }
        }
        control_points.resize(control_points_vector.size() * dim);
        for (int i = 0; i < control_points_vector.size(); i++)
            control_points.segment<dim>(i * dim) = control_points_vector[i];
        control_edge_rest_length.resize(control_edges.size());
        for (int i = 0; i < control_edges.size(); i++)
        {
            TV xi = undeformed.segment<dim>(control_edges[i][0] * dim);
            TV xj = control_points.segment<dim>(control_edges[i][1] * dim);
            control_edge_rest_length[i] = (xi - xj).norm();
        }
        // control_points = Eigen::Map<VectorXT>(control_points_vector.data(), control_points_vector.size());
        std::cout << "# control points " << control_points_vector.size() << " # data points " << n_data_pt << std::endl;
        std::cout << "# control edges " << control_edges.size() << std::endl;
    }
}

template <int dim>
void CellSim<dim>::addMatchingEnergy(T& energy)
{
    if constexpr (dim == 3)
    {
        for (int i = 0; i < control_edges.size(); i++)
        {
            TV xi = deformed.segment<dim>(control_edges[i][0] * dim);
            TV xj = control_points.segment<dim>(control_edges[i][1] * dim);
            T rest_length = control_edge_rest_length[i];
            T ei;
            compute3DMatchingEnergy(xi, xj, rest_length, ei);
            energy += w_matching * ei;
        }
    }
    
    // // if constexpr (dim == 3)
    // {
    //     VectorXT energies(num_cells); energies.setZero();

    //     iterateCellParallel([&](int i)
    //     {
    //         if (is_control_points[i] != -1)
    //         {
    //             TV xi = deformed.segment<dim>(i * dim);
    //             TV target_pos = target_positions.segment<dim>(is_control_points[i] * dim);
    //             // TV target_pos = target_trajectories.col(global_frame).segment<dim>(is_control_points[i] * dim);
    //             if ((target_pos - TV::Constant(-1e10)).norm() > 1e-6)
    //             {
    //                 energies[i] = 0.5 * w_matching * (xi - target_pos).squaredNorm();
    //             }
    //         }
    //     });
    // }
}

template <int dim>
void CellSim<dim>::addMatchingForceEntries(VectorXT& residual)
{
    if constexpr (dim == 3)
    {
        for (int i = 0; i < control_edges.size(); i++)
        {
            TV xi = deformed.segment<dim>(control_edges[i][0] * dim);
            TV xj = control_points.segment<dim>(control_edges[i][1] * dim);
            T rest_length = control_edge_rest_length[i];
            TV dedx;
            compute3DMatchingEnergyGradient(xi, xj, rest_length, dedx);
            addForceEntry<dim>(residual, {control_edges[i][0]}, -w_matching * dedx);
        }
    }
    // {
    //     iterateCellParallel([&](int i)
    //     {
    //         if (is_control_points[i] != -1)
    //         {
    //             TV xi = deformed.segment<dim>(i * dim);
    //             TV target_pos = target_positions.segment<dim>(is_control_points[i] * dim);
    //             // TV target_pos = target_trajectories.col(global_frame).segment<dim>(is_control_points[i] * dim);
    //             if ((target_pos - TV::Constant(-1e10)).norm() > 1e-6)
    //             {
    //                 addForceEntry<dim>(residual, {i}, -w_matching * (xi - target_pos));
    //             }
    //         }
    //     });
    // }
}

template <int dim>
void CellSim<dim>::addMatchingHessianEntries(std::vector<Entry>& entries)
{
    if constexpr (dim == 3)
    {
        for (int i = 0; i < control_edges.size(); i++)
        {
            TV xi = deformed.segment<dim>(control_edges[i][0] * dim);
            TV xj = control_points.segment<dim>(control_edges[i][1] * dim);
            T rest_length = control_edge_rest_length[i];
            TM d2edx2;
            compute3DMatchingEnergyHessian(xi, xj, rest_length, d2edx2);
            addHessianEntry<dim>(entries, {control_edges[i][0]}, w_matching * d2edx2);
        }
    }
    // std::vector<TM> sub_hessian(num_cells, TM::Zero());
    // iterateCellParallel([&](int i)
    // {
    //     if (is_control_points[i] != -1)
    //     {
    //         TV xi = deformed.segment<dim>(i * dim);
    //         // TV target_pos = target_trajectories.col(global_frame).segment<dim>(is_control_points[i] * dim);
    //         TV target_pos = target_positions.segment<dim>(is_control_points[i] * dim);
    //         if ((target_pos - TV::Constant(-1e10)).norm() > 1e-6)
    //         {
    //             sub_hessian[i] = TM::Identity() * w_matching;
    //         }
    //     }
    // });
    // for (int i = 0; i < num_cells; i++)
    // {
    //     addHessianEntry<dim>(entries, {i}, sub_hessian[i]);    
    // }
}
    
template class CellSim<2>;
template class CellSim<3>;