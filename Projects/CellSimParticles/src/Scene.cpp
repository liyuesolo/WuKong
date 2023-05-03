#include "../include/CellSim.h"

template <int dim>
void CellSim<dim>::loadTargetTrajectories()
{
    if constexpr (dim == 3)
    {        
        std::string filename = "/home/yueli/Documents/ETH/WuKong/Projects/CellSim/data/trajectories.dat";
        
        MatrixXi cell_trajectories;
        read_binary<MatrixXi>(filename, cell_trajectories);
        n_frames = cell_trajectories.rows() / 3;
        int n_nucleus = cell_trajectories.cols();

        target_trajectories.resize(n_nucleus * 3, n_frames);
        target_trajectories.setConstant(-1e10);
        tbb::parallel_for(0, n_frames, [&](int frame){
            for (int i = 0; i < n_nucleus; i++)
            {
                IV tmp = cell_trajectories.col(i).segment<3>(frame * 3);
                for (int d = 0; d < 3; d++)
                    target_trajectories(i* 3 + d, frame) = T(tmp[d]);
            }
        }); 
        
        
        Matrix<T, 3, 3> R;
        R << 0.960277, -0.201389, 0.229468, 0.2908, 0.871897, -0.519003, -0.112462, 0.558021, 0.887263;
        Matrix<T, 3, 3> R2 = Eigen::AngleAxis<T>(0.20 * M_PI + 0.5 * M_PI, TV(-1.0, 0.0, 0.0)).toRotationMatrix();
        for (int frame = 0; frame < n_frames; frame++)
        {
            for (int i = 0; i < n_nucleus; i++)
            {
                TV pos = target_trajectories.col(frame).segment<3>(i * 3);
                if ((pos - TV(-1e10, -1e10, -1e10)).norm() > 1e-8)
                {
                    TV updated = (pos - TV(605.877,328.32,319.752)) / 1096.61;
                    updated = R2 * R * updated;
                    // frame_data.segment<3>(i * 3) = updated * 0.935 * 5.0;
                    target_trajectories.col(frame).segment<3>(i * 3) = updated * 0.94 * 5.0;
                }
            }
        }


    }
}



template <int dim>
void CellSim<dim>::updateTargetPointsAsBC()
{
    if constexpr (dim == 3)
    {
        iterateCellSerial([&](int i)
        {
            if (is_control_points[i])
            {
                for (int d = 0; d < dim; d++)
                {
                    T target_pos = target_trajectories.col(global_frame)[is_control_points[i] * dim + d];
                    if (target_pos != -1e10)
                        dirichlet_data[i * dim + d] = target_pos - undeformed[i * dim + d];
                }
                
            }
        });
    }
}

template <int dim>
void CellSim<dim>::initializeFrom3DData()
{
    if constexpr (dim == 3)
    {
        loadTargetTrajectories();
        std::string initial_data_file = "/home/yueli/Documents/ETH/WuKong/Projects/CellSimParticles/data/particles_centroid.txt";
        std::ifstream in(initial_data_file);
        T v;
        std::vector<T> data_vec;
        while(in >> v)
            data_vec.push_back(v);
        in.close();
        undeformed = Eigen::Map<VectorXT>(data_vec.data(), data_vec.size());
        
        num_cells = undeformed.rows() / dim;
        num_nodes = num_cells;
        radius = 0.08;
        constructYolkMesh3D();

        initializeControlPointsData();
        computeInitialNeighbors();
        deformed = undeformed;
        u = VectorXT::Zero(deformed.rows());
        std::cout << "# cells " << num_cells << std::endl;
        collision_dhat = 2.0 * radius;

        w_reg_edge = 10.0; 
        w_rep = 10.0;
        w_adh = 5.0;
        w_matching = 1e3;
        bound_coeff = 1e3;

        // w_reg_edge = 0; 
        // w_rep = 10.0;
        // w_adh = 10.0;
        // w_matching = 1e3;
        // bound_coeff = 0;
        // w_yolk = 0;
        // use_ipc = false;
        
        for (int d = 0; d < dim * 3; d++)
            dirichlet_data[yolk_cell_starts + d] = 0.0;
        woodbury = true;
        use_ipc = true;
        if (use_ipc)
        {
            buildIPCRestData();
        }
        yolk_area_rest = computeYolkArea();
        constructMembraneLevelset();
        // checkMembranePenetration();
        std::cout << "yolk volume rest " << yolk_area_rest << std::endl;
        TV min_corner, max_corner;
        computeBoundingBox(min_corner, max_corner);
        centroid = 0.5 * (min_corner + max_corner);

        max_newton_iter = 3000;
    }
}

template <int dim>
void CellSim<dim>::initializeCells()
{
    radius = 0.05;
    // undeformed.resize(4);
    // undeformed << 0.5, 0.5, 0.55, 0.5;

    num_cells = 40;
    num_nodes = num_cells;
    centroid = TV::Constant(0.5);
    // sdf = SphereSDF(centroid, 0.5);
    
    // std::cout << sdf.center.transpose() << " " << sdf.radius << std::endl;

    T dtheta = 2.0 * M_PI / T(num_cells);
    undeformed.resize(num_cells * dim);

    for (int i = 0; i < num_cells; i++)
    {
        undeformed[i * dim + 0] = centroid[0] + 0.5 * std::cos(T(i) * dtheta);
        undeformed[i * dim + 1] = centroid[1] + 0.5 * std::sin(T(i) * dtheta);
    }
    
    auto inside_yolk = [&](const TV& pos)->bool
    {
        if ((pos - centroid).norm() - 0.45 < 0)
            return true;
        return false;
    };
    
    radius = 0.04;
    w_reg_edge = 10.0;
    constructYolkMesh2D();
    target_positions = undeformed;
    deformed = undeformed;
    u = VectorXT::Zero(deformed.rows());
    initializeControlPointsData();
    computeInitialNeighbors();
    yolk_area_rest = computeYolkArea();
    std::cout << "# cells " << num_cells << std::endl;
    collision_dhat = 2.0 * radius;
    w_rep = 100.0;
    w_adh = 50.0;

    bound_coeff = 1e4;

    for (int d = 0; d < dim; d++)
        dirichlet_data[d] = 0.0;
    
    // std::vector<int> control_points = {9, 10, 11};
    std::vector<int> control_points = {10};
    for (int i : control_points)
    {
        is_control_points[i] = i;
        target_positions[i * dim + 1] = undeformed[i * dim + 1] - 0.4;
    }
    if (use_ipc)
    {
        buildIPCRestData();

    }
}

template <int dim>
void CellSim<dim>::appendCylindersToEdges(const std::vector<std::pair<TV3, TV3>>& edge_pairs, 
        const TV3& color, T radius,
        Eigen::MatrixXd& _V, Eigen::MatrixXi& _F, Eigen::MatrixXd& _C)
{
    int n_div = 10;
    T theta = 2.0 * EIGEN_PI / T(n_div);
    VectorXT points = VectorXT::Zero(n_div * 3);
    for(int i = 0; i < n_div; i++)
        points.segment<3>(i * 3) = TV3(radius * std::cos(theta * T(i)), 
        0.0, radius * std::sin(theta*T(i)));

    int offset_v = n_div * 2;
    int offset_f = n_div * 2;

    int n_row_V = _V.rows();
    int n_row_F = _F.rows();

    int n_edge = edge_pairs.size();

    _V.conservativeResize(n_row_V + offset_v * n_edge, 3);
    _F.conservativeResize(n_row_F + offset_f * n_edge, 3);
    _C.conservativeResize(n_row_F + offset_f * n_edge, 3);

    tbb::parallel_for(0, n_edge, [&](int ei)
    {
        TV3 axis_world = edge_pairs[ei].second - edge_pairs[ei].first;
        TV3 axis_local(0, axis_world.norm(), 0);

        Matrix<T, 3, 3> R = Eigen::Quaternion<T>().setFromTwoVectors(axis_world, axis_local).toRotationMatrix();

        for(int i = 0; i < n_div; i++)
        {
            for(int d = 0; d < 3; d++)
            {
                _V(n_row_V + ei * offset_v + i, d) = points[i * 3 + d];
                _V(n_row_V + ei * offset_v + i+n_div, d) = points[i * 3 + d];
                if (d == 1)
                    _V(n_row_V + ei * offset_v + i+n_div, d) += axis_world.norm();
            }

            // central vertex of the top and bottom face
            _V.row(n_row_V + ei * offset_v + i) = (_V.row(n_row_V + ei * offset_v + i) * R).transpose() + edge_pairs[ei].first;
            _V.row(n_row_V + ei * offset_v + i + n_div) = (_V.row(n_row_V + ei * offset_v + i + n_div) * R).transpose() + edge_pairs[ei].first;

            _F.row(n_row_F + ei * offset_f + i*2 ) = IV3(n_row_V + ei * offset_v + i, 
                                    n_row_V + ei * offset_v + i+n_div, 
                                    n_row_V + ei * offset_v + (i+1)%(n_div));

            _F.row(n_row_F + ei * offset_f + i*2 + 1) = IV3(n_row_V + ei * offset_v + (i+1)%(n_div), 
                                        n_row_V + ei * offset_v + i+n_div, 
                                        n_row_V + + ei * offset_v + (i+1)%(n_div) + n_div);

            _C.row(n_row_F + ei * offset_f + i*2 ) = color;
            _C.row(n_row_F + ei * offset_f + i*2 + 1) = color;
        }
    });
}

template class CellSim<2>;
template class CellSim<3>;