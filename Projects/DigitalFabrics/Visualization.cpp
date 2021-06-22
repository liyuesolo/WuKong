#include "EoLRodSim.h"
#include "igl/colormap.h"
#include "igl/readOBJ.h"
// template<class T, int dim>
// void EoLRodSim<T, dim>::buildMeshFromRodNetwork(Eigen::MatrixXd& V, Eigen::MatrixXi& F, 
//     Eigen::Ref<const DOFStack> q_display, Eigen::Ref<const IV3Stack> rods_display, 
//     Eigen::Ref<const TV3Stack> normal_tile)
// {
//     int n_div = 10;
    
//     T theta = 2.0 * EIGEN_PI / T(n_div);
//     TV3Stack points = TV3Stack::Zero(3, n_div);

//     // bottom face vertices
//     for(int i = 0; i < n_div; i++)
//         points.col(i) = TV3(R * std::cos(theta * T(i)), 0.0, R*std::sin(theta*T(i)));
    
//     int n_ros_draw = rods_display.cols();
    

//     int rod_offset_v = n_div * 2 + 2;
//     int rod_offset_f = n_div * 4;
//     V.resize(n_ros_draw * rod_offset_v, 3);
//     V.setZero();
//     F.resize(n_ros_draw * rod_offset_f, 3);
//     F.setZero();
//     int rod_cnt = 0;
    
//     tbb::parallel_for(0, n_ros_draw, [&](int rod_cnt){
//         int rov = rod_cnt * rod_offset_v;
//         int rof = rod_cnt * rod_offset_f;

//         TV vtx_from_TV = q_display.col(rods_display.col(rod_cnt)[0]).template segment<dim>(0);
//         TV vtx_to_TV = q_display.col(rods_display.col(rod_cnt)[1]).template segment<dim>(0);

//         TV3 vtx_from = TV3::Zero();
//         TV3 vtx_to = TV3::Zero();
//         if constexpr (dim == 3)
//         {
//             vtx_from = vtx_from_TV;
//             vtx_to = vtx_to_TV;
//         }
//         else
//         {
//             // vtx_from = TV3(vtx_from_TV[0], 0, vtx_from_TV[1]);
//             // vtx_to = TV3(vtx_to_TV[0], 0, vtx_to_TV[1]);
//             vtx_from = TV3(vtx_from_TV[0], vtx_from_TV[1], 0);
//             vtx_to = TV3(vtx_to_TV[0], vtx_to_TV[1], 0);
//         }

        
//         TV3 normal_offset = TV3::Zero();
//         if (rods_display.col(rod_cnt)[2] == WARP)
//             normal_offset = normal_tile.col(rod_cnt);
//         else
//             normal_offset = normal_tile.col(rod_cnt);

//         vtx_from += normal_offset * R;
//         vtx_to += normal_offset * R;
        
//         TV3 axis_world = vtx_to - vtx_from;
//         TV3 axis_local(0, axis_world.norm(), 0);

        
//         TM3 R = Eigen::Quaternion<T>().setFromTwoVectors(axis_local, axis_world).toRotationMatrix();
        
//         V(rov + n_div*2+1, 1) = axis_world.norm();
        
//         V.row(rov + n_div*2+1) = (V.row(rov + n_div*2+1) * R).transpose() - vtx_from;
//         V.row(rov + n_div*2) = -vtx_from;
        
//         for(int i = 0; i < n_div; i++)
//         {
//             for(int d = 0; d < 3; d++)
//             {
//                 V(rov + i, d) = points.col(i)[d];
//                 V(rov + i+n_div, d) = points.col(i)[d];
//                 if (d == 1)
//                     V(rov + i+n_div, d) += axis_world.norm();
//             }

//             // central vertex of the top and bottom face
//             V.row(rov + i) = (V.row(rov + i) * R).transpose() - vtx_from;
//             V.row(rov + i + n_div) = (V.row(rov + i + n_div) * R).transpose() - vtx_from;
            
//             //top faces of the cylinder
//             F.row(rof + i) = IV3(rov + n_div*2, rov + i, rov + (i+1)%(n_div));
//             //bottom faces of the cylinder
//             F.row(rof + i+n_div) = IV3(rov + n_div*2+1, rov + n_div + (i+1)%(n_div), rov + i + n_div);
            
//             //side faces of the cylinder
//             F.row(rof + i*2 + 2 * n_div) = IV3(rov + i, rov + i+n_div, rov + (i+1)%(n_div));
//             F.row(rof + i*2 + 1 + 2 * n_div) = IV3(rov + (i+1)%(n_div), rov + i+n_div, rov + (i+1)%(n_div) + n_div);
//         }

//     });
// }

// template<class T, int dim>
// void EoLRodSim<T, dim>::appendSphereMesh(Eigen::MatrixXd& V, Eigen::MatrixXi& F, T scale, Vector<T, 3> shift)
// {
//     Eigen::MatrixXd v_sphere;
//     Eigen::MatrixXi f_sphere;

//     igl::readOBJ("/home/yueli/Documents/ETH/WuKong/Projects/DigitalFabrics/Data/sphere.obj", v_sphere, f_sphere);

//     v_sphere = v_sphere * scale;

//     tbb::parallel_for(0, (int)v_sphere.rows(), [&](int row_idx){
//         v_sphere.row(row_idx) += shift;
//     });

//     int n_vtx_prev = V.rows();
//     int n_face_prev = F.rows();

//     tbb::parallel_for(0, (int)f_sphere.rows(), [&](int row_idx){
//         f_sphere.row(row_idx) += Eigen::Vector3i(n_vtx_prev, n_vtx_prev, n_vtx_prev);
//     });

//     V.conservativeResize(V.rows() + v_sphere.rows(), 3);
//     F.conservativeResize(F.rows() + f_sphere.rows(), 3);

//     V.block(n_vtx_prev, 0, v_sphere.rows(), 3) = v_sphere;
//     F.block(n_face_prev, 0, f_sphere.rows(), 3) = f_sphere;
// }

template<class T, int dim>
void EoLRodSim<T, dim>::buildMeshFromRodNetwork(Eigen::MatrixXd& V, Eigen::MatrixXi& F, 
    Eigen::Ref<const DOFStack> q_display, Eigen::Ref<const IV3Stack> rods_display, 
    Eigen::Ref<const TV3Stack> normal_tile)
{
    int n_div = 10;
    
    T theta = 2.0 * EIGEN_PI / T(n_div);
    TV3Stack points = TV3Stack::Zero(3, n_div);

    T visual_R = 0.01;
    // bottom face vertices
    for(int i = 0; i < n_div; i++)
        points.col(i) = TV3(visual_R * std::cos(theta * T(i)), 0.0, visual_R*std::sin(theta*T(i)));
    
    int n_ros_draw = rods_display.cols();
    

    // int rod_offset_v = n_div * 2 + 2;
    // int rod_offset_f = n_div * 4;

    int rod_offset_v = n_div * 2;
    int rod_offset_f = n_div * 2;
    
    V.resize(n_ros_draw * rod_offset_v, 3);
    V.setZero();
    F.resize(n_ros_draw * rod_offset_f, 3);
    F.setZero();
    int rod_cnt = 0;
    
    tbb::parallel_for(0, n_ros_draw, [&](int rod_cnt){
        int rov = rod_cnt * rod_offset_v;
        int rof = rod_cnt * rod_offset_f;

        TV vtx_from_TV = q_display.col(rods_display.col(rod_cnt)[0]).template segment<dim>(0) / 0.03;
        TV vtx_to_TV = q_display.col(rods_display.col(rod_cnt)[1]).template segment<dim>(0) / 0.03;
        // TV vtx_from_TV = q_display.col(rods_display.col(rod_cnt)[0]).template segment<dim>(0) / 1;
        // TV vtx_to_TV = q_display.col(rods_display.col(rod_cnt)[1]).template segment<dim>(0) / 1;

        // int yarn_type = rods_display.col(rod_cnt)[2];
        // T u_from = q_display(dim + yarn_type, rods_display.col(rod_cnt)[0]);
        // T u_to = q_display(dim + yarn_type, rods_display.col(rod_cnt)[1]);

        // vtx_from_TV = vtx_from_TV + u_from * (vtx_to_TV - vtx_from_TV);
        // vtx_from_TV = vtx_from_TV + u_to * (vtx_to_TV - vtx_from_TV);

        TV3 vtx_from = TV3::Zero();
        TV3 vtx_to = TV3::Zero();
        if constexpr (dim == 3)
        {
            vtx_from = vtx_from_TV;
            vtx_to = vtx_to_TV;
        }
        else
        {
            // vtx_from = TV3(vtx_from_TV[0], 0, vtx_from_TV[1]);
            // vtx_to = TV3(vtx_to_TV[0], 0, vtx_to_TV[1]);
            vtx_from = TV3(vtx_from_TV[0], vtx_from_TV[1], 0);
            vtx_to = TV3(vtx_to_TV[0], vtx_to_TV[1], 0);
        }

        
        TV3 normal_offset = TV3::Zero();
        // if (rods_display.col(rod_cnt)[2] == WARP)
        //     normal_offset = normal_tile.col(rod_cnt);
        // else
        //     normal_offset = normal_tile.col(rod_cnt);

        vtx_from += normal_offset * R;
        vtx_to += normal_offset * R;
        
        TV3 axis_world = vtx_to - vtx_from;
        TV3 axis_local(0, axis_world.norm(), 0);

        
        TM3 R = Eigen::Quaternion<T>().setFromTwoVectors(axis_world, axis_local).toRotationMatrix();
        
        // V(rov + n_div*2+1, 1) = axis_world.norm();
        
        // V.row(rov + n_div*2+1) = (V.row(rov + n_div*2+1) * R).transpose() + vtx_from;
        // V.row(rov + n_div*2) = vtx_from;
        
        for(int i = 0; i < n_div; i++)
        {
            for(int d = 0; d < 3; d++)
            {
                V(rov + i, d) = points.col(i)[d];
                V(rov + i+n_div, d) = points.col(i)[d];
                if (d == 1)
                    V(rov + i+n_div, d) += axis_world.norm();
            }

            // central vertex of the top and bottom face
            V.row(rov + i) = (V.row(rov + i) * R).transpose() + vtx_from;
            V.row(rov + i + n_div) = (V.row(rov + i + n_div) * R).transpose() + vtx_from;
            
            //top faces of the cylinder
            // F.row(rof + i) = IV3(rov + n_div*2, rov + i, rov + (i+1)%(n_div));
            //bottom faces of the cylinder
            // F.row(rof + i+n_div) = IV3(rov + n_div*2+1, rov + n_div + (i+1)%(n_div), rov + i + n_div);
            
            //side faces of the cylinder
            // F.row(rof + i*2 + 2 * n_div) = IV3(rov + i, rov + i+n_div, rov + (i+1)%(n_div));
            // F.row(rof + i*2 + 1 + 2 * n_div) = IV3(rov + (i+1)%(n_div), rov + i+n_div, rov + (i+1)%(n_div) + n_div);

            F.row(rof + i*2 ) = IV3(rov + i, rov + i+n_div, rov + (i+1)%(n_div));
            F.row(rof + i*2 + 1) = IV3(rov + (i+1)%(n_div), rov + i+n_div, rov + (i+1)%(n_div) + n_div);
        }

    });
}

template<class T, int dim>
void EoLRodSim<T, dim>::getEulerianDisplacement(Eigen::MatrixXd& X, Eigen::MatrixXd& x)
{
    int n_vtx = yarns.size() * (yarns[0].size() - 1);
    int cnt = 0;
    X.resize(n_vtx, 3); x.resize(n_vtx, 3);
    // tbb::parallel_for(0, (int)yarns.size(), [&](int i){
    for (int i = 0; i < (int)yarns.size(); i++){
        TV from = q.col(yarns[i][0]).template segment<dim>(0);
        TV to = q.col(yarns[i][yarns[i].size()-2]).template segment<dim>(0);

        int yarn_type = yarns[i][yarns[i].size()-1];

        for(int j = 0; j < yarns[i].size() - 1; j++)
        {
            int node = yarns[i][j];
            T u, u0;
            u = yarn_type == WARP ? q(dim + 0, node) : q(dim + 1, node);
            u0 = yarn_type == WARP ? q0(dim + 0, node) : q0(dim + 1, node);
            
            TV x_lag = from + u * (to - from);
            TV x_eul = from + u0 * (to - from);
            if constexpr (dim == 2)
            {
                X.row(cnt) = Eigen::RowVector3d(x_eul[0], x_eul[1], 0);
                x.row(cnt) = Eigen::RowVector3d(x_lag[0] + 1.0, x_lag[1], 0);
                cnt++;
            }
        }
    }
    // });
    assert(cnt == n_vtx);
}

template<class T, int dim>
void EoLRodSim<T, dim>::markSlidingRange(int idx, int dir, int depth, 
    std::vector<bool>& can_slide, int root)
{
    if (depth > slide_over_n_rods[dir] || idx == -1)
        return;
    T rod_length = (q0.col(rods.col(0)(0)).template segment<dim>(0) - 
        q0.col(rods.col(0)(1)).template segment<dim>(0)).norm();
    // distance root/crossing node travels
    T root_sliding_dis = std::abs(q(dim + dir, root) - q0(dim + dir, root));
    T dis_to_root_rest_state = std::abs(q0(dim + dir, idx) - q0(dim + dir, root));
    T dis_to_root_current = std::abs(q(dim + dir, idx) - q(dim + dir, root));
    
    if (idx == root || root_sliding_dis < 1e-6)
        can_slide[idx * 2 + dir] = true;
    
    if (root_sliding_dis > slide_over_n_rods[dir] * rod_length - 1e-6)
    {
        if(dis_to_root_current > dis_to_root_rest_state)
            can_slide[idx * 2 + dir] = true;
    }
    else
    {
        can_slide[idx * 2 + dir] = true;
    }
    
    
    if(dir == 0)
    {
        markSlidingRange(connections(2, idx), dir, depth + 1, can_slide, root);
        markSlidingRange(connections(3, idx), dir, depth + 1, can_slide, root);
    }
    else
    {
        markSlidingRange(connections(0, idx), dir, depth + 1, can_slide, root);
        markSlidingRange(connections(1, idx), dir, depth + 1, can_slide, root);
    }   
}

template<class T, int dim>
void EoLRodSim<T, dim>::getColorPerYarn(Eigen::MatrixXd& C, int n_rod_per_yarn)
{
    int n_faces = 20;

    std::vector<bool> can_slide(n_nodes * 2, false);

    iterateSlidingNodes([&](int node_idx){
              
        if (dirichlet_data.find(node_idx) != dirichlet_data.end())
        {
            TVDOF mask = dirichlet_data[node_idx].first;
            if(!mask[dim])
                markSlidingRange(node_idx, 0, 0, can_slide, node_idx);
            if(!mask[dim+1])
                markSlidingRange(node_idx, 1, 0, can_slide, node_idx);
        }
        else
        {
            markSlidingRange(node_idx, 0, 0, can_slide, node_idx);
            markSlidingRange(node_idx, 1, 0, can_slide, node_idx);
            
        }
    });

    C.resize(n_rods * n_faces, 3);
    std::vector<Eigen::Vector3d> colors = {
        Eigen::Vector3d(1, 0, 0), Eigen::Vector3d(1, 1, 0), Eigen::Vector3d(0, 1, 0), 
        Eigen::Vector3d(0, 1, 1), Eigen::Vector3d(0, 0, 1), Eigen::Vector3d(1, 0, 1)};
    int n_yarn = yarns.size();
    VectorXT delta_u(n_rods);
    delta_u.setZero();

    tbb::parallel_for(0, n_rods, [&](int rod_idx){
        int v0 = rods(0, rod_idx);
        int v1 = rods(1, rod_idx);
        T dU = (q0.col(v1).template segment<2>(dim) - q0.col(v0).template segment<2>(dim)).norm();
        T du = (q.col(v1).template segment<2>(dim) - q.col(v0).template segment<2>(dim)).norm();
        delta_u[rod_idx] = std::abs(du - dU);
    });
    
    // Eigen::MatrixXd sliding_color(n_rods, 3);
    // sliding_color.setZero();
    // std::cout << "color mapping" << std::endl;
    // igl::colormap(igl::ColorMapType::COLOR_MAP_TYPE_JET, delta_u, true, sliding_color);
    // std::cout << "color mapping done" << std::endl;
    tbb::parallel_for(0, n_rods, [&](int rod_idx){
        for(int i = 0; i < n_faces; i++)
        {

            if(can_slide[rods(0,rod_idx) * 2 + rods(2, rod_idx)] && can_slide[rods(1,rod_idx) * 2 + rods(2, rod_idx)])
                C.row(rod_idx * n_faces + i) = colors[0];
            else
                C.row(rod_idx * n_faces + i) = colors[2];

        }
            // C.row(rod_idx * 40 + i) = sliding_color.row(rod_idx);
            // C.row(rod_idx * 40 + i) = rod_idx % 2 == 0 ? Eigen::Vector3d::Ones() : Eigen::Vector3d::Zero();
            // C.row(rod_idx * 40 + i) = colors[yarn_map[rod_idx]];
    });
}

template<class T, int dim>
void EoLRodSim<T, dim>::getColorFromStretching(
    Eigen::MatrixXd& C)
{
    int n_faces = 20;
    C.resize(n_rods * n_faces, 3);
    DOFStack q_temp = q - q0;
    VectorXT rod_energy(n_rods);
    rod_energy.setZero();
    // tbb::parallel_for(0, n_rods, [&](int rod_idx){
    for (int rod_idx = 0; rod_idx < n_rods; rod_idx++) {
        int node0 = rods.col(rod_idx)[0];
        int node1 = rods.col(rod_idx)[1];
        TV x0 = q.col(node0).template segment<dim>(0);
        TV x1 = q.col(node1).template segment<dim>(0);
        TV2 u0 = q.col(node0).template segment<2>(dim);
        TV2 u1 = q.col(node1).template segment<2>(dim);
        TV2 delta_u = u1 - u0;
        int yarn_type = rods.col(rod_idx)[2];
        

        int uv_offset = yarn_type == WARP ? 0 : 1;
    
        // add elastic potential here 1/2 ks delta_u * (||w|| - 1)^2
        TV w = (x1 - x0) / std::abs(delta_u[uv_offset]);
        // rod_energy[rod_idx] += 0.5 * ks * std::abs(delta_u[uv_offset]) * std::pow(w.norm() - 1.0, 2);
        rod_energy[rod_idx] += std::abs(w.norm() - 1);
        
        std::cout << "Rod " << node0 << "->" << node1 << ": " << std::abs(w.norm() - 1) << std::endl;

    }
    // });

    // iteratePBCReferencePairs([&](int dir_id, int node_i, int node_j){
    //     TV xj = q.col(node_j).template segment<dim>(0);
    //     TV xi = q.col(node_i).template segment<dim>(0);
    //     if constexpr (dim == 2)
    //     {
    //         T theta = 0.285;
    //         TV strain_dir = TV(std::cos(theta), std::sin(theta));
    //         T dij = (xj - xi).dot(strain_dir);
    //         std::cout  << dij << std::endl;
    //         // pbc_strain_data.push_back(std::make_pair(IV2(node_i, node_j), std::make_pair(strain_dir, dij)));
    //     }
    // });

    if(rod_energy.maxCoeff() > 1e-4)
        rod_energy /= rod_energy.maxCoeff();
    else
        rod_energy.setZero();
    
    tbb::parallel_for(0, n_rods, [&](int rod_idx){
        for(int i = 0; i < n_faces; i++)
            C.row(rod_idx * n_faces + i) = Eigen::Vector3d(rod_energy[rod_idx], rod_energy[rod_idx], rod_energy[rod_idx]);
    });

}
template class EoLRodSim<double, 3>;
template class EoLRodSim<double, 2>;