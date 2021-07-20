#include "EoLRodSim.h"

template<class T, int dim>
void EoLRodSim<T, dim>::buildPeriodicNetwork(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& C)
{
    return;
    // auto shift_xy = [&](Eigen::Ref<DOFStack> q_shift, TV shift, int offset)
    // {
    //     tbb::parallel_for(0, n_nodes, [&](int i){
    //         q_shift.col(i + offset).template segment<dim>(0) += shift;
    //     });
    // };

    // auto shift_rod = [&](Eigen::Ref<IV3Stack> rod_shift, int shift, int tile_id)
    // {
    //     tbb::parallel_for(0, n_rods, [&](int i){
    //         rod_shift.col(i + n_rods * tile_id).segment<2>(0) += IV2(shift, shift);
    //     });
    // };
    // int n_faces = 20;
    // DOFStack q_tile = q;
    // IV3Stack rods_tile = rods;
    // TV3Stack normal_tile = normal;

    // int n_tile = 9;
    // q_tile.conservativeResize(dof, n_nodes * n_tile);
    // rods_tile.conservativeResize(3, n_rods * n_tile);
    // normal_tile.conservativeResize(dof, n_nodes * n_tile);

    // tbb::parallel_for(0, n_nodes, [&](int node_idx){
    //     for(int i = 1; i < n_tile; i++)
    //     {
    //         q_tile.col(node_idx + i * n_nodes) = q.col(node_idx);
    //         normal_tile.col(node_idx + i * n_nodes) = normal.col(node_idx);
    //     }
    // });

    // tbb::parallel_for(0, n_rods, [&](int rod_idx){
    //     for(int i = 1; i < n_tile; i++)
    //         rods_tile.col(rod_idx + i * n_rods) = rods.col(rod_idx);
    // });
    
    // if constexpr (dim == 2)
    // {
    //     TV ref0_shift = q.col(pbc_ref_unique[0](0)).template segment<dim>(0) - q.col(pbc_ref_unique[0](1)).template segment<dim>(0);
    //     TV ref1_shift = q.col(pbc_ref_unique[1](0)).template segment<dim>(0) - q.col(pbc_ref_unique[1](1)).template segment<dim>(0);

    //     shift_xy(q_tile, ref0_shift, n_nodes);
    //     shift_rod(rods_tile, n_nodes, 1);

    //     shift_xy(q_tile, ref1_shift, 2 * n_nodes);
    //     shift_rod(rods_tile, 2 * n_nodes, 2);

    //     shift_xy(q_tile, -ref0_shift, 3 * n_nodes);
    //     shift_rod(rods_tile, 3 * n_nodes, 3);

    //     shift_xy(q_tile, -ref1_shift, 4 * n_nodes);
    //     shift_rod(rods_tile, 4 * n_nodes, 4);

    //     shift_xy(q_tile, -ref0_shift, 5 * n_nodes);
    //     shift_xy(q_tile, ref1_shift, 5 * n_nodes);
    //     shift_rod(rods_tile, 5 * n_nodes, 5);

    //     shift_xy(q_tile, -ref0_shift, 6 * n_nodes);
    //     shift_xy(q_tile, -ref1_shift, 6 * n_nodes);
    //     shift_rod(rods_tile, 6 * n_nodes, 6);

    //     shift_xy(q_tile, -ref1_shift, 7 * n_nodes);
    //     shift_xy(q_tile, ref0_shift, 7 * n_nodes);
    //     shift_rod(rods_tile, 7 * n_nodes, 7);

    //     shift_xy(q_tile, ref0_shift, 8 * n_nodes);
    //     shift_xy(q_tile, ref1_shift, 8 * n_nodes);
    //     shift_rod(rods_tile, 8 * n_nodes, 8);
        
    // }

    // buildMeshFromRodNetwork(V, F, q_tile, rods_tile, normal_tile);
    
    // C.resize(F.rows(), F.cols());
    // tbb::parallel_for(0, int(F.rows()), [&](int i){
    //     if (i < n_rods * (n_faces))
    //         C.row(i) = TV3(0, 1, 0);
    //     else
    //         C.row(i) = TV3(1, 1, 0);
    // });
}


template<class T, int dim>
void EoLRodSim<T, dim>::buildPlanePeriodicBCScene3x3Subnodes(int sub_div)
{
    pbc_bending_bn_pairs.clear();
    yarn_map.clear();
    
    buildPlanePeriodicBCScene3x3();
    if (sub_div > 1)
    {
        subdivideRods(sub_div);
    }
}

template<class T, int dim>
void EoLRodSim<T, dim>::subdivideRods(int sub_div)
{
    // subdivide = true;

    // auto setConnection = [&](Eigen::Ref<IV4Stack> cns, int node_i, int node_j, int yarn_type){
    //         if (yarn_type == WEFT)
    //         {
    //             cns(1, node_i) = node_j;
    //             cns(0, node_j) = node_i;
    //         }
    //         else
    //         {
    //             cns(3, node_i) = node_j;
    //             cns(2, node_j) = node_i;
    //         }
    //     };

    // std::vector<IV3> rods_sub;
    // // std::cout << "#nodes " << n_nodes << std::endl;
    // int new_node_cnt = n_nodes;
    // int dof_cnt = n_nodes * dof;
    // std::vector<Eigen::Triplet<T>> w_entry;

    // for (int i = 0; i < n_nodes; i++)
    //     for(int d = 0; d < dof; d++)
    //         w_entry.push_back(Eigen::Triplet<T>(i * dof + d, i * dof + d, 1.0));

    // n_nodes = n_nodes + (sub_div-1) * n_rods;
    // q.conservativeResize(dof, n_nodes);
    // normal.resize(3, n_nodes);
    // normal.setZero();
    // IV4Stack new_connections(4, n_nodes);
    // new_connections.setConstant(-1);
    
    // auto unit_yarn_map = yarn_map;
    // yarn_map.clear();
    // for (int rod_idx = 0; rod_idx < n_rods; rod_idx++)
    // {
    //     IV2 end_points = rods.col(rod_idx).template segment<2>(0);
    //     int node_i = end_points[0];
    //     int node_j = end_points[1];
        
    //     int yarn_type = rods(2, rod_idx);
        
    //     bool sign0 = connections.col(node_i).prod();
    //     int sign1 = connections.col(node_j).prod();
        
    //     // std::cout << "xi: " << q.col(node_i).transpose() << std::endl;
    //     // std::cout << "xj: "<< q.col(node_j).transpose() << std::endl;
    //     T fraction = T(1) / sub_div;
    //     bool new_node_added = false;
    //     bool left_or_bottom_bd = (connections(0, node_i) < 0 || connections(2, node_i) < 0);
    //     bool right_or_top_bd = (connections(1, node_j) < 0 || connections(3, node_j) < 0);
    //     int cnt = 0;
    //     for (int sub_cnt = 1; sub_cnt < sub_div; sub_cnt++)
    //     {
    //         T alpha = sub_cnt * fraction;

    //         // left or bottom boundary
    //         if (left_or_bottom_bd && alpha <= 0.5)
    //             continue;
    //         // right or top boundary
    //         if (right_or_top_bd && alpha >= 0.5)
    //             continue;
            
    //         if (left_or_bottom_bd)
    //             alpha = (alpha - 0.5) / 0.5;
    //         if (right_or_top_bd)
    //             alpha = alpha / 0.5;
            
    //         for(int d = 0; d < dof; d++)
    //         {
    //             if(d < dim)
    //             {
    //                 w_entry.push_back(Entry(new_node_cnt * dof + d, dof_cnt, 1));
    //                 dof_cnt++;
    //             }
    //             else
    //             {
    //                 w_entry.push_back(Entry(new_node_cnt * dof + d, node_i * dof + d, 1-alpha));
    //                 w_entry.push_back(Entry(new_node_cnt * dof + d, node_j * dof + d, alpha));
    //             }
    //         }   

    //         q.col(new_node_cnt) = 
    //             q.col(node_i) * (1 - alpha) + 
    //             q.col(node_j) * alpha;
            
    //         // std::cout << "x sub: "<< q.col(new_node_cnt).transpose() << std::endl;
    //         int n0, n1;
    //         if (cnt == 0)
    //         {
    //             n0 = node_i; n1 = new_node_cnt;
    //         }
    //         else
    //         {
    //             n0 = new_node_cnt-1; n1 = new_node_cnt;
    //         }
    //         rods_sub.push_back(IV3(n0, n1, yarn_type));
    //         yarn_map[rods_sub.size()-1] = unit_yarn_map[rod_idx];
    //         setConnection(new_connections, n0, n1, yarn_type);
    //         // dirichlet_data[new_node_cnt] = std::make_pair(TVDOF::Zero(), fix_eulerian);
    //         new_node_cnt++;
    //         new_node_added = true;
    //         cnt++;
    //     }
    //     if (new_node_added)
    //     {
    //         rods_sub.push_back(IV3(new_node_cnt-1, node_j, yarn_type));
    //         setConnection(new_connections, new_node_cnt-1, node_j, yarn_type);
    //     }
    //     else
    //     {
    //         rods_sub.push_back(IV3(node_i, node_j, yarn_type));
    //         setConnection(new_connections, node_i, node_j, yarn_type);   
    //     }
    //     yarn_map[rods_sub.size()-1] = unit_yarn_map[rod_idx];
        
    // }
    // n_rods = rods_sub.size();
    // rods.resize(3, n_rods);
    // tbb::parallel_for(0, n_rods, [&](int i){
    //     rods.col(i) = rods_sub[i];
    // });
    // connections = new_connections;

    // std::vector<int> init(5, -1);
    // for (int i = 0; i < 6; i++)
    //     pbc_bending_bn_pairs.push_back(init);
    
    // auto add4Nodes = [&](int front, int end, int yarn_id, int rod_id, std::vector<std::vector<int>>& pairs)
    // {
    //     if (rods(0, rod_id) == front)
    //     {
    //         pbc_bending_bn_pairs[yarn_id][0] = front;
    //         pbc_bending_bn_pairs[yarn_id][1] = rods(1, rod_id);
    //         pbc_bending_bn_pairs[yarn_id][4] = rods(2, rod_id);
    //     }
    //     if (rods(1, rod_id) == end)
    //     {
    //         pbc_bending_bn_pairs[yarn_id][3] = end;
    //         pbc_bending_bn_pairs[yarn_id][2] = rods(0, rod_id);
    //         pbc_bending_bn_pairs[yarn_id][4] = rods(2, rod_id);
    //     }
    // };

    // for (int i = 0; i < n_rods; i++)
    // {
    //     add4Nodes(0, 1, 0, i, pbc_bending_bn_pairs);
    //     add4Nodes(7, 8, 1, i, pbc_bending_bn_pairs);
    //     add4Nodes(14, 15, 2, i, pbc_bending_bn_pairs);
    //     add4Nodes(16, 17, 3, i, pbc_bending_bn_pairs);
    //     add4Nodes(9, 10, 4, i, pbc_bending_bn_pairs);
    //     add4Nodes(2, 3, 5, i, pbc_bending_bn_pairs);
    // }
    // q.conservativeResize(dof, new_node_cnt);
    // connections.conservativeResize(dof, new_node_cnt);
    // n_nodes = new_node_cnt;
    
    // normal.conservativeResize(dof, new_node_cnt);
    // is_end_nodes = std::vector<bool>(n_nodes, false);

    // dof_offsets.resize(n_nodes, 0);
    // for(int i = 0; i < 21; i++)
    //     dof_offsets[i] = i * dof;
    
    // n_pb_cons = 0;
    // iteratePBCReferencePairs([&](int yarn_type, int node_i, int node_j){
    //     int ref_i = pbc_ref_unique[yarn_type](0);
    //     int ref_j = pbc_ref_unique[yarn_type](1);

    //     if (ref_i == node_i && ref_j == node_j)
    //         return;
    //     n_pb_cons++;
    // });

    
    // q0 = q;
    // n_dof = dof_cnt;
    // W = StiffnessMatrix(n_nodes * dof, n_dof);
    // W.setFromTriplets(w_entry.begin(), w_entry.end());
    
    // // n_dof = n_nodes * dof;
    // // W = StiffnessMatrix(n_nodes * dof, n_nodes * dof);
    // // W.setIdentity();

    // // do not move out the unit cell
    // slide_over_n_rods = IV2(std::floor(sub_div * 0.25), std::floor(sub_div * 0.25));
    // T rod_length = (q0.col(rods.col(0)(0)).template segment<dim>(0) - 
    //     q0.col(rods.col(0)(1)).template segment<dim>(0)).norm();
    // tunnel_u = slide_over_n_rods[0] * rod_length;
    // tunnel_v = tunnel_u;
}

template<class T, int dim>
void EoLRodSim<T, dim>::buildPlanePeriodicBCScene3x3()
{
    // // pbc_ref_unique.clear();
    // dirichlet_data.clear();
    // // pbc_ref.clear();
    // // pbc_bending_pairs.clear();
    // yarns.clear();

    
    // add_stretching = true;
    // add_bending = true;
    
    // kc = 1e8;
    // add_pbc = true;

    // if(disable_sliding)
    // {
    //     add_shearing = true;
    //     add_eularian_reg = false;
    //     k_pbc = 1e8;
    //     k_strain = 1e8;
    // }
    // else
    // {
    //     add_shearing = false;
    //     add_eularian_reg = true;
    //     ke = 1;
    //     k_pbc = 1e8;
    //     k_strain = 1e8;
    // }
    
    
    // kr = 1e3;
    
    // n_nodes = 21;
    // n_rods = 24;

    // q = DOFStack(dof, n_nodes); q.setZero();
    // rods = IV3Stack(3, n_rods); rods.setZero();
    // connections = IV4Stack(4, n_nodes).setOnes() * -1;
    
    // normal = TV3Stack(3, n_rods);
    // normal.setZero();

    // T u_delta = 1.0 / 3.0, v_delta = 1.0 / 3.0;
    // int cnt = 0;
    // is_end_nodes.resize(n_nodes, true);

    // if constexpr (dim == 2)
    // {
    //     for (int i = 0; i < 3; i++)
    //     {
    //         q.col(cnt).template segment<dim>(0) = TV2(0, 0.5 * v_delta + i * v_delta );
    //         q.col(cnt++).template segment<2>(dim) = TV2(0, 0.5 * v_delta + i * v_delta);

    //         q.col(cnt).template segment<dim>(0) = TV2(1, 0.5 * v_delta + i * v_delta );
    //         q.col(cnt++).template segment<2>(dim) = TV2(1,0.5 * v_delta + i * v_delta);

    //         q.col(cnt).template segment<dim>(0) = TV2(0.5 * u_delta + i * u_delta, 0);
    //         q.col(cnt++).template segment<2>(dim) = TV2(0.5 * u_delta + i * u_delta, 0);

    //         q.col(cnt).template segment<dim>(0) = TV2(0.5 * u_delta + i * u_delta, 1);
    //         q.col(cnt++).template segment<2>(dim) = TV2(0.5 * u_delta + i * u_delta, 1);

    //         for (int j = 0; j < 3; j++)
    //         {
    //             is_end_nodes[cnt] = false;
    //             q.col(cnt).template segment<dim>(0) = TV2(0.5 * u_delta + i * u_delta, 0.5 * v_delta + j * v_delta);
    //             q.col(cnt++).template segment<2>(dim) = TV2(0.5 * u_delta + i * u_delta, 0.5 * v_delta + j * v_delta);
    //         }
    //     }
    //     assert(cnt == n_nodes);
    //     cnt = 0;
        
    //     rods.col(cnt++) = IV3(0, 4, WARP); rods.col(cnt++) = IV3(4, 11, WARP); rods.col(cnt++) = IV3(11, 18, WARP);rods.col(cnt++) = IV3(18, 1, WARP);
    //     rods.col(cnt++) = IV3(7, 5, WARP); rods.col(cnt++) = IV3(5, 12, WARP); rods.col(cnt++) = IV3(12, 19, WARP); rods.col(cnt++) = IV3(19, 8, WARP); 
    //     rods.col(cnt++) = IV3(14, 6, WARP); rods.col(cnt++) = IV3(6, 13, WARP); rods.col(cnt++) = IV3(13, 20, WARP); rods.col(cnt++) = IV3(20, 15, WARP); 

    //     rods.col(cnt++) = IV3(2, 4, WEFT); rods.col(cnt++) = IV3(4, 5, WEFT); rods.col(cnt++) = IV3(5, 6, WEFT); rods.col(cnt++) = IV3(6, 3, WEFT); 
    //     rods.col(cnt++) = IV3(9, 11, WEFT); rods.col(cnt++) = IV3(11, 12, WEFT);  rods.col(cnt++) = IV3(12, 13, WEFT); rods.col(cnt++) = IV3(13, 10, WEFT); 
    //     rods.col(cnt++) = IV3(16, 18, WEFT); rods.col(cnt++) = IV3(18, 19, WEFT); rods.col(cnt++) = IV3(19, 20, WEFT); rods.col(cnt++) = IV3(20, 17, WEFT);
        
    //     for(int i = 0; i < 4; i++)
    //         for(int j = 0; j < 6;j++)
    //             yarn_map[j * 4 + i] = j;

    //     assert(cnt == n_rods);

    //     auto set_left_right = [&](Eigen::Ref<IV4Stack> connections, int idx, int left){
    //         connections(0, idx) = left;
    //         connections(1, left) = idx;
    //     };
    //     auto set_top_bottom = [&](Eigen::Ref<IV4Stack> connections, int idx, int top){
    //         connections(3, idx) = top;
    //         connections(2, top) = idx;
    //     };
        

    //     set_top_bottom(connections, 18, 1); set_top_bottom(connections, 19, 8); set_top_bottom(connections, 20, 15);
    //     set_top_bottom(connections, 11, 18); set_top_bottom(connections, 12, 19); set_top_bottom(connections, 13, 20);
    //     set_top_bottom(connections, 4, 11); set_top_bottom(connections, 5, 12); set_top_bottom(connections, 6, 13);
    //     set_top_bottom(connections, 0, 4); set_top_bottom(connections, 7, 5); set_top_bottom(connections, 14, 6);

    //     set_left_right(connections, 4, 2); set_left_right(connections, 11, 9); set_left_right(connections, 18, 16);
    //     set_left_right(connections, 5, 4); set_left_right(connections, 12, 11); set_left_right(connections, 19, 18);
    //     set_left_right(connections, 6, 5); set_left_right(connections, 13, 12); set_left_right(connections, 20, 19);
    //     set_left_right(connections, 3, 6); set_left_right(connections, 10, 13); set_left_right(connections, 17, 20);

    //     checkConnections();
        
    //     pbc_ref_unique.push_back(IV2(0, 1));
    //     pbc_ref_unique.push_back(IV2(2, 3));

    //     if (disable_sliding)
    //     {
    //         for(int i = 0; i < n_nodes; i++)
    //             dirichlet_data[i] = std::make_pair(TVDOF::Zero(), fix_eulerian);
    //         dirichlet_data[12] = std::make_pair(TVDOF::Zero(), fix_all);
    //     }
    //     else
    //     {
    //         dirichlet_data[2] = std::make_pair(TVDOF::Zero(), fix_eulerian);
    //         dirichlet_data[9] = std::make_pair(TVDOF::Zero(), fix_eulerian);
    //         dirichlet_data[16] = std::make_pair(TVDOF::Zero(), fix_eulerian);
    //         dirichlet_data[3] = std::make_pair(TVDOF::Zero(), fix_eulerian);
    //         dirichlet_data[10] = std::make_pair(TVDOF::Zero(), fix_eulerian);
    //         dirichlet_data[17] = std::make_pair(TVDOF::Zero(), fix_eulerian);

    //         dirichlet_data[1] = std::make_pair(TVDOF::Zero(), fix_eulerian);
    //         dirichlet_data[8] = std::make_pair(TVDOF::Zero(), fix_eulerian);
    //         dirichlet_data[15] = std::make_pair(TVDOF::Zero(), fix_eulerian);
    //         dirichlet_data[0] = std::make_pair(TVDOF::Zero(), fix_eulerian);
    //         dirichlet_data[7] = std::make_pair(TVDOF::Zero(), fix_eulerian);
    //         dirichlet_data[14] = std::make_pair(TVDOF::Zero(), fix_eulerian);

    //         dirichlet_data[12] = std::make_pair(TVDOF::Zero(), fix_lagrangian);
    //         sliding_nodes = {4, 11, 18, 5, 12, 19, 6, 13, 20};

    //         // dirichlet_data[12] = std::make_pair(TVDOF::Zero(), fix_all);
    //         // sliding_nodes = {4, 11, 18, 5, 19, 6, 13, 20};
            
    //         // dirichlet_data[11] = std::make_pair(TVDOF::Zero(), fix_eulerian);
    //         // dirichlet_data[13] = std::make_pair(TVDOF::Zero(), fix_eulerian);
    //         // dirichlet_data[5] = std::make_pair(TVDOF::Zero(), fix_eulerian);
    //         // dirichlet_data[19] = std::make_pair(TVDOF::Zero(), fix_eulerian);
    //         // sliding_nodes = {4, 18, 12, 6, 20};

    //         // dirichlet_data[4] = std::make_pair(TVDOF::Zero(), fix_eulerian);
    //         // dirichlet_data[18] = std::make_pair(TVDOF::Zero(), fix_eulerian);
    //         // dirichlet_data[20] = std::make_pair(TVDOF::Zero(), fix_eulerian);
    //         // dirichlet_data[6] = std::make_pair(TVDOF::Zero(), fix_eulerian);
    //         // sliding_nodes = {5, 13, 12, 11, 19};
    //         // sliding_nodes = {5, 13, 11, 19};

    //     }
            
        
    //     // add reference pairs
    //     pbc_ref.push_back(std::make_pair(WARP, IV2(0, 1)));
    //     pbc_ref.push_back(std::make_pair(WARP, IV2(7, 8)));
    //     pbc_ref.push_back(std::make_pair(WARP, IV2(14, 15)));
    //     pbc_ref.push_back(std::make_pair(WEFT, IV2(2, 3)));
    //     pbc_ref.push_back(std::make_pair(WEFT, IV2(9, 10)));
    //     pbc_ref.push_back(std::make_pair(WEFT, IV2(16, 17)));

    //     // add periodic pairs to shift along peridic direction to compute bending
    //     pbc_bending_pairs.push_back({0, 4, 11, 18, 1, 0});
    //     pbc_bending_pairs.push_back({7, 5, 12, 19, 8, 1});
    //     pbc_bending_pairs.push_back({14, 6, 13, 20, 15, 2});

    //     pbc_bending_pairs.push_back({16, 18, 19, 20, 17, 5});
    //     pbc_bending_pairs.push_back({9, 11, 12, 13, 10, 4});
    //     pbc_bending_pairs.push_back({2, 4, 5, 6, 3, 3});

        

    //     // for coloring
    //     yarns.push_back({0, 4, 11, 18, 1, WARP});
    //     yarns.push_back({7, 5, 12, 19, 8, WARP});
    //     yarns.push_back({14, 6, 13, 20, 15, WARP});

    //     yarns.push_back({16, 18, 19, 20, 17, WEFT});
    //     yarns.push_back({9, 11, 12, 13, 10, WEFT});
    //     yarns.push_back({2, 4, 5, 6, 3, WEFT});

        
    // }
    // else
    // {
    //     std::cout << "3D version this is not implemented" << std::endl;
    //     std::exit(0);
    // }
    // q *= 0.03;
    // q0 = q;
    
    // n_dof = n_nodes * dof;
    // W = StiffnessMatrix(n_nodes * dof, n_dof);
    // W.setIdentity();

    // // add curvature function
    
    // Vector<T, dim + 1> v0 = q.col(0).template segment<dim + 1>(0);
    // Vector<T, dim + 1> v1 = q.col(1).template segment<dim + 1>(0);

    // Vector<T, dim + 1> v2 = q.col(2).template segment<dim + 1>(0);
    // v2[dim] = q(dim + 1, 2);
    // Vector<T, dim + 1> v3 = q.col(3).template segment<dim + 1>(0);
    // v3[dim] = q(dim + 1, 3);

    // curvature_functions.push_back(new LineCurvature<T, dim>(v0, v1));
    // curvature_functions.push_back(new LineCurvature<T, dim>(v2, v3));
}

template<class T, int dim>
void EoLRodSim<T, dim>::checkConnections()
{
    for(int i = 0; i < n_nodes; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            if (connections(j, i) >= n_nodes)
            {
                std::cout << "connections(" << j << ", " << i << ") is larger than " << n_nodes - 1 << std::endl;
            }
        }
    }
    std::cout << "no connection violation" << std::endl;
}

template<class T, int dim>
void EoLRodSim<T, dim>::buildSceneFromUnitPatch(int patch_id)
{
    UnitPatch<T, dim> unit_patch(*this);
    unit_patch.buildScene(patch_id);
}

template class EoLRodSim<double, 3>;
template class EoLRodSim<double, 2>;

//not working for float yet
// template class EoLRodSim<float>;