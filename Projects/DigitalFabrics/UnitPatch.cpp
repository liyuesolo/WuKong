#include "UnitPatch.h"

template<class T, int dim>
void UnitPatch<T, dim>::buildScene(int patch_type)
{
    if (patch_type == 0)
        build3x3StraightRod();
    else if (patch_type == 1)
        buildStraightAndSineScene(64);
    else if (patch_type == 2)
        buildStraightAndHemiCircleScene(8);
    else if (patch_type == 3)
        buildStraightYarnScene(8);
    else if (patch_type == 4)
        buildTwoRodsScene(2);
    else if (patch_type == 5)
        buildSlidingTestScene(8);

}

template<class T, int dim>
void UnitPatch<T, dim>::addRods(std::vector<int>& nodes, int yarn_type, int& cnt, int yarn_idx)
{
    std::vector<int> yarn;
    for (int i = 0; i < nodes.size() - 1; i++)
    {
        sim.yarn_map[cnt] = yarn_idx;
        rods.col(cnt++) = IV3(nodes[i], nodes[i+1], yarn_type);
        if (yarn_type == WEFT)
            set_left_right(nodes[i+1], nodes[i]);
        else
            set_top_bottom(nodes[i], nodes[i+1]);
        yarn.push_back(nodes[i]);
    }
    yarn.push_back(nodes[nodes.size()-1]);
    yarn.push_back(yarn_type);
    
    sim.yarns.push_back(yarn);
}

template<class T, int dim>
void UnitPatch<T, dim>::buildStraightYarnScene(int sub_div)
{
    clearSimData();
    sim.n_nodes = 10;
    sim.n_rods = 9;

    q = DOFStack(sim.dof, sim.n_nodes); q.setZero();
    rods = IV3Stack(3, sim.n_rods); rods.setZero();
    connections = IV4Stack(4, sim.n_nodes).setOnes() * -1;

    sim.normal = TV3Stack(3, sim.n_rods);
    sim.normal.setZero();
    sim.subdivide = true;
    // sim.add_pbc_bending = false;
    if constexpr (dim == 2)
    {
        
        {
            q.col(0).template segment<dim>(0) = TV2(0.25, 0);
            q.col(1).template segment<dim>(0) = TV2(0.25, 0.25);
            q.col(2).template segment<dim>(0) = TV2(0.25, 0.75);
            q.col(3).template segment<dim>(0) = TV2(0.25, 1);
            q.col(4).template segment<dim>(0) = TV2(0.75, 0);
            q.col(5).template segment<dim>(0) = TV2(0.75, 0.25);
            q.col(6).template segment<dim>(0) = TV2(0.75, 0.75);
            q.col(7).template segment<dim>(0) = TV2(0.75, 1.0);
            q.col(8).template segment<dim>(0) = TV2(0, 0.25);
            q.col(9).template segment<dim>(0) = TV2(1.0, 0.25);

            q.block(dim, 0, 2, 10) = q.block(0, 0, 2, 10);
            
            int cnt = 0;
            std::vector<int> rod0 = {0, 1, 2, 3}, 
                             rod1 = {4, 5, 6, 7},
                             rod2 = {2, 6},
                             rod3 = {8, 1},
                             rod4 = {5, 9};
            
            addRods(rod0, WEFT, cnt, 0);
            addRods(rod1, WEFT, cnt, 1);
            addRods(rod2, WARP, cnt, 2);
            addRods(rod3, WARP, cnt, 3);
            addRods(rod4, WARP, cnt, 4);
            
            sim.q0 = q;
            sim.n_dof = sim.n_nodes * sim.dof;

            
            sim.pbc_ref.push_back(std::make_pair(WARP, IV2(8, 9)));
            sim.pbc_ref.push_back(std::make_pair(WEFT, IV2(0, 3)));
            sim.pbc_ref.push_back(std::make_pair(WEFT, IV2(4, 7)));

            sim.pbc_ref_unique.push_back(IV2(8, 9));
            sim.pbc_ref_unique.push_back(IV2(0, 3));

            if (sim.disable_sliding)
            {
                for(int i = 0; i < sim.n_nodes; i++)
                    sim.dirichlet_data[i] = std::make_pair(TVDOF::Zero(), sim.fix_eulerian);
                sim.dirichlet_data[0] = std::make_pair(TVDOF::Zero(), sim.fix_all);
            }
            else
            {
                
                sim.dirichlet_data[8]= std::make_pair(TVDOF::Zero(), sim.fix_eulerian);
                sim.dirichlet_data[9]= std::make_pair(TVDOF::Zero(), sim.fix_eulerian);
                sim.dirichlet_data[3]= std::make_pair(TVDOF::Zero(), sim.fix_eulerian);
                sim.dirichlet_data[7]= std::make_pair(TVDOF::Zero(), sim.fix_eulerian);
                sim.dirichlet_data[4] = std::make_pair(TVDOF::Zero(), sim.fix_eulerian);
                sim.dirichlet_data[1]= std::make_pair(TVDOF::Zero(), sim.fix_u);
                sim.dirichlet_data[5] = std::make_pair(TVDOF::Zero(), sim.fix_u);
                sim.dirichlet_data[6] = std::make_pair(TVDOF::Zero(), sim.fix_u);
                sim.dirichlet_data[2] = std::make_pair(TVDOF::Zero(), sim.fix_u);
                
                sim.dirichlet_data[0] = std::make_pair(TVDOF::Zero(), sim.fix_all);
            }
            sim.n_dof = sim.n_nodes * sim.dof;            
            sim.sliding_nodes = {1, 5, 2, 6};
        }
        if(sub_div > 1)
        {
            auto unit_yarn_map = sim.yarn_map;
            sim.yarn_map.clear();

            std::vector<IV3> rods_sub;
            // std::cout << "#nodes " << n_nodes << std::endl;
            int new_node_cnt = sim.n_nodes;
            int dof_cnt = sim.n_nodes * sim.dof;
            std::vector<Eigen::Triplet<T>> w_entry;

            for (int i = 0; i < sim.n_nodes; i++)
                for(int d = 0; d < sim.dof; d++)
                    w_entry.push_back(Eigen::Triplet<T>(i * sim.dof + d, i * sim.dof + d, 1.0));
                
            
            sim.n_nodes = sim.n_nodes + (sub_div-1) * sim.n_rods;
            q.conservativeResize(sim.dof, sim.n_nodes);
            sim.normal.resize(3, sim.n_nodes);
            sim.normal.setZero();
            IV4Stack new_connections(4, sim.n_nodes);
            new_connections.setConstant(-1);

        
            auto setConnection = [&](Eigen::Ref<IV4Stack> cns, int node_i, int node_j, int yarn_type){
                    if (yarn_type == WEFT)
                    {
                        cns(1, node_i) = node_j;
                        cns(0, node_j) = node_i;
                    }
                    else
                    {
                        cns(3, node_i) = node_j;
                        cns(2, node_j) = node_i;
                    }
                };

            for (int rod_idx = 0; rod_idx < sim.n_rods; rod_idx++)
            {
                IV2 end_points = rods.col(rod_idx).template segment<2>(0);
                int node_i = end_points[0];
                int node_j = end_points[1];
                
                int yarn_type = rods(2, rod_idx);
                
                bool sign0 = connections.col(node_i).prod();
                int sign1 = connections.col(node_j).prod();
                
                // std::cout << "xi: " << q.col(node_i).transpose() << std::endl;
                // std::cout << "xj: "<< q.col(node_j).transpose() << std::endl;
                T fraction = T(1) / sub_div;
                bool new_node_added = false;
                bool left_or_bottom_bd = ((connections(0, node_i) < 0 && connections(1, node_i) != -1 && connections(2, node_i) == -1 && connections(3, node_i) == -1) 
                                            || (connections(2, node_i) < 0 && connections(3, node_i) != -1 && connections(0, node_i) == -1 && connections(1, node_i) == -1));
                bool right_or_top_bd = ((connections(1, node_j) < 0 && connections(0, node_j) != -1 && connections(2, node_j) == -1 && connections(3, node_j) == -1) 
                                            || (connections(3, node_j) < 0 && connections(2, node_j) != -1 && connections(0, node_j) == -1 && connections(1, node_j) == -1));
                int cnt = 0;
                for (int sub_cnt = 1; sub_cnt < sub_div; sub_cnt++)
                {
                    T alpha = sub_cnt * fraction;
                    T mid = 0.5;
                    // left or bottom boundary
                    if (left_or_bottom_bd && (alpha <= mid))
                        continue;
                    // right or top boundary
                    if (right_or_top_bd && (alpha >= mid))
                        continue;
                    
                    if (left_or_bottom_bd)
                        alpha = (alpha - mid) / mid;
                    if (right_or_top_bd)
                        alpha = alpha / mid;
                    // std::cout << "alpha: " << alpha << " " << left_or_bottom_bd << std::endl; 
                    
                    q.col(new_node_cnt) = 
                        q.col(node_i) * (1 - alpha) + 
                        q.col(node_j) * alpha;  

                    for(int d = 0; d < dim; d++)
                    {
                        w_entry.push_back(Entry(new_node_cnt * sim.dof + d, dof_cnt, 1));
                        dof_cnt++;
                    }
                    for(int d = dim; d < sim.dof; d++)
                    {
                        w_entry.push_back(Entry(new_node_cnt * sim.dof + d, node_i * sim.dof + d, 1-alpha));
                        w_entry.push_back(Entry(new_node_cnt * sim.dof + d, node_j * sim.dof + d, alpha));
                    }
                    int n0, n1;
                    if (cnt == 0)
                    {
                        n0 = node_i; n1 = new_node_cnt;
                    }
                    else
                    {
                        n0 = new_node_cnt-1; n1 = new_node_cnt;
                    }
                    rods_sub.push_back(IV3(n0, n1, yarn_type));
                    setConnection(new_connections, n0, n1, yarn_type);
                    sim.yarn_map[rods_sub.size()-1] = unit_yarn_map[rod_idx];
                    // dirichlet_data[new_node_cnt] = std::make_pair(TVDOF::Zero(), fix_eulerian);
                    new_node_cnt++;
                    new_node_added = true;
                    cnt++;    
                }
                if (new_node_added)
                {
                    rods_sub.push_back(IV3(new_node_cnt-1, node_j, yarn_type));
                    setConnection(new_connections, new_node_cnt-1, node_j, yarn_type);
                }
                else
                {
                    rods_sub.push_back(IV3(node_i, node_j, yarn_type));
                    setConnection(new_connections, node_i, node_j, yarn_type);   
                }
                sim.yarn_map[rods_sub.size()-1] = unit_yarn_map[rod_idx];
                
            }

            sim.n_rods = rods_sub.size();
            rods.resize(3, sim.n_rods);
            tbb::parallel_for(0, sim.n_rods, [&](int i){
                rods.col(i) = rods_sub[i];
            });
            connections = new_connections;

            int n_bending_pairs = 3;
            
            std::vector<int> init(sim.N_PBC_BENDING_ELE, -1);
            for (int i = 0; i < n_bending_pairs; i++)
                sim.pbc_bending_bn_pairs.push_back(init);

            for (int i = 0; i < sim.n_rods; i++)
            {
                add4Nodes(8, 9, 0, i);
                add4Nodes(0, 3, 1, i);
                add4Nodes(4, 7, 2, i);
            }
            
            assert(sim.pbc_bending_bn_pairs == n_bending_pairs);

            q.conservativeResize(sim.dof, new_node_cnt);
            connections.conservativeResize(sim.dof, new_node_cnt);
            sim.n_nodes = new_node_cnt;
            
            sim.normal.conservativeResize(sim.dof, new_node_cnt);
            
            sim.q0 = q;
            sim.n_dof = dof_cnt;
            sim.W = StiffnessMatrix(sim.n_nodes * sim.dof, sim.n_dof);
            sim.W.setFromTriplets(w_entry.begin(), w_entry.end());

        }
        else
        {
            sim.q0 = q;
            sim.W = StiffnessMatrix(sim.n_nodes * sim.dof, sim.n_dof);
            sim.W.setIdentity();
        }
        
        sim.is_end_nodes = std::vector<bool>(sim.n_nodes, false);

        T rod_length = (sim.q0.col(rods.col(0)(0)).template segment<dim>(0) - 
            sim.q0.col(rods.col(0)(1)).template segment<dim>(0)).norm();

        sim.tunnel_u = rod_length * 0.25 * T(sub_div);
        sim.tunnel_v = rod_length * 0.25 * T(sub_div);


        sim.curvature_functions.push_back(new LineCurvature<T, dim>());
        sim.curvature_functions.push_back(new LineCurvature<T, dim>());

        // std::cout << q.transpose() << std::endl;
        // std::cout << connections.transpose() << std::endl;
    }
}

template<class T, int dim>
void UnitPatch<T, dim>::buildSlidingTestScene(int sub_div)
{
    bool use_analytical = true;
    clearSimData();
    T r = 0.25;
    sim.n_nodes = 7;
    sim.n_rods = 6;

    sim.add_pbc_bending = false;

    q = DOFStack(sim.dof, sim.n_nodes); q.setZero();
    rods = IV3Stack(3, sim.n_rods); rods.setZero();
    connections = IV4Stack(4, sim.n_nodes).setOnes() * -1;

    sim.normal = TV3Stack(3, sim.n_rods);
    sim.normal.setZero();
    sim.subdivide = true;

    if constexpr (dim == 2)
    {
        
        {
            q.col(0).template segment<dim>(0) = TV2(0.25, 0.125);
            q.col(1).template segment<dim>(0) = TV2(0.25, 0.875);
            q.block(dim, 0, 2, 2) = q.block(0, 0, 2, 2);

            q.col(2).template segment<dim>(0) = TV2(0, 0.5);
            q.col(2).template segment<2>(dim) = TV2(0, 0.5);
            q.col(3).template segment<dim>(0) = TV2(0.25, 0.75);
            q.col(3).template segment<2>(dim) = TV2(r * M_PI/2, 0.75);
            q.col(4).template segment<dim>(0) = TV2(0.5, 0.5);
            q.col(4).template segment<2>(dim) = TV2(r * M_PI, 0.5);

            q.col(5).template segment<dim>(0) = TV2(0.25, 0.25);
            q.col(6).template segment<dim>(0) = TV2(0.25, 0.5);
            q.block(dim, 5, 2, 2) = q.block(0, 5, 2, 2);
            
            int cnt = 0;
            std::vector<int> rod0 = {2, 3, 4}, 
                             rod1 = {0, 5, 6, 3, 1};
            
            addRods(rod0, WARP, cnt, 0);
            addRods(rod1, WEFT, cnt, 1);
            
            sim.q0 = q;
            sim.n_dof = sim.n_nodes * sim.dof;

            sim.pbc_ref.push_back(std::make_pair(WARP, IV2(2, 4)));
            sim.pbc_ref.push_back(std::make_pair(WEFT, IV2(0, 1)));

            sim.pbc_ref_unique.push_back(IV2(2, 4));
            sim.pbc_ref_unique.push_back(IV2(0, 1));

            if (sim.disable_sliding)
            {
                for(int i = 0; i < sim.n_nodes; i++)
                    sim.dirichlet_data[i] = std::make_pair(TVDOF::Zero(), sim.fix_eulerian);
                sim.dirichlet_data[4] = std::make_pair(TVDOF::Zero(), sim.fix_all);
            }
            else
            {
                
                sim.dirichlet_data[0]= std::make_pair(TVDOF::Zero(), sim.fix_eulerian);
                sim.dirichlet_data[1]= std::make_pair(TVDOF::Zero(), sim.fix_eulerian);

                sim.dirichlet_data[5]= std::make_pair(TVDOF::Zero(), sim.fix_eulerian);
                sim.dirichlet_data[6]= std::make_pair(TVDOF::Zero(), sim.fix_eulerian);
            
                sim.dirichlet_data[4] = std::make_pair(TVDOF::Zero(), sim.fix_all);
                // sim.dirichlet_data[3] = std::make_pair(TVDOF::Zero(), sim.fix_eulerian);
                TVDOF shift_x; shift_x.setZero();
                shift_x(2) = -0.05;
                // sim.dirichlet_data[3] = std::make_pair(shift_x, sim.fix_eulerian);
                sim.dirichlet_data[2] = std::make_pair(TVDOF::Zero(), sim.fix_all);
            }
            sim.n_dof = sim.n_nodes * sim.dof;
            // sim.W = StiffnessMatrix(sim.n_nodes * sim.dof, sim.n_dof);
            // sim.W.setIdentity();
            
        }
        auto unit_yarn_map = sim.yarn_map;
        sim.yarn_map.clear();
        if(true)
        {
            std::vector<IV3> rods_sub;
            // std::cout << "#nodes " << n_nodes << std::endl;
            int new_node_cnt = sim.n_nodes;
            int dof_cnt = sim.n_nodes * sim.dof;
            std::vector<Eigen::Triplet<T>> w_entry;

            for (int i = 0; i < sim.n_nodes; i++)
                for(int d = 0; d < sim.dof; d++)
                    w_entry.push_back(Eigen::Triplet<T>(i * sim.dof + d, i * sim.dof + d, 1.0));
                
            
            sim.n_nodes = sim.n_nodes + (sub_div-1) * sim.n_rods;
            q.conservativeResize(sim.dof, sim.n_nodes);
            sim.normal.resize(3, sim.n_nodes);
            sim.normal.setZero();
            IV4Stack new_connections(4, sim.n_nodes);
            new_connections.setConstant(-1);

        
            auto setConnection = [&](Eigen::Ref<IV4Stack> cns, int node_i, int node_j, int yarn_type){
                    if (yarn_type == WEFT)
                    {
                        cns(1, node_i) = node_j;
                        cns(0, node_j) = node_i;
                    }
                    else
                    {
                        cns(3, node_i) = node_j;
                        cns(2, node_j) = node_i;
                    }
                };

            for (int rod_idx = 0; rod_idx < sim.n_rods; rod_idx++)
            {
                IV2 end_points = rods.col(rod_idx).template segment<2>(0);
                int node_i = end_points[0];
                int node_j = end_points[1];
                
                int yarn_type = rods(2, rod_idx);
                
                bool sign0 = connections.col(node_i).prod();
                int sign1 = connections.col(node_j).prod();
                
                // std::cout << "xi: " << q.col(node_i).transpose() << std::endl;
                // std::cout << "xj: "<< q.col(node_j).transpose() << std::endl;
                T fraction = T(1) / sub_div;
                bool new_node_added = false;
                bool left_or_bottom_bd = ((connections(0, node_i) < 0 && connections(1, node_i) != -1) || (connections(2, node_i) < 0 && connections(3, node_i) != -1));
                bool right_or_top_bd = ((connections(1, node_j) < 0 && connections(0, node_j) != -1) || (connections(3, node_j) < 0 && connections(2, node_j) != -1));
                int cnt = 0;
                T arc_length_ij = r * M_PI/2.0;
                int prev_node = node_i;
                std::vector<int> sub_nodes;
                T arc_length_sum = 0;
                for (int sub_cnt = 1; sub_cnt < sub_div; sub_cnt++)
                {
                    T alpha = sub_cnt * fraction;
                    if(yarn_type == WEFT)
                    {
                        T mid = 0.5;
                        // left or bottom boundary
                        if (left_or_bottom_bd && (alpha <= mid))
                            continue;
                        // right or top boundary
                        if (right_or_top_bd && (alpha >= mid))
                            continue;
                        
                        if (left_or_bottom_bd)
                            alpha = (alpha - mid) / mid;
                        if (right_or_top_bd)
                            alpha = alpha / mid;
                        // std::cout << "alpha: " << alpha << " " << left_or_bottom_bd << std::endl; 
                        
                        q.col(new_node_cnt) = 
                            q.col(node_i) * (1 - alpha) + 
                            q.col(node_j) * alpha;      
                    }
                    else
                    {
                        T theta = alpha * M_PI / 2;
                        if(q(0, node_i) == 0)
                        {
                            q.col(new_node_cnt).template segment<dim>(0) = TV(0.25 - r * std::cos(theta), r*std::sin(theta) + 0.5);
                            if (use_analytical) q(dim, new_node_cnt) = r * theta;
                            q(dim + 1, new_node_cnt) = q(dim + 1, node_i) * (1 - alpha) + q(dim + 1, node_j) * alpha;
                        }
                        else if(q(0, node_i) == 0.25)
                        {   
                            q.col(new_node_cnt).template segment<dim>(0) = TV(0.25 + r * std::cos(M_PI/2.0 - theta), r*std::sin(M_PI/2.0 - theta) + 0.5);
                            if (use_analytical) q(dim, new_node_cnt) = r * M_PI/2 + r * theta;
                            q(dim + 1, new_node_cnt) = q(dim + 1, node_i) * (1 - alpha) + q(dim + 1, node_j) * alpha;
                        }
                        if(!use_analytical)
                        {
                            T l_dis = LDis(prev_node, new_node_cnt);
                            q(dim, new_node_cnt) = q(dim, prev_node) + l_dis;
                            arc_length_sum += l_dis;
                            prev_node = new_node_cnt;
                            sub_nodes.push_back(new_node_cnt);
                        }
                    }

                    for(int d = 0; d < dim; d++)
                    {
                        w_entry.push_back(Entry(new_node_cnt * sim.dof + d, dof_cnt, 1));
                        dof_cnt++;
                    }
                    if (use_analytical)
                    {
                        for(int d = dim; d < sim.dof; d++)
                        {
                            w_entry.push_back(Entry(new_node_cnt * sim.dof + d, node_i * sim.dof + d, 1-alpha));
                            w_entry.push_back(Entry(new_node_cnt * sim.dof + d, node_j * sim.dof + d, alpha));
                        }
                    }
                    else
                    {
                        w_entry.push_back(Entry(new_node_cnt * sim.dof + dim + 1, node_i * sim.dof + dim + 1, 1-alpha));
                        w_entry.push_back(Entry(new_node_cnt * sim.dof + dim + 1, node_j * sim.dof + dim + 1, alpha));
                    }

                    // std::cout << "x sub sine: "<< q.col(new_node_cnt).transpose() << std::endl;
                    int n0, n1;
                    if (cnt == 0)
                    {
                        n0 = node_i; n1 = new_node_cnt;
                    }
                    else
                    {
                        n0 = new_node_cnt-1; n1 = new_node_cnt;
                    }
                    rods_sub.push_back(IV3(n0, n1, yarn_type));
                    setConnection(new_connections, n0, n1, yarn_type);
                    sim.yarn_map[rods_sub.size()-1] = unit_yarn_map[rod_idx];
                    // dirichlet_data[new_node_cnt] = std::make_pair(TVDOF::Zero(), fix_eulerian);
                    new_node_cnt++;
                    new_node_added = true;
                    cnt++;
                }
                if (new_node_added)
                {
                    rods_sub.push_back(IV3(new_node_cnt-1, node_j, yarn_type));
                    setConnection(new_connections, new_node_cnt-1, node_j, yarn_type);
                    if(!use_analytical && yarn_type == WARP)
                    {
                        arc_length_sum += LDis(new_node_cnt-1, node_j);
                        for (int id : sub_nodes)
                        {
                            T alpha = (q(2, id) - q(2, node_i)) / arc_length_sum;
                            // std::cout << "id: " << id << " node_i: " << node_i << " dis " << q(2, id) - q(2, node_i) << " " << arc_length_sum << " " << alpha << std::endl;
                            w_entry.push_back(Entry(id * sim.dof + dim, node_i * sim.dof + dim, 1-alpha));
                            w_entry.push_back(Entry(id * sim.dof + dim, node_j * sim.dof + dim, alpha));

                        }
                        q(dim, node_j) = q(dim, new_node_cnt - 1) + LDis(new_node_cnt-1, node_j);  
                    }
                }
                else
                {
                    rods_sub.push_back(IV3(node_i, node_j, yarn_type));
                    setConnection(new_connections, node_i, node_j, yarn_type);   
                }
                sim.yarn_map[rods_sub.size()-1] = unit_yarn_map[rod_idx];
                
            }

            sim.n_rods = rods_sub.size();
            rods.resize(3, sim.n_rods);
            tbb::parallel_for(0, sim.n_rods, [&](int i){
                rods.col(i) = rods_sub[i];
            });
            connections = new_connections;

            int n_bending_pairs = 2;
            
            std::vector<int> init(sim.N_PBC_BENDING_ELE, -1);
            for (int i = 0; i < n_bending_pairs; i++)
                sim.pbc_bending_bn_pairs.push_back(init);

            for (int i = 0; i < sim.n_rods; i++)
            {
                add4Nodes(0, 1, 0, i);
                add4Nodes(2, 4, 1, i);
            }
            
            assert(sim.pbc_bending_bn_pairs == n_bending_pairs);

            q.conservativeResize(sim.dof, new_node_cnt);
            connections.conservativeResize(sim.dof, new_node_cnt);
            sim.n_nodes = new_node_cnt;
            
            sim.normal.conservativeResize(sim.dof, new_node_cnt);
            
            sim.q0 = q;
            sim.n_dof = dof_cnt;
            sim.W = StiffnessMatrix(sim.n_nodes * sim.dof, sim.n_dof);
            sim.W.setFromTriplets(w_entry.begin(), w_entry.end());
        }
        else
        {
            q(dim, 3) = q(dim, 2) + LDis(2, 3);
            q(dim, 4) = q(dim, 3) + LDis(3, 4);
            sim.q0 = q;
            sim.W = StiffnessMatrix(sim.n_nodes * sim.dof, sim.n_dof);
            sim.W.setIdentity();
        }
        sim.is_end_nodes = std::vector<bool>(sim.n_nodes, false);
        T rod_length = (sim.q0.col(rods.col(0)(0)).template segment<dim>(0) - 
            sim.q0.col(rods.col(0)(1)).template segment<dim>(0)).norm();

        sim.tunnel_u = rod_length * 0.2 * T(sub_div);
        sim.tunnel_v = rod_length * 0.2 * T(sub_div);


        sim.curvature_functions.push_back(new CircleCurvature<T, dim>(r));
        sim.curvature_functions.push_back(new LineCurvature<T, dim>());
    }

}

template<class T, int dim>
void UnitPatch<T, dim>::buildTwoRodsScene(int sub_div)
{
    
    clearSimData();
    T r = 0.25;   
    sim.n_nodes = 3;
    sim.n_rods = 2;
    // sim.add_rotation_penalty = false;
    sim.add_pbc_bending = false;
    q = DOFStack(sim.dof, sim.n_nodes); q.setZero();
    rods = IV3Stack(3, sim.n_rods); rods.setZero();
    connections = IV4Stack(4, sim.n_nodes).setOnes() * -1;

    sim.normal = TV3Stack(3, sim.n_rods);
    sim.normal.setZero();

    sim.subdivide = true;

    if constexpr (dim == 2)
    {
        {
            q.col(0).template segment<dim>(0) = TV2(0, 0.5);
            q.col(0).template segment<2>(dim) = TV2(0, 0.5);
            // q.col(1).template segment<dim>(0) = TV2(0.5 - r * std::cos(M_PI/8), 0.75);
            // q.col(1).template segment<2>(dim) = TV2(r * M_PI/2, 0.75);
            q.col(1).template segment<dim>(0) = TV2(0.25 - r * std::cos(M_PI/4), r * std::sin(M_PI/4) + 0.5);
            T l0 = (q.col(1).template segment<dim>(0) - q.col(0).template segment<dim>(0)).norm();
            q.col(1).template segment<2>(dim) = TV2(l0, r * std::sin(M_PI/4) + 0.5);
            q.col(2).template segment<dim>(0) = TV2(0.5, 0.5);
            T l1 = (q.col(2).template segment<dim>(0) - q.col(1).template segment<dim>(0)).norm();
            // q.col(2).template segment<2>(dim) = TV2(r * M_PI, 0.5);
            q.col(2).template segment<2>(dim) = TV2(q(2, 1) + l1, r * std::sin(M_PI/4) + 0.5 + 0.5 - r * std::sin(M_PI/4) + 0.5);

            int cnt = 0;
            std::vector<int> rod0 = {0, 1, 2};

            addRods(rod0, WARP, cnt);
            
            sim.q0 = q;
            sim.n_dof = sim.n_nodes * sim.dof;
            for (int i = 0; i < sim.n_rods; i++) 
                sim.yarn_map[i] = i;
            
            sim.pbc_ref.push_back(std::make_pair(WARP, IV2(0, 2)));
            
            sim.pbc_ref_unique.push_back(IV2(0, 2));
            sim.pbc_ref_unique.push_back(IV2(0, 1));
            
            if (sim.disable_sliding)
            {
                sim.dirichlet_data[0] = std::make_pair(TVDOF::Zero(), sim.fix_eulerian);
                sim.dirichlet_data[1] = std::make_pair(TVDOF::Zero(), sim.fix_eulerian);
                sim.dirichlet_data[2] = std::make_pair(TVDOF::Zero(), sim.fix_all);
            }
            else
            {
                sim.dirichlet_data[0] = std::make_pair(TVDOF::Zero(), sim.fix_eulerian);
                sim.dirichlet_data[1] = std::make_pair(TVDOF::Zero(), sim.fix_v);
                sim.dirichlet_data[2] = std::make_pair(TVDOF::Zero(), sim.fix_all);
            }
            
            sim.n_dof = sim.n_nodes * sim.dof;
        }
        
        {
            std::vector<IV3> rods_sub;
            // std::cout << "#nodes " << n_nodes << std::endl;
            int new_node_cnt = sim.n_nodes;
            int dof_cnt = sim.n_nodes * sim.dof;
            std::vector<Eigen::Triplet<T>> w_entry;

            for (int i = 0; i < sim.n_nodes; i++)
                for(int d = 0; d < sim.dof; d++)
                    w_entry.push_back(Eigen::Triplet<T>(i * sim.dof + d, i * sim.dof + d, 1.0));
                
            
            sim.n_nodes = sim.n_nodes + (sub_div-1) * sim.n_rods;
            q.conservativeResize(sim.dof, sim.n_nodes);
            sim.normal.resize(3, sim.n_nodes);
            sim.normal.setZero();
            IV4Stack new_connections(4, sim.n_nodes);
            new_connections.setConstant(-1);

            sim.subdivide = true;

            auto setConnection = [&](Eigen::Ref<IV4Stack> cns, int node_i, int node_j, int yarn_type){
                    if (yarn_type == WEFT)
                    {
                        cns(1, node_i) = node_j;
                        cns(0, node_j) = node_i;
                    }
                    else
                    {
                        cns(3, node_i) = node_j;
                        cns(2, node_j) = node_i;
                    }
                };

            for (int rod_idx = 0; rod_idx < sim.n_rods; rod_idx++)
            {
                IV2 end_points = rods.col(rod_idx).template segment<2>(0);
                int node_i = end_points[0];
                int node_j = end_points[1];
                
                int yarn_type = rods(2, rod_idx);
                
                bool sign0 = connections.col(node_i).prod();
                int sign1 = connections.col(node_j).prod();
                
                // std::cout << "xi: " << q.col(node_i).transpose() << std::endl;
                // std::cout << "xj: "<< q.col(node_j).transpose() << std::endl;
                T fraction = T(1) / sub_div;
                bool new_node_added = false;
                bool left_or_bottom_bd = ((connections(0, node_i) < 0 && connections(1, node_i) != -1) || (connections(2, node_i) < 0 && connections(3, node_i) != -1));
                bool right_or_top_bd = ((connections(1, node_j) < 0 && connections(0, node_j) != -1) || (connections(3, node_j) < 0 && connections(2, node_j) != -1));
                int cnt = 0;
                T arc_length_ij = r * M_PI/2.0;
                for (int sub_cnt = 1; sub_cnt < sub_div; sub_cnt++)
                {
                    T alpha = sub_cnt * fraction;
                    if(yarn_type == WEFT)
                    {
                        T mid = 0.5;//yarn_type == WARP ? 0.5 * M_PI * 0.5 : 0.5;
                        // std::cout << left_or_bottom_bd  << " " << (alpha < mid) << " " << alpha <<  std::endl;
                        // left or bottom boundary
                        if (left_or_bottom_bd && (alpha <= mid))
                            continue;
                        // right or top boundary
                        if (right_or_top_bd && (alpha >= mid))
                            continue;
                        
                        if (left_or_bottom_bd)
                            alpha = (alpha - mid) / mid;
                        if (right_or_top_bd)
                            alpha = alpha / mid;
                        // std::cout << "alpha: " << alpha << " " << left_or_bottom_bd << std::endl; 
                        
                        q.col(new_node_cnt) = 
                            q.col(node_i) * (1 - alpha) + 
                            q.col(node_j) * alpha;
                        
                        
                    }
                    else
                    {

                        T theta = alpha * M_PI / 2;
                        if(q(0, node_i) == 0)
                        {
                            q.col(new_node_cnt).template segment<dim>(0) = TV(0.25 - r * std::cos(theta), r*std::sin(theta) + 0.5);
                            q(dim, new_node_cnt) = r * theta;
                            q(dim + 1, new_node_cnt) = q(dim + 1, node_i) * (1 - alpha) + q(dim + 1, node_j) * alpha;
                        }
                        else if(q(0, node_i) == 0.25)
                        {   
                            q.col(new_node_cnt).template segment<dim>(0) = TV(0.25 + r * std::cos(M_PI/2.0 - theta), r*std::sin(M_PI/2.0 - theta) + 0.5);
                            q(dim, new_node_cnt) = r * M_PI/2 + r * theta;
                            q(dim + 1, new_node_cnt) = q(dim + 1, node_i) * (1 - alpha) + q(dim + 1, node_j) * alpha;
                        }
                        else if(q(0, node_i) == 0.5)
                        {
                            q.col(new_node_cnt).template segment<dim>(0) = TV(0.75 - r * std::cos(theta), -r*std::sin(theta) + 0.5);
                            q(dim, new_node_cnt) = r * M_PI + r * theta;
                            q(dim + 1, new_node_cnt) = q(dim + 1, node_i) * (1 - alpha) + q(dim + 1, node_j) * alpha;
                        }
                        else if(q(0, node_i) == 0.75)
                        {
                            q.col(new_node_cnt).template segment<dim>(0) = TV(0.75 + r * std::cos(M_PI/2.0 - theta), -r*std::sin(M_PI/2.0 - theta) + 0.5);
                            q(dim, new_node_cnt) = r * 1.5 * M_PI + r * theta;
                            q(dim + 1, new_node_cnt) = q(dim + 1, node_i) * (1 - alpha) + q(dim + 1, node_j) * alpha;
                        }
                    }
                    
                    for(int d = 0; d < dim; d++)
                    {
                        w_entry.push_back(Entry(new_node_cnt * sim.dof + d, dof_cnt, 1));
                        dof_cnt++;
                    }
                    for(int d = dim; d < sim.dof; d++)
                    {
                        w_entry.push_back(Entry(new_node_cnt * sim.dof + d, node_i * sim.dof + d, 1-alpha));
                        w_entry.push_back(Entry(new_node_cnt * sim.dof + d, node_j * sim.dof + d, alpha));
                    }

                    // std::cout << "x sub sine: "<< q.col(new_node_cnt).transpose() << std::endl;
                    int n0, n1;
                    if (cnt == 0)
                    {
                        n0 = node_i; n1 = new_node_cnt;
                    }
                    else
                    {
                        n0 = new_node_cnt-1; n1 = new_node_cnt;
                    }
                    rods_sub.push_back(IV3(n0, n1, yarn_type));
                    sim.yarn_map[rods_sub.size()-1] = rod_idx;
                    setConnection(new_connections, n0, n1, yarn_type);
                    // dirichlet_data[new_node_cnt] = std::make_pair(TVDOF::Zero(), fix_eulerian);
                    new_node_cnt++;
                    new_node_added = true;
                    cnt++;
                }
                if (new_node_added)
                {
                    rods_sub.push_back(IV3(new_node_cnt-1, node_j, yarn_type));
                    setConnection(new_connections, new_node_cnt-1, node_j, yarn_type);
                }
                else
                {
                    rods_sub.push_back(IV3(node_i, node_j, yarn_type));
                    setConnection(new_connections, node_i, node_j, yarn_type);   
                }
                sim.yarn_map[rods_sub.size()-1] = rod_idx;
                
            }

            sim.n_rods = rods_sub.size();
            rods.resize(3, sim.n_rods);
            tbb::parallel_for(0, sim.n_rods, [&](int i){
                rods.col(i) = rods_sub[i];
            });
            connections = new_connections;

            int n_bending_pairs = 1;
            
            std::vector<int> init(sim.N_PBC_BENDING_ELE, -1);
            for (int i = 0; i < n_bending_pairs; i++)
                sim.pbc_bending_bn_pairs.push_back(init);

            for (int i = 0; i < sim.n_rods; i++)
            {
                add4Nodes(0, 2, 0, i);
            }

            q.conservativeResize(sim.dof, new_node_cnt);
            connections.conservativeResize(sim.dof, new_node_cnt);
            sim.n_nodes = new_node_cnt;
            
            sim.normal.conservativeResize(sim.dof, new_node_cnt);
            sim.is_end_nodes = std::vector<bool>(sim.n_nodes, false);

            sim.q0 = q;
            sim.n_dof = dof_cnt;
            sim.W = StiffnessMatrix(sim.n_nodes * sim.dof, sim.n_dof);
            sim.W.setFromTriplets(w_entry.begin(), w_entry.end());
        }

        
        
        sim.is_end_nodes = std::vector<bool>(sim.n_nodes, false);

        T rod_length = (sim.q0.col(rods.col(0)(0)).template segment<dim>(0) - 
            sim.q0.col(rods.col(0)(1)).template segment<dim>(0)).norm();

        sim.tunnel_u = rod_length * T(sub_div) * 0.2;
        sim.tunnel_v = rod_length * T(sub_div) * 0.2;
        // sim.W = StiffnessMatrix(sim.n_nodes * sim.dof, sim.n_dof);
        // sim.W.setIdentity();

        sim.curvature_functions.push_back(new CircleCurvature<T, dim>(r));
        // sim.curvature_functions.push_back(new LineCurvature<T, dim>());
    }
    // std::cout << connections.transpose() << std::endl;
}

template<class T, int dim>
void UnitPatch<T, dim>::subdivide(int sub_div)
{
    
}

template<class T, int dim>
void UnitPatch<T, dim>::clearSimData()
{
    sim.kc = 1e8;
    sim.add_pbc = true;

    if(sim.disable_sliding)
    {
        sim.add_shearing = true;
        sim.add_eularian_reg = false;
        sim.k_pbc = 1e8;
        sim.k_strain = 1e8;
    }
    else
    {
        sim.add_shearing = false;
        sim.add_eularian_reg = true;
        sim.ke = 1e-4;    
        sim.k_yc = 1e8;
    }
    sim.k_pbc = 1e4;
    sim.k_strain = 1e7;
    sim.kr = 1e3;
    
    sim.pbc_ref_unique.clear();
    sim.dirichlet_data.clear();
    sim.pbc_ref.clear();
    sim.pbc_bending_pairs.clear();
    sim.yarns.clear();
}


template<class T, int dim>
void UnitPatch<T, dim>::buildStraightAndHemiCircleScene(int sub_div)
{
    bool use_analytical = true;
    clearSimData();
    T r = 0.25;
    sim.n_nodes = 17;
    sim.n_rods = 16;

    q = DOFStack(sim.dof, sim.n_nodes); q.setZero();
    rods = IV3Stack(3, sim.n_rods); rods.setZero();
    connections = IV4Stack(4, sim.n_nodes).setOnes() * -1;

    sim.normal = TV3Stack(3, sim.n_rods);
    sim.normal.setZero();
    
    if constexpr (dim == 2)
    {
        
        {
            q.col(0).template segment<dim>(0) = TV2(0.25, 0.125);
            q.col(1).template segment<dim>(0) = TV2(0.25, 0.875);
            q.col(2).template segment<dim>(0) = TV2(0.5, 0.125); 
            q.col(3).template segment<dim>(0) = TV2(0.5, 0.875); 
            q.col(4).template segment<dim>(0) = TV2(0.75, 0.125); 
            q.col(5).template segment<dim>(0) = TV2(0.75, 0.875); 
            q.block(dim, 0, 2, 6) = q.block(0, 0, 2, 6);

            
            q.col(6).template segment<dim>(0) = TV2(0, 0.5);
            q.col(6).template segment<2>(dim) = TV2(0, 0.5);
            q.col(7).template segment<dim>(0) = TV2(0.25, 0.75);
            q.col(7).template segment<2>(dim) = TV2(r * M_PI/2, 0.75);
            q.col(8).template segment<dim>(0) = TV2(0.5, 0.5);
            q.col(8).template segment<2>(dim) = TV2(r * M_PI, 0.5);

            q.col(9).template segment<dim>(0) = TV2(0.75, 0.25);
            q.col(9).template segment<2>(dim) = TV2(1.5* M_PI * r, 0.25);
            q.col(10).template segment<dim>(0) = TV2(1, 0.5);
            q.col(10).template segment<2>(dim) = TV2(2 * M_PI * r, 0.5);

            q.col(11).template segment<dim>(0) = TV2(0.25, 0.25);
            q.col(12).template segment<dim>(0) = TV2(0.25, 0.5);
            q.col(13).template segment<dim>(0) = TV2(0.5, 0.25);
            q.col(14).template segment<dim>(0) = TV2(0.5, 0.75); 
            q.col(15).template segment<dim>(0) = TV2(0.75, 0.5);
            q.col(16).template segment<dim>(0) = TV2(0.75, 0.75); 
            q.block(dim, 11, 2, 6) = q.block(0, 11, 2, 6);
            

            int cnt = 0;
            std::vector<int> rod0 = {6, 7, 8, 9, 10}, 
                             rod1 = {0, 11, 12, 7, 1}, 
                             rod2 = {2, 13, 8, 14, 3}, 
                             rod3 = {4, 9, 15, 16, 5};

            addRods(rod0, WARP, cnt, 0);
            addRods(rod1, WEFT, cnt, 1);
            addRods(rod2, WEFT, cnt, 2);
            addRods(rod3, WEFT, cnt, 3);
            
            sim.q0 = q;
            sim.n_dof = sim.n_nodes * sim.dof;

            sim.pbc_ref.push_back(std::make_pair(WEFT, IV2(0, 1)));
            sim.pbc_ref.push_back(std::make_pair(WEFT, IV2(2, 3)));
            sim.pbc_ref.push_back(std::make_pair(WEFT, IV2(4, 5)));

            sim.pbc_ref.push_back(std::make_pair(WARP, IV2(6, 10)));
            sim.pbc_ref_unique.push_back(IV2(6, 10));
            sim.pbc_ref_unique.push_back(IV2(0, 1));

            if (sim.disable_sliding)
            {
                for(int i = 0; i < sim.n_nodes; i++)
                    sim.dirichlet_data[i] = std::make_pair(TVDOF::Zero(), sim.fix_eulerian);
                sim.dirichlet_data[8] = std::make_pair(TVDOF::Zero(), sim.fix_all);
            }
            else
            {
                for (int i = 0; i < sim.n_nodes; i++)
                {
                    if (connections.col(i).prod() < 0)
                        sim.dirichlet_data[i]= std::make_pair(TVDOF::Zero(), sim.fix_eulerian);    
                }
                sim.dirichlet_data[11]= std::make_pair(TVDOF::Zero(), sim.fix_eulerian);
                sim.dirichlet_data[12]= std::make_pair(TVDOF::Zero(), sim.fix_eulerian);
                sim.dirichlet_data[13]= std::make_pair(TVDOF::Zero(), sim.fix_eulerian);
                sim.dirichlet_data[14]= std::make_pair(TVDOF::Zero(), sim.fix_eulerian);
                sim.dirichlet_data[15]= std::make_pair(TVDOF::Zero(), sim.fix_eulerian);
                sim.dirichlet_data[16]= std::make_pair(TVDOF::Zero(), sim.fix_eulerian);

                // sim.dirichlet_data[7]= std::make_pair(TVDOF::Zero(), sim.fix_u);
                // sim.dirichlet_data[9]= std::make_pair(TVDOF::Zero(), sim.fix_u);


                sim.dirichlet_data[8] = std::make_pair(TVDOF::Zero(), sim.fix_all);

                // sim.dirichlet_data[7]= std::make_pair(TVDOF::Zero(), sim.fix_u);
                // sim.dirichlet_data[9]= std::make_pair(TVDOF::Zero(), sim.fix_u);
                // sim.dirichlet_data[1]= std::make_pair(TVDOF::Zero(), sim.fix_eulerian);
            }
            sim.n_dof = sim.n_nodes * sim.dof;
            // sim.W = StiffnessMatrix(sim.n_nodes * sim.dof, sim.n_dof);
            // sim.W.setIdentity();
        }
        auto unit_yarn_map = sim.yarn_map;
        sim.yarn_map.clear();
        // if(false)
        {
            std::vector<IV3> rods_sub;
            // std::cout << "#nodes " << n_nodes << std::endl;
            int new_node_cnt = sim.n_nodes;
            int dof_cnt = sim.n_nodes * sim.dof;
            std::vector<Eigen::Triplet<T>> w_entry;

            for (int i = 0; i < sim.n_nodes; i++)
                for(int d = 0; d < sim.dof; d++)
                    w_entry.push_back(Eigen::Triplet<T>(i * sim.dof + d, i * sim.dof + d, 1.0));
                
            
            sim.n_nodes = sim.n_nodes + (sub_div-1) * sim.n_rods;
            q.conservativeResize(sim.dof, sim.n_nodes);
            sim.normal.resize(3, sim.n_nodes);
            sim.normal.setZero();
            IV4Stack new_connections(4, sim.n_nodes);
            new_connections.setConstant(-1);

            sim.subdivide = true;

            auto setConnection = [&](Eigen::Ref<IV4Stack> cns, int node_i, int node_j, int yarn_type){
                    if (yarn_type == WEFT)
                    {
                        cns(1, node_i) = node_j;
                        cns(0, node_j) = node_i;
                    }
                    else
                    {
                        cns(3, node_i) = node_j;
                        cns(2, node_j) = node_i;
                    }
                };

            for (int rod_idx = 0; rod_idx < sim.n_rods; rod_idx++)
            {
                IV2 end_points = rods.col(rod_idx).template segment<2>(0);
                int node_i = end_points[0];
                int node_j = end_points[1];
                
                int yarn_type = rods(2, rod_idx);
                
                bool sign0 = connections.col(node_i).prod();
                int sign1 = connections.col(node_j).prod();
                
                // std::cout << "xi: " << q.col(node_i).transpose() << std::endl;
                // std::cout << "xj: "<< q.col(node_j).transpose() << std::endl;
                T fraction = T(1) / sub_div;
                bool new_node_added = false;
                bool left_or_bottom_bd = ((connections(0, node_i) < 0 && connections(1, node_i) != -1) || (connections(2, node_i) < 0 && connections(3, node_i) != -1));
                bool right_or_top_bd = ((connections(1, node_j) < 0 && connections(0, node_j) != -1) || (connections(3, node_j) < 0 && connections(2, node_j) != -1));
                int cnt = 0;
                T arc_length_ij = r * M_PI/2.0;
                int prev_node = node_i;
                std::vector<int> sub_nodes;
                T arc_length_sum = 0;

                for (int sub_cnt = 1; sub_cnt < sub_div; sub_cnt++)
                {
                    T alpha = sub_cnt * fraction;
                    if(yarn_type == WEFT)
                    {
                        T mid = 0.5;//yarn_type == WARP ? 0.5 * M_PI * 0.5 : 0.5;
                        // std::cout << left_or_bottom_bd  << " " << (alpha < mid) << " " << alpha <<  std::endl;
                        // left or bottom boundary
                        if (left_or_bottom_bd && (alpha <= mid))
                            continue;
                        // right or top boundary
                        if (right_or_top_bd && (alpha >= mid))
                            continue;
                        
                        if (left_or_bottom_bd)
                            alpha = (alpha - mid) / mid;
                        if (right_or_top_bd)
                            alpha = alpha / mid;
                        // std::cout << "alpha: " << alpha << " " << left_or_bottom_bd << std::endl; 
                        
                        q.col(new_node_cnt) = 
                            q.col(node_i) * (1 - alpha) + 
                            q.col(node_j) * alpha;
                        
                        
                    }
                    else
                    {

                        T theta = alpha * M_PI / 2;
                        if(q(0, node_i) == 0)
                        {
                            q.col(new_node_cnt).template segment<dim>(0) = TV(0.25 - r * std::cos(theta), r*std::sin(theta) + 0.5);
                            if (use_analytical) q(dim, new_node_cnt) = r * theta;
                            q(dim + 1, new_node_cnt) = q(dim + 1, node_i) * (1 - alpha) + q(dim + 1, node_j) * alpha;
                        }
                        else if(q(0, node_i) == 0.25)
                        {   
                            q.col(new_node_cnt).template segment<dim>(0) = TV(0.25 + r * std::cos(M_PI/2.0 - theta), r*std::sin(M_PI/2.0 - theta) + 0.5);
                            if (use_analytical) q(dim, new_node_cnt) = r * M_PI/2 + r * theta;
                            q(dim + 1, new_node_cnt) = q(dim + 1, node_i) * (1 - alpha) + q(dim + 1, node_j) * alpha;
                        }
                        else if(q(0, node_i) == 0.5)
                        {
                            q.col(new_node_cnt).template segment<dim>(0) = TV(0.75 - r * std::cos(theta), -r*std::sin(theta) + 0.5);
                            if (use_analytical) q(dim, new_node_cnt) = r * M_PI + r * theta;
                            q(dim + 1, new_node_cnt) = q(dim + 1, node_i) * (1 - alpha) + q(dim + 1, node_j) * alpha;
                        }
                        else if(q(0, node_i) == 0.75)
                        {
                            q.col(new_node_cnt).template segment<dim>(0) = TV(0.75 + r * std::cos(M_PI/2.0 - theta), -r*std::sin(M_PI/2.0 - theta) + 0.5);
                            if (use_analytical) q(dim, new_node_cnt) = r * 1.5 * M_PI + r * theta;
                            q(dim + 1, new_node_cnt) = q(dim + 1, node_i) * (1 - alpha) + q(dim + 1, node_j) * alpha;
                        }
                    }
                    
                    for(int d = 0; d < dim; d++)
                    {
                        w_entry.push_back(Entry(new_node_cnt * sim.dof + d, dof_cnt, 1));
                        dof_cnt++;
                    }
                    if (use_analytical)
                    {
                        for(int d = dim; d < sim.dof; d++)
                        {
                            w_entry.push_back(Entry(new_node_cnt * sim.dof + d, node_i * sim.dof + d, 1-alpha));
                            w_entry.push_back(Entry(new_node_cnt * sim.dof + d, node_j * sim.dof + d, alpha));
                        }
                    }
                    else
                    {
                        w_entry.push_back(Entry(new_node_cnt * sim.dof + dim + 1, node_i * sim.dof + dim + 1, 1-alpha));
                        w_entry.push_back(Entry(new_node_cnt * sim.dof + dim + 1, node_j * sim.dof + dim + 1, alpha));
                    }

                    // std::cout << "x sub sine: "<< q.col(new_node_cnt).transpose() << std::endl;
                    int n0, n1;
                    if (cnt == 0)
                    {
                        n0 = node_i; n1 = new_node_cnt;
                    }
                    else
                    {
                        n0 = new_node_cnt-1; n1 = new_node_cnt;
                    }
                    rods_sub.push_back(IV3(n0, n1, yarn_type));
                    sim.yarn_map[rods_sub.size()-1] = unit_yarn_map[rod_idx];
                    setConnection(new_connections, n0, n1, yarn_type);
                    // dirichlet_data[new_node_cnt] = std::make_pair(TVDOF::Zero(), fix_eulerian);
                    new_node_cnt++;
                    new_node_added = true;
                    cnt++;
                }
                if (new_node_added)
                {
                    rods_sub.push_back(IV3(new_node_cnt-1, node_j, yarn_type));
                    setConnection(new_connections, new_node_cnt-1, node_j, yarn_type);
                    
                    if(!use_analytical && yarn_type == WARP)
                    {
                        arc_length_sum += LDis(new_node_cnt-1, node_j);
                        for (int id : sub_nodes)
                        {
                            T alpha = (q(2, id) - q(2, node_i)) / arc_length_sum;
                            // std::cout << "id: " << id << " node_i: " << node_i << " dis " << q(2, id) - q(2, node_i) << " " << arc_length_sum << " " << alpha << std::endl;
                            w_entry.push_back(Entry(id * sim.dof + dim, node_i * sim.dof + dim, 1-alpha));
                            w_entry.push_back(Entry(id * sim.dof + dim, node_j * sim.dof + dim, alpha));

                        }
                        q(dim, node_j) = q(dim, new_node_cnt - 1) + LDis(new_node_cnt-1, node_j);  
                    }
                }
                else
                {
                    rods_sub.push_back(IV3(node_i, node_j, yarn_type));
                    setConnection(new_connections, node_i, node_j, yarn_type);   
                }
                sim.yarn_map[rods_sub.size()-1] = unit_yarn_map[rod_idx];
                
            }

            sim.n_rods = rods_sub.size();
            rods.resize(3, sim.n_rods);
            tbb::parallel_for(0, sim.n_rods, [&](int i){
                rods.col(i) = rods_sub[i];
            });
            connections = new_connections;

            int n_bending_pairs = 4;
            
            std::vector<int> init(sim.N_PBC_BENDING_ELE, -1);
            for (int i = 0; i < n_bending_pairs; i++)
                sim.pbc_bending_bn_pairs.push_back(init);

            for (int i = 0; i < sim.n_rods; i++)
            {
                add4Nodes(0, 1, 0, i);
                add4Nodes(2, 3, 1, i);
                add4Nodes(4, 5, 2, i);
                add4Nodes(6, 10, 3, i);
            }
            
            assert(sim.pbc_bending_bn_pairs == n_bending_pairs);

            q.conservativeResize(sim.dof, new_node_cnt);
            connections.conservativeResize(sim.dof, new_node_cnt);
            sim.n_nodes = new_node_cnt;
            
            sim.normal.conservativeResize(sim.dof, new_node_cnt);
            sim.is_end_nodes = std::vector<bool>(sim.n_nodes, false);

            sim.q0 = q;
            sim.n_dof = dof_cnt;
            sim.W = StiffnessMatrix(sim.n_nodes * sim.dof, sim.n_dof);
            sim.W.setFromTriplets(w_entry.begin(), w_entry.end());
        }
        
        T rod_length = (sim.q0.col(rods.col(0)(0)).template segment<dim>(0) - 
            sim.q0.col(rods.col(0)(1)).template segment<dim>(0)).norm();

        sim.tunnel_u = rod_length * 0.2 * T(sub_div);
        sim.tunnel_v = rod_length * 0.2 * T(sub_div);


        sim.curvature_functions.push_back(new CircleCurvature<T, dim>(r));
        sim.curvature_functions.push_back(new LineCurvature<T, dim>());
    }
}


template<class T, int dim>
void UnitPatch<T, dim>::buildStraightAndSineScene(int sub_div)
{
    clearSimData();

    // add base nodes
    T amp = 0.25, period = 2.0 * M_PI, phi = 0.5;
    auto _sin = [=](T theta) { return amp * std::sin(period * theta) + phi; };

    sim.n_nodes = 14;
    sim.n_rods = 13;

    q = DOFStack(sim.dof, sim.n_nodes); q.setZero();
    rods = IV3Stack(3, sim.n_rods); rods.setZero();
    connections = IV4Stack(4, sim.n_nodes).setOnes() * -1;

    sim.normal = TV3Stack(3, sim.n_rods);
    sim.normal.setZero();

    if constexpr (dim == 2)
    {
        auto computeLagDis = [&](int i, int j)
        {
            return q(2, i) + (q.col(i).template segment<dim>(0) - q.col(j).template segment<dim>(0)).norm();
        };
        
        {
            q.col(0).template segment<dim>(0) = TV2(0, 0);
            q.col(1).template segment<dim>(0) = TV2(0, 1);
            q.col(2).template segment<dim>(0) = TV2(M_PI/2.0 /period, 0); 
            q.col(3).template segment<dim>(0) = TV2(M_PI/2.0/period, 1); 
            q.col(4).template segment<dim>(0) = TV2(M_PI/period, 0); 
            q.col(5).template segment<dim>(0) = TV2(M_PI/period, 1); 
            q.col(6).template segment<dim>(0) = TV2(3.0*M_PI/2.0/period, 0);
            q.col(7).template segment<dim>(0) = TV2(3.0*M_PI/2.0/period, 1);
            q.block(dim, 0, 2, 8) = q.block(0, 0, 2, 8);

            
            q.col(8).template segment<dim>(0) = TV2(-M_PI/4.0/period, _sin(-M_PI/4.0/period));
            q.col(8).template segment<2>(dim) = TV2(0, 0.375);
            q.col(9).template segment<dim>(0) = TV2(0, _sin(0));
            q.col(9).template segment<2>(dim) = TV2(computeLagDis(8, 9), 0.5);
            q.col(10).template segment<dim>(0) = TV2(M_PI/2.0/period, _sin(M_PI/2.0/period));
            q.col(10).template segment<2>(dim) = TV2(computeLagDis(9, 10), 0.75);
            q.col(11).template segment<dim>(0) = TV2(M_PI/period, _sin(M_PI/period));
            q.col(11).template segment<2>(dim) = TV2(computeLagDis(10, 11), 0.5);
            q.col(12).template segment<dim>(0) = TV2(3.0 * M_PI/2.0/period, _sin(3.0 * M_PI/2.0/period));
            q.col(12).template segment<2>(dim) = TV2(computeLagDis(11, 12), 0.25);
            q.col(13).template segment<dim>(0) = TV2(7.0 * M_PI/4.0/period, _sin(7.0 * M_PI/4.0/period));
            q.col(13).template segment<2>(dim) = TV2(computeLagDis(12, 13), 0.375);
        }
        
        // q.block(dim, 0, 2, 14) = q.block(0, 0, 2, 14);
        // std::cout << q.transpose() << std::endl;

        int cnt = 0;
        std::vector<int> rod0 = {8,9,10,11,12,13}, 
                         rod1 = {0, 9, 1}, 
                         rod2 = {2, 10, 3}, 
                         rod3 = {4, 11, 5},
                         rod4 = {6, 12, 7};

        addRods(rod0, WARP, cnt);
        addRods(rod1, WEFT, cnt);
        addRods(rod2, WEFT, cnt);
        addRods(rod3, WEFT, cnt);
        addRods(rod4, WEFT, cnt);
        

        sim.q0 = q;    
        sim.n_dof = sim.n_nodes * sim.dof;
        // sim.W = StiffnessMatrix(sim.n_nodes * sim.dof, sim.n_dof);
        // sim.W.setIdentity();

        for (int i = 0; i < sim.n_rods; i++)
            sim.yarn_map[i] = i;

        sim.pbc_ref.push_back(std::make_pair(WEFT, IV2(0, 1)));
        sim.pbc_ref.push_back(std::make_pair(WEFT, IV2(2, 3)));
        sim.pbc_ref.push_back(std::make_pair(WEFT, IV2(4, 5)));
        sim.pbc_ref.push_back(std::make_pair(WEFT, IV2(6, 7)));
        sim.pbc_ref_unique.push_back(IV2(8, 13));
        // pbc_ref_unique[WEFT] = IV2(8, 13)
        // has to push this first, stupid, but that's it for now

        sim.pbc_ref.push_back(std::make_pair(WARP, IV2(8, 13)));
        sim.pbc_ref_unique.push_back(IV2(0, 1));

        if (sim.disable_sliding)
        {
            for(int i = 0; i < sim.n_nodes; i++)
                sim.dirichlet_data[i] = std::make_pair(TVDOF::Zero(), sim.fix_eulerian);
            sim.dirichlet_data[0] = std::make_pair(TVDOF::Zero(), sim.fix_all);
        }
        else
        {
            
            for (int i = 0; i < sim.n_nodes; i++)
            {
                if (connections.col(i).prod() < 0)
                    sim.dirichlet_data[i]= std::make_pair(TVDOF::Zero(), sim.fix_eulerian);    
            }
            sim.dirichlet_data[0] = std::make_pair(TVDOF::Zero(), sim.fix_all);
        }
        // std::cout << "min u: " << q.row(2).minCoeff() << " max u: " << q.row(2).maxCoeff() << std::endl;
        // if(false)
        {

            std::vector<IV3> rods_sub;
            // std::cout << "#nodes " << n_nodes << std::endl;
            int new_node_cnt = sim.n_nodes;
            int dof_cnt = sim.n_nodes * sim.dof;
            std::vector<Eigen::Triplet<T>> w_entry;

            for (int i = 0; i < sim.n_nodes; i++)
                for(int d = 0; d < sim.dof; d++)
                    w_entry.push_back(Eigen::Triplet<T>(i * sim.dof + d, i * sim.dof + d, 1.0));
                
            
            sim.n_nodes = sim.n_nodes + (sub_div-1) * sim.n_rods;
            q.conservativeResize(sim.dof, sim.n_nodes);
            sim.normal.resize(3, sim.n_nodes);
            sim.normal.setZero();
            IV4Stack new_connections(4, sim.n_nodes);
            new_connections.setConstant(-1);

            sim.subdivide = true;

            auto setConnection = [&](Eigen::Ref<IV4Stack> cns, int node_i, int node_j, int yarn_type){
                    if (yarn_type == WEFT)
                    {
                        cns(1, node_i) = node_j;
                        cns(0, node_j) = node_i;
                    }
                    else
                    {
                        cns(3, node_i) = node_j;
                        cns(2, node_j) = node_i;
                    }
                };

            for (int rod_idx = 0; rod_idx < sim.n_rods; rod_idx++)
            {
                IV2 end_points = rods.col(rod_idx).template segment<2>(0);
                int node_i = end_points[0];
                int node_j = end_points[1];
                
                int yarn_type = rods(2, rod_idx);
                
                bool sign0 = connections.col(node_i).prod();
                int sign1 = connections.col(node_j).prod();
                
                // std::cout << "xi: " << q.col(node_i).transpose() << std::endl;
                // std::cout << "xj: "<< q.col(node_j).transpose() << std::endl;
                T fraction = T(1) / sub_div;
                bool new_node_added = false;
                bool left_or_bottom_bd = (connections(0, node_i) < 0 || connections(2, node_i) < 0);
                bool right_or_top_bd = (connections(1, node_j) < 0 || connections(3, node_j) < 0);
                int cnt = 0;
                int prev_node = node_i;
                std::vector<int> sub_nodes;
                T arc_length_sum = 0;
                for (int sub_cnt = 1; sub_cnt < sub_div; sub_cnt++)
                {
                    T alpha = sub_cnt * fraction;

                    T mid = 0.5;//yarn_type == WARP ? 0.5 * M_PI * 0.5 : 0.5;
                    // std::cout << left_or_bottom_bd  << " " << (alpha < mid) << " " << alpha <<  std::endl;
                    // left or bottom boundary
                    if (left_or_bottom_bd && (alpha <= mid))
                        continue;
                    // right or top boundary
                    if (right_or_top_bd && (alpha >= mid))
                        continue;
                    
                    if (left_or_bottom_bd)
                        alpha = (alpha - mid) / mid;
                    if (right_or_top_bd)
                        alpha = alpha / mid;
                    // std::cout << "alpha: " << alpha << " " << left_or_bottom_bd << std::endl;
                    for(int d = 0; d < dim; d++)
                    {
                        w_entry.push_back(Entry(new_node_cnt * sim.dof + d, dof_cnt, 1));
                        dof_cnt++;
                    }   

                    q.col(new_node_cnt) = 
                        q.col(node_i) * (1 - alpha) + 
                        q.col(node_j) * alpha;
                    
                    // std::cout << "x sub: "<< q.col(new_node_cnt).transpose() << std::endl;
                    if(yarn_type == WARP)
                    {
                        w_entry.push_back(Entry(new_node_cnt * sim.dof + dim + 1, node_i * sim.dof + dim + 1, 1-alpha));
                        w_entry.push_back(Entry(new_node_cnt * sim.dof + dim + 1, node_j * sim.dof + dim + 1, alpha));

                        q(1, new_node_cnt) = _sin(q(0, new_node_cnt));
                        q(2, new_node_cnt) = computeLagDis(prev_node, new_node_cnt);
                        arc_length_sum += q(2, new_node_cnt) - q(2, prev_node);

                        prev_node = new_node_cnt;
                        sub_nodes.push_back(new_node_cnt);
                        
                    }
                    else
                    {
                        for(int d = dim; d < sim.dof; d++)
                        {
                            w_entry.push_back(Entry(new_node_cnt * sim.dof + d, node_i * sim.dof + d, 1-alpha));
                            w_entry.push_back(Entry(new_node_cnt * sim.dof + d, node_j * sim.dof + d, alpha));
                        }
                        
                    }

                    // std::cout << "x sub sine: "<< q.col(new_node_cnt).transpose() << std::endl;
                    int n0, n1;
                    if (cnt == 0)
                    {
                        n0 = node_i; n1 = new_node_cnt;
                    }
                    else
                    {
                        n0 = new_node_cnt-1; n1 = new_node_cnt;
                    }
                    rods_sub.push_back(IV3(n0, n1, yarn_type));
                    sim.yarn_map[rods_sub.size()-1] = rod_idx;
                    setConnection(new_connections, n0, n1, yarn_type);
                    // dirichlet_data[new_node_cnt] = std::make_pair(TVDOF::Zero(), fix_eulerian);
                    new_node_cnt++;
                    new_node_added = true;
                    cnt++;
                }
                if (new_node_added)
                {
                    rods_sub.push_back(IV3(new_node_cnt-1, node_j, yarn_type));
                    setConnection(new_connections, new_node_cnt-1, node_j, yarn_type);

                    if(yarn_type == WARP)
                    {
                        arc_length_sum += computeLagDis(new_node_cnt-1, node_j) - q(2, new_node_cnt-1);
                        for (int id : sub_nodes)
                        {
                            T alpha = (q(2, id) - q(2, node_i)) / arc_length_sum;
                            // std::cout << "id: " << id << " node_i: " << node_i << " dis " << q(2, id) - q(2, node_i) << " " << arc_length_sum << " " << alpha << std::endl;
                            w_entry.push_back(Entry(id * sim.dof + dim, node_i * sim.dof + dim, 1-alpha));
                            w_entry.push_back(Entry(id * sim.dof + dim, node_j * sim.dof + dim, alpha));

                        }
                        q(2, node_j) = computeLagDis(new_node_cnt-1, node_j);   
                    }
                }
                else
                {
                    rods_sub.push_back(IV3(node_i, node_j, yarn_type));
                    setConnection(new_connections, node_i, node_j, yarn_type);   
                }
                sim.yarn_map[rods_sub.size()-1] = rod_idx;
                
            }

            sim.n_rods = rods_sub.size();
            rods.resize(3, sim.n_rods);
            tbb::parallel_for(0, sim.n_rods, [&](int i){
                rods.col(i) = rods_sub[i];
            });
            connections = new_connections;

            std::vector<int> init(5, -1);
            for (int i = 0; i < 5; i++)
                sim.pbc_bending_bn_pairs.push_back(init);

            for (int i = 0; i < sim.n_rods; i++)
            {
                add4Nodes(0, 1, 0, i);
                add4Nodes(2, 3, 1, i);
                add4Nodes(4, 5, 2, i);
                add4Nodes(6, 7, 3, i);
                add4Nodes(8, 13, 4, i);
            }
            

            q.conservativeResize(sim.dof, new_node_cnt);
            connections.conservativeResize(sim.dof, new_node_cnt);
            sim.n_nodes = new_node_cnt;
            
            sim.normal.conservativeResize(sim.dof, new_node_cnt);
            sim.is_end_nodes = std::vector<bool>(sim.n_nodes, false);

            sim.q0 = q;
            sim.n_dof = dof_cnt;
            sim.W = StiffnessMatrix(sim.n_nodes * sim.dof, sim.n_dof);
            sim.W.setFromTriplets(w_entry.begin(), w_entry.end());
            
            sim.tunnel_u = (sim.q0.col(rods.col(0)(0)).template segment<dim>(0) - 
            sim.q0.col(rods.col(0)(1)).template segment<dim>(0)).norm() * 2.0;
            sim.tunnel_u = sim.tunnel_v;
        }
    
        sim.curvature_functions.push_back(new SineCurvature<T, dim>(amp, phi, period));
        // sim.curvature_functions.push_back(LineCurvature<T, dim>());
        sim.curvature_functions.push_back(new LineCurvature<T, dim>());

    }


}


template<class T, int dim>
void UnitPatch<T, dim>::build3x3StraightRod()
{

}


template<class T, int dim>
void UnitPatch<T, dim>::buildPlanePeriodicBCScene3x3()
{
    // sim.pbc_ref_unique.clear();
    // sim.dirichlet_data.clear();
    // sim.pbc_ref.clear();
    // sim.pbc_bending_pairs.clear();
    // sim.yarns.clear();

    
    // sim.add_stretching = true;
    // sim.add_bending = true;
    
    // sim.kc = 1e8;
    // sim.add_pbc = true;

    // if(sim.disable_sliding)
    // {
    //     sim.add_shearing = true;
    //     sim.add_eularian_reg = false;
    //     sim.k_pbc = 1e8;
    //     sim.k_strain = 1e8;
    // }
    // else
    // {
    //     sim.add_shearing = false;
    //     sim.add_eularian_reg = true;
    //     sim.ke = 1;
    //     sim.k_pbc = 1e8;
    //     sim.k_strain = 1e8;
    // }
    
    
    // sim.kr = 1e3;
    
    // sim.n_nodes = 21;
    // sim.n_rods = 24;

    // q = DOFStack(sim.dof, sim.n_nodes); q.setZero();
    // rods = IV3Stack(3, sim.n_rods); rods.setZero();
    // connections = IV4Stack(4, sim.n_nodes).setOnes() * -1;
    
    // sim.normal = TV3Stack(3, sim.n_rods);
    // sim.normal.setZero();

    // T u_delta = 1.0 / 3.0, v_delta = 1.0 / 3.0;
    // int cnt = 0;
    // sim.is_end_nodes.resize(sim.n_nodes, true);

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
    //             sim.is_end_nodes[cnt] = false;
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
    //     assert(cnt == n_rods);

    //     set_top_bottom(connections, 18, 1); set_top_bottom(connections, 19, 8); set_top_bottom(connections, 20, 15);
    //     set_top_bottom(connections, 11, 18); set_top_bottom(connections, 12, 19); set_top_bottom(connections, 13, 20);
    //     set_top_bottom(connections, 4, 11); set_top_bottom(connections, 5, 12); set_top_bottom(connections, 6, 13);
    //     set_top_bottom(connections, 0, 4); set_top_bottom(connections, 7, 5); set_top_bottom(connections, 14, 6);

    //     set_left_right(connections, 4, 2); set_left_right(connections, 11, 9); set_left_right(connections, 18, 16);
    //     set_left_right(connections, 5, 4); set_left_right(connections, 12, 11); set_left_right(connections, 19, 18);
    //     set_left_right(connections, 6, 5); set_left_right(connections, 13, 12); set_left_right(connections, 20, 19);
    //     set_left_right(connections, 3, 6); set_left_right(connections, 10, 13); set_left_right(connections, 17, 20);

    //     sim.checkConnections();
        
    //     sim.pbc_ref_unique.push_back(IV2(0, 1));
    //     sim.pbc_ref_unique.push_back(IV2(2, 3));

    //     if (sim.disable_sliding)
    //     {
    //         for(int i = 0; i < sim.n_nodes; i++)
    //             sim.dirichlet_data[i] = std::make_pair(TVDOF::Zero(), sim.fix_eulerian);
    //         sim.dirichlet_data[12] = std::make_pair(TVDOF::Zero(), sim.fix_all);
    //     }
    //     else
    //     {
    //         sim.dirichlet_data[2] = std::make_pair(TVDOF::Zero(), sim.fix_eulerian);
    //         sim.dirichlet_data[9] = std::make_pair(TVDOF::Zero(), sim.fix_eulerian);
    //         sim.dirichlet_data[16] = std::make_pair(TVDOF::Zero(), sim.fix_eulerian);
    //         sim.dirichlet_data[3] = std::make_pair(TVDOF::Zero(), sim.fix_eulerian);
    //         sim.dirichlet_data[10] = std::make_pair(TVDOF::Zero(), sim.fix_eulerian);
    //         sim.dirichlet_data[17] = std::make_pair(TVDOF::Zero(), sim.fix_eulerian);

    //         sim.dirichlet_data[1] = std::make_pair(TVDOF::Zero(), sim.fix_eulerian);
    //         sim.dirichlet_data[8] = std::make_pair(TVDOF::Zero(), sim.fix_eulerian);
    //         sim.dirichlet_data[15] = std::make_pair(TVDOF::Zero(), sim.fix_eulerian);
    //         sim.dirichlet_data[0] = std::make_pair(TVDOF::Zero(), sim.fix_eulerian);
    //         sim.dirichlet_data[7] = std::make_pair(TVDOF::Zero(), sim.fix_eulerian);
    //         sim.dirichlet_data[14] = std::make_pair(TVDOF::Zero(), sim.fix_eulerian);
    //         sim.dirichlet_data[12] = std::make_pair(TVDOF::Zero(), sim.fix_lagrangian);
    //     }
            
        
    //     // add reference pairs
    //     sim.pbc_ref.push_back(std::make_pair(WARP, IV2(0, 1)));
    //     sim.pbc_ref.push_back(std::make_pair(WARP, IV2(7, 8)));
    //     sim.pbc_ref.push_back(std::make_pair(WARP, IV2(14, 15)));
    //     sim.pbc_ref.push_back(std::make_pair(WEFT, IV2(2, 3)));
    //     sim.pbc_ref.push_back(std::make_pair(WEFT, IV2(9, 10)));
    //     sim.pbc_ref.push_back(std::make_pair(WEFT, IV2(16, 17)));

    //     // add periodic pairs to shift along peridic direction to compute bending
    //     sim.pbc_bending_pairs.push_back({0, 4, 11, 18, 1, 0});
    //     sim.pbc_bending_pairs.push_back({7, 5, 12, 19, 8, 1});
    //     sim.pbc_bending_pairs.push_back({14, 6, 13, 20, 15, 2});

    //     sim.pbc_bending_pairs.push_back({16, 18, 19, 20, 17, 5});
    //     sim.pbc_bending_pairs.push_back({9, 11, 12, 13, 10, 4});
    //     sim.pbc_bending_pairs.push_back({2, 4, 5, 6, 3, 3});

        

    //     // for coloring
    //     sim.yarns.push_back({0, 4, 11, 18, 1, WARP});
    //     sim.yarns.push_back({7, 5, 12, 19, 8, WARP});
    //     sim.yarns.push_back({14, 6, 13, 20, 15, WARP});

    //     sim.yarns.push_back({16, 18, 19, 20, 17, WEFT});
    //     sim.yarns.push_back({9, 11, 12, 13, 10, WEFT});
    //     sim.yarns.push_back({2, 4, 5, 6, 3, WEFT});

    //     for (int i = 0; i < sim.n_rods; i++)
    //         sim.yarn_map[i] = i;
        
    // }
    // else
    // {
    //     std::cout << "3D version this is not implemented" << std::endl;
    //     std::exit(0);
    // }
    
    // sim.q0 = q;
    
    // sim.n_dof = sim.n_nodes * sim.dof;
    // sim.W = StiffnessMatrix(sim.n_nodes * sim.dof, sim.n_dof);
    // sim.W.setIdentity();
}

template class UnitPatch<double, 3>;
template class UnitPatch<double, 2>;   