#include "EoLRodSim.h"

template<class T, int dim>
void EoLRodSim<T, dim>::build5NodeTestScene()
{
    add_shearing = true;
    add_stretching = true;
    add_bending = true;
    add_penalty = true;
    add_regularizor = true;

    n_nodes = 5;
    n_rods = 4;

    q = DOFStack(dof, n_nodes); q.setZero();
    rods = IV3Stack(3, n_rods); rods.setZero();
    connections = IV4Stack(4, n_nodes);

    normal = TV3Stack(3, n_rods);
    
    if constexpr (dim == 2)
    {
        q.col(0).template segment<dim>(0) = TV2(0, 0.5);
        q.col(0).template segment<2>(dim) = TV2(0, 0.5);

        q.col(1).template segment<dim>(0) = TV2(1, 0.5);
        q.col(1).template segment<2>(dim) = TV2(1, 0.5);

        q.col(2).template segment<dim>(0) = TV2(0.5, 0);
        q.col(2).template segment<2>(dim) = TV2(0.5, 0);

        q.col(3).template segment<dim>(0) = TV2(0.5, 1);
        q.col(3).template segment<2>(dim) = TV2(0.5, 1);

        q.col(4).template segment<dim>(0) = TV2(0.5, 0.5);
        q.col(4).template segment<2>(dim) = TV2(0.5, 0.5);

        rods.col(0) = IV3(0, 4, WARP);
        rods.col(1) = IV3(4, 1, WARP);
        rods.col(2) = IV3(2, 4, WEFT);
        rods.col(3) = IV3(4, 3, WEFT);

        normal.col(0) = TV3(0, 1, 0);
        normal.col(1) = TV3(0, 1, 0);
        normal.col(2) = TV3(0, -1, 0);
        normal.col(3) = TV3(0, -1, 0);

        TVDOF target, mask;
        mask.setOnes();
        target.setZero();
        
        //fully fix three nodes
        dirichlet_data[0] = std::make_pair(target, mask);
        dirichlet_data[1] = std::make_pair(target, mask);
        dirichlet_data[2] = std::make_pair(target, mask);
        
        //move top vertex
        target.template segment<dim>(0) = TV2(-0.1, 0);
        dirichlet_data[3] = std::make_pair(target, mask);

        // uncomment this to fix uv of the middle node
        // target.setZero();
        // mask.template segment<dim>(0) = TV::Zero();
        // dirichlet_data[4] = std::make_pair(target, mask);

        connections(2, 0) = -1; connections(3, 0) = -1; connections(0, 0) = -1; connections(1, 0) = 4; 
        connections(2, 1) = -1; connections(3, 1) = -1; connections(0, 1) = 4; connections(1, 1) = -1; 
        connections(2, 2) = -1; connections(3, 2) = 4; connections(0, 2) = -1; connections(1, 2) = -1; 
        connections(2, 3) = 4; connections(3, 3) = -1; connections(0, 3) = -1; connections(1, 3) = -1; 
        connections(2, 4) = 2; connections(3, 4) = 3; connections(0, 4) = 0; connections(1, 4) = 1; 
    }
    else
    {
        std::cout << "3D version this is not implemented" << std::endl;
        std::exit(0);
    }
    q0 = q;
}

template<class T, int dim>
void EoLRodSim<T, dim>::buildShearingTest()
{
    add_shearing = true;
    add_stretching = true;
    add_bending = false;
    add_penalty = true;
    add_regularizor = true;
    km = 1e-1;
    kc = 1e3;
    kx = 1;

    n_nodes = 5;
    n_rods = 4;

    q = DOFStack(dof, n_nodes); q.setZero();
    rods = IV3Stack(3, n_rods); rods.setZero();
    connections = IV4Stack(4, n_nodes);

    normal = TV3Stack(3, n_rods);

    TVDOF fix_eulerian, fix_lagrangian;
    fix_eulerian.setOnes();
    fix_lagrangian.setOnes();
    fix_lagrangian.template segment<2>(dim).setZero();
    fix_eulerian.template segment<dim>(0).setZero();

    if constexpr (dim == 2)
    {
        q.col(0).template segment<dim>(0) = TV2(0, 0.5);
        q.col(0).template segment<2>(dim) = TV2(0, 0.5);

        q.col(1).template segment<dim>(0) = TV2(1, 0.5);
        q.col(1).template segment<2>(dim) = TV2(1, 0.5);

        q.col(2).template segment<dim>(0) = TV2(0.75, 0);
        q.col(2).template segment<2>(dim) = TV2(0.75, 0);

        q.col(3).template segment<dim>(0) = TV2(0.25, 1);
        q.col(3).template segment<2>(dim) = TV2(0.25, 1);

        q.col(4).template segment<dim>(0) = TV2(0.5, 0.5);
        q.col(4).template segment<2>(dim) = TV2(0.5, 0.5);

        rods.col(0) = IV3(0, 4, WARP);
        rods.col(1) = IV3(4, 1, WARP);
        rods.col(2) = IV3(2, 4, WEFT);
        rods.col(3) = IV3(4, 3, WEFT);

        normal.col(0) = TV3(0, 1, 0);
        normal.col(1) = TV3(0, 1, 0);
        normal.col(2) = TV3(0, -1, 0);
        normal.col(3) = TV3(0, -1, 0);

        TVDOF target, mask;
        mask.setOnes();
        target.setZero();
        
        
        dirichlet_data[0] = std::make_pair(target, mask);
        dirichlet_data[1] = std::make_pair(target, mask);
        dirichlet_data[4] = std::make_pair(target, mask);
        dirichlet_data[2] = std::make_pair(target, fix_eulerian);
        dirichlet_data[3] = std::make_pair(target, fix_eulerian);


        connections(2, 0) = -1; connections(3, 0) = -1; connections(0, 0) = -1; connections(1, 0) = 4; 
        connections(2, 1) = -1; connections(3, 1) = -1; connections(0, 1) = 4; connections(1, 1) = -1; 
        connections(2, 2) = -1; connections(3, 2) = 4; connections(0, 2) = -1; connections(1, 2) = -1; 
        connections(2, 3) = 4; connections(3, 3) = -1; connections(0, 3) = -1; connections(1, 3) = -1; 
        connections(2, 4) = 2; connections(3, 4) = 3; connections(0, 4) = 0; connections(1, 4) = 1; 
    }
    else
    {
        std::cout << "3D version this is not implemented" << std::endl;
        std::exit(0);
    }
    q0 = q;
}

template<class T, int dim>
void EoLRodSim<T, dim>::buildLongRodForBendingTest()
{
    add_shearing = false;
    add_stretching = true;
    add_bending = true;
    add_penalty = true;
    add_regularizor = true;

    km = 1e-2;
    kc = 1e2;

    int n_row = 4;
    T offset_u = 1.0 / T(n_row + 1);
    T offset_v = 1.0 / 2.0;
    n_nodes = n_row * 3 + 2;
    n_rods = 2 * n_row + n_row + 1;

    q = DOFStack(dof, n_nodes); q.setZero();
    rods = IV3Stack(3, n_rods); rods.setZero();
    connections = IV4Stack(4, n_nodes);

    normal = TV3Stack(3, n_rods);

    TVDOF target, mask;
    mask.setOnes();
    
    target.setZero();

    if constexpr (dim == 2)
    {
        q.col(0).template segment<dim>(0) = TV2(0.5, 0);
        q.col(0).template segment<2>(dim) = TV2(0.5, 0);

        rods.col(0) = IV3(0, 2, WEFT);

        q.col(n_nodes-1).template segment<dim>(0) = TV2(0.5, 1);
        q.col(n_nodes-1).template segment<2>(dim) = TV2(0.5, 1);

        target[0] = -0.1;
        dirichlet_data[0] = std::make_pair(target, TVDOF::Ones());
        dirichlet_data[n_nodes-1] = std::make_pair(target, TVDOF::Ones());

        rods.col(n_rods - 1) = IV3(n_nodes-3, n_nodes-1, WEFT);

        connections(0, 0) = -1; connections(1, 0) = -1; connections(2, 0) = -1; connections(3, 0) = 2;
        connections(0, n_nodes-1) = -1; connections(1, n_nodes-1) = -1; 
        connections(2, n_nodes-1) = n_nodes-3; connections(3, n_nodes-1) = -1;

        int rod_cnt = 1;
        for(int i = 0; i < n_nodes - 2; i++)
        {
            q.col(i+1).template segment<dim>(0) = TV2((i%3) * offset_v, (std::ceil(i/3) + 1) * offset_u);
            q.col(i+1).template segment<2>(dim) = TV2((i%3) * offset_v, (std::ceil(i/3) + 1) * offset_u); 
            if(i%3 == 0)
            {
                dirichlet_data[i+1] = std::make_pair(TVDOF::Zero(), TVDOF::Ones());
                rods.col(rod_cnt++) = IV3(i + 1, i + 2, WARP);
                connections(0, i+1) = -1; connections(1, i+1) = i+2; 
                connections(2, i+1) = i == 0 ? -1 : i - 2; 
                connections(3, i+1) = i == n_nodes - 5 ? -1 : i+4;
            }
            if (i%3 == 1)
            {
                // dirichlet_data[i+1] = std::make_pair(TVDOF::Zero(), uv_mask);
                // if (i == 1 || i == 10)
                //     dirichlet_data[i+1] = std::make_pair(TVDOF::Zero(), fix_lagrangian);
                if (i == 1 || i == 10)
                    dirichlet_data[i+1] = std::make_pair(TVDOF::Zero(), TVDOF::Ones());
                // else
                    // dirichlet_data[i+1] = std::make_pair(TVDOF::Zero(), fix_eulerian);
                rods.col(rod_cnt++) = IV3(i + 1, i + 2, WARP);
                if (i < n_rods - 5)
                    rods.col(rod_cnt++) = IV3(i + 1, i + 4, WEFT);
                connections(0, i+1) = i; connections(1, i+1) = i+2; 
                connections(2, i+1) = i+1 == 2 ? 0 : i-2; 
                connections(3, i+1) = i+1 == n_nodes - 3? n_nodes -1 : i+4;
            }
            if (i%3 == 2)
            {
                dirichlet_data[i+1] = std::make_pair(TVDOF::Zero(), TVDOF::Ones());
                connections(0, i+1) = i; connections(1, i+1) = -1; 
                connections(2, i+1) = i == 2 ? -1 : i - 2; 
                connections(3, i+1) = i == n_nodes - 3 ? -1 : i+4;
            }
        }
        assert(rod_cnt == n_rods-2);
        std::cout << connections << std::endl;
        // std::exit(0);
    }
    else
    {
        std::cout << "3D version this is not implemented" << std::endl;
        std::exit(0);
    }
    q0 = q;
}

template<class T, int dim>
void EoLRodSim<T, dim>::buildRodNetwork(int width, int height)
{
    
    n_nodes = (width + 1) * (height + 1);
    n_rods = (width + 1) * height + (height + 1) * width;
    
    q = DOFStack(dof, n_nodes);
    rods = IV3Stack(3, n_rods);
    normal = TV3Stack(3, n_rods);
    q.setZero();
    rods.setZero();

    connections = IV4Stack(4, n_nodes);

    int cnt = 0;
    for(int i = 0; i < height+1; i++)
    {
        for(int j = 0; j < width+1; j++)
        {
            
            int idx = i * (width+1) + j;
            
            if constexpr (dim == 3)
                q.col(idx).template segment<dim>(0) = TV3(i/T(width), T(0), j/T(height));
            else if constexpr (dim == 2)
                q.col(idx).template segment<dim>(0) = TV2(i/T(width), j/T(height));

            q.col(idx).template segment<2>(dim) = TV2(i/T(width), j/T(height));
            if (i < height)
            {
                normal.col(cnt) = TV3(0, 1, 0);
                rods.col(cnt++) = IV3(i*(width+1) + j, (i + 1)*(width+1) + j, WARP);
            }
            if (j < width)
            {
                normal.col(cnt) = TV3(0, -1, 0);
                rods.col(cnt++) = IV3(i*(width+1) + j, i*(width+1) + j + 1, WEFT);
            }
            IV4 neighbor = IV4::Zero();
            neighbor[2] = (j - 1) < 0 ? -1 : i*(width+1) + j - 1;
            neighbor[3] = (j + 1) > width  ? -1 : i*(width+1) + j + 1;

            neighbor[0] = (i - 1) < 0 ? -1 : (i - 1)*(width+1) + j;
            neighbor[1] = (i + 1) > height  ? -1 : (i + 1)*(width+1) + j;
            connections.col(idx) = neighbor;
        }
    }
    q0 = q;
}


template<class T, int dim>
void EoLRodSim<T, dim>::buildPeriodicNetwork(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& C)
{
    auto shift_xy = [&](Eigen::Ref<DOFStack> q_shift, TV shift, int offset)
    {
        tbb::parallel_for(0, n_nodes, [&](int i){
            q_shift.col(i + offset).template segment<dim>(0) += shift;
        });
    };

    auto shift_rod = [&](Eigen::Ref<IV3Stack> rod_shift, int shift, int tile_id)
    {
        tbb::parallel_for(0, n_rods, [&](int i){
            rod_shift.col(i + n_rods * tile_id).segment<2>(0) += IV2(shift, shift);
        });
    };
    
    DOFStack q_tile = q;
    IV3Stack rods_tile = rods;
    TV3Stack normal_tile = normal;

    int n_tile = 9;
    q_tile.conservativeResize(dof, n_nodes * n_tile);
    rods_tile.conservativeResize(3, n_rods * n_tile);
    normal_tile.conservativeResize(dof, n_nodes * n_tile);

    tbb::parallel_for(0, n_nodes, [&](int node_idx){
        for(int i = 1; i < n_tile; i++)
        {
            q_tile.col(node_idx + i * n_nodes) = q.col(node_idx);
            normal_tile.col(node_idx + i * n_nodes) = normal.col(node_idx);
        }
    });

    tbb::parallel_for(0, n_rods, [&](int rod_idx){
        for(int i = 1; i < n_tile; i++)
            rods_tile.col(rod_idx + i * n_rods) = rods.col(rod_idx);
    });
    
    if constexpr (dim == 2)
    {
        TV ref0_shift = q.col(pbc_ref_unique[0](0)).template segment<dim>(0) - q.col(pbc_ref_unique[0](1)).template segment<dim>(0);
        TV ref1_shift = q.col(pbc_ref_unique[1](0)).template segment<dim>(0) - q.col(pbc_ref_unique[1](1)).template segment<dim>(0);

        shift_xy(q_tile, ref0_shift, n_nodes);
        shift_rod(rods_tile, n_nodes, 1);

        shift_xy(q_tile, ref1_shift, 2 * n_nodes);
        shift_rod(rods_tile, 2 * n_nodes, 2);

        shift_xy(q_tile, -ref0_shift, 3 * n_nodes);
        shift_rod(rods_tile, 3 * n_nodes, 3);

        shift_xy(q_tile, -ref1_shift, 4 * n_nodes);
        shift_rod(rods_tile, 4 * n_nodes, 4);

        shift_xy(q_tile, -ref0_shift, 5 * n_nodes);
        shift_xy(q_tile, ref1_shift, 5 * n_nodes);
        shift_rod(rods_tile, 5 * n_nodes, 5);

        shift_xy(q_tile, -ref0_shift, 6 * n_nodes);
        shift_xy(q_tile, -ref1_shift, 6 * n_nodes);
        shift_rod(rods_tile, 6 * n_nodes, 6);

        shift_xy(q_tile, -ref1_shift, 7 * n_nodes);
        shift_xy(q_tile, ref0_shift, 7 * n_nodes);
        shift_rod(rods_tile, 7 * n_nodes, 7);

        shift_xy(q_tile, ref0_shift, 8 * n_nodes);
        shift_xy(q_tile, ref1_shift, 8 * n_nodes);
        shift_rod(rods_tile, 8 * n_nodes, 8);
        
    }

    buildMeshFromRodNetwork(V, F, q_tile, rods_tile, normal_tile);
    
    C.resize(F.rows(), F.cols());
    tbb::parallel_for(0, int(F.rows()), [&](int i){
        if (i < n_rods * (40))
            C.row(i) = TV3(0, 1, 0);
        else
            C.row(i) = TV3(1, 1, 0);
    });
}


template<class T, int dim>
void EoLRodSim<T, dim>::buildPlanePeriodicBCScene3x3Subnodes(int sub_div)
{
    subdivide = true;
    buildPlanePeriodicBCScene3x3();
    if (sub_div > 1)
        subdivideRods(sub_div);
}

template<class T, int dim>
void EoLRodSim<T, dim>::subdivideRods(int sub_div)
{
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

    std::vector<IV3> rods_sub;
    // std::cout << "#nodes " << n_nodes << std::endl;
    int new_node_cnt = n_nodes;
    n_nodes = n_nodes + (sub_div-1) * n_rods;
    q.conservativeResize(dof, n_nodes);

    normal.resize(3, n_nodes);
    normal.setZero();
    IV4Stack new_connections(4, n_nodes);
    new_connections.setConstant(-1);
    

    for (int rod_idx = 0; rod_idx < n_rods; rod_idx++)
    {
        IV2 end_points = rods.col(rod_idx).template segment<2>(0);
        int node_i = end_points[0];
        int node_j = end_points[1];
        
        int yarn_type = rods(2, rod_idx);
        
        bool sign0 = connections.col(node_i).prod();
        int sign1 = connections.col(node_j).prod();
        
        // std::cout << "xi: " << q.col(node_i).template segment<dim>(0).transpose() << std::endl;
        // std::cout << "xj: "<< q.col(node_j).template segment<dim>(0).transpose() << std::endl;
        T fraction = T(1) / sub_div;
        bool new_node_added = false;
        bool left_or_bottom_bd = (connections(0, node_i) < 0 || connections(2, node_i) < 0);
        bool right_or_top_bd = (connections(1, node_j) < 0 || connections(3, node_j) < 0);
        int cnt = 0;
        for (int sub_cnt = 1; sub_cnt < sub_div; sub_cnt++)
        {
            T alpha = sub_cnt * fraction;

            // left or bottom boundary
            if (left_or_bottom_bd && alpha <= 0.5)
                continue;
            // right or top boundary
            if (right_or_top_bd && alpha >= 0.5)
                continue;
            
            if (left_or_bottom_bd)
                alpha = (alpha - 0.5) / 0.5;
            if (right_or_top_bd)
                alpha = alpha / 0.5;
            // std::cout << alpha << std::endl;
            q.col(new_node_cnt).template segment<dim>(0) = 
                q.col(node_i).template segment<dim>(0) * (1 - alpha) + 
                q.col(node_j).template segment<dim>(0) * alpha;
            q(dim + yarn_type, new_node_cnt) = q(dim + yarn_type, node_i) * (1 - alpha) + 
                q(dim + yarn_type, node_j) * alpha;
            // std::cout << "x sub: "<< q.col(new_node_cnt).template segment<dim>(0).transpose() << std::endl;
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
            yarn_map[rods_sub.size()-1] = rod_idx;
            setConnection(new_connections, n0, n1, yarn_type);
            dirichlet_data[new_node_cnt] = std::make_pair(TVDOF::Zero(), fix_eulerian);
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
        yarn_map[rods_sub.size()-1] = rod_idx;
        
    }
    n_rods = rods_sub.size();
    rods.resize(3, n_rods);
    tbb::parallel_for(0, n_rods, [&](int i){
        rods.col(i) = rods_sub[i];
    });
    connections = new_connections;

    std::vector<int> init(5, -1);
    for (int i = 0; i < 6; i++)
        pbc_bending_bn_pairs.push_back(init);
    
    auto add4Nodes = [&](int front, int end, int yarn_id, int rod_id, std::vector<std::vector<int>>& pairs)
    {
        if (rods(0, rod_id) == front)
        {
            pbc_bending_bn_pairs[yarn_id][0] = front;
            pbc_bending_bn_pairs[yarn_id][1] = rods(1, rod_id);
            pbc_bending_bn_pairs[yarn_id][4] = rods(2, rod_id);
        }
        if (rods(1, rod_id) == end)
        {
            pbc_bending_bn_pairs[yarn_id][3] = end;
            pbc_bending_bn_pairs[yarn_id][2] = rods(0, rod_id);
            pbc_bending_bn_pairs[yarn_id][4] = rods(2, rod_id);
        }
    };

    for (int i = 0; i < n_rods; i++)
    {
        add4Nodes(0, 1, 0, i, pbc_bending_bn_pairs);
        add4Nodes(7, 8, 1, i, pbc_bending_bn_pairs);
        add4Nodes(14, 15, 2, i, pbc_bending_bn_pairs);
        add4Nodes(16, 17, 3, i, pbc_bending_bn_pairs);
        add4Nodes(9, 10, 4, i, pbc_bending_bn_pairs);
        add4Nodes(2, 3, 5, i, pbc_bending_bn_pairs);
    }
    q.conservativeResize(dof, new_node_cnt);
    connections.conservativeResize(dof, new_node_cnt);
    n_nodes = new_node_cnt;
    normal.conservativeResize(dof, new_node_cnt);
    is_end_nodes = std::vector<bool>(n_nodes, false);
    // std::cout << "new # rods: " << n_rods << std::endl;
    // std::cout << rods.transpose() << std::endl;
    // std::cout << connections.transpose() << std::endl;
    // std::cout << new_node_cnt << " " << q.cols() << std::endl;
    
    q0 = q;
    // std::exit(0);
}

template<class T, int dim>
void EoLRodSim<T, dim>::buildPlanePeriodicBCScene3x3()
{
    
    pbc_ref_unique.clear();
    dirichlet_data.clear();
    pbc_ref.clear();
    pbc_bending_pairs.clear();
    yarns.clear();


    add_shearing = true;
    add_stretching = true;
    add_bending = true;
    add_penalty = false;
    add_regularizor = false;
    add_pbc = true;
    add_eularian_reg = false;

    ks = 1e1;
    kb = 1e-1;
    kb_penalty = 1e0;
    ke = 1e-3;
    
    km = 1e-4;
    kx = 1e2;
    kc = 1e3;
    k_pbc = 1e1;
    kr = 1e2;
    
    n_nodes = 21;
    n_rods = 24;

    q = DOFStack(dof, n_nodes); q.setZero();
    rods = IV3Stack(3, n_rods); rods.setZero();
    connections = IV4Stack(4, n_nodes).setOnes() * -1;
    
    normal = TV3Stack(3, n_rods);
    normal.setZero();

    T u_delta = 1.0 / 3.0, v_delta = 1.0 / 3.0;
    int cnt = 0;
    is_end_nodes.resize(n_nodes, true);

    if constexpr (dim == 2)
    {
        for (int i = 0; i < 3; i++)
        {
            q.col(cnt).template segment<dim>(0) = TV2(0, 0.5 * v_delta + i * v_delta );
            q.col(cnt++).template segment<2>(dim) = TV2(0, 0.5 * v_delta + i * v_delta);

            q.col(cnt).template segment<dim>(0) = TV2(1, 0.5 * v_delta + i * v_delta );
            q.col(cnt++).template segment<2>(dim) = TV2(1,0.5 * v_delta + i * v_delta);

            q.col(cnt).template segment<dim>(0) = TV2(0.5 * u_delta + i * u_delta, 0);
            q.col(cnt++).template segment<2>(dim) = TV2(0.5 * u_delta + i * u_delta, 0);

            q.col(cnt).template segment<dim>(0) = TV2(0.5 * u_delta + i * u_delta, 1);
            q.col(cnt++).template segment<2>(dim) = TV2(0.5 * u_delta + i * u_delta, 1);

            for (int j = 0; j < 3; j++)
            {
                is_end_nodes[cnt] = false;
                q.col(cnt).template segment<dim>(0) = TV2(0.5 * u_delta + i * u_delta, 0.5 * v_delta + j * v_delta);
                q.col(cnt++).template segment<2>(dim) = TV2(0.5 * u_delta + i * u_delta, 0.5 * v_delta + j * v_delta);
            }
        }
        assert(cnt == n_nodes);
        cnt = 0;
        
        rods.col(cnt++) = IV3(0, 4, WARP); rods.col(cnt++) = IV3(4, 11, WARP); rods.col(cnt++) = IV3(11, 18, WARP);rods.col(cnt++) = IV3(18, 1, WARP);
        rods.col(cnt++) = IV3(7, 5, WARP); rods.col(cnt++) = IV3(5, 12, WARP); rods.col(cnt++) = IV3(12, 19, WARP); rods.col(cnt++) = IV3(19, 8, WARP); 
        rods.col(cnt++) = IV3(14, 6, WARP); rods.col(cnt++) = IV3(6, 13, WARP); rods.col(cnt++) = IV3(13, 20, WARP); rods.col(cnt++) = IV3(20, 15, WARP); 

        rods.col(cnt++) = IV3(2, 4, WEFT); rods.col(cnt++) = IV3(4, 5, WEFT); rods.col(cnt++) = IV3(5, 6, WEFT); rods.col(cnt++) = IV3(6, 3, WEFT); 
        rods.col(cnt++) = IV3(9, 11, WEFT); rods.col(cnt++) = IV3(11, 12, WEFT);  rods.col(cnt++) = IV3(12, 13, WEFT); rods.col(cnt++) = IV3(13, 10, WEFT); 
        rods.col(cnt++) = IV3(16, 18, WEFT); rods.col(cnt++) = IV3(18, 19, WEFT); rods.col(cnt++) = IV3(19, 20, WEFT); rods.col(cnt++) = IV3(20, 17, WEFT);
        assert(cnt == n_rods);

        auto set_left_right = [&](Eigen::Ref<IV4Stack> connections, int idx, int left){
            connections(0, idx) = left;
            connections(1, left) = idx;
        };
        auto set_top_bottom = [&](Eigen::Ref<IV4Stack> connections, int idx, int top){
            connections(3, idx) = top;
            connections(2, top) = idx;
        };
        

        set_top_bottom(connections, 18, 1); set_top_bottom(connections, 19, 8); set_top_bottom(connections, 20, 15);
        set_top_bottom(connections, 11, 18); set_top_bottom(connections, 12, 19); set_top_bottom(connections, 13, 20);
        set_top_bottom(connections, 4, 11); set_top_bottom(connections, 5, 12); set_top_bottom(connections, 6, 13);
        set_top_bottom(connections, 0, 4); set_top_bottom(connections, 7, 5); set_top_bottom(connections, 14, 6);

        set_left_right(connections, 4, 2); set_left_right(connections, 11, 9); set_left_right(connections, 18, 16);
        set_left_right(connections, 5, 4); set_left_right(connections, 12, 11); set_left_right(connections, 19, 18);
        set_left_right(connections, 6, 5); set_left_right(connections, 13, 12); set_left_right(connections, 20, 19);
        set_left_right(connections, 3, 6); set_left_right(connections, 10, 13); set_left_right(connections, 17, 20);

        checkConnections();
        
        pbc_ref_unique.push_back(IV2(0, 1));
        pbc_ref_unique.push_back(IV2(2, 3));

        if (disable_sliding)
        {
            for(int i = 0; i < n_nodes; i++)
                dirichlet_data[i] = std::make_pair(TVDOF::Zero(), fix_eulerian);
        }
        else
        {
            dirichlet_data[2] = std::make_pair(TVDOF::Zero(), fix_eulerian);
            dirichlet_data[9] = std::make_pair(TVDOF::Zero(), fix_eulerian);
            dirichlet_data[16] = std::make_pair(TVDOF::Zero(), fix_eulerian);
            dirichlet_data[3] = std::make_pair(TVDOF::Zero(), fix_eulerian);
            dirichlet_data[10] = std::make_pair(TVDOF::Zero(), fix_eulerian);
            dirichlet_data[17] = std::make_pair(TVDOF::Zero(), fix_eulerian);

            dirichlet_data[1] = std::make_pair(TVDOF::Zero(), fix_eulerian);
            dirichlet_data[8] = std::make_pair(TVDOF::Zero(), fix_eulerian);
            dirichlet_data[15] = std::make_pair(TVDOF::Zero(), fix_eulerian);
            dirichlet_data[0] = std::make_pair(TVDOF::Zero(), fix_eulerian);
            dirichlet_data[7] = std::make_pair(TVDOF::Zero(), fix_eulerian);
            dirichlet_data[14] = std::make_pair(TVDOF::Zero(), fix_eulerian);
        }
            
        dirichlet_data[12] = std::make_pair(TVDOF::Zero(), fix_lagrangian);
        
        // add reference pairs
        pbc_ref.push_back(std::make_pair(WARP, IV2(0, 1)));
        pbc_ref.push_back(std::make_pair(WARP, IV2(7, 8)));
        pbc_ref.push_back(std::make_pair(WARP, IV2(14, 15)));
        pbc_ref.push_back(std::make_pair(WEFT, IV2(2, 3)));
        pbc_ref.push_back(std::make_pair(WEFT, IV2(9, 10)));
        pbc_ref.push_back(std::make_pair(WEFT, IV2(16, 17)));

        // add periodic pairs to shift along peridic direction to compute bending
        pbc_bending_pairs.push_back({0, 4, 11, 18, 1, 0});
        pbc_bending_pairs.push_back({7, 5, 12, 19, 8, 1});
        pbc_bending_pairs.push_back({14, 6, 13, 20, 15, 2});

        pbc_bending_pairs.push_back({16, 18, 19, 20, 17, 5});
        pbc_bending_pairs.push_back({9, 11, 12, 13, 10, 4});
        pbc_bending_pairs.push_back({2, 4, 5, 6, 3, 3});

        

        // for coloring
        yarns.push_back({0, 4, 11, 18, 1, WARP});
        yarns.push_back({7, 5, 12, 19, 8, WARP});
        yarns.push_back({14, 6, 13, 20, 15, WARP});

        yarns.push_back({16, 18, 19, 20, 17, WEFT});
        yarns.push_back({9, 11, 12, 13, 10, WEFT});
        yarns.push_back({2, 4, 5, 6, 3, WEFT});

        for (int i = 0; i < n_rods; i++)
            yarn_map[i] = i;
    }
    else
    {
        std::cout << "3D version this is not implemented" << std::endl;
        std::exit(0);
    }
    
    q0 = q;

}

template<class T, int dim>
void EoLRodSim<T, dim>::buildPlanePeriodicBCScene1x1()
{
    add_shearing = false;
    add_stretching = true;
    add_bending = true;
    add_penalty = true;
    add_regularizor = true;
    add_pbc = true;

    km = 1e-1;
    kc = 1e3;
    kx = 1e0;
    ks = 1.0;
    kb = 1e-1;
    k_pbc = 1e4;


    n_nodes = 5;
    n_rods = 4;

    q = DOFStack(dof, n_nodes); q.setZero();
    rods = IV3Stack(3, n_rods); rods.setZero();
    connections = IV4Stack(4, n_nodes).setOnes() * -1;
    
    normal = TV3Stack(3, n_rods);
    normal.setZero();
    
    int cnt = 0;
    is_end_nodes.resize(n_nodes, false);

    if constexpr (dim == 2)
    {
        // q.col(cnt).template segment<dim>(0) = TV2(0, 0.5);
        // q.col(cnt++).template segment<2>(dim) = TV2(0, 0.5);
        // q.col(cnt).template segment<dim>(0) = TV2(1, 0.5);
        // q.col(cnt++).template segment<2>(dim) = TV2(1, 0.5);
        // q.col(cnt).template segment<dim>(0) = TV2(0.5, 0.5);
        // q.col(cnt++).template segment<2>(dim) = TV2(0.5, 0.5);
        // q.col(cnt).template segment<dim>(0) = TV2(0.5, 0);
        // q.col(cnt++).template segment<2>(dim) = TV2(0.5, 0);
        // q.col(cnt).template segment<dim>(0) = TV2(0.5, 1);
        // q.col(cnt++).template segment<2>(dim) = TV2(0.5, 1);

        // assert(cnt == n_nodes);
        // cnt = 0;
        // rods.col(cnt++) = IV3(3, 2, WEFT);rods.col(cnt++) = IV3(2, 4, WEFT);
        // rods.col(cnt++) = IV3(0, 2, WARP);rods.col(cnt++) = IV3(2, 1, WARP);

        // assert(cnt == n_rods);

        // auto set_left_right = [&](Eigen::Ref<IV4Stack> connections, int idx, int left){
        //     connections(0, idx) = left;
        //     connections(1, left) = idx;
        // };
        // auto set_top_bottom = [&](Eigen::Ref<IV4Stack> connections, int idx, int top){
        //     connections(3, idx) = top;
        //     connections(2, top) = idx;
        // };
        
        // set_top_bottom(connections, 3, 2);set_top_bottom(connections, 2, 4); 
        // set_left_right(connections, 2, 0); set_left_right(connections, 1, 2);


        // checkConnections();

        // // for(int i = 0; i < n_nodes; i++)
        //     // dirichlet_data[i] = std::make_pair(TVDOF::Zero(), fix_eulerian);
            
        

        // dirichlet_data[0] = std::make_pair(TVDOF::Zero(), fix_eulerian);
        // dirichlet_data[1] = std::make_pair(TVDOF::Zero(), fix_eulerian);
        // dirichlet_data[3] = std::make_pair(TVDOF::Zero(), fix_eulerian);
        // dirichlet_data[4] = std::make_pair(TVDOF::Zero(), fix_eulerian);
        

        // // define BC pairs
        // pbc_ref[0] = IV2(0, 1);
        // pbc_ref[1] = IV2(3, 4);

        // // // define BC distance if required
        // pbc_translation[0] = TVDOF::Zero();
        // pbc_translation[0][0] = -1.5;
        // pbc_translation[0][1] = 0.5;
        // pbc_translation[0][2] = -1.;

        // pbc_pairs[IV2(0, 1)] = 0;
        
    }
    else
    {
        std::cout << "3D version this is not implemented" << std::endl;
        std::exit(0);
    }
    q0 = q;
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



template class EoLRodSim<double, 3>;
template class EoLRodSim<double, 2>;

//not working for float yet
// template class EoLRodSim<float>;