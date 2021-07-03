#include <iostream>
#include <utility>
#include <fstream>
#include <unordered_map>

#include "UnitPatch.h"
#include "HybridC2Curve.h"

template<class T, int dim>
void UnitPatch<T, dim>::buildScene(int patch_type)
{
    if (patch_type == 0)
        build3x3StraightRod();
    else if (patch_type == 1)
        buildStraightAndSineScene(64);
    else if (patch_type == 2)
        buildStraightAndHemiCircleScene(64);
    else if (patch_type == 3)
        buildStraightYarnScene(8);
    else if (patch_type == 4)
        buildTwoRodsScene(16);
    else if (patch_type == 5)
        buildSlidingTestScene(32);
    else if (patch_type == 6)
        buildDragCircleScene(32);
    else if (patch_type == 7)
        buildGripperScene(8);
    else if (patch_type == 8)
        buildUnitFromC2Curves(64);
    else if (patch_type == 9)
        buildCircleCrossScene(4);
}

template<class T, int dim>
void UnitPatch<T, dim>::addKeyPointsDoF(const std::vector<int>& key_points, 
        std::vector<Eigen::Triplet<T>>& w_entry,
        int& dof_cnt)
{
    for (int i = 0; i < key_points.size(); i++)
    {
        sim.dof_offsets[key_points[i]] = dof_cnt;
        for (int d = 0; d < sim.dof; d++)
            w_entry.push_back(Entry(key_points[i] * sim.dof + d, dof_cnt++, 1.0));
    }    
    
}

template<class T, int dim>
void UnitPatch<T, dim>::markCrossingDoF(std::vector<Eigen::Triplet<T>>& w_entry,
        int& dof_cnt)
{
    for (auto& crossing : sim.rod_crossings)
    {
        int node_idx = crossing->node_idx;
        std::vector<int> rods_involved = crossing->rods_involved;

        Vector<int, dim + 1> entry_rod0; 
        sim.Rods[rods_involved.front()]->getEntry(node_idx, entry_rod0);

        // push Lagrangian dof first
        for (int d = 0; d < dim; d++)
        {
            w_entry.push_back(Entry(entry_rod0[d], dof_cnt++, 1.0));
        }
        
        // push Eulerian dof for all rods
        for (int rod_idx : rods_involved)
        {
            sim.Rods[rod_idx]->getEntry(node_idx, entry_rod0);
            w_entry.push_back(Entry(entry_rod0[dim], dof_cnt++, 1.0));
        }
        
    }
}

template<class T, int dim>
void UnitPatch<T, dim>::markDoFSingleStrand(
        const std::vector<int>& nodes_on_strand, 
        const std::vector<int>& key_points_location_on_strand, 
        const std::vector<int>& key_points_location_global, 
        const std::vector<int>& key_points_location_dof, 
        std::vector<Eigen::Triplet<T>>& w_entry,
        int& dof_cnt, int yarn_type, bool closed)
{
    for (int k = 0; k < key_points_location_on_strand.size() - 1; k++)
    {
        int start = key_points_location_on_strand[k] + 1;
        int end = key_points_location_on_strand[k + 1];
        
        int left = key_points_location_dof[k];
        int right = key_points_location_dof[k + 1];
        
        int left_node = key_points_location_global[k];
        int right_node = key_points_location_global[k + 1];

        // std::cout << "start " << start << " end " << end << " left " << left << " right " << right << " left node " << left_node << " right node " << right_node << std::endl;

        for(int node_idx = start; node_idx < end; node_idx++)
        {
            // std::cout << nodes_on_strand[node_idx] << " ";
            sim.dof_offsets[nodes_on_strand[node_idx]] = dof_cnt;
            for(int d = 0; d < dim; d++)
            {
                w_entry.push_back(Eigen::Triplet<T>(nodes_on_strand[node_idx] * sim.dof + d, dof_cnt, 1.0));
                dof_cnt++;
            }   
            for(int d = dim; d < sim.dof; d++)
            {
                T alpha = (q(dim + yarn_type, nodes_on_strand[node_idx]) - q(dim + yarn_type, left_node)) / (q(dim + yarn_type, right_node) - q(dim + yarn_type, left_node));
                if (closed && k == key_points_location_on_strand.size() - 2)
                    alpha *= -1;
                // std::cout << "u current " << q(dim + yarn_type, nodes_on_strand[node_idx]) << " " << q(dim + yarn_type, left_node) << " " << q(dim + yarn_type, right_node) << std::endl;
                // std::cout << "alpha " << alpha << std::endl;
                // w_entry.push_back(Eigen::Triplet<T>(nodes_on_strand[node_idx] * sim.dof + d, left * sim.dof + d, 1 - alpha));
                // w_entry.push_back(Eigen::Triplet<T>(nodes_on_strand[node_idx] * sim.dof + d, right * sim.dof + d, alpha));
                w_entry.push_back(Eigen::Triplet<T>(nodes_on_strand[node_idx] * sim.dof + d, left * sim.dof + d, 1 - alpha));
                w_entry.push_back(Eigen::Triplet<T>(nodes_on_strand[node_idx] * sim.dof + d, right * sim.dof + d, alpha));
            }
        }
        // std::cout << std::endl;
        // std::getchar();
    }
}

// assuming passing points sorted long from to to direction
template<class T, int dim>
void UnitPatch<T, dim>::addStraightYarnCrossNPoints(const TV& from, const TV& to,
    const std::vector<TV>& passing_points, 
    const std::vector<int>& passing_points_id, int sub_div,
    std::vector<TV>& sub_points, std::vector<int>& node_idx, 
    std::vector<int>& key_points_location, 
    int start, bool pbc)
{
    
    int cnt = 1;
    if ((from - passing_points[0]).norm() < 1e-6 )
    {
        node_idx.push_back(passing_points_id[0]);
        cnt = 0;
    }
    else
    {
        node_idx.push_back(start);
        sub_points.push_back(from);
    }
    T length_yarn = (to - from).norm();
    TV length_vec = (to - from).normalized();
    
    TV loop_point = from;
    TV loop_left = from;
    for (int i = 0; i < passing_points.size(); i++)
    {
        if ((from - passing_points[i]).norm() < 1e-6 )
        {
            key_points_location.push_back(0);
            continue;
        }
        T fraction = (passing_points[i] - loop_point).norm() / length_yarn;
        int n_sub_nodes = std::ceil(fraction * sub_div);
        T length_sub = (passing_points[i] - loop_point).norm() / T(n_sub_nodes);
        for (int j = 0; j < n_sub_nodes - 1; j++)
        {
            sub_points.push_back(loop_left + length_sub * length_vec);
            loop_left = sub_points.back();
            node_idx.push_back(start + cnt);
            cnt++;
        }
        node_idx.push_back(passing_points_id[i]);
        key_points_location.push_back(cnt + i);
        loop_point = passing_points[i];
        loop_left = passing_points[i];
    }
    if (passing_points.size())
    {
        if ((passing_points.back() - to).norm() < 1e-6)
            return;
    }
    T fraction;
    int n_sub_nodes;
    T length_sub;
    if( passing_points.size() )
    {
        fraction = (to - passing_points.back()).norm() / length_yarn;
        n_sub_nodes = std::ceil(fraction * sub_div) + 1;
        length_sub = (to - passing_points.back()).norm() / T(n_sub_nodes);
    }
    else
    {
        n_sub_nodes = sub_div + 1;
        length_sub = (to - from).norm() / T(sub_div);
    }
    for (int j = 0; j < n_sub_nodes - 1; j++)
    {
        if (j == 0)
        {
            if(passing_points.size())
            {
                sub_points.push_back(passing_points.back() + length_sub * length_vec);
                loop_left = sub_points.back();
            }
        }
        else
        {
            sub_points.push_back(loop_left + length_sub * length_vec);
            loop_left = sub_points.back();
        }
        if(passing_points.size() == 0 && j == 0)
            continue;
        node_idx.push_back(start + cnt);
        cnt++;
    }
    node_idx.push_back(start + cnt);
    sub_points.push_back(to);
}

template<class T, int dim>
void UnitPatch<T, dim>::buildUnitFromC2Curves(int sub_div)
{
    auto unit_yarn_map = sim.yarn_map;
    sim.yarn_map.clear();
    sim.add_rotation_penalty = false;
    sim.add_pbc_bending = false;
    sim.subdivide = true;

    clearSimData();
    std::vector<Eigen::Triplet<T>> w_entry;

    if constexpr (dim == 2)
    {
        std::string data_points_file = "/home/yueli/Documents/ETH/WuKong/Projects/DigitalFabrics/Data/test.curve";
        T x, y;

        std::ifstream in(data_points_file);

        HybridC2Curve<T, dim>* curve = new HybridC2Curve<T, dim>(sub_div);
        while(in >> x >> y)
        {
            if (sim.run_diff_test)
                curve->data_points.push_back(TV(x, y));
            else
                curve->data_points.push_back(TV(x, y) * 0.03);
        }
        in.close();
        curve->normalizeDataPoints();
        std::vector<TV> points_on_curve;
        // curve->getLinearSegments(points_on_curve);
        curve->sampleCurves(points_on_curve);

        // Rod<T, dim>* r0 = new Rod<T, dim>(sim.deformed_states);

        
        
        sim.n_nodes = points_on_curve.size();
        sim.n_rods = points_on_curve.size() - 1;

        q = DOFStack(sim.dof, sim.n_nodes); q.setZero();
        rods = IV3Stack(3, sim.n_rods); rods.setZero();
        connections = IV4Stack(4, sim.n_nodes).setOnes() * -1;

        sim.normal = TV3Stack(3, sim.n_rods);
        sim.normal.setZero();
        
        int cnt = 0, dof_cnt = 0;
        T arc_length_sum = 0;
        
        std::vector<int> rod0;
        for (int i = 0; i < sim.n_nodes; i++)
        {
            setPos(i, points_on_curve[i], points_on_curve[i]);
            rod0.push_back(i);
            q(dim, i) = T(i) * (curve->data_points.size() - 1 )/ (sim.n_nodes - 1);
        }

        sim.deformed_states.resize(sim.n_nodes * (dim + 1));
        for (int i = 0; i < sim.n_nodes; i++)
        {
            for(int d = 0; d < dim + 1; d++)
            {
                sim.deformed_states[i*(dim + 1) + d] = q(d, i);
            }
        }

        // r0->indices = rod0;
        sim.dof_offsets.resize(sim.n_nodes, 0);
        // rod0 0, points_on_curve.size()

        int offset_cnt = 0;
        for (int i = 0; i < sim.n_nodes; i++)
        {
            sim.dof_offsets[i] = dof_cnt;
            // r0->offset.push_back(offset_cnt);
            offset_cnt += 3;
            for(int d = 0; d < dim; d++)
            {
                w_entry.push_back(Eigen::Triplet<T>(i * sim.dof + d, dof_cnt, 1.0));
                dof_cnt++;
            }
            if (i == 0 || i == sim.n_nodes - 1)
                for(int d = dim; d < sim.dof; d++)
                {
                    w_entry.push_back(Eigen::Triplet<T>(i * sim.dof + d, dof_cnt, 1.0));
                    dof_cnt++;
                }
            else
            {    
                int idx_last = sim.dof + (points_on_curve.size() - 2) * dim;
                
                T alpha = q(dim, i);
                for(int d = dim; d < sim.dof; d++)
                {
                    w_entry.push_back(Eigen::Triplet<T>(i * sim.dof + d, d, 1 - alpha));
                    w_entry.push_back(Eigen::Triplet<T>(i * sim.dof + d, idx_last + d, alpha));
                }
            }

        }
        
        // q *= 0.03;
        
        std::vector<T> data_points_discrete_arc_length;
        
        for(int i = 0; i < curve->data_points.size(); i++)
        {
            data_points_discrete_arc_length.push_back(q(dim, i*sub_div/2));
        }
        
        addRods(rod0, WARP, cnt, 0);

        sim.pbc_ref_unique.push_back(IV2(0, rod0.back()));
        // sim.pbc_ref_unique.push_back(IV2(WEFT, rod0[rod0.size()-1]));

        sim.pbc_ref.push_back(std::make_pair(WARP, IV2(0, rod0.back())));
        // sim.pbc_ref.push_back(std::make_pair(WEFT, IV2(0, rod0[rod0.size()-1])));

        sim.is_end_nodes = std::vector<bool>(sim.n_nodes, false);

        // if (sim.disable_sliding)
        {
            // for(int i = 0; i < sim.n_nodes; i++)
                // sim.dirichlet_data[i] = std::make_pair(TVDOF::Zero(), sim.fix_eulerian);
            sim.dirichlet_data[points_on_curve.size() - 1] = std::make_pair(TVDOF::Zero(), sim.fix_eulerian);
            sim.dirichlet_data[0] = std::make_pair(TVDOF::Zero(), sim.fix_all);
        }
        sim.q0 = q;
        // dof_cnt = sim.n_nodes * sim.dof;
        sim.n_dof = dof_cnt;
        sim.W = StiffnessMatrix(sim.n_nodes * sim.dof, sim.n_dof);
        sim.W.setFromTriplets(w_entry.begin(), w_entry.end());
        // sim.W.setIdentity();
        // std::cout << sim.W << std::endl;
        // for (int i = 0; i < sim.n_nodes; i++)
        //     if(sim.dirichlet_data.find(i) == sim.dirichlet_data.end())
        //         sim.sliding_nodes.push_back(i);

        sim.slide_over_n_rods = IV2(std::floor(sub_div * 0.2), std::floor(sub_div * 0.2));
        T rod_length = (sim.q0.col(rods.col(0)(0)).template segment<dim>(0) - 
            sim.q0.col(rods.col(0)(1)).template segment<dim>(0)).norm();
        sim.tunnel_u = sim.slide_over_n_rods[0] * rod_length;
        sim.tunnel_v = sim.tunnel_u;

        // sim.curvature_functions.push_back(new LineCurvature<T, dim>());
        Vector<T, dim + 1> q0 = q.col(0).template segment<dim + 1>(0);
        Vector<T, dim + 1> q1 = q.col(rod0.back()).template segment<dim + 1>(0);
        
        DiscreteHybridCurvature<T, dim>* curve_func = new DiscreteHybridCurvature<T, dim>(
            q0, q1);
        curve_func->setData(curve, data_points_discrete_arc_length);
        sim.curvature_functions.push_back(curve_func);

        // r0->rest_state = curve_func;


        // std::cout << q.transpose() << std::endl;
        // sim.Rods.push_back(r0);
    }

    // sim.checkMaterialPositionDerivatives();
}



template<class T, int dim>
void UnitPatch<T, dim>::subdivideStraightYarns(int sub_div)
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

    q.conservativeResize(sim.dof, new_node_cnt);
    connections.conservativeResize(sim.dof, new_node_cnt);
    sim.n_nodes = new_node_cnt;
    
    sim.normal.conservativeResize(sim.dof, new_node_cnt);
    
    sim.q0 = q;
    sim.n_dof = dof_cnt;
    sim.W = StiffnessMatrix(sim.n_nodes * sim.dof, sim.n_dof);
    sim.W.setFromTriplets(w_entry.begin(), w_entry.end());
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
void UnitPatch<T, dim>::buildGripperScene(int sub_div)
{
    if (sim.run_diff_test)
        sim.unit = 1;
    auto unit_yarn_map = sim.yarn_map;
    sim.yarn_map.clear();
    sim.add_rotation_penalty = false;
    sim.add_pbc_bending = false;
    sim.subdivide = true;

    clearSimData();
    std::vector<Eigen::Triplet<T>> w_entry;

    if constexpr (dim == 2)
    {   
        T theta = M_PI / 4.0;
        HybridC2Curve<T, dim>* curve = new HybridC2Curve<T, dim>(sub_div);
        curve->data_points.push_back(TV(0.5, -0.5) * sim.unit);
        curve->data_points.push_back(TV(0, 0) * sim.unit);
        curve->data_points.push_back(TV(0.5 - 0.5 * std::cos(theta), 0.5 + 0.5 * std::sin(theta)) * sim.unit);
        curve->data_points.push_back(TV(0.5, 0.0) * sim.unit);
        curve->data_points.push_back(TV(0.5 + 0.5 * std::cos(theta), 0.5 + 0.5 * std::sin(theta)) * sim.unit);
        curve->data_points.push_back(TV(1.0, 0.0) * sim.unit);
        curve->data_points.push_back(TV(0.5, -0.5) * sim.unit);
        std::vector<TV> points_on_curve;
        curve->sampleCurves(points_on_curve);

        sim.n_nodes = points_on_curve.size() - 1;
        int n_points_circle = sim.n_nodes;
        sim.n_rods = points_on_curve.size() - 1;

        q = DOFStack(sim.dof, sim.n_nodes); q.setZero();
        
        int cnt = 0, dof_cnt = 0;

        std::vector<int> rod0;
        for (int i = 0; i < sim.n_nodes; i++)
        {
            setPos(i, points_on_curve[i], points_on_curve[i]);
            rod0.push_back(i);
            q(dim, i) = T(i) * (curve->data_points.size() - 1 )/ (sim.n_nodes);
        }
        rod0.push_back(0);
        
        TV rod1_from(0.5, -1); rod1_from *= sim.unit;
        TV rod1_to = curve->data_points[3];

        std::vector<int> rod1, key_points_location_rod1;
        std::vector<TV> points_rod1;

        addStraightYarnCrossNPoints(rod1_from, rod1_to, 
                {curve->data_points[0], curve->data_points[3]}, 
                {0, 3 * sub_div/2}, 
                sub_div, 
                points_rod1, rod1, 
                key_points_location_rod1,
                sim.n_nodes);
        

        // for(int idx : rod1)
        // {
        //     std::cout << "idx " << idx << " " << q.col(idx).transpose() << std::endl;
        // }

        sim.n_nodes += points_rod1.size();
        sim.n_rods += rod1.size() - 1;

        q.conservativeResize(sim.dof, sim.n_nodes);
        rods = IV3Stack(3, sim.n_rods); rods.setZero();
        connections = IV4Stack(4, sim.n_nodes).setOnes() * -1;   

        for (int i = 0; i < points_rod1.size(); i++)
        {
            int idx = n_points_circle + i;
            setPos(idx, points_rod1[i], points_rod1[i]);
            q(dim + 1, idx) = LDis(rod1.front(), idx) / (rod1_to - rod1_from).norm();
            // std::cout << "idx " << idx << " u " << q.col(idx).transpose() << std::endl;
        }
        
        q(dim + 1, 0) = LDis(rod1.front(), 0) / (rod1_to - rod1_from).norm();
        q(dim + 1, 3 * sub_div / 2) = LDis(rod1.front(), 3 * sub_div / 2) / (rod1_to - rod1_from).norm();

        sim.dof_offsets.resize(sim.n_nodes, 0);

        std::vector<int> key_points = { rod0.front(), 
                                        rod1.front(),
                                        rod1.back()
                                    };

        addKeyPointsDoF(key_points, w_entry, dof_cnt);

        std::vector<int> rod0_kp_on_strand = { 0, 3 * sub_div / 2, n_points_circle};
        std::vector<int> rod0_kp_global = {0, 3 * sub_div / 2, 0};
        std::vector<int> rod0_kp_dof = { 0, 2, 0 };

        markDoFSingleStrand(rod0, rod0_kp_on_strand,
            rod0_kp_global, 
            rod0_kp_dof,
            w_entry, dof_cnt, WARP, true);

        std::vector<int> rod1_kp_on_strand = { 0, key_points_location_rod1[0], key_points_location_rod1[1]};
        std::vector<int> rod1_kp_global = { n_points_circle, 0, 3 * sub_div / 2};
        std::vector<int> rod1_kp_dof = { 1, 0, 2 };
        
        markDoFSingleStrand(rod1, 
            rod1_kp_on_strand,
            rod1_kp_global,
            rod1_kp_dof,
             w_entry, dof_cnt, WEFT);

        std::vector<T> data_points_discrete_arc_length;
        
        for(int i = 0; i < curve->data_points.size(); i++)
        {
            data_points_discrete_arc_length.push_back(q(dim, i*sub_div/2));
        }
        
        addRods(rod0, WARP, cnt, 0);
        addRods(rod1, WEFT, cnt, 1);

        sim.pbc_ref_unique.push_back(IV2(sub_div/2, sub_div/2 * 3));
        sim.pbc_ref_unique.push_back(IV2(rod1.front(), rod1.back()));

        sim.pbc_ref.push_back(std::make_pair(WARP, IV2(sub_div/2, sub_div/2 * 3)));
        sim.pbc_ref.push_back(std::make_pair(WEFT, IV2(rod1.front(), rod1.back())));

        sim.is_end_nodes = std::vector<bool>(sim.n_nodes, false);

        if(true)
        {
            // sim.dirichlet_data[rod0.back()] = std::make_pair(TVDOF::Zero(), sim.fix_all);
            TVDOF mask_xyu = TVDOF::Ones(); mask_xyu[3] = 0.0;

            sim.dirichlet_data[rod0.front()] = std::make_pair(TVDOF::Zero(), mask_xyu);
            
            sim.dirichlet_data[rod1.back()] = std::make_pair(TVDOF::Zero(), sim.fix_eulerian);

            TVDOF shift_x = TVDOF::Zero(), mask_x = TVDOF::Zero();
            shift_x[1] = -1.5 * sim.unit;
            mask_x[1] = 1; mask_x[2] = 1; mask_x[3] = 1;
            sim.dirichlet_data[rod1.front()] = std::make_pair(shift_x, mask_x);
        }
       

        sim.sliding_nodes.push_back(0);

        sim.q0 = q;
        
        sim.n_dof = dof_cnt;
        sim.W = StiffnessMatrix(sim.n_nodes * sim.dof, sim.n_dof);
        sim.W.setFromTriplets(w_entry.begin(), w_entry.end());


        sim.slide_over_n_rods = IV2(0, std::floor(sub_div * 0.4));

        T r0 = 0;
        T r1 = sim.q0(dim + 1, rod1[1]) - sim.q0(dim + 1, rod1[0]);

        sim.tunnel_u = sim.slide_over_n_rods[0] * r0;
        sim.tunnel_v = sim.slide_over_n_rods[1] * r1;


        Vector<T, dim + 1> q0 = q.col(0).template segment<dim + 1>(0);
        Vector<T, dim + 1> q1 = q.col(rod0.back()).template segment<dim + 1>(0);
        
        DiscreteHybridCurvature<T, dim>* curve_func = new DiscreteHybridCurvature<T, dim>(
            q0, q1);
        sim.curvature_functions.push_back(curve_func);
        curve_func->setData(curve, data_points_discrete_arc_length);

        q0.template segment<dim>(0) = rod1_from; q1.template segment<dim>(0) = rod1_to;
        q0(dim) = q(dim + 1, rod1.front()); q1(dim) = q(dim + 1, rod1.back());
        sim.curvature_functions.push_back(new LineCurvature<T, dim>(q0, q1));
        
        // std::cout << q0.transpose() << std::endl;
        // std::cout << q1.transpose() << std::endl;

        sim.normal = TV3Stack(3, sim.n_rods);
        sim.normal.setZero();

        // std::cout << q.transpose() << std::endl;
        // std::cout << rods.transpose() << std::endl;

        // for (int i : rod1)
        //     std::cout << i << " ";
        // std::cout << std::endl;
        // std::cout << sim.W.rows() << " " << sim.W.cols() << std::endl;
        // std::cout << sim.W << std::endl;
    }
}

template<class T, int dim>
void UnitPatch<T, dim>::buildCircleCrossScene(int sub_div)
{
    if (sim.run_diff_test)
        sim.unit = 1;
    auto unit_yarn_map = sim.yarn_map;
    sim.yarn_map.clear();
    sim.add_rotation_penalty = false;
    sim.add_pbc_bending = false;
    sim.subdivide = true;

    clearSimData();
    std::vector<Eigen::Triplet<T>> w_entry;

    if constexpr (dim == 2)
    {   
        int sub_div_2 = sub_div / 2;
        sim.new_frame_work = true;

        T theta = M_PI / 4.0;
        
        TV center = TV(0.5, 0.5) * sim.unit;
        
        HybridC2Curve<T, dim>* curve = new HybridC2Curve<T, dim>(sub_div);
        curve->data_points.push_back(TV(0.5 - 0.5 * std::cos(theta), 0.5 + 0.5 * std::sin(theta)) * sim.unit);
        curve->data_points.push_back(TV(0.5 + 0.5 * std::cos(theta), 0.5 + 0.5 * std::sin(theta)) * sim.unit);
        curve->data_points.push_back(TV(0.5 + 0.5 * std::cos(theta), 0.5 - 0.5 * std::sin(theta)) * sim.unit);
        curve->data_points.push_back(TV(0.5 - 0.5 * std::cos(theta), 0.5 - 0.5 * std::sin(theta)) * sim.unit);
        curve->data_points.push_back(TV(0.5 - 0.5 * std::cos(theta), 0.5 + 0.5 * std::sin(theta)) * sim.unit);

        std::vector<TV> points_on_curve;
        curve->sampleCurves(points_on_curve);

        int full_dof_cnt = 0;
        int node_cnt = 0;

        deformed_states.resize((points_on_curve.size() - 1) * (dim + 1));
        
        // rod0 is the circle
        // there are four nodes crossed by the straight yarns        
        Rod<T, dim>* r0 = new Rod<T, dim>(deformed_states, 0, true);
        std::unordered_map<int, Vector<int, dim + 1>> offset_map;
        std::vector<int> node_index_list, dof_offset;
        std::vector<T> data_points_discrete_arc_length;
        
        for (int i = 0; i < points_on_curve.size() - 1; i++)
        {
            offset_map[i] = Vector<int, dim + 1>::Zero();
            node_cnt++;
            node_index_list.push_back(i);
            //push Lagrangian DoF
            deformed_states.template segment<dim>(full_dof_cnt) = points_on_curve[i];
            offset_map[i][0] = full_dof_cnt++;
            offset_map[i][1] = full_dof_cnt++;
            // push Eulerian DoF
            deformed_states[full_dof_cnt] = T(i) * (curve->data_points.size() - 1 ) / (points_on_curve.size() - 1);
            offset_map[i][2] = full_dof_cnt++;
            
        }
        // std::cout << deformed_states << std::endl;
        // // std::getchar();
        // for (auto& it: offset_map) {
        //     std::cout << it.first << " " << it.second.transpose() << std::endl;
        // }
        // std::getchar();
        

        node_index_list.push_back(0); // closed curve
        r0->offset_map = offset_map;
        r0->indices = node_index_list;
    
        for(int i = 0; i < curve->data_points.size(); i++)
            data_points_discrete_arc_length.push_back(deformed_states[i*sub_div_2*(dim+1)+dim]);

        Vector<T, dim + 1> q0, q1;
        r0->frontDoF(q0); r0->backDoF(q1);

        DiscreteHybridCurvature<T, dim>* rest_state_rod0 = new DiscreteHybridCurvature<T, dim>(q0, q1);
        sim.curvature_functions.push_back(rest_state_rod0);
        rest_state_rod0->setData(curve, data_points_discrete_arc_length);

        r0->rest_state = rest_state_rod0;
        r0->validCheck();
        r0->dof_node_location = {0, 2 * sub_div_2};
        sim.Rods.push_back(r0);

        auto dof_pt0 = offset_map[0];
        auto dof_pt2 = offset_map[sub_div_2*2];
        
        offset_map.clear();
        // ========================================= rod 0 is done now =========================================
        
        // add the center
        int center_id = node_cnt;
        deformed_states.conservativeResize(full_dof_cnt + dim);
        deformed_states.template segment<dim>(full_dof_cnt) = center;
        offset_map[node_cnt] = Vector<int, dim + 1>::Zero();
        offset_map[node_cnt][0] = full_dof_cnt++;
        offset_map[node_cnt][1] = full_dof_cnt++;
        node_cnt++;
        // std::cout<< deformed_states << std::endl;
        // std::getchar();

        Rod<T, dim>* r1 = new Rod<T, dim>(deformed_states, 1);

        TV dir_rod1 = (curve->data_points[2] - curve->data_points[0]).normalized();
        TV rod1_from = curve->data_points[0] - dir_rod1 * 0.1 * sim.unit;
        TV rod1_to = curve->data_points[2] + dir_rod1 * 0.1 * sim.unit;

        std::vector<int> rod1, key_points_location_rod1;
        std::vector<TV> points_rod1;
        
        addStraightYarnCrossNPoints(rod1_from, rod1_to, 
                {curve->data_points[0], center, curve->data_points[2]}, 
                {0, center_id, 2 * sub_div_2}, 
                sub_div, 
                points_rod1, rod1, 
                key_points_location_rod1,
                node_cnt);
        // std::cout << "point1 size " << points_rod1.size() << std::endl;
        // std::cout << "full_dof_cnt " << full_dof_cnt << std::endl;


        // r1->dof_node_location = key_points_location_rod1;
        r1->dof_node_location = { key_points_location_rod1.front(), key_points_location_rod1.back() };
        for(TV pt : points_rod1)
            std::cout << pt.transpose() << std::endl;

        
        deformed_states.conservativeResize(full_dof_cnt + (points_rod1.size() - 1) * (dim + 1) + 3);
        r1->indices = rod1;
        
        
        for (int i = 0; i < points_rod1.size(); i++)
        {
            offset_map[node_cnt] = Vector<int, dim + 1>::Zero();
            // dof_offset.push_back(full_dof_cnt);
            //push Lagrangian DoF
            deformed_states.template segment<dim>(full_dof_cnt) = points_rod1[i];
            offset_map[node_cnt][0] = full_dof_cnt++;
            offset_map[node_cnt][1] = full_dof_cnt++;
            // push Eulerian DoF
            deformed_states[full_dof_cnt] = (points_rod1[i] - rod1_from).norm() / (rod1_to - rod1_from).norm();
            offset_map[node_cnt][2] = full_dof_cnt++;
            node_cnt++;
        }

        deformed_states[full_dof_cnt] = (center - rod1_from).norm() / (rod1_to - rod1_from).norm();
        offset_map[center_id][dim] = full_dof_cnt++;

        deformed_states[full_dof_cnt] = (curve->data_points[0] - rod1_from).norm() / (rod1_to - rod1_from).norm();
        offset_map[0] = Vector<int, dim + 1>::Zero();
        offset_map[0][dim] = full_dof_cnt++;
        offset_map[0].template segment<dim>(0) = dof_pt0.template segment<dim>(0);

        deformed_states[full_dof_cnt] = (curve->data_points[2] - rod1_from).norm() / (rod1_to - rod1_from).norm();
        offset_map[sub_div_2 * 2] = Vector<int, dim + 1>::Zero();
        offset_map[sub_div_2 * 2][dim] = full_dof_cnt++;
        offset_map[sub_div_2 * 2].template segment<dim>(0) = dof_pt2.template segment<dim>(0);

        // std::cout << deformed_states.rows() << std::endl;
        // std::getchar();
        // std::cout << deformed_states << std::endl;
        // 
        // std::getchar();
        // for (int i : rod1)
        //     std::cout << i << " ";
        // std::cout << std::endl;
        // for (auto& it: offset_map) {
        //     std::cout << it.first << " " << it.second.transpose() << std::endl;
        // }
        r1->offset_map = offset_map;
        r1->frontDoF(q0); r1->backDoF(q1);
        
        r1->rest_state = new LineCurvature<T, dim>(q0, q1);
        r1->validCheck();
        sim.Rods.push_back(r1);

        sim.rod_crossings.push_back(new RodCrossing<T, dim>(0, {0, 1} ));
        sim.rod_crossings.push_back(new RodCrossing<T, dim>(sub_div_2 * 2, {0, 1} ));

        int dof_cnt = 0;
        markCrossingDoF(w_entry, dof_cnt);
        r0->markDoF(w_entry, dof_cnt);
        r1->markDoF(w_entry, dof_cnt);

        for (int d = 0; d < dim + 1; d++)
            sim.dirichlet_dof[q0[d]] = 0;
        for (int d = 0; d < dim + 1; d++)
            sim.dirichlet_dof[q1[d]] = 0;
        // std::cout << node_cnt << " " <<  full_dof_cnt << " " << dof_cnt << std::endl;
        // std::vector<int> key_points = { rod0.front(),
        //                                 rod1.front(),
        //                                 rod1.back(),
        //                                 rod2.front()
        //                             };

        // addKeyPointsDoF(key_points, w_entry, dof_cnt);
        
        // std::vector<int> rod0_kp_on_strand = { 0, 2 * sub_div / 2};
        // std::vector<int> rod0_kp_global = {0, 2 * sub_div / 2};
        // std::vector<int> rod0_kp_dof = { 0, 1};

        // markDoFSingleStrand(rod0, rod0_kp_on_strand,
        //     rod0_kp_global, 
        //     rod0_kp_dof,
        //     w_entry, dof_cnt, WARP, true);
        
        // // std::cout << "key_points_location_rod1[0] " << key_points_location_rod1[0] << std::endl;
        // // std::cout << "key_points_location_rod1[1] " << key_points_location_rod1[1] << std::endl;

        // std::vector<int> rod1_kp_on_strand = { key_points_location_rod1[0], int(rod1.size()) - 1 };
        // std::vector<int> rod1_kp_global = { 2 * sub_div / 2, sim.n_nodes - 1};
        // std::vector<int> rod1_kp_dof = { 1, 2 };
        
        // markDoFSingleStrand(rod1, 
        //     rod1_kp_on_strand,
        //     rod1_kp_global,
        //     rod1_kp_dof,
        //      w_entry, dof_cnt, WEFT);

        // std::vector<int> rod2_kp_on_strand = { 0, int(rod2.size()) - 1 };
        // std::vector<int> rod2_kp_global = { node_cnt, 0};
        // std::vector<int> rod2_kp_dof = { 3, 0 };
        
        // markDoFSingleStrand(rod2, 
        //     rod2_kp_on_strand,
        //     rod2_kp_global,
        //     rod2_kp_dof,
        //      w_entry, dof_cnt, WEFT);
        
        
        // std::vector<T> data_points_discrete_arc_length;
        
        // for(int i = 0; i < curve->data_points.size(); i++)
        // {
        //     data_points_discrete_arc_length.push_back(q(dim, i*sub_div/2));
        // }
        
        // addRods(rod0, WARP, cnt, 0);
        // addRods(rod1, WEFT, cnt, 1);
        // addRods(rod2, WEFT, cnt, 2);

        // sim.pbc_ref_unique.push_back(IV2(3 * sub_div / 2, 1 *sub_div/2));
        // sim.pbc_ref_unique.push_back(IV2(rod2.front(), rod1.back()));

        // sim.pbc_ref.push_back(std::make_pair(WARP, IV2(3 * sub_div / 2, 1 *sub_div/2)));
        // sim.pbc_ref.push_back(std::make_pair(WEFT, IV2(rod2.front(), rod1.back())));

        // sim.is_end_nodes = std::vector<bool>(sim.n_nodes, false);

        // if(true)
        // {
        //     // // sim.dirichlet_data[rod0.back()] = std::make_pair(TVDOF::Zero(), sim.fix_all);
        //     // TVDOF mask_xyu = TVDOF::Ones(); mask_xyu[3] = 0.0;

        //     // sim.dirichlet_data[rod0.front()] = std::make_pair(TVDOF::Zero(), mask_xyu);
        //     sim.dirichlet_data[rod0.front()] = std::make_pair(TVDOF::Zero(), sim.fix_eulerian);
        //     sim.dirichlet_data[rod1.front()] = std::make_pair(TVDOF::Zero(), sim.fix_eulerian);
        //     sim.dirichlet_data[rod1.back()] = std::make_pair(TVDOF::Zero(), sim.fix_eulerian);
        //     sim.dirichlet_data[rod2.front()] = std::make_pair(TVDOF::Zero(), sim.fix_all);
        //     // sim.dirichlet_data[sub_div / 2 * 2] = std::make_pair(TVDOF::Zero(), sim.fix_eulerian);
        //     // sim.dirichlet_data[rod1.back()] = std::make_pair(TVDOF::Zero(), sim.fix_eulerian);

        //     // TVDOF shift_x = TVDOF::Zero(), mask_x = TVDOF::Zero();
        //     // shift_x[1] = -1.5 * sim.unit;
        //     // mask_x[1] = 1; mask_x[2] = 1; mask_x[3] = 1;
        //     // sim.dirichlet_data[rod1.front()] = std::make_pair(shift_x, mask_x);
        // }
       

        // sim.sliding_nodes.push_back(0);

        // sim.q0 = q;
        // // dof_cnt = sim.n_nodes * sim.dof;
        // sim.n_dof = dof_cnt;
        sim.W = StiffnessMatrix(full_dof_cnt, dof_cnt);
        sim.W.setFromTriplets(w_entry.begin(), w_entry.end());

        sim.rest_states = deformed_states;
        // std::cout << sim.W << std::endl;

        // sim.slide_over_n_rods = IV2(0, std::floor(sub_div * 0.3));

        // T r0 = 0;
        // T r1 = sim.q0(dim + 1, rod1[1]) - sim.q0(dim + 1, rod1[0]);

        // sim.tunnel_u = sim.slide_over_n_rods[0] * r0;
        // sim.tunnel_v = sim.slide_over_n_rods[1] * r1;


        // q0.template segment<dim>(0) = rod2_from; q1.template segment<dim>(0) = rod2_to;
        // q0(dim) = q(dim + 1, rod2.front()); q1(dim) = q(dim + 1, rod2.back());
        // sim.curvature_functions.push_back(new LineCurvature<T, dim>(q0, q1));
        // std::cout << q0.transpose() << std::endl;
        // std::cout << q1.transpose() << std::endl;

        // sim.normal = TV3Stack(3, sim.n_rods);
        // sim.normal.setZero();

        // std::cout << q.transpose() << std::endl;
        // std::cout << rods.transpose() << std::endl;
        

        // for (int i : rod1)
        //     std::cout << i << " ";
        // std::cout << std::endl;
        // std::cout << sim.W.rows() << " " << sim.W.cols() << std::endl;
        // std::cout << sim.W << std::endl;
        // std::exit(0);
    }
}

template<class T, int dim>
void UnitPatch<T, dim>::buildDragCircleScene(int sub_div)
{
    if (sim.run_diff_test)
        sim.unit = 1;
    auto unit_yarn_map = sim.yarn_map;
    sim.yarn_map.clear();
    sim.add_rotation_penalty = false;
    sim.add_pbc_bending = false;
    sim.subdivide = true;

    clearSimData();
    std::vector<Eigen::Triplet<T>> w_entry;

    if constexpr (dim == 2)
    {   
        HybridC2Curve<T, dim>* curve = new HybridC2Curve<T, dim>(sub_div);
        curve->data_points.push_back(TV(0.5, -0.5) * sim.unit);
        curve->data_points.push_back(TV::Zero() * sim.unit);
        curve->data_points.push_back(TV(0.5, 0.5) * sim.unit);
        curve->data_points.push_back(TV(1.0, 0.0) * sim.unit);
        curve->data_points.push_back(TV(0.5, -0.5) * sim.unit);
        std::vector<TV> points_on_curve;
        curve->sampleCurves(points_on_curve);

        sim.n_nodes = points_on_curve.size() - 1;
        int n_points_circle = sim.n_nodes;
        sim.n_rods = points_on_curve.size() - 1;

        q = DOFStack(sim.dof, sim.n_nodes); q.setZero();
        // rods = IV3Stack(3, sim.n_rods); rods.setZero();
        // connections = IV4Stack(4, sim.n_nodes).setOnes() * -1;

        int cnt = 0, dof_cnt = 0;

        std::vector<int> rod0;
        for (int i = 0; i < sim.n_nodes; i++)
        {
            setPos(i, points_on_curve[i], points_on_curve[i]);
            rod0.push_back(i);
            q(dim, i) = T(i) * (curve->data_points.size() - 1 )/ (sim.n_nodes);
        }
        rod0.push_back(0);
        
        TV rod1_from(0.5, -1.0), rod1_to(0.5, 0.6);
        rod1_from *= sim.unit; rod1_to *= sim.unit;

        std::vector<int> rod1, key_points_location_rod1;
        std::vector<TV> points_rod1;

        addStraightYarnCrossNPoints(rod1_from, rod1_to, 
                {curve->data_points[0], curve->data_points[2]}, 
                {0, 2 * sub_div/2}, 
                sub_div, 
                points_rod1, rod1, 
                key_points_location_rod1,
                sim.n_nodes);
        
        sim.n_nodes += points_rod1.size();
        sim.n_rods += rod1.size() - 1;

        q.conservativeResize(sim.dof, sim.n_nodes);
        rods = IV3Stack(3, sim.n_rods); rods.setZero();
        connections = IV4Stack(4, sim.n_nodes).setOnes() * -1;        
        
        for (int i = 0; i < points_rod1.size(); i++)
        {
            int idx = n_points_circle + i;
            setPos(idx, points_rod1[i], points_rod1[i]);
            q(dim + 1, idx) = LDis(rod1.front(), idx) / (rod1_to - rod1_from).norm();
            // std::cout << "idx " << idx << " u " << q.col(idx).transpose() << std::endl;
        }
        
        q(dim + 1, 0) = LDis(rod1.front(), 0) / (rod1_to - rod1_from).norm();
        q(dim + 1, 2 * sub_div / 2) = LDis(rod1.front(), 2 * sub_div / 2) / (rod1_to - rod1_from).norm();

        // std::cout << "idx 0 " <<  q.col(0).transpose() << " idx " << 2 * sub_div / 2 << " " << q(dim + 1, 2 * sub_div / 2) << std::endl;

        // for(int idx : rod1)
        // {
        //     std::cout << "idx " << idx << " " << q.col(idx).transpose() << std::endl;
        // }

        sim.dof_offsets.resize(sim.n_nodes, 0);

        std::vector<int> key_points = { rod0.front(), 
                                        rod1.front(),
                                        rod1.back(), 
                                        2 * sub_div / 2 
                                    };

        addKeyPointsDoF(key_points, w_entry, dof_cnt);
        
        std::vector<int> rod0_kp_on_strand = { 0, 2 * sub_div / 2, n_points_circle};
        std::vector<int> rod0_kp_global = {0, 2 * sub_div / 2, 0};
        std::vector<int> rod0_kp_dof = { 0, 3, 0 };

        markDoFSingleStrand(rod0, rod0_kp_on_strand,
            rod0_kp_global, 
            rod0_kp_dof,
            w_entry, dof_cnt, WARP, true);
        
        // std::cout << "key_points_location_rod1[0] " << key_points_location_rod1[0] << std::endl;
        // std::cout << "key_points_location_rod1[1] " << key_points_location_rod1[1] << std::endl;

        std::vector<int> rod1_kp_on_strand = { 0, key_points_location_rod1[0], key_points_location_rod1[1], int(rod1.size()) - 1 };
        std::vector<int> rod1_kp_global = { n_points_circle, 0, 2 * sub_div / 2, sim.n_nodes - 1};
        std::vector<int> rod1_kp_dof = { 1, 0, 3, 2 };
        
        markDoFSingleStrand(rod1, 
            rod1_kp_on_strand,
            rod1_kp_global,
            rod1_kp_dof,
             w_entry, dof_cnt, WEFT);
        
        
        std::vector<T> data_points_discrete_arc_length;
        
        for(int i = 0; i < curve->data_points.size(); i++)
        {
            data_points_discrete_arc_length.push_back(q(dim, i*sub_div/2));
        }
        
        addRods(rod0, WARP, cnt, 0);
        addRods(rod1, WEFT, cnt, 1);

        sim.pbc_ref_unique.push_back(IV2(sub_div/2, sub_div/2 * 3));
        sim.pbc_ref_unique.push_back(IV2(rod1.front(), rod1.back()));

        sim.pbc_ref.push_back(std::make_pair(WARP, IV2(sub_div/2, sub_div/2 * 3)));
        sim.pbc_ref.push_back(std::make_pair(WEFT, IV2(rod1.front(), rod1.back())));

        sim.is_end_nodes = std::vector<bool>(sim.n_nodes, false);

        if(true)
        {
            // sim.dirichlet_data[rod0.back()] = std::make_pair(TVDOF::Zero(), sim.fix_all);
            TVDOF mask_xyu = TVDOF::Ones(); mask_xyu[3] = 0.0;

            sim.dirichlet_data[rod0.front()] = std::make_pair(TVDOF::Zero(), mask_xyu);
            
            sim.dirichlet_data[sub_div / 2 * 2] = std::make_pair(TVDOF::Zero(), sim.fix_eulerian);
            sim.dirichlet_data[rod1.back()] = std::make_pair(TVDOF::Zero(), sim.fix_eulerian);

            TVDOF shift_x = TVDOF::Zero(), mask_x = TVDOF::Zero();
            shift_x[1] = -1.5 * sim.unit;
            mask_x[1] = 1; mask_x[2] = 1; mask_x[3] = 1;
            sim.dirichlet_data[rod1.front()] = std::make_pair(shift_x, mask_x);
        }
       

        sim.sliding_nodes.push_back(0);

        sim.q0 = q;
        // dof_cnt = sim.n_nodes * sim.dof;
        sim.n_dof = dof_cnt;
        sim.W = StiffnessMatrix(sim.n_nodes * sim.dof, sim.n_dof);
        sim.W.setFromTriplets(w_entry.begin(), w_entry.end());


        sim.slide_over_n_rods = IV2(0, std::floor(sub_div * 0.3));

        T r0 = 0;
        T r1 = sim.q0(dim + 1, rod1[1]) - sim.q0(dim + 1, rod1[0]);

        sim.tunnel_u = sim.slide_over_n_rods[0] * r0;
        sim.tunnel_v = sim.slide_over_n_rods[1] * r1;


        Vector<T, dim + 1> q0 = q.col(0).template segment<dim + 1>(0);
        Vector<T, dim + 1> q1 = q.col(rod0.back()).template segment<dim + 1>(0);
        
        DiscreteHybridCurvature<T, dim>* curve_func = new DiscreteHybridCurvature<T, dim>(
            q0, q1);
        sim.curvature_functions.push_back(curve_func);
        curve_func->setData(curve, data_points_discrete_arc_length);

        q0.template segment<dim>(0) = rod1_from; q1.template segment<dim>(0) = rod1_to;
        q0(dim) = q(dim + 1, rod1.front()); q1(dim) = q(dim + 1, rod1.back());
        sim.curvature_functions.push_back(new LineCurvature<T, dim>(q0, q1));
        // std::cout << q0.transpose() << std::endl;
        // std::cout << q1.transpose() << std::endl;

        sim.normal = TV3Stack(3, sim.n_rods);
        sim.normal.setZero();

        // std::cout << q.transpose() << std::endl;
        // std::cout << rods.transpose() << std::endl;

        // for (int i : rod1)
        //     std::cout << i << " ";
        // std::cout << std::endl;
        // std::cout << sim.W.rows() << " " << sim.W.cols() << std::endl;
        // std::cout << sim.W << std::endl;
    }
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
    auto unit_yarn_map = sim.yarn_map;
    sim.yarn_map.clear();
    sim.add_rotation_penalty = false;
    sim.add_pbc_bending = false;
    sim.subdivide = true;

    clearSimData();
    std::vector<Eigen::Triplet<T>> w_entry;

    if constexpr (dim == 2)
    {   
        HybridC2Curve<T, dim>* curve = new HybridC2Curve<T, dim>(sub_div);
        curve->data_points.push_back(TV::Zero() * sim.unit);
        curve->data_points.push_back(TV(0.5, 0.5) * sim.unit);
        curve->data_points.push_back(TV(1.0, 0.0) * sim.unit);
        std::vector<TV> points_on_curve;
        curve->sampleCurves(points_on_curve);

        sim.n_nodes = points_on_curve.size();
        sim.n_rods = points_on_curve.size() - 1;

        q = DOFStack(sim.dof, sim.n_nodes); q.setZero();
        // rods = IV3Stack(3, sim.n_rods); rods.setZero();
        // connections = IV4Stack(4, sim.n_nodes).setOnes() * -1;

        int cnt = 0, dof_cnt = 0;

        std::vector<int> rod0;
        for (int i = 0; i < sim.n_nodes; i++)
        {
            setPos(i, points_on_curve[i], points_on_curve[i]);
            rod0.push_back(i);
            q(dim, i) = T(i) * (curve->data_points.size() - 1 )/ (sim.n_nodes - 1);
        }
        
        TV rod1_from(0.5, 0), rod1_to(0.5, 1.0);
        rod1_from *= sim.unit; rod1_to *= sim.unit;

        std::vector<int> rod1, key_points_location_rod1;
        std::vector<TV> points_rod1;

        addStraightYarnCrossNPoints(rod1_from, rod1_to, 
                {curve->data_points[1]}, 
                {sub_div/2}, 
                sub_div, 
                points_rod1, rod1, 
                key_points_location_rod1,
                sim.n_nodes);
        
        sim.n_nodes += points_rod1.size();
        sim.n_rods += rod1.size() - 1;

        q.conservativeResize(sim.dof, sim.n_nodes);
        rods = IV3Stack(3, sim.n_rods); rods.setZero();
        connections = IV4Stack(4, sim.n_nodes).setOnes() * -1;        
        
        for (int i = 0; i < points_rod1.size(); i++)
        {
            int idx = points_on_curve.size() + i;
            setPos(idx, points_rod1[i], points_rod1[i]);
            q(dim + 1, idx) = LDis(rod1.front(), idx) / (rod1_to - rod1_from).norm();
        }
        
        q(dim + 1, sub_div / 2) = LDis(rod1.front(), sub_div / 2) / (rod1_to - rod1_from).norm();

        sim.dof_offsets.resize(sim.n_nodes, 0);

        std::vector<int> key_points = { rod0.front(), 
                                        rod0.back(), 
                                        rod1.front(),
                                        rod1.back(), 
                                        sub_div / 2 
                                    };

        addKeyPointsDoF(key_points, w_entry, dof_cnt);
        
        std::vector<int> rod0_kp_on_strand = { 0, sub_div / 2, int(points_on_curve.size()) - 1 };
        std::vector<int> rod0_kp_global = {0, sub_div / 2, int(points_on_curve.size()) - 1};
        std::vector<int> rod0_kp_dof = { 0, 4, 1 };
        markDoFSingleStrand(rod0, rod0_kp_on_strand,
            rod0_kp_global, 
            rod0_kp_dof,
            w_entry, dof_cnt, WARP);
        
        // std::cout << "key_points_location_rod1[0] " << key_points_location_rod1[0] << std::endl;


        std::vector<int> rod1_kp_on_strand = { 0, key_points_location_rod1[0], int(rod1.size()) - 1 };
        std::vector<int> rod1_kp_global = { int(points_on_curve.size()), sub_div / 2, sim.n_nodes - 1};
        std::vector<int> rod1_kp_dof = { 2, 4, 3 };
        markDoFSingleStrand(rod1, 
            rod1_kp_on_strand,
            rod1_kp_global,
            rod1_kp_dof,
             w_entry, dof_cnt, WEFT);
        
        
        std::vector<T> data_points_discrete_arc_length;
        
        for(int i = 0; i < curve->data_points.size(); i++)
        {
            data_points_discrete_arc_length.push_back(q(dim, i*sub_div/2));
        }
        
        addRods(rod0, WARP, cnt, 0);
        addRods(rod1, WEFT, cnt, 1);

        sim.pbc_ref_unique.push_back(IV2(0, rod0.back()));
        sim.pbc_ref_unique.push_back(IV2(rod1.front(), rod1.back()));

        sim.pbc_ref.push_back(std::make_pair(WARP, IV2(0, rod0.back())));
        sim.pbc_ref.push_back(std::make_pair(WEFT, IV2(rod1.front(), rod1.back())));

        sim.is_end_nodes = std::vector<bool>(sim.n_nodes, false);

        if(true)
        {
            sim.dirichlet_data[rod0.back()] = std::make_pair(TVDOF::Zero(), sim.fix_all);
            sim.dirichlet_data[rod0.front()] = std::make_pair(TVDOF::Zero(), sim.fix_all);

            sim.dirichlet_data[rod1.front()] = std::make_pair(TVDOF::Zero(), sim.fix_all);
            TVDOF shift_x = TVDOF::Zero(), mask_x = TVDOF::Zero();
            shift_x[0] = 0.8 * sim.unit;
            mask_x[0] = 1; mask_x[2] = 1; mask_x[3] = 1;
            sim.dirichlet_data[rod1.back()] = std::make_pair(shift_x, mask_x);
        }
        if(false)
        {
            sim.dirichlet_data[rod0.back()] = std::make_pair(TVDOF::Zero(), sim.fix_all);
            sim.dirichlet_data[rod0.front()] = std::make_pair(TVDOF::Zero(), sim.fix_all);

            sim.dirichlet_data[rod1.back()] = std::make_pair(TVDOF::Zero(), sim.fix_eulerian);
            TVDOF shift_x = TVDOF::Zero(), mask_x = TVDOF::Zero();

            T s = 0.15 * std::sqrt(2);

            shift_x[0] = s * sim.unit;
            // shift_x[1] = -(0.4 + 0.5 - s) * sim.unit;
            shift_x[1] = -0.45 * sim.unit;
            mask_x[0] = 1; mask_x[1] = 1; mask_x[2] = 1; mask_x[3] = 1;
            sim.dirichlet_data[rod1.front()] = std::make_pair(shift_x, mask_x);
        }

        sim.sliding_nodes.push_back(sub_div / 2);

        sim.q0 = q;
        // dof_cnt = sim.n_nodes * sim.dof;
        sim.n_dof = dof_cnt;
        sim.W = StiffnessMatrix(sim.n_nodes * sim.dof, sim.n_dof);
        sim.W.setFromTriplets(w_entry.begin(), w_entry.end());


        sim.slide_over_n_rods = IV2(std::floor(sub_div * 0.4), std::floor(sub_div * 0.4));

        T r0 = sim.q0(dim, rod0[1]) - sim.q0(dim, rod0[0]);
        T r1 = sim.q0(dim + 1, rod1[1]) - sim.q0(dim + 1, rod1[0]);

        sim.tunnel_u = sim.slide_over_n_rods[0] * r0;
        sim.tunnel_v = sim.slide_over_n_rods[1] * r1;


        Vector<T, dim + 1> q0 = q.col(0).template segment<dim + 1>(0);
        Vector<T, dim + 1> q1 = q.col(rod0.back()).template segment<dim + 1>(0);
        
        DiscreteHybridCurvature<T, dim>* curve_func = new DiscreteHybridCurvature<T, dim>(
            q0, q1);
        sim.curvature_functions.push_back(curve_func);
        curve_func->setData(curve, data_points_discrete_arc_length);

        q0.template segment<dim>(0) = rod1_from; q1.template segment<dim>(0) = rod1_to;
        q0(dim) = q(dim + 1, rod1.front()); q1(dim) = q(dim + 1, rod1.back());
        sim.curvature_functions.push_back(new LineCurvature<T, dim>(q0, q1));
        // std::cout << q0.transpose() << std::endl;
        // std::cout << q1.transpose() << std::endl;

        sim.normal = TV3Stack(3, sim.n_rods);
        sim.normal.setZero();

        // std::cout << q.transpose() << std::endl;
        // std::cout << rods.transpose() << std::endl;
        // std::cout << sim.W << std::endl;
        // std::cout << "build scene done" << std::endl;
    }

}

template<class T, int dim>
void UnitPatch<T, dim>::buildTwoRodsScene(int sub_div)
{
    auto unit_yarn_map = sim.yarn_map;
    sim.yarn_map.clear();
    sim.add_rotation_penalty = false;
    sim.add_pbc_bending = false;
    sim.subdivide = true;

    clearSimData();
    std::vector<Eigen::Triplet<T>> w_entry;

    if constexpr (dim == 2)
    {   
        
        sim.n_nodes = 0;

        // q = DOFStack(sim.dof, sim.n_nodes); q.setZero();

        int cnt = 0, dof_cnt = 0;

        TV rod0_from(0.25, 1.0), rod0_to(1.0, 0.0);
        rod0_from *= sim.unit; rod0_to *= sim.unit;

        std::vector<int> rod0, key_points_location_rod0;
        std::vector<TV> points_rod0;

        addStraightYarnCrossNPoints(rod0_from, rod0_to, 
                {}, 
                {}, 
                sub_div, 
                points_rod0, rod0, 
                key_points_location_rod0,
                sim.n_nodes);
        
        sim.n_nodes += points_rod0.size();
        sim.n_rods += rod0.size() - 1;
        
        TV rod1_from(points_rod0[sub_div/2][0], 0), rod1_to(points_rod0[sub_div/2][0], sim.unit);

        std::vector<int> rod1, key_points_location_rod1;
        std::vector<TV> points_rod1;

        addStraightYarnCrossNPoints(rod1_from, rod1_to, 
                {points_rod0[sub_div/2]}, 
                {sub_div/2}, 
                sub_div, 
                points_rod1, rod1, 
                key_points_location_rod1,
                sim.n_nodes);
        
        sim.n_nodes += points_rod1.size();
        sim.n_rods += rod1.size() - 1;

        q = DOFStack(sim.dof, sim.n_nodes); q.setZero();
        rods = IV3Stack(3, sim.n_rods); rods.setZero();
        connections = IV4Stack(4, sim.n_nodes).setOnes() * -1;        

        for (int i = 0; i < points_rod0.size(); i++)
        {
            int idx = i;
            setPos(idx, points_rod0[i], points_rod0[i]);
            q(dim, idx) = LDis(rod0.front(), idx) / (rod0_to - rod0_from).norm();
        }
        
        for (int i = 0; i < points_rod1.size(); i++)
        {
            int idx = points_rod0.size() + i;
            setPos(idx, points_rod1[i], points_rod1[i]);
            q(dim + 1, idx) = LDis(rod1.front(), idx) / (rod1_to - rod1_from).norm();
        }
        
        q(dim + 1, sub_div / 2) = LDis(rod1.front(), sub_div / 2) / (rod1_to - rod1_from).norm();

        sim.dof_offsets.resize(sim.n_nodes, 0);

        std::vector<int> key_points = { rod0.front(), 
                                        rod0.back(), 
                                        rod1.front(),
                                        rod1.back(), 
                                        sub_div / 2 
                                    };

        addKeyPointsDoF(key_points, w_entry, dof_cnt);
        
        std::vector<int> rod0_kp_on_strand = { 0, sub_div / 2, int(points_rod0.size()) - 1 };
        std::vector<int> rod0_kp_global = {0, sub_div / 2, int(points_rod0.size()) - 1};
        std::vector<int> rod0_kp_dof = { 0, 4, 1 };
        markDoFSingleStrand(rod0, rod0_kp_on_strand,
            rod0_kp_global, 
            rod0_kp_dof,
            w_entry, dof_cnt, WARP);
        
        // std::cout << "key_points_location_rod1[0] " << key_points_location_rod1[0] << std::endl;


        std::vector<int> rod1_kp_on_strand = { 0, key_points_location_rod1[0], int(rod1.size()) - 1 };
        std::vector<int> rod1_kp_global = { int(points_rod0.size()), sub_div / 2, sim.n_nodes - 1};
        std::vector<int> rod1_kp_dof = { 2, 4, 3 };
        markDoFSingleStrand(rod1, 
            rod1_kp_on_strand,
            rod1_kp_global,
            rod1_kp_dof,
             w_entry, dof_cnt, WEFT);
        
        
        addRods(rod0, WARP, cnt, 0);
        addRods(rod1, WEFT, cnt, 1);

        sim.pbc_ref_unique.push_back(IV2(rod0.front(), rod0.back()));
        sim.pbc_ref_unique.push_back(IV2(rod1.front(), rod1.back()));

        sim.pbc_ref.push_back(std::make_pair(WARP, IV2(rod0.front(), rod0.back())));
        sim.pbc_ref.push_back(std::make_pair(WEFT, IV2(rod1.front(), rod1.back())));

        sim.is_end_nodes = std::vector<bool>(sim.n_nodes, false);

        sim.dirichlet_data[rod0.back()] = std::make_pair(TVDOF::Zero(), sim.fix_all);
        sim.dirichlet_data[rod0.front()] = std::make_pair(TVDOF::Zero(), sim.fix_all);

        sim.dirichlet_data[rod1.front()] = std::make_pair(TVDOF::Zero(), sim.fix_all);
        // sim.dirichlet_data[rod1.back()] = std::make_pair(TVDOF::Zero(), sim.fix_eulerian);
        // sim.dirichlet_data[sub_div / 2] = std::make_pair(TVDOF::Zero(), sim.fix_eulerian);

        TVDOF shift_x = TVDOF::Zero(), mask_x = TVDOF::Zero();
        shift_x[0] = 0.5 * sim.unit;
        mask_x[0] = 1; mask_x[2] = 1; mask_x[3] = 1;
        // sim.dirichlet_data[rod1.front()] = std::make_pair(shift_x, mask_x);
        sim.dirichlet_data[rod1.back()] = std::make_pair(shift_x, mask_x);

        sim.sliding_nodes.push_back(sub_div / 2);

        sim.q0 = q;
        // dof_cnt = sim.n_nodes * sim.dof;
        sim.n_dof = dof_cnt;
        sim.W = StiffnessMatrix(sim.n_nodes * sim.dof, sim.n_dof);
        sim.W.setFromTriplets(w_entry.begin(), w_entry.end());


        sim.slide_over_n_rods = IV2(std::floor(sub_div * 0.4), std::floor(sub_div * 0.4));

        T r0 = sim.q0(dim, rod0[1]) - sim.q0(dim, rod0[0]);
        T r1 = sim.q0(dim + 1, rod1[1]) - sim.q0(dim + 1, rod1[0]);

        sim.tunnel_u = sim.slide_over_n_rods[0] * r0;
        sim.tunnel_v = sim.slide_over_n_rods[1] * r1;

        Vector<T, dim + 1> q0 = q.col(rod0.front()).template segment<dim + 1>(0);
        Vector<T, dim + 1> q1 = q.col(rod0.back()).template segment<dim + 1>(0);
        sim.curvature_functions.push_back(new LineCurvature<T, dim>(q0, q1));
        
        q0.template segment<dim>(0) = rod1_from; q1.template segment<dim>(0) = rod1_to;
        q0(dim) = q(dim + 1, rod1.front()); q1(dim) = q(dim + 1, rod1.back());
        sim.curvature_functions.push_back(new LineCurvature<T, dim>(q0, q1));
        
        sim.normal = TV3Stack(3, sim.n_rods);
        sim.normal.setZero();

        // std::cout << q.transpose() << std::endl;
        // std::cout << rods.transpose() << std::endl;
        // std::cout << sim.W << std::endl;
    }
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

    sim.subdivide = true;
    
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
                for (int i = 0; i < sim.n_nodes; i++)
                    if(sim.dirichlet_data.find(i) == sim.dirichlet_data.end())
                        sim.sliding_nodes.push_back(i);
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
            q *= 0.03;
            sim.q0 = q;
            sim.n_dof = dof_cnt;
            sim.W = StiffnessMatrix(sim.n_nodes * sim.dof, sim.n_dof);
            sim.W.setFromTriplets(w_entry.begin(), w_entry.end());
        }

        sim.dof_offsets.resize(sim.n_nodes, 0);
        for(int i = 0; i < 17; i++)
            sim.dof_offsets[i] = i * sim.dof;
        
        sim.slide_over_n_rods = IV2(std::floor(sub_div * 0.2), std::floor(sub_div * 0.2));
        T rod_length = (sim.q0.col(rods.col(0)(0)).template segment<dim>(0) - 
            sim.q0.col(rods.col(0)(1)).template segment<dim>(0)).norm();
        sim.tunnel_u = sim.slide_over_n_rods[0] * rod_length;
        sim.tunnel_v = sim.tunnel_u;


        sim.curvature_functions.push_back(new CircleCurvature<T, dim>(r * 0.03));
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