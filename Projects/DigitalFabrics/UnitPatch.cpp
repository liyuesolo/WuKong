#include <iostream>
#include <utility>
#include <fstream>
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
        buildTwoRodsScene(2);
    else if (patch_type == 5)
        buildSlidingTestScene(8);
    else if (patch_type == 6)
        buildStraightYarn3x1(8);
    else if (patch_type == 7)
        buildZigZagScene(64);
    else if (patch_type == 8)
        buildUnitFromC2Curves(64);
}

// assuming passing points sorted long from to to direction
template<class T, int dim>
void UnitPatch<T, dim>::addStraightYarnCrossNPoints(const TV& from, const TV& to,
    const std::vector<TV>& passing_points, int sub_div,
    std::vector<TV>& sub_points, std::vector<int>& node_idx, int start, bool pbc)
{
    node_idx.push_back(start);
    sub_points.push_back(from);
    T length_yarn = (to - from).norm();
    TV length_vec = (to - from).normalized();
    for (int i = 0; i < passing_points.size(); i++)
    {
        T fraction = (passing_points[i] - from).norm() / length_yarn;
        int n_sub_nodes = std::ceil(fraction * sub_div);
        T length_sub = (passing_points[i] - from).norm() / T(n_sub_nodes + 1);
        for (int j = 0; j < n_sub_nodes; j++)
            sub_points.push_back(sub_points.back() + length_sub * length_vec);
        // node_idx
        sub_points.push_back(passing_points[i]);
    }
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

        // curve->derivativeTestdF();

        
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
        
        sim.dof_offsets.resize(sim.n_nodes, 0);
        // rod0 0, points_on_curve.size()

        for (int i = 0; i < sim.n_nodes; i++)
        {
            sim.dof_offsets[i] = dof_cnt;
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
        sim.curvature_functions.push_back(curve_func);
        curve_func->setData(curve, data_points_discrete_arc_length);

        // std::cout << q.transpose() << std::endl;
    }

    // sim.checkMaterialPositionDerivatives();
}



// template<class T, int dim>
// void UnitPatch<T, dim>::buildUnitFromC2Curves(int sub_div)
// {
//     auto unit_yarn_map = sim.yarn_map;
//     sim.yarn_map.clear();
//     sim.add_rotation_penalty = false;
//     sim.add_pbc_bending = false;
//     sim.subdivide = true;

//     clearSimData();
//     std::vector<Eigen::Triplet<T>> w_entry;

//     if constexpr (dim == 2)
//     {
//         std::string data_points_file = "/home/yueli/Documents/ETH/WuKong/Projects/DigitalFabrics/Data/test.curve";
//         T x, y;

//         std::ifstream in(data_points_file);

//         HybridC2Curve<T, dim>* curve = new HybridC2Curve<T, dim>(sub_div);
//         while(in >> x >> y)
//         {
//             curve->data_points.push_back(TV(x, y) * 0.03);
//         }
//         in.close();
//         curve->normalizeDataPoints();
//         std::vector<TV> points_on_curve;
//         // curve->getLinearSegments(points_on_curve);
//         curve->sampleCurves(points_on_curve);

//         sim.n_nodes = points_on_curve.size();
//         sim.n_rods = points_on_curve.size() - 1;

//         q = DOFStack(sim.dof, sim.n_nodes); q.setZero();
//         rods = IV3Stack(3, sim.n_rods); rods.setZero();
//         connections = IV4Stack(4, sim.n_nodes).setOnes() * -1;

//         sim.normal = TV3Stack(3, sim.n_rods);
//         sim.normal.setZero();
        
//         int cnt = 0, dof_cnt = 0;
//         T arc_length_sum = 0;
        
//         std::vector<int> rod0;
//         for (int i = 0; i < sim.n_nodes; i++)
//         {
//             setPos(i, points_on_curve[i], points_on_curve[i]);
//             rod0.push_back(i);
//             if (i == 0) 
//             {
//                 q(dim, 0) = 0.0;
//                 continue;
//             }
//             arc_length_sum += LDis(i-1, i);
//             q(dim, i) = q(dim, i-1) + LDis(i-1, i);
//         }
        
//         q(dim, sim.n_nodes - 1) = arc_length_sum;

//         sim.dof_offsets.resize(sim.n_nodes, 0);
//         // rod0 0, points_on_curve.size()

//         for (int i = 0; i < sim.n_nodes; i++)
//         {
//             sim.dof_offsets[i] = dof_cnt;
//             for(int d = 0; d < dim; d++)
//             {
//                 w_entry.push_back(Eigen::Triplet<T>(i * sim.dof + d, dof_cnt, 1.0));
//                 dof_cnt++;
//             }
//             if (i == 0 || i == sim.n_nodes - 1)
//                 for(int d = dim; d < sim.dof; d++)
//                 {
//                     w_entry.push_back(Eigen::Triplet<T>(i * sim.dof + d, dof_cnt, 1.0));
//                     dof_cnt++;
//                 }
//             else
//             {    
//                 int idx_last = sim.dof + (points_on_curve.size() - 2) * dim;
                
//                 T alpha = q(dim, i) / arc_length_sum;
//                 for(int d = dim; d < sim.dof; d++)
//                 {
//                     w_entry.push_back(Eigen::Triplet<T>(i * sim.dof + d, d, 1 - alpha));
//                     w_entry.push_back(Eigen::Triplet<T>(i * sim.dof + d, idx_last + d, alpha));
//                 }
//             }

//         }
        
//         // q *= 0.03;
        
//         std::vector<T> data_points_discrete_arc_length;
        
//         for(int i = 0; i < curve->data_points.size(); i++)
//         {
//             data_points_discrete_arc_length.push_back(q(dim, i*sub_div/2));
//         }
        
//         addRods(rod0, WARP, cnt, 0);

//         sim.pbc_ref_unique.push_back(IV2(0, rod0.back()));
//         // sim.pbc_ref_unique.push_back(IV2(WEFT, rod0[rod0.size()-1]));

//         sim.pbc_ref.push_back(std::make_pair(WARP, IV2(0, rod0.back())));
//         // sim.pbc_ref.push_back(std::make_pair(WEFT, IV2(0, rod0[rod0.size()-1])));

//         sim.is_end_nodes = std::vector<bool>(sim.n_nodes, false);

//         // if (sim.disable_sliding)
//         {
//             // for(int i = 0; i < sim.n_nodes; i++)
//                 // sim.dirichlet_data[i] = std::make_pair(TVDOF::Zero(), sim.fix_eulerian);
//             sim.dirichlet_data[points_on_curve.size() - 1] = std::make_pair(TVDOF::Zero(), sim.fix_eulerian);
//             sim.dirichlet_data[0] = std::make_pair(TVDOF::Zero(), sim.fix_all);
//         }
//         sim.q0 = q;
//         // dof_cnt = sim.n_nodes * sim.dof;
//         sim.n_dof = dof_cnt;
//         sim.W = StiffnessMatrix(sim.n_nodes * sim.dof, sim.n_dof);
//         sim.W.setFromTriplets(w_entry.begin(), w_entry.end());
//         // sim.W.setIdentity();
//         // std::cout << sim.W << std::endl;
//         // for (int i = 0; i < sim.n_nodes; i++)
//         //     if(sim.dirichlet_data.find(i) == sim.dirichlet_data.end())
//         //         sim.sliding_nodes.push_back(i);

//         sim.slide_over_n_rods = IV2(std::floor(sub_div * 0.2), std::floor(sub_div * 0.2));
//         T rod_length = (sim.q0.col(rods.col(0)(0)).template segment<dim>(0) - 
//             sim.q0.col(rods.col(0)(1)).template segment<dim>(0)).norm();
//         sim.tunnel_u = sim.slide_over_n_rods[0] * rod_length;
//         sim.tunnel_v = sim.tunnel_u;

//         // sim.curvature_functions.push_back(new LineCurvature<T, dim>());
//         Vector<T, dim + 1> q0 = q.col(0).template segment<dim + 1>(0);
//         Vector<T, dim + 1> q1 = q.col(rod0.back()).template segment<dim + 1>(0);
        
//         DiscreteHybridCurvature<T, dim>* curve_func = new DiscreteHybridCurvature<T, dim>(
//             // curve, 
//             // data_points_discrete_arc_length, 
//             q0, q1);
//         sim.curvature_functions.push_back(curve_func);
//         curve_func->setData(curve, data_points_discrete_arc_length);

//         std::cout << q.transpose() << std::endl;
//     }
// }

// template<class T, int dim>
// void UnitPatch<T, dim>::buildUnitFromC2Curves(int sub_div)
// {
//     auto unit_yarn_map = sim.yarn_map;
//     sim.yarn_map.clear();
//     sim.add_rotation_penalty = false;
//     sim.subdivide = true;
//     sim.add_pbc_bending = false;

//     clearSimData();

//     if constexpr (dim == 2)
//     {
//         std::string data_points_file = "/home/yueli/Documents/ETH/WuKong/Projects/DigitalFabrics/Data/test.curve";
//         T x, y;
        
//         std::ifstream in(data_points_file);
        
//         HybridC2Curve<T, dim> curve(sub_div);
//         while(in >> x >> y)
//         {
//             curve.data_points.push_back(TV(x, y));
//         }
//         in.close();
//         curve.normalizeDataPoints();
//         std::vector<TV> points_on_curve;
//         curve.getLinearSegments(points_on_curve);

//         sim.n_nodes = points_on_curve.size();
//         sim.n_rods = points_on_curve.size() - 1;

//         q = DOFStack(sim.dof, sim.n_nodes); q.setZero();
//         rods = IV3Stack(3, sim.n_rods); rods.setZero();
//         connections = IV4Stack(4, sim.n_nodes).setOnes() * -1;

//         sim.normal = TV3Stack(3, sim.n_rods);
//         sim.normal.setZero();
       

//         // set ending points and sliding nodes first
//         // std::vector<int> already_pushed_nodes;

//         int node_cnt = 0;
//         // already_pushed_nodes.push_back(node_cnt);
//         setPos(node_cnt++, points_on_curve.front(), points_on_curve.front());
//         q(dim, node_cnt-1) = 0.0;
//         fixEulerian(node_cnt-1);
        
//         // already_pushed_nodes.push_back(node_cnt);
//         setPos(node_cnt++, points_on_curve.back(), points_on_curve.back());
//         fixEulerian(node_cnt-1);

//         // TV rod1_start(curve.data_points[1][0], 0.2);
//         // TV rod1_end(curve.data_points[1][0], 0.8);

//         // already_pushed_nodes.push_back(node_cnt);
//         // setPos(node_cnt++, rod1_start, rod1_start);
//         // fixEulerian(node_cnt-1);

//         // already_pushed_nodes.push_back(node_cnt);
//         // setPos(node_cnt++, rod1_start, rod1_start);
//         // fixEulerian(node_cnt-1);

//         // already_pushed_nodes.push_back(node_cnt);
//         TV crossing = curve.data_points[1];
//         setPos(node_cnt++, crossing, crossing);
//         std::unordered_map<int, int> existing_nodes_rod0;
//         existing_nodes_rod0[0] = 0;
//         existing_nodes_rod0[points_on_curve.size() - 1] = 1;
//         existing_nodes_rod0[sub_div] = 2;

//         // build yarns noting that the points above are already pushed.
//         int rod_cnt = 0;

//         T arc_length_sum = 0;
//         std::vector<int> rod0;
        
//         for (int i = 0; i < points_on_curve.size(); i++)
//         {
//             //haven't pushed yet
//             bool added = existing_nodes_rod0.find(i) != existing_nodes_rod0.end();
//             if(added) 
//             {
//                 rod0.push_back(existing_nodes_rod0[i]);
//             }
//             else
//             {
//                 setPos(node_cnt, points_on_curve[i], points_on_curve[i]);
//                 rod0.push_back(node_cnt);
//             }
//             if (i == 0)
//             {
//                 // node_cnt++;
//                 continue;
//             }
//             arc_length_sum += LDis(node_cnt-1, node_cnt);
//             if(!added) 
//             {
//                 q(dim, node_cnt) = q(dim, node_cnt-1) + LDis(node_cnt-1, node_cnt);
//                 node_cnt++;
//             }
//         }

//         addRods(rod0, WARP, rod_cnt, 0);

//         // std::vector<TV> nodes0;
    
//         // addStraightYarnCrossNPoints(rod1_start, 
//         //                             rod1_end,
//         //                             {curve.data_points[1]},
//         //                             {4},
//         //                             sub_div, nodes0, false);

//         // q.conservativeResize(q.rows() + nodes0.size());

//         // std::vector<int> rod1;
//         // for (int i = 0; i < nodes0.size(); i++) rod1.push_back(cnt + i);
        
//         // addRods(rod1, WEFT, cnt, 1);
        
//         sim.pbc_ref_unique.push_back(IV2(0, rod0.back()));
//         // sim.pbc_ref_unique.push_back(IV2(rod1.front(), rod1.back()));

//         sim.pbc_ref.push_back(std::make_pair(WARP, IV2(0, rod0.back())));
//         // sim.pbc_ref.push_back(std::make_pair(WEFT, IV2(rod1.front(), rod1.back())));

//         sim.is_end_nodes = std::vector<bool>(sim.n_nodes, false);

//         // if (sim.disable_sliding)
//         {
//             for(int i = 0; i < sim.n_nodes; i++)
//                 sim.dirichlet_data[i] = std::make_pair(TVDOF::Zero(), sim.fix_eulerian);
//             sim.dirichlet_data[0] = std::make_pair(TVDOF::Zero(), sim.fix_all);
//         }

//         q *= 0.03;
//         sim.q0 = q;
//         int dof_cnt = sim.n_nodes * sim.dof;
//         sim.n_dof = dof_cnt;
//         sim.W = StiffnessMatrix(sim.n_nodes * sim.dof, sim.n_dof);
//         // sim.W.setFromTriplets(w_entry.begin(), w_entry.end());
//         sim.W.setIdentity();

//         // for (int i = 0; i < sim.n_nodes; i++)
//         //     if(sim.dirichlet_data.find(i) == sim.dirichlet_data.end())
//         //         sim.sliding_nodes.push_back(i);

//         sim.slide_over_n_rods = IV2(std::floor(sub_div * 0.2), std::floor(sub_div * 0.2));
//         T rod_length = (sim.q0.col(rods.col(0)(0)).template segment<dim>(0) - 
//             sim.q0.col(rods.col(0)(1)).template segment<dim>(0)).norm();
//         sim.tunnel_u = sim.slide_over_n_rods[0] * rod_length;
//         sim.tunnel_v = sim.tunnel_u;

//         sim.curvature_functions.push_back(new LineCurvature<T, dim>());
//     }
//     // std::cout << q.transpose() << std::endl;
// }

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
void UnitPatch<T, dim>::buildZigZagScene(int sub_div)
{
    clearSimData();
    sim.n_nodes = 12;
    sim.n_rods = 12;

    q = DOFStack(sim.dof, sim.n_nodes); q.setZero();
    rods = IV3Stack(3, sim.n_rods); rods.setZero();
    connections = IV4Stack(4, sim.n_nodes).setOnes() * -1;

    sim.normal = TV3Stack(3, sim.n_rods);
    sim.normal.setZero();
    sim.subdivide = true;

    sim.add_pbc_bending = false;

    if constexpr (dim == 2)
    {
        {
            q.col(0).template segment<dim>(0) = TV2(0.125, 0.15);
            q.col(1).template segment<dim>(0) = TV2(0.875, 0.15);
            q.col(2).template segment<dim>(0) = TV2(0.125, 0.05);
            q.col(3).template segment<dim>(0) = TV2(0.875, 0.05);
            q.block(dim, 0, 2, 4) = q.block(0, 0, 2, 4);

            q.col(4).template segment<dim>(0) = TV2(0.3, 0);
            q.col(5).template segment<dim>(0) = TV2(0.3, 0.2);
            q.col(6).template segment<dim>(0) = TV2(0.7, 0);
            q.col(7).template segment<dim>(0) = TV2(0.7, 0.2);
            q.col(8).template segment<dim>(0) = TV2(0.35, 0.15);
            q.col(9).template segment<dim>(0) = TV2(0.75, 0.15);
            q.col(10).template segment<dim>(0) = TV2(0.25, 0.05);
            q.col(11).template segment<dim>(0) = TV2(0.65, 0.05);

            q.col(4).template segment<2>(dim) = TV2(0.3, 0);
            q.col(10).template segment<2>(dim) = TV2(0.25, q(dim + 1, 4) + LDis(4, 10));
            q.col(8).template segment<2>(dim) = TV2(0.35, q(dim + 1, 10) + LDis(10, 8));
            q.col(5).template segment<2>(dim) = TV2(0.3, q(dim + 1, 8) + LDis(8, 5));
            q.col(6).template segment<2>(dim) = TV2(0.7, 0);
            q.col(11).template segment<2>(dim) = TV2(0.65, q(dim + 1, 6) + LDis(6, 11));
            q.col(9).template segment<2>(dim) = TV2(0.75, q(dim + 1, 11) + LDis(11, 9));
            q.col(7).template segment<2>(dim) = TV2(0.7, q(dim + 1, 9) + LDis(9, 7));

            int cnt = 0;
            std::vector<int> rod0 = {0, 8, 9, 1}, 
                             rod1 = {2, 10 ,11, 3},
                             rod2 = {4, 10, 8, 5},
                             rod3 = {6, 11, 9, 7};
            
            addRods(rod0, WARP, cnt, 0);
            addRods(rod1, WARP, cnt, 1);
            addRods(rod2, WEFT, cnt, 2);
            addRods(rod3, WEFT, cnt, 3);
            
            sim.q0 = q;
            sim.n_dof = sim.n_nodes * sim.dof;

            
            sim.pbc_ref.push_back(std::make_pair(WEFT, IV2(4, 5)));
            sim.pbc_ref.push_back(std::make_pair(WEFT, IV2(6, 7)));

            sim.pbc_ref.push_back(std::make_pair(WARP, IV2(0, 1)));
            sim.pbc_ref.push_back(std::make_pair(WARP, IV2(2, 3)));

            sim.pbc_ref_unique.push_back(IV2(0, 1));
            sim.pbc_ref_unique.push_back(IV2(4, 5));

            if (sim.disable_sliding)
            {
                for(int i = 0; i < sim.n_nodes; i++)
                    sim.dirichlet_data[i] = std::make_pair(TVDOF::Zero(), sim.fix_eulerian);
                sim.dirichlet_data[0] = std::make_pair(TVDOF::Zero(), sim.fix_all);
            }
            else
            {
                for(int i = 4; i < 11; i++)
                    sim.dirichlet_data[i] = std::make_pair(TVDOF::Zero(), sim.fix_eulerian);
                
                sim.dirichlet_data[3] = std::make_pair(TVDOF::Zero(), sim.fix_eulerian);
                sim.dirichlet_data[1] = std::make_pair(TVDOF::Zero(), sim.fix_eulerian);

                sim.dirichlet_data[0] = std::make_pair(TVDOF::Zero(), sim.fix_all);
                sim.sliding_nodes = {2};
            }
            sim.n_dof = sim.n_nodes * sim.dof;            
            // sim.sliding_nodes = {1, 2, 3};
            
        }
        if(sub_div > 1)
        {
            subdivideStraightYarns(sub_div);

            int n_bending_pairs = 4;
            
            std::vector<int> init(sim.N_PBC_BENDING_ELE, -1);
            for (int i = 0; i < n_bending_pairs; i++)
                sim.pbc_bending_bn_pairs.push_back(init);

            for (int i = 0; i < sim.n_rods; i++)
            {
                add4Nodes(0, 1, 0, i);
                add4Nodes(2, 3, 1, i);
                add4Nodes(4, 5, 2, i);
                add4Nodes(6, 7, 3, i);
            }
            
            assert(sim.pbc_bending_bn_pairs == n_bending_pairs);

            

        }
        else
        {
            sim.q0 = q;
            sim.W = StiffnessMatrix(sim.n_nodes * sim.dof, sim.n_dof);
            sim.W.setIdentity();
        }
        
        sim.is_end_nodes = std::vector<bool>(sim.n_nodes, false);

        sim.curvature_functions.push_back(new LineCurvature<T, dim>());
        sim.curvature_functions.push_back(new PreBendCurvaure<T, dim>(
            2.0 * std::sqrt(2) * 0.1,
            M_PI/2.0));

        sim.slide_over_n_rods = IV2(std::floor(sub_div * 0.25), std::floor(sub_div * 0.25));
        T rod_length = (sim.q0.col(rods.col(1)(0)).template segment<dim>(0) - 
            sim.q0.col(rods.col(1)(1)).template segment<dim>(0)).norm();

        sim.tunnel_u = sim.slide_over_n_rods[0] * rod_length;
        sim.tunnel_v = sim.tunnel_u;
    }
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
void UnitPatch<T, dim>::buildStraightYarn3x1(int sub_div)
{
    clearSimData();
    sim.n_nodes = 11;
    sim.n_rods = 10;

    q = DOFStack(sim.dof, sim.n_nodes); q.setZero();
    rods = IV3Stack(3, sim.n_rods); rods.setZero();
    connections = IV4Stack(4, sim.n_nodes).setOnes() * -1;

    sim.normal = TV3Stack(3, sim.n_rods);
    sim.normal.setZero();
    sim.subdivide = true;
    

    if constexpr (dim == 2)
    {
        {
            q.col(0).template segment<dim>(0) = TV2(0.125, 0.25);
            q.col(1).template segment<dim>(0) = TV2(0.25, 0.25);
            q.col(2).template segment<dim>(0) = TV2(0.5, 0.25);
            q.col(3).template segment<dim>(0) = TV2(0.75, 0.25);
            q.col(4).template segment<dim>(0) = TV2(0.875, 0.25);

            q.col(5).template segment<dim>(0) = TV2(0.25, 0.125);
            q.col(6).template segment<dim>(0) = TV2(0.25, 0.375);
            q.col(7).template segment<dim>(0) = TV2(0.5, 0.125);
            q.col(8).template segment<dim>(0) = TV2(0.5, 0.375);
            q.col(9).template segment<dim>(0) = TV2(0.75, 0.125);
            q.col(10).template segment<dim>(0) = TV2(0.75, 0.375);

            q.block(dim, 0, 2, 11) = q.block(0, 0, 2, 11);
            
            int cnt = 0;
            std::vector<int> rod0 = {0, 1, 2, 3, 4}, 
                             rod1 = {5, 1, 6},
                             rod2 = {7, 2, 8},
                             rod3 = {9, 3, 10};
            
            addRods(rod0, WARP, cnt, 0);
            addRods(rod1, WEFT, cnt, 1);
            addRods(rod2, WEFT, cnt, 2);
            addRods(rod3, WEFT, cnt, 3);
            
            sim.q0 = q;
            sim.n_dof = sim.n_nodes * sim.dof;

            
            sim.pbc_ref.push_back(std::make_pair(WEFT, IV2(5, 6)));
            sim.pbc_ref.push_back(std::make_pair(WEFT, IV2(7, 8)));
            sim.pbc_ref.push_back(std::make_pair(WEFT, IV2(9, 10)));

            sim.pbc_ref.push_back(std::make_pair(WARP, IV2(0, 4)));
            sim.pbc_ref_unique.push_back(IV2(0, 4));
            sim.pbc_ref_unique.push_back(IV2(5, 6));

            if (sim.disable_sliding)
            {
                for(int i = 0; i < sim.n_nodes; i++)
                    sim.dirichlet_data[i] = std::make_pair(TVDOF::Zero(), sim.fix_eulerian);
                sim.dirichlet_data[0] = std::make_pair(TVDOF::Zero(), sim.fix_all);
            }
            else
            {
                for(int i = 4; i < 11; i++)
                    sim.dirichlet_data[i] = std::make_pair(TVDOF::Zero(), sim.fix_eulerian);
                
                sim.dirichlet_data[3] = std::make_pair(TVDOF::Zero(), sim.fix_eulerian);
                sim.dirichlet_data[1] = std::make_pair(TVDOF::Zero(), sim.fix_eulerian);

                sim.dirichlet_data[0] = std::make_pair(TVDOF::Zero(), sim.fix_all);
            }
            sim.n_dof = sim.n_nodes * sim.dof;            
            // sim.sliding_nodes = {1, 2, 3};
            sim.sliding_nodes = {2};
        }
        if(sub_div > 1)
        {
            subdivideStraightYarns(sub_div);

            int n_bending_pairs = 4;
            
            std::vector<int> init(sim.N_PBC_BENDING_ELE, -1);
            for (int i = 0; i < n_bending_pairs; i++)
                sim.pbc_bending_bn_pairs.push_back(init);

            for (int i = 0; i < sim.n_rods; i++)
            {
                add4Nodes(0, 4, 0, i);
                add4Nodes(5, 6, 1, i);
                add4Nodes(7, 8, 2, i);
                add4Nodes(9, 10, 3, i);
            }
            
            assert(sim.pbc_bending_bn_pairs == n_bending_pairs);

            

        }
        else
        {
            sim.q0 = q;
            sim.W = StiffnessMatrix(sim.n_nodes * sim.dof, sim.n_dof);
            sim.W.setIdentity();
        }
        
        sim.is_end_nodes = std::vector<bool>(sim.n_nodes, false);

        sim.curvature_functions.push_back(new LineCurvature<T, dim>());
        sim.curvature_functions.push_back(new LineCurvature<T, dim>());

        sim.slide_over_n_rods = IV2(std::floor(sub_div * 0.25), std::floor(sub_div * 0.25));
        T rod_length = (sim.q0.col(rods.col(1)(0)).template segment<dim>(0) - 
            sim.q0.col(rods.col(1)(1)).template segment<dim>(0)).norm();

        sim.tunnel_u = sim.slide_over_n_rods[0] * rod_length;
        sim.tunnel_v = sim.tunnel_u;
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
        
        if(false)
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
            // q *= 0.03;
            sim.q0 = q;
            sim.n_dof = dof_cnt;
            sim.W = StiffnessMatrix(sim.n_nodes * sim.dof, sim.n_dof);
            sim.W.setFromTriplets(w_entry.begin(), w_entry.end());
        }
        else
        {
            q(dim, 3) = q(dim, 2) + LDis(2, 3);
            q(dim, 4) = q(dim, 3) + LDis(3, 4);
            // q *= 0.03;
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