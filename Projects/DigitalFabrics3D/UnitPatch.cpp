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
        build3DtestScene(16);
    else if (patch_type == 1)
        buildOneCrossScene(16);
    else if (patch_type == 2)
        buildGridScene(16);
}


template<class T, int dim>
void UnitPatch<T, dim>::addAStraightRod(const TV& from, const TV& to, 
        const std::vector<TV>& passing_points, 
        const std::vector<int>& passing_points_id, 
        int sub_div, int& full_dof_cnt, int& node_cnt, int& rod_cnt, bool closed)
{
    
    std::unordered_map<int, Offset> offset_map;

    std::vector<TV> points_on_curve;
    std::vector<int> rod_indices;
    std::vector<int> key_points_location_rod;
    addStraightYarnCrossNPoints(from, to, passing_points, passing_points_id,
                                sub_div, points_on_curve, rod_indices,
                                key_points_location_rod, node_cnt);
                                
    deformed_states.conservativeResize(full_dof_cnt + (points_on_curve.size()) * (dim + 1));

    Rod<T, dim>* rod = new Rod<T, dim>(deformed_states, sim.rest_states, rod_cnt, closed);

    for (int i = 0; i < points_on_curve.size(); i++)
    {
        offset_map[node_cnt] = Offset::Zero();
        //push Lagrangian DoF    
        deformed_states.template segment<dim>(full_dof_cnt) = points_on_curve[i];
        for (int d = 0; d < dim; d++)
        {
            offset_map[node_cnt][d] = full_dof_cnt++;    
        }
        // push Eulerian DoF
        deformed_states[full_dof_cnt] = (points_on_curve[i] - from).norm() / (to - from).norm();
        offset_map[node_cnt][dim] = full_dof_cnt++;
        node_cnt++;
    }
    
    deformed_states.conservativeResize(full_dof_cnt + passing_points.size());

    for (int i = 0; i < passing_points.size(); i++)
    {
        deformed_states[full_dof_cnt] = (passing_points[i] - from).norm() / (to - from).norm();
        offset_map[passing_points_id[i]] = Offset::Zero();
        offset_map[passing_points_id[i]][dim] = full_dof_cnt++; 
        Vector<int, dim> offset_dof_lag;
        for (int d = 0; d < dim; d++)
        {
            offset_dof_lag[d] = passing_points_id[i] * dim + d;
        }
        offset_map[passing_points_id[i]].template segment<dim>(0) = offset_dof_lag;
    }
    
    rod->offset_map = offset_map;
    rod->indices = rod_indices;
    Vector<T, dim + 1> q0, q1;
    rod->frontDoF(q0); rod->backDoF(q1);

    rod->rest_state = new LineCurvature<T, dim>(q0, q1);
    
    rod->dof_node_location = key_points_location_rod;
    
    // for(int idx : rod->indices)
    // {
    //     TV pos;
    //     rod->x(idx, pos);
    //     std::cout << pos.transpose() << std::endl;
    // }
    sim.Rods.push_back(rod);

    // sim.iterateAllLagrangianDoFs([&](Offset offset)
    // {
    //     T r = static_cast <T> (rand()) / static_cast <T> (RAND_MAX);
    //     deformed_states[offset[dim - 1]] += 0.01 * (r - 0.5) * sim.unit;
    // });
    // sim.rest_states = deformed_states;

}

template<class T, int dim>
void UnitPatch<T, dim>::appendThetaAndJointDoF(std::vector<Entry>& w_entry, 
    int& full_dof_cnt, int& dof_cnt)
{
    for (auto& rod : sim.Rods)
    {
        rod->theta_dof_start_offset = full_dof_cnt;
        rod->theta_reduced_dof_start_offset = dof_cnt;
        deformed_states.conservativeResize(full_dof_cnt + rod->indices.size() - 1);
        for (int i = 0; i < rod->indices.size() - 1; i++)
            w_entry.push_back(Entry(full_dof_cnt++, dof_cnt++, 1.0));
        deformed_states.template segment(rod->theta_dof_start_offset, 
            rod->indices.size() - 1).setZero();
    }   

    deformed_states.conservativeResize(full_dof_cnt + sim.rod_crossings.size() * dim);
    deformed_states.template segment(full_dof_cnt, sim.rod_crossings.size() * dim).setZero();

    for (auto& crossing : sim.rod_crossings)
    {
        crossing->dof_offset = full_dof_cnt;
        crossing->reduced_dof_offset = dof_cnt;
        for (int d = 0; d < dim; d++)
        {
            w_entry.push_back(Entry(full_dof_cnt++, dof_cnt++, 1.0));
        }
    }
}

template<class T, int dim>
void UnitPatch<T, dim>::buildGridScene(int sub_div)
{
    if constexpr (dim == 3)
    {
        auto unit_yarn_map = sim.yarn_map;
        sim.yarn_map.clear();
        sim.add_rotation_penalty = false;
        sim.add_pbc_bending = false;
        sim.new_frame_work = true;

        clearSimData();

        
        std::vector<Eigen::Triplet<T>> w_entry;
        int full_dof_cnt = 0;
        int node_cnt = 0;

        int n_row = 3, n_col = 3;

        // push crossings first 
        T dy = 1.0 / n_row * sim.unit;
        T dx = 1.0 / n_col * sim.unit;
        
        //num of crossing
        deformed_states.resize(n_col * n_row * dim);
        
        std::unordered_map<int, Offset> crossing_offset_copy;

        auto getXY = [=](int row, int col, T& x, T& y)
        {
            if (row == 0) y = 0.5 * dy;
            else if (row == n_row) y = n_row * dy;
            else y = 0.5 * dy + (row ) * dy;
            if (col == 0) x = 0.5 * dx;
            else if (col == n_col) x = n_col * dx;
            else x = 0.5 * dx + (col ) * dx;
        };

        for (int row = 0; row < n_row; row++)
        {
            for (int col = 0; col < n_col; col++)
            {
                T x, y;
                getXY(row, col, x, y);
                deformed_states.template segment<dim>(node_cnt * dim) = TV(x, y, 0);
                
                full_dof_cnt += dim;
                node_cnt ++;       
            }
        }

        int rod_cnt = 0;
        for (int row = 0; row < n_row; row++)
        {
            T x0 = 0.0, x1 = 1.0 * sim.unit;
            T x, y;
            
            std::vector<int> passing_points_id;
            std::vector<TV> passing_points;
            
            for (int col = 0; col < n_col; col++)
            {
                int node_idx = row * n_col + col;
                passing_points_id.push_back(node_idx);
                passing_points.push_back(deformed_states.template segment<dim>(node_idx * dim));
            }

            getXY(row, 0, x, y);

            TV from = TV(x0, y, 0);
            TV to = TV(x1, y, 0);
        
            addAStraightRod(from, to, passing_points, passing_points_id, 
                sub_div, full_dof_cnt, node_cnt, rod_cnt, false);
            rod_cnt ++;
        }
        
        for (int col = 0; col < n_col; col++)
        {
            T y0 = 0.0, y1 = 1.0 * sim.unit;
            T x, y;
            std::vector<int> passing_points_id;
            std::vector<TV> passing_points;
            getXY(0, col, x, y);
            for (int row = 0; row < n_row; row++)
            {
                int node_idx = row * n_col + col;
                passing_points_id.push_back(node_idx);
                passing_points.push_back(deformed_states.template segment<dim>(node_idx * dim));
            }
            
            TV from = TV(x, y0, 0);
            TV to = TV(x, y1, 0);

            addAStraightRod(from, to, passing_points, passing_points_id, sub_div, 
                            full_dof_cnt, node_cnt, rod_cnt, false);
            rod_cnt ++;
        }
        

        
        for (int row = 0; row < n_row; row++)
        {
            for (int col = 0; col < n_col; col++)
            {
                int node_idx = row * n_col + col;
                RodCrossing<T, dim>* crossing = 
                    new RodCrossing<T, dim>(node_idx, {row, n_row + col});

                // std::cout << row << " " << n_row + col << std::endl;

                crossing->sliding_ranges = { Range(0.2, 0.2), Range(0.2, 0.2)};
                
                // std::cout << sim.Rods[row]->dof_node_location[col] << " "
                //             << sim.Rods[n_row + col]->dof_node_location[row] << std::endl;

                crossing->on_rod_idx[row] = sim.Rods[row]->dof_node_location[col];
                crossing->on_rod_idx[n_row + col] = sim.Rods[n_row + col]->dof_node_location[row];
                crossing->is_fixed = true;
                sim.rod_crossings.push_back(crossing);
            }
        }    

        int dof_cnt = 0;
        markCrossingDoF(w_entry, dof_cnt);

        for (auto& rod : sim.Rods) rod->markDoF(w_entry, dof_cnt);
        
        appendThetaAndJointDoF(w_entry, full_dof_cnt, dof_cnt);
        
        // sim.iterateAllLagrangianDoFs([&](Offset offset)
        // {
        //     T r = static_cast <T> (rand()) / static_cast <T> (RAND_MAX);
        //     deformed_states[offset[dim - 1]] += 0.01 * (r - 0.5) * sim.unit;
        // });
        sim.rest_states = deformed_states;

    
        sim.W = StiffnessMatrix(full_dof_cnt, dof_cnt);
        sim.W.setFromTriplets(w_entry.begin(), w_entry.end());
        
        // std::cout << sim.W << std::endl;

        int cnt = 0;
        for (auto& rod : sim.Rods)
        {
            
            rod->fixEndPointEulerian(sim.dirichlet_dof);
            // if (cnt)
                // rod->fixEndPointLagrangian(sim.dirichlet_dof);
            sim.dirichlet_dof[rod->theta_reduced_dof_start_offset] = 0;
            rod->setupBishopFrame();
            cnt++;
            Offset end0, end1;
            rod->frontOffset(end0); rod->backOffset(end1);
            if (rod->rod_id < n_row)
                sim.pbc_pairs.push_back(std::make_pair(0, std::make_pair(end0, end1)));
            else
                sim.pbc_pairs.push_back(std::make_pair(1, std::make_pair(end0, end1)));
        }

        

        Offset of, ob;
        sim.Rods[0]->frontOffsetReduced(of);
        sim.Rods[0]->backOffsetReduced(ob);

        sim.dirichlet_dof[sim.Rods[0]->reduced_map[sim.Rods[0]->offset_map[0][0]]] = 0;
        sim.dirichlet_dof[sim.Rods[0]->reduced_map[sim.Rods[0]->offset_map[0][1]]] = 0;
        sim.dirichlet_dof[sim.Rods[0]->reduced_map[sim.Rods[0]->offset_map[0][2]]] = 0;

        // sim.dirichlet_dof[ob[0]] = 0.1 * sim.unit;
        // sim.dirichlet_dof[ob[1]] = 0.1 * sim.unit;

        // sim.dirichlet_dof[of[0]] = -0.1 * sim.unit;
        // sim.dirichlet_dof[of[1]] = 0.1 * sim.unit;

        Offset end0, end1;
        sim.Rods[0]->frontOffset(end0); sim.Rods[0]->backOffset(end1);
        sim.pbc_pairs_reference[0] = std::make_pair(std::make_pair(end0, end1), 0);
        sim.Rods[1]->frontOffset(end0); sim.Rods[1]->backOffset(end1);
        sim.pbc_pairs_reference[1] = std::make_pair(std::make_pair(end0, end1), 1);

        sim.fixCrossing();
    }
}

template<class T, int dim>
void UnitPatch<T, dim>::buildOneCrossScene(int sub_div)
{
    if constexpr (dim == 3)
    {
        int sub_div_2 = sub_div / 2;
        auto unit_yarn_map = sim.yarn_map;
        sim.yarn_map.clear();
        sim.add_rotation_penalty = false;
        sim.add_pbc_bending = false;
        sim.new_frame_work = true;

        clearSimData();
        std::vector<Eigen::Triplet<T>> w_entry;
        int full_dof_cnt = 0;
        int node_cnt = 0;
        std::unordered_map<int, Offset> offset_map;
        
        TV from(0.0, 0.5, 0.0);
        TV to(1.0, 0.5, 0.0);
        from *= sim.unit; to *= sim.unit;

        TV center = TV(0.5, 0.5, 0.0) * sim.unit;
        int center_id = 0;
        deformed_states.resize(dim);
        deformed_states.template segment<dim>(full_dof_cnt) = center;
        offset_map[node_cnt] = Offset::Zero();
        for (int d = 0; d < dim; d++) offset_map[node_cnt][d] = full_dof_cnt++;
        node_cnt++;
        auto center_offset = offset_map[center_id];

        std::vector<TV> points_on_curve;
        std::vector<int> rod0;
        std::vector<int> key_points_location_rod0;

        addStraightYarnCrossNPoints(from, to, {center}, {0}, sub_div, points_on_curve, rod0, key_points_location_rod0, node_cnt);

        deformed_states.conservativeResize(full_dof_cnt + (points_on_curve.size()) * (dim + 1));

        Rod<T, dim>* r0 = new Rod<T, dim>(deformed_states, sim.rest_states, 0, false);

        for (int i = 0; i < points_on_curve.size(); i++)
        {
            offset_map[node_cnt] = Offset::Zero();
            
            //push Lagrangian DoF
            
            deformed_states.template segment<dim>(full_dof_cnt) = points_on_curve[i];
            
            for (int d = 0; d < dim; d++)
            {
                offset_map[node_cnt][d] = full_dof_cnt++;    
            }
            // push Eulerian DoF
            deformed_states[full_dof_cnt] = (points_on_curve[i] - from).norm() / (to - from).norm();
            offset_map[node_cnt][dim] = full_dof_cnt++;
            node_cnt++;
        }
        deformed_states.conservativeResize(full_dof_cnt + 1);
        deformed_states[full_dof_cnt] = (center - from).norm() / (to - from).norm();
        offset_map[center_id][dim] = full_dof_cnt++;

        r0->offset_map = offset_map;
        r0->indices = rod0;

        Vector<T, dim + 1> q0, q1;
        r0->frontDoF(q0); r0->backDoF(q1);
        r0->rest_state = new LineCurvature<T, dim>(q0, q1);
        
        r0->dof_node_location = key_points_location_rod0;
        sim.Rods.push_back(r0);

        offset_map.clear();
        
        TV rod1_from(0.5, 0.0, 0.0);
        TV rod1_to(0.5, 1.0, 0.0);
        rod1_from *= sim.unit; rod1_to *= sim.unit;

        points_on_curve.clear();
        points_on_curve.resize(0);
        std::vector<int> rod1;
        std::vector<int> key_points_location_rod1;

        addStraightYarnCrossNPoints(rod1_from, rod1_to, {center}, {0}, sub_div, points_on_curve, rod1, key_points_location_rod1, node_cnt);

        deformed_states.conservativeResize(full_dof_cnt + (points_on_curve.size()) * (dim + 1));

        Rod<T, dim>* r1 = new Rod<T, dim>(deformed_states, sim.rest_states, 1, false);
        for (int i = 0; i < points_on_curve.size(); i++)
        {
            offset_map[node_cnt] = Offset::Zero();
            //push Lagrangian DoF
            deformed_states.template segment<dim>(full_dof_cnt) = points_on_curve[i];
            // std::cout << points_on_curve[i].transpose() << std::endl;
            for (int d = 0; d < dim; d++)
            {
                offset_map[node_cnt][d] = full_dof_cnt++;    
            }
            // push Eulerian DoF
            deformed_states[full_dof_cnt] = (points_on_curve[i] - rod1_from).norm() / (rod1_to - rod1_from).norm();
            offset_map[node_cnt][dim] = full_dof_cnt++;
            node_cnt++;
        }

        deformed_states.conservativeResize(full_dof_cnt + 1);

        deformed_states[full_dof_cnt] = (center - rod1_from).norm() / (rod1_to - rod1_from).norm();
        offset_map[center_id] = Offset::Zero();
        offset_map[center_id].template segment<dim>(0) = center_offset.template segment<dim>(0);
        offset_map[center_id][dim] = full_dof_cnt++;

        r1->offset_map = offset_map;
        r1->indices = rod1;

        r1->frontDoF(q0); r1->backDoF(q1);
        r1->rest_state = new LineCurvature<T, dim>(q0, q1);
        
        r1->dof_node_location = key_points_location_rod1;
        sim.Rods.push_back(r1);

        RodCrossing<T, dim>* rc0 = new RodCrossing<T, dim>(0, {0, 1});
        rc0->sliding_ranges = { Range(0.2, 0.2), Range(0.2, 0.2)};
        rc0->on_rod_idx[0] = key_points_location_rod0[0];
        rc0->on_rod_idx[1] =  key_points_location_rod1[0];
        sim.rod_crossings.push_back(rc0);

        

        int dof_cnt = 0;
        markCrossingDoF(w_entry, dof_cnt);
        r0->markDoF(w_entry, dof_cnt);
        r1->markDoF(w_entry, dof_cnt);

        r0->theta_dof_start_offset = full_dof_cnt;
        r0->theta_reduced_dof_start_offset = dof_cnt;        
        int theta_reduced_dof_offset0 = dof_cnt;
        deformed_states.conservativeResize(full_dof_cnt + r0->indices.size() - 1);
        for (int i = 0; i < r0->indices.size() - 1; i++)
        {
            w_entry.push_back(Entry(full_dof_cnt++, dof_cnt++, 1.0));
        }   
        deformed_states.template segment(r0->theta_dof_start_offset, 
            r0->indices.size() - 1).setZero();

        r1->theta_dof_start_offset = full_dof_cnt;
        
        int theta_reduced_dof_offset1 = dof_cnt;
        r1->theta_reduced_dof_start_offset = dof_cnt;
        deformed_states.conservativeResize(full_dof_cnt + r1->indices.size() - 1);
        for (int i = 0; i < r1->indices.size() - 1; i++)
        {
            w_entry.push_back(Entry(full_dof_cnt++, dof_cnt++, 1.0));
        }   
        deformed_states.template segment(r1->theta_dof_start_offset, 
            r1->indices.size() - 1).setZero();

        deformed_states.conservativeResize(full_dof_cnt + sim.rod_crossings.size() * dim);
        deformed_states.template segment(full_dof_cnt, sim.rod_crossings.size() * dim).setZero();

        for (auto& crossing : sim.rod_crossings)
        {
            crossing->dof_offset = full_dof_cnt;
            crossing->reduced_dof_offset = dof_cnt;
            for (int d = 0; d < dim; d++)
            {
                w_entry.push_back(Entry(full_dof_cnt++, dof_cnt++, 1.0));
            }
        }
        
        sim.rest_states = sim.deformed_states;
        sim.W = StiffnessMatrix(full_dof_cnt, dof_cnt);
        sim.W.setFromTriplets(w_entry.begin(), w_entry.end());

        // std::cout << "r0->theta_dof_start_offset " << r0->theta_dof_start_offset << " sim.W.cols() " << sim.W.cols() << std::endl;
        

        Offset ob, of;
        r0->backOffsetReduced(ob);
        r0->frontOffsetReduced(of);

        // std::cout << ob.transpose() << " " << of.transpose() << std::endl;
        r0->fixEndPointEulerian(sim.dirichlet_dof);
        r1->fixEndPointEulerian(sim.dirichlet_dof);

        // r1->fixEndPointLagrangian(sim.dirichlet_dof);

        // sim.fixCrossing();

        sim.dirichlet_dof[ob[0]] = -0.3 * sim.unit;
        sim.dirichlet_dof[ob[1]] = 0.3 * sim.unit;
        // sim.dirichlet_dof[ob[2]] = 0;
        sim.dirichlet_dof[ob[2]] = 0.3 * sim.unit;


        sim.dirichlet_dof[theta_reduced_dof_offset0] = 0;
        sim.dirichlet_dof[theta_reduced_dof_offset1] = 0;

        Offset ob1, of1;
        r1->backOffsetReduced(ob1);
        r1->frontOffsetReduced(of1);


        sim.dirichlet_dof[ob1[0]] = 0.15 * sim.unit;
        sim.dirichlet_dof[ob1[1]] = -0.2 * sim.unit;
        sim.dirichlet_dof[ob1[2]] = -0.1 * sim.unit;

        for (int d = 0; d < dim; d++)
        {
            sim.dirichlet_dof[of[d]] = 0;
            sim.dirichlet_dof[ob1[d]] = 0;
            sim.dirichlet_dof[of1[d]] = 0;
        }

        sim.dirichlet_dof[r0->reduced_map[r0->offset_map[1][0]]] = 0.0 * sim.unit;
        sim.dirichlet_dof[r0->reduced_map[r0->offset_map[1][1]]] = 0.0 * sim.unit;
        sim.dirichlet_dof[r0->reduced_map[r0->offset_map[1][2]]] = 0.0 * sim.unit;
        
        for (auto& rod : sim.Rods)
        {
            rod->setupBishopFrame();
        }
        
    }
}
template<class T, int dim>
void UnitPatch<T, dim>::build3DtestScene(int sub_div)
{
    if constexpr (dim == 3)
    {
        int sub_div_2 = sub_div / 2;
        auto unit_yarn_map = sim.yarn_map;
        sim.yarn_map.clear();
        sim.add_rotation_penalty = false;
        sim.add_pbc_bending = false;
        sim.new_frame_work = true;

        clearSimData();
        std::vector<Eigen::Triplet<T>> w_entry;
        int full_dof_cnt = 0;
        int node_cnt = 0;
        
        TV from(0.0, 0.5, 0.0);
        TV to(1.0, 0.5, 0.0);
        from *= sim.unit; to *= sim.unit;

        std::vector<TV> points_on_curve;
        std::vector<int> rod0;
        std::vector<int> dummy;

        addStraightYarnCrossNPoints(from, to, {}, {}, sub_div, points_on_curve, rod0, dummy, 0);

        // std::cout << points_on_curve.size() << " " << rod0.size() << std::endl;

        deformed_states.resize((points_on_curve.size()) * (dim + 1));

        Rod<T, dim>* r0 = new Rod<T, dim>(deformed_states, sim.rest_states, 0, false);

        std::unordered_map<int, Vector<int, dim + 1>> offset_map;
        std::vector<int> node_index_list;

        std::vector<T> data_points_discrete_arc_length;
        
        for (int i = 0; i < points_on_curve.size(); i++)
        {
            offset_map[i] = Offset::Zero();
            node_cnt++;
            node_index_list.push_back(i);
            //push Lagrangian DoF
            
            deformed_states.template segment<dim>(full_dof_cnt) = points_on_curve[i];
            // std::cout << points_on_curve[i].transpose() << std::endl;
            for (int d = 0; d < dim; d++)
            {
                offset_map[i][d] = full_dof_cnt++;    
            }
            // push Eulerian DoF
            deformed_states[full_dof_cnt] = (points_on_curve[i] - from).norm() / (to - from).norm();
            
            offset_map[i][dim] = full_dof_cnt++;
            
        }

        r0->offset_map = offset_map;
        r0->indices = node_index_list;
        // for (int idx : node_index_list)
        //     std::cout << idx << " ";
        // std::cout << std::endl;
        Vector<T, dim + 1> q0, q1;
        r0->frontDoF(q0); r0->backDoF(q1);
        

        r0->rest_state = new LineCurvature<T, dim>(q0, q1);
        
        r0->dof_node_location = {};
        sim.Rods.push_back(r0);

        int dof_cnt = 0;
        
        r0->markDoF(w_entry, dof_cnt);
        r0->theta_dof_start_offset = full_dof_cnt;
        
        int theta_reduced_dof_offset = dof_cnt;
        deformed_states.conservativeResize(full_dof_cnt + r0->indices.size() - 1);
        for (int i = 0; i < r0->indices.size() - 1; i++)
        {
            w_entry.push_back(Entry(full_dof_cnt++, dof_cnt++, 1.0));
        }   
        deformed_states.template segment(r0->theta_dof_start_offset, 
            r0->indices.size() - 1).setZero();
        
        sim.rest_states = sim.deformed_states;
        sim.W = StiffnessMatrix(full_dof_cnt, dof_cnt);
        sim.W.setFromTriplets(w_entry.begin(), w_entry.end());

        // std::cout << "r0->theta_dof_start_offset " << r0->theta_dof_start_offset << " sim.W.cols() " << sim.W.cols() << std::endl;
        

        Offset ob, of;
        r0->backOffsetReduced(ob);
        r0->frontOffsetReduced(of);

        // std::cout << ob.transpose() << " " << of.transpose() << std::endl;
        r0->fixEndPointEulerian(sim.dirichlet_dof);
        

        sim.dirichlet_dof[ob[0]] = -0.3 * sim.unit;
        sim.dirichlet_dof[ob[1]] = 0.3 * sim.unit;
        // sim.dirichlet_dof[ob[2]] = 0;


        for (int i = theta_reduced_dof_offset; i < dof_cnt; i++)
        {
            sim.dirichlet_dof[i] = 0;
            break;
            // sim.dirichlet_dof[i] = T(i) * M_PI / 4;
        }

        // sim.dirichlet_dof[dof_cnt-1] = M_PI / 2.0;
        // sim.dirichlet_dof[dof_cnt-1] = M_PI;
        // sim.dirichlet_dof[dof_cnt-1] = 0;

        
        // sim.dirichlet_dof[ob[0]] = -0.3 * sim.unit;
        // sim.dirichlet_dof[ob[1]] = 0.1 * sim.unit;
        // sim.dirichlet_dof[ob[2]] = 0.0 * sim.unit;

        for (int d = 0; d < dim; d++)
        {
            sim.dirichlet_dof[of[d]] = 0;
        }

        // sim.dirichlet_dof[r0->reduced_map[r0->offset_map[1][0]]] = 0.0 * sim.unit;
        // sim.dirichlet_dof[r0->reduced_map[r0->offset_map[1][1]]] = 0.0 * sim.unit;
        // sim.dirichlet_dof[r0->reduced_map[r0->offset_map[1][2]]] = 0.0 * sim.unit;


        // sim.dirichlet_dof[r0->reduced_map[r0->offset_map[node_cnt-2][0]]] = -0.19 * sim.unit;
        // sim.dirichlet_dof[r0->reduced_map[r0->offset_map[node_cnt-2][1]]] = 0.18 * sim.unit;
        // sim.dirichlet_dof[r0->reduced_map[r0->offset_map[node_cnt-2][2]]] = 0.18 * sim.unit;
        
        for (auto& rod : sim.Rods)
        {
            rod->setupBishopFrame();
        }
        
        // std::ifstream in("./testdq.txt");
        // for(int i =0; i < deformed_states.rows(); i++)
        //     in>> deformed_states[i];
        
        // in.close();

        
        // VectorXT dq = sim.W.transpose() * deformed_states;
        // sim.testGradient(dq);
        // sim.testHessian(dq);

        // for (auto& it : sim.dirichlet_dof)
        //     std::cout << it.first << " " << it.second << std::endl;
        // std::cout << deformed_states << std::endl;
        // std::cout << sim.W << std::endl;
        // std::cout << sim.W.rows() << " " << sim.W.cols() << " " << deformed_states.rows() << " " << r0->theta_dof_start_offset<< std::endl;
    }
}

template<class T, int dim>
void UnitPatch<T, dim>::markCrossingDoF(std::vector<Eigen::Triplet<T>>& w_entry,
        int& dof_cnt)
{
    for (auto& crossing : sim.rod_crossings)
    {
        int node_idx = crossing->node_idx;
        // std::cout << "node " << node_idx << std::endl;
        std::vector<int> rods_involved = crossing->rods_involved;

        Offset entry_rod0; 
        sim.Rods[rods_involved.front()]->getEntry(node_idx, entry_rod0);

        // push Lagrangian dof first
        for (int d = 0; d < dim; d++)
        {
            
            for (int rod_idx : rods_involved)
            {
                sim.Rods[rod_idx]->reduced_map[entry_rod0[d]] = dof_cnt;
            }    
            w_entry.push_back(Entry(entry_rod0[d], dof_cnt++, 1.0));
        }
        
        // push Eulerian dof for all rods
        for (int rod_idx : rods_involved)
        {
            // std::cout << "dim on rod " <<  rod_idx << std::endl;
            sim.Rods[rod_idx]->getEntry(node_idx, entry_rod0);
            // std::cout << "dim dof on rod " <<  entry_rod0[dim] << std::endl;
            sim.Rods[rod_idx]->reduced_map[entry_rod0[dim]] = dof_cnt;
            w_entry.push_back(Entry(entry_rod0[dim], dof_cnt++, 1.0));
        }
        
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
    
    // sim.pbc_ref_unique.clear();
    // sim.dirichlet_data.clear();
    // sim.pbc_ref.clear();
    // sim.pbc_bending_pairs.clear();
    sim.yarns.clear();
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
    if(passing_points.size())
    {
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


template class UnitPatch<double, 3>;
template class UnitPatch<double, 2>;   