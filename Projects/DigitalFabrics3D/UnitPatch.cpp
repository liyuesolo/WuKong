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
        build3DtestScene(4);
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

        sim.dirichlet_dof[ob[0]] = -0.1 * sim.unit;
        sim.dirichlet_dof[ob[1]] = 0.1 * sim.unit;
        sim.dirichlet_dof[ob[2]] = 0;


        for (int i = theta_reduced_dof_offset; i < dof_cnt; i++)
        {
            sim.dirichlet_dof[i] = 0;
            break;
            // sim.dirichlet_dof[i] = T(i) * M_PI / 4;
        }

        sim.dirichlet_dof[dof_cnt-1] = 0;

        
        // sim.dirichlet_dof[ob[0]] = -0.3 * sim.unit;
        // sim.dirichlet_dof[ob[1]] = 0.1 * sim.unit;
        // sim.dirichlet_dof[ob[2]] = 0.0 * sim.unit;

        for (int d = 0; d < dim; d++)
        {
            sim.dirichlet_dof[of[d]] = 0;
        }

        // sim.dirichlet_dof[r0->reduced_map[r0->offset_map[1][0]]] = 0.01 * sim.unit;
        // sim.dirichlet_dof[r0->reduced_map[r0->offset_map[1][1]]] = 0.01 * sim.unit;
        // sim.dirichlet_dof[r0->reduced_map[r0->offset_map[1][2]]] = 0.01 * sim.unit;


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
    
    sim.pbc_ref_unique.clear();
    sim.dirichlet_data.clear();
    sim.pbc_ref.clear();
    sim.pbc_bending_pairs.clear();
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