#include "Rod.h"

template<class T, int dim>
void Rod<T, dim>::markDoF(std::vector<Entry>& w_entry, int& dof_cnt)
{
    // std::cout << "[Rod" << rod_id << "]" << std::endl;
    int loop_id = 0;
    if (dof_node_location.front() != 0)
    {
        // add first node
        Offset offset = offset_map[indices.front()];
        // std::cout << "node " <<  indices.front() << " added all dof " << std::endl;
        for(int d = 0; d < dim + 1; d++)
        {
            w_entry.push_back(Eigen::Triplet<T>(offset[d], dof_cnt++, 1.0));
        }
    }
    // loop over each segment
    for (int i = 0; i < dof_node_location.size(); i++)
    {   
        // compute weight value for in-between nodes
        // using linear interpolation
        // std::cout << "left node " << indices[loop_id] << " right node " << indices[dof_node_location[i]] << std::endl;
        
        Offset offset_left_node = offset_map[indices[loop_id]];
        Offset offset_right_node = offset_map[indices[dof_node_location[i]]];
        T ui = full_states[offset_left_node[dim]];
        T uj = full_states[offset_right_node[dim]];

        for (int j = loop_id + 1; j < dof_node_location[i]; j++)
        {
            int current_global_idx = indices[j];
            // std::cout << "current node " << current_global_idx << std::endl;
            //push Lagrangian DoF first
            Offset offset = offset_map[current_global_idx];
            for(int d = 0; d < dim; d++)
            {
                w_entry.push_back(Eigen::Triplet<T>(offset[d], dof_cnt++, 1.0));
            }
            // compute Eulerian weight
            T u = full_states[offset[dim]];
            T alpha = (u - ui) / (uj - ui);
            // std::cout << "alpha " << alpha << std::endl;
            w_entry.push_back(Eigen::Triplet<T>(offset[dim], offset_left_node[dim], 1.0 - alpha));
            w_entry.push_back(Eigen::Triplet<T>(offset[dim], offset_right_node[dim], alpha));
        }
        
        loop_id = dof_node_location[i];
    }
    // last segment
    if (loop_id != indices.size() - 1 && !closed)
    {
        //last node is not a crossing node
        Offset offset = offset_map[indices.back()];
        // std::cout << "node " <<  indices.back() << " added all dof " << std::endl;
        for(int d = 0; d < dim + 1; d++)
        {
            w_entry.push_back(Eigen::Triplet<T>(offset[d], dof_cnt++, 1.0));
        }
    }
    for (int j = loop_id + 1; j < indices.size() - 1; j++)
    {
        // std::cout << "left node " << indices[loop_id] << " right node " << indices.back() << std::endl;

        Offset offset_left_node = offset_map[indices[loop_id]];
        Offset offset_right_node = offset_map[indices.back()];

        T ui = full_states[offset_left_node[dim]];
        T uj = full_states[offset_right_node[dim]];
        int current_global_idx = indices[j];
        // std::cout << "current node " << current_global_idx << std::endl;
        //push Lagrangian DoF first
        Offset offset = offset_map[current_global_idx];
        for(int d = 0; d < dim; d++)
        {
            w_entry.push_back(Eigen::Triplet<T>(offset[d], dof_cnt++, 1.0));
        }
        // compute Eulerian weight
        T u = full_states[offset[dim]];
        T alpha = (u - ui) / (uj - ui);
        
        if (closed) alpha *= -1;
        
        // std::cout << "alpha " << alpha << std::endl;
        w_entry.push_back(Eigen::Triplet<T>(offset[dim], offset_left_node[dim], 1.0 - alpha));
        w_entry.push_back(Eigen::Triplet<T>(offset[dim], offset_right_node[dim], alpha));
    }
}
template class Rod<double, 3>;
template class Rod<double, 2>; 