#include "../include/SpatialHash.h"

template <int dim>
void SpatialHash<dim>::build(T _h, const VectorXT& points)
{
    h = _h;
    hash_table.clear();
    int n_points = points.rows() / dim;
    for (int i = 0; i < n_points; i++) 
    {
        const TV& X = points.segment<dim>(i * dim);
        IV cell = IV::Zero();
        for (int d = 0; d < dim; d++)
            cell(d) = (int)(std::floor(X(d) / h));
        if (hash_table.find(cell) != hash_table.end())
            hash_table[cell].push_back(i);
        else 
            hash_table[cell] = { i };
    }
}

template <int dim>
void SpatialHash<dim>::getOneRingNeighbors(const TV& point, std::vector<int>& neighbors)
{
    neighbors.clear();
    IV cell = IV::Zero();
    for (int d = 0; d < dim; d++)
        cell(d) = (int)(std::floor(point(d) / h));
    IV local_min_index = cell.array() - 1;
    IV local_max_index = cell.array() + 2;
    // std::cout << cell.transpose() << " " << local_min_index.transpose() << " " << local_max_index.transpose() << std::endl;
    // std::getchar();
    if constexpr (dim == 3)
    {
        for (int i = local_min_index[0]; i < local_max_index[0]; i++)
        {
            for (int j = local_min_index[1]; j < local_max_index[1]; j++)
            {
                for (int k = local_min_index[2]; k < local_max_index[2]; k++)
                {
                    IV cell(i, j, k);
                    if (hash_table.find(cell) != hash_table.end())
                    {
                        std::vector<int> neighbor = hash_table[cell];
                        neighbors.insert(neighbors.end(), neighbor.begin(), neighbor.end());
                    }
                }
                
            }   
        }
    }
    else if constexpr (dim == 2)
    {
        for (int i = local_min_index[0]; i < local_max_index[0]; i++)
        {
            for (int j = local_min_index[1]; j < local_max_index[1]; j++)
            {
                IV cell(i, j);
                if (hash_table.find(cell) != hash_table.end())
                {
                    std::vector<int> neighbor = hash_table[cell];
                    neighbors.insert(neighbors.end(), neighbor.begin(), neighbor.end());
                }
            }   
        }
    }
    // std::cout << neighbors.size() << std::endl;
}

template class SpatialHash<2>;