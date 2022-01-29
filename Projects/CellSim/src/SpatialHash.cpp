#include "../include/SpatialHash.h"

void SpatialHash::build(T _h, const VectorXT& points)
{
    h = _h;
    hash_table.clear();
    int n_points = points.rows() / 3;
    for (int i = 0; i < n_points; i++) 
    {
        const TV& X = points.segment<3>(i * 3);
        IV cell = IV::Zero();
        for (int d = 0; d < dim; d++)
            cell(d) = (int)(std::floor(X(d) / h));
        if (hash_table.find(cell) != hash_table.end())
            hash_table[cell].push_back(i);
        else 
            hash_table[cell] = { i };
    }
}

void SpatialHash::getOneRingNeighbors(const TV& point, std::vector<int>& neighbors)
{
    neighbors.clear();
    IV cell = IV::Zero();
    for (int d = 0; d < dim; d++)
        cell(d) = (int)(std::floor(point(d) / h));
    IV local_min_index = cell.array() - 1;
    IV local_max_index = cell.array() + 2;
    // std::cout << cell.transpose() << " " << local_min_index.transpose() << " " << local_max_index.transpose() << std::endl;
    // std::getchar();
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
    // std::cout << neighbors.size() << std::endl;
}

void SpatialHash::test()
{
    std::cout << "Spatial Hash Test" << std::endl;

    IV grid_n(5, 5, 1);
    h = 0.05;

    VectorXT data_points(grid_n.prod() * 3);
    int cnt = 0;
    for (int i = 0; i < grid_n[0]; i++)
    {
        for (int j = 0; j < grid_n[1]; j++)
        {
            for (int k = 0; k < grid_n[2]; k++)
            {
                data_points.segment<3>(cnt * 3) = TV(i, j, k) * h - TV(0.1, 0.1, 0);
                std::cout << TV(i, j, k).transpose() * h << std::endl;
                cnt++;
            }
        }
    }
    
    build(h, data_points);

    std::vector<int> neighbors;
    TV test_point = TV(0.2, 0.07, 0.0) - TV(0.1, 0.1, 0);
    getOneRingNeighbors(test_point, neighbors);
    std::cout << neighbors.size() << std::endl;
    for (int idx : neighbors)
        std::cout << data_points.segment<3>(idx * 3).transpose() << std::endl;
}