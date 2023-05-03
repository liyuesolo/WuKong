#include "../include/CellSim.h"
#include "../autodiff/Repulsion.h"

template <int dim>
void CellSim<dim>::addRepulsionEnergy(T& energy)
{
    // cell_hash.build(2.0 * collision_dhat, deformed);
    vv_flag.resize(num_nodes, num_nodes);
    vv_flag.setConstant(0);

    iterateCellSerial([&](int i)
    {
        TV xi = deformed.segment<dim>(i * dim);
        std::vector<int> neighbors;
        cell_hash.getOneRingNeighbors(xi, neighbors);
        
        for (int j : neighbors)
        {
            // if (i == j || vv_flag(i, j) || isYolkParticle(j))
            if (i == j || vv_flag(i, j))
                continue;
            vv_flag(i, j) = 1; vv_flag(j, i) = 1;
            TV xj = deformed.segment<dim>(j * dim);
            T d = (xi - xj).norm();
            
            if (d > collision_dhat)
                continue;
            T e;
            if constexpr (dim == 2)
                computeRepulsion2DCubicEnergy(xi, xj, collision_dhat, e);
            else
                computeRepulsion3DCubicEnergy(xi, xj, collision_dhat, e);
            energy += w_rep * e;
            // std::cout << e << std::endl;
            // std::getchar();
        }
    });
}

template <int dim>
void CellSim<dim>::addRepulsionForceEntries(VectorXT& residual)
{
    // cell_hash.build(2.0 * collision_dhat, deformed);
    vv_flag.resize(num_nodes, num_nodes);
    vv_flag.setConstant(0);

    iterateCellSerial([&](int i)
    {
        TV xi = deformed.segment<dim>(i * dim);
        std::vector<int> neighbors;
        cell_hash.getOneRingNeighbors(xi, neighbors);
        // std::cout << neighbors.size() << std::endl;
        for (int j : neighbors)
        {
            // if (i == j || vv_flag(i, j) || isYolkParticle(j))
            if (i == j || vv_flag(i, j))
                continue;
            vv_flag(i, j) = 1; vv_flag(j, i) = 1;
            
            TV xj = deformed.segment<dim>(j * dim);
            T d = (xi - xj).norm();
            
            // std::cout << i << " " << j << " " << d << std::endl;
            if (d > collision_dhat)
                continue;
            if constexpr (dim == 2)
            {
                Vector<T, 4> dedx;
                computeRepulsion2DCubicEnergyGradient(xi, xj, collision_dhat, dedx);
                
                addForceEntry<4>(residual, {i, j}, -w_rep * dedx);
            }
            else
            {
                Vector<T, 6> dedx;
                computeRepulsion3DCubicEnergyGradient(xi, xj, collision_dhat, dedx);
                addForceEntry<6>(residual, {i, j}, -w_rep * dedx);

                
            }
        }
        
    });
}

template <int dim>
void CellSim<dim>::addRepulsionHessianEntries(std::vector<Entry>& entries, bool projectPD)
{
    // cell_hash.build(2.0 * collision_dhat, deformed);
    vv_flag.resize(num_nodes, num_nodes);
    vv_flag.setConstant(0);
    iterateCellSerial([&](int i)
    {
        
        TV xi = deformed.segment<dim>(i * dim);
        std::vector<int> neighbors;
        cell_hash.getOneRingNeighbors(xi, neighbors);
        
        for (int j : neighbors)
        {
            // if (i == j || vv_flag(i, j) || isYolkParticle(j))
            if (i == j || vv_flag(i, j))
                continue;
            vv_flag(i, j) = 1; vv_flag(j, i) = 1;
            TV xj = deformed.segment<dim>(j * dim);
            T d = (xi - xj).norm();
            
            if (d > collision_dhat)
                continue;
            if constexpr (dim == 2)
            {
                Matrix<T, 4, 4> d2edx2;
                computeRepulsion2DCubicEnergyHessian(xi, xj, collision_dhat, d2edx2);
                addHessianEntry<4>(entries, {i, j}, w_rep * d2edx2);
            }
            else
            {
                Matrix<T, 6, 6> d2edx2;
                computeRepulsion3DCubicEnergyHessian(xi, xj, collision_dhat, d2edx2);
                addHessianEntry<6>(entries, {i, j}, w_rep * d2edx2);
            }
        }
    });
}


template class CellSim<2>;
template class CellSim<3>;