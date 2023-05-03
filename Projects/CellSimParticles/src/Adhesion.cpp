#include "../include/CellSim.h"
#include "../autodiff/Repulsion.h"
#include "../autodiff/EdgeEnergy.h"


template <int dim>
void CellSim<dim>::computeInitialNeighbors()
{
    T distance = 2.0 * radius;
    cell_hash.build(2.0 * distance, undeformed);
    MatrixXi visit(num_nodes, num_nodes); visit.setZero();
    iterateCellSerial([&](int i)
    {
        TV xi = undeformed.segment<dim>(i * dim);
        std::vector<int> neighbors;
        cell_hash.getOneRingNeighbors(xi, neighbors);
        for (int j : neighbors)
        {
            if (i == j || visit(i, j) || isYolkParticle(j))
                continue;
            visit(i, j) = 1; visit(j, i) = 1;
            TV xj = undeformed.segment<dim>(j * dim);
            T d = (xi - xj).norm();
            if (d < distance)
                adhesion_edges.push_back(Edge(i, j));
        }
    });
    std::cout << "# initial adhesion pairs " << adhesion_edges.size() << std::endl;
}

template <int dim>
void CellSim<dim>::addAdhesionEnergy(T& energy)
{
    
    for (auto edge : adhesion_edges)
    {
        TV xi = deformed.segment<dim>(edge[0] * dim);
        TV xj = deformed.segment<dim>(edge[1] * dim);
        TV Xi = undeformed.segment<dim>(edge[0] * dim);
        TV Xj = undeformed.segment<dim>(edge[1] * dim);
        T d = (xi - xj).norm();
        if (d < adhesion_dhat)
            continue;
        T e;
        if constexpr (dim == 2)
            computeRepulsion2DCubicEnergy(xi, xj, adhesion_dhat, e);
        else
        {
            // computeRepulsion3DCubicEnergy(xi, xj, adhesion_dhat, e);
            // energy += w_adh * -e;
            T ei;
            computeEdgeSpringEnergy<dim>(xi, xj, (Xi-Xj).norm(), ei);
            energy += w_adh * ei;

        }
    }
}

template <int dim>
void CellSim<dim>::addAdhesionForceEntries(VectorXT& residual)
{
    for (auto edge : adhesion_edges)
    {
        TV xi = deformed.segment<dim>(edge[0] * dim);
        TV xj = deformed.segment<dim>(edge[1] * dim);
        TV Xi = undeformed.segment<dim>(edge[0] * dim);
        TV Xj = undeformed.segment<dim>(edge[1] * dim);
        T d = (xi - xj).norm();
        if (d < adhesion_dhat)
            continue;
        if constexpr (dim == 2)
        {
            Vector<T, 4> dedx;
            computeRepulsion2DCubicEnergyGradient(xi, xj, adhesion_dhat, dedx);
            
            addForceEntry<4>(residual, {edge[0], edge[1]}, w_adh * dedx);
        }
        else
        {
            // Vector<T, 6> dedx;
            // computeRepulsion3DCubicEnergyGradient(xi, xj, adhesion_dhat, dedx);
            // addForceEntry<6>(residual, {edge[0], edge[1]}, w_adh * dedx);
            Vector<T, dim * 2> dedx;
            computeEdgeSpringEnergyGradient<dim>(xi, xj, (Xi-Xj).norm(), dedx);
            addForceEntry<dim * 2>(residual, {edge[0], edge[1]}, -w_adh * dedx);
        }
    }
}

template <int dim>
void CellSim<dim>::addAdhesionHessianEntries(std::vector<Entry>& entries, bool projectPD)
{
    for (auto edge : adhesion_edges)
    {
        TV xi = deformed.segment<dim>(edge[0] * dim);
        TV xj = deformed.segment<dim>(edge[1] * dim);
        TV Xi = undeformed.segment<dim>(edge[0] * dim);
        TV Xj = undeformed.segment<dim>(edge[1] * dim);
        T d = (xi - xj).norm();
        if (d < adhesion_dhat)
            continue;
        if constexpr (dim == 2)
        {
            Matrix<T, 4, 4> d2edx2;
            computeRepulsion2DCubicEnergyHessian(xi, xj, adhesion_dhat, d2edx2);
            addHessianEntry<4>(entries, {edge[0], edge[1]}, -w_adh * d2edx2);
        }
        else
        {
            // Matrix<T, 6, 6> d2edx2;
            // computeRepulsion3DCubicEnergyHessian(xi, xj, adhesion_dhat, d2edx2);
            // addHessianEntry<6>(entries, {edge[0], edge[1]}, -w_adh * d2edx2);
            Matrix<T, dim * 2, dim * 2> hessian;
            computeEdgeSpringEnergyHessian<dim>(xi, xj, (Xi-Xj).norm(), hessian);
            addHessianEntry<dim * 2>(entries, {edge[0], edge[1]}, w_adh * hessian);
        }
    }
}


template class CellSim<2>;
template class CellSim<3>;