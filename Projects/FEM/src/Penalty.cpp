#include "../include/FEMSolver.h"

template <int dim>
void FEMSolver<dim>::addBCPenaltyEnergy(T& energy)
{
    T penalty_energy = 0.0;
    iterateBCPenaltyPairs([&](int offset, T target)
    {
        penalty_energy += penalty_weight * 0.5 * std::pow(deformed[offset] - target, 2);
    });
    energy += penalty_energy;
}

template <int dim>
void FEMSolver<dim>::addBCPenaltyForceEntries(VectorXT& residual)
{
    iterateBCPenaltyPairs([&](int offset, T target)
    {
        residual[offset] -= penalty_weight * (deformed[offset] - target);
    });
}

template <int dim>
void FEMSolver<dim>::addBCPenaltyHessianEntries(std::vector<Entry>& entries)
{
    iterateBCPenaltyPairs([&](int offset, T target)
    {
        entries.push_back(Entry(offset, offset, penalty_weight));
    });
}

template class FEMSolver<2>;
template class FEMSolver<3>;