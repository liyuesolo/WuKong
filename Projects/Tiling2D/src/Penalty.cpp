#include "../include/FEMSolver.h"

void FEMSolver::addBCPenaltyEnergy(T w, T& energy)
{
    T penalty_energy = 0.0;
    iterateBCPenaltyPairs([&](int offset, T target)
    {
        penalty_energy += w * 0.5 * std::pow(deformed[offset] - target, 2);
    });
    energy += penalty_energy;
}

void FEMSolver::addBCPenaltyForceEntries(T w, VectorXT& residual)
{
    iterateBCPenaltyPairs([&](int offset, T target)
    {
        residual[offset] -= w * (deformed[offset] - target);
    });
}

void FEMSolver::addBCPenaltyHessianEntries(T w, std::vector<Entry>& entries)
{
    iterateBCPenaltyPairs([&](int offset, T target)
    {
        entries.push_back(Entry(offset, offset, w));
    });
}