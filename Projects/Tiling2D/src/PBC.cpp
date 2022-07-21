#include "../include/FEMSolver.h"

void FEMSolver::getPBCPairs3D(std::vector<std::pair<TV3, TV3>>& pairs)
{
    for (int dir = 0; dir < 2; dir++)
        for (IV& pbc_pair : pbc_pairs[dir])
        {
            int idx0 = pbc_pair[0], idx1 = pbc_pair[1];
            TV Xi = deformed.segment<2>(idx0 * 2);
            TV Xj = deformed.segment<2>(idx1 * 2);

            pairs.push_back(std::make_pair(TV3(Xi[0], Xi[1], 0.0), TV3(Xj[0], Xj[1], 0.0)));
        }
}

void FEMSolver::reorderPBCPairs()
{
    std::vector<std::vector<IV>> correct_pbc_pairs(2, std::vector<IV>());
    for (int dir = 0; dir < 2; dir++)
        for (IV& pbc_pair : pbc_pairs[dir])
        {
            int idx0 = pbc_pair[0], idx1 = pbc_pair[1];
            TV Xi = undeformed.segment<2>(idx0 * 2);
            TV Xj = undeformed.segment<2>(idx1 * 2);
            
            T dot_t1 = (Xj - Xi).normalized().dot(t1.normalized());
            T dot_t2 = (Xj - Xi).normalized().dot(t2.normalized());
            
            if (std::abs(std::abs(dot_t1) - 1) < 1e-6)
            {
                if (dot_t1 < 0)
                {
                    pbc_pair[1] = idx0;
                    pbc_pair[0] = idx1;
                    correct_pbc_pairs[0].push_back(pbc_pair);
                }
                else
                {
                    correct_pbc_pairs[0].push_back(pbc_pair);
                }
            }
            else if (std::abs(std::abs(dot_t2) - 1) < 1e-6)
            {
                
                if (dot_t2 < 0)
                {
                    pbc_pair[1] = idx0;
                    pbc_pair[0] = idx1;
                    correct_pbc_pairs[1].push_back(pbc_pair);
                }
                else
                {
                    correct_pbc_pairs[1].push_back(pbc_pair);
                }
            }

        }
    pbc_pairs = correct_pbc_pairs;
}

void FEMSolver::addPBCEnergy(T w, T& energy)
{
    T energy_pbc = 0.0;
    TV strain_dir = TV(std::cos(strain_theta), std::sin(strain_theta));

    auto addPBCEnergyDirection = [&](int dir)
    {
        int cnt = 0;
        for (auto pbc_pair : pbc_pairs[dir])
        {
            // strain term
            int idx0 = pbc_pair[0], idx1 = pbc_pair[1];
            TV Xi = undeformed.segment<2>(idx0 * 2);
            TV Xj = undeformed.segment<2>(idx1 * 2);
            TV xi = deformed.segment<2>(idx0 * 2);
            TV xj = deformed.segment<2>(idx1 * 2);

            T Dij = (Xj - Xi).dot(strain_dir);
            T dij_target = Dij * uniaxial_strain;
            T dij = (xj - xi).dot(strain_dir);

            energy_pbc += 0.5 * w * (dij - dij_target) * (dij - dij_target);
            cnt++;
            if (cnt == 1)
                continue;
            
            // distance term
            TV xi_ref = deformed.segment<2>(pbc_pairs[dir][0][0] * 2);
            TV xj_ref = deformed.segment<2>(pbc_pairs[dir][0][1] * 2);

            TV pair_dis_vec = xj - xi - (xj_ref - xi_ref);
            energy_pbc += 0.5 * w * pair_dis_vec.dot(pair_dis_vec);
        }
    };

    addPBCEnergyDirection(0);
    addPBCEnergyDirection(1);

    energy += energy_pbc;
}

void FEMSolver::addPBCForceEntries(T w, VectorXT& residual)
{
    
    TV strain_dir = TV(std::cos(strain_theta), std::sin(strain_theta));

    auto addPBCForceDirection = [&](int dir)
    {
        int cnt = 0;
        for (auto pbc_pair : pbc_pairs[dir])
        {
            int idx0 = pbc_pair[0], idx1 = pbc_pair[1];
            
            TV Xi = undeformed.segment<2>(idx0 * 2);
            TV Xj = undeformed.segment<2>(idx1 * 2);
            TV xi = deformed.segment<2>(idx0 * 2);
            TV xj = deformed.segment<2>(idx1 * 2);

            T Dij = (Xj - Xi).dot(strain_dir);
            T dij_target = Dij * uniaxial_strain;
            T dij = (xj - xi).dot(strain_dir);

            residual.segment<2>(idx0 * 2) += w * strain_dir * (dij - dij_target);
            residual.segment<2>(idx1 * 2) -= w * strain_dir * (dij - dij_target);

            cnt++;
            if (cnt == 1)
                continue;
            TV xi_ref = deformed.segment<2>(pbc_pairs[dir][0][0] * 2);
            TV xj_ref = deformed.segment<2>(pbc_pairs[dir][0][1] * 2);

            TV pair_dis_vec = xj - xi - (xj_ref - xi_ref);

            residual.segment<2>(idx0 * 2) += w *pair_dis_vec;
            residual.segment<2>(idx1 * 2) += -w *pair_dis_vec;

            residual.segment<2>(pbc_pairs[dir][0][0] * 2) += -w * pair_dis_vec;
            residual.segment<2>(pbc_pairs[dir][0][1] * 2) += w * pair_dis_vec;
        }
    };

    addPBCForceDirection(0);
    addPBCForceDirection(1);
}

void FEMSolver::addPBCHessianEntries(T w, std::vector<Entry>& entries, bool project_PD)
{
    TV strain_dir = TV(std::cos(strain_theta), std::sin(strain_theta));

    auto addPBCHessianDirection = [&](int dir)
    {
        int cnt = 0;
        for (auto pbc_pair : pbc_pairs[dir])
        {
            int idx0 = pbc_pair[0], idx1 = pbc_pair[1];
            TV Xi = undeformed.segment<2>(idx0 * 2);
            TV Xj = undeformed.segment<2>(idx1 * 2);
            TV xi = deformed.segment<2>(idx0 * 2);
            TV xj = deformed.segment<2>(idx1 * 2);

            T Dij = (Xj - Xi).dot(strain_dir);
            T dij_target = Dij * uniaxial_strain;
            T dij = (xj - xi).dot(strain_dir);

            TM Hessian = strain_dir * strain_dir.transpose();
            for(int i = 0; i < dim; i++)
            {
                for(int j = 0; j < dim; j++)
                {
                    entries.push_back(Entry(idx0 * 2 + i, idx0 * 2 + j, w * Hessian(i, j)));
                    entries.push_back(Entry(idx0 * 2 + i, idx1 * 2 + j, -w * Hessian(i, j)));
                    entries.push_back(Entry(idx1 * 2 + i, idx0 * 2 + j, -w * Hessian(i, j)));
                    entries.push_back(Entry(idx1 * 2 + i, idx1 * 2 + j, w * Hessian(i, j)));
                }
            }

            cnt++;
            if (cnt == 1)
                continue;

            std::vector<int> nodes = {idx0, idx1, pbc_pairs[dir][0][0], pbc_pairs[dir][0][1]};
            std::vector<T> sign_J = {-1, 1, 1, -1};
            std::vector<T> sign_F = {1, -1, -1, 1};

            for(int k = 0; k < 4; k++)
                for(int l = 0; l < 4; l++)
                    for(int i = 0; i < 2; i++)
                        entries.push_back(Entry(nodes[k]*2 + i, nodes[l] * 2 + i, -w *sign_F[k]*sign_J[l]));
        }
    };

    addPBCHessianDirection(0);
    addPBCHessianDirection(1);
}