#include "../include/FEMSolver.h"

void FEMSolver::addPBCPairInX()
{
    TV min_corner, max_corner;
    computeBoundingBox(min_corner, max_corner);
    T dx = max_corner[0] - min_corner[0];
    std::vector<int> left_nodes, right_nodes;
    for (int i = 0; i < num_nodes; i++)
    {
        TV xi = undeformed.segment<2>(i * 2);
        if (std::abs(xi[0] - min_corner[0]) < 1e-6)
            left_nodes.push_back(i);
        if (std::abs(xi[0] - max_corner[0]) < 1e-6)
            right_nodes.push_back(i);
    }
    std::sort(left_nodes.begin(), left_nodes.end(), [&](const int a, const int b){
        TV xa = undeformed.segment<2>(a * 2);
        TV xb = undeformed.segment<2>(b * 2);
        return xa[1] < xb[1];
    });
    std::sort(right_nodes.begin(), right_nodes.end(), [&](const int a, const int b){
        TV xa = undeformed.segment<2>(a * 2);
        TV xb = undeformed.segment<2>(b * 2);
        return xa[1] < xb[1];
    });

    if (left_nodes.size() != right_nodes.size())
    {
        std::cout << left_nodes.size() << " " << right_nodes.size() << std::endl;
        // std::exit(0);
    }

    pbc_pairs = {std::vector<IV>(), std::vector<IV>()};
    // pbc_pairs.resize(2, std::vector<IV>());    
    
    for (int i = 0; i < left_nodes.size(); i++)
    // for (int i = 0; i < 2; i++)
    {
        TV x_left = undeformed.segment<2>(left_nodes[i] * 2);
        TV x_right = undeformed.segment<2>(right_nodes[i] * 2);
        // std::cout << "pairs in x " <<  left_nodes[i] << " " << right_nodes[i] << std::endl;
        // std::cout << (x_right - x_left).normalized().dot(TV(1, 0)) << std::endl;
        // std::cout << x_right.transpose() << " " << x_left.transpose() << std::endl;
        pbc_pairs[0].push_back(IV(left_nodes[i], right_nodes[i]));
    }
    
    std::cout << "#pbc pairs " << pbc_pairs[0].size() + pbc_pairs[1].size() << std::endl;
}

void FEMSolver::getPBCPairs3D(std::vector<std::pair<TV3, TV3>>& pairs)
{
    pairs.resize(0);
    for (int dir = 0; dir < 2; dir++)
        for (IV& pbc_pair : pbc_pairs[dir])
        {
            int idx0 = pbc_pair[0], idx1 = pbc_pair[1];
            TV Xi = deformed.segment<2>(idx0 * 2);
            TV Xj = deformed.segment<2>(idx1 * 2);
            // std::cout << idx0 << " " << idx1 << " " << Xi.transpose() << " " << Xj.transpose() << std::endl;
            pairs.push_back(std::make_pair(TV3(Xi[0], Xi[1], 0.0), TV3(Xj[0], Xj[1], 0.0)));
        }
}

void FEMSolver::reorderPBCPairs()
{
    std::vector<std::vector<IV>> correct_pbc_pairs(2, std::vector<IV>());
    for (int dir = 0; dir < 2; dir++)
        for (IV& pbc_pair : pbc_pairs[dir])
        {
            // std::cout << pbc_pair.transpose() << std::endl;
            int idx0 = pbc_pair[0], idx1 = pbc_pair[1];
            TV Xi = undeformed.segment<2>(idx0 * 2);
            TV Xj = undeformed.segment<2>(idx1 * 2);
            
            T dot_t1 = (Xj - Xi).normalized().dot(t1.normalized());
            T dot_t2 = (Xj - Xi).normalized().dot(t2.normalized());
            // std::cout << (Xj - Xi).normalized().transpose() << std::endl;
            // std::cout << t1.transpose() << std::endl;
            // std::cout << t2.transpose() << std::endl;
            // std::cout << t1.normalized().transpose() << std::endl;
            // std::cout << t2.normalized().transpose() << std::endl;
            // std::cout << "dot t1 " << dot_t1 << " dot t2 " << dot_t1 << std::endl;
            // std::getchar();
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
    std::cout << "#pbc pairs " << pbc_pairs[0].size() + pbc_pairs[1].size() << std::endl;
}

void FEMSolver::addPBCEnergy(T& energy)
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
            if (add_pbc_strain)
                energy_pbc += 0.5 * pbc_strain_w * (dij - dij_target) * (dij - dij_target);
            cnt++;
            if (cnt == 1)
                continue;
            
            // distance term
            TV xi_ref = deformed.segment<2>(pbc_pairs[dir][0][0] * 2);
            TV xj_ref = deformed.segment<2>(pbc_pairs[dir][0][1] * 2);

            TV pair_dis_vec = xj - xi - (xj_ref - xi_ref);
            if (pair_dis_vec.norm() < 1e-6)
                continue;
            energy_pbc += 0.5 * pbc_w * pair_dis_vec.dot(pair_dis_vec);
        }
    };

    addPBCEnergyDirection(0);
    addPBCEnergyDirection(1);

    energy += energy_pbc;
}

void FEMSolver::addPBCForceEntries(VectorXT& residual)
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
            if (add_pbc_strain)
            {
                residual.segment<2>(idx0 * 2) += pbc_strain_w * strain_dir * (dij - dij_target);
                residual.segment<2>(idx1 * 2) -= pbc_strain_w * strain_dir * (dij - dij_target);
            }

            cnt++;
            if (cnt == 1)
                continue;
            TV xi_ref = deformed.segment<2>(pbc_pairs[dir][0][0] * 2);
            TV xj_ref = deformed.segment<2>(pbc_pairs[dir][0][1] * 2);

            TV pair_dis_vec = xj - xi - (xj_ref - xi_ref);
            if (pair_dis_vec.norm() < 1e-6)
                continue;
            // std::cout << (xj_ref - xi_ref).norm() << " " << (xj - xi).norm() << std::endl;
            // std::cout << pair_dis_vec.norm() << std::endl;
            // std::getchar();

            residual.segment<2>(idx0 * 2) += pbc_w * pair_dis_vec;
            residual.segment<2>(idx1 * 2) += -pbc_w * pair_dis_vec;

            residual.segment<2>(pbc_pairs[dir][0][0] * 2) += -pbc_w * pair_dis_vec;
            residual.segment<2>(pbc_pairs[dir][0][1] * 2) += pbc_w * pair_dis_vec;
        }
    };

    addPBCForceDirection(0);
    addPBCForceDirection(1);
}

void FEMSolver::addPBCHessianEntries(std::vector<Entry>& entries, bool project_PD)
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
            if (add_pbc_strain)
            {
                for(int i = 0; i < dim; i++)
                {
                    for(int j = 0; j < dim; j++)
                    {
                        entries.push_back(Entry(idx0 * 2 + i, idx0 * 2 + j, pbc_strain_w * Hessian(i, j)));
                        entries.push_back(Entry(idx0 * 2 + i, idx1 * 2 + j, -pbc_strain_w * Hessian(i, j)));
                        entries.push_back(Entry(idx1 * 2 + i, idx0 * 2 + j, -pbc_strain_w * Hessian(i, j)));
                        entries.push_back(Entry(idx1 * 2 + i, idx1 * 2 + j, pbc_strain_w * Hessian(i, j)));
                    }
                }
            }

            cnt++;
            if (cnt == 1)
                continue;

            TV xi_ref = deformed.segment<2>(pbc_pairs[dir][0][0] * 2);
            TV xj_ref = deformed.segment<2>(pbc_pairs[dir][0][1] * 2);
            TV pair_dis_vec = xj - xi - (xj_ref - xi_ref);
            if (pair_dis_vec.norm() < 1e-6)
                continue;
                
            std::vector<int> nodes = {idx0, idx1, pbc_pairs[dir][0][0], pbc_pairs[dir][0][1]};
            std::vector<T> sign_J = {-1, 1, 1, -1};
            std::vector<T> sign_F = {1, -1, -1, 1};

            for(int k = 0; k < 4; k++)
                for(int l = 0; l < 4; l++)
                    for(int i = 0; i < 2; i++)
                        entries.push_back(Entry(nodes[k]*2 + i, nodes[l] * 2 + i, -pbc_w *sign_F[k]*sign_J[l]));
        }
    };

    addPBCHessianDirection(0);
    addPBCHessianDirection(1);
}