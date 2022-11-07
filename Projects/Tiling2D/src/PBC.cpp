#include "../include/FEMSolver.h"
#include "../include/autodiff/PBCEnergy.h"
#include "../include/autodiff/FEMEnergy.h"

bool FEMSolver::addPBCPairsXY()
{
    std::cout << pbc_translation_file << std::endl;
    std::ifstream in(pbc_translation_file);
    TV t1, t2;
    in >> t1[0] >> t1[1] >> t2[0] >> t2[1];
    in.close();


    // rotate the structure to have one translation vector align with X
    // then gether the pairs in Y

    T alpha = angleToXaxis(t1);
    rotate(alpha);
    std::vector<int> dir0_side0, dir0_side1, dir1_side0, dir1_side1;
    getPBCPairsAxisDirection(dir0_side0, dir0_side1, 1);
    rotate(-alpha);
    alpha = angleToXaxis(t2);
    rotate(alpha);
    getPBCPairsAxisDirection(dir1_side0, dir1_side1, 1);
    // rotate(-alpha);

    bool same_num_nodes_dir0 = dir0_side0.size() == dir0_side1.size();
    bool same_num_nodes_dir1 = dir1_side0.size() == dir1_side1.size();
    std::cout << same_num_nodes_dir0 << " " << same_num_nodes_dir1 << std::endl;
    
    pbc_pairs = {std::vector<IV>(), std::vector<IV>()};
    // std::cout <<  dir0_side0.size() << " " << dir0_side1.size()
    //     << " " << dir1_side0.size() << " " << dir1_side1.size() << std::endl;
    
    for (int i = 0; i < std::min(dir0_side0.size(), dir0_side1.size()); i++)
    {
        pbc_pairs[0].push_back(IV(dir0_side0[i], dir0_side1[i]));
    }
    for (int i = 0; i < std::min(dir1_side0.size(), dir1_side1.size()); i++)
    {
        pbc_pairs[1].push_back(IV(dir1_side0[i], dir1_side1[i]));
    }
    return same_num_nodes_dir0 && same_num_nodes_dir1;
}

void FEMSolver::getPBCPairsAxisDirection(std::vector<int>& side0, 
    std::vector<int>& side1, int direction)
{
    T thres_hold = 1e-4;
    int ortho = !direction;
    // std::cout << "dir " << direction << " " << ortho << std::endl;
    side0.clear(); side1.clear();
    TV min_corner, max_corner;
    computeBoundingBox(min_corner, max_corner);
    
    for (int i = 0; i < num_nodes; i++)
    {
        TV xi = undeformed.segment<2>(i * 2);
        if (std::abs(xi[direction] - min_corner[direction]) < thres_hold)
            side0.push_back(i);
        if (std::abs(xi[direction] - max_corner[direction]) < thres_hold)
            side1.push_back(i);
    }
    
    std::sort(side0.begin(), side0.end(), [&](const int a, const int b){
        TV xa = undeformed.segment<2>(a * 2);
        TV xb = undeformed.segment<2>(b * 2);
        return xa[ortho] < xb[ortho];
    });
    std::sort(side1.begin(), side1.end(), [&](const int a, const int b){
        TV xa = undeformed.segment<2>(a * 2);
        TV xb = undeformed.segment<2>(b * 2);
        return xa[ortho] < xb[ortho];
    });
}

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
            std::cout << idx0 << " " << idx1 << " " << Xi.transpose() << " " << Xj.transpose() << std::endl;
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
    TV ortho_dir = TV(-std::sin(strain_theta), std::cos(strain_theta));
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
            
            if (add_pbc_strain && !prescribe_strain_tensor)
                energy_pbc += 0.5 * pbc_strain_w * (dij - dij_target) * (dij - dij_target);
            if (add_pbc_strain && !prescribe_strain_tensor && biaxial)
            {
                Dij = (Xj - Xi).dot(ortho_dir);
                dij_target = Dij * uniaxial_strain_ortho;
                dij = (xj - xi).dot(ortho_dir);
                energy_pbc += 0.5 * pbc_strain_w * (dij - dij_target) * (dij - dij_target);
            }
            cnt++;
            if (cnt == 1)
                continue;
            
            // distance term
            TV xi_ref = deformed.segment<2>(pbc_pairs[dir][0][0] * 2);
            TV xj_ref = deformed.segment<2>(pbc_pairs[dir][0][1] * 2);

            TV pair_dis_vec = xj - xi - (xj_ref - xi_ref);
            // if (pair_dis_vec.norm() < 1e-6)
            //     continue;
            // T e_pbc;
            // compute2DPBCEnergy(pbc_w, xj, xi, xj_ref, xi_ref, e_pbc);
            // energy_pbc += e_pbc;
            energy_pbc += 0.5 * pbc_w * pair_dis_vec.dot(pair_dis_vec);
        }
    };

    addPBCEnergyDirection(0);
    addPBCEnergyDirection(1);

    if (add_pbc_strain && prescribe_strain_tensor)
    {
        Matrix<T, 4, 2> x, X;
        IV4 bd_indices;
        getMarcoBoundaryData(x, X, bd_indices);
        T strain_matching_term;
        computeStrainMatchingEnergy(pbc_strain_w, target_strain, x, X, 
            strain_matching_term);
        energy_pbc += strain_matching_term;
    }
    energy += energy_pbc;
}

void FEMSolver::addPBCForceEntries(VectorXT& residual)
{
    
    TV strain_dir = TV(std::cos(strain_theta), std::sin(strain_theta));
    TV ortho_dir = TV(-std::sin(strain_theta), std::cos(strain_theta));

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
            if (add_pbc_strain && !prescribe_strain_tensor)
            {
                residual.segment<2>(idx0 * 2) += pbc_strain_w * strain_dir * (dij - dij_target);
                residual.segment<2>(idx1 * 2) -= pbc_strain_w * strain_dir * (dij - dij_target);
            }
            if (add_pbc_strain && !prescribe_strain_tensor && biaxial)
            {
                Dij = (Xj - Xi).dot(ortho_dir);
                dij_target = Dij * uniaxial_strain_ortho;
                dij = (xj - xi).dot(ortho_dir);
                residual.segment<2>(idx0 * 2) += pbc_strain_w * ortho_dir * (dij - dij_target);
                residual.segment<2>(idx1 * 2) -= pbc_strain_w * ortho_dir * (dij - dij_target);
            }
            cnt++;
            if (cnt == 1)
                continue;
            TV xi_ref = deformed.segment<2>(pbc_pairs[dir][0][0] * 2);
            TV xj_ref = deformed.segment<2>(pbc_pairs[dir][0][1] * 2);

            TV pair_dis_vec = xj - xi - (xj_ref - xi_ref);
            // if (pair_dis_vec.norm() < 1e-6)
            //     continue;`
            // std::cout << (xj_ref - xi_ref).norm() << " " << (xj - xi).norm() << std::endl;
            // std::cout << pair_dis_vec.norm() << std::endl;
            // std::getchar();
            // Vector<T, 8> dedx;
            // compute2DPBCEnergyGradient(pbc_w, xj, xi, xj_ref, xi_ref, dedx);
            // residual.segment<2>(idx0 * 2) += -dedx.segment<2>(2);
            // residual.segment<2>(idx1 * 2) += -dedx.segment<2>(0);
            // residual.segment<2>(pbc_pairs[dir][0][0] * 2) += -dedx.segment<2>(6);
            // residual.segment<2>(pbc_pairs[dir][0][1] * 2) += -dedx.segment<2>(4);

            residual.segment<2>(idx0 * 2) += pbc_w * pair_dis_vec;
            residual.segment<2>(idx1 * 2) += -pbc_w * pair_dis_vec;

            residual.segment<2>(pbc_pairs[dir][0][0] * 2) += -pbc_w * pair_dis_vec;
            residual.segment<2>(pbc_pairs[dir][0][1] * 2) += pbc_w * pair_dis_vec;
        }
    };

    addPBCForceDirection(0);
    addPBCForceDirection(1);
    if (add_pbc_strain && prescribe_strain_tensor)
    {
        Matrix<T, 4, 2> x, X;
        IV4 bd_indices;
        getMarcoBoundaryData(x, X, bd_indices);
        Vector<T, 8> dedx;
        computeStrainMatchingEnergyGradient(pbc_strain_w, target_strain, x, X, 
            dedx);
        addForceEntry<8>(residual, bd_indices, -dedx);
    }
}

void FEMSolver::addPBCHessianEntries(std::vector<Entry>& entries, bool project_PD)
{
    TV strain_dir = TV(std::cos(strain_theta), std::sin(strain_theta));
    TV ortho_dir = TV(-std::sin(strain_theta), std::cos(strain_theta));
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
            if (add_pbc_strain && !prescribe_strain_tensor)
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
            if (add_pbc_strain && !prescribe_strain_tensor && biaxial)
            {
                Hessian = ortho_dir * ortho_dir.transpose();
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
            // if (pair_dis_vec.norm() < 1e-6)
            //     continue;
                
            // Matrix<T, 8, 8> d2edx2;
            // compute2DPBCEnergyHessian(pbc_w, xj, xi, xj_ref, xi_ref, d2edx2);
            // std::vector<int> nodes = {idx1, idx0, pbc_pairs[dir][0][1], pbc_pairs[dir][0][0]};
            // for(int k = 0; k < 4; k++)
            //     for(int l = 0; l < 4; l++)
            //         for(int i = 0; i < 2; i++)
            //             for(int j = 0; j < 2; j++)
            //                 entries.push_back(Entry(nodes[k]*2 + i, nodes[l] * 2 + j, d2edx2(k * 2 + i, l * 2 + j)));

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
    if (add_pbc_strain && prescribe_strain_tensor)
    {
        Matrix<T, 4, 2> x, X;
        IV4 bd_indices;
        getMarcoBoundaryData(x, X, bd_indices);
        Matrix<T, 8, 8> d2edx2;
        computeStrainMatchingEnergyHessian(pbc_strain_w, target_strain, x, X, 
            d2edx2);
        addHessianEntry<8>(entries, bd_indices, d2edx2);
    }
}

void FEMSolver::computeMarcoBoundaryIndices()
{
    pbc_corners << pbc_pairs[0][0][0], 
        pbc_pairs[0][pbc_pairs[0].size() - 1][0],
        pbc_pairs[0][pbc_pairs[0].size() - 1][1],
        pbc_pairs[0][0][1];
    
    // std::cout << pbc_corners << std::endl;
}

void FEMSolver::getMarcoBoundaryData(Matrix<T, 4, 2>& x, Matrix<T, 4, 2>& X, IV4& bd_indices)
{
    TV xi = deformed.segment<2>(pbc_pairs[0][0][0] * 2);
    TV xj = deformed.segment<2>(pbc_pairs[0][0][1] * 2);
    TV xk = deformed.segment<2>(pbc_pairs[1][0][0] * 2);
    TV xl = deformed.segment<2>(pbc_pairs[1][0][1] * 2);
    x.row(0) = xi; x.row(1) = xj; x.row(2) = xk; x.row(3) = xl;

    TV Xi = undeformed.segment<2>(pbc_pairs[0][0][0] * 2);
    TV Xj = undeformed.segment<2>(pbc_pairs[0][0][1] * 2);
    TV Xk = undeformed.segment<2>(pbc_pairs[1][0][0] * 2);
    TV Xl = undeformed.segment<2>(pbc_pairs[1][0][1] * 2);
    X.row(0) = Xi; X.row(1) = Xj; X.row(2) = Xk; X.row(3) = Xl;

    bd_indices << pbc_pairs[0][0][0], pbc_pairs[0][0][1], pbc_pairs[1][0][0], pbc_pairs[1][0][1];
}

void FEMSolver::computeHomogenizationDataCauchy(TM& cauchy_stress, TM& cauchy_strain, T& energy_density)
{
    TV xi = deformed.segment<2>(pbc_pairs[0][0][0] * 2);
    TV xj = deformed.segment<2>(pbc_pairs[0][0][1] * 2);

    TV xk = deformed.segment<2>(pbc_pairs[1][0][0] * 2);
    TV xl = deformed.segment<2>(pbc_pairs[1][0][1] * 2);


    TV Xi = undeformed.segment<2>(pbc_pairs[0][0][0] * 2);
    TV Xj = undeformed.segment<2>(pbc_pairs[0][0][1] * 2);
    TV Xk = undeformed.segment<2>(pbc_pairs[1][0][0] * 2);
    TV Xl = undeformed.segment<2>(pbc_pairs[1][0][1] * 2);

    VectorXT inner_force(num_nodes * 2);
    inner_force.setZero(); 
    addPBCForceEntries(inner_force);
    
    // computeResidual(u, inner_force);
    TV f0 = TV::Zero(), f1 = TV::Zero();
    T l0 = (xj - xi).norm(), l1 = (xl - xk).norm();
    for (auto pbc_pair : pbc_pairs[0])
    {
        f0 += inner_force.segment<2>(pbc_pair[0] * 2) / l1 / thickness;
    }
    for (auto pbc_pair : pbc_pairs[1])
    {
        f1 += inner_force.segment<2>(pbc_pair[1] * 2) / l0 / thickness;
    }

        
    // f0 /= 1e-6; f1 /= 1e-6;
    // std::cout << f0.norm() << std::endl;
    TM R90 = TM::Zero();
    R90.row(0) = TV(0, -1);
    R90.row(1) = TV(1, 0);

    TM _X = TM::Zero(), _x = TM::Zero();
    _X.col(0) = (Xj - Xi);
    _X.col(1) = (Xk - Xl);

    _x.col(0) = (xj - xi);
    _x.col(1) = (xk - xl);

    TM F_macro = _x * _X.inverse();
    
    cauchy_strain = 0.5 * (F_macro.transpose() + F_macro) - TM::Identity();

    TV n1 = (R90 * (xj - xi)).normalized(), 
        n0 = (R90 * (xl - xk)).normalized();

    if (f1.dot(n1) < 0.0)
    {
        f1 *= -1.0;
        f0 *= -1.0;
    }

    TM f_bc = TM::Zero(), n_bc = TM::Zero();
    
    f_bc.col(0) = f0; f_bc.col(1) = f1;
    
    n_bc.col(0) = n0; n_bc.col(1) = n1;

    cauchy_stress = f_bc * n_bc.inverse();
    TV dx = Xj - Xi, dy = Xk - Xl;
    // std::cout << dx.norm() << " " << dy.norm() << " " << thickness << " " << E << " " << nu << std::endl;
    if ((Xi - Xl).norm() > 1e-6)
    {
        if ((Xk - Xj).norm() > 1e-6)
        {
            std::cout << "ALERT" << std::endl;
            std::cout << (Xi - Xl).norm() << " " << Xi.transpose() << " " << Xl.transpose() << " " << Xk.transpose() << " " << Xj.transpose() << std::endl;
            std::getchar();
        }
        else
        {
            dx = Xi - Xj;
            dy = Xl - Xk;
        }
    }

    T volume = TV3(dx[0], dx[1], 0).cross(TV3(dy[0], dy[1], 0)).norm() * thickness;
    
    T total_energy = computeTotalEnergy(u);
    energy_density = total_energy / volume;
}

void FEMSolver::computeHomogenizationData(TM& secondPK_stress, TM& Green_strain, T& energy_density)
{
    
    TV xi = deformed.segment<2>(pbc_pairs[0][0][0] * 2);
    TV xj = deformed.segment<2>(pbc_pairs[0][0][1] * 2);

    TV xk = deformed.segment<2>(pbc_pairs[1][0][0] * 2);
    TV xl = deformed.segment<2>(pbc_pairs[1][0][1] * 2);


    TV Xi = undeformed.segment<2>(pbc_pairs[0][0][0] * 2);
    TV Xj = undeformed.segment<2>(pbc_pairs[0][0][1] * 2);
    TV Xk = undeformed.segment<2>(pbc_pairs[1][0][0] * 2);
    TV Xl = undeformed.segment<2>(pbc_pairs[1][0][1] * 2);

    T sign = 1.0;

    TV dx = Xj - Xi, dy = Xk - Xl;
    // std::cout << dx.norm() << " " << dy.norm() << " " << thickness << " " << E << " " << nu << std::endl;
    if ((Xi - Xl).norm() > 1e-6)
    {
        if ((Xk - Xj).norm() > 1e-6)
        {
            std::cout << "ALERT" << std::endl;
            std::cout << (Xi - Xl).norm() << " " << Xi.transpose() << " " << Xl.transpose() << " " << Xk.transpose() << " " << Xj.transpose() << std::endl;
            std::getchar();
        }
        else
        {
            sign = -1.0;
            dx = Xi - Xj;
            dy = Xl - Xk;
        }
    }

    VectorXT inner_force(num_nodes * 2);
    inner_force.setZero(); 
    addPBCForceEntries(inner_force);
    
    // computeResidual(u, inner_force);
    TV f0 = TV::Zero(), f1 = TV::Zero();
    T l0 = (xj - xi).norm(), l1 = (xl - xk).norm();
    for (auto pbc_pair : pbc_pairs[0])
    {
        f0 += inner_force.segment<2>(pbc_pair[0] * 2) / l1 / thickness;
    }
    for (auto pbc_pair : pbc_pairs[1])
    {
        f1 += inner_force.segment<2>(pbc_pair[1] * 2) / l0 / thickness;
    }

        
    // f0 /= 1e-6; f1 /= 1e-6;
    // std::cout << f0.norm() << std::endl;
    TM R90 = TM::Zero();
    R90.row(0) = TV(0, -1);
    R90.row(1) = TV(1, 0);

    TM _X = TM::Zero(), _x = TM::Zero();
    _X.col(0) = (Xj - Xi);
    _X.col(1) = (Xk - Xl);

    _x.col(0) = (xj - xi);
    _x.col(1) = (xk - xl);

    TM F_macro = _x * _X.inverse();
    Green_strain = 0.5 * (F_macro.transpose() * F_macro - TM::Identity());
    TM cauchy_strain = 0.5 * (F_macro.transpose() + F_macro) - TM::Identity();
    // std::cout << 0.5 * (F_macro.transpose() + F_macro) - TM::Identity() << std::endl;
    // std::cout << F_macro << std::endl;

    TV n1 = (R90 * (xj - xi)).normalized(), 
        n0 = (R90 * (xl - xk)).normalized();

    f1 *= sign;
    f0 *= sign;
    
    TM f_bc = TM::Zero(), n_bc = TM::Zero();
    
    f_bc.col(0) = f0; f_bc.col(1) = f1;
    
    n_bc.col(0) = n0; n_bc.col(1) = n1;

    TM cauchy_stress = f_bc * n_bc.inverse();
    
    
    //https://engcourses-uofa.ca/books/introduction-to-solid-mechanics/stress/first-and-second-piola-kirchhoff-stress-tensors/
    TM F_inv = F_macro.inverse();
    secondPK_stress = F_macro.determinant() * F_inv * cauchy_stress.transpose() * F_inv.transpose();

    
    T volume = TV3(dx[0], dx[1], 0).cross(TV3(dy[0], dy[1], 0)).norm() * thickness;
    // volume /= 1e-9;
    // std::cout << "volume " << volume << std::endl;
    
    T total_energy = computeTotalEnergy(u);
    // std::cout << "potential " << total_energy << std::endl;
    energy_density = total_energy / volume;

    // Vector<T, 4> Green_strain_vector;
    // Green_strain_vector << Green_strain(0, 0), Green_strain(0, 1), Green_strain(1, 0), Green_strain(1, 1);
    // T energy_AD;
    // computeNHEnergyFromGreenStrain(E, nu, Green_strain_vector, energy_AD);
    // std::cout << "green strain " << std::endl;
    // std::cout << Green_strain << std::endl;
    // std::cout << "energy_density homo " << energy_density << " energy_density AD " << energy_AD << std::endl;
    // Vector<T, 4> secondPK_stress_vector;
    // computeNHEnergyFromGreenStrainGradient(E, nu, Green_strain_vector, secondPK_stress_vector);
    // TM secondPK_stress_AD;
    // secondPK_stress_AD << secondPK_stress_vector(0), secondPK_stress_vector(1), secondPK_stress_vector(2), secondPK_stress_vector(3);
    // std::cout << "second PK homo " << secondPK_stress << std::endl << "second PK AD " << secondPK_stress_AD << std::endl;
    // std::cout << "##############################" << std::endl;
    // std::getchar();
    // auto computedPsidE = [&](const Eigen::Matrix<double,2,2> & Green_strain, 
    //         double& energy, TM& dPsidE)
    // {
    //     T trace = Green_strain(0, 0) + Green_strain(1, 1);
    //     TM E2 = Green_strain * Green_strain;
    //     energy = 0.5 * trace * trace + E2(0, 0) + E2(1, 1);
    //     dPsidE = trace * TM::Identity() + 2.0 * Green_strain; 
    // };

    // auto difftest = [&]()
    // {
    //     T eps = 1e-6;
    //     TM rest;
    //     T E0, E1;
    //     computedPsidE(Green_strain, energy_density, rest);
    //     Green_strain(1, 0) += eps;
    //     computedPsidE(Green_strain, E0, secondPK_stress);
    //     Green_strain(1, 0) -= 2.0 * eps;
    //     computedPsidE(Green_strain, E1, secondPK_stress);
    //     Green_strain(1, 0) += eps;
    //     std::cout << "fd " << (E0 - E1) / 2.0/ eps << " analytic " << rest(1, 0) << std::endl;
    // };
    // computedPsidE(Green_strain, energy_density, secondPK_stress);
    // std::cout << "Green Strain" << std::endl;
    // std::cout << Green_strain << std::endl;
    // std::cout << "second PK Stress" << std::endl;
    // std::cout << secondPK_stress << std::endl;
    // difftest();
    // std::exit(0);
   
}

void FEMSolver::computeHomogenizedStressStrain(TM& sigma, TM& Cauchy_strain, TM& Green_strain)
{
    TV xi = deformed.segment<2>(pbc_pairs[0][0][0] * 2);
    TV xj = deformed.segment<2>(pbc_pairs[0][0][1] * 2);

    TV xk = deformed.segment<2>(pbc_pairs[1][0][0] * 2);
    TV xl = deformed.segment<2>(pbc_pairs[1][0][1] * 2);


    TV Xi = undeformed.segment<2>(pbc_pairs[0][0][0] * 2);
    TV Xj = undeformed.segment<2>(pbc_pairs[0][0][1] * 2);
    TV Xk = undeformed.segment<2>(pbc_pairs[1][0][0] * 2);
    TV Xl = undeformed.segment<2>(pbc_pairs[1][0][1] * 2);

    VectorXT inner_force(num_nodes * 2);
    inner_force.setZero(); 
    addPBCForceEntries(inner_force);
    
    // computeResidual(u, inner_force);
    TV f0 = TV::Zero(), f1 = TV::Zero();
    T l0 = (xj - xi).norm(), l1 = (xl - xk).norm();
    for (auto pbc_pair : pbc_pairs[0])
    {
        f0 += inner_force.segment<2>(pbc_pair[0] * 2) / l1;
    }
    for (auto pbc_pair : pbc_pairs[1])
    {
        f1 += inner_force.segment<2>(pbc_pair[1] * 2) / l0;
    }
    
    TM R90 = TM::Zero();
    R90.row(0) = TV(0, -1);
    R90.row(1) = TV(1, 0);

    TM _X = TM::Zero(), _x = TM::Zero();
    _X.col(0) = (Xi - Xj).template segment<2>(0);
    _X.col(1) = (Xk - Xl).template segment<2>(0);

    _x.col(0) = (xi - xj).template segment<2>(0);
    _x.col(1) = (xk - xl).template segment<2>(0);

    TM F_macro = _x * _X.inverse();
    Cauchy_strain = 0.5 * (F_macro.transpose() + F_macro) - TM::Identity();
    Green_strain = 0.5 * (F_macro.transpose() * F_macro - TM::Identity());
    

    TV n1 = (R90 * (xj - xi)).normalized(), 
        n0 = (R90 * (xl - xk)).normalized();

    TM f_bc = TM::Zero(), n_bc = TM::Zero();
    
    f_bc.col(0) = f0; f_bc.col(1) = f1;
    
    n_bc.col(0) = n0; n_bc.col(1) = n1;

    sigma = f_bc * n_bc.inverse();
}

void FEMSolver::computeHomogenizedStressStrain(TM& sigma, TM& epsilon)
{
    
    TV xi = deformed.segment<2>(pbc_pairs[0][0][0] * 2);
    TV xj = deformed.segment<2>(pbc_pairs[0][0][1] * 2);

    TV xk = deformed.segment<2>(pbc_pairs[1][0][0] * 2);
    TV xl = deformed.segment<2>(pbc_pairs[1][0][1] * 2);


    TV Xi = undeformed.segment<2>(pbc_pairs[0][0][0] * 2);
    TV Xj = undeformed.segment<2>(pbc_pairs[0][0][1] * 2);
    TV Xk = undeformed.segment<2>(pbc_pairs[1][0][0] * 2);
    TV Xl = undeformed.segment<2>(pbc_pairs[1][0][1] * 2);

    VectorXT inner_force(num_nodes * 2);
    inner_force.setZero(); 
    addPBCForceEntries(inner_force);
    
    // computeResidual(u, inner_force);
    TV f0 = TV::Zero(), f1 = TV::Zero();
    T l0 = (xj - xi).norm(), l1 = (xl - xk).norm();
    for (auto pbc_pair : pbc_pairs[0])
    {
        f0 += inner_force.segment<2>(pbc_pair[0] * 2) / l1;
    }
    for (auto pbc_pair : pbc_pairs[1])
    {
        f1 += inner_force.segment<2>(pbc_pair[1] * 2) / l0;
    }
    
    TM R90 = TM::Zero();
    R90.row(0) = TV(0, -1);
    R90.row(1) = TV(1, 0);

    TM _X = TM::Zero(), _x = TM::Zero();
    _X.col(0) = (Xi - Xj).template segment<2>(0);
    _X.col(1) = (Xk - Xl).template segment<2>(0);

    _x.col(0) = (xi - xj).template segment<2>(0);
    _x.col(1) = (xk - xl).template segment<2>(0);

    TM F_macro = _x * _X.inverse();
    // std::cout << "deformation graident" << std::endl;
    // std::cout << F_macro << std::endl;
    
    TM cauchy_strain = 0.5 * (F_macro.transpose() + F_macro) - TM::Identity();
    // std::cout << "cauchy_strain" << std::endl;
    // std::cout << cauchy_strain << std::endl;
    epsilon =  0.5 * (F_macro.transpose() + F_macro) - TM::Identity();

    // Matrix<T, 4, 2> x, X;
    // getMarcoBoundaryData(x, X);

    // T eps_xx = (xl[0] - xk[0]) / (Xl[0] - Xk[0]) - 1.0;
    // T eps_yy = (xj[1] - xi[1]) / (Xj[1] - Xi[1]) - 1.0;
    // T vdu = (xl[1] - xk[1] - (Xl[1] - Xk[1])) / ((Xl[0] - Xk[0]));
    // T udv = (xj[0] - xi[0] - (Xj[0] - Xi[0])) / (Xj[1] - Xi[1]);
    // T eps_xy = 0.5 * (vdu + udv);
    // Matrix<T, 3, 2> dNdb;
    //     dNdb << -1.0, -1.0, 
    //         1.0, 0.0,
    //         0.0, 1.0;
    // EleNodes x_undeformed, x_deformed;
    // x_undeformed << 0,0,1,0,0,1;
    // x_deformed << Xl[0], Xl[1], Xk[0], Xk[1], Xj[0], Xj[1];
    // // std::cout << x_deformed << std::endl;
    // // std::cout << Xl.transpose() << " " << Xk.transpose() << " " << Xj.transpose() << std::endl;
    // TM dXdb = x_undeformed.transpose() * dNdb;
    // TM dxdb = x_deformed.transpose() * dNdb;
    // TM F = dxdb * dXdb.inverse();
    
    // // xi = F.inverse() * xi; xj = F.inverse() * xj;xk = F.inverse() * xk;xl = F.inverse() * xl;
    // // Xi = F.inverse() * Xi; Xj = F.inverse() * Xj;Xk = F.inverse() * Xk;Xl = F.inverse() * Xl;
    // std::ofstream out("test_mesh.obj");
    // // out << "v " << xi.transpose() << " 0" << std::endl;
    // // out << "v " << xj.transpose() << " 0" << std::endl;
    // // out << "v " << xk.transpose() << " 0" << std::endl;
    // out << "v " << Xl.transpose() << " 0" << std::endl;
    // out << "v " << Xk.transpose() << " 0" << std::endl;
    // out << "v " << Xj.transpose() << " 0" << std::endl;

    // out << "v " << xl.transpose() << " 0" << std::endl;
    // out << "v " << xk.transpose() << " 0" << std::endl;
    // out << "v " << xj.transpose() << " 0" << std::endl;
    // // out << "v " << Xi.transpose() << " 0" << std::endl;
    // out << "f 1 2 3" << std::endl;
    // // out << "f 1 2 4" << std::endl;
    // out << "f 4 5 6" << std::endl;
    // out.close();
    // T eps_xx = (xl[0] - xk[0]) / (Xl[0] - Xk[0]) - 1.0;
    // T eps_yy = (xj[1] - xi[1]) / (Xj[1] - Xi[1]) - 1.0;
    // T v_du = (xl[1] - xk[1]) / ((Xl[0] - Xk[0]));
    // T u_dv = (xj[0] - xi[0]) / (Xj[1] - Xi[1]);
    // T eps_xy = 0.5 * (u_dv + v_du);
    // // T eps_xy = (xl[1] - xk[1]) / ((Xl[0] - Xk[0]));
    // // T eps_yx = (xj[0] - xi[0]) / (Xj[1] - Xi[1]);
    
    // epsilon << eps_xx, eps_xy, eps_xy, eps_yy;
    

    // std::cout << strain_macro << std::endl;
    
    TV n1 = (R90 * (xj - xi)).normalized(), 
        n0 = (R90 * (xl - xk)).normalized();

    // std::ofstream out("debug.obj");
    // int cnt = 0;
    // for (auto pbc_pair : pbc_pairs[1])
    // {
        
    //     int idx0 = pbc_pair[0], idx1 = pbc_pair[1];
    //     TV Xi = undeformed.segment<2>(idx0 * 2);
    //     TV Xj = undeformed.segment<2>(idx1 * 2);
    //     out << "v " << Xj.transpose() << " 0" << std::endl;
    //     cnt ++;
    // }
    // for (int i = 0; i < cnt - 1;i++)
    //     out << "l " << i + 1 << " " << i + 2 << std::endl;
    // out << "v " << xi_ref.transpose() << " 0" << std::endl;
    // out << "v " << xj_ref.transpose() << " 0" << std::endl;
    // out << "v " << xk_ref.transpose() << " 0" << std::endl;
    // out << "v " << xl_ref.transpose() << " 0" << std::endl;
    // out << "v " << (xi_ref + n1).transpose() << " 0" << std::endl;
    // out << "v " << (xk_ref + n0).transpose() << " 0" << std::endl;
    // out << "l 1 2" << std::endl;
    // out << "l 1 5" << std::endl;
    // out << "l 3 4" << std::endl;
    // out << "l 3 6" << std::endl; 
    // out.close();
    TM f_bc = TM::Zero(), n_bc = TM::Zero();
    
    f_bc.col(0) = f0; f_bc.col(1) = f1;
    
    n_bc.col(0) = n0; n_bc.col(1) = n1;

    

    sigma = f_bc * n_bc.inverse();

    // TV strain_dir = TV(std::cos(strain_theta), std::sin(strain_theta));
    // std::cout << sigma * (R90 *strain_dir) << std::endl;
    // std::cout << std::endl;
    // std::cout << "stress macro" << std::endl;
    // std::cout << sigma << std::endl;
}