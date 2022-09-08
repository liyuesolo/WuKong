#include "../include/FEMSolver.h"
#include "../include/autodiff/PBCEnergy.h"

void FEMSolver::addPBCPairsXY()
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
    rotate(-alpha);

    pbc_pairs = {std::vector<IV>(), std::vector<IV>()};
    // std::cout <<  dir0_side0.size() << " " << dir0_side1.size()
    //     << " " << dir1_side0.size() << " " << dir1_side1.size() << std::endl;
    
    for (int i = 0; i < dir0_side0.size(); i++)
    {
        pbc_pairs[0].push_back(IV(dir0_side0[i], dir0_side1[i]));
    }
    for (int i = 0; i < dir1_side0.size(); i++)
    {
        pbc_pairs[1].push_back(IV(dir1_side0[i], dir1_side1[i]));
    }

}

void FEMSolver::getPBCPairsAxisDirection(std::vector<int>& side0, 
    std::vector<int>& side1, int direction)
{
    bool ortho = !direction;
    // std::cout << "dir " << direction << " " << ortho << std::endl;
    side0.clear(); side1.clear();
    TV min_corner, max_corner;
    computeBoundingBox(min_corner, max_corner);
    
    for (int i = 0; i < num_nodes; i++)
    {
        TV xi = undeformed.segment<2>(i * 2);
        if (std::abs(xi[direction] - min_corner[direction]) < 1e-6)
            side0.push_back(i);
        if (std::abs(xi[direction] - max_corner[direction]) < 1e-6)
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
            if (add_pbc_strain && prescribe_strain_tensor)
            {
                TV xj_target;
                if (dir == 1)
                    xj_target = xi + TV((Xj-Xi).dot(TV(1, 0))*(target_strain[0] + T(1.0)), 
                                        (Xj-Xi).dot(TV(1, 0))*(target_strain[2]));
                else if (dir == 0)
                    xj_target = xi + TV((Xj-Xi).dot(TV(0, 1))*(target_strain[2]), 
                                        (Xj-Xi).dot(TV(0, 1))*(target_strain[1] + T(1.0)));
                
                energy += 0.5 * pbc_strain_w * (xj_target - xj).dot(xj_target - xj);
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

    // if (add_pbc_strain && prescribe_strain_tensor)
    // {
    //     Matrix<T, 4, 2> x, X;
    //     getMarcoBoundaryData(x, X);
    //     T strain_matching_term;
    //     // computeStrainMatchingEnergy(pbc_strain_w, target_strain, x, X, 
    //     //     strain_matching_term);
    //     TV xa = x.row(0), xb = x.row(1), xc = x.row(2), xd = x.row(3);
	// 	TV Xa = X.row(0), Xb = X.row(1), Xc = X.row(2), Xd = X.row(3);
    //     TV3 epsilon = target_strain;
    //     TV xb_target = xa + TV((Xb[0]-Xa[0]) * (epsilon[0] + 1.0),
    //                             (Xb[0]-Xa[0]) * epsilon[2]);
    //     TV xd_target = xa + TV((Xd[1] - Xa[1]) * epsilon[2],
    //                             (Xd[1] - Xa[1]) * (epsilon[1] + 1.0));
    //     TV xc_target = xa + TV((Xc[0]-Xa[0]) * (epsilon[0] + 1.0) + (Xc[0] - Xc[0]) * epsilon[2],
    //         (Xc[1] - Xa[1]) * epsilon[2] + (Xc[1] - Xa[1]) * (epsilon[1] + 1.0) );
	// 	// TV xb_target = xa + TV(Xb-Xa).dot(TV(1, 0))*(epsilon[0] + T(1.0)), 
	// 	// 									(Xb-Xa).dot(TV(0, 1))*(epsilon[2] + T(1.0)));

	// 	// TV xd_target = xa + TV((Xd-Xa).dot(TV(1, 0))*(epsilon[2] + T(1.0)), 
	// 	// 									(Xd-Xa).dot(TV(0, 1))*(epsilon[1] + T(1.0)));
        

	// 	// TV xc_target = xa + TV((Xc-Xa).dot(TV(1, 0))*(epsilon[0] + epsilon[2] + T(1.0)), 
	// 	// 									(Xc-Xa).dot(TV(0, 1))*(epsilon[2] + epsilon[1] + T(1.0)));

	// 	strain_matching_term = 0.5 * pbc_strain_w * ((xb_target - xb).dot(xb_target - xb) + 
	// 									(xc_target - xc).dot(xc_target - xc) + 
	// 									(xd_target - xd).dot(xd_target - xd));
        
    //     energy += strain_matching_term;
    //     // std::cout << strain_matching_term / pbc_strain_w << std::endl;
    // }

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
            if (add_pbc_strain && prescribe_strain_tensor)
            {
                TV xj_target;
                if (dir == 1)
                    xj_target = xi + TV((Xj-Xi).dot(TV(1, 0))*(target_strain[0] + T(1.0)), 
                                        (Xj-Xi).dot(TV(1, 0))*(target_strain[2]));
                else if (dir == 0)
                    xj_target = xi + TV((Xj-Xi).dot(TV(0, 1))*(target_strain[2]), 
                                        (Xj-Xi).dot(TV(0, 1))*(target_strain[1] + T(1.0)));

                TV dedxi = pbc_strain_w * (xj_target - xj);
                TV dedxj = -pbc_strain_w * (xj_target - xj);
                residual.segment<2>(idx0 * 2) -= dedxi;
                residual.segment<2>(idx1 * 2) -= dedxj;
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

    // if (add_pbc_strain && prescribe_strain_tensor)
    // {
    //     Matrix<T, 4, 2> x, X;
    //     getMarcoBoundaryData(x, X);

    //     TV xa = x.row(0), xb = x.row(1), xc = x.row(2), xd = x.row(3);
	// 	TV Xa = X.row(0), Xb = X.row(1), Xc = X.row(2), Xd = X.row(3);
    //     TV3 epsilon = target_strain;
	// 	// TV xb_target = xa + TV((Xb-Xa).dot(TV(1, 0))*(epsilon[0] + T(1.0)), 
	// 	// 									(Xb-Xa).dot(TV(0, 1))*(epsilon[2] + T(1.0)));

	// 	// TV xd_target = xa + TV((Xd-Xa).dot(TV(1, 0))*(epsilon[2] + T(1.0)), 
	// 	// 									(Xd-Xa).dot(TV(0, 1))*(epsilon[1] + T(1.0)));

	// 	// TV xc_target = xa + TV((Xc-Xa).dot(TV(1, 0))*(epsilon[0] + epsilon[2] + T(1.0)), 
	// 	// 									(Xc-Xa).dot(TV(0, 1))*(epsilon[2] + epsilon[1] + T(1.0)));

	// 	TV xb_target = xa + TV((Xb[0]-Xa[0]) * (epsilon[0] + 1.0),
    //                             (Xb[0]-Xa[0]) * epsilon[2]);
    //     TV xd_target = xa + TV((Xd[1] - Xa[1]) * epsilon[2],
    //                             (Xd[1] - Xa[1]) * (epsilon[1] + 1.0));
    //     TV xc_target = xa + TV((Xc[0]-Xa[0]) * (epsilon[0] + 1.0) + (Xc[0] - Xc[0]) * epsilon[2],
    //         (Xc[1] - Xa[1]) * epsilon[2] + (Xc[1] - Xa[1]) * (epsilon[1] + 1.0) );
        

    //     TV dedxa = pbc_strain_w * ((xb_target - xb) + (xc_target - xc) + (xd_target - xd));
    //     TV dedxb = -pbc_strain_w * (xb_target - xb);
    //     TV dedxc = -pbc_strain_w * (xc_target - xc);
    //     TV dedxd = -pbc_strain_w * (xd_target - xd);

    //     // std::cout << dedxa.transpose() << " " << dedxb.transpose() << " "<< dedxc.transpose() << " "<< dedxd.transpose() << std::endl;
    //     // Vector<T, 8> dedx;
    //     // computeStrainMatchingEnergyGradient(pbc_strain_w, target_strain, x, X, dedx);
    //     // for (int i = 0; i < 4; i++)
    //     // {
    //     //     residual.segment<2>(pbc_corners[i] * 2) -= dedx.segment<2>(i * 2);
    //     // }
        
    //     // std::cout << dedx.transpose() << std::endl;
    //     residual.segment<2>(pbc_corners[0] * 2) -= dedxa;
    //     residual.segment<2>(pbc_corners[1] * 2) -= dedxb;
    //     residual.segment<2>(pbc_corners[2] * 2) -= dedxc;
    //     residual.segment<2>(pbc_corners[3] * 2) -= dedxd;
    //     // std::getchar();
    // }
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
            if (add_pbc_strain && prescribe_strain_tensor)
            {
                Matrix<T, 4, 4> d2edx2;
                d2edx2.setZero();
                d2edx2.block(0, 0, 2, 2) = pbc_strain_w * TM::Identity();
                d2edx2.block(2, 2, 2, 2) = pbc_strain_w * TM::Identity();

                d2edx2.block(2, 0, 2, 2) = -1.0 * pbc_strain_w * TM::Identity();
                d2edx2.block(0, 2, 2, 2) = -1.0 * pbc_strain_w * TM::Identity();

                std::vector<int> ij_global = {idx0, idx1};
                for (int i = 0; i < ij_global.size(); i++)
                {
                    for (int j = 0; j < ij_global.size(); j++)
                    {
                        for (int k = 0; k < 2; k++)
                        {
                            for (int l = 0; l < 2; l++)
                            {
                                entries.push_back(Entry(
                                    ij_global[i] * 2 + k, 
                                    ij_global[j] * 2 + l, 
                                    d2edx2(i * 2 + k, j * 2 + l)));
                            }
                            
                        }   
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

    // if (add_pbc_strain && prescribe_strain_tensor)
    // {
        
    //     Matrix<T, 4, 2> x, X;
    //     getMarcoBoundaryData(x, X);

    //     Matrix<T, 8, 8> d2edx2;
    //     // computeStrainMatchingEnergyHessian(pbc_strain_w, target_strain, x, X, d2edx2);
    //     d2edx2.setZero();
    //     d2edx2.block(0, 0, 2, 2) = 3.0 * pbc_strain_w * TM::Identity();
    //     d2edx2.block(2, 2, 2, 2) = 1.0 * pbc_strain_w * TM::Identity();
    //     d2edx2.block(4, 4, 2, 2) = 1.0 * pbc_strain_w * TM::Identity();
    //     d2edx2.block(6, 6, 2, 2) = 1.0 * pbc_strain_w * TM::Identity();

    //     d2edx2.block(2, 0, 2, 2) = -1.0 * pbc_strain_w * TM::Identity();
    //     d2edx2.block(4, 0, 2, 2) = -1.0 * pbc_strain_w * TM::Identity();
    //     d2edx2.block(6, 0, 2, 2) = -1.0 * pbc_strain_w * TM::Identity();

    //     d2edx2.block(0, 2, 2, 2) = -1.0 * pbc_strain_w * TM::Identity();
    //     d2edx2.block(0, 4, 2, 2) = -1.0 * pbc_strain_w * TM::Identity();
    //     d2edx2.block(0, 6, 2, 2) = -1.0 * pbc_strain_w * TM::Identity();

    //     for (int i = 0; i < 4; i++)
    //     {
    //         for (int j = 0; j < 4; j++)
    //         {
    //             for (int k = 0; k < 2; k++)
    //             {
    //                 for (int l = 0; l < 2; l++)
    //                 {
    //                     entries.push_back(Entry(
    //                         pbc_corners[i] * 2 + k, 
    //                         pbc_corners[j] * 2 + l, 
    //                         d2edx2(i * 2 + k, j * 2 + l)));
    //                 }
                       
    //             }   
    //         }
            
    //     }   
    // }
}

void FEMSolver::computeMarcoBoundaryIndices()
{
    pbc_corners << pbc_pairs[0][0][0], 
        pbc_pairs[0][pbc_pairs[0].size() - 1][0],
        pbc_pairs[0][pbc_pairs[0].size() - 1][1],
        pbc_pairs[0][0][1];
    
    // std::cout << pbc_corners << std::endl;
}

void FEMSolver::getMarcoBoundaryData(Matrix<T, 4, 2>& x, Matrix<T, 4, 2>& X)
{
    x.row(0) = deformed.segment<2>(pbc_corners[0] * 2);
    x.row(1) = deformed.segment<2>(pbc_corners[1] * 2);
    x.row(2) = deformed.segment<2>(pbc_corners[2] * 2);
    x.row(3) = deformed.segment<2>(pbc_corners[3] * 2);

    X.row(0) = undeformed.segment<2>(pbc_corners[0] * 2);
    X.row(1) = undeformed.segment<2>(pbc_corners[1] * 2);
    X.row(2) = undeformed.segment<2>(pbc_corners[2] * 2);
    X.row(3) = undeformed.segment<2>(pbc_corners[3] * 2);
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

    // f0 /= 0.01; f1 /= 0.01;
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
    
    // std::cout << F_macro << std::endl;

    TV n1 = (R90 * (xj - xi)).normalized(), 
        n0 = (R90 * (xl - xk)).normalized();

    TM f_bc = TM::Zero(), n_bc = TM::Zero();
    
    f_bc.col(0) = f0; f_bc.col(1) = f1;
    
    n_bc.col(0) = n0; n_bc.col(1) = n1;

    TM cauchy_stress = f_bc * n_bc.inverse();

    //https://engcourses-uofa.ca/books/introduction-to-solid-mechanics/stress/first-and-second-piola-kirchhoff-stress-tensors/
    TM F_inv = F_macro.inverse();
    secondPK_stress = F_macro.determinant() * F_inv * cauchy_stress.transpose() * F_inv.transpose();

    TV dx = Xj - Xi, dy = Xk - Xl;
    if ((Xi - Xl).norm() > 1e-6)
    {
        std::cout << "ALERT" << std::endl;
        std::getchar();
    }
    T volume = TV3(dx[0], dx[1], 0).cross(TV3(dy[0], dy[1], 0)).norm() * thickness;
    // volume *= 0.001;
    // std::cout << "volume " << volume << std::endl;
    T total_energy = computeTotalEnergy(u);
    // std::cout << "potential " << total_energy;
    energy_density = total_energy / volume;

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