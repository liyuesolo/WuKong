#include "../include/FEMSolver.h"
#include "../include/autodiff/FEMEnergy.h"
bool FEMSolver::addPBCPairs3D()
{
    SpatialHash hash;
    T bb_diag = (max_corner - min_corner).norm();
    hash.build(0.01 * bb_diag, undeformed);
    std::vector<std::vector<int>> boundary_nodes(6);
    getPBCPairsAxisDirection(boundary_nodes[0], boundary_nodes[1], 0, hash);
    getPBCPairsAxisDirection(boundary_nodes[2], boundary_nodes[3], 1, hash);
    getPBCPairsAxisDirection(boundary_nodes[4], boundary_nodes[5], 2, hash);

    // bool same_num_nodes_dir = true;
    // for (int i = 0; i < 3; i++)
    //     same_num_nodes_dir &= (boundary_nodes[i*2+0] == boundary_nodes[i*2+1]);
    
    // if (!same_num_nodes_dir)
    //     return false;
    
    pbc_pairs = {std::vector<Edge>(), std::vector<Edge>(), std::vector<Edge>()};

    for (int i = 0; i < boundary_nodes[0].size(); i++)
    {
        pbc_pairs[0].push_back(Edge(boundary_nodes[0][i], boundary_nodes[1][i]));
    }
    for (int i = 0; i < boundary_nodes[2].size(); i++)
    {
        pbc_pairs[1].push_back(Edge(boundary_nodes[2][i], boundary_nodes[3][i]));
    }
    for (int i = 0; i < boundary_nodes[4].size(); i++)
    {
        pbc_pairs[2].push_back(Edge(boundary_nodes[4][i], boundary_nodes[5][i]));
    }
    std::cout << "# pbc pairs " << pbc_pairs[0].size() + pbc_pairs[1].size() + pbc_pairs[2].size() << std::endl;
    return true;

}

void FEMSolver::getPBCPairsAxisDirection(std::vector<int>& side0, 
    std::vector<int>& side1, int direction, SpatialHash& hash)
{
    T thres_hold = 1e-6;
    side0.clear(); side1.clear();
    
    for (int i = 0; i < num_nodes; i++)
    {
        TV xi = undeformed.segment<3>(i * 3);
        if (std::abs(xi[direction] - min_corner[direction]) < thres_hold)
        {
            std::vector<int> neighbors;
            TV xj = xi; xj[direction] += (max_corner[direction] - min_corner[direction]);
            hash.getOneRingNeighbors(xj, neighbors);
            for (int idx : neighbors)
            {
                TV xk = undeformed.segment<3>(idx * 3);
                // std::cout << xj.transpose() << " " << xk.transpose() << std::endl;
                // std::getchar();
                if ((xj - xk).norm() < 1e-6)
                {
                    side0.push_back(i); side1.push_back(idx);
                    break;
                }
            }
        }
    }
}

void FEMSolver::addPBCEnergy(T& energy)
{
    T energy_pbc = 0.0;
    TV strain_dir = TV(std::sin(theta) * std::cos(phi), 
                        std::sin(theta) * std::sin(phi), std::cos(theta));
    TM rotate_Y; rotate_Y << std::cos(M_PI_2), 0.0, std::sin(M_PI_2), 0.0, 1.0, 0.0, -std::sin(M_PI_2), 0.0, std::cos(M_PI_2);
    TV dir_ortho = rotate_Y * strain_dir;
    TM rotation_third = Eigen::AngleAxis(M_PI_2, dir_ortho).toRotationMatrix();
    TV dir_tri = rotation_third * strain_dir;


    auto addPBCEnergyDirection = [&](int dir)
    {
        int cnt = 0;
        int n_pairs = pbc_pairs[dir].size();
        for (auto pbc_pair : pbc_pairs[dir])
        {
            // strain term
            int idx0 = pbc_pair[0], idx1 = pbc_pair[1];
            TV Xi = undeformed.segment<3>(idx0 * 3);
            TV Xj = undeformed.segment<3>(idx1 * 3);
            TV xi = deformed.segment<3>(idx0 * 3);
            TV xj = deformed.segment<3>(idx1 * 3);

            T Dij = (Xj - Xi).dot(strain_dir);
            T dij = (xj - xi).dot(strain_dir);
            
            // add uniaxial loading first
            T dij_target = Dij * strain_magnitudes[0];
            energy_pbc += 0.5 * pbc_strain_w * (dij - dij_target) * (dij - dij_target) / T(n_pairs);

            if (loading_type == BI_AXIAL || loading_type == TRI_AXIAL)
            {
                Dij = (Xj - Xi).dot(dir_ortho);
                dij = (xj - xi).dot(dir_ortho);
                
                T dij_target = Dij * strain_magnitudes[1];
                energy_pbc += 0.5 * pbc_strain_w * (dij - dij_target) * (dij - dij_target) / T(n_pairs);
            }

            if (loading_type == TRI_AXIAL)
            {
                Dij = (Xj - Xi).dot(dir_tri);
                dij = (xj - xi).dot(dir_tri);
                
                T dij_target = Dij * strain_magnitudes[2];
                energy_pbc += 0.5 * pbc_strain_w * (dij - dij_target) * (dij - dij_target) / T(n_pairs);
            }

            
            cnt++;
            if (cnt == 1)
                continue;
            
            // distance term
            TV xi_ref = deformed.segment<3>(pbc_pairs[dir][0][0] * 3);
            TV xj_ref = deformed.segment<3>(pbc_pairs[dir][0][1] * 3);

            TV pair_dis_vec = xj - xi - (xj_ref - xi_ref);
            energy_pbc += 0.5 * pbc_w * pair_dis_vec.dot(pair_dis_vec) / T(n_pairs);
        }
    };           

    addPBCEnergyDirection(0);
    addPBCEnergyDirection(1);
    addPBCEnergyDirection(2);

    energy += energy_pbc;
}

void FEMSolver::addPBCForceEntries(VectorXT& residual)
{
    VectorXT pbc_force = residual;
    pbc_force.setZero();
    TV strain_dir = TV(std::sin(theta) * std::cos(phi), 
                        std::sin(theta) * std::sin(phi), std::cos(theta));
    TM rotate_Y; rotate_Y << std::cos(M_PI_2), 0.0, std::sin(M_PI_2), 0.0, 1.0, 0.0, -std::sin(M_PI_2), 0.0, std::cos(M_PI_2);
    TV dir_ortho = rotate_Y * strain_dir;
    
    TM rotation_third = Eigen::AngleAxis(M_PI_2, dir_ortho).toRotationMatrix();
    TV dir_tri = rotation_third * strain_dir;
    
    auto addPBCForceDirection = [&](int dir)
    {
        int cnt = 0;
        int n_pairs = pbc_pairs[dir].size();
        for (auto pbc_pair : pbc_pairs[dir])
        {
            int idx0 = pbc_pair[0], idx1 = pbc_pair[1];
            
            TV Xi = undeformed.segment<3>(idx0 * 3);
            TV Xj = undeformed.segment<3>(idx1 * 3);
            TV xi = deformed.segment<3>(idx0 * 3);
            TV xj = deformed.segment<3>(idx1 * 3);

            T Dij = (Xj - Xi).dot(strain_dir);
            T dij_target = Dij * strain_magnitudes[0];
            T dij = (xj - xi).dot(strain_dir);
            pbc_force.segment<3>(idx0 * 3) += pbc_strain_w * strain_dir * (dij - dij_target) / T(n_pairs);
            pbc_force.segment<3>(idx1 * 3) -= pbc_strain_w * strain_dir * (dij - dij_target) / T(n_pairs);

            if (loading_type == BI_AXIAL || loading_type == TRI_AXIAL)
            {
                Dij = (Xj - Xi).dot(dir_ortho);
                dij = (xj - xi).dot(dir_ortho);
                
                T dij_target = Dij * strain_magnitudes[1];
                pbc_force.segment<3>(idx0 * 3) += pbc_strain_w * dir_ortho * (dij - dij_target) / T(n_pairs);
                pbc_force.segment<3>(idx1 * 3) -= pbc_strain_w * dir_ortho * (dij - dij_target) / T(n_pairs);
            }

            if (loading_type == TRI_AXIAL)
            {
                Dij = (Xj - Xi).dot(dir_tri);
                dij = (xj - xi).dot(dir_tri);
                
                T dij_target = Dij * strain_magnitudes[2];
                pbc_force.segment<3>(idx0 * 3) += pbc_strain_w * dir_tri * (dij - dij_target) / T(n_pairs);
                pbc_force.segment<3>(idx1 * 3) -= pbc_strain_w * dir_tri * (dij - dij_target) / T(n_pairs);
            }

            cnt++;
            if (cnt == 1)
                continue;

            TV xi_ref = deformed.segment<3>(pbc_pairs[dir][0][0] * 3);
            TV xj_ref = deformed.segment<3>(pbc_pairs[dir][0][1] * 3);

            TV pair_dis_vec = xj - xi - (xj_ref - xi_ref);
            pbc_force.segment<3>(idx0 * 3) += pbc_w * pair_dis_vec / T(n_pairs);
            pbc_force.segment<3>(idx1 * 3) += -pbc_w * pair_dis_vec / T(n_pairs);

            pbc_force.segment<3>(pbc_pairs[dir][0][0] * 3) += -pbc_w * pair_dis_vec / T(n_pairs);
            pbc_force.segment<3>(pbc_pairs[dir][0][1] * 3) += pbc_w * pair_dis_vec / T(n_pairs);
        }
    };

    addPBCForceDirection(0);
    addPBCForceDirection(1);
    addPBCForceDirection(2);
    residual += pbc_force;
}

void FEMSolver::addPBCHessianEntries(std::vector<Entry>& entries, bool project_PD)
{
    TV strain_dir = TV(std::sin(theta) * std::cos(phi), 
                        std::sin(theta) * std::sin(phi), std::cos(theta));
    TM rotate_Y; rotate_Y << std::cos(M_PI_2), 0.0, std::sin(M_PI_2), 0.0, 1.0, 0.0, -std::sin(M_PI_2), 0.0, std::cos(M_PI_2);
    TV dir_ortho = rotate_Y * strain_dir;
    TM rotation_third = Eigen::AngleAxis(M_PI_2, dir_ortho).toRotationMatrix();
    TV dir_tri = rotation_third * strain_dir;
    auto addPBCHessianDirection = [&](int dir)
    {
        int cnt = 0;
        int n_pairs = pbc_pairs[dir].size();
        for (auto pbc_pair : pbc_pairs[dir])
        {
            int idx0 = pbc_pair[0], idx1 = pbc_pair[1];
            TV Xi = undeformed.segment<3>(idx0 * 3);
            TV Xj = undeformed.segment<3>(idx1 * 3);
            TV xi = deformed.segment<3>(idx0 * 3);
            TV xj = deformed.segment<3>(idx1 * 3);


            TM Hessian = strain_dir * strain_dir.transpose() / T(n_pairs);
            
            for(int i = 0; i < 3; i++)
            {
                for(int j = 0; j < 3; j++)
                {
                    entries.push_back(Entry(idx0 * 3 + i, idx0 * 3 + j, pbc_strain_w * Hessian(i, j)));
                    entries.push_back(Entry(idx0 * 3 + i, idx1 * 3 + j, -pbc_strain_w * Hessian(i, j)));
                    entries.push_back(Entry(idx1 * 3 + i, idx0 * 3 + j, -pbc_strain_w * Hessian(i, j)));
                    entries.push_back(Entry(idx1 * 3 + i, idx1 * 3 + j, pbc_strain_w * Hessian(i, j)));
                }
            }

            if (loading_type == BI_AXIAL || loading_type == TRI_AXIAL)
            {
                Hessian = dir_ortho * dir_ortho.transpose() / T(n_pairs);
                for(int i = 0; i < 3; i++)
                {
                    for(int j = 0; j < 3; j++)
                    {
                        entries.push_back(Entry(idx0 * 3 + i, idx0 * 3 + j, pbc_strain_w * Hessian(i, j)));
                        entries.push_back(Entry(idx0 * 3 + i, idx1 * 3 + j, -pbc_strain_w * Hessian(i, j)));
                        entries.push_back(Entry(idx1 * 3 + i, idx0 * 3 + j, -pbc_strain_w * Hessian(i, j)));
                        entries.push_back(Entry(idx1 * 3 + i, idx1 * 3 + j, pbc_strain_w * Hessian(i, j)));
                    }
                }

            }

            if (loading_type == TRI_AXIAL)
            {
                Hessian = dir_tri * dir_tri.transpose() / T(n_pairs);
                for(int i = 0; i < 3; i++)
                {
                    for(int j = 0; j < 3; j++)
                    {
                        entries.push_back(Entry(idx0 * 3 + i, idx0 * 3 + j, pbc_strain_w * Hessian(i, j)));
                        entries.push_back(Entry(idx0 * 3 + i, idx1 * 3 + j, -pbc_strain_w * Hessian(i, j)));
                        entries.push_back(Entry(idx1 * 3 + i, idx0 * 3 + j, -pbc_strain_w * Hessian(i, j)));
                        entries.push_back(Entry(idx1 * 3 + i, idx1 * 3 + j, pbc_strain_w * Hessian(i, j)));
                    }
                }
            }
            

            cnt++;
            if (cnt == 1)
                continue;

            TV xi_ref = deformed.segment<3>(pbc_pairs[dir][0][0] * 3);
            TV xj_ref = deformed.segment<3>(pbc_pairs[dir][0][1] * 3);
            TV pair_dis_vec = xj - xi - (xj_ref - xi_ref);
            
            std::vector<int> nodes = {idx0, idx1, pbc_pairs[dir][0][0], pbc_pairs[dir][0][1]};
            std::vector<T> sign_J = {-1, 1, 1, -1};
            std::vector<T> sign_F = {1, -1, -1, 1};

            for(int k = 0; k < 4; k++)
                for(int l = 0; l < 4; l++)
                    for(int i = 0; i < 3; i++)
                        entries.push_back(Entry(nodes[k]*3 + i, nodes[l] * 3 + i, -pbc_w *sign_F[k]*sign_J[l] / T(n_pairs)));
        }
    };

    addPBCHessianDirection(0);
    addPBCHessianDirection(1);
    addPBCHessianDirection(2);
}

void FEMSolver::computeHomogenizationData(TM& strain_Green, TM& stress_2ndPK, T& energy_density)
{
    TV xi = deformed.segment<3>(pbc_pairs[0][0][0] * 3);
    TV xj = deformed.segment<3>(pbc_pairs[0][0][1] * 3);
    TV xk = deformed.segment<3>(pbc_pairs[1][0][0] * 3);
    TV xl = deformed.segment<3>(pbc_pairs[1][0][1] * 3);
    TV xm = deformed.segment<3>(pbc_pairs[2][0][0] * 3);
    TV xn = deformed.segment<3>(pbc_pairs[2][0][1] * 3);


    TV Xi = undeformed.segment<3>(pbc_pairs[0][0][0] * 3);
    TV Xj = undeformed.segment<3>(pbc_pairs[0][0][1] * 3);
    TV Xk = undeformed.segment<3>(pbc_pairs[1][0][0] * 3);
    TV Xl = undeformed.segment<3>(pbc_pairs[1][0][1] * 3);
    TV Xm = undeformed.segment<3>(pbc_pairs[2][0][0] * 3);
    TV Xn = undeformed.segment<3>(pbc_pairs[2][0][1] * 3);

    VectorXT inner_force(num_nodes * 3);
    inner_force.setZero(); 
    addPBCForceEntries(inner_force);
    iterateDirichletDoF([&](int offset, T target)
    {
        inner_force[offset] = 0.0;
    });

    TV f0 = TV::Zero(), f1 = TV::Zero(), f2 = TV::Zero();
    T l0 = (xj - xi).norm(), l1 = (xl - xk).norm(), l2 = (xm - xn).norm();
    T area12 = ((xl-xk).cross(xn-xm)).norm();
    T area02 = ((xj-xi).cross(xn-xm)).norm();
    T area01 = ((xj-xi).cross(xl-xk)).norm();
    for (auto pbc_pair : pbc_pairs[0])
    {
        f0 += inner_force.segment<3>(pbc_pair[0] * 3) / area12;
    }
    for (auto pbc_pair : pbc_pairs[1])
    {
        f1 += inner_force.segment<3>(pbc_pair[0] * 3) / area02;
    }
    for (auto pbc_pair : pbc_pairs[2])
    {
        f2 += inner_force.segment<3>(pbc_pair[0] * 3) / area01;
    }

    TM _X = TM::Zero(), _x = TM::Zero();
    _X.col(0) = (Xj - Xi);
    _X.col(1) = (Xk - Xl);
    _X.col(2) = (Xm - Xn);

    _x.col(0) = (xj - xi);
    _x.col(1) = (xk - xl);
    _x.col(2) = (xm - xn);

    TM F_macro = _x * _X.inverse();

    // std::cout << "Deformation gradient " << std::endl;
    // std::cout << F_macro << std::endl;
    // std::cout << std::endl;

    TV n0, n1, n2;
    n0 = ((xn - xm).cross(xl-xk)).normalized();
    if (f0.dot(n0) < 0.0) f0 *= -1.0;
    n1 = ((xj - xi).cross(xn-xm)).normalized();
    if (f1.dot(n1) < 0.0) f1 *= -1.0;
    n2 = ((xj - xi).cross(xl-xk)).normalized();
    if (f2.dot(n2) < 0.0) f2 *= -1.0;

    TM f_bc = TM::Zero(), n_bc = TM::Zero();
    
    f_bc.col(0) = f0; f_bc.col(1) = f1; f_bc.col(2) = f2;
    
    n_bc.col(0) = n0; n_bc.col(1) = n1; n_bc.col(2) = n2;

    strain_Green = 0.5 * (F_macro.transpose() * F_macro - TM::Identity());

    TM cauchy_stress = f_bc * n_bc.inverse();
    // std::cout << "cauchy stress" << std::endl;
    // std::cout << cauchy_stress << std::endl << std::endl;
    TM F_inv = F_macro.inverse();
    stress_2ndPK = F_macro.determinant() * F_inv * cauchy_stress.transpose() * F_inv.transpose();

    
    T volume = std::abs(((Xj - Xi).cross(Xl - Xk)).dot(Xn - Xm));
    T total_energy = computeTotalEnergy(u);
    energy_density = total_energy / volume;

    // Vector<T, 9> strain_vec, stress_vec;
    // for (int i = 0; i < 3; i ++) for (int j = 0;j < 3;j ++) strain_vec[i *3 + j] = strain_Green(i, j);
    
    // T psi;
    // computeNHEnergyFromGreenStrain3D(E, nu, strain_vec, psi, stress_vec);
    // TM stress_GT;
    // for (int i = 0; i < 3; i ++) for (int j = 0;j < 3;j ++) stress_GT(i, j) = stress_vec[i*3 +j];

    // std::cout << "stress_GT " << stress_GT << std::endl << std::endl;
    // std::cout << "energy density GT " << psi << std::endl << std::endl;
}