#include <igl/mosek/mosek_quadprog.h>
#include "../../include/Objectives.h"
#include <Eigen/PardisoSupport>
#include <Eigen/CholmodSupport>
#include <Spectra/SymEigsShiftSolver.h>
#include <Spectra/MatOp/SparseSymShiftSolve.h>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include "../../include/DataIO.h"
#include "../../include/LinearSolver.h"
#include "../../include/autodiff/EdgeEnergy.h"
#include "../../include/autodiff/Deformation.h"

void ObjNucleiTracking::initializeTarget()
{
    // std::vector<int> VF_cell_idx;
    // simulation.cells.getVFCellIds(VF_cell_idx);
    // // std::cout << "VF cells " << VF_cell_idx.size() << std::endl;
    // for (int idx : VF_cell_idx)
    // {
    //     TV cell_centroid;
    //     VtxList face_vtx_list = simulation.cells.faces[idx];
    //     simulation.cells.computeCellCentroid(face_vtx_list, cell_centroid);
    //     target_positions[idx] = cell_centroid;
    // }
    for (int i = 0; i < simulation.cells.basal_face_start; i++)
    {
        target_positions[i] = TV::Zero();
    }
    
}

void ObjNucleiTracking::loadTxtVertices(const std::string& filename, VectorXT& data)
{
    std::ifstream in(filename);
    std::vector<T> data_std_vector;
    T x, y, z;
    while (in >> x >> y >> z)
    {
        data_std_vector.push_back(x);
        data_std_vector.push_back(y);
        data_std_vector.push_back(z);
    }
    in.close();
    data = Eigen::Map<VectorXT>(data_std_vector.data(), data_std_vector.size());
}

void ObjNucleiTracking::loadOBJwithOnlyVertices(const std::string& filename, VectorXT& data)
{
    std::ifstream in(filename);
    std::vector<T> data_std_vector;
    char c; T x, y, z;
    while (in >> c >> x >> y >> z)
    {
        data_std_vector.push_back(x);
        data_std_vector.push_back(y);
        data_std_vector.push_back(z);
    }
    in.close();
    data = Eigen::Map<VectorXT>(data_std_vector.data(), data_std_vector.size());
}

void ObjNucleiTracking::loadWeightedCellTarget(const std::string& filename, bool use_all_points)
{
    target_filename = filename;
    VectorXT data_points, data_point_unfiltered;
    bool success = true;
    success = getTargetTrajectoryFrame(data_point_unfiltered);
    std::string filtered_result_folder = "/home/yueli/Documents/ETH/WuKong/output/FilteredData/";
    // loadOBJwithOnlyVertices(filtered_result_folder + std::to_string(frame) + "_filtered.obj", data_points);
    // loadTxtVertices(filtered_result_folder + std::to_string(frame) + "_denoised.txt", data_points);
    // set in DataIO::loadTrajectories()
    data_points = data_point_unfiltered;
    std::vector<bool> flag(data_point_unfiltered.rows()/ 3, false);
    std::string base_folder = "/home/yueli/Documents/ETH/WuKong/output/cells/stats/data/";
    std::string dis_threshold_string = "1.2";
    std::ifstream in(base_folder + "valid_points_frame_" + std::to_string(frame) +"_thres_"+dis_threshold_string+".txt");
    int idx;
    while(in >> idx)
        flag[idx] = true;
    in.close();
    TV invalid_point = TV::Constant(-1e10);
    // bool use_all_points = false;
    // std::ofstream out("data_points" + std::to_string(frame) + ".obj");

    // for (int i = 0; i < data_points.rows() / 3; i++)
    // {
    //     if ((data_points.segment<3>(i * 3) - invalid_point).norm() < 1e-6)
    //         continue;
    //     out << "v " << data_points.segment<3>(i * 3).transpose() << std::endl;
    // }
    // out.close();
    TV max_corner, min_corner;
    simulation.cells.computeBoundingBox(min_corner, max_corner);

    if (success)
    {
        int n_cells = simulation.cells.basal_face_start;
        std::ifstream in(filename);
        int data_point_idx, cell_idx, nw;
        std::vector<int> visited(n_cells, 0);
        
        while (in >> data_point_idx >> cell_idx >> nw)
        {
            VectorXT w(nw); w.setZero();
            for (int j = 0; j < nw; j++)
                in >> w[j];
            if (!use_all_points)
            {
                if (visited[cell_idx] != 0)
                    continue;
            }
            // if (!flag[cell_idx])
            //     continue;
            
            if (data_point_idx > data_points.rows() / 3)
                continue;
            TV target = data_points.segment<3>(data_point_idx * 3);
            // if (target[1] > min_corner[1] + 0.3 * (max_corner[1] - min_corner[1]))
            //     continue;
            if ((target - invalid_point).norm() < 1e-6)
                continue;
            if ((target - TV::Constant(-1)).norm() < 1e-6)
                continue;
            
            if (!simulation.cells.sdf.inside(target))
                continue;

            VectorXT positions;
            std::vector<int> indices;
            simulation.cells.getCellVtxAndIdx(cell_idx, positions, indices);

            TV current = TV::Zero();
            for (int i = 0; i < nw; i++)
                current += w[i] * positions.segment<3>(i * 3);
            T error = (current - target).norm();
            
            weight_targets.push_back(TargetData(w, data_point_idx, cell_idx, target));
            visited[cell_idx]++;
            // if (error > 0.5)
            // {
            //     std::cout << "target: " << target.transpose() << " data id: " 
            //         << data_point_idx << " cell id: " << cell_idx << " w "
            //         << w.transpose() << std::endl;
            //     std::cout << "raw target " << data_point_unfiltered.segment<3>(data_point_idx * 3).transpose() << std::endl;
            //     std::getchar();
            // }
            // else
            //     std::cout << error << std::endl;
        }
        
        in.close();
        target_obj_weights.resize(n_cells);
        target_obj_weights.setOnes();
        // std::cout << weight_targets.size() << std::endl;
        if (use_all_points)
        {
            for (int i = 0; i < n_cells; i++)
            {
                if (visited[i])
                    target_obj_weights[i] = 1.0 / visited[i];
            }
        }
        for (int i = 0; i < n_cells; i++)
        {
            if (visited[i] == 0)
            {
                if (add_spatial_x)
                {
                    target_obj_weights[i] = 0.0;
                }
                else
                {
                    VectorXT positions;
                    std::vector<int> indices;
                    simulation.cells.getCellVtxAndIdx(i, positions, indices);
                    VectorXT w(indices.size());
                    T avg = 1.0 / T(indices.size());
                    w.setConstant(avg);
                    TV centroid = TV::Zero();
                    for (int j  = 0; j < indices.size(); j++)
                        centroid += positions.segment<3>(j * 3);
                    centroid *= avg;
                    // weight_targets.push_back(TargetData(w, data_point_idx, i, centroid));
                    // target_obj_weights[i] = 1e-3;
                }
            }
        }
        
    }
    target_obj_weights *= w_data;
    
}

void ObjNucleiTracking::loadWeightedTarget(const std::string& filename)
{
    VectorXT data_points;
    bool success = getTargetTrajectoryFrame(data_points);
    if (success)
    {
        std::ifstream in(filename);
        int cell_idx, nw;
        std::vector<bool> visited(simulation.cells.basal_face_start, false);
        while (in >> cell_idx >> nw)
        {
            // std::cout << cell_idx << std::endl;
            std::vector<T> w(nw, 0.0);
            std::vector<int> neighbors(nw, 0);
            for (int j = 0; j < nw; j++)
                in >> neighbors[j] >> w[j];
            TV target = TV::Zero();
            for (int i = 0; i < nw; i++)
            {
                target += w[i] * data_points.segment<3>(neighbors[i] * 3);
            }
            target_positions[cell_idx] = target;
        }
        in.close();
    }
}


void ObjNucleiTracking::setTargetObjWeights()
{
    std::random_device rd;
    std::mt19937 g(rd());
    std::vector<int> cell_list;
    for (int i = 0; i < simulation.cells.basal_face_start; i++)
        cell_list.push_back(i);

    int n_cell_total = simulation.cells.basal_face_start;

    int n_target_cell = std::floor(n_cell_total/2.0);

    std::cout << n_target_cell << "/" << n_cell_total << std::endl;

    std::shuffle(cell_list.begin(), cell_list.end(), g);
  
    // std::copy(cell_list.begin(), cell_list.begin() + n_target_cell, std::ostream_iterator<int>(std::cout, " "));

    // target_obj_weights.setConstant(1e-5);
    target_obj_weights.setConstant(0);

    // std::vector<int> selected = {46, 1, 114, 118, 54, 240, 32, 73, 23, 31};
    std::vector<int> selected = {46, 1, 114, 118, 54,  32, 73, 23, 31};
    // std::vector<int> selected = {114, 118};
    std::vector<int> VF_cell_idx;
    simulation.cells.getVFCellIds(VF_cell_idx);
    // selected.push_back(VF_cell_idx[0]);
    // selected.push_back(VF_cell_idx[4]);
    // for (int idx : selected)
        //  target_obj_weights[idx] = 1.0;

    for (int i = 0; i < n_target_cell; i++)
        target_obj_weights[cell_list[i]] = 1.0;

    // VectorXT cell_centroids;
    // simulation.cells.getAllCellCentroids(cell_centroids);
    // TV min_corner, max_corner;
    // simulation.cells.computeBoundingBox(min_corner, max_corner);
    // T spacing = 0.01 * (max_corner - min_corner).norm();

    // SpatialHash centroid_hash;
    // centroid_hash.build(spacing, cell_centroids);

    // for (int i = 0; i < n_target_cell; i++)
    // {
    //     TV centroid = cell_centroids.segment<3>(cell_list[i] * 3);
    //     std::vector<int> neighbors;
    //     centroid_hash.getOneRingNeighbors(centroid, neighbors);
    //     target_obj_weights[cell_list[i]] = 1.0;
    //     for (int idx : neighbors)
    //         target_obj_weights[idx] = 0.0;
    // }
}

void ObjNucleiTracking::optimizeForStableTargetDeformationGradient(T perturbation)
{
    TV c0, c1;
    simulation.cells.computeCellCentroid(simulation.cells.faces[0], c0);
    simulation.cells.computeCellCentroid(simulation.cells.faces[1], c1);
    
    T length = (c0 - c1).norm();
    
    VectorXT targets_inititial(target_positions.size() * 3);
    VectorXT targets_opt(target_positions.size() * 3);
    VectorXT targets_rest(target_positions.size() * 3);

    std::vector<std::vector<int>> neighbor_cells;

    for (auto data : target_positions)
    {
        targets_rest.segment<3>(data.first * 3) = data.second;
        targets_inititial.segment<3>(data.first * 3) = data.second + perturbation * TV::Random().normalized() * length;
        targets_opt.segment<3>(data.first * 3) = targets_inititial.segment<3>(data.first * 3);
    }

    int n_cells = simulation.cells.n_cells;
    int n_dof = targets_inititial.rows();

    T w_p = 100.0;

    auto fetchEntryFromVector = [&](const std::vector<int>& indices, 
        const VectorXT& vector, VectorXT& entries)
    {
        entries.resize(indices.size() * 3);
        for (int i = 0; i < indices.size(); i++)
            entries.segment<3>(i * 3) = vector.segment<3>(indices[i] * 3);
    };

    auto energyValue = [&](const VectorXT& x)
    {
        T energy = 0.0;
        energy += 0.5 * (x - targets_inititial).dot(x - targets_inititial);

        for (int i = 0; i < n_cells; i++)
        {
            VectorXT undeformed, deformed;
            fetchEntryFromVector(neighbor_cells[i], targets_rest, undeformed);
            fetchEntryFromVector(neighbor_cells[i], x, deformed);
            T ei;
            computeDeformationPenaltyCST3D(w_p, undeformed, deformed, ei);
            energy += ei;
        }
        
        return energy;
    };

    auto energyGradient = [&](const VectorXT& x, VectorXT& dedx)
    {
        dedx = x - targets_inititial;
        for (int i = 0; i < n_cells; i++)
        {
            VectorXT undeformed, deformed;
            fetchEntryFromVector(neighbor_cells[i], targets_rest, undeformed);
            fetchEntryFromVector(neighbor_cells[i], x, deformed);
            Vector<T, 9> dedx_local;
            computeDeformationPenaltyCST3DGradient(w_p, undeformed, deformed, dedx_local);
            if (std::isnan(dedx_local.norm()))
            {
                std::ofstream out("stupid_points.obj");
                for (int i = 0; i < 3; i++)
                    out << "v " << undeformed.segment<3>(i * 3).transpose() << std::endl;
                out.close();
                std::cout << dedx_local.norm() << std::endl;
                std::getchar();
            }
            for (int j = 0; j < neighbor_cells[i].size(); j++)
            {
                dedx.segment<3>(neighbor_cells[i][j] * 3) += dedx_local.segment<3>(j * 3);
            }
        }
        return dedx.norm();
    };

    auto energyHessian = [&](const VectorXT& x, std::vector<Entry>& entries)
    {
        for (int n = 0; n < n_cells; n++)
        {
            entries.push_back(Entry(n, n, 1.0));
            VectorXT undeformed, deformed;
            fetchEntryFromVector(neighbor_cells[n], targets_rest, undeformed);
            fetchEntryFromVector(neighbor_cells[n], x, deformed);
            Matrix<T, 9, 9> hessian_local;
            computeDeformationPenaltyCST3DHessian(w_p, undeformed, deformed, hessian_local);
            for (int i = 0; i < neighbor_cells[n].size(); i++)
            {
                for (int j = 0; j < neighbor_cells[n].size(); j++)
                {
                    for (int k = 0; k < 3; k++)
                    {
                        for (int l = 0; l < 3; l++)
                        {
                            entries.push_back(Entry(neighbor_cells[n][i] * 3 + k, neighbor_cells[n][j] * 3 + l, hessian_local(i * 3 + k, j * 3 + l)));
                        }   
                    }
                }
            }
        }
    };

    SpatialHash hash;
    T spacing = 0.5 * length;
    hash.build(spacing, targets_rest);

    for (int i = 0; i < n_cells; i++)
    {
        std::vector<int> neighbors;
        TV pi = targets_rest.segment<3>(i * 3);
        hash.getOneRingNeighbors(pi, neighbors);
        
        std::vector<T> distance;
        std::vector<int> indices;
        for (int idx : neighbors)
        {
            TV pj = targets_rest.segment<3>(idx * 3);
            if ((pi - pj).norm() < 1e-5)
                continue;
            distance.push_back((pi - pj).norm());
            indices.push_back(idx);
        }
        std::sort(indices.begin(), indices.end(), [&distance](int a, int b){ return distance[a] < distance[b]; } );
        
        std::vector<int> closest_pts;
        closest_pts.push_back(i);
        closest_pts.push_back(indices[0]);
        closest_pts.push_back(indices[1]);
        // closest_pts.push_back(indices[2]);
        neighbor_cells.push_back(closest_pts);
    }

    T tol = 1e-9;
    T g_norm = 1e10;
    int ls_max = 10;
    int opt_iter = 0;

    int max_iter = 100;
    simulation.verbose = false;
    T g_norm0 = 0;
    while (true)
    {
        T O; 
        VectorXT dOdx;
        g_norm = energyGradient(targets_opt, dOdx);
        O = energyValue(targets_opt);
        std::cout << "iter " << opt_iter << " |g|: " << g_norm << " E: " << O << std::endl;
        
        if (opt_iter == 0)
            g_norm0 = g_norm;
        if (g_norm < tol * g_norm0 || opt_iter > max_iter)
            break;
        StiffnessMatrix H(n_dof, n_dof);
        std::vector<Entry> entries;
        energyHessian(targets_opt, entries);
        H.setFromTriplets(entries.begin(), entries.end());
        VectorXT g = -dOdx, dx = VectorXT::Zero(n_dof);
        simulation.linearSolve(H, g, dx);
        T alpha = 1.0;
        int i = 0;
        for (; i < ls_max; i++)
        {
            VectorXT x_ls = targets_opt + alpha * dx;
            T O_ls = energyValue(x_ls);
            if (O_ls < O)
            {
                targets_opt = x_ls;
                break;
            }
            alpha *= 0.5;
        }
        if (i == ls_max)
        {

        }
        std::cout << "#ls " << i << "/" << ls_max << std::endl;
        opt_iter++;
    } 
    std::ofstream out("init.obj");
    for (int i = 0; i < n_cells; i++)
    {
        out << "v " << targets_inititial.segment<3>(i * 3).transpose() << std::endl;
    }
    out.close();
    out.open("opt.obj");
    for (int i = 0; i < n_cells; i++)
    {
        out << "v " << targets_opt.segment<3>(i * 3).transpose() << std::endl;
    }
    out.close();
    out.open("rest.obj");
    for (int i = 0; i < n_cells; i++)
    {
        out << "v " << targets_rest.segment<3>(i * 3).transpose() << std::endl;
    }
    out.close();

    for (auto& data : target_positions)
    {
        data.second = targets_opt.segment<3>(data.first * 3);
    }
}

void ObjNucleiTracking::optimizeForStableTargetSpring(T perturbation)
{
    TV c0, c1;
    simulation.cells.computeCellCentroid(simulation.cells.faces[0], c0);
    simulation.cells.computeCellCentroid(simulation.cells.faces[1], c1);
    
    T length = (c0 - c1).norm();
    
    VectorXT targets_inititial(target_positions.size() * 3);
    VectorXT targets_opt(target_positions.size() * 3);
    VectorXT targets_rest(target_positions.size() * 3);

    for (auto data : target_positions)
    {
        targets_rest.segment<3>(data.first * 3) = data.second;
        targets_inititial.segment<3>(data.first * 3) = data.second + perturbation * TV::Random().normalized() * length;
        targets_opt.segment<3>(data.first * 3) = targets_inititial.segment<3>(data.first * 3);
    }

    int n_cells = simulation.cells.n_cells;
    int n_dof = targets_inititial.rows();
    SpatialHash hash;
    T spacing = 0.5 * length;
    hash.build(spacing, targets_rest);

    struct Spring
    {
        T thres = 0.75;
        // T thres = 1.0;
        T stiffness;
        Vector<T, 6> x;
        Vector<T, 6> X;
        IV2 indices;
        T rest_length = 1.0;
        Spring(T _stiffness, const Vector<T, 6>& _x, 
            const Vector<T, 6>& _X, const IV2& _indices,
            T _rest_length) : 
            stiffness(_stiffness), x(_x), X(_X), 
            indices(_indices), rest_length(_rest_length) {}
        T l0() { return (X.segment<3>(3) - X.segment<3>(0)).norm(); }
        // T l0 () {return rest_length; }
        T l() { return (x.segment<3>(3) - x.segment<3>(0)).norm(); }
        void setx(const Vector<T, 6>& _x) { x = _x; }
        T energy() 
        {
            if (l() > thres * l0())
                return 0.0;
            T e;
            computeSpringUnilateralQubicEnergyRestLength3D(stiffness, l0(), x, e);
            return e;
        }
        void gradient(Vector<T, 6>& g)
        {
            if (l() > thres * l0())
            {
                g.setZero();
                return;
            }
            computeSpringUnilateralQubicEnergyRestLength3DGradient(stiffness, l0(), x, g);
        }
        void hessian(std::vector<Entry>& entries)
        {
            if (l() > thres * l0())
                return;
            Matrix<T, 6, 6> h;
            computeSpringUnilateralQubicEnergyRestLength3DHessian(stiffness, l0(), x, h);
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 2; j++)
                    for (int k = 0; k < 3; k++)
                        for (int l = 0; l < 3; l++)
                            entries.push_back(Entry(indices[i] * 3 + k, indices[j] * 3 + l, h(i * 3 + k, j * 3 + l)));
            
        }
    };

    int n_pt = n_dof / 3;
    MatrixXT visited(n_pt, n_pt);
    visited.setConstant(-1);
    T k = 1e4;
    std::vector<Spring> springs;
    for (int i = 0; i < n_pt; i++)
    {
        TV pi = targets_rest.segment<3>(i * 3);
        std::vector<int> neighbors;
        hash.getOneRingNeighbors(pi, neighbors);
        for (int j : neighbors)
        {
            if (visited(i, j) == -1)
            {
                TV pj = targets_rest.segment<3>(j * 3);
                if ((pi - pj).norm() < 1e-6)
                    continue;
                Vector<T, 6> X; X << pi, pj;
                springs.push_back(Spring(k, X, X, IV2(i, j), length));
                visited(i, j) = 1; visited(j, i) = 1;
            }
        }
    }

    std::cout << "# springs " << springs.size() << std::endl;
    // std::getchar();

    auto energyValue = [&](const VectorXT& x)
    {
        T energy = 0.0;
        energy += 0.5 * (x - targets_inititial).dot(x - targets_inititial);

        for (auto spring : springs)
        {
            Vector<T, 6> x0x1;
            x0x1.segment<3>(0) = x.segment<3>(spring.indices[0] * 3);
            x0x1.segment<3>(3) = x.segment<3>(spring.indices[1] * 3);
            spring.setx(x0x1);
            energy += spring.energy();
        }

        return energy;
    };

    auto energyGradient = [&](const VectorXT& x, VectorXT& dedx)
    {
        dedx = x - targets_inititial;
        for (auto spring : springs)
        {
            Vector<T, 6> x0x1;
            x0x1.segment<3>(0) = x.segment<3>(spring.indices[0] * 3);
            x0x1.segment<3>(3) = x.segment<3>(spring.indices[1] * 3);
            spring.setx(x0x1);
            Vector<T, 6> dedx_local;
            spring.gradient(dedx_local);
            dedx.segment<3>(spring.indices[0] * 3) += dedx_local.segment<3>(0);
            dedx.segment<3>(spring.indices[1] * 3) += dedx_local.segment<3>(3);
        }

        return dedx.norm();
    };

    auto energyHessian = [&](const VectorXT& x, std::vector<Entry>& entries)
    {
        for (int i = 0; i < n_dof; i++)
        {
            entries.push_back(Entry(i , i, 1.0));
        }
        
        for (auto spring : springs)
        {
            Vector<T, 6> x0x1;
            x0x1.segment<3>(0) = x.segment<3>(spring.indices[0] * 3);
            x0x1.segment<3>(3) = x.segment<3>(spring.indices[1] * 3);
            spring.setx(x0x1);
            
            spring.hessian(entries);
        }
    };

    T tol = 1e-7;
    T g_norm = 1e10;
    int ls_max = 10;
    int opt_iter = 0;

    int max_iter = 200;
    simulation.verbose = false;
    T g_norm0 = 0;
    while (true)
    {
        T O; 
        VectorXT dOdx;
        g_norm = energyGradient(targets_opt, dOdx);
        O = energyValue(targets_opt);
        std::cout << "iter " << opt_iter << " |g|: " << g_norm << " E: " << O << std::endl;
        // std::getchar();
        if (opt_iter == 0)
            g_norm0 = g_norm;
        if (g_norm < tol * g_norm0 || opt_iter > max_iter)
            break;
        StiffnessMatrix H(n_dof, n_dof);
        std::vector<Entry> entries;
        energyHessian(targets_opt, entries);
        H.setFromTriplets(entries.begin(), entries.end());
        VectorXT g = -dOdx, dx = VectorXT::Zero(n_dof);
        simulation.linearSolve(H, g, dx);
        
        T alpha = 1.0;
        int i = 0;
        for (; i < ls_max; i++)
        {
            VectorXT x_ls = targets_opt + alpha * dx;
            T O_ls = energyValue(x_ls);
            if (O_ls < O)
            {
                targets_opt = x_ls;
                break;
            }
            alpha *= 0.5;
        }
        if (i == ls_max)
        {

        }
        std::cout << "#ls " << i << "/" << ls_max << std::endl;
        opt_iter++;
    }
    std::ofstream out("init.obj");
    for (int i = 0; i < n_cells; i++)
    {
        out << "v " << targets_inititial.segment<3>(i * 3).transpose() << std::endl;
    }
    out.close();
    out.open("opt.obj");
    for (int i = 0; i < n_cells; i++)
    {
        out << "v " << targets_opt.segment<3>(i * 3).transpose() << std::endl;
    }
    out.close();
    out.open("rest.obj");
    for (int i = 0; i < n_cells; i++)
    {
        out << "v " << targets_rest.segment<3>(i * 3).transpose() << std::endl;
    }
    out.close();

    for (auto& data : target_positions)
    {
        data.second = targets_opt.segment<3>(data.first * 3);
    }
}

void ObjNucleiTracking::optimizeForStableTarget(T perturbation)
{
    TV c0, c1;
    simulation.cells.computeCellCentroid(simulation.cells.faces[0], c0);
    simulation.cells.computeCellCentroid(simulation.cells.faces[1], c1);
    
    T length = (c0 - c1).norm();
    
    VectorXT targets_inititial(target_positions.size() * 3);
    VectorXT targets_opt(target_positions.size() * 3);
    VectorXT targets_rest(target_positions.size() * 3);

    for (auto data : target_positions)
    {
        targets_rest.segment<3>(data.first * 3) = data.second;
        targets_inititial.segment<3>(data.first * 3) = data.second + perturbation * TV::Random().normalized() * length;
        targets_opt.segment<3>(data.first * 3) = targets_inititial.segment<3>(data.first * 3);
    }

    int n_cells = simulation.cells.n_cells;
    int dof = targets_inititial.rows();

    auto fetchEntryFromVector = [&](const std::vector<int>& indices, 
        const VectorXT& vector, VectorXT& entries)
    {
        entries.resize(indices.size() * 3);
        for (int i = 0; i < indices.size(); i++)
            entries.segment<3>(i * 3) = vector.segment<3>(indices[i] * 3);
    };

    T w_p = 1.0;

    VectorXT cell_centroids;
    simulation.cells.getAllCellCentroids(cell_centroids);
    TV min_corner, max_corner;
    simulation.cells.computeBoundingBox(min_corner, max_corner);
    T spacing = 0.05 * (max_corner - min_corner).norm();

    hash.build(spacing, cell_centroids);
    std::vector<std::vector<int>> neighbor_cells;
    for (int i = 0; i < n_cells; i++)
    {
        std::vector<int> neighbors;
        TV ci = cell_centroids.segment<3>(i * 3);
        hash.getOneRingNeighbors(ci, neighbors);
        
        std::vector<T> distance;
        std::vector<int> indices;
        for (int idx : neighbors)
        {
            TV cj = cell_centroids.segment<3>(idx * 3);
            distance.push_back((ci - cj).norm());
            indices.push_back(idx);
        }
        std::sort(indices.begin(), indices.end(), [&distance](int a, int b){ return distance[a] < distance[b]; } );
        
        std::vector<int> closest_three;
        closest_three.push_back(i);
        closest_three.push_back(indices[0]);
        closest_three.push_back(indices[1]);
        // closest_three.push_back(indices[2]);
        neighbor_cells.push_back(closest_three);
    }
    
    auto energyValue = [&](const VectorXT& x)
    {
        T energy = 0.0;
        energy += 0.5 * (x - targets_inititial).dot(x - targets_inititial);

        for (int i = 0; i < n_cells; i++)
        {
            TV ci = cell_centroids.segment<3>(i * 3);
            VectorXT undeformed, deformed;
            fetchEntryFromVector(neighbor_cells[i], targets_rest, undeformed);
            fetchEntryFromVector(neighbor_cells[i], x, deformed);
            T ei;
            computeDeformationPenaltyCST3D(w_p, undeformed, deformed, ei);
            energy += ei;
        }

        return energy;
    };

    auto computeDet = [&](const VectorXT& x, const VectorXT& X)
    {
        TM undeformed, deformed, F;
		for (int i = 0; i < 3; i++)
		{
			undeformed.col(i) = X.segment<3>(3 + i * 3) - X.segment<3>(0);	
			deformed.col(i) = x.segment<3>(3 + i * 3) - x.segment<3>(0);
		}
        
        std::cout << "undeformed" << std::endl;
        std::cout << undeformed << std::endl;
        std::cout << "undeformed inverse" << std::endl;
        std::cout << undeformed.inverse() << std::endl;
		F = deformed * undeformed.inverse();
		T detF = F.determinant();
        return detF;
    };

    auto energyGradient = [&](const VectorXT& x, VectorXT& dedx)
    {
        dedx = x - targets_inititial;
        for (int i = 0; i < n_cells; i++)
        {
            TV ci = cell_centroids.segment<3>(i * 3);
            VectorXT undeformed, deformed;
            fetchEntryFromVector(neighbor_cells[i], targets_rest, undeformed);
            fetchEntryFromVector(neighbor_cells[i], x, deformed);
            // T det = computeDet(undeformed, deformed);
            Vector<T, 9> dedx_local;
            computeDeformationPenaltyCST3DGradient(w_p, undeformed, deformed, dedx_local);
            // std::cout << "det F " << det << std::endl;
            // std::cout << "neighbor cells size: " << neighbor_cells[i].size() << std::endl;
            // std::cout << undeformed.transpose() << std::endl;
            // std::cout << deformed.transpose() << std::endl;
            // std::cout << dedx_local.norm() << std::endl;
            // std::getchar();
            for (int j = 0; j < neighbor_cells[i].size(); j++)
            {
                dedx.segment<3>(neighbor_cells[i][j] * 3) += dedx_local.segment<3>(j * 3);
            }
        }
        std::cout << "norm " << dedx.norm() << std::endl;
        return dedx.norm();
    };

    auto energyHessian = [&](const VectorXT& x, std::vector<Entry>& entries)
    {
        for (int n = 0; n < n_cells; n++)
        {
            entries.push_back(Entry(n, n, 1.0));
            TV ci = cell_centroids.segment<3>(n * 3);
            VectorXT undeformed, deformed;
            fetchEntryFromVector(neighbor_cells[n], targets_rest, undeformed);
            fetchEntryFromVector(neighbor_cells[n], x, deformed);
            Matrix<T, 9, 9> hessian_local;
            computeDeformationPenaltyCST3DHessian(w_p, undeformed, deformed, hessian_local);
            for (int i = 0; i < neighbor_cells[n].size(); i++)
            {
                for (int j = 0; j < neighbor_cells[n].size(); j++)
                {
                    for (int k = 0; k < 3; k++)
                    {
                        for (int l = 0; l < 3; l++)
                        {
                            entries.push_back(Entry(neighbor_cells[n][i] * 3 + k, neighbor_cells[n][j] * 3 + l, hessian_local(i * 3 + k, j * 3 + l)));
                        }   
                    }
                }
            }
        }
    };

    T tol = 1e-7;
    T g_norm = 1e10;
    int ls_max = 10;
    int opt_iter = 0;

    int max_iter = 10000;
    simulation.verbose = false;
    T g_norm0 = 0;
    while (true)
    {
        T O; 
        VectorXT dOdx;
        g_norm = energyGradient(targets_opt, dOdx);
        O = energyValue(targets_opt);
        std::cout << "iter " << opt_iter << " |g|: " << g_norm << " E: " << O << std::endl;
        std::getchar();
        if (opt_iter == 0)
            g_norm0 = g_norm;
        if (g_norm < tol * g_norm0 || opt_iter > max_iter)
            break;
        StiffnessMatrix H(dof, dof);
        std::vector<Entry> entries;
        energyHessian(targets_opt, entries);
        H.setFromTriplets(entries.begin(), entries.end());
        VectorXT g = -dOdx, dx = VectorXT::Zero(dof);
        simulation.linearSolve(H, g, dx);
        T alpha = 1.0;
        int i = 0;
        for (; i < ls_max; i++)
        {
            VectorXT x_ls = targets_opt + alpha * dx;
            T O_ls = energyValue(x_ls);
            if (O_ls < O)
            {
                targets_opt = x_ls;
                break;
            }
            alpha *= 0.5;
        }
        if (i == ls_max)
        {

        }
        std::cout << "#ls " << i << "/" << ls_max << std::endl;
        opt_iter++;
    }
    std::ofstream out("init.obj");
    for (int i = 0; i < n_cells; i++)
    {
        out << "v " << targets_inititial.segment<3>(i * 3).transpose() << std::endl;
    }
    out.close();
    out.open("opt.obj");
    for (int i = 0; i < n_cells; i++)
    {
        out << "v " << targets_opt.segment<3>(i * 3).transpose() << std::endl;
    }
    out.close();
    out.open("rest.obj");
    for (int i = 0; i < n_cells; i++)
    {
        out << "v " << targets_rest.segment<3>(i * 3).transpose() << std::endl;
    }
    out.close();

    for (auto& data : target_positions)
    {
        data.second = targets_opt.segment<3>(data.first * 3);
    }
}

void ObjNucleiTracking::loadTarget(const std::string& filename, T perturbation)
{
    target_filename = filename;
    target_perturbation = perturbation;
    // std::vector<int> VF_cell_idx;
    // simulation.cells.getVFCellIds(VF_cell_idx);
    std::ifstream in(filename);
    int idx; T x, y, z;
    
    while(in >> idx >> x >> y >> z)
    {
        TV perturb = perturbation * TV::Random().normalized();
        target_positions[idx] = TV(x, y, z) + perturb;
        T alpha = 1.0;
        while (true)
        {
            if (simulation.cells.sdf.inside(target_positions[idx]))
                break;
            alpha *= 0.5;
            target_positions[idx] = TV(x, y, z) + alpha * perturb;
        }
        
    }
    in.close();

    target_obj_weights = VectorXT::Ones(simulation.cells.basal_face_start);
    target_obj_weights *= w_data;
}         

void ObjNucleiTracking::rotateTarget(T angle)
{
    TM R = Eigen::AngleAxisd(angle, TV::UnitX()).toRotationMatrix();
    for (auto& data : target_positions)
    {
        data.second = R * data.second;
    }
    
}

void ObjNucleiTracking::buildCentroidStructure()
{
    cell_neighbors;
    VectorXT cell_centroids;
    simulation.cells.getAllCellCentroids(cell_centroids);
    TV min_corner, max_corner;
    simulation.cells.computeBoundingBox(min_corner, max_corner);
    
    VectorXT edge_norm(simulation.cells.edges.size());
    tbb::parallel_for(0, (int)simulation.cells.edges.size(), [&](int i){
        TV vi = simulation.cells.undeformed.segment<3>(simulation.cells.edges[i][0] * 3);
        TV vj = simulation.cells.undeformed.segment<3>(simulation.cells.edges[i][1] * 3);
        edge_norm[i] = (vj - vi).norm();
    });

    T spacing = 1.5 * edge_norm.sum() / edge_norm.rows();

    SpatialHash hash_temp;
    hash_temp.build(spacing, cell_centroids);
    cell_neighbors.resize(simulation.cells.n_cells);
    tbb::parallel_for(0, simulation.cells.n_cells, [&](int i)
    {
        TV ci = cell_centroids.segment<3>(i * 3);
        std::vector<int> neighbors;
        hash_temp.getOneRingNeighbors(ci, neighbors);
        cell_neighbors[i] = neighbors;
    });

}

void ObjNucleiTracking::buildVtxEdgeStructure()
{
    int cnt = 0;
    vtx_edges.resize(simulation.cells.basal_vtx_start, std::vector<int>());
    simulation.cells.iterateApicalEdgeSerial([&](Edge& e)
    {
        vtx_edges[e[0]].push_back(cnt);
        vtx_edges[e[1]].push_back(cnt);
        cnt++;
    });
}

void ObjNucleiTracking::computed2Odx2(const VectorXT& x, std::vector<Entry>& d2Odx2_entries)
{
    simulation.deformed = x;
    int p = power / 2;
    if (match_centroid)
    {
        iterateTargets([&](int cell_idx, TV& target_pos)
        {
            VtxList face_vtx_list = simulation.cells.faces[cell_idx];
            VtxList cell_vtx_list = face_vtx_list;
            for (int idx : face_vtx_list)
                cell_vtx_list.push_back(idx + simulation.cells.basal_vtx_start);
            
            TV centroid;
            simulation.cells.computeCellCentroid(face_vtx_list, centroid);

            TV x_minus_x = centroid - target_pos;
            T xTx = x_minus_x.dot(x_minus_x);
            
            T coeff = cell_vtx_list.size();
            TM dcdx = TM::Identity() / coeff;
            TM d2Odc2;
            TM tensor_term = TM::Zero(); // c is linear in x
            TM local_hessian;
            if (power > 2)
            {
                d2Odc2 = 2.0 * p * std::pow(xTx, p - 1) * TM::Identity();
                d2Odc2 += 2.0 * p * (p - 1) * std::pow(xTx, p - 2) * 2.0 * x_minus_x * x_minus_x.transpose();
                d2Odc2 *= target_obj_weights[cell_idx];
            }
            else
            {
                d2Odc2 = TM::Identity() * 2.0 * target_obj_weights[cell_idx];
            }
            
            local_hessian = dcdx.transpose() * d2Odc2 * dcdx + tensor_term;
            

            for (int idx_i : cell_vtx_list)
                for (int idx_j : cell_vtx_list)
                    for (int d = 0; d < 3; d++)
                        for (int dd = 0; dd < 3; dd++)
                            d2Odx2_entries.push_back(Entry(idx_i * 3 + d, idx_j * 3 + dd, local_hessian(d, dd)));
                        
                    
        });
    }
    else
    {
        iterateWeightedTargets([&](int cell_idx, int data_point_idx, 
            const TV& target, const VectorXT& weights)
        {
            VectorXT positions;
            VtxList indices;
            simulation.cells.getCellVtxAndIdx(cell_idx, positions, indices);

            TV current = TV::Zero();
            int n_pt = weights.rows();

            for (int i = 0; i < n_pt; i++)
                current += weights[i] * positions.segment<3>(i * 3);

            TV x_minus_x = current - target;
            T xTx = x_minus_x.dot(x_minus_x);          

            for (int i = 0; i < indices.size(); i++)
                for (int j = 0; j < indices.size(); j++)
                {
                    TM dcdx0 = TM::Identity() * weights[i];
                    TM dcdx1 = TM::Identity() * weights[j];

                    TM tensor_term = TM::Zero(); // c is linear in x
                    TM local_hessian;

                    TM d2Odc2;
                    if (power == 2)
                        d2Odc2 = TM::Identity() * 2.0 * target_obj_weights[cell_idx];
                    else
                    {
                        d2Odc2 = 2.0 * p * std::pow(xTx, p - 1) * TM::Identity();
                        d2Odc2 += 2.0 * p * (p - 1) * std::pow(xTx, p - 2) * 2.0 * x_minus_x * x_minus_x.transpose();
                        d2Odc2 *= target_obj_weights[cell_idx];
                    }
                    
                    local_hessian = dcdx0.transpose() * d2Odc2 * dcdx1 + tensor_term;

                    for (int d = 0; d < 3; d++)
                        for (int dd = 0; dd < 3; dd++)
                            d2Odx2_entries.push_back(Entry(indices[i] * 3 + d, indices[j] * 3 + dd, local_hessian(d, dd)));
                }
        });
    }
    
    if (add_spatial_x)
    {
        // auto addd2Cdx2 = [&](T _coeff, int cell_idx)
        // {
        //     VectorXT positions;
        //     VtxList indices;
        //     simulation.cells.getCellVtxAndIdx(cell_idx, positions, indices);
        //     T dcdx = 1.0 / T(indices.size());
        //     TM local_hessian = _coeff * TM::Identity() * dcdx;

        //     for (int idx_i : indices)
        //         for (int idx_j : indices)
        //             for (int d = 0; d < 3; d++)
        //                 for (int dd = 0; dd < 3; dd++)
        //                     d2Odx2_entries.push_back(Entry(idx_i * 3 + d, idx_j * 3 + dd, local_hessian(d, dd)));
        // };

        

        // VectorXT cell_terms(simulation.cells.n_cells);
        // for (int i = 0; i < simulation.cells.n_cells; i++)
        // {
        //     TV ci = cell_centroids.segment<3>(i * 3);
        //     TV avg = TV::Zero();
        //     for (int idx : cell_neighbors[i])
        //     {
        //         if (idx == i)
        //             continue;
        //         avg += cell_centroids.segment<3>(idx * 3);
        //     }
        //     avg /= T(cell_neighbors[i].size() - 1);
            
        //     T coeff = 1.0 / T(cell_neighbors[i].size() - 1);
        //     addd2Cdx2(w_reg_x_spacial, i);
        //     for (int idx : cell_neighbors[i])
        //     {
        //         if (idx == i)
        //             continue;
        //         addd2Cdx2(-w_reg_x_spacial * coeff, idx);
        //     }
        // }
    }

    if (add_forward_potential)
    {
        VectorXT dx = simulation.deformed - simulation.undeformed;
        StiffnessMatrix sim_H(n_dof_sim, n_dof_sim);
        simulation.buildSystemMatrix(dx, sim_H);
        sim_H *= w_fp;
        std::vector<Entry> sim_H_entries = simulation.cells.entriesFromSparseMatrix(sim_H);
        d2Odx2_entries.insert(d2Odx2_entries.end(), sim_H_entries.begin(), sim_H_entries.end());
    }
}

void ObjNucleiTracking::computeEnergyAllTerms(const VectorXT& p_curr, std::vector<T>& energies,
        bool simulate, bool use_prev_equil)
{
    T e_matching = 0.0, e_sim = 0.0, e_reg = 0.0;
    
    int p = power / 2;
    if (match_centroid)
    {
        iterateTargets([&](int cell_idx, TV& target_pos)
        {
            VtxList face_vtx_list = simulation.cells.faces[cell_idx];
            TV centroid;
            simulation.cells.computeCellCentroid(face_vtx_list, centroid);
            // Ox += 0.5 * (centroid - target_pos).dot(centroid - target_pos) * target_obj_weights[cell_idx];
            T xTx = (centroid - target_pos).dot(centroid - target_pos);
            e_matching += std::pow(xTx, p) * target_obj_weights[cell_idx];
        });
    }
    else
    {
        iterateWeightedTargets([&](int cell_idx, int data_point_idx, 
            const TV& target, const VectorXT& weights)
        {
            VectorXT positions;
            VtxList indices;
            simulation.cells.getCellVtxAndIdx(cell_idx, positions, indices);

            int n_pt = weights.rows();
            TV current = TV::Zero();
            for (int i = 0; i < n_pt; i++)
                current += weights[i] * positions.segment<3>(i * 3);
            // Ox += 0.5 * (current - target).dot(current - target) * target_obj_weights[cell_idx];
            T xTx = (current - target).dot(current - target);
            
            e_matching += std::pow(xTx, p) * target_obj_weights[cell_idx];
        });
    }

    if (add_forward_potential)
    {
        VectorXT dx = simulation.deformed - simulation.undeformed;
        T simulation_potential = simulation.computeTotalEnergy(dx);
        simulation_potential *= w_fp;
        e_sim += simulation_potential;
    }

    T penalty_term = 0.0;
    if (use_penalty)
    {
        for (int i = 0; i < n_dof_design; i++)
        {
            if (penalty_type == Qubic)
            {
                if (p_curr[i] < bound[0] && mask[0])
                    penalty_term += penalty_weight * std::pow(-(p_curr[i] - bound[0]), 3);
                if (p_curr[i] > bound[1] && mask[1])
                    penalty_term += penalty_weight * std::pow(p_curr[i] - bound[1], 3);
            }
        }
    }

    if (add_spatial_regularizor)
    {
        for (auto& vtx_edge : vtx_edges)
        {
            T w_size = 1.0 / vtx_edge.size();
            for (int i = 0; i < vtx_edge.size(); i++)
            {
                int j = (i + 1) % vtx_edge.size();
                e_reg += 0.5 * w_reg_spacial * w_size * std::pow(p_curr[vtx_edge[i]] - p_curr[vtx_edge[j]], 2); 
            }
        }
    }
    
    energies = { e_matching, e_sim, penalty_term, e_reg };
}

void ObjNucleiTracking::computeOx(const VectorXT& x, T& Ox)
{
    Ox = 0.0;
    simulation.deformed = x;
    int p = power / 2;
    T min_dis = 1e10, max_dis = -1e10, avg_dis = 0;
    int cnt = 0;
    if (match_centroid)
    {
        iterateTargets([&](int cell_idx, TV& target_pos)
        {
            VtxList face_vtx_list = simulation.cells.faces[cell_idx];
            TV centroid;
            simulation.cells.computeCellCentroid(face_vtx_list, centroid);
            // Ox += 0.5 * (centroid - target_pos).dot(centroid - target_pos) * target_obj_weights[cell_idx];
            T xTx = (centroid - target_pos).dot(centroid - target_pos);
            Ox += std::pow(xTx, p) * target_obj_weights[cell_idx];
        });
    }
    else
    {
        iterateWeightedTargets([&](int cell_idx, int data_point_idx, 
            const TV& target, const VectorXT& weights)
        {
            VectorXT positions;
            VtxList indices;
            simulation.cells.getCellVtxAndIdx(cell_idx, positions, indices);

            int n_pt = weights.rows();
            TV current = TV::Zero();
            for (int i = 0; i < n_pt; i++)
                current += weights[i] * positions.segment<3>(i * 3);
            // Ox += 0.5 * (current - target).dot(current - target) * target_obj_weights[cell_idx];
            T xTx = (current - target).dot(current - target);
            
            T dis = (current - target).norm();
            if (dis > max_dis) max_dis = dis;
            if (dis < min_dis) min_dis = dis;
            avg_dis += dis;
            cnt++;
            Ox += std::pow(xTx, p) * target_obj_weights[cell_idx];
        });
        avg_dis /= T(cnt);
    }
    if (add_spatial_x)
    {
        VectorXT cell_centroids;
        simulation.cells.getAllCellCentroids(cell_centroids);

        VectorXT cell_terms(simulation.cells.n_cells);
        cell_terms.setZero();
        tbb::parallel_for(0, simulation.cells.n_cells, [&](int i){
            if (target_obj_weights[i] > 1e-6)
                return;
            TV ci = cell_centroids.segment<3>(i * 3);
            TV avg = TV::Zero();
            for (int idx : cell_neighbors[i])
            {
                if (idx == i)
                    continue;
                avg += cell_centroids.segment<3>(idx * 3);
            }
            avg /= T(cell_neighbors[i].size() - 1);
            cell_terms[i] = 0.5 * w_reg_x_spacial * (ci - avg).dot(ci - avg);
        });
        Ox += cell_terms.sum();
    }
    // std::cout << "======================================" << std::endl;
    // std::cout << "min dis " << min_dis << " max_dis " << max_dis << " avg_dis " << avg_dis << std::endl;
    // std::cout << "min dis power" << std::pow(min_dis, power) << " max_dis power " 
    //     << std::pow(max_dis, power) << " avg_dis power " << std::pow(avg_dis, power) << std::endl;
    // std::cout << "======================================" << std::endl;
    if (add_forward_potential)
    {
        VectorXT dx = simulation.deformed - simulation.undeformed;
        T simulation_potential = simulation.computeTotalEnergy(dx);
        simulation_potential *= w_fp;
        Ox += simulation_potential;
        // std::cout << "constracting energy: " << simulation_potential << std::endl;
    }
}

void ObjNucleiTracking::computedOdx(const VectorXT& x, VectorXT& dOdx)
{
    simulation.deformed = x;
    dOdx.resize(n_dof_sim);
    dOdx.setZero();
    int p = power / 2;
    if (match_centroid)
    {
        iterateTargets([&](int cell_idx, TV& target_pos)
        {
            VtxList face_vtx_list = simulation.cells.faces[cell_idx];
            VtxList cell_vtx_list = face_vtx_list;
            for (int idx : face_vtx_list)
                cell_vtx_list.push_back(idx + simulation.cells.basal_vtx_start);
            TV centroid;
            simulation.cells.computeCellCentroid(face_vtx_list, centroid);
            T dcdx = 1.0 / cell_vtx_list.size();
            TV x_minus_x = centroid - target_pos;
            T xTx = x_minus_x.dot(x_minus_x);
            for (int idx : cell_vtx_list)
            {
                TV dOdc = p * std::pow(xTx, p - 1) * 2.0 * x_minus_x;
                dOdx.segment<3>(idx * 3) += dOdc * dcdx * target_obj_weights[cell_idx];
            }
        });
    }
    else
    {
        iterateWeightedTargets([&](int cell_idx, int data_point_idx, 
            const TV& target, const VectorXT& weights)
        {
            VectorXT positions;
            VtxList indices;
            simulation.cells.getCellVtxAndIdx(cell_idx, positions, indices);

            int n_pt = weights.rows();
            TV current = TV::Zero();
            for (int i = 0; i < n_pt; i++)
                current += weights[i] * positions.segment<3>(i * 3);

            TV x_minus_x = current - target;
            T xTx = x_minus_x.dot(x_minus_x);
            TV dOdc = power / 2 * std::pow(xTx, power / 2 - 1) * 2.0 * x_minus_x;
            for (int i = 0; i < indices.size(); i++)
            {
                T dcdx = weights[i];
                dOdx.segment<3>(indices[i] * 3) += dOdc * dcdx * target_obj_weights[cell_idx];
            }
        });
    }

    if (add_spatial_x)
    {
        auto adddCdx = [&](const TV& dOdc, int cell_idx)
        {
            VectorXT positions;
            VtxList indices;
            simulation.cells.getCellVtxAndIdx(cell_idx, positions, indices);
            T dcdx = 1.0 / T(indices.size());
            for (int i = 0; i < indices.size(); i++)
            {
                dOdx.segment<3>(indices[i] * 3) += dOdc * dcdx;
            }
        };

        VectorXT cell_centroids;
        simulation.cells.getAllCellCentroids(cell_centroids);

        VectorXT cell_terms(simulation.cells.n_cells);
        for (int i = 0; i < simulation.cells.n_cells; i++)
        {
            if (target_obj_weights[i] > 1e-6)
                continue;

            TV ci = cell_centroids.segment<3>(i * 3);
            TV avg = TV::Zero();
            // std::cout << cell_neighbors[i].size() << std::endl;
            // std::ofstream out("neighbor_test.obj");
            // out << "v " << ci.transpose() << std::endl;
            for (int idx : cell_neighbors[i])
            {
                if (idx == i)
                    continue;
                avg += cell_centroids.segment<3>(idx * 3);
                // out << "v " << cell_centroids.segment<3>(idx * 3).transpose() << std::endl;
            }
            avg /= T(cell_neighbors[i].size() - 1);
            // out << "v " << avg.transpose() << std::endl;
            // out.close();
            // std::exit(0);
            T coeff = 1.0 / T(cell_neighbors[i].size() - 1);
            TV dOdci = w_reg_x_spacial * (ci - avg);
            
            adddCdx(dOdci, i);
            for (int idx : cell_neighbors[i])
            {
                if (idx == i)
                    continue;
                adddCdx(dOdci * -coeff, idx);
            }
        }
    }
    
    if (add_forward_potential)
    {
        
        VectorXT cell_forces(n_dof_sim); cell_forces.setZero();
        VectorXT dx = simulation.deformed - simulation.undeformed;
        simulation.computeResidual(dx, cell_forces);
        dOdx -= w_fp * cell_forces;
    }
}

void ObjNucleiTracking::computeOp(const VectorXT& p_curr, T& Op)
{
    Op = 0.0;

    if (add_reg)
    {
        T reg_term = 0.5 * reg_w * (p_curr - prev_params).dot(p_curr - prev_params);
        Op += reg_term;
    }

    if (use_penalty)
    {
        T penalty_term = 0.0;
        for (int i = 0; i < n_dof_design; i++)
        {
            if (penalty_type == Qubic)
            {
                if (p_curr[i] < bound[0] && mask[0])
                    penalty_term += penalty_weight * std::pow(-(p_curr[i] - bound[0]), 3);
                if (p_curr[i] > bound[1] && mask[1])
                    penalty_term += penalty_weight * std::pow(p_curr[i] - bound[1], 3);
            }
        }
        Op += penalty_term;
    }

    if (add_spatial_regularizor)
    {
        for (auto& vtx_edge : vtx_edges)
        {
            T w_size = 1.0 / vtx_edge.size();
            for (int i = 0; i < vtx_edge.size(); i++)
            {
                int j = (i + 1) % vtx_edge.size();
                Op += 0.5 * w_reg_spacial * w_size * std::pow(p_curr[vtx_edge[i]] - p_curr[vtx_edge[j]], 2); 
            }
        }
    }
}

void ObjNucleiTracking::computedOdp(const VectorXT& p_curr, VectorXT& dOdp)
{
    dOdp = VectorXT::Zero(n_dof_design);
    if (add_reg)
    {
        dOdp += reg_w * (p_curr - prev_params);
    }

    if (use_penalty)
    {
        for (int i = 0; i < n_dof_design; i++)
        {
            if (penalty_type == Qubic)
            {
                if (p_curr[i] < bound[0] && mask[0])
                {
                    dOdp[i] += -penalty_weight * 3.0 * std::pow(-(p_curr[i] - bound[0]), 2);
                }
                if (p_curr[i] > bound[1] && mask[1])
                {
                    dOdp[i] += penalty_weight * 3.0 * std::pow((p_curr[i] - bound[1]), 2);
                }
            }
        }
    }
    if (add_spatial_regularizor)
    {
        for (auto& vtx_edge : vtx_edges)
        {
            T w_size = 1.0 / vtx_edge.size();
            for (int i = 0; i < vtx_edge.size(); i++)
            {
                int j = (i + 1) % vtx_edge.size();
                dOdp[vtx_edge[i]] += w_reg_spacial * w_size * (p_curr[vtx_edge[i]] - p_curr[vtx_edge[j]]);
                dOdp[vtx_edge[j]] -= w_reg_spacial * w_size * (p_curr[vtx_edge[i]] - p_curr[vtx_edge[j]]);
            }
        }
    }
}

void ObjNucleiTracking::computed2Odp2(const VectorXT& p_curr, std::vector<Entry>& d2Odp2_entries)
{
    if (add_reg)
    {
        for (int i = 0; i < n_dof_design; i++)
        {
            d2Odp2_entries.push_back(Entry(i, i, reg_w));
        }
    }
    
    if (use_penalty)
    {
        for (int i = 0; i < n_dof_design; i++)
        {
            if (penalty_type == Qubic)
            {
                if (p_curr[i] < bound[0] && mask[0])
                {
                    T vi = -(p_curr[i] - bound[0]);
                    d2Odp2_entries.push_back(Entry(i, i, penalty_weight * 6.0 * vi));
                }
                if (p_curr[i] > bound[1] && mask[1])
                {
                    T vi = (p_curr[i] - bound[1]);
                    d2Odp2_entries.push_back(Entry(i, i, penalty_weight * 6.0 * vi));
                }
            }
        }
    }

    if (add_spatial_regularizor)
    {
        for (auto& vtx_edge : vtx_edges)
        {
            T w_size = 1.0 / vtx_edge.size();
            for (int i = 0; i < vtx_edge.size(); i++)
            {
                int j = (i + 1) % vtx_edge.size();
                d2Odp2_entries.push_back(Entry(vtx_edge[i], vtx_edge[i], w_size * w_reg_spacial));
                d2Odp2_entries.push_back(Entry(vtx_edge[j], vtx_edge[j], w_size * w_reg_spacial));
                d2Odp2_entries.push_back(Entry(vtx_edge[i], vtx_edge[j], -w_size * w_reg_spacial));
                d2Odp2_entries.push_back(Entry(vtx_edge[j], vtx_edge[i], -w_size * w_reg_spacial));
            }
        }
    }
}

void ObjNucleiTracking::checkDistanceMetrics()
{   
    T total_length = 0.0;
    int cnt = 0;
    if (match_centroid)
    {
        iterateTargets([&](int cell_idx, TV& target_pos)
        {
            VtxList face_vtx_list = simulation.cells.faces[cell_idx];
            TV centroid;
            simulation.cells.computeCellCentroid(face_vtx_list, centroid);
            total_length += (centroid - target_pos).norm();
            cnt++;
        });
    }
    else
    {
        iterateWeightedTargets([&](int cell_idx, int data_point_idx, 
            const TV& target, const VectorXT& weights)
        {
            VectorXT positions;
            VtxList indices;
            simulation.cells.getCellVtxAndIdx(cell_idx, positions, indices);

            int n_pt = weights.rows();
            TV current = TV::Zero();
            for (int i = 0; i < n_pt; i++)
                current += weights[i] * positions.segment<3>(i * 3);
            total_length += (current - target).norm();
            cnt++;
        });
    }
    T avg_length = total_length / T(cnt);
    std::cout << "Total distance: " << total_length << " avg distance: " << avg_length << std::endl;
    std::cout << "Total distance in micrometers: " << total_length / simulation.cells.unit 
        << " avg distance in micrometers: " << avg_length / simulation.cells.unit << std::endl;
}

T ObjNucleiTracking::value(const VectorXT& p_curr, bool simulate, bool use_prev_equil)
{
    // simulation.loadDeformedState("current_mesh.obj");
    
    
    updateDesignParameters(p_curr);

    T offset = 0.2 * (bound[1] - bound[0]);
    // std::cout << p_curr.maxCoeff()<< " " << bound[1] + offset << " " << p_curr.minCoeff() << " " << bound[0] - offset << std::endl;
    if (p_curr.maxCoeff() > bound[1] + offset || p_curr.minCoeff() < bound[0] - offset)
        return 1e10;
    // if (p_wrap.maxCoeff() > bound[1] + offset || p_wrap.minCoeff() < -1)
    //     return 1e10;

    if (simulate)
    {
        simulation.reset();
        if (use_prev_equil)
            simulation.u = equilibrium_prev;
        while (true)
        {
            // simulation.loadDeformedState("current_mesh.obj");
            bool forward_simulation_converged = simulation.staticSolve();
            if (!forward_simulation_converged)
            {
                // saveDesignParameters("failed.txt", p_curr);
                // VectorXT deformed_curr = simulation.deformed;
                // simulation.deformed = simulation.undeformed + equilibrium_prev;
                // std::cout << "use_prev_equil " << use_prev_equil << std::endl;
                // std::cout << equilibrium_prev.norm() << std::endl;
                // simulation.saveState("failed.obj", false, false);
                // simulation.deformed = simulation.undeformed + simulation.u;
                // std::exit(0);
                return 1e3;
            }
            if (!perturb)
                break;
            VectorXT negative_eigen_vector;
            T negative_eigen_value;
            bool has_neg_ev = simulation.fetchNegativeEigenVectorIfAny(negative_eigen_value,
                negative_eigen_vector);
            if (has_neg_ev)
            {
                std::cout << "unstable state for the forward problem" << std::endl;
                std::cout << "nodge it along the negative eigen vector" << std::endl;
                VectorXT nodge_direction = negative_eigen_vector;
                T step_size = 0.1 * simulation.cells.computeLineSearchInitStepsize(simulation.u, nodge_direction, false);
                simulation.u += step_size * nodge_direction;
            }
            else
                break;
        }
    }

    T energy = 0.0;
    computeOx(simulation.deformed, energy);
    
    T Op; computeOp(p_curr, Op);
    energy += Op;

    return energy;
}


T ObjNucleiTracking::gradient(const VectorXT& p_curr, VectorXT& dOdp, T& energy, bool simulate, bool use_prev_equil)
{
    // simulation.loadDeformedState("current_mesh.obj");
    
    updateDesignParameters(p_curr);
    if (simulate)
    {
        simulation.reset();
        if (use_prev_equil)
            simulation.u = equilibrium_prev;
        while (true)
        {
            // simulation.loadDeformedState("current_mesh.obj");
            simulation.staticSolve();
            if (!perturb)
                break;
            VectorXT negative_eigen_vector;
            T negative_eigen_value;
            bool has_neg_ev = simulation.fetchNegativeEigenVectorIfAny(negative_eigen_value,
                negative_eigen_vector);
            if (has_neg_ev)
            {
                std::cout << "unstable state for the forward problem" << std::endl;
                std::cout << "nodge it along the negative eigen vector" << std::endl;
                VectorXT nodge_direction = negative_eigen_vector;
                T step_size = 0.1 * simulation.cells.computeLineSearchInitStepsize(simulation.u, nodge_direction, false);
                simulation.u += step_size * nodge_direction;
            }
            else
                break;
        }
    }
    
    energy = 0.0;
    VectorXT dOdx;

    computeOx(simulation.deformed, energy);
    computedOdx(simulation.deformed, dOdx);
    
    StiffnessMatrix d2edx2(n_dof_sim, n_dof_sim);
    simulation.buildSystemMatrix(simulation.u, d2edx2);
    
    simulation.cells.iterateDirichletDoF([&](int offset, T target)
    {
        dOdx[offset] = 0;
    });
    
    VectorXT lambda;
    Eigen::CholmodSupernodalLLT<StiffnessMatrix, Eigen::Lower> solver;
    // Eigen::PardisoLLT<StiffnessMatrix, Eigen::Lower> solver;
    solver.compute(d2edx2);
    lambda = solver.solve(dOdx);

    
    // std::cout << " |gradient| linear solve " << (d2edx2 * lambda - dOdx).norm() / dOdx.norm() << std::endl;
    simulation.cells.dOdpEdgeWeightsFromLambda(lambda, dOdp);
    // std::cout << "|dOdp matching|: " << dOdp.norm();
    // VectorXT dOdp_tmp = dOdp;
    // \partial O \partial p for the edge contracting force
    {
        if (add_forward_potential)
        {
            VectorXT dOdp_force(n_dof_design); dOdp_force.setZero();
            simulation.cells.computededp(dOdp_force);
            dOdp += w_fp * dOdp_force;
        }
    }

    // std::exit(0);
    if (!use_prev_equil)
        equilibrium_prev = simulation.u;

    VectorXT partialO_partialp;
    computedOdp(p_curr, partialO_partialp);

    T Op; computeOp(p_curr, Op);
    dOdp += partialO_partialp;
    energy += Op;
    
    return dOdp.norm();
}

void ObjNucleiTracking::updateDesignParameters(const VectorXT& design_parameters)
{
    simulation.cells.edge_weights = design_parameters;
}

void ObjNucleiTracking::getDesignParameters(VectorXT& design_parameters)
{
    design_parameters = simulation.cells.edge_weights;
}

void ObjNucleiTracking::getSimulationAndDesignDoF(int& _sim_dof, int& _design_dof)
{
    _sim_dof = simulation.num_nodes * 3;
    _design_dof = simulation.cells.edge_weights.rows();
    n_dof_sim = _sim_dof;
    n_dof_design = _design_dof;
}

void ObjNucleiTracking::SGNforQP(const VectorXT& p_curr, 
        StiffnessMatrix& objective_hessian, 
        StiffnessMatrix& constraint_jacobian)
{
    updateDesignParameters(p_curr);
    
    int nx = n_dof_sim;

    StiffnessMatrix dfdx(n_dof_sim, n_dof_sim);
    simulation.buildSystemMatrix(simulation.u, dfdx);
    dfdx *= -1.0;
    StiffnessMatrix dfdp;
    simulation.cells.dfdpWeightsSparse(dfdp);

    std::vector<Entry> entries_constraint;
    for (int i = 0; i < dfdx.outerSize(); i++)
    {
        for (StiffnessMatrix::InnerIterator it(dfdx, i); it; ++it)
        {
            entries_constraint.push_back(Entry(it.row(), it.col(), it.value()));
        }
    }
    
    for (int i = 0; i < dfdp.outerSize(); i++)
    {
        for (StiffnessMatrix::InnerIterator it(dfdp, i); it; ++it)
        {
            entries_constraint.push_back(Entry(it.row(), it.col() + nx, it.value()));
        }
    }
    constraint_jacobian.resize(n_dof_sim, n_dof_design + n_dof_sim);
    constraint_jacobian.setFromTriplets(entries_constraint.begin(), entries_constraint.end());
    std::cout << "constraint_jacobian" << std::endl;
    objective_hessian.resize(n_dof_design + n_dof_sim, n_dof_design + n_dof_sim);
    std::vector<Entry> hessian_entries;
    computed2Odx2(simulation.deformed, hessian_entries);
    
    for (int i = 0; i < n_dof_sim + n_dof_design; i++)
        hessian_entries.push_back(Entry(i, i, 1e-8));

    if (add_reg)
    {
        for (int i = 0; i < n_dof_design; i++)
            hessian_entries.push_back(Entry(i + n_dof_sim, i + n_dof_sim, reg_w));
    }
    if (add_forward_potential)
    {
        for (int i = 0; i < dfdp.outerSize(); i++)
        {
            for (StiffnessMatrix::InnerIterator it(dfdp, i); it; ++it)
            {
                hessian_entries.push_back(Entry(it.row(), it.col() + nx, -w_fp * it.value()));
                hessian_entries.push_back(Entry(it.col() + nx, it.row(), -w_fp * it.value()));
            }
        }
    }

    objective_hessian.setFromTriplets(hessian_entries.begin(), hessian_entries.end());
    std::cout << "objective_hessian" << std::endl;
}

void ObjNucleiTracking::hessianSGN(const VectorXT& p_curr, 
    StiffnessMatrix& H, bool simulate)
{

    updateDesignParameters(p_curr);
    if (simulate)
    {
        simulation.reset();
        simulation.staticSolve();
    }

    std::vector<Entry> d2Odx2_entries;
    computed2Odx2(simulation.deformed, d2Odx2_entries);

    StiffnessMatrix dfdx(n_dof_sim, n_dof_sim);
    simulation.buildSystemMatrix(simulation.u, dfdx);
    dfdx *= -1.0;

    StiffnessMatrix dfdp;
    simulation.cells.dfdpWeightsSparse(dfdp);
    

    StiffnessMatrix d2Odx2_mat(n_dof_sim, n_dof_sim);
    d2Odx2_mat.setFromTriplets(d2Odx2_entries.begin(), d2Odx2_entries.end());

    int nx = n_dof_sim, np = n_dof_design, nxnp = n_dof_sim + n_dof_design;

    H.resize(n_dof_sim * 2 + n_dof_design, n_dof_sim * 2 + n_dof_design);
    std::vector<Entry> entries;
    
    entries.insert(entries.end(), d2Odx2_entries.begin(), d2Odx2_entries.end());

    for (int i = 0; i < dfdx.outerSize(); i++)
    {
        for (StiffnessMatrix::InnerIterator it(dfdx, i); it; ++it)
        {
            entries.push_back(Entry(it.row() + nxnp, it.col(), it.value()));
            entries.push_back(Entry(it.row(), it.col() + nxnp, it.value()));
        }
    }
    
    for (int i = 0; i < dfdp.outerSize(); i++)
    {
        for (StiffnessMatrix::InnerIterator it(dfdp, i); it; ++it)
        {
            entries.push_back(Entry(it.col() + nx, it.row() + nxnp, it.value() * wrapper<1>(p_curr[it.col()])));
            entries.push_back(Entry(it.row() + nxnp, it.col() + nx, it.value() * wrapper<1>(p_curr[it.col()])));

            if (add_forward_potential)
            {
                entries.push_back(Entry(it.row(), it.col() + nx, -w_fp * it.value() * wrapper<1>(p_curr[it.col()])));
                entries.push_back(Entry(it.col() + nx, it.row(), -w_fp * it.value() * wrapper<1>(p_curr[it.col()])));
            }
        }
    }

    std::vector<Entry> d2Odp2_entries;
    computed2Odp2(p_curr, d2Odp2_entries);

    for (auto entry : d2Odp2_entries)
        entries.push_back(Entry(entry.row() + n_dof_sim, 
                                entry.col() + n_dof_sim, 
                                entry.value() * wrapper<2>(p_curr[entry.row()])));

    for (int i = 0; i < n_dof_sim; i++)
        entries.push_back(Entry(i, i, 1e-10));
    for (int i = 0; i < n_dof_design; i++)
        entries.push_back(Entry(i + n_dof_sim, i + n_dof_sim, 1e-10));
    for (int i = 0; i < n_dof_sim; i++)
        entries.push_back(Entry(i + n_dof_sim + n_dof_design, i + n_dof_sim + n_dof_design, -1e-10));
    
    
    H.setFromTriplets(entries.begin(), entries.end());
    H.makeCompressed();
}

void ObjNucleiTracking::hessianGN(const VectorXT& p_curr, MatrixXT& H, bool simulate, bool use_prev_equil)
{
    updateDesignParameters(p_curr);
    
    if (simulate)
    {
        simulation.reset();
        if (use_prev_equil)
            simulation.u = equilibrium_prev;
        while (true)
        {
            // simulation.loadDeformedState("current_mesh.obj");
            simulation.staticSolve();
            if (!perturb)
                break;
            VectorXT negative_eigen_vector;
            T negative_eigen_value;
            bool has_neg_ev = simulation.fetchNegativeEigenVectorIfAny(negative_eigen_value,
                negative_eigen_vector);
            if (has_neg_ev)
            {
                std::cout << "unstable state for the forward problem" << std::endl;
                std::cout << "nodge it along the negative eigen vector" << std::endl;
                VectorXT nodge_direction = negative_eigen_vector;
                T step_size = 0.1 * simulation.cells.computeLineSearchInitStepsize(simulation.u, nodge_direction, false);
                simulation.u += step_size * nodge_direction;
            }
            else
                break;
        }
    }
    
    MatrixXT dxdp;
    
    Timer tt(true);
    simulation.cells.dxdpFromdxdpEdgeWeights(dxdp);
    std::cout << "compute dxdp takes " << tt.elapsed_sec() << "s" << std::endl;
    
    std::vector<Entry> d2Odx2_entries;
    computed2Odx2(simulation.deformed, d2Odx2_entries);
    StiffnessMatrix d2Odx2_matrix(n_dof_sim, n_dof_sim);
    d2Odx2_matrix.setFromTriplets(d2Odx2_entries.begin(), d2Odx2_entries.end());
    simulation.cells.projectDirichletDoFMatrix(d2Odx2_matrix, simulation.cells.dirichlet_data);
    
    // Eigen::JacobiSVD<Eigen::MatrixXd> svd(dxdp, Eigen::ComputeThinU | Eigen::ComputeThinV);
	// MatrixXT U = svd.matrixU();
	// VectorXT Sigma = svd.singularValues();
	// MatrixXT V = svd.matrixV();

    // std::cout << Sigma.tail<10>().transpose() << std::endl;
    // Eigen::SparseMatrix<T, Eigen::RowMajor, long int> d2Odx2_cholmod = d2Odx2_matrix.triangularView<Eigen::Upper>();
    // std::ofstream out("d2Odx2_cholmod.txt");
    // out << d2Odx2_cholmod << std::endl;
    // out.close();
    // MatrixXT d2Odx2_dense = d2Odx2_matrix;
    // std::cout << d2Odx2_dense.minCoeff() << " " << d2Odx2_dense.maxCoeff() << std::endl;
    
    // d2Odx2_matrix.setIdentity();
    // H.noalias() = dxdp.transpose() * d2Odx2_matrix * dxdp;
    tt.restart();

    // cholmod_common cm;
    // cholmod_l_defaults(&cm);
    // cholmod_l_start(&cm);

    // cm.useGPU = false;

    // cholmod_sparse* A = NULL;
    // cholmod_dense* B = NULL;
    // cholmod_dense* AB = NULL;
    // int numRows = d2Odx2_cholmod.rows();
    
    // if (!A) {
    //     A = cholmod_l_allocate_sparse(numRows, numRows, d2Odx2_cholmod.nonZeros(),
    //         true, true, -1, CHOLMOD_REAL, &cm);
    //     A->p = d2Odx2_cholmod.outerIndexPtr();
    //     A->i = d2Odx2_cholmod.innerIndexPtr();
    //     A->x = d2Odx2_cholmod.valuePtr();
    // }
    // if (!B)
    // {
    //     B = cholmod_l_allocate_dense(dxdp.rows(), dxdp.cols(), dxdp.rows(), CHOLMOD_REAL, &cm);
    //     B->x = (void*)dxdp.data();
    // }

    // double alpha[2] = { 1.0, 1.0 }, beta[2] = { 0.0, 0.0 };

    // AB = cholmod_l_allocate_dense(dxdp.rows(), dxdp.cols(), dxdp.rows(), CHOLMOD_REAL, &cm);
    
    // cholmod_l_sdmult(A, 0, alpha, beta, B, AB, &cm);
    
    // MatrixXT d2Odx2_dxdp(dxdp.rows(), dxdp.cols());
    // memcpy(d2Odx2_dxdp.data(), AB->x, dxdp.rows() * dxdp.cols() * sizeof(double));
    
    // // std::cout << "memcpy" << std::endl;
    // cholmod_l_free_dense(&AB, &cm);
    // // MatrixXT gt = d2Odx2_matrix * dxdp;
    // // std::ofstream out("gt.txt");
    // // out << gt << std::endl;
    
    // // out.close();
    // // out.open("mine.txt");
    // // out << d2Odx2_dxdp;
    // // out.close();
    // // std::cout << gt.col(0).segment<20>(0) << std::endl << d2Odx2_dxdp.col(0).segment<20>(0) << std::endl;
    // // std::cout << (gt.col(0) - d2Odx2_dxdp.col(0)).norm() << std::endl; 
    
    // // std::cout << "ERROR " << (d2Odx2_dxdp - gt).norm() << std::endl;

    // if (A) cholmod_l_free_sparse(&A, &cm);
    // // if (B) cholmod_l_free_dense(&B, &cm);    
    // cholmod_l_finish(&cm);
    // H.noalias() = dxdp.transpose() * d2Odx2_dxdp;
    H = dxdp.transpose() * d2Odx2_matrix * dxdp;
    std::cout << "dxdpT H dxdp takes " << tt.elapsed_sec() << "s" << std::endl;

    // 2 dx/dp^T d2O/dxdp
    if (add_forward_potential)
    {
        MatrixXT dfdp;
        simulation.cells.dfdpWeightsDense(dfdp);
        dfdp *= -w_fp;
        // H += dxdp.transpose() * dfdp + dfdp.transpose() * dxdp;
        H.noalias() += 2.0 * dfdp.transpose() * dxdp;
    }

    std::vector<Entry> d2Odp2_entries;
    computed2Odp2(p_curr, d2Odp2_entries);

    for (auto entry : d2Odp2_entries)
        H(entry.row(), entry.col()) += entry.value();
}

T ObjNucleiTracking::maximumStepSize(const VectorXT& dp)
{
    return 1.0;
    VectorXT p_curr, p_wrap;
    getDesignParameters(p_curr);
    
    T step_size = 1.0;
    while (true)
    {
        VectorXT forward = p_curr + step_size * dp;
        p_wrap = forward;
        for (int i = 0; i < n_dof_design; i++)
        {
            p_wrap[i] = wrapper<0>(forward[i]);
        }
        std::cout << "p_wrap min: " << p_wrap.minCoeff() << " max: " << p_wrap.maxCoeff() << std::endl;
        if (p_wrap.maxCoeff() > p_curr.maxCoeff() + bound[1])
            step_size *= 0.5;
        else
            return step_size;
    }
    
}

bool ObjNucleiTracking::getTargetTrajectoryFrame(VectorXT& frame_data)
{
    if (cell_trajectories.rows() == 0)
    {
        std::cout << "load cell trajectory first" << std::endl;
        return false;
    }
    if (frame > cell_trajectories.cols())
    {
        std::cout << "frame exceed " << cell_trajectories.cols() << std::endl;
        return false;
    }
    frame_data = cell_trajectories.col(frame);
    
    int n_pt = frame_data.rows() / 3;
    Matrix<T, 3, 3> R;
    R << 0.960277, -0.201389, 0.229468, 0.2908, 0.871897, -0.519003, -0.112462, 0.558021, 0.887263;
    Matrix<T, 3, 3> R2 = Eigen::AngleAxis<T>(0.20 * M_PI + 0.5 * M_PI, TV(-1.0, 0.0, 0.0)).toRotationMatrix();

    for (int i = 0; i < n_pt; i++)
    {
        TV pos = frame_data.segment<3>(i * 3);
        if ((pos - TV(-1e10, -1e10, -1e10)).norm() > 1e-8)
        {
            TV updated = (pos - TV(605.877,328.32,319.752)) / 1096.61;
            updated = R2 * R * updated;
            // frame_data.segment<3>(i * 3) = updated * 0.9; 
            if (simulation.cells.resolution == 0)
                frame_data.segment<3>(i * 3) = updated * 0.82 * simulation.cells.unit; 
            else if (simulation.cells.resolution == 1)
                frame_data.segment<3>(i * 3) = updated * 0.88 * simulation.cells.unit; 
            else if (simulation.cells.resolution == 2)
                frame_data.segment<3>(i * 3) = updated * 0.9 * simulation.cells.unit; 
        }
    }
    
    return true;
}

void ObjNucleiTracking::loadTargetTrajectory(const std::string& filename, bool filter)
{
    DataIO data_io;
    data_io.loadTrajectories(filename, cell_trajectories);
    if (filter)
    {
        int n_nucleus = cell_trajectories.rows() / 3;
        int n_frames = cell_trajectories.cols();

        MatrixXT filtered_trajectories = cell_trajectories;
        for (int i = 3; i < n_frames - 3; i++)
        {
            filtered_trajectories.col(i) = 1.0 / 64.0 * (2.0 * cell_trajectories.col(i - 3) + 
            7.0 * cell_trajectories.col(i - 2) + 14.0 * cell_trajectories.col(i - 1) 
            + 18.0 * cell_trajectories.col(i) + 14.0 * cell_trajectories.col(i + 1)
            + 7.0 * cell_trajectories.col(i + 2) + 2.0 * cell_trajectories.col(i + 3));
        }

        // std::string base_folder = "/home/yueli/Documents/ETH/WuKong/output/cells/data_points/";
        // for (int i = 0; i < n_frames; i++)
        // {
        //     std::ofstream out(base_folder + std::to_string(i) + ".obj");
        //     for (int j = 0; j < n_nucleus; j++)
        //     {
        //         out << "v " << cell_trajectories.col(i).segment<3>(j * 3).transpose() << std::endl;
        //     }
        //     out.close();
        //     out.open(base_folder + std::to_string(i) + "_filtered.obj");
        //     for (int j = 0; j < n_nucleus; j++)
        //     {
        //         out << "v " << filtered_trajectories.col(i).segment<3>(j * 3).transpose() << std::endl;
        //     }
        //     out.close();
        // }
        cell_trajectories = filtered_trajectories;

        MatrixXT vel(n_nucleus * 3, n_frames), acc(n_nucleus * 3, n_frames);
        tbb::parallel_for(0, n_nucleus, [&](int i)
        {
            for (int j = 0; j < n_frames - 1; j++)
            {
                vel.col(j).segment<3>(i * 3) = cell_trajectories.col(j + 1).segment<3>(i * 3) -
                                                cell_trajectories.col(j).segment<3>(i * 3); 
                if (j > 0)
                    acc.col(j).segment<3>(i * 3) = cell_trajectories.col(j + 1).segment<3>(i * 3) -
                                                        2.0 * cell_trajectories.col(j).segment<3>(i * 3) +
                                                        cell_trajectories.col(j - 1).segment<3>(i * 3);
            }
        });
        vel.col(n_frames - 1) = vel.col(n_frames - 2);
        VectorXT avg_vel(n_nucleus), max_vel(n_nucleus), 
            avg_acc(n_nucleus), max_acc(n_nucleus);
        
        VectorXT mus(n_frames), sigmas(n_frames);
        
        tbb::parallel_for(0, n_frames, [&](int i)
        {
            T sum_vi = 0.0;
            for (int j = 0; j < n_nucleus; j++)
                sum_vi += vel.col(i).segment<3>(j * 3).norm();
            T mean = sum_vi / T(n_nucleus);
            mus[i] = mean;
            T sigma = 0.0;
            for (int j = 0; j < n_nucleus; j++)
            {
                T vi = vel.col(i).segment<3>(j * 3).norm();
                sigma += (vi - mean) * (vi - mean);
            }
            sigma = std::sqrt(sigma / T(n_nucleus));
            sigmas[i] = sigma;

            for (int j = 0; j < n_nucleus; j++)
            {
                T vi = vel.col(i).segment<3>(j * 3).norm();
                if (vi > mean + 2.0 * sigma)
                {
                    cell_trajectories.col(i).segment<3>(j * 3).setConstant(-1e10);
                }
            }
        });

        
        
        
    }
}

void ObjNucleiTracking::filterTrackingData3X3F()
{
    std::string result_folder = "/home/yueli/Documents/ETH/WuKong/output/FilteredData/";
    VectorXT data_previous_frame;
    T h = 0.5;
    int n_nulcei = 0;
    VectorXT targets_inititial, targets_opt, targets_rest;
    std::vector<std::vector<int>> neighbor_cells;
    T w_p = 1.0;
    auto fetchEntryFromVector = [&](const std::vector<int>& indices, 
        const VectorXT& vector, VectorXT& entries)
    {
        entries.resize(indices.size() * 3);
        for (int i = 0; i < indices.size(); i++)
            entries.segment<3>(i * 3) = vector.segment<3>(indices[i] * 3);
    };

    auto computeDet = [&](const VectorXT& x, const VectorXT& X)
    {
        TM undeformed, deformed, F;
		for (int i = 0; i < 3; i++)
		{
			undeformed.col(i) = X.segment<3>(3 + i * 3) - X.segment<3>(0);	
			deformed.col(i) = x.segment<3>(3 + i * 3) - x.segment<3>(0);
		}
        
        // std::cout << "undeformed" << std::endl;
        // std::cout << undeformed << std::endl;
        // std::cout << "undeformed inverse" << std::endl;
        // std::cout << undeformed.inverse() << std::endl;
		F = deformed * undeformed.inverse();
		T detF = F.determinant();
        
        return detF;
    };

    auto energyValue = [&](const VectorXT& x)
    {
        T energy = 0.0;
        energy += 0.5 * (x - targets_inititial).dot(x - targets_inititial);

        for (int i = 0; i < n_nulcei; i++)
        {
            VectorXT undeformed, deformed;
            fetchEntryFromVector(neighbor_cells[i], targets_rest, undeformed);
            fetchEntryFromVector(neighbor_cells[i], x, deformed);
            T ei;
            computeDeformationPenaltyDet3D(w_p, undeformed, deformed, ei);
            energy += ei;
        }

        return energy;
    };

    auto energyGradient = [&](const VectorXT& x, VectorXT& dedx)
    {
        dedx = x - targets_inititial;
        for (int i = 0; i < n_nulcei; i++)
        {
            VectorXT undeformed, deformed;
            fetchEntryFromVector(neighbor_cells[i], targets_rest, undeformed);
            fetchEntryFromVector(neighbor_cells[i], x, deformed);
            Vector<T, 12> dedx_local;
            computeDeformationPenaltyDet3DGradient(w_p, undeformed, deformed, dedx_local);
            T det = computeDet(deformed, undeformed);
            if (dedx_local.norm() > 1e4)
            {
                TM x, X;
                for (int i = 0; i < 3; i++)
                {
                    X.col(i) = undeformed.segment<3>(3 + i * 3) - undeformed.segment<3>(0);	
                    x.col(i) = deformed.segment<3>(3 + i * 3) - deformed.segment<3>(0);
                }
                TV e0 = X.col(0).normalized(), e1 = X.col(1).normalized(), e2 = X.col(2).normalized();
                std::cout << e0.dot(e1) << " " << e0.dot(e2) << std::endl;
                std::ofstream out("stupid_points.obj");
                for (int i = 0; i < 4; i++)
                    out << "v " << undeformed.segment<3>(i * 3).transpose() << std::endl;
                out.close();

                std::cout << "undeformed" << std::endl;
                std::cout << X << std::endl;
                std::cout << "undeformed inverse" << std::endl;
                std::cout << X.inverse() << std::endl;
                std::cout << "det F " << det << std::endl;
                std::cout << "dedx_norm " << dedx_local.norm() << std::endl;
                std::getchar();
            }
            for (int j = 0; j < neighbor_cells[i].size(); j++)
            {
                dedx.segment<3>(neighbor_cells[i][j] * 3) += dedx_local.segment<3>(j * 3);
            }
        }
        return dedx.norm();
    };

    auto energyHessian = [&](const VectorXT& x, std::vector<Entry>& entries)
    {
        for (int n = 0; n < n_nulcei; n++)
        {
            entries.push_back(Entry(n, n, 1.0));
            VectorXT undeformed, deformed;
            fetchEntryFromVector(neighbor_cells[n], targets_rest, undeformed);
            fetchEntryFromVector(neighbor_cells[n], x, deformed);
            Matrix<T, 12, 12> hessian_local;
            computeDeformationPenaltyDet3DHessian(w_p, undeformed, deformed, hessian_local);
            for (int i = 0; i < neighbor_cells[n].size(); i++)
            {
                for (int j = 0; j < neighbor_cells[n].size(); j++)
                {
                    for (int k = 0; k < 3; k++)
                    {
                        for (int l = 0; l < 3; l++)
                        {
                            entries.push_back(Entry(neighbor_cells[n][i] * 3 + k, neighbor_cells[n][j] * 3 + l, hessian_local(i * 3 + k, j * 3 + l)));
                        }   
                    }
                }
            }
        }
    };


    for (int i = 0; i < 40; i++)
    {
        frame = i;
        VectorXT data_points;
        bool success = getTargetTrajectoryFrame(data_points);
        if (frame == 0)
        {
            data_previous_frame = data_points;
            continue;
        }
        SpatialHash hash;
        hash.build(h, data_previous_frame);
        n_nulcei = data_points.rows() / 3;
                std::ofstream out_obj(result_folder + std::to_string(i) + ".obj");
        for (int i = 0; i < n_nulcei; i++)
        {
            out_obj << "v " << data_points.segment<3>(i * 3).transpose() << std::endl;
        }
        out_obj.close();
        out_obj.open(result_folder + std::to_string(i - 1) + "_filtered_3x3.obj");
        for (int i = 0; i < n_nulcei; i++)
        {
            out_obj << "v " << data_previous_frame.segment<3>(i * 3).transpose() << std::endl;
        }
        out_obj.close();
        targets_rest = data_previous_frame;
        targets_inititial = data_points;
        targets_opt = data_points;

        for (int i = 0; i < n_nulcei; i++)
        {
            std::vector<int> neighbors;
            TV pi = data_previous_frame.segment<3>(i * 3);
            hash.getOneRingNeighbors(pi, neighbors);
            
            std::vector<T> distance;
            std::vector<int> indices;
            for (int idx : neighbors)
            {
                TV pj = data_previous_frame.segment<3>(idx * 3);
                if ((pi - pj).norm() < 1e-5)
                    continue;
                distance.push_back((pi - pj).norm());
                indices.push_back(idx);
            }
            // std::sort(indices.begin(), indices.end(), [&distance](int a, int b){ return distance[a] < distance[b]; } );
            
            std::vector<int> closest_pts;
            closest_pts.push_back(i);
            closest_pts.push_back(indices[0]);
            closest_pts.push_back(indices[1]);
            closest_pts.push_back(indices[2]);
            neighbor_cells.push_back(closest_pts);
        }

        T tol = 1e-7;
        T g_norm = 1e10;
        int ls_max = 10;
        int opt_iter = 0;

        int max_iter = 100;
        simulation.verbose = false;
        T g_norm0 = 0;
        int n_dof = targets_opt.rows();
        while (true)
        {
            T O; 
            VectorXT dOdx;
            g_norm = energyGradient(targets_opt, dOdx);
            O = energyValue(targets_opt);
            std::cout << "iter " << opt_iter << " |g|: " << g_norm << " E: " << O << std::endl;
            std::cout << "min " << dOdx.minCoeff() << " max " << dOdx.maxCoeff() << std::endl;
            std::getchar();
            if (opt_iter == 0)
                g_norm0 = g_norm;
            if (g_norm < tol * g_norm0 || opt_iter > max_iter)
                break;
            StiffnessMatrix H(n_dof, n_dof);
            std::vector<Entry> entries;
            energyHessian(targets_opt, entries);
            H.setFromTriplets(entries.begin(), entries.end());
            VectorXT g = -dOdx, dx = VectorXT::Zero(n_dof);
            simulation.linearSolve(H, g, dx);
            T alpha = 1.0;
            int i = 0;
            for (; i < ls_max; i++)
            {
                VectorXT x_ls = targets_opt + alpha * dx;
                T O_ls = energyValue(x_ls);
                if (O_ls < O)
                {
                    targets_opt = x_ls;
                    break;
                }
                alpha *= 0.5;
            }
            if (i == ls_max)
            {

            }
            std::cout << "#ls " << i << "/" << ls_max << std::endl;
            opt_iter++;
        }

        data_previous_frame = targets_opt;
        std::getchar();
    }
}

void ObjNucleiTracking::filterTrackingData3X2F()
{
    std::string result_folder = "/home/yueli/Documents/ETH/WuKong/output/FilteredData/";
    VectorXT data_previous_frame;
    T h = 0.5;
    int n_nulcei = 0;
    VectorXT targets_inititial, targets_opt, targets_rest;
    std::vector<std::vector<int>> neighbor_cells;
    T w_p = 1.0;

    auto fetchEntryFromVector = [&](const std::vector<int>& indices, 
        const VectorXT& vector, VectorXT& entries)
    {
        entries.resize(indices.size() * 3);
        for (int i = 0; i < indices.size(); i++)
            entries.segment<3>(i * 3) = vector.segment<3>(indices[i] * 3);
    };

    auto energyValue = [&](const VectorXT& x)
    {
        T energy = 0.0;
        energy += 0.5 * (x - targets_inititial).dot(x - targets_inititial);

        for (int i = 0; i < n_nulcei; i++)
        {
            VectorXT undeformed, deformed;
            fetchEntryFromVector(neighbor_cells[i], targets_rest, undeformed);
            fetchEntryFromVector(neighbor_cells[i], x, deformed);
            T ei;
            computeDeformationPenaltyCST3D(w_p, undeformed, deformed, ei);
            energy += ei;
        }

        return energy;
    };

    auto energyGradient = [&](const VectorXT& x, VectorXT& dedx)
    {
        dedx = x - targets_inititial;
        for (int i = 0; i < n_nulcei; i++)
        {
            VectorXT undeformed, deformed;
            fetchEntryFromVector(neighbor_cells[i], targets_rest, undeformed);
            fetchEntryFromVector(neighbor_cells[i], x, deformed);
            Vector<T, 9> dedx_local;
            computeDeformationPenaltyCST3DGradient(w_p, undeformed, deformed, dedx_local);
            if (std::isnan(dedx_local.norm()))
            {
                std::ofstream out("stupid_points.obj");
                for (int i = 0; i < 3; i++)
                    out << "v " << undeformed.segment<3>(i * 3).transpose() << std::endl;
                out.close();
                std::cout << dedx_local.norm() << std::endl;
                std::getchar();
            }
            for (int j = 0; j < neighbor_cells[i].size(); j++)
            {
                dedx.segment<3>(neighbor_cells[i][j] * 3) += dedx_local.segment<3>(j * 3);
            }
        }
        return dedx.norm();
    };

    auto energyHessian = [&](const VectorXT& x, std::vector<Entry>& entries)
    {
        for (int n = 0; n < n_nulcei; n++)
        {
            entries.push_back(Entry(n, n, 1.0));
            VectorXT undeformed, deformed;
            fetchEntryFromVector(neighbor_cells[n], targets_rest, undeformed);
            fetchEntryFromVector(neighbor_cells[n], x, deformed);
            Matrix<T, 9, 9> hessian_local;
            computeDeformationPenaltyCST3DHessian(w_p, undeformed, deformed, hessian_local);
            for (int i = 0; i < neighbor_cells[n].size(); i++)
            {
                for (int j = 0; j < neighbor_cells[n].size(); j++)
                {
                    for (int k = 0; k < 3; k++)
                    {
                        for (int l = 0; l < 3; l++)
                        {
                            entries.push_back(Entry(neighbor_cells[n][i] * 3 + k, neighbor_cells[n][j] * 3 + l, hessian_local(i * 3 + k, j * 3 + l)));
                        }   
                    }
                }
            }
        }
    };


    for (int i = 0; i < 50; i++)
    {
        frame = i;
        VectorXT data_points;
        bool success = getTargetTrajectoryFrame(data_points);
        n_nulcei = data_points.rows() / 3;
        
        std::ofstream out_txt(result_folder + std::to_string(i) + ".txt");
        for (int i = 0; i < n_nulcei; i++)
        {
            out_txt << data_points.segment<3>(i * 3).transpose() << std::endl;
        }
        out_txt.close();

        if (frame == 0)
        {
            data_previous_frame = data_points;
            continue;
        }
        SpatialHash hash;
        hash.build(h, data_previous_frame);

        std::ofstream out_obj(result_folder + std::to_string(i) + ".obj");
        for (int i = 0; i < n_nulcei; i++)
        {
            out_obj << "v " << data_points.segment<3>(i * 3).transpose() << std::endl;
        }
        out_obj.close();
        
        out_obj.open(result_folder + std::to_string(i - 1) + "_filtered.obj");
        for (int i = 0; i < n_nulcei; i++)
        {
            out_obj << "v " << data_previous_frame.segment<3>(i * 3).transpose() << std::endl;
        }
        out_obj.close();
        targets_rest = data_previous_frame;
        targets_inititial = data_points;
        targets_opt = data_points;

        for (int i = 0; i < n_nulcei; i++)
        {
            std::vector<int> neighbors;
            TV pi = data_previous_frame.segment<3>(i * 3);
            hash.getOneRingNeighbors(pi, neighbors);
            
            std::vector<T> distance;
            std::vector<int> indices;
            for (int idx : neighbors)
            {
                TV pj = data_previous_frame.segment<3>(idx * 3);
                if ((pi - pj).norm() < 1e-5)
                    continue;
                distance.push_back((pi - pj).norm());
                indices.push_back(idx);
            }
            std::sort(indices.begin(), indices.end(), [&distance](int a, int b){ return distance[a] < distance[b]; } );
            
            std::vector<int> closest_pts;
            closest_pts.push_back(i);
            closest_pts.push_back(indices[0]);
            closest_pts.push_back(indices[1]);
            // closest_pts.push_back(indices[2]);
            neighbor_cells.push_back(closest_pts);
        }

        T tol = 1e-9;
        T g_norm = 1e10;
        int ls_max = 10;
        int opt_iter = 0;

        int max_iter = 100;
        simulation.verbose = false;
        T g_norm0 = 0;
        int n_dof = targets_opt.rows();
        while (true)
        {
            T O; 
            VectorXT dOdx;
            g_norm = energyGradient(targets_opt, dOdx);
            O = energyValue(targets_opt);
            std::cout << "iter " << opt_iter << " |g|: " << g_norm << " E: " << O << std::endl;
            
            if (opt_iter == 0)
                g_norm0 = g_norm;
            if (g_norm < tol * g_norm0 || opt_iter > max_iter)
                break;
            StiffnessMatrix H(n_dof, n_dof);
            std::vector<Entry> entries;
            energyHessian(targets_opt, entries);
            H.setFromTriplets(entries.begin(), entries.end());
            VectorXT g = -dOdx, dx = VectorXT::Zero(n_dof);
            simulation.linearSolve(H, g, dx);
            T alpha = 1.0;
            int i = 0;
            for (; i < ls_max; i++)
            {
                VectorXT x_ls = targets_opt + alpha * dx;
                T O_ls = energyValue(x_ls);
                if (O_ls < O)
                {
                    targets_opt = x_ls;
                    break;
                }
                alpha *= 0.5;
            }
            if (i == ls_max)
            {

            }
            std::cout << "#ls " << i << "/" << ls_max << std::endl;
            opt_iter++;
        }

        data_previous_frame = targets_opt;

        // std::getchar();
    }
}

void ObjNucleiTracking::initializeTargetFromMap(const std::string& filename, int _frame)
{
    VectorXT data_points;
    frame = _frame;
    bool success = getTargetTrajectoryFrame(data_points);
    std::ifstream in(filename);
    int idx0, idx1;
    std::vector<int> vf_cell_indices;
    simulation.cells.getVFCellIds(vf_cell_indices);
    while (in >> idx0 >> idx1)
    {
        if (std::find(vf_cell_indices.begin(), vf_cell_indices.end(), idx0) 
            != vf_cell_indices.end())
            target_positions[idx0] = data_points.segment<3>(idx1 * 3);
    }
    in.close();
}

void ObjNucleiTracking::updateTarget()
{
    target_positions.clear();
    VectorXT cell_centroids;
    simulation.cells.getAllCellCentroids(cell_centroids);
    VectorXT data_points;
    bool success = getTargetTrajectoryFrame(data_points);

    TV min_corner, max_corner;
    simulation.cells.computeBoundingBox(min_corner, max_corner);
    T spacing = 0.05 * (max_corner - min_corner).norm();

    T max_dis = 0.02 * (max_corner - min_corner).norm();
    bool inverse = true;
    std::vector<std::pair<int, int>> pairs;
    if (inverse)
    {
        hash.build(spacing, data_points);

        for (int i = 0; i < cell_centroids.rows() / 3; i++)
        {
            std::vector<int> neighbors;
            TV current = cell_centroids.segment<3>(i * 3);
            hash.getOneRingNeighbors(current, neighbors);
            T min_dis = 1e6;
            int min_dis_pt = -1;
            for (int idx : neighbors)
            {
                TV neighbor = data_points.segment<3>(idx * 3);
                
                T dis = (current - neighbor).norm();
                
                if (dis < min_dis)
                {
                    min_dis = dis;
                    min_dis_pt = idx;
                }
            }
            if (min_dis_pt != -1 && min_dis < max_dis)
            {
                target_positions[i] = data_points.segment<3>(min_dis_pt * 3);
                pairs.push_back(std::make_pair(i, min_dis_pt));
            }
        }   
    }
    else
    {
        hash.build(spacing, cell_centroids);
        for (int i = 0; i < data_points.rows() / 3; i++)
        {
            std::vector<int> neighbors;
            TV current = data_points.segment<3>(i * 3);
            hash.getOneRingNeighbors(current, neighbors);
            T min_dis = 1e6;
            int min_dis_pt = -1;
            for (int idx : neighbors)
            {
                TV neighbor = cell_centroids.segment<3>(idx * 3);
                
                T dis = (current - neighbor).norm();
                
                if (dis < min_dis)
                {
                    min_dis = dis;
                    min_dis_pt = idx;
                }
            }
            if (min_dis_pt != -1 && min_dis < max_dis)
            {
                target_positions[min_dis_pt] = current;
                pairs.push_back(std::make_pair(min_dis_pt, i));
            }
        }   
    }
    // std::ofstream out("idx_map.txt");
    // for (auto pair : pairs)
    //     out << pair.first << " " << pair.second << std::endl;
    // out.close();
}

void ObjNucleiTracking::checkData()
{
    VectorXT cell_centroids;
    simulation.cells.getAllCellCentroids(cell_centroids);
    VectorXT data_points;
    bool success = getTargetTrajectoryFrame(data_points);
    std::ofstream out("data_points.obj");
    for (int i = 0; i < data_points.rows() / 3; i++)
    {
        out << "v " << data_points.segment<3>(i * 3).transpose() << std::endl;
    }
    out.close();
    out.open("cell_centroid.obj");
    for (int i = 0; i < cell_centroids.rows() / 3; i++)
    {
        out << "v " << cell_centroids.segment<3>(i * 3).transpose() << std::endl;
    }
    out.close();
}

void ObjNucleiTracking::computeCellTargetsFromDatapoints(const std::string& filename)
{
    VectorXT data_points;
    bool success = getTargetTrajectoryFrame(data_points);
    
    // std::ofstream out_test_points("data_points" + std::to_string(frame) + "_test.obj");

    // for (int i = 0; i < data_points.rows() / 3; i++)
    // {
    //     out_test_points << "v " << data_points.segment<3>(i * 3).transpose() << std::endl;
    // }
    // out_test_points.close();

    std::cout << "success " << success << std::endl;

    if (success)
    {
        
        std::ofstream out(filename);
        VectorXT cell_centroids;
        simulation.cells.getAllCellCentroids(cell_centroids);
        // out_test_points.open("sim_points" + std::to_string(frame) + "_test.obj");

        // for (int i = 0; i < cell_centroids.rows() / 3; i++)
        // {
        //     out_test_points << "v " << cell_centroids.segment<3>(i * 3).transpose() << std::endl;
        // }
        // out_test_points.close();
        // std::getchar();
        // std::exit(0);
        TV min_corner, max_corner;
        simulation.cells.computeBoundingBox(min_corner, max_corner);
        T spacing = 0.05 * (max_corner - min_corner).norm();
        hash.build(spacing, cell_centroids);
        int n_data_pts = data_points.rows() / 3;
        
        int n_cells = cell_centroids.rows() / 3;

        std::vector<TargetData> target_data;

        for (int i = 0; i < n_data_pts; i++)
        {
            TV pi = data_points.segment<3>(i * 3);
            if ((pi - TV(-1e10, -1e10, -1e10)).norm() < 1e-6)
                continue;
            std::vector<int> neighbors;
            hash.getOneRingNeighbors(pi, neighbors);

            T min_dis = 1e10;
            int min_cell_idx = -1;
            for (int neighbor : neighbors)
            {
                TV centroid = cell_centroids.segment<3>(neighbor * 3);
                T dis = (centroid - pi).norm();
                if (dis < min_dis)
                {
                    min_cell_idx = neighbor;
                    min_dis = dis;
                }
            }
            if (min_cell_idx == -1)
                continue;

            VectorXT positions;
            std::vector<int> indices;
            simulation.cells.getCellVtxAndIdx(min_cell_idx, positions, indices);
            
            int nw = positions.rows() / 3;
            
            MatrixXT C(3, nw);
            for (int i = 0; i < nw; i++)
                C.col(i) = positions.segment<3>(i * 3);
            
            StiffnessMatrix Q = (C.transpose() * C).sparseView();
            VectorXT c = -C.transpose() * pi;
            StiffnessMatrix A(1, nw);
            for (int i = 0; i < nw; i++)
                A.insert(0, i) = 1.0;

            VectorXT lc(1); lc[0] = 1.0;
            VectorXT uc(1); uc[0] = 1.0;

            VectorXT lx(nw); 
            lx.setConstant(1e-4);
            VectorXT ux(nw); 
            ux.setConstant(10);

            VectorXT w;
            igl::mosek::MosekData mosek_data;
            std::vector<VectorXT> lagrange_multipliers;
            igl::mosek::mosek_quadprog(Q, c, 0, A, lc, uc, lx, ux, mosek_data, w, lagrange_multipliers);
            T error = (C * w - pi).norm();
            
            VectorXT dLdp = c + Q * w;
            dLdp -= lagrange_multipliers[0].transpose() * A;
            dLdp += lagrange_multipliers[1].transpose() * A;
            dLdp -= lagrange_multipliers[2];
            dLdp += lagrange_multipliers[3];
            // std::cout << dLdp.norm() << std::endl;
            if (dLdp.norm() > 1e-10 || error > 1e-8)
                continue;
                
            target_data.push_back(TargetData(w, i, min_cell_idx));
        }

        for (auto data : target_data)
        {
            if (data.data_point_idx != -1)
            {
                out << data.data_point_idx << " " << data.cell_idx << " " << data.weights.rows() << " " << data.weights.transpose() << std::endl;
            }
        }

        out.close();
    }

}

void ObjNucleiTracking::computeCellTargetFromDatapoints()
{
    VectorXT data_points;
    bool success = getTargetTrajectoryFrame(data_points);

    if (success)
    {
        std::string base_dir = "/home/yueli/Documents/ETH/WuKong/Projects/CellSim/data/";
        std::ofstream out(base_dir + "weighted_targets.txt");
        VectorXT cell_centroids;
        simulation.cells.getAllCellCentroids(cell_centroids);
        TV min_corner, max_corner;
        simulation.cells.computeBoundingBox(min_corner, max_corner);
        T spacing = 0.05 * (max_corner - min_corner).norm();
        hash.build(spacing, cell_centroids);
        int n_data_pts = data_points.size();
        
        int n_cells = cell_centroids.rows() / 3;

        std::vector<T> errors(cell_centroids.rows() / 3, -1.0);

        std::vector<TargetData> target_data(n_cells, TargetData());

        for (int i = 0; i < n_data_pts; i++)
        {
            TV pi = data_points.segment<3>(i * 3);
            std::vector<int> neighbors;
            hash.getOneRingNeighbors(pi, neighbors);
            
            T min_dis = 1e10;
            int min_cell_idx = -1;
            for (int neighbor : neighbors)
            {
                TV centroid = cell_centroids.segment<3>(neighbor * 3);
                T dis = (centroid - pi).norm();
                if (dis < min_dis)
                {
                    min_cell_idx = neighbor;
                    min_dis = dis;
                }
            }
            if (min_cell_idx == -1)
                continue;
            std::cout << i << "/" << n_data_pts << " min cell " << min_cell_idx << std::endl;
            
            VectorXT positions;
            std::vector<int> indices;
            simulation.cells.getCellVtxAndIdx(min_cell_idx, positions, indices);
            
            int nw = positions.rows() / 3;
            VectorXT weights(nw);
            weights.setConstant(1.0 / nw);
            
            MatrixXT C(3, nw);
            for (int i = 0; i < nw; i++)
                C.col(i) = positions.segment<3>(i * 3);
            
            StiffnessMatrix Q = (C.transpose() * C).sparseView();
            VectorXT c = -C.transpose() * pi;
            StiffnessMatrix A(1, nw);
            for (int i = 0; i < nw; i++)
                A.insert(0, i) = 1.0;

            VectorXT lc(1); lc[0] = 1.0;
            VectorXT uc(1); uc[0] = 1.0;

            VectorXT lx(nw); 
            lx.setConstant(1e-4);
            VectorXT ux(nw); 
            ux.setConstant(1e4);

            VectorXT w;
            igl::mosek::MosekData mosek_data;
            std::vector<VectorXT> lagrange_multipliers;
            igl::mosek::mosek_quadprog(Q, c, 0, A, lc, uc, lx, ux, mosek_data, w, lagrange_multipliers);
            T error = (C * w - pi).norm();
            
            // std::cout << error << " " << w.transpose() << " " << w.sum() << std::endl;
        
            if (error > 1e-6)
                continue;

            if (errors[min_cell_idx] != -1)
            {
                if (error < errors[min_cell_idx])
                {
                    target_data[min_cell_idx] = TargetData(w, i, min_cell_idx);
                    errors[min_cell_idx] = error;
                }
            }
            else
            {
                errors[min_cell_idx] = error;
                target_data[min_cell_idx] = TargetData(w, i, min_cell_idx);
            }
            
        }

        for (auto data : target_data)
        {
            if (data.data_point_idx != -1)
            {
                out << data.data_point_idx << " " << data.cell_idx << " " << data.weights.rows() << " " << data.weights.transpose() << std::endl;
            }
        }
        
        out.close();
    }
}

void ObjNucleiTracking::computeKernelWeights()
{
    VectorXT data_points;
    bool success = getTargetTrajectoryFrame(data_points);
    std::ofstream out("targets_and_weights.txt");
    if (success)
    {
        VectorXT cell_centroids;
        simulation.cells.getAllCellCentroids(cell_centroids);
        TV min_corner, max_corner;
        simulation.cells.computeBoundingBox(min_corner, max_corner);
        T spacing = 0.02 * (max_corner - min_corner).norm();

        hash.build(spacing, data_points);
        int n_cells = cell_centroids.rows() / 3;
        int cons_cnt = 0;
        for (int i = 0; i < n_cells; i++)
        {
            std::vector<int> neighbors;
            TV ci = cell_centroids.segment<3>(i * 3);
            hash.getOneRingNeighbors(ci, neighbors);

            // std::cout << "# of selected point: " << neighbors.size() << std::endl;
            
            int nw = neighbors.size();
            VectorXT weights(nw);
            weights.setConstant(1.0 / nw);
            
            MatrixXT C(3, nw);
            for (int i = 0; i < nw; i++)
                C.col(i) = data_points.segment<3>(neighbors[i] * 3);
            
            StiffnessMatrix Q = (C.transpose() * C).sparseView();
            VectorXT c = -C.transpose() * ci;
            StiffnessMatrix A(1, nw);
            for (int i = 0; i < nw; i++)
                A.insert(0, i) = 1.0;

            VectorXT lc(1); lc[0] = 1.0;
            VectorXT uc(1); uc[0] = 1.0;

            VectorXT lx(nw); 
            lx.setConstant(-2.0);
            VectorXT ux(nw); 
            ux.setConstant(2.0);

            VectorXT w;
            igl::mosek::MosekData mosek_data;
            std::vector<VectorXT> lagrange_multipliers;
            igl::mosek::mosek_quadprog(Q, c, 0, A, lc, uc, lx, ux, mosek_data, w, lagrange_multipliers);
            
            // std::cout << w.transpose() << std::endl;
            // std::cout << "weights sum: " <<  w.sum() << std::endl;
            T error = (C * w - ci).norm();
            // std::cout << "error: " << (C * w - ci).norm() << std::endl;
            if (error < 1e-4)
            {
                cons_cnt++;
                out << i << " " << nw << " ";
                for (int j = 0; j < nw; j++)
                {
                    out << neighbors[j] << " " << w[j] << " ";
                }
                out << std::endl;
            }
            // std::getchar();
        }
        std::cout << cons_cnt << "/" << n_cells << " have targets" << std::endl;
    }
    else
    {
        std::cout << "error with loading cell trajectory data" << std::endl;
    }
}