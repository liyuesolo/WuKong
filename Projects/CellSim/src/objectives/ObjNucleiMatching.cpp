#include <igl/mosek/mosek_quadprog.h>
#include "../../include/Objectives.h"
#include <Eigen/PardisoSupport>
#include <Spectra/SymEigsShiftSolver.h>
#include <Spectra/MatOp/SparseSymShiftSolve.h>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include "../../include/DataIO.h"
#include "../../include/LinearSolver.h"
#include "../../include/eiquadprog.h"
#include "../../include/autodiff/EdgeEnergy.h"

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

void ObjNucleiTracking::loadWeightedCellTarget(const std::string& filename)
{
    target_filename = filename;
    VectorXT data_points;
    bool success = getTargetTrajectoryFrame(data_points);
    
    std::ofstream out("data_points" + std::to_string(frame) + ".obj");

    for (int i = 0; i < data_points.rows() / 3; i++)
    {
        out << "v " << data_points.segment<3>(i * 3).transpose() << std::endl;
    }
    out.close();

    if (success)
    {
        int n_cells = simulation.cells.basal_face_start;
        std::ifstream in(filename);
        int data_point_idx, cell_idx, nw;
        std::vector<bool> visited(n_cells, false);
        while (in >> data_point_idx >> cell_idx >> nw)
        {
            VectorXT w(nw); w.setZero();
            for (int j = 0; j < nw; j++)
                in >> w[j];
            TV target = data_points.segment<3>(data_point_idx * 3);

            VectorXT positions;
            std::vector<int> indices;
            simulation.cells.getCellVtxAndIdx(cell_idx, positions, indices);

            TV current = TV::Zero();
            for (int i = 0; i < nw; i++)
                current += w[i] * positions.segment<3>(i * 3);
            T error = (current - target).norm();
            
            if (error < 0.5)
            {
                weight_targets.push_back(TargetData(w, data_point_idx, cell_idx, target));
                visited[cell_idx] = true;
            }
            // else
            //     std::cout << error << std::endl;
        }
        in.close();
        target_obj_weights.resize(n_cells);
        target_obj_weights.setZero();
        for (int i = 0; i < n_cells; i++)
        {
            if (visited[i] == false)
            {
                VectorXT positions;
                std::vector<int> indices;
                simulation.cells.getCellVtxAndIdx(i, positions, indices);
                TV centroid = TV::Zero();
                int n_pt = positions.rows() / 3;
                for (int i = 0; i < n_pt; i++)
                {
                    centroid += positions.segment<3>(i * 3);
                }
                centroid /= T(n_pt);
                VectorXT w = VectorXT::Constant(n_pt, 1.0/n_pt);
                
                weight_targets.push_back(TargetData(w, -1, i, centroid));
                target_obj_weights[i] = 1e-4;
            }
            else
            {
                target_obj_weights[i] = 1.0;
            }
        }
        
    }
    
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
        TV perturb = perturbation * TV::Random();
        target_positions[idx] = TV(x, y, z) + perturb;
    }
    in.close();

    target_obj_weights = VectorXT::Ones(simulation.cells.basal_face_start);

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
                    // for (int d = 0; d < 3; d++)
                    // {
                    //     d2Odx2_entries.push_back(Entry(indices[i] * 3 + d, 
                    //         indices[j] * 3 + d, 
                    //         weights[i] * weights[j] * target_obj_weights[cell_idx])); 
                    // }
        });

        // if (!running_diff_test)
        //     for (int i = 0; i < n_dof_sim; i++)
        //         d2Odx2_entries.push_back(Entry(i, i, 1.0));
    }
    if (add_forward_potential)
    {
        std::vector<Entry> sim_H_entries;
        VectorXT dx = simulation.deformed - simulation.undeformed;
        StiffnessMatrix sim_H(n_dof_sim, n_dof_sim);
        simulation.buildSystemMatrix(dx, sim_H);
        sim_H *= w_fp;
        std::vector<Entry> sim_potential_H_entries = simulation.cells.entriesFromSparseMatrix(sim_H);
        d2Odx2_entries.insert(d2Odx2_entries.end(), sim_potential_H_entries.begin(), sim_potential_H_entries.end());

        // int cnt = 0;
        // simulation.cells.iterateApicalEdgeSerial([&](Edge& e){
        //     TV vi = simulation.cells.deformed.segment<3>(e[0] * 3);
        //     TV vj = simulation.cells.deformed.segment<3>(e[1] * 3);
        //     Matrix<T, 6, 6> hessian;
        //     computeEdgeSquaredNormHessian(vi, vj, hessian);
        //     hessian *= simulation.cells.edge_weights[cnt++];
        //     simulation.cells.addHessianEntry<6>(d2Odx2_entries, {e[0], e[1]}, w_fp * hessian);
        // });

        // std::vector<Entry> sim_H_entries;
        // simulation.cells.addCellVolumePreservationHessianEntries(sim_H_entries);
        // MatrixXT dummy;
        // simulation.woodbury = false;
        // simulation.cells.addYolkVolumePreservationHessianEntries(sim_H_entries, dummy);
        // simulation.woodbury = true;
        // StiffnessMatrix sim_H(n_dof_sim, n_dof_sim);
        // sim_H.setFromTriplets(sim_H_entries.begin(), sim_H_entries.end());
        // sim_H *= w_fp;
        // std::vector<Entry> sim_potential_H_entries = simulation.cells.entriesFromSparseMatrix(sim_H);
        // d2Odx2_entries.insert(d2Odx2_entries.end(), sim_potential_H_entries.begin(), sim_potential_H_entries.end());
    }
}

void ObjNucleiTracking::computeOx(const VectorXT& x, T& Ox)
{
    Ox = 0.0;
    simulation.deformed = x;
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
            Ox += std::pow(xTx, p) * target_obj_weights[cell_idx];
        });
    }

    if (add_forward_potential)
    {
        // T simulation_potential = 0.0;
        // simulation.cells.addPerEdgeEnergy(simulation_potential);
        // simulation.cells.addCellVolumePreservationEnergy(simulation_potential);
        // simulation.cells.addYolkVolumePreservationEnergy(simulation_potential);
        VectorXT dx = simulation.deformed - simulation.undeformed;
        T simulation_potential = simulation.computeTotalEnergy(dx);
        simulation_potential *= w_fp;
        Ox += simulation_potential;
        std::cout << "constracting energy: " << simulation_potential << std::endl;
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
    
    if (add_forward_potential)
    {
        
        VectorXT cell_forces(n_dof_sim); cell_forces.setZero();
        // simulation.cells.addPerEdgeForceEntries(cell_forces);
        // simulation.cells.addCellVolumePreservationForceEntries(cell_forces);
        // simulation.cells.addYolkVolumePreservationForceEntries(cell_forces);
        VectorXT dx = simulation.deformed - simulation.undeformed;
        simulation.computeResidual(dx, cell_forces);
        dOdx -= w_fp * cell_forces;

        // std::cout << "&&&&&&f: " << dOdx[0] << " " << contracting_force[0] << " " << contracting_force.norm() << std::endl;
    }
}


T ObjNucleiTracking::value(const VectorXT& p_curr, bool simulate, bool use_prev_equil)
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
                T step_size = simulation.cells.computeLineSearchInitStepsize(simulation.u, nodge_direction, false);
                simulation.u += step_size * nodge_direction;
            }
            else
                break;
        }
    }

    T energy = 0.0;
    computeOx(simulation.deformed, energy);
    
    if (add_reg)
    {
        T reg_term = 0.5 * reg_w * (p_curr - prev_params).dot(p_curr - prev_params);
        std::cout << "\t reg term: " << reg_term << std::endl;
        energy += reg_term;
    }


    return energy;
}

T ObjNucleiTracking::gradient(const VectorXT& p_curr, VectorXT& dOdp, T& energy, bool simulate)
{
    // simulation.loadDeformedState("current_mesh.obj");
    updateDesignParameters(p_curr);
    if (simulate)
    {
        simulation.reset();
        // simulation.staticSolve();
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
                T step_size = simulation.cells.computeLineSearchInitStepsize(simulation.u, nodge_direction, false);
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
    Eigen::PardisoLLT<StiffnessMatrix> solver;
    solver.analyzePattern(d2edx2);
    T alpha = 1.0;
    for (int i = 0; i < 50; i++)
    {
        solver.factorize(d2edx2);
        if (solver.info() == Eigen::NumericalIssue)
        { 
            std::cout << "[ERROR] simulation Hessian indefinite" << std::endl;
            d2edx2.diagonal().array() += alpha;
            alpha *= 10;
            continue;
        }
        break;    
    }
    lambda = solver.solve(dOdx);

    
    // std::cout << " |gradient| linear solve " << (d2edx2 * lambda - dOdx).norm() / dOdx.norm() << std::endl;
    simulation.cells.dOdpEdgeWeightsFromLambda(lambda, dOdp);
    
    // \partial O \partial p
    {
        if (add_forward_potential)
        {
            VectorXT dOdp_force(n_dof_design); dOdp_force.setZero();
            simulation.cells.computededp(dOdp_force);
            dOdp += w_fp * dOdp_force;
        }
    }

    // std::exit(0);
    equilibrium_prev = simulation.u;

    // MatrixXT dxdp;
    // simulation.cells.dxdpFromdxdpEdgeWeights(dxdp);
    // VectorXT dOdp_SA = dOdx.transpose() * dxdp;

    // std::cout << (dOdp_SA - dOdp).norm() << std::endl;
    // std::exit(0);
    if (add_reg)
    {
        T reg_term = 0.5 * reg_w * (p_curr - prev_params).dot(p_curr - prev_params);
        energy += reg_term;
        dOdp += reg_w * (p_curr - prev_params);
    }

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
            entries.push_back(Entry(it.col() + nx, it.row() + nxnp, it.value()));
            entries.push_back(Entry(it.row() + nxnp, it.col() + nx, it.value()));
        }
    }

    if (use_penalty)
    {
        T penalty_energy = 0.0;
        for (int i = 0; i < n_dof_design; i++)
        {
            T d_hat_max = bound[1] - p_curr[i];
            T d_hat_min = p_curr[i] - bound[0];
            if (penalty_type == LogBarrier)
            {
                if (d_hat_max < barrier_distance)
                {
                    entries.push_back(Entry(i + n_dof_sim, i + n_dof_sim, 
                        penalty_weight * barrier<2>(d_hat_max, barrier_distance)));
                }
                if (d_hat_min < barrier_distance)
                {
                    entries.push_back(Entry(i + n_dof_sim, i + n_dof_sim, 
                        penalty_weight * barrier<2>(d_hat_min, barrier_distance)));
                }
            }    
            else if (penalty_type == Qubic)
            {
                if (d_hat_max < barrier_distance)
                {
                     entries.push_back(Entry(i + n_dof_sim, i + n_dof_sim, 
                        penalty_weight * 6.0 * d_hat_max));
                }
                if (d_hat_min < barrier_distance)
                {
                    entries.push_back(Entry(i + n_dof_sim, i + n_dof_sim, 
                        penalty_weight * 6.0 * d_hat_min));
                }
            }
        }
    }

    for (int i = 0; i < n_dof_sim; i++)
        entries.push_back(Entry(i, i, 1e-10));
    for (int i = 0; i < n_dof_design; i++)
        entries.push_back(Entry(i + n_dof_sim, i + n_dof_sim, 1e-10));
    for (int i = 0; i < n_dof_sim; i++)
        entries.push_back(Entry(i + n_dof_sim + n_dof_design, i + n_dof_sim + n_dof_design, -1e-10));
    
    // for (int i = 0; i < nxnp; i++)
    //     entries.push_back(Entry(i, i, 1e-4));
    // for (int i = nx; i < nxnp; i++)
    //     entries.push_back(Entry(i, i, 1e-8));
    // for (int i = nxnp; i < nxnp + n_dof_sim; i++)
    //     entries.push_back(Entry(i, i, -1e-6));
    
    H.setFromTriplets(entries.begin(), entries.end());
    H.makeCompressed();
}

void ObjNucleiTracking::hessianGN(const VectorXT& p_curr, MatrixXT& H, bool simulate)
{
    updateDesignParameters(p_curr);
    
    if (simulate)
    {
        simulation.reset();
        simulation.staticSolve();
    }
    
    MatrixXT dxdp;
    
    simulation.cells.dxdpFromdxdpEdgeWeights(dxdp);
    
    std::vector<Entry> d2Odx2_entries;
    computed2Odx2(simulation.deformed, d2Odx2_entries);
    StiffnessMatrix d2Odx2_matrix(n_dof_sim, n_dof_sim);
    d2Odx2_matrix.setFromTriplets(d2Odx2_entries.begin(), d2Odx2_entries.end());
    simulation.cells.projectDirichletDoFMatrix(d2Odx2_matrix, simulation.cells.dirichlet_data);
    
    // MatrixXT d2Odx2_dense = d2Odx2_matrix;
    // std::cout << d2Odx2_dense.minCoeff() << " " << d2Odx2_dense.maxCoeff() << std::endl;
    
    MatrixXT dxdpTHdxdp = dxdp.transpose() * d2Odx2_matrix * dxdp;

    // StiffnessMatrix d2edx2(n_dof_sim, n_dof_sim);
    // simulation.buildSystemMatrix(simulation.u, d2edx2);
    // T scaling_weight = 1e-1;
    // MatrixXT dxdpTHdxdp = dxdp.transpose() * (d2Odx2_matrix - scaling_weight * d2edx2) * dxdp;
    
    // MatrixXT dxdpTHdxdp = dxdp.transpose() * dxdp;
    
    // Eigen::JacobiSVD<Eigen::MatrixXd> svd(d2Odx2_dense, Eigen::ComputeThinU | Eigen::ComputeThinV);
    // MatrixXT U = svd.matrixU();
	// VectorXT Sigma = svd.singularValues();
	// MatrixXT V = svd.matrixV();
    // std::cout << "d2Odx2_dense singular value last 5: " << Sigma.tail<5>().transpose() << std::endl;
    // std::cout << d2Odx2_dense.diagonal().transpose() << std::endl;
    
    // std::ofstream out("cell_d2odx_singular_vectors.txt");
    // out << U.rows() << " " << U.cols() << std::endl;
    // for (int i = 0; i < n_dof_design; i++)
    //     out << Sigma[i] << " ";
    // out << std::endl;
    // for (int i = 0; i < n_dof_sim; i++)
    // {
    //     for (int j = 0; j < n_dof_design; j++)
    //         out << U(i, j) << " ";
    //     out << std::endl;
    // }
    // out << V.rows() << " " << V.cols() << std::endl;
    // for (int i = 0; i < n_dof_design; i++)
    // {
    //     for (int j = 0; j < n_dof_design; j++)
    //         out << V(i, j) << " ";
    //     out << std::endl;
    // }
    // out << std::endl;
    // out.close();
   
    // simulation.saveState("d2odx2_check.obj");
    // std::exit(0);
    H = dxdpTHdxdp;

    if (add_reg)
    {
        H.diagonal().array() += reg_w;
    }
    
}

T ObjNucleiTracking::maximumStepSize(const VectorXT& dp)
{
    VectorXT p_curr;
    getDesignParameters(p_curr);
    T step_size = 1.0;
    if (use_penalty && penalty_type == LogBarrier)
    {
        while (true)
        {
            VectorXT forward = p_curr + step_size * dp;
            if (forward.minCoeff() < bound[0] || forward.maxCoeff() > bound[1])
                step_size *= 0.8;
            else
                return step_size;
        }
    }
    else
        return step_size;
    
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
    std::cout << "fetching frame: " << frame << std::endl;
    int n_pt = frame_data.rows() / 3;
    Matrix<T, 3, 3> R;
    R << 0.960277, -0.201389, 0.229468, 0.2908, 0.871897, -0.519003, -0.112462, 0.558021, 0.887263;
    Matrix<T, 3, 3> R2 = Eigen::AngleAxis<T>(0.20 * M_PI + 0.5 * M_PI, TV(-1.0, 0.0, 0.0)).toRotationMatrix();

    for (int i = 0; i < n_pt; i++)
    {
        TV pos = frame_data.segment<3>(i * 3);
        TV updated = (pos - TV(605.877,328.32,319.752)) / 1096.61;
        updated = R2 * R * updated;
        // frame_data.segment<3>(i * 3) = updated * 0.8 * simulation.cells.unit; 
        frame_data.segment<3>(i * 3) = updated * 0.9 * simulation.cells.unit; 
    }
    
    return true;
}

void ObjNucleiTracking::loadTargetTrajectory(const std::string& filename)
{
    DataIO data_io;
    data_io.loadTrajectories(filename, cell_trajectories);
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