#include <Eigen/PardisoSupport>
#include <Eigen/CholmodSupport>
#include <Spectra/SymEigsShiftSolver.h>
#include <Spectra/MatOp/SparseSymShiftSolve.h>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include "../include/Simulation.h"
#include "../../../Solver/CHOLMODSolver.hpp"
#include <igl/readOBJ.h>

#include <iomanip>
#include <ipc/ipc.hpp>

#define FOREVER 30000

void Simulation::computeEigenValueSpectraSparse(StiffnessMatrix& A, int nmodes, VectorXT& modes, T shift)
{
    Spectra::SparseSymShiftSolve<T, Eigen::Upper> op(A);

    Spectra::SymEigsShiftSolver<T, 
        Spectra::LARGEST_MAGN, 
        Spectra::SparseSymShiftSolve<T, Eigen::Upper> > 
        eigs(&op, nmodes, 2 * nmodes, shift);

    eigs.init();

    int nconv = eigs.compute();

    Eigen::VectorXd eigen_values = eigs.eigenvalues().real();
    int n_entries = eigen_values.rows();
    modes = eigen_values.segment(n_entries - nmodes - 1, nmodes).transpose();
}

bool Simulation::fetchNegativeEigenVectorIfAny(T& negative_eigen_value, VectorXT& negative_eigen_vector)
{
    int nmodes = 20;
    int n_dof_sim = deformed.rows();
    VectorXT residual(n_dof_sim); residual.setZero();
    computeResidual(u, residual);
    if (residual.norm() > newton_tol)
        return false;
        
    StiffnessMatrix d2edx2(n_dof_sim, n_dof_sim);
    buildSystemMatrix(u, d2edx2);
    
    Eigen::PardisoLLT<StiffnessMatrix> solver;
    solver.analyzePattern(d2edx2); 
    solver.factorize(d2edx2);
    
    if (solver.info() != Eigen::NumericalIssue)
    {
        return false;
    }

    Spectra::SparseSymShiftSolve<T, Eigen::Upper> op(d2edx2);

        //0 cannot cannot be used as a shift
    T shift = -10;
    Spectra::SymEigsShiftSolver<T, 
        Spectra::LARGEST_MAGN, 
        Spectra::SparseSymShiftSolve<T, Eigen::Upper> > 
        eigs(&op, nmodes, 2 * nmodes, shift);

    eigs.init();

    int nconv = eigs.compute();

    if (eigs.info() == Spectra::SUCCESSFUL)
    {
        // std::cout << "Spectra successful" << std::endl;
        Eigen::MatrixXd eigen_vectors = eigs.eigenvectors().real();
        Eigen::VectorXd eigen_values = eigs.eigenvalues().real();
        int last_col = eigen_values.rows() - 1;
        // std::cout << eigen_values.transpose() << std::endl;
        if (eigen_values[last_col] < 0.0)
        {
            negative_eigen_vector = eigen_vectors.col(last_col);
            negative_eigen_value = eigen_values[last_col]; 
            return true;
        }   
        return false;
    }
    else
    {
        std::cout << "Spectra failed" << std::endl;
        return false;
    }
}

void Simulation::checkHessianPD(bool save_txt)
{
    int nmodes = 10;
    int n_dof_sim = deformed.rows();
    StiffnessMatrix d2edx2(n_dof_sim, n_dof_sim);
    buildSystemMatrix(u, d2edx2);
    bool use_Spectra = true;

    Eigen::PardisoLLT<StiffnessMatrix> solver;
    solver.analyzePattern(d2edx2); 
    solver.factorize(d2edx2);
    bool indefinite = false;
    if (solver.info() == Eigen::NumericalIssue)
    {
        std::cout << "!!!indefinite matrix!!!" << std::endl;
        indefinite = true;
    }
    
    if (use_Spectra)
    {

        Spectra::SparseSymShiftSolve<T, Eigen::Upper> op(d2edx2);

        //0 cannot cannot be used as a shift
        // T shift = indefinite ? -1e2 : -1e-4;
        T shift = -1e-4;
        Spectra::SymEigsShiftSolver<T, 
            Spectra::LARGEST_MAGN, 
            Spectra::SparseSymShiftSolve<T, Eigen::Upper> > 
            eigs(&op, nmodes, 2 * nmodes, shift);

        eigs.init();

        int nconv = eigs.compute();

        if (eigs.info() == Spectra::SUCCESSFUL)
        {
            Eigen::MatrixXd eigen_vectors = eigs.eigenvectors().real();
            Eigen::VectorXd eigen_values = eigs.eigenvalues().real();
            std::cout << eigen_values.transpose() << std::endl;
            if (save_txt)
            {
                std::ofstream out("cell_eigen_vectors.txt");
                out << eigen_vectors.rows() << " " << eigen_vectors.cols() << std::endl;
                for (int i = 0; i < eigen_vectors.cols(); i++)
                    out << eigen_values[eigen_vectors.cols() - 1 - i] << " ";
                out << std::endl;
                for (int i = 0; i < eigen_vectors.rows(); i++)
                {
                    // for (int j = 0; j < eigen_vectors.cols(); j++)
                    for (int j = eigen_vectors.cols() - 1; j >-1 ; j--)
                        out << eigen_vectors(i, j) << " ";
                    out << std::endl;
                }       
                out << std::endl;
                out.close();
            }
        }
        else
        {
            std::cout << "Eigen decomposition failed" << std::endl;
        }
    }
    else
    {
        Eigen::MatrixXd A_dense = d2edx2;
        Eigen::EigenSolver<Eigen::MatrixXd> eigen_solver;
        eigen_solver.compute(A_dense, /* computeEigenvectors = */ true);
        auto eigen_values = eigen_solver.eigenvalues();
        auto eigen_vectors = eigen_solver.eigenvectors();
        
        std::vector<T> ev_all(A_dense.cols());
        for (int i = 0; i < A_dense.cols(); i++)
        {
            ev_all[i] = eigen_values[i].real();
        }
        
        std::vector<int> indices;
        for (int i = 0; i < A_dense.cols(); i++)
        {
            indices.push_back(i);    
        }
        std::sort(indices.begin(), indices.end(), [&ev_all](int a, int b){ return ev_all[a] < ev_all[b]; } );
        // std::sort(ev_all.begin(), ev_all.end());

        for (int i = 0; i < nmodes; i++)
            std::cout << ev_all[indices[i]] << std::endl;
        
        if (save_txt)
        {
            std::ofstream out("cell_eigen_vectors.txt");
            out << nmodes << " " << A_dense.cols() << std::endl;
            for (int i = 0; i < nmodes; i++)
                out << ev_all[indices[i]] << " ";
            out << std::endl;
            for (int i = 0; i < nmodes; i++)
            {
                out << eigen_vectors.col(indices[i]).real().transpose() << std::endl;
            }
            out.close();
        }
    }

}

void Simulation::computeLinearModes()
{
    cells.computeLinearModes();
}

void Simulation::initializeCells()
{
    woodbury = true;
    cells.use_alm_on_cell_volume = false;

    std::string surface_mesh_file;
    cells.scene_type = 2;
    if (cells.resolution == 0)
    {
        surface_mesh_file = "/home/yueli/Documents/ETH/WuKong/Projects/CellSim/data/drosophila_real_124_remesh.obj";
    }
    else if (cells.resolution == 1)
    {
        surface_mesh_file = "/home/yueli/Documents/ETH/WuKong/Projects/CellSim/data/drosophila_real_463_remesh.obj";
    }
    else if (cells.resolution == 2)
    {
        surface_mesh_file = "/home/yueli/Documents/ETH/WuKong/Projects/CellSim/data/drosophila_real_1.5k_remesh.obj";
    }
    else if (cells.resolution == 3)
    {
        surface_mesh_file = "/home/yueli/Documents/ETH/WuKong/Projects/CellSim/data/drosophila_real_6k_remesh.obj";
    }
    else if (cells.resolution == -1)
    {
        surface_mesh_file = "/home/yueli/Documents/ETH/WuKong/Projects/CellSim/data/drosophila_real_59_remesh.obj";
    }
    if (cells.use_test_mesh)
    {
        cells.addTestPrismGrid(2, 5);
        woodbury = false;
    }
    else
        cells.vertexModelFromMesh(surface_mesh_file); 
    // cells.addTestPrism(6);
    // cells.addTestPrismGrid(10, 10);
    
    // cells.dynamics = true;
    // vtx_vel = VectorXT::Zero(undeformed.rows());
    // cells.computeNodalMass();
    // cells.vtx_vel.setRandom();
    // cells.vtx_vel/=cells.vtx_vel.norm();
    // cells.checkTotalGradient(true);
    // cells.checkTotalGradientScale(true);
    // cells.checkTotalHessianScale(true);
    // cells.checkTotalHessian(true);
    
    max_newton_iter = FOREVER;
    // verbose = true;
    cells.print_force_norm = true;
    // save_mesh = true;
    // cells.project_block_hessian_PD = true;
    
}
void Simulation::reinitializeCells()
{
    
}

void Simulation::sampleBoundingSurface(Eigen::MatrixXd& V)
{
    cells.sampleBoundingSurface(V);
}

void Simulation::generateMeshForRendering(Eigen::MatrixXd& V, 
    Eigen::MatrixXi& F, Eigen::MatrixXd& C, 
    bool show_deformed, bool show_rest, 
    bool split, bool split_a_bit, bool yolk_only)
{
    // deformed = undeformed + 1.0 * u;
    V.resize(0, 0);
    F.resize(0, 0);
    C.resize(0, 0);
    
    Eigen::MatrixXd V_rest, C_rest;
    Eigen::MatrixXi F_rest, offset;
    
    if (show_deformed)
        cells.generateMeshForRendering(V, F, C);
    int nv = V.rows(), nf = F.rows();
    if (show_rest)
    {
        cells.generateMeshForRendering(V_rest, F_rest, C_rest, true);
        int nv_rest = V_rest.rows(), nf_rest = F_rest.rows();
        V.conservativeResize(V.rows() + V_rest.rows(), 3);
        F.conservativeResize(F.rows() + F_rest.rows(), 3);
        C.conservativeResize(C.rows() + C_rest.rows(), 3);
        C_rest.col(0).setConstant(1.0);
        C_rest.col(1).setConstant(1.0);
        C_rest.col(2).setConstant(0.0);
        offset = F_rest;
        offset.setConstant(nv);
        V.block(nv, 0, nv_rest, 3) = V_rest;
        F.block(nf, 0, nf_rest, 3) = F_rest + offset;
        C.block(nf, 0, nf_rest, 3) = C_rest;
    }
    if (split || split_a_bit)
    {
        cells.splitCellsForRendering(V, F, C, split_a_bit);
    }
    if (yolk_only)
    {
        if (split || split_a_bit)
        {
            cells.splitYolkForRendering(V, F, C, split_a_bit);
        }
        else
        {
            if (show_deformed)
                cells.getYolkForRendering(V, F, C);
            int nv = V.rows(), nf = F.rows();
            if (show_rest)
            {
                cells.getYolkForRendering(V_rest, F_rest, C_rest, true);
                int nv_rest = V_rest.rows(), nf_rest = F_rest.rows();
                V.conservativeResize(V.rows() + V_rest.rows(), 3);
                F.conservativeResize(F.rows() + F_rest.rows(), 3);
                C.conservativeResize(C.rows() + C_rest.rows(), 3);
                C_rest.col(0).setConstant(1.0);
                C_rest.col(1).setConstant(1.0);
                C_rest.col(2).setConstant(0.0);
                offset = F_rest;
                offset.setConstant(nv);
                V.block(nv, 0, nv_rest, 3) = V_rest;
                F.block(nf, 0, nf_rest, 3) = F_rest + offset;
                C.block(nf, 0, nf_rest, 3) = C_rest;
            }
        }
    }
}

bool Simulation::impliciteUpdate(VectorXT& _u)
{
    cells.iterateDirichletDoF([&](int offset, T target)
    {
        f[offset] = 0;
    });

    T residual_norm = 1e10, dq_norm = 1e10;
    int cnt = 0;
    while (true)
    {
        VectorXT residual(deformed.rows());
        residual.setZero();
        
        if (cells.use_ipc_contact)
            cells.updateIPCVertices(_u);

        residual_norm = computeResidual(_u, residual);
        
        std::cout << "iter " << cnt << "/" << max_newton_iter << ": residual_norm " << residual.norm() << " tol: " << newton_tol << std::endl;
        
        if (residual_norm < newton_tol)
            break;

        dq_norm = lineSearchNewton(_u, residual, 20, true);
        
        if(cnt == max_newton_iter || dq_norm > 1e10 || dq_norm < 1e-6)
            return true;

        cnt ++;
    }
    
    return false;
}

bool Simulation::advanceOneStep(int step)
{
    if (dynamic)
    {
        std::cout << "###########################TIME STEP " << 
            current_time << "s/" << simulation_time 
            << "s ###########################" << std::endl;

        impliciteUpdate(u);
        cells.saveCellMesh(step);
        // vtx_vel = u / dt;
        std::cout << "\t\t" << u.norm() / dt / cells.eta << std::endl;
        if (u.norm() < 1e-6)
            return true;
        update();
        current_time += dt;
        std::cout << "###############################################################" << std::endl;
        std::cout << std::endl;
        if (current_time < simulation_time)
            return false;
        return true;
    }
    else
    {
        Timer step_timer(true);
        cells.iterateDirichletDoF([&](int offset, T target)
        {
            f[offset] = 0;
        });

        VectorXT residual(deformed.rows());
        residual.setZero();
        
        if (cells.use_ipc_contact)
            cells.updateIPCVertices(u);

        T residual_norm = computeResidual(u, residual);
        std::cout << "[Newton] computeResidual takes " << step_timer.elapsed_sec() << "s" << std::endl;
        step_timer.restart();
        if (save_mesh)
            cells.saveCellMesh(step);
        // std::cout << "[Newton] saveCellMesh takes " << step_timer.elapsed_sec() << "s" << std::endl;
        if (verbose)
            std::cout << "[Newton] iter " << step << "/" << max_newton_iter << ": residual_norm " << residual.norm() << " tol: " << newton_tol << std::endl;

        if (residual_norm < newton_tol)
            return true;

        T dq_norm = lineSearchNewton(u, residual);
        step_timer.stop();
        if (verbose)
            std::cout << "[Newton] step takes " << step_timer.elapsed_sec() << "s" << std::endl;

        if(step == max_newton_iter || dq_norm > 1e10)
            return true;
        
        return false;    
        
    }
}

void Simulation::appendCylindersToEdges(const std::vector<std::pair<TV, TV>>& edge_pairs, 
        T radius, Eigen::MatrixXd& _V, Eigen::MatrixXi& _F)
{
    int n_div = 10;
    T theta = 2.0 * EIGEN_PI / T(n_div);
    VectorXT points = VectorXT::Zero(n_div * 3);
    for(int i = 0; i < n_div; i++)
        points.segment<3>(i * 3) = TV(radius * std::cos(theta * T(i)), 
        0.0, radius * std::sin(theta*T(i)));

    int offset_v = n_div * 2;
    int offset_f = n_div * 2;

    int n_row_V = _V.rows();
    int n_row_F = _F.rows();

    int n_edge = edge_pairs.size();

    _V.conservativeResize(n_row_V + offset_v * n_edge, 3);
    _F.conservativeResize(n_row_F + offset_f * n_edge, 3);

    tbb::parallel_for(0, n_edge, [&](int ei)
    {
        TV axis_world = edge_pairs[ei].second - edge_pairs[ei].first;
        TV axis_local(0, axis_world.norm(), 0);

        Matrix<T, 3, 3> R = Eigen::Quaternion<T>().setFromTwoVectors(axis_world, axis_local).toRotationMatrix();

        for(int i = 0; i < n_div; i++)
        {
            for(int d = 0; d < 3; d++)
            {
                _V(n_row_V + ei * offset_v + i, d) = points[i * 3 + d];
                _V(n_row_V + ei * offset_v + i+n_div, d) = points[i * 3 + d];
                if (d == 1)
                    _V(n_row_V + ei * offset_v + i+n_div, d) += axis_world.norm();
            }

            // central vertex of the top and bottom face
            _V.row(n_row_V + ei * offset_v + i) = (_V.row(n_row_V + ei * offset_v + i) * R).transpose() + edge_pairs[ei].first;
            _V.row(n_row_V + ei * offset_v + i + n_div) = (_V.row(n_row_V + ei * offset_v + i + n_div) * R).transpose() + edge_pairs[ei].first;

            _F.row(n_row_F + ei * offset_f + i*2 ) = IV(n_row_V + ei * offset_v + i, 
                                    n_row_V + ei * offset_v + i+n_div, 
                                    n_row_V + ei * offset_v + (i+1)%(n_div));

            _F.row(n_row_F + ei * offset_f + i*2 + 1) = IV(n_row_V + ei * offset_v + (i+1)%(n_div), 
                                        n_row_V + ei * offset_v + i+n_div, 
                                        n_row_V + + ei * offset_v + (i+1)%(n_div) + n_div);

        }
    });
}

void Simulation::saveState(const std::string& filename, bool save_edges)
{
    std::ofstream out(filename);
    Eigen::MatrixXd V, C;
    Eigen::MatrixXi F;
    cells.generateMeshForRendering(V, F, C, false);
    if (save_edges)
    {
        int cnt = 0;
        std::vector<std::pair<TV, TV>> end_points;
        cells.iterateEdgeSerial([&](Edge& edge)
        {
            TV from = deformed.segment<3>(edge[0] * 3);
            TV to = deformed.segment<3>(edge[1] * 3);
            end_points.push_back(std::make_pair(from, to));
            cnt++;
        });
        appendCylindersToEdges(end_points, 0.005, V, F);
    }
    for (int i = 0; i < V.rows(); i++)
    {
        out << "v " << std::setprecision(20) << V.row(i) << std::endl;
    }
    for (int i = 0; i < F.rows(); i++)
    {
        IV obj_face = F.row(i).transpose() + IV::Ones();
        out << "f " << obj_face.transpose() << std::endl;
    }
    out.close();
}

void Simulation::reset()
{
    deformed = undeformed;
    u.setZero();
    if (cells.use_ipc_contact)
    {
        // cells.computeIPCRestData();
        for (int i = 0; i < cells.basal_vtx_start; i++)
            cells.ipc_vertices.row(i) = undeformed.segment<3>(i * 3);
    }
}

void Simulation::update()
{
    undeformed = deformed;
    u.setZero();
    if (cells.use_ipc_contact)
    {
        cells.computeIPCRestData();
    }
}

void Simulation::initializeDynamicsData(T _dt, T total_time)
{
    vtx_vel = VectorXT::Zero(undeformed.rows());
    dt = _dt;
    simulation_time = total_time;
    cells.computeNodalMass();
}

bool Simulation::staticSolve()
{
    
    // cells.saveHexTetsStep(0);
    // std::exit(0);
    Timer sim_timer(true);
    VectorXT cell_volume_initial;
    cells.computeVolumeAllCells(cell_volume_initial);
    T yolk_volume_init = 0.0;
    if (cells.add_yolk_volume)
    {
        yolk_volume_init = cells.computeYolkVolume(false);
        // std::cout << "yolk volume initial: " << yolk_volume_init << std::endl;
    }

    T total_volume_apical_surface = cells.computeTotalVolumeFromApicalSurface();
    
    
    // std::cout << cells.computeTotalEnergy(u, true) << std::endl;
    int cnt = 0;
    T residual_norm = 1e10, dq_norm = 1e10;

    cells.iterateDirichletDoF([&](int offset, T target)
    {
        f[offset] = 0;
    });

    T residual_norm_init = 0.0;
    while (true)
    {
        VectorXT residual(deformed.rows());
        residual.setZero();
        if (cells.use_fixed_centroid)
            cells.updateFixedCentroids();
        
        residual_norm = computeResidual(u, residual);
        if (cnt == 0)
            residual_norm_init = residual_norm;
        if (cells.use_ipc_contact)
            cells.updateIPCVertices(u);
        if (!cells.single_prism && save_mesh)
            cells.saveCellMesh(cnt);
        
        
        if (verbose)
            std::cout << "iter " << cnt << "/" << max_newton_iter << ": residual_norm " << residual.norm() << " tol: " << newton_tol << std::endl;
            
        if (residual_norm < newton_tol)
            break;
        
        // t.start();
        dq_norm = lineSearchNewton(u, residual, 20, true);
        cells.updateALMData(u);
        // t.stop();
        // std::cout << "newton single step costs " << t.elapsed_sec() << "s" << std::endl;

        if(cnt == max_newton_iter || dq_norm > 1e10 || dq_norm < 1e-12)
            break;
        cnt++;
    }

    cells.iterateDirichletDoF([&](int offset, T target)
    {
        u[offset] = target;
    });

    // T total_energy_final = cells.computeTotalEnergy(u, true);

    deformed = undeformed + u;
    // cells.saveIPCData();
    if (verbose)
    {
        VectorXT cell_volume_final;
        cells.computeVolumeAllCells(cell_volume_final);

        std::cout << "============================================================================" << std::endl;
        std::cout << std::endl;
        std::cout << "========================= Solver Info ================================="<< std::endl;
        std::cout << "# of system DoF " << deformed.rows() << std::endl;
        std::cout << "# of newton iter: " << cnt << " exited with |g|: " 
            << residual_norm << " |ddu|: " << dq_norm  
            << " |g_init|: " << residual_norm_init << std::endl;
        // std::cout << "Smallest 15 eigenvalues " << std::endl;
        // cells.computeLinearModes();
        std::cout << std::endl;
        std::cout << "========================= Cell Info =================================" << std::endl;
        std::cout << "\tcell volume sum initial " << cell_volume_initial.sum() << std::endl;
        std::cout << "\tcell volume sum final " << cell_volume_final.sum() << std::endl;
        if (cells.add_yolk_volume)
        {
            T yolk_volume = cells.computeYolkVolume(false);
            std::cout << "\tyolk volume initial: " << yolk_volume_init << std::endl;
            std::cout << "\tyolk volume final: " << yolk_volume << std::endl;
        }
        
        std::cout << "\ttotal volume initial from apical surface: " << total_volume_apical_surface << std::endl;
        std::cout << "\ttotal volume final from apical surface: " << cells.computeTotalVolumeFromApicalSurface() << std::endl;
        T total_energy_final = cells.computeTotalEnergy(u, true);
        std::cout << "\ttotal energy final: " << total_energy_final << std::endl;
        std::cout << "============================================================================" << std::endl;

    }
    sim_timer.stop();
    std::cout << "# of newton iter: " << cnt << " exited with |g|: " 
            << residual_norm << " |ddu|: " << dq_norm  
            << " |g_init|: " << residual_norm_init
            << " takes " << sim_timer.elapsed_sec() << "s" << std::endl;
    // checkHessianPD(false);

    VectorXT cell_volume_final;
    cells.computeVolumeAllCells(cell_volume_final);
    int compressed_cell_cnt = 0;
    T compression = 0.0;
    T max_compression = -1e10, min_compression = 1e10;
    for (int i = 0; i < cell_volume_final.rows(); i++)
    {
        if (cell_volume_final[i] < cell_volume_initial[i])
        {
            compressed_cell_cnt++;
            T delta = cell_volume_final[i] - cell_volume_initial[i];
            compression += delta;
            if (std::abs(delta) > max_compression)
                max_compression = delta;
            if (std::abs(delta) < min_compression)
                min_compression = delta;
        }
    }
    std::cout << compressed_cell_cnt << "/" << cells.basal_face_start 
        << " cells are compressed. avg compression: " 
        << compression / T(compressed_cell_cnt) 
        << " max compression: " << max_compression
        << " min compression: " << min_compression
        << std::endl;
    if (cells.has_rest_shape)
    {
        VectorXT edge_compression(cells.edges.size()); 
        edge_compression.setZero();
        tbb::parallel_for(0, (int)cells.edges.size(), [&](int i)
        {
            TV Xi = undeformed.segment<3>(cells.edges[i][0] * 3);
            TV Xj = undeformed.segment<3>(cells.edges[i][1] * 3);
            TV xi = deformed.segment<3>(cells.edges[i][0] * 3);
            TV xj = deformed.segment<3>(cells.edges[i][1] * 3);
            T l0 = (Xj - Xi).norm();
            T l1 = (xj - xi).norm();
            if (l1 < l0)
            {
                edge_compression[i] = l1 - l0;
            }
        });
        std::cout << "edge compression sum " << edge_compression.sum()
                    << " edge compression max " << edge_compression.maxCoeff() 
                    << " edge compression min " << edge_compression.minCoeff() << std::endl;
    }

    if (cnt == max_newton_iter || dq_norm > 1e10 || residual_norm > 1)
        return false;
    return true;
}

void Simulation::checkInfoForSA()
{
    VectorXT cell_hessian_evs;
    cells.computeCellVolumeHessianEigenValues(cell_hessian_evs);
    std::cout << cell_hessian_evs.sum() / T(cell_hessian_evs.rows()) << std::endl;
    std::vector<int> indices;
    for (int i = 0; i < cell_hessian_evs.rows(); i++)
    {
        indices.push_back(i);    
    }
    std::sort(indices.begin(), indices.end(), [&cell_hessian_evs](int a, int b){ return cell_hessian_evs[a] < cell_hessian_evs[b]; } );

    for (int i = 0; i < cell_hessian_evs.rows(); i++)
    {
        std::cout << "cell: " << indices[i] << " " << cell_hessian_evs[indices[i]] << " ";
        // VectorXT positions; 
        // std::vector<int> vtx_idx;
        // cells.getCellVtxAndIdx(indices[i], positions, vtx_idx);
        // cells.saveSingleCellEdges("cell" + std::to_string(indices[i]) + ".obj", vtx_idx, positions);
    }
    std::cout << std::endl;
    std::vector<Entry> yolk_hessian_entries;
    MatrixXT UV;
    woodbury = false;
    cells.addYolkVolumePreservationHessianEntries(yolk_hessian_entries, UV, false);
    woodbury = true;
    StiffnessMatrix global_hessian_yolk(num_nodes * 3, num_nodes * 3);
    global_hessian_yolk.setFromTriplets(yolk_hessian_entries.begin(), yolk_hessian_entries.end());
    cells.projectDirichletDoFMatrix(global_hessian_yolk, cells.dirichlet_data);
    MatrixXT yolk_hessian_dense = global_hessian_yolk.block(num_nodes * 3 - cells.basal_vtx_start * 3 - 1, 
        num_nodes * 3 - cells.basal_vtx_start * 3 - 1, 
        cells.basal_vtx_start * 3, cells.basal_vtx_start * 3);
    
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(yolk_hessian_dense, Eigen::ComputeThinU | Eigen::ComputeThinV);
    VectorXT Sigma = svd.singularValues();
    std::cout << "yolk term singular values: " << Sigma.tail<10>().transpose() << std::endl;
    
    Eigen::EigenSolver<Eigen::MatrixXd> eigen_solver;
    eigen_solver.compute(yolk_hessian_dense, /* computeEigenvectors = */ false);
    auto eigen_values = eigen_solver.eigenvalues();
    std::vector<T> ev_all(yolk_hessian_dense.cols());
    for (int i = 0; i < yolk_hessian_dense.cols(); i++)
    {
        ev_all[i] = eigen_values[i].real();
    }
    
    std::vector<int> sort_idx;
    for (int i = 0; i < yolk_hessian_dense.cols(); i++)
    {
        sort_idx.push_back(i);    
    }
    std::sort(sort_idx.begin(), sort_idx.end(), [&ev_all](int a, int b){ return ev_all[a] < ev_all[b]; } );
    std::cout << "yolk term eigen values: " << std::endl;
    for (int i = 0; i < 10; i++)
        std::cout << ev_all[sort_idx[i]] << " ";
    std::cout << std::endl;
    // VectorXT modes;
    // computeEigenValueSpectraSparse(sparse_mat, 20, modes, -1);
    // std::cout << modes << std::endl;
}



bool Simulation::solveWoodburyCholmod(StiffnessMatrix& K, MatrixXT& UV,
         VectorXT& residual, VectorXT& du)
{
    
    Eigen::SparseMatrix<T, Eigen::RowMajor, long int> K_prime = K;
    Timer t(true);
    Noether::CHOLMODSolver<long int> solver;
    T alpha = 10e-6;
    solver.set_pattern(K_prime);
    
    solver.analyze_pattern();    
    
    int i = 0;
    int indefinite_count_reg_cnt = 0, invalid_search_dir_cnt = 0, invalid_residual_cnt = 0;
    for (; i < 50; i++)
    {
        if (!solver.factorize())
        {
            // tbb::parallel_for(0, (int)K.rows(), [&](int row)    
            // {
            //     K_prime.coeffRef(row, row) += alpha;
            // }); 
            K_prime.diagonal().array() += alpha; 
            alpha *= 10;
            indefinite_count_reg_cnt++;
            continue;
        }
        
        // sherman morrison
        if (UV.cols() == 1)
        {
            VectorXT v = UV.col(0);
            VectorXT A_inv_g = VectorXT::Zero(du.rows());
            VectorXT A_inv_u = VectorXT::Zero(du.rows());
            solver.solve(residual.data(), A_inv_g.data(), true);
            solver.solve(v.data(), A_inv_u.data(), true);

            T dem = 1.0 + v.dot(A_inv_u);

            du.noalias() = A_inv_g - (A_inv_g.dot(v)) * A_inv_u / dem;
        }
        // UV is actually only U, since UV is the same in the case
        // C is assume to be Identity
        else // Woodbury https://en.wikipedia.org/wiki/Woodbury_matrix_identity
        {
            VectorXT A_inv_g = VectorXT::Zero(du.rows());
            solver.solve(residual.data(), A_inv_g.data(), true);
            // VectorXT A_inv_g = solver.solve(residual);

            MatrixXT A_inv_U(UV.rows(), UV.cols());
            for (int col = 0; col < UV.cols(); col++)
                solver.solve(UV.col(col).data(), A_inv_U.col(col).data(), true);
                // A_inv_U.col(col) = solver.solve(UV.col(col));
            
            MatrixXT C(UV.cols(), UV.cols());
            C.setIdentity();
            C += UV.transpose() * A_inv_U;
            du.noalias() = A_inv_g - A_inv_U * C.inverse() * UV.transpose() * A_inv_g;
        }
        

        T dot_dx_g = du.normalized().dot(residual.normalized());

        int num_negative_eigen_values = 0;
        int num_zero_eigen_value = 0;

        bool positive_definte = num_negative_eigen_values == 0;
        bool search_dir_correct_sign = dot_dx_g > 1e-6;
        if (!search_dir_correct_sign)
            invalid_search_dir_cnt++;
        bool solve_success = true;
        if (!solve_success)
            invalid_residual_cnt++;
        // std::cout << "PD: " << positive_definte << " direction " 
        //     << search_dir_correct_sign << " solve " << solve_success << std::endl;

        if (positive_definte && search_dir_correct_sign && solve_success)
        {
            t.stop();
            if (verbose)
            {
                std::cout << "\t===== Linear Solve ===== " << std::endl;
                std::cout << "\tnnz: " << K_prime.nonZeros() << std::endl;
                std::cout << "\t takes " << t.elapsed_sec() << "s" << std::endl;
                std::cout << "\t# regularization step " << i 
                    << " indefinite " << indefinite_count_reg_cnt 
                    << " invalid search dir " << invalid_search_dir_cnt
                    << " invalid solve " << invalid_residual_cnt << std::endl;
                std::cout << "\tdot(search, -gradient) " << dot_dx_g << std::endl;
                std::cout << "\t======================== " << std::endl;
                
            }
            return true;
            
        }
        else
        {
            // K = H + alpha * I;       
            // tbb::parallel_for(0, (int)K.rows(), [&](int row)    
            // {
            //     K.coeffRef(row, row) += alpha;
            // });  
            K.diagonal().array() += alpha; 
            alpha *= 10;
        }
    }
    return false;
}


bool Simulation::WoodburySolveNaive(StiffnessMatrix& A, const MatrixXT& UV,
        const VectorXT& b, VectorXT& x)
{
    Eigen::PardisoLLT<Eigen::SparseMatrix<T, Eigen::ColMajor, int>> solver;
    solver.analyzePattern(A);
    solver.factorize(A);
    if (UV.cols() == 1)
    {
        VectorXT v = UV.col(0);
        VectorXT A_inv_g = solver.solve(b);
        VectorXT A_inv_u = solver.solve(v);

        T dem = 1.0 + v.dot(A_inv_u);

        x = A_inv_g - (A_inv_g.dot(v)) * A_inv_u / dem;
    }
    // UV is actually only U, since UV is the same in the case
    // C is assume to be Identity
    else // Woodbury https://en.wikipedia.org/wiki/Woodbury_matrix_identity
    {
        VectorXT A_inv_g = solver.solve(b);

        MatrixXT A_inv_U(UV.rows(), UV.cols());
        for (int col = 0; col < UV.cols(); col++)
            A_inv_U.col(col) = solver.solve(UV.col(col));
        
        MatrixXT C(UV.cols(), UV.cols());
        C.setIdentity();
        C += UV.transpose() * A_inv_U;
        x = A_inv_g - A_inv_U * C.inverse() * UV.transpose() * A_inv_g;
    }
    return true;
}

bool Simulation::WoodburySolve(StiffnessMatrix& K, const MatrixXT& UV,
         VectorXT& residual, VectorXT& du)
{
    MatrixXT UVT= UV.transpose();
    bool use_cholmod = true;
    Timer t(true);
    // Eigen::SimplicialLLT<StiffnessMatrix> solver;
    Eigen::PardisoLLT<Eigen::SparseMatrix<T, Eigen::ColMajor, int>> solver;
    solver.pardisoParameterArray()[6] = 0;
    // solver.pardisoParameterArray()[59] = 1;
    auto& iparm = solver.pardisoParameterArray();
	iparm.setZero();
	iparm[0] = 1; /// not default values, set everything
	iparm[1] = 2; // ordering
	iparm[7] = 2;
	iparm[9] = 13; //pivot perturbation

	iparm[10] = 2; //scaling diagonals/vectors   this can only be used in conjunction with iparm[12] = 1
	iparm[12] = 1;

	iparm[17] = -1;
	iparm[18] = -1;
	iparm[20] = 1; //https://software.intel.com/en-us/mkl-developer-reference-c-pardiso-iparm-parameter

	iparm[23] = 1; // parallel solve
	iparm[24] = 1; // parallel backsubst.
	iparm[26] = 1; // check matrix
	iparm[34] = 1; // 0 based indexing


    T alpha = 10e-6;
    solver.analyzePattern(K);
    // T time_analyze = t.elapsed_sec();
    // std::cout << "\t analyzePattern takes " << time_analyze << "s" << std::endl;
    int i = 0;
    int indefinite_count_reg_cnt = 0, invalid_search_dir_cnt = 0, invalid_residual_cnt = 0;
    for (; i < 50; i++)
    {
        // std::cout << i << std::endl;

        solver.factorize(K);
        // T time_factorize = t.elapsed_sec() - time_analyze;
        // std::cout << "\t factorize takes " << time_factorize << "s" << std::endl;
        // std::cout << "-----factorization takes " << t.elapsed_sec() << "s----" << std::endl;
        if (solver.info() == Eigen::NumericalIssue)
        {
            // K = H + alpha * I;        
            // tbb::parallel_for(0, (int)K.rows(), [&](int row)    
            // {
            //     K.coeffRef(row, row) += alpha;
            // }); 
            K.diagonal().array() += alpha;
            alpha *= 10;
            indefinite_count_reg_cnt++;
            continue;
        }

        // sherman morrison
        if (UV.cols() == 1)
        {
            VectorXT v = UV.col(0);
            MatrixXT rhs(K.rows(), 2); rhs.col(0) = residual; rhs.col(1) = v;
            // VectorXT A_inv_g = solver.solve(residual);
            // VectorXT A_inv_u = solver.solve(v);
            MatrixXT A_inv_gu = solver.solve(rhs);

            T dem = 1.0 + v.dot(A_inv_gu.col(1));

            du.noalias() = A_inv_gu.col(0) - (A_inv_gu.col(0).dot(v)) * A_inv_gu.col(1) / dem;
        }
        // UV is actually only U, since UV is the same in the case
        // C is assume to be Identity
        else // Woodbury https://en.wikipedia.org/wiki/Woodbury_matrix_identity
        {
            VectorXT A_inv_g = solver.solve(residual);

            MatrixXT A_inv_U(UV.rows(), UV.cols());
            // for (int col = 0; col < UV.cols(); col++)
                // A_inv_U.col(col) = solver.solve(UV.col(col));
            A_inv_U = solver.solve(UV);

            MatrixXT C(UV.cols(), UV.cols());
            C.setIdentity();
            C += UVT * A_inv_U;
            du = A_inv_g - A_inv_U * C.inverse() * UVT * A_inv_g;
        }
        

        T dot_dx_g = du.normalized().dot(residual.normalized());

        int num_negative_eigen_values = 0;
        int num_zero_eigen_value = 0;

        bool positive_definte = num_negative_eigen_values == 0;
        bool search_dir_correct_sign = dot_dx_g > 1e-3;
        if (!search_dir_correct_sign)
            invalid_search_dir_cnt++;
        
        bool solve_success = true;//(K * du + UV * UV.transpose()*du - residual).norm() < 1e-6 && solver.info() == Eigen::Success;

        if (!solve_success)
            invalid_residual_cnt++;
        // std::cout << "PD: " << positive_definte << " direction " 
        //     << search_dir_correct_sign << " solve " << solve_success << std::endl;

        if (positive_definte && search_dir_correct_sign && solve_success)
        {
            t.stop();
            if (verbose)
            {
                std::cout << "\t===== Linear Solve ===== " << std::endl;
                std::cout << "\tnnz: " << K.nonZeros() << std::endl;
                std::cout << "\t takes " << t.elapsed_sec() << "s" << std::endl;
                std::cout << "\t# regularization step " << i 
                    << " indefinite " << indefinite_count_reg_cnt 
                    << " invalid search dir " << invalid_search_dir_cnt
                    << " invalid solve " << invalid_residual_cnt << std::endl;
                std::cout << "\tdot(search, -gradient) " << dot_dx_g << std::endl;
                std::cout << "\t======================== " << std::endl;
            }
            return true;
        }
        else
        {
            // K = H + alpha * I;       
            // tbb::parallel_for(0, (int)K.rows(), [&](int row)    
            // {
            //     K.coeffRef(row, row) += alpha;
            // });  
            K.diagonal().array() += alpha;
            alpha *= 10;
        }
    }
    return false;
}

bool Simulation::linearSolveNaive(StiffnessMatrix& A, const VectorXT& b, VectorXT& x)
{
    Eigen::PardisoLLT<StiffnessMatrix> solver;
    solver.pardisoParameterArray()[6] = 0;
    // solver.pardisoParameterArray()[59] = 1;
    auto& iparm = solver.pardisoParameterArray();
	iparm.setZero();
	iparm[0] = 1; /// not default values, set everything
	iparm[1] = 2; // ordering
	iparm[7] = 2;
	iparm[9] = 13; //pivot perturbation

	iparm[10] = 2; //scaling diagonals/vectors   this can only be used in conjunction with iparm[12] = 1
	iparm[12] = 1;

	iparm[17] = -1;
	iparm[18] = -1;
	iparm[20] = 1; //https://software.intel.com/en-us/mkl-developer-reference-c-pardiso-iparm-parameter

	iparm[23] = 1; // parallel solve
	iparm[24] = 1; // parallel backsubst.
	iparm[26] = 1; // check matrix
	iparm[34] = 1; // 0 based indexing

    solver.analyzePattern(A);
    solver.factorize(A);
    x = solver.solve(b);
}

bool Simulation::linearSolve(StiffnessMatrix& K, VectorXT& residual, VectorXT& du)
{
    Timer timer(true);
    
#define USE_PARDISO


#ifdef USE_PARDISO
    Eigen::PardisoLLT<StiffnessMatrix> solver;
    solver.pardisoParameterArray()[6] = 0;
    // solver.pardisoParameterArray()[59] = 1;
    auto& iparm = solver.pardisoParameterArray();
	iparm.setZero();
	iparm[0] = 1; /// not default values, set everything
	iparm[1] = 2; // ordering
	iparm[7] = 2;
	iparm[9] = 13; //pivot perturbation

	iparm[10] = 2; //scaling diagonals/vectors   this can only be used in conjunction with iparm[12] = 1
	iparm[12] = 1;

	iparm[17] = -1;
	iparm[18] = -1;
	iparm[20] = 1; //https://software.intel.com/en-us/mkl-developer-reference-c-pardiso-iparm-parameter

	iparm[23] = 1; // parallel solve
	iparm[24] = 1; // parallel backsubst.
	iparm[26] = 1; // check matrix
	iparm[34] = 1; // 0 based indexing

    // Eigen::PardisoLDLT<StiffnessMatrix> solver;
#else
    Eigen::SimplicialLDLT<StiffnessMatrix> solver;
    // Eigen::CholmodSimplicialLLT<StiffnessMatrix> solver;
#endif

    T alpha = 10e-6;
    // std::cout << "analyzePattern" << std::endl;
    solver.analyzePattern(K);
    int i = 0;
    for (; i < 50; i++)
    {
        
        solver.factorize(K);
        if (solver.info() == Eigen::NumericalIssue)
        {
            // std::cout << "indefinite" << std::endl;
            // tbb::parallel_for(0, (int)K.rows(), [&](int row)    
            // {
            //     K.coeffRef(row, row) += alpha;
            // });  
            K.diagonal().array() += alpha; 
            alpha *= 10;
            continue;
        }
        
        du = solver.solve(residual);

        T dot_dx_g = du.normalized().dot(residual.normalized());

        int num_negative_eigen_values = 0;
        int num_zero_eigen_value = 0;
#ifndef USE_PARDISO
        VectorXT d_vector = solver.vectorD();
        // std::cout << d_vector << std::endl;
        // std::getchar();
        for (int i = 0; i < d_vector.size(); i++)
        {
            if (d_vector[i] < 0)
            {
                num_negative_eigen_values++;
                // break;
            }
            if (std::abs(d_vector[i]) < 1e-6)
                num_zero_eigen_value++;
        }
        if (num_zero_eigen_value > 0)
        {
            std::cout << "num_zero_eigen_value " << num_zero_eigen_value << std::endl;
            return false;
        }
#endif
        bool positive_definte = num_negative_eigen_values == 0;
        bool search_dir_correct_sign = dot_dx_g > 1e-6;
        bool solve_success = (K*du - residual).norm() < 1e-6 && solver.info() == Eigen::Success;
        // bool solve_success = true;
        // std::cout << "PD: " << positive_definte << " direction " 
        //     << search_dir_correct_sign << " solve " << solve_success << std::endl;

        if (positive_definte && search_dir_correct_sign && solve_success)
        {
            timer.stop();
            if (verbose)
            {
                std::cout << "\t===== Linear Solve ===== " << std::endl;
                std::cout << "\ttakes " << timer.elapsed_sec() << "s" << std::endl;
                std::cout << "\t# regularization step " << i << std::endl;
                std::cout << "\tdot(search, -gradient) " << dot_dx_g << std::endl;
                std::cout << "\t======================== " << std::endl;
            }
            return true;
        }
        else
        {
            // tbb::parallel_for(0, (int)K.rows(), [&](int row)    
            // {
            //     K.coeffRef(row, row) += alpha;
            // });
            K.diagonal().array() += alpha; 
            alpha *= 10;
        }
    }
    return false;
}

void Simulation::buildSystemMatrix(const VectorXT& _u, StiffnessMatrix& K)
{
    cells.buildSystemMatrix(_u, K);
}

T Simulation::computeTotalEnergy(const VectorXT& _u, bool add_to_deform)
{
    T energy = cells.computeTotalEnergy(_u, false, add_to_deform);
    return energy;
}

T Simulation::computeResidual(const VectorXT& _u,  VectorXT& residual)
{
    return cells.computeResidual(_u, residual, verbose);
}


void Simulation::sampleEnergyWithSearchAndGradientDirection(
    const VectorXT& _u,  
    const VectorXT& search_direction,
    const VectorXT& negative_gradient)
{
    T E0 = computeTotalEnergy(_u);
    
    std::cout << std::setprecision(12) << "E0 " << E0 << std::endl;
    // T step_size = 5e-5;
    // int step = 200;

    T step_size = 1e-2;
    int step = 100; 

    // T step_size = 1e0;
    // int step = 50;

    

    std::vector<T> energies;
    std::vector<T> energies_gd;
    std::vector<T> steps;
    int step_cnt = 1;
    for (T xi = -T(step/2) * step_size; xi < T(step/2) * step_size; xi+=step_size)
    {
        // cells.use_sphere_radius_bound = false;
        // cells.add_contraction_term = false;
        cells.use_ipc_contact = false;
        // cells.weights_all_edges = 0.0;
        cells.sigma = 0;
        cells.gamma = 0;
        cells.alpha = 0.0;
        cells.B = 0;
        cells.By = 0;
        cells.Bp = 0;
        cells.add_tet_vol_barrier = false;
        cells.use_sphere_radius_bound = false;
        dynamic = false;
        T Ei = computeTotalEnergy(_u + xi * search_direction);
        
        // T Ei = cells.computeAreaEnergy(_u + xi * search_direction);
        // if (std::abs(xi) < 1e-6)
        //     std::getchar();
        energies.push_back(Ei);
        steps.push_back(xi);
    }
    
    for (T e : energies)
    {
        std::cout << std::setprecision(12) <<  e << " ";
    }
    std::cout << std::endl;
    for (T e : energies_gd)
    {
        std::cout << e << " ";
    }
    std::cout << std::endl;
    for (T idx : steps)
    {
        std::cout << idx << " ";
    }
    std::cout << std::endl;

}

void Simulation::buildSystemMatrixWoodbury(const VectorXT& _u, StiffnessMatrix& K, MatrixXT& UV)
{
    cells.buildSystemMatrixWoodbury(u, K, UV);
}

T Simulation::lineSearchNewton(VectorXT& _u,  VectorXT& residual, int ls_max, bool wolfe_condition)
{
    // for wolfe condition
    T c1 = 10e-4, c2 = 0.9;

    VectorXT du = residual;
    du.setZero();

    StiffnessMatrix K(residual.rows(), residual.rows());
    
    bool success = false;
    Timer ti(true);
    if (woodbury)
    {
        MatrixXT UV;
        buildSystemMatrixWoodbury(_u, K, UV);
        std::cout << "build system takes: " << ti.elapsed_sec() << "s" << std::endl;
        ti.restart();
        // ti.restart();
        // success = WoodburySolve(K, UV, residual, du);   
        success = solveWoodburyCholmod(K, UV, residual, du); 
        std::cout << "solve takes: " << ti.elapsed_sec() << "s" << std::endl;
        ti.restart();
    }
    else
    {
        buildSystemMatrix(_u, K);
        // std::cout << "built system" << std::endl;
        success = linearSolve(K, residual, du);    
    }
    if (!success)
    {
        std::cout << "linear solve failed" << std::endl;
        return 1e16;
    }

    T norm = du.norm();
    
    T alpha = cells.computeLineSearchInitStepsize(_u, du, verbose);
    std::cout << "computeLineSearchInitStepsize: " << ti.elapsed_sec() << std::endl;
    ti.restart();
    T E0 = computeTotalEnergy(_u);
    // std::cout << "E0 " << E0 << std::endl;
    // std::getchar();
    int cnt = 1;
    std::vector<T> ls_energies;
    while (true)
    {
        VectorXT u_ls = _u + alpha * du;
        T E1 = computeTotalEnergy(u_ls);
        ls_energies.push_back(E1);
        // std::cout << "ls# " << cnt << " E1 " << E1 << " alpha " << alpha << std::endl;
        // std::getchar();
        // cells.computeTotalEnergy(u_ls, true);
        // if (wolfe_condition)
        if (false)
        {
            bool Armijo = E1 <= E0 + c1 * alpha * du.dot(-residual);
            // std::cout << c1 * alpha * du.dot(-residual) << std::endl;
            VectorXT gradient_forward = VectorXT::Zero(deformed.rows());
            computeResidual(u_ls, gradient_forward);
            bool curvature = -du.dot(-gradient_forward) <= -c2 * du.dot(-residual);
            // std::cout << "wolfe Armijo " << Armijo << " curvature " << curvature << std::endl;
            if ((Armijo && curvature) || cnt > ls_max)
            {
                _u = u_ls;
                if (cnt > ls_max)
                {
                    if (verbose)
                        std::cout << "---ls max---" << std::endl;
                    // std::cout << "step size: " << alpha << std::endl;
                    // sampleEnergyWithSearchAndGradientDirection(_u, du, residual);
                    // cells.computeTotalEnergy(u_ls, true);
                    // cells.checkTotalGradientScale();
                    // cells.checkTotalHessianScale();
                    // return 1e16;
                }
                std::cout << "# ls " << cnt << std::endl;
                break;
            }
        }
        else
        {
            if (E1 - E0 < 0 || cnt > ls_max)
            {
                _u = u_ls;
                if (cnt > ls_max)
                {
                    if (verbose)
                        std::cout << "---ls max---" << std::endl;
                    // std::cout << "step size: " << alpha << std::endl;
                    // sampleEnergyWithSearchAndGradientDirection(_u, residual, residual);
                    // cells.checkTotalGradientScale();
                    // cells.print_force_norm = false;
                    // cells.checkTotalHessianScale();
                    // cells.print_force_norm = true;
                    // std::cout << "|du|: " << du.norm() << std::endl;
                    // std::cout << "E0: " << E0 << " E1 " << E1 << std::endl;
                    // for (T ei : ls_energies)
                    //     std::cout << std::setprecision(6) << ei << std::endl;
                    // std::getchar();
                    // cells.saveLowVolumeTets("low_vol_tet.obj");
                    // cells.saveBasalSurfaceMesh("low_vol_tet_basal_surface.obj");
                    // return 1e16;
                }
                if (verbose)
                    std::cout << "# ls " << cnt << " |du| " << alpha * du.norm() << std::endl;
                break;
            }
        }
        alpha *= 0.5;
        cnt += 1;
    }
    // std::cout << "line search: " << ti.elapsed_sec() << std::endl;
    ti.restart();
    // std::exit(0);
    return norm;
    if (cnt > ls_max)
    {
        // try gradien step
        std::cout << "taking gradient step " << std::endl;
        // std::cout << "|du|: " << du.norm() << " |g| " << residual.norm() << std::endl;
        // std::cout << "E0 " << E0 << std::endl;
        VectorXT negative_gradient_direction = residual.normalized();
        alpha = 1.0;
        cnt = 1;
        while (true)
        {
            VectorXT u_ls = _u + alpha * negative_gradient_direction;
            // _u = u_ls;
            // return 1e16;
            T E1 = computeTotalEnergy(u_ls);
            // std::cout << "ls gd # " << cnt << " E1 " << E1 << std::endl;
            if (E1 - E0 < 0 || cnt > 30)
            {
                _u = u_ls;
                if (cnt > 30)
                {
                    std::cout << "---gradient ls max---" << std::endl;
                    // cells.checkTotalGradient();
                    // std::cout << "|g|: " <<  residual.norm() << std::endl;
                    // cells.checkTotalGradientScale();
                    sampleEnergyWithSearchAndGradientDirection(_u, negative_gradient_direction, residual);
                    return 1e16;
                }
                // std::cout << "# ls " << cnt << std::endl;
                break;
            }
            alpha *= 0.5;
            cnt += 1;
        }
        
    }
    
    return norm;
}


void Simulation::loadDeformedState(const std::string& filename)
{
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    igl::readOBJ(filename, V, F);

    for (int i = 0; i < num_nodes; i++)
    {
        deformed.segment<3>(i * 3) = V.row(i);
    }
    u = deformed - undeformed;
    if (cells.use_ipc_contact)
    {
        cells.updateIPCVertices(u);
    }
    if (verbose)
        cells.computeCellInfo();
}

void Simulation::loadEdgeWeights(const std::string& filename, VectorXT& weights)
{
    std::ifstream in(filename);
    std::vector<T> weights_std_vec;
    T w;
    while (in >> w)
        weights_std_vec.push_back(w);
    weights = Eigen::Map<VectorXT>(weights_std_vec.data(), weights_std_vec.size());
    in.close();
}

void Simulation::loadVector(const std::string& filename, VectorXT& vector)
{
    std::ifstream in(filename);
    int n_entry;
    in >> n_entry;
    vector.resize(n_entry);
    for (int i = 0; i < n_entry; i++)
        in >> vector[i];
    in.close();
}