#include "../include/Simulation.h"
#include <Eigen/PardisoSupport>
#include <Eigen/CholmodSupport>
#include "../../../Solver/CHOLMODSolver.hpp"

#include <igl/readOBJ.h>

#include <iomanip>
#include <ipc/ipc.hpp>

#define FOREVER 30000


void generatePolygonRendering(Eigen::MatrixXd& V, Eigen::MatrixXi& F, 
        Eigen::MatrixXd& C)
{

}

void Simulation::computeLinearModes()
{
    cells.computeLinearModes();
}

void Simulation::initializeCells()
{
    woodbury = true;
    cells.use_alm_on_cell_volume = false;

    std::string sphere_file;
    cells.scene_type = 1;

    if (cells.scene_type == 1 || cells.scene_type == 2)
        // sphere_file = "/home/yueli/Documents/ETH/WuKong/Projects/CellSim/data/sphere_2k.obj";
        // sphere_file = "/home/yueli/Documents/ETH/WuKong/Projects/CellSim/data/sphere.obj";
        // sphere_file = "/home/yueli/Documents/ETH/WuKong/Projects/CellSim/data/drosophila_embryo_4k.obj";
        // sphere_file = "/home/yueli/Documents/ETH/WuKong/Projects/CellSim/data/drosophila_embryo_1k.obj";
        // sphere_file = "/home/yueli/Documents/ETH/WuKong/Projects/CellSim/data/drosophila_embryo_476.obj";
        // sphere_file = "/home/yueli/Documents/ETH/WuKong/Projects/CellSim/data/drosophila_embryo_120.obj";
        // sphere_file = "/home/yueli/Documents/ETH/WuKong/Projects/CellSim/data/drosophila_real_1.5k.obj";
        // sphere_file = "/home/yueli/Documents/ETH/WuKong/Projects/CellSim/data/drosophila_real_486.obj";
        sphere_file = "/home/yueli/Documents/ETH/WuKong/Projects/CellSim/data/drosophila_real_241.obj";
        
    else if(cells.scene_type == 0)
        sphere_file = "/home/yueli/Documents/ETH/WuKong/Projects/CellSim/data/sphere_lowres.obj";
    cells.vertexModelFromMesh(sphere_file);
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
    
}

// void Simulation::setViewer(igl::opengl::glfw::Viewer& viewer)
// {
//     igl::opengl::glfw::imgui::ImGuiMenu menu;

//     viewer.plugins.push_back(&menu);

//     menu.callback_draw_viewer_menu = [&]()
//     {
//         if (ImGui::CollapsingHeader("Visualization", ImGuiTreeNodeFlags_DefaultOpen))
//         {
//             if (ImGui::Checkbox("SelectVertex", &viewer_data.enable_selection))
//             {
//                 updateScreen(viewer);
//             }
//             if (ImGui::Checkbox("ShowCurrent", &viewer_data.show_current))
//             {
//                 updateScreen(viewer);
//             }
//             if (ImGui::Checkbox("ShowRest", &viewer_data.show_rest))
//             {
//                 updateScreen(viewer);
//             }
//             if (ImGui::Checkbox("SplitPrism", &viewer_data.split))
//             {
//                 updateScreen(viewer);
//             }
//             if (ImGui::Checkbox("SplitPrismABit", &viewer_data.split_a_bit))
//             {
//                 updateScreen(viewer);
//             }
//             if (ImGui::Checkbox("ShowMembrane", &viewer_data.show_membrane))
//             {
//                 updateScreen(viewer);
//             }
//             if (ImGui::Checkbox("YolkOnly", &viewer_data.yolk_only))
//             {
//                 updateScreen(viewer);
//             }
//             if (ImGui::Checkbox("ContractingEdges", &viewer_data.show_contracting_edges))
//             {
//                 updateScreen(viewer);
//             }
//             if (ImGui::Checkbox("ShowOutsideVtx", &viewer_data.show_outside_vtx))
//             {
//                 updateScreen(viewer);
//             }
//             if (ImGui::Checkbox("ComputeEnergy", &viewer_data.compute_energy))
//             {
//                 updateScreen(viewer);
//             }
//         }
//         if (ImGui::CollapsingHeader("Simulation", ImGuiTreeNodeFlags_DefaultOpen))
//         {
//             if (ImGui::Checkbox("Dynamics", &dynamic))
//             {
//                 if (dynamic)
//                     initializeDynamicsData(1e-2, 5e-2);
//             }
//         }
//         if (ImGui::Button("StaticSolve", ImVec2(-1,0)))
//         {
//             staticSolve();
//             updateScreen(viewer);
//         }
//         if (ImGui::Button("Reset", ImVec2(-1,0)))
//         {
//             deformed = undeformed;
//             u.setZero();
//             updateScreen(viewer);
//         }
//         if (ImGui::Button("SaveMesh", ImVec2(-1,0)))
//         {
//             igl::writeOBJ("current_mesh.obj", viewer_data.V, viewer_data.F);
//         }
//     };

//     viewer.callback_mouse_down = [&](igl::opengl::glfw::Viewer&, int, int)->bool
//     {
//         if (!viewer_data.enable_selection)
//             return false;
//         double x = viewer.current_mouse_x;
//         double y = viewer.core().viewport(3) - viewer.current_mouse_y;

//         for (int i = 0; i < cells.num_nodes; i++)
//         {
//             Vector<T, 3> pos = deformed.template segment<3>(i * 3);
//             Eigen::MatrixXd x3d(1, 3); x3d.setZero();
//             x3d.row(0).template segment<3>(0) = pos;

//             Eigen::MatrixXd pxy(1, 3);
//             igl::project(x3d, viewer.core().view, viewer.core().proj, viewer.core().viewport, pxy);
//             if(abs(pxy.row(0)[0]-x)<20 && abs(pxy.row(0)[1]-y)<20)
//             {
//                 std::cout << "selected " << i << std::endl;
//                 return true;
//             }
//         }
//         return false;
//     };

//     viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer &) -> bool
//     {
//         if(viewer.core().is_animating && viewer_data.check_modes)
//         {
//             deformed = undeformed + u + viewer_data.evectors.col(viewer_data.modes) * std::sin(viewer_data.t);
//             if (viewer_data.compute_energy)
//             {
//                 verbose = false;
//                 T energy = computeTotalEnergy(u, false);
//                 verbose = false;
//                 std::cout << std::setprecision(8) << "E: " << energy << std::endl;
//             }
//             viewer_data.t += 0.1;
//             viewer_data.compute_energy_cnt++;
            
//             viewer.data().clear();
//             generateMeshForRendering(viewer_data.V, viewer_data.F, viewer_data.C, 
//                 viewer_data.show_current, viewer_data.show_rest, viewer_data.split, 
//                 viewer_data.split_a_bit, viewer_data.yolk_only);
//             viewer.data().set_mesh(viewer_data.V, viewer_data.F);     
//             viewer.data().set_colors(viewer_data.C);
//             if (viewer_data.show_membrane)
//             {
//                 viewer.data().set_points(viewer_data.bounding_surface_samples, viewer_data.bounding_surface_samples_color);
//             }
//         }
//         return false;
//     };

//     viewer.callback_post_draw = [&](igl::opengl::glfw::Viewer &) -> bool
//     {
//         if(viewer.core().is_animating && !viewer_data.check_modes)
//         {
//             bool finished = advanceOneStep(viewer_data.static_solve_step);
//             if (finished)
//             {
//                 viewer.core().is_animating = false;
//             }
//             else 
//                 viewer_data.static_solve_step++;
//             updateScreen(viewer);
//         }
//         return false;
//     };

//     viewer.callback_key_pressed = 
//         [&](igl::opengl::glfw::Viewer &,unsigned int key,int mods)->bool
//     {
//         VectorXT residual(num_nodes * 3);
//         residual.setZero();
//         switch(key)
//         {
//         default: 
//             return false;
//         case ' ':
//             viewer.core().is_animating = true;
//             return true;
//         case '0':
//             viewer_data.check_modes = true;
//             // loadEigenVectors("/home/yueli/Documents/ETH/WuKong/cell_svd_vectors.txt");
//             // loadEigenVectors("/home/yueli/Documents/ETH/WuKong/dxdp.txt");
//             viewer_data.modes = 0;
//             std::cout << "modes " << viewer_data.modes << std::endl;
//             return true;
//         case '1':
//             viewer_data.check_modes = true;
//             computeLinearModes();
//             // loadEigenVectors("/home/yueli/Documents/ETH/WuKong/cell_eigen_vectors.txt");
            
//             for (int i = 0; i < viewer_data.evalues.rows(); i++)
//             {
//                 if (viewer_data.evalues[i] > 1e-6)
//                 {
//                     viewer_data.modes = i;
//                     return true;
//                 }
//             }
//             return true;
//         case '2':
//             viewer_data.modes++;
//             viewer_data.modes = (viewer_data.modes + viewer_data.evectors.cols()) % viewer_data.evectors.cols();
//             std::cout << "modes " << viewer_data.modes << std::endl;
//             return true;
//         case '3': //check modes at equilirium after static solve
//             std::cout << "state: " << viewer_data.load_obj_iter_cnt << std::endl;
//             loadDeformedState("output/cells/cell/cell_mesh_iter_" + std::to_string(viewer_data.load_obj_iter_cnt) + ".obj");
//             std::cout << computeResidual(u, residual) << std::endl;
//             updateScreen(viewer);
//             return true;
//         case 'a':
//             viewer.core().is_animating = !viewer.core().is_animating;
//             return true;
//         case 'n':
//             viewer_data.load_obj_iter_cnt++;
//             std::cout << "state: " << viewer_data.load_obj_iter_cnt << std::endl;
//             loadDeformedState("output/cells/cell/cell_mesh_iter_" + std::to_string(viewer_data.load_obj_iter_cnt) + ".obj");
//             updateScreen(viewer);
//             return true;
//         case 'l':
//             viewer_data.load_obj_iter_cnt--;
//             viewer_data.load_obj_iter_cnt = std::max(0, viewer_data.load_obj_iter_cnt);
//             std::cout << "state: " << viewer_data.load_obj_iter_cnt << std::endl;
//             loadDeformedState("output/cells/cell/cell_mesh_iter_" + std::to_string(viewer_data.load_obj_iter_cnt) + ".obj");
//             updateScreen(viewer);
//             return true;
//         }
//     };

//     initializeCells();
//     dynamic = false;
//     if (dynamic)
//         initializeDynamicsData(1e0, 10000);

//     sampleBoundingSurface(viewer_data.bounding_surface_samples);
//     viewer_data.sdf_test_sample_idx_offset = viewer_data.bounding_surface_samples.rows();
//     viewer_data.bounding_surface_samples_color = viewer_data.bounding_surface_samples;
//     for (int i = 0; i < viewer_data.bounding_surface_samples.rows(); i++)
//         viewer_data.bounding_surface_samples_color.row(i) = TV(0.1, 1.0, 0.1);

//     // cells.loadMeshAndSaveCentroid("output/cells/cell_drosophila_4k_with_cephalic", 0, 1117);
//     // verbose = true;
//     cells.print_force_norm = false;
//     // sa.initialize();
//     // sa.svdOnSensitivityMatrix();
//     // sa.optimizePerEdgeWeigths();
    
    
//     updateScreen(viewer);

//     viewer.core().background_color.setOnes();
//     viewer.data().set_face_based(true);
//     viewer.data().shininess = 1.0;
//     viewer.data().point_size = 10.0;

//     viewer.data().set_mesh(viewer_data.V, viewer_data.F);     
//     viewer.data().set_colors(viewer_data.C);

//     viewer.core().align_camera_center(viewer_data.V);

//     viewer.launch();
// }

// void Simulation::updateScreen(igl::opengl::glfw::Viewer& viewer)
// {
//     generateMeshForRendering(viewer_data.V, viewer_data.F, viewer_data.C, 
//                 viewer_data.show_current, viewer_data.show_rest, viewer_data.split, 
//                 viewer_data.split_a_bit, viewer_data.yolk_only);

//     viewer.data().clear();
//     // viewer.data().set_mesh(V, F);
//     // viewer.data().set_colors(C);

//     if (viewer_data.show_contracting_edges)
//     {
//         // viewer.data().clear();
//         cells.appendCylinderOnContractingEdges(viewer_data.V, viewer_data.F, viewer_data.C);
//     }
        
//     if (viewer_data.show_membrane)
//     {
//         viewer.data().set_points(viewer_data.bounding_surface_samples, viewer_data.bounding_surface_samples_color);
//     }
//     if (viewer_data.show_outside_vtx)
//     {
//         cells.getOutsideVtx(viewer_data.bounding_surface_samples, 
//             viewer_data.bounding_surface_samples_color, viewer_data.sdf_test_sample_idx_offset);
//         viewer.data().set_points(viewer_data.bounding_surface_samples, viewer_data.bounding_surface_samples_color);
//     }
//     viewer.data().set_mesh(viewer_data.V, viewer_data.F);
//     viewer.data().set_colors(viewer_data.C);   
// }

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
        // std::cout << "[Newton] computeResidual takes " << step_timer.elapsed_sec() << "s" << std::endl;
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

void Simulation::saveState(const std::string& filename)
{
    std::ofstream out(filename);
    Eigen::MatrixXd V, C;
    Eigen::MatrixXi F;
    cells.generateMeshForRendering(V, F, C, false);
    for (int i = 0; i < V.rows(); i++)
    {
        out << "v " << V.row(i) << std::endl;
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
        cells.computeIPCRestData();
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

        if(cnt == max_newton_iter || dq_norm > 1e10 || dq_norm < 1e-8)
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
    if (cnt == max_newton_iter || dq_norm > 1e10 || residual_norm > 1)
        return false;
    return true;
}

bool Simulation::solveWoodburyCholmod(StiffnessMatrix& K, MatrixXT& UV,
         VectorXT& residual, VectorXT& du)
{
    
    Timer t(true);
    
    Noether::CHOLMODSolver<typename StiffnessMatrix::StorageIndex> solver;
    T alpha = 10e-6;
    solver.set_pattern(K);
    solver.analyze_pattern();    
    int i = 0;
    int indefinite_count_reg_cnt = 0, invalid_search_dir_cnt = 0, invalid_residual_cnt = 0;
    for (; i < 50; i++)
    {
        if (!solver.factorize())
        {
            tbb::parallel_for(0, (int)K.rows(), [&](int row)    
            {
                K.coeffRef(row, row) += alpha;
            }); 
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

            du = A_inv_g - (A_inv_g.dot(v)) * A_inv_u / dem;
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
            du = A_inv_g - A_inv_U * C.inverse() * UV.transpose() * A_inv_g;
        }
        

        T dot_dx_g = du.normalized().dot(residual.normalized());

        int num_negative_eigen_values = 0;
        int num_zero_eigen_value = 0;

        bool positive_definte = num_negative_eigen_values == 0;
        bool search_dir_correct_sign = dot_dx_g > 1e-3;
        if (!search_dir_correct_sign)
            invalid_search_dir_cnt++;
        bool solve_success = ((K + UV * UV.transpose())*du - residual).norm() < 1e-6;
        if (!solve_success)
            invalid_residual_cnt++;
        // std::cout << "PD: " << positive_definte << " direction " 
        //     << search_dir_correct_sign << " solve " << solve_success << std::endl;

        if (positive_definte && search_dir_correct_sign && solve_success)
        {
            t.stop();
            std::cout << "\t===== Linear Solve ===== " << std::endl;
            std::cout << "\tnnz: " << K.nonZeros() << std::endl;
            std::cout << "\t takes " << t.elapsed_sec() << "s" << std::endl;
            std::cout << "\t# regularization step " << i 
                << " indefinite " << indefinite_count_reg_cnt 
                << " invalid search dir " << invalid_search_dir_cnt
                << " invalid solve " << invalid_residual_cnt << std::endl;
            std::cout << "\tdot(search, -gradient) " << dot_dx_g << std::endl;
            std::cout << "\t======================== " << std::endl;
            return true;
        }
        else
        {
            // K = H + alpha * I;       
            tbb::parallel_for(0, (int)K.rows(), [&](int row)    
            {
                K.coeffRef(row, row) += alpha;
            });  
            alpha *= 10;
        }
    }
    return false;
}


bool Simulation::WoodburySolve(StiffnessMatrix& K, const MatrixXT& UV,
         VectorXT& residual, VectorXT& du)
{
    bool use_cholmod = true;
    Timer t(true);

    Eigen::PardisoLLT<Eigen::SparseMatrix<T, Eigen::ColMajor, int>> solver;

    T alpha = 10e-6;
    solver.analyzePattern(K);
    int i = 0;
    int indefinite_count_reg_cnt = 0, invalid_search_dir_cnt = 0, invalid_residual_cnt = 0;
    for (; i < 50; i++)
    {
        // std::cout << i << std::endl;

        solver.factorize(K);
        // std::cout << "-----factorization takes " << t.elapsed_sec() << "s----" << std::endl;
        if (solver.info() == Eigen::NumericalIssue)
        {
            // K = H + alpha * I;        
            tbb::parallel_for(0, (int)K.rows(), [&](int row)    
            {
                K.coeffRef(row, row) += alpha;
            }); 
            alpha *= 10;
            indefinite_count_reg_cnt++;
            continue;
        }

        // sherman morrison
        if (UV.cols() == 1)
        {
            VectorXT v = UV.col(0);
            VectorXT A_inv_g = solver.solve(residual);
            VectorXT A_inv_u = solver.solve(v);

            T dem = 1.0 + v.dot(A_inv_u);

            du = A_inv_g - (A_inv_g.dot(v)) * A_inv_u / dem;
        }
        // UV is actually only U, since UV is the same in the case
        // C is assume to be Identity
        else // Woodbury https://en.wikipedia.org/wiki/Woodbury_matrix_identity
        {
            VectorXT A_inv_g = solver.solve(residual);

            MatrixXT A_inv_U(UV.rows(), UV.cols());
            for (int col = 0; col < UV.cols(); col++)
                A_inv_U.col(col) = solver.solve(UV.col(col));
            
            MatrixXT C(UV.cols(), UV.cols());
            C.setIdentity();
            C += UV.transpose() * A_inv_U;
            du = A_inv_g - A_inv_U * C.inverse() * UV.transpose() * A_inv_g;
        }
        

        T dot_dx_g = du.normalized().dot(residual.normalized());

        int num_negative_eigen_values = 0;
        int num_zero_eigen_value = 0;

        bool positive_definte = num_negative_eigen_values == 0;
        bool search_dir_correct_sign = dot_dx_g > 1e-3;
        if (!search_dir_correct_sign)
            invalid_search_dir_cnt++;
        bool solve_success = (K * du + UV * UV.transpose()*du - residual).norm() < 1e-6 && solver.info() == Eigen::Success;
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
            tbb::parallel_for(0, (int)K.rows(), [&](int row)    
            {
                K.coeffRef(row, row) += alpha;
            });  
            alpha *= 10;
        }
    }
    return false;
}

bool Simulation::linearSolve(StiffnessMatrix& K, VectorXT& residual, VectorXT& du)
{
    Timer timer(true);
#define USE_PARDISO


#ifdef USE_PARDISO
    Eigen::PardisoLLT<Eigen::SparseMatrix<T, Eigen::ColMajor, int>> solver;
#else
    Eigen::SimplicialLDLT<StiffnessMatrix> solver;
    // Eigen::CholmodSimplicialLLT<StiffnessMatrix> solver;
#endif

    T alpha = 10e-6;
    solver.analyzePattern(K);
    int i = 0;
    for (; i < 50; i++)
    {
        
        solver.factorize(K);
        if (solver.info() == Eigen::NumericalIssue)
        {
            // std::cout << "indefinite" << std::endl;
            tbb::parallel_for(0, (int)K.rows(), [&](int row)    
            {
                K.coeffRef(row, row) += alpha;
            });  
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
            tbb::parallel_for(0, (int)K.rows(), [&](int row)    
            {
                K.coeffRef(row, row) += alpha;
            });        
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
        // std::cout << "build system: " << ti.elapsed_sec() << std::endl;
        // ti.restart();
        success = WoodburySolve(K, UV, residual, du);   
        // success = solveWoodburyCholmod(K, UV, residual, du); 
        // std::cout << "solve: " << ti.elapsed_sec() << std::endl;
        // ti.restart();
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
    // std::cout << "computeLineSearchInitStepsize: " << ti.elapsed_sec() << std::endl;
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
    if (verbose)
        cells.computeCellInfo();
}