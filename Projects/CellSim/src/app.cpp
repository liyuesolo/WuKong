#include "../include/app.h"

void SimulationApp::updateScreen(igl::opengl::glfw::Viewer& viewer)
{
    simulation.generateMeshForRendering(V, F, C, show_current, show_rest, split, split_a_bit, yolk_only);

    viewer.data().clear();
    // viewer.data().set_mesh(V, F);
    // viewer.data().set_colors(C);

    if (show_contracting_edges)
    {
        // viewer.data().clear();
        simulation.cells.appendCylinderOnContractingEdges(V, F, C);
    }
        
    if (show_membrane)
    {
        viewer.data().set_points(bounding_surface_samples, bounding_surface_samples_color);
    }
    if (show_outside_vtx)
    {
        simulation.cells.getOutsideVtx(bounding_surface_samples, 
            bounding_surface_samples_color, sdf_test_sample_idx_offset);
        viewer.data().set_points(bounding_surface_samples, bounding_surface_samples_color);
    }
    viewer.data().set_mesh(V, F);
    viewer.data().set_colors(C);   
}

void SimulationApp::loadDisplacementVectors(const std::string& filename)
{
    std::ifstream in(filename);
    int row, col;
    in >> row >> col;
    evectors.resize(row, col);
    evalues.resize(col);
    double entry;
    for (int i = 0; i < col; i++)
        in >> evalues[i];
    for (int i = 0; i < row; i++)
        for (int j = 0; j < col; j++)
            in >> evectors(i, j);
    in.close();
}

void SimulationApp::setMenu(igl::opengl::glfw::Viewer& viewer, 
    igl::opengl::glfw::imgui::ImGuiMenu& menu)
{
    menu.callback_draw_viewer_menu = [&]()
    {
        if (ImGui::CollapsingHeader("Visualization", ImGuiTreeNodeFlags_DefaultOpen))
        {
            if (ImGui::Checkbox("SelectVertex", &enable_selection))
            {
                updateScreen(viewer);
            }
            if (ImGui::Checkbox("ShowCurrent", &show_current))
            {
                updateScreen(viewer);
            }
            if (ImGui::Checkbox("ShowRest", &show_rest))
            {
                updateScreen(viewer);
            }
            if (ImGui::Checkbox("SplitPrism", &split))
            {
                updateScreen(viewer);
            }
            if (ImGui::Checkbox("SplitPrismABit", &split_a_bit))
            {
                updateScreen(viewer);
            }
            if (ImGui::Checkbox("ShowMembrane", &show_membrane))
            {
                updateScreen(viewer);
            }
            if (ImGui::Checkbox("YolkOnly", &yolk_only))
            {
                updateScreen(viewer);
            }
            if (ImGui::Checkbox("ContractingEdges", &show_contracting_edges))
            {
                updateScreen(viewer);
            }
            if (ImGui::Checkbox("ShowOutsideVtx", &show_outside_vtx))
            {
                updateScreen(viewer);
            }
            if (ImGui::Checkbox("ComputeEnergy", &compute_energy))
            {
                updateScreen(viewer);
            }
        }
        if (ImGui::CollapsingHeader("Simulation", ImGuiTreeNodeFlags_DefaultOpen))
        {
            if (ImGui::Checkbox("Dynamics", &simulation.dynamic))
            {
                if (simulation.dynamic)
                    simulation.initializeDynamicsData(1e-2, 5e-2);
            }
        }
        if (ImGui::Button("StaticSolve", ImVec2(-1,0)))
        {
            simulation.staticSolve();
            updateScreen(viewer);
        }
        if (ImGui::Button("Reset", ImVec2(-1,0)))
        {
            simulation.deformed = simulation.undeformed;
            simulation.u.setZero();
            updateScreen(viewer);
        }
        if (ImGui::Button("SaveMesh", ImVec2(-1,0)))
        {
            igl::writeOBJ("current_mesh.obj", V, F);
        }
    };
}

void SimulationApp::setMouseDown(igl::opengl::glfw::Viewer& viewer)
{
    viewer.callback_mouse_down = [&](igl::opengl::glfw::Viewer& viewer, int, int)->bool
    {
        if (!enable_selection)
            return false;
        double x = viewer.current_mouse_x;
        double y = viewer.core().viewport(3) - viewer.current_mouse_y;

        for (int i = 0; i < simulation.cells.num_nodes; i++)
        {
            Vector<T, 3> pos = simulation.deformed.template segment<3>(i * 3);
            Eigen::MatrixXd x3d(1, 3); x3d.setZero();
            x3d.row(0).template segment<3>(0) = pos;

            Eigen::MatrixXd pxy(1, 3);
            igl::project(x3d, viewer.core().view, viewer.core().proj, viewer.core().viewport, pxy);
            if(abs(pxy.row(0)[0]-x)<20 && abs(pxy.row(0)[1]-y)<20)
            {
                std::cout << "selected " << i << std::endl;
                return true;
            }
        }
        return false;
    };
}

void SimulationApp::setViewer(igl::opengl::glfw::Viewer& viewer, igl::opengl::glfw::imgui::ImGuiMenu& menu)
{
    
    setMenu(viewer, menu);

    setMouseDown(viewer);

    viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer &) -> bool
    {
        if(viewer.core().is_animating && check_modes)
        {
            simulation.deformed = simulation.undeformed + simulation.u + evectors.col(modes) * std::sin(t);
            if (compute_energy)
            {
                simulation.verbose = false;
                T energy = simulation.computeTotalEnergy(simulation.u, false);
                simulation.verbose = false;
                std::cout << std::setprecision(8) << "E: " << energy << std::endl;
            }
            t += 0.1;
            compute_energy_cnt++;
            
            viewer.data().clear();
            simulation.generateMeshForRendering(V, F, C, show_current, show_rest, split, split_a_bit, yolk_only);
            viewer.data().set_mesh(V, F);     
            viewer.data().set_colors(C);
            if (show_membrane)
            {
                viewer.data().set_points(bounding_surface_samples, bounding_surface_samples_color);
            }
        }
        return false;
    };

    viewer.callback_post_draw = [&](igl::opengl::glfw::Viewer &) -> bool
    {
        if(viewer.core().is_animating && !check_modes)
        {
            bool finished = simulation.advanceOneStep(static_solve_step);
            if (finished)
            {
                viewer.core().is_animating = false;
            }
            else 
                static_solve_step++;
            updateScreen(viewer);
        }
        return false;
    };

    viewer.callback_key_pressed = 
        [&](igl::opengl::glfw::Viewer & viewer,unsigned int key,int mods)->bool
    {
        VectorXT residual(simulation.num_nodes * 3);
        residual.setZero();
        switch(key)
        {
        default: 
            return false;
        case ' ':
            viewer.core().is_animating = true;
            return true;
        case '0':
            check_modes = true;
            loadDisplacementVectors("/home/yueli/Documents/ETH/WuKong/cell_svd_vectors.txt");
            // loadDisplacementVectors("/home/yueli/Documents/ETH/WuKong/dxdp.txt");
            modes = 0;
            std::cout << "modes " << modes << std::endl;
            return true;
        case '1':
            check_modes = true;
            simulation.computeLinearModes();
            loadDisplacementVectors("/home/yueli/Documents/ETH/WuKong/cell_eigen_vectors.txt");
            
            for (int i = 0; i < evalues.rows(); i++)
            {
                if (evalues[i] > 1e-6)
                {
                    modes = i;
                    return true;
                }
            }
            return true;
        case '2':
            modes++;
            modes = (modes + evectors.cols()) % evectors.cols();
            std::cout << "modes " << modes << std::endl;
            return true;
        case '3': //check modes at equilirium after static solve
            std::cout << "state: " << load_obj_iter_cnt << std::endl;
            simulation.loadDeformedState("output/cells/cell/cell_mesh_iter_" + std::to_string(load_obj_iter_cnt) + ".obj");
            std::cout << simulation.computeResidual(simulation.u, residual) << std::endl;
            updateScreen(viewer);
            return true;
        case 'a':
            viewer.core().is_animating = !viewer.core().is_animating;
            return true;
        case 'n':
            load_obj_iter_cnt++;
            std::cout << "state: " << load_obj_iter_cnt << std::endl;
            simulation.loadDeformedState("output/cells/cell/cell_mesh_iter_" + std::to_string(load_obj_iter_cnt) + ".obj");
            updateScreen(viewer);
            return true;
        case 'l':
            load_obj_iter_cnt--;
            load_obj_iter_cnt = std::max(0, load_obj_iter_cnt);
            std::cout << "state: " << load_obj_iter_cnt << std::endl;
            simulation.loadDeformedState("output/cells/cell/cell_mesh_iter_" + std::to_string(load_obj_iter_cnt) + ".obj");
            updateScreen(viewer);
            return true;
        }
    };

    simulation.initializeCells();
    simulation.dynamic = false;
    if (simulation.dynamic)
        simulation.initializeDynamicsData(1e0, 10000);

    simulation.sampleBoundingSurface(bounding_surface_samples);
    sdf_test_sample_idx_offset = bounding_surface_samples.rows();
    bounding_surface_samples_color = bounding_surface_samples;
    for (int i = 0; i < bounding_surface_samples.rows(); i++)
        bounding_surface_samples_color.row(i) = TV(0.1, 1.0, 0.1);

    // simulation.cells.loadMeshAndSaveCentroid("output/cells/cell_drosophila_4k_with_cephalic", 0, 1117);
    // simulation.verbose = true;
    simulation.cells.print_force_norm = false;
    // sa.initialize();
    // sa.svdOnSensitivityMatrix();
    // sa.optimizePerEdgeWeigths();
    

    updateScreen(viewer);

    viewer.core().background_color.setOnes();
    viewer.data().set_face_based(true);
    viewer.data().shininess = 1.0;
    viewer.data().point_size = 10.0;

    viewer.data().set_mesh(V, F);     
    viewer.data().set_colors(C);

    viewer.core().align_camera_center(V);
}

void DiffSimApp::runOptimization()
{
    sa.initialize();
    sa.optimizePerEdgeWeigths();
}

void DiffSimApp::setViewer(igl::opengl::glfw::Viewer& viewer, igl::opengl::glfw::imgui::ImGuiMenu& menu)
{
    setMenu(viewer, menu);
    setMouseDown(viewer);

    viewer.callback_key_pressed = 
        [&](igl::opengl::glfw::Viewer & viewer,unsigned int key,int mods)->bool
    {
        VectorXT dx, dp;
        switch(key)
        {
        default: 
            return false;
        case '0':
            check_modes = true;
            loaddxdp("dxdp.txt", dx, dp);
            modes = 0;
            std::cout << "modes " << modes << std::endl;
            return true;
        case ' ':
            viewer.core().is_animating = true;
            return true;
        case 'a':
            viewer.core().is_animating = !viewer.core().is_animating;
            return true;
        }
    };

    viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer &) -> bool
    {
        if(viewer.core().is_animating && check_modes)
        {
            simulation.deformed = simulation.undeformed + simulation.u + evectors.col(modes) * std::sin(t);
            
            t += 0.1;
            
            viewer.data().clear();
            simulation.generateMeshForRendering(V, F, C, show_current, show_rest, split, split_a_bit, yolk_only);
            viewer.data().set_mesh(V, F);     
            viewer.data().set_colors(C);
            if (show_membrane)
            {
                viewer.data().set_points(bounding_surface_samples, bounding_surface_samples_color);
            }
        }
        return false;
    };

    viewer.callback_post_draw = [&](igl::opengl::glfw::Viewer &) -> bool
    {
        if(viewer.core().is_animating && !check_modes)
        {
            // bool finished = sa.optimizeOneStep(opt_step);
            // if (finished)
            // {
            //     viewer.core().is_animating = false;
            // }
            // else 
            //     opt_step++;
            // updateScreen(viewer);
        }
        return false;
    };

    // simulation.initializeCells();

    simulation.sampleBoundingSurface(bounding_surface_samples);
    sdf_test_sample_idx_offset = bounding_surface_samples.rows();
    bounding_surface_samples_color = bounding_surface_samples;
    for (int i = 0; i < bounding_surface_samples.rows(); i++)
        bounding_surface_samples_color.row(i) = TV(0.1, 1.0, 0.1);


    updateScreen(viewer);

    viewer.core().background_color.setOnes();
    viewer.data().set_face_based(true);
    viewer.data().shininess = 1.0;
    viewer.data().point_size = 10.0;

    viewer.data().set_mesh(V, F);     
    viewer.data().set_colors(C);

    viewer.core().align_camera_center(V);
}

void DiffSimApp::updateScreen(igl::opengl::glfw::Viewer& viewer)
{
    simulation.generateMeshForRendering(V, F, C, show_current, show_rest, split, split_a_bit, yolk_only);

    viewer.data().clear();

    if (show_contracting_edges)
        simulation.cells.appendCylinderOnContractingEdges(V, F, C);
        
    if (show_membrane)
        viewer.data().set_points(bounding_surface_samples, bounding_surface_samples_color);

    if (show_outside_vtx)
    {
        simulation.cells.getOutsideVtx(bounding_surface_samples, 
            bounding_surface_samples_color, sdf_test_sample_idx_offset);
        viewer.data().set_points(bounding_surface_samples, bounding_surface_samples_color);
    }
    viewer.data().set_mesh(V, F);
    viewer.data().set_colors(C);  
}

void DiffSimApp::loaddxdp(const std::string& filename, VectorXT& dx, VectorXT& dp)
{
    int n_dof_sim = sa.n_dof_sim;
    int n_dof_design = sa.n_dof_design;
    dx = VectorXT::Zero(n_dof_sim);
    dp = VectorXT::Zero(n_dof_design);

    std::ifstream in(filename);
    for (int i = 0; i < n_dof_sim; i++)
        in >> dx[i];
    for (int i = 0; i < n_dof_design; i++)
        in >> dp[i];
    in.close();

    evectors = dx;
}