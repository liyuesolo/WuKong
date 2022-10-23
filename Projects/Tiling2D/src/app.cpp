#include "../include/app.h"

void App::appendCylindersToEdges(const std::vector<std::pair<TV3, TV3>>& edge_pairs, 
        const std::vector<TV3>& color, T radius,
        Eigen::MatrixXd& _V, Eigen::MatrixXi& _F, Eigen::MatrixXd& _C)
{
    int n_div = 10;
    T theta = 2.0 * EIGEN_PI / T(n_div);
    VectorXT points = VectorXT::Zero(n_div * 3);
    for(int i = 0; i < n_div; i++)
        points.segment<3>(i * 3) = TV3(radius * std::cos(theta * T(i)), 
        0.0, radius * std::sin(theta*T(i)));

    int offset_v = n_div * 2;
    int offset_f = n_div * 2;

    int n_row_V = _V.rows();
    int n_row_F = _F.rows();

    int n_edge = edge_pairs.size();

    _V.conservativeResize(n_row_V + offset_v * n_edge, 3);
    _F.conservativeResize(n_row_F + offset_f * n_edge, 3);
    _C.conservativeResize(n_row_F + offset_f * n_edge, 3);

    tbb::parallel_for(0, n_edge, [&](int ei)
    {
        TV3 axis_world = edge_pairs[ei].second - edge_pairs[ei].first;
        TV3 axis_local(0, axis_world.norm(), 0);

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

            _F.row(n_row_F + ei * offset_f + i*2 ) = IV3(n_row_V + ei * offset_v + i, 
                                    n_row_V + ei * offset_v + i+n_div, 
                                    n_row_V + ei * offset_v + (i+1)%(n_div));

            _F.row(n_row_F + ei * offset_f + i*2 + 1) = IV3(n_row_V + ei * offset_v + (i+1)%(n_div), 
                                        n_row_V + ei * offset_v + i+n_div, 
                                        n_row_V + + ei * offset_v + (i+1)%(n_div) + n_div);

            _C.row(n_row_F + ei * offset_f + i*2 ) = color[ei];
            _C.row(n_row_F + ei * offset_f + i*2 + 1) = color[ei];
        }
    });
}

void App::loadDisplacementVectors(const std::string& filename)
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

void SimulationApp::updateScreen(igl::opengl::glfw::Viewer& viewer)
{
    tiling.generateMeshForRendering(V, F, C, show_PKstress);

    if (tile_in_x_only)
        tiling.tilingMeshInX(V, F, C);

    if (tile_XY)
        tiling.tileUnitCell(V, F, C, 2);


    if (connect_pbc_pairs)
    {
        std::vector<std::pair<TV3, TV3>> end_points;
        tiling.solver.getPBCPairs3D(end_points);
        T ref_dis = (end_points[0].first - end_points[1].second).norm();
        std::vector<TV3> colors;
        for (int i = 0; i < end_points.size(); i++)
            colors.push_back(TV3(1.0, 0.3, 0.0));
        appendCylindersToEdges(end_points, colors, 0.001 * ref_dis, V, F, C);
    }    
    
    viewer.data().clear();
    viewer.data().set_mesh(V, F);
    viewer.data().set_colors(C);
}

void SimulationApp::setViewer(igl::opengl::glfw::Viewer& viewer,
    igl::opengl::glfw::imgui::ImGuiMenu& menu)
{
    menu.callback_draw_viewer_menu = [&]()
    {
        if (ImGui::CollapsingHeader("Visualization", ImGuiTreeNodeFlags_DefaultOpen))
        {
            if (ImGui::Checkbox("ShowStrain", &show_PKstress))
            {
                updateScreen(viewer);
            } 
        }
        if (ImGui::CollapsingHeader("Simulation", ImGuiTreeNodeFlags_DefaultOpen))
        {
            if (ImGui::Checkbox("PBC", &tiling.solver.add_pbc))
            {

            } 
            if (tiling.solver.add_pbc)
            {
                if (ImGui::Checkbox("ConnectPBC", &connect_pbc_pairs))
                {
                    // tiling.solver.addPBCPairInX();
                    updateScreen(viewer);
                }
                if (ImGui::Checkbox("TileInXOnly", &tile_in_x_only))
                {
                    updateScreen(viewer);
                }
                if (ImGui::Checkbox("TileUnitCell", &tile_XY))
                {
                    updateScreen(viewer);
                }
                
            }
        }
        if (ImGui::Button("GenerateOne", ImVec2(-1,0)))
        {
            tiling.generateOneStructure();
            updateScreen(viewer);
            viewer.core().align_camera_center(V);
        }
        if (ImGui::Button("GeneratePeriodicUnit", ImVec2(-1,0)))
        {
            tiling.generateOnePerodicUnit();
            updateScreen(viewer);
            viewer.core().align_camera_center(V);
        }
        if (ImGui::Button("GenerateWithRotation", ImVec2(-1,0)))
        {
            tiling.generateOneStructureWithRotation();
            updateScreen(viewer);
            viewer.core().align_camera_center(V);
        }
        if (ImGui::Button("GenerateNonPeriodic", ImVec2(-1,0)))
        {
            tiling.generateOneNonperiodicStructure();
            updateScreen(viewer);
            viewer.core().align_camera_center(V);
        }
        if (ImGui::Button("GenerateBatch", ImVec2(-1,0)))
        {
            // tiling.generateSandwichStructureBatch();
            tiling.generateSandwichBatchChangingTilingParams();
        }
        if (ImGui::Button("StaticSolve", ImVec2(-1,0)))
        {
            tiling.solver.staticSolve();
            updateScreen(viewer);
        }
        if (ImGui::Button("Reset", ImVec2(-1,0)))
        {
            tiling.solver.reset();
            static_solve_step = 0;
            updateScreen(viewer);
        }
        if (ImGui::Button("LoadVTK", ImVec2(-1,0)))
        {
            std::string fname = igl::file_dialog_open();
            if (fname.length() != 0)
            {
                tiling.initializeSimulationDataFromFiles(fname, PBC_X);
                updateScreen(viewer);
            }
        }
        if (ImGui::Button("LoadMesh", ImVec2(-1,0)))
        {
            std::string fname = igl::file_dialog_open();
            if (fname.length() != 0)
            {
                tiling.solver.loadOBJ(fname);
                updateScreen(viewer);
            }
        }
        if (ImGui::Button("LoadUndeformedMesh", ImVec2(-1,0)))
        {
            std::string fname = igl::file_dialog_open();
            if (fname.length() != 0)
            {
                tiling.solver.loadOBJ(fname, true);
                updateScreen(viewer);
            }
        }
        if (ImGui::Button("SaveForces", ImVec2(-1,0)))
        {
            tiling.solver.savePenaltyForces("force.txt");
        }
        if (ImGui::Button("SaveMesh", ImVec2(-1,0)))
        {
            igl::writeOBJ("/home/yueli/Documents/ETH/WuKong/build/Projects/Tiling2D/current_mesh.obj", V, F);
        }
        if (ImGui::Button("SaveIPCMesh", ImVec2(-1,0)))
        {
            if (tiling.solver.use_ipc)
            {
                tiling.solver.saveIPCMesh("/home/yueli/Documents/ETH/WuKong/build/Projects/Tiling2D/ipc_mesh.obj");
            }    
        }
    };    

    viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer &) -> bool
    {
        if(viewer.core().is_animating && check_modes)
        {
            tiling.solver.deformed = tiling.solver.undeformed + tiling.solver.u + evectors.col(modes) * std::sin(t);
            updateScreen(viewer);
            t += 0.1;
        }
        return false;
    };

    viewer.callback_post_draw = [&](igl::opengl::glfw::Viewer &) -> bool
    {
        if(viewer.core().is_animating && !check_modes)
        {
            
            bool finished = tiling.solver.staticSolveStep(static_solve_step);
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
        
        switch(key)
        {
        default: 
            return false;
        case 's':
            tiling.solver.staticSolveStep(static_solve_step);
            updateScreen(viewer);
            return true;
        case ' ':
            viewer.core().is_animating = true;
            // tiling.solver.optimizeIPOPT();
            return true;
        case '1':
            check_modes = true;
            tiling.solver.checkHessianPD(true);
            // tiling.solver.computeLinearModes();
            loadDisplacementVectors("eigen_vectors.txt");
            
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
        case 'a':
            viewer.core().is_animating = !viewer.core().is_animating;
            return true;
        case 'd':
            // tiling.solver.checkTotalGradient(false);
            if (tiling.solver.use_ipc && tiling.solver.ipc_vertices.rows() == 0)
            {
                tiling.solver.computeIPCRestData();
            }

            // tiling.solver.checkTotalGradient(false);
            tiling.solver.checkTotalGradientScale(true);
            tiling.solver.checkTotalHessianScale(true);
            // tiling.solver.checkTotalHessian(false);
            return true;

        }
    };
    // tiling.initializeSimulationDataFromVTKFile(tiling.data_folder + "thickshell.vtk");
    // tiling.initializeSimulationDataFromFiles("thickshellPatchPeriodicInX", true);
    // tiling.initializeSimulationDataFromFiles("/home/yueli/Documents/ETH/SandwichStructure/TilingVTK/50.vtk", true);
    // tiling.generateForceDisplacementCurve("/home/yueli/Documents/ETH/WuKong/build/Projects/Tiling2D/results/");
    
    updateScreen(viewer);
    viewer.core().background_color.setOnes();
    viewer.data().set_face_based(true);
    viewer.data().shininess = 1.0;
    viewer.data().point_size = 25.0;

    viewer.core().align_camera_center(V);
    // viewer.core().toggle(viewer.data().show_lines);
    viewer.core().animation_max_fps = 24.;
    // key_down(viewer,'0',0);
    viewer.core().is_animating = false;
}

void Simulation3DApp::updateScreen(igl::opengl::glfw::Viewer& viewer)
{
    solver.generateMeshForRendering(V, F, C);
    viewer.data().clear();
    viewer.data().set_mesh(V, F);
    viewer.data().set_colors(C);
}

void Simulation3DApp::setViewer(igl::opengl::glfw::Viewer& viewer,
    igl::opengl::glfw::imgui::ImGuiMenu& menu)
{
    menu.callback_draw_viewer_menu = [&]()
    {
        if (ImGui::CollapsingHeader("Visualization", ImGuiTreeNodeFlags_DefaultOpen))
        {
            
        }
        if (ImGui::CollapsingHeader("Simulation", ImGuiTreeNodeFlags_DefaultOpen))
        {
            if (ImGui::Checkbox("Kirchhoff-Love", &solver.plain_strain))
            {
                std::cout << solver.plain_strain << std::endl;
            }
            if (ImGui::Checkbox("StVK", &solver.stvk))
            {

            } 
        }
        if (ImGui::Button("Reset", ImVec2(-1,0)))
        {
            solver.reset();
            static_solve_step = 0;
            updateScreen(viewer);
        }
    };

    viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer &) -> bool
    {
        if(viewer.core().is_animating && check_modes)
        {
            solver.deformed = solver.undeformed + solver.u + evectors.col(modes) * std::sin(t);
            updateScreen(viewer);
            t += 0.1;
        }
        return false;
    };

    viewer.callback_post_draw = [&](igl::opengl::glfw::Viewer &) -> bool
    {
        if(viewer.core().is_animating && !check_modes)
        {
            
            bool finished = solver.staticSolveStep(static_solve_step);
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
        
        switch(key)
        {
        default: 
            return false;
        case 's':
            solver.staticSolveStep(static_solve_step);
            updateScreen(viewer);
            return true;
        case ' ':
            viewer.core().is_animating = true;
            
            return true;
        case '1':
            check_modes = true;
            solver.checkHessianPD(true);
            loadDisplacementVectors("eigen_vectors.txt");
            
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
        case 'a':
            viewer.core().is_animating = !viewer.core().is_animating;
            return true;
        case 'd':
            // solver.checkTotalGradient(true);
            // solver.checkTotalHessian(true);
            return true;

        }
    };

    updateScreen(viewer);
    viewer.core().background_color.setOnes();
    viewer.data().set_face_based(true);
    viewer.data().shininess = 1.0;
    viewer.data().point_size = 25.0;

    viewer.core().align_camera_center(V);
    // viewer.core().toggle(viewer.data().show_lines);
    viewer.core().animation_max_fps = 24.;
    // key_down(viewer,'0',0);
    viewer.core().is_animating = false;
}
