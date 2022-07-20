#include "../include/app.h"

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
    tiling.generateMeshForRendering(V, F, C);
        
    
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
            
        }
        if (ImGui::CollapsingHeader("Simulation", ImGuiTreeNodeFlags_DefaultOpen))
        {
            
        }
        if (ImGui::Button("StaticSolve", ImVec2(-1,0)))
        {
            
        }
        if (ImGui::Button("Reset", ImVec2(-1,0)))
        {
            
        }
        if (ImGui::Button("SaveMesh", ImVec2(-1,0)))
        {
            igl::writeOBJ("current_mesh.obj", V, F);
        }
    };    

    viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer &) -> bool
    {
        if(viewer.core().is_animating && check_modes)
        {
            // tiling.solver.deformed = tiling.solver.undeformed + tiling.solver.u + evectors.col(modes) * std::sin(t);
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
            return true;
        case ' ':
            viewer.core().is_animating = true;
            
            updateScreen(viewer);
            return true;
        case '1':
            check_modes = true;
            // tiling.solver.computeLinearModes();
            // loadDisplacementVectors("/home/yueli/Documents/ETH/WuKong/fem_eigen_vectors.txt");
            
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
        }
    };
    tiling.initializeSimulationDataFromVTKFile("thickshell.vtk");

    updateScreen(viewer);
    viewer.core().background_color.setOnes();
    viewer.data().set_face_based(true);
    viewer.data().shininess = 1.0;
    viewer.data().point_size = 25.0;

    viewer.core().align_camera_center(V);
    viewer.core().animation_max_fps = 24.;
    // key_down(viewer,'0',0);
    viewer.core().is_animating = false;
}

void TilingViewerApp::updateScreen(igl::opengl::glfw::Viewer& viewer)
{
    if (show_unit)
    {

    }
}

void TilingViewerApp::setViewer(igl::opengl::glfw::Viewer& viewer,
    igl::opengl::glfw::imgui::ImGuiMenu& menu)
{
    menu.callback_draw_viewer_menu = [&]()
    {
        if (ImGui::CollapsingHeader("Visualization", ImGuiTreeNodeFlags_DefaultOpen))
        {
            if (ImGui::Checkbox("ShowUnit", &show_unit))
            {
                updateScreen(viewer);
            }
        }
    };   
    VectorXT vertices;
    std::vector<Vector<int, 2>> edge_list;
    
    tiling.getPBCUnit(vertices, edge_list);
    updateScreen(viewer);
    viewer.core().background_color.setOnes();
    viewer.data().set_face_based(true);
    viewer.data().shininess = 1.0;
    viewer.data().point_size = 25.0;

    viewer.core().align_camera_center(V);
    viewer.core().animation_max_fps = 24.;
    // key_down(viewer,'0',0);
    viewer.core().is_animating = false;
}