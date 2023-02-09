#include <igl/colormap.h>
#include <igl/copyleft/tetgen/tetrahedralize.h>
#include <igl/readOBJ.h>
#include <igl/edges.h>
#include <igl/slice.h>
#include "../include/App.h"
#include <igl/png/writePNG.h>
#include <igl/png/readPNG.h>
#include<random>
#include<cmath>
#include<chrono>

void SimulationApp::updateScreen(igl::opengl::glfw::Viewer& viewer)
{
    simulation.generateMeshForRendering(V, F, C, yolk_only, cells_only);
    viewer.data().clear();
    viewer.data().set_mesh(V, F);
    viewer.data().set_colors(C);   
}

void SimulationApp::setMenu(igl::opengl::glfw::Viewer& viewer, 
    igl::opengl::glfw::imgui::ImGuiMenu& menu)
{
    menu.callback_draw_viewer_menu = [&]()
    {
        if (ImGui::CollapsingHeader("Visualization", ImGuiTreeNodeFlags_DefaultOpen))
        {
            if (ImGui::Checkbox("YolkOnly", &yolk_only))
            {
                updateScreen(viewer);
            }
            if (ImGui::Checkbox("CellsOnly", &cells_only))
            {
                updateScreen(viewer);
            }
        }
        if (ImGui::Button("SaveMesh", ImVec2(-1,0)))
        {
            igl::writeOBJ("current_mesh.obj", V, F);
        }
    };
}

void SimulationApp::setViewer(igl::opengl::glfw::Viewer& viewer, igl::opengl::glfw::imgui::ImGuiMenu& menu)
{
    
    setMenu(viewer, menu);

    viewer.callback_post_draw = [&](igl::opengl::glfw::Viewer &) -> bool
    {
        if(viewer.core().is_animating && !check_modes)
        {
            bool finished = simulation.advanceOneStep(static_solve_step);
            if (finished)
            {
                viewer.core().is_animating = false;
                // simulation.checkHessianPD(false);
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
        case 's':
            simulation.advanceOneStep(static_solve_step++);
            updateScreen(viewer);
            return true;
        case 'd':
            simulation.checkTotalGradientScale();
            simulation.checkTotalHessianScale();
            return true;
        }
    };

    updateScreen(viewer);
    
    viewer.core().background_color.setOnes();
    viewer.data().set_face_based(true);
    viewer.data().shininess = 1.0;
    viewer.data().point_size = 10.0;

    viewer.data().set_mesh(V, F);     
    viewer.data().set_colors(C);

    viewer.core().align_camera_center(V);
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

void SimulationApp::loadSVDData(const std::string& filename)
{
    std::ifstream in(filename);
    int row, col;
    in >> row >> col;
    svd_U.resize(row, col);
    svd_Sigma.resize(col);
    double entry;
    for (int i = 0; i < col; i++)
        in >> svd_Sigma[i];
    for (int i = 0; i < row; i++)
        for (int j = 0; j < col; j++)
            in >> svd_U(i, j);
    in >> row >> col;
    
    svd_V.resize(row, col);
    for (int i = 0; i < row; i++)
        for (int j = 0; j < col; j++)
            in >> svd_V(i, j);
    in.close();
}