#include <igl/opengl/glfw/Viewer.h>
#include <igl/project.h>
#include <igl/unproject_on_plane.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>
#include <igl/png/writePNG.h>
#include <igl/writeOBJ.h>
#include <igl/readOBJ.h>
#include <igl/jet.h>

#include <boost/filesystem.hpp>

#include "../include/App.h"
// #include "../include/VoronoiCells.h"
// #include "../include/IntrinsicSimulation.h"

inline bool fileExist (const std::string& name) {
    std::ifstream f(name.c_str());
    return f.good();
}

using CMat = Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic>;
using TV = Vector<T, 3>;
using TV3 = Vector<T, 3>;
using TM3 = Matrix<T, 3, 3>;
using VectorXT = Eigen::Matrix<T, Eigen::Dynamic, 1>;


int main(int argc, char** argv)
{
    std::string exp = "bumpy_cube";
    bool with_viewer = true;
    bool render_seq = false;
    bool run_sim = true;
    bool intrinsic_sim = false;
    if (intrinsic_sim)
    {
        IntrinsicSimulation intrinsic_simulation;
        // intrinsic_simulation.initializeMassSpringDebugScene();
        // intrinsic_simulation.initializeTriangleDebugScene();
        // intrinsic_simulation.movePointsPlotEnergy();
        intrinsic_simulation.initializeMassSpringSceneExactGeodesic();
        // intrinsic_simulation.initializeSceneCheckingSmoothness();
        // intrinsic_simulation.movePointsPlotEnergy();
        if (with_viewer)
        {
            igl::opengl::glfw::Viewer viewer;
            igl::opengl::glfw::imgui::ImGuiMenu menu;

            viewer.plugins.push_back(&menu);
            

            GeodesicSimApp app(intrinsic_simulation);    
            app.setViewer(viewer, menu);
            if (render_seq)
            {
                viewer.core().camera_zoom *= 1.8;
                viewer.launch_init(true, false, "WuKong viewer", 2000, 1600);
                for(int i = 0; i < 50; i++)
                {
                    app.updateScreen(viewer);
                    int w = viewer.core().viewport(2), h = viewer.core().viewport(3);
                    CMat R(w,h), G(w,h), B(w,h), A(w,h);
                    viewer.core().draw_buffer(viewer.data(),true,R,G,B,A);
                    A.setConstant(255);
                    igl::png::writePNG(R,G,B,A, exp+"_frame"+std::to_string(i)+".png");
                    bool converged = intrinsic_simulation.advanceOneStep(i);
                    if (converged)
                        break;
                }
            }
            else
            {
                viewer.launch(true, false, "WuKong viewer", 2000, 1600);
            }
        }
        else
        {
            if (!run_sim)
                return 0;
            for (int i = 0; i < intrinsic_simulation.max_newton_iter; i++)
            {
                bool converged = intrinsic_simulation.advanceOneStep(i);
                if (converged)
                    break;
            }
            
        }
    }
    else
    {
        VoronoiCells voronoi_cells;
        
        
        VoronoiApp app(voronoi_cells);
        igl::opengl::glfw::Viewer viewer;
        igl::opengl::glfw::imgui::ImGuiMenu menu;

        viewer.plugins.push_back(&menu);
        app.setViewer(viewer, menu);
        viewer.launch(true, false, "WuKong viewer", 2000, 1600);
    }

    


    return 0;
}