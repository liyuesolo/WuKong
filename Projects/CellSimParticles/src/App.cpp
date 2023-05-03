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

template<int dim>
void SimulationApp<dim>::updateScreen(igl::opengl::glfw::Viewer& viewer)
{
    viewer.data().clear();
    V.resize(0, 0); F.resize(0, 0); C.resize(0, 0);
    if (show_cell_center)
    {
        if (show_collision_sphere)
            simulation.generateMeshForRendering(V, F, C, false, simulation.radius, simulation.radius * 0.3);
        else
            simulation.generateMeshForRendering(V, F, C, false, simulation.radius * 0.3, simulation.radius * 0.3);
    }
    if (show_membrane)
    {
        // if (simulation.use_surface_membrane)
        if (false)
        {
            int n_vtx_current = V.rows(); int n_face_curernt = F.rows();
            V.conservativeResize(n_vtx_current + simulation.membrane_vtx.rows(), 3);
            F.conservativeResize(n_face_curernt + simulation.membrane_faces.rows(), 3);
            C.conservativeResize(n_face_curernt + simulation.membrane_faces.rows(), 3);
            V.block(n_vtx_current, 0, simulation.membrane_vtx.rows(), 3) = simulation.membrane_vtx;
            MatrixXi offset(simulation.membrane_faces.rows(), 3); 
            offset.setConstant(n_vtx_current);
            F.block(n_face_curernt, 0, simulation.membrane_faces.rows(), 3) = simulation.membrane_faces + offset;
            C.block(n_face_curernt, 0, simulation.membrane_faces.rows(), 1).setOnes();
            C.block(n_face_curernt, 1, simulation.membrane_faces.rows(), 1).setConstant(0.3);
            C.block(n_face_curernt, 2, simulation.membrane_faces.rows(), 1).setZero();
        }
        else
        {
            viewer.data().set_points(bounding_surface_samples, bounding_surface_samples_color);
        }
    }
    if (show_matching_pairs)
    {
        auto edges = simulation.control_edges;
        std::vector<std::pair<TV3, TV3>> edge_pairs(edges.size());
        tbb::parallel_for(0, (int)edges.size(), [&](int i)
        {
            TV vi = simulation.deformed.template segment<dim>(edges[i][0] * dim);
            TV vj = simulation.control_points.template segment<dim>(edges[i][1] * dim);
            if constexpr (dim == 3)
                edge_pairs[i] = std::make_pair(vi, vj);
            else
                edge_pairs[i] = std::make_pair(TV3(vi[0], vi[1], 0), TV3(vj[0], vj[1], 0));
        });
        simulation.appendCylindersToEdges(edge_pairs, TV3(0, 1, 0), 0.2 * simulation.radius, V, F, C);
    }
    if (show_adh_pairs)
    {
        auto edges = simulation.adhesion_edges;
        std::vector<std::pair<TV3, TV3>> edge_pairs(edges.size());
        tbb::parallel_for(0, (int)edges.size(), [&](int i)
        {
            TV vi = simulation.deformed.template segment<dim>(edges[i][0] * dim);
            TV vj = simulation.deformed.template segment<dim>(edges[i][1] * dim);
            if constexpr (dim == 3)
                edge_pairs[i] = std::make_pair(vi, vj);
            else
                edge_pairs[i] = std::make_pair(TV3(vi[0], vi[1], 0), TV3(vj[0], vj[1], 0));
        });
        simulation.appendCylindersToEdges(edge_pairs, TV3(0, 1, 0), 0.2 * simulation.radius, V, F, C);
    }
    if (show_control_points)
    {
        simulation.appendSphereToPositionVector(simulation.control_points, simulation.radius*0.3, TV3(1, 0, 0), V, F, C);
    }
    if (show_yolk_surface)
    {
        if (simulation.yolk_cell_starts != -1)
        {
            if constexpr (dim == 2)
            {
                std::vector<std::pair<TV3, TV3>> edge_pairs(simulation.num_nodes - simulation.yolk_cell_starts);
                tbb::parallel_for(simulation.yolk_cell_starts, simulation.num_nodes, [&](int i)
                {
                    int j;
                    if (i == simulation.num_nodes - 1)
                        j = simulation.yolk_cell_starts;
                    else
                        j = i + 1;

                    TV vi = simulation.deformed.template segment<dim>(i * dim);
                    TV vj = simulation.deformed.template segment<dim>(j * dim);
                    if constexpr (dim == 3)
                        edge_pairs[i - simulation.yolk_cell_starts] = std::make_pair(vi, vj);
                    else
                        edge_pairs[i - simulation.yolk_cell_starts] = std::make_pair(TV3(vi[0], vi[1], 0), TV3(vj[0], vj[1], 0));
                });
                simulation.appendCylindersToEdges(edge_pairs, TV3(0, 1, 0), 0.1 * simulation.radius, V, F, C);
            }
            else
            {
                
                int n_vtx_current = V.rows(); int n_face_curernt = F.rows();
                V.conservativeResize(n_vtx_current + simulation.num_nodes - simulation.yolk_cell_starts, 3);
                F.conservativeResize(n_face_curernt + simulation.ipc_faces.rows(), 3);
                C.conservativeResize(n_face_curernt + simulation.ipc_faces.rows(), 3);
                for (int i = simulation.yolk_cell_starts; i < simulation.num_nodes; i++)
                {
                    V.row(n_vtx_current + i - simulation.yolk_cell_starts) = 
                        simulation.deformed.template segment<dim>(i * dim);
                }
                MatrixXi offset(simulation.ipc_faces.rows(), 3); 
                offset.setConstant(n_vtx_current - simulation.yolk_cell_starts);
                F.block(n_face_curernt, 0, simulation.ipc_faces.rows(), 3) = simulation.ipc_faces + offset;
                C.block(n_face_curernt, 0, simulation.ipc_faces.rows(), 1).setZero();
                C.block(n_face_curernt, 1, simulation.ipc_faces.rows(), 1).setOnes();
                C.block(n_face_curernt, 2, simulation.ipc_faces.rows(), 1).setZero();
                
            }
        }
    }
    if (show_yolk_triangle)
    {
        std::vector<std::pair<TV3, TV3>> edge_pairs(simulation.num_nodes - simulation.yolk_cell_starts);
        tbb::parallel_for(simulation.yolk_cell_starts, simulation.num_nodes, [&](int i)
        {
            TV vi = simulation.deformed.template segment<dim>(i * dim);
            TV vj = simulation.centroid.template head<dim>();
            if constexpr (dim == 3)
                edge_pairs[i - simulation.yolk_cell_starts] = std::make_pair(vi, vj);
            else
                edge_pairs[i - simulation.yolk_cell_starts] = std::make_pair(TV3(vi[0], vi[1], 0), TV3(vj[0], vj[1], 0));
        });
        simulation.appendCylindersToEdges(edge_pairs, TV3(0, 0, 0), 0.02 * simulation.radius, V, F, C);
    }
    
    if (show_multiview && !show_voronoi)
    {
        
        TM3 R90z; R90z << std::cos(0.5 * M_PI),  -std::sin(0.5 * M_PI), 0, std::sin(0.5 * M_PI), std::cos(0.5 * M_PI), 0, 0, 0, 1.0;
        TM3 R90y; R90y << std::cos(0.5 * M_PI),  0, std::sin(0.5 * M_PI), 0, 1, 0, -std::sin(0.5 * M_PI), 0, std::cos(0.5 * M_PI);
        TM3 nR90y; nR90y << std::cos(-0.5 * M_PI),  0, std::sin(-0.5 * M_PI), 0, 1, 0, -std::sin(-0.5 * M_PI), 0, std::cos(-0.5 * M_PI);
        MatrixXT V_lateral_left = V;
        MatrixXT V_dorsal = V;
        MatrixXT V_ventral = V;
        MatrixXT V_lateral_right = V;
        for (int i = 0; i < V.rows(); i++)
        {
            V_lateral_left.row(i) = (R90z * V.row(i).transpose()).transpose();
            V_dorsal.row(i) = (R90y * V_lateral_left.row(i).transpose()).transpose();
            V_lateral_right.row(i) = (R90y * V_dorsal.row(i).transpose()).transpose();
            V_ventral.row(i) = (nR90y * V_lateral_left.row(i).transpose()).transpose();
        }
        int n_vtx_current = V.rows();
        int n_face_current = F.rows();
        TV min_corner, max_corner;
        simulation.computeBoundingBox(min_corner, max_corner);
        MatrixXT offset_vertice = V;
        offset_vertice.setZero();
        offset_vertice.col(0).setConstant((max_corner-min_corner).norm() * 0.5);
        V.resize(n_vtx_current * 4, 3); 
        // V.block(0, 0, n_vtx_current, 3) = V_ventral - 1.8 * offset_vertice;
        // V.block(n_vtx_current, 0, n_vtx_current, 3) = V_dorsal - 0.8 * offset_vertice;
        // V.block(2 * n_vtx_current, 0, n_vtx_current, 3) = V_lateral_left + 0.2 * offset_vertice;
        // V.block(3 * n_vtx_current, 0, n_vtx_current, 3) = V_lateral_right + offset_vertice * 1.0;
        V.block(0, 0, n_vtx_current, 3) = V_ventral ;
        V.block(n_vtx_current, 0, n_vtx_current, 3) = V_dorsal + 1.0 * offset_vertice;
        V.block(2 * n_vtx_current, 0, n_vtx_current, 3) = V_lateral_left + 2.0 * offset_vertice;
        V.block(3 * n_vtx_current, 0, n_vtx_current, 3) = V_lateral_right + offset_vertice * 3.0;
        
        MatrixXi offset(n_face_current, 3);
        offset.setConstant(n_vtx_current);
        MatrixXi F_current = F;
        
        MatrixXT C_current = C;
        F.resize(n_face_current * 4, 3);
        C.resize(n_face_current * 4, 3);
        for (int i = 0; i < 4; i++)
        {
            F.block(i * n_face_current, 0, n_face_current, 3) = F_current + i * offset;    
            C.block(i * n_face_current, 0, n_face_current, 3) = C_current;    
        }
        
        // viewer.core().align_camera_center(V);
        // viewer.core().camera_zoom *= 1.2;
    }
    if (show_voronoi)
    {
        if constexpr (dim == 3)
        {
            std::cout << "show voronoi" << std::endl;
            SpatialHash<dim> hash_table;
            TV min_corner, max_corner;
            simulation.computeBoundingBox(min_corner, max_corner);
            T ref_dis = 0.02 * (max_corner-min_corner).norm();
            VectorXT cell_centers = simulation.deformed.segment(0, simulation.num_cells * dim);
            hash_table.build(1.5 * ref_dis, cell_centers);
            int n_samples_per_cell = 1000;
            voronoi_samples.resize(n_samples_per_cell * simulation.num_cells, 3);
            voronoi_samples_colors.resize(n_samples_per_cell * simulation.num_cells, 3);
            tbb::parallel_for(0, simulation.num_cells, [&](int i)
            {
                TV pt = cell_centers.segment<dim>(i * dim);
                std::vector<TV3> samples;
                sampleSphere(pt, 1.0 * ref_dis, n_samples_per_cell, samples);
                for (int j = 0; j < samples.size(); j++)
                {
                    std::vector<int> neighbors;
                    hash_table.getOneRingNeighbors(samples[j], neighbors);
                    T min_dis = 1e10; int min_idx = -1;
                    for (int idx : neighbors)
                    {
                        TV vj = cell_centers.segment<dim>(idx * dim);
                        T dis = (vj - samples[j]).norm();
                        if (dis < min_dis)
                        {
                            min_dis = dis;
                            min_idx = idx;
                        }
                    }
                    // std::cout << min_dis << " " << min_idx << std::endl;
                    // std::getchar();
                    voronoi_samples.row(i * n_samples_per_cell + j) = samples[j];
                    if (min_idx == -1)
                        voronoi_samples_colors.row(i * n_samples_per_cell + j) = cell_colors[i];
                    else
                        voronoi_samples_colors.row(i * n_samples_per_cell + j) = cell_colors[min_idx];
                }
            });
            if (show_multiview)
            {
                TM3 R90z; R90z << std::cos(0.5 * M_PI),  -std::sin(0.5 * M_PI), 0, std::sin(0.5 * M_PI), std::cos(0.5 * M_PI), 0, 0, 0, 1.0;
                TM3 R90y; R90y << std::cos(0.5 * M_PI),  0, std::sin(0.5 * M_PI), 0, 1, 0, -std::sin(0.5 * M_PI), 0, std::cos(0.5 * M_PI);
                TM3 nR90y; nR90y << std::cos(-0.5 * M_PI),  0, std::sin(-0.5 * M_PI), 0, 1, 0, -std::sin(-0.5 * M_PI), 0, std::cos(-0.5 * M_PI);
                MatrixXT V_lateral_left = voronoi_samples;
                MatrixXT V_dorsal = voronoi_samples;
                MatrixXT V_ventral = voronoi_samples;
                MatrixXT V_lateral_right = voronoi_samples;
                for (int i = 0; i < voronoi_samples.rows(); i++)
                {
                    V_lateral_left.row(i) = (R90z * voronoi_samples.row(i).transpose()).transpose();
                    V_dorsal.row(i) = (R90y * V_lateral_left.row(i).transpose()).transpose();
                    V_lateral_right.row(i) = (R90y * V_dorsal.row(i).transpose()).transpose();
                    V_ventral.row(i) = (nR90y * V_lateral_left.row(i).transpose()).transpose();
                }
                int n_vtx_current = voronoi_samples.rows();
                TV min_corner, max_corner;
                simulation.computeBoundingBox(min_corner, max_corner);
                MatrixXT offset_vertice = voronoi_samples;
                offset_vertice.setZero();
                offset_vertice.col(0).setConstant((max_corner-min_corner).norm() * 0.5);
                voronoi_samples.resize(n_vtx_current * 4, 3); 
                voronoi_samples.block(0, 0, n_vtx_current, 3) = V_ventral;
                voronoi_samples.block(n_vtx_current, 0, n_vtx_current, 3) = V_dorsal + 1.0 * offset_vertice;
                voronoi_samples.block(2 * n_vtx_current, 0, n_vtx_current, 3) = V_lateral_left + 2.0 * offset_vertice;
                voronoi_samples.block(3 * n_vtx_current, 0, n_vtx_current, 3) = V_lateral_right + offset_vertice * 3.0;
                
                Eigen::MatrixXd colors_updated(4 * n_vtx_current, 3);
                for (int i = 0; i < 4; i++)
                    colors_updated.block(i * n_vtx_current, 0, n_vtx_current, 3) = voronoi_samples_colors;
                voronoi_samples_colors = colors_updated;

            }
            viewer.data().set_points(voronoi_samples, voronoi_samples_colors);
        }
    }
    if (!show_voronoi)
    {
        viewer.data().set_mesh(V, F);
        viewer.data().set_colors(C);   
    }
}

template<int dim>
void SimulationApp<dim>::setMenu(igl::opengl::glfw::Viewer& viewer, 
    igl::opengl::glfw::imgui::ImGuiMenu& menu)
{
    menu.callback_draw_viewer_menu = [&]()
    {
        if (ImGui::CollapsingHeader("Visualization", ImGuiTreeNodeFlags_DefaultOpen))
        {
            if (ImGui::Checkbox("MultiView", &show_multiview))
            {
                updateScreen(viewer);
            }
            if (ImGui::Checkbox("ShowCellCenter", &show_cell_center))
            {
                updateScreen(viewer);
            }
            if (ImGui::Checkbox("CollisionSphere", &show_collision_sphere))
            {
                updateScreen(viewer);
            }
            if (ImGui::Checkbox("ControlPoints", &show_control_points))
            {
                updateScreen(viewer);
            }
            if (ImGui::Checkbox("ShowMembrane", &show_membrane))
            {
                updateScreen(viewer);
            }
            if (ImGui::Checkbox("ShowMatchingPairs", &show_matching_pairs))
            {
                updateScreen(viewer);
            }
            if (ImGui::Checkbox("ShowAdhesionPairs", &show_adh_pairs))
            {
                updateScreen(viewer);
            }
            if (ImGui::Checkbox("ShowYolkSurface", &show_yolk_surface))
            {
                updateScreen(viewer);
            }
            if (ImGui::Checkbox("ShowYolkTriangle", &show_yolk_triangle))
            {
                updateScreen(viewer);
            }
            if (ImGui::Checkbox("ShowVoronoi", &show_voronoi))
            {
                show_cell_center = !show_voronoi;
                unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
                std::mt19937 generator (seed);
                std::uniform_real_distribution<T> uniform01(0.0, 1.0);
                cell_colors.resize(simulation.num_cells);
                for (int i = 0; i < simulation.num_cells; i++)
                {
                    T r = std::min(uniform01(generator) + 0.1, 1.0);
                    T g = std::min(uniform01(generator) + 0.1, 1.0);
                    T b = std::min(uniform01(generator) + 0.1, 1.0);
                    TV3 random_color = TV3(r, g, b);
                    cell_colors[i] = random_color;
                }

                updateScreen(viewer);
            }
        }
        if (ImGui::Button("RunSequence", ImVec2(-1,0)))
        {
            for (int i = 0; i < simulation.n_frames; i++)
            {
                simulation.verbose = false;
                simulation.advanceOneFrame();
                updateScreen(viewer);
                simulation.global_frame++;    
            }
        }
        if (ImGui::Button("StaticSolve", ImVec2(-1,0)))
        {
            simulation.staticSolve();
            updateScreen(viewer);
        }
        if (ImGui::Button("LoadStates", ImVec2(-1,0)))
        {
            simulation.loadStates("results/"+std::to_string(simulation.global_frame)+".obj");
            updateScreen(viewer);
        }
        if (ImGui::Button("RenderTrajectory", ImVec2(-1,0)))
        {
            int width = viewer.core().viewport(2);
            int height = viewer.core().viewport(3);
            CMat R(width,height), G(width,height), B(width,height), A(width,height);
            show_multiview = true;
            show_yolk_surface = false;
            show_voronoi = true;
            show_cell_center = false;
            for (int i = 0; i < simulation.n_frames; i++)
            // for (int i = 0; i < 5; i++)
            {
                simulation.loadStates("results/"+std::to_string(i)+".obj");
                updateScreen(viewer);
                std::cout << "here" << std::endl;
                if (i == 0)
                {
                    // if (show_voronoi)
                    //     viewer.core().align_camera_center(voronoi_samples);
                    // else
                        viewer.core().align_camera_center(V);
                    viewer.core().camera_zoom *= 1.3;
                }
                viewer.core().draw_buffer(viewer.data(),true,R,G,B,A);
                A.setConstant(255);
                if (show_voronoi)
                    igl::png::writePNG(R,G,B,A, "image/voro_" + std::to_string(i)+".png");
                else
                    igl::png::writePNG(R,G,B,A, "image/" + std::to_string(i)+".png");
            }
        }
        if (ImGui::Button("Reset", ImVec2(-1,0)))
        {
            simulation.reset();
            updateScreen(viewer);
        }
        if (ImGui::Button("SaveIPCMesh", ImVec2(-1,0)))
        {
            simulation.saveIPCMesh();
        }
        if (ImGui::Button("SaveMesh", ImVec2(-1,0)))
        {
            igl::writeOBJ("current_mesh.obj", V, F);
        }
    };
}

template<int dim>
void SimulationApp<dim>::setViewer(igl::opengl::glfw::Viewer& viewer, igl::opengl::glfw::imgui::ImGuiMenu& menu)
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
        
        switch(key)
        {
        default: 
            return false;
        case ' ':
            static_solve_step = 0;
            viewer.core().is_animating = true;
            return true;
        case 's':
            simulation.advanceOneStep(static_solve_step++);
            updateScreen(viewer);
            return true;
        case 'n':
            simulation.global_frame++;
            simulation.updatePerFrameData();
            updateScreen(viewer);
            std::cout << "frame " << simulation.global_frame << std::endl;
            return true;
        case 'm':
            simulation.global_frame--;
            simulation.updatePerFrameData();
            updateScreen(viewer);
            std::cout << "frame " << simulation.global_frame << std::endl;
            return true;
        case 'd':
            simulation.checkTotalGradientScale();
            simulation.checkTotalHessianScale();
            return true;
        }
    };

    viewer.callback_mouse_down = [&](igl::opengl::glfw::Viewer& viewer, int, int)->bool
    {
        return false;
        
        double x = viewer.current_mouse_x;
        double y = viewer.core().viewport(3) - viewer.current_mouse_y;
        for (int i = 0; i < simulation.num_cells; i++)
        {
            Eigen::MatrixXd pxy(1, 3);
            Eigen::MatrixXd x3d(1, 3); x3d.setZero();
            x3d.row(0).template segment<dim>(0) = simulation.deformed.template segment<dim>(i * dim);
            igl::project(x3d, viewer.core().view, viewer.core().proj, viewer.core().viewport, pxy);
            
            if(abs(pxy.row(0)[0]-x)<20 && abs(pxy.row(0)[1]-y)<20)
            {
                selected = i;
                x0 = x;
                y0 = y;
                std::cout << "selected " << selected << std::endl;
                return true;
            }
        }
        return false;
    };

    viewer.callback_mouse_up = [&](igl::opengl::glfw::Viewer& viewer, int, int)->bool
	  {
		if(selected!=-1)
		{
            simulation.is_control_points(selected) = -1;
			selected = -1;
            simulation.undeformed = simulation.deformed;
			return true;
		}
	    return false;
	  };

    viewer.callback_mouse_move =
        [&](igl::opengl::glfw::Viewer& viewer, int, int)->bool
    {
        if(selected!=-1)
        {
            simulation.is_control_points(selected) = selected;
            double x = viewer.current_mouse_x;
            double y = viewer.core().viewport(3) - viewer.current_mouse_y;
        
            double delta_x = (x - x0) / viewer.core().viewport(2);
            double delta_y = (y - y0) / viewer.core().viewport(3);

            simulation.target_positions.template segment<2>(selected * dim) = 
                simulation.undeformed.template segment<2>(selected * dim) + TV2(delta_x, delta_y);

            // simulation.deformed.template segment<2>(selected * dim) = 
            //     simulation.undeformed.template segment<2>(selected * dim) + TV2(delta_x, delta_y);

            simulation.staticSolve();
            // simulation.undeformed = simulation.deformed;
            updateScreen(viewer);
            return true;
        }
        return false;
    };

    simulation.sampleBoundingSurface(bounding_surface_samples);
    sdf_test_sample_idx_offset = bounding_surface_samples.rows();
    bounding_surface_samples_color = bounding_surface_samples;
    for (int i = 0; i < bounding_surface_samples.rows(); i++)
        bounding_surface_samples_color.row(i) = TV3(0.1, 1.0, 0.1);

    updateScreen(viewer);
    
    viewer.core().background_color.setOnes();
    viewer.core().viewport(2) = 3000; viewer.core().viewport(3) = 2200;
    viewer.data().set_face_based(true);
    viewer.data().shininess = 1.0;
    viewer.data().point_size = 10.0;

    viewer.data().set_mesh(V, F);     
    viewer.data().set_colors(C);

    viewer.core().align_camera_center(V);
    viewer.core().camera_zoom *= 2.0;
}

template<int dim>
void SimulationApp<dim>::loadDisplacementVectors(const std::string& filename)
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

template<int dim>
void SimulationApp<dim>::loadSVDData(const std::string& filename)
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

template<int dim>
void SimulationApp<dim>::sampleSphere(const TV3& center, T radius, 
    int n_samples, std::vector<TV3>& samples)
{
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 generator (seed);
    std::uniform_real_distribution<T> uniform01(0.0, 1.0);
    samples.resize(n_samples);
    tbb::parallel_for(0, n_samples, [&](int i){
        T r = radius * uniform01(generator);
        T theta = 2 * M_PI * uniform01(generator);
        T phi = std::acos(1 - 2 * uniform01(generator));
        T x = r * std::sin(phi) * std::cos(theta);
        T y = r * std::sin(phi) * std::sin(theta);
        T z = r * std::cos(phi);
        samples[i] = center + TV3(x, y, z);
    });

}

template class SimulationApp<2>;
template class SimulationApp<3>;