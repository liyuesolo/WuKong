#include <igl/boundary_loop.h>
#include <igl/edge_lengths.h>
#include <igl/project.h>
#include <igl/unproject_onto_mesh.h>
#include <igl/unproject_on_plane.h>
#include "../include/App.h"

void SimulationApp::appendCylindersToEdges(const std::vector<std::pair<TV3, TV3>>& edge_pairs, 
        const std::vector<TV3>& color, T radius,
        Eigen::MatrixXd& _V, Eigen::MatrixXi& _F, Eigen::MatrixXd& _C)
{
    if (!edge_pairs.size())
        return;
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

void SimulationApp::appendSpheresToPositions(const VectorXT& position, T radius, const TV& color,
        Eigen::MatrixXd& _V, Eigen::MatrixXi& _F, Eigen::MatrixXd& _C)
{
    if (!position.rows())
        return;
    int n_pt = position.rows() / 3;

    Eigen::MatrixXd v_sphere;
    Eigen::MatrixXi f_sphere;

    igl::readOBJ("/home/yueli/Documents/ETH/WuKong/Projects/DigitalFabrics/Data/sphere162.obj", v_sphere, f_sphere);
    
    Eigen::MatrixXd c_sphere(f_sphere.rows(), f_sphere.cols());
    
    v_sphere = v_sphere * radius;

    int n_vtx_prev = _V.rows();
    int n_face_prev = _F.rows();

    _V.conservativeResize(_V.rows() + v_sphere.rows() * n_pt, 3);
    _F.conservativeResize(_F.rows() + f_sphere.rows() * n_pt, 3);
    _C.conservativeResize(_C.rows() + f_sphere.rows() * n_pt, 3);

    tbb::parallel_for(0, n_pt, [&](int i)
    {
        Eigen::MatrixXd v_sphere_i = v_sphere;
        Eigen::MatrixXi f_sphere_i = f_sphere;
        Eigen::MatrixXd c_sphere_i = c_sphere;

        tbb::parallel_for(0, (int)v_sphere.rows(), [&](int row_idx){
            v_sphere_i.row(row_idx) += position.segment<3>(i * 3);
        });


        int offset_v = n_vtx_prev + i * v_sphere.rows();
        int offset_f = n_face_prev + i * f_sphere.rows();

        tbb::parallel_for(0, (int)f_sphere.rows(), [&](int row_idx){
            f_sphere_i.row(row_idx) += Eigen::Vector3i(offset_v, offset_v, offset_v);
        });

        tbb::parallel_for(0, (int)f_sphere.rows(), [&](int row_idx){
            c_sphere_i.row(row_idx) = color;
        });

        _V.block(offset_v, 0, v_sphere.rows(), 3) = v_sphere_i;
        _F.block(offset_f, 0, f_sphere.rows(), 3) = f_sphere_i;
        _C.block(offset_f, 0, f_sphere.rows(), 3) = c_sphere_i;
    });
}


void SimulationApp::appendSpheresToPositions(const VectorXT& position, T radius, const std::vector<TV>& colors,
        Eigen::MatrixXd& _V, Eigen::MatrixXi& _F, Eigen::MatrixXd& _C)
{
    if (!position.rows())
        return;
    int n_pt = position.rows() / 3;

    Eigen::MatrixXd v_sphere;
    Eigen::MatrixXi f_sphere;

    igl::readOBJ("/home/yueli/Documents/ETH/WuKong/Projects/DigitalFabrics/Data/sphere162.obj", v_sphere, f_sphere);
    
    Eigen::MatrixXd c_sphere(f_sphere.rows(), f_sphere.cols());
    
    v_sphere = v_sphere * radius;

    int n_vtx_prev = _V.rows();
    int n_face_prev = _F.rows();

    _V.conservativeResize(_V.rows() + v_sphere.rows() * n_pt, 3);
    _F.conservativeResize(_F.rows() + f_sphere.rows() * n_pt, 3);
    _C.conservativeResize(_C.rows() + f_sphere.rows() * n_pt, 3);

    tbb::parallel_for(0, n_pt, [&](int i)
    {
        Eigen::MatrixXd v_sphere_i = v_sphere;
        Eigen::MatrixXi f_sphere_i = f_sphere;
        Eigen::MatrixXd c_sphere_i = c_sphere;

        tbb::parallel_for(0, (int)v_sphere.rows(), [&](int row_idx){
            v_sphere_i.row(row_idx) += position.segment<3>(i * 3);
        });


        int offset_v = n_vtx_prev + i * v_sphere.rows();
        int offset_f = n_face_prev + i * f_sphere.rows();

        tbb::parallel_for(0, (int)f_sphere.rows(), [&](int row_idx){
            f_sphere_i.row(row_idx) += Eigen::Vector3i(offset_v, offset_v, offset_v);
        });

        tbb::parallel_for(0, (int)f_sphere.rows(), [&](int row_idx){
            c_sphere_i.row(row_idx) = colors[i];
        });

        _V.block(offset_v, 0, v_sphere.rows(), 3) = v_sphere_i;
        _F.block(offset_f, 0, f_sphere.rows(), 3) = f_sphere_i;
        _C.block(offset_f, 0, f_sphere.rows(), 3) = c_sphere_i;
    });
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

void VoronoiApp::updateScreen(igl::opengl::glfw::Viewer& viewer)
{
    viewer.data().clear();
    voronoi_cells.generateMeshForRendering(V, F, C);
    int n_face_base = F.rows();

    if (V.rows())
    {
        // MatrixXi mesh_edges;
        // igl::edges(F, mesh_edges);
        // viewer.data().lines.resize(mesh_edges.rows(), 9);
        // for (int i = 0; i < mesh_edges.rows(); i++)
        // {
        //     viewer.data().lines.row(i).segment<3>(0) = V.row(mesh_edges(i, 0));
        //     viewer.data().lines.row(i).segment<3>(3) = V.row(mesh_edges(i, 1));
        //     viewer.data().lines.row(i).segment<3>(6) = TV::Zero();
        // }
        // viewer.data().line_width = 3.5;
    }

    T reference_length = 0.1;
    if (V.rows())
    {
        TV min_corner = V.colwise().minCoeff();
        TV max_corner = V.colwise().maxCoeff();
        T bb_diag = (max_corner - min_corner).norm();
        reference_length = 0.1 * bb_diag;
    }
    

    TV point_color(1.0, 0.0, 0.0);
    std::vector<TV> point_colors;
    if (voronoi_cells.n_sites)
    {
        point_colors.resize(voronoi_cells.n_sites, point_color);
        // point_colors[0] = TV(0,1,0);
        // point_colors[1] = TV(0,1,0);
        // point_colors[2] = TV(0,1,0);
        appendSpheresToPositions(voronoi_cells.voronoi_sites, 0.01 * reference_length, point_colors, V, F, C);
    }
    // appendSpheresToPositions(voronoi_cells.voronoi_sites, 0.025 * reference_length, point_color, V, F, C);

    // TV site_vtx_color(0.0, 0.0, 0.0);
    // appendSpheresToPositions(voronoi_cells.voronoi_cell_vertices, 0.05 * reference_length, site_vtx_color, V, F, C);
    
    if (compute_dual)
    {
        std::vector<std::pair<TV, TV>> idt_edges;
        std::vector<IV> idt_indices;
        voronoi_cells.computeDualIDT(idt_edges, idt_indices);
        std::vector<TV3> colors(idt_edges.size(), TV3(1.0,0.3,0.0));
        appendCylindersToEdges(idt_edges, colors, 0.008 * reference_length, V, F, C);
    }
    else
    {
        std::vector<TV3> colors(voronoi_cells.voronoi_edges.size(), TV3(1.0,0.3,0.0));
        appendCylindersToEdges(voronoi_cells.voronoi_edges, colors, 0.008 * reference_length, V, F, C);
    }

    
    viewer.data().set_mesh(V, F);
    viewer.data().set_colors(C);
    if (C.rows())
    {
        MatrixXT white = C; white.setOnes();
        viewer.data().F_material_ambient.block(0, 0, n_face_base, 3).array() += 0.2;
        // viewer.data().F_material_diffuse.block(0, 0, n_face_base, 3).array() += 0.2;
        // viewer.data().F_material_specular.block(0, 0, n_face_base, 3).array() += 0.2;
    }
}

void VoronoiApp::setViewer(igl::opengl::glfw::Viewer& viewer,
    igl::opengl::glfw::imgui::ImGuiMenu& menu)
{
    menu.callback_draw_viewer_menu = [&]()
    {
        if (ImGui::CollapsingHeader("Settings", ImGuiTreeNodeFlags_DefaultOpen))
        {
            if (ImGui::Checkbox("geodesic", &geodesic))
            {
                update_geodesic = true;
            }
            if (ImGui::Checkbox("exact", &exact))
            {
                update_exact = true;
            }
            if (ImGui::Checkbox("Perimeter", &perimeter))
            {
                if (perimeter)
                    voronoi_cells.objective = Perimeter;
                voronoi_cells.add_peri = perimeter;
                update_perimeter = true;
            }
            if (ImGui::Checkbox("Centroidal", &CGVD))
            {
                if (CGVD)
                    voronoi_cells.objective = Centroidal;
                voronoi_cells.add_centroid = CGVD;
                update_CGVD = true;
            }
            if (ImGui::Checkbox("Regularizer", &reg))
            {
                voronoi_cells.add_reg = reg;
                voronoi_cells.w_reg = 1e-6;
            }
            
        }
        if (ImGui::CollapsingHeader("Save Data", ImGuiTreeNodeFlags_DefaultOpen))
        {
            if (ImGui::Checkbox("save_IDT", &save_idt))
            {

            }
            if (ImGui::Checkbox("save_sites", &save_sites))
            {
                
            }
        }
        if (ImGui::CollapsingHeader("Visualization", ImGuiTreeNodeFlags_DefaultOpen))
        {
            if (ImGui::Checkbox("compute_dual", &compute_dual))
            {
                updateScreen(viewer);
            }
        }
        
        if (ImGui::Button("Generate", ImVec2(-1,0)))
        {
            if (geodesic)
                voronoi_cells.metric = Geodesic;
            else
                voronoi_cells.metric = Euclidean;
            voronoi_cells.samples.clear();
            voronoi_cells.source_data.clear();
            voronoi_cells.voronoi_edges.clear();
            voronoi_cells.voronoi_cells.clear();
            voronoi_cells.loadGeometry();
            voronoi_cells.constructVoronoiDiagram(exact, false);
            update_CGVD = false;
            update_exact = false;
            update_geodesic = false;
            updateScreen(viewer);
        }
        if (ImGui::Button("Compute", ImVec2(-1,0)))
        {
            if (update_geodesic)
                voronoi_cells.constructVoronoiDiagram(exact, false);
            if (update_exact)
                voronoi_cells.optimizeForExactVD();
            if (update_CGVD)
                voronoi_cells.optimizeForCentroidalVD();
            if (update_perimeter)
                voronoi_cells.perimeterMinimizationVD();
            update_CGVD = false;
            update_exact = false;
            update_geodesic = false;
            update_perimeter = false;
            updateScreen(viewer);
        }
        if (ImGui::Button("Resample", ImVec2(-1,0)))
        {
            voronoi_cells.resample(1.0);
            voronoi_cells.constructVoronoiDiagram(exact, false);
            updateScreen(viewer);
        }
        if (ImGui::Button("Save", ImVec2(-1,0)))
        {
            if (save_sites)
                voronoi_cells.saveVoronoiDiagram();
            if (save_idt)
            {
                std::vector<std::pair<TV, TV>> idt_edges;
                std::vector<IV> idt_indices;
                voronoi_cells.computeDualIDT(idt_edges, idt_indices);
                std::ofstream out("idt_triangulation.txt");
                out << idt_indices.size() << std::endl;
                for (const IV& tri : idt_indices)
                    out << tri[0] << " " << tri[1] << " " << tri[2] << " " << std::endl;
                out.close();
            }
            updateScreen(viewer);
        }
        if (ImGui::Button("Reset", ImVec2(-1,0)))
        {
            voronoi_cells.reset();
            voronoi_cells.constructVoronoiDiagram(exact, false);
            updateScreen(viewer);
        }
        if (ImGui::Button("Render", ImVec2(-1,0)))
        {
            int w = viewer.core().viewport(2), h = viewer.core().viewport(3);
            CMat R(w,h), G(w,h), B(w,h), A(w,h);
            viewer.core().draw_buffer(viewer.data(),true,R,G,B,A);
            A.setConstant(255);
            igl::png::writePNG(R,G,B,A, "./current_window.png");
        }
        
        if (ImGui::Button("SaveMesh", ImVec2(-1,0)))
        {
            igl::writeOBJ("./current_mesh.obj", V, F);
        }
        
    }; 

    
    viewer.callback_mouse_down = [&](igl::opengl::glfw::Viewer& viewer, int, int)->bool
    {
        return false;
        int fid;
        Eigen::Vector3f bc;
        // Cast a ray in the view direction starting from the mouse position
        double x = viewer.current_mouse_x;
        double y = viewer.core().viewport(3) - viewer.current_mouse_y;
        if(igl::unproject_onto_mesh(Eigen::Vector2f(x,y), viewer.core().view,
            viewer.core().proj, viewer.core().viewport, V, F, fid, bc))
        {
            // paint hit red
            voronoi_cells.saveFacePrism(fid);
            std::cout << "hit face " << fid << std::endl;
            return true;
        }
        return false;
    };
    
    viewer.callback_post_draw = [&](igl::opengl::glfw::Viewer &) -> bool
    {
        if(viewer.core().is_animating && !check_modes)
        {
            bool finished = voronoi_cells.advanceOneStep(static_solve_step);
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
            voronoi_cells.advanceOneStep(static_solve_step++);
            updateScreen(viewer);
            return true;
        case ' ':
            viewer.core().is_animating = !viewer.core().is_animating;
            return true;
        case 'd':
            voronoi_cells.diffTestScale();
            return true;
        }
        return true;
    };
    
    updateScreen(viewer);
    viewer.core().background_color.setOnes();
    viewer.data().set_face_based(true);
    viewer.data().shininess = 1.0;
    viewer.data().point_size = 25.0;

    viewer.core().align_camera_center(V);
    viewer.core().toggle(viewer.data().show_lines);
    viewer.core().animation_max_fps = 24.;
    viewer.core().is_animating = false;
}

void GeodesicSimApp::updateScreen(igl::opengl::glfw::Viewer& viewer)
{
    viewer.data().clear();
    simulation.generateMeshForRendering(V, F, C);
    int n_face_base = F.rows();
    // std::cout << viewer.data().lines.rows() << " " << viewer.data().lines.cols() << std::endl;
    MatrixXi mesh_edges;
    igl::edges(F, mesh_edges);
    viewer.data().lines.resize(mesh_edges.rows(), 9);
    for (int i = 0; i < mesh_edges.rows(); i++)
    {
        viewer.data().lines.row(i).segment<3>(0) = V.row(mesh_edges(i, 0));
        viewer.data().lines.row(i).segment<3>(3) = V.row(mesh_edges(i, 1));
        viewer.data().lines.row(i).segment<3>(6) = TV::Zero();
    }
    // viewer.data().show_lines = true;
    viewer.data().line_width = 3.5;
    // std::cout << viewer.data().lines.rows() << " " << viewer.data().lines.cols() << std::endl;

    simulation.updateVisualization(all_edges);

    std::vector<TV3> colors(simulation.all_intrinsic_edges.size(), TV3(1.0, 0.3, 0.0));
    appendCylindersToEdges(simulation.all_intrinsic_edges, colors, 0.03 * simulation.ref_dis, V, F, C);
    
    VectorXT sites;
    simulation.getAllPointsPosition(sites);
    appendSpheresToPositions(sites, simulation.ref_dis * 0.05, TV::Ones(), V, F, C);
    
    VectorXT markers;
    simulation.getMarkerPointsPosition(markers);
    appendSpheresToPositions(markers, simulation.ref_dis * 0.07, TV(1.0, 0.0, 0.0), V, F, C);

    viewer.data().set_mesh(V, F);
    viewer.data().set_colors(C);
    if (C.rows())
    {
        MatrixXT white = C; white.setOnes();
        viewer.data().F_material_ambient.block(0, 0, n_face_base, 3).array() += 0.2;
        // viewer.data().F_material_diffuse.block(0, 0, n_face_base, 3).array() += 0.2;
        // viewer.data().F_material_specular.block(0, 0, n_face_base, 3).array() += 0.2;
    }
    
}

void GeodesicSimApp::saveMesh(const std::string& folder)
{
    if (save_sites)
    {
        VectorXT sites;
        simulation.getAllPointsPosition(sites);
        MatrixXT vertices, colors;
        MatrixXi faces;
        appendSpheresToPositions(sites, simulation.ref_dis * 0.05, TV::Ones(), vertices, faces, colors);
        igl::writeOBJ(folder + "/sites.obj", vertices, faces);
    }
    if (save_mesh)
    {
        
        MatrixXT vertices, colors;
        MatrixXi faces;
        simulation.generateMeshForRendering(vertices, faces, colors);
        igl::writeOBJ(folder + "/surface_mesh.obj", vertices, faces);
    }
    if (save_curve)
    {
        MatrixXT vertices, _colors;
        MatrixXi faces;
        std::vector<TV> colors(simulation.all_intrinsic_edges.size(), TV3(0.95,103.0/255.0,0.0/255.0));
        appendCylindersToEdges(simulation.all_intrinsic_edges, colors, 0.03 * simulation.ref_dis, vertices, faces, _colors);
        
        igl::writeOBJ(folder + "/path.obj", vertices, faces);
    }
    if (save_all)
        igl::writeOBJ(folder + "/current_mesh.obj", V, F);
}

void GeodesicSimApp::setViewer(igl::opengl::glfw::Viewer& viewer,
    igl::opengl::glfw::imgui::ImGuiMenu& menu)
{
    menu.callback_draw_viewer_menu = [&]()
    {
        if (ImGui::CollapsingHeader("Visualization", ImGuiTreeNodeFlags_DefaultOpen))
        {
            if (ImGui::Checkbox("AllEdges", &all_edges))
            {
                updateScreen(viewer);
            }
        }
        if (ImGui::CollapsingHeader("Simulaiton", ImGuiTreeNodeFlags_DefaultOpen))
        {
            if (ImGui::Checkbox("TwoWayCoupling", &simulation.two_way_coupling))
            {
                
            }
            if (ImGui::Checkbox("PseudoCTensor", &simulation.add_geo_elasticity))
            {
                
            }
            if (ImGui::Checkbox("PseudoArea", &simulation.add_area_term))
            {
                
            }
            if (ImGui::Checkbox("Eclidean", &simulation.Euclidean))
            {
                
            }
            if (ImGui::Checkbox("mollifier", &simulation.use_t_wrapper))
            {
                
            }
        }

        if (ImGui::CollapsingHeader("SaveStates", ImGuiTreeNodeFlags_DefaultOpen))
        {
            if (ImGui::Checkbox("CurveNetwork", &save_curve))
            {
                
            }
            if (ImGui::Checkbox("EmbeddedMesh", &save_mesh))
            {
                
            }
            if (ImGui::Checkbox("Sites", &save_sites))
            {
                
            }
            if (ImGui::Checkbox("All", &save_all))
            {
                
            }
        }
        
        // if (ImGui::Button("StaticSolve", ImVec2(-1,0)))
        // {
            
        //     updateScreen(viewer);
        // }
        if (ImGui::Button("LoadState", ImVec2(-1,0)))
        {
            step_along_search_direction = true;
            static_solve_step = 0;
            std::ifstream in("search_direction.txt");
            int length; in >> length;
            search_direction.resize(length);
            for (int i = 0; i < length; i++)
                in >> search_direction[i];
            
            int n_springs; in >> n_springs;
            std::vector<Vector<int, 2>> edges(n_springs, Vector<int, 2>());
            for (int i = 0; i < n_springs; i++)
            {
                in >> edges[i][0] >> edges[i][1];
            }
            std::cout << "load edges" <<std::endl;
            int n_pts; in >> n_pts;
            simulation.mass_surface_points.resize(n_pts);
            for (int i = 0; i < n_pts; i++)
            {
                TV bary; in >> bary[0] >> bary[1] >> bary[2];
                int face_idx; in >> face_idx;
                gcs::Face f = simulation.mesh->face(face_idx);
                simulation.mass_surface_points[i].second = f;
                simulation.mass_surface_points[i].first = 
                    gcs::SurfacePoint(f, gc::Vector3{bary[0],bary[1], bary[2]});
            }
            std::cout << "load points" <<std::endl;
            if (simulation.two_way_coupling)
            {
                simulation.undeformed.conservativeResize(length);
                simulation.shell_dof_start = n_pts * 2;
                simulation.undeformed.segment(simulation.shell_dof_start, 
                    length - simulation.shell_dof_start)
                = simulation.extrinsic_vertices;
                simulation.deformed = simulation.undeformed;
                simulation.delta_u = simulation.undeformed; 
                simulation.delta_u.setZero();
                int n_dof; in >> n_dof;
                for (int i = 0; i < n_dof; i++)
                    in >> simulation.deformed[i];
            }
            std::cout << "load mesh" <<std::endl;
            
            in.close();
            // simulation.initializeNetworkData(edges);
            
            simulation.updateCurrentState();
            simulation.mass_surface_points_undeformed = simulation.mass_surface_points;
            temporary_vector = simulation.deformed;
            updateScreen(viewer);
        }
        if (ImGui::Button("StaticSolve", ImVec2(-1,0)))
        {
            // tiling.solver.staticSolve();
            updateScreen(viewer);
        }
        
        if (ImGui::Button("Reset", ImVec2(-1,0)))
        {
            simulation.reset();
            static_solve_step = 0;
            updateScreen(viewer);
        }
        
        
        if (ImGui::Button("Render", ImVec2(-1,0)))
        {
            int w = viewer.core().viewport(2), h = viewer.core().viewport(3);
            CMat R(w,h), G(w,h), B(w,h), A(w,h);
            viewer.core().draw_buffer(viewer.data(),true,R,G,B,A);
            A.setConstant(255);
            igl::png::writePNG(R,G,B,A, "./current_window.png");
        }
        if (ImGui::Button("SaveMesh", ImVec2(-1,0)))
        {
            // igl::writeOBJ("current_mesh.obj", V, F);
            saveMesh("./");
        }
        
    };    

    viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer &) -> bool
    {
        if(viewer.core().is_animating && check_modes)
        {
            simulation.mass_surface_points = simulation.mass_surface_points_temp;
            simulation.deformed = simulation.deformed_temp;
            simulation.delta_u = evectors.col(modes) * std::sin(t);
            simulation.updateCurrentState();
            t += 0.1;
            updateScreen(viewer);
        }
        return false;
    };

    viewer.callback_mouse_down = [&](igl::opengl::glfw::Viewer& viewer, int, int)->bool
    {
        // return false;
        double x = viewer.current_mouse_x;
        double y = viewer.core().viewport(3) - viewer.current_mouse_y;
        for (int i = 0; i < simulation.mass_surface_points.size(); i++)
        {
            Eigen::MatrixXd pxy(1, 3);
            TV x3d;
            simulation.massPointPosition(i, x3d);
            // std::cout << x3d.transpose() << std::endl;
            igl::project(x3d.transpose(), viewer.core().view, viewer.core().proj, viewer.core().viewport, pxy);
            // std::cout << pxy << std::endl;
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
			selected = -1;
			return true;
		}
	    return false;
	  };

    viewer.callback_mouse_move =
        [&](igl::opengl::glfw::Viewer& viewer, int, int)->bool
    {
        if(selected!=-1)
        {
            double x = viewer.current_mouse_x;
            double y = viewer.core().viewport(3) - viewer.current_mouse_y;
        
            double delta_x = (x - x0) / viewer.core().viewport(2);
            double delta_y = (y - y0) / viewer.core().viewport(3);

            return true;
        }
        return false;
    };

    viewer.callback_post_draw = [&](igl::opengl::glfw::Viewer &) -> bool
    {
        if(viewer.core().is_animating && !check_modes)
        {
            // bool finished = true;
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
        bool succeed = false;
        switch(key)
        {
        default: 
            return false;
        case 's':
            succeed = simulation.advanceOneStep(static_solve_step++);
            updateScreen(viewer);
            return true;
        case 'm':
            // simulation.moveMassPoint(selected, 0);
            // simulation.updateVisualization(all_edges);
            if (step_along_search_direction)
            {
                simulation.mass_surface_points = simulation.mass_surface_points_undeformed;
                simulation.delta_u = (static_solve_step) * search_direction;
                simulation.deformed = temporary_vector;
                simulation.updateCurrentState();
                T total_energy = simulation.computeTotalEnergy();
                std::cout << "energy : " << total_energy << std::endl;
                static_solve_step++;
            }
            updateScreen(viewer);
            return true;
        case 'n':
            if (step_along_search_direction)
            {
                simulation.mass_surface_points = simulation.mass_surface_points_undeformed;
                simulation.delta_u = (static_solve_step-1) * search_direction;
                simulation.deformed = temporary_vector;
                simulation.updateCurrentState();
                T total_energy = simulation.computeTotalEnergy();
                std::cout << "energy : " << total_energy << std::endl;
                static_solve_step--;
            }
            updateScreen(viewer);
            return true;
        case 'd':
            simulation.checkTotalGradientScale(true);
            simulation.checkTotalHessianScale(true);
            // simulation.checkTotalGradient(true);
            // simulation.checkTotalHessian(true);
            // simulation.checkLengthDerivativesScale();
            updateScreen(viewer);
            return true;
        case 'c':
            // simulation.checkHessian();
            simulation.checkInformation();
            return true;
        case 'g':
            if (use_temp_vec)
            {
                static_solve_step++;
                simulation.mass_surface_points = simulation.mass_surface_points_undeformed;
                simulation.delta_u = (static_solve_step*0.1) * temporary_vector;
                simulation.updateCurrentState();
                T total_energy = simulation.computeTotalEnergy();
                std::cout << "energy : " << total_energy << std::endl;
            }
            else
            {
                // use_temp_vec = true;
                simulation.run_diff_test = true;
                static_solve_step++;
                temporary_vector.resize(simulation.undeformed.rows());
                temporary_vector <<  -0.0044444,  0.00888878, -0.00444453,  0.00888904;
                // temporary_vector.setZero();
                // simulation.computeResidual(temporary_vector); temporary_vector *= -1.0;
                // T a = temporary_vector[2],
                //     b = temporary_vector[3];
                // temporary_vector[2] = -b;
                // temporary_vector[3] = a;
                // a = temporary_vector[0],
                // b = temporary_vector[1];
                // temporary_vector[0] = b;
                // temporary_vector[1] = -a;
                // std::cout << temporary_vector.transpose() << std::endl;
                simulation.mass_surface_points = simulation.mass_surface_points_undeformed;
                simulation.delta_u = (static_solve_step) * temporary_vector;
                simulation.updateCurrentState();
                // T total_energy = simulation.computeTotalEnergy();
                T total_energy = simulation.current_length[0];
                std::cout << "length : " << total_energy << " " << static_solve_step << std::endl;
            }
            updateScreen(viewer);
            return true;
        case 'h':
            if (use_temp_vec)
            {
                static_solve_step--;
                simulation.mass_surface_points = simulation.mass_surface_points_undeformed;
                simulation.delta_u = (static_solve_step*0.1) * temporary_vector;
                simulation.updateCurrentState();
                T total_energy = simulation.computeTotalEnergy();
                std::cout << "energy : " << total_energy << std::endl;
            }
            updateScreen(viewer);
            return true;
        case 'r':
            simulation.reset();
            updateScreen(viewer);
            return true;
        case ' ':
            viewer.core().is_animating = !viewer.core().is_animating;
            return true;
        case '1':
            check_modes = true;
            simulation.mass_surface_points_temp = simulation.mass_surface_points;
            simulation.deformed_temp = simulation.deformed;
            simulation.checkHessianPD(true);
            loadDisplacementVectors("eigen_vectors.txt");
            std::cout << "modes " << modes << " singular value: " << evalues(modes) << std::endl;            
            return true;
        case '2':
            modes++;
            modes = (modes + evectors.cols()) % evectors.cols();
            std::cout << "modes " << modes << " singular value: " << evalues(modes) << std::endl;
            return true;
        case 'a':
            viewer.core().is_animating = !viewer.core().is_animating;
            return true;
        case '+':
            simulation.expandBaseMesh(0.05);
            updateScreen(viewer);
            return true;
        }
    };
    
    updateScreen(viewer);
    viewer.core().background_color.setOnes();
    viewer.data().set_face_based(true);
    viewer.data().shininess = 1.0;
    viewer.data().point_size = 25.0;
    

    viewer.core().align_camera_center(V);
    viewer.data().show_lines = false;
    // viewer.core().toggle(viewer.data().show_lines);
    viewer.core().animation_max_fps = 24.;
    viewer.core().is_animating = false;
}