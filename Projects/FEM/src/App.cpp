#include "../include/App.h"

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

void App::appendCylindersToEdges(const std::vector<std::pair<TV, TV>>& edge_pairs, 
        const std::vector<TV>& color, T radius,
        Eigen::MatrixXd& _V, Eigen::MatrixXi& _F, Eigen::MatrixXd& _C)
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
    _C.conservativeResize(n_row_F + offset_f * n_edge, 3);

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

            _C.row(n_row_F + ei * offset_f + i*2 ) = color[ei];
            _C.row(n_row_F + ei * offset_f + i*2 + 1) = color[ei];
        }
    });
}

void SimulationApp::updateScreen(igl::opengl::glfw::Viewer& viewer)
{
    solver.generateMeshForRendering(V, F, C);
        
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
            if (ImGui::Checkbox("SelectVertex", &enable_selection))
            {
                
            }
        }
        if (ImGui::Button("StaticSolve", ImVec2(-1,0)))
        {
            solver.staticSolve();
            updateScreen(viewer);
        }
        if (ImGui::Button("Reset", ImVec2(-1,0)))
        {
            solver.deformed = solver.undeformed;
            solver.u.setZero();
            updateScreen(viewer);
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
        case 'a':
            viewer.core().is_animating = !viewer.core().is_animating;
            return true;
        case 's':
            solver.staticSolveStep(static_solve_step++);
            return true;
        case 'd':
            solver.checkTotalGradientScale(true);
            solver.checkTotalHessianScale(true);
        case ' ':
            viewer.core().is_animating = true;
            updateScreen(viewer);
            return true;
        case '1':
            check_modes = true;
            solver.computeLinearModes();
            loadDisplacementVectors("./fem_eigen_vectors.txt");
            
            for (int i = 0; i < evalues.rows(); i++)
            {
                if (evalues[i] > 1e-6) // skip rigid modes
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

    viewer.callback_mouse_down = [&](igl::opengl::glfw::Viewer& viewer, int, int)->bool
    {
        if (!enable_selection)
            return false;
        double x = viewer.current_mouse_x;
        double y = viewer.core().viewport(3) - viewer.current_mouse_y;

        for (int i = 0; i < solver.num_nodes; i++)
        {
            Vector<T, 3> pos = solver.deformed.segment<3>(i * 3);
            Eigen::MatrixXd x3d(1, 3); x3d.setZero();
            x3d.row(0).segment<3>(0) = pos;

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