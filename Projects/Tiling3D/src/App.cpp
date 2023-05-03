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

void App::appendMesh(const Eigen::MatrixXd& V1, const Eigen::MatrixXi& F1, const Eigen::MatrixXd& C1,
        Eigen::MatrixXd& V2, Eigen::MatrixXi& F2, Eigen::MatrixXd& C2)
{
    int n_vtx = V2.rows(), n_face = F2.rows();
    V2.conservativeResize(n_vtx + V1.rows(), 3);
    F2.conservativeResize(n_face + F1.rows(), 3); 
    C2.conservativeResize(n_face + F1.rows(), 3); 
    Eigen::MatrixXi offset(F1.rows(), 3);
    offset.setConstant(n_vtx);
    V2.block(n_vtx, 0, V1.rows(), 3) = V1;
    F2.block(n_face, 0, F1.rows(), 3) = F1 + offset;
    C2.block(n_face, 0, F1.rows(), 3) = C1;
}

void SimulationApp::updateScreen(igl::opengl::glfw::Viewer& viewer)
{
    tiling.solver.generateMeshForRendering(V, F, C);

    if (show_rest)
    {
        Eigen::MatrixXd V_rest, C_rest;
        Eigen::MatrixXi F_rest;
        tiling.solver.generateMeshForRendering(V_rest, F_rest, C_rest, true);
        C_rest.setConstant(0.0);
        C_rest.col(0).setConstant(1.0);
        appendMesh(V_rest, F_rest, C_rest, V, F, C);
    }

    if (tiling.solver.add_pbc && tile_unit)
    {
        int n_unit = 27;
        Eigen::MatrixXd V_tile(V.rows() * n_unit, 3);
        Eigen::MatrixXi F_tile(F.rows() * n_unit, 3);
        Eigen::MatrixXd C_tile(F.rows() * n_unit, 3);

        TV left0 = tiling.solver.deformed.segment<3>(tiling.solver.pbc_pairs[0][0][0] * 3);
        TV right0 = tiling.solver.deformed.segment<3>(tiling.solver.pbc_pairs[0][0][1] * 3);
        TV top0 = tiling.solver.deformed.segment<3>(tiling.solver.pbc_pairs[1][0][1] * 3);
        TV bottom0 = tiling.solver.deformed.segment<3>(tiling.solver.pbc_pairs[1][0][0] * 3);
        TV front0 = tiling.solver.deformed.segment<3>(tiling.solver.pbc_pairs[2][0][1] * 3);
        TV back0 = tiling.solver.deformed.segment<3>(tiling.solver.pbc_pairs[2][0][0] * 3);
        TV dx = (right0 - left0);
        TV dy = (top0 - bottom0);
        TV dz = front0 - back0;

        int n_unit_dir = std::pow(n_unit, 1.0/3.0);

        int n_face = F.rows(), n_vtx = V.rows();
        for (int i = 0; i < n_unit; i++)
            V_tile.block(i * n_vtx, 0, n_vtx, 3) = V;
        
        int start = (n_unit_dir - 1) / 2;
        int cnt = 0;
        for (int left = -start; left < start + 1; left++)
        {
            for (int bottom = -start; bottom < start + 1; bottom++)
            {
                for (int back = -start; back < start + 1; back++)
                {
                    tbb::parallel_for(0, n_vtx, [&](int i){
                        V_tile.row(cnt * n_vtx + i).head<3>() += T(left) * dx + T(bottom) * dy + T(back) * dz;
                    });
                    cnt++;
                }
            }
        }
        V = V_tile;
        Eigen::MatrixXi offset(n_face, 3);
        offset.setConstant(n_vtx);

        for (int i = 0; i < n_unit; i++)
            F_tile.block(i * n_face, 0, n_face, 3) = F + i * offset;
        
        F = F_tile;

        Eigen::MatrixXd C_unit = C;
        C_unit.col(2).setConstant(0.3); C_unit.col(1).setConstant(1.0);
        C_tile.block(0, 0, n_face, 3) = C;
        for (int i = 1; i < n_unit; i++)
            C_tile.block(i * n_face, 0, n_face, 3) = C;
        C = C_tile;
    }
        
    if (tiling.solver.add_pbc && connect_pbc)
    // std::cout << "tiling.solver.add_pbc " << tiling.solver.add_pbc << " connect_pbc " << connect_pbc << std::endl; 
    {
        std::vector<std::pair<TV, TV>> end_points;
        for (int dir = 0; dir < 3; dir++)
            for (const Edge& pbc_pair : tiling.solver.pbc_pairs[dir])
            {
                int idx0 = pbc_pair[0], idx1 = pbc_pair[1];
                TV xi = tiling.solver.deformed.segment<3>(idx0 * 3);
                TV xj = tiling.solver.deformed.segment<3>(idx1 * 3);
                // std::cout << idx0 << " " << idx1 << " " << xi.transpose() << " " << xj.transpose() << std::endl;
                end_points.push_back(std::make_pair(xi, xj));
            }
        T ref_dis = (end_points[0].first - end_points[1].second).norm();
        std::vector<TV> colors;
        for (int i = 0; i < end_points.size(); i++)
            colors.push_back(TV(1.0, 0.3, 0.0));
        appendCylindersToEdges(end_points, colors, 0.001 * ref_dis, V, F, C);
    }
    
    // if (show_bc)
    // {
    //     int nf_current = F.rows();
    //     tiling.solver.updateSphere();
    //     tiling.solver.appendMesh(V, F, tiling.solver.sphere_vertices, tiling.solver.sphere_faces);
    //     int nf_sphere = tiling.solver.sphere_faces.rows();
    //     MatrixXd sphere_color(nf_sphere, 3);
    //     sphere_color.setZero();
    //     sphere_color.col(0).setConstant(1.0);
    //     C.conservativeResize(F.rows(), 3);
    //     C.block(nf_current, 0, nf_sphere, 3) = sphere_color;
    // }

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
                // updateScreen(viewer);
            }
            if (ImGui::Checkbox("ShowRest", &show_rest))
            {
                updateScreen(viewer);
            }
            // if (ImGui::Checkbox("ShowBC", &show_bc))
            // {
            //     updateScreen(viewer);
            // } 
        }
        if (ImGui::CollapsingHeader("Simulation", ImGuiTreeNodeFlags_DefaultOpen))
        {
            if (ImGui::Checkbox("PBC", &tiling.solver.add_pbc))
            {
                updateScreen(viewer);
            }
            if (tiling.solver.add_pbc)
            {
                if (ImGui::Checkbox("ShowPairs", &connect_pbc))
                {
                    updateScreen(viewer);
                }
                if (ImGui::Checkbox("TileUnit", &tile_unit))
                {
                    updateScreen(viewer);
                }
            }
        }
        // if (ImGui::CollapsingHeader("Simulation", ImGuiTreeNodeFlags_DefaultOpen))
        // {
        //     if (ImGui::Checkbox("TetGen", &tetgen))
        //     {
                
        //     }
        //     if (ImGui::Checkbox("IncrementalLoading", &incremental))
        //     {
                
        //     }
        // }
        if (ImGui::Button("StaticSolve", ImVec2(-1,0)))
        {
            tiling.solver.staticSolve();
            updateScreen(viewer);
        }
        if (ImGui::Button("Reset", ImVec2(-1,0)))
        {
            tiling.solver.deformed = tiling.solver.undeformed;
            tiling.solver.u.setZero();
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
                // T bending_stiffness = tiling.solver.computeBendingStiffness();
                // std::cout << "bending stiffness " << bending_stiffness << std::endl;
                VectorXT interal_force(tiling.solver.deformed.rows());
                interal_force.setZero();
                tiling.solver.addBCPenaltyForceEntries(interal_force);
                std::cout << interal_force.norm() << std::endl;
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
            if (incremental)
                tiling.solver.incrementalLoading();
            else
                viewer.core().is_animating = true;
                // tiling.solver.staticSolve();    
            updateScreen(viewer);
            return true;
        case '1':
            check_modes = true;
            tiling.solver.computeLinearModes();
            loadDisplacementVectors("/home/yueli/Documents/ETH/WuKong/fem_eigen_vectors.txt");
            
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

    viewer.callback_mouse_down = [&](igl::opengl::glfw::Viewer& viewer, int, int)->bool
    {
        if (!enable_selection)
            return false;
        double x = viewer.current_mouse_x;
        double y = viewer.core().viewport(3) - viewer.current_mouse_y;

        for (int i = 0; i < tiling.solver.num_nodes; i++)
        {
            Vector<T, 3> pos = tiling.solver.deformed.segment<3>(i * 3);
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

    // tiling.initializeSimulationData(tetgen);
    // tiling.solver.runForceCurvatureExperiment();
    // tiling.solver.runForceDisplacementExperiment();
    // tiling.solver.loadForceDisplacementResults();
    // tiling.solver.checkTotalGradientScale(true);
    // tiling.solver.checkTotalHessianScale(true);
    // tiling.solver.runBendingHomogenization();
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
    // viewer.data().set_face_based(true);
    // viewer.data().shininess = 1.0;
    // viewer.data().point_size = 25.0;

    // viewer.core().align_camera_center(V);
    // viewer.core().animation_max_fps = 24.;
    // // key_down(viewer,'0',0);
    // viewer.core().is_animating = false;
}