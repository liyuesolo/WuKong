#include <igl/colormap.h>
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
            
            updateScreen(viewer);
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
                simulation.checkHessianPD(false);
                
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
            loadDisplacementVectors("/home/yueli/Documents/ETH/WuKong/cell_d2odx_singular_vectors.txt");
            // loadDisplacementVectors("/home/yueli/Documents/ETH/WuKong/dxdp.txt");
            modes = 0;
            std::cout << "modes " << modes << std::endl;
            return true;
        case '1':
            check_modes = true;
            // simulation.computeLinearModes();
            simulation.checkHessianPD(true);
            loadDisplacementVectors("/home/yueli/Documents/ETH/WuKong/cell_eigen_vectors.txt");
            std::cout << "modes " << modes << " singular value: " << evalues(modes) << std::endl;
            // for (int i = 0; i < evalues.rows(); i++)
            // {
            //     if (evalues[i] > 1e-6)
            //     {
            //         modes = i;
            //         return true;
            //     }
            // }
            return true;
        case '2':
            modes++;
            modes = (modes + evectors.cols()) % evectors.cols();
            std::cout << "modes " << modes << " singular value: " << evalues(modes) << std::endl;
            return true;
        case '3': //check modes at equilirium after static solve
            std::cout << "state: " << load_obj_iter_cnt << std::endl;
            simulation.loadDeformedState("output/cells/cell/cell_mesh_iter_" + std::to_string(load_obj_iter_cnt) + ".obj");
            std::cout << simulation.computeResidual(simulation.u, residual) << std::endl;
            updateScreen(viewer);
            return true;
        case '4':
            check_modes = true;
            // loadDisplacementVectors("/home/yueli/Documents/ETH/WuKong/dxdp_eigen_vectors.txt");
            loadDisplacementVectors("/home/yueli/Documents/ETH/WuKong/cell_edge_weights_svd_vectors.txt");
            modes = 0;
            std::cout << "modes " << modes << " singular value: " << evalues(modes) << std::endl;
            return true;
        case 'a':
            viewer.core().is_animating = !viewer.core().is_animating;
            return true;
        case 'n':
            load_obj_iter_cnt++;
            std::cout << "state: " << load_obj_iter_cnt << std::endl;
            // simulation.loadDeformedState("output/cells/cell/cell_mesh_iter_" + std::to_string(load_obj_iter_cnt) + ".obj");
            simulation.loadDeformedState("output/cells/debug_debug/" + std::to_string(load_obj_iter_cnt) + ".obj");
            updateScreen(viewer);
            return true;
        case 'l':
            load_obj_iter_cnt--;
            load_obj_iter_cnt = std::max(0, load_obj_iter_cnt);
            std::cout << "state: " << load_obj_iter_cnt << std::endl;
            // simulation.loadDeformedState("output/cells/cell/cell_mesh_iter_" + std::to_string(load_obj_iter_cnt) + ".obj");
            simulation.loadDeformedState("output/cells/debug_debug/" + std::to_string(load_obj_iter_cnt) + ".obj");
            updateScreen(viewer);
            return true;
        case 'c':
            simulation.cells.computeCellInfo();
            return true;
        }
    };

    // simulation.initializeCells();
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

    show_target_current = true;
    show_target = true;
    show_edges = false;

    color.resize(sa.n_dof_design);
    color.setZero();
    
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
        case '1':
            check_modes = true;
            // loadDisplacementVectors("/home/yueli/Documents/ETH/WuKong/dxdp_eigen_vectors.txt");
            // loadDisplacementVectors("/home/yueli/Documents/ETH/WuKong/cell_edge_weights_svd_vectors.txt");
            loadSVDData("cell_d2odx_singular_vectors.txt");
            color.resize(svd_V.cols());
            color.setZero();
            modes = 0;
            std::cout << "modes " << modes << " singular value: " << svd_Sigma(modes) << std::endl;
            return true;
        case '2':
            modes++;
            modes = (modes + svd_U.cols()) % svd_U.cols();
            std::cout << "modes " << modes << " singular value: " << svd_Sigma(modes) << std::endl;
            return true;
        case ' ':
            viewer.core().is_animating = true;
            return true;
        case 'a':
            viewer.core().is_animating = !viewer.core().is_animating;
            return true;
        case 'n':
            load_obj_iter_cnt++;
            std::cout << "state: " << load_obj_iter_cnt << std::endl;
            if (load_opt_state)
            {
                simulation.loadDeformedState("output/cells/15/SQP_iter_" + std::to_string(load_obj_iter_cnt) + ".obj");
                simulation.loadEdgeWeights("output/cells/15/SQP_iter_" + std::to_string(load_obj_iter_cnt) + ".txt", edge_weights);
                updateScreen(viewer);
                return true;
            }
            else if (load_debug_state)
            {
                simulation.loadDeformedState("output/cells/debug/" + std::to_string(load_obj_iter_cnt) + ".obj");
                simulation.loadEdgeWeights("output/cells/debug/" + std::to_string(load_obj_iter_cnt) + ".txt", edge_weights);
                
                updateScreen(viewer);
                return true;
            }
            else
                return false;
        case 'l':
            load_obj_iter_cnt--;
            load_obj_iter_cnt = std::max(0, load_obj_iter_cnt);
            std::cout << "state: " << load_obj_iter_cnt << std::endl;
            if (load_opt_state)
            {
                simulation.loadDeformedState("output/cells/15/SQP_iter_" + std::to_string(load_obj_iter_cnt) + ".obj");
                simulation.loadEdgeWeights("output/cells/15/SQP_iter_" + std::to_string(load_obj_iter_cnt) + ".txt", edge_weights);
                updateScreen(viewer);
                return true;
            }
            else if (load_debug_state)
            {
                simulation.loadDeformedState("output/cells/debug/" + std::to_string(load_obj_iter_cnt) + ".obj");
                simulation.loadEdgeWeights("output/cells/debug/" + std::to_string(load_obj_iter_cnt) + ".txt", edge_weights);
                updateScreen(viewer);
                return true;
            }
            else
                return false;
        }
    };

    viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer &) -> bool
    {
        if(viewer.core().is_animating && check_modes)
        {
            simulation.deformed = simulation.undeformed + simulation.u + svd_U.col(modes) * std::sin(t);
            color.resize(svd_V.cols());
            color = VectorXT::Constant(svd_V.cols(), 0.5) + 10.0 * svd_V.col(modes) * std::sin(t);
            t += 0.1;
            updateScreen(viewer);
        }
        return false;
    };

    viewer.callback_post_draw = [&](igl::opengl::glfw::Viewer &) -> bool
    {
        if(viewer.core().is_animating && !check_modes)
        {
            bool finished = sa.optimizeOneStep(opt_step, sa.objective.default_optimizer);
            if (finished)
            {
                viewer.core().is_animating = false;
            }
            else 
                opt_step++;
            updateScreen(viewer);
        }
        return false;
    };

    simulation.sampleBoundingSurface(bounding_surface_samples);
    sdf_test_sample_idx_offset = bounding_surface_samples.rows();
    bounding_surface_samples_color = bounding_surface_samples;
    for (int i = 0; i < bounding_surface_samples.rows(); i++)
        bounding_surface_samples_color.row(i) = TV(0.1, 1.0, 0.1);

    // simulation.loadDeformedState("/home/yueli/Documents/ETH/WuKong/output/cells/cell/result.obj");

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
    TV max_corner, min_corner;
    simulation.cells.computeBoundingBox(min_corner, max_corner);
    TV shift = TV(1.2 * (max_corner[0] - min_corner[0]), 0, 0);
    T sphere_radius = 0.01 * (max_corner - min_corner).norm();
    simulation.generateMeshForRendering(V, F, C, show_current, show_rest, split, split_a_bit, yolk_only);
    if (use_debug_color)
    {
        C.col(0).setZero(); C.col(1).setOnes(); C.col(2).setZero();
    }
    viewer.data().clear();

    if (show_edge_weights)
    {
        if (svd_V.rows() == 0)
            loadSVDMatrixV("cell_edge_weights_svd_vectors_V.txt");   
        appendCylinderToEdges(svd_V.col(modes), V, F, C);
    }
    if (show_target)
    {
        TV color(1, 0, 0);
        std::vector<TV> target_positions_std_vec;
        if (sa.objective.match_centroid)
            sa.objective.iterateTargets([&](int cell_idx, TV& target)
            {
                if (sa.objective.target_obj_weights[cell_idx] > 1e-2)
                    target_positions_std_vec.push_back(target + shift);
                    // appendSphereToPosition(target + shift, sphere_radius, color, V, F, C);
            });
        else
            sa.objective.iterateWeightedTargets([&](int cell_idx, int data_point_idx, 
                const TV& target, const VectorXT& weights)
            {
                if (sa.objective.target_obj_weights[cell_idx] > 1e-2)
                    target_positions_std_vec.push_back(target + shift);
                    // appendSphereToPosition(target + shift, sphere_radius, color, V, F, C);
            });
        int n_sphere = target_positions_std_vec.size();
        VectorXT target_positions(n_sphere * 3);
        tbb::parallel_for(0, n_sphere, [&](int i){
            target_positions.segment<3>(i * 3) = target_positions_std_vec[i];
        });
        appendSphereToPositionVector(target_positions, 0.4 * sphere_radius, color, V, F, C);

    }
    if (show_target_current)
    {
        TV color(0, 1, 0);  
        std::vector<TV> positions_std_vec;
        if (sa.objective.match_centroid)
        {
            sa.objective.iterateTargets([&](int cell_idx, TV& target){
                TV current;
                if (sa.objective.target_obj_weights[cell_idx] > 1e-2)
                {
                    sa.simulation.cells.computeCellCentroid(simulation.cells.faces[cell_idx], current);
                    // appendSphereToPosition(current + shift, sphere_radius, color, V, F, C);
                    positions_std_vec.push_back(current + shift);
                }
            });
        }
        else
        {
            sa.objective.iterateWeightedTargets([&](int cell_idx, int data_point_idx, 
                const TV& target, const VectorXT& weights)
            {
                if (sa.objective.target_obj_weights[cell_idx] > 1e-2)
                {
                    VectorXT positions;
                    std::vector<int> indices;
                    simulation.cells.getCellVtxAndIdx(cell_idx, positions, indices);
                    int n_pt = weights.rows();
                    // std::cout << "n_pt: " <<  n_pt << " w sum " << weights.sum() << std::endl;
                    TV current = TV::Zero();
                    for (int i = 0; i < n_pt; i++)
                        current += weights[i] * positions.segment<3>(i * 3);
                    // std::cout << current.transpose() << std::endl;
                    // appendSphereToPosition(current + shift, sphere_radius, color, V, F, C);
                    positions_std_vec.push_back(current + shift);
                }
            });
        }
        int n_sphere = positions_std_vec.size();
        VectorXT positions(n_sphere * 3);
        tbb::parallel_for(0, n_sphere, [&](int i){
            positions.segment<3>(i * 3) = positions_std_vec[i];
        });
        appendSphereToPositionVector(positions, 0.4 * sphere_radius, color, V, F, C);
    }
    if (show_target && show_target_current)
    {
        TV color(0, 1, 1);
        std::vector<std::pair<TV, TV>> end_points;
        if (sa.objective.match_centroid)
        {
            sa.objective.iterateTargets([&](int cell_idx, TV& target)
            {
                TV current;
                if (sa.objective.target_obj_weights[cell_idx] > 1e-2)
                {
                    sa.simulation.cells.computeCellCentroid(simulation.cells.faces[cell_idx], current);
                    // appendCylinderToEdge(current + shift, target + shift, color, sphere_radius * 0.25, V, F, C);
                    end_points.push_back(std::make_pair(current + shift, target + shift));
                }
            });
        }
        else
        {
            sa.objective.iterateWeightedTargets([&](int cell_idx, int data_point_idx, 
                const TV& target, const VectorXT& weights)
            {
                if (sa.objective.target_obj_weights[cell_idx] > 1e-2)
                {
                    VectorXT positions;
                    std::vector<int> indices;
                    simulation.cells.getCellVtxAndIdx(cell_idx, positions, indices);
                    int n_pt = weights.rows();
                    TV current = TV::Zero();
                    for (int i = 0; i < n_pt; i++)
                        current += weights[i] * positions.segment<3>(i * 3);
                    // appendCylinderToEdge(current + shift, target + shift, color, sphere_radius * 0.25, V, F, C);
                    end_points.push_back(std::make_pair(current + shift, target + shift));
                }
            });
        }
        appendCylindersToEdges(end_points, color, sphere_radius * 0.15, V, F, C);
    }
    if (show_edges)
    {
        TV color(1, 1, 0);
        int cnt = 0;
        std::vector<std::pair<TV, TV>> end_points;
        simulation.cells.iterateEdgeSerial([&](Edge& edge)
        {
            TV from = simulation.deformed.segment<3>(edge[0] * 3);
            TV to = simulation.deformed.segment<3>(edge[1] * 3);
            end_points.push_back(std::make_pair(from + shift, to + shift));
            // appendCylinderToEdge(from + shift, to + shift, color, 0.005, V, F, C);
            cnt++;
        });
        appendCylindersToEdges(end_points, color, 0.005, V, F, C);
    }
    if (show_edge_weights_opt && edge_weights.rows() != 0 && !show_undeformed)
    {
        VectorXT ewn = edge_weights.normalized();
        Eigen::MatrixXd colors;
        igl::colormap(igl::COLOR_MAP_TYPE_TURBO, edge_weights, false, colors);
        int cnt = 0;
        
        T max_w = sa.design_parameter_bound[1], min_w = sa.design_parameter_bound[0];
        // T max_w = edge_weights.maxCoeff(), min_w = edge_weights.minCoeff();

        T epsilon = min_w + (max_w - min_w) * threshold;

        simulation.cells.iterateApicalEdgeSerial([&](Edge& edge){
            TV from = simulation.deformed.segment<3>(edge[0] * 3);
            TV to = simulation.deformed.segment<3>(edge[1] * 3);
            if (edge_weights[cnt] < epsilon)
            {
                TV color = (edge_weights[cnt] - min_w) / (epsilon - min_w) * TV::Ones();
                if (use_debug_color)
                    color.segment<2>(1).setZero();
                appendCylinderToEdge(from, to, color, 0.01, V, F, C);
            }
            else
                appendCylinderToEdge(from, to, TV(1, 1, 1), 0.01, V, F, C);
            if (edge_weights[cnt] < max_w - 1e-3)
                appendCylinderToEdge(from, to, TV(1, 1, 0), 0.01, V, F, C);
            cnt++;
        });
    }

    if (show_undeformed)
    {
        appendRestShapeShifted(V, F, C, shift);
        
        int cnt = 0;
        if (show_edge_weights_opt)
        {
            T max_w = sa.design_parameter_bound[1], min_w = sa.design_parameter_bound[0];
            T epsilon = min_w + (max_w - min_w) * threshold;
            simulation.cells.iterateApicalEdgeSerial([&](Edge& edge){
                TV from = simulation.undeformed.segment<3>(edge[0] * 3);
                TV to = simulation.undeformed.segment<3>(edge[1] * 3);

                TV from_deformed = simulation.deformed.segment<3>(edge[0] * 3);
                TV to_deformed = simulation.deformed.segment<3>(edge[1] * 3);
                TV _color = (edge_weights[cnt] - min_w) / (epsilon - min_w) * TV::Ones();
                if (use_debug_color)
                    _color.segment<2>(1).setZero();
                appendCylinderToEdge(from_deformed, to_deformed, _color, 0.02, V, F, C);
                appendCylinderToEdge(from + shift, to + shift, _color, 0.02, V, F, C);
                cnt++;
            });
        }
        else
        {
            simulation.cells.iterateApicalEdgeSerial([&](Edge& edge){
                TV from = simulation.undeformed.segment<3>(edge[0] * 3);
                TV to = simulation.undeformed.segment<3>(edge[1] * 3);
                TV cc = color[cnt] * TV::Ones();
                appendCylinderToEdge(from + shift, to + shift, cc, 0.01, V, F, C);
                cnt++;
            });
        }
    }

    
    viewer.data().set_mesh(V, F);
    viewer.data().set_colors(C);  
}

void DiffSimApp::loadSVDMatrixV(const std::string& filename)
{
    std::ifstream in(filename);
    int row;
    in >> row;
    svd_V.resize(row, row); svd_V.setZero();
    for (int i = 0; i < row; i++)
        for (int j = 0; j < row; j++)
            in >> svd_V(i, j);
    in.close();
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

void DiffSimApp::setMenu(igl::opengl::glfw::Viewer& viewer,
        igl::opengl::glfw::imgui::ImGuiMenu& menu)
{
    menu.callback_draw_viewer_menu = [&]()
    {
        if (ImGui::CollapsingHeader("Visualization", ImGuiTreeNodeFlags_DefaultOpen))
        {
            if (ImGui::Checkbox("ShowCurrent", &show_current))
            {
                updateScreen(viewer);
            }
            if (ImGui::Checkbox("ShowTarget", &show_target))
            {
                updateScreen(viewer);
            }
            if (ImGui::Checkbox("ShowTargetCurrent", &show_target_current))
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
            if (ImGui::Checkbox("ShowOutsideVtx", &show_outside_vtx))
            {
                updateScreen(viewer);
            }
            if (ImGui::Checkbox("ShowEdgeWeight", &show_edge_weights))
            {
                updateScreen(viewer);
            }
            if (ImGui::Checkbox("ShowEdgeWeightOptimization", &show_edge_weights_opt))
            {
                updateScreen(viewer);
            }
            if (ImGui::Checkbox("ShowEdges", &show_edges))
            {
                updateScreen(viewer);
            }
            if (ImGui::Checkbox("DebugColor", &use_debug_color))
            {
                updateScreen(viewer);
            }
            if (ImGui::Checkbox("Undeformed", &show_undeformed))
            {
                if (show_undeformed)
                {
                    show_target = false;
                    show_target_current = false;
                }
                updateScreen(viewer);
            }
            if (ImGui::Checkbox("LoadOptStep", &load_opt_state))
            {
                load_obj_iter_cnt = 0;
                load_debug_state = !load_opt_state;
                if (load_opt_state)
                {
                    simulation.loadDeformedState("output/cells/15/SQP_iter_" + std::to_string(0) + ".obj");
                    simulation.loadEdgeWeights("output/cells/15/SQP_iter_" + std::to_string(0) + ".txt", edge_weights);
                }
                updateScreen(viewer);
            }
            if (ImGui::Checkbox("LoadDebugStep", &load_debug_state))
            {
                load_obj_iter_cnt = 0;
                load_opt_state = !load_debug_state;
                if (load_debug_state)
                {
                    simulation.loadDeformedState("output/cells/debug/" + std::to_string(0) + ".obj");
                    simulation.loadEdgeWeights("output/cells/debug/" + std::to_string(0) + ".txt", edge_weights);
                }
                updateScreen(viewer);
            }
            if (ImGui::DragFloat("Threshold", &(threshold), 0.1f, 0.01f, 1.0f))
            {
                updateScreen(viewer);
            }
        }
        if (ImGui::CollapsingHeader("LoadData", ImGuiTreeNodeFlags_DefaultOpen))
        {
            float w = ImGui::GetContentRegionAvailWidth();
            float p = ImGui::GetStyle().FramePadding.x;
            if (ImGui::Button("Weights", ImVec2((w-p)/2.f, 0)))
            {
                std::string fname = igl::file_dialog_open();
                if (fname.length() != 0)
                {
                    simulation.loadEdgeWeights(fname, edge_weights);
                }
            }
            ImGui::SameLine(0, p);
            if (ImGui::Button("State", ImVec2((w-p)/2.f, 0)))
            {
                std::string fname = igl::file_dialog_open();
                if (fname.length() != 0)
                {
                    simulation.loadDeformedState(fname);
                }
            }
            // ImGui::SameLine(0, p);
        }
        if (ImGui::Button("ComputeGraident", ImVec2(-1,0)))
        {
            VectorXT gradient;
            T energy = 0.0;
            sa.objective.gradient(sa.design_parameters, gradient, energy, false);
            std::cout << "E " << energy << " |g| " << gradient.norm() << std::endl;
            std::ofstream out("gradient.txt");
            out << gradient.rows() << " " << gradient.transpose() << std::endl;
            out.close();
        }
        if (ImGui::Button("SaveMesh", ImVec2(-1,0)))
        {
            igl::writeOBJ("current_mesh.obj", V, F);
        }
    };

    if (show_edge_weights)
        loadSVDMatrixV("cell_edge_weights_svd_vectors_V.txt");
}

void DiffSimApp::appendRestShapeShifted(Eigen::MatrixXd& _V, 
    Eigen::MatrixXi& _F, Eigen::MatrixXd& _C, const TV& shift)
{
    Eigen::MatrixXd V_rest, C_rest;
    Eigen::MatrixXi F_rest;
    
    simulation.cells.generateMeshForRendering(V_rest, F_rest, C_rest, true);
    if (use_debug_color)
    {
        C_rest.col(0).setZero(); C_rest.col(1).setOnes(); C_rest.col(2).setZero();
    }
    int n_vtx_prev = V_rest.rows();
    int n_face_prev = F_rest.rows();

    tbb::parallel_for(0, (int)V_rest.rows(), [&](int row_idx){
        V_rest.row(row_idx) += shift;
    });

    tbb::parallel_for(0, (int)F_rest.rows(), [&](int row_idx){
        F_rest.row(row_idx) += Eigen::Vector3i(n_vtx_prev, n_vtx_prev, n_vtx_prev);
    });

    V.conservativeResize(V.rows() + V_rest.rows(), 3);
    F.conservativeResize(F.rows() + F_rest.rows(), 3);
    C.conservativeResize(C.rows() + F_rest.rows(), 3);


    V.block(n_vtx_prev, 0, V_rest.rows(), 3) = V_rest;
    F.block(n_face_prev, 0, F_rest.rows(), 3) = F_rest;
    C.block(n_face_prev, 0, F_rest.rows(), 3) = C_rest;
}

void DiffSimApp::appendCylinderToEdges(const VectorXT weights_vector, 
        Eigen::MatrixXd& _V, Eigen::MatrixXi& _F, Eigen::MatrixXd& _C)
{
       
    std::vector<Edge> contracting_edges;

    simulation.cells.iterateApicalEdgeSerial([&](Edge& e)
    {
        contracting_edges.push_back(e);
    });

    T visual_R = 0.01;
    int n_div = 10;
    T theta = 2.0 * EIGEN_PI / T(n_div);
    VectorXT points = VectorXT::Zero(n_div * 3);
    for(int i = 0; i < n_div; i++)
        points.segment<3>(i * 3) = TV(visual_R * std::cos(theta * T(i)), 
        0.0, visual_R*std::sin(theta*T(i)));
    
    int rod_offset_v = n_div * 2;
    int rod_offset_f = n_div * 2;

    int n_row_V = V.rows();
    int n_row_F = F.rows();

    int n_contracting_edges = contracting_edges.size();
    
    V.conservativeResize(n_row_V + n_contracting_edges * rod_offset_v, 3);
    F.conservativeResize(n_row_F + n_contracting_edges * rod_offset_f, 3);
    C.conservativeResize(n_row_F + n_contracting_edges * rod_offset_f, 3);

    
    for (int j = 0; j < n_contracting_edges; j++)
    {
        int rov = n_row_V + j * rod_offset_v;
        int rof = n_row_F + j * rod_offset_f;

        TV vtx_from = simulation.deformed.segment<3>(contracting_edges[j][0] * 3);
        TV vtx_to = simulation.deformed.segment<3>(contracting_edges[j][1] * 3);

        TV axis_world = vtx_to - vtx_from;
        TV axis_local(0, axis_world.norm(), 0);

        Matrix<T, 3, 3> R = Eigen::Quaternion<T>().setFromTwoVectors(axis_world, axis_local).toRotationMatrix();

        for(int i = 0; i < n_div; i++)
        {
            for(int d = 0; d < 3; d++)
            {
                V(rov + i, d) = points[i * 3 + d];
                V(rov + i+n_div, d) = points[i * 3 + d];
                if (d == 1)
                    V(rov + i+n_div, d) += axis_world.norm();
            }

            // central vertex of the top and bottom face
            V.row(rov + i) = (V.row(rov + i) * R).transpose() + vtx_from;
            V.row(rov + i + n_div) = (V.row(rov + i + n_div) * R).transpose() + vtx_from;

            F.row(rof + i*2 ) = IV(rov + i, rov + i+n_div, rov + (i+1)%(n_div));
            F.row(rof + i*2 + 1) = IV(rov + (i+1)%(n_div), rov + i+n_div, rov + (i+1)%(n_div) + n_div);

            if (weights_vector[j] < 0)
            {
                C.row(rof + i*2 ) = TV(1.0, 0.0, 0.0);
                C.row(rof + i*2 + 1) = TV(1.0, 0.0, 0.0);
            }
            else
            {
                C.row(rof + i*2 ) = TV(0.0, 1.0, 0.0);
                C.row(rof + i*2 + 1) = TV(0.0, 1.0, 0.0);
            }
            // C.row(rof + i*2 ) = TV(1.0, 1.0, 1.0) * weights_vector[j];
            // C.row(rof + i*2 + 1) = TV(1.0, 1.0, 1.0) * weights_vector[j];
        }
    }   
}


void SimulationApp::appendCylindersToEdges(const std::vector<std::pair<TV, TV>>& edge_pairs, 
        const TV& color, T radius,
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

            _C.row(n_row_F + ei * offset_f + i*2 ) = color;
            _C.row(n_row_F + ei * offset_f + i*2 + 1) = color;
        }
    });
}

void SimulationApp::appendCylinderToEdge(const TV& vtx_from, const TV& vtx_to, const TV& color,
        T radius, Eigen::MatrixXd& _V, Eigen::MatrixXi& _F, Eigen::MatrixXd& _C)
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

    _V.conservativeResize(n_row_V + offset_v, 3);
    _F.conservativeResize(n_row_F + offset_f, 3);
    _C.conservativeResize(n_row_F + offset_f, 3);

    TV axis_world = vtx_to - vtx_from;
    TV axis_local(0, axis_world.norm(), 0);

    Matrix<T, 3, 3> R = Eigen::Quaternion<T>().setFromTwoVectors(axis_world, axis_local).toRotationMatrix();

    for(int i = 0; i < n_div; i++)
    {
        for(int d = 0; d < 3; d++)
        {
            _V(n_row_V + i, d) = points[i * 3 + d];
            _V(n_row_V + i+n_div, d) = points[i * 3 + d];
            if (d == 1)
                _V(n_row_V + i+n_div, d) += axis_world.norm();
        }

        // central vertex of the top and bottom face
        _V.row(n_row_V + i) = (_V.row(n_row_V + i) * R).transpose() + vtx_from;
        _V.row(n_row_V + i + n_div) = (_V.row(n_row_V + i + n_div) * R).transpose() + vtx_from;

        _F.row(n_row_F + i*2 ) = IV(n_row_V + i, n_row_V + i+n_div, n_row_V + (i+1)%(n_div));
        _F.row(n_row_F + i*2 + 1) = IV(n_row_V + (i+1)%(n_div), n_row_V + i+n_div, n_row_V + (i+1)%(n_div) + n_div);

        _C.row(n_row_F + i*2 ) = color;
        _C.row(n_row_F + i*2 + 1) = color;
    }
}

void DiffSimApp::loaddpAndAppendCylinder(const std::string& filename, 
        Eigen::MatrixXd& _V, Eigen::MatrixXi& _F, Eigen::MatrixXd& _C)
{
    VectorXT p(sa.n_dof_design);
    std::ifstream in(filename);
    for (int i = 0; i < sa.n_dof_design; i++)
        in >> p[i];
    in.close();
    p  = p.array() + p.minCoeff();
    p.normalize();
    std::vector<Edge> contracting_edges;

    simulation.cells.iterateApicalEdgeSerial([&](Edge& e)
    {
        contracting_edges.push_back(e);
    });

    T visual_R = 0.01;
    int n_div = 10;
    T theta = 2.0 * EIGEN_PI / T(n_div);
    VectorXT points = VectorXT::Zero(n_div * 3);
    for(int i = 0; i < n_div; i++)
        points.segment<3>(i * 3) = TV(visual_R * std::cos(theta * T(i)), 
        0.0, visual_R*std::sin(theta*T(i)));
    
    int rod_offset_v = n_div * 2;
    int rod_offset_f = n_div * 2;

    int n_row_V = V.rows();
    int n_row_F = F.rows();

    int n_contracting_edges = contracting_edges.size();
    
    V.conservativeResize(n_row_V + n_contracting_edges * rod_offset_v, 3);
    F.conservativeResize(n_row_F + n_contracting_edges * rod_offset_f, 3);
    C.conservativeResize(n_row_F + n_contracting_edges * rod_offset_f, 3);


    for (int j = 0; j < n_contracting_edges; j++)
    {
        int rov = n_row_V + j * rod_offset_v;
        int rof = n_row_F + j * rod_offset_f;

        TV vtx_from = simulation.deformed.segment<3>(contracting_edges[j][0] * 3);
        TV vtx_to = simulation.deformed.segment<3>(contracting_edges[j][1] * 3);

        TV axis_world = vtx_to - vtx_from;
        TV axis_local(0, axis_world.norm(), 0);

        Matrix<T, 3, 3> R = Eigen::Quaternion<T>().setFromTwoVectors(axis_world, axis_local).toRotationMatrix();

        for(int i = 0; i < n_div; i++)
        {
            for(int d = 0; d < 3; d++)
            {
                V(rov + i, d) = points[i * 3 + d];
                V(rov + i+n_div, d) = points[i * 3 + d];
                if (d == 1)
                    V(rov + i+n_div, d) += axis_world.norm();
            }

            // central vertex of the top and bottom face
            V.row(rov + i) = (V.row(rov + i) * R).transpose() + vtx_from;
            V.row(rov + i + n_div) = (V.row(rov + i + n_div) * R).transpose() + vtx_from;

            F.row(rof + i*2 ) = IV(rov + i, rov + i+n_div, rov + (i+1)%(n_div));
            F.row(rof + i*2 + 1) = IV(rov + (i+1)%(n_div), rov + i+n_div, rov + (i+1)%(n_div) + n_div);

            C.row(rof + i*2 ) = TV(1.0, 1.0, 1.0) * p[j];
            C.row(rof + i*2 + 1) = TV(1.0, 1.0, 1.0) * p[j];
        }
    }   
}

void SimulationApp::appendSphereToPosition(const TV& position, T radius, const TV& color,
        Eigen::MatrixXd& _V, Eigen::MatrixXi& _F, Eigen::MatrixXd& _C)
{
    Eigen::MatrixXd v_sphere;
    Eigen::MatrixXi f_sphere;

    igl::readOBJ("/home/yueli/Documents/ETH/WuKong/Projects/DigitalFabrics/Data/sphere162.obj", v_sphere, f_sphere);
    
    Eigen::MatrixXd c_sphere(f_sphere.rows(), f_sphere.cols());
    
    v_sphere = v_sphere * radius;

    tbb::parallel_for(0, (int)v_sphere.rows(), [&](int row_idx){
        v_sphere.row(row_idx) += position;
    });

    int n_vtx_prev = V.rows();
    int n_face_prev = F.rows();

    tbb::parallel_for(0, (int)f_sphere.rows(), [&](int row_idx){
        f_sphere.row(row_idx) += Eigen::Vector3i(n_vtx_prev, n_vtx_prev, n_vtx_prev);
    });

    tbb::parallel_for(0, (int)f_sphere.rows(), [&](int row_idx){
        c_sphere.row(row_idx) = color;
    });

    V.conservativeResize(V.rows() + v_sphere.rows(), 3);
    F.conservativeResize(F.rows() + f_sphere.rows(), 3);
    C.conservativeResize(C.rows() + f_sphere.rows(), 3);


    V.block(n_vtx_prev, 0, v_sphere.rows(), 3) = v_sphere;
    F.block(n_face_prev, 0, f_sphere.rows(), 3) = f_sphere;
    C.block(n_face_prev, 0, f_sphere.rows(), 3) = c_sphere;
}

void SimulationApp::appendSphereToPositionVector(const VectorXT& position, T radius, const TV& color,
    Eigen::MatrixXd& _V, Eigen::MatrixXi& _F, Eigen::MatrixXd& _C)
{
    int n_pt = position.rows() / 3;

    Eigen::MatrixXd v_sphere;
    Eigen::MatrixXi f_sphere;

    igl::readOBJ("/home/yueli/Documents/ETH/WuKong/Projects/DigitalFabrics/Data/sphere162.obj", v_sphere, f_sphere);
    
    Eigen::MatrixXd c_sphere(f_sphere.rows(), f_sphere.cols());
    
    v_sphere = v_sphere * radius;

    int n_vtx_prev = V.rows();
    int n_face_prev = F.rows();

    V.conservativeResize(V.rows() + v_sphere.rows() * n_pt, 3);
    F.conservativeResize(F.rows() + f_sphere.rows() * n_pt, 3);
    C.conservativeResize(C.rows() + f_sphere.rows() * n_pt, 3);

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

        V.block(offset_v, 0, v_sphere.rows(), 3) = v_sphere_i;
        F.block(offset_f, 0, f_sphere.rows(), 3) = f_sphere_i;
        C.block(offset_f, 0, f_sphere.rows(), 3) = c_sphere_i;
    });
}


void DataViewerApp::loadRawData()
{
    
    data_io.loadDataFromBinary("/home/yueli/Downloads/drosophila_data/drosophila_side2_time_xyz.dat", 
            "/home/yueli/Downloads/drosophila_data/drosophila_side2_ids.dat",
            "/home/yueli/Downloads/drosophila_data/drosophila_side2_scores.dat");
}

void DataViewerApp::loadFilteredData()
{
    std::string filename = "/home/yueli/Documents/ETH/WuKong/Projects/CellSim/data/trajectories.dat";
    data_io.loadTrajectories(filename, cell_trajectories);
}

void DataViewerApp::updateScreen(igl::opengl::glfw::Viewer& viewer)
{
    viewer.data().clear();
    VectorXT frame_data;
    loadFrameData(frame_cnt, frame_data);
    V.resize(0, 0); F.resize(0, 0); C.resize(0, 0);
    TV color = TV(0, 1, 0);
    appendSphereToPositionVector(frame_data, 0.025, color, V, F, C);
    viewer.data().set_mesh(V, F);
    viewer.data().set_colors(C);   
}

void DataViewerApp::loadFrameData(int frame, VectorXT& frame_data)
{
    frame_data = cell_trajectories.col(frame);
    
    int n_pt = frame_data.rows() / 3;
    Matrix<T, 3, 3> R;
    R << 0.960277, -0.201389, 0.229468, 0.2908, 0.871897, -0.519003, -0.112462, 0.558021, 0.887263;
    Matrix<T, 3, 3> R2 = Eigen::AngleAxis<T>(0.20 * M_PI + 0.5 * M_PI, TV(-1.0, 0.0, 0.0)).toRotationMatrix();

    for (int i = 0; i < n_pt; i++)
    {
        TV pos = frame_data.segment<3>(i * 3);
        TV updated = (pos - TV(605.877,328.32,319.752)) / 1096.61;
        updated = R2 * R * updated;
        // frame_data.segment<3>(i * 3) = updated * 0.8 * simulation.cells.unit; 
        frame_data.segment<3>(i * 3) = updated * 0.9 * simulation.cells.unit; 
    }
}

void DataViewerApp::setViewer(igl::opengl::glfw::Viewer& viewer,
        igl::opengl::glfw::imgui::ImGuiMenu& menu)
{
    menu.callback_draw_viewer_menu = [&]()
    {
        if (ImGui::CollapsingHeader("Visualization", ImGuiTreeNodeFlags_DefaultOpen))
        {
            if (ImGui::Checkbox("RawData", &raw_data))
            {
                if (raw_data)
                    loadRawData();
                else
                    loadFilteredData();
                updateScreen(viewer);
            }
        }
    };

    viewer.callback_key_pressed = 
        [&](igl::opengl::glfw::Viewer & viewer,unsigned int key,int mods)->bool
    {
        switch(key)
        {
        default: 
            return false;
        case 'n':
            frame_cnt++;
            std::cout << "frame " << frame_cnt << std::endl;
            updateScreen(viewer);
            return false;
        case 'l':
            frame_cnt--;
            std::cout << "frame " << frame_cnt << std::endl;
            updateScreen(viewer);
            return false;
        }
    };

    loadFilteredData();

    updateScreen(viewer);

    viewer.core().background_color.setOnes();
    viewer.data().set_face_based(true);
    viewer.data().shininess = 1.0;
    viewer.data().point_size = 10.0;

    viewer.data().set_mesh(V, F);     
    viewer.data().set_colors(C);

    viewer.core().align_camera_center(V);
}