#include "../include/Foam3DApp.h"
#include "../src/implot/implot.h"

#include <filesystem>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/core/eigen.hpp>

#include "../include/Tessellation/Power.h"

void Foam3DApp::setViewer(igl::opengl::glfw::Viewer &viewer,
                          igl::opengl::glfw::imgui::ImGuiMenu &menu) {
    menu.callback_draw_viewer_menu = [&]() {
        if (ImGui::Checkbox("Optimize", &optimize)) {

        }

        std::vector<std::string> optTypes;
        optTypes.push_back("Gradient Descent");
        optTypes.push_back("Newton's Method");
        optTypes.push_back("BFGS");
        ImGui::Combo("Optimizer", &optimizer, optTypes);

        ImGui::Spacing();
        ImGui::Spacing();

        std::vector<std::string> colorModes;
        colorModes.push_back("Random");
        if (ImGui::Combo("Colors", &colormode, colorModes)) {
            updateViewerData(viewer);
        }

        ImGui::Spacing();
        ImGui::Spacing();

        std::vector<std::string> sliceModes;
        sliceModes.push_back("Behind plane");
        sliceModes.push_back("Intersecting plane");
        sliceModes.push_back("No slice");
        if (ImGui::Combo("Slice Mode", &slice_mode, sliceModes)) {
            updateViewerData(viewer);
        }
        ImGui::Checkbox("Show plane", &slice_visible);
        ImGui::Checkbox("Move plane", &slice_follow);

        ImGui::Spacing();
        ImGui::Spacing();

        ImGui::Text("Simulation Setup");

        std::vector<std::string> scenarios;
        scenarios.push_back("No Boundary");
        scenarios.push_back("Cube");

        if (ImGui::Combo("Scenario", &generate_scenario_type, scenarios)) {

        }
        if (generate_scenario_type > -1) {
            ImGui::SetNextItemWidth(ImGui::GetWindowWidth() * 0.5);
            ImGui::InputInt("Cells", &generate_scenario_num_sites, 10, 100);
        }
        if (ImGui::Button("Generate")) {
            generateScenario();
            updateViewerData(viewer);
        }
    };

    // Draw additional windows
    menu.callback_draw_custom_window = [&]() {
        if (!ImPlot::GetCurrentContext()) {
            auto ctx = ImPlot::CreateContext();
            ImPlot::SetCurrentContext(ctx);
        }

        updatePlotData();
    };

    viewer.callback_key_pressed =
            [&](igl::opengl::glfw::Viewer &viewer, unsigned int key, int mods) -> bool {
                switch (key) {
                    case GLFW_KEY_SPACE:
                        optimize = !optimize;
                        return false;
                    default:
                        return false;
                }
            };

    viewer.callback_mouse_scroll =
            [&](igl::opengl::glfw::Viewer &viewer, float t) -> bool {
                return false;
            };

    viewer.callback_mouse_down =
            [&](igl::opengl::glfw::Viewer &viewer, int a, int b) -> bool {
                return false;
            };

    viewer.callback_mouse_up =
            [&](igl::opengl::glfw::Viewer &viewer, int a, int b) -> bool {
                return false;
            };

    viewer.callback_mouse_move =
            [&](igl::opengl::glfw::Viewer &viewer, int a, int b) -> bool {
                return false;
            };

    viewer.callback_pre_draw =
            [&](igl::opengl::glfw::Viewer &viewer) -> bool {
                updateViewerData(viewer);
                return false;
            };

    generateScenario();

    viewer.core().viewport = Eigen::Vector4f(0, 0, 1500, 1000);
//    viewer.core().camera_zoom = 2.07 * 1.5;
    viewer.data(0).show_lines = 0;
    viewer.core().background_color.setOnes();
    viewer.data(0).point_size = 10;
    viewer.core().is_animating = true;
//    viewer.data(0).shininess = 0;
//    viewer.core().lighting_factor = 0;

    updateViewerData(viewer);
}

void Foam3DApp::generateScenario() {
    switch (generate_scenario_type) {
        case 0:
            break;
        default:
            std::cout << "Error: scenario not implemented!";
            break;
    }

    double infp = 10;
    VectorXd infbox(8 * 3);
    infbox << -infp, -infp, -infp,
            -infp, -infp, infp,
            -infp, infp, -infp,
            -infp, infp, infp,
            infp, -infp, -infp,
            infp, -infp, infp,
            infp, infp, -infp,
            infp, infp, infp;

    srand(time(NULL));
//    int bb = rand();
//    std::cout << bb << std::endl;
//    srand(bb);
//    srand(1504356941);
    VectorXd vertices((8 + generate_scenario_num_sites) * 3);
    vertices << VectorXd::Random(generate_scenario_num_sites * 3), infbox;
    VectorXd params = VectorXd::Zero(8 + generate_scenario_num_sites);

    foam.vertices = vertices;
    foam.params = params;

    double bound = 1;
    MatrixXd bbox(8, 3);
    bbox << -bound, -bound, -bound,
            bound, -bound, -bound,
            -bound, bound, -bound,
            bound, bound, -bound,
            -bound, -bound, bound,
            bound, -bound, bound,
            -bound, bound, bound,
            bound, bound, bound;

    MatrixXi btri(12, 3);
    btri << 0, 2, 1,
            2, 3, 1,
            0, 1, 4,
            1, 5, 4,
            0, 4, 2,
            4, 6, 2,
            4, 5, 6,
            5, 7, 6,
            1, 3, 5,
            3, 7, 5,
            2, 6, 3,
            6, 7, 3;

    foam.tessellation.bv.resize(bbox.rows());
    foam.tessellation.bf.resize(btri.rows());
    for (int i = 0; i < bbox.rows(); i++) {
        foam.tessellation.bv[i].pos = bbox.row(i);
    }
    for (int i = 0; i < btri.rows(); i++) {
        foam.tessellation.bf[i].vertices = btri.row(i);
    }

    foam.tessellation.tessellate(foam.vertices, foam.params);
}

void Foam3DApp::updateViewerData(igl::opengl::glfw::Viewer &viewer) {
    Eigen::Matrix<float, 4, 1> eye;
    eye << viewer.core().camera_eye, 1.0f;
    const Eigen::Matrix<double, 3, 1> camera_pos = (viewer.core().view.inverse() * eye).cast<double>().head(3);
    if (camera_pos.hasNaN()) return;

    Eigen::Matrix<double, -1, -1> points;
    Eigen::Matrix<double, -1, -1> points_c;
    Eigen::Matrix<double, -1, -1> nodes;
    Eigen::Matrix<int, -1, -1> lines;
    Eigen::Matrix<double, -1, -1> lines_c;
    Eigen::Matrix<double, -1, -1> V;
    Eigen::Matrix<int, -1, -1> F;
    Eigen::Matrix<double, -1, -1> Fc;

    {
        int nx = foam.tessellation.nodes.size();
        std::map<Node, int> nodeIndices;
        nodes.resize(nx, 3);
        int ix = 0;
        for (std::pair<Node, NodePosition> p: foam.tessellation.nodes) {
            nodes.row(ix) = p.second.pos;
            nodeIndices[p.first] = ix;
            ix++;
        }
        V = nodes;

        int nc = foam.tessellation.c.rows() / 4;
        bool cellVisible[nc];
        bool cellHasNodeBehind[nc];
        bool cellHasNodeFront[nc];
        for (int i = 0; i < nc; i++) {
            cellHasNodeBehind[i] = false;
            cellHasNodeFront[i] = false;
        }

//        for (std::pair<Node, NodePosition> p: foam.tessellation.nodes) {
//            TV3 node = p.second.pos;
//            bool behind = node.dot(slice_normal) < slice_offset.dot(slice_normal);
//            int startj;
//            switch (p.first.type) {
//                case STANDARD:
//                    startj = 0;
//                    break;
//                case B_FACE:
//                    startj = 1;
//                    break;
//                case B_EDGE:
//                    startj = 2;
//                    break;
//                default:
//                    startj = 4;
//            }
//            for (int j = startj; j < 4; j++) {
//                int cell = p.first.gen[j];
//                cellHasNodeBehind[cell] = cellHasNodeBehind[cell] || behind;
//                cellHasNodeFront[cell] = cellHasNodeFront[cell] || !behind;
//            }
//        }

        for (Face face: foam.tessellation.faces) {
            for (Node node: face.nodes) {
                TV3 pos = foam.tessellation.nodes.at(node).pos;
                bool behind = pos.dot(slice_normal) < slice_offset.dot(slice_normal);

                cellHasNodeBehind[face.site0] = cellHasNodeBehind[face.site0] || behind;
                cellHasNodeFront[face.site0] = cellHasNodeFront[face.site0] || !behind;
                if (face.site1 >= 0) {
                    cellHasNodeBehind[face.site1] = cellHasNodeBehind[face.site1] || behind;
                    cellHasNodeFront[face.site1] = cellHasNodeFront[face.site1] || !behind;
                }
            }
        }

        for (int i = 0; i < nc; i++) {
            if (i >= nc - 8) {
                cellVisible[i] = false;
            } else {
                switch (slice_mode) {
                    case 0:
                        cellVisible[i] = cellHasNodeBehind[i];
                        break;
                    case 1:
                        cellVisible[i] = cellHasNodeBehind[i] && cellHasNodeFront[i];
                        break;
                    case 2:
                        cellVisible[i] = true;
                        break;
                    default:
                        break;
                }
            }
        }

        points.resize(nc, 3);
        for (int i = 0; i < nc; i++) {
            points.row(i) = foam.tessellation.c.segment<3>(i * 4);
        }
        points_c = 0 * points;

        srand(0);
        MatrixXd colors = 0.5 * (MatrixXd::Random(nc, 3) + MatrixXd::Ones(nc, 3));

        int ne = 0;
        int ntri = 0;
        for (Face face: foam.tessellation.faces) {
            bool orientation;
            if (face.site1 >= 0) {
                TV3 c0 = foam.vertices.segment<3>(face.site0 * 3);
                TV3 c1 = foam.vertices.segment<3>(face.site1 * 3);
                orientation = (c1 - c0).dot(camera_pos - c0) > 0;
            } else {
                orientation = true;
            }

            int cell = orientation ? face.site0 : face.site1;
            if (cell == -1 || !cellVisible[cell]) continue;
            ne += face.nodes.size();
            ntri += face.nodes.size() - 2;
        }
        lines.resize(ne, 2);
        F.resize(ntri, 3);
        Fc.resize(ntri, 3);
        int e = 0;
        int f = 0;
        for (Face face: foam.tessellation.faces) {
            bool orientation;
            if (face.site1 >= 0) {
                TV3 c0 = foam.vertices.segment<3>(face.site0 * 3);
                TV3 c1 = foam.vertices.segment<3>(face.site1 * 3);
                orientation = (c1 - c0).dot(camera_pos - c0) > 0;
            } else {
                orientation = true;
            }

            int cell = orientation ? face.site0 : face.site1;
            if (cell == -1 || !cellVisible[cell]) continue;

            for (int i = 0; i < face.nodes.size(); i++) {
                lines(e, 0) = nodeIndices.at(face.nodes[i]);
                lines(e, 1) = nodeIndices.at(face.nodes[(i + 1) % face.nodes.size()]);
                e++;
            }
            for (int i = 1; i < face.nodes.size() - 1; i++) {
                if (orientation) {
                    F(f, 0) = nodeIndices.at(face.nodes[0]);
                    F(f, 1) = nodeIndices.at(face.nodes[i]);
                    F(f, 2) = nodeIndices.at(face.nodes[i + 1]);
                    Fc.row(f) = colors.row(cell);
                } else {
                    F(f, 0) = nodeIndices.at(face.nodes[0]);
                    F(f, 2) = nodeIndices.at(face.nodes[i]);
                    F(f, 1) = nodeIndices.at(face.nodes[i + 1]);
                    Fc.row(f) = colors.row(cell);
                }
                f++;
            }
        }
        lines_c = MatrixXd::Zero(ne, 3);

        viewer.data(0).clear();
        viewer.data(0).set_mesh(V, F);
        viewer.data(0).set_colors(Fc);
        viewer.data(0).show_lines = false;

//        viewer.data(0).set_points(points, points_c);
        viewer.data(0).set_edges(nodes, lines, lines_c);

        if (viewer.data_list.size() < 2) {
            viewer.append_mesh(true);
        }

        if (slice_follow) {
            slice_normal = camera_pos * 10;
            if (camera_pos.norm() > 10) {
                slice_offset = camera_pos.normalized() * (camera_pos.norm() - 10) * 0.05;
            } else {
                slice_offset = TV3::Zero();
            }
        }

        MatrixXd V1(4, 3);
        MatrixXi F1(2, 3);
        TV3 planeCorner = slice_normal.cross(TV3(0, 0, 1)).normalized() * slice_normal.norm() + slice_offset;
        V1.row(0) = planeCorner;
        planeCorner = planeCorner.cross(slice_normal).normalized() * slice_normal.norm() + slice_offset;
        V1.row(1) = planeCorner;
        planeCorner = planeCorner.cross(slice_normal).normalized() * slice_normal.norm() + slice_offset;
        V1.row(2) = planeCorner;
        planeCorner = planeCorner.cross(slice_normal).normalized() * slice_normal.norm() + slice_offset;
        V1.row(3) = planeCorner;
        F1 << 0, 1, 3, 1, 2, 3;
        MatrixXd F1c = MatrixXd::Constant(2, 3, 0.8);

        viewer.data(1).clear();
        viewer.data(1).set_mesh(V1, F1);
        viewer.data(1).set_colors(F1c);
        viewer.data(1).show_lines = 0;
        viewer.data(1).is_visible = slice_visible;
    }

//    else {
//        V = ts.V;
//        F = ts.F;
//        Fc = ts.Fc;
//
////        Fc = MatrixXT::Zero(F.rows(), 3);
////        Fc.col(0).setConstant(1);
//
//        viewer.data(0).clear();
//        viewer.data(0).set_mesh(V, F);
//        viewer.data(0).set_colors(Fc);
//        viewer.data(0).show_lines = true;
//    }
}

void Foam3DApp::updatePlotData() {

}
