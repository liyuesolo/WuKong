#include <igl/opengl/glfw/Viewer.h>
#include <igl/project.h>
#include <igl/unproject_on_plane.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>
#include <igl/png/writePNG.h>
#include <igl/png/readPNG.h>
#include "igl/colormap.h"

#include "UI.h"
#include "EoLRodSim.h"
#include "Homogenization.h"
#include "HybridC2Curve.h"

#define T double
#define dim 2


bool USE_VIEWER = true;

EoLRodSim<T, dim> eol_sim;
HybridC2Curve<T, dim> hybrid_curve;
Homogenization<T, dim> homogenizer(eol_sim);

Eigen::MatrixXd V;
Eigen::MatrixXi F;
Eigen::MatrixXd C;

Eigen::MatrixXd nodes;


Eigen::MatrixXd V_drawing;
Eigen::MatrixXi F_drawing;

static bool tileUnit = false;
static bool showUnit = false;
static bool showStretching = false;
static bool show_index = false;
static bool show_original = false;
static bool per_yarn = true;
static bool slide = false;
static bool draw_unit = false;

static bool drawing = true;
static bool editing = !drawing;
static bool show_data_points = true;

static float theta_pbc = 0;
static float strain = 1.0;
static int n_rod_per_yarn = 4;

int n_faces = 20;

std::vector<HybridC2Curve<T, dim>> hybrid_curves;

auto updateScreen = [&](igl::opengl::glfw::Viewer& viewer)
{
    viewer.data().clear();
    if (draw_unit)
    {
        if(V_drawing.size())
        {
            viewer.data().set_mesh(V_drawing, F_drawing);
        }
    }
    else
    {
        if(tileUnit)
            eol_sim.buildPeriodicNetwork(V, F, C);
        else
            eol_sim.buildMeshFromRodNetwork(V, F, eol_sim.q, eol_sim.rods, eol_sim.normal);
        viewer.data().set_mesh(V, F);
        if(showUnit)
            viewer.data().set_colors(C);
        if (per_yarn)
        {
            eol_sim.getColorPerYarn(C, n_rod_per_yarn);
            viewer.data().set_colors(C);
            if(tileUnit)
            {
                eol_sim.getColorPerYarn(C, n_rod_per_yarn);
                C.conservativeResize(F.rows(), 3);
                tbb::parallel_for(0, eol_sim.n_rods * n_faces, [&](int i){
                    for(int j = 1; j < std::floor(F.rows()/eol_sim.n_rods/40); j++)
                    {
                        C.row(j * eol_sim.n_rods * n_faces + i) = C.row(i);
                    }
                });
                viewer.data().set_colors(C);
            }
        }
        if(show_original && !tileUnit)
        {
            Eigen::MatrixXd X, x;
            eol_sim.getEulerianDisplacement(X, x);
            for (int i = 0; i < X.rows(); i++)
            {
                // viewer.data().add_edges(X.row(i), x.row(i), Eigen::RowVector3d(1, 1, 1));
            }
            viewer.data().add_points(X, Eigen::RowVector3d(1,1,1));
            // viewer.data().add_points(x, Eigen::RowVector3d(0,0,0));  
        }
    }
    
};

bool key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier)
{
    if (key == ' ')
    {
        if (draw_unit)
        {
            std::vector<Vector<T, 2>> points_on_curve;
            hybrid_curve.getLinearSegments(points_on_curve);
            appendCylinderMesh(viewer, V_drawing, F_drawing, points_on_curve, true);
        }
        else
        {
            eol_sim.advanceOneStep();   
        }
        updateScreen(viewer);
    }
    return false;
}

enum TestCase{
    DrawUnit, StaticSolve, BatchRendering
};

const char* test_case_names[] = {
    "DrawUnit", "StaticSolve", "BatchRendering"
};


int main(int argc, char *argv[])
{
    using RGBMat = Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic>;
    // RGBMat R, G, B, A;
    // igl::png::readPNG("checkerboard.png", R, G, B, A);

    int n_test_case = sizeof(test_case_names)/sizeof(const char*);
    
    int selected = -1;
    double u0 = 0.0, x0 = 0.0, y0 = 0.0;

    static TestCase test = BatchRendering;
    TestCase test_current = StaticSolve; // set to be a different from above or change the above one to be a random one

    auto setupScene = [&](igl::opengl::glfw::Viewer& viewer)
    {   
        if (test_current == DrawUnit)
        {
            draw_unit = true;
            // viewer.core().camera_zoom = 0.1;
        }
        else if (test_current == StaticSolve)
        {
            homogenizer.testOneSample();
            draw_unit = false;
        }
        updateScreen(viewer);
    };

    igl::opengl::glfw::Viewer viewer;
    
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    
    if (test_current == BatchRendering)
    {
        menu.callback_draw_viewer_menu = [&](){
            viewer.core().align_camera_center(viewer.data().V, viewer.data().F);
        };
    }
    else if (test_current == StaticSolve)
    {
        viewer.plugins.push_back(&menu);
        
        menu.callback_draw_viewer_menu = [&]()
        {   
            // menu.draw_viewer_menu();
            if (ImGui::CollapsingHeader("Scene", ImGuiTreeNodeFlags_DefaultOpen))
            {   
                ImGui::Combo("TestCase", (int *)(&test_current), test_case_names, n_test_case);
                // if(test != test_current)
                // {
                //     test = test_current;
                //     setupScene(viewer);
                // }
            }
            if (ImGui::CollapsingHeader("PeriodicBC", ImGuiTreeNodeFlags_DefaultOpen))
            {   
                if (ImGui::DragFloat("Angle", &(eol_sim.theta), 0.f, 0.1f, M_PI * 2.f))
                {
                    eol_sim.resetScene();
                    Vector<T, dim> strain_dir, ortho_dir;
                    eol_sim.setUniaxialStrain(eol_sim.theta, 1.1, strain_dir, ortho_dir);
                    eol_sim.advanceOneStep();
                    updateScreen(viewer);
                }
                if (ImGui::DragFloat("Strain", &(strain), 1.f, 0.02f, 1.1f))
                {
                    eol_sim.resetScene();
                    Vector<T, dim> strain_dir, ortho_dir;
                    eol_sim.setUniaxialStrain(eol_sim.theta, strain, strain_dir, ortho_dir);
                    eol_sim.advanceOneStep();
                    updateScreen(viewer);
                }
                if (ImGui::Checkbox("FixEulerian", &eol_sim.disable_sliding))
                {
                    if(!eol_sim.disable_sliding)
                    {
                        eol_sim.freeEulerian();
                        eol_sim.resetScene();
                        updateScreen(viewer);
                    }
                    else
                    {
                        eol_sim.fixEulerian();
                        eol_sim.resetScene();
                        updateScreen(viewer);
                    }
                }
                if (ImGui::Checkbox("RegularizeEulerian", &eol_sim.add_eularian_reg))
                {
                    eol_sim.resetScene();
                    updateScreen(viewer);
                }
                if (ImGui::Checkbox("Tunnel", &eol_sim.add_contact_penalty))
                {
                    eol_sim.resetScene();
                    updateScreen(viewer);
                }
                if (ImGui::Checkbox("Shearing", &eol_sim.add_shearing))
                {
                    eol_sim.resetScene();
                    updateScreen(viewer);
                }
                if (ImGui::Checkbox("Stretching", &eol_sim.add_stretching))
                {
                    eol_sim.resetScene();
                    updateScreen(viewer);
                }
                if (ImGui::Checkbox("Bending", &eol_sim.add_bending))
                {
                    eol_sim.resetScene();
                    updateScreen(viewer);
                }
                if (ImGui::Checkbox("TileUnit", &tileUnit))
                {
                    updateScreen(viewer);
                }
                if (ImGui::Checkbox("ShowIndex", &show_index))
                {
                    if(show_index)
                    {
                        for (int i = 0; i < eol_sim.n_nodes; i++)
                            viewer.data().add_label(Eigen::Vector3d(eol_sim.q(0, i), eol_sim.q(1, i), 0), std::to_string(i));
                        viewer.data().show_custom_labels = true;
                    }
                }
                if (ImGui::Checkbox("ShowEulerianRest", &show_original))
                {   
                    updateScreen(viewer);
                }
            }
            if (ImGui::CollapsingHeader("ColorScheme", ImGuiTreeNodeFlags_DefaultOpen))
            {
                if (ImGui::Checkbox("ShowStretching", &showStretching))
                {
                    viewer.data().clear();
                    viewer.data().set_mesh(V, F);
                    if (showStretching)
                    {
                        eol_sim.getColorFromStretching(C);
                        viewer.data().set_colors(C);
                    }   
                }
                if (ImGui::Checkbox("PerYarn", &per_yarn))
                {
                    updateScreen(viewer);
                }   
            }
            if (ImGui::Button("Solve", ImVec2(-1,0)))
            {
                eol_sim.advanceOneStep();
                
                updateScreen(viewer);
            }
            if (ImGui::Button("Reset", ImVec2(-1,0)))
            {
                eol_sim.resetScene();
                updateScreen(viewer);
            }
        };
    }
    else if (test_current == DrawUnit)
    {
        viewer.plugins.push_back(&menu);
        menu.callback_draw_viewer_menu = [&]()
        {
            if (ImGui::CollapsingHeader("CurveIO", ImGuiTreeNodeFlags_DefaultOpen))
            {
                float w = ImGui::GetContentRegionAvailWidth();
                float p = ImGui::GetStyle().FramePadding.x;
                if (ImGui::Button("Load##Curve##Data##Points", ImVec2((w-p)/2.f, 0)))
                {
                    std::string fname = igl::file_dialog_open();
                    if (fname.length() != 0)
                    {
                        hybrid_curve.data_points.clear();
                        std::ifstream in(fname);
                        double x, y;
                        while(in >> x >> y)
                            hybrid_curve.data_points.push_back(Vector<T, 2>(x, y));
                        in.close();
                    }
                }
                ImGui::SameLine(0, p);
                if (ImGui::Button("Save##Curve##Data##Points", ImVec2((w-p)/2.f, 0)))
                {
                    std::string fname = igl::file_dialog_save();

                    if (fname.length() != 0)
                    {
                        std::ofstream out(fname);
                        for (auto pt : hybrid_curve.data_points)
                        {
                            out << pt.transpose() << std::endl;
                        }
                        out.close();
                    }
                }
            }

            if (ImGui::CollapsingHeader("Drawing Options", ImGuiTreeNodeFlags_DefaultOpen))
            {
                if (ImGui::DragInt("SubDivision", &(hybrid_curve.sub_div), 1.f, 8, 64))
                {
                    std::vector<Vector<T, 2>> points_on_curve;
                    hybrid_curve.getLinearSegments(points_on_curve);
                    appendCylinderMesh(viewer, V_drawing, F_drawing, points_on_curve);
                    updateScreen(viewer);
                }
                if (ImGui::Checkbox("Drawing", &drawing))
                {
                    editing = !drawing;
                }
                if (ImGui::Checkbox("Editing", &editing))
                {
                    drawing = !editing;
                }
                if (ImGui::Checkbox("ShowDataPoints", &show_data_points))
                {

                }
                float w = ImGui::GetContentRegionAvailWidth();
                float p = ImGui::GetStyle().FramePadding.x;
                if (ImGui::Button("Add##Curve", ImVec2((w-p)/2.f, 0)))
                {
                    hybrid_curves.push_back(hybrid_curve);
                    hybrid_curve = HybridC2Curve<T, 2>();
                }
                ImGui::SameLine(0, p);
                if (ImGui::Button("Remove##Curve", ImVec2((w-p)/2.f, 0)))
                {
                    hybrid_curves.pop_back();
                }
            }
        };
    }
    
    auto draw_unit_func = [&](igl::opengl::glfw::Viewer& viewer, int button, int)->bool
    {
        if (!drawing)
            return false;
        double x = viewer.current_mouse_x;
        double y = viewer.core().viewport(3) - viewer.current_mouse_y;
        Eigen::Vector4f eye_n = (viewer.core().view).inverse().col(3);
        // Eigen::Vector3d point;
        // igl::unproject_on_plane(Eigen::Vector2d(x,y), viewer.core().proj*viewer.core().view, viewer.core().viewport, eye_n, point);
        
        if (button == 0) // left button
        {
            hybrid_curve.data_points.push_back(Vector<T, 2>(x, y));
            std::vector<Vector<T, 2>> points_on_curve;
            // for(auto curve : hybrid_curves)
            // {
            //     std::vector<Vector<T, 2>> points;
            //     hybrid_curve.getLinearSegments(points);
            //     points_on_curve.insert(points_on_curve.end(), points.begin(), points.end());
            // }
            hybrid_curve.getLinearSegments(points_on_curve);
            appendCylinderMesh(viewer, V_drawing, F_drawing, points_on_curve);
            if (show_data_points)
            {
                for (auto pt : hybrid_curve.data_points)
                {
                    Eigen::Vector3d point;
                    igl::unproject_on_plane(pt, viewer.core().proj*viewer.core().view, viewer.core().viewport, eye_n, point);
                    appendSphereMesh(V_drawing, F_drawing, 0.05, point);
                }
            }
        }
        else if (button == 2) // right button
        {
            // removeSphereMesh(V_drawing, F_drawing);
            if(hybrid_curve.data_points.size())
            {
                hybrid_curve.data_points.pop_back();
                std::vector<Vector<T, 2>> points_on_curve;
                hybrid_curve.getLinearSegments(points_on_curve);
                appendCylinderMesh(viewer, V_drawing, F_drawing, points_on_curve, true);
                if (show_data_points)
                {
                    for (auto pt : hybrid_curve.data_points)
                    {
                        Eigen::Vector3d point;
                        igl::unproject_on_plane(pt, viewer.core().proj*viewer.core().view, viewer.core().viewport, eye_n, point);
                        appendSphereMesh(V_drawing, F_drawing, 0.05, point);
                    }
                }
            }
        }
        updateScreen(viewer);
        return false;
    };

    if (test_current == DrawUnit)
        viewer.callback_mouse_down = draw_unit_func;
    else
        viewer.callback_mouse_down = [&](igl::opengl::glfw::Viewer& viewer, int, int)->bool
        {
            double x = viewer.current_mouse_x;
            double y = viewer.core().viewport(3) - viewer.current_mouse_y;
            
            Eigen::MatrixXd pxy = eol_sim.q.transpose().block(0, 0, eol_sim.n_nodes, 2) / eol_sim.unit;
            Eigen::MatrixXd rod_v(eol_sim.n_nodes, 3);
            rod_v.setZero();
            rod_v.block(0, 0, eol_sim.n_nodes, 2) = pxy;

            igl::project(rod_v, viewer.core().view, viewer.core().proj, viewer.core().viewport, pxy);

            for(int i=0; i<pxy.rows(); ++i)
            {
                if(abs(pxy.row(i)[0]-x)<20 && abs(pxy.row(i)[1]-y)<20)
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
            eol_sim.q0 = eol_sim.q;
            // eol_sim.q0.transpose().block(0, 0, eol_sim.n_nodes, dim)  = eol_sim.q.transpose().block(0, 0, eol_sim.n_nodes, dim);
			return true;
		}
	    return false;
	  };

    if (test_current == DrawUnit)
    {
        
        viewer.callback_mouse_move =
        [&](igl::opengl::glfw::Viewer& viewer, int, int)->bool
        {
            double x = viewer.current_mouse_x;
            double y = viewer.core().viewport(3) - viewer.current_mouse_y;
        
            return false;
        };
    }
    else
        viewer.callback_mouse_move =
        [&](igl::opengl::glfw::Viewer& viewer, int, int)->bool
        {
            if(selected!=-1)
            {
                double x = viewer.current_mouse_x;
                double y = viewer.core().viewport(3) - viewer.current_mouse_y;
            
                double delta_x = (x - x0) / viewer.core().viewport(2);
                double delta_y = (y - y0) / viewer.core().viewport(3);
                // eol_sim.q(dim, 3) = u0;
                Eigen::VectorXd delta_dof(4); delta_dof.setZero();
                auto zero_delta = delta_dof;
                Eigen::VectorXd mask_dof(4); mask_dof.setZero();
                
                // delta_dof(0) = delta_x * eol_sim.unit;
                delta_dof(1) = delta_y * eol_sim.unit;
                
                // mask_dof(0) = 1;
                mask_dof(1) = 1;
                mask_dof(2) = 1;
                mask_dof(3) = 1;
                
                eol_sim.dirichlet_data[selected] = std::make_pair(delta_dof, mask_dof);
                eol_sim.advanceOneStep();
                updateScreen(viewer);
                std::cout << delta_x << " " << delta_y << std::endl;
                return true;
            }

            return false;
        };
    //================== Run GUI ==================
    
    

    int width = 800, height = 800;
    
    if (test_current == BatchRendering)
    {
        RGBMat R(width,height), G(width,height), B(width,height), A(width,height);

        T s = 1.1;
        int n_angles = 40;
        T cycle = 2.0 * M_PI;
        int cnt = 0;
        // viewer.core().camera_eye += Eigen::Vector3f(1, 1, 0);
        // std::cout << viewer.core().camera_eye.transpose() << std::endl;
        // for (T theta = 0; theta <= cycle; theta += cycle/(T)n_angles)
        T theta = strtod(argv[1], NULL);
        {
            eol_sim.buildPlanePeriodicBCScene3x3Subnodes();
            // viewer.data().set_mesh(V, F);
            viewer.data().set_face_based(true);
            viewer.data().clear();
            viewer.launch_init();
            // viewer.draw();
            per_yarn = false;
            Vector<T, dim> strain_dir, ortho_dir;
            eol_sim.setUniaxialStrain(theta, s, strain_dir, ortho_dir);
            eol_sim.advanceOneStep();
            eol_sim.buildMeshFromRodNetwork(V, F, eol_sim.q, eol_sim.rods, eol_sim.normal);
            
            viewer.data().clear();
            viewer.data().set_mesh(V, F);
            eol_sim.getColorPerYarn(C, n_rod_per_yarn);
            // eol_sim.getColorFromStretching(C);
            viewer.data().set_colors(C);   
            
            viewer.data().shininess = 1.0;
            
            // viewer.draw();
            
            viewer.core().draw_buffer(viewer.data(),true,R,G,B,A);
            // Save it to a PNG
            igl::png::writePNG(R,G,B,A,"output/strain_"+std::to_string(theta)+".png");
            
            eol_sim.resetScene();
            cnt++;
            viewer.launch_shut();
        }
        
    }
    else if (test_current == StaticSolve)
    {
        viewer.data().set_face_based(true);
        viewer.data().shininess = 1.0;
        viewer.data().point_size = 25.0;
        setupScene(viewer);
        viewer.callback_key_down = &key_down;
        viewer.core().align_camera_center(V);
        key_down(viewer,'0',0);
        viewer.launch();
    }
    else if (test_current == DrawUnit)
    {
        
        viewer.data().set_face_based(true);
        viewer.data().shininess = 1.0;
        // viewer.data().point_size = 25.0;
        setupScene(viewer);
        viewer.callback_key_down = &key_down;
        key_down(viewer,'0',0);
        // viewer.launch();
    }

    //================== Run Diff Test ==================
    // eol_sim.buildPlanePeriodicBCScene3x3();
    // homogenizer.initialize();
    // eol_sim.buildPlanePeriodicBCScene3x3();
    // eol_sim.runDerivativeTest();

    // eol_sim.resetScene();
    // homogenizer.computeYoungsModulusPoissonRatioBatch();
    // homogenizer.fitComplianceFullTensor();
    // homogenizer.fitComplianceTensor();
    return 0;
}