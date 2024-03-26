#include <igl/opengl/glfw/Viewer.h>
#include <igl/project.h>
#include <igl/unproject_on_plane.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui.h>
#include <igl/marching_cubes.h>
#include <igl/voxel_grid.h>
#include <igl/writeOBJ.h>

#include "../include/FEMSolver.h"
#include "../include/Contact2D.h"
#include "../include/Contact3D.h"
#include "../ImGui/implot.h"
#include "../ImGui/implot_internal.h"
#include "../include/vtk_io.h"

#define T double

using namespace std;

Eigen::MatrixXd V;
Eigen::MatrixXi F;
Eigen::MatrixXd C;

int num_time_step = 300;
bool static_solve = false;
float DISPLAYSMENT = -0.031;
int RES = 1;
int RES_2 = 3;
bool BILATERAL = 0;
bool use_NTS_AR = 0;
double FORCE = 0.1;
float WIDTH_2 = 0.5;
float SPRING_SCALE = 1.0;
int SPRING_BASE = 1e3;
bool USE_NONUNIFORM_MESH = 0;
int MODE = 0;
float GAP = 0.001;
bool USE_IMLS = 0;
bool IMLS_BOTH = 0;
bool USE_MORE_POINTS = 0;
bool USE_FROM_STL = 1;
bool USE_MORTAR_METHOD = 0;
bool USE_MORTAR_IMLS = 0;
bool TEST = 0;
int TEST_CASE = 0;
bool USE_VIRTUAL_NODE = 0;
bool CALCULATE_IMLS_PROJECTION = 0;
bool USE_IPC_3D = 1;
bool IMLS_3D_VIRTUAL = 0;
bool USE_TRUE_IPC_2D = 0;
bool USE_NEW_FORMULATION = 0;
bool SLIDING_TEST = 0;
int sliding_res = 10;
bool PULL_TEST = 0;
bool BARRIER_ENERGY = 1;
bool USE_DYNAMICS = 1;
bool USE_SHELL = 0;
bool USE_RIMLS = 0;
bool USE_FRICTION = 0;
VectorXa xz;

float k2 = 1000.;
float theta1 = 0.1;
float theta2 = 0.2;


double t = 0.0;
double h = 0.00002;
using TV = Vector<double, 3>;
using VectorXT = Matrix<double, Eigen::Dynamic, 1>;

Contact3D contact;
//Contact2D contact;

const int dim = 3;

int static_solve_step = 0;
static bool enable_selection = false;
static bool stress_graph = false;
float max_force_color = 0.;

bool compute_pressure = false;
bool display_IMLS = false;
bool show_upper = true;
int L2p = 0;
int IMLSp = 0;
float spring_length = 5;

std::vector<Eigen::MatrixXd> grid_points(2);
std::vector<Eigen::MatrixXd> grid_points_3d(2);
std::vector<Eigen::MatrixXd> grid_values(2);
std::vector<Eigen::MatrixXd> grid_colors(2);

// Customize the menu
double doubleVariable = 0.1f; // Shared between two menus

void ShowDemo_LinePlots() {
    static float xs1[1001], ys1[1001];
    for (int i = 0; i < 1001; ++i) {
        xs1[i] = i * 0.001f;
        ys1[i] = 0.5f + 0.5f * sinf(50 * (xs1[i] + (float)ImGui::GetTime() / 10));
    }
    static double xs2[11], ys2[11];
    for (int i = 0; i < 11; ++i) {
        xs2[i] = i * 0.1f;
        ys2[i] = xs2[i] * xs2[i];
    }
    ImGui::BulletText("Anti-aliasing can be enabled from the plot's context menu (see Help).");
    if (ImPlot::BeginPlot("Line Plot")) {
        ImPlot::SetupAxes("x","f(x)");
        ImPlot::PlotLine("sin(x)", xs1, ys1, 1001);
        ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle);
        ImPlot::PlotLine("x^2", xs2, ys2, 11);
        ImPlot::EndPlot();
    }
}


int main()
{
    Eigen::MatrixXd evectors;
    Eigen::VectorXd evalues;
    bool check_modes = false;
    int modes = 0;
    
    igl::opengl::glfw::Viewer viewer;

    igl::opengl::glfw::imgui::ImGuiPlugin plugin;
    igl::opengl::glfw::imgui::ImGuiMenu menu;

    viewer.plugins.push_back(&plugin);
    plugin.widgets.push_back(&menu);

    auto updateScreen = [&](igl::opengl::glfw::Viewer& viewer)
    {
        if(stress_graph)
            contact.solver.generateMeshForRenderingStress(V, F, C, max_force_color);
        else
            contact.solver.generateMeshForRendering(V, F, C);

        viewer.data().clear();

        // viewer.data().point_size = 10.0;
        // Vector3a pos = contact.solver.deformed.segment<3>(3*2434).transpose();
        // viewer.data().add_points(pos, Vector3a(1,0,0));

        if(display_IMLS)
        {
            //std::cout<<"true"<<std::endl;
            viewer.data().point_size = 5.0;
            if(show_upper)
                viewer.data().add_points(grid_points_3d[0], grid_colors[0]);
            else
                viewer.data().add_points(grid_points_3d[1], grid_colors[1]);
        }
        if(SLIDING_TEST)
        {
            // viewer.data().point_size = 10.0;
            // Eigen::MatrixXd P(2,2);
            // P(0,0) = contact.solver.deformed(2*(2*(contact.solver.sliding_res+1)));
            // P(0,1) = contact.solver.deformed(2*(2*(contact.solver.sliding_res+1))+1);
            // P(1,0) = contact.solver.r2*cos(contact.solver.theta2);
            // P(1,1) = -contact.solver.r2*sin(contact.solver.theta2);
            // Eigen::MatrixXd CP(2,3);
            // CP.setZero();
            // viewer.data().add_points(P,CP);
        }
        viewer.data().set_mesh(V, F);
        viewer.data().set_colors(C);
        igl::writeOBJ("current.obj",V,F);
        // std::cout<<V<<std::endl;
        // std::cout<<"___________________________________"<<std::endl;
        // std::cout<<F<<std::endl;
    };

    auto loadDisplacementVectors = [&](const std::string& filename)
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
    };

    menu.callback_draw_viewer_menu = [&]()
    {
        if (ImGui::CollapsingHeader("Visualization", ImGuiTreeNodeFlags_DefaultOpen))
        {
            if (ImGui::Checkbox("SelectVertex", &enable_selection))
            {
            }
            if (ImGui::Checkbox("Dynamics", &USE_DYNAMICS))
            {

            }
            if (ImGui::Checkbox("Display Stress", &stress_graph))
            {
                updateScreen(viewer);
            }
            if (ImGui::Button("Compute Pressure"))
            {
                contact.solver.computeContactPressure();
                //contact.solver.computeCauchyStress();
                //compute_pressure = true;
            }

            float bd = 1e-5;
            if(ImGui::InputFloat("barrier distance", &bd, 1e-6, 1e0))
            {
                contact.solver.barrier_distance = bd;
            }

            if(ImGui::InputFloat("Virtual Spring Length", &spring_length, 1, 1000))
            {
                contact.solver.virtual_spring_stiffness = spring_length;
                //std::cout<<"Virtual Spring Stiffness: "<<contact.solver.virtual_spring_stiffness<<std::endl;
            }

            if(ImGui::InputInt("Num Time Steps", &num_time_step, 1, 500))
            {
               
            }
            if(ImGui::InputInt("L2 Param", &L2p, -12, 12))
            {
                contact.solver.L2D_param = pow(10,L2p);
            }

            if(ImGui::InputInt("IMLS Param", &IMLSp, 0, 12))
            {
                contact.solver.IMLS_param = pow(10,IMLSp);
            }
            if(ImGui::InputInt("Sliding res", &sliding_res, 3, 20))
            {
                contact.solver.sliding_res = sliding_res;
            }


            if(ImGui::InputDouble("Max Force Applied", &FORCE, 0, 10000))
            {
                updateScreen(viewer);
            }
            if(ImGui::InputDouble("time step", &h, 1e-7, 1e0))
            {
                contact.solver.h = h;
                updateScreen(viewer);
            }
            
        }

        if (ImGui::Button("StaticSolve", ImVec2(-1,0)))
        {
            contact.solver.staticSolve();
            updateScreen(viewer);
        }
        if (ImGui::Button("DynamicSolve", ImVec2(-1,0)))
        {
            USE_DYNAMICS = true;
            contact.solver.sigma_r = 0.5;
            contact.solver.sigma_n = 1.5;
            contact.solver.h = 0.001;
            contact.solver.simulation_time = h*num_time_step;
            std::vector<std::string> attr_names = {};
			std::vector<VTK_ATTRIBUTE_TYPE> attr_types = {};
			std::vector<Eigen::VectorXd> attr_values;
			std::vector<std::vector<std::pair<int, int>>> attr_paths;

            contact.solver.barrier_distance  = 0.01;

            updateScreen(viewer);
            string file_name = "results/knot_iimls20_"+to_string(FORCE)+"current_"+to_string(0)+".vtk";
            WriteVTK(file_name, V, F, attr_names, attr_types, attr_values, attr_paths);

            xz = VectorXT(4);
            xz.setZero();
            //xz(0) = V.block(7786,0,contact.solver.num_nodes-7786,3).col(0).maxCoeff()+0.1;
            xz(0) = V.col(0).maxCoeff();
            // contact.solver.xz(1) = V.block(169,0,contact.solver.num_nodes-169,3).col(0).minCoeff();
            // contact.solver.xz(2) = V.block(169,0,contact.solver.num_nodes-169,3).col(2).maxCoeff();
            // contact.solver.xz(3) = V.block(169,0,contact.solver.num_nodes-169,3).col(2).minCoeff();
            // contact.solver.y_bar = V.block(169,0,contact.solver.num_nodes-169,3).col(1).maxCoeff();



			viewer.core().is_animating = true;
            ofstream file;
            file.open("results/knot_iimls20_timing_"+to_string(FORCE)+".txt");
            file<<"Number of Vertices: "<<contact.solver.deformed.size()/3<<std::endl;
            
            for(int i=0; i<num_time_step; ++i)
            {
                // if(i<8) contact.solver.addRotationalDirichletBC((i)*5*M_PI/180);
                // else contact.solver.addRotationalDirichletBC((7)*20*M_PI/180+(i-7)*5*M_PI/180);
                //contact.solver.addRotationalDirichletBC((i)*10*M_PI/180);
                // if( i<=50)
                //     contact.solver.addRotationalDirichletBC((i)/20.);
                // else
                //     contact.solver.addRotationalDirichletBC(2.5);
                contact.solver.addRotationalDirichletBC((i)/20.);
                //contact.solver.y_bar-=0.05;
                // if(i<=100)
                //     contact.solver.addRotationalDirichletBC(i/30.);
                // else
                //     contact.solver.addRotationalDirichletBC(10./3.+(i-100)/60.);
                std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

                contact.solver.staticSolve();

                std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
                std::cout << "Time difference (elastic hessian) = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
                file << "Num of iteration = " << contact.solver.num_cnt << std::endl;
                file << "Time difference (elastic hessian) = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
                T energy;
                if(USE_RIMLS) contact.solver.addFastRIMLSSCEnergy(energy);
                else contact.solver.addFastIMLSSCEnergy(energy);
                file << "Num of contact = " << contact.solver.dist_info.size()<< std::endl;
                // bool finished = false;
                // static_solve_step = 0;
                // while(!finished)
                // {
                //     finished = contact.solver.staticSolveStep(static_solve_step);
                //     static_solve_step++;
                //     updateScreen(viewer);
                // }
                contact.solver.v_prev = (contact.solver.deformed-contact.solver.x_prev)/contact.solver.h;
                contact.solver.x_prev = contact.solver.deformed;
                if(USE_FRICTION)
                {
                    contact.solver.prev_contact_force.setZero();
                    contact.solver.addFastRIMLSSCForceEntries(contact.solver.prev_contact_force);
                }
                

                updateScreen(viewer);
                string file_name = "results/knot_iimls20"+to_string(FORCE)+"current_"+to_string(i+1)+".vtk";
                string file_name2 = "results/knot_iimls20"+to_string(FORCE)+"current_"+to_string(i+1)+".obj";
                WriteVTK(file_name, V, F, attr_names, attr_types, attr_values, attr_paths);
                igl::writeOBJ(file_name2,V,F);
                // contact.solver.undeformed = contact.solver.deformed;
                // contact.solver.u.setZero();
            }
            file.close();
            viewer.core().is_animating = false;
            std::cout<<contact.solver.v_prev.norm()<<std::endl;
            
            
        }
        if (ImGui::Button("BoundaryInfo", ImVec2(-1,0)))
        {
            contact.solver.displayBoundaryInfo();
            updateScreen(viewer);
        }
        if (ImGui::Button("ShowLeftRight", ImVec2(-1,0)))
        {
            contact.solver.showLeftRight();
            updateScreen(viewer);
        }
        if (ImGui::Button("Reset", ImVec2(-1,0)))
        {
           
            if(!SLIDING_TEST && !PULL_TEST)
            {
                // if(MODE == 3)
                // {
                //     contact.solver.use_pos_penalty = true;
                //     contact.solver.use_virtual_spring = true;
                // }else{
                //     contact.solver.use_pos_penalty = true;
                //     contact.solver.use_virtual_spring = false;
                // }
                // if(!TEST)
                //     contact.solver.use_pos_penalty = false;
                // //contact.initializeSimulationDataFromSTL();
                // if(USE_FROM_STL)
                //     contact.initializeSimulationDataFromSTL();
                // else
                //     contact.initializeSimulationData();
                // contact.initializeSimulationDataBunnyFunnel();

                contact.search_radius = 0.1;
                contact.solver.radius = 0.02;
                contact.solver.friction_mu = 0.2;
                contact.solver.friction_epsilon = 1e-5;
                //contact.mesh_names = {{"../../../data/test7.obj",2}};
                //contact.mesh_names = {{"../../../data/square_32_32.obj",2}};
                //contact.mesh_names = {{"iter_186.obj",2}};
                
                //contact.mesh_names = {{"../../../data/square_32_32.obj",2},{"../../../data/spring4.obj",2}};
                //contact.mesh_names = {{"../../../data/knot2-1.obj",3}};
               // contact.mesh_names = {{"../../../data/mat100x100t40.mesh",3}};
                //contact.mesh_names = {{"../../../data/rod.mesh",3},{"../../../data/rod.mesh",3},{"../../../data/rod.mesh",3},{"../../../data/rod.mesh",3}};
                //contact.mesh_names = {{"../../../data/knot4.2-hi.obj",3}};
                //contact.mesh_names = {{"../../../data/dolphin5K.obj",3},{"../../../data/armedillo.obj",3}};
                //contact.mesh_names = {{"../../../data/cliff_hi.obj",2},{"../../../data/cube_hi.mesh",3}};
                //contact.mesh_names = {{"../../../data/dolphin_new_new3.obj",3}};
                contact.mesh_names = {{"../../../data/square64x64.obj",2},{"../../../data/torus.obj",3}};
                contact.initializeSimulationSelfContact();

                //if(!contact.solver.use_virtual_spring)
                if(USE_FROM_STL)
                {
                    // contact.solver.addNeumannBCFromSTL();
                    //  contact.solver.addDirichletBCFromSTL();
                    contact.solver.addRotationalDirichletBC(1);
                }
                else
                {
                    contact.solver.addNeumannBC();
                    contact.solver.addDirichletBC();
                }
            }
            else if(SLIDING_TEST)
            {
                contact.initializeSimulationData();
                contact.solver.use_elasticity = false;
                contact.solver.addDirichletBC();
            }
            else if(PULL_TEST)
            {
                contact.initializeSimulationDataFromSTL();
                contact.solver.addNeumannBC();
                contact.solver.addDirichletBC();
            }
            
            

            // if(TEST)
            // {
            //     contact.solver.use_pos_penalty = true;
            //     contact.solver.use_rotational_penalty = false;
            // }
            // if(USE_IMLS)
            //     contact.solver.findProjectionIMLS(contact.solver.ipc_vertices,true,false);
            // contact.solver.checkTotalGradientScale(1);
            // contact.solver.checkTotalGradient(1);
            
            
            // contact.solver.checkTotalGradient(1);
            // contact.solver.checkTotalHessian(1);

            // contact.solver.checkTotalGradientScale(1);
            // contact.solver.checkTotalHessianScale(1);

            static_solve_step = 0;
            compute_pressure = false;
            contact.solver.k1 = SPRING_BASE;
            contact.solver.k2  = SPRING_SCALE*contact.solver.k1;
            contact.solver.k2  = k2;
            //std::cout<<contact.solver.k2<<std::endl;

            check_modes = 0;
            viewer.core().is_animating = false;

            updateScreen(viewer);
        }
        if (ImGui::Button("Test Inverse Derivative", ImVec2(-1,0)))
        {
            //contact.solver.checkDerivative();
            double tar = 0.2001*M_PI;
            double dummy;
            contact.solver.InverseDesign(tar,dummy);
        }
        if (ImGui::Button("SaveMesh", ImVec2(-1,0)))
        {
            igl::writeOBJ("current.obj", V, F);
        }
        if (ImGui::Button("Refresh", ImVec2(-1,0)))
        {
            updateScreen(viewer);
        }
        if (ImGui::Button("Check Gradient/Hessian", ImVec2(-1,0)))
        {
            contact.solver.checkTotalGradientScale(1);
            contact.solver.checkTotalHessianScale(1);
        }
        if (ImGui::Button("Check Gradient/Hessian analytical", ImVec2(-1,0)))
        {
            //contact.solver.checkTotalGradient(1);
            contact.solver.checkTotalHessian(1);
        }
        if (ImGui::Button("Compute Center", ImVec2(-1,0)))
        {
            contact.solver.computeUpperCenter();
        }
        if (ImGui::Button("Master Y Var", ImVec2(-1,0)))
        {
            contact.solver.checkMasterVariance();
        }
        
        
        if (ImGui::Button("Compute Grid", ImVec2(-1,0)))
        {
            // double width = WIDTH_2+0.2;
            // double height = HEIGHT_2+0.2;

            // double width_2 = WIDTH_2+0.2;
            // double height_2 = HEIGHT_2/3.+0.2;

            // if(USE_FROM_STL)
            // {
            //     width = 1.4;
            //     height = 0.5;

            //     width_2 = 1.4;
            //     height_2 = 0.5;
            // }
            // else if(TEST)
            // {
            //     width = 13;
            //     height = 5;
            // }

            // contact.solver.computeSlaveCenter();

            // int x_num = 800; 
            // int y_num = 400;
            // int x_num_2 = 800;
            // int y_num_2 = 400;
            // grid_points[0].resize(x_num*y_num, 2);
            // grid_points[1].resize(x_num_2*y_num_2, 2);

            // std::cout<<contact.solver.slave_center<<std::endl;

            // for(int i=0; i<x_num; ++i)
            // {
            //     for(int j=0; j<y_num; ++j)
            //     {
            //         if(USE_FROM_STL)
            //         {
            //             // grid_points[0](i*y_num+j,0) = contact.solver.slave_center(0)-0.5-0.1+width*(i)/(double(x_num));
            //             // grid_points[0](i*y_num+j,1) = contact.solver.slave_center(1)-0.5-0.1+1e-3+height*(j)/(double(y_num));

            //             grid_points[0](i*y_num+j,0) = -0.7+width*(i)/(double(x_num));
            //             grid_points[0](i*y_num+j,1) = 0.9+height*(j)/(double(y_num));

            //             if(TEST_CASE == 2) 
            //                 grid_points[0](i*y_num+j,1) = -0.25+height*(j)/(double(y_num));
            //         }
            //         else if(TEST)
            //         {
            //             grid_points[0](i*y_num+j,0) = -0.5+width*(i)/(double(x_num));
            //             grid_points[0](i*y_num+j,1) = -0.5+height*(j)/(double(y_num));
            //         }
            //         else if(SLIDING_TEST)
            //         {
            //             grid_points[0](i*y_num+j,0) = -0.5+2.*(i)/(double(x_num));
            //             grid_points[0](i*y_num+j,1) = -1.5+2.*(j)/(double(y_num));
            //         }
            //         else if(PULL_TEST)
            //         {
            //             grid_points[0](i*y_num+j,0) = 5.6+5.*(i)/(double(x_num));
            //             grid_points[0](i*y_num+j,1) = -2-5.*(j)/(double(y_num));
            //         }
            //         else
            //         {
            //             grid_points[0](i*y_num+j,0) = contact.solver.slave_center(0)-WIDTH_2/2.-0.1+width*(i)/(double(x_num));
            //             grid_points[0](i*y_num+j,1) = contact.solver.slave_center(1)-1./6.*height+1e-3+height*(j)/(double(y_num));
            //         }
                    
            //     }
            // }

            // for(int i=0; i<x_num_2; ++i)
            // {
            //     for(int j=0; j<y_num_2; ++j)
            //     {
            //         if(USE_FROM_STL)
            //         {
            //             grid_points[1](i*y_num_2+j,0) = -0.7+width*(i)/(double(x_num));
            //             grid_points[1](i*y_num_2+j,1) =  0.9+height*(j)/(double(y_num));

            //             if(TEST_CASE == 2) 
            //                 grid_points[1](i*y_num+j,1) = -0.25+height*(j)/(double(y_num));
            //         }
            //         else if(TEST)
            //         {
            //             grid_points[1](i*y_num_2+j,0) = -0.5+width*(i)/(double(x_num_2));
            //             grid_points[1](i*y_num_2+j,1) = -0.5+height*(j)/(double(y_num_2));
            //         }
            //         else
            //         {
            //             grid_points[1](i*y_num_2+j,0) = contact.solver.slave_center(0)-WIDTH_2/2.-0.1+width*(i)/(double(x_num));
            //             grid_points[1](i*y_num_2+j,1) = contact.solver.slave_center(1)-1./6.*height+1e-3+height*(j)/(double(y_num));
            //         }
                    
            //     }
            // }

            // compute values
            std::cout<<"Creating grid..."<<std::endl;
            // number of vertices on the largest side
            const int s = 100;
            // create grid
            MatrixXd GV;
            Eigen::RowVector3i res;
            Eigen:MatrixXd used_V = V.block(0,0,7786,3);
            igl::voxel_grid(used_V,0,s,1,GV,res);
            std::cout<<"Computing distances... "<<GV.rows()<<" "<<GV.col(0).maxCoeff()<<" "<<GV.col(1).maxCoeff()<<" "<<GV.col(2).maxCoeff()<<std::endl;
            Eigen::VectorXd S(GV.rows()),B;
            
            
            // contact.solver.testIMLS(GV,0);
            // S = contact.solver.result_values[0];
            
            std::unordered_map<int,int> hm;
            contact.solver.num_nodes = 7786+GV.rows();
            contact.solver.deformed.resize(3*((7786+GV.rows())));
            for(int i=0; i<3*7786; ++i) contact.solver.deformed(i) = contact.solver.undeformed(i);
            for(int i=0; i<GV.rows(); ++i)
            {
                //std::cout<<GV.row(i)<<std::endl;
                hm[i+7786] = i;
                contact.solver.deformed.segment<3>(3*(7786+i)) = GV.row(i);
                S(i) = 1e8;
            }
            std::cout<<"Marching cubes... "<<std::endl;
            //contact.solver.slave_nodes_3d[0] = hm;
            
            T energy;
            std::cout<<"Marching cubes..."<<std::endl;
            contact.solver.sigma_r = 0.5;
            contact.solver.sigma_n = 1.5;
            contact.solver.radius = 0.25;
            contact.solver.addFastRIMLSSCTestEnergy(energy);
            std::cout<<"Marching cubes... "<<contact.solver.dist_info.size()<<std::endl;
            for(int i=0; i<contact.solver.dist_info.size(); ++i)
            {
                S(contact.solver.dist_info[i].first-7786) = contact.solver.dist_info[i].second;
            }
            
            std::cout<<"Marching cubes..."<<std::endl;
            MatrixXd SV,BV;
            MatrixXi SF,BF;
            igl::marching_cubes(S,GV,res(0),res(1),res(2),0,SV,SF);
            igl::writeOBJ("marching_cube.obj",SV,SF);


            
            // if(IMLS_BOTH)
            // {
            //     contact.solver.testIMLS(grid_points[0],1);
            //     grid_values[0] = contact.solver.result_values[1];
            //     contact.solver.testIMLS(grid_points[1],0);
            //     grid_values[1] = contact.solver.result_values[0];
            // }
            // else
            // {
            //     contact.solver.testIMLS(grid_points[0],0);
            //     grid_values[0] = contact.solver.result_values[0];
            // }
                

            // int num_boundary = 0;
            // std::vector<int> indices;
            // for(int i=0; i<grid_points[0].rows(); ++i)
            // {
            //     double value = grid_values[0](i);
            //     if(true)
            //     {
            //         num_boundary++;
            //         indices.push_back(i);
            //     }
            // }

            // grid_colors[0].setZero(num_boundary, 3);
            // grid_points_3d[0].setZero(num_boundary, 3);

            // for (int k = 0; k < indices.size(); ++k) {
            //     int i = indices[k];
            //     double value = grid_values[0](i);
            //     if(fabs(value) == 1e8)
            //     {
            //         grid_colors[0](i, 2) = 0;
            //     }else if (fabs(value) < 1e-3) {
            //         grid_colors[0](i, 1) = 1;
            //     }
            //     else {
            //         if (value > 0)
            //             grid_colors[0](i, 0) = 1;
            //         else
            //             grid_colors[0](i, 2) = 1;
            //     }
            //     // grid_colors(k,0) = 1;
            //     grid_points_3d[0](k,0) = grid_points[0](i,0);
            //     grid_points_3d[0](k,1) = grid_points[0](i,1);
            // }

            // if(IMLS_BOTH)
            // {
            //     num_boundary = 0;
            //     indices.clear();
            //     for(int i=0; i<grid_points[1].rows(); ++i)
            //     {
            //         double value = grid_values[1](i);
            //         if(true)
            //         {
            //             num_boundary++;
            //             indices.push_back(i);
            //         }
            //     }

            //     grid_colors[1].setZero(num_boundary, 3);
            //     grid_points_3d[1].setZero(num_boundary, 3);

            //     for (int k = 0; k < indices.size(); ++k) {
            //         int i = indices[k];
            //         double value = grid_values[1](i);
            //         if(fabs(value) == 1e8)
            //         {
            //             grid_colors[1](i, 2) = 0;
            //         }else if (fabs(value) < 1e-3) {
            //             grid_colors[1](i, 1) = 1;
            //             grid_colors[1](i, 1) = 1;
            //             grid_colors[1](i, 2) = 1;
            //         }
            //         else {
            //             if (value > 0)
            //             {
            //                 grid_colors[1](i, 0) = 1;
            //                 grid_colors[1](i, 1) = 1;
            //             }   
            //             else
            //             {
            //                 grid_colors[1](i, 1) = 1;
            //                 grid_colors[1](i, 2) = 1;
            //             }
                           
            //         }
            //         // grid_colors(k,0) = 1;
            //         grid_points_3d[1](k,0) = grid_points[1](i,0);
            //         grid_points_3d[1](k,1) = grid_points[1](i,1);
            //     }
            // }
        }
        if(ImGui::InputFloat("Displacement", &DISPLAYSMENT, 0.0f,  WIDTH_1-WIDTH_2, "%.5f"))
        {
        }
        if(ImGui::InputFloat("Width 2", &WIDTH_2, 0.0f,  4.0f, "%.3f"))
        {
        }
        if(ImGui::InputFloat("GAP", &GAP, -1.0f,  1.0f, "%.3f"))
        {
        }
        if(ImGui::InputFloat("Spring Scale", &SPRING_SCALE, 0.001f,  1000.f, "%.3f"))
        {
        }
        if(ImGui::InputInt("Spring Base", &SPRING_BASE, 1,  1e5))
        {
        }
        if(ImGui::InputFloat("k2", &k2, 1.0,  1e5))
        {
        }
        if(ImGui::InputFloat("theta1", &theta1, 0,  0.5))
        {
            contact.solver.theta1 = theta1*M_PI;
        }
        if(ImGui::InputFloat("theta2", &theta2, 0,  0.5))
        {
            contact.solver.theta2 = theta2*M_PI;
        }
        if(ImGui::InputInt("MODE", &MODE, 0,  3))
        {
        }
        if(ImGui::SliderInt("Resolution", &RES, 1,  16))
        {
        }
        if(ImGui::SliderInt("Test Case", &TEST_CASE, 0,  4))
        {
        }
        if(ImGui::Checkbox("Show Upper", &show_upper))
        {

        }
        if(ImGui::Checkbox("Bilateral", &BILATERAL))
        {

        }
        if(ImGui::Checkbox("Display IMLS", &display_IMLS))
        {

        }
        if(ImGui::Checkbox("Area Regularization", &use_NTS_AR))
        {

        }
        if(ImGui::Checkbox("Non uniform mesh", &USE_NONUNIFORM_MESH))
        {

        }
        if(ImGui::Checkbox("Rotational Penalty", &contact.solver.use_rotational_penalty))
        {

        }
        if(ImGui::Checkbox("use imls", &USE_IMLS))
        {

        }
        if(ImGui::Checkbox("Bilateral IMLS", &IMLS_BOTH))
        {

        }
        if(ImGui::Checkbox("Improved IMLS both", &USE_MORE_POINTS))
        {

        }
        if(ImGui::Checkbox("Imported Mesh", &USE_FROM_STL))
        {

        }
        if(ImGui::Checkbox("Mortar Method", &USE_MORTAR_METHOD))
        {

        }
        if(ImGui::Checkbox("USE MC integration", &USE_MORTAR_IMLS))
        {
            contact.solver.mortar.use_mortar_IMLS = USE_MORTAR_IMLS;
        }
        if(ImGui::Checkbox("USE Virtual Node", &USE_VIRTUAL_NODE))
        {
            
        }
        if(ImGui::Checkbox("TEST", &TEST))
        {

        }
        if(ImGui::Checkbox("USE IMLS PROJECTION", &CALCULATE_IMLS_PROJECTION))
        {

        }
        if(ImGui::Checkbox("USE IPC 3D", &USE_IPC_3D))
        {

        }

        if(ImGui::Checkbox("USE IPC 2D", &USE_TRUE_IPC_2D))
        {

        }

        if(ImGui::Checkbox("USE ROBUST IMLS", &USE_RIMLS))
        {

        }

        if(ImGui::Checkbox("USE NEW FORMULATION", &USE_NEW_FORMULATION))
        {

        }
        if(ImGui::Checkbox("Sliding Test", &SLIDING_TEST))
        {

        }
        
    };
    
    viewer.callback_post_draw = [&](igl::opengl::glfw::Viewer &) -> bool
    {
        //USE_DYNAMICS = false;
        if(viewer.core().is_animating && !check_modes)
        {
            bool finished = contact.solver.staticSolveStep(static_solve_step);
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
        case ' ':
            viewer.core().is_animating = true;
            updateScreen(viewer);
            return true;
        case '1':
            check_modes = true;
            contact.solver.checkDeformationGradient();
            contact.solver.checkHessianPD(true);
            loadDisplacementVectors("cell_eigen_vectors_2d.txt");
            // std::cout << "modes " << modes << " singular value: " << evalues(modes) << std::endl;
            return true;
        case '2':
            check_modes = false;
            return true;
        case 'a':
            viewer.core().is_animating = !viewer.core().is_animating;
            return true;
        }
    };

    viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer &) -> bool
    {
        if(viewer.core().is_animating && check_modes)
        {
            contact.solver.deformed = contact.solver.undeformed + contact.solver.u + 0.1 * evectors.col(modes) * std::sin(t);
            t += 0.05;
            updateScreen(viewer);
        }
        return false;
    };

    viewer.callback_mouse_down = [&](igl::opengl::glfw::Viewer& viewer, int, int)->bool
    {
        if (!enable_selection)
            return false;
        double x = viewer.current_mouse_x;
        double y = viewer.core().viewport(3) - viewer.current_mouse_y;

        for (int i = 0; i < contact.solver.num_nodes; i++)
        {
            Vector<T, 3> pos;
            pos(0) = contact.solver.deformed(i * dim);
            pos(1) = contact.solver.deformed(i * dim + 1);
            if constexpr (dim == 3)
                pos(2) = contact.solver.deformed(i * dim + 2);
            else pos(2) = 0.0;

            Eigen::MatrixXd x3d(1, 3); x3d.setZero();
            x3d.row(0).segment<3>(0) = pos;

            Eigen::MatrixXd pxy(1, 3);
            igl::project(x3d, viewer.core().view, viewer.core().proj, viewer.core().viewport, pxy);

            if(abs(pxy.row(0)[0]-x)<20 && abs(pxy.row(0)[1]-y)<20)
            {
                std::cout << "selected " << i << std::endl;
                for(int j=0; j<dim; ++j)
                {
                    std::cout<<contact.solver.deformed(dim*i+j)<<" ";
                }
                std::cout<<std::endl;
                
                //return true;
            }
        }
        return false;
    };

     menu.callback_draw_custom_window = [&]()
    {
        // Define next window position + size
        ImGui::SetNextWindowPos(ImVec2(180.f * menu.menu_scaling(), 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(200, 160), ImGuiCond_FirstUseEver);
        ImGui::Begin(
            "Plot of Pressure", nullptr,
            ImGuiWindowFlags_NoSavedSettings
        );

        // Expose the same variable directly ...

        if(compute_pressure)
        {
            int contact_size = contact.solver.ContactPressure.size();
            double xs2[contact_size], ys2[contact_size];
            double xs1[contact_size], ys1[contact_size];
            for (int i = 0; i < contact_size; ++i) {
                xs2[i] = double(i)/double(contact_size-1)*1.0f;
                ys2[i] = contact.solver.ContactPressure(i);
            }
            for (int i = 0; i < contact_size; ++i) {
                xs1[i] = double(i)/double(contact_size-1)*1.0f;
                ys1[i] = contact.solver.ContactTraction(i);
            }

            int contact_size_2 = contact.solver.ContactPenetration.size();
            double xs[contact_size_2], ys[contact_size_2];
            for (int i = 0; i < contact_size_2; ++i) {
                xs[i] = double(i)/double(contact_size_2-1)*1.0f;
                ys[i] = contact.solver.ContactPenetration(i);
            }
            
            if(dim == 2)
            {
                //double max_y = std::max(1.2*contact.solver.ContactPressure.maxCoeff(),1.2*contact.solver.ContactTraction.maxCoeff());
                double max_y = 1.2*contact.solver.ContactTraction.maxCoeff();
                double max_y_2 = 1.2*contact.solver.ContactPenetration.maxCoeff();
                ImPlot::CreateContext();

                ImGui::BulletText("Penetration on the Contact Surface");
                if (ImPlot::BeginPlot("Penetration Plot")) {
                    ImPlot::SetupAxes("x","f(x)");
                    ImPlot::SetupAxesLimits(0.0,1.0,0.0,max_y_2);
                    ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle);
                    ImPlot::PlotLine("Penetration at Contact", xs, ys, contact_size_2);
                    ImPlot::EndPlot();
                }
                ImPlot::DestroyContext();

                ImPlot::CreateContext();

                ImGui::BulletText("Pressure on the Contact Surface");
                if (ImPlot::BeginPlot("Pressure Plot")) {
                    ImPlot::SetupAxes("x","f(x)");
                    ImPlot::SetupAxesLimits(0.0,1.0,0.0,max_y);
                    ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle);
                    //ImPlot::PlotLine("Pressure at Contact", xs2, ys2, contact_size);
                    ImPlot::PlotLine("Normal Stress at Contact", xs1, ys1, contact_size);
                    ImPlot::EndPlot();
                }
                ImPlot::DestroyContext();
                }
            
        }
    
        ImGui::End();
    };


    //contact.solver.IMLSProjectionTest();
    

    if(SLIDING_TEST) {
        contact.solver.sample_res = 8;
        contact.solver.sliding_res = 5;
        sliding_res = 5;
    }
    if(PULL_TEST) {
        contact.solver.sample_res = 3;
        // contact.solver.sliding_res = 5;
        // sliding_res = 5;
    }
    //contact.initializeSimulationDataFromSTL();
    contact.solver.spring_length = 1000;
    contact.solver.add_bending = true;
    contact.solver.add_stretching = true;
    contact.solver.bar_param = 1e5;
    contact.solver.y_bar = 2.;
    xz = Eigen::VectorXd(4);
    xz.setZero();
    xz(0) = 3;

    //contact.solver.use_elasticity = false;
    //contact.initializeSimulationDataBunnyFunnel();
    contact.solver.radius = 0.05;
    contact.search_radius = 0.15;
    //contact.mesh_names = {{"../../../data/square_32_32.obj",2},{"../../../data/spring4.obj",2}};
    contact.mesh_names = {{"../../../data/rod.obj",2},{"../../../data/rod.obj",2}};
    //contact.mesh_names = {{"../../../data/test7.obj",2}};
    contact.initializeSimulationSelfContact();

    // if(USE_FROM_STL || PULL_TEST)
    //     contact.initializeSimulationDataFromSTL();
    // else
    //     contact.initializeSimulationData();

    contact.solver.rou = 1.0e-4;
    contact.solver.newton_tol = 1e-5;
    
    contact.solver.barrier_distance  = 0.001;
    contact.solver.IMLS_param = 1e-1;
    contact.solver.max_newton_iter = 2000;

    std::cout<<xz.size()<<std::endl;

    VectorXT u = contact.solver.u;
    //contact.solver.addL2CorrectForceEntries(u);
    
    //contact.solver.checkTotalHessianScale(1);

    // if(!contact.solver.use_virtual_spring)
    // if(USE_FROM_STL)
    // {
    //     contact.solver.addNeumannBCFromSTL();
    //     contact.solver.addDirichletBCFromSTL();
    // }
    // else
    // {
    //     contact.solver.addNeumannBC();
    //     contact.solver.addDirichletBC();
    // }
    //contact.solver.mortar.MortarMethod();

    // if(USE_IMLS)
    //     contact.solver.findProjectionIMLS(contact.solver.ipc_vertices,true,false);
    // contact.solver.checkTotalHessian(1);
    // contact.solver.checkTotalGradientScale(1);
    // contact.solver.checkTotalGradient(1);
    // contact.solver.checkTotalGradientScale(1);
    // contact.solver.checkTotalHessianScale(1);

    //contact.solver.testDerivativeIMLS();

    // contact.solver.addDirichletBC();
    // contact.solver.use_rotational_penalty = true;
    // double scale = 0.01/sliding_res;
    // double a = 1;
    // double b = 1000;
    // double mid = (a+b)/2;
    // while((b-a)>1e-2)
    // {
    //     mid = (a+b)/2;
    //    contact.solver.virtual_spring_stiffness = mid/100.;

    //     contact.initializeSimulationData();
    //     contact.solver.addDirichletBC();
    //     contact.solver.addNeumannBC();
    //     static_solve_step = 0;

    //     while(true)
    //     {
    //         bool finished = contact.solver.staticSolveStep(static_solve_step);
    //         if (finished)
    //         {
    //             break;
    //         }
    //         else 
    //             static_solve_step++;
    //     }
    //     Eigen::VectorXd p = contact.solver.deformed.segment<dim>(dim*(15+RES_2));
    //     std::ofstream file( "IPC_TEST1.csv", std::ios::app );
    //     std::cout<<" "<<mid<<" "<<p.transpose()<<" "<<std::endl;
    //     file<<mid<<","<<p.transpose()<<","<<contact.solver.barrier_weight<<std::endl;
    //     if(p(0) > 5)
    //     {
    //         b = mid;
    //     }else{
    //         a = mid;
    //     }

    // }
    // std::cout<<"FINAL VALUE: "<<mid<<" Barrier Weight"<<contact.solver.barrier_weight<<std::endl;

    // double scale = 0.00001;
    // for(double i=1.3; i<=1.7; i += scale)
    // double i = 1.3;
    // {
    //     // if(MODE == 3)
    //     // {
    //     //     contact.solver.use_pos_penalty = false;
    //     //     contact.solver.use_virtual_spring = true;
    //     // }else{
    //     //     contact.solver.use_pos_penalty = true;
    //     //     contact.solver.use_virtual_spring = false;
    //     // }

    //     //DISPLAYSMENT = WIDTH_1*i/(SPRING_BASE+i);
    //     DISPLAYSMENT = i;
    //     //USE_MORTAR_METHOD = 0;

    //     if(MODE == 3)
    //     {
    //         contact.solver.use_pos_penalty = true;
    //         contact.solver.use_virtual_spring = true;
    //     }else{
    //         contact.solver.use_pos_penalty = true;
    //         contact.solver.use_virtual_spring = false;
    //     }
    //         contact.initializeSimulationData();

    //         //if(!contact.solver.use_virtual_spring)
    //         contact.solver.addNeumannBC();
    //         contact.solver.addDirichletBC();
    //         //contact.solver.checkTotalGradient(1);
    //         //contact.solver.checkTotalGradientScale(1);
    //         //contact.solver.checkTotalHessianScale(1);
    //         //contact.solver.checkTotalHessian(1);

    //         static_solve_step = 0;
    //         compute_pressure = false;
    //         contact.solver.k1 = SPRING_BASE;
    //         contact.solver.k2  = SPRING_SCALE*contact.solver.k1;
    //         contact.solver.k2  = i;
    //         //std::cout<<contact.solver.k2<<std::endl;

    //         check_modes = 0;
    //         viewer.core().is_animating = false;
    //         updateScreen(viewer);
        
    //     double c1 = contact.solver.computeUpperCenter();

    //     while(true)
    //     {
    //         bool finished = contact.solver.staticSolveStep(static_solve_step);
    //         if (finished)
    //         {
    //             break;
    //         }
    //         else 
    //             static_solve_step++;
    //     }
       
    //     //contact.solver.showLeftRight();
    //     //contact.solver.staticSolve();
    //     //contact.solver.showLeftRight();
    //     // double c2 = contact.solver.computeUpperCenter();
    //     // std::ofstream file( "center_y.csv", std::ios::app ) ;
    //     // file<<DISPLAYSMENT<<","<<c1<<","<<c2<<std::endl;

    //     //double var = contact.solver.checkMasterVariance();

    //     double c2 = contact.solver.computeUpperCenter();
    //     double ipc_e = 0;
    //     contact.solver.addIPC2DtrueEnergy(ipc_e);
    //     std::ofstream file( "upper_rest_x_IPC_all.csv", std::ios::app ) ;
    //     //file<<DISPLAYSMENT+WIDTH_2/2.0<<","<<c2<<std::endl;
    //     file<<DISPLAYSMENT<<","<<contact.solver.deformed(2*16)<<","<<ipc_e<<std::endl;

    //     // updateScreen(viewer);
    //     // viewer.core().background_color.setOnes();
    //     // viewer.data().set_face_based(true);
    //     // viewer.data().shininess = 1.0;
    //     // viewer.data().point_size = 5.0;

    //     // viewer.core().align_camera_center(V);
    //     // viewer.core().animation_max_fps = 24.;
    //     // // key_down(viewer,'0',0);
    //     // viewer.core().is_animating = false;
    //     // viewer.launch();
        
    // }



    // double eps = 1e-5;
    // contact.solver.k1  = 500;
    // if(MODE == 3)
    // {
    //     contact.solver.use_pos_penalty = false;
    //     contact.solver.use_virtual_spring = true;
    // }else{
    //     contact.solver.use_pos_penalty = true;
    //     contact.solver.use_virtual_spring = false;
    // }

    // contact.solver.ForwardSim();
    // double r1 = contact.solver.R;
    // double d = contact.solver.dRdp(0);

    // contact.initializeSimulationData();

    // contact.solver.addNeumannBC();
    // contact.solver.addDirichletBC();

    // static_solve_step = 0;
    // compute_pressure = false;
    // contact.solver.k1  += eps;

    // contact.solver.ForwardSim();
    // double r2 = contact.solver.R;
    // std::cout<<(r2-r1)/eps<<" "<<d<<std::endl;

    //contact.solver.checkDerivative();

    // if(display_IMLS)
    // {
    //     double width = WIDTH_1+0.2;
    //     double height = HEIGHT_1+0.2;

    //     int x_num = 200;
    //     int y_num = 200;
    //     grid_points[0].resize(x_num*y_num, 2);

    //     for(int i=0; i<x_num; ++i)
    //     {
    //         for(int j=0; j<y_num; ++j)
    //         {
    //             grid_points[0](i*y_num+j,0) = -0.1+width*(i)/(double(x_num));
    //             grid_points[0](i*y_num+j,1) = -0.1+height*(j)/(double(y_num));
    //         }
    //     }
    //     contact.solver.testIMLS(grid_points[0]);
    //     grid_values[0] = contact.solver.result_values[1];
    // }

    


    updateScreen(viewer);
    viewer.core().background_color.setOnes();
    viewer.data().set_face_based(true);
    viewer.data().shininess = 1.0;
    xz(0) = V.col(0).maxCoeff();
    if(SLIDING_TEST)
        viewer.data().point_size = 10.0;
    else
        viewer.data().point_size = 5.0;

    viewer.core().align_camera_center(V);
    viewer.core().animation_max_fps = 24.;
    // key_down(viewer,'0',0);
    viewer.core().is_animating = false;
    viewer.launch();


    return 0;
}