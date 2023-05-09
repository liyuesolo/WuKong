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

#include "../include/app.h"
#include "../include/SensitivityAnalysis.h"
#include "../include/TilingObjectives.h"
#include "../include/HexFEMSolver.h"
#include <boost/filesystem.hpp>
#include "../include/TorchModel.h"
#include "../include/PoissonDisk.h"

inline bool fileExist (const std::string& name) {
    std::ifstream f(name.c_str());
    return f.good();
}

using CMat = Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic>;
using TV = Vector<T, 2>;
using TV3 = Vector<T, 3>;
using VectorXT = Eigen::Matrix<T, Eigen::Dynamic, 1>;

int main(int argc, char** argv)
{
    
    FEMSolver fem_solver;
    Tiling2D tiling(fem_solver);
    
    if (argc > 1)
    {
        int IH = std::stoi(argv[1]);
        std::string result_folder = argv[2];
        if (IH == -1)
        {
            T pi = std::stod(argv[3]);
            std::vector<T> params = {pi};
            tiling.generateGreenStrainSecondPKPairsServerToyExample(params, result_folder);
        }
        else
        {
            int n_params = std::stoi(argv[3]);
            std::vector<T> params(n_params);
            for (int i = 0; i < n_params; i++)
            {
                params[i] = std::stod(argv[4+i]);
            }
            int resume_start = std::stoi(argv[4 + n_params]);
            tiling.generateGreenStrainSecondPKPairsServer(params, IH, "", result_folder, resume_start);
        }
        
    }
    else
    {
        auto runSimApp = [&]()
        {
            igl::opengl::glfw::Viewer viewer;
            igl::opengl::glfw::imgui::ImGuiMenu menu;

            viewer.plugins.push_back(&menu);
            SimulationApp app(tiling);
                
            app.setViewer(viewer, menu);
            
            viewer.launch(true, false, "WuKong viewer", 2000, 1600);
        };
        
        auto run3DSim = [&]()
        {
            
            HexFEMSolver hex_fem_solver;
            igl::opengl::glfw::Viewer viewer;
            igl::opengl::glfw::imgui::ImGuiMenu menu;

            viewer.plugins.push_back(&menu);
            Simulation3DApp app(hex_fem_solver);
            T dx = 0.1;
            hex_fem_solver.buildGrid3D(TV3::Zero(), TV3(1.0, dx, 1.0), dx);
            Vector<bool, 4> flag;
            flag << true, false, true, false;
            hex_fem_solver.E = 0.0;
            // hex_fem_solver.nu = 0.3;
            hex_fem_solver.updateLameParams();
            hex_fem_solver.KL_stiffness = 1e6;
            hex_fem_solver.KL_stiffness_shear = 1e6;
            hex_fem_solver.addCornerVtxToDirichletVertices(flag);
            // hex_fem_solver.setBCBendCorner(4.0, 0.0);
            hex_fem_solver.penaltyInPlane(0, 0.1);
            app.setViewer(viewer, menu);
            viewer.launch();
        };

        auto renderScene = [&]()
        {
            igl::opengl::glfw::Viewer viewer;
            igl::opengl::glfw::imgui::ImGuiMenu menu;

            viewer.plugins.push_back(&menu);
            SimulationApp app(tiling);
            
            app.setViewer(viewer, menu);
            std::string folder = "/home/yueli/Documents/ETH/SandwichStructure/SampleStrain/";
            int width = 2000, height = 2000;
            CMat R(width,height), G(width,height), B(width,height), A(width,height);
            viewer.core().background_color.setOnes();
            viewer.data().set_face_based(true);
            viewer.data().shininess = 1.0;
            viewer.data().point_size = 10.0;
            viewer.core().camera_zoom *= 1.4;
            viewer.launch_init();
            int n_sp_strain = 50;
            TV range_strain(0.001, 0.2);
            T delta_strain = (range_strain[1] - range_strain[0]) / T(n_sp_strain);
            int cnt = 0;
            for (T strain = range_strain[0]; strain < range_strain[1] + delta_strain; strain += delta_strain)
            {
                igl::readOBJ(folder + std::to_string(strain) + ".obj", app.V, app.F);
                    
                viewer.data().clear();
                viewer.data().set_mesh(app.V, app.F);
                app.C.resize(app.F.rows(), 3);
                app.C.col(0).setConstant(0.0); app.C.col(1).setConstant(0.3); app.C.col(2).setConstant(1.0);
                viewer.data().set_colors(app.C);
                viewer.core().align_camera_center(app.V);
                viewer.core().draw_buffer(viewer.data(),true,R,G,B,A);
                A.setConstant(255);
                igl::png::writePNG(R,G,B,A, folder + std::to_string(cnt++)+".png");
            }
        };


        auto testNeuralConstitutiveModel = [&]()
        {
            TorchModel ncm;
            ncm.load("/home/yueli/Documents/ETH/WuKong/Projects/Tiling2D/python/model_scripted.pt");
            ncm.test();
        };

        auto inverseDesign = [&]()
        {
            UniaxialStressObjective ti_obj(tiling);
            SensitivityAnalysis sa(fem_solver, ti_obj);
            VectorXT strain_samples;
            strain_samples.resize(3);
            // strain_samples << 1.0-0.025, 1.0+0.025, 1.0+0.085;
            strain_samples << 1.0 - 0.025, 1.0 + 0.025, 1.0 + 0.085;
            // TV strain_range(-0.05, 0.1);
            // int n_sp_strain = 10;
            // strain_samples.resize(n_sp_strain);
            // for (int i = 0; i < n_sp_strain; i++)
            // {
            //     strain_samples[i] = 1.0 + strain_range[0] + T(i) * (strain_range[1] - strain_range[0]) / T(n_sp_strain);
            //     // std::cout << strain_samples[i] - 1.0 << ", ";
            // }
            // std::exit(0);
            ti_obj.strain_samples = strain_samples;
            // VectorXT stress_samples;
            // ti_obj.computeStressForDifferentStrain(TV(0.115, 0.75), stress_samples);
            // ti_obj.computeStressForDifferentStrain(TV(0.104123,  0.53023), stress_samples);
            // for (int i = 0; i < strain_samples.size(); i++)
                // std::cout << std::setprecision(12) << stress_samples[i] << ", ";
            
            // std::cout << ti_obj.generateSingleTarget(ti) << std::endl;
            // sa.optimizeMMA();
            // sa.optimizeLBFGSB();
            // sa.optimizeGradientDescent();
            //-0.00700946   0.0239969   0.0720543
            sa.sampleGradientDirection();

        };

        auto generatePoisonDiskSample = [&]()
        {
            PoissonDisk pd;
            // IH 01
            // Vector<T, 4> min_corner; min_corner << 0.05, 0.25, 0.05, 0.4;
            // Vector<T, 4> max_corner; max_corner << 0.3, 0.75, 0.15, 0.8;
            // IH 03
            // Vector<T, 4> min_corner; min_corner << 0.05, 0.2, 0.08, 0.4;
            // Vector<T, 4> max_corner; max_corner << 0.5, 0.8, 0.5, 0.8;
            // IH 21
            // Vector<T, 2> min_corner; min_corner << 0.05, 0.3;
            // Vector<T, 2> max_corner; max_corner << 0.3, 0.9;
            // IH 28
            Vector<T, 2> min_corner; min_corner << 0.005, 0.005;
            Vector<T, 2> max_corner; max_corner << 0.8, 1.0;

            VectorXT samples;
            pd.sampleNDBox<2>(min_corner, max_corner, 100, samples);
            // return;
            std::ofstream out("PD_IH28_100.txt");
            out << "[ ";
            int n_tiling_paras = 2;
            for (int i = 0; i < 400; i++)
            {
                out << "[";
                for (int j = 0; j < n_tiling_paras - 1; j++)
                    out << std::setprecision(12) << samples[i * n_tiling_paras  + j] << ", ";
                out << samples[i * n_tiling_paras  + n_tiling_paras - 1] << "], " << std::endl;
            }
            out << "]";
            out.close();
        };

        auto renderMeshSequence = [&](int IH)
        {
            igl::opengl::glfw::Viewer viewer;
            igl::opengl::glfw::imgui::ImGuiMenu menu;

            viewer.plugins.push_back(&menu);
            SimulationApp app(tiling);
            
            app.setViewer(viewer, menu);
            std::string folder = "/home/yueli/Documents/ETH/NCM/PaperData/StrainStress/IH"+std::to_string(IH) + "/";
            int width = 2000, height = 2000;
            CMat R(width,height), G(width,height), B(width,height), A(width,height);
            viewer.core().background_color.setOnes();
            viewer.data().set_face_based(true);
            viewer.data().shininess = 1.0;
            viewer.data().point_size = 10.0;
            viewer.core().camera_zoom *= 1.4;
            viewer.launch_init();
            igl::readOBJ(folder + "0.obj", app.V, app.F);
            viewer.core().align_camera_center(app.V);
            for (int i = 0; i < 15; i++)
            {
                igl::readOBJ(folder + std::to_string(i) + ".obj", app.V, app.F);
                    
                viewer.data().clear();
                viewer.data().set_mesh(app.V, app.F);
                app.C.resize(app.F.rows(), 3);
                app.C.col(0).setConstant(0.0); app.C.col(1).setConstant(0.3); app.C.col(2).setConstant(1.0);
                viewer.data().set_colors(app.C);
                // viewer.core().align_camera_center(app.V);
                viewer.core().draw_buffer(viewer.data(),true,R,G,B,A);
                A.setConstant(255);
                igl::png::writePNG(R,G,B,A, folder + std::to_string(i)+".png");
            }
        };

        auto renderToyExample = [&]()
        {
            igl::opengl::glfw::Viewer viewer;
            igl::opengl::glfw::imgui::ImGuiMenu menu;

            viewer.plugins.push_back(&menu);
            SimulationApp app(tiling);
            
            app.setViewer(viewer, menu);
            std::string folder = "/home/yueli/Documents/ETH/SandwichStructure/ServerToy/";
            int width = 3000, height = 3000;
            CMat R(width,height), G(width,height), B(width,height), A(width,height);
            viewer.core().background_color.setOnes();
            viewer.data().set_face_based(true);
            viewer.data().shininess = 1.0;
            viewer.data().point_size = 10.0;
            viewer.core().camera_zoom *= 1.4;
            viewer.launch_init();
            tiling.initializeSimulationDataFromFiles(folder + "0/structure.vtk", PBC_XY);
            app.thicken_edges = true;
            app.updateScreen(viewer);
            viewer.core().align_camera_center(app.V);
            for (int i = 0; i < 400; i++)
            {
                tiling.initializeSimulationDataFromFiles(folder + std::to_string(i) + "/structure.vtk", PBC_XY);
                app.updateScreen(viewer);
                // viewer.core().align_camera_center(app.V);
                    
                viewer.data().clear();
                viewer.data().set_mesh(app.V, app.F);
                app.C.resize(app.F.rows(), 3);
                app.C.col(0).setConstant(0.0); app.C.col(1).setConstant(0.3); app.C.col(2).setConstant(1.0);
                viewer.data().set_colors(app.C);
                // viewer.core().align_camera_center(app.V);
                viewer.core().draw_buffer(viewer.data(),true,R,G,B,A);
                A.setConstant(255);
                igl::png::writePNG(R,G,B,A, folder + std::to_string(i)+".png");
            }
        };
        
        // auto renderPaperStructure = [&]()
        // {
        //     igl::opengl::glfw::Viewer viewer;
        //     igl::opengl::glfw::imgui::ImGuiMenu menu;

        //     viewer.plugins.push_back(&menu);
        //     SimulationApp app(tiling);
            
        //     app.setViewer(viewer, menu);
        //     std::string base_folder = "/home/yueli/Documents/ETH/WuKong/Projects/Tiling2D/paper_data/strain_stress/";
        //     int width = 2000, height = 2000;
        //     CMat R(width,height), G(width,height), B(width,height), A(width,height);
        //     viewer.core().background_color.setOnes();
        //     viewer.data().set_face_based(true);
        //     viewer.data().shininess = 1.0;
        //     viewer.data().point_size = 10.0;
        //     viewer.core().camera_zoom *= 1.4;
        //     viewer.launch_init();

        //     for (int IH : {1, 21, 22, 28, 50, 67})
        //     {

        //         // tiling.generateOneStructureSquarePatch(IH, );
        //         updateScreen(viewer);
        //         viewer.core().align_camera_center(V);
        //         viewer.core().viewport(2) = 2000; viewer.core().viewport(3) = 1600;
        //         viewer.core().camera_zoom *= 8.0; //IH01
        //         // viewer.core().camera_zoom *= 6.0;
        //     }
        // };

        // save3DMesh();
        // testNeuralConstitutiveModel();
        // inverseDesign();
        // renderScene();
        // run3DSim();
        // tiling.generateNHHomogenousData("/home/yueli/Documents/ETH/SandwichStructure/Homo/");
        // tiling.sampleDirectionWithUniaxialStrain("/home/yueli/Documents/ETH/SandwichStructure/SampleStrain/", 50, TV(0, M_PI), 1.05);
        // Matrix<T, 3, 3> elasticity_tensor;
        // tiling.solver.computeHomogenizationElasticityTensor(0.0, 1.05, elasticity_tensor);
        // tiling.sampleUniAxialStrainAlongDirection("/home/yueli/Documents/ETH/SandwichStructure/StableStructure/", 50, TV(0.75, 1.7), 0.);
        // tiling.runSimUniAxialStrainAlongDirection("/home/yueli/Documents/ETH/SandwichStructure/StableStructure/", 0, 50, TV(1.05, 1.1), 0.0, {0.1224,  0.5254, 0.1433, 0.49});
        // tiling.generatseGreenStrainSecondPKPairs("/home/yueli/Documents/ETH/SandwichStructure/TrainingData/WithEnergy/");
        // fem_solver.pbc_translation_file = "/home/yueli/Documents/ETH/SandwichStructure/Server/0/structure_translation.txt";
        // tiling.initializeSimulationDataFromFiles("/home/yueli/Documents/ETH/SandwichStructure/Server/0/structure.vtk", PBC_XY);
        // tiling.sampleFixedTilingParamsAlongStrain("/home/yueli/Documents/ETH/SandwichStructure/SampleStrain/");
        // runSimApp();
        // tiling.generateStrainStressSimulationData("/home/yueli/Documents/ETH/WuKong/Projects/Tiling2D/paper_data/", 1, 100);
        // tiling.generateStrainStressSimulationData("/home/yueli/Documents/ETH/WuKong/Projects/Tiling2D/paper_data/", 67, 100);
        // tiling.generateStrainStressSimulationData("/home/yueli/Documents/ETH/WuKong/Projects/Tiling2D/paper_data/", 50, 100);
        // tiling.generateStrainStressSimulationData("/home/yueli/Documents/ETH/WuKong/Projects/Tiling2D/paper_data/", 22, 100);
        // tiling.generateStrainStressSimulationDataFromFile("/home/yueli/Documents/ETH/WuKong/Projects/Tiling2D/paper_data/", 
        //     "/home/yueli/Documents/ETH/WuKong/Projects/Tiling2D/paper_data/IH_1_strain_stress.txt", 
        //     "projectPD", 1, 50);
        // tiling.generateStrainStressSimulationDataFromFile("/home/yueli/Documents/ETH/WuKong/Projects/Tiling2D/paper_data/", 
        //     "/home/yueli/Documents/ETH/WuKong/Projects/Tiling2D/paper_data/IH_21_strain_stress.txt", 
        //     "projectPD", 21, 50);
        // tiling.generateStrainStressSimulationDataFromFile("/home/yueli/Documents/ETH/WuKong/Projects/Tiling2D/paper_data/", 
        //     "/home/yueli/Documents/ETH/WuKong/Projects/Tiling2D/paper_data/IH_50_strain_stress.txt", 
        //     "projectPD", 50, 50);
        // tiling.generateStrainStressSimulationDataFromFile("/home/yueli/Documents/ETH/WuKong/Projects/Tiling2D/paper_data/", 
        //     "/home/yueli/Documents/ETH/WuKong/Projects/Tiling2D/paper_data/IH_28_strain_stress.txt", 
        //     "projectPD", 28, 50);
        // tiling.generateStrainStressSimulationData("/home/yueli/Documents/ETH/WuKong/Projects/Tiling2D/paper_data/", 28, 100);
        // tiling.generateStrainStressSimulationData("/home/yueli/Documents/ETH/WuKong/Projects/Tiling2D/paper_data/", 29, 50);
        // tiling.generateStrainStressSimulationData("/home/yueli/Documents/ETH/WuKong/Projects/Tiling2D/paper_data/", 21, 100);
        // tiling.generateStrainStressSimulationData("/home/yueli/Documents/ETH/WuKong/Projects/Tiling2D/paper_data/", 28, 50);
        // for (int IH : {1, 21, 22, 28, 29, 50, 67})
        // tiling.generateStrainStressDataFromParams("/home/yueli/Documents/ETH/WuKong/Projects/Tiling2D/paper_data/strain_stress/", 50);
        tiling.generateStrainStressDataFromParams("/home/yueli/Documents/ETH/WuKong/Projects/Tiling2D/paper_data/strain_stress/", 21,
            0, {0.10887782216199968, 0.6526880237650166}, TV(0.9, 1.2), 25, false);
        tiling.generateStrainStressDataFromParams("/home/yueli/Documents/ETH/WuKong/Projects/Tiling2D/paper_data/strain_stress/", 50,
            0, {0.21330992098074827, 0.6013053081575949}, TV(0.9, 1.2), 25, false);
        tiling.generateStrainStressDataFromParams("/home/yueli/Documents/ETH/WuKong/Projects/Tiling2D/paper_data/strain_stress/", 67,
            0.5 * M_PI, {0.1499092468720663, 0.7400465501354314}, TV(0.9, 1.2), 25, false);
        tiling.generateStrainStressDataFromParams("/home/yueli/Documents/ETH/WuKong/Projects/Tiling2D/paper_data/strain_stress/", 28,
            0.5 * M_PI, {0.219595270751497, 0.397364995280736}, TV(0.9, 1.2), 25, false);
        tiling.generateStrainStressDataFromParams("/home/yueli/Documents/ETH/WuKong/Projects/Tiling2D/paper_data/strain_stress/", 1,
            0.25 * M_PI, {0.07592219809002378, 0.6738283023035684, 0.13498317834561663, 0.5710111040053688}, TV(0.9, 1.2), 25, false);
        tiling.generateStrainStressDataFromParams("/home/yueli/Documents/ETH/WuKong/Projects/Tiling2D/paper_data/strain_stress/", 22,
            0, {0.22090099256867987, 0.6187215051849427, 0.1571724148917844}, TV(0.9, 1.2), 25, false);
        

        // tiling.generateStrainStressSimulationDataFromFile("/home/yueli/Documents/ETH/WuKong/Projects/Tiling2D/paper_data/strain_stress/", "sim_uni", 21, 25);

        // std::exit(0);
        // tiling.generateOnePerodicUnit();

        // tiling.solver.diffTestdxdE(TV3(0.05, 0.05, 0.001));
        // tiling.solver.diffTestdfdE(TV3(0.05, 0.05, 0.001));
        // tiling.solver.diffTestdxdEScale(TV3(0.05, 0.05, 0.001));
        // tiling.solver.diffTestdfdEScale(TV3(0.05, 0.05, 0.001));
        // tiling.solver.diffTestdxdE(TV3(0.0186157, -0.0105733,  0.0898344));
        // Matrix<T, 3, 3> elasticity_tensor;
        // tiling.solver.computeHomogenizationElasticityTensorSA(M_PI * 0.0, 1.05, elasticity_tensor);
        // std::cout << elasticity_tensor << std::endl;
        // std::exit(0);
        // std::string base_folder = "/home/yueli/Documents/ETH/WuKong/Projects/Tiling2D/paper_data/poisson_ratio/";
        // for (int IH : {22})
            // tiling.generatePoissonRatioDataFromParams(base_folder, IH);
            // tiling.generateStiffnessDataFromParams(base_folder, IH);
        // base_folder = "/home/yueli/Documents/ETH/WuKong/Projects/Tiling2D/paper_data/stiffness/";
        //     tiling.generateStiffnessDataFromParams(base_folder, 28);
        // renderToyExample();
        // tiling.sampleSingleStructurePoissonDisk("/home/yueli/Documents/ETH/SandwichStructure/IH21_PoissonDisk/", TV(0.7, 1.5), TV(0.9, 1.2), TV(0, M_PI), 100, 19);
        // generatePoisonDiskSample();
        // tiling.generateTenPointUniaxialStrainData("/home/yueli/Documents/ETH/NCM/PaperData/StrainStress/IH21/",
        //     19, 0.0, TV(0.95, 1.1), 0.01, {0.115, 0.765});
        // renderMeshSequence(21);
    }
    return 0;
}