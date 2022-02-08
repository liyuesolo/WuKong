#ifndef SIMULATION_H
#define SIMULATION_H

// #include <igl/opengl/glfw/Viewer.h>
// #include <igl/project.h>
// #include <igl/unproject_on_plane.h>
// #include <igl/opengl/glfw/imgui/ImGuiMenu.h>
// #include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
// #include <imgui/imgui.h>

#include "VertexModel.h"
#include "Timer.h"

using T = double;

// struct ViewerData
// {
//     bool show_rest = false;
//     bool show_current = true;
//     bool show_membrane = false;
//     bool split = false;
//     bool split_a_bit = false;
//     bool yolk_only = false;
//     bool show_apical_polygon = false;
//     bool show_basal_polygon = false;
//     bool show_contracting_edges = true;
//     bool show_outside_vtx = false;
//     int modes = 0;
//     bool enable_selection = false;
//     bool compute_energy = false;
//     double t = 0.0;
//     int compute_energy_cnt = 0;

//     int static_solve_step = 0;

//     int opt_step = 0;
//     bool check_modes = false;

//     int load_obj_iter_cnt = 0;

//     Eigen::MatrixXd evectors;
//     Eigen::VectorXd evalues;

//     Eigen::MatrixXd bounding_surface_samples;
//     Eigen::MatrixXd bounding_surface_samples_color;
//     int sdf_test_sample_idx_offset = 0;

//     Eigen::MatrixXd V;
//     Eigen::MatrixXi F;
//     Eigen::MatrixXd C;
// };

class Simulation
{
public:
    // using T = double;
    using TV = Vector<double, 3>;
    using TV2 = Vector<double, 2>;
    using TM2 = Matrix<double, 2, 2>;
    using IV = Vector<int, 3>;

    using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
    using MatrixXT = Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorXi = Vector<int, Eigen::Dynamic>;

    using CellModel = VertexModel;
    // using StiffnessMatrix = Eigen::SparseMatrix<T>;
    typedef long StorageIndex;
    using StiffnessMatrix = Eigen::SparseMatrix<T, Eigen::RowMajor, StorageIndex>;

    T newton_tol = 1e-6;
    int max_newton_iter = 500;
    bool verbose = false;
    

public:
    CellModel cells;

    VectorXT& undeformed = cells.undeformed;
    VectorXT& deformed = cells.deformed;
    
    // dynamics
    VectorXT& vtx_vel = cells.vtx_vel;
    T& dt = cells.dt;
    T current_time = 0.0;
    T simulation_time;

    VectorXT& u = cells.u;
    VectorXT& f = cells.f;
    bool& woodbury = cells.woodbury;
    int& num_nodes = cells.num_nodes;

    Timer t;
    
    bool& dynamic = cells.dynamics;

    bool save_mesh = true;
    
    // ViewerData viewer_data;
public:
    void initializeCells();
    void reinitializeCells();

    void generateMeshForRendering(
        Eigen::MatrixXd& V, Eigen::MatrixXi& F, 
        Eigen::MatrixXd& C, 
        bool show_deformed = true,
        bool show_rest = false,
        bool split = false,
        bool split_a_bit = false,
        bool yolk_only = false);

    void generatePolygonRendering(Eigen::MatrixXd& V, Eigen::MatrixXi& F, 
        Eigen::MatrixXd& C);
    
    void sampleBoundingSurface(Eigen::MatrixXd& V);

    bool staticSolve();

    void initializeDynamicsData(T _dt, T total_time);

    void reset();
    void update();

    bool advanceOneStep(int step);

    bool impliciteUpdate(VectorXT& _u);

    void computeLinearModes();

    bool linearSolve(StiffnessMatrix& K, VectorXT& residual, VectorXT& du);

    bool WoodburySolve(StiffnessMatrix& K, const MatrixXT& UV,
         VectorXT& residual, VectorXT& du);

    void buildSystemMatrixWoodbury(const VectorXT& _u, 
        StiffnessMatrix& K, MatrixXT& UV);
    
    bool solveWoodburyCholmod(StiffnessMatrix& K, MatrixXT& UV,
         VectorXT& residual, VectorXT& du);

    void buildSystemMatrix(const VectorXT& _u, StiffnessMatrix& K);
    T computeTotalEnergy(const VectorXT& _u, bool add_to_deform = true);
    T computeResidual(const VectorXT& _u,  VectorXT& residual);
    T lineSearchNewton(VectorXT& _u,  VectorXT& residual, int ls_max = 15, bool wolfe_condition = false);

    // void setViewer(igl::opengl::glfw::Viewer& viewer);
    // void updateScreen(igl::opengl::glfw::Viewer& viewer);

    void sampleEnergyWithSearchAndGradientDirection(
        const VectorXT& _u,  
        const VectorXT& search_direction,
        const VectorXT& negative_gradient
    );

    void loadDeformedState(const std::string& filename);

    void saveState(const std::string& filename);

public:
    Simulation() {}
    ~Simulation() {}
};

#endif