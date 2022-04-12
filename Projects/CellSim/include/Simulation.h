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
    using Entry = Eigen::Triplet<T>;
    typedef int StorageIndex;
    using StiffnessMatrix = Eigen::SparseMatrix<T, Eigen::ColMajor, StorageIndex>;

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
    bool fetchNegativeEigenVectorIfAny(T& negative_eigen_value, VectorXT& negative_eigen_vector);

    bool linearSolve(StiffnessMatrix& K, VectorXT& residual, VectorXT& du);
    bool linearSolveNaive(StiffnessMatrix& A, const VectorXT& b, VectorXT& x);

    bool WoodburySolve(StiffnessMatrix& K, const MatrixXT& UV,
         VectorXT& residual, VectorXT& du);
        
    bool WoodburySolveNaive(StiffnessMatrix& A, const MatrixXT& UV,
         const VectorXT& b, VectorXT& x);

    void buildSystemMatrixWoodbury(const VectorXT& _u, 
        StiffnessMatrix& K, MatrixXT& UV);
    
    bool solveWoodburyCholmod(StiffnessMatrix& K, MatrixXT& UV,
         VectorXT& residual, VectorXT& du);

    void buildSystemMatrix(const VectorXT& _u, StiffnessMatrix& K);
    T computeTotalEnergy(const VectorXT& _u, bool add_to_deform = true);
    T computeResidual(const VectorXT& _u,  VectorXT& residual);
    T lineSearchNewton(VectorXT& _u,  VectorXT& residual, int ls_max = 15, bool wolfe_condition = false);
    void checkInfoForSA();
    void sampleEnergyWithSearchAndGradientDirection(
        const VectorXT& _u,  
        const VectorXT& search_direction,
        const VectorXT& negative_gradient
    );

    void checkHessianPD(bool save_txt = false);
    void computeEigenValueSpectraSparse(StiffnessMatrix& A, int nmodes, VectorXT& modes, T shift = 1e-4);
    void loadDeformedState(const std::string& filename);
    void loadEdgeWeights(const std::string& filename, VectorXT& weights);
    void saveState(const std::string& filename);

    void loadVector(const std::string& filename, VectorXT& vector);

public:
    Simulation() {}
    ~Simulation() {}
};

#endif