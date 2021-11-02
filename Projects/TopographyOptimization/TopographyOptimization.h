#ifndef TOPOGRAPHY_OPTIMIZATION_H
#define TOPOGRAPHY_OPTIMIZATION_H


#include "Timer.h"

#include "FEMSolver.h"
#include "ShellFEMSolver.h"

template<class T, int dim>
class FEMSolver;

template<class T, int dim>
class ShellFEMSolver;

enum OptimizationMethod
{
    GD, MMA
};

enum BeadType
{
    None, BeadRib, VRib, DiagonalBead, FourParts, Circle, CurveBD, Drawing, Levelset
};

template<class T, int dim, class Solver>
class TopographyOptimization
{
public:
    // using Solver = FEMSolver<T, dim>;

    using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
    using TV = Vector<T, dim>;
    using TV2 = Vector<T, 2>;
    using IV = Vector<int, dim>;
    using TM = Matrix<T, dim, dim>;
    using StiffnessMatrix = Eigen::SparseMatrix<T>;

    
    Solver& solver;

    VectorXT& deformed = solver.deformed;
    VectorXT& undeformed = solver.undeformed;
    
    OptimizationMethod om = GD;

    Timer timer;


    std::vector<std::function<bool(const TV&, T&)>> bead_levelsets;

public:
    void initializeScene(BeadType type);
    
    void addBeadLevelset(int type);
    
    void addBeads(BeadType bead_type, const TV& min_corner, const TV& max_corner, const TV& dx);     

    void forward();

    void inverseRestShape();

    void generateMeshForRendering(Eigen::MatrixXd& V, 
                                Eigen::MatrixXi& F, 
                                Eigen::MatrixXd& C) 
        { return solver.generateMeshForRendering(V, F, C); }


public:
    TopographyOptimization(Solver& _solver) : solver(_solver) {}
    ~TopographyOptimization() {}
};


#endif