#ifndef SIMULATION_H
#define SIMULATION_H

#include "VertexModel.h"

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
    using VectorXi = Vector<int, Eigen::Dynamic>;

    using CellModel = VertexModel;
    using StiffnessMatrix = Eigen::SparseMatrix<T>;

    T newton_tol = 1e-6;
    int max_newton_iter = 500;
    bool verbose = false;

public:
    CellModel cells;

    VectorXT& undeformed = cells.undeformed;
    VectorXT& deformed = cells.deformed;
    VectorXT& u = cells.u;

public:
    void initializeCells();
    void generateMeshForRendering(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& C);

    void advanceOneStep();

    bool staticSolve();

    bool linearSolve(StiffnessMatrix& K, VectorXT& residual, VectorXT& du);
    void buildSystemMatrix(const VectorXT& _u, StiffnessMatrix& K);
    T computeTotalEnergy(const VectorXT& _u);
    T computeResidual(const VectorXT& _u,  VectorXT& residual);
    T lineSearchNewton(VectorXT& _u,  VectorXT& residual, int ls_max = 15);

public:
    Simulation() {}
    ~Simulation() {}
};

#endif