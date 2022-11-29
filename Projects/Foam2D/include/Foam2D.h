#ifndef FOAM2D_H
#define FOAM2D_H

#include <utility>
#include <iostream>
#include <fstream>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>

#include "VecMatDef.h"
#include "Projects/Foam2D/include/Tessellation/Tessellation.h"
#include "../include/Energy/EnergyObjective.h"
#include "../include/Energy/DynamicObjective.h"
#include "../include/TrajectoryOpt/TrajectoryOptNLP.h"
#include "../src/optLib/GradientDescentMinimizer.h"

using TV = Vector<double, 2>;
using TV3 = Vector<double, 3>;
using TM = Matrix<double, 2, 2>;
using IV3 = Vector<int, 3>;
using IV = Vector<int, 2>;

using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
using MatrixXT = Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
using MatrixXi = Matrix<int, Eigen::Dynamic, Eigen::Dynamic>;
using VectorXi = Vector<int, Eigen::Dynamic>;
using VectorXf = Vector<float, Eigen::Dynamic>;

class Foam2D {
public:
    std::vector<Tessellation *> tessellations;
    int tessellation = 1;
    std::vector<GradientDescentLineSearch *> minimizers;
    int opttype = 1;

    EnergyObjective energyObjective;
    DynamicObjective dynamicObjective;

    TrajectoryOptNLP trajOptNLP;

    VectorXT vertices;
    VectorXT params;

    int n_free;
    int n_fixed;

    VectorXT boundary;

public:

    void initRandomSitesInCircle(int n_free_in, int n_fixed_in);

    void initBasicTestCase();

    void initRandomCellsInBox(int n_free_in);

    void initImageMatch(MatrixXi &markers);

    void dynamicsInit(double dt, double m, double mu);

    void dynamicsNewStep();

    void optimize(bool dynamic);

    void moveVertex(int idx, const TV &pos);

    int getClosestMovablePointThreshold(const TV &p, double threshold);

    void resetVertexParams();

    void getTessellationViewerData(MatrixXT &S, MatrixXT &X, MatrixXi &E, MatrixXT &Sc, MatrixXT &Ec, MatrixXT &V,
                                   MatrixXi &F, MatrixXT &Fc);

    void getTriangulationViewerData(MatrixXT &S, MatrixXT &X, MatrixXi &E, MatrixXT &Sc, MatrixXT &Ec, MatrixXT &V,
                                    MatrixXi &F, MatrixXT &Fc);

    void addTrajectoryOptViewerData(MatrixXT &S, MatrixXT &X, MatrixXi &E, MatrixXT &Sc, MatrixXT &Ec, MatrixXT &V,
                                    MatrixXi &F, MatrixXT &Fc);

    void getPlotAreaHistogram(VectorXT &areas);

    void getPlotObjectiveStats(bool dynamics, double &obj_value, double &gradient_norm, bool &hessian_pd);

    void getPlotObjectiveFunctionLandscape(int selected_vertex, int type, int image_size, double range, VectorXf &obj,
                                           double &obj_min, double &obj_max);

    bool isConvergedDynamic(double tol);

    void trajectoryOptSetInit();

    void trajectoryOptOptimizeIPOPT(int N);

    void trajectoryOptGetFrame(int frame);

    void trajectoryOptGetForces(VectorXd &forceX, VectorXd &forceY);

    void trajectoryOptStop();

    void imageMatchOptimizeIPOPT(MatrixXi &markers);

public:

    Foam2D();

    ~Foam2D() {}
};

#endif
