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
#include "../include/Objective/AreaLengthObjective.h"
#include "../include/Objective/AreaLengthObjective.h"
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

class Foam2D {
public:
    using Edge = Vector<int, 2>;

    typedef int StorageIndex;
    using StiffnessMatrix = Eigen::SparseMatrix<T, Eigen::ColMajor, StorageIndex>;
    using Entry = Eigen::Triplet<T>;

    std::vector<Tessellation *> tessellations;
    int tesselation = 0;
    std::vector<GradientDescentLineSearch *> minimizers;
    int opttype = 0;

    AreaLengthObjective objective;

    VectorXT vertices;
    VectorXT params;
    int dim = 2;
public:

    void generateRandomVoronoi();

    void getTessellationViewerData(MatrixXT &S, MatrixXT &X, MatrixXi &E, MatrixXT &V, MatrixXi &F, MatrixXT &C);

    void getTriangulationViewerData(MatrixXT &S, MatrixXT &X, MatrixXi &E, MatrixXT &V, MatrixXi &F, MatrixXT &C);

    void optimize();

    void moveVertex(int idx, const TV &pos);

    int getClosestMovablePointThreshold(const TV &p, double threshold);

    void checkGradients();

    void resetVertexParams();

public:

    Foam2D();

    ~Foam2D() {}
};

#endif
