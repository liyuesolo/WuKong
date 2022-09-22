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


class Foam2D {
public:
    using TV = Vector<double, 2>;
    using TV3 = Vector<double, 3>;
    using TM = Matrix<double, 2, 2>;
    using IV3 = Vector<int, 3>;
    using IV = Vector<int, 2>;

    using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
    using MatrixXT = Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using MatrixXi = Matrix<int, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorXi = Vector<int, Eigen::Dynamic>;
    
    using Edge = Vector<int, 2>;

    typedef int StorageIndex;
    using StiffnessMatrix = Eigen::SparseMatrix<T, Eigen::ColMajor, StorageIndex>;
    using Entry = Eigen::Triplet<T>;

    VectorXT vertices;
    VectorXi tri_face_indices;
    int dim = 2;
public:

    void generateRandomVoronoi();

    void generateVoronoiDiagramForVisualization(MatrixXT &C, MatrixXT &X, MatrixXi &E);

    void optimize(double area_target);

    void retriangulate();

    void moveVertex(int idx, const TV &pos);

    int getClosestMovablePointThreshold(const TV &p, double threshold);

    void testCasadiCode();

public:
    Foam2D() {}

    ~Foam2D() {}
};

#endif