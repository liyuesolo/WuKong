#ifndef FOAM2D_H
#define FOAM2D_H

#include <utility>
#include <iostream>
#include <fstream>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <tbb/tbb.h>



#include "VecMatDef.h"

template<int dim = 2>
class Foam2D
{
public:
    using TV = Vector<double, dim>;
    using TV3 = Vector<double, 3>;
    using TM = Matrix<double, dim, dim>;
    using IV = Vector<int, dim>;

    using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
    using MatrixXT = Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorXi = Vector<int, Eigen::Dynamic>;

    using Edge = Vector<int, 2>;

    typedef int StorageIndex;
    using StiffnessMatrix = Eigen::SparseMatrix<T, Eigen::ColMajor, StorageIndex>;
    using Entry = Eigen::Triplet<T>;

public:

    void generateMeshForRendering(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& C);
    void appendCylindersToEdges(const std::vector<std::pair<TV, TV>>& edge_pairs, 
        const std::vector<TV3>& color, T radius,
        Eigen::MatrixXd& _V, Eigen::MatrixXi& _F, Eigen::MatrixXd& _C);
public:
    Foam2D() {}
    ~Foam2D() {}
};
#endif