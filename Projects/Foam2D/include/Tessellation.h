#pragma once

#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>

#include "VecMatDef.h"

enum TessellationType {
    VORONOI,
    SECTIONAL,
};

using TV = Vector<double, 2>;
using TV3 = Vector<double, 3>;
using TM = Matrix<double, 2, 2>;
using IV3 = Vector<int, 3>;
using IV = Vector<int, 2>;

using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
using MatrixXT = Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
using MatrixXi = Matrix<int, Eigen::Dynamic, Eigen::Dynamic>;
using VectorXi = Vector<int, Eigen::Dynamic>;

class Tessellation {
public:
    Tessellation() {}

    // Returns the dual graph of the tesselation. For standard Voronoi tessellation, this is the Delaunay triangulation.
    virtual VectorXi getDualGraph(const VectorXT &vertices) = 0;

    // Returns the tesselation nodes corresponding to faces in the dual graph. For standard Voronoi tessellation, these are the circumcentres of Delaunay triangles.
    virtual VectorXT getNodes(const VectorXT &vertices, const VectorXi &dual) = 0;

    // Wrapper for getNodes which computes the dual graph internally.
    VectorXT getNodes(const VectorXT &vertices) { return getNodes(vertices, getDualGraph(vertices)); }

    // Gradient of nodes with respect to vertices.
    virtual Eigen::SparseMatrix<double> getNodesGradient(const VectorXT &vertices, const VectorXi &dual) = 0;

    // Wrapper for getNodesGradient which computes the dual graph internally.
    Eigen::SparseMatrix<double> getNodesGradient(const VectorXT &vertices) {
        return getNodesGradient(vertices, getDualGraph(vertices));
    }

    // Computes list of indices of nodes bounding each cell, ordered counterclockwise.
    std::vector<std::vector<int>> getCells(const VectorXT &vertices, const VectorXi &dual, const VectorXT &nodes);

    // Wrapper for getCells which computes the dual graph internally.
    std::vector<std::vector<int>> getCells(const VectorXT &vertices) {
        VectorXi dual = getDualGraph(vertices);
        return getCells(vertices, dual, getNodes(vertices, dual));
    }
};
