#pragma once

#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>

#include "Projects/Foam2D/include/VecMatDef.h"

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
    virtual VectorXi getDualGraph(const VectorXT &vertices, const VectorXT &params) = 0;

    // Returns the tesselation nodes corresponding to faces in the dual graph. For standard Voronoi tessellation, these are the circumcentres of Delaunay triangles.
    virtual VectorXT getNodes(const VectorXT &vertices, const VectorXT &params, const VectorXi &dual) = 0;

//    // Wrapper for getNodes which computes the dual graph internally.
//    VectorXT getNodes(const VectorXT &vertices, const VectorXT &params) {
//        return getNodes(vertices, params, getDualGraph(vertices, params));
//    }

    // Computes list of indices of nodes bounding each cell, ordered counterclockwise.
    std::vector<std::vector<int>> getCells(const VectorXT &vertices, const VectorXi &dual, const VectorXT &nodes);

    // Computes list of indices of neighboring sites, ordered counterclockwise.
    std::vector<std::vector<int>> getNeighbors(const VectorXT &vertices, const VectorXi &dual, int n_cells);

//    // Wrapper for getCells which computes the dual graph internally.
//    std::vector<std::vector<int>> getCells(const VectorXT &vertices, const VectorXT &params) {
//        VectorXi dual = getDualGraph(vertices, params);
//        return getCells(vertices, dual, getNodes(vertices, params, dual));
//    }

    virtual int getNumVertexParams() = 0;

    virtual VectorXT getDefaultVertexParams(const VectorXT &vertices) = 0;

    VectorXT combineVerticesParams(const VectorXT &vertices, const VectorXT &params) {
        int n_vtx = vertices.rows() / 2;
        int n_param = getNumVertexParams();
        int n_dims = 2 + n_param;
        VectorXT combined(n_vtx * n_dims);
        for (int i = 0; i < n_vtx; i++) {
            combined.segment<2>(i * n_dims) = vertices.segment<2>(i * 2);
            for (int j = 0; j < n_param; j++) {
                combined(i * n_dims + 2 + j) = params(i * n_param + j);
            }
        }
        return combined;
    }

    void separateVerticesParams(const VectorXT &combined, VectorXT &vertices, VectorXT &params) {
        int n_param = getNumVertexParams();
        int n_dims = 2 + n_param;
        int n_vtx = combined.rows() / n_dims;

        vertices.resize(n_vtx * 2);
        params.resize(n_vtx * n_param);
        for (int i = 0; i < n_vtx; i++) {
            vertices.segment<2>(i * 2) = combined.segment<2>(i * n_dims);
            for (int j = 0; j < n_param; j++) {
                params(i * n_param + j) = combined(i * n_dims + 2 + j);
            }
        }
    }
};
