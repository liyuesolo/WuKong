#pragma once

#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>

#include "Projects/Foam2D/include/VecMatDef.h"

enum TessellationType {
    VORONOI,
    POWER
};

using TV = Vector<double, 2>;
using TV3 = Vector<double, 3>;
using TM = Matrix<double, 2, 2>;
using IV = Vector<int, 2>;
using IV3 = Vector<int, 3>;
using IV4 = Vector<int, 4>;

using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
using MatrixXT = Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
using MatrixXi = Matrix<int, Eigen::Dynamic, Eigen::Dynamic>;
using VectorXi = Vector<int, Eigen::Dynamic>;

class CellFunction;

class CellInfo;

class Boundary;

class BoundaryEdge;

struct CellEdge {
    int startNode;
    int neighbor;
    int flag;
    int nextEdge;
};

struct Cell {
    std::vector<CellEdge> edges;
};

class Tessellation {

public:
    VectorXi dual;
    std::vector<Cell> cells;
//    std::vector<VectorXi> cells;
//    std::vector<std::vector<int>> neighborhoods;
//    std::vector<std::vector<int>> neighborhoodFlags;
//    std::vector<VectorXi> segment;
    VectorXT c;
    VectorXT x;

    bool isValid = false;
private:
    VectorXT last_boundary;
    Boundary *bdry;

//    MatrixXT dxdc;
    Eigen::SparseMatrix<double> dxdc;
    Eigen::SparseMatrix<double> dxdv;
    Eigen::SparseMatrix<double> dxdq;
    std::vector<MatrixXT> d2xdy2;

    // Computes list of indices of neighboring sites, ordered counterclockwise.
    std::vector<std::vector<int>> getNeighbors(const VectorXT &vertices, const VectorXi &dual, int n_cells);

    // Get the tessellation node at the intersection of three cells.
    virtual void getNode(const VectorXT &v0, const VectorXT &v1, const VectorXT &v2, VectorXT &node) = 0;

    virtual void
    getNodeGradient(const VectorXT &v0, const VectorXT &v1, const VectorXT &v2, MatrixXT &nodeGrad) = 0;

    virtual void
    getNodeHessian(const VectorXT &v0, const VectorXT &v1, const VectorXT &v2,
                   std::vector<MatrixXT> &nodeHess) = 0;

    void getNodeWrapper(int i0, int i1, int i2, int flag, VectorXT &node, MatrixXT &nodeGrad,
                        std::vector<MatrixXT> &nodeHess, int &mode);

    // Get the tessellation node at the intersection of two cells and a domain boundary.
    virtual void
    getBoundaryNode(const VectorXT &v0, const VectorXT &v1, const TV &b0, const TV &b1, VectorXT &node) = 0;

    virtual void
    getBoundaryNodeGradient(const VectorXT &v0, const VectorXT &v1, const TV &b0, const TV &b1, MatrixXT &nodeGrad) = 0;

    virtual void
    getBoundaryNodeHessian(const VectorXT &v0, const VectorXT &v1, const TV &b0, const TV &b1,
                           std::vector<MatrixXT> &nodeHess) = 0;

    // Get the tessellation node at the intersection of two cells and a domain boundary.
    virtual void
    getArcBoundaryNode(const VectorXT &v0, const VectorXT &v1, const TV &b0, const TV &b1, double r, int flag,
                       VectorXT &node) = 0;

    virtual void
    getArcBoundaryNodeGradient(const VectorXT &v0, const VectorXT &v1, const TV &b0, const TV &b1, double r, int flag,
                               MatrixXT &nodeGrad) = 0;

    virtual void
    getArcBoundaryNodeHessian(const VectorXT &v0, const VectorXT &v1, const TV &b0, const TV &b1, double r, int flag,
                              std::vector<MatrixXT> &nodeHess) = 0;

public:
    Tessellation() {}

    // Returns the dual graph of the tesselation. For standard Voronoi tessellation, this is the Delaunay triangulation.
    virtual VectorXi getDualGraph(const VectorXT &vertices, const VectorXT &params) = 0;

    void
    addSingleCellFunctionValue(int cellIndex, const CellFunction &function, double &value, const CellInfo *cellInfo);

    void
    addSingleCellFunctionGradient(int cellIndex, const CellFunction &function, VectorXT &gradient,
                                  const CellInfo *cellInfo);

    void addFunctionValue(const CellFunction &function, double &value, std::vector<CellInfo> cellInfos);

    void addFunctionGradient(const CellFunction &function, VectorXT &gradient,
                             std::vector<CellInfo> cellInfos);

    void
    addFunctionHessian(const CellFunction &function, MatrixXT &hessian, std::vector<CellInfo> cellInfos);

    // Computes list of indices of neighboring sites and boundary edges, ordered counterclockwise.
    bool
    getNeighborsClipped(const VectorXT &vertices, const VectorXT &params, const VectorXi &dual, int n_cells);

    void tessellate(const VectorXT &vertices, const VectorXT &params, Boundary *bdry_new, int n_free);

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

    virtual TessellationType getTessellationType() = 0;
};
