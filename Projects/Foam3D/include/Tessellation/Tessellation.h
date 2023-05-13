#pragma once

#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>

#include "Projects/Foam2D/include/VecMatDef.h"

enum TessellationType {
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

enum NodeType {
    STANDARD,
    B_FACE,
    B_EDGE,
    B_VERTEX,
};

struct Node {
    NodeType type;
    int gen[4];
};

bool operator<(const Node &a, const Node &b);

struct NodePosition {
    TV3 pos;
};

struct Face {
    int site0;
    int site1;
    std::vector<Node> nodes;
};

struct BoundaryVertex {
    TV3 pos;
};
struct BoundaryFace {
    IV3 vertices;
};

struct TempStruct {
    MatrixXT V;
    MatrixXi F;
    MatrixXT Fc;
};

class Tessellation {

public:
    VectorXT c;

    std::vector<Face> faces;
    std::map<Node, NodePosition> nodes;

    std::vector<BoundaryVertex> bv;
    std::vector<BoundaryFace> bf;

    bool isValid = false;
private:

public:
    Tessellation() {}

    void clipFaces(TempStruct &ts);

    virtual void
    getDualGraph() = 0;

    virtual void
    getNode(const VectorXT &v0, const VectorXT &v1, const VectorXT &v2, const VectorXT &v3, TV3 &node) = 0;

    virtual void
    getNodeGradient(const VectorXT &v0, const VectorXT &v1, const VectorXT &v2, const VectorXT &v3,
                    MatrixXT &nodeGrad) = 0;

    virtual void
    getNodeHessian(const VectorXT &v0, const VectorXT &v1, const VectorXT &v2, const VectorXT &v3,
                   std::vector<MatrixXT> &nodeHess) = 0;


    void tessellate(const VectorXT &vertices, const VectorXT &params, int n_cells);

    virtual int getNumVertexParams() = 0;

    virtual VectorXT getDefaultVertexParams(const VectorXT &vertices) = 0;

    VectorXT combineVerticesParams(const VectorXT &vertices, const VectorXT &params) {
        int n_vtx = vertices.rows() / 3;
        int n_param = getNumVertexParams();
        int n_dims = 3 + n_param;
        VectorXT combined(n_vtx * n_dims);
        for (int i = 0; i < n_vtx; i++) {
            combined.segment<3>(i * n_dims) = vertices.segment<3>(i * 3);
            for (int j = 0; j < n_param; j++) {
                combined(i * n_dims + 3 + j) = params(i * n_param + j);
            }
        }
        return combined;
    }

    void separateVerticesParams(const VectorXT &combined, VectorXT &vertices, VectorXT &params) {
        int n_param = getNumVertexParams();
        int n_dims = 3 + n_param;
        int n_vtx = combined.rows() / n_dims;

        vertices.resize(n_vtx * 3);
        params.resize(n_vtx * n_param);
        for (int i = 0; i < n_vtx; i++) {
            vertices.segment<3>(i * 3) = combined.segment<3>(i * n_dims);
            for (int j = 0; j < n_param; j++) {
                params(i * n_param + j) = combined(i * n_dims + 3 + j);
            }
        }
    }

    int getDims() {
        return 3 + getNumVertexParams();
    }

    virtual TessellationType getTessellationType() = 0;
};
