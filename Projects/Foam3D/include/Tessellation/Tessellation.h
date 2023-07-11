#pragma once

#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>

#include "Projects/Foam3D/include/VecMatDef.h"
#include "../Boundary/Boundary.h"

enum TessellationType {
    POWER
};

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
    MatrixXT grad;
    MatrixXT hess[3];
};

struct Face {
    int site0;
    int site1;
    int bface = -1;
    std::vector<Node> nodes;
};

struct Cell {
    int cellIndex;
    std::vector<int> facesPos;
    std::vector<int> facesNeg;
    std::map<Node, int> nodeIndices;
};

struct CellInfo {
    double adhesion = 0;
};

class Tessellation {

public:
    VectorXT c;
    std::vector<CellInfo> cellInfos;

    std::vector<Cell> cells;

    std::vector<Face> faces;
    std::map<Node, NodePosition> nodes;

    Boundary *boundary;

    bool isValid = false;
private:

public:
    Tessellation() {}

    void clipFaces();

    void computeCellData();

    virtual void
    getDualGraph() = 0;

    virtual void
    getNode(const VectorXT &v0, const VectorXT &v1, const VectorXT &v2, const VectorXT &v3, NodePosition &nodePos) = 0;

    virtual void
    getNodeGradient(const VectorXT &v0, const VectorXT &v1, const VectorXT &v2, const VectorXT &v3,
                    NodePosition &nodePos) = 0;

    virtual void
    getNodeHessian(const VectorXT &v0, const VectorXT &v1, const VectorXT &v2, const VectorXT &v3,
                   NodePosition &nodePos) = 0;

    virtual void
    getNodeBFace(const TV3 &b0, const TV3 &b1, const TV3 &b2, const VectorXT &v0,
                 const VectorXT &v1, const VectorXT &v2, NodePosition &nodePos) = 0;

    virtual void
    getNodeBFaceGradient(const TV3 &b0, const TV3 &b1, const TV3 &b2, const VectorXT &v0,
                         const VectorXT &v1, const VectorXT &v2, NodePosition &nodePos) = 0;

    virtual void
    getNodeBFaceHessian(const TV3 &b0, const TV3 &b1, const TV3 &b2, const VectorXT &v0,
                        const VectorXT &v1, const VectorXT &v2, NodePosition &nodePos) = 0;

    virtual void
    getNodeBEdge(const TV3 &b0, const TV3 &b1, const VectorXT &v0,
                 const VectorXT &v1, NodePosition &nodePos) = 0;

    virtual void
    getNodeBEdgeGradient(const TV3 &b0, const TV3 &b1, const VectorXT &v0,
                         const VectorXT &v1, NodePosition &nodePos) = 0;

    virtual void
    getNodeBEdgeHessian(const TV3 &b0, const TV3 &b1, const VectorXT &v0,
                        const VectorXT &v1, NodePosition &nodePos) = 0;

    void tessellate(const VectorXT &vertices, const VectorXT &params, const VectorXT &p_free);

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
