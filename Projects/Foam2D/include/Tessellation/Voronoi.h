#pragma once

#include "Projects/Foam2D/include/Tessellation/Tessellation.h"

class Voronoi : public Tessellation {
public:
    Voronoi() {}

    VectorXi delaunayJRS(const VectorXT &vertices);

//    VectorXi delaunayNaive(const VectorXT &vertices);

    virtual VectorXi getDualGraph(const VectorXT &vertices, const VectorXT &params);

    virtual void getStandardNode(const VectorXT &v0, const VectorXT &v1, const VectorXT &v2, VectorXT &node);

    virtual void
    getStandardNodeGradient(const VectorXT &v0, const VectorXT &v1, const VectorXT &v2, MatrixXT &nodeGrad);

    virtual void
    getStandardNodeHessian(const VectorXT &v0, const VectorXT &v1, const VectorXT &v2,
                           std::vector<MatrixXT> &nodeHess);

    virtual void
    getBoundaryNode(const VectorXT &v0, const VectorXT &v1, const TV &b0, const TV &b1, VectorXT &node);

    virtual void
    getBoundaryNodeGradient(const VectorXT &v0, const VectorXT &v1, const TV &b0, const TV &b1, MatrixXT &nodeGrad);

    virtual void
    getBoundaryNodeHessian(const VectorXT &v0, const VectorXT &v1, const TV &b0, const TV &b1,
                           std::vector<MatrixXT> &nodeHess);

    virtual void
    getArcBoundaryNode(const VectorXT &v0, const VectorXT &v1, const TV &b0, const TV &b1, double r, int flag,
                       VectorXT &node);

    virtual void
    getArcBoundaryNodeGradient(const VectorXT &v0, const VectorXT &v1, const TV &b0, const TV &b1, double r, int flag,
                               MatrixXT &nodeGrad);

    virtual void
    getArcBoundaryNodeHessian(const VectorXT &v0, const VectorXT &v1, const TV &b0, const TV &b1, double r, int flag,
                              std::vector<MatrixXT> &nodeHess);

    virtual void
    getBezierBoundaryNode(const VectorXT &v0, const VectorXT &v1, const TV &b0, const TV &b1, double q0, double q1,
                          int flag,
                          VectorXT &node);

    virtual void
    getBezierBoundaryNodeGradient(const VectorXT &v0, const VectorXT &v1, const TV &b0, const TV &b1, double q0,
                                  double q1,
                                  int flag,
                                  MatrixXT &nodeGrad);

    virtual void
    getBezierBoundaryNodeHessian(const VectorXT &v0, const VectorXT &v1, const TV &b0, const TV &b1, double q0,
                                 double q1, int flag,
                                 std::vector<MatrixXT> &nodeHess);

    virtual int getNumVertexParams() { return 0; }

    virtual VectorXT getDefaultVertexParams(const VectorXT &vertices);

    virtual TessellationType getTessellationType() { return VORONOI; }
};
