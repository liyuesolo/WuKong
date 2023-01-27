#pragma once

#include "Projects/Foam2D/include/Tessellation/Tessellation.h"

class Voronoi : public Tessellation {
public:
    Voronoi() {}

    VectorXi delaunayJRS(const VectorXT &vertices);

//    VectorXi delaunayNaive(const VectorXT &vertices);

    virtual VectorXi getDualGraph(const VectorXT &vertices, const VectorXT &params);

    virtual void getNode(const VectorXT &v0, const VectorXT &v1, const VectorXT &v2, TV &node);

    virtual void
    getNodeGradient(const VectorXT &v0, const VectorXT &v1, const VectorXT &v2, VectorXT &gradX, VectorXT &gradY);

    virtual void
    getNodeHessian(const VectorXT &v0, const VectorXT &v1, const VectorXT &v2, MatrixXT &hessX, MatrixXT &hessY);

    virtual void getBoundaryNode(const VectorXT &v0, const VectorXT &v1, const TV &b0, const TV &b1, TV &node);

    virtual void
    getBoundaryNodeGradient(const VectorXT &v0, const VectorXT &v1, const TV &b0, const TV &b1, VectorXT &gradX,
                            VectorXT &gradY);

    virtual void
    getBoundaryNodeHessian(const VectorXT &v0, const VectorXT &v1, const TV &b0, const TV &b1, MatrixXT &hessX,
                           MatrixXT &hessY);

    virtual void
    getArcBoundaryNode(const VectorXT &v0, const VectorXT &v1, const TV &b0, const TV &b1, double r, int flag,
                       TV &node);

    virtual void
    getArcBoundaryNodeGradient(const VectorXT &v0, const VectorXT &v1, const TV &b0, const TV &b1, double r, int flag,
                               VectorXT &gradX,
                               VectorXT &gradY);

    virtual void
    getArcBoundaryNodeHessian(const VectorXT &v0, const VectorXT &v1, const TV &b0, const TV &b1, double r, int flag,
                              MatrixXT &hessX,
                              MatrixXT &hessY);

    virtual int getNumVertexParams() { return 0; }

    virtual VectorXT getDefaultVertexParams(const VectorXT &vertices);

    virtual TessellationType getTessellationType() { return VORONOI; }
};
