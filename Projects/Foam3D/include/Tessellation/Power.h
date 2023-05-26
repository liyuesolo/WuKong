#pragma once

#include "Projects/Foam3D/include/Tessellation/Tessellation.h"

class Power : public Tessellation {
private:
    double wmul = 0.5;

public:
    Power() {}

    virtual void
    getDualGraph();

    virtual void
    getNode(const VectorXT &v0, const VectorXT &v1, const VectorXT &v2, const VectorXT &v3, NodePosition &nodePos);

    virtual void
    getNodeGradient(const VectorXT &v0, const VectorXT &v1, const VectorXT &v2, const VectorXT &v3,
                    NodePosition &nodePos);

    virtual void
    getNodeHessian(const VectorXT &v0, const VectorXT &v1, const VectorXT &v2, const VectorXT &v3,
                   NodePosition &nodePos);

    virtual void
    getNodeBFace(const TV3 &b0, const TV3 &b1, const TV3 &b2, const VectorXT &v0,
                 const VectorXT &v1, const VectorXT &v2, NodePosition &nodePos);

    virtual void
    getNodeBFaceGradient(const TV3 &b0, const TV3 &b1, const TV3 &b2, const VectorXT &v0,
                         const VectorXT &v1, const VectorXT &v2, NodePosition &nodePos);

    virtual void
    getNodeBFaceHessian(const TV3 &b0, const TV3 &b1, const TV3 &b2, const VectorXT &v0,
                        const VectorXT &v1, const VectorXT &v2, NodePosition &nodePos);

    virtual void
    getNodeBEdge(const TV3 &b0, const TV3 &b1, const VectorXT &v0,
                 const VectorXT &v1, NodePosition &nodePos);

    virtual void
    getNodeBEdgeGradient(const TV3 &b0, const TV3 &b1, const VectorXT &v0,
                         const VectorXT &v1, NodePosition &nodePos);

    virtual void
    getNodeBEdgeHessian(const TV3 &b0, const TV3 &b1, const VectorXT &v0,
                        const VectorXT &v1, NodePosition &nodePos);

    virtual int getNumVertexParams() { return 1; }

    virtual VectorXT getDefaultVertexParams(const VectorXT &vertices);

    virtual TessellationType getTessellationType() { return POWER; }
};
