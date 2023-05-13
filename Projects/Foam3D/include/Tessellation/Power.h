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
    getNode(const VectorXT &v0, const VectorXT &v1, const VectorXT &v2, const VectorXT &v3, TV3 &node);

    virtual void
    getNodeGradient(const VectorXT &v0, const VectorXT &v1, const VectorXT &v2, const VectorXT &v3, MatrixXT &nodeGrad);

    virtual void
    getNodeHessian(const VectorXT &v0, const VectorXT &v1, const VectorXT &v2, const VectorXT &v3,
                   std::vector<MatrixXT> &nodeHess);

    virtual int getNumVertexParams() { return 1; }

    virtual VectorXT getDefaultVertexParams(const VectorXT &vertices);

    virtual TessellationType getTessellationType() { return POWER; }
};
