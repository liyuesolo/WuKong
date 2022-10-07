#pragma once

#include "Projects/Foam2D/include/Tessellation/Tessellation.h"

class Sectional : public Tessellation {
public:
    Sectional() {}

    VectorXi sectionalDualNaive(const VectorXT &vertices3d);

    virtual VectorXi getDualGraph(const VectorXT &vertices, const VectorXT &params);

    virtual VectorXT getNodes(const VectorXT &vertices, const VectorXT &params, const VectorXi &dual);

    virtual int getNumVertexParams() { return 1; }

    virtual VectorXT getDefaultVertexParams(const VectorXT &vertices);

    virtual TessellationType getTessellationType() { return SECTIONAL; }
};
