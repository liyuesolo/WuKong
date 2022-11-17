#pragma once

#include "Projects/Foam2D/include/Tessellation/Tessellation.h"

class Power : public Tessellation {
public:
    Power() {}

    VectorXi powerDualNaive(const VectorXT &vertices3d);

    VectorXi powerDualCGAL(const VectorXT &vertices3d);

    virtual VectorXi getDualGraph(const VectorXT &vertices, const VectorXT &params);

    virtual VectorXT getNodes(const VectorXT &vertices, const VectorXT &params, const VectorXi &dual);

    virtual TV getNode(const VectorXT &v0, const VectorXT &v1, const VectorXT &v2);

    virtual TV getBoundaryNode(const VectorXT &v0, const VectorXT &v1, const TV &b0, const TV &b1);

    virtual int getNumVertexParams() { return 1; }

    virtual VectorXT getDefaultVertexParams(const VectorXT &vertices);

    virtual TessellationType getTessellationType() { return POWER; }
};
