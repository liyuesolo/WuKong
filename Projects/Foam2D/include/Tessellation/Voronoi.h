#pragma once

#include "Projects/Foam2D/include/Tessellation/Tessellation.h"

class Voronoi : public Tessellation {
public:
    Voronoi() {}

    VectorXi delaunayJRS(const VectorXT &vertices);

    VectorXi delaunayNaive(const VectorXT &vertices);

    virtual VectorXi getDualGraph(const VectorXT &vertices, const VectorXT &params);

    virtual VectorXT getNodes(const VectorXT &vertices, const VectorXT &params, const VectorXi &dual);

    virtual int getNumVertexParams() { return 0; }

    virtual VectorXT getDefaultVertexParams(const VectorXT &vertices);
};
