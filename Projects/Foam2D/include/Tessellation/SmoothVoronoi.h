#pragma once

#include "Projects/Foam2D/include/Tessellation/Voronoi.h"

class SmoothVoronoi : public Voronoi {
    double d0 = 5e-2;

    double getWeight(double d);

    void getNodeLSQ(Node &node, NodePosition &nodePos);

public:
    SmoothVoronoi() {}

    virtual void getNodeWrapper(Node &node, NodePosition &nodePos);

    virtual TessellationType getTessellationType() { return SMOOTH_VORONOI; }
};
