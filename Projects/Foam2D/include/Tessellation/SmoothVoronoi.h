#pragma once

#include "Projects/Foam2D/include/Tessellation/Voronoi.h"

struct SmoothVoronoiChainRuleStruct {
    VectorXT u;

    MatrixXT pupt;
    MatrixXT pupc;
    std::vector<MatrixXT> p2upt2;

    VectorXT t;

    MatrixXT ptpq;
    MatrixXT ptpc;
    std::vector<MatrixXT> p2tpq2;
    std::vector<MatrixXT> p2tpqpc;
    std::vector<MatrixXT> p2tpc2;

    VectorXT q;

    MatrixXT dqdc;
    std::vector<MatrixXT> d2qdc2;
};

class SmoothVoronoi : public Voronoi {
    double d0 = 5e-2;

    TV3 getWeightDerivatives(double d);

    void getChainRuleMatrices(Node &node, SmoothVoronoiChainRuleStruct &chainRuleMatrices);

    void getNodeLSQ(Node &node, NodePosition &nodePos);

public:
    SmoothVoronoi() {}

    virtual void getNodeWrapper(Node &node, NodePosition &nodePos);

    virtual TessellationType getTessellationType() { return SMOOTH_VORONOI; }
};
