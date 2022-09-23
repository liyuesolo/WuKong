#pragma once

#include "Tessellation.h"

class Voronoi : public Tessellation {
public:
    Voronoi() {}

    virtual VectorXi getDualGraph(const VectorXT &vertices);

    virtual VectorXT getNodes(const VectorXT &vertices, const VectorXi &dual);

    virtual Eigen::SparseMatrix<double> getNodesGradient(const VectorXT &vertices, const VectorXi &dual);
};
