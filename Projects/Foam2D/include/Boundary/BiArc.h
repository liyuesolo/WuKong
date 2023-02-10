#pragma once

#include "Boundary.h"

namespace BiArc {
    void getBiArcValues(const VectorXT &inputs, VectorXT &output);

    void getBiArcGradient(const VectorXT &inputs, MatrixXT &output);

    void getBiArcHessian(const VectorXT &inputs, std::vector<MatrixXT> &output);
}
