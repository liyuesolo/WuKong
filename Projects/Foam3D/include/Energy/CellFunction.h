#pragma once

#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>

#include "Projects/Foam3D/include/VecMatDef.h"
#include "Projects/Foam3D/include/Tessellation/Tessellation.h"

struct CellValue {
    Cell cell;

    double value;
    VectorXT gradient;
    MatrixXT hessian;

    CellValue(Cell &_cell) {
        cell = _cell;

        int nvars = cell.nodeIndices.size() * 3 + 4;

        value = 0;
        gradient = VectorXT::Zero(nvars);
        hessian = MatrixXT::Zero(nvars, nvars);
    }
};

class CellFunction {
public:
    virtual void
    getValue(Tessellation *tessellation, CellValue &value) const = 0;

    virtual void
    getGradient(Tessellation *tessellation, CellValue &value) const = 0;

    virtual void
    getHessian(Tessellation *tessellation, CellValue &value) const = 0;

public:
};
