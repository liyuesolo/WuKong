#pragma once

#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>

#include "Projects/Foam2D/include/VecMatDef.h"
#include "Tessellation.h"

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
        hessian = VectorXT::Zero(nvars, nvars);
    }
};

class CellFunction {
public:
    virtual void
    addValue(Tessellation *tessellation, CellValue &value) const = 0;

    virtual void
    addGradient(Tessellation *tessellation, CellValue &value) const = 0;

    virtual void
    addHessian(Tessellation *tessellation, CellValue &value) const = 0;

public:
};
