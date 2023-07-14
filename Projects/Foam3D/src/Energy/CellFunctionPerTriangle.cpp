#include "Projects/Foam3D/include/Energy/CellFunctionPerTriangle.h"

void CellFunctionPerTriangle::getValue(Tessellation *tessellation, CellValue &value) const {
    value.value = 0;

    for (Face f: value.cell.faces) {
        for (int i = 1; i < f.nodes.size() - 1; i++) {
            TriangleValue triValue;
            triValue.v0 = tessellation->nodes[f.nodes[0]].pos;
            triValue.v1 = tessellation->nodes[f.nodes[i]].pos;
            triValue.v2 = tessellation->nodes[f.nodes[i + 1]].pos;

            perTriangleFunction->getValue(triValue);
            if (std::isnan(triValue.value) || std::isinf(triValue.value)) continue;
            value.value += triValue.value;
        }
    }
}

void CellFunctionPerTriangle::getGradient(Tessellation *tessellation, CellValue &value) const {
    value.gradient.setZero();

    for (Face f: value.cell.faces) {
        for (int i = 1; i < f.nodes.size() - 1; i++) {
            TriangleValue triValue;
            Node tempNodes[3] = {f.nodes[0], f.nodes[i], f.nodes[i + 1]};
            triValue.v0 = tessellation->nodes[tempNodes[0]].pos;
            triValue.v1 = tessellation->nodes[tempNodes[1]].pos;
            triValue.v2 = tessellation->nodes[tempNodes[2]].pos;

            perTriangleFunction->getGradient(triValue);
            if (std::isnan(triValue.gradient.norm()) || std::isinf(triValue.gradient.norm())) continue;
            for (int ii = 0; ii < 3; ii++) {
                value.gradient.segment<3>(value.cell.nodeIndices[tempNodes[ii]] * 3) +=
                        triValue.gradient.segment<3>(ii * 3);
            }
        }
    }
}

void CellFunctionPerTriangle::getHessian(Tessellation *tessellation, CellValue &value) const {
    value.hessian.setZero();

    for (Face f: value.cell.faces) {
        for (int i = 1; i < f.nodes.size() - 1; i++) {
            TriangleValue triValue;
            Node tempNodes[3] = {f.nodes[0], f.nodes[i], f.nodes[i + 1]};
            triValue.v0 = tessellation->nodes[tempNodes[0]].pos;
            triValue.v1 = tessellation->nodes[tempNodes[1]].pos;
            triValue.v2 = tessellation->nodes[tempNodes[2]].pos;

            perTriangleFunction->getHessian(triValue);
            if (std::isnan(triValue.hessian.norm()) || std::isinf(triValue.hessian.norm())) continue;
            for (int ii = 0; ii < 3; ii++) {
                for (int jj = 0; jj < 3; jj++) {
                    value.hessian.block<3, 3>(value.cell.nodeIndices[tempNodes[ii]] * 3,
                                              value.cell.nodeIndices[tempNodes[jj]] * 3) +=
                            triValue.hessian.block<3, 3>(ii * 3, jj * 3);
                }
            }
        }
    }
}
