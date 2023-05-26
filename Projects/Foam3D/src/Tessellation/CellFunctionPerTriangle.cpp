#include "../../include/Tessellation/CellFunctionPerTriangle.h"

void CellFunctionPerTriangle::addValue(Tessellation *tessellation, CellValue &value) const {
    double mul = 1;
    auto func = [&](const int &iF) {
        Face f = tessellation->faces[iF];
        for (int i = 1; i < f.nodes.size() - 1; i++) {
            TriangleValue triValue;
            triValue.v0 = tessellation->nodes[f.nodes[0]].pos;
            triValue.v1 = tessellation->nodes[f.nodes[i]].pos;
            triValue.v2 = tessellation->nodes[f.nodes[i + 1]].pos;

            perTriangleFunction->addValue(triValue);
            value.value += mul * triValue.value;
        }
    };
    std::for_each(value.cell.facesPos.begin(), value.cell.facesPos.end(), func);
    mul = -1;
    std::for_each(value.cell.facesNeg.begin(), value.cell.facesNeg.end(), func);
}

void CellFunctionPerTriangle::addGradient(Tessellation *tessellation, CellValue &value) const {
    value.gradient = VectorXT::Zero(value.cell.nodeIndices.size() * 3);

    double mul = 1;
    auto func = [&](const int &iF) {
        Face f = tessellation->faces[iF];
        for (int i = 1; i < f.nodes.size() - 1; i++) {
            TriangleValue triValue;
            Node tempNodes[3] = {f.nodes[0], f.nodes[i], f.nodes[i + 1]};
            triValue.v0 = tessellation->nodes[tempNodes[0]].pos;
            triValue.v1 = tessellation->nodes[tempNodes[1]].pos;
            triValue.v2 = tessellation->nodes[tempNodes[2]].pos;

            perTriangleFunction->addGradient(triValue);
            for (int ii = 0; ii < 3; ii++) {
                value.gradient.segment<3>(value.cell.nodeIndices[tempNodes[ii]] * 3) +=
                        mul * triValue.gradient.segment<3>(ii * 3);
            }
        }
    };
    std::for_each(value.cell.facesPos.begin(), value.cell.facesPos.end(), func);
    mul = -1;
    std::for_each(value.cell.facesNeg.begin(), value.cell.facesNeg.end(), func);
}

void CellFunctionPerTriangle::addHessian(Tessellation *tessellation, CellValue &value) const {
    value.hessian = MatrixXT::Zero(value.cell.nodeIndices.size() * 3, value.cell.nodeIndices.size() * 3);

    double mul = 1;
    auto func = [&](const int &iF) {
        Face f = tessellation->faces[iF];
        for (int i = 1; i < f.nodes.size() - 1; i++) {
            TriangleValue triValue;
            Node tempNodes[3] = {f.nodes[0], f.nodes[i], f.nodes[i + 1]};
            triValue.v0 = tessellation->nodes[tempNodes[0]].pos;
            triValue.v1 = tessellation->nodes[tempNodes[1]].pos;
            triValue.v2 = tessellation->nodes[tempNodes[2]].pos;

            perTriangleFunction->addHessian(triValue);
            for (int ii = 0; ii < 3; ii++) {
                for (int jj = 0; jj < 3; jj++) {
                    value.hessian.block<3, 3>(value.cell.nodeIndices[tempNodes[ii]] * 3,
                                              value.cell.nodeIndices[tempNodes[jj]] * 3) +=
                            mul * triValue.hessian.block<3, 3>(ii * 3, jj * 3);
                }
            }
        }
    };
    std::for_each(value.cell.facesPos.begin(), value.cell.facesPos.end(), func);
    mul = -1;
    std::for_each(value.cell.facesNeg.begin(), value.cell.facesNeg.end(), func);
}
