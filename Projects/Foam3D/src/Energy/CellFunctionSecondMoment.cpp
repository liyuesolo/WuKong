#include "Projects/Foam3D/include/Energy/CellFunctionSecondMoment.h"

void CellFunctionSecondMoment::getValue(Tessellation *tessellation, CellValue &value) const {
    CellValue centroidVals[3] = {value, value, value};
    for (int i = 0; i < 3; i++) {
        CellValue &centroid = centroidVals[i];
        centroidFuncs[i]->getValue(tessellation, centroid);
    }

    for (Face f: value.cell.faces) {
        for (int i = 1; i < f.nodes.size() - 1; i++) {
            TriangleValue triValue;
            triValue.v0 = tessellation->nodes[f.nodes[0]].pos;
            triValue.v1 = tessellation->nodes[f.nodes[i]].pos;
            triValue.v2 = tessellation->nodes[f.nodes[i + 1]].pos;

            TV3 centroid;
            for (int j = 0; j < 3; j++) {
                CellValue &centroidVal = centroidVals[j];
                centroid(j) = centroidVal.value;
            }

            perTriangleFunction.getValue(triValue, centroid);
            if (std::isnan(triValue.value) || std::isinf(triValue.value)) continue;
            value.value += triValue.value;
        }
    }
}

void CellFunctionSecondMoment::getGradient(Tessellation *tessellation, CellValue &value) const {
    value.gradient.setZero();

    CellValue centroidVals[3] = {value, value, value};
    for (int i = 0; i < 3; i++) {
        CellValue &centroid = centroidVals[i];
        centroidFuncs[i]->getValue(tessellation, centroid);
        centroidFuncs[i]->getGradient(tessellation, centroid);
    }
    VectorXT gradient_centroid = VectorXT::Zero(3);

    for (Face f: value.cell.faces) {
        for (int i = 1; i < f.nodes.size() - 1; i++) {
            TriangleValue triValue;
            Node tempNodes[3] = {f.nodes[0], f.nodes[i], f.nodes[i + 1]};
            triValue.v0 = tessellation->nodes[tempNodes[0]].pos;
            triValue.v1 = tessellation->nodes[tempNodes[1]].pos;
            triValue.v2 = tessellation->nodes[tempNodes[2]].pos;

            TV3 centroid;
            for (int j = 0; j < 3; j++) {
                CellValue &centroidVal = centroidVals[j];
                centroid(j) = centroidVal.value;
            }

            perTriangleFunction.getGradient(triValue, centroid);
            if (std::isnan(triValue.gradient.norm()) || std::isinf(triValue.gradient.norm())) continue;
            for (int ii = 0; ii < 3; ii++) {
                value.gradient.segment<3>(value.cell.nodeIndices[tempNodes[ii]] * 3) +=
                        triValue.gradient.segment<3>(ii * 3);
            }
            gradient_centroid += triValue.gradient.segment<3>(9);
        }
    }

    MatrixXT d_centroid_d_x(3, value.gradient.rows());
    for (int i = 0; i < 3; i++) {
        d_centroid_d_x.row(i) = centroidVals[i].gradient;
    }
    value.gradient += gradient_centroid.transpose() * d_centroid_d_x;
}

void CellFunctionSecondMoment::getHessian(Tessellation *tessellation, CellValue &value) const {
    value.hessian.setZero();

    CellValue centroidVals[3] = {value, value, value};
    for (int i = 0; i < 3; i++) {
        CellValue &centroid = centroidVals[i];
        centroidFuncs[i]->getValue(tessellation, centroid);
        centroidFuncs[i]->getGradient(tessellation, centroid);
        centroidFuncs[i]->getHessian(tessellation, centroid);
    }
    VectorXT gradient_centroid = VectorXT::Zero(3);
    MatrixXT hess_Cx = MatrixXT::Zero(3, value.gradient.rows());
    MatrixXT hess_CC = MatrixXT::Zero(3, 3);

    for (Face f: value.cell.faces) {
        for (int i = 1; i < f.nodes.size() - 1; i++) {
            TriangleValue triValue;
            Node tempNodes[3] = {f.nodes[0], f.nodes[i], f.nodes[i + 1]};
            triValue.v0 = tessellation->nodes[tempNodes[0]].pos;
            triValue.v1 = tessellation->nodes[tempNodes[1]].pos;
            triValue.v2 = tessellation->nodes[tempNodes[2]].pos;

            TV3 centroid;
            for (int j = 0; j < 3; j++) {
                CellValue &centroidVal = centroidVals[j];
                centroid(j) = centroidVal.value;
            }

            perTriangleFunction.getGradient(triValue, centroid);
            perTriangleFunction.getHessian(triValue, centroid);
            if (std::isnan(triValue.hessian.norm()) || std::isinf(triValue.hessian.norm())) continue;
            for (int ii = 0; ii < 3; ii++) {
                hess_Cx.block<3, 3>(0, value.cell.nodeIndices[tempNodes[ii]] * 3) +=
                        triValue.hessian.block<3, 3>(9, ii * 3);
                for (int jj = 0; jj < 3; jj++) {
                    value.hessian.block<3, 3>(value.cell.nodeIndices[tempNodes[ii]] * 3,
                                              value.cell.nodeIndices[tempNodes[jj]] * 3) +=
                            triValue.hessian.block<3, 3>(ii * 3, jj * 3);
                }
            }
            hess_CC += triValue.hessian.block<3, 3>(9, 9);
            gradient_centroid += triValue.gradient.segment<3>(9);

        }
    }

    MatrixXT d_centroid_d_x(3, value.gradient.rows());
    for (int i = 0; i < 3; i++) {
        d_centroid_d_x.row(i) = centroidVals[i].gradient;
    }

    value.hessian += d_centroid_d_x.transpose() * hess_CC * d_centroid_d_x + d_centroid_d_x.transpose() * hess_Cx +
                     hess_Cx.transpose() * d_centroid_d_x;
    for (int i = 0; i < 3; i++) {
        value.hessian += gradient_centroid(i) * centroidVals[i].hessian;
    }
}
