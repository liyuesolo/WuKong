#pragma once

#include "../../src/optLib/ObjectiveFunction.h"
#include "../../include/Tessellation/Tessellation.h"
#include "../../include/CodeGen.h"
#include "../../include/Constants.h"

static VectorXi getAreaTriangles(std::vector<std::vector<int>> cells) {
    VectorXi area_triangles(NAREA * 3);

    int edge = 0;
    for (size_t i = 0; i < NFREE; i++) {
        std::vector<int> &cell = cells[i];
        size_t degree = cell.size();

        for (size_t j = 0; j < degree; j++) {
            area_triangles[edge * 3 + 0] = i;
            area_triangles[edge * 3 + 1] = cell[j];
            area_triangles[edge * 3 + 2] = cell[(j + 1) % degree];
            edge++;
        }
    }

    for (int i = edge; i < NAREA; i++) {
        area_triangles[i * 3 + 0] = 0; // TODO: this is a hack, degenerate triangle with area 0...
        area_triangles[i * 3 + 1] = 0;
        area_triangles[i * 3 + 2] = 0;
    }

    return area_triangles;
}

class AreaLengthObjective : public ObjectiveFunction {

public:
    Tessellation *tessellation;

    VectorXd c_fixed;

    double area_target = 0.05;
    double area_weight = 2;
    double length_weight = 0.01;

public:
    virtual double evaluate(const VectorXd &c_free) const {
        VectorXi tri;
        VectorXT x;
        VectorXi e;
        VectorXT A;
        double L;

        VectorXd c(c_free.size() + c_fixed.size());
        c << c_free, c_fixed;

        VectorXT vertices;
        VectorXT params;
        tessellation->separateVerticesParams(c, vertices, params);

        tri = tessellation->getDualGraph(vertices, params);
        x = tessellation->getNodes(vertices, params, tri);

        e = getAreaTriangles(tessellation->getCells(vertices, tri, x));

        double O = 0;
        if (area_weight > 0) {
            A = evaluate_A(c_free, x, e);
            for (int i = 0; i < A.rows(); i++) {
                O += area_weight * (A(i) - area_target) * (A(i) - area_target);
            }
        }
        if (length_weight > 0) {
            L = evaluate_L(x, e);
            O += length_weight * L;
        }

        return O;
    }

    virtual void addGradientTo(const VectorXd &c_free, VectorXd &grad) const {
        grad += get_DODc(c_free);
    }

    VectorXd get_DODc(const VectorXd &c_free) const {
        VectorXi tri;
        VectorXT x;
        Eigen::SparseMatrix<double> dxdc;
        VectorXi e;
        VectorXT A;
        Eigen::SparseMatrix<double> dAdx;
        Eigen::SparseMatrix<double> dLdx;
        MatrixXT dOdA;

        VectorXd c(c_free.size() + c_fixed.size());
        c << c_free, c_fixed;

        VectorXT vertices;
        VectorXT params;
        tessellation->separateVerticesParams(c, vertices, params);

        tri = tessellation->getDualGraph(vertices, params);
        x = tessellation->getNodes(vertices, params, tri);
        dxdc = tessellation->getNodesGradient(vertices, params, tri);

        e = getAreaTriangles(tessellation->getCells(vertices, tri, x));

        VectorXT dOdc = VectorXT::Zero(c_free.rows());

        if (area_weight > 0) {
            A = evaluate_A(c_free, x, e);
            dAdx = evaluate_dAdx(c_free, x, e);

            VectorXT targets;
            targets.resize(A.rows());
            targets.setOnes();
            targets = targets * area_target;
            dOdA = area_weight * 2 * (A - targets).transpose();

            dOdc += (dOdA * dAdx * dxdc).transpose();
        }
        if (length_weight > 0) {
            dLdx = evaluate_dLdx(x, e);

            double dOdL = length_weight;
            dOdc += (dOdL * dLdx * dxdc).transpose();
        }

        return dOdc.transpose();
    }
};
