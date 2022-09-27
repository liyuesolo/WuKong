#pragma once

#include "../../src/optLib/ObjectiveFunction.h"
#include "../../include/Tessellation/Tessellation.h"
#include "../../include/CodeGen.h"

#define NAREA 300
#define NCELLS 40

static VectorXi getAreaTriangles(std::vector<std::vector<int>> cells) {
    VectorXi area_triangles(NAREA * 3);

    int edge = 0;
    for (size_t i = 0; i < NCELLS; i++) {
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

    double area_target = 0.1;
    double area_weight = 2;
    double length_weight = 0.01;

public:
    virtual double evaluate(const VectorXd &c) const {
        VectorXi tri;
        VectorXT x;
        VectorXi e;
        VectorXT A;
        double L;

        VectorXT vertices;
        VectorXT params;
        tessellation->separateVerticesParams(c, vertices, params);

        tri = tessellation->getDualGraph(vertices, params);
        x = tessellation->getNodes(vertices, params, tri);

        e = getAreaTriangles(tessellation->getCells(vertices, tri, x));

        double O = 0;
        if (area_weight > 0) {
            A = evaluate_A(vertices.segment<NCELLS * 2>(0), x, e);
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

    virtual void addGradientTo(const VectorXd &c, VectorXd &grad) const {
        grad += get_DODc(c);
    }

    VectorXd get_DODc(const VectorXd &c) const {
        VectorXi tri;
        VectorXT x;
        Eigen::SparseMatrix<double> dxdc;
        VectorXi e;
        VectorXT A;
        Eigen::SparseMatrix<double> dAdx;
        Eigen::SparseMatrix<double> dLdx;
        MatrixXT dOdA;

        VectorXT vertices;
        VectorXT params;
        tessellation->separateVerticesParams(c, vertices, params);

        tri = tessellation->getDualGraph(vertices, params);
        x = tessellation->getNodes(vertices, params, tri);
        dxdc = tessellation->getNodesGradient(vertices, params, tri);

        e = getAreaTriangles(tessellation->getCells(vertices, tri, x));

        VectorXT dOdc = VectorXT::Zero(c.rows());

        if (area_weight > 0) {
            A = evaluate_A(vertices.segment<NCELLS * 2>(0), x, e);
            dAdx = evaluate_dAdx(vertices.segment<NCELLS * 2>(0), x, e);

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

        // Fix boundary sites...
        for (int i = NCELLS * (2 + tessellation->getNumVertexParams()); i < dOdc.rows(); i++) {
            dOdc(i) = 0;
        }

        return dOdc.transpose();
    }
};
