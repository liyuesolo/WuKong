#pragma once

#include "ObjectiveFunction.h"
#include "GradientDescentMinimizer.h"

class FancyBFGSMinimizer : public GradientDescentLineSearch {

public:
    FancyBFGSMinimizer(int maxIterations = 100, double solveResidual = 0.0001, int maxLineSearchIterations = 15)
            : GradientDescentLineSearch(maxIterations, solveResidual, maxLineSearchIterations) {}

    virtual ~FancyBFGSMinimizer() {}

public:
    Solver solver;
    int nMaxStabSteps = 10;
    double stabValue = 1e-4;

    virtual void computeSearchDirection(const ObjectiveFunction *function, const VectorXd &x, VectorXd &dx) {
        VectorXd g = function->getGradient(x);
        if (inverseHessian.rows() != x.rows()) {
            initializeHessian(function, x);
        } else {
            VectorXd deltaX = x - x0;
            VectorXd y = g - g0;

            if (y.norm() > 1e-10 && deltaX.norm() > 1e-10) {
                MatrixXd I = MatrixXd::Identity(x.rows(), x.rows());
                MatrixXd A = (I - (deltaX * y.transpose()) / (y.transpose() * deltaX));
                MatrixXd B = (I - (y * deltaX.transpose()) / (y.transpose() * deltaX));
                MatrixXd C = (deltaX * deltaX.transpose()) / (y.transpose() * deltaX);
                inverseHessian = A * inverseHessian * B + C;

                std::cout << "quasinewton " << (inverseHessian * y - deltaX).norm() << std::endl;
            }
        }
        x0 = x;
        g0 = g;

        dx = inverseHessian * g;
        std::cout << "dx norm " << dx.norm() << std::endl;
        if (dx.dot(g) < 0) {
            std::cout << "Search direction is bad!!!" << std::endl;
        }
        if (dx.norm() > 1) {
            dx = dx.normalized();
        }
    }

public:
    MatrixXd inverseHessian;
    VectorXd x0;
    VectorXd g0;

private:
    void initializeHessian(const ObjectiveFunction *function, const VectorXd &x) {
        inverseHessian = MatrixXd::Identity(x.rows(), x.rows());
    }
};


