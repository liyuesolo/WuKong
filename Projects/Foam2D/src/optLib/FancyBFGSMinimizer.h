    #pragma once

#include "ObjectiveFunction.h"
#include "GradientDescentMinimizer.h"

class FancyBFGSMinimizer : public GradientDescentLineSearch {

public:
    FancyBFGSMinimizer(int maxIterations = 100, double solveResidual = 0.0001, int maxLineSearchIterations = 15)
        : GradientDescentLineSearch(maxIterations, solveResidual, maxLineSearchIterations) {    }

    virtual ~FancyBFGSMinimizer() {}

public:
	Solver solver;
	int nMaxStabSteps = 10;
	double stabValue = 1e-4;
	virtual void computeSearchDirection(const ObjectiveFunction *function, const VectorXd &x, VectorXd& dx) {
        if (inverseHessian.rows() != x.rows()) {
            initializeHessian(function, x);
        }

		VectorXd g = function->getGradient(x);
		dx = inverseHessian * g;
    }

    virtual void step(const ObjectiveFunction *function, const VectorXd &dx, VectorXd &x) {
        VectorXd x0 = x;
        VectorXd g0 = function->getGradient(x);
        GradientDescentLineSearch::step(function, dx, x);
        VectorXd x1 = x;
        VectorXd g1 = function->getGradient(x);

        VectorXd deltaX = x1 - x0;
        VectorXd y = g1 - g0;

        MatrixXd I = MatrixXd::Identity(x.rows(), x.rows());
        MatrixXd A = (I - (deltaX * y.transpose()) / (y.transpose() * deltaX));
        MatrixXd B = (I - (y * deltaX.transpose()) / (y.transpose() * deltaX));
        MatrixXd C = (deltaX * deltaX.transpose()) / (y.transpose() * deltaX);
        inverseHessian = A * inverseHessian * B + C;
    }

public:
    MatrixXd inverseHessian;

private:
    void initializeHessian(const ObjectiveFunction *function, const VectorXd &x) {
        inverseHessian = MatrixXd::Identity(x.rows(), x.rows());
    }
};


