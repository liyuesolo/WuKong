#pragma once

#include "ObjectiveFunction.h"
#include "Minimizer.h"

class GradientDescentFixedStep : public Minimizer {
public:
    GradientDescentFixedStep(int maxIterations = 100, double solveResidual = 1e-5)
            : maxIterations(maxIterations), solveResidual(solveResidual) {
    }

    int getLastIterations() { return lastIterations; }

    virtual bool minimize(const ObjectiveFunction *function, VectorXd &x) {

        bool optimizationConverged = false;

        VectorXd dx(x.size());

        int i = 0;
        for (; i < maxIterations; i++) {
            dx.setZero();
            computeSearchDirection(function, x, dx);

            if (dx.norm() < solveResidual) {
                optimizationConverged = true;
                break;
            }

            step(function, dx, x);
        }

        lastIterations = i;

        return optimizationConverged;
    }

public:
    virtual void computeSearchDirection(const ObjectiveFunction *function, const VectorXd &x, VectorXd &dx) {
        function->addGradientTo(x, dx);
    }

    // Given the objective `function` and the search direction `x`, update the candidate `x`
    virtual void step(const ObjectiveFunction *function, const VectorXd &dx, VectorXd &x) {
        x = x - stepSize * dx;
    }

public:
    int maxIterations = 1;
    double solveResidual = 1e-5;
    double stepSize = 0.001;

    // some stats about the last time `minimize` was called
    int lastIterations = -1;
};


class GradientDescentLineSearch : public GradientDescentFixedStep {
public:
    double alpha_start = 1.;

public:
    GradientDescentLineSearch(int maxIterations = 100, double solveResidual = 1e-5, int maxLineSearchIterations = 15)
            : GradientDescentFixedStep(maxIterations, solveResidual), maxLineSearchIterations(maxLineSearchIterations) {
    }

public:
    virtual void step(const ObjectiveFunction *function, const VectorXd &dx, VectorXd &x) {
        double alpha_nominal = alpha_start;
        double O0 = function->evaluate(x);
        double gradnorm = function->getGradient(x).norm();
        std::cout << "Optimization step gradient norm: " << gradnorm << ", Objective: " << O0 << ", Alpha: "
                  << alpha_start << std::endl;
        VectorXd x_cand = x;

        double Omin = O0;
        for (int i = 0; i < maxLineSearchIterations; ++i) {
            double alpha = alpha_nominal * pow(.5, i);
            x_cand = x - alpha * dx;
            double O = function->evaluate(x_cand);
            std::cout << "Line search iteration dx norm: " << dx.norm() << ", Objective: " << O << ", Previous: " << O0
                      << std::endl;
            if (O < O0) {
                // TODO: Logan
//                double alpha2 = alpha * .5;
//                VectorXd x_cand2 = x - alpha2 * dx;
//                if (function->evaluate(x_cand2) < function->evaluate(x_cand)) x_cand = x_cand2;
                // TODO: Logan
                x = x_cand;
//                alpha_start = std::min(2 * alpha, 1.0);
//                std::cout << "new alpha " << alpha_start << std::endl;
                return;
            }
//            if (O < Omin) {
//                Omin = O;
//                x = x_cand;
//            }
        }
    }

public:
    int maxLineSearchIterations = 15;
};
