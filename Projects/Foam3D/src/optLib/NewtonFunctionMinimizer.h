#pragma once

#include "ObjectiveFunction.h"
#include "GradientDescentMinimizer.h"
#include <chrono>

#define PRINT_INTERMEDIATE_TIMES_NEWTON true
#define PRINT_TOTAL_TIME_NEWTON true

static void
printTimeNewton(std::chrono::high_resolution_clock::time_point tstart, std::string description = "",
                bool final = false) {
    if (PRINT_INTERMEDIATE_TIMES_NEWTON || (final && PRINT_TOTAL_TIME_NEWTON)) {
        const auto tcurr = std::chrono::high_resolution_clock::now();
        std::cout << description << "Time: "
                  << std::chrono::duration_cast<std::chrono::microseconds>(tcurr - tstart).count() * 1.0e-6
                  << std::endl;
    }
}

class NewtonFunctionMinimizer : public GradientDescentLineSearch {

public:
    NewtonFunctionMinimizer(int maxIterations = 100, double solveResidual = 0.0001, int maxLineSearchIterations = 15)
            : GradientDescentLineSearch(maxIterations, solveResidual, maxLineSearchIterations) {}

    virtual ~NewtonFunctionMinimizer() {}

public:
    Solver solver;
    int nMaxStabSteps = 10;
    double stabValue = 1e-4;

    virtual void computeSearchDirection(const ObjectiveFunction *function, const VectorXd &x, VectorXd &dx) {
        SparseMatrixd H;
        function->getHessian(x, H);
        VectorXd g = function->getGradient(x);

        auto tstart = std::chrono::high_resolution_clock::now();
        solver.compute(H);
        dx = solver.solve(g);
        printTimeNewton(tstart, "Linear solve ", true);

        if (dx.dot(g) <= 0) {
            double currStabValue = stabValue;
            for (int i = 0; i < nMaxStabSteps; ++i) {
                for (int j = 0; j < x.size(); ++j) { H.coeffRef(j, j) += currStabValue; }
                currStabValue *= 10;
                solver.compute(H);
                dx = solver.solve(g);
                if (dx.dot(g) > 0) { break; }
            }
            std::cout << "Regularizer " << currStabValue << std::endl;
        }
    }

public:
    SparseMatrixd hessian;
    std::vector<Triplet<double>> hessianEntries;
    double reg = 1.0;
};
