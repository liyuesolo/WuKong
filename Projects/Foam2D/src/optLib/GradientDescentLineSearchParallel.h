#pragma once

#include "ObjectiveFunction.h"
#include "GradientDescentMinimizer.h"
#include <tbb/tbb.h>

class GradientDescentLineSearchParallel : public GradientDescentLineSearch {
public:
    GradientDescentLineSearchParallel(int maxIterations = 100, double solveResidual = 1e-5, int maxLineSearchIterations = 15)
            : GradientDescentLineSearch(maxIterations, solveResidual, maxLineSearchIterations) {
    }

    double alpha_nominal = 1.;

public:
    virtual void step(const ObjectiveFunction *function, const VectorXd &dx, VectorXd &x) {
        tbb::task_group g;
        double O0;
        g.run([&]{O0 = function->evaluate(x);});

        VectorXd obj = VectorXd::Zero(maxLineSearchIterations);
        tbb::parallel_for(0, maxLineSearchIterations, [&](int i) {
            double alpha = alpha_nominal * pow(.5, i - 3);
            VectorXd x_cand = x - alpha * dx;
            obj(i) = function->evaluate(x_cand);
//            std::cout << i << std::endl;
        });
        g.wait();

        int idx;
        if (obj.minCoeff(&idx) < O0) {
            alpha_nominal = alpha_nominal * pow(.5, idx - 3);
            x = x - alpha_nominal * dx;
        }
        else {
            alpha_nominal = alpha_nominal * pow(.5, maxLineSearchIterations - 1);
        }
        std::cout << "New alpha " << alpha_nominal << std::endl;
    }
};
