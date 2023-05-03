#ifndef NEURAL_ELEMENT_H
#define NEURAL_ELEMENT_H

#include <iomanip>

#include "VecMatDef.h"

#include "FEMSolver.h"
template <int dim>
class FEMSolver;

template <int dim>
class NeuralElement
{
public:
    using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
    using VectorXi = Vector<int, Eigen::Dynamic>;
    using TV = Vector<T, dim>;
    using TV6 = Vector<T, 6>;
    using TV2 = Vector<T, 2>;
    using IV = Vector<int, dim>;
    using IV3 = Vector<int, 3>;
    using IV4 = Vector<int, 4>;
    using TV3 = Vector<T, 3>;
    using TM = Matrix<T, dim, dim>;
    using TM3 = Matrix<T, 3, 3>;
    using TM32 = Matrix<T, 3, 2>;
    using TM2 = Matrix<T, 2, 2>;
    using MatrixXT = Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using MatrixXi = Matrix<int, Eigen::Dynamic, Eigen::Dynamic>;

    using Entry = Eigen::Triplet<T>;

    using QuadEleNodes = Matrix<T, 6, 2>;
    using QuadEleIdx = Vector<int, 6>;

    using EleNodes = Matrix<T, 3, dim>;
    using EleIdx = VectorXi;

public:
    FEMSolver<dim>& solver;

public:
    void generateBeamSceneTrainingData(const std::string& folder);
public:
    NeuralElement(FEMSolver<dim>& _solver) : solver(_solver) {}
    ~NeuralElement() {}
};

#endif