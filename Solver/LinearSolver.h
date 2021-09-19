#pragma once
#include <Ziran/CS/Util/Forward.h>
#include <Ziran/CS/Util/Timer.h>

namespace ZIRAN {
template <class StiffnessMatrix, class T>
void DirectSolver(StiffnessMatrix & A, Eigen::Ref<Matrix<T, Eigen::Dynamic, 1>> x, Eigen::Ref<Matrix<T, Eigen::Dynamic, 1>> residual, bool spd=true);

template <class StiffnessMatrix, class T>
bool IterativeSolver(const StiffnessMatrix& A, Eigen::Ref<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> residual, Eigen::Ref<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> x, int block_size, T linear_tol, bool spd);

template <class StiffnessMatrix>
void outputSparse(std::string filename, const StiffnessMatrix& A);

template <class T>
void outputDense(std::string filename, Eigen::Ref<const Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> b);

} // namespace ZIRAN