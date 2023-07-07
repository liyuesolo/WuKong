#include <Eigen/Core>

template<typename T, int dim>
using Vector = Eigen::Matrix<T, dim, 1, 0, dim, 1>;

template<typename T, int n, int m>
using Matrix = Eigen::Matrix<T, n, m, 0, n, m>;

using T = double;

using TV = Vector<double, 2>;
using TV3 = Vector<double, 3>;
using TM = Matrix<double, 2, 2>;
using IV = Vector<int, 2>;
using IV3 = Vector<int, 3>;
using IV4 = Vector<int, 4>;

using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
using MatrixXT = Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
using MatrixXi = Matrix<int, Eigen::Dynamic, Eigen::Dynamic>;
using VectorXi = Vector<int, Eigen::Dynamic>;
