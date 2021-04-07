template <typename T, int dim>
using Vector = Eigen::Matrix<T, dim, 1, 0, dim, 1>;

template <typename T, int n, int m>
using Matrix = Eigen::Matrix<T, n, m, 0, n, m>;

template <typename T>
using TV3 = Vector<T, 3>;

template <typename T>
using TV2 = Vector<T, 2>;

template <typename T>
using VectorXT = Matrix<T, Eigen::Dynamic, 1>;

