#include <iosfwd>
#include <vector>
#include <cppad/cg.hpp>

using namespace CppAD;
using namespace CppAD::cg;

using CGD = CG<double>;
using ADCG = AD<CGD>;


template <typename T>
T d_mid(const T* a, const T* b, const T* c, const T* d) {
	const T dx = 0.5*(*a + *b - *c - *d);
	const T dy = 0.5*(*(a+1) + *(b+1) - *(c+1) - *(d+1));
	return sqrt((dx*dx) + (dy*dy));
}

template <typename T>
T l_mid(const T* a, const T* b, const T* c, const T* d) {
	const T dx = 0.5*(*a + *c - *b - *d);
	const T dy = 0.5*(*(a+1) + *(c+1) - *(b+1) - *(d+1));
	return sqrt((dx*dx) + (dy*dy));
}

template <typename T>
T f(const T d, const T d0) {
	const T d_sq = d*d, d0_sq = d0*d0;
	const T v = 3*d_sq*d_sq -8*d0*d*d_sq + 6*d0_sq*d_sq - d0_sq*d0_sq; 
	return v/12;
}

template <int order>
void potential(const double* a, const double* b, const double* c, const double* d, const double d0, double *y);

template <typename T>
T limit_l(const T l, const T logi_k) {
	return 2 /(1+exp(-logi_k * l))-1;
}

template <typename T>
inline void potential(const T* a, const T* b, const T* c, const T* d, const T d0, const T logi_k, T *y) {
	const T dm = d_mid(a, b, c, d);
	const T lm = l_mid(a, b, c, d);
	*y = f(dm, d0) * limit_l(lm, logi_k);
}

template <typename T>
void vertex_line_potential(const T* p, const T* a, const T* b, T* y, T d0) {
	const T abx = b[0]-a[0], aby = b[1]-a[1];
	const T apx = p[0]-a[0], apy = p[1]-a[1];
	const T l_ab = sqrt(pow(abx, 2) + pow(aby, 2));
	const T area = apx*aby - apy*abx;
	const T dist = abs(area / l_ab);
	const T potential = -pow(dist-d0, 2) * log(dist/d0);
	*y = potential;
}


int main() {

	constexpr int N = 10;
	CppAD::vector<ADCG> x(N);

	auto *a = x.data(), *b = x.data()+2, *c = x.data()+4, *d = x.data()+6, *d0 = x.data()+8, *logi_k = x.data()+9;

	for (int i = 0; i < N; ++i)
		x[i] = 0;
	*d0 = 1;
	*logi_k = 1;

	Independent(x);
	CppAD::vector<ADCG> y(1);

	potential(a, b, c, d, *d0, *logi_k, y.data());
	ADFun<CGD> fun(x, y);
	CodeHandler<double> handler;

	CppAD::vector<CGD> indVars(N);
	handler.makeVariables(indVars);

	LanguageC<double> langC("double");
	LangCDefaultVariableNameGenerator<double> nameGen;

	std::ostringstream code;

# if JACOBIAN
	CppAD::vector<CGD> jac = fun.Jacobian(indVars);
	handler.generateCode(code, langC, jac, nameGen);
# else
	CppAD::vector<CGD> hess = fun.Hessian(indVars, 0);
	handler.generateCode(code, langC, hess, nameGen);
#endif

	std::cout << code.str();
}
