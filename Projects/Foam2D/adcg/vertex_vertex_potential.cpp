#include <iosfwd>
#include <vector>
#include <cppad/cg.hpp>

using namespace CppAD;
using namespace CppAD::cg;

using CGD = CG<double>;
using ADCG = AD<CGD>;



template <typename T>
void vertex_vertex_potential(const T* p, const T* a, const T* b, T* y, T* d0) {
# if VERTA
	const T dist = sqrt(pow(a[0]-p[0], 2) + pow(a[1]-p[1], 2));
# else
	const T dist = sqrt(pow(b[0]-p[0], 2) + pow(b[1]-p[1], 2));
# endif
	const T potential = -pow(dist-*d0, 2) * log(dist/ *d0);
	*y = potential;
}


int main() {

	constexpr int N = 7;
	CppAD::vector<ADCG> x(N);
	x[0] = 0.5;
	x[1] = 1;
	x[2] = 0;
	x[3] = 0;
	x[4] = 1;
	x[5] = 0;
	x[6] = 1;
	CppAD::vector<ADCG> y(1);

	Independent(x);

	vertex_vertex_potential(x.data(), x.data()+2, x.data()+4, y.data(), x.data()+6);
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
