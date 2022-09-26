#include "../include/constants.hpp"

#include <iosfwd>
#include <vector>
#include <cppad/cg.hpp>
//#include <cppad/example/cppad_eigen.hpp>


using namespace CppAD;
using namespace CppAD::cg;

using CGD = CG<double>;
using ADCG = AD<CGD>;

template <typename T>
// vertices are stored as x0, y0, x1, ..., y2
T triangleArea(const std::vector<T>& x) {
	// vertices is a vector of vertices stored as 2d coordinates
	T a_h = x[4] - x[2], a_v = x[5] - x[3];
	T b_h = x[4] - x[0], b_v = x[5] - x[1];
	T c_h = x[2] - x[0], c_v = x[3] - x[1];

	T area = abs(a_h * b_v - b_h * a_v)/2;
	return area;
}


int main() {

	std::vector<ADCG> x(6);
	Independent(x);

	std::vector<ADCG> y(1);

	y[0] = triangleArea(x);

	ADFun<CGD> fun(x, y);

	CodeHandler<double> handler;

	CppAD::vector<CGD> indVars(6);
	handler.makeVariables(indVars);

	LanguageC<double> langC("double");
	LangCDefaultVariableNameGenerator<double> nameGen;

	std::ostringstream code;
#define JACOBIAN 1

# if JACOBIAN
	CppAD::vector<CGD> jac = fun.Jacobian(indVars);
	handler.generateCode(code, langC, jac, nameGen);
# else
	CppAD::vector<CGD> hess = fun.Hessian(indVars, 0);
	handler.generateCode(code, langC, hess, nameGen);
#endif
	std::cout << code.str();
}
