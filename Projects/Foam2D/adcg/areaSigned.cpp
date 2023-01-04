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
T triangleAreaSigned(const std::vector<T>& x) {
	// vertices is a vector of vertices stored as 2d coordinates
	T a_b_h = x[2] - x[0], a_b_v = x[3] - x[1];
	T a_c_h = x[4] - x[0], a_c_v = x[5] - x[1];

	T area = (a_b_h * a_c_v - a_c_h * a_b_v)/2;
	return area;
}


int main() {

	std::vector<ADCG> x(6);
	Independent(x);

	std::vector<ADCG> y(1);

	y[0] = triangleAreaSigned(x);

	ADFun<CGD> fun(x, y);

	CodeHandler<double> handler;

	CppAD::vector<CGD> indVars(6);
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
