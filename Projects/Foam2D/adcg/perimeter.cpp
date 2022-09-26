#include "../include/constants.hpp"

#include <iosfwd>
#include <array>
#include <cppad/cg.hpp>
//#include <cppad/example/cppad_eigen.hpp>


using namespace CppAD;
using namespace CppAD::cg;

using CGD = CG<double>;
using ADCG = AD<CGD>;

template <typename T>
// vertices are stored as x1, y1, x2, ..., xN_SEGMENTS, yN_SEGMENTS
T LineLength(const std::array<T,4>& vertices) {
	// vertices is an array of vertices stored as 2d coordinates
	T horz = vertices[2] - vertices[0];
	T vert = vertices[3] - vertices[1];
	return sqrt(pow(horz, 2) + pow(vert, 2));
}


int main() {

	CppAD::vector<ADCG> x(4);

	CppAD::vector<ADCG> y(1);

	Independent(x);
	std::array<ADCG, 4> x_a = {x[0], x[1], x[2], x[3]};
	y[0] = LineLength(x_a);
	ADFun<CGD> fun(x, y);

	CodeHandler<double> handler;

	CppAD::vector<CGD> indVars(4);
	handler.makeVariables(indVars);


	LanguageC<double> langC("double");
	LangCDefaultVariableNameGenerator<double> nameGen;

	std::ostringstream code;

#define JACOBIAN 0

# if JACOBIAN
	CppAD::vector<CGD> jac = fun.Jacobian(indVars);
	handler.generateCode(code, langC, jac, nameGen);
# else
	CppAD::vector<CGD> hess = fun.Hessian(indVars, 0);
	handler.generateCode(code, langC, hess, nameGen);
#endif

	std::cout << code.str();
}
