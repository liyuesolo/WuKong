#include "../include/constants.hpp"

#include <iosfwd>
#include <vector>
#include <cppad/cg.hpp>

using namespace CppAD;
using namespace CppAD::cg;

using CGD = CG<double>;
using ADCG = AD<CGD>;



template <typename T>
inline void triangle_area_signed(const T* a, const T* b, const T* c, T* y) {
	const T abx = b[0] - a[0], aby = b[1] - a[1];
	const T acx = c[0] - a[0], acy = c[1] - a[1];
	*y = abx * acy - aby * acx;
}


int main() {

	constexpr int N = 6;
	CppAD::vector<ADCG> x(N);
	CppAD::vector<ADCG> y(1);

	Independent(x);
	triangle_area_signed(x.data(), x.data()+2, x.data()+4, y.data());
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
