#include "../include/constants.hpp"

#include <iosfwd>
#include <vector>
#include <cppad/cg.hpp>

using namespace CppAD;
using namespace CppAD::cg;

using CGD = CG<double>;
using ADCG = AD<CGD>;



template <typename T>
inline void line_length(const T* a, const T* b, T* y) {
	const T abx = b[0]-a[0], aby = b[1]-a[1];
	*y = sqrt(pow(abx, 2) + pow(aby, 2));
}


int main() {

	constexpr int N = 4;
	CppAD::vector<ADCG> x(N);
	CppAD::vector<ADCG> y(1);

	Independent(x);
	line_length(x.data(), x.data()+2, y.data());
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
