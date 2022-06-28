#include <fem.hpp> // Fortran EMulation library of fable module

namespace lbfgsb
{

using namespace fem::major_types;
using fem::common;

void
setulb(
	common& cmn,
	int const& n,
	int const& m,
	arr_ref<double> x,
	arr_cref<double> l,
	arr_cref<double> u,
	arr_cref<int> nbd,
	double& f,
	arr_ref<double> g,
	double const& factr,
	double const& pgtol,
	arr_ref<double> wa,
	arr_ref<int> iwa,
	str_ref task,
	int const& iprint,
	str_ref csave,
	arr_ref<bool> lsave,
	arr_ref<int> isave,
	arr_ref<double> dsave);

//void mainlb(
//	common& cmn,
//	int const& n,
//	int const& m,
//	arr_ref<double> x,
//	arr_cref<double> l,
//	arr_cref<double> u,
//	arr_cref<int> nbd,
//	double& f,
//	arr_ref<double> g,
//	double const& factr,
//	double const& pgtol,
//	arr_ref<double, 2> ws,
//	arr_ref<double, 2> wy,
//	arr_ref<double, 2> sy,
//	arr_ref<double, 2> ss,
//	arr_ref<double, 2> wt,
//	arr_ref<double, 2> wn,
//	arr_ref<double, 2> snd,
//	arr_ref<double> z,
//	arr_ref<double> r,
//	arr_ref<double> d,
//	arr_ref<double> t,
//	arr_ref<double> xp,
//	arr_ref<double> wa,
//	arr_ref<int> index,
//	arr_ref<int> iwhere,
//	arr_ref<int> indx2,
//	str_ref task,
//	int const& iprint,
//	str_ref csave,
//	arr_ref<bool> lsave,
//	arr_ref<int> isave,
//	arr_ref<double> dsave);

}