#include "LinearSolver.h"
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/CholmodSupport>

#include <amgcl/adapter/eigen.hpp>
#include <amgcl/backend/builtin.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/coarsening/plain_aggregates.hpp>
#include <amgcl/coarsening/aggregation.hpp>
#include <amgcl/coarsening/ruge_stuben.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/relaxation/gauss_seidel.hpp>
#include <amgcl/solver/cg.hpp>
#include <amgcl/solver/bicgstab.hpp>
#include <amgcl/solver/gmres.hpp>
#include <amgcl/profiler.hpp>
#include <amgcl/adapter/block_matrix.hpp>
#include <amgcl/value_type/static_matrix.hpp>
#include <amgcl/io/mm.hpp>
#include <amgcl/relaxation/chebyshev.hpp>
#include <amgcl/solver/runtime.hpp>
#include <amgcl/coarsening/runtime.hpp>
#include <amgcl/relaxation/runtime.hpp>
#include <amgcl/preconditioner/runtime.hpp>
#ifdef ENABLE_AMGCL_CUDA
#include <amgcl/backend/vexcl.hpp>
#include <amgcl/backend/vexcl_static_matrix.hpp>
#endif
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/adapter/reorder.hpp>

#include <boost/program_options.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/range/iterator_range.hpp>
#include <boost/preprocessor/seq/for_each.hpp>


// #ifdef USE_CHOLMOD
#include "CHOLMODSolver.hpp"
// #endif

namespace ZIRAN {

template <class StiffnessMatrix, class T>
void DirectSolver(StiffnessMatrix & A, Eigen::Ref<Matrix<T, Eigen::Dynamic, 1>> x, Eigen::Ref<Matrix<T, Eigen::Dynamic, 1>> residual, bool spd)
{
    ZIRAN_TIMER();
    using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
// #ifdef USE_CHOLMOD
    Noether::CHOLMODSolver<typename StiffnessMatrix::StorageIndex> solver;
    solver.set_pattern(A);
    if (spd) {
        solver.analyze_pattern();
        solver.factorize();
    }
    solver.solve(residual.data(), x.data(), spd);
// #else
//     if (spd) {
//         Eigen::SimplicialLDLT<Eigen::SparseMatrix<T, Eigen::ColMajor, typename StiffnessMatrix::StorageIndex>> solver;
//         solver.compute(A);
//         const auto& rhs = Eigen::Map<const VectorXT>(residual.data(), residual.size());
//         Eigen::Map<VectorXT>(x.data(), x.size()) = solver.solve(rhs);
//     }
//     else {
//         Eigen::SparseLU<Eigen::SparseMatrix<T, Eigen::ColMajor, typename StiffnessMatrix::StorageIndex>> solver;
//         solver.compute(A);
//         const auto& rhs = Eigen::Map<const VectorXT>(residual.data(), residual.size());
//         Eigen::Map<VectorXT>(x.data(), x.size()) = solver.solve(rhs);
//     }
// #endif
}

template <class StiffnessMatrix, class T>
bool IterativeSolver(const StiffnessMatrix& A, Eigen::Ref<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> residual, Eigen::Ref<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> x, int block_size, T linear_tol, bool spd)
{
    x.setZero();
    boost::property_tree::ptree prm;
    prm.put("solver.tol", linear_tol); // relative
    prm.put("solver.maxiter", 200);
    prm.put("precond.class", "amg");
    prm.put("precond.relax.type", "chebyshev");
    prm.put("precond.relax.degree", 16);
    prm.put("precond.relax.power_iters", 100);
    prm.put("precond.relax.higher", 2.0f);
    prm.put("precond.relax.lower", 1.0f / 120.0f);
    prm.put("precond.relax.scale", true);
    prm.put("precond.max_levels", 6);
    prm.put("precond.direct_coarse", false);
    prm.put("precond.ncycle", 2);
    if (spd) {
        prm.put("precond.coarsening.type", "smoothed_aggregation");
        prm.put("precond.coarsening.estimate_spectral_radius", true);
        prm.put("precond.coarsening.relax", 1.0f);
        // prm.put("precond.coarsening.power_iters", 100);
    } else {
        prm.put("precond.coarsening.type", "aggregation");
    }
    prm.put("precond.coarsening.aggr.eps_strong", 0.0);
    prm.put("precond.coarsening.aggr.block_size", block_size);
    // if (!spd) {
    prm.put("solver.type", "lgmres");
    prm.put("solver.M", 100);
    // }
    // else
        // prm.put("solver.type", "cg");

#ifdef ENABLE_AMGCL_CUDA
    typedef amgcl::backend::vexcl<T> Backend;
#else
    typedef amgcl::backend::builtin<T> Backend;
#endif

    using Solver = amgcl::make_solver<
        amgcl::runtime::preconditioner<Backend>,
        amgcl::runtime::solver::wrapper<Backend>>;

    typename Backend::params bprm;

    amgcl::adapter::reorder<> perm(A);
#ifdef ENABLE_AMGCL_CUDA
    vex::Context ctx(vex::Filter::Env);
    // std::cout << ctx << std::endl;
    bprm.q = ctx;
    Solver amgcl_solver(perm(A), prm, bprm);
#else
    Solver amgcl_solver(perm(A), prm);
#endif

    // ZIRAN_INFO(amgcl_solver.precond());

    for (int c = 0; c < residual.cols(); ++c) {
        size_t iters;
        double resid;

        std::vector<T> F(&residual(0, c), &residual(0, c) + residual.rows());
        std::vector<T> X(&x(0,c), &x(0,c) + x.rows());
        std::vector<T> tmp(A.rows());

        perm.forward(F, tmp);
        auto f_b = Backend::copy_vector(tmp, bprm);
        perm.forward(X, tmp);
        auto x_b = Backend::copy_vector(tmp, bprm);

        std::tie(iters, resid) = amgcl_solver(*f_b, *x_b);

        if (iters == prm.get<unsigned>("solver.maxiter")) {
            ZIRAN_INFO("Iterations: ", iters);
            ZIRAN_INFO("Error:      ", resid);
            return false;
        }
        else {
#ifdef ENABLE_AMGCL_CUDA
            vex::copy(*x_b, tmp);
#else
            std::copy(&(*x_b)[0], &(*x_b)[0] + x.rows(), tmp.data());
#endif
            T* xptr = &x(0,c);
            perm.inverse(tmp, xptr);
            return true;
        }
    }
}

template <class StiffnessMatrix>
void outputSparse(std::string filename, const StiffnessMatrix& A)
{
    amgcl::io::mm_write(filename, A);
}

template <class T>
void outputDense(std::string filename, Eigen::Ref<const Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> b)
{
    amgcl::io::mm_write(filename, b.data(), b.size());
}

template void DirectSolver(Eigen::SparseMatrix<double, Eigen::RowMajor, long>& A, Eigen::Ref<Matrix<double, Eigen::Dynamic, 1>> x, Eigen::Ref<Matrix<double, Eigen::Dynamic, 1>> residual, bool spd);
template bool IterativeSolver(const Eigen::SparseMatrix<double, Eigen::RowMajor, long>& A, Eigen::Ref<Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> residual, Eigen::Ref<Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> x, int block_size, double linear_tol, bool spd);
template void outputSparse(std::string, const Eigen::SparseMatrix<double, Eigen::RowMajor, long>&);
template void outputDense(std::string, const Eigen::Ref<const Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>);

// template void DirectSolver(Eigen::SparseMatrix<double, Eigen::RowMajor, int>& A, Eigen::Ref<Matrix<double, Eigen::Dynamic, 1>> x, Eigen::Ref<Matrix<double, Eigen::Dynamic, 1>> residual, bool spd);
// template bool IterativeSolver(const Eigen::SparseMatrix<double, Eigen::RowMajor, int>& A, Eigen::Ref<Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> residual, Eigen::Ref<Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> x, int block_size, double linear_tol, bool spd);
// template void outputSparse(std::string, const Eigen::SparseMatrix<double, Eigen::RowMajor, int>&);
}