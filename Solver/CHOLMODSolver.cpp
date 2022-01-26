//
//  CHOLMODSolver.cpp
//  Noether
//
//  Created by Minchen Li on 6/22/18.
//

#include "CHOLMODSolver.hpp"
#include "SuiteSparseQR.hpp"
// #include "getRSS.hpp"
#include <iostream>

namespace Noether {

template <typename StorageIndex>
CHOLMODSolver<StorageIndex>::CHOLMODSolver(void)
{
    cholmod_l_defaults(&cm);
    cholmod_l_start(&cm);
    cm.supernodal = CHOLMOD_SUPERNODAL;
    A = NULL;
    L = NULL;
    b = NULL;
    x_cd = y_cd = NULL;

    Ai = Ap = Ax = NULL;
    bx = NULL;
    solutionx = x_cdx = y_cdx = NULL;
}

template <typename StorageIndex>
CHOLMODSolver<StorageIndex>::~CHOLMODSolver(void)
{
    if (A) {
        A->i = Ai;
        A->p = Ap;
        A->x = Ax;
        cholmod_l_free_sparse(&A, &cm);
    }

    cholmod_l_free_factor(&L, &cm);

    if (b) {
        b->x = bx;
        cholmod_l_free_dense(&b, &cm);
    }

    if (x_cd) {
        x_cd->x = x_cdx;
        cholmod_l_free_dense(&x_cd, &cm);
    }

    if (y_cd) {
        y_cd->x = y_cdx;
        cholmod_l_free_dense(&y_cd, &cm);
    }

    cholmod_l_finish(&cm);
}

template <typename StorageIndex>
void CHOLMODSolver<StorageIndex>::set_pattern(std::vector<StorageIndex>& ptr, std::vector<StorageIndex>& col, std::vector<double>& value)
{
    numRows = ptr.size() - 1;
    if (!A) {
        A = cholmod_l_allocate_sparse(numRows, numRows, ptr.back(),
            true, true, -1, CHOLMOD_REAL, &cm);
        Ax = A->x;
        Ap = A->p;
        Ai = A->i;
        // -1: upper right part will be ignored during computation

        A->p = ptr.data();
        A->i = col.data();
        A->x = value.data();
    }
}

template <typename StorageIndex>
void CHOLMODSolver<StorageIndex>::set_pattern(Eigen::SparseMatrix<double, Eigen::RowMajor, StorageIndex>& mat)
{
    numRows = mat.rows();
    if (!A) {
        A = cholmod_l_allocate_sparse(numRows, numRows, mat.nonZeros(),
            true, true, -1, CHOLMOD_REAL, &cm);
        Ax = A->x;
        Ap = A->p;
        Ai = A->i;

        // -1: upper right part will be ignored during computation

        A->p = mat.outerIndexPtr();
        A->i = mat.innerIndexPtr();
        A->x = mat.valuePtr();
    }
}

template <typename StorageIndex>
void CHOLMODSolver<StorageIndex>::analyze_pattern(void)
{
    cholmod_l_free_factor(&L, &cm);
    L = cholmod_l_analyze(A, &cm);
}

template <typename StorageIndex>
bool CHOLMODSolver<StorageIndex>::factorize(void)
{
    cholmod_l_factorize(A, L, &cm);
    return cm.status != CHOLMOD_NOT_POSDEF;
}

template <typename StorageIndex>
void CHOLMODSolver<StorageIndex>::solve(double* rhs, double* result, bool spd)
{
    //TODO: directly point to rhs?
    if (!b) {
        b = cholmod_l_allocate_dense(numRows, 1, numRows, CHOLMOD_REAL, &cm);
        bx = b->x;
    }
    b->x = rhs;
    cholmod_dense* x;
    if (spd) {
        x = cholmod_l_solve(CHOLMOD_A, L, b, &cm);
    }
    else
    {
        A->stype = 0;
        x = SuiteSparseQR<double> (A, b, &cm);
    }
    memcpy(result, x->x, numRows * sizeof(double));
    cholmod_l_free_dense(&x, &cm);
}



template <typename StorageIndex>
void CHOLMODSolver<StorageIndex>::outputFactorization(const std::string& filePath)
{
    cholmod_sparse* spm = cholmod_l_factor_to_sparse(L, &cm);

    FILE* out = fopen(filePath.c_str(), "w");
    assert(out);

    cholmod_l_write_sparse(out, spm, NULL, "", &cm);

    fclose(out);
}

template <typename StorageIndex>
void CHOLMODSolver<StorageIndex>::outputMatrix(const std::string& filePath)
{

    FILE* out = fopen(filePath.c_str(), "w");
    assert(out);
    A->stype = 0;
    cholmod_l_write_sparse(out, A, NULL, "", &cm);
    A->stype = -1;
    fclose(out);
}

template <typename StorageIndex>
void CHOLMODSolver<StorageIndex>::readMatrix(const std::string& filePath)
{
    if (A) {
        A->i = Ai;
        A->p = Ap;
        A->x = Ax;
        cholmod_l_free_sparse(&A, &cm);
    }
    FILE* out = fopen(filePath.c_str(), "r");
    assert(out);
    A = cholmod_l_read_sparse(out, &cm);
    A->stype = -1;
    numRows = A->nrow;
    fclose(out);
}

template <typename StorageIndex>
void CHOLMODSolver<StorageIndex>::multiply(const double* x,
    double* Ax)
{
    if (!x_cd) {
        x_cd = cholmod_l_allocate_dense(numRows, 1, numRows,
            CHOLMOD_REAL, &cm);
        x_cdx = x_cd->x;
    }
    x_cd->x = (void*)x;

    if (!y_cd) {
        y_cd = cholmod_l_allocate_dense(numRows, 1, numRows,
            CHOLMOD_REAL, &cm);
        y_cdx = y_cd->x;
    }
    y_cd->x = (void*)Ax;

    double alpha[2] = { 1.0, 1.0 }, beta[2] = { 0.0, 0.0 };

    cholmod_l_sdmult(A, 0, alpha, beta, x_cd, y_cd, &cm);
}

template class CHOLMODSolver<long>;
template class CHOLMODSolver<int>;
} // namespace Noether
