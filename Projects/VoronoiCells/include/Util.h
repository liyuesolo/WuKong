#ifndef UTIL_H
#define UTIL_H

#include "VecMatDef.h"



template <typename _type, int _n_col>
void vectorToIGLMatrix(const Matrix<_type, Eigen::Dynamic, 1>& vec, 
    Matrix<_type, Eigen::Dynamic, Eigen::Dynamic>& mat)
{
    int n_rows = vec.rows() / _n_col;
    mat.resize(n_rows, _n_col);
    for (int i = 0; i < n_rows; i++)
        mat.row(i) = vec.template segment<_n_col>(i * _n_col);
}

template <typename _type, int _n_col>
void iglMatrixFatten(const Matrix<_type, Eigen::Dynamic, Eigen::Dynamic>& mat, 
    Matrix<_type, Eigen::Dynamic, 1>& vec)
{
    int n_rows = mat.rows();
    vec.resize(n_rows * _n_col);
    for (int i = 0; i < n_rows; i++)
        vec.template segment<_n_col>(i * _n_col) = mat.row(i);
}

void triangulatePointCloud(const Eigen::VectorXd& points, Eigen::VectorXi& triangle_indices);

#endif