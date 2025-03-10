#include "../include/FEMSolver.h"
#include "../include/L2diff.h"
#include "pallas/differential_evolution.h"
#include <stdexcept>
#include <omp.h>

template <typename Scalar, int size>
void ProjectSPD_IGL(Eigen::Matrix<Scalar, size, size>& symMtr, AScalar eps = 0.)
{
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Scalar, size, size>> eigenSolver(symMtr);
    if (eigenSolver.eigenvalues()[0] >= 0.0) {
        return;
    }
    Eigen::DiagonalMatrix<Scalar, size> D(eigenSolver.eigenvalues());
    int rows = ((size == Eigen::Dynamic) ? symMtr.rows() : size);
    for (int i = 0; i < rows; i++) {
        if (D.diagonal()[i] <= 0.0) {
            D.diagonal()[i] = eps;
        }
        else {
            break;
        }
    }
    symMtr = eigenSolver.eigenvectors() * D * eigenSolver.eigenvectors().transpose();
}

template <int dim>
void FEMSolver<dim>::IMLS_local_gradient_to_global_gradient(
    VectorXa& local_grad,
    Eigen::VectorXi ids,
    int dim1,
    Eigen::VectorXd& grad)
{
    for (int i = 0; i < ids.size(); i++) 
    {
        if(ids(i) < 0) continue;
        grad.segment(dim * ids(i), dim) += local_grad.segment(dim * i, dim);
    }
}

template <int dim>
void FEMSolver<dim>:: IMLS_local_gradient_to_global_gradient_and_multiply(
    VectorXa& local_grad,
    Eigen::VectorXi ids,
    int dim1,
    AScalar scale,
    VectorXa& global_vector,
    std::vector<Eigen::Triplet<double>>& triplets)
{
    for (int i = 0; i < ids.size(); i++) {
        if(ids(i) < 0) continue;
        for (int j = 0; j < global_vector.size(); j++) {
            if(global_vector(j) == 0) continue;
            for (int k = 0; k < dim; k++) {
                triplets.emplace_back(
                    dim * ids(i) + k, j,
                    scale*local_grad(dim * i + k)*global_vector(j));
            }
        }
    }

    for (int i = 0; i < global_vector.size(); i++) {
        if(global_vector(i) == 0) continue;
        for (int j = 0; j < ids.size(); j++) {
            if(ids(j) < 0) continue;
            for (int k = 0; k < dim; k++) {
                triplets.emplace_back(
                    i,  dim * ids(j) + k,
                     scale*local_grad(dim*j+k)*global_vector(i));
            }
        }
    }
}

template <int dim>
void FEMSolver<dim>::IMLS_local_hessian_to_global_triplets(
    VectorXa& local_hessian,
    Eigen::VectorXi ids,
    int dim1,
    std::vector<Eigen::Triplet<double>>& triplets)
{
    for (int i = 0; i < ids.size(); i++) {
        if(ids(i) < 0) continue;
        for (int j = 0; j < ids.size(); j++) {
            if(ids(j) < 0) continue;
            for (int k = 0; k < dim; k++) {
                for (int l = 0; l < dim; l++) {
                    triplets.emplace_back(
                        dim * ids(i) + k, dim * ids(j) + l,
                        local_hessian(dim*ids.size()*(dim * i + k)+dim * j + l));
                }
            }
        }
    }
}

template <int dim>
void FEMSolver<dim>::IMLS_local_hessian_matrix_to_global_triplets(
    MatrixXa& local_hessian,
    Eigen::VectorXi ids,
    int dim1,
    std::vector<Eigen::Triplet<double>>& triplets)
{
    for (int i = 0; i < ids.size(); i++) {
        if(ids(i) < 0) continue;
        for (int j = 0; j < ids.size(); j++) {
            if(ids(j) < 0) continue;
            for (int k = 0; k < dim; k++) {
                for (int l = 0; l < dim; l++) {
                    triplets.emplace_back(
                        dim * ids(i) + k, dim * ids(j) + l,
                        local_hessian(dim*i+k,dim*j+l));
                }
            }
        }
    }
}


template <int dim>
void FEMSolver<dim>::IMLS_vector_muliplication_to_triplets(
    VectorXa& v1,
    VectorXa& v2,
    AScalar scale,
    std::vector<Eigen::Triplet<double>>& triplets)
{
    assert(v1.size() == v2.size());
    for(int i=0; i<v1.size(); ++i)
    {
        if(v1(i) == 0) continue;
        for(int j=0; j<v2.size(); ++j)
        {
            if(v2(j) == 0) continue;
            triplets.emplace_back(
            i, j, scale*v1(i)*v2(j));
        }
    }
}

Vector24a glue_gradient(Vector1a& A, Vector1a& B, Vector24a& Adq, Vector24a& Bdq)
{
    Vector24a depsilon;

    AScalar t1 = 0.1e1 / B[0];
    AScalar t2 = pow(t1, 0.2e1);
    AScalar t3 = A[0] * t1;
    t1 = A[0] * A[0] * t1 * t2;
    t2 = t1 * (-t3 * Bdq[11] + Adq[11]);
    AScalar t4 = t1 * (-t3 * Bdq[12] + Adq[12]);
    AScalar t5 = t1 * (-t3 * Bdq[13] + Adq[13]);
    AScalar t6 = t1 * (-t3 * Bdq[14] + Adq[14]);
    AScalar t7 = t1 * (-t3 * Bdq[15] + Adq[15]);
    AScalar t8 = t1 * (-t3 * Bdq[17] + Adq[17]);
    AScalar t9 = t1 * (-t3 * Bdq[18] + Adq[18]);
    AScalar t10 = t1 * (-t3 * Bdq[19] + Adq[19]);
    AScalar t11 = t1 * (-t3 * Bdq[20] + Adq[20]);
    AScalar t12 = t1 * (-t3 * Bdq[21] + Adq[21]);
    AScalar t13 = t1 * (-t3 * Bdq[23] + Adq[23]);
    depsilon[0] = 0.3e1 * t1 * (-t3 * Bdq[0] + Adq[0]);
    depsilon[1] = 0.3e1 * t1 * (-t3 * Bdq[1] + Adq[1]);
    depsilon[2] = 0.3e1 * t1 * (-t3 * Bdq[2] + Adq[2]);
    depsilon[3] = 0.3e1 * t1 * (-t3 * Bdq[3] + Adq[3]);
    depsilon[4] = 0.3e1 * t1 * (-t3 * Bdq[4] + Adq[4]);
    depsilon[5] = 0.3e1 * t1 * (-t3 * Bdq[5] + Adq[5]);
    depsilon[6] = 0.3e1 * t1 * (-t3 * Bdq[6] + Adq[6]);
    depsilon[7] = 0.3e1 * t1 * (-t3 * Bdq[7] + Adq[7]);
    depsilon[8] = 0.3e1 * t1 * (-t3 * Bdq[8] + Adq[8]);
    depsilon[9] = 0.3e1 * t1 * (-t3 * Bdq[9] + Adq[9]);
    depsilon[10] = 0.3e1 * t1 * (-t3 * Bdq[10] + Adq[10]);
    depsilon[11] = 0.3e1 * t2;
    depsilon[12] = 0.3e1 * t4;
    depsilon[13] = 0.3e1 * t5;
    depsilon[14] = 0.3e1 * t6;
    depsilon[15] = 0.3e1 * t7;
    depsilon[16] = 0.3e1 * t1 * (-t3 * Bdq[16] + Adq[16]);
    depsilon[17] = 0.3e1 * t8;
    depsilon[18] = 0.3e1 * t9;
    depsilon[19] = 0.3e1 * t10;
    depsilon[20] = 0.3e1 * t11;
    depsilon[21] = 0.3e1 * t12;
    depsilon[22] = 0.3e1 * t1 * (-t3 * Bdq[22] + Adq[22]);
    depsilon[23] = 0.3e1 * t13;

    return depsilon;
}

Matrix24a glue_hessian(Vector1a& A, Vector1a& B, Vector24a& Adq, Vector24a& Bdq, VectorXa& Ad2q, VectorXa& Bd2q)
{
    AScalar hessian[576];

    AScalar t1 = 0.1e1 / B[0];
    AScalar t2 = pow(t1, 0.2e1);
    AScalar t3 = t1 * t2;
    AScalar t4 = A[0] * A[0];
    t1 = A[0] * t1;
    AScalar t5 = t4 * t3;
    AScalar t6 = 0.12e2 * t4 * t2;
    AScalar t7 = -0.18e2 * t1;
    t3 = A[0] * t3;
    AScalar t8 = 0.3e1 * t5 * (-t1 * Bd2q[0] + Ad2q[0]) + t3 * (t6 * Bdq[0] * Bdq[0] + t7 * Adq[0] * Bdq[0] + 0.6e1 * Adq[0] * Adq[0]);
    t2 = t4 * pow(t2, 0.2e1);
    t4 = 0.6e1 * Adq[0];
    AScalar t9 = t6 * Bdq[0];
    AScalar t10 = -0.9e1 * t2 * (Adq[0] * Bdq[1] + Bdq[0] * Adq[1]) + 0.3e1 * t5 * (-t1 * Bd2q[1] + Ad2q[1]) + t3 * (t4 * Adq[1] + t9 * Bdq[1]);
    AScalar t11 = -0.9e1 * t2 * (Adq[0] * Bdq[2] + Bdq[0] * Adq[2]) + 0.3e1 * t5 * (-t1 * Bd2q[2] + Ad2q[2]) + t3 * (t4 * Adq[2] + t9 * Bdq[2]);
    AScalar t12 = -0.9e1 * t2 * (Adq[0] * Bdq[3] + Bdq[0] * Adq[3]) + 0.3e1 * t5 * (-t1 * Bd2q[3] + Ad2q[3]) + t3 * (t4 * Adq[3] + t9 * Bdq[3]);
    AScalar t13 = -0.9e1 * t2 * (Adq[0] * Bdq[4] + Bdq[0] * Adq[4]) + 0.3e1 * t5 * (-t1 * Bd2q[4] + Ad2q[4]) + t3 * (t4 * Adq[4] + t9 * Bdq[4]);
    AScalar t14 = -0.9e1 * t2 * (Adq[0] * Bdq[5] + Bdq[0] * Adq[5]) + 0.3e1 * t5 * (-t1 * Bd2q[5] + Ad2q[5]) + t3 * (t4 * Adq[5] + t9 * Bdq[5]);
    AScalar t15 = 0.3e1 * t5 * (-t1 * Bd2q[6] + Ad2q[6]) - 0.9e1 * t2 * (Adq[0] * Bdq[6] + Bdq[0] * Adq[6]) + t3 * (t4 * Adq[6] + t9 * Bdq[6]);
    AScalar t16 = 0.3e1 * t5 * (-t1 * Bd2q[7] + Ad2q[7]) - 0.9e1 * t2 * (Adq[0] * Bdq[7] + Bdq[0] * Adq[7]) + t3 * (t4 * Adq[7] + t9 * Bdq[7]);
    AScalar t17 = 0.3e1 * t5 * (-t1 * Bd2q[8] + Ad2q[8]) - 0.9e1 * t2 * (Adq[0] * Bdq[8] + Bdq[0] * Adq[8]) + t3 * (t4 * Adq[8] + t9 * Bdq[8]);
    AScalar t18 = 0.3e1 * t5 * (-t1 * Bd2q[9] + Ad2q[9]) - 0.9e1 * t2 * (Adq[0] * Bdq[9] + Bdq[0] * Adq[9]) + t3 * (t4 * Adq[9] + t9 * Bdq[9]);
    AScalar t19 = 0.3e1 * t5 * (-t1 * Bd2q[10] + Ad2q[10]) - 0.9e1 * t2 * (Adq[0] * Bdq[10] + Bdq[0] * Adq[10]) + t3 * (t4 * Adq[10] + t9 * Bdq[10]);
    AScalar t20 = 0.3e1 * t5 * (-t1 * Bd2q[11] + Ad2q[11]) - 0.9e1 * t2 * (Adq[0] * Bdq[11] + Bdq[0] * Adq[11]) + t3 * (t4 * Adq[11] + t9 * Bdq[11]);
    AScalar t21 = 0.3e1 * t5 * (-t1 * Bd2q[12] + Ad2q[12]) - 0.9e1 * t2 * (Adq[0] * Bdq[12] + Bdq[0] * Adq[12]) + t3 * (t4 * Adq[12] + t9 * Bdq[12]);
    AScalar t22 = 0.3e1 * t5 * (-t1 * Bd2q[13] + Ad2q[13]) - 0.9e1 * t2 * (Adq[0] * Bdq[13] + Bdq[0] * Adq[13]) + t3 * (t4 * Adq[13] + t9 * Bdq[13]);
    AScalar t23 = -0.9e1 * t2 * (Adq[0] * Bdq[14] + Bdq[0] * Adq[14]) + 0.3e1 * t5 * (-t1 * Bd2q[14] + Ad2q[14]) + t3 * (t4 * Adq[14] + t9 * Bdq[14]);
    AScalar t24 = -0.9e1 * t2 * (Adq[0] * Bdq[15] + Bdq[0] * Adq[15]) + 0.3e1 * t5 * (-t1 * Bd2q[15] + Ad2q[15]) + t3 * (t4 * Adq[15] + t9 * Bdq[15]);
    AScalar t25 = -0.9e1 * t2 * (Adq[0] * Bdq[16] + Bdq[0] * Adq[16]) + 0.3e1 * t5 * (-t1 * Bd2q[16] + Ad2q[16]) + t3 * (t4 * Adq[16] + t9 * Bdq[16]);
    AScalar t26 = -0.9e1 * t2 * (Adq[0] * Bdq[17] + Bdq[0] * Adq[17]) + 0.3e1 * t5 * (-t1 * Bd2q[17] + Ad2q[17]) + t3 * (t4 * Adq[17] + t9 * Bdq[17]);
    AScalar t27 = -0.9e1 * t2 * (Adq[0] * Bdq[18] + Bdq[0] * Adq[18]) + 0.3e1 * t5 * (-t1 * Bd2q[18] + Ad2q[18]) + t3 * (t4 * Adq[18] + t9 * Bdq[18]);
    AScalar t28 = -0.9e1 * t2 * (Adq[0] * Bdq[19] + Bdq[0] * Adq[19]) + 0.3e1 * t5 * (-t1 * Bd2q[19] + Ad2q[19]) + t3 * (t4 * Adq[19] + t9 * Bdq[19]);
    AScalar t29 = -0.9e1 * t2 * (Adq[0] * Bdq[20] + Bdq[0] * Adq[20]) + 0.3e1 * t5 * (-t1 * Bd2q[20] + Ad2q[20]) + t3 * (t4 * Adq[20] + t9 * Bdq[20]);
    AScalar t30 = -0.9e1 * t2 * (Adq[0] * Bdq[21] + Bdq[0] * Adq[21]) + 0.3e1 * t5 * (-t1 * Bd2q[21] + Ad2q[21]) + t3 * (t4 * Adq[21] + t9 * Bdq[21]);
    AScalar t31 = -0.9e1 * t2 * (Adq[0] * Bdq[22] + Bdq[0] * Adq[22]) + 0.3e1 * t5 * (-t1 * Bd2q[22] + Ad2q[22]) + t3 * (t4 * Adq[22] + t9 * Bdq[22]);
    t4 = -0.9e1 * t2 * (Adq[0] * Bdq[23] + Bdq[0] * Adq[23]) + 0.3e1 * t5 * (-t1 * Bd2q[23] + Ad2q[23]) + t3 * (t4 * Adq[23] + t9 * Bdq[23]);
    t9 = 0.3e1 * t5 * (-t1 * Bd2q[25] + Ad2q[25]) + t3 * (t6 * Bdq[1] * Bdq[1] + t7 * Adq[1] * Bdq[1] + 0.6e1 * Adq[1] * Adq[1]);
    AScalar t32 = t6 * Bdq[1];
    AScalar t33 = 0.6e1 * Adq[1];
    AScalar t34 = -0.9e1 * t2 * (Adq[1] * Bdq[2] + Bdq[1] * Adq[2]) + 0.3e1 * t5 * (-t1 * Bd2q[26] + Ad2q[26]) + t3 * (t32 * Bdq[2] + t33 * Adq[2]);
    AScalar t35 = -0.9e1 * t2 * (Adq[1] * Bdq[3] + Bdq[1] * Adq[3]) + 0.3e1 * t5 * (-t1 * Bd2q[27] + Ad2q[27]) + t3 * (t32 * Bdq[3] + t33 * Adq[3]);
    AScalar t36 = -0.9e1 * t2 * (Adq[1] * Bdq[4] + Bdq[1] * Adq[4]) + 0.3e1 * t5 * (-t1 * Bd2q[28] + Ad2q[28]) + t3 * (t32 * Bdq[4] + t33 * Adq[4]);
    AScalar t37 = 0.3e1 * t5 * (-t1 * Bd2q[29] + Ad2q[29]) - 0.9e1 * t2 * (Adq[1] * Bdq[5] + Bdq[1] * Adq[5]) + t3 * (t32 * Bdq[5] + t33 * Adq[5]);
    AScalar t38 = 0.3e1 * t5 * (-t1 * Bd2q[30] + Ad2q[30]) - 0.9e1 * t2 * (Adq[1] * Bdq[6] + Bdq[1] * Adq[6]) + t3 * (t32 * Bdq[6] + t33 * Adq[6]);
    AScalar t39 = 0.3e1 * t5 * (-t1 * Bd2q[31] + Ad2q[31]) - 0.9e1 * t2 * (Adq[1] * Bdq[7] + Bdq[1] * Adq[7]) + t3 * (t32 * Bdq[7] + t33 * Adq[7]);
    AScalar t40 = 0.3e1 * t5 * (-t1 * Bd2q[32] + Ad2q[32]) - 0.9e1 * t2 * (Adq[1] * Bdq[8] + Bdq[1] * Adq[8]) + t3 * (t32 * Bdq[8] + t33 * Adq[8]);
    AScalar t41 = 0.3e1 * t5 * (-t1 * Bd2q[33] + Ad2q[33]) - 0.9e1 * t2 * (Adq[1] * Bdq[9] + Bdq[1] * Adq[9]) + t3 * (t32 * Bdq[9] + t33 * Adq[9]);
    AScalar t42 = 0.3e1 * t5 * (-t1 * Bd2q[34] + Ad2q[34]) - 0.9e1 * t2 * (Adq[1] * Bdq[10] + Bdq[1] * Adq[10]) + t3 * (t32 * Bdq[10] + t33 * Adq[10]);
    AScalar t43 = 0.3e1 * t5 * (-t1 * Bd2q[35] + Ad2q[35]) - 0.9e1 * t2 * (Adq[1] * Bdq[11] + Bdq[1] * Adq[11]) + t3 * (t32 * Bdq[11] + t33 * Adq[11]);
    AScalar t44 = 0.3e1 * t5 * (-t1 * Bd2q[36] + Ad2q[36]) - 0.9e1 * t2 * (Adq[1] * Bdq[12] + Bdq[1] * Adq[12]) + t3 * (t32 * Bdq[12] + t33 * Adq[12]);
    AScalar t45 = 0.3e1 * t5 * (-t1 * Bd2q[37] + Ad2q[37]) - 0.9e1 * t2 * (Adq[1] * Bdq[13] + Bdq[1] * Adq[13]) + t3 * (t32 * Bdq[13] + t33 * Adq[13]);
    AScalar t46 = -0.9e1 * t2 * (Adq[1] * Bdq[14] + Bdq[1] * Adq[14]) + 0.3e1 * t5 * (-t1 * Bd2q[38] + Ad2q[38]) + t3 * (t32 * Bdq[14] + t33 * Adq[14]);
    AScalar t47 = -0.9e1 * t2 * (Adq[1] * Bdq[15] + Bdq[1] * Adq[15]) + 0.3e1 * t5 * (-t1 * Bd2q[39] + Ad2q[39]) + t3 * (t32 * Bdq[15] + t33 * Adq[15]);
    AScalar t48 = -0.9e1 * t2 * (Adq[1] * Bdq[16] + Bdq[1] * Adq[16]) + 0.3e1 * t5 * (-t1 * Bd2q[40] + Ad2q[40]) + t3 * (t32 * Bdq[16] + t33 * Adq[16]);
    AScalar t49 = -0.9e1 * t2 * (Adq[1] * Bdq[17] + Bdq[1] * Adq[17]) + 0.3e1 * t5 * (-t1 * Bd2q[41] + Ad2q[41]) + t3 * (t32 * Bdq[17] + t33 * Adq[17]);
    AScalar t50 = -0.9e1 * t2 * (Adq[1] * Bdq[18] + Bdq[1] * Adq[18]) + 0.3e1 * t5 * (-t1 * Bd2q[42] + Ad2q[42]) + t3 * (t32 * Bdq[18] + t33 * Adq[18]);
    AScalar t51 = -0.9e1 * t2 * (Adq[1] * Bdq[19] + Bdq[1] * Adq[19]) + 0.3e1 * t5 * (-t1 * Bd2q[43] + Ad2q[43]) + t3 * (t32 * Bdq[19] + t33 * Adq[19]);
    AScalar t52 = -0.9e1 * t2 * (Adq[1] * Bdq[20] + Bdq[1] * Adq[20]) + 0.3e1 * t5 * (-t1 * Bd2q[44] + Ad2q[44]) + t3 * (t32 * Bdq[20] + t33 * Adq[20]);
    AScalar t53 = -0.9e1 * t2 * (Adq[1] * Bdq[21] + Bdq[1] * Adq[21]) + 0.3e1 * t5 * (-t1 * Bd2q[45] + Ad2q[45]) + t3 * (t32 * Bdq[21] + t33 * Adq[21]);
    AScalar t54 = -0.9e1 * t2 * (Adq[1] * Bdq[22] + Bdq[1] * Adq[22]) + 0.3e1 * t5 * (-t1 * Bd2q[46] + Ad2q[46]) + t3 * (t32 * Bdq[22] + t33 * Adq[22]);
    t32 = -0.9e1 * t2 * (Adq[1] * Bdq[23] + Bdq[1] * Adq[23]) + 0.3e1 * t5 * (-t1 * Bd2q[47] + Ad2q[47]) + t3 * (t32 * Bdq[23] + t33 * Adq[23]);
    t33 = 0.3e1 * t5 * (-t1 * Bd2q[50] + Ad2q[50]) + t3 * (t6 * Bdq[2] * Bdq[2] + t7 * Adq[2] * Bdq[2] + 0.6e1 * Adq[2] * Adq[2]);
    AScalar t55 = t6 * Bdq[2];
    AScalar t56 = 0.6e1 * Adq[2];
    AScalar t57 = -0.9e1 * t2 * (Adq[2] * Bdq[3] + Bdq[2] * Adq[3]) + 0.3e1 * t5 * (-t1 * Bd2q[51] + Ad2q[51]) + t3 * (t55 * Bdq[3] + t56 * Adq[3]);
    AScalar t58 = -0.9e1 * t2 * (Adq[2] * Bdq[4] + Bdq[2] * Adq[4]) + 0.3e1 * t5 * (-t1 * Bd2q[52] + Ad2q[52]) + t3 * (t55 * Bdq[4] + t56 * Adq[4]);
    AScalar t59 = 0.3e1 * t5 * (-t1 * Bd2q[53] + Ad2q[53]) - 0.9e1 * t2 * (Adq[2] * Bdq[5] + Bdq[2] * Adq[5]) + t3 * (t55 * Bdq[5] + t56 * Adq[5]);
    AScalar t60 = 0.3e1 * t5 * (-t1 * Bd2q[54] + Ad2q[54]) - 0.9e1 * t2 * (Adq[2] * Bdq[6] + Bdq[2] * Adq[6]) + t3 * (t55 * Bdq[6] + t56 * Adq[6]);
    AScalar t61 = 0.3e1 * t5 * (-t1 * Bd2q[55] + Ad2q[55]) - 0.9e1 * t2 * (Adq[2] * Bdq[7] + Bdq[2] * Adq[7]) + t3 * (t55 * Bdq[7] + t56 * Adq[7]);
    AScalar t62 = 0.3e1 * t5 * (-t1 * Bd2q[56] + Ad2q[56]) - 0.9e1 * t2 * (Adq[2] * Bdq[8] + Bdq[2] * Adq[8]) + t3 * (t55 * Bdq[8] + t56 * Adq[8]);
    AScalar t63 = 0.3e1 * t5 * (-t1 * Bd2q[57] + Ad2q[57]) - 0.9e1 * t2 * (Adq[2] * Bdq[9] + Bdq[2] * Adq[9]) + t3 * (t55 * Bdq[9] + t56 * Adq[9]);
    AScalar t64 = 0.3e1 * t5 * (-t1 * Bd2q[58] + Ad2q[58]) - 0.9e1 * t2 * (Adq[2] * Bdq[10] + Bdq[2] * Adq[10]) + t3 * (t55 * Bdq[10] + t56 * Adq[10]);
    AScalar t65 = 0.3e1 * t5 * (-t1 * Bd2q[59] + Ad2q[59]) - 0.9e1 * t2 * (Adq[2] * Bdq[11] + Bdq[2] * Adq[11]) + t3 * (t55 * Bdq[11] + t56 * Adq[11]);
    AScalar t66 = 0.3e1 * t5 * (-t1 * Bd2q[60] + Ad2q[60]) - 0.9e1 * t2 * (Adq[2] * Bdq[12] + Bdq[2] * Adq[12]) + t3 * (t55 * Bdq[12] + t56 * Adq[12]);
    AScalar t67 = 0.3e1 * t5 * (-t1 * Bd2q[61] + Ad2q[61]) - 0.9e1 * t2 * (Adq[2] * Bdq[13] + Bdq[2] * Adq[13]) + t3 * (t55 * Bdq[13] + t56 * Adq[13]);
    AScalar t68 = -0.9e1 * t2 * (Adq[2] * Bdq[14] + Bdq[2] * Adq[14]) + 0.3e1 * t5 * (-t1 * Bd2q[62] + Ad2q[62]) + t3 * (t55 * Bdq[14] + t56 * Adq[14]);
    AScalar t69 = -0.9e1 * t2 * (Adq[2] * Bdq[15] + Bdq[2] * Adq[15]) + 0.3e1 * t5 * (-t1 * Bd2q[63] + Ad2q[63]) + t3 * (t55 * Bdq[15] + t56 * Adq[15]);
    AScalar t70 = -0.9e1 * t2 * (Adq[2] * Bdq[16] + Bdq[2] * Adq[16]) + 0.3e1 * t5 * (-t1 * Bd2q[64] + Ad2q[64]) + t3 * (t55 * Bdq[16] + t56 * Adq[16]);
    AScalar t71 = -0.9e1 * t2 * (Adq[2] * Bdq[17] + Bdq[2] * Adq[17]) + 0.3e1 * t5 * (-t1 * Bd2q[65] + Ad2q[65]) + t3 * (t55 * Bdq[17] + t56 * Adq[17]);
    AScalar t72 = -0.9e1 * t2 * (Adq[2] * Bdq[18] + Bdq[2] * Adq[18]) + 0.3e1 * t5 * (-t1 * Bd2q[66] + Ad2q[66]) + t3 * (t55 * Bdq[18] + t56 * Adq[18]);
    AScalar t73 = -0.9e1 * t2 * (Adq[2] * Bdq[19] + Bdq[2] * Adq[19]) + 0.3e1 * t5 * (-t1 * Bd2q[67] + Ad2q[67]) + t3 * (t55 * Bdq[19] + t56 * Adq[19]);
    AScalar t74 = -0.9e1 * t2 * (Adq[2] * Bdq[20] + Bdq[2] * Adq[20]) + 0.3e1 * t5 * (-t1 * Bd2q[68] + Ad2q[68]) + t3 * (t55 * Bdq[20] + t56 * Adq[20]);
    AScalar t75 = -0.9e1 * t2 * (Adq[2] * Bdq[21] + Bdq[2] * Adq[21]) + 0.3e1 * t5 * (-t1 * Bd2q[69] + Ad2q[69]) + t3 * (t55 * Bdq[21] + t56 * Adq[21]);
    AScalar t76 = -0.9e1 * t2 * (Adq[2] * Bdq[22] + Bdq[2] * Adq[22]) + 0.3e1 * t5 * (-t1 * Bd2q[70] + Ad2q[70]) + t3 * (t55 * Bdq[22] + t56 * Adq[22]);
    t55 = -0.9e1 * t2 * (Adq[2] * Bdq[23] + Bdq[2] * Adq[23]) + 0.3e1 * t5 * (-t1 * Bd2q[71] + Ad2q[71]) + t3 * (t55 * Bdq[23] + t56 * Adq[23]);
    t56 = 0.3e1 * t5 * (-t1 * Bd2q[75] + Ad2q[75]) + t3 * (t6 * Bdq[3] * Bdq[3] + t7 * Adq[3] * Bdq[3] + 0.6e1 * Adq[3] * Adq[3]);
    AScalar t77 = t6 * Bdq[3];
    AScalar t78 = 0.6e1 * Adq[3];
    AScalar t79 = -0.9e1 * t2 * (Adq[3] * Bdq[4] + Bdq[3] * Adq[4]) + 0.3e1 * t5 * (-t1 * Bd2q[76] + Ad2q[76]) + t3 * (t77 * Bdq[4] + t78 * Adq[4]);
    AScalar t80 = 0.3e1 * t5 * (-t1 * Bd2q[77] + Ad2q[77]) - 0.9e1 * t2 * (Adq[3] * Bdq[5] + Bdq[3] * Adq[5]) + t3 * (t77 * Bdq[5] + t78 * Adq[5]);
    AScalar t81 = 0.3e1 * t5 * (-t1 * Bd2q[78] + Ad2q[78]) - 0.9e1 * t2 * (Adq[3] * Bdq[6] + Bdq[3] * Adq[6]) + t3 * (t77 * Bdq[6] + t78 * Adq[6]);
    AScalar t82 = 0.3e1 * t5 * (-t1 * Bd2q[79] + Ad2q[79]) - 0.9e1 * t2 * (Adq[3] * Bdq[7] + Bdq[3] * Adq[7]) + t3 * (t77 * Bdq[7] + t78 * Adq[7]);
    AScalar t83 = 0.3e1 * t5 * (-t1 * Bd2q[80] + Ad2q[80]) - 0.9e1 * t2 * (Adq[3] * Bdq[8] + Bdq[3] * Adq[8]) + t3 * (t77 * Bdq[8] + t78 * Adq[8]);
    AScalar t84 = 0.3e1 * t5 * (-t1 * Bd2q[81] + Ad2q[81]) - 0.9e1 * t2 * (Adq[3] * Bdq[9] + Bdq[3] * Adq[9]) + t3 * (t77 * Bdq[9] + t78 * Adq[9]);
    AScalar t85 = 0.3e1 * t5 * (-t1 * Bd2q[82] + Ad2q[82]) - 0.9e1 * t2 * (Adq[3] * Bdq[10] + Bdq[3] * Adq[10]) + t3 * (t77 * Bdq[10] + t78 * Adq[10]);
    AScalar t86 = 0.3e1 * t5 * (-t1 * Bd2q[83] + Ad2q[83]) - 0.9e1 * t2 * (Adq[3] * Bdq[11] + Bdq[3] * Adq[11]) + t3 * (t77 * Bdq[11] + t78 * Adq[11]);
    AScalar t87 = 0.3e1 * t5 * (-t1 * Bd2q[84] + Ad2q[84]) - 0.9e1 * t2 * (Adq[3] * Bdq[12] + Bdq[3] * Adq[12]) + t3 * (t77 * Bdq[12] + t78 * Adq[12]);
    AScalar t88 = 0.3e1 * t5 * (-t1 * Bd2q[85] + Ad2q[85]) - 0.9e1 * t2 * (Adq[3] * Bdq[13] + Bdq[3] * Adq[13]) + t3 * (t77 * Bdq[13] + t78 * Adq[13]);
    AScalar t89 = -0.9e1 * t2 * (Adq[3] * Bdq[14] + Bdq[3] * Adq[14]) + 0.3e1 * t5 * (-t1 * Bd2q[86] + Ad2q[86]) + t3 * (t77 * Bdq[14] + t78 * Adq[14]);
    AScalar t90 = -0.9e1 * t2 * (Adq[3] * Bdq[15] + Bdq[3] * Adq[15]) + 0.3e1 * t5 * (-t1 * Bd2q[87] + Ad2q[87]) + t3 * (t77 * Bdq[15] + t78 * Adq[15]);
    AScalar t91 = -0.9e1 * t2 * (Adq[3] * Bdq[16] + Bdq[3] * Adq[16]) + 0.3e1 * t5 * (-t1 * Bd2q[88] + Ad2q[88]) + t3 * (t77 * Bdq[16] + t78 * Adq[16]);
    AScalar t92 = -0.9e1 * t2 * (Adq[3] * Bdq[17] + Bdq[3] * Adq[17]) + 0.3e1 * t5 * (-t1 * Bd2q[89] + Ad2q[89]) + t3 * (t77 * Bdq[17] + t78 * Adq[17]);
    AScalar t93 = -0.9e1 * t2 * (Adq[3] * Bdq[18] + Bdq[3] * Adq[18]) + 0.3e1 * t5 * (-t1 * Bd2q[90] + Ad2q[90]) + t3 * (t77 * Bdq[18] + t78 * Adq[18]);
    AScalar t94 = -0.9e1 * t2 * (Adq[3] * Bdq[19] + Bdq[3] * Adq[19]) + 0.3e1 * t5 * (-t1 * Bd2q[91] + Ad2q[91]) + t3 * (t77 * Bdq[19] + t78 * Adq[19]);
    AScalar t95 = -0.9e1 * t2 * (Adq[3] * Bdq[20] + Bdq[3] * Adq[20]) + 0.3e1 * t5 * (-t1 * Bd2q[92] + Ad2q[92]) + t3 * (t77 * Bdq[20] + t78 * Adq[20]);
    AScalar t96 = -0.9e1 * t2 * (Adq[3] * Bdq[21] + Bdq[3] * Adq[21]) + 0.3e1 * t5 * (-t1 * Bd2q[93] + Ad2q[93]) + t3 * (t77 * Bdq[21] + t78 * Adq[21]);
    AScalar t97 = -0.9e1 * t2 * (Adq[3] * Bdq[22] + Bdq[3] * Adq[22]) + 0.3e1 * t5 * (-t1 * Bd2q[94] + Ad2q[94]) + t3 * (t77 * Bdq[22] + t78 * Adq[22]);
    t77 = -0.9e1 * t2 * (Adq[3] * Bdq[23] + Bdq[3] * Adq[23]) + 0.3e1 * t5 * (-t1 * Bd2q[95] + Ad2q[95]) + t3 * (t77 * Bdq[23] + t78 * Adq[23]);
    t78 = 0.3e1 * t5 * (-t1 * Bd2q[100] + Ad2q[100]) + t3 * (t6 * Bdq[4] * Bdq[4] + t7 * Adq[4] * Bdq[4] + 0.6e1 * Adq[4] * Adq[4]);
    AScalar t98 = t6 * Bdq[4];
    AScalar t99 = 0.6e1 * Adq[4];
    AScalar t100 = 0.3e1 * t5 * (-t1 * Bd2q[101] + Ad2q[101]) - 0.9e1 * t2 * (Adq[4] * Bdq[5] + Bdq[4] * Adq[5]) + t3 * (t98 * Bdq[5] + t99 * Adq[5]);
    AScalar t101 = 0.3e1 * t5 * (-t1 * Bd2q[102] + Ad2q[102]) - 0.9e1 * t2 * (Adq[4] * Bdq[6] + Bdq[4] * Adq[6]) + t3 * (t98 * Bdq[6] + t99 * Adq[6]);
    AScalar t102 = 0.3e1 * t5 * (-t1 * Bd2q[103] + Ad2q[103]) - 0.9e1 * t2 * (Adq[4] * Bdq[7] + Bdq[4] * Adq[7]) + t3 * (t98 * Bdq[7] + t99 * Adq[7]);
    AScalar t103 = 0.3e1 * t5 * (-t1 * Bd2q[104] + Ad2q[104]) - 0.9e1 * t2 * (Adq[4] * Bdq[8] + Bdq[4] * Adq[8]) + t3 * (t98 * Bdq[8] + t99 * Adq[8]);
    AScalar t104 = 0.3e1 * t5 * (-t1 * Bd2q[105] + Ad2q[105]) - 0.9e1 * t2 * (Adq[4] * Bdq[9] + Bdq[4] * Adq[9]) + t3 * (t98 * Bdq[9] + t99 * Adq[9]);
    AScalar t105 = 0.3e1 * t5 * (-t1 * Bd2q[106] + Ad2q[106]) - 0.9e1 * t2 * (Adq[4] * Bdq[10] + Bdq[4] * Adq[10]) + t3 * (t98 * Bdq[10] + t99 * Adq[10]);
    AScalar t106 = 0.3e1 * t5 * (-t1 * Bd2q[107] + Ad2q[107]) - 0.9e1 * t2 * (Adq[4] * Bdq[11] + Bdq[4] * Adq[11]) + t3 * (t98 * Bdq[11] + t99 * Adq[11]);
    AScalar t107 = 0.3e1 * t5 * (-t1 * Bd2q[108] + Ad2q[108]) - 0.9e1 * t2 * (Adq[4] * Bdq[12] + Bdq[4] * Adq[12]) + t3 * (t98 * Bdq[12] + t99 * Adq[12]);
    AScalar t108 = 0.3e1 * t5 * (-t1 * Bd2q[109] + Ad2q[109]) - 0.9e1 * t2 * (Adq[4] * Bdq[13] + Bdq[4] * Adq[13]) + t3 * (t98 * Bdq[13] + t99 * Adq[13]);
    AScalar t109 = -0.9e1 * t2 * (Adq[4] * Bdq[14] + Bdq[4] * Adq[14]) + 0.3e1 * t5 * (-t1 * Bd2q[110] + Ad2q[110]) + t3 * (t98 * Bdq[14] + t99 * Adq[14]);
    AScalar t110 = -0.9e1 * t2 * (Adq[4] * Bdq[15] + Bdq[4] * Adq[15]) + 0.3e1 * t5 * (-t1 * Bd2q[111] + Ad2q[111]) + t3 * (t98 * Bdq[15] + t99 * Adq[15]);
    AScalar t111 = -0.9e1 * t2 * (Adq[4] * Bdq[16] + Bdq[4] * Adq[16]) + 0.3e1 * t5 * (-t1 * Bd2q[112] + Ad2q[112]) + t3 * (t98 * Bdq[16] + t99 * Adq[16]);
    AScalar t112 = -0.9e1 * t2 * (Adq[4] * Bdq[17] + Bdq[4] * Adq[17]) + 0.3e1 * t5 * (-t1 * Bd2q[113] + Ad2q[113]) + t3 * (t98 * Bdq[17] + t99 * Adq[17]);
    AScalar t113 = -0.9e1 * t2 * (Adq[4] * Bdq[18] + Bdq[4] * Adq[18]) + 0.3e1 * t5 * (-t1 * Bd2q[114] + Ad2q[114]) + t3 * (t98 * Bdq[18] + t99 * Adq[18]);
    AScalar t114 = -0.9e1 * t2 * (Adq[4] * Bdq[19] + Bdq[4] * Adq[19]) + 0.3e1 * t5 * (-t1 * Bd2q[115] + Ad2q[115]) + t3 * (t98 * Bdq[19] + t99 * Adq[19]);
    AScalar t115 = -0.9e1 * t2 * (Adq[4] * Bdq[20] + Bdq[4] * Adq[20]) + 0.3e1 * t5 * (-t1 * Bd2q[116] + Ad2q[116]) + t3 * (t98 * Bdq[20] + t99 * Adq[20]);
    AScalar t116 = -0.9e1 * t2 * (Adq[4] * Bdq[21] + Bdq[4] * Adq[21]) + 0.3e1 * t5 * (-t1 * Bd2q[117] + Ad2q[117]) + t3 * (t98 * Bdq[21] + t99 * Adq[21]);
    AScalar t117 = -0.9e1 * t2 * (Adq[4] * Bdq[22] + Bdq[4] * Adq[22]) + 0.3e1 * t5 * (-t1 * Bd2q[118] + Ad2q[118]) + t3 * (t98 * Bdq[22] + t99 * Adq[22]);
    t98 = -0.9e1 * t2 * (Adq[4] * Bdq[23] + Bdq[4] * Adq[23]) + 0.3e1 * t5 * (-t1 * Bd2q[119] + Ad2q[119]) + t3 * (t98 * Bdq[23] + t99 * Adq[23]);
    t99 = 0.3e1 * t5 * (-t1 * Bd2q[125] + Ad2q[125]) + t3 * (t6 * Bdq[5] * Bdq[5] + t7 * Adq[5] * Bdq[5] + 0.6e1 * Adq[5] * Adq[5]);
    AScalar t118 = t6 * Bdq[5];
    AScalar t119 = 0.6e1 * Adq[5];
    AScalar t120 = 0.3e1 * t5 * (-t1 * Bd2q[126] + Ad2q[126]) - 0.9e1 * t2 * (Adq[5] * Bdq[6] + Bdq[5] * Adq[6]) + t3 * (t118 * Bdq[6] + t119 * Adq[6]);
    AScalar t121 = 0.3e1 * t5 * (-t1 * Bd2q[127] + Ad2q[127]) - 0.9e1 * t2 * (Adq[5] * Bdq[7] + Bdq[5] * Adq[7]) + t3 * (t118 * Bdq[7] + t119 * Adq[7]);
    AScalar t122 = 0.3e1 * t5 * (-t1 * Bd2q[128] + Ad2q[128]) - 0.9e1 * t2 * (Adq[5] * Bdq[8] + Bdq[5] * Adq[8]) + t3 * (t118 * Bdq[8] + t119 * Adq[8]);
    AScalar t123 = 0.3e1 * t5 * (-t1 * Bd2q[129] + Ad2q[129]) - 0.9e1 * t2 * (Adq[5] * Bdq[9] + Bdq[5] * Adq[9]) + t3 * (t118 * Bdq[9] + t119 * Adq[9]);
    AScalar t124 = 0.3e1 * t5 * (-t1 * Bd2q[130] + Ad2q[130]) - 0.9e1 * t2 * (Adq[5] * Bdq[10] + Bdq[5] * Adq[10]) + t3 * (t118 * Bdq[10] + t119 * Adq[10]);
    AScalar t125 = 0.3e1 * t5 * (-t1 * Bd2q[131] + Ad2q[131]) - 0.9e1 * t2 * (Adq[5] * Bdq[11] + Bdq[5] * Adq[11]) + t3 * (t118 * Bdq[11] + t119 * Adq[11]);
    AScalar t126 = 0.3e1 * t5 * (-t1 * Bd2q[132] + Ad2q[132]) - 0.9e1 * t2 * (Adq[5] * Bdq[12] + Bdq[5] * Adq[12]) + t3 * (t118 * Bdq[12] + t119 * Adq[12]);
    AScalar t127 = -0.3e1 * t5 * (t1 * Bd2q[133] - Ad2q[133]) - 0.9e1 * t2 * (Adq[5] * Bdq[13] + Bdq[5] * Adq[13]) + t3 * (t118 * Bdq[13] + t119 * Adq[13]);
    AScalar t128 = -0.9e1 * t2 * (Adq[5] * Bdq[14] + Bdq[5] * Adq[14]) + 0.3e1 * t5 * (-t1 * Bd2q[134] + Ad2q[134]) + t3 * (t118 * Bdq[14] + t119 * Adq[14]);
    AScalar t129 = -0.9e1 * t2 * (Adq[5] * Bdq[15] + Bdq[5] * Adq[15]) + 0.3e1 * t5 * (-t1 * Bd2q[135] + Ad2q[135]) + t3 * (t118 * Bdq[15] + t119 * Adq[15]);
    AScalar t130 = -0.9e1 * t2 * (Adq[5] * Bdq[16] + Bdq[5] * Adq[16]) + 0.3e1 * t5 * (-t1 * Bd2q[136] + Ad2q[136]) + t3 * (t118 * Bdq[16] + t119 * Adq[16]);
    AScalar t131 = -0.9e1 * t2 * (Adq[5] * Bdq[17] + Bdq[5] * Adq[17]) + 0.3e1 * t5 * (-t1 * Bd2q[137] + Ad2q[137]) + t3 * (t118 * Bdq[17] + t119 * Adq[17]);
    AScalar t132 = -0.9e1 * t2 * (Adq[5] * Bdq[18] + Bdq[5] * Adq[18]) + 0.3e1 * t5 * (-t1 * Bd2q[138] + Ad2q[138]) + t3 * (t118 * Bdq[18] + t119 * Adq[18]);
    AScalar t133 = -0.9e1 * t2 * (Adq[5] * Bdq[19] + Bdq[5] * Adq[19]) + 0.3e1 * t5 * (-t1 * Bd2q[139] + Ad2q[139]) + t3 * (t118 * Bdq[19] + t119 * Adq[19]);
    AScalar t134 = -0.9e1 * t2 * (Adq[5] * Bdq[20] + Bdq[5] * Adq[20]) + 0.3e1 * t5 * (-t1 * Bd2q[140] + Ad2q[140]) + t3 * (t118 * Bdq[20] + t119 * Adq[20]);
    AScalar t135 = -0.9e1 * t2 * (Adq[5] * Bdq[21] + Bdq[5] * Adq[21]) + 0.3e1 * t5 * (-t1 * Bd2q[141] + Ad2q[141]) + t3 * (t118 * Bdq[21] + t119 * Adq[21]);
    AScalar t136 = -0.9e1 * t2 * (Adq[5] * Bdq[22] + Bdq[5] * Adq[22]) + 0.3e1 * t5 * (-t1 * Bd2q[142] + Ad2q[142]) + t3 * (t118 * Bdq[22] + t119 * Adq[22]);
    t118 = -0.9e1 * t2 * (Adq[5] * Bdq[23] + Bdq[5] * Adq[23]) + 0.3e1 * t5 * (-t1 * Bd2q[143] + Ad2q[143]) + t3 * (t118 * Bdq[23] + t119 * Adq[23]);
    t119 = 0.3e1 * t5 * (-t1 * Bd2q[150] + Ad2q[150]) + t3 * (t6 * Bdq[6] * Bdq[6] + t7 * Adq[6] * Bdq[6] + 0.6e1 * Adq[6] * Adq[6]);
    AScalar t137 = t6 * Bdq[6];
    AScalar t138 = 0.6e1 * Adq[6];
    AScalar t139 = 0.3e1 * t5 * (-t1 * Bd2q[151] + Ad2q[151]) - 0.9e1 * t2 * (Adq[6] * Bdq[7] + Bdq[6] * Adq[7]) + t3 * (t137 * Bdq[7] + t138 * Adq[7]);
    AScalar t140 = 0.3e1 * t5 * (-t1 * Bd2q[152] + Ad2q[152]) - 0.9e1 * t2 * (Adq[6] * Bdq[8] + Bdq[6] * Adq[8]) + t3 * (t137 * Bdq[8] + t138 * Adq[8]);
    AScalar t141 = 0.3e1 * t5 * (-t1 * Bd2q[153] + Ad2q[153]) - 0.9e1 * t2 * (Adq[6] * Bdq[9] + Bdq[6] * Adq[9]) + t3 * (t137 * Bdq[9] + t138 * Adq[9]);
    AScalar t142 = 0.3e1 * t5 * (-t1 * Bd2q[154] + Ad2q[154]) - 0.9e1 * t2 * (Adq[6] * Bdq[10] + Bdq[6] * Adq[10]) + t3 * (t137 * Bdq[10] + t138 * Adq[10]);
    AScalar t143 = 0.3e1 * t5 * (-t1 * Bd2q[155] + Ad2q[155]) - 0.9e1 * t2 * (Adq[6] * Bdq[11] + Bdq[6] * Adq[11]) + t3 * (t137 * Bdq[11] + t138 * Adq[11]);
    AScalar t144 = 0.3e1 * t5 * (-t1 * Bd2q[156] + Ad2q[156]) - 0.9e1 * t2 * (Adq[6] * Bdq[12] + Bdq[6] * Adq[12]) + t3 * (t137 * Bdq[12] + t138 * Adq[12]);
    AScalar t145 = -0.9e1 * t2 * (Adq[6] * Bdq[13] + Bdq[6] * Adq[13]) + 0.3e1 * t5 * (-t1 * Bd2q[157] + Ad2q[157]) + t3 * (t137 * Bdq[13] + t138 * Adq[13]);
    AScalar t146 = -0.9e1 * t2 * (Adq[6] * Bdq[14] + Bdq[6] * Adq[14]) + 0.3e1 * t5 * (-t1 * Bd2q[158] + Ad2q[158]) + t3 * (t137 * Bdq[14] + t138 * Adq[14]);
    AScalar t147 = -0.9e1 * t2 * (Adq[6] * Bdq[15] + Bdq[6] * Adq[15]) + 0.3e1 * t5 * (-t1 * Bd2q[159] + Ad2q[159]) + t3 * (t137 * Bdq[15] + t138 * Adq[15]);
    AScalar t148 = -0.9e1 * t2 * (Adq[6] * Bdq[16] + Bdq[6] * Adq[16]) + 0.3e1 * t5 * (-t1 * Bd2q[160] + Ad2q[160]) + t3 * (t137 * Bdq[16] + t138 * Adq[16]);
    AScalar t149 = -0.9e1 * t2 * (Adq[6] * Bdq[17] + Bdq[6] * Adq[17]) + 0.3e1 * t5 * (-t1 * Bd2q[161] + Ad2q[161]) + t3 * (t137 * Bdq[17] + t138 * Adq[17]);
    AScalar t150 = -0.9e1 * t2 * (Adq[6] * Bdq[18] + Bdq[6] * Adq[18]) + 0.3e1 * t5 * (-t1 * Bd2q[162] + Ad2q[162]) + t3 * (t137 * Bdq[18] + t138 * Adq[18]);
    AScalar t151 = -0.9e1 * t2 * (Adq[6] * Bdq[19] + Bdq[6] * Adq[19]) + 0.3e1 * t5 * (-t1 * Bd2q[163] + Ad2q[163]) + t3 * (t137 * Bdq[19] + t138 * Adq[19]);
    AScalar t152 = -0.9e1 * t2 * (Adq[6] * Bdq[20] + Bdq[6] * Adq[20]) + 0.3e1 * t5 * (-t1 * Bd2q[164] + Ad2q[164]) + t3 * (t137 * Bdq[20] + t138 * Adq[20]);
    AScalar t153 = -0.9e1 * t2 * (Adq[6] * Bdq[21] + Bdq[6] * Adq[21]) + 0.3e1 * t5 * (-t1 * Bd2q[165] + Ad2q[165]) + t3 * (t137 * Bdq[21] + t138 * Adq[21]);
    AScalar t154 = -0.9e1 * t2 * (Adq[6] * Bdq[22] + Bdq[6] * Adq[22]) + 0.3e1 * t5 * (-t1 * Bd2q[166] + Ad2q[166]) + t3 * (t137 * Bdq[22] + t138 * Adq[22]);
    t137 = -0.9e1 * t2 * (Adq[6] * Bdq[23] + Bdq[6] * Adq[23]) + 0.3e1 * t5 * (-t1 * Bd2q[167] + Ad2q[167]) + t3 * (t137 * Bdq[23] + t138 * Adq[23]);
    t138 = 0.3e1 * t5 * (-t1 * Bd2q[175] + Ad2q[175]) + t3 * (t6 * Bdq[7] * Bdq[7] + t7 * Adq[7] * Bdq[7] + 0.6e1 * Adq[7] * Adq[7]);
    AScalar t155 = t6 * Bdq[7];
    AScalar t156 = 0.6e1 * Adq[7];
    AScalar t157 = 0.3e1 * t5 * (-t1 * Bd2q[176] + Ad2q[176]) - 0.9e1 * t2 * (Adq[7] * Bdq[8] + Bdq[7] * Adq[8]) + t3 * (t155 * Bdq[8] + t156 * Adq[8]);
    AScalar t158 = 0.3e1 * t5 * (-t1 * Bd2q[177] + Ad2q[177]) - 0.9e1 * t2 * (Adq[7] * Bdq[9] + Bdq[7] * Adq[9]) + t3 * (t155 * Bdq[9] + t156 * Adq[9]);
    AScalar t159 = 0.3e1 * t5 * (-t1 * Bd2q[178] + Ad2q[178]) - 0.9e1 * t2 * (Adq[7] * Bdq[10] + Bdq[7] * Adq[10]) + t3 * (t155 * Bdq[10] + t156 * Adq[10]);
    AScalar t160 = 0.3e1 * t5 * (-t1 * Bd2q[179] + Ad2q[179]) - 0.9e1 * t2 * (Adq[7] * Bdq[11] + Bdq[7] * Adq[11]) + t3 * (t155 * Bdq[11] + t156 * Adq[11]);
    AScalar t161 = 0.3e1 * t5 * (-t1 * Bd2q[180] + Ad2q[180]) - 0.9e1 * t2 * (Adq[7] * Bdq[12] + Bdq[7] * Adq[12]) + t3 * (t155 * Bdq[12] + t156 * Adq[12]);
    AScalar t162 = -0.9e1 * t2 * (Adq[7] * Bdq[13] + Bdq[7] * Adq[13]) + 0.3e1 * t5 * (-t1 * Bd2q[181] + Ad2q[181]) + t3 * (t155 * Bdq[13] + t156 * Adq[13]);
    AScalar t163 = -0.9e1 * t2 * (Adq[7] * Bdq[14] + Bdq[7] * Adq[14]) + 0.3e1 * t5 * (-t1 * Bd2q[182] + Ad2q[182]) + t3 * (t155 * Bdq[14] + t156 * Adq[14]);
    AScalar t164 = -0.9e1 * t2 * (Adq[7] * Bdq[15] + Bdq[7] * Adq[15]) + 0.3e1 * t5 * (-t1 * Bd2q[183] + Ad2q[183]) + t3 * (t155 * Bdq[15] + t156 * Adq[15]);
    AScalar t165 = -0.9e1 * t2 * (Adq[7] * Bdq[16] + Bdq[7] * Adq[16]) + 0.3e1 * t5 * (-t1 * Bd2q[184] + Ad2q[184]) + t3 * (t155 * Bdq[16] + t156 * Adq[16]);
    AScalar t166 = -0.9e1 * t2 * (Adq[7] * Bdq[17] + Bdq[7] * Adq[17]) + 0.3e1 * t5 * (-t1 * Bd2q[185] + Ad2q[185]) + t3 * (t155 * Bdq[17] + t156 * Adq[17]);
    AScalar t167 = -0.9e1 * t2 * (Adq[7] * Bdq[18] + Bdq[7] * Adq[18]) + 0.3e1 * t5 * (-t1 * Bd2q[186] + Ad2q[186]) + t3 * (t155 * Bdq[18] + t156 * Adq[18]);
    AScalar t168 = -0.9e1 * t2 * (Adq[7] * Bdq[19] + Bdq[7] * Adq[19]) + 0.3e1 * t5 * (-t1 * Bd2q[187] + Ad2q[187]) + t3 * (t155 * Bdq[19] + t156 * Adq[19]);
    AScalar t169 = -0.9e1 * t2 * (Adq[7] * Bdq[20] + Bdq[7] * Adq[20]) + 0.3e1 * t5 * (-t1 * Bd2q[188] + Ad2q[188]) + t3 * (t155 * Bdq[20] + t156 * Adq[20]);
    AScalar t170 = -0.9e1 * t2 * (Adq[7] * Bdq[21] + Bdq[7] * Adq[21]) + 0.3e1 * t5 * (-t1 * Bd2q[189] + Ad2q[189]) + t3 * (t155 * Bdq[21] + t156 * Adq[21]);
    AScalar t171 = -0.9e1 * t2 * (Adq[7] * Bdq[22] + Bdq[7] * Adq[22]) + 0.3e1 * t5 * (-t1 * Bd2q[190] + Ad2q[190]) + t3 * (t155 * Bdq[22] + t156 * Adq[22]);
    t155 = -0.9e1 * t2 * (Adq[7] * Bdq[23] + Bdq[7] * Adq[23]) + 0.3e1 * t5 * (-t1 * Bd2q[191] + Ad2q[191]) + t3 * (t155 * Bdq[23] + t156 * Adq[23]);
    t156 = 0.3e1 * t5 * (-t1 * Bd2q[200] + Ad2q[200]) + t3 * (t6 * Bdq[8] * Bdq[8] + t7 * Adq[8] * Bdq[8] + 0.6e1 * Adq[8] * Adq[8]);
    AScalar t172 = t6 * Bdq[8];
    AScalar t173 = 0.6e1 * Adq[8];
    AScalar t174 = 0.3e1 * t5 * (-t1 * Bd2q[201] + Ad2q[201]) - 0.9e1 * t2 * (Adq[8] * Bdq[9] + Bdq[8] * Adq[9]) + t3 * (t172 * Bdq[9] + t173 * Adq[9]);
    AScalar t175 = 0.3e1 * t5 * (-t1 * Bd2q[202] + Ad2q[202]) - 0.9e1 * t2 * (Adq[8] * Bdq[10] + Bdq[8] * Adq[10]) + t3 * (t172 * Bdq[10] + t173 * Adq[10]);
    AScalar t176 = 0.3e1 * t5 * (-t1 * Bd2q[203] + Ad2q[203]) - 0.9e1 * t2 * (Adq[8] * Bdq[11] + Bdq[8] * Adq[11]) + t3 * (t172 * Bdq[11] + t173 * Adq[11]);
    AScalar t177 = 0.3e1 * t5 * (-t1 * Bd2q[204] + Ad2q[204]) - 0.9e1 * t2 * (Adq[8] * Bdq[12] + Bdq[8] * Adq[12]) + t3 * (t172 * Bdq[12] + t173 * Adq[12]);
    AScalar t178 = -0.9e1 * t2 * (Adq[8] * Bdq[13] + Bdq[8] * Adq[13]) + 0.3e1 * t5 * (-t1 * Bd2q[205] + Ad2q[205]) + t3 * (t172 * Bdq[13] + t173 * Adq[13]);
    AScalar t179 = -0.9e1 * t2 * (Adq[8] * Bdq[14] + Bdq[8] * Adq[14]) + 0.3e1 * t5 * (-t1 * Bd2q[206] + Ad2q[206]) + t3 * (t172 * Bdq[14] + t173 * Adq[14]);
    AScalar t180 = -0.9e1 * t2 * (Adq[8] * Bdq[15] + Bdq[8] * Adq[15]) + 0.3e1 * t5 * (-t1 * Bd2q[207] + Ad2q[207]) + t3 * (t172 * Bdq[15] + t173 * Adq[15]);
    AScalar t181 = -0.9e1 * t2 * (Adq[8] * Bdq[16] + Bdq[8] * Adq[16]) + 0.3e1 * t5 * (-t1 * Bd2q[208] + Ad2q[208]) + t3 * (t172 * Bdq[16] + t173 * Adq[16]);
    AScalar t182 = -0.9e1 * t2 * (Adq[8] * Bdq[17] + Bdq[8] * Adq[17]) + 0.3e1 * t5 * (-t1 * Bd2q[209] + Ad2q[209]) + t3 * (t172 * Bdq[17] + t173 * Adq[17]);
    AScalar t183 = -0.9e1 * t2 * (Adq[8] * Bdq[18] + Bdq[8] * Adq[18]) + 0.3e1 * t5 * (-t1 * Bd2q[210] + Ad2q[210]) + t3 * (t172 * Bdq[18] + t173 * Adq[18]);
    AScalar t184 = -0.9e1 * t2 * (Adq[8] * Bdq[19] + Bdq[8] * Adq[19]) + 0.3e1 * t5 * (-t1 * Bd2q[211] + Ad2q[211]) + t3 * (t172 * Bdq[19] + t173 * Adq[19]);
    AScalar t185 = -0.9e1 * t2 * (Adq[8] * Bdq[20] + Bdq[8] * Adq[20]) + 0.3e1 * t5 * (-t1 * Bd2q[212] + Ad2q[212]) + t3 * (t172 * Bdq[20] + t173 * Adq[20]);
    AScalar t186 = -0.9e1 * t2 * (Adq[8] * Bdq[21] + Bdq[8] * Adq[21]) + 0.3e1 * t5 * (-t1 * Bd2q[213] + Ad2q[213]) + t3 * (t172 * Bdq[21] + t173 * Adq[21]);
    AScalar t187 = -0.9e1 * t2 * (Adq[8] * Bdq[22] + Bdq[8] * Adq[22]) + 0.3e1 * t5 * (-t1 * Bd2q[214] + Ad2q[214]) + t3 * (t172 * Bdq[22] + t173 * Adq[22]);
    t172 = -0.9e1 * t2 * (Adq[8] * Bdq[23] + Bdq[8] * Adq[23]) + 0.3e1 * t5 * (-t1 * Bd2q[215] + Ad2q[215]) + t3 * (t172 * Bdq[23] + t173 * Adq[23]);
    t173 = 0.3e1 * t5 * (-t1 * Bd2q[225] + Ad2q[225]) + t3 * (t6 * Bdq[9] * Bdq[9] + t7 * Adq[9] * Bdq[9] + 0.6e1 * Adq[9] * Adq[9]);
    AScalar t188 = t6 * Bdq[9];
    AScalar t189 = 0.6e1 * Adq[9];
    AScalar t190 = 0.3e1 * t5 * (-t1 * Bd2q[226] + Ad2q[226]) - 0.9e1 * t2 * (Adq[9] * Bdq[10] + Bdq[9] * Adq[10]) + t3 * (t188 * Bdq[10] + t189 * Adq[10]);
    AScalar t191 = 0.3e1 * t5 * (-t1 * Bd2q[227] + Ad2q[227]) - 0.9e1 * t2 * (Adq[9] * Bdq[11] + Bdq[9] * Adq[11]) + t3 * (t188 * Bdq[11] + t189 * Adq[11]);
    AScalar t192 = 0.3e1 * t5 * (-t1 * Bd2q[228] + Ad2q[228]) - 0.9e1 * t2 * (Adq[9] * Bdq[12] + Bdq[9] * Adq[12]) + t3 * (t188 * Bdq[12] + t189 * Adq[12]);
    AScalar t193 = -0.9e1 * t2 * (Adq[9] * Bdq[13] + Bdq[9] * Adq[13]) + 0.3e1 * t5 * (-t1 * Bd2q[229] + Ad2q[229]) + t3 * (t188 * Bdq[13] + t189 * Adq[13]);
    AScalar t194 = -0.9e1 * t2 * (Adq[9] * Bdq[14] + Bdq[9] * Adq[14]) + 0.3e1 * t5 * (-t1 * Bd2q[230] + Ad2q[230]) + t3 * (t188 * Bdq[14] + t189 * Adq[14]);
    AScalar t195 = -0.9e1 * t2 * (Adq[9] * Bdq[15] + Bdq[9] * Adq[15]) + 0.3e1 * t5 * (-t1 * Bd2q[231] + Ad2q[231]) + t3 * (t188 * Bdq[15] + t189 * Adq[15]);
    AScalar t196 = -0.9e1 * t2 * (Adq[9] * Bdq[16] + Bdq[9] * Adq[16]) + 0.3e1 * t5 * (-t1 * Bd2q[232] + Ad2q[232]) + t3 * (t188 * Bdq[16] + t189 * Adq[16]);
    AScalar t197 = -0.9e1 * t2 * (Adq[9] * Bdq[17] + Bdq[9] * Adq[17]) + 0.3e1 * t5 * (-t1 * Bd2q[233] + Ad2q[233]) + t3 * (t188 * Bdq[17] + t189 * Adq[17]);
    AScalar t198 = -0.9e1 * t2 * (Adq[9] * Bdq[18] + Bdq[9] * Adq[18]) + 0.3e1 * t5 * (-t1 * Bd2q[234] + Ad2q[234]) + t3 * (t188 * Bdq[18] + t189 * Adq[18]);
    AScalar t199 = -0.9e1 * t2 * (Adq[9] * Bdq[19] + Bdq[9] * Adq[19]) + 0.3e1 * t5 * (-t1 * Bd2q[235] + Ad2q[235]) + t3 * (t188 * Bdq[19] + t189 * Adq[19]);
    AScalar t200 = -0.9e1 * t2 * (Adq[9] * Bdq[20] + Bdq[9] * Adq[20]) + 0.3e1 * t5 * (-t1 * Bd2q[236] + Ad2q[236]) + t3 * (t188 * Bdq[20] + t189 * Adq[20]);
    AScalar t201 = -0.9e1 * t2 * (Adq[9] * Bdq[21] + Bdq[9] * Adq[21]) + 0.3e1 * t5 * (-t1 * Bd2q[237] + Ad2q[237]) + t3 * (t188 * Bdq[21] + t189 * Adq[21]);
    AScalar t202 = -0.9e1 * t2 * (Adq[9] * Bdq[22] + Bdq[9] * Adq[22]) + 0.3e1 * t5 * (-t1 * Bd2q[238] + Ad2q[238]) + t3 * (t188 * Bdq[22] + t189 * Adq[22]);
    t188 = -0.9e1 * t2 * (Adq[9] * Bdq[23] + Bdq[9] * Adq[23]) + 0.3e1 * t5 * (-t1 * Bd2q[239] + Ad2q[239]) + t3 * (t188 * Bdq[23] + t189 * Adq[23]);
    t189 = 0.3e1 * t5 * (-t1 * Bd2q[250] + Ad2q[250]) + t3 * (t6 * Bdq[10] * Bdq[10] + t7 * Adq[10] * Bdq[10] + 0.6e1 * Adq[10] * Adq[10]);
    AScalar t203 = t6 * Bdq[10];
    AScalar t204 = 0.6e1 * Adq[10];
    AScalar t205 = 0.3e1 * t5 * (-t1 * Bd2q[251] + Ad2q[251]) - 0.9e1 * t2 * (Adq[10] * Bdq[11] + Bdq[10] * Adq[11]) + t3 * (t203 * Bdq[11] + t204 * Adq[11]);
    AScalar t206 = 0.3e1 * t5 * (-t1 * Bd2q[252] + Ad2q[252]) - 0.9e1 * t2 * (Adq[10] * Bdq[12] + Bdq[10] * Adq[12]) + t3 * (t203 * Bdq[12] + t204 * Adq[12]);
    AScalar t207 = -0.9e1 * t2 * (Adq[10] * Bdq[13] + Bdq[10] * Adq[13]) + 0.3e1 * t5 * (-t1 * Bd2q[253] + Ad2q[253]) + t3 * (t203 * Bdq[13] + t204 * Adq[13]);
    AScalar t208 = -0.9e1 * t2 * (Adq[10] * Bdq[14] + Bdq[10] * Adq[14]) + 0.3e1 * t5 * (-t1 * Bd2q[254] + Ad2q[254]) + t3 * (t203 * Bdq[14] + t204 * Adq[14]);
    AScalar t209 = -0.9e1 * t2 * (Adq[10] * Bdq[15] + Bdq[10] * Adq[15]) + 0.3e1 * t5 * (-t1 * Bd2q[255] + Ad2q[255]) + t3 * (t203 * Bdq[15] + t204 * Adq[15]);
    AScalar t210 = -0.9e1 * t2 * (Adq[10] * Bdq[16] + Bdq[10] * Adq[16]) + 0.3e1 * t5 * (-t1 * Bd2q[256] + Ad2q[256]) + t3 * (t203 * Bdq[16] + t204 * Adq[16]);
    AScalar t211 = -0.9e1 * t2 * (Adq[10] * Bdq[17] + Bdq[10] * Adq[17]) + 0.3e1 * t5 * (-t1 * Bd2q[257] + Ad2q[257]) + t3 * (t203 * Bdq[17] + t204 * Adq[17]);
    AScalar t212 = -0.9e1 * t2 * (Adq[10] * Bdq[18] + Bdq[10] * Adq[18]) + 0.3e1 * t5 * (-t1 * Bd2q[258] + Ad2q[258]) + t3 * (t203 * Bdq[18] + t204 * Adq[18]);
    AScalar t213 = -0.9e1 * t2 * (Adq[10] * Bdq[19] + Bdq[10] * Adq[19]) + 0.3e1 * t5 * (-t1 * Bd2q[259] + Ad2q[259]) + t3 * (t203 * Bdq[19] + t204 * Adq[19]);
    AScalar t214 = -0.9e1 * t2 * (Adq[10] * Bdq[20] + Bdq[10] * Adq[20]) + 0.3e1 * t5 * (-t1 * Bd2q[260] + Ad2q[260]) + t3 * (t203 * Bdq[20] + t204 * Adq[20]);
    AScalar t215 = -0.9e1 * t2 * (Adq[10] * Bdq[21] + Bdq[10] * Adq[21]) + 0.3e1 * t5 * (-t1 * Bd2q[261] + Ad2q[261]) + t3 * (t203 * Bdq[21] + t204 * Adq[21]);
    AScalar t216 = -0.9e1 * t2 * (Adq[10] * Bdq[22] + Bdq[10] * Adq[22]) + 0.3e1 * t5 * (-t1 * Bd2q[262] + Ad2q[262]) + t3 * (t203 * Bdq[22] + t204 * Adq[22]);
    t203 = -0.9e1 * t2 * (Adq[10] * Bdq[23] + Bdq[10] * Adq[23]) + 0.3e1 * t5 * (-t1 * Bd2q[263] + Ad2q[263]) + t3 * (t203 * Bdq[23] + t204 * Adq[23]);
    t204 = 0.3e1 * t5 * (-t1 * Bd2q[275] + Ad2q[275]) + t3 * (t6 * Bdq[11] * Bdq[11] + t7 * Adq[11] * Bdq[11] + 0.6e1 * Adq[11] * Adq[11]);
    AScalar t217 = t6 * Bdq[11];
    AScalar t218 = 0.6e1 * Adq[11];
    AScalar t219 = 0.3e1 * t5 * (-t1 * Bd2q[276] + Ad2q[276]) - 0.9e1 * t2 * (Adq[11] * Bdq[12] + Bdq[11] * Adq[12]) + t3 * (t217 * Bdq[12] + t218 * Adq[12]);
    AScalar t220 = -0.9e1 * t2 * (Adq[11] * Bdq[13] + Bdq[11] * Adq[13]) + 0.3e1 * t5 * (-t1 * Bd2q[277] + Ad2q[277]) + t3 * (t217 * Bdq[13] + t218 * Adq[13]);
    AScalar t221 = -0.9e1 * t2 * (Adq[11] * Bdq[14] + Bdq[11] * Adq[14]) + 0.3e1 * t5 * (-t1 * Bd2q[278] + Ad2q[278]) + t3 * (t217 * Bdq[14] + t218 * Adq[14]);
    AScalar t222 = -0.9e1 * t2 * (Adq[11] * Bdq[15] + Bdq[11] * Adq[15]) + 0.3e1 * t5 * (-t1 * Bd2q[279] + Ad2q[279]) + t3 * (t217 * Bdq[15] + t218 * Adq[15]);
    AScalar t223 = -0.9e1 * t2 * (Adq[11] * Bdq[16] + Bdq[11] * Adq[16]) + 0.3e1 * t5 * (-t1 * Bd2q[280] + Ad2q[280]) + t3 * (t217 * Bdq[16] + t218 * Adq[16]);
    AScalar t224 = -0.9e1 * t2 * (Adq[11] * Bdq[17] + Bdq[11] * Adq[17]) + 0.3e1 * t5 * (-t1 * Bd2q[281] + Ad2q[281]) + t3 * (t217 * Bdq[17] + t218 * Adq[17]);
    AScalar t225 = -0.9e1 * t2 * (Adq[11] * Bdq[18] + Bdq[11] * Adq[18]) + 0.3e1 * t5 * (-t1 * Bd2q[282] + Ad2q[282]) + t3 * (t217 * Bdq[18] + t218 * Adq[18]);
    AScalar t226 = -0.9e1 * t2 * (Adq[11] * Bdq[19] + Bdq[11] * Adq[19]) + 0.3e1 * t5 * (-t1 * Bd2q[283] + Ad2q[283]) + t3 * (t217 * Bdq[19] + t218 * Adq[19]);
    AScalar t227 = -0.9e1 * t2 * (Adq[11] * Bdq[20] + Bdq[11] * Adq[20]) - 0.3e1 * t5 * (t1 * Bd2q[284] - Ad2q[284]) + t3 * (t217 * Bdq[20] + t218 * Adq[20]);
    AScalar t228 = -0.9e1 * t2 * (Adq[11] * Bdq[21] + Bdq[11] * Adq[21]) + 0.3e1 * t5 * (-t1 * Bd2q[285] + Ad2q[285]) + t3 * (t217 * Bdq[21] + t218 * Adq[21]);
    AScalar t229 = -0.9e1 * t2 * (Adq[11] * Bdq[22] + Bdq[11] * Adq[22]) + 0.3e1 * t5 * (-t1 * Bd2q[286] + Ad2q[286]) + t3 * (t217 * Bdq[22] + t218 * Adq[22]);
    t217 = -0.9e1 * t2 * (Adq[11] * Bdq[23] + Bdq[11] * Adq[23]) + 0.3e1 * t5 * (-t1 * Bd2q[287] + Ad2q[287]) + t3 * (t217 * Bdq[23] + t218 * Adq[23]);
    t218 = 0.3e1 * t5 * (-t1 * Bd2q[300] + Ad2q[300]) + t3 * (t6 * Bdq[12] * Bdq[12] + t7 * Adq[12] * Bdq[12] + 0.6e1 * Adq[12] * Adq[12]);
    AScalar t230 = t6 * Bdq[12];
    AScalar t231 = 0.6e1 * Adq[12];
    AScalar t232 = -0.9e1 * t2 * (Adq[12] * Bdq[13] + Bdq[12] * Adq[13]) + 0.3e1 * t5 * (-t1 * Bd2q[301] + Ad2q[301]) + t3 * (t230 * Bdq[13] + t231 * Adq[13]);
    AScalar t233 = -0.9e1 * t2 * (Adq[12] * Bdq[14] + Bdq[12] * Adq[14]) + 0.3e1 * t5 * (-t1 * Bd2q[302] + Ad2q[302]) + t3 * (t230 * Bdq[14] + t231 * Adq[14]);
    AScalar t234 = -0.9e1 * t2 * (Adq[12] * Bdq[15] + Bdq[12] * Adq[15]) + 0.3e1 * t5 * (-t1 * Bd2q[303] + Ad2q[303]) + t3 * (t230 * Bdq[15] + t231 * Adq[15]);
    AScalar t235 = -0.9e1 * t2 * (Adq[12] * Bdq[16] + Bdq[12] * Adq[16]) + 0.3e1 * t5 * (-t1 * Bd2q[304] + Ad2q[304]) + t3 * (t230 * Bdq[16] + t231 * Adq[16]);
    AScalar t236 = -0.9e1 * t2 * (Adq[12] * Bdq[17] + Bdq[12] * Adq[17]) + 0.3e1 * t5 * (-t1 * Bd2q[305] + Ad2q[305]) + t3 * (t230 * Bdq[17] + t231 * Adq[17]);
    AScalar t237 = -0.9e1 * t2 * (Adq[12] * Bdq[18] + Bdq[12] * Adq[18]) + 0.3e1 * t5 * (-t1 * Bd2q[306] + Ad2q[306]) + t3 * (t230 * Bdq[18] + t231 * Adq[18]);
    AScalar t238 = -0.9e1 * t2 * (Adq[12] * Bdq[19] + Bdq[12] * Adq[19]) + 0.3e1 * t5 * (-t1 * Bd2q[307] + Ad2q[307]) + t3 * (t230 * Bdq[19] + t231 * Adq[19]);
    AScalar t239 = -0.9e1 * t2 * (Adq[12] * Bdq[20] + Bdq[12] * Adq[20]) + 0.3e1 * t5 * (-t1 * Bd2q[308] + Ad2q[308]) + t3 * (t230 * Bdq[20] + t231 * Adq[20]);
    AScalar t240 = -0.9e1 * t2 * (Adq[12] * Bdq[21] + Bdq[12] * Adq[21]) + 0.3e1 * t5 * (-t1 * Bd2q[309] + Ad2q[309]) + t3 * (t230 * Bdq[21] + t231 * Adq[21]);
    AScalar t241 = -0.9e1 * t2 * (Adq[12] * Bdq[22] + Bdq[12] * Adq[22]) + 0.3e1 * t5 * (-t1 * Bd2q[310] + Ad2q[310]) + t3 * (t230 * Bdq[22] + t231 * Adq[22]);
    t230 = -0.9e1 * t2 * (Adq[12] * Bdq[23] + Bdq[12] * Adq[23]) + 0.3e1 * t5 * (-t1 * Bd2q[311] + Ad2q[311]) + t3 * (t230 * Bdq[23] + t231 * Adq[23]);
    t231 = 0.3e1 * t5 * (-t1 * Bd2q[325] + Ad2q[325]) + t3 * (t6 * Bdq[13] * Bdq[13] + t7 * Adq[13] * Bdq[13] + 0.6e1 * Adq[13] * Adq[13]);
    AScalar t242 = t6 * Bdq[13];
    AScalar t243 = 0.6e1 * Adq[13];
    AScalar t244 = -0.9e1 * t2 * (Adq[13] * Bdq[14] + Bdq[13] * Adq[14]) + 0.3e1 * t5 * (-t1 * Bd2q[326] + Ad2q[326]) + t3 * (t242 * Bdq[14] + t243 * Adq[14]);
    AScalar t245 = -0.9e1 * t2 * (Adq[13] * Bdq[15] + Bdq[13] * Adq[15]) + 0.3e1 * t5 * (-t1 * Bd2q[327] + Ad2q[327]) + t3 * (t242 * Bdq[15] + t243 * Adq[15]);
    AScalar t246 = -0.9e1 * t2 * (Adq[13] * Bdq[16] + Bdq[13] * Adq[16]) + 0.3e1 * t5 * (-t1 * Bd2q[328] + Ad2q[328]) + t3 * (t242 * Bdq[16] + t243 * Adq[16]);
    AScalar t247 = -0.9e1 * t2 * (Adq[13] * Bdq[17] + Bdq[13] * Adq[17]) + 0.3e1 * t5 * (-t1 * Bd2q[329] + Ad2q[329]) + t3 * (t242 * Bdq[17] + t243 * Adq[17]);
    AScalar t248 = -0.9e1 * t2 * (Adq[13] * Bdq[18] + Bdq[13] * Adq[18]) + 0.3e1 * t5 * (-t1 * Bd2q[330] + Ad2q[330]) + t3 * (t242 * Bdq[18] + t243 * Adq[18]);
    AScalar t249 = -0.9e1 * t2 * (Adq[13] * Bdq[19] + Bdq[13] * Adq[19]) + 0.3e1 * t5 * (-t1 * Bd2q[331] + Ad2q[331]) + t3 * (t242 * Bdq[19] + t243 * Adq[19]);
    AScalar t250 = -0.9e1 * t2 * (Adq[13] * Bdq[20] + Bdq[13] * Adq[20]) + 0.3e1 * t5 * (-t1 * Bd2q[332] + Ad2q[332]) + t3 * (t242 * Bdq[20] + t243 * Adq[20]);
    AScalar t251 = -0.9e1 * t2 * (Adq[13] * Bdq[21] + Bdq[13] * Adq[21]) + 0.3e1 * t5 * (-t1 * Bd2q[333] + Ad2q[333]) + t3 * (t242 * Bdq[21] + t243 * Adq[21]);
    AScalar t252 = -0.9e1 * t2 * (Adq[13] * Bdq[22] + Bdq[13] * Adq[22]) + 0.3e1 * t5 * (-t1 * Bd2q[334] + Ad2q[334]) + t3 * (t242 * Bdq[22] + t243 * Adq[22]);
    t242 = -0.9e1 * t2 * (Adq[13] * Bdq[23] + Bdq[13] * Adq[23]) + 0.3e1 * t5 * (-t1 * Bd2q[335] + Ad2q[335]) + t3 * (t242 * Bdq[23] + t243 * Adq[23]);
    t243 = 0.3e1 * t5 * (-t1 * Bd2q[350] + Ad2q[350]) + t3 * (t6 * Bdq[14] * Bdq[14] + t7 * Adq[14] * Bdq[14] + 0.6e1 * Adq[14] * Adq[14]);
    AScalar t253 = t6 * Bdq[14];
    AScalar t254 = 0.6e1 * Adq[14];
    AScalar t255 = -0.9e1 * t2 * (Adq[14] * Bdq[15] + Bdq[14] * Adq[15]) + 0.3e1 * t5 * (-t1 * Bd2q[351] + Ad2q[351]) + t3 * (t253 * Bdq[15] + t254 * Adq[15]);
    AScalar t256 = -0.9e1 * t2 * (Adq[14] * Bdq[16] + Bdq[14] * Adq[16]) + 0.3e1 * t5 * (-t1 * Bd2q[352] + Ad2q[352]) + t3 * (t253 * Bdq[16] + t254 * Adq[16]);
    AScalar t257 = -0.9e1 * t2 * (Adq[14] * Bdq[17] + Bdq[14] * Adq[17]) + 0.3e1 * t5 * (-t1 * Bd2q[353] + Ad2q[353]) + t3 * (t253 * Bdq[17] + t254 * Adq[17]);
    AScalar t258 = -0.9e1 * t2 * (Adq[14] * Bdq[18] + Bdq[14] * Adq[18]) + 0.3e1 * t5 * (-t1 * Bd2q[354] + Ad2q[354]) + t3 * (t253 * Bdq[18] + t254 * Adq[18]);
    AScalar t259 = -0.9e1 * t2 * (Adq[14] * Bdq[19] + Bdq[14] * Adq[19]) + 0.3e1 * t5 * (-t1 * Bd2q[355] + Ad2q[355]) + t3 * (t253 * Bdq[19] + t254 * Adq[19]);
    AScalar t260 = -0.9e1 * t2 * (Adq[14] * Bdq[20] + Bdq[14] * Adq[20]) + 0.3e1 * t5 * (-t1 * Bd2q[356] + Ad2q[356]) + t3 * (t253 * Bdq[20] + t254 * Adq[20]);
    AScalar t261 = -0.9e1 * t2 * (Adq[14] * Bdq[21] + Bdq[14] * Adq[21]) + 0.3e1 * t5 * (-t1 * Bd2q[357] + Ad2q[357]) + t3 * (t253 * Bdq[21] + t254 * Adq[21]);
    AScalar t262 = -0.9e1 * t2 * (Adq[14] * Bdq[22] + Bdq[14] * Adq[22]) + 0.3e1 * t5 * (-t1 * Bd2q[358] + Ad2q[358]) + t3 * (t253 * Bdq[22] + t254 * Adq[22]);
    t253 = -0.9e1 * t2 * (Adq[14] * Bdq[23] + Bdq[14] * Adq[23]) + 0.3e1 * t5 * (-t1 * Bd2q[359] + Ad2q[359]) + t3 * (t253 * Bdq[23] + t254 * Adq[23]);
    t254 = 0.3e1 * t5 * (-t1 * Bd2q[375] + Ad2q[375]) + t3 * (t6 * Bdq[15] * Bdq[15] + t7 * Adq[15] * Bdq[15] + 0.6e1 * Adq[15] * Adq[15]);
    AScalar t263 = t6 * Bdq[15];
    AScalar t264 = 0.6e1 * Adq[15];
    AScalar t265 = -0.9e1 * t2 * (Adq[15] * Bdq[16] + Bdq[15] * Adq[16]) + 0.3e1 * t5 * (-t1 * Bd2q[376] + Ad2q[376]) + t3 * (t263 * Bdq[16] + t264 * Adq[16]);
    AScalar t266 = -0.9e1 * t2 * (Adq[15] * Bdq[17] + Bdq[15] * Adq[17]) + 0.3e1 * t5 * (-t1 * Bd2q[377] + Ad2q[377]) + t3 * (t263 * Bdq[17] + t264 * Adq[17]);
    AScalar t267 = -0.9e1 * t2 * (Adq[15] * Bdq[18] + Bdq[15] * Adq[18]) + 0.3e1 * t5 * (-t1 * Bd2q[378] + Ad2q[378]) + t3 * (t263 * Bdq[18] + t264 * Adq[18]);
    AScalar t268 = -0.9e1 * t2 * (Adq[15] * Bdq[19] + Bdq[15] * Adq[19]) + 0.3e1 * t5 * (-t1 * Bd2q[379] + Ad2q[379]) + t3 * (t263 * Bdq[19] + t264 * Adq[19]);
    AScalar t269 = -0.9e1 * t2 * (Adq[15] * Bdq[20] + Bdq[15] * Adq[20]) + 0.3e1 * t5 * (-t1 * Bd2q[380] + Ad2q[380]) + t3 * (t263 * Bdq[20] + t264 * Adq[20]);
    AScalar t270 = -0.9e1 * t2 * (Adq[15] * Bdq[21] + Bdq[15] * Adq[21]) + 0.3e1 * t5 * (-t1 * Bd2q[381] + Ad2q[381]) + t3 * (t263 * Bdq[21] + t264 * Adq[21]);
    AScalar t271 = -0.9e1 * t2 * (Adq[15] * Bdq[22] + Adq[22] * Bdq[15]) + 0.3e1 * t5 * (-t1 * Bd2q[382] + Ad2q[382]) + t3 * (t263 * Bdq[22] + t264 * Adq[22]);
    t263 = -0.9e1 * t2 * (Adq[15] * Bdq[23] + Adq[23] * Bdq[15]) - 0.3e1 * t5 * (t1 * Bd2q[383] - Ad2q[383]) + t3 * (t263 * Bdq[23] + t264 * Adq[23]);
    t264 = 0.3e1 * t5 * (-t1 * Bd2q[400] + Ad2q[400]) + t3 * (t6 * Bdq[16] * Bdq[16] + t7 * Adq[16] * Bdq[16] + 0.6e1 * Adq[16] * Adq[16]);
    AScalar t272 = t6 * Bdq[16];
    AScalar t273 = 0.6e1 * Adq[16];
    AScalar t274 = -0.9e1 * t2 * (Adq[16] * Bdq[17] + Adq[17] * Bdq[16]) + 0.3e1 * t5 * (-t1 * Bd2q[401] + Ad2q[401]) + t3 * (t272 * Bdq[17] + t273 * Adq[17]);
    AScalar t275 = -0.9e1 * t2 * (Adq[16] * Bdq[18] + Adq[18] * Bdq[16]) + 0.3e1 * t5 * (-t1 * Bd2q[402] + Ad2q[402]) + t3 * (t272 * Bdq[18] + t273 * Adq[18]);
    AScalar t276 = -0.9e1 * t2 * (Adq[16] * Bdq[19] + Adq[19] * Bdq[16]) + 0.3e1 * t5 * (-t1 * Bd2q[403] + Ad2q[403]) + t3 * (t272 * Bdq[19] + t273 * Adq[19]);
    AScalar t277 = -0.9e1 * t2 * (Adq[16] * Bdq[20] + Bdq[16] * Adq[20]) + 0.3e1 * t5 * (-t1 * Bd2q[404] + Ad2q[404]) + t3 * (t272 * Bdq[20] + t273 * Adq[20]);
    AScalar t278 = -0.9e1 * t2 * (Adq[16] * Bdq[21] + Adq[21] * Bdq[16]) + 0.3e1 * t5 * (-t1 * Bd2q[405] + Ad2q[405]) + t3 * (t272 * Bdq[21] + t273 * Adq[21]);
    AScalar t279 = -0.9e1 * t2 * (Adq[16] * Bdq[22] + Adq[22] * Bdq[16]) + 0.3e1 * t5 * (-t1 * Bd2q[406] + Ad2q[406]) + t3 * (t272 * Bdq[22] + t273 * Adq[22]);
    t272 = -0.9e1 * t2 * (Adq[16] * Bdq[23] + Adq[23] * Bdq[16]) + 0.3e1 * t5 * (-t1 * Bd2q[407] + Ad2q[407]) + t3 * (t272 * Bdq[23] + t273 * Adq[23]);
    t273 = 0.3e1 * t5 * (-t1 * Bd2q[425] + Ad2q[425]) + t3 * (t6 * Bdq[17] * Bdq[17] + t7 * Adq[17] * Bdq[17] + 0.6e1 * Adq[17] * Adq[17]);
    AScalar t280 = t6 * Bdq[17];
    AScalar t281 = 0.6e1 * Adq[17];
    AScalar t282 = -0.9e1 * t2 * (Adq[17] * Bdq[18] + Adq[18] * Bdq[17]) + 0.3e1 * t5 * (-t1 * Bd2q[426] + Ad2q[426]) + t3 * (t280 * Bdq[18] + t281 * Adq[18]);
    AScalar t283 = -0.9e1 * t2 * (Adq[17] * Bdq[19] + Adq[19] * Bdq[17]) + 0.3e1 * t5 * (-t1 * Bd2q[427] + Ad2q[427]) + t3 * (t280 * Bdq[19] + t281 * Adq[19]);
    AScalar t284 = -0.9e1 * t2 * (Adq[17] * Bdq[20] + Adq[20] * Bdq[17]) + 0.3e1 * t5 * (-t1 * Bd2q[428] + Ad2q[428]) + t3 * (t280 * Bdq[20] + t281 * Adq[20]);
    AScalar t285 = -0.9e1 * t2 * (Adq[17] * Bdq[21] + Adq[21] * Bdq[17]) + 0.3e1 * t5 * (-t1 * Bd2q[429] + Ad2q[429]) + t3 * (t280 * Bdq[21] + t281 * Adq[21]);
    AScalar t286 = -0.9e1 * t2 * (Adq[17] * Bdq[22] + Adq[22] * Bdq[17]) + 0.3e1 * t5 * (-t1 * Bd2q[430] + Ad2q[430]) + t3 * (t280 * Bdq[22] + t281 * Adq[22]);
    t280 = -0.9e1 * t2 * (Adq[17] * Bdq[23] + Adq[23] * Bdq[17]) + 0.3e1 * t5 * (-t1 * Bd2q[431] + Ad2q[431]) + t3 * (t280 * Bdq[23] + t281 * Adq[23]);
    t281 = 0.3e1 * t5 * (-t1 * Bd2q[450] + Ad2q[450]) + t3 * (t6 * Bdq[18] * Bdq[18] + t7 * Adq[18] * Bdq[18] + 0.6e1 * Adq[18] * Adq[18]);
    AScalar t287 = t6 * Bdq[18];
    AScalar t288 = 0.6e1 * Adq[18];
    AScalar t289 = -0.9e1 * t2 * (Adq[18] * Bdq[19] + Adq[19] * Bdq[18]) + 0.3e1 * t5 * (-t1 * Bd2q[451] + Ad2q[451]) + t3 * (t287 * Bdq[19] + t288 * Adq[19]);
    AScalar t290 = -0.9e1 * t2 * (Adq[18] * Bdq[20] + Adq[20] * Bdq[18]) + 0.3e1 * t5 * (-t1 * Bd2q[452] + Ad2q[452]) + t3 * (t287 * Bdq[20] + t288 * Adq[20]);
    AScalar t291 = -0.9e1 * t2 * (Adq[18] * Bdq[21] + Adq[21] * Bdq[18]) + 0.3e1 * t5 * (-t1 * Bd2q[453] + Ad2q[453]) + t3 * (t287 * Bdq[21] + t288 * Adq[21]);
    AScalar t292 = -0.9e1 * t2 * (Adq[18] * Bdq[22] + Adq[22] * Bdq[18]) + 0.3e1 * t5 * (-t1 * Bd2q[454] + Ad2q[454]) + t3 * (t287 * Bdq[22] + t288 * Adq[22]);
    t287 = -0.9e1 * t2 * (Adq[18] * Bdq[23] + Adq[23] * Bdq[18]) + 0.3e1 * t5 * (-t1 * Bd2q[455] + Ad2q[455]) + t3 * (t287 * Bdq[23] + t288 * Adq[23]);
    t288 = 0.3e1 * t5 * (-t1 * Bd2q[475] + Ad2q[475]) + t3 * (t6 * Bdq[19] * Bdq[19] + t7 * Adq[19] * Bdq[19] + 0.6e1 * Adq[19] * Adq[19]);
    AScalar t293 = t6 * Bdq[19];
    AScalar t294 = 0.6e1 * Adq[19];
    AScalar t295 = -0.9e1 * t2 * (Adq[19] * Bdq[20] + Adq[20] * Bdq[19]) + 0.3e1 * t5 * (-t1 * Bd2q[476] + Ad2q[476]) + t3 * (t293 * Bdq[20] + t294 * Adq[20]);
    AScalar t296 = -0.9e1 * t2 * (Adq[19] * Bdq[21] + Adq[21] * Bdq[19]) + 0.3e1 * t5 * (-t1 * Bd2q[477] + Ad2q[477]) + t3 * (t293 * Bdq[21] + t294 * Adq[21]);
    AScalar t297 = -0.9e1 * t2 * (Adq[19] * Bdq[22] + Adq[22] * Bdq[19]) + 0.3e1 * t5 * (-t1 * Bd2q[478] + Ad2q[478]) + t3 * (t293 * Bdq[22] + t294 * Adq[22]);
    t293 = -0.9e1 * t2 * (Adq[19] * Bdq[23] + Bdq[19] * Adq[23]) + 0.3e1 * t5 * (-t1 * Bd2q[479] + Ad2q[479]) + t3 * (t293 * Bdq[23] + t294 * Adq[23]);
    t294 = 0.3e1 * t5 * (-t1 * Bd2q[500] + Ad2q[500]) + t3 * (t6 * Bdq[20] * Bdq[20] + t7 * Adq[20] * Bdq[20] + 0.6e1 * Adq[20] * Adq[20]);
    AScalar t298 = t6 * Bdq[20];
    AScalar t299 = 0.6e1 * Adq[20];
    AScalar t300 = -0.9e1 * t2 * (Adq[20] * Bdq[21] + Bdq[20] * Adq[21]) + 0.3e1 * t5 * (-t1 * Bd2q[501] + Ad2q[501]) + t3 * (t298 * Bdq[21] + t299 * Adq[21]);
    AScalar t301 = -0.9e1 * t2 * (Adq[20] * Bdq[22] + Adq[22] * Bdq[20]) + 0.3e1 * t5 * (-t1 * Bd2q[502] + Ad2q[502]) + t3 * (t298 * Bdq[22] + t299 * Adq[22]);
    t298 = -0.9e1 * t2 * (Adq[20] * Bdq[23] + Adq[23] * Bdq[20]) + 0.3e1 * t5 * (-t1 * Bd2q[503] + Ad2q[503]) + t3 * (t298 * Bdq[23] + t299 * Adq[23]);
    t299 = 0.3e1 * t5 * (-t1 * Bd2q[525] + Ad2q[525]) + t3 * (t6 * Bdq[21] * Bdq[21] + t7 * Adq[21] * Bdq[21] + 0.6e1 * Adq[21] * Adq[21]);
    AScalar t302 = t6 * Bdq[21];
    AScalar t303 = 0.6e1 * Adq[21];
    AScalar t304 = -0.9e1 * t2 * (Adq[21] * Bdq[22] + Adq[22] * Bdq[21]) + 0.3e1 * t5 * (-t1 * Bd2q[526] + Ad2q[526]) + t3 * (t302 * Bdq[22] + t303 * Adq[22]);
    t302 = -0.9e1 * t2 * (Adq[21] * Bdq[23] + Adq[23] * Bdq[21]) + 0.3e1 * t5 * (-t1 * Bd2q[527] + Ad2q[527]) + t3 * (t302 * Bdq[23] + t303 * Adq[23]);
    t303 = 0.3e1 * t5 * (-t1 * Bd2q[550] + Ad2q[550]) + t3 * (t6 * Bdq[22] * Bdq[22] + t7 * Adq[22] * Bdq[22] + 0.6e1 * Adq[22] * Adq[22]);
    t2 = -0.9e1 * t2 * (Adq[22] * Bdq[23] + Adq[23] * Bdq[22]) + 0.3e1 * t5 * (-t1 * Bd2q[551] + Ad2q[551]) + t3 * (t6 * Bdq[22] * Bdq[23] + 0.6e1 * Adq[22] * Adq[23]);
    t1 = 0.3e1 * t5 * (-t1 * Bd2q[575] + Ad2q[575]) + t3 * (t6 * Bdq[23] * Bdq[23] + t7 * Adq[23] * Bdq[23] + 0.6e1 * Adq[23] * Adq[23]);
    hessian[0] = t8;
    hessian[1] = t10;
    hessian[2] = t11;
    hessian[3] = t12;
    hessian[4] = t13;
    hessian[5] = t14;
    hessian[6] = t15;
    hessian[7] = t16;
    hessian[8] = t17;
    hessian[9] = t18;
    hessian[10] = t19;
    hessian[11] = t20;
    hessian[12] = t21;
    hessian[13] = t22;
    hessian[14] = t23;
    hessian[15] = t24;
    hessian[16] = t25;
    hessian[17] = t26;
    hessian[18] = t27;
    hessian[19] = t28;
    hessian[20] = t29;
    hessian[21] = t30;
    hessian[22] = t31;
    hessian[23] = t4;
    hessian[24] = t10;
    hessian[25] = t9;
    hessian[26] = t34;
    hessian[27] = t35;
    hessian[28] = t36;
    hessian[29] = t37;
    hessian[30] = t38;
    hessian[31] = t39;
    hessian[32] = t40;
    hessian[33] = t41;
    hessian[34] = t42;
    hessian[35] = t43;
    hessian[36] = t44;
    hessian[37] = t45;
    hessian[38] = t46;
    hessian[39] = t47;
    hessian[40] = t48;
    hessian[41] = t49;
    hessian[42] = t50;
    hessian[43] = t51;
    hessian[44] = t52;
    hessian[45] = t53;
    hessian[46] = t54;
    hessian[47] = t32;
    hessian[48] = t11;
    hessian[49] = t34;
    hessian[50] = t33;
    hessian[51] = t57;
    hessian[52] = t58;
    hessian[53] = t59;
    hessian[54] = t60;
    hessian[55] = t61;
    hessian[56] = t62;
    hessian[57] = t63;
    hessian[58] = t64;
    hessian[59] = t65;
    hessian[60] = t66;
    hessian[61] = t67;
    hessian[62] = t68;
    hessian[63] = t69;
    hessian[64] = t70;
    hessian[65] = t71;
    hessian[66] = t72;
    hessian[67] = t73;
    hessian[68] = t74;
    hessian[69] = t75;
    hessian[70] = t76;
    hessian[71] = t55;
    hessian[72] = t12;
    hessian[73] = t35;
    hessian[74] = t57;
    hessian[75] = t56;
    hessian[76] = t79;
    hessian[77] = t80;
    hessian[78] = t81;
    hessian[79] = t82;
    hessian[80] = t83;
    hessian[81] = t84;
    hessian[82] = t85;
    hessian[83] = t86;
    hessian[84] = t87;
    hessian[85] = t88;
    hessian[86] = t89;
    hessian[87] = t90;
    hessian[88] = t91;
    hessian[89] = t92;
    hessian[90] = t93;
    hessian[91] = t94;
    hessian[92] = t95;
    hessian[93] = t96;
    hessian[94] = t97;
    hessian[95] = t77;
    hessian[96] = t13;
    hessian[97] = t36;
    hessian[98] = t58;
    hessian[99] = t79;
    hessian[100] = t78;
    hessian[101] = t100;
    hessian[102] = t101;
    hessian[103] = t102;
    hessian[104] = t103;
    hessian[105] = t104;
    hessian[106] = t105;
    hessian[107] = t106;
    hessian[108] = t107;
    hessian[109] = t108;
    hessian[110] = t109;
    hessian[111] = t110;
    hessian[112] = t111;
    hessian[113] = t112;
    hessian[114] = t113;
    hessian[115] = t114;
    hessian[116] = t115;
    hessian[117] = t116;
    hessian[118] = t117;
    hessian[119] = t98;
    hessian[120] = t14;
    hessian[121] = t37;
    hessian[122] = t59;
    hessian[123] = t80;
    hessian[124] = t100;
    hessian[125] = t99;
    hessian[126] = t120;
    hessian[127] = t121;
    hessian[128] = t122;
    hessian[129] = t123;
    hessian[130] = t124;
    hessian[131] = t125;
    hessian[132] = t126;
    hessian[133] = t127;
    hessian[134] = t128;
    hessian[135] = t129;
    hessian[136] = t130;
    hessian[137] = t131;
    hessian[138] = t132;
    hessian[139] = t133;
    hessian[140] = t134;
    hessian[141] = t135;
    hessian[142] = t136;
    hessian[143] = t118;
    hessian[144] = t15;
    hessian[145] = t38;
    hessian[146] = t60;
    hessian[147] = t81;
    hessian[148] = t101;
    hessian[149] = t120;
    hessian[150] = t119;
    hessian[151] = t139;
    hessian[152] = t140;
    hessian[153] = t141;
    hessian[154] = t142;
    hessian[155] = t143;
    hessian[156] = t144;
    hessian[157] = t145;
    hessian[158] = t146;
    hessian[159] = t147;
    hessian[160] = t148;
    hessian[161] = t149;
    hessian[162] = t150;
    hessian[163] = t151;
    hessian[164] = t152;
    hessian[165] = t153;
    hessian[166] = t154;
    hessian[167] = t137;
    hessian[168] = t16;
    hessian[169] = t39;
    hessian[170] = t61;
    hessian[171] = t82;
    hessian[172] = t102;
    hessian[173] = t121;
    hessian[174] = t139;
    hessian[175] = t138;
    hessian[176] = t157;
    hessian[177] = t158;
    hessian[178] = t159;
    hessian[179] = t160;
    hessian[180] = t161;
    hessian[181] = t162;
    hessian[182] = t163;
    hessian[183] = t164;
    hessian[184] = t165;
    hessian[185] = t166;
    hessian[186] = t167;
    hessian[187] = t168;
    hessian[188] = t169;
    hessian[189] = t170;
    hessian[190] = t171;
    hessian[191] = t155;
    hessian[192] = t17;
    hessian[193] = t40;
    hessian[194] = t62;
    hessian[195] = t83;
    hessian[196] = t103;
    hessian[197] = t122;
    hessian[198] = t140;
    hessian[199] = t157;
    hessian[200] = t156;
    hessian[201] = t174;
    hessian[202] = t175;
    hessian[203] = t176;
    hessian[204] = t177;
    hessian[205] = t178;
    hessian[206] = t179;
    hessian[207] = t180;
    hessian[208] = t181;
    hessian[209] = t182;
    hessian[210] = t183;
    hessian[211] = t184;
    hessian[212] = t185;
    hessian[213] = t186;
    hessian[214] = t187;
    hessian[215] = t172;
    hessian[216] = t18;
    hessian[217] = t41;
    hessian[218] = t63;
    hessian[219] = t84;
    hessian[220] = t104;
    hessian[221] = t123;
    hessian[222] = t141;
    hessian[223] = t158;
    hessian[224] = t174;
    hessian[225] = t173;
    hessian[226] = t190;
    hessian[227] = t191;
    hessian[228] = t192;
    hessian[229] = t193;
    hessian[230] = t194;
    hessian[231] = t195;
    hessian[232] = t196;
    hessian[233] = t197;
    hessian[234] = t198;
    hessian[235] = t199;
    hessian[236] = t200;
    hessian[237] = t201;
    hessian[238] = t202;
    hessian[239] = t188;
    hessian[240] = t19;
    hessian[241] = t42;
    hessian[242] = t64;
    hessian[243] = t85;
    hessian[244] = t105;
    hessian[245] = t124;
    hessian[246] = t142;
    hessian[247] = t159;
    hessian[248] = t175;
    hessian[249] = t190;
    hessian[250] = t189;
    hessian[251] = t205;
    hessian[252] = t206;
    hessian[253] = t207;
    hessian[254] = t208;
    hessian[255] = t209;
    hessian[256] = t210;
    hessian[257] = t211;
    hessian[258] = t212;
    hessian[259] = t213;
    hessian[260] = t214;
    hessian[261] = t215;
    hessian[262] = t216;
    hessian[263] = t203;
    hessian[264] = t20;
    hessian[265] = t43;
    hessian[266] = t65;
    hessian[267] = t86;
    hessian[268] = t106;
    hessian[269] = t125;
    hessian[270] = t143;
    hessian[271] = t160;
    hessian[272] = t176;
    hessian[273] = t191;
    hessian[274] = t205;
    hessian[275] = t204;
    hessian[276] = t219;
    hessian[277] = t220;
    hessian[278] = t221;
    hessian[279] = t222;
    hessian[280] = t223;
    hessian[281] = t224;
    hessian[282] = t225;
    hessian[283] = t226;
    hessian[284] = t227;
    hessian[285] = t228;
    hessian[286] = t229;
    hessian[287] = t217;
    hessian[288] = t21;
    hessian[289] = t44;
    hessian[290] = t66;
    hessian[291] = t87;
    hessian[292] = t107;
    hessian[293] = t126;
    hessian[294] = t144;
    hessian[295] = t161;
    hessian[296] = t177;
    hessian[297] = t192;
    hessian[298] = t206;
    hessian[299] = t219;
    hessian[300] = t218;
    hessian[301] = t232;
    hessian[302] = t233;
    hessian[303] = t234;
    hessian[304] = t235;
    hessian[305] = t236;
    hessian[306] = t237;
    hessian[307] = t238;
    hessian[308] = t239;
    hessian[309] = t240;
    hessian[310] = t241;
    hessian[311] = t230;
    hessian[312] = t22;
    hessian[313] = t45;
    hessian[314] = t67;
    hessian[315] = t88;
    hessian[316] = t108;
    hessian[317] = t127;
    hessian[318] = t145;
    hessian[319] = t162;
    hessian[320] = t178;
    hessian[321] = t193;
    hessian[322] = t207;
    hessian[323] = t220;
    hessian[324] = t232;
    hessian[325] = t231;
    hessian[326] = t244;
    hessian[327] = t245;
    hessian[328] = t246;
    hessian[329] = t247;
    hessian[330] = t248;
    hessian[331] = t249;
    hessian[332] = t250;
    hessian[333] = t251;
    hessian[334] = t252;
    hessian[335] = t242;
    hessian[336] = t23;
    hessian[337] = t46;
    hessian[338] = t68;
    hessian[339] = t89;
    hessian[340] = t109;
    hessian[341] = t128;
    hessian[342] = t146;
    hessian[343] = t163;
    hessian[344] = t179;
    hessian[345] = t194;
    hessian[346] = t208;
    hessian[347] = t221;
    hessian[348] = t233;
    hessian[349] = t244;
    hessian[350] = t243;
    hessian[351] = t255;
    hessian[352] = t256;
    hessian[353] = t257;
    hessian[354] = t258;
    hessian[355] = t259;
    hessian[356] = t260;
    hessian[357] = t261;
    hessian[358] = t262;
    hessian[359] = t253;
    hessian[360] = t24;
    hessian[361] = t47;
    hessian[362] = t69;
    hessian[363] = t90;
    hessian[364] = t110;
    hessian[365] = t129;
    hessian[366] = t147;
    hessian[367] = t164;
    hessian[368] = t180;
    hessian[369] = t195;
    hessian[370] = t209;
    hessian[371] = t222;
    hessian[372] = t234;
    hessian[373] = t245;
    hessian[374] = t255;
    hessian[375] = t254;
    hessian[376] = t265;
    hessian[377] = t266;
    hessian[378] = t267;
    hessian[379] = t268;
    hessian[380] = t269;
    hessian[381] = t270;
    hessian[382] = t271;
    hessian[383] = t263;
    hessian[384] = t25;
    hessian[385] = t48;
    hessian[386] = t70;
    hessian[387] = t91;
    hessian[388] = t111;
    hessian[389] = t130;
    hessian[390] = t148;
    hessian[391] = t165;
    hessian[392] = t181;
    hessian[393] = t196;
    hessian[394] = t210;
    hessian[395] = t223;
    hessian[396] = t235;
    hessian[397] = t246;
    hessian[398] = t256;
    hessian[399] = t265;
    hessian[400] = t264;
    hessian[401] = t274;
    hessian[402] = t275;
    hessian[403] = t276;
    hessian[404] = t277;
    hessian[405] = t278;
    hessian[406] = t279;
    hessian[407] = t272;
    hessian[408] = t26;
    hessian[409] = t49;
    hessian[410] = t71;
    hessian[411] = t92;
    hessian[412] = t112;
    hessian[413] = t131;
    hessian[414] = t149;
    hessian[415] = t166;
    hessian[416] = t182;
    hessian[417] = t197;
    hessian[418] = t211;
    hessian[419] = t224;
    hessian[420] = t236;
    hessian[421] = t247;
    hessian[422] = t257;
    hessian[423] = t266;
    hessian[424] = t274;
    hessian[425] = t273;
    hessian[426] = t282;
    hessian[427] = t283;
    hessian[428] = t284;
    hessian[429] = t285;
    hessian[430] = t286;
    hessian[431] = t280;
    hessian[432] = t27;
    hessian[433] = t50;
    hessian[434] = t72;
    hessian[435] = t93;
    hessian[436] = t113;
    hessian[437] = t132;
    hessian[438] = t150;
    hessian[439] = t167;
    hessian[440] = t183;
    hessian[441] = t198;
    hessian[442] = t212;
    hessian[443] = t225;
    hessian[444] = t237;
    hessian[445] = t248;
    hessian[446] = t258;
    hessian[447] = t267;
    hessian[448] = t275;
    hessian[449] = t282;
    hessian[450] = t281;
    hessian[451] = t289;
    hessian[452] = t290;
    hessian[453] = t291;
    hessian[454] = t292;
    hessian[455] = t287;
    hessian[456] = t28;
    hessian[457] = t51;
    hessian[458] = t73;
    hessian[459] = t94;
    hessian[460] = t114;
    hessian[461] = t133;
    hessian[462] = t151;
    hessian[463] = t168;
    hessian[464] = t184;
    hessian[465] = t199;
    hessian[466] = t213;
    hessian[467] = t226;
    hessian[468] = t238;
    hessian[469] = t249;
    hessian[470] = t259;
    hessian[471] = t268;
    hessian[472] = t276;
    hessian[473] = t283;
    hessian[474] = t289;
    hessian[475] = t288;
    hessian[476] = t295;
    hessian[477] = t296;
    hessian[478] = t297;
    hessian[479] = t293;
    hessian[480] = t29;
    hessian[481] = t52;
    hessian[482] = t74;
    hessian[483] = t95;
    hessian[484] = t115;
    hessian[485] = t134;
    hessian[486] = t152;
    hessian[487] = t169;
    hessian[488] = t185;
    hessian[489] = t200;
    hessian[490] = t214;
    hessian[491] = t227;
    hessian[492] = t239;
    hessian[493] = t250;
    hessian[494] = t260;
    hessian[495] = t269;
    hessian[496] = t277;
    hessian[497] = t284;
    hessian[498] = t290;
    hessian[499] = t295;
    hessian[500] = t294;
    hessian[501] = t300;
    hessian[502] = t301;
    hessian[503] = t298;
    hessian[504] = t30;
    hessian[505] = t53;
    hessian[506] = t75;
    hessian[507] = t96;
    hessian[508] = t116;
    hessian[509] = t135;
    hessian[510] = t153;
    hessian[511] = t170;
    hessian[512] = t186;
    hessian[513] = t201;
    hessian[514] = t215;
    hessian[515] = t228;
    hessian[516] = t240;
    hessian[517] = t251;
    hessian[518] = t261;
    hessian[519] = t270;
    hessian[520] = t278;
    hessian[521] = t285;
    hessian[522] = t291;
    hessian[523] = t296;
    hessian[524] = t300;
    hessian[525] = t299;
    hessian[526] = t304;
    hessian[527] = t302;
    hessian[528] = t31;
    hessian[529] = t54;
    hessian[530] = t76;
    hessian[531] = t97;
    hessian[532] = t117;
    hessian[533] = t136;
    hessian[534] = t154;
    hessian[535] = t171;
    hessian[536] = t187;
    hessian[537] = t202;
    hessian[538] = t216;
    hessian[539] = t229;
    hessian[540] = t241;
    hessian[541] = t252;
    hessian[542] = t262;
    hessian[543] = t271;
    hessian[544] = t279;
    hessian[545] = t286;
    hessian[546] = t292;
    hessian[547] = t297;
    hessian[548] = t301;
    hessian[549] = t304;
    hessian[550] = t303;
    hessian[551] = t2;
    hessian[552] = t4;
    hessian[553] = t32;
    hessian[554] = t55;
    hessian[555] = t77;
    hessian[556] = t98;
    hessian[557] = t118;
    hessian[558] = t137;
    hessian[559] = t155;
    hessian[560] = t172;
    hessian[561] = t188;
    hessian[562] = t203;
    hessian[563] = t217;
    hessian[564] = t230;
    hessian[565] = t242;
    hessian[566] = t253;
    hessian[567] = t263;
    hessian[568] = t272;
    hessian[569] = t280;
    hessian[570] = t287;
    hessian[571] = t293;
    hessian[572] = t298;
    hessian[573] = t302;
    hessian[574] = t2;
    hessian[575] = t1;

    return Eigen::Matrix<AScalar, 24, 24>(Eigen::Map<Eigen::Matrix<AScalar,24,24,Eigen::ColMajor> >(hessian));
}


template <int dim>
void FEMSolver<dim>::BuildAcceleratorTree(bool is_master)
{
    accelerator.clear();

    if(is_master)
    {
        auto it = master_nodes_3d[0].begin();
        for(int i=0; i<master_nodes_3d[0].size(); ++i)
        {
            Eigen::VectorXd point = deformed.segment<dim>(dim*it->first);
            accelerator.insert(std::make_tuple(Point_d(point[0], point[1], point[2]), it->first));
            it++;
        }
    }
    else
    {
        auto it = slave_nodes_3d[0].begin();
        for(int i=0; i<slave_nodes_3d[0].size(); ++i)
        {
            Eigen::VectorXd point = deformed.segment<dim>(dim*it->first);
            accelerator.insert(std::make_tuple(Point_d(point[0], point[1], point[2]), it->first));
            it++;
        }
    }
    
}

template <int dim>
void FEMSolver<dim>::BuildAcceleratorTreeSC()
{
    accelerator.clear();

    for(int i=0; i<num_nodes; ++i)
    {
        if(is_surface_vertex[i] == 0) continue;
        Eigen::VectorXd point = deformed.segment<dim>(dim*i);
        accelerator.insert(std::make_tuple(Point_d(point[0], point[1], point[2]), i));
    }
}

template <int dim>
void FEMSolver<dim>::BuildAcceleratorTreeSCTest()
{
    accelerator.clear();

    for(int i=0; i<7786; ++i)
    {
        if(is_surface_vertex[i] == 0) continue;
        Eigen::VectorXd point = deformed.segment<dim>(dim*i);
        accelerator.insert(std::make_tuple(Point_d(point[0], point[1], point[2]), i));
    }
}

template <int dim>
void FEMSolver<dim>::addFastIMLSEnergy(T& energy)
{
    BuildAcceleratorTree();
	current_indices.clear();
    close_slave_nodes.clear();
    auto it = slave_nodes_3d[0].begin();

    dist_info.clear();
	for(int i=0; i<slave_nodes_3d[0].size(); ++i)
	{
		current_indices.push_back(std::vector<int>());

		Vector3a xi = deformed.segment<3>(3*it->first);

		Fuzzy_sphere fs(Point_d(xi[0], xi[1], xi[2]), radius+radius*0.2, radius*0.2);

		std::vector<std::tuple<Point_d, int>> points_query;
		accelerator.search(std::back_inserter(points_query), fs);

		for(int k=0; k<points_query.size(); ++k)
			current_indices.back().push_back(std::get<1>(points_query[k]));

        it++;
	}

	it = slave_nodes_3d[0].begin();
	for(int i=0; i<slave_nodes_3d[0].size(); ++i)
	{
		AScalar fx = 0;
		AScalar gx = 0;

		Vector3a xi = deformed.segment<3>(3*it->first);

		for(int k=0; k<current_indices[i].size(); ++k)
		{
            Vector3a ck = deformed.segment<3>(current_indices[i][k]*3);
            //f(i == 1567) std::cout<<k<<" k out of "<<current_indices[i].size()<<std::endl;
            int valence = master_nodes_adjacency[current_indices[i][k]].size();
            VectorXa vs;

            if(valence <= 6)
            {
                vs.resize(18);
                for(int a=0; a<6; ++a)
                {
                    vs.segment<3>(a*3) = ck;
                }
            }else
            {
                vs.resize(33);
                for(int a=0; a<11; ++a)
                {
                    vs.segment<3>(a*3) = ck;
                }
            }

           
            for(int a=0; a<master_nodes_adjacency[current_indices[i][k]].size(); ++a)
            {
                vs.segment<3>(a*3) = deformed.segment<3>(master_nodes_adjacency[current_indices[i][k]][a]*3);
            }
            
            if((ck-xi).norm()<=radius) {
                if(valence <= 6)
                {
                    fx += fx_func(xi, vs, ck, radius);
                    gx += gx_func(xi, ck, radius);
                }else
                {
                    fx += fx_func11(xi, vs, ck, radius);
                    gx += gx_func11(xi, ck, radius);
                }
            }
		}

		AScalar dist = fx/gx;

        

		if(abs(gx)>1e-6)
        {
            //std::cout<<dist<<std::endl;
            if(BARRIER_ENERGY)
            {
                if(dist > 0)
                {
                    if(dist <= barrier_distance)
                        energy += -IMLS_param*pow((dist-barrier_distance),2)*log(dist/barrier_distance);
                }
                else
                    std::cout<<"ERROR!!! Negative Distance!"<<std::endl;
            }
            else if(!BARRIER_ENERGY && dist<0)
                energy += -IMLS_param*pow((fx/gx),3);
            //dist_info.push_back(std::pair<int,double>(it->first,dist));
            close_slave_nodes.push_back(it->second);
            
        }
			
        
        it++;
	}
    std::cout<<"close slave nodes: "<<close_slave_nodes.size()<<std::endl;

    if(!IMLS_BOTH) return;

    BuildAcceleratorTree(false);
	current_indices.clear();
    it = master_nodes_3d[0].begin();

	for(int i=0; i<master_nodes_3d[0].size(); ++i)
	{
		current_indices.push_back(std::vector<int>());

		Vector3a xi = deformed.segment<3>(3*it->first);

		Fuzzy_sphere fs(Point_d(xi[0], xi[1], xi[2]), radius+radius*0.2, radius*0.2);

		std::vector<std::tuple<Point_d, int>> points_query;
		accelerator.search(std::back_inserter(points_query), fs);

		for(int k=0; k<points_query.size(); ++k)
			current_indices.back().push_back(std::get<1>(points_query[k]));

        it++;
	}

	it = master_nodes_3d[0].begin();
	for(int i=0; i<master_nodes_3d[0].size(); ++i)
	{
		AScalar fx = 0;
		AScalar gx = 0;

		Vector3a xi = deformed.segment<3>(3*it->first);

		for(int k=0; k<current_indices[i].size(); ++k)
		{
            Vector3a ck = deformed.segment<3>(current_indices[i][k]*3);
            //f(i == 1567) std::cout<<k<<" k out of "<<current_indices[i].size()<<std::endl;
            int valence = slave_nodes_adjacency[current_indices[i][k]-slave_nodes_shift].size();
            VectorXa vs;

            if(valence <= 6)
            {
                vs.resize(18);
                for(int a=0; a<6; ++a)
                {
                    vs.segment<3>(a*3) = ck;
                }
            }else
            {
                vs.resize(33);
                for(int a=0; a<11; ++a)
                {
                    vs.segment<3>(a*3) = ck;
                }
            }

           
            for(int a=0; a<slave_nodes_adjacency[current_indices[i][k]-slave_nodes_shift].size(); ++a)
            {
                vs.segment<3>(a*3) = deformed.segment<3>(slave_nodes_adjacency[current_indices[i][k]-slave_nodes_shift][a]*3);
            }
            
            if((ck-xi).norm()<=radius) {
                if(valence <= 6)
                {
                    fx += fx_func(xi, vs, ck, radius);
                    gx += gx_func(xi, ck, radius);
                }else
                {
                    fx += fx_func11(xi, vs, ck, radius);
                    gx += gx_func11(xi, ck, radius);
                }
            }
		}

		AScalar dist = fx/gx;

		if(abs(gx)>1e-6)
        {
            //std::cout<<dist<<std::endl;
            if(BARRIER_ENERGY)
            {
               if(dist > 0)
                {
                    if(dist <= barrier_distance)
                        energy += -IMLS_param*pow((dist-barrier_distance),2)*log(dist/barrier_distance);
                }
                else
                    std::cout<<"ERROR!!! Negative Distance!"<<std::endl;
            }
            else if(!BARRIER_ENERGY && dist<0)
                energy += -IMLS_param*pow((fx/gx),3);
            //dist_info.push_back(std::pair<int,double>(it->first,dist));
        }
        it++;
	}

}

template <int dim>
void FEMSolver<dim>::addFastIMLSSCEnergy(T& energy, bool test)
{

    BuildAcceleratorTreeSC();
	current_indices.resize(num_nodes);
    dist_info.clear();
   // dist_info.resize(num_nodes,std::pair<int,double>(-1,-1));

    // #pragma omp parallel for
	// for(int i=0; i<num_nodes; ++i)
	// {
    //     //current_indices.push_back(std::vector<int>());
    //     if(is_surface_vertex[i] == 0) continue;
	
	// 	Vector3a xi = deformed.segment<3>(3*i);

	// 	Fuzzy_sphere fs(Point_d(xi[0], xi[1], xi[2]), radius+radius*0.2, radius*0.2);

	// 	std::vector<std::tuple<Point_d, int>> points_query;
	// 	accelerator.search(std::back_inserter(points_query), fs);

    //     //std::cout<<points_query.size()<<" ";
	// 	for(int k=0; k<points_query.size(); ++k)
    //     {
    //         int index = std::get<1>(points_query[k]);
    //         if((geodist_close_matrix.coeff(i,index) == 0))
    //         {
    //             //std::cout<<i<<" "<<index<<std::endl;
    //             current_indices[i].push_back(index);
    //         }
               
    //     }
    //     //std::cout<<current_indices.back().size()<<"\n";
	// }

    std::vector<std::vector<int>> tbb_current_indices(num_nodes);
    tbb::parallel_for(
    tbb::blocked_range<size_t>(size_t(0), num_nodes),
    [&](const tbb::blocked_range<size_t>& r) {
        for (size_t i = r.begin(); i < r.end(); i++) {
            auto& local_current_index = tbb_current_indices[i];
            local_current_index.clear();
            //std::cout<<i<<std::endl;
            if(is_surface_vertex[i] != 0)
            {
                Vector3a xi = deformed.segment<3>(3*i);

                Fuzzy_sphere fs(Point_d(xi[0], xi[1], xi[2]), radius+radius*0.2, radius*0.2);

                std::vector<std::tuple<Point_d, int>> points_query;
                accelerator.search(std::back_inserter(points_query), fs);

                for(int k=0; k<points_query.size(); ++k)
                {
                   
                    int index = std::get<1>(points_query[k]);
                    if((geodist_close_matrix.coeff(i,index) == 0))
                    {
                        //  if(i == 1510)
                        // {
                        //     std::cout<<index<<" "; 
                        // }
                        local_current_index.push_back(index);
                    }
                        
                }
                // if(i == 1510)
                //     std::cout<<std::endl;
            }
        }   
    });

    // std::cout<<tbb_current_indices.size()<<std::endl;
    // int i=0;
    // for(const auto& t: tbb_current_indices)
    // {
    //     std::cout<<i<<" "<<t.size()<<std::endl;
    //     i++;
    // }

    std::vector<double> temp_energies(num_nodes,0);
    tbb::enumerable_thread_specific<double> tbb_temp_energies(0);
    std::vector<std::pair<int,double>> tbb_dist_info(num_nodes);
    // // #pragma omp parallel for
	// // for(int i=0; i<num_nodes; ++i)
	// // {
    // //     if(is_surface_vertex[i] == 0) continue;
	// // 	AScalar fx = 0;
	// // 	AScalar gx = 0;

	// // 	Vector3a xi = deformed.segment<3>(3*i);

	// // 	for(int k=0; k<current_indices[i].size(); ++k)
	// // 	{
    // //         Vector3a ck = deformed.segment<3>(current_indices[i][k]*3);
    // //         //f(i == 1567) std::cout<<k<<" k out of "<<current_indices[i].size()<<std::endl;
    // //         int valence = nodes_adjacency[current_indices[i][k]].size();
    // //         if(valence > 11) std::cout<<"WTF!!!!"<<std::endl;
    // //         VectorXa vs;

    // //         if(valence <= 6)
    // //         {
    // //             vs.resize(18);
    // //             for(int a=0; a<6; ++a)
    // //             {
    // //                 vs.segment<3>(a*3) = ck;
    // //             }
    // //         }else
    // //         {
    // //             vs.resize(33);
    // //             for(int a=0; a<11; ++a)
    // //             {
    // //                 vs.segment<3>(a*3) = ck;
    // //             }
    // //         }

    // //         //if(nodes_adjacency[current_indices[i][k]].size() == 0) std::cout<<"Zero Neighbors\n"<<std::endl;

    // //         for(int a=0; a<nodes_adjacency[current_indices[i][k]].size(); ++a)
    // //         {
    // //             vs.segment<3>(a*3) = deformed.segment<3>(nodes_adjacency[current_indices[i][k]][a]*3);
    // //         }
            
    // //         if((ck-xi).norm()<=radius) {
    // //             if(valence <= 6)
    // //             {
    // //                 fx += fx_func(xi, vs, ck, radius);
    // //                 gx += gx_func(xi, ck, radius);
    // //             }else
    // //             {
    // //                 fx += fx_func11(xi, vs, ck, radius);
    // //                 gx += gx_func11(xi, ck, radius);
    // //             }
    // //         }
	// // 	}

	// // 	AScalar dist = fx/gx;

        

	// // 	if(abs(gx)>1e-6)
    // //     {
    // //         //std::cout<<i<<" "<<fx<<" "<<gx<<" "<<dist<<std::endl;
    // //         if(BARRIER_ENERGY)
    // //         {
    // //             if(dist > 0)
    // //             {
    // //                 if(dist <= barrier_distance)
    // //                     temp_energies[i] = -IMLS_param*pow((dist-barrier_distance),2)*log(dist/barrier_distance);
    // //             }
    // //             else
    // //                 std::cout<<"ERROR!!! Negative Distance!"<<std::endl;
    // //         }
    // //         else if(!BARRIER_ENERGY && dist<0)
    // //             temp_energies[i] = -IMLS_param*pow((fx/gx),3);
    // //         dist_info[i] = (std::pair<int,double>(i,dist));
    // //         // close_slave_nodes.push_back(it->second);
    // //     }
	// // }

    tbb::parallel_for(
    tbb::blocked_range<size_t>(size_t(0), num_nodes),
    [&](const tbb::blocked_range<size_t>& r) {
        auto& local_energy_temp = tbb_temp_energies.local();
        for (size_t i = r.begin(); i < r.end(); i++) {
            auto& local_current_index = tbb_current_indices[i];
            auto& local_dist_info = tbb_dist_info[i];
            if(is_surface_vertex[i] != 0)
            {
                AScalar fx = 0;
                AScalar gx = 0;

                Vector3a xi = deformed.segment<3>(3*i);

                for(int k=0; k<local_current_index.size(); ++k)
                {
                    Vector3a ck = deformed.segment<3>(local_current_index[k]*3);
                    //f(i == 1567) std::cout<<k<<" k out of "<<current_indices[i].size()<<std::endl;
                    int valence = nodes_adjacency[local_current_index[k]].size();
                    if(valence > 11) std::cout<<"WTF!!!!"<<std::endl;
                    VectorXa vs;

                    if(valence <= 6)
                    {
                        vs.resize(18);
                        for(int a=0; a<6; ++a)
                        {
                            vs.segment<3>(a*3) = ck;
                        }
                    }else
                    {
                        vs.resize(33);
                        for(int a=0; a<11; ++a)
                        {
                            vs.segment<3>(a*3) = ck;
                        }
                    }

                    //if(nodes_adjacency[current_indices[i][k]].size() == 0) std::cout<<"Zero Neighbors\n"<<std::endl;

                    for(int a=0; a<nodes_adjacency[local_current_index[k]].size(); ++a)
                    {
                        vs.segment<3>(a*3) = deformed.segment<3>(nodes_adjacency[local_current_index[k]][a]*3);
                    }
                    
                    if((ck-xi).norm()<=radius) {
                        if(valence <= 6)
                        {
                            fx += fx_func(xi, vs, ck, radius);
                            gx += gx_func(xi, ck, radius);
                        }else
                        {
                            fx += fx_func11(xi, vs, ck, radius);
                            gx += gx_func11(xi, ck, radius);
                        }
                    }
                }

                //AScalar dist = pow(fx/gx,2); 
                AScalar dist = fabs(fx/gx);     
                //if(i == 1291) std::cout<<fx<<" "<<gx<<" "<<fx/gx<<std::endl; 

                if(abs(gx)>1e-6)
                {
                   
                    // if(i == 2434)
                    // {
                    //     std::cout<<i<<" "<<fx<<" "<<gx<<" "<<dist<<" "<<xi<<std::endl;
                    //     for(int k=0; k<local_current_index.size(); ++k)
                    //     {
                    //         Vector3a ck = deformed.segment<3>(local_current_index[k]*3);
                    //         std::cout<<gx_func11(xi, ck, radius)<<" "<<local_current_index[k]<<" ";
                    //     }
                            
                    //     std::cout<<std::endl;
                    // }    
                    if(BARRIER_ENERGY)
                    {
                        // if(dist > 0)
                        // {
                        //     if(dist <= barrier_distance)
                        //         local_energy_temp += -IMLS_param*pow((dist-barrier_distance),2)*log(dist/barrier_distance);
                        // }
                        // else if(!test)
                        // {
                        //     std::cerr<<"ERROR!!! Negative Distance! "<<i<<" "<< dist<<std::endl;
                        //     throw std::invalid_argument("Negative Distance");
                        // }
                        // std::cout<<dist<<std::endl;
                        if(dist <= barrier_distance)
                            local_energy_temp += -IMLS_param*pow((dist-barrier_distance),2)*log(dist/barrier_distance);
                    }
                    else if(!BARRIER_ENERGY && dist<0)
                        local_energy_temp += -IMLS_param*pow((fx/gx),3);
                    local_dist_info = (std::pair<int,double>(i,fx/gx));
                    // close_slave_nodes.push_back(it->second);
                }
            }
        }   
    });

    for(auto& local_energy_temp: tbb_temp_energies)
    {
        energy += local_energy_temp;
    }
    for(auto& local_dist_info: tbb_dist_info)
    {
        if(local_dist_info.first > 0)
            dist_info.push_back(local_dist_info);
    }
}

template <int dim>
void FEMSolver<dim>::addFastIMLSForceEntries(VectorXT& residual)
{
    BuildAcceleratorTree();

    VectorXa gradient(num_nodes*3);
	gradient.setZero();
    close_slave_nodes.clear();

	current_indices.clear();
    auto it = slave_nodes_3d[0].begin();
	for(int i=0; i<slave_nodes_3d[0].size(); ++i)
	{
		current_indices.push_back(std::vector<int>());

		Vector3a xi = deformed.segment<3>(3*it->first);

		Fuzzy_sphere fs(Point_d(xi[0], xi[1], xi[2]), radius+radius*0.1, radius*0.1);

		std::vector<std::tuple<Point_d, int>> points_query;
		accelerator.search(std::back_inserter(points_query), fs);

		for(int k=0; k<points_query.size(); ++k)
			current_indices.back().push_back(std::get<1>(points_query[k]));

        it++;
	}

	it = slave_nodes_3d[0].begin();
	for(int i=0; i<slave_nodes_3d[0].size(); ++i)
	{
		VectorXa sum_dfdx(dim*num_nodes);
        sum_dfdx.setZero();
        VectorXa sum_dgdx(dim*num_nodes);
        sum_dgdx.setZero();

        //std::cout<<i<<" out of "<<slave_nodes_3d[0].size()<<std::endl;
		AScalar fx = 0;
		AScalar gx = 0;
        Vector3a xi = deformed.segment<3>(3*it->first);

        VectorXa ele_dfdx; 
    	VectorXa ele_dgdx;
        //std::cout<<i<<" out of "<<slave_nodes_3d[0].size()<<std::endl;

		for(int k=0; k<current_indices[i].size(); ++k)
		{
            Vector3a ck = deformed.segment<3>(current_indices[i][k]*3);
            //f(i == 1567) std::cout<<k<<" k out of "<<current_indices[i].size()<<std::endl;
            int valence = master_nodes_adjacency[current_indices[i][k]].size();
            Eigen::VectorXi valence_indices;
            VectorXa vs;

            if(valence <= 6)
            {
                ele_dfdx.resize(24);
                ele_dgdx.resize(24);
                vs.resize(18);
                valence_indices.resize(8);
                for(int a=0; a<6; ++a)
                {
                    vs.segment<3>(a*3) = ck;
                    valence_indices(a+2)=(current_indices[i][k]);
                }
            }else
            {
                ele_dfdx.resize(39);
                ele_dgdx.resize(39);
                vs.resize(33);
                valence_indices.resize(13);
                for(int a=0; a<11; ++a)
                {
                    vs.segment<3>(a*3) = ck;
                    valence_indices(a+2)=(current_indices[i][k]);
                }
            }
           

            valence_indices(0)=(it->first);
            valence_indices(1)=(current_indices[i][k]);

           
            for(int a=0; a<master_nodes_adjacency[current_indices[i][k]].size(); ++a)
            {
                vs.segment<3>(a*3) = deformed.segment<3>(master_nodes_adjacency[current_indices[i][k]][a]*3);
                valence_indices(a+2) = master_nodes_adjacency[current_indices[i][k]][a];
            }
            
            if((ck-xi).norm()<=radius) {
                if(valence <= 6)
                {
                    ele_dfdx = dfdx_func(xi, vs, ck, radius);
                    ele_dgdx = dgdx_func(xi, ck, radius);
                    fx += fx_func(xi, vs, ck, radius);
                    gx += gx_func(xi, ck, radius);
                }else
                {
                    ele_dfdx = dfdx_func11(xi, vs, ck, radius);
                    ele_dgdx = dgdx_func11(xi, ck, radius);
                    fx += fx_func11(xi, vs, ck, radius);
                    gx += gx_func11(xi, ck, radius);
                }
                IMLS_local_gradient_to_global_gradient(ele_dfdx,valence_indices,dim,sum_dfdx);
                IMLS_local_gradient_to_global_gradient(ele_dgdx,valence_indices,dim,sum_dgdx);
            }
		}
        AScalar dist = fx/gx;

        if(abs(gx) > 1e-6) close_slave_nodes.push_back(it->second);
        if(abs(gx) > 1e-6)
        {
            gradient.setZero();
            // for(int k=0; k<current_indices[i].size(); ++k)
		    // {
            //     // for(int a=0; a<8; ++a)
            //     //     std::cout<<valence_indices[k][a]<<" ";
            //     // std::cout<<"\n";
            //    // std::vector<int> temp = valence_indices[k];
            //     Vector1a A,B;
            //     A(0) = fx; B(0) = gx;
            //     Vector24a local_grad = -IMLS_param*glue_gradient(A,B,ele_dfdx[k],ele_dgdx[k]);
            //     //std::cout<<((gx * ele_dfdx[k] - fx * ele_dgdx[k]) / (gx * gx)-glue_gradient(A,B,ele_dfdx[k],ele_dgdx[k])).norm()<<std::endl;
            //     // IMLS_local_gradient_to_global_gradient(
            //     // local_grad,temp, dim, gradient);
            //     for (int a = 0; a < 8; a++) 
            //     {   
            //         //std::cout<<valence_indices[k][a]<<" ";
            //         gradient.segment(dim * valence_indices(k,a), dim) += local_grad.segment(dim * a, dim);
            //     }
            //     //std::cout<<"\n";
                
            // }
            if(BARRIER_ENERGY)
            {
                if(dist > 0)
                {
                    if(dist<=barrier_distance)
                        gradient = IMLS_param*((barrier_distance-dist)*(2*log(dist/barrier_distance)-barrier_distance/dist+1))*(gx*sum_dfdx-fx*sum_dgdx)/(pow(gx,2));
                }          
                else
                    std::cout<<"ERROR!!! Negative Distance!"<<std::endl;
            }
            else if(!BARRIER_ENERGY && dist<0)
                gradient = -3*IMLS_param*pow(fx/gx,2)*(gx*sum_dfdx-fx*sum_dgdx)/(pow(gx,2));
            
            residual.segment(0, num_nodes * dim) += -gradient.segment(0, num_nodes * dim);
		}
        it++;
	}
    //residual.segment(0, num_nodes * dim) += -gradient.segment(0, num_nodes * dim);

    if(!IMLS_BOTH) return;

    BuildAcceleratorTree(false);
	gradient.setZero();

	current_indices.clear();
    it = master_nodes_3d[0].begin();
	for(int i=0; i<master_nodes_3d[0].size(); ++i)
	{
		current_indices.push_back(std::vector<int>());

		Vector3a xi = deformed.segment<3>(3*it->first);

		Fuzzy_sphere fs(Point_d(xi[0], xi[1], xi[2]), radius+radius*0.1, radius*0.1);

		std::vector<std::tuple<Point_d, int>> points_query;
		accelerator.search(std::back_inserter(points_query), fs);

		for(int k=0; k<points_query.size(); ++k)
			current_indices.back().push_back(std::get<1>(points_query[k]));

        it++;
	}

	it = master_nodes_3d[0].begin();
	for(int i=0; i<master_nodes_3d[0].size(); ++i)
	{
		VectorXa sum_dfdx(dim*num_nodes);
        sum_dfdx.setZero();
        VectorXa sum_dgdx(dim*num_nodes);
        sum_dgdx.setZero();

        //std::cout<<i<<" out of "<<slave_nodes_3d[0].size()<<std::endl;
		AScalar fx = 0;
		AScalar gx = 0;
        Vector3a xi = deformed.segment<3>(3*it->first);

        VectorXa ele_dfdx; 
    	VectorXa ele_dgdx;
        //std::cout<<i<<" out of "<<slave_nodes_3d[0].size()<<std::endl;

		for(int k=0; k<current_indices[i].size(); ++k)
		{
            Vector3a ck = deformed.segment<3>(current_indices[i][k]*3);
            //f(i == 1567) std::cout<<k<<" k out of "<<current_indices[i].size()<<std::endl;
            int valence = slave_nodes_adjacency[current_indices[i][k]-slave_nodes_shift].size();
            Eigen::VectorXi valence_indices;
            VectorXa vs;

            if(valence <= 6)
            {
                ele_dfdx.resize(24);
                ele_dgdx.resize(24);
                vs.resize(18);
                valence_indices.resize(8);
                for(int a=0; a<6; ++a)
                {
                    vs.segment<3>(a*3) = ck;
                    valence_indices(a+2)=(current_indices[i][k]);
                }
            }else
            {
                ele_dfdx.resize(39);
                ele_dgdx.resize(39);
                vs.resize(33);
                valence_indices.resize(13);
                for(int a=0; a<11; ++a)
                {
                    vs.segment<3>(a*3) = ck;
                    valence_indices(a+2)=(current_indices[i][k]);
                }
            }
           

            valence_indices(0)=(it->first);
            valence_indices(1)=(current_indices[i][k]);

           
            for(int a=0; a<slave_nodes_adjacency[current_indices[i][k]-slave_nodes_shift].size(); ++a)
            {
                vs.segment<3>(a*3) = deformed.segment<3>(slave_nodes_adjacency[current_indices[i][k]-slave_nodes_shift][a]*3);
                valence_indices(a+2) = slave_nodes_adjacency[current_indices[i][k]-slave_nodes_shift][a];
            }
            
            if((ck-xi).norm()<=radius) {
                if(valence <= 6)
                {
                    ele_dfdx = dfdx_func(xi, vs, ck, radius);
                    ele_dgdx = dgdx_func(xi, ck, radius);
                    fx += fx_func(xi, vs, ck, radius);
                    gx += gx_func(xi, ck, radius);
                }else
                {
                    ele_dfdx = dfdx_func11(xi, vs, ck, radius);
                    ele_dgdx = dgdx_func11(xi, ck, radius);
                    fx += fx_func11(xi, vs, ck, radius);
                    gx += gx_func11(xi, ck, radius);
                }
                IMLS_local_gradient_to_global_gradient(ele_dfdx,valence_indices,dim,sum_dfdx);
                IMLS_local_gradient_to_global_gradient(ele_dgdx,valence_indices,dim,sum_dgdx);
            }
		}
        AScalar dist = pow(fx/gx,2);

        if(abs(gx)>1e-6)
        {
            gradient.setZero();
            // for(int k=0; k<current_indices[i].size(); ++k)
		    // {
            //     // for(int a=0; a<8; ++a)
            //     //     std::cout<<valence_indices[k][a]<<" ";
            //     // std::cout<<"\n";
            //    // std::vector<int> temp = valence_indices[k];
            //     Vector1a A,B;
            //     A(0) = fx; B(0) = gx;
            //     Vector24a local_grad = -IMLS_param*glue_gradient(A,B,ele_dfdx[k],ele_dgdx[k]);
            //     //std::cout<<((gx * ele_dfdx[k] - fx * ele_dgdx[k]) / (gx * gx)-glue_gradient(A,B,ele_dfdx[k],ele_dgdx[k])).norm()<<std::endl;
            //     // IMLS_local_gradient_to_global_gradient(
            //     // local_grad,temp, dim, gradient);
            //     for (int a = 0; a < 8; a++) 
            //     {   
            //         //std::cout<<valence_indices[k][a]<<" ";
            //         gradient.segment(dim * valence_indices(k,a), dim) += local_grad.segment(dim * a, dim);
            //     }
            //     //std::cout<<"\n";
                
            // }
            if(BARRIER_ENERGY)
            {
                if(dist > 0)
                {
                    if(dist<=barrier_distance)
                        gradient = IMLS_param*((barrier_distance-dist)*(2*log(dist/barrier_distance)-barrier_distance/dist+1))*(gx*sum_dfdx-fx*sum_dgdx)/(pow(gx,2));
                }          
                else
                    std::cerr<<"ERROR!!! Negative Distance!"<<std::endl;
                 
            }
            else if(!BARRIER_ENERGY && dist<0)
                gradient = -3*IMLS_param*pow(fx/gx,2)*(gx*sum_dfdx-fx*sum_dgdx)/(pow(gx,2));
            residual.segment(0, num_nodes * dim) += -gradient.segment(0, num_nodes * dim);
		}
        it++;
	}
}

template <int dim>
void FEMSolver<dim>::addFastIMLSSCForceEntries(VectorXT& residual)
{
    BuildAcceleratorTreeSC();
	current_indices.resize(num_nodes);
    dist_info.clear();

    //#pragma omp parallel for
	// for(int i=0; i<num_nodes; ++i)
	// {
    //     //current_indices.push_back(std::vector<int>());
    //     if(is_surface_vertex[i] == 0) continue;
		

	// 	Vector3a xi = deformed.segment<3>(3*i);

	// 	Fuzzy_sphere fs(Point_d(xi[0], xi[1], xi[2]), radius+radius*0.2, radius*0.2);

	// 	std::vector<std::tuple<Point_d, int>> points_query;
	// 	accelerator.search(std::back_inserter(points_query), fs);

	// 	for(int k=0; k<points_query.size(); ++k)
    //     {
    //         int index = std::get<1>(points_query[k]);
    //         if((geodist_close_matrix.coeff(i,index) == 0))
    //             current_indices[i].push_back(index);
    //     }
	// }

    std::vector<std::vector<int>> tbb_current_indices(num_nodes);
    tbb::parallel_for(
    tbb::blocked_range<size_t>(size_t(0), num_nodes),
    [&](const tbb::blocked_range<size_t>& r) {
        for (size_t i = r.begin(); i < r.end(); i++) {
            auto& local_current_index = tbb_current_indices[i];
            local_current_index.clear();
            //std::cout<<i<<std::endl;
            if(is_surface_vertex[i] != 0)
            {
                Vector3a xi = deformed.segment<3>(3*i);

                Fuzzy_sphere fs(Point_d(xi[0], xi[1], xi[2]), radius+radius*0.2, radius*0.2);

                std::vector<std::tuple<Point_d, int>> points_query;
                accelerator.search(std::back_inserter(points_query), fs);

                for(int k=0; k<points_query.size(); ++k)
                {
                    int index = std::get<1>(points_query[k]);
                    if((geodist_close_matrix.coeff(i,index) == 0))
                        local_current_index.push_back(index);
                }
            }
        }   
    });

    std::vector<std::vector<std::pair<int,double>>> vector_temps(num_nodes);
    tbb::enumerable_thread_specific<std::vector<std::pair<int,double>>> tbb_vector_temps;
    //#pragma omp parallel for
	// for(int i=0; i<num_nodes; ++i)
	// {
    //     if(is_surface_vertex[i] == 0) continue;
    //     VectorXa gradient(num_nodes*3);
	//     gradient.setZero();
	// 	VectorXa sum_dfdx(dim*num_nodes);
    //     sum_dfdx.setZero();
    //     VectorXa sum_dgdx(dim*num_nodes);
    //     sum_dgdx.setZero();

    //     //std::cout<<i<<" out of "<<slave_nodes_3d[0].size()<<std::endl;
	// 	AScalar fx = 0;
	// 	AScalar gx = 0;
    //     Vector3a xi = deformed.segment<3>(3*i);

    //     VectorXa ele_dfdx; 
    // 	VectorXa ele_dgdx;
    //     //std::cout<<i<<" out of "<<slave_nodes_3d[0].size()<<std::endl;

	// 	for(int k=0; k<current_indices[i].size(); ++k)
	// 	{
    //         Vector3a ck = deformed.segment<3>(current_indices[i][k]*3);
    //         //f(i == 1567) std::cout<<k<<" k out of "<<current_indices[i].size()<<std::endl;
    //         int valence = nodes_adjacency[current_indices[i][k]].size();
    //         Eigen::VectorXi valence_indices;
    //         VectorXa vs;

    //         if(valence <= 6)
    //         {
    //             ele_dfdx.resize(24);
    //             ele_dgdx.resize(24);
    //             vs.resize(18);
    //             valence_indices.resize(8);
    //             for(int a=0; a<6; ++a)
    //             {
    //                 vs.segment<3>(a*3) = ck;
    //                 valence_indices(a+2)=(current_indices[i][k]);
    //             }
    //         }else
    //         {
    //             ele_dfdx.resize(39);
    //             ele_dgdx.resize(39);
    //             vs.resize(33);
    //             valence_indices.resize(13);
    //             for(int a=0; a<11; ++a)
    //             {
    //                 vs.segment<3>(a*3) = ck;
    //                 valence_indices(a+2)=(current_indices[i][k]);
    //             }
    //         }
           

    //         valence_indices(0)=(i);
    //         valence_indices(1)=(current_indices[i][k]);

           
    //         for(int a=0; a<nodes_adjacency[current_indices[i][k]].size(); ++a)
    //         {
    //             vs.segment<3>(a*3) = deformed.segment<3>(nodes_adjacency[current_indices[i][k]][a]*3);
    //             valence_indices(a+2) = nodes_adjacency[current_indices[i][k]][a];
    //         }
            
    //         if((ck-xi).norm()<=radius) {
    //             if(valence <= 6)
    //             {
    //                 ele_dfdx = dfdx_func(xi, vs, ck, radius);
    //                 ele_dgdx = dgdx_func(xi, ck, radius);
    //                 fx += fx_func(xi, vs, ck, radius);
    //                 gx += gx_func(xi, ck, radius);
    //             }else
    //             {
    //                 ele_dfdx = dfdx_func11(xi, vs, ck, radius);
    //                 ele_dgdx = dgdx_func11(xi, ck, radius);
    //                 fx += fx_func11(xi, vs, ck, radius);
    //                 gx += gx_func11(xi, ck, radius);
    //             }
    //             IMLS_local_gradient_to_global_gradient(ele_dfdx,valence_indices,dim,sum_dfdx);
    //             IMLS_local_gradient_to_global_gradient(ele_dgdx,valence_indices,dim,sum_dgdx);
    //         }
	// 	}
    //     AScalar dist = fx/gx;

    //     //if(abs(gx) > 1e-6) close_slave_nodes.push_back(it->second);
    //     if(abs(gx) > 1e-6)
    //     {
    //         gradient.setZero();
    //         // for(int k=0; k<current_indices[i].size(); ++k)
	// 	    // {
    //         //     // for(int a=0; a<8; ++a)
    //         //     //     std::cout<<valence_indices[k][a]<<" ";
    //         //     // std::cout<<"\n";
    //         //    // std::vector<int> temp = valence_indices[k];
    //         //     Vector1a A,B;
    //         //     A(0) = fx; B(0) = gx;
    //         //     Vector24a local_grad = -IMLS_param*glue_gradient(A,B,ele_dfdx[k],ele_dgdx[k]);
    //         //     //std::cout<<((gx * ele_dfdx[k] - fx * ele_dgdx[k]) / (gx * gx)-glue_gradient(A,B,ele_dfdx[k],ele_dgdx[k])).norm()<<std::endl;
    //         //     // IMLS_local_gradient_to_global_gradient(
    //         //     // local_grad,temp, dim, gradient);
    //         //     for (int a = 0; a < 8; a++) 
    //         //     {   
    //         //         //std::cout<<valence_indices[k][a]<<" ";
    //         //         gradient.segment(dim * valence_indices(k,a), dim) += local_grad.segment(dim * a, dim);
    //         //     }
    //         //     //std::cout<<"\n";
                
    //         // }
    //         if(BARRIER_ENERGY)
    //         {
    //             if(dist > 0)
    //             {
    //                 if(dist<=barrier_distance)
    //                     gradient = IMLS_param*((barrier_distance-dist)*(2*log(dist/barrier_distance)-barrier_distance/dist+1))*(gx*sum_dfdx-fx*sum_dgdx)/(pow(gx,2));
    //             }          
    //             else
    //                 std::cout<<"ERROR!!! Negative Distance!"<<std::endl;
    //         }
    //         else if(!BARRIER_ENERGY && dist<0)
    //             gradient = -3*IMLS_param*pow(fx/gx,2)*(gx*sum_dfdx-fx*sum_dgdx)/(pow(gx,2));
            
    //         //residual.segment(0, num_nodes * dim) += -gradient.segment(0, num_nodes * dim);
    //         for(int j=0; j<num_nodes; ++j)
    //         {
    //             if(gradient(j) != 0) vector_temps[i].push_back(std::pair<int, double>(j,gradient(j)));
    //         }
	// 	}
	// }

    tbb::parallel_for(
    tbb::blocked_range<size_t>(size_t(0), num_nodes),
    [&](const tbb::blocked_range<size_t>& r) {
        auto& local_vector_temp = tbb_vector_temps.local();
        for (size_t i = r.begin(); i < r.end(); i++) {
            auto& local_current_index = tbb_current_indices[i];
            if(is_surface_vertex[i] != 0)
            {
                VectorXa gradient(num_nodes*dim);
                gradient.setZero();
                VectorXa sum_dfdx(dim*num_nodes);
                sum_dfdx.setZero();
                VectorXa sum_dgdx(dim*num_nodes);
                sum_dgdx.setZero();

                //std::cout<<i<<" out of "<<slave_nodes_3d[0].size()<<std::endl;
                AScalar fx = 0;
                AScalar gx = 0;
                Vector3a xi = deformed.segment<3>(3*i);

                VectorXa ele_dfdx; 
                VectorXa ele_dgdx;
                //std::cout<<i<<" out of "<<slave_nodes_3d[0].size()<<std::endl;

                for(int k=0; k<local_current_index.size(); ++k)
                {
                    Vector3a ck = deformed.segment<3>(local_current_index[k]*3);
                    //f(i == 1567) std::cout<<k<<" k out of "<<local_current_index.size()<<std::endl;
                    int valence = nodes_adjacency[local_current_index[k]].size();
                    Eigen::VectorXi valence_indices;
                    VectorXa vs;

                    if(valence <= 6)
                    {
                        ele_dfdx.resize(24);
                        ele_dgdx.resize(24);
                        vs.resize(18);
                        valence_indices.resize(8);
                        for(int a=0; a<6; ++a)
                        {
                            vs.segment<3>(a*3) = ck;
                            valence_indices(a+2)=(local_current_index[k]);
                        }
                    }else
                    {
                        ele_dfdx.resize(39);
                        ele_dgdx.resize(39);
                        vs.resize(33);
                        valence_indices.resize(13);
                        for(int a=0; a<11; ++a)
                        {
                            vs.segment<3>(a*3) = ck;
                            valence_indices(a+2)=(local_current_index[k]);
                        }
                    }
                

                    valence_indices(0)=(i);
                    valence_indices(1)=(local_current_index[k]);

                
                    for(int a=0; a<nodes_adjacency[local_current_index[k]].size(); ++a)
                    {
                        vs.segment<3>(a*3) = deformed.segment<3>(nodes_adjacency[local_current_index[k]][a]*3);
                        valence_indices(a+2) = nodes_adjacency[local_current_index[k]][a];
                    }
                    
                    if((ck-xi).norm()<=radius) {
                        if(valence <= 6)
                        {
                            ele_dfdx = dfdx_func(xi, vs, ck, radius);
                            ele_dgdx = dgdx_func(xi, ck, radius);
                            fx += fx_func(xi, vs, ck, radius);
                            gx += gx_func(xi, ck, radius);
                        }else
                        {
                            ele_dfdx = dfdx_func11(xi, vs, ck, radius);
                            ele_dgdx = dgdx_func11(xi, ck, radius);
                            fx += fx_func11(xi, vs, ck, radius);
                            gx += gx_func11(xi, ck, radius);
                        }
                        IMLS_local_gradient_to_global_gradient(ele_dfdx,valence_indices,dim,sum_dfdx);
                        IMLS_local_gradient_to_global_gradient(ele_dgdx,valence_indices,dim,sum_dgdx);
                    }
                }
                //AScalar dist = pow(fx/gx,2);
                AScalar dist = fabs(fx/gx);
                int sign = 1;
                if(fx/gx < 0) sign = -1;

                //if(abs(gx) > 1e-6) close_slave_nodes.push_back(it->second);
                if(abs(gx) > 1e-6)
                {
                    gradient.setZero();
                    // for(int k=0; k<current_indices[i].size(); ++k)
                    // {
                    //     // for(int a=0; a<8; ++a)
                    //     //     std::cout<<valence_indices[k][a]<<" ";
                    //     // std::cout<<"\n";
                    //    // std::vector<int> temp = valence_indices[k];
                    //     Vector1a A,B;
                    //     A(0) = fx; B(0) = gx;
                    //     Vector24a local_grad = -IMLS_param*glue_gradient(A,B,ele_dfdx[k],ele_dgdx[k]);
                    //     //std::cout<<((gx * ele_dfdx[k] - fx * ele_dgdx[k]) / (gx * gx)-glue_gradient(A,B,ele_dfdx[k],ele_dgdx[k])).norm()<<std::endl;
                    //     // IMLS_local_gradient_to_global_gradient(
                    //     // local_grad,temp, dim, gradient);
                    //     for (int a = 0; a < 8; a++) 
                    //     {   
                    //         //std::cout<<valence_indices[k][a]<<" ";
                    //         gradient.segment(dim * valence_indices(k,a), dim) += local_grad.segment(dim * a, dim);
                    //     }
                    //     //std::cout<<"\n";
                        
                    // }
                    if(BARRIER_ENERGY)
                    {
                        // if(dist > 0)
                        // {
                        //     if(dist<=barrier_distance)
                        //         gradient = IMLS_param*((barrier_distance-dist)*(2*log(dist/barrier_distance)-barrier_distance/dist+1))*(gx*sum_dfdx-fx*sum_dgdx)/(pow(gx,2));
                        // }          
                        // else
                        //     std::cout<<"ERROR!!! Negative Distance!"<<std::endl;
                        if(dist<=barrier_distance)
                            gradient = sign*IMLS_param*((barrier_distance-dist)*(2*log(dist/barrier_distance)-barrier_distance/dist+1))*(gx*sum_dfdx-fx*sum_dgdx)/(pow(gx,2));
                    }
                    else if(!BARRIER_ENERGY && dist<0)
                        gradient = -3*IMLS_param*pow(fx/gx,2)*(gx*sum_dfdx-fx*sum_dgdx)/(pow(gx,2));
                    
                    //residual.segment(0, num_nodes * dim) += -gradient.segment(0, num_nodes * dim);
                    for(int j=0; j<dim*num_nodes; ++j)
                    {
                        if(gradient(j) != 0) local_vector_temp.push_back(std::pair<int, double>(j,gradient(j)));
                    }
                }
            }
        }   
    });

    for (const auto& local_vector_temp : tbb_vector_temps)
    {
        for(int j=0; j<local_vector_temp.size(); ++j)
        {
            residual(local_vector_temp[j].first) -= local_vector_temp[j].second;
        }
    }
}

template <int dim>
void FEMSolver<dim>::addFastIMLSHessianEntries(std::vector<Entry>& entries,bool project_PD)
{
    BuildAcceleratorTree();
	current_indices.clear();
    auto it = slave_nodes_3d[0].begin();
    close_slave_nodes.clear();

	for(int i=0; i<slave_nodes_3d[0].size(); ++i)
	{
		current_indices.push_back(std::vector<int>());

		Vector3a xi = deformed.segment<3>(3*it->first);

		Fuzzy_sphere fs(Point_d(xi[0], xi[1], xi[2]), radius+radius*0.1, radius*0.1);

		std::vector<std::tuple<Point_d, int>> points_query;
		accelerator.search(std::back_inserter(points_query), fs);

		for(int k=0; k<points_query.size(); ++k)
			current_indices.back().push_back(std::get<1>(points_query[k]));

        it++;
	}

	it = slave_nodes_3d[0].begin();
	for(int i=0; i<slave_nodes_3d[0].size(); ++i)
	{
        VectorXa sum_dfdx(dim*num_nodes);
        sum_dfdx.setZero();
        VectorXa sum_dgdx(dim*num_nodes);
        sum_dgdx.setZero();
        std::vector<Entry> sum_d2fdx2;
        std::vector<Entry> sum_d2gdx2;

        //std::cout<<i<<" out of "<<slave_nodes_3d[0].size()<<std::endl;
		AScalar fx = 0;
		AScalar gx = 0;
        Vector3a xi = deformed.segment<3>(3*it->first);

        VectorXa ele_dfdx; 
    	VectorXa ele_dgdx;
        VectorXa ele_d2fdx; 
    	VectorXa ele_d2gdx;
        //std::cout<<i<<" out of "<<slave_nodes_3d[0].size()<<std::endl;

        Eigen::MatrixXd valence_indices;
        valence_indices.resize(current_indices[i].size(),8);

		for(int k=0; k<current_indices[i].size(); ++k)
		{
            Vector3a ck = deformed.segment<3>(current_indices[i][k]*3);
            //f(i == 1567) std::cout<<k<<" k out of "<<current_indices[i].size()<<std::endl;
            int valence = master_nodes_adjacency[current_indices[i][k]].size();
            Eigen::VectorXi valence_indices;
            VectorXa vs;

            if(valence <= 6)
            {
                ele_dfdx.resize(24);
                ele_dgdx.resize(24);
                ele_d2fdx.resize(24*24);
                ele_d2gdx.resize(24*24);
                vs.resize(18);
                valence_indices.resize(8);
                for(int a=0; a<6; ++a)
                {
                    vs.segment<3>(a*3) = ck;
                    valence_indices(a+2)=(current_indices[i][k]);
                }
            }else
            {
                ele_dfdx.resize(39);
                ele_dgdx.resize(39);
                ele_d2fdx.resize(39*39);
                ele_d2gdx.resize(39*39);
                vs.resize(33);
                valence_indices.resize(13);
                for(int a=0; a<11; ++a)
                {
                    vs.segment<3>(a*3) = ck;
                    valence_indices(a+2)=(current_indices[i][k]);
                }
            }
           

            valence_indices(0)=(it->first);
            valence_indices(1)=(current_indices[i][k]);

           
            for(int a=0; a<master_nodes_adjacency[current_indices[i][k]].size(); ++a)
            {
                vs.segment<3>(a*3) = deformed.segment<3>(master_nodes_adjacency[current_indices[i][k]][a]*3);
                valence_indices(a+2) = master_nodes_adjacency[current_indices[i][k]][a];
            }
            
            
            if((ck-xi).norm()<=radius) {

                if(valence <= 6)
                {
                    ele_d2fdx = d2fdx_func(xi, vs, ck, radius);
                    ele_d2gdx = d2gdx_func(xi, ck, radius);
                    ele_dfdx = dfdx_func(xi, vs, ck, radius);
                    ele_dgdx = dgdx_func(xi, ck, radius);
                    fx += fx_func(xi, vs, ck, radius);
                    gx += gx_func(xi, ck, radius);
                }else
                {
                    ele_d2fdx = d2fdx_func11(xi, vs, ck, radius);
                    ele_d2gdx = d2gdx_func11(xi, ck, radius);
                    ele_dfdx = dfdx_func11(xi, vs, ck, radius);
                    ele_dgdx = dgdx_func11(xi, ck, radius);
                    fx += fx_func11(xi, vs, ck, radius);
                    gx += gx_func11(xi, ck, radius);
                }
                

                IMLS_local_gradient_to_global_gradient(ele_dfdx,valence_indices,dim,sum_dfdx);
                IMLS_local_gradient_to_global_gradient(ele_dgdx,valence_indices,dim,sum_dgdx);
                IMLS_local_hessian_to_global_triplets(ele_d2fdx,valence_indices,dim,sum_d2fdx2);
                IMLS_local_hessian_to_global_triplets(ele_d2gdx,valence_indices,dim,sum_d2gdx2);
            }
		}
        
		AScalar dist = fx/gx;
        if(abs(gx) > 1e-6) close_slave_nodes.push_back(it->second);
        if(abs(gx)>1e-6)
        {
            if(BARRIER_ENERGY && dist <= 0)
            {
                it++;
                std::cout<<"NEGATIVE DISTANCE!!!"<<std::endl;
                continue;
            } 
            if(BARRIER_ENERGY && dist > barrier_distance)
            {
                it++;
                continue;
            } 
            if(!BARRIER_ENERGY && dist > 0)
            {
                it++;
                continue;
            }
             
            // for(int k=0; k<current_indices[i].size(); ++k)
		    // {
            //     Vector1a A,B;
            //     A(0) = fx; B(0) = gx;
            //     Matrix24a local_hess = -IMLS_param*glue_hessian(A,B,ele_dfdx[k],ele_dgdx[k],ele_d2fdx[k],ele_d2gdx[k]);
            //     // IMLS_local_hessian_to_global_triplets(
            //     // local_hess,valence_indices[k], dim, entries);

            //     for (int a = 0; a < 8; a++) {
            //         for (int b = 0; b < 8; b++) {
            //             for (int c = 0; c < dim; c++) {
            //                 for (int l = 0; l < dim; l++) {
            //                     entries.emplace_back(
            //                         dim * valence_indices(k,a)+ c, dim * valence_indices(k,b) + l,
            //                         local_hess(dim * a + c, dim * b + l));
            //                 }
            //             }
            //         }
            //     }
            //     //std::cout<<" \n";
            // }
            // Eigen::VectorXd gradient(dim*num_nodes); 
            // gradient.setZero();
            // for(int k=0; k<current_indices[i].size(); ++k)
		    // {
            //     // for(int a=0; a<8; ++a)
            //     //     std::cout<<valence_indices[k][a]<<" ";
            //     // std::cout<<"\n";
            //    // std::vector<int> temp = valence_indices[k];
            //     Vector1a A,B;
            //     A(0) = fx; B(0) = gx;
            //     Vector24a local_grad = glue_gradient(A,B,ele_dfdx[k],ele_dgdx[k]);
            //     //std::cout<<((gx * ele_dfdx[k] - fx * ele_dgdx[k]) / (gx * gx)-glue_gradient(A,B,ele_dfdx[k],ele_dgdx[k])).norm()<<std::endl;
            //     // IMLS_local_gradient_to_global_gradient(
            //     // local_grad,temp, dim, gradient);
            //     for (int a = 0; a < 8; a++) 
            //     {   
            //         //std::cout<<valence_indices[k][a]<<" ";
            //         gradient.segment(dim * valence_indices(k,a), dim) += local_grad.segment(dim * a, dim);
            //     }
            //     //std::cout<<"\n";
                
            // }

            // Compute d dist/ dx
            Eigen::VectorXd ddist_dx = (gx*sum_dfdx-fx*sum_dgdx)/(pow(gx,2));
            
            //std::cout<<"Norm: "<<(3*pow(fx/gx,2)*ddist_dx-gradient).norm()<<std::endl;

            // StiffnessMatrix Hessian(dim*num_nodes,dim*num_nodes);
            std::vector<Eigen::Triplet<double>> Hessian_t;

            //Part 1 6(f/g) (ddistdx)(ddistdx)^T
            if(!BARRIER_ENERGY)
                IMLS_vector_muliplication_to_triplets(ddist_dx,ddist_dx,-IMLS_param*6*(fx/gx),Hessian_t);
            else
                IMLS_vector_muliplication_to_triplets(ddist_dx,ddist_dx,IMLS_param*((barrier_distance/dist+2)*(barrier_distance/dist)-2*log(dist/barrier_distance)-3),Hessian_t);
            entries.insert(entries.end(), Hessian_t.begin(), Hessian_t.end());

            //Part 2 3(f/g)^2 d2distdx2
            // std::vector<Entry> contact_entries = entriesFromSparseMatrix(hess.block(0, 0, num_nodes * dim , num_nodes * dim));
            // entries.insert(entries.end(), contact_entries.begin(), contact_entries.end());
            // A.setFromTriplets(coefficients.begin(), coefficients.end());

            // h1 = d2f/dx2
            StiffnessMatrix h1(dim*num_nodes,dim*num_nodes);
            h1.setFromTriplets(sum_d2fdx2.begin(), sum_d2fdx2.end());

            // h2 = d2g/dx2
            StiffnessMatrix h2(dim*num_nodes,dim*num_nodes);
            h2.setFromTriplets(sum_d2gdx2.begin(), sum_d2gdx2.end());

            //h3 = dfdx*dgdx^T
            std::vector<Eigen::Triplet<double>> h3_t;
            IMLS_vector_muliplication_to_triplets(sum_dfdx,sum_dgdx,1.,h3_t);
            StiffnessMatrix h3(dim*num_nodes,dim*num_nodes);
            h3.setFromTriplets(h3_t.begin(), h3_t.end());

            //h4 = dgdx*dgdx^T
            std::vector<Eigen::Triplet<double>> h4_t;
            IMLS_vector_muliplication_to_triplets(sum_dgdx,sum_dgdx,1.,h4_t);
            StiffnessMatrix h4(dim*num_nodes,dim*num_nodes);
            h4.setFromTriplets(h4_t.begin(), h4_t.end());

            //h6 = dgdx*dfdx^T
            std::vector<Eigen::Triplet<double>> h6_t;
            IMLS_vector_muliplication_to_triplets(sum_dgdx,sum_dfdx,1.,h6_t);
            StiffnessMatrix h6(dim*num_nodes,dim*num_nodes);
            h6.setFromTriplets(h6_t.begin(), h6_t.end());

            //h5 = 3(f/g)^2 1/g h1 -2/g^2 h3+2f/g^3 h4 -f/g^2 h2
            StiffnessMatrix h5(dim*num_nodes,dim*num_nodes);
            if(!BARRIER_ENERGY)
                h5 = -IMLS_param*(1./gx*h1 - 1./pow(gx,2)*h3 -1./pow(gx,2)*h6 + 2*fx/pow(gx,3)*h4-fx/pow(gx,2)*h2)*3*pow(fx/gx,2);
            else
                h5 = IMLS_param*(1./gx*h1 - 1./pow(gx,2)*h3 -1./pow(gx,2)*h6 + 2*fx/pow(gx,3)*h4-fx/pow(gx,2)*h2)*((barrier_distance-dist)*(2*log(dist/barrier_distance)-barrier_distance/dist+1));
            std::vector<Entry> h5e = entriesFromSparseMatrix(h5.block(0, 0, num_nodes * dim , num_nodes * dim));
            entries.insert(entries.end(), h5e.begin(), h5e.end());
		}
        it++;
        
	}
    if(!IMLS_BOTH) return;

    BuildAcceleratorTree(false);
	current_indices.clear();
    it = master_nodes_3d[0].begin();

	for(int i=0; i<master_nodes_3d[0].size(); ++i)
	{
		current_indices.push_back(std::vector<int>());

		Vector3a xi = deformed.segment<3>(3*it->first);

		Fuzzy_sphere fs(Point_d(xi[0], xi[1], xi[2]), radius+radius*0.1, radius*0.1);

		std::vector<std::tuple<Point_d, int>> points_query;
		accelerator.search(std::back_inserter(points_query), fs);

		for(int k=0; k<points_query.size(); ++k)
			current_indices.back().push_back(std::get<1>(points_query[k]));

        it++;
	}

	it = master_nodes_3d[0].begin();
	for(int i=0; i<master_nodes_3d[0].size(); ++i)
	{
        VectorXa sum_dfdx(dim*num_nodes);
        sum_dfdx.setZero();
        VectorXa sum_dgdx(dim*num_nodes);
        sum_dgdx.setZero();
        std::vector<Entry> sum_d2fdx2;
        std::vector<Entry> sum_d2gdx2;

        //std::cout<<i<<" out of "<<slave_nodes_3d[0].size()<<std::endl;
		AScalar fx = 0;
		AScalar gx = 0;
        Vector3a xi = deformed.segment<3>(3*it->first);

        VectorXa ele_dfdx; 
    	VectorXa ele_dgdx;
        VectorXa ele_d2fdx; 
    	VectorXa ele_d2gdx;
        //std::cout<<i<<" out of "<<slave_nodes_3d[0].size()<<std::endl;

        Eigen::MatrixXd valence_indices;
        valence_indices.resize(current_indices[i].size(),8);

		for(int k=0; k<current_indices[i].size(); ++k)
		{
            Vector3a ck = deformed.segment<3>(current_indices[i][k]*3);
            //f(i == 1567) std::cout<<k<<" k out of "<<current_indices[i].size()<<std::endl;
            int valence = slave_nodes_adjacency[current_indices[i][k]-slave_nodes_shift].size();
            Eigen::VectorXi valence_indices;
            VectorXa vs;

            if(valence <= 6)
            {
                ele_dfdx.resize(24);
                ele_dgdx.resize(24);
                ele_d2fdx.resize(24*24);
                ele_d2gdx.resize(24*24);
                vs.resize(18);
                valence_indices.resize(8);
                for(int a=0; a<6; ++a)
                {
                    vs.segment<3>(a*3) = ck;
                    valence_indices(a+2)=(current_indices[i][k]);
                }
            }else
            {
                ele_dfdx.resize(39);
                ele_dgdx.resize(39);
                ele_d2fdx.resize(39*39);
                ele_d2gdx.resize(39*39);
                vs.resize(33);
                valence_indices.resize(13);
                for(int a=0; a<11; ++a)
                {
                    vs.segment<3>(a*3) = ck;
                    valence_indices(a+2)=(current_indices[i][k]);
                }
            }
           

            valence_indices(0)=(it->first);
            valence_indices(1)=(current_indices[i][k]);

           
            for(int a=0; a<slave_nodes_adjacency[current_indices[i][k]-slave_nodes_shift].size(); ++a)
            {
                vs.segment<3>(a*3) = deformed.segment<3>(slave_nodes_adjacency[current_indices[i][k]-slave_nodes_shift][a]*3);
                valence_indices(a+2) = slave_nodes_adjacency[current_indices[i][k]-slave_nodes_shift][a];
            }
            
            
            if((ck-xi).norm()<=radius) {

                if(valence <= 6)
                {
                    ele_d2fdx = d2fdx_func(xi, vs, ck, radius);
                    ele_d2gdx = d2gdx_func(xi, ck, radius);
                    ele_dfdx = dfdx_func(xi, vs, ck, radius);
                    ele_dgdx = dgdx_func(xi, ck, radius);
                    fx += fx_func(xi, vs, ck, radius);
                    gx += gx_func(xi, ck, radius);
                }else
                {
                    ele_d2fdx = d2fdx_func11(xi, vs, ck, radius);
                    ele_d2gdx = d2gdx_func11(xi, ck, radius);
                    ele_dfdx = dfdx_func11(xi, vs, ck, radius);
                    ele_dgdx = dgdx_func11(xi, ck, radius);
                    fx += fx_func11(xi, vs, ck, radius);
                    gx += gx_func11(xi, ck, radius);
                }
                

                IMLS_local_gradient_to_global_gradient(ele_dfdx,valence_indices,dim,sum_dfdx);
                IMLS_local_gradient_to_global_gradient(ele_dgdx,valence_indices,dim,sum_dgdx);
                IMLS_local_hessian_to_global_triplets(ele_d2fdx,valence_indices,dim,sum_d2fdx2);
                IMLS_local_hessian_to_global_triplets(ele_d2gdx,valence_indices,dim,sum_d2gdx2);
            }
		}
        
		AScalar dist = fx/gx;
        if(abs(gx)>1e-6)
        {
            if(BARRIER_ENERGY && dist < 0)
            {
                std::cout<<"NEGATIVE DISTANCE!!!"<<std::endl;
                it++;
                continue;
            } 
            if(BARRIER_ENERGY && dist > barrier_distance)
            {
                it++;
                continue;
            } 
            if(!BARRIER_ENERGY && dist > 0)
            {
                it++;
                continue;
            } 
            // for(int k=0; k<current_indices[i].size(); ++k)
		    // {
            //     Vector1a A,B;
            //     A(0) = fx; B(0) = gx;
            //     Matrix24a local_hess = -IMLS_param*glue_hessian(A,B,ele_dfdx[k],ele_dgdx[k],ele_d2fdx[k],ele_d2gdx[k]);
            //     // IMLS_local_hessian_to_global_triplets(
            //     // local_hess,valence_indices[k], dim, entries);

            //     for (int a = 0; a < 8; a++) {
            //         for (int b = 0; b < 8; b++) {
            //             for (int c = 0; c < dim; c++) {
            //                 for (int l = 0; l < dim; l++) {
            //                     entries.emplace_back(
            //                         dim * valence_indices(k,a)+ c, dim * valence_indices(k,b) + l,
            //                         local_hess(dim * a + c, dim * b + l));
            //                 }
            //             }
            //         }
            //     }
            //     //std::cout<<" \n";
            // }
            // Eigen::VectorXd gradient(dim*num_nodes); 
            // gradient.setZero();
            // for(int k=0; k<current_indices[i].size(); ++k)
		    // {
            //     // for(int a=0; a<8; ++a)
            //     //     std::cout<<valence_indices[k][a]<<" ";
            //     // std::cout<<"\n";
            //    // std::vector<int> temp = valence_indices[k];
            //     Vector1a A,B;
            //     A(0) = fx; B(0) = gx;
            //     Vector24a local_grad = glue_gradient(A,B,ele_dfdx[k],ele_dgdx[k]);
            //     //std::cout<<((gx * ele_dfdx[k] - fx * ele_dgdx[k]) / (gx * gx)-glue_gradient(A,B,ele_dfdx[k],ele_dgdx[k])).norm()<<std::endl;
            //     // IMLS_local_gradient_to_global_gradient(
            //     // local_grad,temp, dim, gradient);
            //     for (int a = 0; a < 8; a++) 
            //     {   
            //         //std::cout<<valence_indices[k][a]<<" ";
            //         gradient.segment(dim * valence_indices(k,a), dim) += local_grad.segment(dim * a, dim);
            //     }
            //     //std::cout<<"\n";
                
            // }

            // Compute d dist/ dx
            Eigen::VectorXd ddist_dx = (gx*sum_dfdx-fx*sum_dgdx)/(pow(gx,2));
            //std::cout<<"Norm: "<<(3*pow(fx/gx,2)*ddist_dx-gradient).norm()<<std::endl;

            // StiffnessMatrix Hessian(dim*num_nodes,dim*num_nodes);
            std::vector<Eigen::Triplet<double>> Hessian_t;

            //Part 1 6(f/g) (ddistdx)(ddistdx)^T
            if(!BARRIER_ENERGY)
                IMLS_vector_muliplication_to_triplets(ddist_dx,ddist_dx,-IMLS_param*6*(fx/gx),Hessian_t);
            else
                IMLS_vector_muliplication_to_triplets(ddist_dx,ddist_dx,IMLS_param*((barrier_distance/dist+2)*(barrier_distance/dist)-2*log(dist/barrier_distance)-3),Hessian_t);
            entries.insert(entries.end(), Hessian_t.begin(), Hessian_t.end());

            //Part 2 3(f/g)^2 d2distdx2
            // std::vector<Entry> contact_entries = entriesFromSparseMatrix(hess.block(0, 0, num_nodes * dim , num_nodes * dim));
            // entries.insert(entries.end(), contact_entries.begin(), contact_entries.end());
            // A.setFromTriplets(coefficients.begin(), coefficients.end());

            // h1 = d2f/dx2
            StiffnessMatrix h1(dim*num_nodes,dim*num_nodes);
            h1.setFromTriplets(sum_d2fdx2.begin(), sum_d2fdx2.end());

            // h2 = d2g/dx2
            StiffnessMatrix h2(dim*num_nodes,dim*num_nodes);
            h2.setFromTriplets(sum_d2gdx2.begin(), sum_d2gdx2.end());

            //h3 = dfdx*dgdx^T
            std::vector<Eigen::Triplet<double>> h3_t;
            IMLS_vector_muliplication_to_triplets(sum_dfdx,sum_dgdx,1.,h3_t);
            StiffnessMatrix h3(dim*num_nodes,dim*num_nodes);
            h3.setFromTriplets(h3_t.begin(), h3_t.end());

            //h4 = dgdx*dgdx^T
            std::vector<Eigen::Triplet<double>> h4_t;
            IMLS_vector_muliplication_to_triplets(sum_dgdx,sum_dgdx,1.,h4_t);
            StiffnessMatrix h4(dim*num_nodes,dim*num_nodes);
            h4.setFromTriplets(h4_t.begin(), h4_t.end());

            //h6 = dgdx*dfdx^T
            std::vector<Eigen::Triplet<double>> h6_t;
            IMLS_vector_muliplication_to_triplets(sum_dgdx,sum_dfdx,1.,h6_t);
            StiffnessMatrix h6(dim*num_nodes,dim*num_nodes);
            h6.setFromTriplets(h6_t.begin(), h6_t.end());

            //h5 = 3(f/g)^2 1/g h1 -2/g^2 h3+2f/g^3 h4 -f/g^2 h2
            StiffnessMatrix h5(dim*num_nodes,dim*num_nodes);
            if(!BARRIER_ENERGY)
                h5 = -IMLS_param*(1./gx*h1 - 1./pow(gx,2)*h3 -1./pow(gx,2)*h6 + 2*fx/pow(gx,3)*h4-fx/pow(gx,2)*h2)*3*pow(fx/gx,2);
            else
                h5 = IMLS_param*(1./gx*h1 - 1./pow(gx,2)*h3 -1./pow(gx,2)*h6 + 2*fx/pow(gx,3)*h4-fx/pow(gx,2)*h2)*((barrier_distance-dist)*(2*log(dist/barrier_distance)-barrier_distance/dist+1));
            std::vector<Entry> h5e = entriesFromSparseMatrix(h5.block(0, 0, num_nodes * dim , num_nodes * dim));
            entries.insert(entries.end(), h5e.begin(), h5e.end());
		}
        it++;
	}
}

template <int dim>
void FEMSolver<dim>::addFastIMLSSCHessianEntries(std::vector<Entry>& entries,bool project_PD)
{
    BuildAcceleratorTreeSC();
	current_indices.resize(num_nodes);
    close_slave_nodes.clear();

    std::vector<std::vector<int>> tbb_current_indices(num_nodes);
    tbb::parallel_for(
    tbb::blocked_range<size_t>(size_t(0), num_nodes),
    [&](const tbb::blocked_range<size_t>& r) {
        for (size_t i = r.begin(); i < r.end(); i++) {
            auto& local_current_index = tbb_current_indices[i];
            local_current_index.clear();
            //std::cout<<i<<std::endl;
            if(is_surface_vertex[i] != 0)
            {
                Vector3a xi = deformed.segment<3>(3*i);

                Fuzzy_sphere fs(Point_d(xi[0], xi[1], xi[2]), radius+radius*0.2, radius*0.2);

                std::vector<std::tuple<Point_d, int>> points_query;
                accelerator.search(std::back_inserter(points_query), fs);

                for(int k=0; k<points_query.size(); ++k)
                {
                    int index = std::get<1>(points_query[k]);
                    if((geodist_close_matrix.coeff(i,index) == 0))
                        local_current_index.push_back(index);
                }
            }
        }   
    });

	// for(int i=0; i<num_nodes; ++i)
	// {
    //     //current_indices.push_back(std::vector<int>());
    //     if(is_surface_vertex[i] == 0) continue;

	// 	Vector3a xi = deformed.segment<3>(3*i);

	// 	Fuzzy_sphere fs(Point_d(xi[0], xi[1], xi[2]), radius+radius*0.2, radius*0.2);

	// 	std::vector<std::tuple<Point_d, int>> points_query;
	// 	accelerator.search(std::back_inserter(points_query), fs);

	// 	for(int k=0; k<points_query.size(); ++k)
    //     {
    //         int index = std::get<1>(points_query[k]);
    //         if((geodist_close_matrix.coeff(i,index) == 0))
    //             current_indices[i].push_back(index);
    //     }
	// }

    std::vector<std::vector<Entry>> temp_triplets(num_nodes);
    tbb::enumerable_thread_specific<std::vector<Eigen::Triplet<double>>> tbb_temp_triplets;
    //#pragma omp parallel for
	// for(int i=0; i<num_nodes; ++i)
	// {
    //     if(is_surface_vertex[i] == 0) continue;
    //     VectorXa sum_dfdx(dim*num_nodes);
    //     sum_dfdx.setZero();
    //     VectorXa sum_dgdx(dim*num_nodes);
    //     sum_dgdx.setZero();
    //     std::vector<Entry> sum_d2fdx2;
    //     std::vector<Entry> sum_d2gdx2;

    //     //std::cout<<i<<" out of "<<slave_nodes_3d[0].size()<<std::endl;
	// 	AScalar fx = 0;
	// 	AScalar gx = 0;
    //     Vector3a xi = deformed.segment<3>(3*i);

    //     VectorXa ele_dfdx; 
    // 	VectorXa ele_dgdx;
    //     VectorXa ele_d2fdx; 
    // 	VectorXa ele_d2gdx;
    //     MatrixXa ele_d2fdxm;
    //     MatrixXa ele_d2gdxm;
    //     //std::cout<<i<<" out of "<<slave_nodes_3d[0].size()<<std::endl;

    //     Eigen::MatrixXd valence_indices;
    //     valence_indices.resize(current_indices[i].size(),8);

	// 	for(int k=0; k<current_indices[i].size(); ++k)
	// 	{
    //         Vector3a ck = deformed.segment<3>(current_indices[i][k]*3);
    //         //f(i == 1567) std::cout<<k<<" k out of "<<current_indices[i].size()<<std::endl;
    //         int valence = nodes_adjacency[current_indices[i][k]].size();
    //         Eigen::VectorXi valence_indices;
    //         VectorXa vs;

    //         if(valence <= 6)
    //         {
    //             ele_dfdx.resize(24);
    //             ele_dgdx.resize(24);
    //             ele_d2fdx.resize(24*24);
    //             ele_d2gdx.resize(24*24);
    //             ele_d2fdxm.resize(24,24);
    //             ele_d2gdxm.resize(24,24);
    //             vs.resize(18);
    //             valence_indices.resize(8);
    //             for(int a=0; a<6; ++a)
    //             {
    //                 vs.segment<3>(a*3) = ck;
    //                 valence_indices(a+2)=(current_indices[i][k]);
    //             }
    //         }else
    //         {
    //             ele_dfdx.resize(39);
    //             ele_dgdx.resize(39);
    //             ele_d2fdx.resize(39*39);
    //             ele_d2gdx.resize(39*39);
    //             ele_d2fdxm.resize(39,39);
    //             ele_d2gdxm.resize(39,39);
    //             vs.resize(33);
    //             valence_indices.resize(13);
    //             for(int a=0; a<11; ++a)
    //             {
    //                 vs.segment<3>(a*3) = ck;
    //                 valence_indices(a+2)=(current_indices[i][k]);
    //             }
    //         }
           

    //         valence_indices(0)=i;
    //         valence_indices(1)=(current_indices[i][k]);

           
    //         for(int a=0; a<nodes_adjacency[current_indices[i][k]].size(); ++a)
    //         {
    //             vs.segment<3>(a*3) = deformed.segment<3>(nodes_adjacency[current_indices[i][k]][a]*3);
    //             valence_indices(a+2) = nodes_adjacency[current_indices[i][k]][a];
    //         }
            
            
    //         if((ck-xi).norm()<=radius) {

    //             if(valence <= 6)
    //             {
    //                 ele_d2fdx = (d2fdx_func(xi, vs, ck, radius));
    //                 // ele_d2fdxm = ele_d2fdx.reshaped(24,24);
    //                 // ProjectSPD_IGL(ele_d2fdxm);
    //                 ele_d2gdx = (d2gdx_func(xi, ck, radius));
    //                 // ele_d2gdxm = ele_d2fdx.reshaped(24,24);
    //                 // ProjectSPD_IGL(ele_d2gdxm);
    //                 ele_dfdx = dfdx_func(xi, vs, ck, radius);
    //                 ele_dgdx = dgdx_func(xi, ck, radius);
    //                 fx += fx_func(xi, vs, ck, radius);
    //                 gx += gx_func(xi, ck, radius);
    //             }else
    //             {
    //                 ele_d2fdx = (d2fdx_func11(xi, vs, ck, radius));
    //                 //ele_d2fdxm = ele_d2fdx.reshaped(39,39);
    //                 //ProjectSPD_IGL(ele_d2fdxm);
    //                 ele_d2gdx = (d2gdx_func11(xi, ck, radius));
    //                 //ele_d2gdxm = ele_d2fdx.reshaped(39,39);
    //                 //ProjectSPD_IGL(ele_d2gdxm);
    //                 ele_dfdx = dfdx_func11(xi, vs, ck, radius);
    //                 ele_dgdx = dgdx_func11(xi, ck, radius);
    //                 fx += fx_func11(xi, vs, ck, radius);
    //                 gx += gx_func11(xi, ck, radius);
    //             }
                

    //             IMLS_local_gradient_to_global_gradient(ele_dfdx,valence_indices,dim,sum_dfdx);
    //             IMLS_local_gradient_to_global_gradient(ele_dgdx,valence_indices,dim,sum_dgdx);
    //             IMLS_local_hessian_to_global_triplets(ele_d2fdx,valence_indices,dim,sum_d2fdx2);
    //             IMLS_local_hessian_to_global_triplets(ele_d2gdx,valence_indices,dim,sum_d2gdx2);
    //             // IMLS_local_hessian_matrix_to_global_triplets(ele_d2fdxm,valence_indices,dim,sum_d2fdx2);
    //             // IMLS_local_hessian_matrix_to_global_triplets(ele_d2gdxm,valence_indices,dim,sum_d2gdx2);
    //         }
	// 	}
        
	// 	AScalar dist = fx/gx;
    //     if(abs(gx)>1e-6)
    //     {
    //         if(BARRIER_ENERGY && dist <= 0)
    //         {
    //             std::cout<<"NEGATIVE DISTANCE!!!"<<std::endl;
    //             continue;
    //         } 
    //         if(BARRIER_ENERGY && dist > barrier_distance)
    //         {
    //             continue;
    //         } 
    //         if(!BARRIER_ENERGY && dist > 0)
    //         {
    //             continue;
    //         }
             
    //         // for(int k=0; k<current_indices[i].size(); ++k)
	// 	    // {
    //         //     Vector1a A,B;
    //         //     A(0) = fx; B(0) = gx;
    //         //     Matrix24a local_hess = -IMLS_param*glue_hessian(A,B,ele_dfdx[k],ele_dgdx[k],ele_d2fdx[k],ele_d2gdx[k]);
    //         //     // IMLS_local_hessian_to_global_triplets(
    //         //     // local_hess,valence_indices[k], dim, entries);

    //         //     for (int a = 0; a < 8; a++) {
    //         //         for (int b = 0; b < 8; b++) {
    //         //             for (int c = 0; c < dim; c++) {
    //         //                 for (int l = 0; l < dim; l++) {
    //         //                     entries.emplace_back(
    //         //                         dim * valence_indices(k,a)+ c, dim * valence_indices(k,b) + l,
    //         //                         local_hess(dim * a + c, dim * b + l));
    //         //                 }
    //         //             }
    //         //         }
    //         //     }
    //         //     //std::cout<<" \n";
    //         // }
    //         // Eigen::VectorXd gradient(dim*num_nodes); 
    //         // gradient.setZero();
    //         // for(int k=0; k<current_indices[i].size(); ++k)
	// 	    // {
    //         //     // for(int a=0; a<8; ++a)
    //         //     //     std::cout<<valence_indices[k][a]<<" ";
    //         //     // std::cout<<"\n";
    //         //    // std::vector<int> temp = valence_indices[k];
    //         //     Vector1a A,B;
    //         //     A(0) = fx; B(0) = gx;
    //         //     Vector24a local_grad = glue_gradient(A,B,ele_dfdx[k],ele_dgdx[k]);
    //         //     //std::cout<<((gx * ele_dfdx[k] - fx * ele_dgdx[k]) / (gx * gx)-glue_gradient(A,B,ele_dfdx[k],ele_dgdx[k])).norm()<<std::endl;
    //         //     // IMLS_local_gradient_to_global_gradient(
    //         //     // local_grad,temp, dim, gradient);
    //         //     for (int a = 0; a < 8; a++) 
    //         //     {   
    //         //         //std::cout<<valence_indices[k][a]<<" ";
    //         //         gradient.segment(dim * valence_indices(k,a), dim) += local_grad.segment(dim * a, dim);
    //         //     }
    //         //     //std::cout<<"\n";
                
    //         // }

    //         // Compute d dist/ dx
    //         Eigen::VectorXd ddist_dx = (gx*sum_dfdx-fx*sum_dgdx)/(pow(gx,2));
            
    //         //std::cout<<"Norm: "<<(3*pow(fx/gx,2)*ddist_dx-gradient).norm()<<std::endl;

    //         // StiffnessMatrix Hessian(dim*num_nodes,dim*num_nodes);
    //         std::vector<Eigen::Triplet<double>> Hessian_t;

    //         //Part 1 6(f/g) (ddistdx)(ddistdx)^T
    //         if(!BARRIER_ENERGY)
    //             IMLS_vector_muliplication_to_triplets(ddist_dx,ddist_dx,-IMLS_param*6*(fx/gx),Hessian_t);
    //         else
    //             IMLS_vector_muliplication_to_triplets(ddist_dx,ddist_dx,IMLS_param*((barrier_distance/dist+2)*(barrier_distance/dist)-2*log(dist/barrier_distance)-3),Hessian_t);
    //         temp_triplets[i].insert(temp_triplets[i].end(), Hessian_t.begin(), Hessian_t.end());

    //         //Part 2 3(f/g)^2 d2distdx2
    //         // std::vector<Entry> contact_entries = entriesFromSparseMatrix(hess.block(0, 0, num_nodes * dim , num_nodes * dim));
    //         // entries.insert(entries.end(), contact_entries.begin(), contact_entries.end());
    //         // A.setFromTriplets(coefficients.begin(), coefficients.end());

    //         // h1 = d2f/dx2
    //         StiffnessMatrix h1(dim*num_nodes,dim*num_nodes);
    //         h1.setFromTriplets(sum_d2fdx2.begin(), sum_d2fdx2.end());

    //         // h2 = d2g/dx2
    //         StiffnessMatrix h2(dim*num_nodes,dim*num_nodes);
    //         h2.setFromTriplets(sum_d2gdx2.begin(), sum_d2gdx2.end());

    //         //h3 = dfdx*dgdx^T
    //         std::vector<Eigen::Triplet<double>> h3_t;
    //         IMLS_vector_muliplication_to_triplets(sum_dfdx,sum_dgdx,1.,h3_t);
    //         StiffnessMatrix h3(dim*num_nodes,dim*num_nodes);
    //         h3.setFromTriplets(h3_t.begin(), h3_t.end());

    //         //h4 = dgdx*dgdx^T
    //         std::vector<Eigen::Triplet<double>> h4_t;
    //         IMLS_vector_muliplication_to_triplets(sum_dgdx,sum_dgdx,1.,h4_t);
    //         StiffnessMatrix h4(dim*num_nodes,dim*num_nodes);
    //         h4.setFromTriplets(h4_t.begin(), h4_t.end());

    //         //h6 = dgdx*dfdx^T
    //         std::vector<Eigen::Triplet<double>> h6_t;
    //         IMLS_vector_muliplication_to_triplets(sum_dgdx,sum_dfdx,1.,h6_t);
    //         StiffnessMatrix h6(dim*num_nodes,dim*num_nodes);
    //         h6.setFromTriplets(h6_t.begin(), h6_t.end());

    //         //h5 = 3(f/g)^2 1/g h1 -2/g^2 h3+2f/g^3 h4 -f/g^2 h2
    //         StiffnessMatrix h5(dim*num_nodes,dim*num_nodes);
    //         if(!BARRIER_ENERGY)
    //             h5 = -IMLS_param*(1./gx*h1 - 1./pow(gx,2)*h3 -1./pow(gx,2)*h6 + 2*fx/pow(gx,3)*h4-fx/pow(gx,2)*h2)*3*pow(fx/gx,2);
    //         else
    //             h5 = IMLS_param*(1./gx*h1 - 1./pow(gx,2)*h3 -1./pow(gx,2)*h6 + 2*fx/pow(gx,3)*h4-fx/pow(gx,2)*h2)*((barrier_distance-dist)*(2*log(dist/barrier_distance)-barrier_distance/dist+1));
    //         std::vector<Entry> h5e = entriesFromSparseMatrix(h5.block(0, 0, num_nodes * dim , num_nodes * dim));
    //         temp_triplets[i].insert(temp_triplets[i].end(), h5e.begin(), h5e.end());
	// 	}
	// }

    std::cout<<num_nodes<<" "<<is_surface_vertex.size()<<std::endl;

    tbb::parallel_for(
    tbb::blocked_range<size_t>(size_t(0), num_nodes),
    [&](const tbb::blocked_range<size_t>& r) {
        auto& local_temp_triplets = tbb_temp_triplets.local();
        for (size_t i = r.begin(); i < r.end(); i++) {
            auto& local_current_index = tbb_current_indices[i];
            //std::cout<<"HESSIAN size: "<<local_temp_triplets.size()<<std::endl;
            if(is_surface_vertex[i] != 0)
            {
                VectorXa sum_dfdx(dim*num_nodes);
                sum_dfdx.setZero();
                VectorXa sum_dgdx(dim*num_nodes);
                sum_dgdx.setZero();
                std::vector<Entry> sum_d2fdx2;
                std::vector<Entry> sum_d2gdx2;

                //std::cout<<i<<" out of "<<slave_nodes_3d[0].size()<<std::endl;
                AScalar fx = 0;
                AScalar gx = 0;
                Vector3a xi = deformed.segment<3>(3*i);

                VectorXa ele_dfdx; 
                VectorXa ele_dgdx;
                VectorXa ele_d2fdx; 
                VectorXa ele_d2gdx;
                MatrixXa ele_d2fdxm;
                MatrixXa ele_d2gdxm;
                //std::cout<<i<<" out of "<<slave_nodes_3d[0].size()<<std::endl;

                Eigen::MatrixXd valence_indices;
                valence_indices.resize(local_current_index.size(),8);

                for(int k=0; k<local_current_index.size(); ++k)
                {
                    Vector3a ck = deformed.segment<3>(local_current_index[k]*3);
                    //f(i == 1567) std::cout<<k<<" k out of "<<local_current_index.size()<<std::endl;
                    int valence = nodes_adjacency[local_current_index[k]].size();
                    Eigen::VectorXi valence_indices;
                    VectorXa vs;

                    if(valence <= 6)
                    {
                        ele_dfdx.resize(24);
                        ele_dgdx.resize(24);
                        ele_d2fdx.resize(24*24);
                        ele_d2gdx.resize(24*24);
                        ele_d2fdxm.resize(24,24);
                        ele_d2gdxm.resize(24,24);
                        vs.resize(18);
                        valence_indices.resize(8);
                        for(int a=0; a<6; ++a)
                        {
                            vs.segment<3>(a*3) = ck;
                            valence_indices(a+2)=(local_current_index[k]);
                        }
                    }else
                    {
                        ele_dfdx.resize(39);
                        ele_dgdx.resize(39);
                        ele_d2fdx.resize(39*39);
                        ele_d2gdx.resize(39*39);
                        ele_d2fdxm.resize(39,39);
                        ele_d2gdxm.resize(39,39);
                        vs.resize(33);
                        valence_indices.resize(13);
                        for(int a=0; a<11; ++a)
                        {
                            vs.segment<3>(a*3) = ck;
                            valence_indices(a+2)=(local_current_index[k]);
                        }
                    }
                

                    valence_indices(0)=i;
                    valence_indices(1)=(local_current_index[k]);

                
                    for(int a=0; a<nodes_adjacency[local_current_index[k]].size(); ++a)
                    {
                        vs.segment<3>(a*3) = deformed.segment<3>(nodes_adjacency[local_current_index[k]][a]*3);
                        valence_indices(a+2) = nodes_adjacency[local_current_index[k]][a];
                    }
                    
                    
                    if((ck-xi).norm()<=radius) {

                        if(valence <= 6)
                        {
                            ele_d2fdx = (d2fdx_func(xi, vs, ck, radius));
                            // ele_d2fdxm = ele_d2fdx.reshaped(24,24);
                            // ProjectSPD_IGL(ele_d2fdxm);
                            ele_d2gdx = (d2gdx_func(xi, ck, radius));
                            // ele_d2gdxm = ele_d2fdx.reshaped(24,24);
                            // ProjectSPD_IGL(ele_d2gdxm);
                            ele_dfdx = dfdx_func(xi, vs, ck, radius);
                            ele_dgdx = dgdx_func(xi, ck, radius);
                            fx += fx_func(xi, vs, ck, radius);
                            gx += gx_func(xi, ck, radius);
                        }else
                        {
                            ele_d2fdx = (d2fdx_func11(xi, vs, ck, radius));
                            //ele_d2fdxm = ele_d2fdx.reshaped(39,39);
                            //ProjectSPD_IGL(ele_d2fdxm);
                            ele_d2gdx = (d2gdx_func11(xi, ck, radius));
                            //ele_d2gdxm = ele_d2fdx.reshaped(39,39);
                            //ProjectSPD_IGL(ele_d2gdxm);
                            ele_dfdx = dfdx_func11(xi, vs, ck, radius);
                            ele_dgdx = dgdx_func11(xi, ck, radius);
                            fx += fx_func11(xi, vs, ck, radius);
                            gx += gx_func11(xi, ck, radius);
                        }
                        

                        IMLS_local_gradient_to_global_gradient(ele_dfdx,valence_indices,dim,sum_dfdx);
                        IMLS_local_gradient_to_global_gradient(ele_dgdx,valence_indices,dim,sum_dgdx);
                        IMLS_local_hessian_to_global_triplets(ele_d2fdx,valence_indices,dim,sum_d2fdx2);
                        IMLS_local_hessian_to_global_triplets(ele_d2gdx,valence_indices,dim,sum_d2gdx2);
                        // IMLS_local_hessian_matrix_to_global_triplets(ele_d2fdxm,valence_indices,dim,sum_d2fdx2);
                        // IMLS_local_hessian_matrix_to_global_triplets(ele_d2gdxm,valence_indices,dim,sum_d2gdx2);
                    }
                }
                
                //AScalar dist = pow(fx/gx,2);
                AScalar dist = fabs(fx/gx);
                int sign = 1;
                if(fx/gx < 0) sign = -1;
                if(abs(gx)>1e-6)
                {
                    if(BARRIER_ENERGY && dist <= 0)
                    {
                        std::cout<<"NEGATIVE DISTANCE!!!"<<std::endl;
                        continue;
                    } 
                    if(BARRIER_ENERGY && dist > barrier_distance)
                    {
                        continue;
                    } 
                    if(!BARRIER_ENERGY && dist > 0)
                    {
                        continue;
                    }
                    
                    // for(int k=0; k<local_current_index.size(); ++k)
                    // {
                    //     Vector1a A,B;
                    //     A(0) = fx; B(0) = gx;
                    //     Matrix24a local_hess = -IMLS_param*glue_hessian(A,B,ele_dfdx[k],ele_dgdx[k],ele_d2fdx[k],ele_d2gdx[k]);
                    //     // IMLS_local_hessian_to_global_triplets(
                    //     // local_hess,valence_indices[k], dim, entries);

                    //     for (int a = 0; a < 8; a++) {
                    //         for (int b = 0; b < 8; b++) {
                    //             for (int c = 0; c < dim; c++) {
                    //                 for (int l = 0; l < dim; l++) {
                    //                     entries.emplace_back(
                    //                         dim * valence_indices(k,a)+ c, dim * valence_indices(k,b) + l,
                    //                         local_hess(dim * a + c, dim * b + l));
                    //                 }
                    //             }
                    //         }
                    //     }
                    //     //std::cout<<" \n";
                    // }
                    // Eigen::VectorXd gradient(dim*num_nodes); 
                    // gradient.setZero();
                    // for(int k=0; k<local_current_index.size(); ++k)
                    // {
                    //     // for(int a=0; a<8; ++a)
                    //     //     std::cout<<valence_indices[k][a]<<" ";
                    //     // std::cout<<"\n";
                    //    // std::vector<int> temp = valence_indices[k];
                    //     Vector1a A,B;
                    //     A(0) = fx; B(0) = gx;
                    //     Vector24a local_grad = glue_gradient(A,B,ele_dfdx[k],ele_dgdx[k]);
                    //     //std::cout<<((gx * ele_dfdx[k] - fx * ele_dgdx[k]) / (gx * gx)-glue_gradient(A,B,ele_dfdx[k],ele_dgdx[k])).norm()<<std::endl;
                    //     // IMLS_local_gradient_to_global_gradient(
                    //     // local_grad,temp, dim, gradient);
                    //     for (int a = 0; a < 8; a++) 
                    //     {   
                    //         //std::cout<<valence_indices[k][a]<<" ";
                    //         gradient.segment(dim * valence_indices(k,a), dim) += local_grad.segment(dim * a, dim);
                    //     }
                    //     //std::cout<<"\n";
                        
                    // }

                    // Compute d dist/ dx
                    Eigen::VectorXd ddist_dx = sign*(gx*sum_dfdx-fx*sum_dgdx)/(pow(gx,2));
                    Eigen::VectorXd ddist0_dx = (gx*sum_dfdx-fx*sum_dgdx)/(pow(gx,2));
                    
                    //std::cout<<"Norm: "<<(3*pow(fx/gx,2)*ddist_dx-gradient).norm()<<std::endl;

                    // StiffnessMatrix Hessian(dim*num_nodes,dim*num_nodes);
                    std::vector<Eigen::Triplet<double>> Hessian_t;

                    //Part 1 6(f/g) (ddistdx)(ddistdx)^T
                    if(!BARRIER_ENERGY)
                        IMLS_vector_muliplication_to_triplets(ddist_dx,ddist_dx,-IMLS_param*6*(fx/gx),Hessian_t);
                    else
                    {
                        IMLS_vector_muliplication_to_triplets(ddist_dx,ddist_dx,IMLS_param*((barrier_distance/dist+2)*(barrier_distance/dist)-2*log(dist/barrier_distance)-3),Hessian_t);
                        //IMLS_vector_muliplication_to_triplets(ddist0_dx,ddist0_dx,2*((barrier_distance-dist)*(2*log(dist/barrier_distance)-barrier_distance/dist+1)),Hessian_t);
                    }

                    std::vector<Entry> local;   
                    local.insert(local.end(), Hessian_t.begin(), Hessian_t.end());

                    //std::cout<<"HESSIAN size: "<<local.size()<<std::endl;

                    //Part 2 3(f/g)^2 d2distdx2
                    // std::vector<Entry> contact_entries = entriesFromSparseMatrix(hess.block(0, 0, num_nodes * dim , num_nodes * dim));
                    // entries.insert(entries.end(), contact_entries.begin(), contact_entries.end());
                    // A.setFromTriplets(coefficients.begin(), coefficients.end());

                    // h1 = d2f/dx2
                    StiffnessMatrix h1(dim*num_nodes,dim*num_nodes);
                    h1.setFromTriplets(sum_d2fdx2.begin(), sum_d2fdx2.end());

                    // h2 = d2g/dx2
                    StiffnessMatrix h2(dim*num_nodes,dim*num_nodes);
                    h2.setFromTriplets(sum_d2gdx2.begin(), sum_d2gdx2.end());

                    //h3 = dfdx*dgdx^T
                    std::vector<Eigen::Triplet<double>> h3_t;
                    IMLS_vector_muliplication_to_triplets(sum_dfdx,sum_dgdx,1.,h3_t);
                    StiffnessMatrix h3(dim*num_nodes,dim*num_nodes);
                    h3.setFromTriplets(h3_t.begin(), h3_t.end());

                    //h4 = dgdx*dgdx^T
                    std::vector<Eigen::Triplet<double>> h4_t;
                    IMLS_vector_muliplication_to_triplets(sum_dgdx,sum_dgdx,1.,h4_t);
                    StiffnessMatrix h4(dim*num_nodes,dim*num_nodes);
                    h4.setFromTriplets(h4_t.begin(), h4_t.end());

                    //h6 = dgdx*dfdx^T
                    std::vector<Eigen::Triplet<double>> h6_t;
                    IMLS_vector_muliplication_to_triplets(sum_dgdx,sum_dfdx,1.,h6_t);
                    StiffnessMatrix h6(dim*num_nodes,dim*num_nodes);
                    h6.setFromTriplets(h6_t.begin(), h6_t.end());

                    //h5 = 3(f/g)^2 1/g h1 -2/g^2 h3+2f/g^3 h4 -f/g^2 h2
                    StiffnessMatrix h5(dim*num_nodes,dim*num_nodes);
                    if(!BARRIER_ENERGY)
                        h5 = -IMLS_param*(1./gx*h1 - 1./pow(gx,2)*h3 -1./pow(gx,2)*h6 + 2*fx/pow(gx,3)*h4-fx/pow(gx,2)*h2)*3*pow(fx/gx,2);
                    else
                        h5 = sign*IMLS_param*(1./gx*h1 - 1./pow(gx,2)*h3 -1./pow(gx,2)*h6 + 2*fx/pow(gx,3)*h4-fx/pow(gx,2)*h2)*((barrier_distance-dist)*(2*log(dist/barrier_distance)-barrier_distance/dist+1));
                    std::vector<Entry> h5e = entriesFromSparseMatrix(h5.block(0, 0, num_nodes * dim , num_nodes * dim));
                    local.insert(local.end(), h5e.begin(), h5e.end());
                    h5.setFromTriplets(local.begin(),local.end());
                    std::vector<Entry> h5f = entriesFromSparseMatrix(h5.block(0, 0, num_nodes * dim , num_nodes * dim));
                    //std::cout<<"HESSIAN: "<<local_temp_triplets.size()<<std::endl;
                    local_temp_triplets.insert(local_temp_triplets.end(),h5f.begin(),h5f.end());
                    
                }   
            }
        }   
    });

    //std::vector<Entry> test;

    for (const auto& local_temp_triplets : tbb_temp_triplets)
    {
        if(local_temp_triplets.size() > 0)
            entries.insert(entries.end(),local_temp_triplets.begin(),local_temp_triplets.end());
        //test.insert(test.end(),local_temp_triplets.begin(),local_temp_triplets.end());
    }

    // StiffnessMatrix testm(dim*num_nodes,dim*num_nodes);
    // if(test.size()>0)
    // {
    //     testm.setFromTriplets(test.begin(),test.end());
    //     Eigen::SimplicialLDLT<StiffnessMatrix> ldltoft(testm);
    //     if(ldltoft.info() == Eigen::NumericalIssue)
    //     {
    //         std::cout<<"IMLS Hessian Non-PSD"<<std::endl;
    //     }
    // }
}


template <int dim>
void FEMSolver<dim>::addFastIMLSSameSideEnergy(T& energy)
{
    BuildAcceleratorTree(false);
	current_indices.clear();
    dist_info.clear();

    
	for(int i=0; i<close_slave_nodes.size(); ++i)
	{
		current_indices.push_back(std::vector<int>());

		Vector3a xi = deformed.segment<3>(3*(close_slave_nodes[i]+num_nodes-1));

		Fuzzy_sphere fs(Point_d(xi[0], xi[1], xi[2]), radius+radius*0.2, radius*0.2);

		std::vector<std::tuple<Point_d, int>> points_query;
		accelerator.search(std::back_inserter(points_query), fs);

		for(int k=0; k<points_query.size(); ++k)
			current_indices.back().push_back(std::get<1>(points_query[k]));
	}

	for(int i=0; i<close_slave_nodes.size(); ++i)
	{
		AScalar fx = 0;
		AScalar gx = 0;

		Vector3a xi = deformed.segment<3>(3*(close_slave_nodes[i]+num_nodes-1));

		for(int k=0; k<current_indices[i].size(); ++k)
		{
            Vector3a ck = deformed.segment<3>(current_indices[i][k]*3);
            //f(i == 1567) std::cout<<k<<" k out of "<<current_indices[i].size()<<std::endl;
            int valence = slave_nodes_adjacency[current_indices[i][k]-slave_nodes_shift].size();
            VectorXa vs;

            if(valence <= 6)
            {
                vs.resize(18);
                for(int a=0; a<6; ++a)
                {
                    vs.segment<3>(a*3) = ck;
                }
            }else
            {
                vs.resize(33);
                for(int a=0; a<11; ++a)
                {
                    vs.segment<3>(a*3) = ck;
                }
            }

           
            for(int a=0; a<slave_nodes_adjacency[current_indices[i][k]-slave_nodes_shift].size(); ++a)
            {
                vs.segment<3>(a*3) = deformed.segment<3>(slave_nodes_adjacency[current_indices[i][k]-slave_nodes_shift][a]*3);
            }
            
            if((ck-xi).norm()<=radius) {
                if(valence <= 6)
                {
                    fx += fx_func(xi, vs, ck, radius);
                    gx += gx_func(xi, ck, radius);
                }else
                {
                    fx += fx_func11(xi, vs, ck, radius);
                    gx += gx_func11(xi, ck, radius);
                }
            }
		}

		AScalar dist = fx/gx;

        

		if(abs(gx)>1e-6)
        {
            std::cout<<"Current Distance: "<<dist<<std::endl;
            if(dist<0)
                energy += -IMLS_pen_scale*IMLS_param*pow((fx/gx),3);
            else
                energy += IMLS_pen_scale*IMLS_param*pow((fx/gx),3);
        }
	}
}

template <int dim>
void FEMSolver<dim>::addFastIMLSSameSideForceEntries(VectorXT& residual)
{
    BuildAcceleratorTree(false);

    VectorXa gradient((additional_dof+num_nodes)*3);
	gradient.setZero();

	current_indices.clear();
    
	for(int i=0; i<close_slave_nodes.size(); ++i)
	{
		current_indices.push_back(std::vector<int>());

		Vector3a xi = deformed.segment<3>(3*(close_slave_nodes[i]+num_nodes-1));

		Fuzzy_sphere fs(Point_d(xi[0], xi[1], xi[2]), radius+radius*0.1, radius*0.1);

		std::vector<std::tuple<Point_d, int>> points_query;
		accelerator.search(std::back_inserter(points_query), fs);

		for(int k=0; k<points_query.size(); ++k)
			current_indices.back().push_back(std::get<1>(points_query[k]));

	}

	for(int i=0; i<close_slave_nodes.size(); ++i)
	{
		VectorXa sum_dfdx((additional_dof+num_nodes)*3);
        sum_dfdx.setZero();
        VectorXa sum_dgdx((additional_dof+num_nodes)*3);
        sum_dgdx.setZero();

        //std::cout<<i<<" out of "<<slave_nodes_3d[0].size()<<std::endl;
		AScalar fx = 0;
		AScalar gx = 0;
        Vector3a xi = deformed.segment<3>(3*(close_slave_nodes[i]+num_nodes-1));

        VectorXa ele_dfdx; 
    	VectorXa ele_dgdx;
        //std::cout<<i<<" out of "<<slave_nodes_3d[0].size()<<std::endl;

		for(int k=0; k<current_indices[i].size(); ++k)
		{
            Vector3a ck = deformed.segment<3>(current_indices[i][k]*3);
            //f(i == 1567) std::cout<<k<<" k out of "<<current_indices[i].size()<<std::endl;
            int valence = slave_nodes_adjacency[current_indices[i][k]-slave_nodes_shift].size();
            Eigen::VectorXi valence_indices;
            VectorXa vs;

            if(valence <= 6)
            {
                ele_dfdx.resize(24);
                ele_dgdx.resize(24);
                vs.resize(18);
                valence_indices.resize(8);
                for(int a=0; a<6; ++a)
                {
                    vs.segment<3>(a*3) = ck;
                    valence_indices(a+2)=(current_indices[i][k]);
                }
            }else
            {
                ele_dfdx.resize(39);
                ele_dgdx.resize(39);
                vs.resize(33);
                valence_indices.resize(13);
                for(int a=0; a<11; ++a)
                {
                    vs.segment<3>(a*3) = ck;
                    valence_indices(a+2)=(current_indices[i][k]);
                }
            }
           

            valence_indices(0)=(close_slave_nodes[i]+num_nodes-1);
            valence_indices(1)=(current_indices[i][k]);

           
            for(int a=0; a<slave_nodes_adjacency[current_indices[i][k]-slave_nodes_shift].size(); ++a)
            {
                vs.segment<3>(a*3) = deformed.segment<3>(slave_nodes_adjacency[current_indices[i][k]-slave_nodes_shift][a]*3);
                valence_indices(a+2) = slave_nodes_adjacency[current_indices[i][k]-slave_nodes_shift][a];
            }
            
            if((ck-xi).norm()<=radius) {
                if(valence <= 6)
                {
                    ele_dfdx = dfdx_func(xi, vs, ck, radius);
                    ele_dgdx = dgdx_func(xi, ck, radius);
                    fx += fx_func(xi, vs, ck, radius);
                    gx += gx_func(xi, ck, radius);
                }else
                {
                    ele_dfdx = dfdx_func11(xi, vs, ck, radius);
                    ele_dgdx = dgdx_func11(xi, ck, radius);
                    fx += fx_func11(xi, vs, ck, radius);
                    gx += gx_func11(xi, ck, radius);
                }
                IMLS_local_gradient_to_global_gradient(ele_dfdx,valence_indices,dim,sum_dfdx);
                IMLS_local_gradient_to_global_gradient(ele_dgdx,valence_indices,dim,sum_dgdx);
            }
		}
        AScalar dist = fx/gx;

        if(abs(gx)>1e-6)
        {
            // for(int k=0; k<current_indices[i].size(); ++k)
		    // {
            //     // for(int a=0; a<8; ++a)
            //     //     std::cout<<valence_indices[k][a]<<" ";
            //     // std::cout<<"\n";
            //    // std::vector<int> temp = valence_indices[k];
            //     Vector1a A,B;
            //     A(0) = fx; B(0) = gx;
            //     Vector24a local_grad = -IMLS_param*glue_gradient(A,B,ele_dfdx[k],ele_dgdx[k]);
            //     //std::cout<<((gx * ele_dfdx[k] - fx * ele_dgdx[k]) / (gx * gx)-glue_gradient(A,B,ele_dfdx[k],ele_dgdx[k])).norm()<<std::endl;
            //     // IMLS_local_gradient_to_global_gradient(
            //     // local_grad,temp, dim, gradient);
            //     for (int a = 0; a < 8; a++) 
            //     {   
            //         //std::cout<<valence_indices[k][a]<<" ";
            //         gradient.segment(dim * valence_indices(k,a), dim) += local_grad.segment(dim * a, dim);
            //     }
            //     //std::cout<<"\n";
                
            // }
            if(dist<0)
                gradient = -3*IMLS_pen_scale*IMLS_param*pow(fx/gx,2)*(gx*sum_dfdx-fx*sum_dgdx)/(pow(gx,2));
            else
                gradient = 3*IMLS_pen_scale*IMLS_param*pow(fx/gx,2)*(gx*sum_dfdx-fx*sum_dgdx)/(pow(gx,2));
            residual.segment(0, (additional_dof+num_nodes) * dim) += -gradient.segment(0, (additional_dof+num_nodes) * dim);
		}
	}
    //residual.segment(0, num_nodes * dim) += -gradient.segment(0, num_nodes * dim);
}

template <int dim>
void FEMSolver<dim>::addFastIMLSSameSideHessianEntries(std::vector<Entry>& entries,bool project_PD)
{
    BuildAcceleratorTree(false);
	current_indices.clear();


	for(int i=0; i<close_slave_nodes.size(); ++i)
	{
		current_indices.push_back(std::vector<int>());

		Vector3a xi = deformed.segment<3>(3*(close_slave_nodes[i]+num_nodes-1));

		Fuzzy_sphere fs(Point_d(xi[0], xi[1], xi[2]), radius+radius*0.1, radius*0.1);

		std::vector<std::tuple<Point_d, int>> points_query;
		accelerator.search(std::back_inserter(points_query), fs);

		for(int k=0; k<points_query.size(); ++k)
			current_indices.back().push_back(std::get<1>(points_query[k]));
	}

	for(int i=0; i<close_slave_nodes.size(); ++i)
	{
        VectorXa sum_dfdx(dim*(additional_dof+num_nodes)*3);
        sum_dfdx.setZero();
        VectorXa sum_dgdx(dim*(additional_dof+num_nodes)*3);
        sum_dgdx.setZero();
        std::vector<Entry> sum_d2fdx2;
        std::vector<Entry> sum_d2gdx2;

        //std::cout<<i<<" out of "<<slave_nodes_3d[0].size()<<std::endl;
		AScalar fx = 0;
		AScalar gx = 0;
        Vector3a xi = deformed.segment<3>(3*(close_slave_nodes[i]+num_nodes-1));

        VectorXa ele_dfdx; 
    	VectorXa ele_dgdx;
        VectorXa ele_d2fdx; 
    	VectorXa ele_d2gdx;
        //std::cout<<i<<" out of "<<slave_nodes_3d[0].size()<<std::endl;

        Eigen::MatrixXd valence_indices;
        valence_indices.resize(current_indices[i].size(),8);

		for(int k=0; k<current_indices[i].size(); ++k)
		{
            Vector3a ck = deformed.segment<3>(current_indices[i][k]*3);
            //f(i == 1567) std::cout<<k<<" k out of "<<current_indices[i].size()<<std::endl;
            int valence = slave_nodes_adjacency[current_indices[i][k]-slave_nodes_shift].size();
            Eigen::VectorXi valence_indices;
            VectorXa vs;

            if(valence <= 6)
            {
                ele_dfdx.resize(24);
                ele_dgdx.resize(24);
                ele_d2fdx.resize(24*24);
                ele_d2gdx.resize(24*24);
                vs.resize(18);
                valence_indices.resize(8);
                for(int a=0; a<6; ++a)
                {
                    vs.segment<3>(a*3) = ck;
                    valence_indices(a+2)=(current_indices[i][k]);
                }
            }else
            {
                ele_dfdx.resize(39);
                ele_dgdx.resize(39);
                ele_d2fdx.resize(39*39);
                ele_d2gdx.resize(39*39);
                vs.resize(33);
                valence_indices.resize(13);
                for(int a=0; a<11; ++a)
                {
                    vs.segment<3>(a*3) = ck;
                    valence_indices(a+2)=(current_indices[i][k]);
                }
            }
           

            valence_indices(0)=(close_slave_nodes[i]+num_nodes-1);
            valence_indices(1)=(current_indices[i][k]);

           
            for(int a=0; a<slave_nodes_adjacency[current_indices[i][k]-slave_nodes_shift].size(); ++a)
            {
                vs.segment<3>(a*3) = deformed.segment<3>(slave_nodes_adjacency[current_indices[i][k]-slave_nodes_shift][a]*3);
                valence_indices(a+2) = slave_nodes_adjacency[current_indices[i][k]-slave_nodes_shift][a];
            }
            
            
            if((ck-xi).norm()<=radius) {

                if(valence <= 6)
                {
                    ele_d2fdx = d2fdx_func(xi, vs, ck, radius);
                    ele_d2gdx = d2gdx_func(xi, ck, radius);
                    ele_dfdx = dfdx_func(xi, vs, ck, radius);
                    ele_dgdx = dgdx_func(xi, ck, radius);
                    fx += fx_func(xi, vs, ck, radius);
                    gx += gx_func(xi, ck, radius);
                }else
                {
                    ele_d2fdx = d2fdx_func11(xi, vs, ck, radius);
                    ele_d2gdx = d2gdx_func11(xi, ck, radius);
                    ele_dfdx = dfdx_func11(xi, vs, ck, radius);
                    ele_dgdx = dgdx_func11(xi, ck, radius);
                    fx += fx_func11(xi, vs, ck, radius);
                    gx += gx_func11(xi, ck, radius);
                }
                

                IMLS_local_gradient_to_global_gradient(ele_dfdx,valence_indices,dim,sum_dfdx);
                IMLS_local_gradient_to_global_gradient(ele_dgdx,valence_indices,dim,sum_dgdx);
                IMLS_local_hessian_to_global_triplets(ele_d2fdx,valence_indices,dim,sum_d2fdx2);
                IMLS_local_hessian_to_global_triplets(ele_d2gdx,valence_indices,dim,sum_d2gdx2);
            }
		}
        
		AScalar dist = fx/gx;
        if(abs(gx)>1e-6)
        {
            // for(int k=0; k<current_indices[i].size(); ++k)
		    // {
            //     Vector1a A,B;
            //     A(0) = fx; B(0) = gx;
            //     Matrix24a local_hess = -IMLS_param*glue_hessian(A,B,ele_dfdx[k],ele_dgdx[k],ele_d2fdx[k],ele_d2gdx[k]);
            //     // IMLS_local_hessian_to_global_triplets(
            //     // local_hess,valence_indices[k], dim, entries);

            //     for (int a = 0; a < 8; a++) {
            //         for (int b = 0; b < 8; b++) {
            //             for (int c = 0; c < dim; c++) {
            //                 for (int l = 0; l < dim; l++) {
            //                     entries.emplace_back(
            //                         dim * valence_indices(k,a)+ c, dim * valence_indices(k,b) + l,
            //                         local_hess(dim * a + c, dim * b + l));
            //                 }
            //             }
            //         }
            //     }
            //     //std::cout<<" \n";
            // }
            // Eigen::VectorXd gradient(dim*num_nodes); 
            // gradient.setZero();
            // for(int k=0; k<current_indices[i].size(); ++k)
		    // {
            //     // for(int a=0; a<8; ++a)
            //     //     std::cout<<valence_indices[k][a]<<" ";
            //     // std::cout<<"\n";
            //    // std::vector<int> temp = valence_indices[k];
            //     Vector1a A,B;
            //     A(0) = fx; B(0) = gx;
            //     Vector24a local_grad = glue_gradient(A,B,ele_dfdx[k],ele_dgdx[k]);
            //     //std::cout<<((gx * ele_dfdx[k] - fx * ele_dgdx[k]) / (gx * gx)-glue_gradient(A,B,ele_dfdx[k],ele_dgdx[k])).norm()<<std::endl;
            //     // IMLS_local_gradient_to_global_gradient(
            //     // local_grad,temp, dim, gradient);
            //     for (int a = 0; a < 8; a++) 
            //     {   
            //         //std::cout<<valence_indices[k][a]<<" ";
            //         gradient.segment(dim * valence_indices(k,a), dim) += local_grad.segment(dim * a, dim);
            //     }
            //     //std::cout<<"\n";
                
            // }

            int sign = -1;
            if(dist>0) sign = 1;

            // Compute d dist/ dx
            Eigen::VectorXd ddist_dx = (gx*sum_dfdx-fx*sum_dgdx)/(pow(gx,2));
            //std::cout<<"Norm: "<<(3*pow(fx/gx,2)*ddist_dx-gradient).norm()<<std::endl;

            // StiffnessMatrix Hessian(dim*num_nodes,dim*num_nodes);
            std::vector<Eigen::Triplet<double>> Hessian_t;

            //Part 1 6(f/g) (ddistdx)(ddistdx)^T
            IMLS_vector_muliplication_to_triplets(ddist_dx,ddist_dx,sign*IMLS_pen_scale*IMLS_param*6*(fx/gx),Hessian_t);
            entries.insert(entries.end(), Hessian_t.begin(), Hessian_t.end());

            //Part 2 3(f/g)^2 d2distdx2
            // std::vector<Entry> contact_entries = entriesFromSparseMatrix(hess.block(0, 0, num_nodes * dim , num_nodes * dim));
            // entries.insert(entries.end(), contact_entries.begin(), contact_entries.end());
            // A.setFromTriplets(coefficients.begin(), coefficients.end());

            // h1 = d2f/dx2
            StiffnessMatrix h1((additional_dof+num_nodes)*3,(additional_dof+num_nodes)*3);
            h1.setFromTriplets(sum_d2fdx2.begin(), sum_d2fdx2.end());

            // h2 = d2g/dx2
            StiffnessMatrix h2((additional_dof+num_nodes)*3,(additional_dof+num_nodes)*3);
            h2.setFromTriplets(sum_d2gdx2.begin(), sum_d2gdx2.end());

            //h3 = dfdx*dgdx^T
            std::vector<Eigen::Triplet<double>> h3_t;
            IMLS_vector_muliplication_to_triplets(sum_dfdx,sum_dgdx,1.,h3_t);
            StiffnessMatrix h3((additional_dof+num_nodes)*3,(additional_dof+num_nodes)*3);
            h3.setFromTriplets(h3_t.begin(), h3_t.end());

            //h4 = dgdx*dgdx^T
            std::vector<Eigen::Triplet<double>> h4_t;
            IMLS_vector_muliplication_to_triplets(sum_dgdx,sum_dgdx,1.,h4_t);
            StiffnessMatrix h4((additional_dof+num_nodes)*3,(additional_dof+num_nodes)*3);
            h4.setFromTriplets(h4_t.begin(), h4_t.end());

            //h6 = dgdx*dfdx^T
            std::vector<Eigen::Triplet<double>> h6_t;
            IMLS_vector_muliplication_to_triplets(sum_dgdx,sum_dfdx,1.,h6_t);
            StiffnessMatrix h6((additional_dof+num_nodes)*3,(additional_dof+num_nodes)*3);
            h6.setFromTriplets(h6_t.begin(), h6_t.end());

            //h5 = 3(f/g)^2 1/g h1 -2/g^2 h3+2f/g^3 h4 -f/g^2 h2
            StiffnessMatrix h5((additional_dof+num_nodes)*3,(additional_dof+num_nodes)*3);
            h5 = sign*IMLS_pen_scale*IMLS_param*(1./gx*h1 - 1./pow(gx,2)*h3 -1./pow(gx,2)*h6 + 2*fx/pow(gx,3)*h4-fx/pow(gx,2)*h2)*3*pow(fx/gx,2);
            std::vector<Entry> h5e = entriesFromSparseMatrix(h5.block(0, 0, (additional_dof+num_nodes) * dim , (additional_dof+num_nodes) * dim));
            entries.insert(entries.end(), h5e.begin(), h5e.end());
		}
	}
}

template <int dim>
void FEMSolver<dim>::addFastIMLS12Energy(T& energy)
{
    BuildAcceleratorTree();
	current_indices.clear();
    //close_slave_nodes.clear();
    auto it = slave_nodes_3d[0].begin();
    dist_info.clear();
	for(int i=0; i<slave_nodes_3d[0].size(); ++i)
	{
		current_indices.push_back(std::vector<int>());

		Vector3a xi = deformed.segment<3>(3*(it->second+num_nodes-1));

		Fuzzy_sphere fs(Point_d(xi[0], xi[1], xi[2]), radius+radius*0.2, radius*0.2);

		std::vector<std::tuple<Point_d, int>> points_query;
		accelerator.search(std::back_inserter(points_query), fs);

		for(int k=0; k<points_query.size(); ++k)
			current_indices.back().push_back(std::get<1>(points_query[k]));

        it++;
	}

	it = slave_nodes_3d[0].begin();
	for(int i=0; i<slave_nodes_3d[0].size(); ++i)
	{
		AScalar fx = 0;
		AScalar gx = 0;

		Vector3a xi = deformed.segment<3>(3*(it->second+num_nodes-1));

		for(int k=0; k<current_indices[i].size(); ++k)
		{
            Vector3a ck = deformed.segment<3>(current_indices[i][k]*3);
            //f(i == 1567) std::cout<<k<<" k out of "<<current_indices[i].size()<<std::endl;
            int valence = master_nodes_adjacency[current_indices[i][k]].size();
            VectorXa vs;

            if(valence <= 6)
            {
                vs.resize(18);
                for(int a=0; a<6; ++a)
                {
                    vs.segment<3>(a*3) = ck;
                }
            }else
            {
                vs.resize(33);
                for(int a=0; a<11; ++a)
                {
                    vs.segment<3>(a*3) = ck;
                }
            }

           
            for(int a=0; a<master_nodes_adjacency[current_indices[i][k]].size(); ++a)
            {
                vs.segment<3>(a*3) = deformed.segment<3>(master_nodes_adjacency[current_indices[i][k]][a]*3);
            }
            
            if((ck-xi).norm()<=radius) {
                if(valence <= 6)
                {
                    fx += fx_func(xi, vs, ck, radius);
                    gx += gx_func(xi, ck, radius);
                }else
                {
                    fx += fx_func11(xi, vs, ck, radius);
                    gx += gx_func11(xi, ck, radius);
                }
            }
		}

		AScalar dist = fx/gx;

        

		if(abs(gx)>1e-6)
        {
            //std::cout<<dist<<std::endl;
            if(dist<0)
                energy += -IMLS_param*pow((fx/gx),3);
            //dist_info.push_back(std::pair<int,double>(it->first,dist));
            //close_slave_nodes.push_back(it->first);
        }
			

        it++;
	}
}

template <int dim>
void FEMSolver<dim>::addFastIMLS12ForceEntries(VectorXT& residual)
{
    BuildAcceleratorTree();

    VectorXa gradient(num_nodes*3);
	gradient.setZero();
    //close_slave_nodes.clear();

	current_indices.clear();
    auto it = slave_nodes_3d[0].begin();
	for(int i=0; i<slave_nodes_3d[0].size(); ++i)
	{
		current_indices.push_back(std::vector<int>());

		Vector3a xi = deformed.segment<3>(3*(it->second+num_nodes-1));

		Fuzzy_sphere fs(Point_d(xi[0], xi[1], xi[2]), radius+radius*0.1, radius*0.1);

		std::vector<std::tuple<Point_d, int>> points_query;
		accelerator.search(std::back_inserter(points_query), fs);

		for(int k=0; k<points_query.size(); ++k)
			current_indices.back().push_back(std::get<1>(points_query[k]));

        it++;
	}

	it = slave_nodes_3d[0].begin();
	for(int i=0; i<slave_nodes_3d[0].size(); ++i)
	{
		VectorXa sum_dfdx((additional_dof+num_nodes)*3);
        sum_dfdx.setZero();
        VectorXa sum_dgdx((additional_dof+num_nodes)*3);
        sum_dgdx.setZero();

        //std::cout<<i<<" out of "<<slave_nodes_3d[0].size()<<std::endl;
		AScalar fx = 0;
		AScalar gx = 0;
        Vector3a xi = deformed.segment<3>(3*(it->second+num_nodes-1));

        VectorXa ele_dfdx; 
    	VectorXa ele_dgdx;
        //std::cout<<i<<" out of "<<slave_nodes_3d[0].size()<<std::endl;

		for(int k=0; k<current_indices[i].size(); ++k)
		{
            Vector3a ck = deformed.segment<3>(current_indices[i][k]*3);
            //f(i == 1567) std::cout<<k<<" k out of "<<current_indices[i].size()<<std::endl;
            int valence = master_nodes_adjacency[current_indices[i][k]].size();
            Eigen::VectorXi valence_indices;
            VectorXa vs;

            if(valence <= 6)
            {
                ele_dfdx.resize(24);
                ele_dgdx.resize(24);
                vs.resize(18);
                valence_indices.resize(8);
                for(int a=0; a<6; ++a)
                {
                    vs.segment<3>(a*3) = ck;
                    valence_indices(a+2)=(current_indices[i][k]);
                }
            }else
            {
                ele_dfdx.resize(39);
                ele_dgdx.resize(39);
                vs.resize(33);
                valence_indices.resize(13);
                for(int a=0; a<11; ++a)
                {
                    vs.segment<3>(a*3) = ck;
                    valence_indices(a+2)=(current_indices[i][k]);
                }
            }
           

            valence_indices(0)=(it->second+num_nodes-1);
            valence_indices(1)=(current_indices[i][k]);

           
            for(int a=0; a<master_nodes_adjacency[current_indices[i][k]].size(); ++a)
            {
                vs.segment<3>(a*3) = deformed.segment<3>(master_nodes_adjacency[current_indices[i][k]][a]*3);
                valence_indices(a+2) = master_nodes_adjacency[current_indices[i][k]][a];
            }
            
            if((ck-xi).norm()<=radius) {
                if(valence <= 6)
                {
                    ele_dfdx = dfdx_func(xi, vs, ck, radius);
                    ele_dgdx = dgdx_func(xi, ck, radius);
                    fx += fx_func(xi, vs, ck, radius);
                    gx += gx_func(xi, ck, radius);
                }else
                {
                    ele_dfdx = dfdx_func11(xi, vs, ck, radius);
                    ele_dgdx = dgdx_func11(xi, ck, radius);
                    fx += fx_func11(xi, vs, ck, radius);
                    gx += gx_func11(xi, ck, radius);
                }
                IMLS_local_gradient_to_global_gradient(ele_dfdx,valence_indices,dim,sum_dfdx);
                IMLS_local_gradient_to_global_gradient(ele_dgdx,valence_indices,dim,sum_dgdx);
            }
		}
        AScalar dist = fx/gx;

        //if(abs(gx) > 1e-6) close_slave_nodes.push_back(it->first);
        if(dist< 0 && abs(gx)>1e-6)
        {
            // for(int k=0; k<current_indices[i].size(); ++k)
		    // {
            //     // for(int a=0; a<8; ++a)
            //     //     std::cout<<valence_indices[k][a]<<" ";
            //     // std::cout<<"\n";
            //    // std::vector<int> temp = valence_indices[k];
            //     Vector1a A,B;
            //     A(0) = fx; B(0) = gx;
            //     Vector24a local_grad = -IMLS_param*glue_gradient(A,B,ele_dfdx[k],ele_dgdx[k]);
            //     //std::cout<<((gx * ele_dfdx[k] - fx * ele_dgdx[k]) / (gx * gx)-glue_gradient(A,B,ele_dfdx[k],ele_dgdx[k])).norm()<<std::endl;
            //     // IMLS_local_gradient_to_global_gradient(
            //     // local_grad,temp, dim, gradient);
            //     for (int a = 0; a < 8; a++) 
            //     {   
            //         //std::cout<<valence_indices[k][a]<<" ";
            //         gradient.segment(dim * valence_indices(k,a), dim) += local_grad.segment(dim * a, dim);
            //     }
            //     //std::cout<<"\n";
                
            // }
            gradient = -3*IMLS_param*pow(fx/gx,2)*(gx*sum_dfdx-fx*sum_dgdx)/(pow(gx,2));
            residual.segment(0, (additional_dof+num_nodes) * dim) += -gradient.segment(0, (additional_dof+num_nodes) * dim);
		}
        it++;
	}
    //residual.segment(0, num_nodes * dim) += -gradient.segment(0, num_nodes * dim);
}

template <int dim>
void FEMSolver<dim>::addFastIMLS12HessianEntries(std::vector<Entry>& entries,bool project_PD)
{
    BuildAcceleratorTree();
	current_indices.clear();
    auto it = slave_nodes_3d[0].begin();

	for(int i=0; i<slave_nodes_3d[0].size(); ++i)
	{
		current_indices.push_back(std::vector<int>());

		Vector3a xi = deformed.segment<3>(3*(it->second+num_nodes-1));

		Fuzzy_sphere fs(Point_d(xi[0], xi[1], xi[2]), radius+radius*0.1, radius*0.1);

		std::vector<std::tuple<Point_d, int>> points_query;
		accelerator.search(std::back_inserter(points_query), fs);

		for(int k=0; k<points_query.size(); ++k)
			current_indices.back().push_back(std::get<1>(points_query[k]));

        it++;
	}

	it = slave_nodes_3d[0].begin();
	for(int i=0; i<slave_nodes_3d[0].size(); ++i)
	{
        VectorXa sum_dfdx((additional_dof+num_nodes)*3);
        sum_dfdx.setZero();
        VectorXa sum_dgdx((additional_dof+num_nodes)*3);
        sum_dgdx.setZero();
        std::vector<Entry> sum_d2fdx2;
        std::vector<Entry> sum_d2gdx2;

        //std::cout<<i<<" out of "<<slave_nodes_3d[0].size()<<std::endl;
		AScalar fx = 0;
		AScalar gx = 0;
        Vector3a xi = deformed.segment<3>(3*(it->second+num_nodes-1));

        VectorXa ele_dfdx; 
    	VectorXa ele_dgdx;
        VectorXa ele_d2fdx; 
    	VectorXa ele_d2gdx;
        //std::cout<<i<<" out of "<<slave_nodes_3d[0].size()<<std::endl;

        Eigen::MatrixXd valence_indices;
        valence_indices.resize(current_indices[i].size(),8);

		for(int k=0; k<current_indices[i].size(); ++k)
		{
            Vector3a ck = deformed.segment<3>(current_indices[i][k]*3);
            //f(i == 1567) std::cout<<k<<" k out of "<<current_indices[i].size()<<std::endl;
            int valence = master_nodes_adjacency[current_indices[i][k]].size();
            Eigen::VectorXi valence_indices;
            VectorXa vs;

            if(valence <= 6)
            {
                ele_dfdx.resize(24);
                ele_dgdx.resize(24);
                ele_d2fdx.resize(24*24);
                ele_d2gdx.resize(24*24);
                vs.resize(18);
                valence_indices.resize(8);
                for(int a=0; a<6; ++a)
                {
                    vs.segment<3>(a*3) = ck;
                    valence_indices(a+2)=(current_indices[i][k]);
                }
            }else
            {
                ele_dfdx.resize(39);
                ele_dgdx.resize(39);
                ele_d2fdx.resize(39*39);
                ele_d2gdx.resize(39*39);
                vs.resize(33);
                valence_indices.resize(13);
                for(int a=0; a<11; ++a)
                {
                    vs.segment<3>(a*3) = ck;
                    valence_indices(a+2)=(current_indices[i][k]);
                }
            }
           

            valence_indices(0)=(it->second+num_nodes-1);
            valence_indices(1)=(current_indices[i][k]);

           
            for(int a=0; a<master_nodes_adjacency[current_indices[i][k]].size(); ++a)
            {
                vs.segment<3>(a*3) = deformed.segment<3>(master_nodes_adjacency[current_indices[i][k]][a]*3);
                valence_indices(a+2) = master_nodes_adjacency[current_indices[i][k]][a];
            }
            
            
            if((ck-xi).norm()<=radius) {

                if(valence <= 6)
                {
                    ele_d2fdx = d2fdx_func(xi, vs, ck, radius);
                    ele_d2gdx = d2gdx_func(xi, ck, radius);
                    ele_dfdx = dfdx_func(xi, vs, ck, radius);
                    ele_dgdx = dgdx_func(xi, ck, radius);
                    fx += fx_func(xi, vs, ck, radius);
                    gx += gx_func(xi, ck, radius);
                }else
                {
                    ele_d2fdx = d2fdx_func11(xi, vs, ck, radius);
                    ele_d2gdx = d2gdx_func11(xi, ck, radius);
                    ele_dfdx = dfdx_func11(xi, vs, ck, radius);
                    ele_dgdx = dgdx_func11(xi, ck, radius);
                    fx += fx_func11(xi, vs, ck, radius);
                    gx += gx_func11(xi, ck, radius);
                }
                

                IMLS_local_gradient_to_global_gradient(ele_dfdx,valence_indices,dim,sum_dfdx);
                IMLS_local_gradient_to_global_gradient(ele_dgdx,valence_indices,dim,sum_dgdx);
                IMLS_local_hessian_to_global_triplets(ele_d2fdx,valence_indices,dim,sum_d2fdx2);
                IMLS_local_hessian_to_global_triplets(ele_d2gdx,valence_indices,dim,sum_d2gdx2);
            }
		}
        
		AScalar dist = fx/gx;
        //if(abs(gx) > 1e-6) close_slave_nodes.push_back(it->first);
        if(dist< 0 && abs(gx)>1e-6)
        {
            
            // for(int k=0; k<current_indices[i].size(); ++k)
		    // {
            //     Vector1a A,B;
            //     A(0) = fx; B(0) = gx;
            //     Matrix24a local_hess = -IMLS_param*glue_hessian(A,B,ele_dfdx[k],ele_dgdx[k],ele_d2fdx[k],ele_d2gdx[k]);
            //     // IMLS_local_hessian_to_global_triplets(
            //     // local_hess,valence_indices[k], dim, entries);

            //     for (int a = 0; a < 8; a++) {
            //         for (int b = 0; b < 8; b++) {
            //             for (int c = 0; c < dim; c++) {
            //                 for (int l = 0; l < dim; l++) {
            //                     entries.emplace_back(
            //                         dim * valence_indices(k,a)+ c, dim * valence_indices(k,b) + l,
            //                         local_hess(dim * a + c, dim * b + l));
            //                 }
            //             }
            //         }
            //     }
            //     //std::cout<<" \n";
            // }
            // Eigen::VectorXd gradient((additional_dof+num_nodes)*3); 
            // gradient.setZero();
            // for(int k=0; k<current_indices[i].size(); ++k)
		    // {
            //     // for(int a=0; a<8; ++a)
            //     //     std::cout<<valence_indices[k][a]<<" ";
            //     // std::cout<<"\n";
            //    // std::vector<int> temp = valence_indices[k];
            //     Vector1a A,B;
            //     A(0) = fx; B(0) = gx;
            //     Vector24a local_grad = glue_gradient(A,B,ele_dfdx[k],ele_dgdx[k]);
            //     //std::cout<<((gx * ele_dfdx[k] - fx * ele_dgdx[k]) / (gx * gx)-glue_gradient(A,B,ele_dfdx[k],ele_dgdx[k])).norm()<<std::endl;
            //     // IMLS_local_gradient_to_global_gradient(
            //     // local_grad,temp, dim, gradient);
            //     for (int a = 0; a < 8; a++) 
            //     {   
            //         //std::cout<<valence_indices[k][a]<<" ";
            //         gradient.segment(dim * valence_indices(k,a), dim) += local_grad.segment(dim * a, dim);
            //     }
            //     //std::cout<<"\n";
                
            // }

            // Compute d dist/ dx
            Eigen::VectorXd ddist_dx = (gx*sum_dfdx-fx*sum_dgdx)/(pow(gx,2));
            //std::cout<<"Norm: "<<(3*pow(fx/gx,2)*ddist_dx-gradient).norm()<<std::endl;

            // StiffnessMatrix Hessian((additional_dof+num_nodes)*3,(additional_dof+num_nodes)*3);
            std::vector<Eigen::Triplet<double>> Hessian_t;

            //Part 1 6(f/g) (ddistdx)(ddistdx)^T
            IMLS_vector_muliplication_to_triplets(ddist_dx,ddist_dx,-IMLS_param*6*(fx/gx),Hessian_t);
            entries.insert(entries.end(), Hessian_t.begin(), Hessian_t.end());

            //Part 2 3(f/g)^2 d2distdx2
            // std::vector<Entry> contact_entries = entriesFromSparseMatrix(hess.block(0, 0, num_nodes * dim , num_nodes * dim));
            // entries.insert(entries.end(), contact_entries.begin(), contact_entries.end());
            // A.setFromTriplets(coefficients.begin(), coefficients.end());

            // h1 = d2f/dx2
            StiffnessMatrix h1((additional_dof+num_nodes)*3,(additional_dof+num_nodes)*3);
            h1.setFromTriplets(sum_d2fdx2.begin(), sum_d2fdx2.end());

            // h2 = d2g/dx2
            StiffnessMatrix h2((additional_dof+num_nodes)*3,(additional_dof+num_nodes)*3);
            h2.setFromTriplets(sum_d2gdx2.begin(), sum_d2gdx2.end());

            //h3 = dfdx*dgdx^T
            std::vector<Eigen::Triplet<double>> h3_t;
            IMLS_vector_muliplication_to_triplets(sum_dfdx,sum_dgdx,1.,h3_t);
            StiffnessMatrix h3((additional_dof+num_nodes)*3,(additional_dof+num_nodes)*3);
            h3.setFromTriplets(h3_t.begin(), h3_t.end());

            //h4 = dgdx*dgdx^T
            std::vector<Eigen::Triplet<double>> h4_t;
            IMLS_vector_muliplication_to_triplets(sum_dgdx,sum_dgdx,1.,h4_t);
            StiffnessMatrix h4((additional_dof+num_nodes)*3,(additional_dof+num_nodes)*3);
            h4.setFromTriplets(h4_t.begin(), h4_t.end());

            //h6 = dgdx*dfdx^T
            std::vector<Eigen::Triplet<double>> h6_t;
            IMLS_vector_muliplication_to_triplets(sum_dgdx,sum_dfdx,1.,h6_t);
            StiffnessMatrix h6((additional_dof+num_nodes)*3,(additional_dof+num_nodes)*3);
            h6.setFromTriplets(h6_t.begin(), h6_t.end());

            //h5 = 3(f/g)^2 1/g h1 -2/g^2 h3+2f/g^3 h4 -f/g^2 h2
            StiffnessMatrix h5((additional_dof+num_nodes)*3,(additional_dof+num_nodes)*3);
            h5 = -IMLS_param*(1./gx*h1 - 1./pow(gx,2)*h3 -1./pow(gx,2)*h6 + 2*fx/pow(gx,3)*h4-fx/pow(gx,2)*h2)*3*pow(fx/gx,2);
            std::vector<Entry> h5e = entriesFromSparseMatrix(h5.block(0, 0, (additional_dof+num_nodes) * dim , (additional_dof+num_nodes) * dim));
            entries.insert(entries.end(), h5e.begin(), h5e.end());
		}
        it++;
        
	}
}

template <int dim>
void FEMSolver<dim>::FASTIMLSTestHessian()
{
    VectorXT x(45);
    x.setRandom();

    x *= 1.0 / x.norm();

    Vector3a xi = x.segment<3>(0);
    Vector3a ck = x.segment<3>(3);
    Vector3a ck2 = x.segment<3>(6);
    VectorXa vs = x.segment<18>(9);
    VectorXa vs2 = x.segment<18>(27);


    VectorXa ele_d2fdx = d2fdx_func(xi, vs, ck, radius);
    VectorXa ele_d2gdx = d2gdx_func(xi, ck, radius);
    Vector24a ele_dfdx = dfdx_func(xi, vs, ck, radius);
    Vector24a ele_dgdx = dgdx_func(xi, ck, radius);
    VectorXa ele_d2fdx2 = d2fdx_func(xi, vs2, ck2, radius);
    VectorXa ele_d2gdx2 = d2gdx_func(xi, ck2, radius);
    Vector24a ele_dfdx2 = dfdx_func(xi, vs2, ck2, radius);
    Vector24a ele_dgdx2 = dgdx_func(xi, ck2, radius);
    AScalar fx = 0; 
    fx += fx_func(xi, vs, ck, radius);
    fx += fx_func(xi, vs2, ck2, radius);
    AScalar gx = 0; 
    gx += gx_func(xi, ck, radius);
    gx += gx_func(xi, ck2, radius);

    Vector1a A,B;
    A(0) = fx; B(0) = gx;
    AScalar dist = pow(fx/gx,3);
    Matrix24a local_hess = glue_hessian(A,B,ele_dfdx,ele_dgdx,ele_d2fdx,ele_d2gdx);
    Vector24a local_grad = glue_gradient(A,B,ele_dfdx,ele_dgdx);
    Matrix24a local_hess2 = glue_hessian(A,B,ele_dfdx2,ele_dgdx2,ele_d2fdx2,ele_d2gdx2);
    Vector24a local_grad2 = glue_gradient(A,B,ele_dfdx2,ele_dgdx2);
    std::vector<int> s1 = {0,1,3,4,5,6,7,8};
    std::vector<int> s2 = {0,2,9,10,11,12,13,14};

    MatrixXa hess(45,45);
    hess.setZero();
    VectorXa grad(45);
    grad.setZero();
    for(int i=0; i<8; ++i)
    {
        for(int j=0; j<8; ++j)
        {
            for(int k=0; k<3; ++k)
            {
                for(int l=0; l<3; ++l)
                {
                    hess(3*s1[i]+k,3*s1[j]+l) += local_hess(3*i+k,3*j+l);
                    hess(3*s2[i]+k,3*s2[j]+l) += local_hess2(3*i+k,3*j+l);
                }
            }
        }
    }

    for(int i=0; i<8; ++i)
    {
        for(int k = 0; k<3; ++k)
        {
            grad(3*s1[i]+k)+=local_grad(3*i+k);
            grad(3*s2[i]+k)+=local_grad2(3*i+k);
        }
    }

    AScalar eps = 1e-5;
    for(int a=0; a<45; ++a)
    {
        x(a) += eps;
        Vector3a xi = x.segment<3>(0);
        Vector3a ck = x.segment<3>(3);
        Vector3a ck2 = x.segment<3>(6);
        VectorXa vs = x.segment<18>(9);
        VectorXa vs2 = x.segment<18>(27);


        VectorXa ele_d2fdx = d2fdx_func(xi, vs, ck, radius);
        VectorXa ele_d2gdx = d2gdx_func(xi, ck, radius);
        Vector24a ele_dfdx = dfdx_func(xi, vs, ck, radius);
        Vector24a ele_dgdx = dgdx_func(xi, ck, radius);
        VectorXa ele_d2fdx2 = d2fdx_func(xi, vs2, ck2, radius);
        VectorXa ele_d2gdx2 = d2gdx_func(xi, ck2, radius);
        Vector24a ele_dfdx2 = dfdx_func(xi, vs2, ck2, radius);
        Vector24a ele_dgdx2 = dgdx_func(xi, ck2, radius);
        AScalar fx = 0; 
        fx += fx_func(xi, vs, ck, radius);
        fx += fx_func(xi, vs2, ck2, radius);
        AScalar gx = 0; 
        gx += gx_func(xi, ck, radius);
        gx += gx_func(xi, ck2, radius);

        Vector1a A,B;
        A(0) = fx; B(0) = gx;
        Matrix24a local_hess = glue_hessian(A,B,ele_dfdx,ele_dgdx,ele_d2fdx,ele_d2gdx);
        Vector24a local_grad = glue_gradient(A,B,ele_dfdx,ele_dgdx);
        Matrix24a local_hess2 = glue_hessian(A,B,ele_dfdx2,ele_dgdx2,ele_d2fdx2,ele_d2gdx2);
        Vector24a local_grad2 = glue_gradient(A,B,ele_dfdx2,ele_dgdx2);

        AScalar distp = pow(fx/gx,3);

        VectorXa gradp(45);
        gradp.setZero();
        for(int i=0; i<8; ++i)
        {
            for(int k = 0; k<3; ++k)
            {
                gradp(3*s1[i]+k)+=local_grad(3*i+k);
                gradp(3*s2[i]+k)+=local_grad2(3*i+k);
            }
        }
        //std::cout<<a<<" Numerical: "<<(distp-dist)/eps<<" Analytical: "<<grad(a)<<std::endl;
        for(int b=0; b<45; ++b)
        {
            std::cout<<a<<" "<<b<<" Numerical: "<<(gradp(b)-grad(b))/eps<<" Analytical: "<<hess(a,b)<<std::endl;
        }
        x(a) -= eps;
    }
}

template <int dim>
void FEMSolver<dim>::addL2CorrectEnergy(T& energy)
{
    T L2_energy = 0.0;

    auto it = slave_nodes_3d[0].begin();
	for(int i=0; i<slave_nodes_3d[0].size(); ++i)
	{
        bool is_close = false;
        for(int j=0; j<close_slave_nodes.size(); ++j)
        {
           if(close_slave_nodes[j] == it->second)
            {
                is_close = true;
                break;
            }
        }
        if(!is_close)
        {
            deformed.segment<3>(3*(it->second+num_nodes-1)) = deformed.segment<3>(3*(it->first));
            it++;
            continue;
        } 
		Vector3a xi = deformed.segment<3>(3*(it->second+num_nodes-1));
        Vector3a ck = deformed.segment<3>(3*(it->first));
        

        VectorXa vs;
        vs.resize(33);
        for(int a=0; a<11; ++a)
        {
            vs.segment<3>(a*3) = ck;
        }

        for(int a=0; a<slave_nodes_adjacency[it->first-slave_nodes_shift].size(); ++a)
        {
            vs.segment<3>(a*3) = deformed.segment<3>(slave_nodes_adjacency[it->first-slave_nodes_shift][a]*3);
        }

        L2_energy += normalPen(xi, vs, ck, 1);
        std::cout<<" Projected Point: "<<xi.transpose()<<" Query Point: "<<ck.transpose()<<" Energy: "<<normalPen(xi, vs, ck, 1)<<std::endl;
        // std::cout<<"[ ";
        // for(int i=0; i<33; ++i)
        // {
        //     std::cout<<vs[i]<<", ";
        // }
        // std::cout<<"]";

        it++;
	}

    energy += L2_energy;

    // if(!IMLS_BOTH) return;

    // L2_energy = 0;
}

template <int dim>
void FEMSolver<dim>::addL2CorrectForceEntries(VectorXT& residual)
{
    int num_nodes_all = additional_dof+num_nodes;
    Eigen::VectorXd L2_gradient(num_nodes_all*dim);
    L2_gradient.setZero();

    auto it = slave_nodes_3d[0].begin();
	for(int i=0; i<slave_nodes_3d[0].size(); ++i)
	{
        bool is_close = false;
        for(int j=0; j<close_slave_nodes.size(); ++j)
        {
            if(close_slave_nodes[j] == it->second)
            {
                is_close = true;
                break;
            }
            
        }
        if(!is_close)
        {
            deformed.segment<3>(3*(it->second+num_nodes-1)) = deformed.segment<3>(3*(it->first));
            it++;
            continue;
        } 
		Vector3a xi = deformed.segment<3>(3*(it->second+num_nodes-1));
        Vector3a ck = deformed.segment<3>(3*(it->first));
        Eigen::VectorXi valence_indices;

        // std::cout<<it->first<<" "<<it->second<<std::endl;
        // std::cout<<xi.transpose()<<" ";
        // std::cout<<ck.transpose()<<std::endl;

        VectorXa vs;
        vs.resize(33);
        valence_indices.resize(13);
        for(int a=0; a<11; ++a)
        {
            vs.segment<3>(a*3) = ck;
            valence_indices(a+2)=(it->first);
        }
        

        valence_indices(0)=(it->second+num_nodes-1);
        valence_indices(1)=(it->first);

        
        for(int a=0; a<slave_nodes_adjacency[it->first-slave_nodes_shift].size(); ++a)
        {
            vs.segment<3>(a*3) = deformed.segment<3>(slave_nodes_adjacency[it->first-slave_nodes_shift][a]*3);
            valence_indices(a+2) = slave_nodes_adjacency[it->first-slave_nodes_shift][a];
        }

        VectorXa gradient =  normalPenGradient(xi, vs, ck, 1);
        IMLS_local_gradient_to_global_gradient(gradient,valence_indices,dim,L2_gradient);

        it++;
	}
    residual.segment(0, num_nodes_all * dim) += -L2_gradient.segment(0, num_nodes_all * dim);
    //std::cout<<"done!"<<std::endl;
}

template <int dim>
void FEMSolver<dim>::addL2CorrectHessianEntries(std::vector<Entry>& entries,bool project_PD)
{
    int num_nodes_all = (additional_dof+num_nodes);
    StiffnessMatrix L2_hessian(dim*num_nodes_all,dim*num_nodes_all);
    std::vector<Eigen::Triplet<double>> L2_hessian_triplets;
    auto it = slave_nodes_3d[0].begin();
	for(int i=0; i<slave_nodes_3d[0].size(); ++i)
	{
        bool is_close = false;
        for(int j=0; j<close_slave_nodes.size(); ++j)
        {
           if(close_slave_nodes[j] == it->second)
            {
                is_close = true;
                break;
            }
        }
        if(!is_close)
        {
            deformed.segment<3>(3*(it->second+num_nodes-1)) = deformed.segment<3>(3*(it->first));
            for(int j=0; j<3; ++j)
                L2_hessian_triplets.push_back(Eigen::Triplet<double>(3*(it->second+num_nodes-1)+j,3*(it->second+num_nodes-1)+j,1));
            it++;
            continue;
        } 
		Vector3a xi = deformed.segment<3>(3*(it->second+num_nodes-1));
        Vector3a ck = deformed.segment<3>(3*(it->first));
        Eigen::VectorXi valence_indices;

        VectorXa vs;
        vs.resize(33);
        valence_indices.resize(13);
        for(int a=0; a<11; ++a)
        {
            vs.segment<3>(a*3) = ck;
            valence_indices(a+2)=(it->first);
        }
        

        valence_indices(0)=(it->second+num_nodes-1);
        valence_indices(1)=(it->first);

        
        for(int a=0; a<slave_nodes_adjacency[it->first-slave_nodes_shift].size(); ++a)
        {
            vs.segment<3>(a*3) = deformed.segment<3>(slave_nodes_adjacency[it->first-slave_nodes_shift][a]*3);
            valence_indices(a+2) = slave_nodes_adjacency[it->first-slave_nodes_shift][a];
        }

        VectorXa hessian =  normalPenHessian(xi, vs, ck, 1);
        //std::cout<<valence_indices(0)<<hessian[0]<<" "<<hessian[40]<<" "<<hessian[80]<<std::endl;
        IMLS_local_hessian_to_global_triplets(hessian,valence_indices,dim,L2_hessian_triplets);

        it++;
	}

    L2_hessian.setFromTriplets(L2_hessian_triplets.begin(), L2_hessian_triplets.end());
    //std::cout<<L2_hessian<<std::endl;

    std::vector<Entry> L2_entries = entriesFromSparseMatrix(L2_hessian.block(0, 0, num_nodes_all * dim , num_nodes_all * dim));
    //std::cout<<contact_hessian<<std::endl;
    
    entries.insert(entries.end(), L2_entries.begin(), L2_entries.end());
}

template class FEMSolver<2>;
template class FEMSolver<3>;