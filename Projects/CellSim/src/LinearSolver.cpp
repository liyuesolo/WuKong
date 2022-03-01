#include "../include/LinearSolver.h"

void EigenLUSolver::compute()
{
    solver.compute(A);
}

void EigenLUSolver::solve(const Eigen::VectorXd &b, VectorXT &x)
{
    x = solver.solve(b);
}


void PardisoLDLTSolver::compute()
{
    solver.analyzePattern(A);
    if (!use_default)
        solver.pardisoParameterArray()[13] = 0;
    solver.factorize(A);

}

void PardisoLDLTSolver::solve(const Eigen::VectorXd &b, VectorXT &x)
{
    if(!use_default)
        solver.pardisoParameterArray()[6] = 0;
    x = solver.solve(b);
    VectorXT error = A * x - b;
    // std::cout << error.norm() / b.norm() << std::endl;
}

// From Jonas Zehnder
void PardisoLDLTSolver::setDefaultLDLTPardisoSolverParameters()
{
    auto& iparm = solver.pardisoParameterArray();
	iparm.setZero();
	iparm[0] = 1; /// not default values, set everything
	iparm[1] = 2; // ordering
	iparm[7] = 2;
	iparm[9] = 13; //pivot perturbation

	iparm[10] = 2; //scaling diagonals/vectors   this can only be used in conjunction with iparm[12] = 1
	iparm[12] = 1;

	iparm[17] = -1;
	iparm[18] = -1;
	iparm[20] = 1; //https://software.intel.com/en-us/mkl-developer-reference-c-pardiso-iparm-parameter

	iparm[23] = 1; // parallel solve
	iparm[24] = 1; // parallel backsubst.
	iparm[26] = 1; // check matrix
	iparm[34] = 1; // 0 based indexing
	//https://github.com/coin-or/Ipopt/blob/c1719e17a4a79bc717b14444a931e67b971df6a8/Ipopt/src/Algorithm/LinearSolvers/IpPardisoSolverInterface.cpp
}

void PardisoLLTSolver::solve(const Eigen::VectorXd &b, VectorXT &x)
{
    T alpha = 10e-6;
    solver.analyzePattern(A);
    int i = 0;
    int indefinite_count_reg_cnt = 0, invalid_search_dir_cnt = 0, invalid_residual_cnt = 0;
    for (; i < 50; i++)
    {
        // std::cout << i << std::endl;

        solver.factorize(A);
        // std::cout << "-----factorization takes " << t.elapsed_sec() << "s----" << std::endl;
        if (solver.info() == Eigen::NumericalIssue)
        {
            // K = H + alpha * I;        
            // tbb::parallel_for(0, (int)K.rows(), [&](int row)    
            // {
            //     K.coeffRef(row, row) += alpha;
            // }); 
            A.diagonal().array() += alpha;
            alpha *= 10;
            indefinite_count_reg_cnt++;
            continue;
        }

        // sherman morrison
        if (UV.cols() == 1)
        {
            VectorXT v = UV.col(0);
            VectorXT A_inv_g = solver.solve(b);
            VectorXT A_inv_u = solver.solve(v);

            T dem = 1.0 + v.dot(A_inv_u);

            x = A_inv_g - (A_inv_g.dot(v)) * A_inv_u / dem;
        }
        // UV is actually only U, since UV is the same in the case
        // C is assume to be Identity
        else // Woodbury https://en.wikipedia.org/wiki/Woodbury_matrix_identity
        {
            VectorXT A_inv_g = solver.solve(b);

            MatrixXT A_inv_U(UV.rows(), UV.cols());
            for (int col = 0; col < UV.cols(); col++)
                A_inv_U.col(col) = solver.solve(UV.col(col));
            
            MatrixXT C(UV.cols(), UV.cols());
            C.setIdentity();
            C += UV.transpose() * A_inv_U;
            x = A_inv_g - A_inv_U * C.inverse() * UV.transpose() * A_inv_g;
        }
    }
}