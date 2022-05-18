#include "../include/LinearSolver.h"

void EigenLUSolver::compute()
{
    solver.compute(A);
}

void EigenLUSolver::solve(const Eigen::VectorXd &b, VectorXT &x)
{
    compute();
    x = solver.solve(b);
    VectorXT error = A * x - b;
    std::cout << "\t[" << name << "] |Ax-b|/|b|: " << error.norm() / b.norm() << std::endl;
    T search_direction_dot = x.normalized().dot(b.normalized());
    std::cout << "\t[" << name << "] dot(dx, -g) " << search_direction_dot << std::endl;
}


void PardisoLDLTSolver::compute()
{
    solver.analyzePattern(A);
    if (!use_default)
        solver.pardisoParameterArray()[13] = 0;
    solver.factorize(A);
    
    bool refactorize = (solver.pardisoParameterArray()[22] != num_neg_ev);
    refactorize |= (solver.pardisoParameterArray()[21] != num_pos_ev);
    T alpha = 1e-6;
    while (refactorize)
    {
        A.diagonal().segment(0, num_pos_ev).array() += alpha;
        A.diagonal().segment(num_pos_ev, num_neg_ev).array() -= alpha;      
        alpha *= 10.0;
        solver.factorize(A);
        refactorize = (solver.pardisoParameterArray()[22] != num_neg_ev);
        refactorize |= (solver.pardisoParameterArray()[21] != num_pos_ev);
        // int num_pos_ev = solver.pardisoParameterArray()[21];
        std::cout << "\t[" << name << "] # positive eigen values " << solver.pardisoParameterArray()[21] << 
            " #negative eigen values: " << solver.pardisoParameterArray()[22] << std::endl;
    }
    std::cout << "\t[" << name << "] # positive eigen values " << solver.pardisoParameterArray()[21] << 
            " #negative eigen values: " << solver.pardisoParameterArray()[22] << std::endl;
}

void PardisoLDLTSolver::solve(const Eigen::VectorXd &b, VectorXT &x)
{
    compute();
    if(!use_default)
        solver.pardisoParameterArray()[6] = 0;
    T alpha = 1e-6;
    while (true)
    {
        x = solver.solve(b);
        VectorXT error = A * x - b;
        T rel_err = error.norm() / b.norm();
        if (rel_err < 1e-6)
            break;
        A.diagonal().segment(0, num_pos_ev).array() += alpha;
        A.diagonal().segment(num_pos_ev, num_neg_ev).array() -= alpha;      
        alpha *= 10.0;
        solver.factorize(A);
    }
    // x = solver.solve(b);
    // VectorXT error = A * x - b;
    // std::cout << "\t[" << name << "] |Ax-b|/|b|: " << error.norm() / b.norm() << std::endl;
    // T search_direction_dot = x.normalized().dot(b.normalized());
    // std::cout << "\t[" << name << "] dot(dx, -g) " << search_direction_dot << std::endl;
}

void PardisoLUSolver::compute()
{
    solver.analyzePattern(A);
    if (!use_default)
        solver.pardisoParameterArray()[13] = 0;
    bool refactorize = false;
    solver.factorize(A);
}

void PardisoLUSolver::solve(const Eigen::VectorXd &b, VectorXT &x)
{
    compute();
    if(!use_default)
        solver.pardisoParameterArray()[6] = 0;
    x = solver.solve(b);
    VectorXT error = A * x - b;
    std::cout << "\t[" << name << "] |Ax-b|/|b|: " << error.norm() / b.norm() << std::endl;
    T search_direction_dot = x.normalized().dot(b.normalized());
    std::cout << "\t[" << name << "] dot(dx, -g) " << search_direction_dot << std::endl;
}

void PardisoLUSolver::setDefaultPardisoSolverParameters()
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
            std::cout << "A is indefinite" << std::endl;
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