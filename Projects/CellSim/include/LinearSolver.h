#ifndef LINEAR_SOLVER_H
#define LINEAR_SOLVER_H


#include <utility>
#include <iostream>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <tbb/tbb.h>
#include <Eigen/PardisoSupport>

#include "VecMatDef.h"

class LinearSolver
{
public:
    using StiffnessMatrix = Eigen::SparseMatrix<T>;
    using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
    using MatrixXT = Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using Entry = Eigen::Triplet<T>;    

    virtual void compute() = 0;
	virtual void solve(const Eigen::VectorXd &b, VectorXT &x) = 0;

    StiffnessMatrix& A;

public:
    LinearSolver(StiffnessMatrix& _A) : A(_A) {}
    ~LinearSolver() {}
};

class PardisoLDLTSolver : public LinearSolver
{
public:
    Eigen::PardisoLDLT<StiffnessMatrix> solver;
    bool use_default = true;
public:
    
    void compute();
	void solve(const Eigen::VectorXd &b, VectorXT &x);

private:
	void setDefaultLDLTPardisoSolverParameters();

public:
    PardisoLDLTSolver(StiffnessMatrix& _A, bool _use_default) : use_default(_use_default), LinearSolver(_A) 
    {
        setDefaultLDLTPardisoSolverParameters();
    }
    PardisoLDLTSolver(StiffnessMatrix& _A) : use_default(true), LinearSolver(_A) {}
    ~PardisoLDLTSolver() {}
};

class PardisoLLTSolver : public LinearSolver
{
public:
    Eigen::PardisoLLT<StiffnessMatrix> solver;
    MatrixXT UV;
    bool woodbury;
public:
    void setWoodburyMatrix(const MatrixXT& _UV) { UV = _UV; woodbury = true; }

    void compute() {}
	void solve(const Eigen::VectorXd &b, VectorXT &x);

public:
    PardisoLLTSolver(StiffnessMatrix& _A) : woodbury(false), LinearSolver(_A) {}
    PardisoLLTSolver(StiffnessMatrix& _A, MatrixXT& _UV) : UV(_UV), woodbury(true), LinearSolver(_A) {}
    ~PardisoLLTSolver() {}
};

class EigenLUSolver : public LinearSolver
{
public:
    Eigen::SparseLU<StiffnessMatrix> solver;
public:
    
    void compute();
	void solve(const Eigen::VectorXd &b, VectorXT &x);


public:
    EigenLUSolver(StiffnessMatrix& _A) : LinearSolver(_A) {}
    ~EigenLUSolver() {}
};

#endif