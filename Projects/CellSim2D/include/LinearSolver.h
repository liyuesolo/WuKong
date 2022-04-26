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
    int regularization_start_entry;
    int regularization_offset;
    std::string name = "LinearSolver";
    virtual void setRegularizationIndices(int start, int offset) 
    {
        regularization_start_entry = start;
        regularization_offset = offset;
    }

public:
    LinearSolver(StiffnessMatrix& _A) : A(_A), regularization_start_entry(0), regularization_offset(_A.rows())
    {
        
    }
    ~LinearSolver() {}
};

class PardisoLDLTSolver : public LinearSolver
{
public:
    Eigen::PardisoLDLT<StiffnessMatrix> solver;
    bool use_default = true;
    int num_pos_ev, num_neg_ev;
    
public:
    
    void compute();
	void solve(const Eigen::VectorXd &b, VectorXT &x);
    void setPositiveNegativeEigenValueNumber(int _num_pos_ev, int _num_neg_ev)
    {
        num_pos_ev = _num_pos_ev;
        num_neg_ev = _num_neg_ev;
    }
private:
	void setDefaultLDLTPardisoSolverParameters();

public:
    PardisoLDLTSolver(StiffnessMatrix& _A, bool _use_default) : use_default(_use_default), LinearSolver(_A) 
    {
        name = "PardisoLDLT";
        if (!_use_default)
            setDefaultLDLTPardisoSolverParameters();
    }
    PardisoLDLTSolver(StiffnessMatrix& _A) : use_default(true), LinearSolver(_A) {}
    ~PardisoLDLTSolver() {}
};

class PardisoLUSolver : public LinearSolver
{
public:
    
    Eigen::PardisoLU<StiffnessMatrix> solver;
    bool use_default = true;
public:
    
    void compute();
	void solve(const Eigen::VectorXd &b, VectorXT &x);
private:
	void setDefaultPardisoSolverParameters();

public:
    PardisoLUSolver(StiffnessMatrix& _A, bool _use_default) : use_default(_use_default), LinearSolver(_A) 
    {
        name = "PardisoLU";
        if (!_use_default)
            setDefaultPardisoSolverParameters();
    }
    PardisoLUSolver(StiffnessMatrix& _A) : use_default(true), LinearSolver(_A) {}
    ~PardisoLUSolver() {}
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
    EigenLUSolver(StiffnessMatrix& _A) : LinearSolver(_A) 
    {
        name = "EigenLU";
    }
    ~EigenLUSolver() {}
};

#endif