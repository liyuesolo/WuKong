//
//  CHOLMODSolver.hpp
//  Noether
//
//  Created by Minchen Li on 6/22/18.
//  Modified by Xuan Li on 1/31/20.
//

#ifndef CHOLMODSolver_hpp
#define CHOLMODSolver_hpp

#include "cholmod.h"
#include <Eigen/Eigen>
#include <vector>
#include <set>

namespace Noether {

template <typename StorageIndex>
class CHOLMODSolver {

protected:
    int numRows;
    cholmod_common cm;
    cholmod_sparse* A;
    cholmod_factor* L;
    cholmod_dense *b, *solution;
    cholmod_dense *x_cd, *y_cd; // for multiply
    void *Ai, *Ap, *Ax, *bx, *solutionx, *x_cdx, *y_cdx;

public:
    CHOLMODSolver(void);
    ~CHOLMODSolver(void);

    void set_pattern(std::vector<StorageIndex>& ptr, std::vector<StorageIndex>& col, std::vector<double>& value); //NOTE: mtr must be SPD

    void set_pattern(Eigen::SparseMatrix<double, Eigen::RowMajor, StorageIndex>& mat);

    void analyze_pattern(void);

    bool factorize(void);

    void solve(double* rhs, double* result, bool spd);

    void outputFactorization(const std::string& filePath);

    void outputMatrix(const std::string& filePath);

    void readMatrix(const std::string& filePath);

    void multiply(const double* x, double* Ax);

    int rows() { return numRows; }
};

} // namespace Noether

#endif /* CHOLMODSolver_hpp */
