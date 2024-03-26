#ifndef CONTACT2D_H
#define CONTACT2D_H


#include <utility>
#include <iostream>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <tbb/tbb.h>
#include <unordered_map>
#include <chrono>

#include "VecMatDef.h"

#include "FEMSolver.h"

template <int dim>
struct VectorHash
{
    typedef Vector<int, dim> IV;
    size_t operator()(const IV& a) const{
        std::size_t h = 0;
        for (int d = 0; d < dim; ++d) {
            h ^= std::hash<int>{}(a(d)) + 0x9e3779b9 + (h << 6) + (h >> 2); 
        }
        return h;
    }
};

class Contact2D
{
public:
    using T = double;
    using TV = Vector<double, 3>;
    using TV2 = Vector<double, 2>;
    using TM2 = Matrix<double, 2, 2>;
    using IV = Vector<int, 3>;

    using PointLoops = std::vector<TV2>;
    using IdList = std::vector<int>;
    using Face = Vector<int, 3>;
    using Edge = Vector<int, 2>;
    
    FEMSolver<2> solver;
    std::vector<std::pair<int, int>> boundary_pairs;
    std::vector<int> boundary_indices;

public:
    Contact2D() {}
    ~Contact2D() {}

    void initializeSimulationData();
    void initializeSimulationDataFromSTL();
    //void initializeBoundaryInfo();

};


#endif