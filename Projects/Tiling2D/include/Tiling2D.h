#ifndef TILING2D_H
#define TILING2D_H

#include "../tactile/tiling.hpp"
#include "../../Libs/clipper/clipper.hpp"

#include <utility>
#include <iostream>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <tbb/tbb.h>
#include <unordered_map>

#include "VecMatDef.h"
#include "Util.h"
#include "FEMSolver.h"
#include <random>
#include <cmath>
#include <fstream>

#include <gmsh.h>

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

class Tiling2D
{
public: 
    using T = double;
    using TV = Vector<double, 2>;
    using TV2 = Vector<double, 2>;
    using TV3 = Vector<double, 3>;
    using TM2 = Matrix<double, 2, 2>;
    using IV3 = Vector<int, 3>;
    using IV = Vector<int, 2>;

    using PointLoops = std::vector<TV2>;
    using IdList = std::vector<int>;
    using Face = Vector<int, 3>;
    using Edge = Vector<int, 2>;
    using EdgeList = std::vector<Edge>;
    using VectorXT = Matrix<double, Eigen::Dynamic, 1>;
    
    FEMSolver& solver;

    std::string data_folder = "/home/yueli/Documents/ETH/WuKong/Projects/Tiling2D/data/";
public:
    Tiling2D(FEMSolver& _solver) : solver(_solver) {}
    ~Tiling2D() {}

    // ########################## Tiling2D.cpp ########################## 
    void initializeSimulationDataFromVTKFile(const std::string& filename);
    void initializeSimulationDataFromFiles(const std::string& filename, bool periodic = false);
    void generateMeshForRendering(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& C);
    void tilingMeshInX(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& C);
    
    void generateForceDisplacementCurve(const std::string& result_folder);

    // ########################## UnitPatch.cpp ########################## 
    // generate periodic mesh
    void generatePeriodicMesh(std::vector<std::vector<TV2>>& polygons, std::vector<TV2>& pbc_corners);
    void fetchUnitCellFromOneFamily(int IH, std::vector<std::vector<TV2>>& eigen_polygons,
        std::vector<TV2>& eigen_base, int n_unit = 1, bool random = false);
    void getPBCUnit(VectorXT& vertices, EdgeList& edge_list);

    
    void generateSandwichMeshPerodicInX(std::vector<std::vector<TV2>>& polygons, std::vector<TV2>& pbc_corners);

    void generateSandwichStructureBatch();
    void fetchSandwichFromOneFamilyFromParamsDilation(int IH, 
        std::vector<T> params,
        std::vector<std::vector<TV2>>& eigen_polygons,
        std::vector<TV2>& eigen_base, bool random = false,
        bool save_to_file = false, std::string filename = "");
};


#endif