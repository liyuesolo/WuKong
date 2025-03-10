#ifndef TILING3D_H
#define TILING3D_H

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

class Tiling3D
{
public:
    using T = double;
    using TV = Vector<double, 3>;
    using TV2 = Vector<double, 2>;
    using TM2 = Matrix<double, 2, 2>;
    using TM = Matrix<double, 3, 3>;
    using IV = Vector<int, 3>;

    using PointLoops = std::vector<TV2>;
    using IdList = std::vector<int>;
    using Face = Vector<int, 3>;
    using Edge = Vector<int, 2>;
    using EdgeList = std::vector<Edge>;
    using VectorXT = Matrix<double, Eigen::Dynamic, 1>;
    
    FEMSolver& solver;
public:
    Tiling3D(FEMSolver& _solver) : solver(_solver) {}
    ~Tiling3D() {}

    // training
    void generateGreenStrainSecondPKPairsServerToyExample(const std::vector<T>& params,
        const std::string& result_folder, int loading_type, bool generate_mesh = false);

    void generatePeriodicMesh(std::vector<std::vector<TV2>>& polygons, std::vector<TV2>& pbc_corners);
    void fetchUnitCellFromOneFamily(int IH, std::vector<std::vector<TV2>>& eigen_polygons,
        std::vector<TV2>& eigen_base, int n_unit = 1, bool random = false);

    void getPBCUnit(VectorXT& vertices, EdgeList& edge_list);

    void initializeSimulationData(bool tetgen);

    void buildSimulationMeshFromTilingInfo(int IH, T* params,
        Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& C);

    void fetchOneFamily(int IH, T* params, TV2& T1, TV2& T2, 
        PointLoops& raw_points, T width, T height);

    void fetchTilingCropped(int IH, T* params, std::vector<TV2>& valid_points, 
        std::vector<Edge>& edge_pairs,
        T square_width);
    
    void fetchOneFamilyFillRegion(int IH, T* params, 
        std::vector<PointLoops>& raw_points, T width, T height);

    void thickenLinesToSurface(const std::vector<PointLoops>& raw_points, 
        T thickness, std::vector<TV>& mesh_vertices,
        std::vector<Face>& mesh_faces);

    void extrudeToMesh(const std::vector<PointLoops>& raw_points, 
        T width, T height, std::string filename);
    void getMeshForPrinting(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& C);
    void getMeshForPrintingWithLines(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& C);
    
    void test();

    void fetchTilingVtxLoop(std::vector<PointLoops>& raw_points);

    void buildSimulationMesh(const std::vector<PointLoops>& raw_points,
        Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& C);

    void clapBottomLayerWithSquare(
        int IH, T* params, 
        PointLoops& point_loop_unit,
        std::vector<TV2>& valid_points, 
        std::vector<Vector<int, 2>>& edge_pairs,
        T square_width);

    void thickenLines(const std::vector<TV2>& valid_points, 
        const std::vector<Vector<int, 2>>& edge_pairs,
        std::vector<TV>& vertices, std::vector<Face>& faces, T thickness,
        std::vector<IdList>& boundary_indices);

    void extrudeInZ(std::vector<TV>& vertices, std::vector<Face>& faces, 
        T height, std::vector<IdList>& boundary_indices);

    void cropTranslationalUnitByparallelogram(const std::vector<PointLoops>& input_points,
    std::vector<TV2>& output_points, const TV2& top_left, const TV2& top_right,
    const TV2& bottom_right, const TV2& bottom_left, std::vector<Vector<int, 2>>& edge_pairs);
};


#endif