#ifndef TILING3D_H
#define TILING3D_H

#include "tactile/tiling.hpp"

#include <utility>
#include <iostream>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <tbb/tbb.h>
#include <unordered_map>

#include "VecMatDef.h"


class Tiling3D
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
    
    
public:
    Tiling3D() {}
    ~Tiling3D() {}

    void fetchOneFamily(int IH, T* params, TV2& T1, TV2& T2, 
        PointLoops& raw_points, T width, T height);
    
    void fetchOneFamilyFillRegion(int IH, T* params, 
        std::vector<PointLoops>& raw_points, T width, T height);

    void extrudeToMesh(const std::vector<PointLoops>& raw_points, 
        T width, T height, std::string filename);
    void getMeshForPrinting(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& C);
    void getMeshForPrintingWithLines(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& C);
    
    void test();

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