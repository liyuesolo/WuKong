#pragma once

#include <utility>
#include <iostream>
#include <fstream>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>

#include "VecMatDef.h"

using TV = Vector<double, 2>;
using TV3 = Vector<double, 3>;
using TM = Matrix<double, 2, 2>;
using IV3 = Vector<int, 3>;
using IV = Vector<int, 2>;

using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
using MatrixXT = Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
using MatrixXi = Matrix<int, Eigen::Dynamic, Eigen::Dynamic>;
using VectorXi = Vector<int, Eigen::Dynamic>;
using VectorXf = Vector<float, Eigen::Dynamic>;

struct MetricPoint {
    double theta;
    double a;
};

class Site {
public:
    TV position;
    std::vector<MetricPoint> metric;

public:
    double getMetricDistance(const TV &p) const;
};

class VoronoiMetrics {
public:
    std::vector<Site> sites;

public:

    int getClosestSiteToSelect(const TV &p, double threshold) const;

    int getClosestSiteByMetric(const TV &p) const;

    void createSite(const TV &p);

    void deleteSite(int index);

    void moveSite(int index, const TV &p);

public:

    VoronoiMetrics();

    ~VoronoiMetrics() {}
};
