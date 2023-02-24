#pragma once

#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>

#include "Projects/Foam2D/include/VecMatDef.h"
#include <iostream>

using TV = Vector<double, 2>;
using TV3 = Vector<double, 3>;
using TM = Matrix<double, 2, 2>;
using IV3 = Vector<int, 3>;
using IV = Vector<int, 2>;

using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
using MatrixXT = Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
using MatrixXi = Matrix<int, Eigen::Dynamic, Eigen::Dynamic>;
using VectorXi = Vector<int, Eigen::Dynamic>;

struct BoundaryIntersection {
    int i_cell;
    double t_cell;
    int i_bdry;
    double t_bdry;
    int flag;
};

struct BoundaryEdge {
    int nextEdge;

    int btype; // 0 = straight, 1 = arc, 2 = quadratic bezier.
    int q_idx;
};

class Boundary {

public:
    int np;
    int nfree;

    VectorXT p;
    VectorXi free_idx;
    VectorXi free_map;

    VectorXT v;
    MatrixXT dvdp;
    std::vector<MatrixXT> d2vdp2;

    std::vector<BoundaryEdge> edges;

    VectorXT q; // Curve parameter - radius for circular arcs, tangent at start vertex for quadratic bezier.
    MatrixXT dqdp;
    std::vector<MatrixXT> d2qdp2;

    MatrixXT holes;

private:

    virtual void computeVertices() = 0;

    virtual void computeGradient() = 0;

    virtual void computeHessian() = 0;

    bool straightBoundaryIntersection(const TV &p0, const TV &p1, int v_idx, BoundaryIntersection &intersect);

    void
    arcBoundaryIntersection(const TV &p0, const TV &p1, int v_idx, bool &isInt0, BoundaryIntersection &intersect0,
                            bool &isInt1, BoundaryIntersection &intersect1);

    void
    bezierBoundaryIntersection(const TV &p0, const TV &p1, int v_idx, bool &isInt0, BoundaryIntersection &intersect0,
                               bool &isInt1, BoundaryIntersection &intersect1);

public:
    virtual bool checkValid() { return true; };

    virtual double computeEnergy() { return 0; };

    virtual VectorXT computeEnergyGradient() { return VectorXT::Zero(nfree); };

    virtual MatrixXT computeEnergyHessian() { return MatrixXT::Zero(nfree, nfree); };

protected:
    inline void addGradientEntry(int iv, int ip, double value) {
        if (free_map(ip) >= 0) {
            dvdp(iv, free_map(ip)) += value;
        }
    }

    inline void addHessianEntry(int iv, int ip0, int ip1, double value) {
        if (free_map(ip0) >= 0 && free_map(ip1) >= 0) {
            d2vdp2[iv](free_map(ip0), free_map(ip1)) += value;
        }
    }

    inline void addQGradientEntry(int ir, int ip, double value) {
        if (free_map(ip) >= 0) {
            dqdp(ir, free_map(ip)) += value;
        }
    }

    inline void addQHessianEntry(int ir, int ip0, int ip1, double value) {
        if (free_map(ip0) >= 0 && free_map(ip1) >= 0) {
            d2qdp2[ir](free_map(ip0), free_map(ip1)) += value;
        }
    }

public:
    Boundary(const VectorXT &p_, const VectorXi &free_);

    void compute(const VectorXT &p_free);

    VectorXT get_p_free();

    bool pointInBounds(const TV &point);

    bool getCellIntersections(const std::vector<TV> &nodes, std::vector<BoundaryIntersection> &intersections);
};
