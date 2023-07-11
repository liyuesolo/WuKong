#include <igl/fast_find_self_intersections.h>
#include "../../include/Boundary/MeshSpringBoundary.h"

MeshSpringBoundary::MeshSpringBoundary(MatrixXT &v_, const MatrixXi &f_, const VectorXi &free_) : Boundary(
        Eigen::Map<VectorXT>(MatrixXT(v_.transpose()).data(), v_.size()), free_) {
    initialize(v_.rows(), f_);

    for (int i = 0; i < f.size(); i++) {
        IV3 verts = f[i].vertices;
        for (int j = 0; j < 3; j++) {
            int v0 = verts[j];
            int v1 = verts[(j + 1) % 3];
            if (v0 < v1) {
                double d = (p.segment<3>(v1 * 3) - p.segment<3>(v0 * 3)).norm();
                springs.emplace_back(v0, v1, d);
            }
        }
    }
}

bool MeshSpringBoundary::checkValid() {
    MatrixXT V(v.size(), 3);
    for (int i = 0; i < v.size(); i++) {
        V.row(i) = v[i].pos;
    }
    MatrixXi F(f.size(), 3);
    for (int i = 0; i < f.size(); i++) {
        F.row(i) = f[i].vertices;
    }
    MatrixXi I;

    return !igl::fast_find_self_intersections(V, F, I);
}

void MeshSpringBoundary::computeVertices() {
    for (int i = 0; i < v.size(); i++) {
        v[i].pos = p.segment<3>(i * 3);

        for (int j = 0; j < 3; j++) {
            addGradientEntry(i, j, i * 3 + j, 1.0);
        }
    }
}

double MeshSpringBoundary::computeEnergy() {
    double energy = 0;

    for (BoundaryEdgeSpring spring: springs) {
        double d = (v[spring.v1].pos - v[spring.v0].pos).norm();
        energy += 0.5 * k * pow(d - spring.len, 2.0);
    }

    return energy;
}

VectorXT MeshSpringBoundary::computeEnergyGradient() {
    Eigen::SparseVector<double> gradient(nfree);

    for (BoundaryEdgeSpring spring: springs) {
        int v0 = spring.v0;
        int v1 = spring.v1;
        double len = spring.len;

        int i0 = v0 * 3 + 0;
        int i1 = v0 * 3 + 1;
        int i2 = v0 * 3 + 2;
        int i3 = v1 * 3 + 0;
        int i4 = v1 * 3 + 1;
        int i5 = v1 * 3 + 2;

        double x0 = p(i0);
        double y0 = p(i1);
        double z0 = p(i2);
        double x1 = p(i3);
        double y1 = p(i4);
        double z1 = p(i5);

        double unknown[6];

        double t1 = x1 - x0;
        double t2 = t1 * t1;
        double t3 = y1 - y0;
        double t4 = t3 * t3;
        double t5 = z1 - z0;
        double t6 = t5 * t5;
        double t8 = sqrt(t2 + t4 + t6);
        double t10 = (t8 - len) * k;
        double t11 = 0.1e1 / t8;
        unknown[0] = -t1 * t11 * t10;
        unknown[1] = -t3 * t11 * t10;
        unknown[2] = -t5 * t11 * t10;
        unknown[3] = t1 * t11 * t10;
        unknown[4] = t3 * t11 * t10;
        unknown[5] = t5 * t11 * t10;

        int idx[6] = {i0, i1, i2, i3, i4, i5};
        for (int i = 0; i < 6; i++) {
            addEnergyGradientEntry(gradient, idx[i], unknown[i]);
        }
    }

    return gradient;
}

MatrixXT MeshSpringBoundary::computeEnergyHessian() {
    Eigen::SparseMatrix<double> hessian(nfree, nfree);

    for (BoundaryEdgeSpring spring: springs) {
        int v0 = spring.v0;
        int v1 = spring.v1;
        double len = spring.len;

        int i0 = v0 * 3 + 0;
        int i1 = v0 * 3 + 1;
        int i2 = v0 * 3 + 2;
        int i3 = v1 * 3 + 0;
        int i4 = v1 * 3 + 1;
        int i5 = v1 * 3 + 2;

        double x0 = p(i0);
        double y0 = p(i1);
        double z0 = p(i2);
        double x1 = p(i3);
        double y1 = p(i4);
        double z1 = p(i5);

        double unknown[6][6];

        double t1 = x1 - x0;
        double t2 = t1 * t1;
        double t3 = y1 - y0;
        double t4 = t3 * t3;
        double t5 = z1 - z0;
        double t6 = t5 * t5;
        double t7 = t2 + t4 + t6;
        double t9 = 0.1e1 / t7 * k;
        double t10 = 0.4e1 * t1 * t1;
        double t13 = sqrt(t7);
        double t15 = (t13 - len) * k;
        double t17 = 0.1e1 / t13 / t7;
        double t22 = 0.1e1 / t13 * t15;
        double t23 = t10 * t9 / 0.4e1 - t10 * t17 * t15 / 0.4e1 + t22;
        double t25 = 0.4e1 * t1 * t3 * t9;
        double t26 = -0.2e1 * t1 * t17;
        double t29 = 0.2e1 * t3 * t26 * t15 + t25;
        double t31 = 0.4e1 * t1 * t5 * t9;
        double t34 = 0.2e1 * t5 * t26 * t15 + t31;
        double t41 = -t1 * t1 * t9 - t1 * t26 * t15 / 0.2e1 - t22;
        double t43 = -0.4e1 * t1 * t3 * t9;
        double t46 = -0.2e1 * t3 * t26 * t15 + t43;
        double t48 = -0.4e1 * t1 * t5 * t9;
        double t51 = -0.2e1 * t5 * t26 * t15 + t48;
        double t52 = 0.4e1 * t3 * t3;
        double t58 = t52 * t9 / 0.4e1 - t52 * t17 * t15 / 0.4e1 + t22;
        double t60 = 0.4e1 * t3 * t5 * t9;
        double t61 = -0.2e1 * t3 * t17;
        double t64 = 0.2e1 * t5 * t61 * t15 + t60;
        double t67 = -0.2e1 * t1 * t61 * t15 + t43;
        double t74 = -t3 * t3 * t9 - t3 * t61 * t15 / 0.2e1 - t22;
        double t76 = -0.4e1 * t3 * t5 * t9;
        double t79 = -0.2e1 * t5 * t61 * t15 + t76;
        double t80 = 0.4e1 * t5 * t5;
        double t86 = t80 * t9 / 0.4e1 - t80 * t17 * t15 / 0.4e1 + t22;
        double t87 = -0.2e1 * t5 * t17;
        double t90 = -0.2e1 * t1 * t87 * t15 + t48;
        double t93 = -0.2e1 * t3 * t87 * t15 + t76;
        double t100 = -t5 * t5 * t9 - t5 * t87 * t15 / 0.2e1 - t22;
        double t101 = 0.2e1 * t1 * t17;
        double t104 = -0.2e1 * t3 * t101 * t15 + t25;
        double t107 = -0.2e1 * t5 * t101 * t15 + t31;
        double t111 = -0.4e1 * t5 * t3 * t17 * t15 + t60;
        unknown[0][0] = t23;
        unknown[0][1] = t29 / 0.4e1;
        unknown[0][2] = t34 / 0.4e1;
        unknown[0][3] = t41;
        unknown[0][4] = t46 / 0.4e1;
        unknown[0][5] = t51 / 0.4e1;
        unknown[1][0] = t29 / 0.4e1;
        unknown[1][1] = t58;
        unknown[1][2] = t64 / 0.4e1;
        unknown[1][3] = t67 / 0.4e1;
        unknown[1][4] = t74;
        unknown[1][5] = t79 / 0.4e1;
        unknown[2][0] = t34 / 0.4e1;
        unknown[2][1] = t64 / 0.4e1;
        unknown[2][2] = t86;
        unknown[2][3] = t90 / 0.4e1;
        unknown[2][4] = t93 / 0.4e1;
        unknown[2][5] = t100;
        unknown[3][0] = t41;
        unknown[3][1] = t67 / 0.4e1;
        unknown[3][2] = t90 / 0.4e1;
        unknown[3][3] = t23;
        unknown[3][4] = t104 / 0.4e1;
        unknown[3][5] = t107 / 0.4e1;
        unknown[4][0] = t46 / 0.4e1;
        unknown[4][1] = t74;
        unknown[4][2] = t93 / 0.4e1;
        unknown[4][3] = t104 / 0.4e1;
        unknown[4][4] = t58;
        unknown[4][5] = t111 / 0.4e1;
        unknown[5][0] = t51 / 0.4e1;
        unknown[5][1] = t79 / 0.4e1;
        unknown[5][2] = t100;
        unknown[5][3] = t107 / 0.4e1;
        unknown[5][4] = t111 / 0.4e1;
        unknown[5][5] = t86;

        int idx[6] = {i0, i1, i2, i3, i4, i5};
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                addEnergyHessianEntry(hessian, idx[i], idx[j], unknown[i][j]);
            }
        }
    }

    return hessian;
}

