#pragma once

#include <iostream>
#include <vector>
#include <array>
#include <string>
#include <memory>
#include <stdexcept>
#include <Eigen/Core>
#include <Eigen/Sparse>

using Eigen::Vector2d;
using Eigen::Vector2f;
using Eigen::VectorXd;
using Eigen::Matrix2d;
using Eigen::MatrixXd;
typedef Eigen::SparseMatrix<double> SparseMatrixd;
using Eigen::Triplet;
typedef Eigen::SimplicialLDLT<SparseMatrixd, Eigen::Lower> Solver;
using std::cout;
using std::endl;
using std::string;
using std::vector;
using std::pair;
using std::array;
using std::min;
using std::max;

template<typename T>
void pop_front(std::vector<T> &vec) { vec.erase(vec.begin()); }

template<typename T>
void push_back_pop_front(std::vector<T> &vec, const T &el) {
    vec.push_back(el);
    pop_front(vec);
}

template<typename T>
void concat_in_place(vector<T> &modify_me, const vector<T> &vector2) {
    modify_me.insert(modify_me.end(), vector2.begin(), vector2.end());
}

const auto min_element_wrapper = [](const auto &v) { return *std::min_element(v.begin(), v.end()); };
const auto max_element_wrapper = [](const auto &v) { return *std::max_element(v.begin(), v.end()); };

static void toggle(bool &b) { b = !b; };

template<typename T>
T lerp(const double &f, const T &p, const T &q) { return p + (q - p) * f; };

template<typename ... Args>
string string_format(const std::string &format, Args ... args) {
    // https://stackoverflow.com/questions/2342162/stdstring-formatting-like-sprintf#comment61134428_2342176
    unsigned int size = snprintf(nullptr, 0, format.c_str(), args ...) + 1; // Extra space for '\0'
    std::unique_ptr<char[]> buf(new char[size]);
    snprintf(buf.get(), size, format.c_str(), args ...);
    return string(buf.get(), buf.get() + size - 1); // We don't want the '\0' inside
}

template<typename T>
vector<T> linspace(const int N, const T &left, const T &right) {
    if (N == 1) { return vector<T>({left}); }
    vector<T> X;
    for (int i = 0; i < N; ++i) {
        double f = double(i) / (N - 1);
        T x = lerp<T>(f, left, right);
        X.push_back(x);
    }
    return X;
}

const auto vecFD = [](VectorXd s0, auto O_of_s, double d = 1e-5) -> VectorXd {
    // ~ dOds|s0
    int N = int(s0.size());
    VectorXd dOds;
    dOds.setZero(N);
    for (int i = 0; i < N; ++i) {
        double s0i = s0[i];
        s0[i] -= d;
        double lft = O_of_s(s0);
        s0[i] = s0i;
        s0[i] += d;
        double ryt = O_of_s(s0);
        s0[i] = s0i;
        dOds[i] = (ryt - lft) / (2. * d);
    }
    return dOds;
};

const auto matFD = [](VectorXd s0, auto G_of_s, double d = 1e-5) -> MatrixXd {
    // ~ dGds|s0
    int R = int(G_of_s(s0).size());
    int C = int(s0.size());
    MatrixXd dGds;
    dGds.setZero(R, C);
    for (int j = 0; j < C; ++j) {
        double s0j = s0[j];
        s0[j] -= d;
        auto lft = G_of_s(s0);
        s0[j] = s0j;
        s0[j] += d;
        auto ryt = G_of_s(s0);
        s0[j] = s0j;
        // dGds(i, j) = Dj(Gi) <=> dGds.col(j) = Dj(G)
        auto DjG = (ryt - lft) / (2. * d);
        dGds.col(j) = DjG;
    }
    return dGds;
};

template<class MATType>
void
writeSparseMatrixDenseBlockAdd(SparseMatrixd &hes, int startX, int startY, const MATType &block, bool dropZeroes = true,
                               bool writeOnlyLowerDiagonalValues = false) {
    for (int i = 0; i < block.rows(); i++) {
        for (int j = 0; j < block.cols(); j++) {
            if (startX + i >= startY + j || !writeOnlyLowerDiagonalValues) {
                if (dropZeroes && fabs(block(i, j)) < 1e-10) { continue; }
                hes.coeffRef(startX + i, startY + j) += block(i, j);
            }
        }
    }
}

