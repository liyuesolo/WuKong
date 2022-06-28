#ifndef DATA_IO_H
#define DATA_IO_H

#include <utility>
#include <iostream>
#include <fstream>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <tbb/tbb.h>


// https://gist.github.com/zishun/da277d30f4604108029d06db0e804773
template<class Matrix>
inline void write_binary(const std::string& filename, const Matrix& matrix){
    std::ofstream out(filename, std::ios::out | std::ios::binary | std::ios::trunc);
    if(out.is_open()) {
        typename Matrix::Index rows=matrix.rows(), cols=matrix.cols();
        out.write(reinterpret_cast<char*>(&rows), sizeof(typename Matrix::Index));
        out.write(reinterpret_cast<char*>(&cols), sizeof(typename Matrix::Index));
        out.write(reinterpret_cast<const char*>(matrix.data()), rows*cols*static_cast<typename Matrix::Index>(sizeof(typename Matrix::Scalar)) );
        out.close();
    }
    else {
        std::cout << "Can not write to file: " << filename << std::endl;
    }
}

template<class Matrix>
inline void read_binary(const std::string& filename, Matrix& matrix){
    std::ifstream in(filename, std::ios::in | std::ios::binary);
    if (in.is_open()) {
        typename Matrix::Index rows=0, cols=0;
        in.read(reinterpret_cast<char*>(&rows),sizeof(typename Matrix::Index));
        in.read(reinterpret_cast<char*>(&cols),sizeof(typename Matrix::Index));
        matrix.resize(rows, cols);
        in.read(reinterpret_cast<char*>(matrix.data()), rows*cols*static_cast<typename Matrix::Index>(sizeof(typename Matrix::Scalar)) );
        in.close();
    }
    else {
        std::cout << "Can not open binary matrix file: " << filename << std::endl;
    }
}

#include "VecMatDef.h"

struct Nucleus
{
    using VectorXi = Matrix<int, Eigen::Dynamic, 1>;
    int idx, parent_idx;
    int start_frame, end_frame;
    VectorXi positions;
    T score;
};


class DataIO
{
public:
    using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
    using MatrixXT = Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using MatrixXi = Matrix<int, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorXi = Matrix<int, Eigen::Dynamic, 1>;
    using VectorXli = Matrix<long int, Eigen::Dynamic, 1>;

    using TV = Vector<T, 3>;
    using IV = Vector<int, 3>;

    VectorXi time_stamp, positions;
    VectorXli cell_ids, parent_ids;
    VectorXT node_scores, edge_scores;

    std::vector<Nucleus> nuclei;

public:

    void loadDataFromTxt(const std::string& filename);
    void loadDataFromBinary(const std::string& int_data_file, 
        const std::string& long_int_data_file,
        const std::string& float_data_file);
    
    void trackCells();
    void processData();
    void filterWithVelocity();
    void loadTrajectories(const std::string& filename, MatrixXT& trajectories, bool filter = false);
    void runStatsOnTrajectories(const MatrixXT& trajectories, const std::string& filename);
    void loadData(const std::string& filename, VectorXT& data);

    DataIO() {}
    ~DataIO() {}


};

#endif