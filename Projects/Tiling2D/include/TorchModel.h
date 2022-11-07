#ifndef TORCH_MODEL_H
#define TORCH_MODEL_H

#include <torch/torch.h>
#include <iostream>
#include <map>
#include <fstream>
#include <string>

#include <torch/script.h>

using namespace torch;
#include "VecMatDef.h"

class TorchModel
{
public:
    using TV3 = Vector<T, 3>;
    using TV = Vector<T, 2>;
    using TM = Matrix<T, 2, 2>;
    using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
    using MatrixXT = Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

private:
    template <int _Rows, int _Cols>
    void toEigen(const torch::Tensor& tensor,
                    Eigen::Matrix<double, _Rows, _Cols>& matrix) {
        if (tensor.dim() == 1) {  //Vector
            
            Eigen::Map<Eigen::Matrix<double, _Rows, _Cols>> map(
                tensor.data_ptr<double>(), tensor.size(0), 1);
            //matrix = map.template cast<double>().transpose();
            matrix = map;
        } else if (tensor.dim() == 2) {  //Matrix
            
            Eigen::Map<Eigen::Matrix<double, _Rows, _Cols>> map(
                tensor.data_ptr<double>(), tensor.size(1), tensor.size(0));
            //matrix = map.template cast<double>().transpose();
            matrix = map.transpose();
        } 
    }


    inline void toEigen(const torch::Tensor& tensor, Eigen::MatrixXd& matrix) {
        toEigen<-1, -1>(tensor, matrix);
    }

    inline void toEigen(const torch::Tensor& tensor, T& scalar) {
        scalar = tensor.item().to<T>();
    }

    inline void toEigen(const torch::Tensor& tensor, VectorXT& vector) {
        toEigen<-1, 1>(tensor, vector);
    }
    
    inline void toEigen(const torch::Tensor& tensor, TV& vector) {
        toEigen<2, 1>(tensor, vector);
    }

    inline void toEigen(const torch::Tensor& tensor, TV3& vector) {
        toEigen<3, 1>(tensor, vector);
    }


    inline void toTorch(const Eigen::MatrixXd& matrix, torch::Tensor& tensor, bool require_grad = true) {
        Eigen::MatrixXd mat = matrix.cast<double>().transpose();
        auto options =
            torch::TensorOptions().requires_grad(require_grad).dtype(torch::kFloat64);
        if (mat.rows() == 1) {  //Vector
            tensor = torch::from_blob(mat.data(), {mat.cols()}, options).clone();
        } else {
            tensor = torch::from_blob(mat.data(), {mat.cols(), mat.rows()}, options)
                        .clone();
        }
    }

    inline void toTorchGPU(const Eigen::MatrixXd& matrix,
                           torch::Tensor& tensor, bool require_grad = true) {
        Eigen::MatrixXd mat = matrix.transpose();
        auto options =
            torch::TensorOptions().requires_grad(require_grad).dtype(torch::kFloat64);
        if (mat.rows() == 1) {  //Vector
            tensor = torch::from_blob(mat.data(), {mat.cols()}, options)
                        .to(torch::kCUDA);
        } else {
            tensor = torch::from_blob(mat.data(), {mat.cols(), mat.rows()}, options)
                        .to(torch::kCUDA);
        }
    }


public:
    torch::jit::script::Module torch_module;
    

    void load(const std::string& filename);

    void test();

    T psi(const TV3& Green_strain, const VectorXT& ti);

    void dpsi(const TV3& Green_strain, const VectorXT& ti, TV3& dPsidE);
    void dpsiBatch(const VectorXT& Green_strain, const VectorXT& ti, VectorXT& dPsidE);

    void psiBatch(const VectorXT& Green_strain, 
        VectorXT& psi_batch, const VectorXT& ti); 
    
public:
    TorchModel() {}
    ~TorchModel() {}
};

#endif