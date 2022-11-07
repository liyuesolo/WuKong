#include "../include/TorchModel.h"

using namespace torch::indexing;

void TorchModel::load(const std::string& filename)
{
    torch_module = torch::jit::load(filename);
    
    // torch::Device device(torch::kCPU);
    // torch::Device device(torch::kCUDA);
    // torch_module.to(device);
}

void TorchModel::test()
{
    TV3 Green_strain(0.1, 0.04, 0.002);
    TV ti(0.15, 0.6);
    // T psi_i = psi(Green_strain, ti);
    TV3 dPsidE;
    dpsi(Green_strain, ti, dPsidE);
    int n_sample = 10;
    VectorXT Green_strain_batch(3 * n_sample);
    for (int i = 0; i < n_sample; i++)
    {
        Green_strain_batch.segment<3>(i * 3) = Green_strain;
    }
    VectorXT psi_batch(n_sample);
    // psiBatch(Green_strain_batch, psi_batch, ti);
    VectorXT dPsidE_batch;
    dpsiBatch(Green_strain_batch, ti, dPsidE_batch);
}

void TorchModel::dpsiBatch(const VectorXT& Green_strain, const VectorXT& ti, VectorXT& dPsidE)
{
    int n_samples = Green_strain.rows() / 3;
    std::vector<torch::jit::IValue> input;
    torch::Tensor nn_input_tensor;
    int n_tiling_params = ti.rows();

    MatrixXT nn_input(n_samples, 3 + n_tiling_params);
    for (int i = 0; i < n_samples; i++)
    {
        nn_input.row(i).segment(0, n_tiling_params) = ti;
        nn_input.row(i).segment(n_tiling_params, 3) = Green_strain;
    }
    toTorchGPU(nn_input, nn_input_tensor, true);

    input.push_back(nn_input_tensor);
    auto output = torch_module.forward(input);
    torch::Tensor psi_torch = output.toTensor(); 

    int input_dim = n_samples;
    int output_dim = 1;
    auto options = torch::TensorOptions().dtype(torch::kFloat64);
    // torch::Tensor J = torch::zeros({output_dim, input_dim}, options).to(torch::kCUDA);
    torch::Tensor grad_output = torch::ones({input_dim, output_dim}, options).to(torch::kCUDA);
    auto gradient = torch::autograd::grad({psi_torch},
                                        {nn_input_tensor},
                                        /*grad_outputs=*/{grad_output},
                                        /*retain_graph=*/true,
                                        /*create_graph=*/true);
    // torch::Tensor gradient = gradient.to(torch::kCPU); 
    
    // MatrixXT gradient_eigen;
    // toEigen(gradient, gradient_eigen);
    std::cout << gradient << std::endl;
}

void TorchModel::dpsi(const TV3& Green_strain, const VectorXT& ti, TV3& dPsidE)
{
    // std::vector<torch::jit::IValue> input;
    // torch::Tensor nn_input_tensor;
    // int n_tiling_params = ti.rows();

    // MatrixXT nn_input(n_samples, 3 + n_tiling_params);
    // for (int i = 0; i < n_samples; i++)
    // {
    //     nn_input.row(i).segment(0, n_tiling_params) = ti;
    //     nn_input.row(i).segment(n_tiling_params, 3) = Green_strain;
    // }
    // toTorchGPU(nn_input, nn_input_tensor, true);

    // input.push_back(nn_input_tensor);
    // auto output = torch_module.forward(input);
    // torch::Tensor psi_torch = output.toTensor(); 

    // int input_dim = (n_tiling_params + 3) * n_samples;
    // int output_dim = 1;
    // auto options = torch::TensorOptions().dtype(torch::kFloat64);
    // // torch::Tensor J = torch::zeros({output_dim, input_dim}, options).to(torch::kCUDA);
    // torch::Tensor grad_output = torch::ones({output_dim, input_dim}, options).to(torch::kCUDA);
    // auto gradient = torch::autograd::grad({psi_torch},
    //                                     {nn_input_tensor},
    //                                     /*grad_outputs=*/{grad_output},
    //                                     /*retain_graph=*/true,
    //                                     /*create_graph=*/true);

    // for (int i = 0; i < output_dim; i++)
    // {
    //   auto grad_output = torch::zeros({output_dim}, options);
    //   grad_output.index_put_({i}, 1);

    
    //   auto gradient = torch::autograd::grad({psi_torch},
    //                                         {nn_input_tensor},
    //                                         /*grad_outputs=*/{grad_output},
    //                                         /*retain_graph=*/true,
    //                                         /*create_graph=*/true);
    //   auto grad = gradient[0];

    //   J.index({i, Slice(0, input_dim, 1)})=grad;
        
    // }
    // torch::Tensor derivative = J;
    // std::cout << derivative << std::endl;
}

T TorchModel::psi(const TV3& Green_strain, const VectorXT& ti)
{
    std::vector<torch::jit::IValue> input;
    torch::Tensor nn_input_tensor;
    int n_tiling_params = ti.rows();
    VectorXT nn_input(3 + n_tiling_params);
    nn_input.segment(0, n_tiling_params) = ti;
    nn_input.segment(n_tiling_params, 3) = Green_strain;
    toTorchGPU(nn_input, nn_input_tensor, false);
    input.push_back(nn_input_tensor);
    auto output = torch_module.forward(input);
    torch::Tensor psi_torch = output.toTensor(); 
    psi_torch.detach();
    T psi_eigen;
    toEigen(psi_torch, psi_eigen);
    return psi_eigen;
}

void TorchModel::psiBatch(const VectorXT& Green_strain, 
    VectorXT& psi_batch, const VectorXT& ti)
{
    int n_samples = Green_strain.rows() / 3;
    std::vector<torch::jit::IValue> input;
    torch::Tensor nn_input_tensor;
    int n_tiling_params = ti.rows();
    MatrixXT nn_input(n_samples, 3 + n_tiling_params);
    for (int i = 0; i < n_samples; i++)
    {
        nn_input.row(i).segment(0, n_tiling_params) = ti;
        nn_input.row(i).segment(n_tiling_params, 3) = Green_strain;
    }
    toTorchGPU(nn_input, nn_input_tensor, false);
    input.push_back(nn_input_tensor);
    auto output = torch_module.forward(input);
    torch::Tensor psi_torch = output.toTensor().to(torch::kCPU); 
    for (int i = 0; i < psi_torch.dim(); i++)
        std::cout << psi_torch.size(i) << std::endl;
    std::cout << psi_torch << std::endl;
    MatrixXT _psi_eigen;
    toEigen(psi_torch, _psi_eigen);
    psi_torch.detach();
    std::cout << _psi_eigen<< std::endl;
}