// #ifndef NEURAL_CONSTITUTIVE_MODEL_H
// #define NEURAL_CONSTITUTIVE_MODEL_H

// #include <torch/torch.h>
// #include <iostream>
// #include <map>
// #include <fstream>
// #include <string>

// #include <utility>
// #include <Eigen/Geometry>
// #include <Eigen/Core>
// #include <Eigen/Sparse>
// #include <Eigen/Dense>
// #include <tbb/tbb.h>
// #include <unordered_set>
// #include <cassert>

// #include "VecMatDef.h"
// using namespace torch;

// void first_layer_initializer(nn::Module& m, int num_input)
// {
//     if ((typeid(m) == typeid(nn::Linear))) {
//         auto p = m.named_parameters(false);
//         auto w = p.find("weight");
//         auto b = p.find("bias");
//         double  k_sqrt_first = sqrt(1.0/num_input);

//         if (w != nullptr) torch::nn::init::uniform_(*w, -1.0/num_input, 1.0/num_input);
//         if (b != nullptr) torch::nn::init::uniform_(*b, -k_sqrt_first, k_sqrt_first);
//     }
// }
// void middle_layer_initializer(nn::Module& m, int num_hidden)
// {
//     if ((typeid(m) == typeid(nn::Linear))) {
//         auto p = m.named_parameters(false);
//         auto w = p.find("weight");
//         auto b = p.find("bias");
//         double k_sqrt_middle = sqrt(1.0/num_hidden);

//         if (w != nullptr) torch::nn::init::uniform_(*w, -sqrt(6 / num_hidden), sqrt(6 / num_hidden));
//         if (b != nullptr) torch::nn::init::uniform_(*b, -k_sqrt_middle, k_sqrt_middle);
//     }
// }  

// void last_layer_initializer(nn::Module& m, int num_hidden, double last_layer_init_scale)
// {
//     if ((typeid(m) == typeid(nn::Linear))) {
//         auto p = m.named_parameters(false);
//         auto w = p.find("weight");
//         auto b = p.find("bias");
//         double k_sqrt_middle = sqrt(1.0/num_hidden);

//         if (w != nullptr) torch::nn::init::uniform_(*w, -sqrt(6 / num_hidden)*last_layer_init_scale, sqrt(6 / num_hidden)*last_layer_init_scale);
//         if (b != nullptr) torch::nn::init::constant_(*b, 0.0);
//     }
// }  
// struct DenseSIRENNetImpl : nn::Module {
//     DenseSIRENNetImpl(int inputDim, int hiddenDim, int outputDim, double last_layer_init_scale, double omega)
//             : lin1(nn::Linear(inputDim, hiddenDim)),
//               lin2(nn::Linear(hiddenDim, hiddenDim)),
//               lin3(nn::Linear(hiddenDim, hiddenDim)),
//               lin4(nn::Linear(hiddenDim, hiddenDim)),
//               lin5(nn::Linear(hiddenDim, hiddenDim)),
//               lin6(nn::Linear(hiddenDim, outputDim)),
//               omega0(omega)
//     {
//         // register_module() is needed if we want to use the parameters() method later on
//         auto m1 = register_module("lin1", lin1);
//         auto m2 = register_module("lin2", lin2);
//         auto m3 = register_module("lin3", lin3);
//         auto m4 = register_module("lin4", lin4);
//         auto m5 = register_module("lin5", lin5);
//         auto m6 = register_module("lin6", lin6);

//         first_layer_initializer(*m1, inputDim);
//         middle_layer_initializer(*m2, hiddenDim);
//         middle_layer_initializer(*m3, hiddenDim);
//         middle_layer_initializer(*m4, hiddenDim);
//         middle_layer_initializer(*m5, hiddenDim);
//         last_layer_initializer(*m6, hiddenDim, last_layer_init_scale);
//     }

//     torch::Tensor forward(torch::Tensor x) {
//         auto x1 = torch::sin(omega0*lin1(x));
//         auto x2 = torch::sin(lin2(x1));
//         auto x3 = torch::sin(lin3(x2));
//         auto x4 = torch::sin(lin2(x3));
//         auto x5 = torch::sin(lin3(x4));
//         auto x6 = torch::sin(lin3(x5));
//         return x6;
//     }

//     nn::Linear lin1, lin2, lin3, lin4, lin5, lin6;
//     double omega0;
// };

// TORCH_MODULE(DenseSIRENNet);

// class NeuralConstitutiveModel
// {
// public:
    

// private:    
    

// public:
//     NeuralConstitutiveModel() {}
//     ~NeuralConstitutiveModel() {}
// };

// #endif