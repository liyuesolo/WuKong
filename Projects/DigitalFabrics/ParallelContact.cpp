#include "EoLRodSim.h"

template<class T, int dim>
void EoLRodSim<T, dim>::addParallelContactK(Eigen::Ref<const DOFStack> q_temp, std::vector<Eigen::Triplet<T>>& entry_K)
{
    iterateYarnCrossingsSerial([&](int middle, int bottom, int top, int left, int right){
        if (left != -1 && right != -1 && top != -1 && bottom != -1)
        {
            auto singleDirHessian = [&, middle](int dir)
            {
                T delta_u = std::abs(q_temp(dir, middle) - q0(dir, middle));
                if (delta_u < tunnel_R)
                    return;
                entry_K.push_back(Entry(middle * dof + dir, middle * dof + dir, k_yc));
            };
            singleDirHessian(dim);
            singleDirHessian(dim+1);
        }
    });
}

template<class T, int dim>
void EoLRodSim<T, dim>::addParallelContactForce(Eigen::Ref<const DOFStack> q_temp, Eigen::Ref<DOFStack> residual)
{
    DOFStack residual_cp = residual;
    iterateYarnCrossingsSerial([&](int middle, int bottom, int top, int left, int right){
        if (left != -1 && right != -1 && top != -1 && bottom != -1)
        {
            auto singleDirGrad = [&, middle](int dir)
            {
                T delta_u = q_temp(dir, middle) - q0(dir, middle);
                if (std::abs(delta_u) < tunnel_R)
                    return;
                if (delta_u < 0)
                    residual(dir, middle) += -k_yc * (q_temp(dir, middle) - q0(dir, middle) + tunnel_R);
                else
                    residual(dir, middle) += -k_yc * (q_temp(dir, middle) - q0(dir, middle) - tunnel_R);
            };
            singleDirGrad(dim);
            singleDirGrad(dim + 1);
        }
    });
    if (print_force_mag)
        std::cout << "contact force norm: " << (residual - residual_cp).norm() << std::endl;
}

template<class T, int dim>
T EoLRodSim<T, dim>::addParallelContactEnergy(Eigen::Ref<const DOFStack> q_temp)
{
    T energy = 0.0;
    
    VectorXT crossing_energy(n_nodes);
    crossing_energy.setZero();
     
    iterateYarnCrossingsSerial([&](int middle, int bottom, int top, int left, int right){
        
        if (left != -1 && right != -1 && top != -1 && bottom != -1)
        {
            auto singleDirEnergy = [&, middle](int dir)
            {
                T delta_u = q_temp(dir, middle) - q0(dir, middle);
                std::cout << delta_u << " " << tunnel_R << std::endl;   
                if (std::abs(delta_u) < tunnel_R)
                    return;
                if (delta_u < 0)
                    crossing_energy[middle] += 0.5 * k_yc * std::pow(q_temp(dir, middle) - q0(dir, middle) + tunnel_R, 2);
                else
                    crossing_energy[middle] += 0.5 * k_yc * std::pow(q_temp(dir, middle) - q0(dir, middle) - tunnel_R, 2);
            };
            singleDirEnergy(dim);
            singleDirEnergy(dim+1);
        }
    });
    energy += crossing_energy.sum();
    return energy;
}


template class EoLRodSim<double, 3>;
template class EoLRodSim<double, 2>;