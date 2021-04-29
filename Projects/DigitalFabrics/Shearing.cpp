#include "EoLRodSim.h"

template<class T, int dim>
void EoLRodSim<T, dim>::addShearingK(Eigen::Ref<const DOFStack> q_temp, std::vector<Eigen::Triplet<T>>& entry_K, bool top_right)
{
    iterateYarnCrossingsSerial([&](int middle, int bottom, int top, int left, int right){
        if (left == -1 || bottom == -1 || top == -1 && right == -1)
            return;
        int n1 = top_right ? right : left;
        int n3 = top_right ? top : bottom;

        TV x0 = q_temp.col(middle).template segment<dim>(0);
        TV x1 = q_temp.col(n1).template segment<dim>(0);
        TV x3 = q_temp.col(n3).template segment<dim>(0);

        TV d3 = (x3 - x0).normalized();
        TV d1 = (x1 - x0).normalized();

        T l1 = (x1 - x0).norm(), l3 = (x3 - x0).norm();

        T phi = std::acos(d1.dot(d3));

        if(std::abs(phi) > 1e-6)
        {
            TM P1 = TM::Identity() - d1 * d1.transpose();
            TM P3 = TM::Identity() - d3 * d3.transpose();

            TM Fx1dx1 = kx * L / l1 / l1 / std::sin(phi)*((phi - M_PI / 2.0)*(-P1 * d3*d1.transpose() + std::cos(phi) / std::sin(phi) / std::sin(phi)*P1*d3*d3.transpose()*P1 - std::cos(phi)*P1 - d1 * d3.transpose()*P1) - 1 / std::sin(phi)*P1*d3*d3.transpose()*P1);
            TM Fx1dx3 = kx * L / l3 / l1 / std::sin(phi)*((phi - M_PI / 2.0)*(std::cos(phi) / std::sin(phi) / std::sin(phi)*P1*d3*d1.transpose() + P1) - 1 / std::sin(phi)*P1*d3*d1.transpose())*P3;
            TM Fx3dx1 = kx * L / l1 / l3 / std::sin(phi)*((phi - M_PI / 2.0)*(std::cos(phi) / std::sin(phi) / std::sin(phi)*P3*d1*d3.transpose() + P3) - 1 / std::sin(phi)*P3*d1*d3.transpose())*P1;
            TM Fx3dx3 = kx * L / l3 / l3 / std::sin(phi)*((phi - M_PI / 2.0)*(-P3 * d1*d3.transpose() + std::cos(phi) / std::sin(phi) / std::sin(phi)*P3*d1*d1.transpose()*P3 - std::cos(phi)*P3 - d3 * d1.transpose()*P3) - 1 / std::sin(phi)*P3*d1*d1.transpose()*P3);

            TM Fx1dx0 = -(Fx1dx1 + Fx1dx3);
            TM Fx3dx0 = -(Fx3dx1 + Fx3dx3);
            TM Fx0dx1 = -(Fx1dx1 + Fx3dx1);
            TM Fx0dx3 = -(Fx1dx3 + Fx3dx3);
            TM Fx0dx0 = -(Fx1dx0 + Fx3dx0);

            for (int i = 0; i < dim; i++)
            {
                for (int j = 0; j < dim; j++)
                {
                     //Fx0dx0
                    entry_K.push_back(Eigen::Triplet<T>(middle * dof + i, middle * dof + j, -Fx0dx0(i, j)));
                    //Fx0dx1
                    entry_K.push_back(Eigen::Triplet<T>(middle * dof + i, n1 * dof + j, -Fx0dx1(i, j)));
                    //Fx0dx2
                    entry_K.push_back(Eigen::Triplet<T>(middle * dof + i, n3 * dof + j, -Fx0dx3(i, j)));

                    //Fx1dx0
                    entry_K.push_back(Eigen::Triplet<T>(n1 * dof + i, middle * dof + j, -Fx1dx0(i, j)));
                    //Fx1dx1
                    entry_K.push_back(Eigen::Triplet<T>(n1 * dof + i, n1 * dof + j, -Fx1dx1(i, j)));
                    //Fx1dx2
                    entry_K.push_back(Eigen::Triplet<T>(n1 * dof + i, n3 * dof + j, -Fx1dx3(i, j)));

                    //Fx2dx0
                    entry_K.push_back(Eigen::Triplet<T>(n3 * dof + i, middle * dof + j, -Fx3dx0(i, j)));
                    //Fx2dx1
                    entry_K.push_back(Eigen::Triplet<T>(n3 * dof + i, n1 * dof + j, -Fx3dx1(i, j)));
                    //Fx2dx2
                    entry_K.push_back(Eigen::Triplet<T>(n3 * dof + i, n3 * dof + j, -Fx3dx3(i, j)));
                }
            }
        }
    });
}

template<class T, int dim>
void EoLRodSim<T, dim>::addShearingForce(Eigen::Ref<const DOFStack> q_temp, Eigen::Ref<DOFStack> residual, bool top_right)
{
    DOFStack residual_cp = residual;
    iterateYarnCrossingsSerial([&](int middle, int bottom, int top, int left, int right){
        if (left == -1 || bottom == -1 || top == -1 && right == -1)
            return;
        int n1 = top_right ? right : left;
        int n3 = top_right ? top : bottom;

        TV x0 = q_temp.col(middle).template segment<dim>(0);
        TV x1 = q_temp.col(n1).template segment<dim>(0);
        TV x3 = q_temp.col(n3).template segment<dim>(0);

        T l1 = (x1 - x0).norm(), l3 = (x3 - x0).norm();
        TV d3 = (x3 - x0).normalized();
        TV d1 = (x1 - x0).normalized();

        T phi = std::acos(d1.dot(d3));

        if(std::abs(phi) > 1e-6)
        {
            TM P1 = TM::Identity() - d1 * d1.transpose();
            TM P3 = TM::Identity() - d3 * d3.transpose();

            TV Fx1 = kx * L*(phi - M_PI / 2.0) / l1 / std::sin(phi)*P1*d3;
            TV Fx3 = kx * L*(phi - M_PI / 2.0) / l3 / std::sin(phi)*P3*d1;
            TV Fx0 = -(Fx1 + Fx3);

            residual.col(middle).template segment<dim>(0) += Fx0;
            residual.col(n3).template segment<dim>(0) += Fx3;
            residual.col(n1).template segment<dim>(0) += Fx1;   
        }
    });
    if (print_force_mag)
        std::cout << "shearing norm: " << (residual - residual_cp).norm() << std::endl;
}

template<class T, int dim>
T EoLRodSim<T, dim>::addShearingEnergy(Eigen::Ref<const DOFStack> q_temp, bool top_right)
{
    VectorXT crossing_energy(n_nodes);
    crossing_energy.setZero();
    iterateYarnCrossingsParallel([&](int middle, int bottom, int top, int left, int right){
        if (left == -1 || bottom == -1 || top == -1 && right == -1)
            return;

        int n1 = top_right ? right : left;
        int n3 = top_right ? top : bottom;

        TV x0 = q_temp.col(middle).template segment<dim>(0);
        TV x1 = q_temp.col(n1).template segment<dim>(0);
        TV x3 = q_temp.col(n3).template segment<dim>(0);

        TV d3 = (x3 - x0).normalized();
        TV d1 = (x1 - x0).normalized();
        
        T phi = std::acos(d1.dot(d3));

        if(std::abs(phi) > 1e-6)
        {
            crossing_energy[middle] += 0.5 * kx * L * std::pow(phi - M_PI/T(2), 2);
        }
    });
    return crossing_energy.sum();
}

template class EoLRodSim<double, 3>;
template class EoLRodSim<double, 2>;