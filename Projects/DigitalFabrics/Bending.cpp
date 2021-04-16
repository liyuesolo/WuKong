#include "EoLRodSim.h"

template<class T, int dim>
void EoLRodSim<T, dim>::entryHelperBending(std::vector<TV>& x, T u1, T u2, 
    std::vector<Eigen::Triplet<T>>& entry_K, int n0, int n1, int n2, int uv_offset)
{
    T J[8][8];
    memset(J, 0, sizeof(J));
    #include "Maple/YarnBendJ.mcg"
    for(int i = 0; i < dim; i++)
    {
        for(int j = 0; j < dim; j++)   
        {
            //Fx0dx0
            entry_K.push_back(Eigen::Triplet<T>(n0 * dof + i, n0 * dof + j, -J[i][j]));
            //Fx0dx1
            entry_K.push_back(Eigen::Triplet<T>(n0 * dof + i, n1 * dof + j, -J[i][dim + j]));
            //Fx0dx2
            entry_K.push_back(Eigen::Triplet<T>(n0 * dof + i, n2 * dof + j, -J[i][2 * dim + 1 + j]));

            //Fx1dx0
            entry_K.push_back(Eigen::Triplet<T>(n1 * dof + i, n0 * dof + j, -J[dim + i][j]));
            //Fx1dx1
            entry_K.push_back(Eigen::Triplet<T>(n1 * dof + i, n1 * dof + j, -J[dim + i][dim + j]));
            //Fx1dx2
            entry_K.push_back(Eigen::Triplet<T>(n1 * dof + i, n2 * dof + j, -J[dim + i][2 * dim + 1 + j]));

            //Fx2dx0
            entry_K.push_back(Eigen::Triplet<T>(n2 * dof + i, n0 * dof + j, -J[2 * dim + 1 + i][j]));
            //Fx2dx1
            entry_K.push_back(Eigen::Triplet<T>(n2 * dof + i, n1 * dof + j, -J[2 * dim + 1 + i][dim + j]));
            //Fx2dx2
            entry_K.push_back(Eigen::Triplet<T>(n2 * dof + i, n2 * dof + j, -J[2 * dim + 1 + i][2 * dim + 1 + j]));
        }
        //Fx0du1
        entry_K.push_back(Eigen::Triplet<T>(n0 * dof + i, n1 * dof + uv_offset, -J[i][2*dim]));
        //Fx0du2
        entry_K.push_back(Eigen::Triplet<T>(n0 * dof + i, n2 * dof + uv_offset, -J[i][7]));

        //Fx1du1
        entry_K.push_back(Eigen::Triplet<T>(n1 * dof + i, n1 * dof + uv_offset, -J[dim + i][2*dim]));
        //Fx1du2
        entry_K.push_back(Eigen::Triplet<T>(n1 * dof + i, n2 * dof + uv_offset, -J[dim + i][7]));

        //Fx2du1
        entry_K.push_back(Eigen::Triplet<T>(n2 * dof + i, n1 * dof + uv_offset, -J[2 * dim + 1 + i][2*dim]));
        //Fx2du2
        entry_K.push_back(Eigen::Triplet<T>(n2 * dof + i, n2 * dof + uv_offset, -J[2 * dim + 1 + i][7]));


        //Fu1dx0
        entry_K.push_back(Eigen::Triplet<T>(n1 * dof + uv_offset, n0 * dof + i, -J[2*dim][i]));
        //Fu1dx1
        entry_K.push_back(Eigen::Triplet<T>(n1 * dof + uv_offset, n1 * dof + i, -J[2*dim][dim + i]));
        //Fu1dx2
        entry_K.push_back(Eigen::Triplet<T>(n1 * dof + uv_offset, n2 * dof + i, -J[2*dim][2 * dim + 1 + i]));

        //Fu2dx0
        entry_K.push_back(Eigen::Triplet<T>(n2 * dof + uv_offset, n0 * dof + i, -J[7][i]));
        //Fu2dx1
        entry_K.push_back(Eigen::Triplet<T>(n2 * dof + uv_offset, n1 * dof + i, -J[7][dim + i]));
        //Fu2dx2
        entry_K.push_back(Eigen::Triplet<T>(n2 * dof + uv_offset, n2 * dof + i, -J[7][2 * dim + 1 + i]));
    }
    
    entry_K.push_back(Eigen::Triplet<T>(n2 * dof + uv_offset, n2 * dof + uv_offset, -J[7][7]));
    entry_K.push_back(Eigen::Triplet<T>(n1 * dof + uv_offset, n1 * dof + uv_offset, -J[2*dim][2*dim]));
    entry_K.push_back(Eigen::Triplet<T>(n2 * dof + uv_offset, n1 * dof + uv_offset, -J[2*dim][7]));
    entry_K.push_back(Eigen::Triplet<T>(n1 * dof + uv_offset, n2 * dof + uv_offset, -J[7][2*dim]));
}


template<class T, int dim>
void EoLRodSim<T, dim>::addBendingK(Eigen::Ref<const DOFStack> q_temp, std::vector<Eigen::Triplet<T>>& entry_K)
{
    iterateYarnCrossingsSerial([&](int middle, int bottom, int top, int left, int right){
        if (left != -1 && right != -1)
        {
            TV x0 = q_temp.col(middle).template segment<dim>(0);
            TV x1 = q_temp.col(right).template segment<dim>(0);
            TV x2 = q_temp.col(left).template segment<dim>(0);

            T u1 = q_temp(dim + 1, right);
            T u2 = q_temp(dim + 1, left);
            T l1 = (x1 - x0).norm(), l2 = (x2 - x0).norm();
            TV d1 = (x1 - x0).normalized(), d2 = (x2 - x0).normalized();
            T theta = std::acos(-d1.dot(d2));
            if(std::abs(theta) > 1e-6)
            {
                std::vector<TV> x(3);
                x[0] = x0; x[1] = x1; x[2] = x2;
                entryHelperBending(x, u1, u2, entry_K, middle, right, left, dim + 1);
            }
        }
        if (top != -1 && bottom != -1)
        {
            TV x0 = q_temp.col(middle).template segment<dim>(0);
            TV x1 = q_temp.col(top).template segment<dim>(0);
            TV x2 = q_temp.col(bottom).template segment<dim>(0);
            T u1 = q_temp(dim, top);
            T u2 = q_temp(dim, bottom);

            T l1 = (x1 - x0).norm(), l2 = (x2 - x0).norm();
            TV d1 = (x1 - x0).normalized(), d2 = (x2 - x0).normalized();
            T theta = std::acos(-d1.dot(d2));   
            
            if(std::abs(theta) > 1e-6)
            {
                std::vector<TV> x(3);
                x[0] = x0; x[1] = x1; x[2] = x2;
                entryHelperBending(x, u1, u2, entry_K, middle, top, bottom, dim);
            }
        }
            
            
    });
}

template<class T, int dim>
void EoLRodSim<T, dim>::addBendingForce(Eigen::Ref<const DOFStack> q_temp, Eigen::Ref<DOFStack> residual)
{
    iterateYarnCrossingsSerial([&](int middle, int bottom, int top, int left, int right){
        if (left != -1 && right != -1)
        {
            TV x0 = q_temp.col(middle).template segment<dim>(0);
            TV x1 = q_temp.col(right).template segment<dim>(0);
            TV x2 = q_temp.col(left).template segment<dim>(0);
            T u1 = q_temp(dim + 1, right);
            T u2 = q_temp(dim + 1, left);
            TV d1 = (x1 - x0).normalized(), d2 = (x2 - x0).normalized();
            
            T theta = std::acos(-d1.dot(d2));

            if(std::abs(theta) > 1e-6)
            {
                std::vector<TV> x(3);
                x[0] = x0;
                x[1] = x1;
                x[2] = x2;
                Vector<T, 8> F;
                #include "Maple/YarnBendF.mcg"
                residual.col(middle).template segment<dim>(0) += F.template segment<dim>(0);
                residual.col(right).template segment<dim>(0) += F.template segment<dim>(dim);
                residual.col(left).template segment<dim>(0) += F.template segment<dim>(dim*2+1);

                residual(dim + 1, right) += F[4];                    
                residual(dim + 1, left) += F[7];
            }    
        }
        if (top != -1 && bottom != -1)
        {
            TV x0 = q_temp.col(middle).template segment<dim>(0);
            TV x1 = q_temp.col(top).template segment<dim>(0);
            TV x2 = q_temp.col(bottom).template segment<dim>(0);
            T u1 = q_temp(dim, top);
            T u2 = q_temp(dim, bottom);
            TV d1 = (x1 - x0).normalized(), d2 = (x2 - x0).normalized();
            
            T theta = std::acos(-d1.dot(d2));   
            
            if(std::abs(theta) > 1e-6)
            {
                std::vector<TV> x(3);
                x[0] = x0;
                x[1] = x1;
                x[2] = x2;
                Vector<T, 8> F;
                #include "Maple/YarnBendF.mcg"
                residual.col(middle).template segment<dim>(0) += F.template segment<dim>(0);
                residual.col(top).template segment<dim>(0) += F.template segment<dim>(dim);
                residual.col(bottom).template segment<dim>(0) += F.template segment<dim>(dim*2+1);

                residual(dim, top) += F[4];                    
                residual(dim, bottom) += F[7];
            }              
        }
    });
}

template<class T, int dim>
T EoLRodSim<T, dim>::addBendingEnergy(Eigen::Ref<const DOFStack> q_temp)
{
    VectorXT crossing_energy(n_nodes);
    crossing_energy.setZero();
    iterateYarnCrossingsParallel([&](int middle, int bottom, int top, int left, int right){
        if (left != -1 && right != -1)
        {
            TV x0 = q_temp.col(middle).template segment<dim>(0);
            TV x1 = q_temp.col(right).template segment<dim>(0);
            TV x2 = q_temp.col(left).template segment<dim>(0);
            T u1 = q_temp(dim + 1, right);
            T u2 = q_temp(dim + 1, left);
            TV d1 = (x1 - x0).normalized(), d2 = (x2 - x0).normalized();
            T theta = std::acos(-d1.dot(d2));
            if(std::abs(theta) > 1e-6)
            {
                std::vector<TV> x(3);
                x[0] = x0;
                x[1] = x1;
                x[2] = x2;
                T V[1];
                #include "Maple/YarnBendV.mcg"
                crossing_energy[middle] += V[0];
            }
        }
        if (top != -1 && bottom != -1)
        {
            TV x0 = q_temp.col(middle).template segment<dim>(0);
            TV x1 = q_temp.col(top).template segment<dim>(0);
            TV x2 = q_temp.col(bottom).template segment<dim>(0);
            T u1 = q_temp(dim, top);
            T u2 = q_temp(dim, bottom);
            TV d1 = (x1 - x0).normalized(), d2 = (x2 - x0).normalized();
            T theta = std::acos(-d1.dot(d2));
            if(std::abs(theta) > 1e-6)
            {
                std::vector<TV> x(3);
                x[0] = x0;
                x[1] = x1;
                x[2] = x2;
                T V[1];
                #include "Maple/YarnBendV.mcg"
                crossing_energy[middle] += V[0];
            }
        }
    });
    return crossing_energy.sum();
}

template class EoLRodSim<double, 3>;
template class EoLRodSim<double, 2>;