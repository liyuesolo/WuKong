#include "EoLRodSim.h"
template<class T, int dim>
void EoLRodSim<T, dim>::addStretchingK(Eigen::Ref<const DOFStack> q_temp, std::vector<Eigen::Triplet<T>>& entry_K)
{
    for (int rod_idx = 0; rod_idx < n_rods; rod_idx++)
    {
        int node0 = rods.col(rod_idx)[0];
        int node1 = rods.col(rod_idx)[1];
        TV x0 = q_temp.col(node0).template segment<dim>(0);
        TV x1 = q_temp.col(node1).template segment<dim>(0);
        TV2 u0 = q_temp.col(node0).template segment<2>(dim);
        TV2 u1 = q_temp.col(node1).template segment<2>(dim);
        TV2 delta_u = u1 - u0;

        T l = (x1 - x0).norm();
        TV d = (x1 - x0).normalized();

        TM P = TM::Identity() - d * d.transpose();
        int yarn_type = rods.col(rod_idx)[2];

        int uv_offset = yarn_type == WARP ? 0 : 1;

        TV w = (x1 - x0) / std::abs(delta_u[uv_offset]);
        
        // add streching K here
        {
            TM dfxdx = -1.0 * (ks/l * P - ks / std::abs(delta_u[uv_offset]) * TM::Identity());
            TV dfxdu = -1.0 * (ks * w.norm() / std::abs(delta_u[uv_offset]) * d);
            T dfudu = -1.0 * (-ks * w.squaredNorm() / std::abs(delta_u[uv_offset]));
            TV dfudx = -1.0 * (ks / std::abs(delta_u[uv_offset]) * w);

            for(int i = 0; i < dim; i++)
            {
                //dfx/dx
                for(int j = 0; j < dim; j++)
                {
                    //dfx0/dx0
                    entry_K.push_back(Eigen::Triplet<T>(node0 * dof + i, node0 * dof + j, dfxdx(i, j)));
                    //dfx1/dx1
                    entry_K.push_back(Eigen::Triplet<T>(node1 * dof + i, node1 * dof + j, dfxdx(i, j)));
                    //dfx0/dx1
                    entry_K.push_back(Eigen::Triplet<T>(node0 * dof + i, node1 * dof + j, -dfxdx(i, j)));
                    //dfx1/dx0
                    entry_K.push_back(Eigen::Triplet<T>(node1 * dof + i, node0 * dof + j, -dfxdx(i, j)));
                }
                // dfx1/du1
                entry_K.push_back(Eigen::Triplet<T>(node1 * dof + i, node1 * dof + dim + uv_offset, dfxdu(i)));
                // dfx1/du0
                entry_K.push_back(Eigen::Triplet<T>(node1 * dof + i, node0 * dof + dim + uv_offset, -dfxdu(i)));
                // dfx0/du1
                entry_K.push_back(Eigen::Triplet<T>(node0 * dof + i, node1 * dof + dim + uv_offset, -dfxdu(i)));
                // dfx0/du0
                entry_K.push_back(Eigen::Triplet<T>(node0 * dof + i, node0 * dof + dim + uv_offset, dfxdu(i)));

                // dfu0/dx0
                entry_K.push_back(Eigen::Triplet<T>(node0 * dof + dim + uv_offset, node0 * dof + i, dfudx(i)));
                // dfu1/dx1
                entry_K.push_back(Eigen::Triplet<T>(node1 * dof + dim + uv_offset, node1 * dof + i, dfudx(i)));
                // dfu1/dx0
                entry_K.push_back(Eigen::Triplet<T>(node1 * dof + dim + uv_offset, node0 * dof + i, -dfudx(i)));
                // dfu0/dx1
                entry_K.push_back(Eigen::Triplet<T>(node0 * dof + dim + uv_offset, node1 * dof + i, -dfudx(i)));
            }
            
            //dfu0/du0
            entry_K.push_back(Eigen::Triplet<T>(node0 * dof + dim + uv_offset, node0 * dof + dim + uv_offset, dfudu));
            //dfu1/du1
            entry_K.push_back(Eigen::Triplet<T>(node1 * dof + dim + uv_offset, node1 * dof + dim + uv_offset, dfudu));
            //dfu1/du0
            entry_K.push_back(Eigen::Triplet<T>(node1 * dof + dim + uv_offset, node0 * dof + dim + uv_offset, -dfudu));
            //dfu0/du1
            entry_K.push_back(Eigen::Triplet<T>(node0 * dof + dim + uv_offset, node1 * dof + dim + uv_offset, -dfudu));
        }   
    }
}
template<class T, int dim>
void EoLRodSim<T, dim>::addStretchingForce(Eigen::Ref<const DOFStack> q_temp, Eigen::Ref<DOFStack> residual)
{
    for (int rod_idx = 0; rod_idx < n_rods; rod_idx++)
    {
        int node0 = rods.col(rod_idx)[0];
        int node1 = rods.col(rod_idx)[1];
        TV x0 = q_temp.col(node0).template segment<dim>(0);
        TV x1 = q_temp.col(node1).template segment<dim>(0);
        TV2 u0 = q_temp.col(node0).template segment<2>(dim);
        TV2 u1 = q_temp.col(node1).template segment<2>(dim);
        TV2 delta_u = u1 - u0;
        
        T l = (x1 - x0).norm();
        TV d = (x1 - x0).normalized();

        int yarn_type = rods.col(rod_idx)[2];

        int uv_offset = yarn_type == WARP ? 0 : 1;

        TV w = (x1 - x0) / std::abs(delta_u[uv_offset]);
        //fx
        residual.col(node0).template segment<dim>(0) += ks * (w.norm() - 1.0) * d;
        residual.col(node1).template segment<dim>(0) += -ks * (w.norm() - 1.0) * d;
        //fu
        residual.col(node0)[dim + uv_offset] += -0.5 * ks * (std::pow(w.norm(), 2) - 1.0);
        residual.col(node1)[dim + uv_offset] += 0.5 * ks * (std::pow(w.norm(), 2) - 1.0);
    }
}
template<class T, int dim>
T EoLRodSim<T, dim>::addStretchingEnergy(Eigen::Ref<const DOFStack> q_temp)
{
    VectorXT rod_energy(n_rods);
    rod_energy.setZero();

    tbb::parallel_for(0, n_rods, [&](int rod_idx){
    // for (int rod_idx = 0; rod_idx < n_rods; rod_idx++) {
        int node0 = rods.col(rod_idx)[0];
        int node1 = rods.col(rod_idx)[1];
        TV x0 = q_temp.col(node0).template segment<dim>(0);
        TV x1 = q_temp.col(node1).template segment<dim>(0);
        TV2 u0 = q_temp.col(node0).template segment<2>(dim);
        TV2 u1 = q_temp.col(node1).template segment<2>(dim);
        TV2 delta_u = u1 - u0;

        int yarn_type = rods.col(rod_idx)[2];

        int uv_offset = yarn_type == WARP ? 0 : 1;
    
        // add elastic potential here 1/2 ks delta_u * (||w|| - 1)^2
        TV w = (x1 - x0) / std::abs(delta_u[uv_offset]);
        rod_energy[rod_idx] += 0.5 * ks * std::abs(delta_u[uv_offset]) * std::pow(w.norm() - 1.0, 2);
        
    // }
    });
    return rod_energy.sum();
}

template class EoLRodSim<double, 3>;
template class EoLRodSim<double, 2>;