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

        std::vector<TV> X;
        int yarn_type = rods.col(rod_idx)[2];
        std::vector<int> nodes = { node0, node1 };
        getMaterialPositions(q_temp, nodes, X, yarn_type);

        std::vector<TV> x(4);
        x[0] = x0; x[1] = x1; x[2] = X[0]; x[3] = X[1];

        T J[8][8];
        memset(J, 0, sizeof(J));
        #include "Maple/YarnStretchingJ.mcg"
        for(int k = 0; k < nodes.size(); k++)
            for(int l = 0; l < nodes.size(); l++)
                for(int i = 0; i < dim; i++)
                    for (int j = 0; j < dim; j++)
                        {
                            entry_K.push_back(Eigen::Triplet<T>(nodes[k] * dof + i, nodes[l] * dof + j, -J[k*dim + i][l * dim + j]));
                        }
    }
}
template<class T, int dim>
void EoLRodSim<T, dim>::addStretchingForce(Eigen::Ref<const DOFStack> q_temp, Eigen::Ref<DOFStack> residual)
{
    DOFStack residual_cp = residual;
    for (int rod_idx = 0; rod_idx < n_rods; rod_idx++)
    {
        int node0 = rods.col(rod_idx)[0];
        int node1 = rods.col(rod_idx)[1];
        TV x0 = q_temp.col(node0).template segment<dim>(0);
        TV x1 = q_temp.col(node1).template segment<dim>(0);
        
        std::vector<TV> X;
        int yarn_type = rods.col(rod_idx)[2];
        std::vector<int> nodes = { node0, node1 };
        getMaterialPositions(q_temp, nodes, X, yarn_type);

        std::vector<TV> x(4);
        x[0] = x0; x[1] = x1; x[2] = X[0]; x[3] = X[1];

        Vector<T, 8> F;
        F.setZero();
        #include "Maple/YarnStretchingF.mcg"
        
        int cnt = 0;
        for (int node : nodes)
        {
            residual.col(node).template segment<dim>(0) += F.template segment<dim>(cnt*dim);
            cnt++;
        } 

    }
    if(print_force_mag)
        std::cout << "stretching norm: " << (residual - residual_cp).norm() << std::endl;
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

        std::vector<TV> X;
        int yarn_type = rods.col(rod_idx)[2];
        std::vector<int> nodes = { node0, node1 };
        getMaterialPositions(q_temp, nodes, X, yarn_type);

        std::vector<TV> x(4);
        x[0] = x0; x[1] = x1; x[2] = X[0]; x[3] = X[1];

        T V[1];
        #include "Maple/YarnStretchingV.mcg"
        rod_energy[rod_idx] += V[0];

    });
    return rod_energy.sum();
}

template class EoLRodSim<double, 3>;
template class EoLRodSim<double, 2>;