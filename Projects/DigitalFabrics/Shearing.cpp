#include "EoLRodSim.h"

template<class T, int dim>
void EoLRodSim<T, dim>::addShearingK(Eigen::Ref<const DOFStack> q_temp, std::vector<Eigen::Triplet<T>>& entry_K, bool top_right)
{
    iterateYarnCrossingsSerial([&](int middle, int bottom, int top, int left, int right){
        if (left == -1 || bottom == -1 || top == -1 && right == -1)
            return;
        int n1 = top_right ? right : left;
        int n3 = top_right ? top : bottom;

        std::vector<Vector<T, dim>> x(3);
        std::vector<int> nodes = {middle, n1, n3};
        toMapleNodesVector(x, q_temp, nodes);

        T J[6][6];
        memset(J, 0, sizeof(J));
        #include "Maple/YarnShearingJ.mcg"
        
        for(int k = 0; k < nodes.size(); k++)
            for(int l = 0; l < nodes.size(); l++)
                for(int i = 0; i < dim; i++)
                    for (int j = 0; j < dim; j++)
                        {
                            entry_K.push_back(Eigen::Triplet<T>(nodes[k] * dof + i, nodes[l] * dof + j, -J[k*dim + i][l * dim + j]));
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

        std::vector<Vector<T, dim>> x(3);
        std::vector<int> nodes = {middle, n1, n3};
        toMapleNodesVector(x, q_temp, nodes);

        Vector<T, 6> F;
        #include "Maple/YarnShearingF.mcg"
        int cnt = 0;
        for (int node : nodes)
        {
            residual.col(node).template segment<dim>(0) += F.template segment<dim>(cnt*dim);
            cnt++;
        } 
    });
    if (print_force_mag)
        std::cout << "shearing norm: " << (residual - residual_cp).norm() << std::endl;
}

template<class T, int dim>
void EoLRodSim<T, dim>::toMapleNodesVector(std::vector<Vector<T, dim>>& x, Eigen::Ref<const DOFStack> q_temp,
        std::vector<int>& nodes)
{
    int cnt = 0;
    x.resize(nodes.size(), Vector<T, dim>::Zero());
    for (int node : nodes)
    {
        x[cnt++] = q_temp.col(node).template segment<dim>(0);
    }
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
        
        std::vector<Vector<T, dim>> x(3);
        std::vector<int> nodes = {middle, n1, n3};
        toMapleNodesVector(x, q_temp, nodes);

        T V[1];
        #include "Maple/YarnShearingV.mcg"
        crossing_energy[middle] += V[0];
    });
    return crossing_energy.sum();
}

template class EoLRodSim<double, 3>;
template class EoLRodSim<double, 2>;