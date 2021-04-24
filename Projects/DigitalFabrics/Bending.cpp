#include "EoLRodSim.h"

template<class T, int dim>
void EoLRodSim<T, dim>::entryHelperBending(Eigen::Ref<const DOFStack> q_temp, 
    std::vector<Eigen::Triplet<T>>& entry_K, int n0, int n1, int n2, int uv_offset)
{
    
    TV x0 = q_temp.col(n0).template segment<dim>(0);
    TV x1 = q_temp.col(n1).template segment<dim>(0);
    TV x2 = q_temp.col(n2).template segment<dim>(0);

    T u1 = q_temp(uv_offset, n1);
    T u2 = q_temp(uv_offset, n2);
    
    TV d1 = (x1 - x0).normalized(), d2 = (x2 - x0).normalized();
    T theta = std::acos(-d1.dot(d2));


    if(std::abs(theta) > 1e-6)
    {
        
        assert(u1-u2);
        assert(std::sin(theta));
        std::vector<TV> x(3);
        x[0] = x0; x[1] = x1; x[2] = x2;
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
        entry_K.push_back(Eigen::Triplet<T>(n1 * dof + uv_offset, n2 * dof + uv_offset, -J[2*dim][7]));
        entry_K.push_back(Eigen::Triplet<T>(n2 * dof + uv_offset, n1 * dof + uv_offset, -J[7][2*dim]));
    }
    
}


template<class T, int dim>
void EoLRodSim<T, dim>::addBendingK(Eigen::Ref<const DOFStack> q_temp, std::vector<Eigen::Triplet<T>>& entry_K)
{

    iterateYarnCrossingsSerial([&](int middle, int bottom, int top, int left, int right){
        if (left != -1 && right != -1)
            if(!is_end_nodes[middle] && !is_end_nodes[right] && !is_end_nodes[left])
                entryHelperBending(q_temp, entry_K, middle, right, left, dim);
        if (top != -1 && bottom != -1)
            if(!is_end_nodes[middle] && !is_end_nodes[top] && !is_end_nodes[bottom])
                entryHelperBending(q_temp, entry_K, middle, top, bottom, dim+1);
    });
    
    iteratePBCBendingPairs([&](std::vector<int> nodes, int pair_id){
        int yarn_type = pbc_ref[pair_id].first == WARP ? 0 : 1;
        std::vector<Vector<T, dim + 1>> x(nodes.size());
        toMapleVector5Nodes(x, q_temp, nodes, yarn_type);
        T J[15][15];
        memset(J, 0, sizeof(J));
        #include "Maple/YarnBendPBCJ.mcg"
        for(int k = 0; k < nodes.size(); k++)
            for(int l = 0; l < nodes.size(); l++)
                for(int i = 0; i < dim + 1; i++)
                    for (int j = 0; j < dim + 1; j++)
                        {
                            int u_offset = i == dim ? dim + yarn_type : i;
                            int v_offset = j == dim ? dim + yarn_type : j;
    
                            entry_K.push_back(Eigen::Triplet<T>(nodes[k] * dof + u_offset, nodes[l] * dof + v_offset, -J[k*(dim + 1) + i][l * (dim + 1) + j]));
                        }
    });
    
}

template<class T, int dim>
void EoLRodSim<T, dim>::addBendingForceSingleDirection(Eigen::Ref<const DOFStack> q_temp, Eigen::Ref<DOFStack> residual,
        int n0, int n1, int n2, int uv_offset)
{
    
    TV x0 = q_temp.col(n0).template segment<dim>(0);
    TV x1 = q_temp.col(n1).template segment<dim>(0);
    TV x2 = q_temp.col(n2).template segment<dim>(0);
    T u1 = q_temp(dim + uv_offset, n1);
    T u2 = q_temp(dim + uv_offset, n2);
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
        residual.col(n0).template segment<dim>(0) += F.template segment<dim>(0);
        residual.col(n1).template segment<dim>(0) += F.template segment<dim>(dim);
        residual.col(n2).template segment<dim>(0) += F.template segment<dim>(dim*2+1);

        residual(dim + uv_offset, n1) += F[4];                    
        residual(dim + uv_offset, n2) += F[7];
    }    
    
}        

template<class T, int dim>
void EoLRodSim<T, dim>::addBendingForce(Eigen::Ref<const DOFStack> q_temp, Eigen::Ref<DOFStack> residual)
{
    iterateYarnCrossingsSerial([&](int middle, int bottom, int top, int left, int right){
        if (left != -1 && right != -1)
            if(!is_end_nodes[middle] && !is_end_nodes[right] && !is_end_nodes[left])
                addBendingForceSingleDirection(q_temp, residual, middle, right, left, 0);
        if (top != -1 && bottom != -1)
            if(!is_end_nodes[middle] && !is_end_nodes[top] && !is_end_nodes[bottom])
                addBendingForceSingleDirection(q_temp, residual, middle, top, bottom, 1);
    });   

    iteratePBCBendingPairs([&](std::vector<int> nodes, int pair_id){
        int yarn_type = pbc_ref[pair_id].first == WARP ? 0 : 1;
        std::vector<Vector<T, dim + 1>> x(nodes.size());
        toMapleVector5Nodes(x, q_temp, nodes, yarn_type);
        Vector<T, 15> F;
        F.setZero();
        #include "Maple/YarnBendPBCF.mcg"
        
        int cnt = 0;
        for (int node : nodes)
        {
            
            residual.col(node).template segment<dim>(0) += F.template segment<dim>(cnt*(dim+1));
            residual(dim + yarn_type, node) += F[cnt*(dim+1)+dim];
            cnt++;
        }
    });

}

template<class T, int dim>
T EoLRodSim<T, dim>::bendingEnergySingleDirection(Eigen::Ref<const DOFStack> q_temp, int n0, int n1, int n2, int uv_offset)
{
    TV x0 = q_temp.col(n0).template segment<dim>(0);
    TV x1 = q_temp.col(n1).template segment<dim>(0);
    TV x2 = q_temp.col(n2).template segment<dim>(0);
    T u1 = q_temp(dim + uv_offset, n1);
    T u2 = q_temp(dim + uv_offset, n2);
    TV d1 = (x1 - x0).normalized(), d2 = (x2 - x0).normalized();
    
    T theta = std::acos(-d1.dot(d2));
    assert(u1 - u2);
    if(std::abs(theta) > 1e-6)
    {
        std::vector<TV> x(3);
        x[0] = x0;
        x[1] = x1;
        x[2] = x2;
        T V[1];
        #include "Maple/YarnBendV.mcg"
        return V[0];
    }
    
    return 0;
}


template<class T, int dim>
void EoLRodSim<T, dim>::toMapleVector5Nodes(std::vector<Vector<T, dim + 1>>& x, Eigen::Ref<const DOFStack> q_temp,
    std::vector<int>& nodes, int yarn_type)
{
    int cnt = 0;
    for (int node : nodes)
    {
        Vector<T, dim + 1> xu;
        xu.template segment<dim>(0) = q_temp.col(node).template segment<dim>(0);
        xu[dim] = q_temp(dim + yarn_type, node);
        x[cnt++] = xu;
    }
}


template<class T, int dim>
T EoLRodSim<T, dim>::addBendingEnergy(Eigen::Ref<const DOFStack> q_temp)
{
    T energy = 0.0;
    
    VectorXT crossing_energy(n_nodes);
    crossing_energy.setZero();
     
    iterateYarnCrossingsParallel([&](int middle, int bottom, int top, int left, int right){
        if (left != -1 && right != -1)
        {
            if(!is_end_nodes[middle] && !is_end_nodes[right] && !is_end_nodes[left])
                crossing_energy[middle] += bendingEnergySingleDirection(q_temp, middle, right, left, 0);
        }
        
        if (top != -1 && bottom != -1)
            if(!is_end_nodes[middle] && !is_end_nodes[top] && !is_end_nodes[bottom])
                crossing_energy[middle] += bendingEnergySingleDirection(q_temp, middle, top, bottom, 1);
    });
    energy += crossing_energy.sum();
    T before = energy;
    iteratePBCBendingPairs([&](std::vector<int> nodes, int pair_id){
        int yarn_type = pbc_ref[pair_id].first == WARP ? 0 : 1;
        std::vector<Vector<T, dim + 1>> x(nodes.size());
        toMapleVector5Nodes(x, q_temp, nodes, yarn_type);
        T V[1];
        #include "Maple/YarnBendPBCV.mcg"
        energy += V[0];
    });
    // std::cout << energy - before << std::endl;
    return energy;
}

template class EoLRodSim<double, 3>;
template class EoLRodSim<double, 2>;