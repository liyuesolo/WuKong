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

    DOFStack q_temp_hack = q_temp;

    // ****************************** HACK ALERT ****************************** 
    T kb0 = kb;
    iteratePBCBendingPairs([&](int n0, int n1, int n2, int n3, int n4, int direction){
        q_temp_hack.col(n3) -= -pbc_translation[direction+ 100];
        kb = kb * 0.5;
        entryHelperBending(q_temp_hack, entry_K, n0, n1, n3, dim+direction);
        kb = kb0;
        entryHelperBending(q_temp_hack, entry_K, n1, n2, n3, dim+direction);
        q_temp_hack.col(n3) += -pbc_translation[direction+ 100];
        q_temp_hack.col(n1) += -pbc_translation[direction+ 100];
        kb = kb * 0.5;
        entryHelperBending(q_temp_hack, entry_K, n4, n1, n3, dim+direction);
        kb = kb0;
        entryHelperBending(q_temp_hack, entry_K, n3, n1, n2, dim+direction);
        q_temp_hack.col(n1) -= -pbc_translation[direction+ 100];
    });
    
    // std::cout << "K done" << std::endl;
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
    // std::cout << theta << std::endl;
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

    DOFStack q_temp_hack = q_temp;
    T kb0 = kb;
    // ****************************** HACK ALERT ****************************** 
    iteratePBCBendingPairs([&](int n0, int n1, int n2, int n3, int n4, int direction){
        q_temp_hack.col(n3) -= -pbc_translation[direction+ 100];
        kb = kb * 0.5;
        addBendingForceSingleDirection(q_temp_hack, residual, n0, n1, n3, direction);
        kb = kb0;
        addBendingForceSingleDirection(q_temp_hack, residual, n1, n2, n3, direction);
        q_temp_hack.col(n3) += -pbc_translation[direction+ 100];
        q_temp_hack.col(n1) += -pbc_translation[direction+ 100];
        kb = kb * 0.5;
        addBendingForceSingleDirection(q_temp_hack, residual, n4, n1, n3, direction);
        kb = kb0;
        addBendingForceSingleDirection(q_temp_hack, residual, n3, n1, n2, direction);
        q_temp_hack.col(n1) -= -pbc_translation[direction+ 100];
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
T EoLRodSim<T, dim>::addBendingEnergy(Eigen::Ref<const DOFStack> q_temp)
{
    
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
    T before = crossing_energy.sum();
    DOFStack q_temp_hack = q_temp;

    // ****************************** HACK ALERT ****************************** 
    T kb0 = kb;
    iteratePBCBendingPairs([&](int n0, int n1, int n2, int n3, int n4, int direction){
        q_temp_hack.col(n3) -= -pbc_translation[direction+ 100];
        kb = kb * 0.5;
        crossing_energy[n0] += bendingEnergySingleDirection(q_temp_hack, n0, n1, n3, direction);
        kb = kb0;
        crossing_energy[n1] += bendingEnergySingleDirection(q_temp_hack, n1, n2, n3, direction);
        q_temp_hack.col(n3) += -pbc_translation[direction+ 100];
        q_temp_hack.col(n1) += -pbc_translation[direction+ 100];
        kb = kb * 0.5;
        crossing_energy[n4] += bendingEnergySingleDirection(q_temp_hack, n4, n1, n3, direction);
        kb = kb0;
        crossing_energy[n3] += bendingEnergySingleDirection(q_temp_hack, n3, n1, n2, direction);
        q_temp_hack.col(n1) -= -pbc_translation[direction+ 100];
    });
    
    // std::cout << "middle rod: " << crossing_energy.sum() - before << std::endl;
    // std::cout << crossing_energy.sum() << std::endl;
    // std::cout << "crossing_energy: " << crossing_energy << std::endl;
    return crossing_energy.sum();
}

template class EoLRodSim<double, 3>;
template class EoLRodSim<double, 2>;