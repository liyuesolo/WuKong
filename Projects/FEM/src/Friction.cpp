#include "../include/FEMSolver.h"
#include "../include/VecMatDef.h"

AScalar friction1(const Vector3a& q, const Vector3a& q0, const Vector3a& fn, const AScalar epsilon, const AScalar mu)
{
    AScalar energy;
    AScalar t1 = fn[0];
    AScalar t2 = t1 * t1;
    AScalar t3 = fn[1];
    AScalar t4 = t3 * t3;
    AScalar t5 = fn[2];
    AScalar t6 = t5 * t5;
    AScalar t8 = sqrt(t2 + t4 + t6);
    AScalar t10 = q[0];
    AScalar t11 = q0[0];
    AScalar t12 = 0.1e1 / t8;
    AScalar t17 = q[1];
    AScalar t18 = q0[1];
    AScalar t22 = q[2];
    AScalar t23 = q0[2];
    AScalar t27 = t12 * ((t10 - t11) * t1 * t12 + (t17 - t18) * t3 * t12 + (t22 - t23) * t5 * t12);
    AScalar t30 = pow(-t1 * t27 + t10 - t11, 0.2e1);
    AScalar t33 = pow(-t3 * t27 + t17 - t18, 0.2e1);
    AScalar t36 = pow(-t5 * t27 + t22 - t23, 0.2e1);
    AScalar t37 = t30 + t33 + t36;
    AScalar t38 = t37 * t37;
    AScalar t40 = epsilon * epsilon;
    AScalar t41 = t40 * t40;
    AScalar t45 = sqrt(t37);
    energy = (0.1e1 / epsilon / t41 * t37 * t38 - 0.3e1 / t41 * t45 * t38 + 0.5e1 / 0.2e1 / epsilon / t40 * t38) * t8 * mu;
    return energy;
}

Vector3a frictionForce1(const Vector3a& q, const Vector3a& q0, const Vector3a& fn, const AScalar epsilon, const AScalar mu)
{
    Vector3a gradient;

    AScalar t1 = fn[1] * fn[1];
    AScalar t2 = fn[2] * fn[2];
    AScalar t3 = fn[0] * fn[0];
    AScalar t4 = t1 + t2 + t3;
    AScalar t5 = pow(t4, -0.1e1 / 0.2e1);
    AScalar t6 = (fn[0] * (q[0] - q0[0]) + fn[1] * (q[1] - q0[1]) + fn[2] * (q[2] - q0[2])) * pow(t5, 0.2e1);
    AScalar t7 = -t6 * fn[0] + q[0] - q0[0];
    AScalar t8 = -t6 * fn[1] + q[1] - q0[1];
    t6 = -t6 * fn[2] + q[2] - q0[2];
    AScalar t9 = pow(t6, 0.2e1) + pow(t7, 0.2e1) + pow(t8, 0.2e1);
    AScalar t10 = 0.1e1 / t4;
    AScalar t11 = t8 * fn[1];
    AScalar t12 = t6 * fn[2];
    AScalar t13 = 0.1e1 / epsilon;
    t9 = t13 * (-0.15e2 * pow(t9, 0.3e1 / 0.2e1) + 0.6e1 * pow(t9, 0.2e1) * t13) + 0.10e2 * t9;
    AScalar t14 = pow(t13, 0.2e1);
    t13 = t13 * t14;
    t14 = t7 * fn[0];
    t4 = mu * t4 * t5;
    gradient[0] = t4 * t13 * (t7 * (-t10 * t3 + 0.1e1) - t10 * fn[0] * (t11 + t12)) * t9;
    gradient[1] = t4 * t13 * (t8 * (-t1 * t10 + 0.1e1) - t10 * fn[1] * (t14 + t12)) * t9;
    gradient[2] = t4 * t13 * (t6 * (-t10 * t2 + 0.1e1) - t10 * fn[2] * (t14 + t11)) * t9;

    return gradient;
}

Matrix3a frictionHessian1(const Vector3a& q, const Vector3a& q0, const Vector3a& fn, const AScalar epsilon, const AScalar mu)
{
    AScalar hessian[9];

    AScalar t1 = fn[1] * fn[1];
    AScalar t2 = fn[2] * fn[2];
    AScalar t3 = fn[0] * fn[0];
    AScalar t4 = t1 + t2 + t3;
    AScalar t5 = pow(t4, -0.1e1 / 0.2e1);
    AScalar t6 = (fn[0] * (q[0] - q0[0]) + fn[1] * (q[1] - q0[1]) + fn[2] * (q[2] - q0[2])) * pow(t5, 0.2e1);
    AScalar t7 = -t6 * fn[0] + q[0] - q0[0];
    AScalar t8 = -t6 * fn[1] + q[1] - q0[1];
    t6 = -t6 * fn[2] + q[2] - q0[2];
    AScalar t9 = pow(t6, 0.2e1) + pow(t7, 0.2e1) + pow(t8, 0.2e1);
    AScalar t10 = 0.1e1 / t4;
    AScalar t11 = t10 * t3;
    AScalar t12 = 0.1e1 - t11;
    AScalar t13 = t6 * fn[2];
    AScalar t14 = t8 * fn[1];
    AScalar t15 = t10 * fn[0];
    AScalar t16 = -t12 * t7 + t15 * (t13 + t14);
    AScalar t17 = pow(t10, 0.2e1);
    AScalar t18 = sqrt(t9);
    AScalar t19 = 0.1e1 / epsilon;
    AScalar t20 = 0.6e1 * pow(t9, 0.2e1) * t19;
    AScalar t21 = 0.15e2 * t9 * t18;
    AScalar t22 = 0.10e2 * t9;
    AScalar t23 = t19 * (-t21 + t20) + t22;
    t9 = t19 * (0.24e2 * t9 * t19 - 0.45e2 * t18) + 0.20e2;
    t18 = pow(t19, 0.2e1);
    t18 = t19 * t18;
    AScalar t24 = t10 * t1;
    AScalar t25 = 0.1e1 - t24;
    t7 = t7 * fn[0];
    AScalar t26 = t10 * fn[1];
    t8 = -t25 * t8 + t26 * (t7 + t13);
    t13 = t10 * t2;
    t19 = t19 * (t21 - t20) - t22;
    t20 = t9 * t8;
    t21 = 0.1e1 - t13;
    t6 = -t21 * t6 + t10 * fn[2] * (t7 + t14);
    t4 = mu * t4 * t5;
    t5 = t4 * t18 * (t23 * t26 * fn[2] * (t11 - t25 - t21) + t20 * t6);
    t7 = t4 * t18 * (t15 * fn[1] * (-t13 + t12 + t25) * t19 + t20 * t16);
    t10 = t4 * t18 * (t19 * t15 * fn[2] * (-t24 + t12 + t21) + t9 * t6 * t16);

    hessian[0] = t4 * t18 * (t23 * (pow(t12, 0.2e1) + t17 * t3 * (t1 + t2)) + t9 * pow(t16, 0.2e1));
    hessian[1] = t7;
    hessian[2] = t10;
    hessian[3] = t7;
    hessian[4] = t4 * t18 * (t23 * (pow(t25, 0.2e1) + t17 * t1 * (t2 + t3)) + t9 * pow(t8, 0.2e1));
    hessian[5] = t5;
    hessian[6] = t10;
    hessian[7] = t5;
    hessian[8] = t4 * t18 * (t23 * (pow(t21, 0.2e1) + t17 * t2 * (t1 + t3)) + t9 * pow(t6, 0.2e1));

    return Matrix3a(Eigen::Map<Eigen::Matrix<AScalar,3,3,Eigen::ColMajor> >(hessian));
}

AScalar friction2(const Vector3a& q, const Vector3a& q0, const Vector3a& fn, const AScalar epsilon, const AScalar mu)
{
    AScalar energy;
    AScalar t1 = fn[0];
    AScalar t2 = t1 * t1;
    AScalar t3 = fn[1];
    AScalar t4 = t3 * t3;
    AScalar t5 = fn[2];
    AScalar t6 = t5 * t5;
    AScalar t8 = sqrt(t2 + t4 + t6);
    AScalar t10 = q[0];
    AScalar t11 = q0[0];
    AScalar t12 = 0.1e1 / t8;
    AScalar t17 = q[1];
    AScalar t18 = q0[1];
    AScalar t22 = q[2];
    AScalar t23 = q0[2];
    AScalar t27 = t12 * ((t10 - t11) * t1 * t12 + (t17 - t18) * t3 * t12 + (t22 - t23) * t5 * t12);
    AScalar t30 = pow(-t1 * t27 + t10 - t11, 0.2e1);
    AScalar t33 = pow(-t3 * t27 + t17 - t18, 0.2e1);
    AScalar t36 = pow(-t5 * t27 + t22 - t23, 0.2e1);
    AScalar t38 = sqrt(t30 + t33 + t36);
    energy = (t38 - epsilon / 0.2e1) * t8 * mu;
    return energy;
}

Vector3a frictionForce2(const Vector3a& q, const Vector3a& q0, const Vector3a& fn, const AScalar epsilon, const AScalar mu)
{
    Vector3a gradient;

    AScalar t1 = fn[1] * fn[1];
    AScalar t2 = fn[2] * fn[2];
    AScalar t3 = fn[0] * fn[0];
    AScalar t4 = t1 + t2 + t3;
    AScalar t5 = pow(t4, -0.1e1 / 0.2e1);
    AScalar t6 = (fn[0] * (q[0] - q0[0]) + fn[1] * (q[1] - q0[1]) + fn[2] * (q[2] - q0[2])) * pow(t5, 0.2e1);
    AScalar t7 = -t6 * fn[0] + q[0] - q0[0];
    AScalar t8 = -t6 * fn[1] + q[1] - q0[1];
    t6 = -t6 * fn[2] + q[2] - q0[2];
    AScalar t9 = pow(t6, 0.2e1) + pow(t7, 0.2e1) + pow(t8, 0.2e1);
    AScalar t10 = 0.1e1 / t4;
    AScalar t11 = t8 * fn[1];
    AScalar t12 = t6 * fn[2];
    AScalar t13 = t7 * fn[0];
    t4 = mu * t4 * t5 * pow(t9, -0.1e1 / 0.2e1);
    gradient[0] = t4 * (t7 * (-t10 * t3 + 0.1e1) - t10 * fn[0] * (t11 + t12));
    gradient[1] = t4 * (t8 * (-t1 * t10 + 0.1e1) - t10 * fn[1] * (t12 + t13));
    gradient[2] = t4 * (t6 * (-t10 * t2 + 0.1e1) - t10 * fn[2] * (t11 + t13));

    return gradient;
}

Matrix3a frictionHessian2(const Vector3a& q, const Vector3a& q0, const Vector3a& fn, const AScalar epsilon, const AScalar mu)
{
    AScalar hessian[9];

    AScalar t1 = fn[1] * fn[1];
    AScalar t2 = fn[2] * fn[2];
    AScalar t3 = fn[0] * fn[0];
    AScalar t4 = t1 + t2 + t3;
    AScalar t5 = pow(t4, -0.1e1 / 0.2e1);
    AScalar t6 = (fn[0] * (q[0] - q0[0]) + fn[1] * (q[1] - q0[1]) + fn[2] * (q[2] - q0[2])) * pow(t5, 0.2e1);
    AScalar t7 = -t6 * fn[0] + q[0] - q0[0];
    AScalar t8 = -t6 * fn[1] + q[1] - q0[1];
    t6 = -t6 * fn[2] + q[2] - q0[2];
    AScalar t9 = pow(t6, 0.2e1) + pow(t7, 0.2e1) + pow(t8, 0.2e1);
    AScalar t10 = pow(t9, -0.3e1 / 0.2e1);
    AScalar t11 = 0.1e1 / t4;
    AScalar t12 = t11 * t3;
    AScalar t13 = 0.1e1 - t12;
    AScalar t14 = t6 * fn[2];
    AScalar t15 = t8 * fn[1];
    AScalar t16 = t11 * fn[0];
    AScalar t17 = -t13 * t7 + t16 * (t14 + t15);
    t9 = t9 * t10;
    AScalar t18 = pow(t11, 0.2e1);
    t4 = mu * t4 * t5;
    t5 = t11 * t1;
    AScalar t19 = 0.1e1 - t5;
    t7 = t7 * fn[0];
    AScalar t20 = t11 * fn[1];
    t8 = -t19 * t8 + t20 * (t7 + t14);
    t14 = t11 * t2;
    AScalar t21 = t10 * t17;
    AScalar t22 = t4 * (t9 * t16 * fn[1] * (-t14 + t13 + t19) + t21 * t8);
    t14 = 0.1e1 - t14;
    t6 = -t14 * t6 + t11 * fn[2] * (t7 + t15);
    t5 = t4 * (t9 * t16 * fn[2] * (-t5 + t13 + t14) + t21 * t6);
    t7 = t4 * (t9 * t20 * fn[2] * (t12 - t19 - t14) - t10 * t8 * t6);
    hessian[0] = t4 * (-t10 * pow(t17, 0.2e1) + t9 * (pow(t13, 0.2e1) + t18 * t3 * (t1 + t2)));
    hessian[1] = -t22;
    hessian[2] = -t5;
    hessian[3] = -t22;
    hessian[4] = -t4 * (t10 * pow(t8, 0.2e1) - t9 * (pow(t19, 0.2e1) + t18 * t1 * (t2 + t3)));
    hessian[5] = t7;
    hessian[6] = -t5;
    hessian[7] = t7;
    hessian[8] = -t4 * (t10 * pow(t6, 0.2e1) - t9 * (pow(t14, 0.2e1) + t18 * t2 * (t1 + t3)));

    return Matrix3a(Eigen::Map<Eigen::Matrix<AScalar,3,3,Eigen::ColMajor> >(hessian));
}

AScalar mut(const Vector3a& q, const Vector3a& q0, const Vector3a& fn, const AScalar epsilon, const AScalar mu)
{
    AScalar flen;
    AScalar t1 = q[0];
    AScalar t2 = q0[0];
    AScalar t3 = fn[0];
    AScalar t4 = t3 * t3;
    AScalar t5 = fn[1];
    AScalar t6 = t5 * t5;
    AScalar t7 = fn[2];
    AScalar t8 = t7 * t7;
    AScalar t10 = sqrt(t4 + t6 + t8);
    AScalar t11 = 0.1e1 / t10;
    AScalar t16 = q[1];
    AScalar t17 = q0[1];
    AScalar t21 = q[2];
    AScalar t22 = q0[2];
    AScalar t26 = t11 * ((t1 - t2) * t3 * t11 + (t16 - t17) * t5 * t11 + (t21 - t22) * t7 * t11);
    AScalar t29 = pow(-t3 * t26 + t1 - t2, 0.2e1);
    AScalar t32 = pow(-t5 * t26 + t16 - t17, 0.2e1);
    AScalar t35 = pow(-t7 * t26 + t21 - t22, 0.2e1);
    flen = sqrt(t29 + t32 + t35);
    return flen;
}

template <int dim>
void FEMSolver<dim>::addFrictionEnergy(T& energy)
{
    tbb::enumerable_thread_specific<double> tbb_temp_energies(0);
    tbb::parallel_for(
    tbb::blocked_range<size_t>(size_t(0), num_nodes),
    [&](const tbb::blocked_range<size_t>& r) {
        auto& local_energy_temp = tbb_temp_energies.local();
        for (size_t i = r.begin(); i < r.end(); i++) {
            if(is_surface_vertex[i] != 0)
            {
                Vector3a q = deformed.segment<3>(3*i);
                Vector3a q0 = x_prev.segment<3>(3*i);
                //if(i == 0) std::cout<<" sliding distance: "<< (q-q0).norm()<<std::endl;
                Vector3a fn = prev_contact_force.segment<3>(3*i);
                if(fn.norm() > 0){
                    AScalar flen = mut(q,q0,fn,friction_epsilon,friction_mu);
                    if(flen < friction_epsilon)
                        local_energy_temp += friction1(q,q0,fn,friction_epsilon,friction_mu);
                    else
                        local_energy_temp += friction2(q,q0,fn,friction_epsilon,friction_mu);
                }
                
            }
        }
    });

    for(auto& local_energy_temp: tbb_temp_energies)
    {
        energy += local_energy_temp;
    }
}   

template <int dim>
void FEMSolver<dim>::addFrictionForceEntries(VectorXT& residual)
{
    tbb::enumerable_thread_specific<std::vector<std::pair<int,Vector3a>>> tbb_vector_temps;

    tbb::parallel_for(
    tbb::blocked_range<size_t>(size_t(0), num_nodes),
    [&](const tbb::blocked_range<size_t>& r) {
        auto& local_vector_temp = tbb_vector_temps.local();
        for (size_t i = r.begin(); i < r.end(); i++) {
            if(is_surface_vertex[i] != 0)
            {
                Vector3a q = deformed.segment<3>(3*i);
                Vector3a q0 = x_prev.segment<3>(3*i);
                Vector3a fn = prev_contact_force.segment<3>(3*i);
                if(fn.norm() > 0)
                {
                    AScalar flen = mut(q,q0,fn,friction_epsilon,friction_mu);
                    Vector3a force;
                    if(flen < friction_epsilon)
                        force = frictionForce1(q,q0,fn,friction_epsilon,friction_mu);
                    else
                        force = frictionForce2(q,q0,fn,friction_epsilon,friction_mu);
                    
                    local_vector_temp.push_back(std::pair<int,Vector3a>(i,force));
                }   
            }
        }
    });

    for(auto& local_vector_temp: tbb_vector_temps)
    {
        for(int j=0; j<local_vector_temp.size(); ++j)
        {
            residual.segment<3>(3*local_vector_temp[j].first) -= local_vector_temp[j].second;
        }
    }
}

template <int dim>
void FEMSolver<dim>::addFrictionHessianEntries(std::vector<Entry>& entries,bool project_PD)
{
    tbb::enumerable_thread_specific<std::vector<Entry>> tbb_vector_temps;

    tbb::parallel_for(
    tbb::blocked_range<size_t>(size_t(0), num_nodes),
    [&](const tbb::blocked_range<size_t>& r) {
        auto& local_vector_temp = tbb_vector_temps.local();
        for (size_t i = r.begin(); i < r.end(); i++) {
            if(is_surface_vertex[i] != 0)
            {
                Vector3a q = deformed.segment<3>(3*i);
                Vector3a q0 = x_prev.segment<3>(3*i);
                Vector3a fn = prev_contact_force.segment<3>(3*i);
                
                if(fn.norm() > 0)
                {
                    Matrix3a hessian;
                    AScalar flen = mut(q,q0,fn,friction_epsilon,friction_mu);
                    Vector3a force;
                    if(flen < friction_epsilon)
                        hessian = frictionHessian1(q,q0,fn,friction_epsilon,friction_mu);
                    else
                        hessian = frictionHessian2(q,q0,fn,friction_epsilon,friction_mu);
                    for(int a=0; a<3; ++a)
                    {
                        for(int b=0; b<3; ++b)
                        {
                            local_vector_temp.push_back(Entry(i*dim+a,i*dim+b,hessian(a,b)));
                        }
                    }
                }
            }
        }
    });

    for(auto& local_vector_temp: tbb_vector_temps)
    {
       entries.insert(entries.end(),local_vector_temp.begin(),local_vector_temp.end());
    }
}

template class FEMSolver<2>;
template class FEMSolver<3>;