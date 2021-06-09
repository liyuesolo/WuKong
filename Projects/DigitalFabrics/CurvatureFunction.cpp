#include "CurvatureFunction.h"

// template<class T, int dim>
// T DiscreteCurvature<T, dim>::value(std::vector<T>& us) 
// {
//     auto func = [&](T x)
//     {
//         if (x < 0.5)
//             return std::sqrt(r * r - (x - 0.25) * (x - 0.25)) + 0.5;
//     };
//     T y0 = func(us[0]), y1 = func(us[0]), y2 = func(us[0]);
//     if constexpr (dim == 2)
//     {
//         std::vector<Vector<T, dim + 1>> x(3);
//         x[0][0] = us[0]; x[0][1] = y0; x[0][2] = us[0];
//     }
// }

template<class T, int dim>
T CircleCurvature<T, dim>::value(T u) 
{ 
    // left most / right most boundary
    if (u < 1e-6 || std::abs(u - 2* M_PI * r) < 1e-6)
        return 0;
    T middle_point_arclength = M_PI * r;
    T dis = (u - middle_point_arclength);
    if (dis < -1e-6)
        return 1.0 / r; 
    else if(std::abs(dis) < 1e-6)
        return 0;
    else if(dis > 1e-6)
        return -1.0 / r; 
    else
        std::cout << "undefined curvature for u=" << u << std::endl;
}

template<class T, int dim>
void CircleCurvature<T, dim>::gradient(T u, T& dedu) 
{
     dedu = 0; 
}
template<class T, int dim>
void CircleCurvature<T, dim>::hessian(T u, T& de2du2) 
{ 
    de2du2 = 0; 
}

template<class T, int dim>
T SineCurvature<T, dim>::value(T u) 
{ 
    T t1 = period*period;
    T t3 = period*u;
    T t4 = std::sin(t3);
    T t5 = amp*amp;
    T t7 = std::cos(t3);
    T t8 = t7*t7;
    T t11 = std::pow(t5*t1*t8+1.0,-0.15E1);
    return -amp*t1*t4*t11;
}

//-g -> f
template<class T, int dim>
void SineCurvature<T, dim>::gradient(T u, T& dedu)
{
    T t1 = period*period;
    T t4 = period*u;
    T t5 = std::cos(t4);
    T t6 = amp*amp;
    T t8 = t5*t5;
    T t10 = t6*t1*t8+1.0;
    T t11 = std::pow(t10,-0.15E1);
    T t15 = t1*t1;
    T t18 = std::sin(t4);
    T t19 = t18*t18;
    T t20 = std::pow(t10,-0.25E1);
    dedu = amp*t1*period*t5*t11+0.3E1*t6*amp*t15*period*t19*t20*t5;
}
// -J -> Hessian
template<class T, int dim>
void SineCurvature<T, dim>::hessian(T u, T& de2du2)
{
    T t1 = period*period;
    T t2 = t1*t1;
    T t4 = period*u;
    T t5 = std::sin(t4);
    T t6 = amp*amp;
    T t8 = std::cos(t4);
    T t9 = t8*t8;
    T t11 = t6*t1*t9+1.0;
    T t12 = std::pow(t11,-0.15E1);
    T t17 = t6*amp*t2*t1;
    T t18 = std::pow(t11,-0.25E1);
    T t23 = t6*t6;
    T t25 = t2*t2;
    T t27 = t5*t5;
    T t28 = t27*t5;
    T t29 = std::pow(t11,-0.35E1);
    de2du2 = amp*t2*t5*t12+0.9E1*t17*t9*t18*t5+0.15E2*t23*amp*t25*t28*t29*t9-0.3E1*t17*t28*t18;
}

template class SineCurvature<double, 3>;
template class SineCurvature<double, 2>;   

template class CircleCurvature<double, 3>;
template class CircleCurvature<double, 2>;   


// template class DiscreteCurvature<double, 3>;
// template class DiscreteCurvature<double, 2>;   
