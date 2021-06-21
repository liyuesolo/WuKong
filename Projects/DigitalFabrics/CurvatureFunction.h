#ifndef CURVATURE_FUNCTION_H
#define CURVATURE_FUNCTION_H

#include <cmath>
#include <iostream>
#include <utility>
#include <vector>

// #include "VecMatDef.h"

template<class T, int dim>
class CurvatureFunction
{
public:
    
public:
    CurvatureFunction() {}
    ~CurvatureFunction() {}

    virtual T value(T u) {return 0;}
    virtual void gradient(T u, T& dedu) { dedu = 0; }
    virtual void hessian(T u, T& de2du2) { de2du2 = 0; }
};

template<class T, int dim>
class LineCurvature : public CurvatureFunction<T, dim>
{
public:

};

template<class T, int dim>
class PreBendCurvaure : public CurvatureFunction<T, dim>
{
    T length;
    T theta;
public:
    PreBendCurvaure(T _length, T _theta) : length(_length), theta(_theta) {}
    virtual T value(T u)
    {
        return theta / length;
    }
};

template<class T, int dim>
class CircleCurvature : public CurvatureFunction<T, dim>
{
private:
    T r;
public:
    CircleCurvature(T _r) : r(_r) {}

    virtual T value(T u);
    virtual void gradient(T u, T& dedu);
    virtual void hessian(T u, T& de2du2);
};

template<class T, int dim>
class SineCurvature : public CurvatureFunction<T, dim>
{
private:
    T amp;
    T phi;
    T period;

public:
    SineCurvature(T _amp, T _phi, T _period) : amp(_amp), phi(_phi), period(_period) {}
    
    virtual T value(T u);
    //-g -> f
    virtual void gradient(T u, T& dedu);
    // -J -> Hessian
    virtual void hessian(T u, T& de2du2);
};

#endif