#ifndef GCODE_GENERATOR_H
#define GCODE_GENERATOR_H

#include "EoLRodSim.h"

template<class T, int dim>
class EoLRodSim;

template<class T, int dim>
class GCodeGenerator
{
private:
    EoLRodSim<T, dim>& sim;

public:
    GCodeGenerator(const EoLRodSim<T, dim>& _sim) : sim(sim) {}
    ~GCodeGenerator() {}

public:
    void generateGCodeFromRods();
};


#endif