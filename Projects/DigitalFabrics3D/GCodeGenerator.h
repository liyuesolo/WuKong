#ifndef GCODE_GENERATOR_H
#define GCODE_GENERATOR_H

#include <fstream>

#include "EoLRodSim.h"
template<class T, int dim>
class EoLRodSim;

enum PrinterType
{
    PrusaI3, UltiMaker
};

enum ExtrusionMode
{
    Absolute, Relative
};

template<class T, int dim>
class GCodeGenerator
{
public:
    using TV = Vector<T, dim>;
    using TV3 = Vector<T, 3>;
    using TV2 = Vector<T, 2>;
private:
    const EoLRodSim<T, dim>& sim;
    std::string gcode_file;
	T nozzle_diameter;
    T filament_diameter;
    T extruder_temperature;
    T bed_temperature;
    T first_layer_height;
    T layer_height;
    T feed_rate_move;
    T feed_rate_print;
    PrinterType printer;
	ExtrusionMode extrusion_mode;

    T current_E;

    std::ofstream gcode;
public:
    GCodeGenerator(const EoLRodSim<T, dim>& _sim) : sim(_sim), gcode_file("./rod.gcode") {}
    GCodeGenerator(const EoLRodSim<T, dim>& _sim, const std::string& filename);
    ~GCodeGenerator() {}

public:
    void generateGCodeFromRodsCurveGripperHardCoded();

    void writeLine(const TV& from, const TV& to, bool is_first_layer);
    void moveTo(const TV& to);

    void addSingleTunnel(const TV& from, const TV& to, T height);

private:
    T computeExtrusionAmount() const;
    T extrusionWidth() const;
    T crossSectionArea(bool is_first_layer) const;
    void retract(T E);
    void writeHeader();
    void writeFooter();

    // to millimeters and shift towards center
    void scaleAndShift(TV& x);
    
};


#endif