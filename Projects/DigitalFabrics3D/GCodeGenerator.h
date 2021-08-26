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
    using Range = Vector<T, 2>;
    using Offset = Vector<int, dim + 1>;

// generic printing stuff
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

// specific parameters
private:
    T tunnel_height = 0.2; //0.2mm
public:
    GCodeGenerator(const EoLRodSim<T, dim>& _sim) : sim(_sim), gcode_file("./rod.gcode") {}
    GCodeGenerator(const EoLRodSim<T, dim>& _sim, const std::string& filename);
    ~GCodeGenerator() {}

public:
    
    void slidingBlocksGCode(int n_row, int n_col, int type);

    void activeTexticleGCode(bool fused = false);
    void activeTexticleGCode2(bool fused = false);

    void crossingTest();

    void generateGCodeFromRodsCurveGripperHardCoded();

    void generateGCodeFromRodsGridGripperHardCoded();
    void generateGCodeFromRodsFixedGridGripperHardCoded();

    void generateGCodeFromRodsShelterHardCoded();
    void generateGCodeFromRodsGridHardCoded(int n_row, int n_col, bool fused);

    void generateGCodeFromRodsNoTunnel();

    void writeLine(const TV& from, const TV& to, T rod_radius, T speed = 600.0);
    void moveTo(const TV& to, T speed = 2000.0, bool do_retract = true);

    void addSingleTunnel(const TV& from, const TV& to, T height);

    void addSingleTunnelOnCrossing(int crossing_id, const TV3& heights, 
        int direction, std::function<void(TV&)> scaleAndShift);
    
    void addSingleTunnelOnCrossingWithFixedRange(int crossing_id, const TV3& heights, 
        int direction, std::function<void(TV&)> scaleAndShift,
        const Range& range);

    void generateCodeSingleRod(int rod_idx, std::function<void(TV&)> scaleAndShift, 
        bool is_first_layer,
        T bd_height = 0.3, T inner_height = 0.3, T buffer_percentage = 0.3);
private:

    T computeExtrusionAmount() const;
    T crossSectionAreaFromRod(T rod_radius) const;

    T extrusionWidth() const;
    T crossSectionArea(bool is_first_layer) const;
    void retract(T E);
    void extrude(T E);
    void writeHeader();
    void writeFooter();

    
};


#endif