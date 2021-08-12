#include "GCodeGenerator.h"

template<class T, int dim>
GCodeGenerator<T, dim>::GCodeGenerator(const EoLRodSim<T, dim>& _sim, 
    const std::string& filename) 
    : sim(_sim), 
    gcode_file(filename),
    nozzle_diameter(0.4),
    filament_diameter(1.75),
    extruder_temperature(180),
    bed_temperature(60),
    first_layer_height(0.3),
    layer_height(0.2),
    feed_rate_move(5000),
	feed_rate_print(1000),
    printer(PrusaI3),
    extrusion_mode(Absolute),
    current_E(0.0)
{

}

template<class T, int dim>
void GCodeGenerator<T, dim>::scaleAndShift(TV& x)
{
    x *= 1e3;
    x.template segment<2>(0) += Vector<T, 2>(50, 50);
}

template<class T, int dim>
void GCodeGenerator<T, dim>::addSingleTunnel(const TV& from, const TV& to, T height)
{
    if constexpr (dim == 3)
    {
        TV3 mid_point = TV3::Zero();
        mid_point.template segment<dim>(0) = 0.5 * (from + to);
        mid_point[2] += height;    
        moveTo(from);
        writeLine(from, mid_point, false);
        writeLine(mid_point, to, false);
    }
}

template<class T, int dim>
void GCodeGenerator<T, dim>::generateGCodeFromRodsCurveGripperHardCoded()
{
    writeHeader();
    for (auto& rod : sim.Rods)
    {
        TV x0; rod->x(rod->indices.front(), x0);
        scaleAndShift(x0);
        x0[dim - 1] = first_layer_height;
        moveTo(x0);
        rod->iterateSegments([&](int node_i, int node_j, int rod_idx)
        {
            TV xi, xj;
            rod->x(node_i, xi); rod->x(node_j, xj);
            
            scaleAndShift(xi); scaleAndShift(xj);
            if (rod->rod_id == 0 || rod_idx == rod->numSeg() - 1)
            {
                xi[dim - 1] = first_layer_height;
                xj[dim - 1] = first_layer_height;
            }
            else
            {
                xi[dim - 1] = first_layer_height + layer_height * 4.0;
                xj[dim - 1] = first_layer_height + layer_height * 4.0;
            }
            writeLine(xi, xj, true);
        });
    }
    int n0 = sim.Rods[0]->indices[1];
    int n1 = sim.Rods[0]->indices[sim.Rods[0]->indices.size()-2];
    TV left, right;
    sim.Rods[0]->x(n0, left);
    sim.Rods[0]->x(n1, right);
    scaleAndShift(left); scaleAndShift(right);
    left[dim - 1] = first_layer_height;
    right[dim - 1] = first_layer_height;
    addSingleTunnel(left, right, 2.0);
    writeFooter();
}

template<class T, int dim>
void GCodeGenerator<T, dim>::writeLine(const TV& from, const TV& to, bool is_first_layer)
{
    
    const T cross_section_area = crossSectionArea(is_first_layer);
	T amount = (to - from).norm();
    amount *= cross_section_area / (M_PI * filament_diameter * filament_diameter * 0.25);
    T current_amount = extrusion_mode == Absolute ? current_E + amount : amount;
    std::string cmd;
    if constexpr (dim == 3)
        cmd += "G1 F" + std::to_string(feed_rate_print) + " X" + 
            std::to_string(to[0]) + " Y" + std::to_string(to[1]) +
            " Z" + std::to_string(to[2]) +
            " E" + std::to_string(current_amount) + "\n";
    else if constexpr (dim == 2)
        cmd += "G1 F" + std::to_string(feed_rate_print) + " X" + 
        std::to_string(to[0]) + " Y" + std::to_string(to[1]) +
            " Z" + std::to_string(first_layer_height) +
            " E" + std::to_string(current_amount) + "\n";
    if (extrusion_mode == Absolute)
        current_E += amount;
    gcode << cmd;
}

template<class T, int dim>  
void GCodeGenerator<T, dim>::retract(T E)
{
    gcode << "G1 E" << std::to_string(E) << " F2100.0" << std::endl;
    current_E = E;
}

template<class T, int dim>  
void GCodeGenerator<T, dim>::moveTo(const TV& to)
{
    std::string cmd;
    if (extrusion_mode == Absolute)
    {
        retract(current_E - 0.8);
        cmd += "G1 F" + std::to_string(feed_rate_move) + " X" + 
            std::to_string(to[0]) + " Y" + std::to_string(to[1]) +
            " Z" + std::to_string(to[2]) + "\n";
        gcode << cmd;
        retract(current_E + 0.8);
    }
    else if (extrusion_mode == Relative)
    {
        retract(-0.8);
        cmd += "G1 F" + std::to_string(feed_rate_move) + " X" + 
            std::to_string(to[0]) + " Y" + std::to_string(to[1]) +
            " Z" + std::to_string(to[2]) + "\n";
        gcode << cmd;
        retract(0.8);
    }
}

template<class T, int dim>
T GCodeGenerator<T, dim>::extrusionWidth() const 
{
    return 1.2 * nozzle_diameter;
}

template<class T, int dim>
T GCodeGenerator<T, dim>::crossSectionArea(bool is_first_layer) const
{
	// Approximating cross sectino area, based on http://hydraraptor.blogspot.ch/2014/06/why-slicers-get-dimensions-wrong.html
	T extrusion_width = extrusionWidth();
	if (is_first_layer)
		return M_PI * first_layer_height * first_layer_height / 4 + first_layer_height * (extrusion_width - first_layer_height);
	else
		return M_PI * layer_height * layer_height / 4 + layer_height * (extrusion_width - layer_height);
}

template<class T, int dim>
void GCodeGenerator<T, dim>::writeHeader()
{
    gcode.open(gcode_file);
    if (printer == PrusaI3)
    {
        gcode << "G21 ; set units to millimeters\n";
        gcode << "G90 ; use absolute positioning\n";
        gcode << "M104 S" << std::to_string(extruder_temperature) << " ;set extruder temp\n";
        gcode << "M140 S"<< std::to_string(bed_temperature) << " ; set bed temp\n";
        gcode << "M190 S"<< std::to_string(bed_temperature) << " ; wait for bed temp\n";
        gcode << "M109 S" << std::to_string(extruder_temperature) << " ;wait for extruder temp\n";
        
        gcode << "G28 W ; home all without mesh bed level\n";
        gcode << "G80 ; mesh bed leveling\n";
        gcode << "G1 Y-3.0 F1000.0 ; go outside print area\n";
        gcode << "G92 E0.0 ; reset extruder distance position\n";
        gcode << "G1 X100.0 E9.0 F1000.0 ; intro line\n";
        gcode << "G92 E0.0 ; reset extruder distance position\n";
        if (extrusion_mode == Absolute)
            gcode << "M82 ;absolute extrusion mode\n";
        else if (extrusion_mode == Relative)
            gcode << "M83 ;relative extrusion mode\n";
        else
        {
            std::cout << "unexpected extrusion mode" << std::endl;
            std::exit(0);
        }
        
    }
    else
    {
        std::cout << "unrecognized printer type" << std::endl;
        std::exit(0);
    }
}

template<class T, int dim>
void GCodeGenerator<T, dim>::writeFooter()
{
    if (printer == PrusaI3)
    {
        gcode << "M107\n";
        gcode << "G4 ; wait\n";
        gcode << "M221 S100 ; reset flow\n";
        gcode << "M104 S0 ; turn off extruder\n";
        gcode << "M140 S0 ; turn off heatbed\n";
        gcode << "M107 ; turn off fan\n";
        gcode << "G1 Z33.6 ; Move print head up\n";
        gcode << "G1 X0 Y210; home X axis and push Y forward\n";
        gcode << "M84 ; disable motors\n";
        gcode << "M73 P100 R0\n";
        gcode << "M73 Q100 S0\n";
    }
    else
    {
        std::cout << "unrecognized printer type" << std::endl;
        gcode.close();
        std::exit(0);
    }
    gcode.close();
    
}

template class GCodeGenerator<double, 3>;
template class GCodeGenerator<double, 2>;  