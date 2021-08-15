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
    first_layer_height(0.2), 
    layer_height(0.2),
    feed_rate_move(5200),
	feed_rate_print(1000),
    printer(PrusaI3),
    extrusion_mode(Absolute),
    current_E(0.0)
{

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
        
        writeLine(from, mid_point, tunnel_height);
        writeLine(mid_point, to, tunnel_height);
    }
}

template<class T, int dim>
void GCodeGenerator<T, dim>::generateCodeSingleRod(int rod_idx, 
    std::function<void(TV&)> scaleAndShift, bool is_first_layer,
    T bd_height, T inner_height)
{
    auto rod = sim.Rods[rod_idx];

    T rod_radius_in_mm;
    rod_radius_in_mm = rod->a * 1e3;
    
    TV x0; rod->x(rod->indices.front(), x0);
    TV front, back;
    rod->x(rod->indices.front(), front); rod->x(rod->indices.back(), back);

    TV extend = back + (back - front).normalized() * 0.3 * (front - back).norm();
    scaleAndShift(extend);

    //move slightly out of domain
    x0 -= (back - front).normalized() * 0.2 * (front - back).norm();
    scaleAndShift(x0);
    TV front_scaled = front;
    scaleAndShift(front_scaled);
    front_scaled[dim-1] = bd_height;
    x0[dim - 1] = 2.0;
    moveTo(x0);

    x0[dim - 1] = 0.2;
    moveTo(x0, 100);

    writeLine(x0, front_scaled, rod_radius_in_mm);

    // writeLine(x0, front_scaled, rod_radius_in_mm);
    rod->iterateSegments([&](int node_i, int node_j, int rod_idx)
    {
        TV xi, xj;
        rod->x(node_i, xi); rod->x(node_j, xj);
        // if (rod_idx == rod->numSeg() - 1)
        //     xj += (back - front).normalized() * 0.05 * (front - back).norm();
        scaleAndShift(xi); scaleAndShift(xj);
        if (rod_idx > rod->numSeg() - 2 || rod_idx < 2)
        {
            xi[dim - 1] = bd_height;
            xj[dim - 1] = bd_height;
        }
        else
        {
            xi[dim - 1] = inner_height;
            xj[dim - 1] = inner_height;
        }
        writeLine(xi, xj, rod_radius_in_mm);
    });

    scaleAndShift(back);
    back[dim - 1] += 0.2;
    moveTo(back, 100);
    back[dim - 1] = 2.0;
    moveTo(back, 100);

    extend[dim - 1] = 2.0;
    moveTo(extend);
}


template<class T, int dim>
void GCodeGenerator<T, dim>::addSingleTunnelOnCrossingWithFixedRange(int crossing_id, const TV3& heights, 
        int direction, std::function<void(TV&)> scaleAndShift,
        const Range& range)
{
    if constexpr (dim == 3)
    {
        tunnel_height = 0.3;
        auto crossing = sim.rod_crossings[crossing_id];
        auto rod = sim.Rods[crossing->rods_involved[direction]];
        TV front, back;
        rod->x(rod->indices.front(), front); rod->x(rod->indices.back(), back);
        T rod_length = (front - back).norm();
        TV rod_dir = (back - front).normalized();
        Range absolute_dx_in_mm = range * rod_length;
        TV x_crossing;
        rod->x(crossing->node_idx, x_crossing);
        TV left = x_crossing - rod_dir * absolute_dx_in_mm(0);
        TV right = x_crossing + rod_dir * absolute_dx_in_mm(1);

        scaleAndShift(left); scaleAndShift(right); 
        left[dim - 1] = heights[0];
        right[dim - 1] = heights[1];

        TV3 mid_point = TV3::Zero();
        mid_point.template segment<dim>(0) = 0.5 * (left + right);
        right += (right - left).normalized() * 0.3 * (right - left).norm();
        mid_point[2] += heights[2];  
        left[dim - 1] += 2.0;
        moveTo(left);
        left[dim - 1] -= 2.0;
        moveTo(left, 100);
        
        writeLine(left, mid_point, tunnel_height, 200);
        writeLine(mid_point, right, tunnel_height, 1000);
        right[dim - 1] += 2.0;
        moveTo(right);
    }
}

template<class T, int dim>
void GCodeGenerator<T, dim>::addSingleTunnelOnCrossing(int crossing_id, const TV3& heights,
    int direction,
    std::function<void(TV&)> scaleAndShift)
{
    if constexpr (dim == 3)
    {
        auto crossing = sim.rod_crossings[crossing_id];
        Range range = crossing->sliding_ranges[direction];
        auto rod = sim.Rods[crossing->rods_involved[direction]];
        TV front, back;
        rod->x(rod->indices.front(), front); rod->x(rod->indices.back(), back);
        T rod_length = (front - back).norm();
        TV rod_dir = (back - front).normalized();
        Range absolute_dx_in_mm = range * rod_length;
        TV x_crossing;
        rod->x(crossing->node_idx, x_crossing);
        TV left = x_crossing - rod_dir * absolute_dx_in_mm(0);
        TV right = x_crossing + rod_dir * absolute_dx_in_mm(1);
        
        scaleAndShift(left); scaleAndShift(right); 
        left[dim - 1] = heights[0];
        right[dim - 1] = heights[1];

        TV3 mid_point = TV3::Zero();
        mid_point.template segment<dim>(0) = 0.5 * (left + right);
        right += (right - left).normalized() * 0.3 * (right - left).norm();
        mid_point[2] += heights[2];  
        left[dim - 1] += 2.0;
        moveTo(left);
        left[dim - 1] -= 2.0;
        moveTo(left);

        writeLine(left, mid_point, tunnel_height);
        writeLine(mid_point, right, tunnel_height);

        
        right[dim - 1] += 2.0;
        moveTo(right);
    }

}

template<class T, int dim>
void GCodeGenerator<T, dim>::generateGCodeFromRodsFixedGridGripperHardCoded()
{
    auto scaleAndShift = [](TV& x)->void
    {
        x *= 1e3;
        x.template segment<2>(0) += Vector<T, 2>(40, 80);
    };

    writeHeader();

    T rod_radius_in_mm = sim.Rods[0]->a * 1e3;

    // step one bottom layers
    for (int rod_idx  : {0, 1, 2, 3, 4, 5, 6})
        generateCodeSingleRod(rod_idx, scaleAndShift, true, rod_radius_in_mm, rod_radius_in_mm);
    
    for (int rod_idx  : {7, 8, 9, 11, 12, 13})
        generateCodeSingleRod(rod_idx, scaleAndShift, true, 1.2 * rod_radius_in_mm, 1.2 * rod_radius_in_mm);

    generateCodeSingleRod(10, scaleAndShift, true, rod_radius_in_mm, 
        4.0 * rod_radius_in_mm);

    // TV3 heights = TV3(rod_radius_in_mm, rod_radius_in_mm, 4.0 * rod_radius_in_mm);
    TV3 heights = TV3(first_layer_height, first_layer_height, 14.0 * first_layer_height);
    for (int crossing_id : {3, 10})
    {
        addSingleTunnelOnCrossingWithFixedRange(crossing_id, heights, 0, scaleAndShift, Range(0.03, 0.03));
    }

    heights = TV3(first_layer_height * 2.0, first_layer_height * 2.0, 20.0 * first_layer_height);
    // for (int crossing_id : {3, 10})
    // {
    //     addSingleTunnelOnCrossingWithFixedRange(crossing_id, heights, 1, scaleAndShift, Range(0.06, 0.06));
    // }

    // addSingleTunnelOnCrossingWithFixedRange(3, heights, 1, scaleAndShift, Range(0.08, 0.08));
    addSingleTunnelOnCrossingWithFixedRange(10, heights, 1, scaleAndShift, Range(0.04, 0.04));

    writeFooter();

}


template<class T, int dim>
void GCodeGenerator<T, dim>::generateGCodeFromRodsGridGripperHardCoded()
{
    auto scaleAndShift = [](TV& x)->void
    {
        x *= 1e3;
        x.template segment<2>(0) += Vector<T, 2>(40, 80);
    };

    writeHeader();

    T rod_radius_in_mm = sim.Rods[0]->a * 1e3;

    // step one bottom layers
    for (int rod_idx  : {0, 1, 2, 3, 4, 5, 6, 7, 13})
        generateCodeSingleRod(rod_idx, scaleAndShift, true, rod_radius_in_mm, rod_radius_in_mm);
    // step two second layers with tunnels in y
    // print heigher in the middle for easy removal of unfixed crossings
    for (int rod_idx  : {8, 9, 11, 12})
        generateCodeSingleRod(rod_idx, scaleAndShift, true, rod_radius_in_mm, 
            4.0 * rod_radius_in_mm);
    
    // add tunnel
    // T base_height = first_layer_height + layer_height;
    TV3 heights = TV3(first_layer_height, first_layer_height, 14.0 * first_layer_height);
    for (int crossing_id : {8, 9, 11, 12, 15, 16, 18, 19})
    {
        // addSingleTunnelOnCrossing(crossing_id, heights1, 0, scaleAndShift);
        addSingleTunnelOnCrossingWithFixedRange(crossing_id, heights, 0, scaleAndShift, Range(0.04, 0.04));
    }

    heights = TV3(first_layer_height, first_layer_height, 14.0 * first_layer_height);
    for (int crossing_id : {22, 25})
    {
        // addSingleTunnelOnCrossing(crossing_id, heights1, 0, scaleAndShift);
        addSingleTunnelOnCrossingWithFixedRange(crossing_id, heights, 0, scaleAndShift, Range(0.1, 0.1));
    }

    generateCodeSingleRod(10, scaleAndShift, true, rod_radius_in_mm, 
        4.0 * rod_radius_in_mm);

    heights = TV3(rod_radius_in_mm, rod_radius_in_mm, 4.0 * rod_radius_in_mm);
    for (int crossing_id : {3, 10})
    {
        addSingleTunnelOnCrossingWithFixedRange(crossing_id, heights, 0, scaleAndShift, Range(0.03, 0.03));
    }

    for (int crossing_id : {23, 24})
    {
        auto crossing = sim.rod_crossings[crossing_id];
        Offset offset;
        sim.Rods[crossing->rods_involved.front()]->getEntry(crossing->node_idx, offset);
        TV x = sim.deformed_states.template segment<dim>(offset[0]);
        scaleAndShift(x);
        x[dim - 1] = rod_radius_in_mm + 2.0;
        moveTo(x);
        x[dim - 1] = rod_radius_in_mm;
        moveTo(x);
        extrude(0.2);
        x[dim - 1] += rod_radius_in_mm + 2.0;
        moveTo(x);
    }
    
    writeFooter();
}

template<class T, int dim>
void GCodeGenerator<T, dim>::generateGCodeFromRodsNoTunnel()
{
    auto scaleAndShift = [&](TV& x)
    {
        x *= 1e3;
        x.template segment<2>(0) += Vector<T, 2>(50, 50);
    };

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
            
            T rod_radius_in_mm = rod->a * 1e3;
            writeLine(xi, xj, rod_radius_in_mm);
        });
    }
    writeFooter();
}

template<class T, int dim>
void GCodeGenerator<T, dim>::generateGCodeFromRodsCurveGripperHardCoded()
{
    auto scaleAndShift = [&](TV& x)
    {
        x *= 1e3;
        x.template segment<2>(0) += Vector<T, 2>(50, 50);
    };

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
            T rod_radius_in_mm = rod->a * 1e3;
            writeLine(xi, xj, rod_radius_in_mm);
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
void GCodeGenerator<T, dim>::writeLine(const TV& from, const TV& to, T rod_radius, T speed)
{
    
    T cross_section_area = crossSectionAreaFromRod(rod_radius);
	T amount = (to - from).norm();
    amount *= cross_section_area / (M_PI * filament_diameter * filament_diameter * 0.25);
    T current_amount = extrusion_mode == Absolute ? current_E + amount : amount;
    std::string cmd;
    if constexpr (dim == 3)
        cmd += "G1 F" + std::to_string(speed) + " X" + 
            std::to_string(to[0]) + " Y" + std::to_string(to[1]) +
            " Z" + std::to_string(to[2]) +
            " E" + std::to_string(current_amount) + "\n";
    else if constexpr (dim == 2)
        cmd += "G1 F" + std::to_string(speed) + " X" + 
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
void GCodeGenerator<T, dim>::extrude(T E)
{
    T current_amout = current_E + E;
    gcode << "G1 E" << std::to_string(current_amout) << " F2100.0" << std::endl;
    current_E = current_amout;
}

template<class T, int dim>  
void GCodeGenerator<T, dim>::moveTo(const TV& to, T speed)
{
    std::string cmd;
    if (extrusion_mode == Absolute)
    {
        retract(current_E - 0.6);
        cmd += "G1 F" + std::to_string(speed) + " X" + 
            std::to_string(to[0]) + " Y" + std::to_string(to[1]) +
            " Z" + std::to_string(to[2]) + "\n";
        gcode << cmd;
        retract(current_E + 0.6);
    }
    else if (extrusion_mode == Relative)
    {
        retract(-0.8);
        cmd += "G1 F" + std::to_string(speed) + " X" + 
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
T GCodeGenerator<T, dim>::crossSectionAreaFromRod(T rod_radius) const
{
	T extrusion_width = extrusionWidth();
	return M_PI * rod_radius * rod_radius / 4 + rod_radius * (extrusion_width - rod_radius);
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