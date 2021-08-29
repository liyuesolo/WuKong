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
void GCodeGenerator<T, dim>::slidingBlocksGCode(int n_row, int n_col, int type)
{
    auto scaleAndShift = [](TV& x)->void
    {
        x *= 1e3;
        x.template segment<2>(0) += Vector<T, 2>(50, 40);
    };

    if constexpr (dim == 3)
    {
        writeHeader();
        T rod_radius_in_mm = sim.Rods[0]->a * 1e3 * 2.0;
        T extend_percentage = 0.1;
        T inner_height = 3.5 * rod_radius_in_mm;
        // x sliding
        if (type == 0)
        {
            int rod_cnt = 0; 
            int boundary_rod_cnt = 0;
            // outer contour
            for (int col = 0; col < n_col; col++)
            {
                generateCodeSingleRod(col, scaleAndShift, true, rod_radius_in_mm, rod_radius_in_mm, extend_percentage);
                rod_cnt++;
            }
            for (int row = 0; row < n_row; row++)
            {
                generateCodeSingleRod(n_col * 2 + n_row + row, scaleAndShift, true, rod_radius_in_mm, rod_radius_in_mm, extend_percentage);
                rod_cnt++;
            }
            for (int col = 0; col < n_col; col++)
            {
                generateCodeSingleRod(n_col + col, scaleAndShift, true, rod_radius_in_mm, rod_radius_in_mm, extend_percentage);
                rod_cnt++;
            }
            for (int row = 0; row < n_row; row++)
            {
                generateCodeSingleRod(n_col * 2 + row , scaleAndShift, true, rod_radius_in_mm, rod_radius_in_mm, extend_percentage);
                rod_cnt++;
            }
            boundary_rod_cnt = rod_cnt;
            while (rod_cnt < boundary_rod_cnt + 2 * (n_col - 1) * n_row)
            {
                generateCodeSingleRod(rod_cnt++, scaleAndShift, true, rod_radius_in_mm, rod_radius_in_mm, extend_percentage);
            }
            
            int temp = rod_cnt + 2 * (n_row - 1) * n_col;
            for (int col = 0; col < n_col - 1; col++)
            {
                for (int row = 0; row < n_row - 1; row++)
                {
                    int idx = row * (n_col - 1) + col;
                    if (row == n_row - 2)
                        generateCodeSingleRod(temp + idx * 4 + 1, scaleAndShift, true, rod_radius_in_mm, rod_radius_in_mm, extend_percentage, false, true);
                    else
                        generateCodeSingleRod(temp + idx * 4 + 1, scaleAndShift, true, rod_radius_in_mm, rod_radius_in_mm, extend_percentage);
                }
            }
            for (int col = 0; col < n_col - 1; col++)
            {
                for (int row = n_row - 2; row > -1; row--)
                {
                    int idx = row * (n_col - 1) + col;
                    if (row == 0)
                        generateCodeSingleRod(temp + idx * 4 + 3, scaleAndShift, true, rod_radius_in_mm, rod_radius_in_mm, extend_percentage, false, true);
                    else
                        generateCodeSingleRod(temp + idx * 4 + 3, scaleAndShift, true, rod_radius_in_mm, rod_radius_in_mm, extend_percentage);
                }
            }
            
            boundary_rod_cnt = rod_cnt;
            while (rod_cnt < boundary_rod_cnt + 2 * (n_row - 1) * n_col)
                generateCodeSingleRod(rod_cnt++, scaleAndShift, true, rod_radius_in_mm, inner_height, extend_percentage, true);


            
            temp = rod_cnt;
            for (int col = 0; col < n_col - 1; col++)
            {
                for (int row = 0; row < n_row - 1; row++)
                {
                    int idx = row * (n_col - 1) + col;
                    generateCodeSingleRod(temp + idx * 4 + 0, scaleAndShift, true, rod_radius_in_mm, inner_height, extend_percentage, true);
                }
            }
            for (int col = 0; col < n_col - 1; col++)
            {
                for (int row = n_row - 2; row > -1; row--)
                {
                    int idx = row * (n_col - 1) + col;
                    generateCodeSingleRod(temp + idx * 4 + 2, scaleAndShift, true, rod_radius_in_mm, inner_height, extend_percentage, true);
                }
            }

            TV3 heights = TV3(rod_radius_in_mm, rod_radius_in_mm, 4.0 * rod_radius_in_mm);
            T width;
            if (sim.unit == 0.05)
                width = 0.12;
            tunnel_height = 0.14;
            for (int row = 0; row < n_row - 1; row++)
            {
                for (int col = 0; col < n_col - 1; col++)
                {
                    int base = n_row * n_col * 4 + (row * (n_col - 1) + col) * 12;
                    for (int corner = 0; corner < 4; corner++)
                    {
                        int crossing_id = base + corner * 3 + 1;
                        addSingleTunnelOnCrossingWithFixedRange(crossing_id, heights, 0, scaleAndShift, Range(width, width), 0.2, 150, 200);
                        crossing_id = base + corner * 3 + 2;
                        addSingleTunnelOnCrossingWithFixedRange(crossing_id, heights, 1, scaleAndShift, Range(width, width), 0.2, 150, 200);
                    }
                }
            }
        }
        else if (type == 1)
        {
            int rod_cnt = 0; 
            for (int row = 0; row < n_row; row++)
            {
                for (int col = 0; col < n_col; col++)
                {
                    TV bottom_left;
                    sim.getCrossingPosition((row * n_col + col) * 4, bottom_left);

                    scaleAndShift(bottom_left);
                    bottom_left[dim-1] = rod_radius_in_mm; 
                    
                    for (int corner = 0; corner < 4; corner++)
                    {
                        int from_idx = (row * n_col + col) * 4 + corner;
                        int to_idx = (row * n_col + col) * 4 + (corner + 1) % 4;
                        TV from, to;
                        sim.getCrossingPosition(from_idx, from);
                        sim.getCrossingPosition(to_idx, to);
                        
                        TV left, right;
                        left = from - (to - from) * extend_percentage;
                        right = to + (to - from) * extend_percentage;

                        scaleAndShift(left); scaleAndShift(right);
                        left[dim-1] = rod_radius_in_mm; right[dim-1] = rod_radius_in_mm;
                        moveTo(left);
                        writeLine(left, right, rod_radius_in_mm);
                        // for (int i = 0; i < 8; i++)
                        // {
                        //     writeLine(from + (to - from) * i/8.0, from + (to - from) * (i + 1) /8.0, rod_radius_in_mm);
                        // }
                        
                        
                    }
                    
                }
            }
            for (int col = 0; col < n_col; col++) rod_cnt+=2;
            for (int row = 0; row < n_row; row++) rod_cnt+=2;
            for (int row = 0; row < n_row - 1; row++)
            {
                for (int col = 0; col < n_col - 1; col++)
                {
                    if (row == 0) rod_cnt += 2;
                    if (row == n_row - 2) rod_cnt += 2;
                    if (row != n_row - 2) rod_cnt += 2;
                    if (col == 0) rod_cnt += 2;
                    if (col == n_col - 2) rod_cnt += 2;
                    if (n_col - 2 != col) rod_cnt += 2;
                }
            }
            for (int row = 0; row < n_row - 1; row++)
            {
                for (int col = 0; col < n_col - 1; col++)
                {
                    // if (row == 0 && col == 0)
                    //     generateCodeSingleRod(rod_cnt++, scaleAndShift, true, 0.5 * rod_radius_in_mm, 3.0 * rod_radius_in_mm, 0.0);
                    // else
                        generateCodeSingleRod(rod_cnt++, scaleAndShift, true, rod_radius_in_mm, inner_height, extend_percentage);
                    generateCodeSingleRod(rod_cnt++, scaleAndShift, true, rod_radius_in_mm, inner_height, extend_percentage);
                    generateCodeSingleRod(rod_cnt++, scaleAndShift, true, rod_radius_in_mm, inner_height, extend_percentage);
                    generateCodeSingleRod(rod_cnt++, scaleAndShift, true, rod_radius_in_mm, inner_height, extend_percentage, false, true);
                }
            }

            TV3 heights = TV3(rod_radius_in_mm, rod_radius_in_mm, 4.0 * rod_radius_in_mm);
            T width;
            tunnel_height = 0.14;
            if (sim.unit == 0.05)
                width = 0.12;
            for (int row = 0; row < n_row - 1; row++)
            {
                for (int col = 0; col < n_col - 1; col++)
                {
                    int base = n_row * n_col * 4 + (row * (n_col - 1) + col) * 12;
                    for (int corner = 0; corner < 4; corner++)
                    {
                        int crossing_id = base + corner * 3 + 1;
                        addSingleTunnelOnCrossingWithFixedRange(crossing_id, heights, 0, scaleAndShift, Range(width, width));
                        crossing_id = base + corner * 3 + 2;
                        addSingleTunnelOnCrossingWithFixedRange(crossing_id, heights, 0, scaleAndShift, Range(width, width));
                    }
                }
            }
        }
        else if (type == 2)
        {
            int rod_cnt = 0; 
            int boundary_rod_cnt = 0;
            // outer contour
            for (int col = 0; col < n_col; col++)
            {
                generateCodeSingleRod(col, scaleAndShift, true, rod_radius_in_mm, rod_radius_in_mm, extend_percentage);
                rod_cnt++;
            }
            for (int row = 0; row < n_row; row++)
            {
                generateCodeSingleRod(n_col * 2 + n_row + row, scaleAndShift, true, rod_radius_in_mm, rod_radius_in_mm, extend_percentage);
                rod_cnt++;
            }
            for (int col = 0; col < n_col; col++)
            {
                generateCodeSingleRod(n_col + col, scaleAndShift, true, rod_radius_in_mm, rod_radius_in_mm, extend_percentage);
                rod_cnt++;
            }
            for (int row = 0; row < n_row; row++)
            {
                generateCodeSingleRod(n_col * 2 + row , scaleAndShift, true, rod_radius_in_mm, rod_radius_in_mm, extend_percentage);
                rod_cnt++;
            }
            boundary_rod_cnt = rod_cnt;
            while (rod_cnt < boundary_rod_cnt + 2 * (n_col - 1) * n_row)
            {
                generateCodeSingleRod(rod_cnt++, scaleAndShift, true, rod_radius_in_mm, rod_radius_in_mm, extend_percentage);
            }
            
            int temp = rod_cnt + 2 * (n_row - 1) * n_col;
            for (int col = 0; col < n_col - 1; col++)
            {
                for (int row = 0; row < n_row - 1; row++)
                {
                    int idx = row * (n_col - 1) + col;
                    if (row == n_row - 2)
                        generateCodeSingleRod(temp + idx * 4 + 1, scaleAndShift, true, rod_radius_in_mm, rod_radius_in_mm, extend_percentage, false, true);
                    else
                        generateCodeSingleRod(temp + idx * 4 + 1, scaleAndShift, true, rod_radius_in_mm, rod_radius_in_mm, extend_percentage);
                }
            }
            for (int col = 0; col < n_col - 1; col++)
            {
                for (int row = n_row - 2; row > -1; row--)
                {
                    int idx = row * (n_col - 1) + col;
                    if (row == 0)
                        generateCodeSingleRod(temp + idx * 4 + 3, scaleAndShift, true, rod_radius_in_mm, rod_radius_in_mm, extend_percentage, false, true);
                    else
                        generateCodeSingleRod(temp + idx * 4 + 3, scaleAndShift, true, rod_radius_in_mm, rod_radius_in_mm, extend_percentage);
                }
            }
            
            boundary_rod_cnt = rod_cnt;
            while (rod_cnt < boundary_rod_cnt + 2 * (n_row - 1) * n_col)
                generateCodeSingleRod(rod_cnt++, scaleAndShift, true, rod_radius_in_mm, rod_radius_in_mm, extend_percentage);


            
            temp = rod_cnt;
            for (int col = 0; col < n_col - 1; col++)
            {
                for (int row = 0; row < n_row - 1; row++)
                {
                    int idx = row * (n_col - 1) + col;
                    generateCodeSingleRod(temp + idx * 4 + 0, scaleAndShift, true, rod_radius_in_mm, rod_radius_in_mm, extend_percentage);
                }
            }
            for (int col = 0; col < n_col - 1; col++)
            {
                for (int row = n_row - 2; row > -1; row--)
                {
                    int idx = row * (n_col - 1) + col;
                    generateCodeSingleRod(temp + idx * 4 + 2, scaleAndShift, true, rod_radius_in_mm, rod_radius_in_mm, extend_percentage);
                }
            }

            
        }
        else if (type == 3)
        {
            int rod_cnt = 0; 
            for (int row = 0; row < n_row; row++)
            {
                for (int col = 0; col < n_col; col++)
                {
                    TV bottom_left;
                    sim.getCrossingPosition((row * n_col + col) * 4, bottom_left);
                    scaleAndShift(bottom_left);
                    bottom_left[dim-1] = rod_radius_in_mm; 
                    // moveTo(bottom_left);
                    for (int corner = 0; corner < 4; corner++)
                    {
                        int from_idx = (row * n_col + col) * 4 + corner;
                        int to_idx = (row * n_col + col) * 4 + (corner + 1) % 4;
                        TV from, to;
                        sim.getCrossingPosition(from_idx, from);
                        sim.getCrossingPosition(to_idx, to);

                        TV left, right;
                        left = from - (to - from) * extend_percentage;
                        right = to + (to - from) * extend_percentage;

                        scaleAndShift(left); scaleAndShift(right);
                        left[dim-1] = rod_radius_in_mm; right[dim-1] = rod_radius_in_mm;
                        
                        moveTo(left);
                        writeLine(left, right, rod_radius_in_mm);
                        // for (int i = 0; i < 8; i++)
                        // {
                        //     writeLine(from + (to - from) * i/8.0, from + (to - from) * (i + 1) /8.0, rod_radius_in_mm);
                        // }
                    }
                    
                }
            }
            for (int col = 0; col < n_col; col++) rod_cnt+=2;
            for (int row = 0; row < n_row; row++) rod_cnt+=2;
            for (int row = 0; row < n_row - 1; row++)
            {
                for (int col = 0; col < n_col - 1; col++)
                {
                    if (row == 0) rod_cnt += 2;
                    if (row == n_row - 2) rod_cnt += 2;
                    if (row != n_row - 2) rod_cnt += 2;
                    if (col == 0) rod_cnt += 2;
                    if (col == n_col - 2) rod_cnt += 2;
                    if (n_col - 2 != col) rod_cnt += 2;
                }
            }
            for (int row = 0; row < n_row - 1; row++)
            {
                for (int col = 0; col < n_col - 1; col++)
                {
                    // if (row == 0 && col == 0)
                    //     generateCodeSingleRod(rod_cnt++, scaleAndShift, true, 0.5 * rod_radius_in_mm, 3.0 * rod_radius_in_mm, 0.0);
                    // else
                        generateCodeSingleRod(rod_cnt++, scaleAndShift, true, rod_radius_in_mm, inner_height, extend_percentage);
                    generateCodeSingleRod(rod_cnt++, scaleAndShift, true, rod_radius_in_mm, inner_height, extend_percentage);
                    generateCodeSingleRod(rod_cnt++, scaleAndShift, true, rod_radius_in_mm, inner_height, extend_percentage);
                    generateCodeSingleRod(rod_cnt++, scaleAndShift, true, rod_radius_in_mm, inner_height, extend_percentage);
                }
            }

            TV3 heights = TV3(rod_radius_in_mm, rod_radius_in_mm, 4.0 * rod_radius_in_mm);
            T width;
            tunnel_height = 0.17;
            if (sim.unit == 0.05)
                width = 0.12;
            for (int row = 0; row < n_row - 1; row++)
            {
                for (int col = 0; col < n_col - 1; col++)
                {
                    int base = n_row * n_col * 4 + (row * (n_col - 1) + col) * 12;
                    for (int corner = 0; corner < 4; corner++)
                    {
                        if (corner == 0)
                            continue;
                        int crossing_id = base + corner * 3 + 1;
                        addSingleTunnelOnCrossingWithFixedRange(crossing_id, heights, 0, scaleAndShift, Range(width, width), 0.2, 150, 200);
                        crossing_id = base + corner * 3 + 2;
                        addSingleTunnelOnCrossingWithFixedRange(crossing_id, heights, 0, scaleAndShift, Range(width, width), 0.2, 150, 200);
                    }
                }
            }
        }
        writeFooter();
    }
}

template<class T, int dim>
void GCodeGenerator<T, dim>::generateGCodeFromRodsGridHardCoded(int n_row, int n_col, bool fused)
{
    auto scaleAndShift = [](TV& x)->void
    {
        x *= 1e3;
        x.template segment<2>(0) += Vector<T, 2>(50, 40);
    };

    T rod_radius_in_mm = sim.Rods[0]->a * 1e3 * 2.0;
    
    if constexpr (dim == 3)
    {
        writeHeader();
        T extend_percentage = 0.3;
        TV bottom_left, top_right;
        sim.computeBoundingBox(bottom_left, top_right);

        scaleAndShift(bottom_left); scaleAndShift(top_right);
        bottom_left[dim-1] = rod_radius_in_mm;
        top_right[dim-1] = rod_radius_in_mm;

        TV bottom_right = bottom_left;
        bottom_right[0] = top_right[0];
        TV top_left = top_right;
        top_left[0] = bottom_left[0];

        TV bottom_left_extend = bottom_left - (bottom_right - bottom_left) * 0.2;

        
        
        if (fused)
        {
            for (int row = 0; row < n_row; row++)
            {
                generateCodeSingleRod(row, scaleAndShift, true, rod_radius_in_mm, rod_radius_in_mm);
            }

            for (int col = 0; col < n_col; col++)
            {
                generateCodeSingleRod(col + n_row, scaleAndShift, true, 1.5 * rod_radius_in_mm, 1.5 * rod_radius_in_mm);
            }            
        }
        else
        {
            for (int row = 0; row < n_row; row++)
            {
                // generateCodeSingleRod(row, scaleAndShift, true, rod_radius_in_mm, rod_radius_in_mm, 0.3, false, true);
                auto rod = sim.Rods[row];
                TV from, to, left, right;
                rod->x(rod->indices.front(), from); rod->x(rod->indices.back(), to);
                left = from - (to - from) * extend_percentage;
                right = to + (to - from) * extend_percentage;

                scaleAndShift(left); scaleAndShift(right);
                scaleAndShift(from); scaleAndShift(to);
                left[dim-1] = rod_radius_in_mm; right[dim-1] = rod_radius_in_mm;
                from[dim-1] = rod_radius_in_mm; to[dim-1] = rod_radius_in_mm;
                
                if (row % 2 == 0)
                {
                    moveTo(left, 3000, true);
                    writeLine(left, from, rod_radius_in_mm, 600);
                    writeLine(from, to, rod_radius_in_mm, 1200);
                    writeLine(to, right, rod_radius_in_mm, 600);
                    TV extend = right;
                    extend += (right - left) * 0.1;
                    moveTo(extend);
                }
                else
                {
                    moveTo(right, 3000, true);
                    writeLine(right, to, rod_radius_in_mm, 600);
                    writeLine(to, from, rod_radius_in_mm, 1200);
                    writeLine(from, left, rod_radius_in_mm, 600);
                    TV extend = left;
                    extend -= (right - left) * 0.1;
                    moveTo(extend);
                }
            }

            for (int col = 0; col < n_col; col++)
            {
                // generateCodeSingleRod(col + n_row, scaleAndShift, true, rod_radius_in_mm, 3.0 * rod_radius_in_mm, 0.3, false, true);
                auto rod = sim.Rods[col + n_row];
                TV from, to, left, right;
                rod->x(rod->indices.front(), from); rod->x(rod->indices.back(), to);
                left = from - (to - from) * extend_percentage;
                right = to + (to - from) * extend_percentage;

                scaleAndShift(left); scaleAndShift(right);
                left[dim-1] = rod_radius_in_mm; right[dim-1] = rod_radius_in_mm;
                scaleAndShift(from); scaleAndShift(to);
                from[dim-1] = 3.0 * rod_radius_in_mm; to[dim-1] = 3.0 * rod_radius_in_mm;
                
                if (col % 2 == 0)
                {
                    
                    moveTo(right);
                    right[dim-1] = 0.2;
                    moveTo(right);
                    to[1] = top_right[1];
                    TV temp = 0.5 * (right + to);
                    temp[2] = 0.2;

                    writeLine(right, temp, rod_radius_in_mm, 200);
                    writeLine(temp, to, rod_radius_in_mm, 400);
                    writeLine(to, from, 4.0 * rod_radius_in_mm, 600);
                    temp = 0.5 * (from + left);
                    temp[2] = 0.2;
                    writeLine(from, temp, rod_radius_in_mm, 200);
                    writeLine(temp, left, rod_radius_in_mm, 200);
                }
                else
                {
                    moveTo(left);
                    left[dim-1] = 0.2;
                    moveTo(left);
                    from[1] = bottom_left[1];
                    TV temp = 0.5 * (left + from);
                    temp[2] = 0.2;
                    writeLine(left, temp, rod_radius_in_mm, 200);
                    writeLine(temp, from, rod_radius_in_mm, 400);
                    writeLine(from, to, 4.0 * rod_radius_in_mm, 600);
                    temp = 0.5 * (to + right);
                    temp[2] = 0.2;
                    writeLine(to, temp, rod_radius_in_mm, 200);
                    writeLine(temp, right, rod_radius_in_mm, 200);
                }
            }

            TV3 heights = TV3(rod_radius_in_mm, rod_radius_in_mm, 4.0 * rod_radius_in_mm);
            // T width = 0.034;
            T width = 0.015;
            tunnel_height = 0.2;
            for (int row = 0; row < n_row; row++)
            {
                for (int col = 0; col < n_col; col++)
                {
                    int crossing_id = row * n_col + col;
                    addSingleTunnelOnCrossingWithFixedRange(crossing_id, heights, 0, scaleAndShift, Range(width, width), 0.2, 100, 200);
                }
            }
        }

        moveTo(bottom_left_extend);
        writeLine(bottom_left_extend, bottom_right, rod_radius_in_mm, 1500);
        writeLine(bottom_right, top_right, rod_radius_in_mm, 1500);
        writeLine(top_right, top_left, rod_radius_in_mm, 1500);
        writeLine(top_left, bottom_left, rod_radius_in_mm, 1500);
        

        writeFooter();
    }
}


template<class T, int dim>
void GCodeGenerator<T, dim>::activeTexticleGCode(bool fused)
{
    auto scaleAndShift = [](TV& x)->void
    {
        x *= 1e3;
        x.template segment<2>(0) += Vector<T, 2>(40, 80);
    };


    if constexpr (dim == 3)
    {
        writeHeader();
        T rod_radius_in_mm = sim.Rods[0]->a * 1e3;

        if (fused)
        {
            // step one bottom layers
            for (int rod_idx  : {0, 1, 2, 3, 4, 5, 6, 8, 9})
                generateCodeSingleRod(rod_idx, scaleAndShift, true, rod_radius_in_mm, rod_radius_in_mm);
            
            for (int rod_idx  : {7})
                generateCodeSingleRod(rod_idx, scaleAndShift, true, rod_radius_in_mm, 
                    4.0 * rod_radius_in_mm);

            
            TV3 heights = TV3(first_layer_height, first_layer_height, 14.0 * first_layer_height);

            std::vector<int> crossings = {7, 12, 17, 22};

            for (int crossing_id : crossings)
            {
                addSingleTunnelOnCrossingWithFixedRange(crossing_id, heights, 0, scaleAndShift, Range(0.03, 0.03));
            }

            
        }
        else
        {
            // step one bottom layers
            for (int rod_idx  : {0, 1, 2, 3, 4, 5, 9})
                generateCodeSingleRod(rod_idx, scaleAndShift, true, rod_radius_in_mm, rod_radius_in_mm);
            
            for (int rod_idx  : {6, 7, 8})
                generateCodeSingleRod(rod_idx, scaleAndShift, true, rod_radius_in_mm, 
                    4.0 * rod_radius_in_mm);

            
            TV3 heights = TV3(first_layer_height, first_layer_height, 14.0 * first_layer_height);

            std::vector<int> crossings = {6, 7, 8, 11, 12, 13, 16, 17, 18, 21, 22, 23};

            for (int crossing_id : crossings)
            {
                addSingleTunnelOnCrossingWithFixedRange(crossing_id, heights, 0, scaleAndShift, Range(0.03, 0.03));
            }
        }
        writeFooter();
    }
}

template<class T, int dim>
void GCodeGenerator<T, dim>::activeTexticleGCode2(bool fused)
{
    auto scaleAndShift = [](TV& x)->void
    {
        x *= 1e3;
        x.template segment<2>(0) += Vector<T, 2>(40, 80);
    };


    if constexpr (dim == 3)
    {
        writeHeader();
        T rod_radius_in_mm = sim.Rods[0]->a * 1e3;

        // if (fused)
        // {
        //     for (int rod_idx  : {5})
        //         generateCodeSingleRod(rod_idx, scaleAndShift, true, rod_radius_in_mm, 
        //             rod_radius_in_mm);

        //     // step one bottom layers
        //     for (int rod_idx  : {0, 1, 2, 3, 4, 6, 8, 9})
        //         generateCodeSingleRod(rod_idx, scaleAndShift, true, rod_radius_in_mm, 4.0 * rod_radius_in_mm);
            
        //     for (int rod_idx  : {7})
        //         generateCodeSingleRod(rod_idx, scaleAndShift, true, rod_radius_in_mm, 
        //             4.0 * rod_radius_in_mm);

        //     TV3 heights = TV3(first_layer_height, first_layer_height, 14.0 * first_layer_height);

        //     std::vector<int> crossings = {7, 12, 17, 22};

        //     for (int crossing_id : crossings)
        //     {
        //         addSingleTunnelOnCrossingWithFixedRange(crossing_id, heights, 0, scaleAndShift, Range(0.03, 0.03));
        //     }

        //     crossings = {10};
        //     for (int crossing_id : crossings)
        //     {
        //         addSingleTunnelOnCrossingWithFixedRange(crossing_id, heights, 1, scaleAndShift, Range(0.03, 0.03));
        //     }
            
        // }
        if (fused)
        {
            // step one bottom layers
            for (int rod_idx  : {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
                generateCodeSingleRod(rod_idx, scaleAndShift, true, rod_radius_in_mm, rod_radius_in_mm);
               
        }
        else
        {

            for (int rod_idx  : {5})
                generateCodeSingleRod(rod_idx, scaleAndShift, true, rod_radius_in_mm, 
                    rod_radius_in_mm);

            // step one bottom layers
            for (int rod_idx  : {0, 1, 2, 3, 4})
                generateCodeSingleRod(rod_idx, scaleAndShift, true, rod_radius_in_mm, 4.0 * rod_radius_in_mm);

            for (int rod_idx  : {9})
                generateCodeSingleRod(rod_idx, scaleAndShift, true, rod_radius_in_mm, rod_radius_in_mm);
            
            for (int rod_idx  : {6, 7, 8})
                generateCodeSingleRod(rod_idx, scaleAndShift, true, rod_radius_in_mm, 
                    4.0 * rod_radius_in_mm);

            
            TV3 heights = TV3(first_layer_height, first_layer_height, 7.0 * first_layer_height);

            std::vector<int> crossings = {6, 7, 8, 11, 12, 13, 16, 17, 18, 21, 22, 23};

            for (int crossing_id : crossings)
            {
                addSingleTunnelOnCrossingWithFixedRange(crossing_id, heights, 0, scaleAndShift, Range(0.04, 0.04));
            }

            crossings = {5, 10, 15};
            for (int crossing_id : crossings)
            {
                addSingleTunnelOnCrossingWithFixedRange(crossing_id, heights, 1, scaleAndShift, Range(0.04, 0.04));
            }
        }
        writeFooter();
    }
}

template<class T, int dim>
void GCodeGenerator<T, dim>::crossingTest()
{
    if constexpr (dim == 3)
    {
        layer_height = 0.3;
        writeHeader();
        TV from = TV(20, 40, layer_height);
        TV to = TV(60, 40, layer_height);
        TV extend = TV(80, 40, layer_height);

        TV left(38, 40, layer_height);
        TV right(42, 40, layer_height);

        moveTo(from);
        writeLine(from, to, layer_height, 300);
        moveTo(extend);
        extend[dim - 1] += 4.0;
        moveTo(extend);
        extend[dim - 1] -= 4.0;

        left[dim - 1] += 2.0;
        moveTo(left);
        left[dim - 1] -= 2.0;
        // addSingleTunnel(left, right, 2.0);
        
        right[dim - 1] += 4.0;
        moveTo(right);
        right[dim - 1] -= 4.0;
        
        for (int i = 0; i < 9; i++)
        {
            from[dim - 1] += layer_height;
            left[dim - 1] += layer_height;
            
            from[dim - 1] += 4.0;
            moveTo(from);
            from[dim - 1] -= 4.0;
            moveTo(from);

            writeLine(from, left, layer_height, 300);
            left[dim - 1] += 2.0;
            moveTo(left);
            left[dim - 1] -= 2.0;
        }

        for (int i = 0; i < 9; i++)
        {
            to[dim - 1] += layer_height;
            right[dim - 1] += layer_height;
            
            right[dim - 1] += 4.0;
            moveTo(right);
            right[dim - 1] -= 4.0;
            moveTo(right);
            
            writeLine(right, to, layer_height, 300);
            to[dim - 1] += 2.0;
            moveTo(to);
            to[dim - 1] -= 2.0;
        }

        

        from = TV(40, 20, layer_height);
        to = TV(40, 60, layer_height);
        extend = TV(40, 80, layer_height);

        TV middle = TV(40, 40, layer_height * 4);
        TV middle0 = TV(40, 30, layer_height);
        TV middle1 = TV(40, 50, layer_height);
        
        for (int i = 0; i < 9; i++)
        {
            from[dim - 1] += layer_height;
            to[dim - 1] += layer_height;
            middle[dim - 1] += layer_height;
            middle0[dim - 1] += layer_height;
            middle1[dim - 1] += layer_height;

            from[dim - 1] += 4.0;
            moveTo(from);
            from[dim - 1] -= 4.0;
            moveTo(from);

            writeLine(from, middle0, layer_height, 300);
            writeLine(middle0, middle, layer_height, 300);
            writeLine(middle, middle1, layer_height, 300);
            writeLine(middle1, to, layer_height, 300);

            to[dim - 1] += 2.0;
            moveTo(to);
            to[dim - 1] -= 2.0;
        }
    
        left[dim - 1] += 8.0;
        moveTo(left);
        left[dim - 1] -= 8.0;
        right[0] += 5;
        addSingleTunnel(left, right, 3.0);
        
        writeFooter();
    }
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
        
        writeLine(from, mid_point, 0.3, 100);
        writeLine(mid_point, to, 0.3, 200);
    }
}

template<class T, int dim>
void GCodeGenerator<T, dim>::generateCodeSingleRod(int rod_idx, 
    std::function<void(TV&)> scaleAndShift, bool is_first_layer,
    T bd_height, T inner_height, T buffer_percentage, T less, T extend)
{
    auto rod = sim.Rods[rod_idx];

    T rod_radius_in_mm;
    rod_radius_in_mm = rod->a * 1e3 * 2.0;
    
    TV x0; rod->x(rod->indices.front(), x0);
    TV front, back;
    rod->x(rod->indices.front(), front); rod->x(rod->indices.back(), back);

    TV extension = back + (back - front).normalized() * 0.6 * (front - back).norm();
    scaleAndShift(extension);

    //move slightly out of domain in case it doesn't stick at the beginning
    x0 -= (back - front).normalized() * buffer_percentage * (front - back).norm();
    scaleAndShift(x0);
    TV front_scaled = front;
    scaleAndShift(front_scaled);
    front_scaled[dim-1] = bd_height;

    // 2.0 is to avoid nozzle touching existing rods
    // x0[dim - 1] = 1.5;
    // retract(current_E - 0.5);
    // moveTo(x0);

    // // 0.2 is used for better sticking at the beginning
    // x0[dim - 1] = 0.2;
    // moveTo(x0, 200, false);
    // retract(current_E + 0.5);

    
    x0[dim - 1] = bd_height;
    moveTo(x0);

    if (buffer_percentage > 1e-6)
        writeLine(x0, front_scaled, rod_radius_in_mm);


    // writeLine(x0, front_scaled, rod_radius_in_mm);
    int running_cnt =0;

    std::vector<bool> is_fused;
    rod->iterateSegments([&](int node_i, int node_j, int rod_idx)
    {
        is_fused.push_back(rod->isFixedNodeForPrinting(node_i, rod_idx));
        if (rod_idx == rod->numSeg() - 1)
            is_fused.push_back(rod->isFixedNodeForPrinting(node_j, rod_idx));        
        // std::cout << "is_fused " << std::endl;
        // std::cout << is_fused.back() << std::endl;
    }); 
    
    bool lift_head = false;

    std::vector<bool> fused_buffer = is_fused;
    for (int i = 0; i < is_fused.size(); i++)
    {
        if (!is_fused[i])
        {
            lift_head = true;
            break;       
            for (int j = i - 2 ; j < i + 3; j++)
            {
                if (j >= 0 && j < rod->numSeg())
                {
                    fused_buffer[j] = false;
                }
            }
        }
    }



    int node_cnt = 0;
    rod->iterateSegments([&](int node_i, int node_j, int rod_idx)
    {
        // std::cout << is_fused[node_cnt] << std::endl;
        TV xi, xj;
        rod->x(node_i, xi); rod->x(node_j, xj);
        // if (rod_idx == rod->numSeg() - 1 && buffer_percentage > 1e-6)
        //     xj += (back - front).normalized() * buffer_percentage * (front - back).norm();
        // else if (rod_idx == 0 && buffer_percentage > 1e-6)
        //     xi -= (back - front).normalized() * buffer_percentage * (front - back).norm();
        scaleAndShift(xi); scaleAndShift(xj);
        
        // if (fused_buffer[node_cnt]) xi[dim - 1] = bd_height;
        // else xi[dim - 1] = inner_height; 
        
        // if (fused_buffer[node_cnt + 1]) xj[dim - 1] = bd_height;
        // else xj[dim - 1] = inner_height; 
        // node_cnt++;

        if (lift_head && rod_idx != 0 && rod_idx != rod->numSeg()-1)
        {
             xj[dim - 1] = inner_height; 
             xi[dim - 1] = inner_height; 
        }
        else
        {
            xj[dim - 1] = bd_height; 
            xi[dim - 1] = bd_height; 
        }
        if (less)
            writeLine(xi, xj, 0.7 * rod_radius_in_mm);
        else
            writeLine(xi, xj, rod_radius_in_mm);
    });
    
    
    TV xn = back + (back - front).normalized() * buffer_percentage * (front - back).norm();
    // 
    scaleAndShift(xn);
    scaleAndShift(back);
    xn[dim - 1] = bd_height;
    // // moveTo(xn, 100);
    if (buffer_percentage > 1e-6)
        writeLine(back, xn, rod_radius_in_mm);
    // xn[dim - 1] = 2.0;
    // retract(current_E - 0.5);
    // moveTo(xn, 500, false);

    // // move nozzle along printing direction to avoid detaching of current print
    extension[dim - 1] = bd_height;
    if (extend)
        moveTo(extension);
    // retract(current_E + 0.5);
}


// template<class T, int dim>
// void GCodeGenerator<T, dim>::generateCodeSingleRod(int rod_idx, 
//     std::function<void(TV&)> scaleAndShift, bool is_first_layer,
//     T bd_height, T inner_height, T buffer_percentage)
// {
//     auto rod = sim.Rods[rod_idx];

//     T rod_radius_in_mm;
//     rod_radius_in_mm = rod->a * 1e3 * 2.0;
    
//     TV x0; rod->x(rod->indices.front(), x0);
//     TV front, back;
//     rod->x(rod->indices.front(), front); rod->x(rod->indices.back(), back);

//     TV extend = back + (back - front).normalized() * buffer_percentage * (front - back).norm();
//     scaleAndShift(extend);

//     //move slightly out of domain in case it doesn't stick at the beginning
//     x0 -= (back - front).normalized() * buffer_percentage * (front - back).norm();
//     scaleAndShift(x0);
//     TV front_scaled = front;
//     scaleAndShift(front_scaled);
//     front_scaled[dim-1] = bd_height;

//     // 2.0 is to avoid nozzle touching existing rods
//     // x0[dim - 1] = 2.0;
//     // retract(current_E - 0.5);
//     // moveTo(x0, 2000, false);

//     // // 0.2 is used for better sticking at the beginning
//     // x0[dim - 1] = 0.2;
//     // moveTo(x0, 50, false);
//     // retract(current_E + 0.5);
//     // if (buffer_percentage > 1e-6)
//     //     writeLine(x0, front_scaled, rod_radius_in_mm);

//     x0[dim - 1] = rod_radius_in_mm;
//     moveTo(x0);

//     // writeLine(x0, front_scaled, rod_radius_in_mm);
//     int running_cnt =0;

//     std::vector<bool> is_fused;
//     rod->iterateSegments([&](int node_i, int node_j, int rod_idx)
//     {
//         is_fused.push_back(rod->isFixedNodeForPrinting(node_i, rod_idx));
//         if (rod_idx == rod->numSeg() - 1)
//             is_fused.push_back(rod->isFixedNodeForPrinting(node_j, rod_idx));        
//         // std::cout << "is_fused " << std::endl;
//         // std::cout << is_fused.back() << std::endl;
//     }); 
    
//     std::vector<bool> fused_buffer = is_fused;
//     for (int i = 0; i < is_fused.size(); i++)
//     {
//         if (!is_fused[i])
//         {
            
//             for (int j = i - 5 ; j < i + 6; j++)
//             {
//                 if (j >= 0 && j < rod->numSeg())
//                 {
//                     fused_buffer[j] = false;
//                 }
//             }
//         }
//     }

//     int node_cnt = 0;
//     rod->iterateSegments([&](int node_i, int node_j, int rod_idx)
//     {
//         // std::cout << is_fused[node_cnt] << std::endl;
//         TV xi, xj;
//         rod->x(node_i, xi); rod->x(node_j, xj);
//         // if (rod_idx == rod->numSeg() - 1)
//         //     xj += (back - front).normalized() * 0.05 * (front - back).norm();
//         scaleAndShift(xi); scaleAndShift(xj);
//         if (is_fused[node_cnt]) xi[dim - 1] = bd_height;
//         else xi[dim - 1] = inner_height; 
        
//         if (is_fused[node_cnt + 1]) xj[dim - 1] = bd_height;
//         else xj[dim - 1] = inner_height; 
//         node_cnt++;

//         // if (rod_idx > rod->numSeg() - 2 || rod_idx < 2)
//         // {
//         //     xi[dim - 1] = bd_height;
//         //     xj[dim - 1] = bd_height;
//         // }
//         // else
//         // {
//         //     xi[dim - 1] = inner_height;
//         //     xj[dim - 1] = inner_height;
//         // }
//         writeLine(xi, xj, rod_radius_in_mm);
//     });
    
    
//     TV xn = back + (back - front).normalized() * buffer_percentage * (front - back).norm();
//     // 
//     // scaleAndShift(xn);
//     // scaleAndShift(back);
//     // xn[dim - 1] += 0.2;
//     // // moveTo(xn, 100);
//     // if (buffer_percentage > 1e-6)
//     //     writeLine(back, xn, rod_radius_in_mm);
//     // xn[dim - 1] = 2.0;
//     // retract(current_E - 0.5);
//     // moveTo(xn, 100, false);

//     // // move nozzle along printing direction to avoid detaching of current print
//     // extend[dim - 1] = 2.0;
//     // moveTo(extend, 2000, false);
//     // retract(current_E + 0.5);
// }


template<class T, int dim>
void GCodeGenerator<T, dim>::addSingleTunnelOnCrossingWithFixedRange(int crossing_id, const TV3& heights, 
        int direction, std::function<void(TV&)> scaleAndShift,
        const Range& range, T extend_right, T speed_first_half, T speed_second_half)
{
    if constexpr (dim == 3)
    {
        
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
        right += (right - left).normalized() * extend_right * (right - left).norm();
        mid_point[2] += heights[2];  
        left[dim - 1] += 2.0;
        moveTo(left);
        left[dim - 1] -= 2.0;
        moveTo(left, 100);
        
        writeLine(left, mid_point, tunnel_height, speed_first_half);
        writeLine(mid_point, right, tunnel_height, speed_second_half);
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
void GCodeGenerator<T, dim>::generateGCodeFromRodsShelterHardCoded()
{
    auto scaleAndShift = [](TV& x)->void
    {
        x *= 1e3;
        x.template segment<2>(0) += Vector<T, 2>(50, 50);
    };

    writeHeader();

    T rod_radius_in_mm = sim.Rods[0]->a * 1e3;
    for (int rod_idx  : {0, 1, 2, 3})
        generateCodeSingleRod(rod_idx, scaleAndShift, true, rod_radius_in_mm, rod_radius_in_mm);
    
    for (int rod_idx  : {5, 6})
        generateCodeSingleRod(rod_idx, scaleAndShift, true,  0.8 * rod_radius_in_mm, 1.2 * rod_radius_in_mm);

    for (int rod_idx  : {4, 7})
        generateCodeSingleRod(rod_idx, scaleAndShift, true,  0.8 * rod_radius_in_mm, 4.0 * rod_radius_in_mm);
    
    TV3 heights = TV3(first_layer_height, first_layer_height, 14.0 * first_layer_height);

    for (int crossing_id : {0, 3, 4, 7, 8, 11})
    {
        addSingleTunnelOnCrossingWithFixedRange(crossing_id, heights, 0, scaleAndShift, Range(0.03, 0.03));
    }

    writeFooter();
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
    // addSingleTunnelOnCrossingWithFixedRange(10, heights, 1, scaleAndShift, Range(0.04, 0.04));

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
            
            T rod_radius_in_mm = rod->a * 1e3 * 2.0;
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
            T rod_radius_in_mm = rod->a * 1e3 * 2.0;
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
    current_position = to;
}

template<class T, int dim>  
void GCodeGenerator<T, dim>::retract(T E)
{
    // gcode << "G1 E" << std::to_string(E) << " F2100.0" << std::endl;
    gcode << "G1 E" << std::to_string(E) << std::endl;
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
void GCodeGenerator<T, dim>::moveTo(const TV& to, T speed, bool do_retract)
{
    if ((current_position - to).norm() < 1e-6)
        return;
    std::string cmd;
    if (extrusion_mode == Absolute)
    {
        if (do_retract)
            retract(current_E - 0.4);
        cmd += "G1 F" + std::to_string(speed) + " X" + 
            std::to_string(to[0]) + " Y" + std::to_string(to[1]) +
            " Z" + std::to_string(to[2]) + "\n";
        gcode << cmd;
        if (do_retract)
            retract(current_E + 0.4);
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
    current_position = to;
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