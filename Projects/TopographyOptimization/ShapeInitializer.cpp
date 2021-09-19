#include "TopoOptSimulation.h"

namespace ZIRAN {

template <class T, int dim>
void TopographyOptimization<T, dim>::initializeDesignPad(T dx, TV min_corner, TV max_corner)
{
    ZIRAN_TIMER();
    if (this->dx > 0) { ZIRAN_ASSERT(this->dx == dx); } else { this->dx = dx; }
    this->vol = std::pow(0.5 * dx, dim);
    grid = StaticGrid<T, dim>();
    grid.id = 0;
    
    grid.min_corner = (min_corner / dx).array().round().template cast<int>();
    grid.max_corner = (max_corner / dx).array().round().template cast<int>();
    ZIRAN_INFO("Create grid between ", grid.min_corner.transpose(), " and ", grid.max_corner.transpose());
    
    if constexpr (dim == 2)
        for (int i = grid.min_corner(0); i <= grid.max_corner(0); ++i)
            for (int j = grid.min_corner(1); j <= grid.max_corner(1); ++j) {
                IV index = IV(i, j);
                grid[index].touched = true;
                grid[index].cell_idx = -2;
                grid[index].grid_idx = -1;
                if (i == grid.max_corner(0) || j == grid.max_corner(1))
                    grid[index].cell_idx = -3; // boundary grid, not for cell
            }
    else
        for (int i = grid.min_corner(0); i <= grid.max_corner(0); ++i)
            for (int j = grid.min_corner(1); j <= grid.max_corner(1); ++j)
                for (int k = grid.min_corner(2); k <= grid.max_corner(2); ++k) {
                    IV index = IV(i, j, k);
                    grid[index].touched = true;
                    grid[index].cell_idx = -2;
                    grid[index].grid_idx = -1;
                    if (i == grid.max_corner(0) || j == grid.max_corner(1) || k == grid.max_corner(2))
                        grid[index].cell_idx = -3;
                }
}

template <class T, int dim>
void TopographyOptimization<T, dim>::addLambdaShape(std::function<bool(const TV&)> &cell_helper, T E, T nu, T density, bool fixed_density) {
    
    
    grid.iterateGrid([&](const IV& base_node, auto& cell) {
        if (cell.cell_idx == -3) return; // boundary grid
        TV cell_base = dx * base_node.template cast<T>();
        TV cell_center = cell_base.array() + 0.5 * dx;
        if (cell_helper(cell_center)) {
            if (density == 0)
                cell.cell_idx = -2;
            else {
                cell.cell_idx = -1;
                cell.E = E;
                cell.nu = nu;
                cell.fixed_density = fixed_density;
                cell.density = density; 
            }
        }
    });
}

template <class T, int dim>
void TopographyOptimization<T, dim>::addBox(TV min_corner, TV max_corner, T E, T nu, T density, bool fixed_density) {
    std::function<bool(const TV&)> shape_function = [&](const TV& x) -> bool {
        return (x - min_corner).minCoeff() >= -0.01 * dx && (max_corner - x).minCoeff() >= -0.01 * dx;
    };
    addLambdaShape(shape_function, E, nu, density, fixed_density);
}

template <class T, int dim>
void TopographyOptimization<T, dim>::addSphere(TV center, T radius, T E, T nu, T density, bool fixed_density)
{
    std::function<bool(const TV&)> shape_function = [&](const TV& x) -> bool {
        return (x - center).norm() <= radius + 0.01 * dx;
    };
    addLambdaShape(shape_function, E, nu, density, fixed_density);
}

template <class T, int dim>
void TopographyOptimization<T, dim>::finalizeDesignDomain()
{
    int cnt = 0;
    
    grid.iterateGridSerial([&](const IV& base_node, auto& cell) {
        if (cell.cell_idx == -1)
            cell.cell_idx = cnt++;
    });
    num_cells = cnt;
    ZIRAN_INFO("Cells: ", num_cells);
}

template class TopographyOptimization<double, 2>;
template class TopographyOptimization<double, 3>;

} // end of ZIRAN