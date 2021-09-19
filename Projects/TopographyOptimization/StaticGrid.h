#ifndef TOPOGRID_H
#define TOPOGRID_H

#include <tbb/tbb.h>
#include <Ziran/CS/Util/Forward.h>
#include <SPGrid/Core/SPGrid_Allocator.h>
#include <SPGrid/Core/SPGrid_Page_Map.h>

namespace ZIRAN {

inline constexpr bool is_power_of_two(size_t x)
{
    return x > 0 && (x & (x - 1)) == 0;
}

inline std::array<int, 2> to_std_array(Vector<int, 2> v)
{
    return std::array<int, 2>{ v[0], v[1] };
}

inline std::array<int, 3> to_std_array(Vector<int, 3> v)
{
    return std::array<int, 3>{ v[0], v[1], v[2] };
}

template <class T, int dim>
class StaticGridState {
public:
    int cell_idx;
    int grid_idx;
    float E;
    float nu;
    float density;
    bool active_cell;
    bool fixed_density;
    bool touched;
    bool contact_interface;
    bool parent_active;
    bool unused[7];
};

template <class T, int dim>
class StaticGrid {
public:
    static constexpr int log2_page = 12;
    static constexpr int spgrid_size = 4096;
    using DataType = StaticGridState<T, dim>;
    using SparseGrid = SPGrid::SPGrid_Allocator<DataType, dim, log2_page>;
    using SparseMask = typename SparseGrid::template Array_mask<>;
    std::unique_ptr<SparseGrid> grid;
    typedef Vector<int, dim> IV;
    typedef Vector<T, dim> TV;
    typedef Vector<T, dim + 1> IVT;

    StdVector<IV> offset_vector;
    IV max_corner;
    IV min_corner;
    int id = -1; // for multimesh indexing;
    bool use_gauss_quad = false;

    StaticGrid()
    {
        ZIRAN_ASSERT(is_power_of_two(sizeof(DataType)), "Type size must be POT");
        if constexpr (dim == 2) {
            grid = std::make_unique<SparseGrid>(spgrid_size*2, spgrid_size*2);
        }
        else {
            grid = std::make_unique<SparseGrid>(spgrid_size, spgrid_size, spgrid_size);
        }
        offset_vector.clear();
        if constexpr (dim == 2)
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 2; ++j)
                    offset_vector.push_back(IV(i, j));
        else
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 2; ++j)
                    for (int k = 0; k < 2; ++k)
                        offset_vector.push_back(IV(i, j, k));
    }
    DataType& operator[](const Vector<int, dim>& v)
    {
        Vector<int, dim> node = v - min_corner;
        return grid->Get_Array()(to_std_array(node));
    }
    const DataType& operator[](const Vector<int, dim>& v) const
    {
        Vector<int, dim> node = v - min_corner;
        return grid->Get_Array()(to_std_array(node));
    }

    bool allocated(const Vector<int, dim>& v) const
    {
        if ((v - min_corner).minCoeff() >= 0 && (max_corner - v).minCoeff() >= 0)
            return true;
        else
            return false;
    }

    uint64_t linearOffset(const Vector<int, dim>& v)
    {
        Vector<int, dim> node = v - min_corner;
        return SparseMask::Linear_Offset(to_std_array(node));
    }

    template <typename OP>
    void iterateGrid(const OP& target)
    {
        IV region = max_corner - min_corner;
        region.array() += 1;
        if constexpr (dim == 2)
            tbb::parallel_for(0, (int)region.prod(), [&](int index) {
                int j = index % region[1];
                int i = index / region[1];
                IV node = Vector<int, 2>(i, j) + min_corner;
                target(node, (*this)[node]);
            });
        else
            tbb::parallel_for(0, (int)region.prod(), [&](int index) {
                int k = index % region[2];
                index /= region[2];
                int j = index % region[1];
                int i = index / region[1];
                IV node = Vector<int, 3>(i, j, k) + min_corner;
                target(node, (*this)[node]);
            });
    }

    template <typename OP>
    void iterateGridSerial(const OP& target)
    {
        IV region = max_corner - min_corner;
        region.array() += 1;
        if constexpr (dim == 2)
            for (int i = 0; i < region[0]; ++i)
                for (int j = 0; j < region[1]; ++j) {
                    IV node = Vector<int, 2>(i, j) + min_corner;
                    target(node, (*this)[node]);
                }
        else
            for (int i = 0; i < region[0]; ++i)
                for (int j = 0; j < region[1]; ++j)
                    for (int k = 0; k < region[2]; ++k) {
                        IV node = Vector<int, 3>(i, j, k) + min_corner;
                        target(node, (*this)[node]);
                    }
    }

    template <typename OP>
    void iterateNLaynerNeighbor(const Vector<int, 2>& base_node, int n_layer, const OP& target)
    {
        StdVector<IV> corners;
        for (int i = base_node[0] - n_layer; i <= base_node[0] + n_layer; ++i)
            for (int j = base_node[1] - n_layer; j <= base_node[1] + n_layer; ++j) {
                IV node = Vector<int, 2>(i, j);
                if ((node - base_node).cwiseAbs().sum() > 1) {
                    corners.push_back(node);
                    continue;
                }
                if (allocated(node))
                    target(node, (*this)[node]);
            }
        for (auto corner: corners) {
            if (allocated(corner))
                target(corner, (*this)[corner]);
        }
    }

    template <typename OP>
    void iterateNLaynerNeighbor(const Vector<int, 3>& base_node, int n_layer, const OP& target)
    {
        StdVector<IV> corners;
        for (int i = base_node[0] - n_layer; i <= base_node[0] + n_layer; ++i)
            for (int j = base_node[1] - n_layer; j <= base_node[1] + n_layer; ++j)
                for (int k = base_node[2] - n_layer; k <= base_node[2] + n_layer; ++k) {
                    IV node = Vector<int, 3>(i, j, k);
                    if ((node - base_node).cwiseAbs().sum() > 1) {
                        corners.push_back(node);
                        continue;
                    }
                    if (allocated(node))
                        target(node, (*this)[node]);
                }
        for (auto corner: corners) {
            if (allocated(corner))
                target(corner, (*this)[corner]);
        }
    }

    template <typename OP>
    void iterateNodeCell(const Vector<int, 2>& base_node, const OP& target)
    {
        for (int i = base_node[0] - 1; i <= base_node[0]; ++i)
            for (int j = base_node[1] - 1; j <= base_node[1]; ++j) {
                IV node = Vector<int, 2>(i, j);
                if (allocated(node))
                    target(node, (*this)[node]);
            }
    }

    template <typename OP>
    void iterateNodeCell(const Vector<int, 3>& base_node, const OP& target)
    {
        for (int i = base_node[0] - 1; i <= base_node[0]; ++i)
            for (int j = base_node[1] - 1; j <= base_node[1]; ++j)
                for (int k = base_node[2] - 1; k <= base_node[2]; ++k) {
                    IV node = Vector<int, 3>(i, j, k);
                    if (allocated(node))
                        target(node, (*this)[node]);
                }
    }

    template <typename OP>
    void iterateAxisAlignedOneLaynerNeighbor(const Vector<int, 2>& base_node, const OP& target)
    {
        int n_layer = 1;
        for (int i = base_node[0] - n_layer; i <= base_node[0] + n_layer; ++i)
            for (int j = base_node[1] - n_layer; j <= base_node[1] + n_layer; ++j) {
                IV node = Vector<int, 2>(i, j);
                if (!allocated(node)) continue;
                if ((node - base_node).prod() != 0) continue;
                target(node, (*this)[node]);
            }
    }

    template <typename OP>
    void iterateAxisAlignedOneLaynerNeighbor(const Vector<int, 3>& base_node, const OP& target)
    {
        int n_layer = 1;
        for (int i = base_node[0] - n_layer; i <= base_node[0] + n_layer; ++i)
            for (int j = base_node[1] - n_layer; j <= base_node[1] + n_layer; ++j)
                for (int k = base_node[2] - n_layer; k <= base_node[2] + n_layer; ++k) {
                    IV node = Vector<int, 3>(i, j, k);
                    if (!allocated(node)) continue;
                    if ((node - base_node).cwiseAbs().sum() != 1) continue;
                    target(node, (*this)[node]);
                }
    }

    template <typename OP>
    void iterateSubRegion(const Vector<int, 2>& base_node, int range, const OP& target)
    {
        for (int i = base_node[0]; i < base_node[0] + range; ++i)
            for (int j = base_node[1]; j < base_node[1] + range; ++j) {
                IV node = Vector<int, 2>(i, j);
                if (allocated(node))
                    target((*this)[node]);
            }
    }

    template <typename OP>
    void iterateSubRegion(const Vector<int, 3>& base_node, int range, const OP& target)
    {
        for (int i = base_node[0]; i < base_node[0] + range; ++i)
            for (int j = base_node[1]; j < base_node[1] + range; ++j)
                for (int k = base_node[2]; k < base_node[2] + range; ++k) {
                    IV node = Vector<int, 3>(i, j, k);
                    if (allocated(node))
                        target((*this)[node]);
                }
    }

    template <typename OP>
    void iterateCellKernel(const IV& base_node, const OP& target)
    {
        for (auto& delta : offset_vector) {
            IV node = base_node + delta;
            target(node, (*this)[node]);
        }
    }

    template <typename OP>
    void iterateCellKernel(const IV& base_node, const TV& X, T dx, const OP& target)
    {
        // target should be target(id, node, w, dw)
        TV Xp_rel = X - base_node.template cast<T>();
        TV w[2] = { -Xp_rel.array() + 1, Xp_rel };
        T dw[2] = { T(-1.) / dx, T(1.) / dx };
        for (auto& delta : offset_vector) {
            IV node = base_node + delta;
            T weight = 1;
            TV dweight;
            for (int i = 0; i < dim; ++i) {
                weight *= w[delta[i]](i);
            }
            if (weight == 0)
                dweight.setZero();
            else
                for (int i = 0; i < dim; ++i) {
                    dweight(i) = weight / w[delta[i]](i) * dw[delta[i]];
                }
            target(node, (*this)[node], weight, dweight);
        }
    }


    template <bool return_state=false, typename OP>
    void iterateKernel(const IV& base_node, int quadrature_loc_index, T dx, const OP& target)
    {
        // target should be target(id, node, w, dw)
        TV Xp_rel;
        if (use_gauss_quad)
            Xp_rel = (1./std::sqrt(3.) * offset_vector[quadrature_loc_index].template cast<T>()).array() + (0.5 - 0.5 / std::sqrt(3.));
        else
            Xp_rel = (0.5 * offset_vector[quadrature_loc_index].template cast<T>()).array() + 0.25;
        TV w[2] = { -Xp_rel.array() + 1, Xp_rel };
        T dw[2] = { T(-1.) / dx, T(1.) / dx };
        for (auto& delta : offset_vector) {
            IV node = base_node + delta;
            T weight = 1;
            TV dweight;
            for (int i = 0; i < dim; ++i) {
                weight *= w[delta[i]](i);
            }
            if (weight == 0)
                dweight.setZero();
            else
                for (int i = 0; i < dim; ++i) {
                    dweight(i) = weight / w[delta[i]](i) * dw[delta[i]];
                }
            if constexpr (return_state)
                target((*this)[node], node, weight, dweight);
            else
                target((*this)[node].grid_idx, node, weight, dweight);
        }
    }
};

} // namespace ZIRAN

#endif