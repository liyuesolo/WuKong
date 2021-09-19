#ifndef ZIRAN_TOPOSPACIALHASH_H
#define ZIRAN_TOPOSPACIALHASH_H
#include <Ziran/CS/Util/Forward.h>
#include <fstream>
#include <Ziran/CS/DataStructure/SpatialHash.h>

namespace ZIRAN {

template <class T, int dim>
class TopoSpatialHash : public SpatialHash<T, dim> {
public:
    typedef Vector<T, dim> TV;
    typedef Vector<int, dim> IV;
    using Base = SpatialHash<T, dim>;
    TopoSpatialHash()
        : Base()
    {
    }

    template <class PointArray>
    void rebuild(const T h_input, const PointArray& Xs)
    {
        ZIRAN_ASSERT(h_input > 0);

        Base::h = h_input;
        Base::hash.clear();
        for (size_t i = 0; i < Xs.size(); ++i) {
            const auto& X = Xs[i];
            IV cell = IV::Zero();
            for (int d = 0; d < dim; d++)
                cell(d) = (int)(std::floor(X(d) / Base::h));
            Base::hash[cell].push_back(i);
        }
    }

    template <class PointArray>
    void rebuildEigen(const T h_input, const PointArray& Xs, const TV& min_corner)
    {
        ZIRAN_ASSERT(h_input > 0);

        Base::h = h_input;
        Base::hash.clear();
        for (int i = 0; i < (int)Xs.cols(); ++i) {
            IV cell = IV::Zero();
            for (int d = 0; d < dim; d++)
                cell(d) = (int)(std::floor((Xs(d, i) - min_corner(d)) / Base::h));
            Base::hash[cell].push_back(i);
        }
    }

    template <class PointArray>
    void oneLayerNeighborsWithinL1Radius(const TV& X, const PointArray& points, const T radius, StdVector<int>& neighbors)
    {
        ZIRAN_ASSERT(Base::h > 0);
        neighbors.clear();
        IV cell = IV::Zero();
        for (int d = 0; d < dim; d++)
            cell(d) = (int)(std::floor(X(d) / Base::h));
        IV local_min_index = cell.array() - 1;
        IV local_max_index = cell.array() + 2;
        Box<int, dim> local_box(local_min_index, local_max_index);
        for (MaxExclusiveBoxIterator<dim> it(local_box); it.valid(); ++it) {
            StdVector<int>* v = Base::hash.get(it.index);
            if (v != nullptr) {
                for (auto i : *v) {
                    if ((points[i].segment(0, dim) - X).cwiseAbs().maxCoeff() <= radius)
                        neighbors.push_back(i);
                }
            }
        }
    }

    template <class PointArray>
    void oneLayerNeighborsWithinRadius(const TV& X, const PointArray& points, const T radius, StdVector<int>& neighbors)
    {
        ZIRAN_ASSERT(Base::h > 0);
        neighbors.clear();
        IV cell = IV::Zero();
        for (int d = 0; d < dim; d++)
            cell(d) = (int)(std::floor(X(d) / Base::h));
        IV local_min_index = cell.array() - 1;
        IV local_max_index = cell.array() + 2;
        Box<int, dim> local_box(local_min_index, local_max_index);
        for (MaxExclusiveBoxIterator<dim> it(local_box); it.valid(); ++it) {
            StdVector<int>* v = Base::hash.get(it.index);
            if (v != nullptr) {
                for (auto i : *v) {
                    if ((points[i].segment(0, dim) - X).norm() <= radius)
                        neighbors.push_back(i);
                }
            }
        }
    }

    void nLayerNeighbors(const TV& X, int n_layers, StdVector<int>& neighbors)
    {
        ZIRAN_ASSERT(Base::h > 0);
        neighbors.clear();
        IV cell = IV::Zero();
        for (int d = 0; d < dim; d++)
            cell(d) = (int)(std::floor(X(d) / Base::h));
        IV local_min_index = cell.array() - n_layers;
        IV local_max_index = cell.array() + n_layers + 1;
        Box<int, dim> local_box(local_min_index, local_max_index);
        for (MaxExclusiveBoxIterator<dim> it(local_box); it.valid(); ++it) {
            StdVector<int>* v = Base::hash.get(it.index);
            if (v != nullptr) {
                neighbors.insert(neighbors.end(), v->begin(), v->end());
            }
        }
    }

    template <class PointArray>
    void neighborsWithinRadius(const TV& X, const PointArray& points, const T radius, StdVector<int>& neighbors)
    {
        ZIRAN_ASSERT(Base::h > 0);
        neighbors.clear();
        IV cell = IV::Zero();
        for (int d = 0; d < dim; d++)
            cell(d) = (int)(std::floor(X(d) / Base::h));
        int layers = std::ceil(T(radius) / Base::h);
        // std::cout << layers << std::endl;
        IV local_min_index = cell.array() - layers;
        IV local_max_index = cell.array() + layers + 1;
        Box<int, dim> local_box(local_min_index, local_max_index);
        for (MaxExclusiveBoxIterator<dim> it(local_box); it.valid(); ++it) {
            StdVector<int>* v = Base::hash.get(it.index);
            if (v != nullptr) {
                for (auto i : *v) {
                    if ((points[i].segment(0, dim) - X).norm() <= radius)
                        neighbors.push_back(i);
                }
            }
        }
    }

    void neighborsInCell(const TV& X, StdVector<int>& neighbors)
    {
        ZIRAN_ASSERT(Base::h > 0);
        neighbors.clear();
        IV cell = IV::Zero();
        for (int d = 0; d < dim; d++)
            cell(d) = (int)(std::floor(X(d) / Base::h));
        neighbors = Base::hash[cell];
    }

    StdVector<int>& operator[](const IV& v)
    {
        return Base::hash[v];
    }
};

} // namespace ZIRAN

#endif //ZIRAN_TOPOSPACIALHASH_H
