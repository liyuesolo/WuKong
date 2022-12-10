#pragma once

#include "../../include/Tessellation/CellFunctionWeightedSum.h"
#include "../../include/Foam2DInfo.h"

#include "../../include/Energy/CellFunctionAreaTarget.h"
#include "../../include/Energy/CellFunctionAreaBarrier.h"
#include "../../include/Energy/CellFunctionPerimeter.h"
#include "../../include/Energy/CellFunctionCentroidXTarget.h"
#include "../../include/Energy/CellFunctionCentroidYTarget.h"
#include "../../include/Energy/CellFunctionPositionTarget.h"

class CellFunctionEnergy : public CellFunctionWeightedSum {
public:
    CellFunctionAreaTarget area_target_function;
    CellFunctionPerimeter perimeter_function;
    CellFunctionAreaBarrier area_barrier_function;
    CellFunctionCentroidXTarget centroid_x_function;
    CellFunctionCentroidYTarget centroid_y_function;
    CellFunctionPositionTarget position_target_function;
public:
    CellFunctionEnergy(Foam2DInfo *info) : CellFunctionWeightedSum() {
        functions.push_back(&area_target_function);
        weights.push_back(info->energy_area_weight);

        functions.push_back(&perimeter_function);
        weights.push_back(info->energy_length_weight);

        functions.push_back(&centroid_x_function);
        weights.push_back(info->energy_centroid_weight);

        functions.push_back(&centroid_y_function);
        weights.push_back(info->energy_centroid_weight);

        functions.push_back(&position_target_function);
        weights.push_back(info->energy_drag_target_weight);

        functions.push_back(&area_barrier_function);
        weights.push_back(1.0);
    };
};
