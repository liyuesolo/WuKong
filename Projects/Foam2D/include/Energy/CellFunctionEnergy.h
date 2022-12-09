#pragma once

#include "../../include/Tessellation/CellFunctionWeightedSum.h"
#include "../../include/Foam2DInfo.h"

#include "../../include/Energy/CellFunctionAreaTarget.h"

class CellFunctionEnergy : public CellFunctionWeightedSum {
public:
    CellFunctionAreaTarget area_target_function;
public:
    CellFunctionEnergy(Foam2DInfo *info) : CellFunctionWeightedSum() {
        area_target_function.target_reciprocal = 1.0 / 0.05;

        functions.push_back(&area_target_function);
        weights.push_back(info->energy_area_weight);
    };
};
