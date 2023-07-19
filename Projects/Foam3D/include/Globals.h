#pragma once

// CellFunctionEnergy

extern double volumeTargetWeight;//100;
extern double surfaceAreaTargetWeight;
extern double siteCentroidWeight;//1;
extern double volumeBarrierWeight;
extern double wPenaltyWeight;
extern double secondMomentWeight;//100;
extern double adhesionWeight;

extern double cellRadiusTarget;

// GastrulationBoundary

extern double kNeighborhood;
extern double kVol;
extern double volTarget;
extern double outerRadius;

// DynamicObjective

extern double dt;
extern double m;
extern double eta;
