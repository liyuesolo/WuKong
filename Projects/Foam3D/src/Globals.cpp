#include "../include/Globals.h"

// CellFunctionEnergy

double volumeTargetWeight = 500;//100;
double surfaceAreaTargetWeight = 0.0;
double siteCentroidWeight = 1;//1;
double volumeBarrierWeight = 1;
double wPenaltyWeight = 1;
double secondMomentWeight = 500;//100;
double adhesionWeight = 50;

double cellRadiusTarget = 0.4;

// GastrulationBoundary

double kNeighborhood = 10;
double kYolk = 2;
double kOuterFluid = 5;
double yolkTarget = 40.666;
double outerFluidTarget = 10;
double outerRadius = 2.81;

// DynamicObjective

double dt = 0.003;
double m = 0.001;
double eta = 0.1;
