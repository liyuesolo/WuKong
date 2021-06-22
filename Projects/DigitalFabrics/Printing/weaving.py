import random
import copy
from math import *
import numpy as np


Pi = 3.14159265359

g_x_min = 10
g_x_max = 90
g_y_min = 10
g_y_max = 90
g_deltaZ = 0.5
g_ddZ = .40
g_retraction = 2.0

g_initZ = 0.15
g_jumpDistance = 0.5
g_nozzle_diam = 0.4
g_filament_diam = 1.75
g_temp = 195

PatternPlain   = [[0,1],[1,0]]
PatternBasket  = [[0,0,1,1],[0,0,1,1],[1,1,0,0],[1,1,0,0]]
Pattern22Twill = [[0,0,1,1],[0,1,1,0],[1,1,0,0],[1,0,0,1]]
Pattern31Twill = [[0,1,1,1],[1,1,1,0],[1,1,0,1],[1,0,1,1]]


def createCurves(f,pattern,repetition,caseSize,intersectionSize,startingPoint):
  curves = []
  for x_rep in range(0,repetition):
    for y_rep in range(0,repetition):
      for j in range(0,len(pattern)):
        for k in range(0,len(pattern)):
          if(pattern[j][k] == 0):#LR bottom
            p0 = startingPoint[0] + (x_rep*len(pattern) + j)*caseSize - intersectionSize,startingPoint[1] + (y_rep*len(pattern) + k)*caseSize + caseSize/2,g_initZ
            p1 = startingPoint[0] + (x_rep*len(pattern) + j+1)*caseSize + intersectionSize,startingPoint[1] + (y_rep*len(pattern) + k)*caseSize + caseSize/2,g_initZ
            curveLR = (p0,p1)
            curves.append(curveLR)
          else:#LR top
            p2 = startingPoint[0] + (x_rep*len(pattern) + j)*caseSize + caseSize/2,startingPoint[1] + (y_rep*len(pattern) + k)*caseSize - intersectionSize,g_initZ
            p3 = startingPoint[0] + (x_rep*len(pattern) + j)*caseSize + caseSize/2,startingPoint[1] + (y_rep*len(pattern) + k+1)*caseSize + intersectionSize,g_initZ
            curveTB = (p2,p3)
            curves.append(curveTB)

  for x_rep in range(0,repetition):
    for y_rep in range(0,repetition):
      for j in range(0,len(pattern)):
        for k in range(0,len(pattern)):
          if(pattern[j][k] == 0):#LR bottom
            p2 = startingPoint[0] + (x_rep*len(pattern) + j)*caseSize + caseSize/2,startingPoint[1] + (y_rep*len(pattern) + k)*caseSize - intersectionSize,2*g_initZ
            p3 = startingPoint[0] + (x_rep*len(pattern) + j)*caseSize + caseSize/2,startingPoint[1] + (y_rep*len(pattern) + k+1)*caseSize + intersectionSize,2*g_initZ
            p25 = (p2[0] + p3[0] )/2,(p2[1] + p3[1] )/2,4*g_initZ            
            
            curveTB = (p2,p25,p3)
            curves.append(curveTB)
          else:#LR top
            p0 = startingPoint[0] + (x_rep*len(pattern) + j)*caseSize - intersectionSize,startingPoint[1] + (y_rep*len(pattern) + k)*caseSize + caseSize/2,2*g_initZ
            p1 = startingPoint[0] + (x_rep*len(pattern) + j+1)*caseSize + intersectionSize,startingPoint[1] + (y_rep*len(pattern) + k)*caseSize + caseSize/2,2*g_initZ
            p05 = (p0[0] + p1[0] )/2,(p0[1] + p1[1] )/2,4*g_initZ    
            curveLR = (p0,p05,p1)
            curves.append(curveLR)
  retracted = False
  E = 0
  for curve3d in curves:
    #go to first point
    cpoint = curve3d[0]
    if(not retracted):
      E = E - g_retraction
      f.write("G1 E" + str(E)+ "\n")#retraction
      retracted = True
    f.write("G1 X" + str(cpoint[0]) + " Y" + str(cpoint[1]) + " Z" + str(g_initZ*2)+ "\n")
    f.write("G1 X" + str(cpoint[0]) + " Y" + str(cpoint[1]) + " Z" + str(cpoint[2]) + "\n")
    E = E + g_retraction
    f.write("G1 E" + str(E)+ "\n")
    retracted = False
    for pt3d in curve3d:
      if(cpoint == pt3d):
        continue
      E = E + computeE(cpoint,pt3d,g_initZ)
      f.write("G1 X" + str(pt3d[0]) + " Y" + str(pt3d[1]) + " Z" + str(pt3d[2])+ " E" + str(E) +"\n")
      cpoint = pt3d
    if(not retracted):
      E = E - g_retraction
      f.write("G1 E" + str(E)+ "\n")#retraction
      retracted = True
    f.write("G1 X" + str(cpoint[0]) + " Y" + str(cpoint[1]) + " Z" + str(g_initZ*2)+ "\n")


def dist(pA,pB):
  s = 0
  for i in range(0,len(pA)):
    s = s+(pA[i] - pB[i])*(pA[i] - pB[i])
  return sqrt(s)

def computeE(A, B,lHeight):
  srf = dist(A,B) * lHeight
  cSection = (g_filament_diam * g_filament_diam / 4.0) * Pi
  return srf * g_nozzle_diam / cSection
  
def writeHeader(f):
  f.write("M82 ;absolute extrusion mode\n")
  f.write("G21 ; set units to millimeters\n")
  f.write("G90 ; use absolute positioning\n")
  f.write("M82 ; absolute extrusion mode\n")
  f.write("M104 S" +str(g_temp) + " ;set extruder temp\n")
  f.write("M140 S60.0 ; set bed temp\n")
  f.write("M190 S60.0 ; wait for bed temp\n")
  f.write("M109 S" +str(g_temp) + " ;set extruder temp\n")
  f.write("G28 W ; home all without mesh bed level\n")
  f.write("G80 ; mesh bed leveling\n")
  f.write("G92 E0.0 ; reset extruder distance position\n")
  f.write("G1 Y-3.0 F1000.0 ; go outside print area\n")
  f.write("G1 X60.0 E9.0 F1000.0 ; intro line\n")
  f.write("G1 X100.0 E21.5 F600.0 ; intro line\n")
  f.write("G1 F600.0 ; intro line\n")
  f.write("G92 E0.0 ; reset extruder distance position\n")
  f.write("G92 E0\n")
  f.write("M107\n")

def writePointsToFile(sFile):
  E = 0.0

  with open(sFile, "w") as f: 
    writeHeader(f)
    createCurves(f,PatternPlain,1,2.5,0.4,(10,10))
    f.write("M107\n")
    f.write("M104 S0 ; turn off extruder\n")
    f.write("M140 S0 ; turn off heatbed\n")
    f.write("M107 ; turn off fan\n")
    f.write("G1 X0 Y210; home X axis and push Y forward\n")
    f.write("M84 ; disable motors\n")
    f.write("M82 ;absolute extrusion mode\n")
    f.write("M104 S0\n")


writePointsToFile("weave.gcode")