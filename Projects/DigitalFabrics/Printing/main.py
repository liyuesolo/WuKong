import random
import copy
from math import *
import numpy as np


Pi = 3.14159265359

g_x_min = 10
g_x_max = 90
g_y_min = 10
g_y_max = 90
g_deltaZ = 1.0
g_ddZ = .8
g_speed = 400
g_retraction = 0.0

g_initZ = 0.35
g_jumpDistance = 1.0
g_nozzle_diam = 0.4
g_filament_diam = 1.75
g_temp = 250
points = []
edges = []
curves = []

  
def Reverse(tuples): 
    new_tup = tuples[::-1] 
    return new_tup 
    
def dist(pA,pB):
  s = 0
  for i in range(0,len(pA)):
    s = s+(pA[i] - pB[i])*(pA[i] - pB[i])
  assert(s>0)
  return sqrt(s)

def computeE(A, B,lHeight):
  srf = dist(A,B) * lHeight
  cSection = (g_filament_diam * g_filament_diam / 4.0) * Pi
  assert(srf * g_nozzle_diam / cSection > 0)
  return 1.2*srf * g_nozzle_diam / cSection
  
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
  f.write("G1 Y-3.0 F" + str(g_temp) +" ; go outside print area\n")
  f.write("G1 X60.0 E5.0 F"+str(g_temp) +"; intro line\n")
  f.write("G1 X100.0 E10.5 F"+str(g_temp) +" ; intro line\n")
  f.write("G1 F"+str(g_temp) +" ; intro line\n")
  f.write("G92 E0.0 ; reset extruder distance position\n")
  f.write("G92 E0\n")
  f.write("M107\n")


def outside(x):
  if(x[0]<x_min):
    return True
  if(x[0]>x_max):
    return True
  if(x[1]<y_min):
    return True
  if(x[1]>y_max):
    return True
  return False
  
def clip(x):
  if(x[0]<x_min):
    x[0] = x_min
  if(x[0]>x_max):
    x[0]=x_max
  if(x[1]<y_min):
    x[1]=y_min
  if(x[1]>y_max):
    x[1]=y_max  
    

def line(p1, p2):
  A = (p1[1] - p2[1])
  B = (p2[0] - p1[0])
  C = (p1[0]*p2[1] - p2[0]*p1[1])
  return A, B, -C

def cross(v0,v1):
  return v0[0] * v1[1] - v0[1] * v1[0]
    
def intersectionPtPt(pA0,pA1,pB0,pB1):
  ad = pA1[0] - pA0[0],pA1[1] - pA0[1] 
  bd = pB1[0] - pB0[0],pB1[1] - pB0[1] 
  o = cross(ad, bd);
  if (abs(o) < 1e-6):
    return 0,0,False
  b0a0 = pB0[0] - pA0[0],pB0[1] - pA0[1]
  
  t = cross(b0a0, bd) / o
  u = cross(b0a0, ad) / o
  _at = t
  _bt = u
  return _at,_bt,(t >= 0 and t <= 1 and u >= 0 and u <= 1);
  

def getBC(posA,posB,t ):
  return posA[0]  + t *(posB[0] - posA[0]),posA[1]  + t *(posB[1] - posA[1])
  
  
def geodesicDistance(points,curve,intersection):
  distance = 0
  
  for i in range(0,intersection[0]):
    pA,pB = points[curve[i]],points[curve[i+1]]
    distance = distance + dist(pA,pB)
  subDistance = 0
  if(intersection[1] > 0):
    pA,pB = points[curve[intersection[0]]],points[curve[intersection[0]+1]]
    subDistance = intersection[1] * dist(pA,pB)
  return distance+subDistance

  
def computePosOnCurve0(points,curve,intersection):
  posA = points[curve[intersection[0]]]
  posB = points[curve[intersection[0]+1]]
  return getBC(posA,posB,intersection[1])

def computePosNextOnCurve(points,curve,currentRelPos,delta):

  posC = points[curve[currentRelPos[0]]]
  posD = points[curve[currentRelPos[0]+1]]
  r = currentRelPos[0]

  
  assert(delta <= dist(posC,posD))
  dst = delta / (dist(posC,posD))
  dst = dst + currentRelPos[1]
  assert (dst >= 0)
  assert (dst <= 1)
  return getBC(posC,posD,dst),(r,dst)
  
def computePosOnCurve(points,curve,currentRelPos,jumpingDistance,intersection):

  if(intersection[1] > 1e-6):
    posA = points[curve[intersection[0]]]
    posB = points[curve[intersection[0]+1]]
  else:
    posA = points[curve[intersection[0]-1]]
    posB = points[curve[intersection[0]]]
  posC = points[curve[currentRelPos[0]]]
  posD = points[curve[currentRelPos[0]+1]]
  r = currentRelPos[0]
  
  a = geodesicDistance(points,curve,intersection)
  b = geodesicDistance(points,curve,currentRelPos)
  v = a - b - jumpingDistance
  dst = v / dist(posC,posD)
  print(intersection)
  assert (dst >= 0)
  
  dst +=currentRelPos[1]
  assert(dst >= currentRelPos[1])
  assert (dst <= 1)
  return getBC(posC,posD,dst),(r,dst)
  

def computeJump(points,curves, curveI,currentRelPos,Zj,initZ,intersection):
  jumpingCurve = []
  curve = curves[curveI]
  cdist = geodesicDistance(points,curve,currentRelPos)
  ndist = geodesicDistance(points,curve,intersection)
  dist2Intersection = ndist - cdist
  print(dist2Intersection)
  assert(dist2Intersection >= 0)
  init = currentRelPos[0]
  for i in range(init,intersection[0]+1):
    pt = points[curve[i]]
    cDist2Intersection = ndist - geodesicDistance(points,curve,currentRelPos)
    if(dist2Intersection-cDist2Intersection < 1e-6):
      z = initZ
    else:
      z = initZ + (Zj / (dist2Intersection) ) * (dist2Intersection-cDist2Intersection)
    pt2d = getBC(points[curve[i]],points[curve[i+1]],currentRelPos[1])
    pt = pt2d[0],pt2d[1],z
    jumpingCurve.append(pt)
    currentRelPos = currentRelPos[0]+1,0
  
  #add lastpoint to last Z
  pt2d = computePosOnCurve0(points,curve,intersection)
  jumpingCurve.append((pt2d[0],pt2d[1],initZ+Zj))
  #advance a littlebit after intersection
  return jumpingCurve,intersection

def computePointsBefore(points,curves,curveI,currentRelPos,curZ,intersection,jumpingDistance):
  curve = curves[curveI]
  cdist = geodesicDistance(points,curve,currentRelPos)
  ndist = geodesicDistance(points,curve,intersection)
  jumpingCurve = []
  precRelPos = currentRelPos
  dmy = False
  print("Before first point")
  print(currentRelPos)
  while (ndist - cdist > jumpingDistance):
    pt2d = getBC(points[curve[currentRelPos[0]]],points[curve[currentRelPos[0]+1]],currentRelPos[1])
    pt = pt2d[0],pt2d[1],curZ
    precRelPos = currentRelPos
    jumpingCurve.append(pt)
    currentRelPos = currentRelPos[0]+1,0
    cdist = geodesicDistance(points,curve,currentRelPos)
    dmy = True
  if(dmy):
    currentRelPos = precRelPos

  pt3d,currentRelPos = computePosOnCurve(points,curve,currentRelPos,jumpingDistance,intersection)
  jumpingCurve.append((pt3d[0],pt3d[1],curZ))
  print("Before last point")
  print(currentRelPos)
  return jumpingCurve,currentRelPos

  
def sortIntersectionAlongCurve(intersections):
  intersections.sort(key=lambda x: (x[0],x[1]))  
  for inter in intersections:
    print(inter)

def intersectionsCurveCurve(points,curves, curveI, curveJ,type):
  intersections = []
  for i in range(0,len(curves[curveI])-1):
    for j in range (0,len(curves[curveJ])-1):
      if(curveI == curveJ):#special case, we don't want to consider self intersection on edges that share a point
        if(i == j):continue
        if(j == i + 1):continue
        if(j + 1 == i):continue
        if(i == 0 and j == len(curves[curveJ])-2):continue
        if(i == len(curves[curveI])-2 and j == 0):continue

      edge = curves[curveI][i],curves[curveI][i+1]
      edge2 = curves[curveJ][j],curves[curveJ][j+1]
      p0 = points[edge[0]]
      p1 = points[edge[1]]
      p2 = points[edge2[0]]
      p3 = points[edge2[1]]
      t,_u,b = intersectionPtPt(p0,p1,p2,p3)
      if(b and t >  1e-6):
        if( not (i,t,type) in intersections):
          intersections.append((i,t,type))#edgeIndice and position along the edge
  return intersections

def computePointsLanding(points,curves,curveI,currentRelPos,initZ,dZ,jumpDistance):
  curve = curves[curveI]
  precDist = geodesicDistance(points,curve,currentRelPos)
  cdist = geodesicDistance(points,curve,currentRelPos)

  jumpingCurve = []
  dmy = False
  delta = 0
  precRP = currentRelPos
  while(cdist - precDist <= g_nozzle_diam):
    A = cdist - precDist
    pt2d = getBC(points[curve[currentRelPos[0]]],points[curve[currentRelPos[0]+1]],currentRelPos[1])
    pt = pt2d[0],pt2d[1],initZ+dZ
    jumpingCurve.append(pt)
    delta = cdist - precDist
    precRP = currentRelPos
    currentRelPos = currentRelPos[0]+1,0
    cdist = geodesicDistance(points,curve,currentRelPos)
    dmy = True
  if(dmy):
    currentRelPos = precRP

  pt3d,currentRelPos = computePosNextOnCurve(points,curve,currentRelPos,g_nozzle_diam - delta)
  jumpingCurve.append((pt3d[0],pt3d[1],initZ+dZ))
  dmy = False
  delta = 0
  precDist = geodesicDistance(points,curve,currentRelPos)
  cdist = geodesicDistance(points,curve,currentRelPos)
  while(cdist - precDist <= jumpDistance):
    if(currentRelPos[0] >= len(curve)-1):break
    A = cdist - precDist
    z = initZ + (dZ / jumpDistance) * (jumpDistance - (A))
    pt2d = getBC(points[curve[currentRelPos[0]]],points[curve[currentRelPos[0]+1]],currentRelPos[1])
    pt = pt2d[0],pt2d[1],z
    jumpingCurve.append(pt)
    delta = cdist - precDist
    precRP = currentRelPos
    currentRelPos = currentRelPos[0]+1,0
    cdist = geodesicDistance(points,curve,currentRelPos)
    dmy = True
  if(dmy):
    currentRelPos = precRP
  pt3d,currentRelPos = computePosNextOnCurve(points,curve,currentRelPos,jumpDistance - delta)
  jumpingCurve.append((pt3d[0],pt3d[1],initZ))
  return jumpingCurve,currentRelPos

    
def computeSingle3DCurve(points,curves,curveId,intersections,jumpDistance,Z1,dZ):
  curve3d = []
  print ("computeSingle3DCurve")
  curve = curves[curveId]
  if len(intersections) == 0:
    for pointInd in curves[curveId]:  
      curve3d.append(points[pointInd] + (Z1,))
    return curve3d
  
  nextIntersection = 0
  currentPointRelPos = (0,0)
  totalDist = geodesicDistance(points,curves[curveId],(len(curves[curveId]) - 2,1))
  Zpos = 0
  precInterType = intersections[0]
  for i in range(0,len(intersections)):
    geoDist = geodesicDistance(points,curves[curveId],intersections[i]) - geodesicDistance(points,curves[curveId],currentPointRelPos)
    if(geoDist < 0.4):#skip
      continue
    assert(geoDist >= 0)
    ldz = dZ
    if(intersections[i][2]):ldz = ldz + g_ddZ
    if(Zpos == 0):
      if(geoDist < jumpDistance): #jump Anyway
        res = computeJump(points,curves,curveId,currentPointRelPos,ldz,Z1,intersections[i])
        curve3d = curve3d + res[0]
        currentPointRelPos = res[1]
      else:#stay done untill we have to jump up
        res = computePointsBefore(points,curves,curveId,currentPointRelPos,Z1,intersections[i],jumpDistance)
        curve3d = curve3d + res[0]
        currentPointRelPos = res[1]
        res = computeJump(points,curves,curveId,currentPointRelPos,ldz,Z1,intersections[i])
        curve3d = curve3d + res[0]
        currentPointRelPos = res[1]
        Zpos = 1
    else:
      if(geoDist < 3*jumpDistance+g_nozzle_diam):
        ljumpDistance = (geoDist-g_nozzle_diam)/3
        if(precInterType[2]):
          res = computePointsLanding(points,curves,curveId,precInterType,Z1,dZ+g_ddZ,ljumpDistance)
          curve3d = curve3d + res[0]
          currentPointRelPos = res[1]
        else:
          res = computePointsLanding(points,curves,curveId,precInterType,Z1,dZ,ljumpDistance)
          curve3d = curve3d + res[0]
          currentPointRelPos = res[1]
        res = computePointsBefore(points,curves,curveId,currentPointRelPos,Z1,intersections[i],ljumpDistance)
        curve3d = curve3d + res[0]
        currentPointRelPos = res[1]
        res = computeJump(points,curves,curveId,currentPointRelPos,ldz,Z1,intersections[i])
        curve3d = curve3d + res[0]
        currentPointRelPos = res[1]
        Zpos = 1
      else:#head is up, and can go down then up stay up
        if(precInterType[2]):
          res = computePointsLanding(points,curves,curveId,precInterType,Z1,dZ+g_ddZ,jumpDistance)
          curve3d = curve3d + res[0]
          currentPointRelPos = res[1]
        else:
          res = computePointsLanding(points,curves,curveId,precInterType,Z1,dZ,jumpDistance)
          curve3d = curve3d + res[0]
          currentPointRelPos = res[1]
        res = computePointsBefore(points,curves,curveId,currentPointRelPos,Z1,intersections[i],jumpDistance)
        curve3d = curve3d + res[0]
        currentPointRelPos = res[1]
        res = computeJump(points,curves,curveId,currentPointRelPos,ldz,Z1,intersections[i])
        curve3d = curve3d + res[0]
        currentPointRelPos = res[1]
        Zpos = 1
    precInterType = intersections[i]
  
  if(currentPointRelPos[0] < len(curve)-1):
    if(Zpos == 1):
      if(precInterType[2]):
        ljumpDistance = min(jumpDistance,geodesicDistance(points,curves[curveId],(len(curve)-2,0.99))-geodesicDistance(points,curves[curveId],precInterType)-g_nozzle_diam )
        if(ljumpDistance>0):
          res = computePointsLanding(points,curves,curveId,precInterType,Z1,dZ+g_ddZ,ljumpDistance)
          curve3d = curve3d + res[0]
          currentPointRelPos = res[1]
      else:
        ljumpDistance = min(jumpDistance,geodesicDistance(points,curves[curveId],(len(curve)-2,0.99))-geodesicDistance(points,curves[curveId],precInterType)-g_nozzle_diam )
        if(ljumpDistance>0):
          res = computePointsLanding(points,curves,curveId,precInterType,Z1,dZ,ljumpDistance)
          curve3d = curve3d + res[0]
          currentPointRelPos = res[1]
      
  for i in range(currentPointRelPos[0]+1,len(curve)):
    curve3d.append((points[curve[i]][0],points[curve[i]][1],Z1))
  return curve3d
    
   
def computeGcode(points,curves,f,initZ,deltaZ,jumpDistance):
  curves3D = []
  print("computeGcode")
  for i in range(0,len(curves)):
    intersections = []
    curve3D = []
    for j in range(0,len(curves)):
      if(j > i):continue
      intersections = intersections + intersectionsCurveCurve(points,curves,i,j,False)
    sortIntersectionAlongCurve(intersections)

    curve3D = computeSingle3DCurve(points,curves,i,intersections,jumpDistance,initZ,deltaZ)
    curves3D.append(curve3D)
    
  curves.reverse()
  for i in range(0,len(curves)):
    curves[i].reverse()
  for i in range(0,len(curves)):
    intersections = []
    curve3D = []
    for j in range(0,len(curves)):
      intersections = intersections + intersectionsCurveCurve(points,curves,i,j,i>j)
    sortIntersectionAlongCurve(intersections)
    curve3D = computeSingle3DCurve(points,curves,i,intersections,jumpDistance,2*initZ,deltaZ)
    curves3D.append(curve3D)
  E = 0
    
  retracted = False
  for curve3d in curves3D:
    #go to first point
    cpoint = curve3d[0]   
    f.write("G1 X" + str(cpoint[0]) + " Y" + str(cpoint[1]) + " Z" + str(g_initZ*2+g_deltaZ*2+g_ddZ)+ "\n")
    f.write("G1 X" + str(cpoint[0]) + " Y" + str(cpoint[1]) + " Z" + str(cpoint[2]) + "\n")
    for pt3d in curve3d:
      if(cpoint == pt3d):
        continue
      E = E + computeE(cpoint,pt3d,initZ)
      f.write("G1 X" + str(pt3d[0]) + " Y" + str(pt3d[1]) + " Z" + str(pt3d[2])+ " E" + str(E) +"\n")
      cpoint = pt3d
    f.write("G1 X" + str(cpoint[0]) + " Y" + str(cpoint[1]) + " Z" + str(g_initZ*2+g_deltaZ*2+g_ddZ)+ "\n")



def writeGeomDefinition(sFile,points,curves):

  intersections = []
  curve3D = []
  perCurveIntersections = []
  
  with open(sFile, "w") as f: 
    f.write(str(len(curves)) +"\n")#number of curves
    for i in  range(0,len(curves)):
      intersections = []
      perCurveIntersection = []
      for j in range(0,len(curves)):
        lint = intersectionsCurveCurve(points,curves,i,j,False)
        intersections = intersections + lint
      sortIntersectionAlongCurve(intersections)
      dpos = 1
      for intersection in intersections:
        curves.insert(dpos+ intersection[0], computePosOnCurve0(p0,p1,intersection))
        perCurveIntersections.append(dpos+intersection[0],i,j)
        dpos = dpos + 1

    for i in range(0,len(curves)):
      f.write(str(len(curve[i])) + "\n")#number of points per curve
      for j in range(0,curves[i]):
        f.write(str(points[curves[i][j]]) + "\n")#point

    f.write(str(len(perCurveIntersections))+"\n")
    for perCurveIntersection in perCurveIntersections:
      f.write(str(perCurveIntersection) + "\n")#intersection

def writePointsToFile(sFile,points,curves,initZ,deltaZ,jumpDistance):
  E = 0.0

  with open(sFile, "w") as f: 
    writeHeader(f)
    computeGcode(points,curves,f,initZ,deltaZ,jumpDistance)
    f.write("M107\n")
    f.write("M104 S0 ; turn off extruder\n")
    f.write("M140 S0 ; turn off heatbed\n")
    f.write("M107 ; turn off fan\n")
    f.write("G1 X0 Y210; home X axis and push Y forward\n")
    f.write("M84 ; disable motors\n")
    f.write("M82 ;absolute extrusion mode\n")
    f.write("M104 S0\n")


import sys
script = __import__(sys.argv[1]) 

points = script.points
curves = script.curves
print(str(len(points)))
print(str(len(curves)))
writePointsToFile(sys.argv[2],points,curves,g_initZ,g_deltaZ,g_jumpDistance)


#{createPoints3(points,edges)    
#createInterlace(points,edges)

