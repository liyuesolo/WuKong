from tkinter import*
import random
import copy
from math import *
import numpy as np


Pi = 3.14159265359
root = Tk()
root.title('Graphic')
canvas = Canvas(root, width=1350, height=800, bg='sky blue')
canvas.pack()

x_min = 10
x_max = 90
y_min = 10
y_max = 90
deltaZ = 0.2
initZ = 0.2

nozzle_diam = 0.4
filament_diam = 1.75

points = []
edges = []

def Reverse(tuples): 
    new_tup = tuples[::-1] 
    return new_tup 
    
def dist(pA,pB):
  asq = (pA[0] - pB[0])*(pA[0] - pB[0])
  bsq = (pA[1] - pB[1])*(pA[1] - pB[1])
  return sqrt(asq + bsq)

def computeE(A, B,lHeight):
  x = A[0] - B[0]
  y = A[1] - B[1]
  srf = sqrt(x*x+y*y) * lHeight
  cSection = (filament_diam / 2.0)*(filament_diam / 2.0) * Pi
  nozzleSection = (nozzle_diam * nozzle_diam)/4.0 * pi
  return srf * nozzleSection / cSection
  
def writeHeader(f):
  f.write("M82 ;absolute extrusion mode\n")
  f.write("G21 ; set units to millimeters\n")
  f.write("G90 ; use absolute positioning\n")
  f.write("M82 ; absolute extrusion mode\n")
  f.write("M104 S230.0 ; set extruder temp\n")
  f.write("M140 S60.0 ; set bed temp\n")
  f.write("M190 S60.0 ; wait for bed temp\n")
  f.write("M109 S230.0 ; wait for extruder temp\n")
  f.write("G28 W ; home all without mesh bed level\n")
  f.write("G80 ; mesh bed leveling\n")
  f.write("G92 E0.0 ; reset extruder distance position\n")
  f.write("G1 Y-3.0 F1000.0 ; go outside print area\n")
  f.write("G1 X60.0 E9.0 F1000.0 ; intro line\n")
  f.write("G1 X100.0 E21.5 F1000.0 ; intro line\n")
  f.write("G92 E0.0 ; reset extruder distance position\n")
  f.write("G92 E0\n")
  f.write("G1 F2100 E-0.2\n")
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
  
def computeNextPos(posA,posB,size):
  c = (posB[0]-posA[0])/dist(posB,posA),(posB[1]-posA[1])/dist(posB,posA)
  return posA[0] + c[0]*size, posA[1] + c[1]*size


def sortAlong(intersections):
  intersections.sort(key=lambda x: x[2])  

def compute1Part(points,edge,intersections,cpos,f,Z1,deltaZ,E):
  distanceBeforeUp = 1.0
  lastPoint = points[edge[0]]
  #print(cpos, edge[0])
  f.write(";start edge\n")#go to the first point on the edge; 

  if(cpos != edge[0]):
    f.write("G1 X" + str(lastPoint[0]) + " Y" + str(lastPoint[1]) + " Z" + str(Z1+deltaZ)+"\n")#go to the first point on the edge; 
    f.write("G1 Z" + str(Z1) + "\n")
  posZ = 0
  for intersect in intersections:
    distance = dist(lastPoint,intersect[1])      
    if(posZ==0):
      if(distance > distanceBeforeUp):#head is done, I have the time to print something before getting up
        ToPoint = computeNextPos(lastPoint,intersect[1],distance-distanceBeforeUp)
        dE = computeE(lastPoint,ToPoint,deltaZ)
        E = E + dE
        E = round(E,3)
        f.write("G1 X" + str(ToPoint[0]) + " Y" + str(ToPoint[1]) + " Z" + str(Z1)+ " E" + str(E) +"\n")
        lastPoint = ToPoint
      dE = computeE(intersect[1],lastPoint,deltaZ)
      E = E + dE
      E = round(E,3)
      if(intersect[0]):
        f.write("G1 X" + str(intersect[1][0]) + " Y" + str(intersect[1][1]) + " Z" + str(Z1+2*deltaZ)+ " E" + str(E)+"\n")
      else:
        f.write("G1 X" + str(intersect[1][0]) + " Y" + str(intersect[1][1]) + " Z" + str(Z1+deltaZ)+ " E" + str(E)+"\n")
      posZ = 1
      
    else:
      if(distance > 2*distanceBeforeUp):#head is done, I have the time to print something before getting up,I can get down, advance, then get Up again
        ToPoint = computeNextPos(lastPoint,intersect[1],distanceBeforeUp)#get Down 
        dE = computeE(lastPoint,ToPoint,deltaZ)
        E = E + dE
        E = round(E,3)
        f.write("G1 X" + str(ToPoint[0]) + " Y" + str(ToPoint[1]) + " Z" + str(Z1)+ " E" + str(E)+"\n")
        ToPoint2 = computeNextPos(lastPoint,intersect[1],distance-distanceBeforeUp)#advance
        dE = computeE(ToPoint,ToPoint2,deltaZ)
        E = E + dE
        E = round(E,3)
        f.write("G1 X" + str(ToPoint2[0]) + " Y" + str(ToPoint2[1]) + " Z" + str(Z1)+ " E" + str(E)+"\n")
        lastPoint = ToPoint2
      dE = computeE(intersect[1],lastPoint,deltaZ)
      E = E + dE
      E = round(E,3)
      if(intersect[0]):
        f.write("G1 X" + str(intersect[1][0]) + " Y" + str(intersect[1][1]) + " Z" + str(Z1+2*deltaZ)+ " E" + str(E)+"\n")
      else:
        f.write("G1 X" + str(intersect[1][0]) + " Y" + str(intersect[1][1]) + " Z" + str(Z1+deltaZ)+ " E" + str(E)+"\n")
    lastPoint = intersect[1]
    posZ = 1
    #finish the past by going down, and to the lastPoint
  finishingPoint = points[edge[1]]
  dE = computeE(lastPoint,finishingPoint,deltaZ)
  E = E + dE
  E = round(E,3)
  f.write("G1 X" + str(finishingPoint[0]) + " Y" + str(finishingPoint[1]) + " Z" + str(Z1)+ " E" + str(E)+"\n")
  cpos = edge[1]
  return E,cpos
    
    
def computeGcode(points,edges,f):
  intersections = []
  i = 0
  E = 0
  cpos = -1
  #first, do all the edges with getting up when needed
  for edge in edges:
    intersections = []
    j = 0
    for edge2 in edges:
      if(j >= i or i == j + 1):
        j = j + 1
        continue
      inter = intersectionPtPt(points[edge[0]],points[edge[1]],points[edge2[0]],points[edge2[1]])
      if(not inter[2]):
        j = j + 1
        continue
      intersections.append((False,getBC(points[edge[0]],points[edge[1]],inter[0]),inter[0]))
      j = j + 1
    sortAlong(intersections)
    E,cpos = compute1Part(points,edge,intersections,cpos,f,initZ,deltaZ,E)
    i = i + 1
  edges.reverse()
  i = 0
  #not exactly same thing and reversed order
  f.write(";REVERSE ORDER\n")
  cpos = -1

  for edge in edges:
    intersections = []

    _edge = edge[1], edge[0], edge[2]
    #_edge = edge
    j = 0
    for edge2 in edges:
      _edge2 = edge2[1], edge2[0], edge2[2]
      if(j == i or j == i + 1 or i == j + 1):
        j = j + 1
        continue
      inter = intersectionPtPt(points[_edge[0]],points[_edge[1]],points[_edge2[0]],points[_edge2[1]])
      interType = j < i#not the same height in this case
      if(not inter[2]):
        j = j + 1
        continue
      intersections.append((interType,getBC(points[_edge[0]],points[_edge[1]],inter[0]),inter[0]))
      j = j + 1
    sortAlong(intersections)
    E,cpos = compute1Part(points,_edge,intersections,cpos,f,initZ+deltaZ,deltaZ,E)
    i = i + 1
  edges.reverse()
  
           
def writePointsToFile(sFile,points,edges):
  E = 0.0
  Z = 0.5

  with open(sFile, "w") as f: 
    writeHeader(f)
    computeGcode(points,edges,f)
    f.write("M107\n")
    f.write("M104 S0 ; turn off extruder\n")
    f.write("M140 S0 ; turn off heatbed\n")
    f.write("M107 ; turn off fan\n")
    f.write("G1 X0 Y210; home X axis and push Y forward\n")
    f.write("M84 ; disable motors\n")
    f.write("M82 ;absolute extrusion mode\n")
    f.write("M104 S0\n")

def createInterlaceTest(points,edges):
  pt0 = (0,0,0)
  pt1 = (10,10,0)
  
  pt2 = (0,10,0)
  pt3 = (10,0,0)
  
  ed0 = (0,1,1.0)  
  ed1 = (2,3,1.0)  
  points.append(pt0) 
  points.append(pt1) 
  points.append(pt2)
  points.append(pt3) 
  edges.append(ed0)
  edges.append(ed1)
    
    
def createInterlaceTest(points,edges):
  pt0 = (0,0,0)
  pt1 = (10,10,0)
  
  pt2 = (0,10,0)
  pt3 = (10,0,0)
  
  ed0 = (0,1,1.0)  
  ed1 = (2,3,1.0)  
  points.append(pt0) 
  points.append(pt1) 
  points.append(pt2)
  points.append(pt3) 
  edges.append(ed0)
  edges.append(ed1)

  
def createInterlace(points,edges):
  xs = 50
  ys = 50
  dx = 2.5
  dy = 2.5
  pos = 0
  for i in np.arange(0, xs,dx):
    pt0 = (i,0)
    pt1 = (i,xs)
    points.append(pt0) 
    points.append(pt1) 
    edges.append((pos, pos+1,1.0))
    pos = pos + 2

  for i in np.arange(0, ys,dy):
    pt0 = (0,i)
    pt1 = (ys,i)
    points.append(pt0) 
    points.append(pt1) 
    edges.append((pos, pos+1,1.0))
    pos = pos + 2

def randomDir(vn):
  x = random.randint(1,1000)
  y = random.randint(1,1000)
  x -= 500
  y -= 500
  len = sqrt(x*x+y*y)
  vn[0] = x/len
  vn[1] = y/len
  
def createPoints3(points,edges):
  x_lin = [0.5*(x_max-x_min),0.5*(y_max-y_min)]
  x0 = copy.copy(x_lin)
  v_lin = [1,1]
  randomDir(v_lin)
  t = 0.0
  dt = 1.0
  t_end = 100
  v_ang = 0.2
  v_lin_mag = 0.1
  v_lin[0] *= v_lin_mag
  v_lin[1] *= v_lin_mag
  r = 10.0
  x1 = copy.copy(x0)
  while(t<t_end):
    x_lin[0] += v_lin[0]*dt
    x_lin[1] += v_lin[1]*dt
    if outside(x_lin):
      clip(x_lin)
      randomDir(v_lin)
      v_lin[0] *= v_lin_mag
      v_lin[1] *= v_lin_mag
    x1[0] = x_lin[0] + 1.2*r*cos(v_ang*t)
    x1[1] = x_lin[1] + 1.0*r*sin(v_ang*t)
    points.append([x1[0],x1[1]])
    if(len(points)>=2):edges.append((len(points)-2,len(points)-1,1.0))
    t += 1.0
  print(len(points))
  print(len(edges))
  
import sys
script = __import__(sys.argv[1]) 

points = script.points
edges = script.edges

writePointsToFile(sys.argv[2],points,edges)


#{createPoints3(points,edges)    
#createInterlace(points,edges)

