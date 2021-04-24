from tkinter import*
import random
import copy
from math import *
import numpy as np

Pi = 3.14159265359
points = []
curves = []

x_min = 10
x_max = 90
y_min = 10
y_max = 90


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


    
def circles(points,curves,center, r):
  i_point = len(points)
  curve = []
  pushed = 0
  for i in np.arange(0.1, 2*Pi+0.1,2*Pi/40):
    pt = cos(i)*r + center[0],sin(i)*r+ center[1]
    points.append(pt)
    pushed = pushed + 1
    curve.append(len(points)-1)      
  curve.append(i_point)
  curves.append(curve)

def circlesAlong(points,curves,Icenter,line,N,ray):
  for i in range(1,N+1):
    for j in range(1,N+1):
      center = Icenter[0]+i*line[0],Icenter[1]+j*line[1]
      circles(points,curves,center, ray)
    

def lines(points,curves,center,r,deltaCenter,offset):
  for i in np.arange(offset,offset+ 2*Pi,2*Pi/4):
    curve = []
    pt = cos(i)*r + center[0],sin(i)*r+ center[1]
    pt2 = cos(i)*deltaCenter + center[0],sin(i)*deltaCenter+ center[1]
    points.append(pt)
    points.append(pt2)
    curve.append(len(points)-2)
    curve.append(len(points)-1)      
    curves.append(curve)
 

def rosace(center,ray,points,curves):
  i_point = len(points)
  pushed = 0
  for i in np.arange(0.0, 2*Pi,2*Pi):
    pt = cos(i)*ray/2 + center[0],sin(i)*ray/2+ center[1]
    circles(points,curves,pt,ray/2)
    
center = 100,100
r = 5.0

def rasterCircle(center,ray,points,curves,step):

  sens = 0
  curve = []

  for i in np.arange(-ray+step,ray-step,step):
    print(i)
    x0 = sqrt(ray*ray-(i*i))
    x1 = -x0
    pt = x0+center[0],i+center[1]
    pt2 = x1+center[0],i+center[1]
    points.append(pt)
    points.append(pt2)
    if(sens == 0):
      curve.append(len(points)-2)
      curve.append(len(points)-1)      
    if(sens == 1):
      curve.append(len(points)-1)
      curve.append(len(points)-2) 
    sens = sens+1
    sens = sens%2
  
  curve2 = []
  for i in np.arange(-ray+step,ray-step,step):
    print(i)
    x0 = sqrt(ray*ray-(i*i))
    x1 = -x0
    pt = i+center[1],x0+center[0]
    pt2 = i+center[1],x1+center[0]
    points.append(pt)
    points.append(pt2)
    if(sens == 0):
      curve2.append(len(points)-2)
      curve2.append(len(points)-1)      
    if(sens == 1):
      curve2.append(len(points)-1)
      curve2.append(len(points)-2) 
    sens = sens+1
    sens = sens%2  
    
  curves.append(curve)
  curves.append(curve2)

r = 5.0

#circles(points,edges,middle,50)
#rosace(middle,50,points,curves)  
Icenter = 100,100
#circlesAlong(points,curves,Icenter,(10,10),8,9.0)
rasterCircle(Icenter,40,points,curves,4.0)

#createInterlace(points,edges)

