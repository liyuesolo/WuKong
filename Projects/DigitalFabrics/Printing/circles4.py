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

def toto():
  for i in range(0,10):
    circles(points,curves,center,r*i)
      
  lines(points,curves,center,r*10+1.0,3.0,0)
  lines(points,curves,center,r*10+1.0,8.0,2*Pi/16)
  lines(points,curves,center,r*10+1.0,8.0,4*Pi/16)
  lines(points,curves,center,r*10+1.0,8.0,6*Pi/16)

  lines(points,curves,center,r*10+1.0,28.0,1*Pi/32)
  lines(points,curves,center,r*10+1.0,13.0,2*Pi/32)
  lines(points,curves,center,r*10+1.0,28.0,3*Pi/32)

  lines(points,curves,center,r*10+1.0,28.0,5*Pi/32)
  lines(points,curves,center,r*10+1.0,13.0,6*Pi/32)
  lines(points,curves,center,r*10+1.0,28.0,7*Pi/32)

  lines(points,curves,center,r*10+1.0,28.0,9*Pi/32)
  lines(points,curves,center,r*10+1.0,13.0,10*Pi/32)
  lines(points,curves,center,r*10+1.0,28.0,11*Pi/32)

  lines(points,curves,center,r*10+1.0,28.0,13*Pi/32)
  lines(points,curves,center,r*10+1.0,13.0,14*Pi/32)
  lines(points,curves,center,r*10+1.0,28.0,15*Pi/32)

middle = 100,100
r = 5.0

#circles(points,edges,middle,50)
#rosace(middle,50,points,curves)  
Icenter = 30,30
#circlesAlong(points,curves,Icenter,(10,10),8,9.0)


#createInterlace(points,edges)

