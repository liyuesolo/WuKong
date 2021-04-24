from tkinter import*
import random
import copy
from math import *
import numpy as np

Pi = 3.14159265359
points = []
edges = []

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


def lines(points,edges,center,r):
  for i in np.arange(0, Pi,0.3):
    pt = cos(i)*r + center[0],sin(i)*r+ center[1]
    pt2 = -cos(i)*r + center[0],-sin(i)*r+ center[1]

    points.append(pt)
    points.append(pt2)
    edges.append((len(points)-2,len(points)-1,1.0))      
  
def circles(points,edges,center, r):
  i_point = len(points)
  
  pushed = 0
  for i in np.arange(0.1, 2*Pi+0.1,0.1):
    pt = cos(i)*r + center[0],sin(i)*r+ center[1]
    points.append(pt)
    pushed = pushed + 1
    if(pushed > 1):
      edges.append((len(points)-2,len(points)-1,1.0))      
  edges.append((i_point,len(points)-1,1.0))
  
center = 100,100
r = 5.0


for i in range(0,10):
  circles(points,edges,center,r*i)
lines(points,edges,center,r*10+1.0)

#createInterlace(points,edges)

