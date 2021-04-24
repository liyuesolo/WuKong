from tkinter import*
import random
import copy
from math import *
import numpy as np


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
    

def line(p1, p2):
  A = (p1[1] - p2[1])
  B = (p2[0] - p1[0])
  C = (p1[0]*p2[1] - p2[0]*p1[1])
  return A, B, -C

def cross(v0,v1):
  return v0[0] * v1[1] - v0[1] * v1[0]
    

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
  t_end = 10000
  v_ang = .1
  v_lin_mag = 0.5
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
  


print("toto")
createPoints3(points,edges)    
#createInterlace(points,edges)

