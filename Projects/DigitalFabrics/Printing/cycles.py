from tkinter import*
import random
import copy
from math import *
Pi = 3.14159265359
root = Tk()
root.title('Graphic')
canvas = Canvas(root, width=1350, height=800, bg='sky blue')
canvas.pack()

x_min = 10
x_max = 90
y_min = 10
y_max = 90

def drawPoints(points):
  p0 = points[0]
  for p1 in points:
    #canvas.create_line(p0[0],p0[1],p1[0],p1[1], width=4)
    canvas.create_line(10.0*p0[0],10.0*p0[1],10.0*p1[0],10.0*p1[1], width=4)
    p0 = copy.copy(p1)

    
    
points = []

def randomDir(vn):
  x = random.randint(1,1000)
  y = random.randint(1,1000)
  x -= 500
  y -= 500
  len = sqrt(x*x+y*y)
  vn[0] = x/len
  vn[1] = y/len
  #print("vn=(" +str(vn[0]) + "," + str(vn[1]) + ")")
  #vn[0] = -vn[1]
  #vn[1] = vn[0]
  
def computeE(A, B,lHeight):
  x = A[0] - B[0]
  y = A[1] - B[1]
  srf = sqrt(x*x+y*y) * lHeight
  cSection = (1.75 / 2.0)*(1.75 / 2.0) * Pi
  return srf / cSection
  
def writePointsToFile(sFile,points):
  E = 0.0
  Z = 0.5
  with open(sFile, "w") as f:
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
    f.write("G0 F7200 X63.704 Y106.828 Z" + str(Z) + "\n")  
    #f.write("G0 F4000 Z10.0\n")  
    f.write("G0 F2000 E" + str(E) + "\n")  
    precP = points[0]
    for p in points:
      dE = computeE(p,precP,Z)
      E += dE
      f.write("G1 X" + str(p[0]) + " Y" + str(p[1]) + " E" + str(E) + "\n")
      precP  = p
      
    f.write("M107\n")
    f.write("M104 S0 ; turn off extruder\n")
    f.write("M140 S0 ; turn off heatbed\n")
    f.write("M107 ; turn off fan\n")
    f.write("G1 X0 Y210; home X axis and push Y forward\n")
    f.write("M84 ; disable motors\n")
    f.write("M82 ;absolute extrusion mode\n")
    f.write("M104 S0\n")
  
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


def createPoints1(points):
  for i in range(10):
    x = random.randint(1,1350)
    y = random.randint(1,800)
    points.append([x,y])

def createPoints2(points):
  p0 = [400,400]
  v0 = [2,0]
  alpha_max =  Pi/100.
  alpha0 = 0.01
  for i in range(10000):
    pc = copy.copy(p0)
    pc[0] += v0[0]
    pc[1] += v0[1]
    points.append([pc[0],pc[1]])
    p0 = copy.copy(pc)
    v1 = [0.,0.]    
    v1[0] = cos(alpha0)*v0[0] - sin(alpha0)*v0[1]
    v1[1] = sin(alpha0)*v0[0] + cos(alpha0)*v0[1]
    if i%40 == 0:
      dalpha = random.randint(-180,180)/180
      dalpha *= alpha_max/4.
      alpha = alpha0+dalpha
      if alpha > alpha_max:
        alpha = alpha_max
      if alpha < -alpha_max:
        alpha = -alpha_max
      alpha0 = alpha
    v0 = copy.copy(v1)

def createPoints3(points):
  x_lin = [0.5*(x_max-x_min),0.5*(y_max-y_min)]
  x0 = copy.copy(x_lin)
  v_lin = [1,1]
  randomDir(v_lin)
  t = 0.0
  dt = 1.0
  t_end = 1000
  v_ang = 1.0
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
    t += 1.0
    

def createInterlace(points):
  xMax = 50
  yMin = 50
  x0 = 0
  fqcy = 2.0;
  while(x0<xMax):
  pt
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
    t += 1.0
createPoints3(points)
drawPoints(points)  
writePointsToFile("Circles.gcode", points)

root.mainloop()

for i in range(36):
  x = 50 + (i*40)
  canvas.create_line(x,800,x,-850, width=4)
  
for i in range(24):
  y = 100 - (i*40)
  canvas.create_line(1600,-y,10,-y, width=4)
