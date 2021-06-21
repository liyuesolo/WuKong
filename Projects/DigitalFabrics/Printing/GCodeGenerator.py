import numpy as np
import os

# G1 Z0.800 F10800.000

class GCodeGenerator:

    def __init__(self, gcode_file, temperature = 215, printer_type = "PrusaI3"):
        
        self.temperature = temperature
        self.printer_type = printer_type
        self.gcode_file = gcode_file

        self.header_written = False
        self.footer_written = False

        self.absolute_extrusion = False

        self.last_E = 0.0

        if(self.printer_type == "PrusaI3"):
            self.nozzle_diameter = 0.4
            self.filament_diameter = 0.175
        else:
            print("Please add printer and filament info for this printer....")
    
    def writeHeader(self):
        if(self.printer_type == "PrusaI3"):
            if self.absolute_extrusion:
                self.gcode_file.write("M82 ;absolute extrusion mode\n")
            else:
                self.gcode_file.write("M83 ;relative extrusion mode\n")
            self.gcode_file.write("G21 ; set units to millimeters\n")
            self.gcode_file.write("G90 ; use absolute positioning\n")
            self.gcode_file.write("M104 S" +str(self.temperature) + " ;set extruder temp\n")
            self.gcode_file.write("M140 S60.0 ; set bed temp\n")
            self.gcode_file.write("M190 S60.0 ; wait for bed temp\n")
            self.gcode_file.write("M109 S" +str(self.temperature) + " ;set extruder temp\n")
            self.gcode_file.write("G28 W ; home all without mesh bed level\n")
            self.gcode_file.write("G80 ; mesh bed leveling\n")
            self.gcode_file.write("G92 E0.0 ; reset extruder distance position\n")
            self.gcode_file.write("G1 Y-3.0 F1000.0 ; go outside print area\n")
            self.gcode_file.write("G1 X60.0 E9.0 F1000.0 ; intro line\n")
            self.gcode_file.write("G1 X100.0 E15.0 F600.0 ; intro line\n")
            self.gcode_file.write("G1 F600.0 ; intro line\n")
            self.gcode_file.write("G1 Z1 F600.0 ; move up nozzle\n")
            self.gcode_file.write("G92 E0.0 ; reset extruder distance position\n")
            self.gcode_file.write("M107\n")
            self.header_written = True
    
    def writeFooter(self):
        if(self.printer_type == "PrusaI3"):
            self.gcode_file.write("M107\n")
            self.gcode_file.write("M104 S0 ; turn off extruder\n")
            self.gcode_file.write("M140 S0 ; turn off heatbed\n")
            self.gcode_file.write("M107 ; turn off fan\n")
            self.gcode_file.write("G1 Z33.6 ; Move print head up\n")
            self.gcode_file.write("G1 X0 Y210; home X axis and push Y forward\n")
            self.gcode_file.write("M84 ; disable motors\n")
            self.gcode_file.write("M73 P100 R0\n")
            self.gcode_file.write("M73 Q100 S0\n")
            self.footer_written = True
    
    def moveTo(self, X, Y, Z, F=None):
        cmd = "G1 X" + str(X) + " Y" + str(Y) + " Z" + str(Z)
        if F is not None:
            cmd += " F" + str(F)
        cmd += "\n"
        self.gcode_file.write(cmd)

    def retract(self, E):
        self.gcode_file.write("G1 E" + str(E) + "\n")
    

    def computeExtrusion(self, X, Y, Z, material_height):

        length = np.sqrt(
                (X[1] - X[0]) * (X[1] - X[0]) + 
                (Y[1] - Y[0]) * (Y[1] - Y[0]) + 
                (Z[1] - Z[0]) * (Z[1] - Z[0]))

        rf = self.filament_diameter / 2.0
        
        filameter_area_cross_section = np.pi * rf * rf
        
        # E * filameter_area_cross_section = self.nozzle_diameter * material_height * material_height_length
        # print(length, self.nozzle_diameter, material_height, filameter_area_cross_section)
        extrusion_amount = length * self.nozzle_diameter * material_height / filameter_area_cross_section
        # print(extrusion_amount)
        
        return extrusion_amount


    def writeLine(self, X, Y, Z, material_height, E=None, F=None, move_up_nozzle=True, move_up_nozzle2=True):
        
        if move_up_nozzle:
            self.moveTo(X[0], Y[0], Z[0] + 2.5, F = 2000)

        self.moveTo(X[0], Y[0], Z[0], F = 2000)
        self.retract(0.4)
        cmd = "G1 X" + str(X[1]) + " Y" + str(Y[1]) + " Z" + str(Z[1])
        if self.absolute_extrusion:
            cmd += " E" + str(self.last_E + E)
            self.last_E += E
        else:
            # cmd += " E" + str(E)
            cmd += " E" + str(self.computeExtrusion(X, Y, Z, material_height))
        cmd += " F" + str(F)
        self.gcode_file.write(cmd+"\n")
        self.retract(-0.4)
        if move_up_nozzle2:
            self.moveTo(X[1], Y[1], Z[1] + 2.5, F = 2000)

    def checkValid(self):
        print(self.header_written and self.footer_written)
        
    

            


    
